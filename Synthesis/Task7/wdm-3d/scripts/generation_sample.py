"""
A script for sampling from a diffusion model for unconditional image generation.
"""

import argparse
import nibabel as nib
import numpy as np
import os
import pathlib
import random
import sys
import torch as th

sys.path.append(".")
sys.path.insert(1, '/projects/brats2023_a_f/BraTS2024_cluster/7_MissingMRI/src')   
from guided_diffusion.c_bratsloader import c_BraTSVolumes
from guided_diffusion import (dist_util,
                              logger)
from guided_diffusion.script_util import (model_and_diffusion_defaults,
                                          create_model_and_diffusion,
                                          add_dict_to_argparser,
                                          args_to_dict,
                                          )
from DWT_IDWT.DWT_IDWT_layer import DWT_3D, IDWT_3D


def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min)/ (_max - _min)
    return normalized_img


def dice_score(pred, targs):
    pred = (pred>0).float()
    return 2. * (pred*targs).sum() / (pred+targs).sum()


def main():
    args = create_argparser().parse_args()
    seed = args.seed
    logger.configure()

    args.devices = [th.cuda.current_device()]
    dist_util.setup_dist(devices=args.devices)
    print(f"Devices: {args.devices}")

    logger.log("Creating model and diffusion...")
    args.diffusion_steps = int(args.sampling_steps)
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    logger.log("Load model from: {}".format(args.model_path))
    model.load_state_dict(dist_util.load_state_dict(args.model_path, map_location="cpu"))
    model.to(dist_util.dev([0, 1]) if len(args.devices) > 1 else dist_util.dev())  # allow for 2 devices

    if args.use_fp16:
        raise ValueError("fp16 currently not implemented")

    model.eval()
    idwt = IDWT_3D("haar")
    dwt = DWT_3D('haar')

    # Creating data loader
    datal, ds = c_BraTSVolumes(directory=args.data_dir, 
                          batch_size=args.batch_size,
                          num_workers=int(args.num_workers), 
                          mode=args.mode,
                          img_size=args.image_size,
                          data_split_json=args.data_split_json).get_dl_ds()
    # Creating iterator
    iterator_data = iter(datal)

    for ind in range(args.num_samples // args.batch_size):
        th.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        # print(f"Reseeded (in for loop) to {seed}")
        seed += 1
        # Folder to save the output
        args.output_dir = f"./results/{args.train_mode}_{args.sampling_steps}"
        pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)

        list_of_all_modals_to_gen = ["t1n", "t1c", "t2f", "t2w"]

        for modal_to_generate in list_of_all_modals_to_gen:
            # Getting next batch
            batch = next(iterator_data)
            t1n_modal = batch["t1n"].to(dist_util.dev())
            t1c_modal = batch["t1c"].to(dist_util.dev())
            t2f_modal = batch["t2f"].to(dist_util.dev())
            t2w_modal = batch["t2w"].to(dist_util.dev())
            seg = batch["seg"].to(dist_util.dev())
            #modal_to_generate = "t2w"

            # Replace one of the modal with noise
            # Create a NIfTI image
            nifti_image = nib.Nifti1Image(np.squeeze(batch[modal_to_generate].numpy()), affine=np.eye(4))
            # Save the NIfTI image to a file
            nib.save(nifti_image, f'{args.output_dir}/{modal_to_generate}_modal_{ind}.nii.gz') 
            ######
            noise_as_input = th.randn(args.batch_size,         # Batch size
                        1,                       # 32 wavelet coefficients
                        args.image_size,      # Half spatial resolution (D)
                        args.image_size,      # Half spatial resolution (H)
                        args.image_size,      # Half spatial resolution (W)
                        ).to(dist_util.dev())
            if modal_to_generate=="t1n":
                t1n_modal = noise_as_input
            elif modal_to_generate=="t1c":
                t1c_modal = noise_as_input
            elif modal_to_generate=="t2f":
                t2f_modal = noise_as_input
            elif modal_to_generate=="t2w":
                t2w_modal = noise_as_input
            else:
                print("This modal does not exist")
            
        
            if args.train_mode=="default" or args.train_mode=="known_all_time" or args.train_mode=="known_3_to_gen_1":
                print(f"Doing mode default")
                combined_mri = th.cat((t1n_modal, t1c_modal, t2f_modal, t2w_modal), dim=1)
                # Wavelet transform the input image
                LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = dwt(combined_mri)
                x_start_dwt = th.cat([LLL / 3., LLH, LHL, LHH, HLL, HLH, HHL, HHH], dim=1) 
                print(f"x_start_dwt: {x_start_dwt.shape}")
                img = x_start_dwt

            else:
                # Even with the same noise in all 4 modalities, the results are bad :D
                noise_base = th.randn(args.batch_size,         # Batch size
                            32,                       # 32 wavelet coefficients
                            args.image_size//2,      # Half spatial resolution (D)
                            args.image_size//2,      # Half spatial resolution (H)
                            args.image_size//2,      # Half spatial resolution (W)
                            ).to(dist_util.dev())
                modal_to_generate = None
                
            model_kwargs = {}

            sample_fn = diffusion.p_sample_loop

            sample = sample_fn(model=model,
                            time=int(args.sampling_steps),
                            shape=img.shape,
                            input_volume=img,
                            clip_denoised=args.clip_denoised,
                            model_kwargs=model_kwargs,
                            modal_to_generate=modal_to_generate,
                            mode=args.train_mode,
                            )

            B, C, D, H, W = sample.size()

            print(f"sample.shape: {sample.shape}")
            modal_list_name = ['t1n', 't1c', 't2f', 't2w']
            if args.train_mode=="default" or args.train_mode=="known_all_time":
                for modal_idx, modal in enumerate(modal_list_name):
                    new_sample = idwt(sample[:, 0+modal_idx, :, :, :].view(B, 1, D, H, W) * 3.,
                                    sample[:, 4+modal_idx, :, :, :].view(B, 1, D, H, W),
                                    sample[:, 8+modal_idx, :, :, :].view(B, 1, D, H, W),
                                    sample[:, 12+modal_idx, :, :, :].view(B, 1, D, H, W),
                                    sample[:, 16+modal_idx, :, :, :].view(B, 1, D, H, W),
                                    sample[:, 20+modal_idx, :, :, :].view(B, 1, D, H, W),
                                    sample[:, 24+modal_idx, :, :, :].view(B, 1, D, H, W),
                                    sample[:, 28+modal_idx, :, :, :].view(B, 1, D, H, W))

                    #new_sample = (new_sample + 1) / 2.

                    if len(new_sample.shape) == 5:
                        new_sample = new_sample.squeeze(dim=1)  # don't squeeze batch dimension for bs 1

                    for i in range(new_sample.shape[0]):
                        output_name = os.path.join(args.output_dir, f'sample_{ind}_{i}_{modal}.nii.gz')
                        img = nib.Nifti1Image(new_sample.detach().cpu().numpy()[i, :, :, :], np.eye(4))
                        nib.save(img=img, filename=output_name)
                        print(f'Saved to {output_name}')
            else:
                new_sample = idwt(sample[:, 0, :, :, :].view(B, 1, D, H, W) * 3.,
                                sample[:, 1, :, :, :].view(B, 1, D, H, W),
                                sample[:, 2, :, :, :].view(B, 1, D, H, W),
                                sample[:, 3, :, :, :].view(B, 1, D, H, W),
                                sample[:, 4, :, :, :].view(B, 1, D, H, W),
                                sample[:, 5, :, :, :].view(B, 1, D, H, W),
                                sample[:, 6, :, :, :].view(B, 1, D, H, W),
                                sample[:, 7, :, :, :].view(B, 1, D, H, W))
                if len(new_sample.shape) == 5:
                        new_sample = new_sample.squeeze(dim=1)  # don't squeeze batch dimension for bs 1

                for i in range(new_sample.shape[0]):
                    output_name = os.path.join(args.output_dir, f'sample_{ind}_{i}_{modal_to_generate}.nii.gz')
                    img = nib.Nifti1Image(new_sample.detach().cpu().numpy()[i, :, :, :], np.eye(4))
                    nib.save(img=img, filename=output_name)
                    print(f'Saved to {output_name}')
        
                


def create_argparser():
    defaults = dict(
        seed=0,
        data_dir="",
        data_mode='validation',
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=False,
        class_cond=False,
        sampling_steps=0,
        model_path="",
        devices=[0],
        output_dir='./results',
        mode=None,
        renormalize=False,
        image_size=256,
        half_res_crop=False,
        concat_coords=False, # if true, add 3 (for 3d) or 2 (for 2d) to in_channels
        data_split_json=None,
        num_workers=None, 
        train_mode=None, 
    )
    defaults.update({k:v for k, v in model_and_diffusion_defaults().items() if k not in defaults})
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
