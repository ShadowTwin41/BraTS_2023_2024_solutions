"""
A script for sampling from a diffusion model for unconditional image generation.
"""

import argparse
import nibabel as nib
import numpy as np
import os
import shutil
import zipfile
import pathlib
import random
import sys
import torch as th
from scipy.stats import linregress

sys.path.append(".")
sys.path.insert(1, '/projects/brats2023_a_f/BraTS2024_cluster/8_InPainting/src')   
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

def dilate_a_bit(img):
    """
    makes the mask bigger to avoid problems with the edges. 
    We are using a wavelet to downsample the image and label, so the edges have problems :(
    The upside is train speed and no need of much VRAM
    """
    import scipy 
    while len(img.shape)>3:
        img = img[0]
    dilation_struct = scipy.ndimage.generate_binary_structure(3, 2)
    dil_factor = 2

    gt_mat_dilation = scipy.ndimage.binary_dilation(input=img, structure=dilation_struct, iterations=dil_factor)
    gt_mat_dilation = np.expand_dims(gt_mat_dilation, axis=0)

    return gt_mat_dilation

def crop_to_240_240_155(tensor):
    # Calculate starting indices for cropping
    start_x = (256 - 240) // 2  # for the second dimension
    start_y = (256 - 240) // 2  # for the third dimension
    start_z = (256 - 155) // 2  # for the fourth dimension

    # Crop the tensor to shape [1, 240, 240, 155]
    cropped_tensor = tensor[start_x:start_x + 240, 
                            start_y:start_y + 240, 
                            start_z:start_z + 155]
    return cropped_tensor

def get_affine(scan_path):
    '''
    Get the metada from the nifti file
    '''
    scan = nib.load(scan_path)
    header = scan.header
    affine = scan.affine
    return affine, header

def rescale_array_numpy(arr, mask, minv, maxv): #monai function adapted
    """
    Rescale the values of numpy array `arr` to be from `minv` to `maxv`.
    """
    # Get the max and min only considering the non-voided region (i.e., the real region)
    mina = np.min(arr*(1-mask))
    maxa = np.max(arr*(1-mask))
    if mina == maxa:
        return arr * minv
    # normalize the array first
    norm = (arr - mina) / (maxa - mina) 
    # rescale by minv and maxv, which is the normalized array by default 
    return (norm * (maxv - minv)) + minv  

def main():
    args = create_argparser().parse_args()

    args.use_conditional_model = True if (args.use_conditional_model=="True" or args.use_conditional_model==True) else False
    args.use_label_cond = True if (args.use_label_cond=="True" or args.use_label_cond==True) else False
    args.use_label_cond_dilated = True if (args.use_label_cond_dilated=="True" or args.use_label_cond_dilated==True) else False
    args.validation = True if (args.validation=="True" or args.validation==True) else False

    seed = args.seed
    args.devices = [th.cuda.current_device()]
    dist_util.setup_dist(devices=args.devices)
    print(f"Devices: {args.devices}")

    logger.configure()

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
    if args.use_label_cond:
        datal, ds = c_BraTSVolumes(directory=args.data_dir, 
                            batch_size=args.batch_size,
                            num_workers=int(args.num_workers), 
                            mode=args.mode,
                            img_size=args.image_size,
                            use_label_cond=args.use_label_cond,
                            use_label_cond_dilated=args.use_label_cond_dilated,
                            data_split_json=args.data_split_json,
                            validation=args.validation,
                            train_mode=args.train_mode,
                            beg_case=args.beg_case,
                            end_case=args.end_case).get_dl_ds()

        iterator_data = iter(datal)

    print("Starting inference...")

    for ind, batch in enumerate(iterator_data):
        try: 
            th.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            # print(f"Reseeded (in for loop) to {seed}")
            print(f"Doing number {ind} out of {args.num_samples // args.batch_size}")
            # TODO -> the seed should be the same? It might not have a great impact anyway
            seed += 1

            if args.use_label_cond==True and args.use_conditional_model==True:
                print("##############################")
                print(f"# Using use_label_cond: {args.use_label_cond} #")
                print(f"# Using use_conditional_model: {args.use_conditional_model} #")
                print("##############################")
                # Get a new batch
                #batch = next(iterator_data)

                # Get mask healthy to inpaint
                label_cond = batch["mask"].to(dist_util.dev())
                full_res_label_cond_dilated_th = None
                LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = dwt(label_cond)
                label_cond_dwt = th.cat([LLL / 3., LLH, LHL, LHH, HLL, HLH, HHL, HHH], dim=1)

                # Get real case to inpaint
                t1n_voided = batch["t1n_voided"].to(dist_util.dev())

                #t1n_voided = t1n * (1-label_cond)

                file_name = batch["t1n_voided_meta_dict"]["filename_or_obj"][0].split("/")[-2]


                # Create input with noise in region of interest
                noise = th.randn(args.batch_size, 1, args.image_size, args.image_size, args.image_size).to(dist_util.dev()) * label_cond
                noise_n_t1n = t1n_voided + noise
                LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = dwt(noise_n_t1n)
                # Noise only in the region of interest :D
                noise_dwt = th.cat([LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH], dim=1)  # Wavelet transformed noise
                
            elif args.use_label_cond==True and args.use_conditional_model==False:
                print("##############################")
                print(f"# Using use_label_cond: {args.use_label_cond} #")
                print(f"# Not using use_conditional_model: {args.use_conditional_model} #")
                print("##############################")

                # Get mask to inpaint (not usefull here as the model is not conditional)
                label_cond = batch["mask"].to(dist_util.dev())
                
                LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = dwt(label_cond)
                label_cond_dwt = th.cat([LLL / 3., LLH, LHL, LHH, HLL, HLH, HHL, HHH], dim=1)

                # Get real case
                #t1n = batch["t1n"].to(dist_util.dev())

                t1n_voided = batch["t1n_voided"].to(dist_util.dev()) #batch["t1n_voided"].to(dist_util.dev()) #t1n #* (1-label_cond) # TODO: the voided will be the one for the validation and testing
                
                file_name = batch["t1n_voided_meta_dict"]["filename_or_obj"][0].split("/")[-2]

                noise_dwt = th.randn(args.batch_size,         # Batch size
                        8,                       # 8 wavelet coefficients
                        args.image_size//2,      # Half spatial resolution (D)
                        args.image_size//2,      # Half spatial resolution (H)
                        args.image_size//2,      # Half spatial resolution (W)
                        ).to(dist_util.dev())

            else:
                # Just regular noise as input :D
                noise_dwt = th.randn(args.batch_size,         # Batch size
                        8,                       # 8 wavelet coefficients
                        args.image_size//2,      # Half spatial resolution (D)
                        args.image_size//2,      # Half spatial resolution (H)
                        args.image_size//2,      # Half spatial resolution (W)
                        ).to(dist_util.dev())

            model_kwargs = {}

            if args.validation:
                progress = True
            else:
                progress = True
            sample_fn = diffusion.p_sample_loop
            sample = sample_fn(model=model,
                            shape=noise_dwt.shape,
                            noise=noise_dwt,
                            time=args.diffusion_steps,
                            full_res_input=t1n_voided,
                            use_conditional_model=args.use_conditional_model,
                            label_cond_dwt=label_cond_dwt,
                            full_res_label_cond=label_cond,
                            full_res_label_cond_dilated=full_res_label_cond_dilated_th,
                            train_mode=args.train_mode,
                            clip_denoised=args.clip_denoised,
                            model_kwargs=model_kwargs,
                            progress=progress,
                            steps_scheduler=args.steps_scheduler,
                            )

            B, _, D, H, W = sample.size()
            
            sample = idwt(sample[:, 0, :, :, :].view(B, 1, D, H, W) * 3.,
                        sample[:, 1, :, :, :].view(B, 1, D, H, W),
                        sample[:, 2, :, :, :].view(B, 1, D, H, W),
                        sample[:, 3, :, :, :].view(B, 1, D, H, W),
                        sample[:, 4, :, :, :].view(B, 1, D, H, W),
                        sample[:, 5, :, :, :].view(B, 1, D, H, W),
                        sample[:, 6, :, :, :].view(B, 1, D, H, W),
                        sample[:, 7, :, :, :].view(B, 1, D, H, W))

            # Getting metadata
            target_t1n_voided_path = os.path.join("/projects/brats2023_a_f/BraTS2024_cluster/data/ASNR-MICCAI-BraTS2023-Local-Synthesis-Challenge-Training", file_name, f"{file_name}-t1n-voided.nii.gz")
            if not os.path.exists(target_t1n_voided_path):
                target_t1n_voided_path = os.path.join("/projects/brats2023_a_f/BraTS2024_cluster/data/ASNR-MICCAI-BraTS2023-Local-Synthesis-Challenge-Validation", file_name, f"{file_name}-t1n-voided.nii.gz")
            
            target_t1n_voided_data = nib.load(target_t1n_voided_path).get_fdata()
            affine, header = get_affine(scan_path=target_t1n_voided_path)

            # The new correction of intensity starts here #########################
            #########################
            # We need the label used for the prediction, so we don't use these voxels

            sample = (sample + 1) / 2. # the sample intensity will range [0,1]
            t1n_voided = (t1n_voided + 1) / 2 # variable with the real data (used as input of the model for prediction) with intensity ranging [0,1]

            t1n_voided_dilated = t1n_voided * (1-label_cond) # getting all real values, outside of the conditional label # full_res_label_cond_dilated_th before TODO
            sample_dilated = sample * (1-label_cond) # getting all sampled values, outside of the conditional label # full_res_label_cond_dilated_th before TODO

            non_zero_mask = ((1-label_cond)[0][0].cpu().numpy() != 0)  # full_res_label_cond_dilated_th before TODO

            roi = sample*label_cond # Region predicted

            final_prediction = t1n_voided.detach().clone()

            print(f"final_prediction: {final_prediction.max()}")
            print(f"final_prediction: {final_prediction.min()}")

            # Create a boolean mask where y is non-zero
            mask = label_cond != 0 # full_res_label_cond_dilated_th before TODO
            # Use the mask to assign values from p to x
            final_prediction[mask] = roi[mask]

            #########################
            ######################### Until here #########################

            ## Send everything to cpu and numpy
            final_prediction = final_prediction.detach().cpu().numpy()[0][0]
            label_cond = label_cond.detach().cpu().numpy()[0][0]
            roi = roi.detach().cpu().numpy()[0][0]
            sample = sample.detach().cpu().numpy()[0][0]
            
            # Setting folder to save files
            pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)
            
            output_name = os.path.join(args.output_dir, f'{file_name}-t1n-inference.nii.gz')
            final_prediction = rescale_array_numpy(arr=final_prediction, mask=label_cond, minv=target_t1n_voided_data.min(), maxv=target_t1n_voided_data.max()) 
            final_prediction = crop_to_240_240_155(tensor=final_prediction)
            final_prediction = np.flip(np.flip(final_prediction, axis=0), axis=1)
            img = nib.Nifti1Image(final_prediction, affine=affine, header=header)
            nib.save(img=img, filename=output_name)
            print(f'Saved to {output_name}')

            output_name = os.path.join(args.output_dir, f'{file_name}-sample.nii.gz')
            sample = crop_to_240_240_155(tensor=sample)
            sample = np.flip(np.flip(sample, axis=0), axis=1)
            img = nib.Nifti1Image(sample, affine=affine, header=header)
            nib.save(img=img, filename=output_name)
            print(f'Saved to {output_name}')

            output_name = os.path.join(args.output_dir, f'{file_name}-sample-roi.nii.gz')
            roi = crop_to_240_240_155(tensor=roi)
            roi = np.flip(np.flip(roi, axis=0), axis=1)
            img = nib.Nifti1Image(roi, affine=affine, header=header)
            nib.save(img=img, filename=output_name)
            print(f'Saved to {output_name}')

            output_name = os.path.join(args.output_dir, f'{file_name}_label.nii.gz')
            label_cond = crop_to_240_240_155(tensor=label_cond)
            label_cond = np.flip(np.flip(label_cond, axis=0), axis=1)
            label_cond = nib.Nifti1Image(label_cond, affine=affine, header=header)
            nib.save(img=label_cond, filename=output_name)
            print(f'Saved to {output_name}')

        except Exception as e:
            print("An error occurred:", e)
            print("INFERENCE FINISHED")
            break
            
    def copy_files_with_suffix(source_folder, destination_folder, suffix):
        # Create the destination folder if it doesn't exist
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)
            print(f"Created directory: {destination_folder}")
        else:
            print(f"Directory already exists: {destination_folder}")

        copied_files = []

        # Iterate over all files in the source folder
        for filename in os.listdir(source_folder):
            # Check if the file ends with the specified suffix
            if filename.endswith(suffix):
                source_file = os.path.join(source_folder, filename)
                destination_file = os.path.join(destination_folder, filename)
                # Copy the file to the destination folder
                shutil.copy2(source_file, destination_file)
                print(f"Copied: {source_file} to {destination_file}")
                copied_files.append(filename)
        # Zip all copied files
        zip_filename = os.path.join(destination_folder, f"{args.output_dir.split('/')[-1]}_submit.zip")
        with zipfile.ZipFile(zip_filename, 'w') as zipf:
            for file in copied_files:
                zipf.write(os.path.join(destination_folder, file), file)
        print(f"Zipped all copied files to: {zip_filename}")

    destination_folder = os.path.join(args.output_dir, f"{args.output_dir.split('/')[-1]}_submit")
    copy_files_with_suffix(source_folder=args.output_dir, destination_folder=destination_folder, suffix='t1n-inference.nii.gz')





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
        sampling_steps=None,
        model_path="",
        devices=[0],
        output_dir='./results',
        mode=None,
        renormalize=False,
        image_size=256,
        half_res_crop=False,
        concat_coords=False, # if true, add 3 (for 3d) or 2 (for 2d) to in_channels
        num_workers=None,
        data_split_json=None,
        use_conditional_model=None,
        validation=False,
        train_mode=None,
        beg_case=0,
        end_case="end",
        steps_scheduler=None
    )
    defaults.update({k:v for k, v in model_and_diffusion_defaults().items() if k not in defaults})
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
