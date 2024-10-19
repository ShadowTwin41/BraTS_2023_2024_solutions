import os
import torch
from os import listdir
from os.path import join
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
import json

sys.path.append("../wdm-3d/") 

from guided_diffusion.c_bratsloader import c_BraTSVolumes
from guided_diffusion import (dist_util,
                              logger)
from guided_diffusion.script_util import (model_and_diffusion_defaults,
                                          create_model_and_diffusion,
                                          add_dict_to_argparser,
                                          args_to_dict,
                                          )
from DWT_IDWT.DWT_IDWT_layer import DWT_3D, IDWT_3D


def crop_to_240_240_155(tensor):
    """
    Crops the tensor back to the opriginal size of 240x240x155
    """
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

def rescale_array_numpy_masked(arr, mask, minv, maxv): #monai function adapted
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

def rescale_array_numpy(arr, minv, maxv): #monai function adapted
    """
    Rescale the values of numpy array `arr` to be from `minv` to `maxv`.
    """
    # Get the max and min only considering the non-voided region (i.e., the real region)
    mina = np.min(arr)
    maxa = np.max(arr)
    if mina == maxa:
        return arr * minv
    # normalize the array first
    norm = (arr - mina) / (maxa - mina) 
    # rescale by minv and maxv, which is the normalized array by default 
    return (norm * (maxv - minv)) + minv 

def apply_linear_correction(
        sample_path,
        mask_path,
        target_t1n_voided_path):
    """
    Computes the linear equation top match the output of the model 
    with the original scan to correct the roi.

    IN:
        sample_path: output of the model, normalised between 0 and 1
        mask_path: path to the mask of the ROI
        target_t1n_voided_path: original voided scan
    OUT:
        corrected_inference: inference linearly corrected
    """
    pred_image = nib.load(sample_path).get_fdata()
    label_image = nib.load(mask_path).get_fdata()
    real_voided_case_image = nib.load(target_t1n_voided_path).get_fdata()

    real_voided_case_image_norm = rescale_array_numpy(arr=real_voided_case_image, minv=0, maxv=1)
    pred_image = rescale_array_numpy(arr=pred_image, minv=0, maxv=1)
    
    pred_t1_voided_not_roi = pred_image * (1-label_image)
    real_t1_voided_not_roi = real_voided_case_image_norm * (1-label_image)
    non_zero_mask = ((1-label_image) != 0)
    
    x = pred_t1_voided_not_roi[non_zero_mask]
    y = real_t1_voided_not_roi[non_zero_mask]
    a = np.sum(x * y) / np.sum(x * x)
    print(f"Linear equation: y = {a} * x")

    roi = pred_image*label_image # Region predicted
    roi = roi*a # correct with linear equation

    final_prediction = np.copy(real_voided_case_image_norm)
    mask = label_image != 0 
    # Use the mask to assign values from p to x
    final_prediction[mask] = roi[mask]

    final_prediction = rescale_array_numpy_masked(arr=final_prediction, mask=label_image, minv=real_voided_case_image.min(), maxv=real_voided_case_image.max())

    return final_prediction


def create_data_split(data_path):
    # The data will be in "data_path". In the format:
    # BraTS-GLI-00020-000
    #    BraTS-GLI-00020-000-mask.nii.gz
    #    BraTS-GLI-00020-000-t1n-voided.nii.gz
    """
    Json entry example
    {
        "testing": [
            {
                "mask": "BraTS-GLI-01274-000/BraTS-GLI-01274-000-mask.nii.gz",
                "t1n_voided": "BraTS-GLI-01274-000/BraTS-GLI-01274-000-t1n-voided.nii.gz",
            }
        ]
    }
    """
    training_list = []

    cases_folder_L = listdir(data_path)
    dataset_name = data_path.split("/")[-1]
    cases_folder_L.sort()
    already_done = f"./predictions/{dataset_name}"
    os.makedirs(already_done, exist_ok=True)
    already_done = listdir(already_done)

    for case_folder in cases_folder_L:
        dict_entry = {}
        # T1n-voided
        t1n_voided_path = join(case_folder, f"{case_folder}-t1n-voided.nii.gz")
        dict_entry['t1n_voided'] = t1n_voided_path
        # Mask
        mask_path = join(case_folder, f"{case_folder}-mask.nii.gz")
        dict_entry['mask'] = mask_path

        if f"{case_folder}-t1n-inference.nii.gz" in already_done:
            print(f"Case {case_folder} already done.")
        else:
            training_list.append(dict_entry)
    
    
    final_json = {"testing":training_list}
    with open(f'./DataSet/{dataset_name}.json', 'w') as json_file:
        json.dump(final_json, json_file, indent=4)

def main(data_path, output_path, model_path, beg_case, end_case):
    print(f"Doing from case {sys.argv[1]} until case {sys.argv[2]}")
    ## Defining IMPORTANT args
    use_conditional_model = True
    use_label_cond = True
    use_label_cond_dilated = False
    validation = False
    testing = True
    sampling_steps = 5000 
    diffusion_steps = int(sampling_steps)
    progress = True
    in_channels = 16
    out_channels = 8
    noise_schedule = "linear"
    steps_scheduler = "linear"
    mode = "c_sample"
    train_mode = "Conditional_always_known_only_healthy"
    batch_size = 1
    num_workers = 2
    image_size = 256
    dataset_name = data_path.split("/")[-1]
    data_split_json =  f'./DataSet/{dataset_name}.json'
    num_channels = 64 
   
    # Fixed args
    class_cond=False
    num_res_blocks=2
    num_heads=1
    learn_sigma=False
    use_scale_shift_norm=False
    channel_mult='1,2,2,4,4,4'
    rescale_learned_sigmas=False
    rescale_timesteps=False
    dims=3
    num_groups=32
    bottleneck_attention=False
    resample_2d=False
    renormalize=True
    additive_skips=True
    use_freq=False
    predict_xstart=True
    clip_denoised=True
    attention_resolutions = ''
    dataset = 'c_brats'
    
    ## Other args
    data_dir = data_path
    output_dir = join(output_path, dataset_name)
    use_fp16=False
    use_ddim=False

    # Define seed
    seed = 42
    seed = seed

    # Define Device
    devices = [th.cuda.current_device()]
    dist_util.setup_dist(devices=devices)
    print(f"Devices: {devices}")

    print("Creating data splits json")
    create_data_split(data_path=data_dir)
    print("Creating model and diffusion...")
    
    args_in_a_dict = {
        'image_size': image_size, 
        'num_channels': num_channels, 
        'num_res_blocks': num_res_blocks, 
        'num_heads': num_heads, 
        'num_heads_upsample': -1, 
        'num_head_channels': -1, 
        'attention_resolutions': attention_resolutions, 
        'channel_mult': channel_mult, 
        'dropout': 0.0, 
        'class_cond': class_cond, 
        'use_checkpoint': False, 
        'use_scale_shift_norm': use_scale_shift_norm, 
        'resblock_updown': True, 
        'use_fp16': use_fp16, 
        'use_new_attention_order': False, 
        'dims': dims, 
        'num_groups': num_groups, 
        'in_channels': in_channels, 
        'out_channels': out_channels, 
        'bottleneck_attention': bottleneck_attention, 
        'resample_2d': resample_2d, 
        'additive_skips': additive_skips, 
        'mode': mode, 
        'use_freq': use_freq, 
        'predict_xstart': predict_xstart, 
        'use_label_cond': use_label_cond, 
        'use_label_cond_dilated': use_label_cond_dilated, 
        'use_conditional_model': use_conditional_model, 
        'learn_sigma': learn_sigma, 
        'diffusion_steps': diffusion_steps, 
        'noise_schedule': noise_schedule, 
        'timestep_respacing': '', 
        'use_kl': False, 
        'rescale_timesteps': rescale_timesteps, 
        'rescale_learned_sigmas': rescale_learned_sigmas, 
        'dataset': dataset
    }
    
    model, diffusion = create_model_and_diffusion(**args_in_a_dict)
    print("Load model from: {}".format(model_path))

    model.load_state_dict(dist_util.load_state_dict(model_path, map_location="cpu"))
    model.to(dist_util.dev([0, 1]) if len(devices) > 1 else dist_util.dev())  # allow for 2 devices

    model.eval()
    idwt = IDWT_3D("haar")
    dwt = DWT_3D('haar')

    datal, ds = c_BraTSVolumes(
        directory=data_dir, 
        batch_size=batch_size,
        num_workers=int(num_workers), 
        mode=mode,
        img_size=image_size,
        use_label_cond=use_label_cond,
        data_split_json=data_split_json,
        validation=validation,
        testing=testing,
        train_mode=train_mode,
        beg_case=beg_case,
        end_case=end_case
        ).get_dl_ds()


    if end_case=="end":
        end_case = len(ds)
    else:
        end_case = int(end_case)
    iterator_data = iter(datal)

    print("Starting inference...")
    inter_output_dir = f"./iter_output/{dataset_name}" 
    pathlib.Path(inter_output_dir).mkdir(parents=True, exist_ok=True)

    for ind, batch in enumerate(iterator_data):

        th.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        print(f"Doing number {ind} out of {len(iterator_data)}")
        seed += 1 

        print("##############################")
        print(f"# Using use_label_cond: {use_label_cond} #")
        print(f"# Using use_conditional_model: {use_conditional_model} #")
        print("##############################")

        # Get mask healthy to inpaint
        label_cond = batch["mask"].to(dist_util.dev())

        # Get real case to inpaint
        t1n_voided = batch["t1n_voided"].to(dist_util.dev())

        file_name = batch["t1n_voided_meta_dict"]["filename_or_obj"][0].split("/")[-2]

        # Create input with noise in region of interest
        noise = th.randn(batch_size, 1, image_size, image_size, image_size).to(dist_util.dev()) * label_cond
        noise_n_t1n = t1n_voided + noise
        LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = dwt(noise_n_t1n)
        # Noise only in the region of interest :D
        noise_dwt = th.cat([LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH], dim=1)  # Wavelet transformed noise
        
        LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = dwt(label_cond)
        label_cond_dwt = th.cat([LLL / 3., LLH, LHL, LHH, HLL, HLH, HHL, HHH], dim=1)
        
        model_kwargs = {}

        sample_fn = diffusion.p_sample_loop
        sample = sample_fn(model=model,
                        shape=noise_dwt.shape,
                        noise=noise_dwt,
                        time=diffusion_steps,
                        full_res_input=t1n_voided,
                        use_conditional_model=use_conditional_model,
                        label_cond_dwt=label_cond_dwt,
                        full_res_label_cond=label_cond,
                        full_res_label_cond_dilated=None,
                        train_mode=train_mode,
                        clip_denoised=clip_denoised,
                        model_kwargs=model_kwargs,
                        progress=progress,
                        steps_scheduler=steps_scheduler,
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
        target_t1n_voided_path = os.path.join(data_dir, file_name, f"{file_name}-t1n-voided.nii.gz")
        target_t1n_voided_data = nib.load(target_t1n_voided_path).get_fdata()
        affine, header = get_affine(scan_path=target_t1n_voided_path)

        # The new correction of intensity starts here #########################
        #########################
        # We need the label used for the prediction, so we don't use these voxels

        sample = (sample + 1) / 2. # the sample intensity will range [0,1]
        t1n_voided = (t1n_voided + 1) / 2. # variable with the real data (used as input of the model for prediction) with intensity ranging [0,1]

        sample = sample[0][0].cpu().numpy()
        t1n_voided = t1n_voided[0][0].cpu().numpy()
        label_cond = label_cond[0][0].cpu().numpy()

        roi = sample*label_cond # Region predicted
        
        final_prediction = np.copy(t1n_voided)

        # Create a boolean mask where y is non-zero
        mask = label_cond != 0 
        # Use the mask to assign values from roi to final_prediction
        final_prediction[mask] = roi[mask]

        #########################
        ######################### Until here #########################
        
        # Setting folder to save files

        inter_output_name_sample = os.path.join(inter_output_dir, f'{file_name}-sample.nii.gz')
        sample = crop_to_240_240_155(tensor=sample)
        sample = np.flip(np.flip(sample, axis=0), axis=1)
        img = nib.Nifti1Image(sample, affine=affine, header=header)
        nib.save(img=img, filename=inter_output_name_sample)
        print(f'Saved to {inter_output_name_sample}')
            

        # Perform linear correction
        mask_path = os.path.join(data_dir, file_name, f"{file_name}-mask.nii.gz")
        final_prediction_corrected = apply_linear_correction(
            sample_path = inter_output_name_sample,
            mask_path = mask_path,
            target_t1n_voided_path = target_t1n_voided_path
            )
        
        output_name_corrected = os.path.join(output_dir, f'{file_name}-t1n-inference.nii.gz')
        img = nib.Nifti1Image(final_prediction_corrected, affine=affine, header=header)
        nib.save(img=img, filename=output_name_corrected)
        print(f'Final inference corrected saved to {output_name_corrected}')
    print("INFERENCE FINISHED")
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segmentation inference")
    parser.add_argument("--data_path", type=str, help="Path with the raw cases, following the BraTS 2024 Inpaint challenge")
    parser.add_argument("--output_path", type=str, help="Path to save the predictions")
    parser.add_argument("--model_path", type=str, help="Path to the model to be used for inference. E.g., ../wdm-3d/runs/Conditional_always_known_only_healthy_31_7_2024_12:35:11/checkpoints/c_brats_3000000.pt")
    parser.add_argument("--beg_case", type=int, default=0, help="Case to start. Set 0 to start from begining.")
    parser.add_argument("--end_case", type=str, default="end", help="Case to finish. Set end to select all cases.")
    args = parser.parse_args()
    main(data_path=args.data_path, output_path=args.output_path, model_path=args.model_path, beg_case=args.beg_case, end_case=args.end_case)

