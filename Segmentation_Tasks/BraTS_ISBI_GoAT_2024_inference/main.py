import os
import argparse
import torch
from os import listdir
from os.path import join
from infer_low_disk import convert_data_step, perform_inference_step, thresholding_step, convert_back_BraTS_step

def infer(
    data_path,
    output_path,
    nnUNet_results
):
    
    # Setting the path for the weights
    os.environ['nnUNet_results'] = nnUNet_results
    #os.environ['nnUNet_preprocessed'] = 'NONE'
    #os.environ['nnUNet_raw'] = 'NONE'

    if torch.cuda.is_available():
        print("GPU is available")
    else:
        print("GPU is not available")

    # First step - Convert dataset from BraTS2023 to nnUNet format
    print("Doing first step: convert_data_step")
    input_folder_nnunet = './converted_dataset/'
    convert_data_step(input_folder_nnunet=input_folder_nnunet, raw_dataset=data_path)
    print(f"Number of files in input_folder_nnunet: {len(listdir(input_folder_nnunet))}")

    # Second step - Performing inference for each model
    print("Doing second step: perform_inference_step")
    #ensemble_code = 'rGB_rGL_rGS'
    ensemble_code = 'rGB_rGS_rGL'
    inference_folder =  "./inference/"
    perform_inference_step(inference_folder=inference_folder, input_folder_nnunet=input_folder_nnunet, ensemble_code=ensemble_code)
    print("________________________ Second step completed ________________________")
    
    # Third step - Doing the ensemble
    print("Doing third step")
    print("Step assemble already done before. skipping") # This step is ignored in the low disk script

    # Fourth step - Thresholding
    print("Doing fourth step: thresholding_step")
    min_volume_threshold_WT = 250
    min_volume_threshold_TC = 150
    min_volume_threshold_ET = 100
    thresholding_step(
        min_volume_threshold_WT=min_volume_threshold_WT, 
        min_volume_threshold_TC=min_volume_threshold_TC, 
        min_volume_threshold_ET=min_volume_threshold_ET,
        inference_folder=inference_folder,
        ensemble_code=ensemble_code
        )

    # Fifth step - Converting back from nnUnet to BraTS2023
    print("Doing fifth step: convert_back_BraTS_step")
    convert_back_BraTS_step(
        min_volume_threshold_WT=min_volume_threshold_WT, 
        min_volume_threshold_TC=min_volume_threshold_TC, 
        min_volume_threshold_ET=min_volume_threshold_ET, 
        inference_folder=inference_folder, 
        ensemble_code=ensemble_code,
        brats_final_inference=output_path
        )
    
    print("DONE")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segmentation inference")
    parser.add_argument("--data_path", type=str, help="Path with the raw cases, following the BraTS 2024-ISBI GoAT challenge")
    parser.add_argument("--output_path", type=str, help="Path to save the predictions")
    parser.add_argument("--nnUNet_results", type=str, help="Path to the results of the nnUNet training")
    args = parser.parse_args()
    infer(data_path=args.data_path,output_path=args.output_path, nnUNet_results=args.nnUNet_results)

