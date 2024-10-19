
import sys
sys.path.insert(1, '.')    
from utils.data_creator_utils import New_Voided, Label_Generator

import torch
import monai
from monai.data import CacheDataset, load_decathlon_datalist, DataLoader
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    EnsureTyped,
    CropForegroundd,
    Resized,
    ResizeWithPadOrCropd,
    RandAffined,
    RandFlipd,
    RandRotate90d,
    CropForegroundd,
    RandRotated,
    ToTensord,
)
import os
from os import listdir
from os.path import join
import argparse
import nibabel as nib
import numpy as np
import random
import json

def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True  # To ensure deterministic results
    torch.backends.cudnn.benchmark = False  # May slow down performance

set_seed(1)

def get_data_loader(DATA_LIST_FILE_PATH, DATA_LIST_KEY, DATA_DIR, NUM_WORKERS, BATCH_SIZE, begin, end):
    # Creating the basic data loader
    train_transforms = Compose([LoadImaged(keys=['mask_healthy', 'mask_unhealthy', 'mask', 't1n_voided', 't1n'], image_only=False),
                            EnsureChannelFirstd(keys=['mask_healthy', 'mask_unhealthy', 'mask', 't1n_voided', 't1n']),
                            EnsureTyped(keys=['mask_healthy', 'mask_unhealthy', 'mask', 't1n_voided', 't1n']),
                            ToTensord(keys=['mask_healthy', 'mask_unhealthy', 'mask', 't1n_voided', 't1n'], dtype="float32")
                        ])
        
    # Get training data dict 
    all_data = load_decathlon_datalist(
            DATA_LIST_FILE_PATH,
            is_segmentation=True,
            data_list_key=DATA_LIST_KEY,
            base_dir=DATA_DIR,
        )

    # Creating traing dataset
    print(all_data[:1])
    try:
        if end.lower()=='end':
            end = len(all_data)
    except:
        if int(end) > len(all_data):
            end = len(all_data)
    
    train_ds = CacheDataset( 
        data=all_data[begin:end],
        transform=train_transforms,
        cache_rate=0,
        copy_cache=False,
        progress=True, 
        num_workers=NUM_WORKERS, 
    )

    # Creating data loader
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        shuffle=False,
        #collate_fn=no_collation,
    )
    print(f"Start: {begin} -> Finish: {end}")
    print(f"Number of cases in train_loader: {len(train_loader)}")
    print(f"Batch size: {BATCH_SIZE}")
    return train_loader

def get_seg_loader(DATA_LIST_FILE_PATH, DATA_LIST_KEY, DATA_DIR, NUM_WORKERS, BATCH_SIZE):
    # Creating the basic data loader
    train_transforms = Compose([LoadImaged(keys=['mask_healthy'], image_only=False),
                            EnsureChannelFirstd(keys=['mask_healthy']),
                            EnsureTyped(keys=['mask_healthy']),
                            RandFlipd(keys=['mask_healthy'], prob=0.5, lazy=True),
                            RandRotated(keys=['mask_healthy'], prob=0.5, range_x=np.radians(360), range_y=np.radians(360), range_z=np.radians(360), mode='nearest', lazy=True),
                            CropForegroundd(keys=['mask_healthy'], source_key='mask_healthy', lazy=True),
                            ToTensord(keys=['mask_healthy'], dtype="float32")
                        ])
        
    # Get training data dict 
    all_data = load_decathlon_datalist(
            DATA_LIST_FILE_PATH,
            is_segmentation=True,
            data_list_key=DATA_LIST_KEY,
            base_dir=DATA_DIR,
        )


    # Creating traing dataset
    print(all_data[:1])
    train_ds = CacheDataset( 
        data=all_data[:],
        transform=train_transforms,
        cache_rate=0,
        copy_cache=False,
        progress=True, 
        num_workers=NUM_WORKERS, 
    )

    # Creating data loader
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        shuffle=True,
        #collate_fn=no_collation,
    )
    print(f"Number of cases in seg_loader: {len(train_loader)}")
    print(f"Batch size: {BATCH_SIZE}")
    return train_loader


def get_affine(scan_path):
    '''
    Get the metada from the nifti file
    '''
    scan = nib.load(scan_path)
    header = scan.header
    affine = scan.affine
    return affine, header

def save_nii(image, scan_path=None, save=False):
    '''
    Show the scans in the 3 axis, in the chosen slices or in the center of mass.
        Parameters:
                image (pytorch tensor): Pytorch tensor of the scan to save
                save (str) : path to save the scan as .nii 
        Returns:
                None
    '''
    affine, header =  get_affine(scan_path)
    feat = np.squeeze((image).data.numpy())
    feat = nib.Nifti1Image(feat, affine=affine, header=header)
    nib.save(feat, save)

def main_n_new_tumours(model_path, train_loader, save_folder, only_real_seg, only_fake_seg, new_cases_per_case, begin, end):
    print(f"only_real_seg: {only_real_seg}")
    print(f"only_fake_seg: {only_fake_seg}")
    # option and prob to add a number of tumours
    options = [1, 2, 3]
    weights = [0.45, 0.45, 0.1]
    # Dict with all informations of each individual fake case
    all_info_dict = {} 
    obj_label_gen = Label_Generator(path=model_path, device="cpu")
    Label_G = obj_label_gen.load_label_generator()
    new_voided_creator = New_Voided(device="cpu")
    
    for case_number, batch in enumerate(train_loader):
        mask_unhealthy = batch['mask_unhealthy']
        t1n = batch['t1n']
        t1n_complete_path = batch['t1n_meta_dict']['filename_or_obj'][0]
        mask_unhealth_complete_path = batch['mask_unhealthy_meta_dict']['filename_or_obj'][0]
        name = t1n_complete_path.split("/")[-2]
        print(f"CASE: {case_number}: {name}")
        for i in range(new_cases_per_case):
            info_per_case_dict = {}
            GIVE_UP = False
            correct_new_seg = mask_unhealthy[0][0] # old seg, not the complete yet!
            # Select a number random of tumours to insert!
            n_tumours = int(random.choices(options, weights)[0])
            info_per_case_dict["n_tumours"] = n_tumours
            for n_tumour in range(n_tumours):
                print(f"Tumour number: {n_tumour}")
                counter = 0
                while True:
                    # Decide what seg to use
                    if only_real_seg:
                        fake_seg = False
                    if only_fake_seg:
                        fake_seg=True
                    if (not only_real_seg and not only_fake_seg) or (only_real_seg and only_fake_seg):
                        fake_seg = random.choice([True, False])
                    if fake_seg:
                        # Generate a fake seg from the label generator
                        fake_WT_label = obj_label_gen.get_WT_label(Label_G=Label_G, Th=0.5) # Create a new fake label (new seg)
                    else:
                        #Get random label from the dataset
                        fake_WT_label = torch.zeros(1)
                        while torch.sum(fake_WT_label)==0:
                            try:
                                next_fake_WT_label = next(iter_seg_loader)
                                fake_WT_label = next_fake_WT_label['mask_healthy']
                            except:
                                print("New iterator iter_seg_loader")
                                iter_seg_loader = iter(seg_loader)
                                next_fake_WT_label = next(iter_seg_loader)
                                fake_WT_label = next_fake_WT_label['mask_healthy']

                    new_seg = fake_WT_label[0][0] # new seg
                    if torch.sum(new_seg)==0:
                        print(next_fake_WT_label["mask_healthy_meta_dict"]['filename_or_obj'][0].split("/")[-2])
                        print(f"torch.sum(new_seg): {torch.sum(new_seg)}")
                        GIVE_UP = True
                        break
                    counter += 1
                    # Testing patient
                    if counter==10:
                        print(f"Try number: {counter}")
                    if counter==20:
                        GIVE_UP = True
                        break
                    # Check if the correct_new_seg has only three dimentions for processing
                    if len(correct_new_seg.shape) == 5:
                        correct_new_seg = correct_new_seg[0][0].numpy()
                        
                    if n_tumour+1==n_tumours:
                        unhealthy_seg = mask_unhealthy[0][0]
                    else:
                        unhealthy_seg = None
                    # Creating new case
                    SUCCESS, correct_new_seg = new_voided_creator.get_complete_seg(new_seg=new_seg, old_seg=correct_new_seg, image=t1n[0][0])
    
                    if SUCCESS:
                        # Save if a fake_seg or real_seg was used
                        info_per_case_dict[f"fake_seg_{n_tumour}"] = fake_seg
                        # Creating voided volume
                        voided_volume, correct_new_seg = new_voided_creator.get_new_voided(complete_seg=correct_new_seg, image=t1n[0][0], old_seg=unhealthy_seg)
                        break
                if GIVE_UP:
                    break
                
            if GIVE_UP:
                info_per_case_dict["GIVE_UP"] = True
                print(f"Case {i} GAVE UP!")
            else:
                info_per_case_dict["GIVE_UP"] = False
                save_nii(image=voided_volume, scan_path=t1n_complete_path, save=join(save_folder, f"{name}-t1n-voided-fake_{only_real_seg}_{only_fake_seg}_{i}.nii.gz"))
                save_nii(image=correct_new_seg, scan_path=mask_unhealth_complete_path, save=join(save_folder, f"{name}-mask-healthy-fake_{only_real_seg}_{only_fake_seg}_{i}.nii.gz"))
                print(f"DONE: {i}")

            # Add info into the dict
            all_info_dict[f"{name}-t1n-voided-fake_{only_real_seg}_{only_fake_seg}_{i}.nii.gz"] = info_per_case_dict
        
        print("#######################")
    # Save the dictionary to a JSON file with an indentation of 4 spaces
    os.makedirs("./metadata", exist_ok=True)
    with open(f"./metadata/begin_{begin}__end_{end}__only_real_seg_{only_real_seg}__only_fake_seg_{only_fake_seg}.json", "w") as json_file:
        json.dump(all_info_dict, json_file, indent=4)
   

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segmentation inference")
    parser.add_argument("--data_dir", type=str, help="Path to original dataset, e.g., ../../DataSet/ASNR-MICCAI-BraTS2023-Local-Synthesis-Challenge-Training")
    parser.add_argument("--json_file", type=str, help="Path to the json file, e.g., ./ASNR-MICCAI-BraTS2023-Local-Synthesis-Challenge-Training.json")
    parser.add_argument("--save_folder", type=str, help="Path to save the new dataset, e.g., ./New_DataSet")
    parser.add_argument("--model_path", type=str, help="Path to the model to be used for inference. E.g., ./weights/LabelGenerator.pth")
    parser.add_argument("--only_real_seg", type=str, help="If True, uses only real segmentations")
    parser.add_argument("--only_fake_seg", type=str, help="If True, uses only fake segmentations")
    parser.add_argument("--begin", type=str, help="Begin case")
    parser.add_argument("--end", type=str, help="End case. Write end to select all.")
    parser.add_argument("--new_cases_per_case", type=str, help="Number of new cases per case.")
    
    args = parser.parse_args()

    ### Data loaders
    DATA_LIST_KEY = 'training'
    NUM_WORKERS = 4
    BATCH_SIZE = 1

    print("Loading train_loader")
    train_loader = get_data_loader(args.json_file, DATA_LIST_KEY, args.data_dir, NUM_WORKERS, BATCH_SIZE, int(args.begin), args.end)
    seg_loader = get_seg_loader(args.json_file, DATA_LIST_KEY, args.data_dir, NUM_WORKERS, BATCH_SIZE)
    iter_seg_loader = iter(seg_loader)
    print("Done loading train_loader")
    
    if args.only_real_seg == "True":
        only_real_seg = True
    else:
        only_real_seg = False

    if args.only_fake_seg == "True":
        only_fake_seg = True
    else:
        only_fake_seg = False

    os.makedirs(args.save_folder, exist_ok=True)
    main_n_new_tumours(
        model_path = args.model_path,
        train_loader = train_loader, 
        save_folder = args.save_folder, 
        only_real_seg = only_real_seg, 
        only_fake_seg = only_fake_seg,
        new_cases_per_case = int(args.new_cases_per_case),
        begin=args.begin,
        end=args.end)

