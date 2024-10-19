import os
import torch
import numpy as np
from monai.data import CSVDataset, CacheDataset, DataLoader, Dataset, DistributedSampler, SmartCacheDataset, load_decathlon_datalist
from monai.data.utils import pad_list_data_collate
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    EnsureTyped,
    ScaleIntensityd,
    CopyItemsd,
    CropForegroundd,
    SpatialCropd,
    ToTensord,
    ResizeWithPadOrCropd,
)
import warnings
from src.utils.gaussian_noise_tumour_extended import GaussianNoiseTumourExtended
from src.utils.gaussian_noise_tumour import GaussianNoiseTumour

def get_loader(args): 
    NUM_WORKERS = int(args.num_workers)
    if args.csv_path == "":
        for file_name in os.listdir(f"../../Checkpoint/{args.logdir}"):
            if file_name.endswith("csv"):
                CSV_PATH = os.path.join(f"../../Checkpoint/{args.logdir}", file_name)
    else:
        CSV_PATH = args.csv_path
    print(f"CSV_PATH: {CSV_PATH}")
    
    if args.modality == "t1ce":
        scan_name = "scan_t1ce"
        col_names = ['scan_t1ce', 'label', 'center_x', 'center_y', 'center_z', 'x_extreme_min', 'x_extreme_max', 'y_extreme_min', 'y_extreme_max', 'z_extreme_min', 'z_extreme_max', 'x_size', 'y_size', 'z_size']
        col_types= {'center_x': {'type': int}, 'center_y': {'type': int}, 'center_z': {'type': int}, 'x_extreme_min': {'type': int}, 'x_extreme_max': {'type': int}, 'y_extreme_min': {'type': int}, 'y_extreme_max': {'type': int}, 'z_extreme_min': {'type': int}, 'z_extreme_max': {'type': int}, 'x_size': {'type': int}, 'y_size': {'type': int}, 'z_size': {'type': int}}
    elif args.modality == "t1":
        scan_name = "scan_t1"
        col_names = ['scan_t1', 'label', 'center_x', 'center_y', 'center_z', 'x_extreme_min', 'x_extreme_max', 'y_extreme_min', 'y_extreme_max', 'z_extreme_min', 'z_extreme_max', 'x_size', 'y_size', 'z_size']
        col_types= {'center_x': {'type': int}, 'center_y': {'type': int}, 'center_z': {'type': int}, 'x_extreme_min': {'type': int}, 'x_extreme_max': {'type': int}, 'y_extreme_min': {'type': int}, 'y_extreme_max': {'type': int}, 'z_extreme_min': {'type': int}, 'z_extreme_max': {'type': int}, 'x_size': {'type': int}, 'y_size': {'type': int}, 'z_size': {'type': int}}
    elif args.modality == "t2":
        scan_name = "scan_t2"
        col_names = ['scan_t2', 'label', 'center_x', 'center_y', 'center_z', 'x_extreme_min', 'x_extreme_max', 'y_extreme_min', 'y_extreme_max', 'z_extreme_min', 'z_extreme_max', 'x_size', 'y_size', 'z_size']
        col_types= {'center_x': {'type': int}, 'center_y': {'type': int}, 'center_z': {'type': int}, 'x_extreme_min': {'type': int}, 'x_extreme_max': {'type': int}, 'y_extreme_min': {'type': int}, 'y_extreme_max': {'type': int}, 'z_extreme_min': {'type': int}, 'z_extreme_max': {'type': int}, 'x_size': {'type': int}, 'y_size': {'type': int}, 'z_size': {'type': int}}
    elif args.modality == "flair":
        scan_name = "scan_flair"
        col_names = ['scan_flair', 'label', 'center_x', 'center_y', 'center_z', 'x_extreme_min', 'x_extreme_max', 'y_extreme_min', 'y_extreme_max', 'z_extreme_min', 'z_extreme_max', 'x_size', 'y_size', 'z_size']
        col_types= {'center_x': {'type': int}, 'center_y': {'type': int}, 'center_z': {'type': int}, 'x_extreme_min': {'type': int}, 'x_extreme_max': {'type': int}, 'y_extreme_min': {'type': int}, 'y_extreme_max': {'type': int}, 'z_extreme_min': {'type': int}, 'z_extreme_max': {'type': int}, 'x_size': {'type': int}, 'y_size': {'type': int}, 'z_size': {'type': int}}    
    print(f"Scan Modality: {scan_name}")


    if args.dataset=="BRATS_2023" or args.dataset=="BRATS_GOAT_2024":
        if args.dataset=="BRATS_2023":
            print(f"Using dataset: BRATS_2023")
        else:
             print(f"Using dataset: BRATS_GOAT_2024")
        from src.utils.convert_to_multi_channel_based_on_brats_classes import ConvertToMultiChannelBasedOnBratsGliomaClasses2023d as LABEL_TRANSFORM
        if int(args.in_channels)!=4:
            print("YOU WILL HAVE AN ERROR IN THE DATA LOADER. Change in_channels to 4")
    elif args.dataset=="BRATS_2024":
        print(f"Using dataset: BRATS_2024")
        from src.utils.convert_to_multi_channel_based_on_brats_classes import ConvertToMultiChannelBasedOnBratsGliomaPosTreatClasses2024d as LABEL_TRANSFORM
        if int(args.in_channels)!=5:
            print("YOU WILL HAVE AN ERROR IN THE DATA LOADER. Change in_channels to 5")
    elif args.dataset=="BRATS_2024_MENINGIOMA":
        print(f"Using dataset: BRATS_2024_MENINGIOMA")
        from src.utils.convert_to_multi_channel_based_on_brats_classes import ConvertToMultiChannelBasedOnBratsMeningiomaClasses2024d as LABEL_TRANSFORM
        if int(args.in_channels)!=2:
            print("YOU WILL HAVE AN ERROR IN THE DATA LOADER. Change in_channels to 2")
    else:
        raise ValueError("The dataset must be from BraTS: BRATS_GOAT_2024, BRATS_2024, BRATS_2023 or BRATS_2024_MENINGIOMA")

    if args.noise_type=="gaussian_extended":
        print("Using Gaussian noise with noise in the surrounding tissue")
        train_transforms = Compose(
                    [
                        LoadImaged(keys=[scan_name, 'label'], image_only=False),
                        EnsureChannelFirstd(keys=[scan_name, "label"]),
                        EnsureTyped(keys=[scan_name, "label"]),
                        # TODO uncomment if not found a solution around 
                        #ResizeWithPadOrCropd(  # TODO: In principle this is not need for the Brats2023 and BratsGOAT2024, however the Brats2024 glioma requires this (original shape 182, 218, 182)...
                        #    keys=[scan_name, 'label'],
                        #    spatial_size=(240,240,155),
                        #    mode="constant",
                        #    value=0,
                        #    lazy=False,
                        #),
                        LABEL_TRANSFORM(keys="label"), 
                        GaussianNoiseTumourExtended(keys=scan_name),
                        ToTensord(keys=[scan_name, 'scan_t1ce_crop', 'scan_t1ce_crop_pad', 'scan_t1ce_noisy', 'label', 'label_crop_pad', 'center_x', 'center_y', 'center_z', 'x_extreme_min', 'x_extreme_max', 'y_extreme_min', 'y_extreme_max', 'z_extreme_min', 'z_extreme_max', 'x_size', 'y_size', 'z_size']),
                    ]
                )
        
    elif args.noise_type=="gaussian_tumour":
        print("Using Gaussian noise only in the tumour zone")
        train_transforms = Compose(
                    [
                        LoadImaged(keys=[scan_name, 'label'], image_only=False),
                        EnsureChannelFirstd(keys=[scan_name, "label"]),
                        EnsureTyped(keys=[scan_name, "label"]),
                        # TODO uncomment if not found a solution around 
                        #ResizeWithPadOrCropd( # In principle this is not need for the Brats2023 and BratsGOAT2024, however the Brats2024 glioma requires this (original shape 182, 218, 182)...
                        #    keys=[scan_name, 'label'],
                        #    spatial_size=(240,240,155),
                        #    mode="constant",
                        #    value=0,
                        #    lazy=False,
                        #),
                        LABEL_TRANSFORM(keys="label"), 
                        GaussianNoiseTumour(keys=scan_name),
                        ToTensord(keys=[scan_name, 'scan_t1ce_crop', 'scan_t1ce_crop_pad', 'scan_t1ce_noisy', 'label', 'label_crop_pad', 'center_x', 'center_y', 'center_z', 'x_extreme_min', 'x_extreme_max', 'y_extreme_min', 'y_extreme_max', 'z_extreme_min', 'z_extreme_max', 'x_size', 'y_size', 'z_size']),
                    ]
                )
    
    # USING THE WHOLE DATASET
    train_CSVdataset = CSVDataset(src=CSV_PATH, col_names=col_names, col_types=col_types)
    print(f"Number of training images: {len(train_CSVdataset)}")
    warnings.warn(f"The data loader will load all labels to memory. In case it fails due to lack of memory, reduce the 'cache_rate' in the function 'get_loader()'.")
    
    train_ds = CacheDataset( 
        data=train_CSVdataset, 
        transform=train_transforms,
        cache_rate=1, 
        copy_cache=False,
        progress=True,
        num_workers=NUM_WORKERS,
    )
    # Using args.batch_size*2 so I can use distinct images for training the Generator and the Discriminator
    train_loader = DataLoader(train_ds, batch_size=int(args.batch_size*2), num_workers=NUM_WORKERS, drop_last=True, shuffle=True, collate_fn=pad_list_data_collate)
    print(f'Dataset training: number of batches: {len(train_loader)}')
    print("Leaving the data loader. Good luck!") 
    return train_loader