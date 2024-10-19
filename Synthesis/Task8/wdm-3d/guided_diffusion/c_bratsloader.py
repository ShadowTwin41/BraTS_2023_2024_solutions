import torch
import torch.nn as nn
import torch.utils.data
import numpy as np
import os
import os.path
import nibabel
import json
import sys

from monai.data import load_decathlon_datalist, DataLoader, CacheDataset
import scipy
from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms.transform import MapTransform, Transform
from monai.config import IndexSelection, KeysCollection, SequenceStr
from collections.abc import Callable, Hashable, Mapping, Sequence
from monai.utils import TransformBackends

from monai.transforms import (
    Compose, 
    LoadImaged,
    EnsureChannelFirstd, 
    EnsureTyped,
    Orientationd,
    Resized,
    ScaleIntensityRanged, 
    ResizeWithPadOrCropd,
    RandFlipd,
    RandAffined,
    RandGaussianNoised,
    RandGaussianSharpend,
    RandAdjustContrastd,
    RandRotate90d,
    ScaleIntensityRangePercentilesd,
    CopyItemsd,
    ScaleIntensityd,
    )
from utils.data_loader_utils import QuantileAndScaleIntensityd

class c_BraTSVolumes(torch.utils.data.Dataset):
    def __init__(self, directory, batch_size, num_workers, mode=None, img_size=256, use_label_cond=None, use_label_cond_dilated=None, data_split_json=None, validation=False, testing=None, train_mode=None, beg_case=None, end_case=None):
        print(f"directory: {directory}")

        self.directory = directory
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mode = mode
        self.img_size = img_size
        self.use_label_cond = use_label_cond
        self.use_label_cond_dilated = use_label_cond_dilated
        self.data_split_json = data_split_json
        self.validation = validation
        self.train_mode = train_mode
        self.testing = testing
        self.beg_case = beg_case
        self.end_case = end_case

    def generate_detection_train_transform(self,
        image_size,
    ):
        """
        Generate training transform for the GAN.

        ARGS:
            image_size: final image size for resizing 

        RETURN:
            training transform for the WDM
        """


        if self.use_label_cond:
            print(f"USING LABEL COND")
            if self.train_mode=="Conditional_default":
                # In case of using a label as condition
                in_keys = ['t1n', 'mask_healthy']    #['t1n', 'mask_healthy', 't1n_voided', 'mask']  
                all_image_keys = ['t1n']        #['t1n', 't1n_voided'] 

            elif self.train_mode=="Conditional_always_known_only_healthy":
                # In case of using a label as condition
                in_keys = ['t1n', 'mask_healthy', 'mask_unhealthy']
                all_image_keys = ['t1n']

            else:
                in_keys = ['t1n', 'mask_healthy']
                all_image_keys = ['t1n']
        else:
            in_keys = ['t1n']
            all_image_keys = ['t1n']
        
        if self.validation or self.mode=="c_sample":
            print(f"DOING sampling")
            # In case of using a label as condition
            in_keys = ['t1n_voided', 'mask']
            all_image_keys = ['t1n_voided']

        
        compute_dtype = torch.float32
        # Define the input transforms and basic transforms with no change
        all_transforms = [
                    LoadImaged(keys=in_keys, meta_key_postfix="meta_dict", image_only=False),
                    EnsureChannelFirstd(keys=in_keys),
                    EnsureTyped(keys=in_keys, dtype=torch.float32),
                    Orientationd(keys=in_keys, axcodes="RAS", lazy=True,),
                    ResizeWithPadOrCropd(
                            keys=in_keys,
                            spatial_size=image_size,
                            mode="constant",
                            value=0,
                            lazy=True,
                        )]
        all_transforms.append(QuantileAndScaleIntensityd(keys=all_image_keys)) # a_min=-1, a_max=1)

        if self.use_label_cond and not self.validation:
            # In case of using a label as condition.
            # This is never used for validation
            if self.use_label_cond_dilated:
                print(f"use_label_cond_dilated: {self.use_label_cond_dilated}")
                # Creating a new tensor, similar with the mask healthy but a bit dilated
                # These two decide the size and shape of the dilation
                dilation_struct = scipy.ndimage.generate_binary_structure(3, 2)
                dil_factor = 3
                all_transforms.append(CopyItemsd(keys=['mask_healthy'], names=["mask_healthy_dilated"]))
                all_transforms.append(Resized(keys=['mask_healthy_dilated'], spatial_size=(128,128,128), mode="nearest", lazy=False))
                all_transforms.append(Dilation3Dd(keys=['mask_healthy_dilated'], dilation_struct=dilation_struct, dil_factor=dil_factor))
        # Adding out transforms
        all_transforms.append(EnsureTyped(keys=in_keys, dtype=compute_dtype))
        return Compose(all_transforms)
    

    def get_loader(self, directory, batch_size, num_workers, mode='train', img_size=256):
        """
        ARGS:
            directory: root directory for the dataset
            test_flag: Batch size
            
        RETURN:
            train_loader: data loader
            train_data: dict of the data loaded 
        """
        # Get train transforms
        transforms = self.generate_detection_train_transform(
                image_size = (img_size,img_size,img_size),
            )

        if self.testing:
            data_list_key = "testing"
        else:
            data_list_key = "training"
            
        # Get training data dict 
        data_set = load_decathlon_datalist(
                self.data_split_json,
                is_segmentation=True,
                data_list_key=data_list_key,
                base_dir=directory,
            )

        print(f"Training cases: {len(data_set)}")

        if self.validation or self.testing or self.mode=="c_sample":
            print(f"Shuffle: False")
            shuffle = False
            beg = int(self.beg_case)
            if self.end_case == "end":
                end = len(data_set)
            else:
                end = int(self.end_case)
        else:
            beg = 0
            end = len(data_set)
            print(f"Training with all data: beg={beg} end={end}")
            print(f"Shuffle: True")
            shuffle = True
        print(f"Training with data: beg={beg} end={end}")

        print(data_set[-1:])
        # Creating traing dataset
        ds = CacheDataset( 
            data=data_set[beg:end], 
            transform=transforms,
            cache_rate=0, 
            copy_cache=False,
            progress=True,
            num_workers=num_workers,
        )
        
        # Creating data loader
        dl = DataLoader(
            ds,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            shuffle=shuffle, 
            #collate_fn=no_collation,
        )
        print(f"Batch size: {batch_size}")
        return dl, ds

    def get_dl_ds(self):
        dl, ds = self.get_loader(self.directory, self.batch_size, self.num_workers, self.mode, self.img_size)
        return dl, ds

class Dilation3D(Transform):
    """
    Applies a convolution filter to the input image.

    Args:
        NONE

    dilation_struct = scipy.ndimage.generate_binary_structure(3, 2)
    out:
        [[[False  True False]
          [ True  True  True]
          [False  True False]]

          [[ True  True  True]
           [ True  True  True]
           [ True  True  True]]

          [[False  True False]
           [ True  True  True]
           [False  True False]]]
    
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, dilation_struct, dil_factor) -> None:
        self.dilation_struct = dilation_struct
        self.dil_factor = dil_factor

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Args:
            img: torch tensor data to apply filter to with shape: [channels, height, width[, depth]]

        Returns:
            A MetaTensor with the same shape as `img` and identical metadata and DILATED
        """
        
        ## Performing Dilation
        while len(img.shape)>3:
            img = img[0]

        gt_mat_dilation = scipy.ndimage.binary_dilation(input=img, structure=self.dilation_struct, iterations=self.dil_factor)
        gt_mat_dilation = np.expand_dims(gt_mat_dilation, axis=0)

        return gt_mat_dilation

class Dilation3Dd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.Dilation3D`.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: monai.transforms.MapTransform

        dilation_struct: 
            Matrix to apply in dilation

        dil_factor: 
            Number of iterations (basicaly how much it grows)
        
        allow_missing_keys:
            Don't raise exception if key is missing.
    """

    backend = Dilation3D.backend

    def __init__(self, keys: KeysCollection, dilation_struct, dil_factor, allow_missing_keys: bool = False, **kwargs,) -> None:
        super().__init__(keys, allow_missing_keys)
        self.filter = Dilation3D(dilation_struct, dil_factor)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.filter(d[key])
        return d

