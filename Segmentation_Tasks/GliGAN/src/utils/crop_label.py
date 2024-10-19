import math
import torch
from monai.config import KeysCollection
from monai.transforms.compose import MapTransform
from torch import clone as clone
import numpy as np
import random
from scipy import ndimage

class CropLabel(MapTransform):
    def __init__(self, keys: KeysCollection):
        super().__init__(keys)

    def __call__(self, data):
        d = dict(data)
        label = d["label"]
        _, max_x, max_y, max_z = label.shape
        label_crop = clone(label)

        x_extreme_dif = d["x_extreme_max"] - d["x_extreme_min"]
        y_extreme_dif = d["y_extreme_max"] - d["y_extreme_min"]
        z_extreme_dif = d["z_extreme_max"] - d["z_extreme_min"]

        x_pad = (96 - x_extreme_dif) / 2
        y_pad = (96 - y_extreme_dif) / 2
        z_pad = (96 - z_extreme_dif) / 2

        if x_pad < 0:
            C_x = -0.5
        else:
            C_x = 0.5

        if y_pad < 0:
            C_y = -0.5
        else:
            C_y = 0.5

        if z_pad < 0:
            C_z = -0.5
        else:
            C_z = 0.5

        x_base = d["x_extreme_min"] - int(x_pad)
        x_top = d["x_extreme_max"] + int(x_pad+C_x) 
        y_base = d["y_extreme_min"] - int(y_pad) 
        y_top = d["y_extreme_max"] + int(y_pad+C_y) 
        z_base = d["z_extreme_min"] - int(z_pad) 
        z_top = d["z_extreme_max"] + int(z_pad+C_z) 
        
        # Verifying the need for padding
        x_base_pad = 0
        y_base_pad = 0
        z_base_pad = 0
        x_top_pad = 0
        y_top_pad = 0
        z_top_pad = 0

        if x_base < 0:
            x_base_pad = -x_base
            x_base = 0
            
        if y_base < 0:
            y_base_pad = -y_base
            y_base = 0
            
        if z_base < 0:
            z_base_pad = -z_base
            z_base = 0
            
        if x_top > max_x:
            x_top_pad = x_top-max_x
            x_top = max_x
            
        if y_top > max_y:
            y_top_pad = y_top-max_y
            y_top = max_y
            
        if z_top > max_z:
            z_top_pad = z_top-max_z
            z_top = max_z
        ##################################
        # Crop the label
        label_crop = label_crop[:, x_base : x_top, y_base : y_top, z_base : z_top]
        
        
        label_crop_pad = clone(label_crop)
        label_crop_pad = np.pad(label_crop_pad, pad_width=((0,0), (x_base_pad,x_top_pad), (y_base_pad,y_top_pad), (z_base_pad,z_top_pad)), mode='constant', constant_values=(0, 0))
       

        d["label_crop_pad"] = label_crop_pad

        return d

    