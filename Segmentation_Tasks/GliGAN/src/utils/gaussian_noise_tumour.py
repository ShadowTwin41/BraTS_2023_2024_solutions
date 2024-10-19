import math
import torch
from monai.config import KeysCollection
from monai.transforms.compose import MapTransform
from torch import clone as clone
import numpy as np
import random

class GaussianNoiseTumour(MapTransform):
    def __init__(self, keys: KeysCollection):
        super().__init__(keys)
        self.keys = keys
    def __call__(self, data):
        d = dict(data)
        scan_t1ce = d[self.keys]
        _, max_x, max_y, max_z = scan_t1ce.shape
        scan_t1ce_crop = clone(scan_t1ce)
        label = d["label"]
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
        
        # Crop and Normalise the scan
        scan_t1ce_crop = scan_t1ce_crop[:, x_base : x_top, y_base : y_top, z_base : z_top]
        scan_t1ce_crop = self.rescale_array(arr=scan_t1ce_crop, minv=-1, maxv=1)
        d["scan_t1ce_crop"] = scan_t1ce_crop
        
        # Scan and label with padding for 96, 96, 96 (if needed)
        scan_t1ce_crop_pad = clone(scan_t1ce_crop)
        scan_t1ce_crop_pad = np.pad(scan_t1ce_crop_pad, pad_width=((0,0), (x_base_pad,x_top_pad), (y_base_pad,y_top_pad), (z_base_pad,z_top_pad)), mode='constant', constant_values=(-1, -1))
        label_crop_pad = clone(label_crop)
        label_crop_pad = np.pad(label_crop_pad, pad_width=((0,0), (x_base_pad,x_top_pad), (y_base_pad,y_top_pad), (z_base_pad,z_top_pad)), mode='constant', constant_values=(0, 0))

        # Computing the noise size
        max_size = max(d['x_size'], d["y_size"], d["z_size"])
        exp_base = self.norm_exp_base(value=max_size)
        scan_t1ce_noisy = self.add_gaussian_noise_tumour(scan=scan_t1ce_crop_pad, label=label_crop_pad, exp_base=exp_base)
        scan_t1ce_noisy = self.rescale_array_numpy(arr=scan_t1ce_noisy, minv=-1, maxv=1)

        d[self.keys] = scan_t1ce
        d["scan_t1ce_crop_pad"] = scan_t1ce_crop_pad
        d["scan_t1ce_noisy"] = scan_t1ce_noisy
        d["label_crop"] = label_crop  
        d["label_crop_pad"] = label_crop_pad

        return d

    def rescale_array(self, arr, minv, maxv): #monai function adapted
        """
        Rescale the values of numpy array `arr` to be from `minv` to `maxv`.
        """
        mina = torch.min(arr)
        maxa = torch.max(arr)
        if mina == maxa:
            return arr * minv
        # normalize the array first
        norm = (arr - mina) / (maxa - mina) 
        # rescale by minv and maxv, which is the normalized array by default 
        return (norm * (maxv - minv)) + minv  

    def rescale_array_numpy(self, arr, minv, maxv): #monai function adapted
        """
        Rescale the values of numpy array `arr` to be from `minv` to `maxv`.
        """
        mina = np.min(arr)
        maxa = np.max(arr)
        if mina == maxa:
            return arr * minv
        # normalize the array first
        norm = (arr - mina) / (maxa - mina) 
        # rescale by minv and maxv, which is the normalized array by default 
        return (norm * (maxv - minv)) + minv  
    
    def distance_3d(self, point1, point2):
        """
        Compute the distance between two points
        Parameters:
                point1 (tuple): Point 1 coordinates
                point2 (tuple): Point 2 coordinates
        Returns:
                distance (float): Distance between the two points
                """
        x1, y1, z1 = point1
        x2, y2, z2 = point2
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
        return distance
    
    def norm_exp_base(self, value):
        """
        Rescale the value to fit between 1.1 and 1.3, having as max 96 and min 28.
        """
        m = - 0.2/68
        c = 1.1 - 96*m
        return (m)*value + c

    def add_gaussian_noise_tumour(self, scan, label, exp_base):
        """
        Adds Gaussian noise to the scan to mask the tumour
            Parameters:
                    scan (array): Scan to add Gaussian noise
            Returns:
                    scan (array): Scan with Gaussian noise
        """
        scan_noisy = np.copy(scan)
        noise =  np.full((1,96,96,96), 1000.)
        point1 = (48,48,48) # Point in the center
        for x_axis in range(0, 96):
            for y_axis in range(0, 96):
                for z_axis in range(0, 96):
                    if True in label[:, x_axis, y_axis, z_axis]:
                        noise[0,x_axis,y_axis,z_axis] = torch.randn(1)
                    
        #noise = rescale_gaussian_noise(noise, -1, 1)
        
        np.copyto(scan_noisy, noise, where= np.logical_and(noise<100 , scan_noisy!=-1))
        return scan_noisy
    
    