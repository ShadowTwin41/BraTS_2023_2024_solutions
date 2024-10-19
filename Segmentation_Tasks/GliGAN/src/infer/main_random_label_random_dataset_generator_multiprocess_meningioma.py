import os
import argparse
import time
import multiprocessing
import random

import torch
from torch.autograd import Variable

import nibabel as nib
from nilearn import plotting
from nilearn.plotting import find_xyz_cut_coords, view_img, glass_brain, plot_anat, plot_epi

from monai.networks.nets.swin_unetr import SwinUNETR
from monai.transforms import Resize, EnsureChannelFirst
from skimage.filters import threshold_otsu
from scipy.ndimage import binary_closing, binary_fill_holes, generate_binary_structure, label, find_objects
from scipy import ndimage
from scipy.ndimage import label

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


import sys
sys.path.append("./")
sys.path.append("../")
sys.path.append("../../")
from src.networks.LabelGAN import *
from src.utils.data_utils import get_loader


def get_affine(scan_path):
    '''
    Get the metada from the nifti file
    '''
    scan = nib.load(scan_path)
    header = scan.header
    affine = scan.affine
    return affine, header

def save_nifti(image, name, path_metadata, save_path):
    ### Saving nifti without plotting 
    feat = np.squeeze((image).data)  
    affine, header =  get_affine(path_metadata)
    feat = nib.Nifti1Image(feat,affine=affine, header=header)
    nib.save(feat, f"{save_path}/{name}.nii.gz")

def visualization(image, reality, normalize=False, scan_path=None,  x_axis=None, y_axis=None, z_axis=None, save=False):
    '''
    Show the scans in the 3 axis, in the chosen slices or in the center of mass.
        Parameters:
                image (pytorch tensor): Pytorch tensor of the scan to show
                reality (str) : name of the scan to save/show
                normalize (bool) : True to normalise the scan for better visualization
                x_axis (int) : x slice to show 
                y_axis (int) : y slice to show
                z_axis (int) : z slice to show
                save (str) : path to save the scan as .nii with the name of the variable ´reality´
        Returns:
                None
    '''
    if image.shape[0]==3:
        # To deal with the 3 channel label
        print(f"This one has 3 channels! Shape: {image.shape}")
        image = torch.sum(a=image, axis=0)   
        image = image.type(torch.float32)     
    if normalize:
        image = rescale_array(image.detach(),0,1)
    try:
        if image.is_cuda:
            feat = np.squeeze((image).data.cpu().numpy()) # transfer data to the cpu and numpy
        else:
            feat = np.squeeze((image).data)
    except:
        feat = np.squeeze((image).data)

    if x_axis==None:
        if scan_path==None:
            feat = nib.Nifti1Image(feat,affine = np.eye(4))
            display = plotting.plot_img(feat,title=reality)
            if save:
                nib.save(feat, f"{save}/{reality}.nii.gz")
        else:
            affine, header =  get_affine(scan_path)
            feat = nib.Nifti1Image(feat, affine=affine, header=header)
            display = plotting.plot_img(feat,title=reality)
            if save:
                nib.save(feat, f"{save}/{reality}.nii.gz")
    else:
        if scan_path==None:
            feat = nib.Nifti1Image(feat,affine = np.eye(4))
            display = plotting.plot_img(feat,title=reality, cut_coords = (x_axis, y_axis, z_axis))
            if save:
                nib.save(feat, f"{save}/{reality}.nii.gz")
            
        else:
            affine, header =  get_affine(scan_path)
            feat_2 = nib.Nifti1Image(feat,affine = np.eye(4))
            display = plotting.plot_img(feat_2,title=reality, cut_coords = (x_axis, y_axis, z_axis))
            feat = nib.Nifti1Image(feat,affine=affine, header=header)
            if save:
                nib.save(feat, f"{save}/{reality}.nii.gz")

def rescale_array(arr: np.ndarray, minv: float = 0.0, maxv: float = 1.0): #monai function adapted
    """
    Rescale the values of numpy array `arr` to be from `minv` to `maxv`.
    maxa = max_value
    mina = 0
    """  
    try:
        mina = torch.min(arr)
        maxa = torch.max(arr)
    except:
        mina = np.amin(arr)
        maxa = np.amax(arr)
    if mina == maxa:
        return arr * minv
    # normalize the array first
    norm = (arr - mina) / (maxa - mina) 
    # rescale by minv and maxv, which is the normalized array by default 
    return (norm * (maxv - minv)) + minv  

def get_generator(args, MODEL_PATH):
    """
    Loading the pre-trained generator's weights
    """
    generator = SwinUNETR(
            img_size=(96, 96, 96),
            in_channels=args.in_channels_tumour,
            out_channels=args.out_channels_tumour,
            feature_size=args.feature_size,
            use_checkpoint=args.use_checkpoint,
        )
    generator_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))["state_dict"]
    generator.load_state_dict(generator_dict)
    return generator

def create_dirs(save_dir, dir_name):
    """
    Creating fake_scans dir
    """
    save_path = os.path.join(save_dir, dir_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

def read_csv_dataset(args):
    """
    Readin the csv with the datset
    """
    args.df = pd.read_csv(args.csv_path)
    return args.df

def rescale_gaussian_noise(arr, minv, maxv): #monai function adapted
        """
        Rescale the values of numpy array `arr` to be from `minv` to `maxv`.
        """
        maxa = np.unique(arr)[-2]
        mina = np.amin(arr)
        if mina == maxa:
            return arr * minv
        # normalize the array first
        norm = (arr - mina) / (maxa - mina) 
        # rescale by minv and maxv, which is the normalized array by default 
        return (norm * (maxv - minv)) + minv 

def add_gaussian_noise_tumour(scan, label):
    """
    Adds Gaussian noise to the scan to mask the tumour
        Parameters:
                scan (array): Scan to add Gaussian noise
        Returns:
                scan (array): Scan with Gaussian noise
    """
    scan_noisy = np.copy(scan)
    noise =  np.full((96,96,96), 1000.)
    for x_axis in range(0, 96):
        for y_axis in range(0, 96):
            for z_axis in range(0, 96):
                if label[x_axis, y_axis, z_axis]!=0:
                    noise[x_axis,y_axis,z_axis] = torch.randn(1)
                
    #noise = rescale_gaussian_noise(noise, -1, 1)
    
    np.copyto(scan_noisy, noise, where=np.logical_and(noise<100 , scan_noisy!=-1))
    scan_noisy = rescale_gaussian_noise(scan_noisy, -1, 1)
    noise[noise > 100] = 0
    return scan_noisy, noise

def correct_label(label, healthy_scan, original_label):
    """
    Corrects the label for input by ensuring it is not outside of the brain 
    and does not overlap existent tumours
    """
    print(f"correct_label, healthy_scan: {healthy_scan.shape}")
    print(f"correct_label, original_label: {original_label.shape}")
    for x_axis in range(0, 96):
            for y_axis in range(0, 96):
                for z_axis in range(0, 96):
                    # the healthy scan is normalised 
                    # So the background is -1
                    if (healthy_scan[x_axis, y_axis, z_axis]==0) or (original_label[x_axis, y_axis, z_axis]!=0): 
                        label[x_axis, y_axis, z_axis]=0
    return label

def correct_background(healthy_crop_pad, imgs_recon):
    """
    (Of the generated image) Ensures that the values outsite of the brain and values under -1 are -1,
    and values higher than 1 are 1
    """
    imgs_recon_corrected = np.copy(imgs_recon)
    for x_axis in range(0, 96):
            for y_axis in range(0, 96):
                for z_axis in range(0, 96):
                    # the healthy scan is normalised 
                    # So the background is -1
                    if (healthy_crop_pad[x_axis, y_axis, z_axis]==-1) or (imgs_recon_corrected[x_axis, y_axis, z_axis]<-1): 
                        imgs_recon_corrected[x_axis, y_axis, z_axis] = -1
                    if (imgs_recon_corrected[x_axis, y_axis, z_axis] > 1):
                        imgs_recon_corrected[x_axis, y_axis, z_axis] = 1
    return imgs_recon_corrected

def linear_interpolation(final_recons ,healthy_scan_crop, untouch_x_axis, untouch_y_axis, untouch_z_axis):
    """
    Linear filter to correct the intensity values of the reconstructed scan
    Parameters:
        y_1 and y_2 -> intensity values of two points from the original scan
        x_1 and x_2 -> intensity values of the reconstructed scan that must be the same as the original scan
           
    Returns:
        inten_scan -> scan with intensities corrected
    """
    x_list = []
    y_list = []
    y_0=0
    x_0=-1
    x_list.append(x_0)
    y_list.append(y_0)
    
    for i in range(len(untouch_x_axis)):
        y_1 = healthy_scan_crop[untouch_x_axis[i], untouch_y_axis[i], untouch_z_axis[i]]
        x_1 = final_recons[untouch_x_axis[i], untouch_y_axis[i], untouch_z_axis[i]]
        x_list.append(x_1)
        y_list.append(y_1)
        
    coefficients = np.polyfit(x_list, y_list, 1)
    best_m, best_b = coefficients
    
    #print("Best-fit line equation: y =", best_m, "x +", best_b)
    #y =  m*x + b
    inten_scan = best_m*final_recons + best_b

    for z_axis in range(0, 96):
            for y_axis in range(0, 96):
                for x_axis in range(0, 96):
                    if (healthy_scan_crop[x_axis, y_axis, z_axis]) == 0:
                        inten_scan[x_axis, y_axis, z_axis] = 0
                    if (inten_scan[x_axis, y_axis, z_axis] < 0):
                        inten_scan[x_axis, y_axis, z_axis] = 0

    return inten_scan

def get_inten_coord(healthy_scan_crop, original_label_crop, noise):
    """
    Selects all the points (x,y,z coordenates) in the bonding cube 
    which is inside the brain and does not touch the original tumour 
    neither the new tumour
    """
    untouch_x_axis = list()
    untouch_y_axis = list()
    untouch_z_axis = list()
    constant = 5

    def verify(x_axis, y_axis, z_axis, untouch_x_axis,untouch_y_axis,untouch_z_axis):
        if (healthy_scan_crop[x_axis, y_axis, z_axis])!=0 and original_label_crop[x_axis, y_axis, z_axis]==0 and (noise[x_axis, y_axis, z_axis])==0:
            mini_x = max(x_axis-constant, 0)
            maxi_x = min(x_axis+constant, 95)
            mini_y = max(y_axis-constant, 0)
            maxi_y = min(y_axis+constant, 95)
            mini_z = max(z_axis-constant, 0)
            maxi_z = min(z_axis+constant, 95)

            zeros = np.sum(noise[mini_x:maxi_x, mini_y:maxi_y, mini_z:maxi_z])

            #print(f"zeros: {zeros}")
            if (zeros)==0:
                untouch_x_axis.append(x_axis)
                untouch_y_axis.append(y_axis)
                untouch_z_axis.append(z_axis)
        return (untouch_x_axis,untouch_y_axis,untouch_z_axis)
                    
    y_axis=0
    for x_axis in range(0, 96):
        for z_axis in range(0, 96):
            untouch_x_axis,untouch_y_axis,untouch_z_axis = verify(x_axis, y_axis, z_axis, untouch_x_axis,untouch_y_axis,untouch_z_axis)           
    y_axis=95
    for x_axis in range(0, 96):
        for z_axis in range(0, 96):
                untouch_x_axis,untouch_y_axis,untouch_z_axis = verify(x_axis, y_axis, z_axis, untouch_x_axis,untouch_y_axis,untouch_z_axis)
    
    z_axis=0
    for x_axis in range(0, 96):
        for y_axis in range(0, 96):
                untouch_x_axis,untouch_y_axis,untouch_z_axis = verify(x_axis, y_axis, z_axis, untouch_x_axis,untouch_y_axis,untouch_z_axis)
    z_axis=95
    for x_axis in range(0, 96):
        for y_axis in range(0, 96):
                untouch_x_axis,untouch_y_axis,untouch_z_axis = verify(x_axis, y_axis, z_axis, untouch_x_axis,untouch_y_axis,untouch_z_axis)
    
    x_axis=0
    for y_axis in range(0, 96):
        for z_axis in range(0, 96):
                untouch_x_axis,untouch_y_axis,untouch_z_axis = verify(x_axis, y_axis, z_axis, untouch_x_axis,untouch_y_axis,untouch_z_axis)
    x_axis=95
    for y_axis in range(0, 96):
        for z_axis in range(0, 96):
                untouch_x_axis,untouch_y_axis,untouch_z_axis = verify(x_axis, y_axis, z_axis, untouch_x_axis,untouch_y_axis,untouch_z_axis)
    
    return (untouch_x_axis,untouch_y_axis,untouch_z_axis)       

def load_generators(args):
    t1_ce_gen_path = f"{args.t1ce_logdir}/weights/generator_{args.g_t1ce_n}.pt"
    generator_t1ce = get_generator(args=args, MODEL_PATH=t1_ce_gen_path)
    generator_t1ce.eval()
    print(f"{args.t1ce_logdir}/weights/generator_{args.g_t1ce_n}.pt")

    # Load label Generator weights
    G_label = Generator(noise=args.latent_dim, out_channels=int(args.out_channels_label))
    G_label_dict = torch.load(f'{args.labelGAN_logdir}/weights/G_iter_{args.g_label_n}.pt', map_location=torch.device('cpu'))
    G_label.load_state_dict(G_label_dict["state_dict"])
    print(f"Label generator loaded from: {args.labelGAN_logdir}/weights/G_iter_{args.g_label_n}.pt")

    if torch.cuda.is_available():
        generator_t1ce.cuda()
    generators_dic = {'scan_t1ce':generator_t1ce}
    return generators_dic, G_label

def prepar_label_to_gen(args, new_label, healthy_scan_crop_dic, original_label_crop):
    """
    Preparing the label for the generator
    """
    # The new label is fixed so it doesn't overlap the old one and doesn't stick out of the brain
    label_crop = correct_label(label=np.copy(new_label), healthy_scan=next(iter(healthy_scan_crop_dic.values())), original_label=original_label_crop)
    ####################################################################################
    label_crop_cuda = torch.from_numpy(np.copy(label_crop))
    
    label_crop_cuda = args.LABEL_TRANSFORM()(label_crop_cuda)
    label_crop_cuda = torch.reshape(input=label_crop_cuda, shape=(1,int(args.out_channels_label),96,96,96))

    label_crop_cuda = label_crop_cuda.cuda()
    #print(f"label_crop_cuda: {torch.sum(label_crop_cuda)}")
    return label_crop_cuda, label_crop

def prepar_healthy_scan_to_gen(x, y, z, healthy_scan_dic, original_label):
    ####################################################################################
    #Cropping the scan considering the coordinates used as input
    x_min = x - 48
    x_max = x + 48
    y_min = y - 48
    y_max = y + 48
    z_min = z - 48
    z_max = z + 48

    def check_boundaries(min_value, max_value, max_boundary):
        if min_value > max_boundary or max_value < 0:
            raise Exception(f"Choose coordinates inside the boundaries [0:{max_boundary}]") 
        if min_value < 0:
            max_value = max_value + (-min_value)
            min_value = 0
            return min_value, max_value
        elif max_value > max_boundary:
            min_value = min_value - (max_value - max_boundary)
            max_value = max_boundary
            return min_value, max_value
        else:
            return min_value, max_value

    x_min, x_max = check_boundaries(min_value=x_min, max_value=x_max, max_boundary=original_label.shape[0])
    y_min, y_max = check_boundaries(min_value=y_min, max_value=y_max, max_boundary=original_label.shape[1])
    z_min, z_max = check_boundaries(min_value=z_min, max_value=z_max, max_boundary=original_label.shape[2])

    print("#############################################")
    print("The tumour will be inserted into the block")
    print(f"##### x_min: {x_min}, x_max: {x_max} #####")
    print(f"##### y_min: {y_min}, y_max: {y_max} #####")
    print(f"##### z_min: {z_min}, z_max: {z_max} #####")
    print("#############################################")
    healthy_scan_crop_dic = {}
    for key in list(healthy_scan_dic.keys()):
        healthy_scan_crop_dic[key] = np.copy(healthy_scan_dic[key][x_min:x_max, y_min:y_max, z_min:z_max])

    original_label_crop = np.copy(original_label[x_min:x_max, y_min:y_max, z_min:z_max])
    return (healthy_scan_crop_dic, original_label_crop, x_min, x_max, y_min, y_max, z_min, z_max)

def prepar_real_label_to_gen(args, new_label, healthy_scan_crop_dic, original_label_crop, registo_lista_label):
    """
    Preparing the label for the generator
    """
    # Getting the boundaries of the label
    label_x_min = registo_lista_label["x_extreme_min"] -1 # -1 so when cutting, the edges are not removed
    label_x_max = registo_lista_label["x_extreme_max"] 
    label_y_min = registo_lista_label["y_extreme_min"] -1
    label_y_max = registo_lista_label["y_extreme_max"]  
    label_z_min = registo_lista_label["z_extreme_min"] -1
    label_z_max = registo_lista_label["z_extreme_max"] 

    if label_x_max - label_x_min > 96:
        label_x_min = registo_lista_label["x_extreme_min"]
    if label_y_max - label_y_min > 96:
        label_y_min = registo_lista_label["y_extreme_min"]
    if label_z_max - label_z_min > 96:
        label_z_min = registo_lista_label["z_extreme_min"]
    
    # Cropping the label
    label_crop = np.copy(new_label[label_x_min:label_x_max, label_y_min:label_y_max, label_z_min:label_z_max])
    
    ## Padding the label to 96x96x96
    def padding_need(min_value, max_value):
        # Compute the padding needed for the final size to be 96x96x96
        diff = max_value - min_value
        base = int((96 - diff)/2)
        top = int((96 - diff)/2 +0.5)
        return base, top

    x_base_pad, x_top_pad = padding_need(min_value=label_x_min, max_value=label_x_max)
    y_base_pad, y_top_pad = padding_need(min_value=label_y_min, max_value=label_y_max)
    z_base_pad, z_top_pad = padding_need(min_value=label_z_min, max_value=label_z_max)
    label_crop = np.pad(np.copy(label_crop), pad_width=((x_base_pad,x_top_pad), (y_base_pad,y_top_pad), (z_base_pad,z_top_pad)), mode='constant', constant_values=(0, 0))
    
    # The new label is fixed so it doesn't overlap the old one and doesn't stick out of the brain
    label_crop = correct_label(label=np.copy(label_crop), healthy_scan=next(iter(healthy_scan_crop_dic.values())), original_label=original_label_crop)
    ####################################################################################
    
    label_crop_cuda = torch.from_numpy(np.copy(label_crop))
    label_crop_cuda = args.LABEL_TRANSFORM()(label_crop_cuda)
    label_crop_cuda = torch.reshape(input=label_crop_cuda, shape=(1,int(args.out_channels_label),96,96,96))
    label_crop_cuda = label_crop_cuda.cuda()
    #print(f"label_crop_cuda: {torch.sum(label_crop_cuda)}")
    return label_crop_cuda, label_crop

def insert_tumour(args, generators_dic, healthy_scan_dic, original_label, new_label, x, y, z, registo_lista_label, source_original_label, source_original_healthy_scan):
    ### Getting the portion of the brain to insert the tumour (for input of the generator)
    healthy_scan_crop_dic, original_label_crop, x_min, x_max, y_min, y_max, z_min, z_max = prepar_healthy_scan_to_gen(x, y, z, healthy_scan_dic, original_label)
    print(f"healthy_scan_crop_dic.shape: {healthy_scan_crop_dic['scan_t1ce'].shape}")
    print(f"original_label_crop.shape: {original_label_crop.shape}")
    
    ### Getting the label for input of the generator
    if registo_lista_label is None:
        label_crop_cuda, label_crop = prepar_label_to_gen(args, new_label, healthy_scan_crop_dic, original_label_crop)
    else:
        label_crop_cuda, label_crop = prepar_real_label_to_gen(args, new_label, healthy_scan_crop_dic, original_label_crop, registo_lista_label)
    
    print(f"torch.sum(label_crop_cuda): {torch.sum(label_crop_cuda)}")
    ### Getting the generators for the 4 modalities
    scan_final_dic={}
    for idx, key in enumerate(list(healthy_scan_crop_dic.keys())):
        healthy_scan_crop = healthy_scan_crop_dic[key]
        ## Adding noise to the tumour zone
        healthy_scan_crop_norm = rescale_array(arr=np.copy(healthy_scan_crop), minv=-1, maxv=1) # Normalise scan
        healthy_noisy, noise = add_gaussian_noise_tumour(scan=healthy_scan_crop_norm, label=label_crop) 
        healthy_noisy_cuda = torch.from_numpy(healthy_noisy)
        healthy_noisy_cuda = torch.reshape(input=healthy_noisy_cuda, shape=(1,1,96,96,96))
        healthy_noisy_cuda = healthy_noisy_cuda.cuda()
        # Creating the generator input
        input_cat = torch.cat([healthy_noisy_cuda, label_crop_cuda], dim=1) 
        input_cat = input_cat.type(torch.float32)
        # Generated tumour
        imgs_recon = generators_dic[key](input_cat)
        # Correcting the background (to -1) and values outside [-1,1]
        imgs_recon = imgs_recon.data.cpu().numpy().reshape(96,96,96)            
        imgs_recon_corrected = correct_background(healthy_crop_pad=healthy_scan_crop_norm, imgs_recon=imgs_recon)
        # Denormalise the portion generated to fit the surroudings 
        untouch_x_axis, untouch_y_axis, untouch_z_axis = get_inten_coord(healthy_scan_crop=healthy_scan_crop, original_label_crop=original_label_crop, noise=noise)
        final_recons = linear_interpolation(final_recons=imgs_recon_corrected, healthy_scan_crop=healthy_scan_crop, untouch_x_axis=untouch_x_axis, untouch_y_axis=untouch_y_axis, untouch_z_axis=untouch_z_axis)
        # Final full resolution scan with synthetic tumour
        scan_final = np.copy(source_original_healthy_scan)#healthy_scan_dic[key])
        print(f"scan_final: {scan_final.shape}")
        print(f"x_min: {x_min}")
        print(f"x_max: {x_max}")
        print(f"y_min: {y_min}")
        print(f"y_max: {y_max}")
        print(f"z_min: {z_min}")
        print(f"z_max: {z_max}")
        print(f"final_recons: {final_recons.shape}")
        scan_final[x_min:x_max, y_min:y_max, z_min:z_max] = final_recons
        scan_final_dic[key] = scan_final
        if idx!=0:
            print("ERROR, this code is not prepared for more than one image per case!")
        
    # Final label with new tumour labeled
    complete_label_crop = np.copy(label_crop) + np.copy(original_label_crop)
    label_final = np.copy(original_label)
    label_final[x_min:x_max, y_min:y_max, z_min:z_max] = complete_label_crop

    return scan_final_dic, label_final

    #visualization(image=label_final, reality=(f"label_{ficheiro_scan[i].split('.')[-3].split('_')[-2]}_{ficheiro_label.split('.')[-3].split('_')[-2]}"), scan_path=ficheiro_scan[i], x_axis=x, y_axis=y, z_axis=z, save="/projects/brats2023_a_f/BRAINTUMOUR/trash")
    #visualization(image=scan_final_list[0], reality=(f"scan_{ficheiro_scan[i].split('.')[-3].split('_')[-1]}_{ficheiro_scan[idx].split('.')[-3].split('_')[-2]}_{ficheiro_label.split('.')[-3].split('_')[-2]}"), scan_path=ficheiro_label, x_axis=x, y_axis=y, z_axis=z, save="/projects/brats2023_a_f/BRAINTUMOUR/trash")

def create_brain_mask(brain_numpy):
    """
    Takes as input a numpy array of the brain and returns the brain mask.
    This is important to select correctly the place for center of the tumour.
    """
    # Find what is the best threshold
    threshold = threshold_otsu(brain_numpy)
    brain_mask = brain_numpy > threshold
    # Close the holes inside of the brain
    structure = generate_binary_structure(3, 3)
    brain_mask = binary_closing(brain_mask, structure=structure, iterations=10)  # You can adjust the number of iterations

    # Label connected components
    labeled_array, num_features = label(brain_mask)

    # Find the largest connected component
    sizes = np.bincount(labeled_array.ravel())
    sizes[0] = 0  # Ignore the background
    largest_label = sizes.argmax()

    # Create a mask for the largest connected component
    brain_mask = labeled_array == largest_label
    return brain_mask


    
def random_center_tumour_voxel(healthy_scan, original_label):
    """
    # This portion of code selects the first voxel in the brain
    for z_axis in range (0,96):
        for y_axis in range (0,96):
            for x_axis in range (0,96):

    # for Brats glioma 2023 the shape was 240x240x155
    # however, the brats glioma post treatment cases have shape 182x218x182
    # therefore, the cases are padded to 256x256x256 so it fits both datasets 
    """
    
    for i in range(1000):
        x_axis = np.random.randint(low=0, high=original_label.shape[0], size=2, dtype=int)[0]
        y_axis = np.random.randint(low=0, high=original_label.shape[1], size=2, dtype=int)[0] 
        z_axis = np.random.randint(low=0, high=original_label.shape[2], size=1, dtype=int)[0]
        if healthy_scan[x_axis,y_axis,z_axis]!=0:
            x_min = x_axis - 48
            x_max = x_axis + 48
            y_min = y_axis - 48
            y_max = y_axis + 48
            z_min = z_axis - 48
            z_max = z_axis + 48
            if (np.sum(original_label[x_min:x_max, y_min:y_max, z_min:z_max]))==0:
                return x_axis, y_axis, z_axis
    print("No space for a new tumour")
    return None, None, None

def get_a_random_label(args):
    G = args.G_label
    if torch.cuda.is_available():
        G.cuda()   
        z_rand = Variable(torch.randn((1,100))).cuda() #random vector
    else:
        z_rand = Variable(torch.randn((1,100))) #random vector

    x_rand = G(z_rand)

    if torch.cuda.is_available():
        x_rand_cpu = np.squeeze((x_rand).data.cpu().numpy())
    else:
        x_rand_cpu = np.squeeze((x_rand).data.numpy())

    rand_label_0 = EnsureChannelFirst(channel_dim="no_channel")(x_rand_cpu)
    
    rand_label_trilinear_0 = Resize(spatial_size=(96, 96, 96), mode='trilinear')(rand_label_0)

    rand_label_trilinear_0 = np.squeeze((rand_label_trilinear_0).data.numpy())
    
    rand_label_trilinear_0[rand_label_trilinear_0 > 0.5] = 1
    rand_label_trilinear_0[rand_label_trilinear_0 <= 0.5] = 0

    return rand_label_trilinear_0

def process_ids_label(args, ids_label, healthy_scan_dic, ficheiro_scan_dic, original_label, new_random_label, registo_lista_label, source_original_label, source_original_healthy_scan):
        print(f"Label number {ids_label}")
        x, y, z = random_center_tumour_voxel(healthy_scan=next(iter(healthy_scan_dic.values())), original_label=original_label)
        print(f"Selected x:{x}, y:{y}, z:{z}")
        if x is None:
            print(f"------------ sample number {ids_label} not generated ------------")  
            return None
        else:
            
            # Creating a new case
            if new_random_label is None:
                print("USING RANDOM LABEL")
                new_random_label = get_a_random_label(args)
                final_naming = "fake"
                print(f"FAKE np.sum(new_random_label): {np.sum(new_random_label)}")
            else:
                print("USING REAL LABEL")
                final_naming = "real"
                print(f"REAL np.sum(new_random_label): {np.sum(new_random_label)}")
            
            # Creating the directory to save each patient
            if args.dataset=="BRATS_2024_MENINGIOMA":
                create_dirs(args.save_folder, f"BraTS-{next(iter(ficheiro_scan_dic.values())).split('/')[-2][-13:]}_{final_naming}_label_{ids_label}") # .split('/')[-2][-10:] -> GoAT-00000
                save_path = f"{args.save_folder}/BraTS-{next(iter(ficheiro_scan_dic.values())).split('/')[-2][-13:]}_{final_naming}_label_{ids_label}" # ids_label -> Number of the fake label
            else:
                raise ValueError("Datasets available: BRATS_2024_MENINGIOMA")


            scan_final_dic, label_final = insert_tumour(args=args, generators_dic=args.generators_dic, healthy_scan_dic=healthy_scan_dic, original_label=original_label, new_label=new_random_label, x=x, y=y, z=z, registo_lista_label=registo_lista_label, source_original_label=source_original_label, source_original_healthy_scan=source_original_healthy_scan)
            # Save each modality in the patient's folder
            for idx, key in enumerate(list(scan_final_dic.keys())):
                image = scan_final_dic[key]

                if args.dataset=="BRATS_2024_MENINGIOMA":
                    name = f"BraTS-{ficheiro_scan_dic[key].split('/')[-2][-13:]}_{final_naming}_label_{ids_label}-{key}"
                    image = crop_to_shape(arr=image, original_shape=source_original_label.shape)

                path_metadata = ficheiro_scan_dic[key]
                save_nifti(image=image, name=name, path_metadata=path_metadata, save_path=save_path)

            # Saving the new label
            if args.dataset=="BRATS_2024_MENINGIOMA":
                name = f"BraTS-{ficheiro_scan_dic[key].split('/')[-2][-13:]}_{final_naming}_label_{ids_label}-seg"


            if args.dataset=="BRATS_2024_MENINGIOMA":
                label_final = crop_to_shape(arr=label_final, original_shape=source_original_label.shape) 

            else:
                raise ValueError("Datasets available: BRATS_2024_MENINGIOMA")
            save_nifti(image=label_final.astype(int), name=name, path_metadata=path_metadata, save_path=save_path)
            print(f"The files were saved in: {save_path}")

            print(f"------------ sample generated ------------")  

def pad_to_shape(arr, target_shape=(256, 256, 256)):
    """
    Pad the array to have the target_shape
    """
    current_shape = arr.shape
    pad_width = []

    for current_size, target_size in zip(current_shape, target_shape):
        if target_size > current_size:
            total_padding = target_size - current_size
            half_padding = total_padding // 2
            pad_width.append((half_padding, total_padding - half_padding))
        else:
            pad_width.append((0, 0))  # No padding if target size is smaller or equal
    
    padded_arr = np.pad(arr, pad_width, mode='constant', constant_values=0)
    return padded_arr

def crop_to_shape(arr, original_shape):
    current_shape = arr.shape
    crop_slices = []

    for current_size, target_size in zip(current_shape, original_shape):
        total_crop = current_size - target_size
        half_crop = total_crop // 2
        crop_slices.append(slice(half_crop, half_crop + target_size))
    
    cropped_arr = arr[tuple(crop_slices)]
    return cropped_arr

def generating(args):
    # Getting all label ids from the csv file
    label_ids_list = args.df.sort_values(by="id")["id"].tolist()
    print(f"Number of IDs: {len(label_ids_list)}")
    if args.end_case == "end":
        label_ids_list = label_ids_list[int(args.start_case):]
    else:
        label_ids_list = label_ids_list[int(args.start_case):int(args.end_case)]
    print(f"New Number of IDs: {len(label_ids_list)}")
    print(f"Staring on {args.start_case} and ending on {args.end_case}")

    for ids_scan_n, ids_scan in enumerate(label_ids_list):
        # For each case in the csv
        print(f"Number {ids_scan_n}")
        registo_scan = args.df.loc[args.df['id'] == ids_scan]
        registo_lista_scan=registo_scan.iloc[0]
        ficheiro_scan_dic={'scan_t1ce':registo_lista_scan['scan_t1ce']}
        healthy_scan_dic={}

        for idx, key in enumerate(list(ficheiro_scan_dic.keys())):
            # Load all modalities to memory 
            numpy_arr = nib.load(ficheiro_scan_dic[key]).get_fdata()
            # the meningioma dataset has very distinct image shapes
            # Therefore, let's pad all images to the biggest size possible
            healthy_scan_dic[key] = pad_to_shape(arr=numpy_arr, target_shape=(800, 800, 512)) 
            source_original_healthy_scan = np.copy(healthy_scan_dic[key])
            # Removing the background no skull/brain
            binary_brain_mask = create_brain_mask(brain_numpy=healthy_scan_dic[key])
            healthy_scan_dic[key] = healthy_scan_dic[key]*binary_brain_mask
            
        
        print(f"Scan path: {next(iter(ficheiro_scan_dic.values()))}")
        
        original_label = registo_lista_scan['label']
        original_label = nib.load(original_label)
        original_label = original_label.get_fdata()
        source_original_label = np.copy(original_label)
        original_label = pad_to_shape(arr=original_label, target_shape=(800, 800, 512))


        # Choose randomly between 0 and 1
        new_random_label_list = []
        registo_lista_label_list = []
        for index in range(0,int(args.new_n)): 
            use_random_label = random.choice([True, False])
            if use_random_label:
                new_random_label = None
                registo_lista_label = None
                registo_lista_label_list.append(registo_lista_label)
                new_random_label_list.append(new_random_label)
            else:
                random_idx = np.random.randint(0, high=len(label_ids_list), size=1, dtype=int)
                id_label_random_idx = label_ids_list[random_idx[0]] 
                registo_label=args.df.loc[args.df['id'] == id_label_random_idx]
                registo_lista_label=registo_label.iloc[0]
                ficheiro_label = registo_lista_label['label']
                new_random_label = nib.load(ficheiro_label)
                new_random_label = new_random_label.get_fdata()
                # new_random_label = pad_to_shape(arr=new_random_label, target_shape=(800, 800, 512)) # The label is padded to the biggest shape possible # TODO this cannot happen, when cropping the ROI, padding is not taking in consideration :D
                new_random_label_list.append(new_random_label)
                print(f"Case number {index} will use label: {ficheiro_label}")
                registo_lista_label_list.append(registo_lista_label)
                
        # Set the start method to 'spawn' before creating the pool
        multiprocessing.set_start_method('spawn', force=True)
        pool = multiprocessing.Pool(processes=int(args.num_process))
        pool.starmap(process_ids_label, [(args, ids_label, healthy_scan_dic, ficheiro_scan_dic, original_label, new_random_label_list[ids_label], registo_lista_label_list[ids_label], source_original_label, source_original_healthy_scan) for ids_label in range(0, int(args.new_n))])
        pool.close()
        pool.join()

def __main__():
    parser = argparse.ArgumentParser(description="PyTorch Training")
    parser.add_argument("--logdir", type=str, help="Directory with the experiment")
    parser.add_argument("--batch_size", default=1, type=int, help="Batch size")
    parser.add_argument("--in_channels_tumour", default=4, type=int, help="Number of input channels for the tumour GAN (Default 4 -> 3 channel label + 1 channel scan with noise)")
    parser.add_argument("--out_channels_tumour", default=1, type=int, help="Number of output channels for the tumour GAN (Default 1 -> Scan with reconstructed tumour)")
    parser.add_argument("--out_channels_label", default=3, type=int, help="Number of output channels for the label GAN (Default 3 -> Three regions)")
    parser.add_argument("--feature_size", default=48, type=int, help="Feature size")
    parser.add_argument("--use_checkpoint", action="store_true", help="Use gradient checkpointing to save memory")
    parser.add_argument("--csv_path", default='', type=str, help="Path to the CSV file. If empty, will use the csv in the logdir")
    parser.add_argument("--dataset", type=str, help="What dataset and from what year. E.g. Brats_2023. Not prepared for versions before 2023 or other dataset")
    parser.add_argument("--save_folder", default="", type=str, help="Folder to save the generated data")
    parser.add_argument("--g_t1ce_n", type=int, help="Number of the generator t1ce")
    parser.add_argument("--t1ce_logdir", default="", type=str, help="Directory with the t1ce checkpoint")
    parser.add_argument("--g_label_n", type=int, help="Number of the generator label GAN")
    parser.add_argument("--labelGAN_logdir", default="", type=str, help="Directory with the label GAN checkpoint")
    parser.add_argument("--num_process", default="", type=int, help="Number of processes. It must be equal to or less than the number of CPU cores. If you're not sure, just use 1.")
    parser.add_argument("--latent_dim", default=100, type=int, help="Size of the latend dim (random input vector of the label generator)")
    parser.add_argument("--start_case", default=0, type=str, help="Size of the latend dim (random input vector of the label generator)")
    parser.add_argument("--end_case", default="end", type=str, help="Size of the latend dim (random input vector of the label generator)")
    parser.add_argument("--new_n", default=17, type=int, help="Number of new synthetic cases")
    args = parser.parse_args()

    args.HOME_DIR = f"../../Checkpoint/{args.logdir}"

    if args.csv_path == "":
        for file_name in os.listdir(f"../../Checkpoint/{args.logdir}"):
            if file_name.endswith("csv"):
                args.csv_path = os.path.join(f"../../Checkpoint/{args.logdir}", file_name)
    else:
        args.csv_path = args.csv_path
    print(f"CSV_PATH: {args.csv_path}")

    if args.save_folder == "":
        args.save_folder = f"../../Checkpoint/{args.logdir}/Synthetic_dataset_random_labels"
    else:
        args.save_folder = args.save_folder
    print(f"Fake dataset folder: {args.save_folder}")

    ## Setting default checkpoint dir
    if args.t1ce_logdir == "":
        args.t1ce_logdir = f"../../Checkpoint/{args.logdir}/t1ce"

    if args.labelGAN_logdir == "":
        args.labelGAN_logdir = f"../../Checkpoint/{args.logdir}/label"

    print(f"Checkpoint dirs:\nT1ce: {args.t1ce_logdir}")


    if ("2024" in args.dataset.lower()) and ("meningioma" in args.dataset.lower()) and ("brats" in args.dataset.lower()):
        args.dataset = "BRATS_2024_MENINGIOMA"
    else:
        raise ValueError("The dataset must be from BRATS_2024_MENINGIOMA")

    if args.dataset=="BRATS_2024_MENINGIOMA":
        print(f"Using dataset: BRATS_2024_MENINGIOMA")
        from src.utils.convert_to_multi_channel_based_on_brats_classes import ConvertToMultiChannelBasedOnBratsMeningiomaClasses2024 as ConvertToMultiChannelBasedOnBratsGliomaClasses
        args.LABEL_TRANSFORM = ConvertToMultiChannelBasedOnBratsGliomaClasses
    else:
        raise ValueError("The dataset must be from BRATS_2024_MENINGIOMA")

    print("Loading generators")
    args.generators_dic, args.G_label = load_generators(args)
    print("Loading csv")
    args.df=read_csv_dataset(args)
    print(f"Generating cases now!")
    generating(args)
    
    

if __name__ == "__main__":
    #python main_random_dataset_generator.py --csv_path /projects/brats2023_a_f/BRAINTUMOUR/data/brats21_96_2023.csv --save_folder /projects/brats2023_a_f/Synthetic_dataset --g_t1ce_n 400000  --g_t1_n 400000 --g_t2_n 400000  --g_flair_n 400000
    __main__()
        
        
