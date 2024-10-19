import sys
sys.path.insert(1, './')    
from networks.alpha_GAN import Generator as L_G
import torch
from torch.autograd import Variable
from monai.transforms.spatial import functional as F
import monai
import numpy as np
import random
class New_Voided:
    def __init__(self, device):
        self.device = device

    def __compute_bounding_box__(self, segmentation):
        """
        Computes the bounding box coordinates for a segmentation tensor.

        Args:
            segmentation (torch.Tensor): A 3D segmentation tensor of shape (256, 256, 256).

        Returns:
            tuple: A tuple containing the bounding box coordinates (xmin, ymin, zmin, xmax, ymax, zmax).
        """
        # Find the non-zero pixels (indicating the object)
        nonzero_indices = np.nonzero(segmentation)
        # Get minimum and maximum coordinates for each dimension
        ymin = nonzero_indices[0].min()  # Access the first element directly
        ymax = nonzero_indices[0].max()  # Access the first element directly
        xmin = nonzero_indices[1].min()  # Access the second element directly
        xmax = nonzero_indices[1].max()  # Access the second element directly
        zmin = nonzero_indices[2].min()  # Access the third element directly
        zmax = nonzero_indices[2].max()  # Access the third element directly

        return xmin, ymin, zmin, xmax, ymax, zmax

    def __outside_existent_label_F__(self, old_seg, x_center, y_center, z_center, new_seg_width, new_seg_height, new_seg_depth):
        """Check if outside the real label"""
        #if (x_center+(new_seg_width//2))<xmin_2 or (x_center-(new_seg_width//2))>xmax_2 or (y_center+(new_seg_height//2))<ymin_2 or y_center-(new_seg_height//2)>ymax_2 or (z_center+(new_seg_depth//2))<zmin_2 or z_center-(new_seg_depth//2)>zmax_2:
        ROI = old_seg[y_center-new_seg_height//2:y_center+new_seg_height//2, x_center-new_seg_width//2:x_center+new_seg_width//2, z_center-new_seg_depth//2:z_center+new_seg_depth//2]
        if np.sum(ROI) == 0:
            return True
        else:
            return False

    def __inside_image_F__(self, image, x_center, y_center, z_center):
        """Check if inside the image"""
        if image[y_center, x_center, z_center]!=0:
            return True
        else:
            return False

    def __inside_borders_F__(self, image, x_center, y_center, z_center, new_seg_width, new_seg_height, new_seg_depth):
        """Check if inside of borders"""
        if ((x_center+(new_seg_width//2))<image.shape[1]) and ((x_center-(new_seg_width//2))>0) and ((y_center+(new_seg_height//2))<image.shape[0]) and ((y_center-(new_seg_height//2))>0) and ((z_center+(new_seg_depth//2))<image.shape[2]) and ((z_center-(new_seg_depth//2))>0):
            return True
        else:
            return False

    def __ensure_even__(self, number):
        """
        Ensures a variable is an even integer (par) and returns the nearest even number.

        Args:
            number: The number to check and potentially modify.

        Returns:
            int: The even version of the input number.
        """
        return number if number % 2 == 0 else number + 1

    def __crop_and_paste_segmentation__(self, new_seg, old_seg, image):
        """
        Crops the segmentation from the given label based on the bounding box
        and pastes it onto another label.

        Args:
            new_seg (torch.Tensor): A 3D new_seg tensor of shape (256, 256, 256).
            old_seg (torch.Tensor): A 3D old_seg tensor of shape (256, 256, 256).
            image (torch.Tensor): A 3D image T1n tensor of shape (256, 256, 256).


        Returns:
            torch.Tensor: A new img tensor with the cropped new_seg pasted onto it.
        """
        # Convert tensors to NumPy arrays for easier manipulation
        new_seg_np = new_seg
        old_seg_np = old_seg

        # Compute the bounding for cropping the new segmentation
        xmin_1, ymin_1, zmin_1, xmax_1, ymax_1, zmax_1= self.__compute_bounding_box__(new_seg_np)
        
        # Making sure the dimentions are even integers
        new_seg_height = ymax_1 - ymin_1
        if new_seg_height % 2 != 0:
            ymax_1 = ymax_1+1
            new_seg_height = ymax_1 - ymin_1

        new_seg_width = xmax_1 - xmin_1
        if new_seg_width % 2 != 0:
            xmax_1 = xmax_1+1
            new_seg_width = xmax_1 - xmin_1

        new_seg_depth = zmax_1 - zmin_1
        if new_seg_depth % 2 != 0:
            zmax_1 = zmax_1+1
            new_seg_depth = zmax_1 - zmin_1
        cropped_new_seg_np = new_seg_np[ymin_1:ymax_1, xmin_1:xmax_1, zmin_1:zmax_1]

        # These are the dimentions of the new segmentation we want to insert in the already existent segmentation
        new_seg_height, new_seg_width, new_seg_depth =  self.__ensure_even__(new_seg_height), self.__ensure_even__(new_seg_width), self.__ensure_even__(new_seg_depth)

        # Get the bounding box of the second label, to place the first segmentation outside of it
        xmin_2, ymin_2, zmin_2, xmax_2, ymax_2, zmax_2 = self.__compute_bounding_box__(old_seg_np)

        # Find a nice center i.e., where the new label is not inside of the old label, has center inside of the brain and inside of the image
        NICE_CENTER = False
        GIVE_UP = False
        patient = 10000
        while not NICE_CENTER:
            x_center, y_center, z_center = random.randint(0, old_seg_np.shape[1]-1),  random.randint(0, old_seg_np.shape[0]-1),  random.randint(0, old_seg_np.shape[2]-1)
            # Check if chosen center is a good place
            outside_existent_label = self.__outside_existent_label_F__(old_seg_np, x_center, y_center, z_center, new_seg_width, new_seg_height, new_seg_depth)
            inside_image = self.__inside_image_F__(image, x_center, y_center, z_center)
            inside_borders = self.__inside_borders_F__(image, x_center, y_center, z_center, new_seg_width, new_seg_height, new_seg_depth)
            if outside_existent_label and inside_image and inside_borders:
                NICE_CENTER = True
                #print(f"Center point {x_center, y_center, z_center}")
            patient-=1
            if patient==0:
                GIVE_UP = True
                break

        if not GIVE_UP:
            # Create a new image with the same channels and dimensions as the original old_seg
            complete_seg = np.copy(old_seg_np)

            # Paste the cropped new_seg onto the new image 
            complete_seg[y_center-new_seg_height//2:y_center+new_seg_height//2, x_center-new_seg_width//2:x_center+new_seg_width//2, z_center-new_seg_depth//2:z_center+new_seg_depth//2] = cropped_new_seg_np

            # Convert the pasted image back to a PyTorch tensor
            #complete_seg_tensor = torch.from_numpy(complete_seg)
            
            return complete_seg
        else:
            return None

    def __remove_region_from_volume__(self, complete_seg, volume):
        """
        Get a new volume with a voided region

        Args:
            complete_seg: Tensor of an both real and fake tumour segmentations
            image: Tensor T1n scan of image from where a tumour will be cropped.
        Returns:
            new_volume: Tensor -> Volume without the unhealthy and healthy regions 
        """
        # Find voxels belonging to the tumour (where tumour_mask is 1)
        seg_indices = np.where(complete_seg == 1)
        # Set those voxels in the volume array to zero
        new_volume = volume.clone()
        new_volume[seg_indices] = 0
        return new_volume
    
    def get_complete_seg(self, new_seg, old_seg, image):
        """
        Get a new volume with a voided region
        Args:
            new_seg: Tensor of a new synthetic tumour label
            old_seg: Tensor of a old tumour label 
            image: Tensor T1n scan of image 
        Returns:
            complete_seg: Tensor -> Complete segmentation of the tumour (old and new)
        """
        complete_seg = self.__crop_and_paste_segmentation__(new_seg=new_seg, old_seg=old_seg, image=image)
        if complete_seg is not None:
            return True, complete_seg
        else:
            return False, old_seg

    def get_new_voided(self, complete_seg, image, old_seg=None):
        """
        Get a new volume with a voided region
        Args:
            complete_seg: Tensor of a new synthetic tumour label to define the location and shape of the region to crop from the T1n volume
            image: Tensor T1n scan of image from where a tumour will be cropped.
        Returns:
            voided_bolume: Tensor -> Volume without the unhealthy and healthy regions 
        """
        if old_seg is not None:
            # In case we want to remove the original unhealthy seg
            complete_seg = complete_seg - old_seg
        voided_volume = self.__remove_region_from_volume__(complete_seg=complete_seg, volume=image)
        voided_volume = torch.unsqueeze(torch.unsqueeze(voided_volume, dim=0), dim=0).to(self.device)
        correct_new_seg = torch.from_numpy(complete_seg)
        correct_new_seg = torch.unsqueeze(torch.unsqueeze(correct_new_seg, dim=0), dim=0).to(self.device)
    
        return voided_volume, correct_new_seg
    
    

class Label_Generator:
    def __init__(self, path, device):
        self.path = path
        self.device = device

    def load_label_generator(self):
        """
        Loading the weights of the label generator
        """
        Label_G = L_G(noise=100)  
        Label_G.to(self.device)
        Label_G.load_state_dict(torch.load(self.path, map_location=torch.device(self.device)))
        return Label_G


    def __rezize_to_128__(self, fake_seg):
        """
        Resize to the original size 128
        """
        full_res_fake_seg = torch.nn.functional.interpolate(
            input=fake_seg, 
            size=(128,128,128), 
            scale_factor=None, 
            mode='nearest', 
            align_corners=None,
            recompute_scale_factor=None, 
            antialias=False)
        return full_res_fake_seg

    def _remove_background_(self, full_res_fake_seg):
        """
        Removes the borders, leaving only the label values and the background
        """
        def __threshold_at_one__(x):
            # threshold at 0
            return x > 0
        remover_background = monai.transforms.CropForeground(
            select_fn=__threshold_at_one__, 
            channel_indices=None, 
            margin=0, 
            allow_smaller=True, 
            return_coords=False, 
            k_divisible=1, 
            mode="constant", 
            lazy=False)

        cropped_full_res_fake_seg = remover_background(full_res_fake_seg)
        return cropped_full_res_fake_seg
        
    def get_WT_label(self, Label_G, Th):
        """
        Returns the WT label from the label generator
        """
        z_rand = Variable(torch.randn((1,100)), requires_grad=False)
        fake_seg = Label_G(z_rand)
        fake_seg = torch.unsqueeze(torch.unsqueeze(fake_seg[0][1], dim=0), dim=0)
        full_res_fake_seg = self.__rezize_to_128__(fake_seg=fake_seg)
        final_fake_seg = (full_res_fake_seg>Th).int()
        final_fake_seg = self._remove_background_(full_res_fake_seg=final_fake_seg)
        return final_fake_seg