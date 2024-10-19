import os
import csv
import numpy as np
from scipy import ndimage
import nibabel as nib
import argparse

def center_of_mass(mask_data):
    """
    Compute the center of mass of a binary mask
    Returns:
        x, y and z center of mass
    """
    x, y, z = ndimage.center_of_mass(mask_data) # This gives the center of mass 
    return round(x), round(y), round(z)

def x_extremes(matrix): 
    '''
    This function gives the extremes of the bounding box in the x axis.
    The min and max are the first and the last slice where the tumour label is non zero.
        Parameters:
                matrix (array): numpy array of the label
        Returns:
                min_x (int): x position of the first slice with non zero tumour voxel 
                max_x (int): x position of the last slice with non zero tumour voxel 
    '''
    min_x = 0
    max_x = 500
    for x_idx, x_slice in enumerate(matrix):
        if sum(sum(x_slice))>=1:
                min_x = x_idx+1
                break
    for x_idx, x_slice in enumerate((np.fliplr(matrix))):
       if sum(sum(x_slice))>=1:
        max_x = x_idx+1
    return min_x, max_x

def y_extremes(matrix): 
    '''
    This function gives the extremes of the bounding box in the y axis.
    The min and max are the first and the last slice where the tumour label is non zero.
        Parameters:
                matrix (array): numpy array of the label
        Returns:
                min_y (int): y position of the first slice with non zero tumour voxel 
                max_y (int): y position of the last slice with non zero tumour voxel 
    '''
    min_y = 0
    max_y = 500
    for y_idx, y_slice in enumerate(matrix.transpose(1,0,2)):
        if sum(sum(y_slice))>=1:
                min_y = y_idx+1
                break
    for y_idx, y_slice in enumerate((np.fliplr(matrix.transpose(1,0,2)))):
       if sum(sum(y_slice))>=1:
        max_y = y_idx+1
    return min_y, max_y

def z_extremes(matrix): 
    '''
    This function gives the extremes of the bounding box in the z axis.
    The min and max are the first and the last slice where the tumour label is non zero.
        Parameters:
                matrix (array): numpy array of the label
        Returns:
                min_z (int): z position of the first slice with non zero tumour voxel 
                max_z (int): z position of the last slice with non zero tumour voxel 
    '''
    min_z = 0
    max_z = 500
    for z_idx, z_slice in enumerate(matrix.transpose(2,1,0)):
        if sum(sum(z_slice))>=1:
                min_z = z_idx+1
                break
    for z_idx, z_slice in enumerate((np.fliplr(matrix.transpose(2,1,0)))):
       if sum(sum(z_slice))>=1:
        max_z = z_idx + 1
    return min_z, max_z

def get_training_dict(args, datadir):
    '''
    Creates a dictionary with the scans and the lables
        Parameters:
                datadir (str): path to the data directory
        Returns:
                training_dict (dict): dictionary with image:path and label:path
    '''
    training = []
    for sub_dir in os.listdir(datadir):
        images = []
        for file in os.listdir(os.path.join(datadir, sub_dir)):
            if file.endswith("seg.nii.gz") or file.endswith(args.seg_ending):
                label = os.path.join(datadir, sub_dir, file)
            else:
                images.append(os.path.join(datadir, sub_dir, file))
        dict_entry = {
            "image" : images,
            "label" : label
        }
        training.append(dict_entry)
    training_dict = {"training" : training}
    return training_dict

def modal_paths(args, mask_path):
                """
                Returns the path to the respective modal
                """
                scan_path_t1ce = None
                scan_path_t2 = None
                scan_path_flair = None
                scan_path_t1 = None
                for scan_paths in mask_path['image']: 
                    if args.t1c_ending in scan_paths:
                        scan_path_t1ce = scan_paths
                    elif args.t2w_ending in scan_paths:
                        scan_path_t2 = scan_paths
                    elif args.t2f_ending in scan_paths:
                        scan_path_flair = scan_paths
                    elif args.t1n_ending in scan_paths:
                        scan_path_t1 = scan_paths
                label_path = mask_path['label']
                return scan_path_t1ce, scan_path_t2, scan_path_flair, scan_path_t1, label_path

def create_csv(args, DATASET_NAME, CSV_PATH, DATADIR):
    """ 
    Creation of the complete CSV dataset with size smaller than or equal to 96 in all directions
    """
    # Define the header for the csv file
    header = ['id', 'scan_t1ce', 'scan_t2', 'scan_flair', 'scan_t1', 'label', 'center_x', 'center_y', 'center_z', 
            'x_extreme_min', 'x_extreme_max', 'y_extreme_min', 'y_extreme_max', 'z_extreme_min', 'z_extreme_max', 'x_size', 'y_size', 'z_size']

    # Getting all files in a folder DATADIR
    ### Create a dictionary with scans and labels paths
    training_dict = get_training_dict(args, DATADIR)
    training = training_dict['training']
    with open(CSV_PATH, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for mask_path in training:
            if ("brats" in DATASET_NAME.lower()) and ("goat" in DATASET_NAME.lower()) and ("2024" in DATASET_NAME.lower()):
                id = mask_path['label'].split("/")[-2][-5:] # For BraTS 2024 GOAT
            elif ("brats" in DATASET_NAME.lower()) and ("2023" in DATASET_NAME.lower()) and ("goat" not in DATASET_NAME.lower()) and ("meningioma" not in DATASET_NAME.lower()):
                id = mask_path['label'].split("/")[-2][-9:] # For BraTS 2023
            elif ("brats" in DATASET_NAME.lower()) and ("2024" in DATASET_NAME.lower()) and ("goat" not in DATASET_NAME.lower()) and ("meningioma" not in DATASET_NAME.lower()):
                id = mask_path['label'].split("/")[-2][-9:] # For BraTS 2024
            elif  ("brats" in DATASET_NAME.lower()) and ("meningioma" in DATASET_NAME.lower()):
                id = mask_path['label'].split("/")[-2][-6:] # For BraTS Meningioma Radiotherapy 2024
            else:
                raise ValueError("Datasets available: Brats2023, Brats_goat2024, Brats2024 or Brats2024_Meningioma")

            print(f"Doing case ID: {id}")
            # Load mask data
            mask = nib.load(mask_path['label'])
            mask_data = np.asarray(mask.get_fdata())
            # Binary mask
            mask_data = np.where(mask_data > 0.5, 1, 0)
            mask_data = mask_data > 0 
            # Dividing by modalities
            scan_path_t1ce, scan_path_t2, scan_path_flair, scan_path_t1, label_path = modal_paths(args, mask_path)

            ## These functions give the center of mass and extremes of x, y, z ##
            ### With this is possible to know the exact position of the center of the tumour and bounding box
            center_x, center_y, center_z = center_of_mass(mask_data)
            min_x, max_x = x_extremes(mask_data)
            min_y, max_y = y_extremes(mask_data)
            min_z, max_z = z_extremes(mask_data)
            x_size = max_x - min_x
            y_size = max_y - min_y
            z_size = max_z - min_z

            # Only cases with a size smaller than or equal to 96 in all directions are used.
            if x_size<=96 and y_size<=96 and  z_size<=96:
                # Save one line in the csv
                row = [id, scan_path_t1ce, scan_path_t2, scan_path_flair, scan_path_t1, label_path, center_x, center_y, center_z, min_x, max_x, min_y, max_y, min_z, max_z, x_size, y_size, z_size]
                writer.writerow(row)
            else:
                print(f"Case id: {id} was skipped (one dimention bigger than 96)")
        print(f"Done. Saved in {CSV_PATH}")
    return training

def __main__():
    parser = argparse.ArgumentParser(description="Label generator Training")
    parser.add_argument("--logdir", default="test", type=str, help="Directory to save the experiment (save CSV)")
    parser.add_argument("--dataset", type=str, help="What dataset and from what year. E.g. Brats_2023. Not prepared for versions before 2023 or other dataset")
    parser.add_argument("--datadir", type=str, help="Complete or relative path of the dataset")
    parser.add_argument("--debug", default="True", type=str, help="If want to show some output for debugging")
    parser.add_argument("--csv_path", default="", type=str, help="Path to the CSV with all cases to use when training")
    parser.add_argument("--seg_ending", default="seg.nii.gz", type=str, help="Ending to the segmentation file. Important to locate the correct segmentation file. Default: seg.nii.gz")
    parser.add_argument("--t1n_ending", default="t1n.nii.gz", type=str, help="Ending to the t1n file. Important to locate the correct t1n file. Default: t1n.nii.gz")
    parser.add_argument("--t1c_ending", default="t1c.nii.gz", type=str, help="Ending to the t1c file. Important to locate the correct t1c file. Default: t1c.nii.gz")
    parser.add_argument("--t2w_ending", default="t2w.nii.gz", type=str, help="Ending to the t2w file. Important to locate the correct t2w file. Default: t2w.nii.gz")
    parser.add_argument("--t2f_ending", default="t2f.nii.gz", type=str, help="Ending to the t2f file. Important to locate the correct t2f file. Default: t2f.nii.gz")
    args = parser.parse_args()

    # Making dir to save the csv
    HOME_DIR = f'../../Checkpoint/{args.logdir}'
    if not os.path.exists(HOME_DIR):
        os.makedirs(HOME_DIR)
        print(f"Directory {HOME_DIR} created")
    else:
        print(f"Directory {HOME_DIR} already exists")
    if not os.path.exists(f"{HOME_DIR}/debug"):
        os.makedirs(f"{HOME_DIR}/debug")
        print(f"Directory {HOME_DIR}/debug created")
    else:
        print(f"Directory {HOME_DIR}/debug already exists")

    if args.csv_path == "":
        CSV_PATH = f'../../Checkpoint/{args.logdir}/{args.logdir}.csv'
    else:
        CSV_PATH = args.csv_path
    print(f"CSV_PATH: {CSV_PATH}")

    training = create_csv(args=args, DATASET_NAME=args.dataset, CSV_PATH=CSV_PATH, DATADIR=args.datadir)

    
    if args.debug=="True":
        print("####################################")
        print(f"Output for debug")
        print(f"Number of cases in the training dict: {len(training)}")
        # Check CSV file
        import pandas as pd
        df = pd.read_csv(CSV_PATH)
        print(f"Number of rows in the csv file: {len(df)}")
        print(f"If the {len(training)}!={len(df)}, it is normal, as some cases have tumours bigger than 96.\nIn case you did not see any line saying 'Case id: id was skipped (one dimention bigger than 96)' something is wrong.")
        print(f"### Some rows of the csv file ###")
        print(df)

        # Getting some data from the dataframe
        # Get the scan list
        print(f"Path to the t1ce file: {df['scan_t1ce'][0]}")
        print(f"Path to the t2 file: {df['scan_t2'][0]}")
        print(f"Path to the falir file: {df['scan_flair'][0]}")
        print(f"Path to the t1 file: {df['scan_t1'][0]}")
        # Get the label
        print(f"Path to the label file: {df['label'][0]}")
        # Get the center
        print(f"Center of mass -> x: {df['center_x'][0]}, y: {df['center_y'][0]}, z: {df['center_z'][0]}")

        # Create an image with a sample (several slices)
        print(f"Creating an image with a sample (several slices)")
        import nibabel as nib
        import numpy as np
        import matplotlib.pyplot as plt

        def visualize_sample(idx, slice, types=('scan_t1ce','scan_t2', 'scan_flair', 'scan_t1')):
            plt.figure(figsize=(16, 5))
            for i, t in enumerate(types, 1):
                data = nib.load(df[t][idx])
                data = np.asarray(data.get_fdata())
                plt.subplot(1, 4, i)
                plt.imshow(data[:,:,slice], cmap='gray')
                plt.title(f'{t}', fontsize=16)
                plt.axis('off')
            plt.suptitle(f'idx: {idx}', fontsize=16)
            plt.savefig(f"{HOME_DIR}/debug/sample_{slice}.png", format='png')
            plt.close()
        
        if ("brats" in args.dataset.lower()) and ("meningioma" in args.dataset.lower()):
            types = ['scan_t1ce']
        else:
            types=('scan_t1ce','scan_t2', 'scan_flair', 'scan_t1')
        for slice in range (5):
            visualize_sample(idx=0, slice=100+slice*5, types=types)

if __name__ == "__main__":
    __main__()
    print("Finished!")
