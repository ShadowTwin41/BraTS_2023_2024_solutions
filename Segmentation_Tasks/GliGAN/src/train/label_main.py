## %pylab inline
#start a timer
import time
start_time = time.time()
import argparse
import warnings

#import operating system 
import os
from os import listdir
from os.path import isfile, join

#To Show Images
import numpy as np
import nibabel as nib
from nilearn import plotting
import pickle
import matplotlib.pyplot as plt

# Import Pytorch
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch import autograd
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from pathlib import Path
#Monai (Import data and transform data)
from monai.transforms import \
    Compose, ScaleIntensity, ToTensor, Resize, RandRotate, RandFlip, RandScaleIntensity, RandZoom, RandGaussianNoise, RandAffine, ResizeWithPadOrCrop, EnsureChannelFirst
from monai.data import CacheDataset, ImageDataset

from monai.data.utils import pad_list_data_collate
from monai.data import CSVDataset
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    EnsureTyped,
    ScaleIntensityd,
    CopyItemsd,
    CropForegroundd,
    SpatialCropd,
    Resized,
    ToTensord,
    ResizeWithPadOrCropd,
)
from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms.transform import Transform
from monai.utils.enums import TransformBackends
from monai.config import DtypeLike, KeysCollection
from monai.transforms.transform import MapTransform
from collections.abc import Callable, Hashable, Mapping
# My imports
import sys
sys.path.append("./")
sys.path.append("../")
sys.path.append("../../")
from src.networks.LabelGAN import *
from src.utils.crop_label import CropLabel

def create_train_loader(args, CSV_PATH, BATCH_SIZE,  WORKERS, LABEL_TRANSFORM):
    """
    Get data loader.
    It downsamples the cropped labels to 64x64x64.
    """
    col_names = ['scan_t1ce', 'label', 'center_x', 'center_y', 'center_z', 'x_extreme_min', 'x_extreme_max', 'y_extreme_min', 'y_extreme_max', 'z_extreme_min', 'z_extreme_max', 'x_size', 'y_size', 'z_size']
    col_types= {'center_x': {'type': int}, 'center_y': {'type': int}, 'center_z': {'type': int}, 'x_extreme_min': {'type': int}, 'x_extreme_max': {'type': int}, 'y_extreme_min': {'type': int}, 'y_extreme_max': {'type': int}, 'z_extreme_min': {'type': int}, 'z_extreme_max': {'type': int}, 'x_size': {'type': int}, 'y_size': {'type': int}, 'z_size': {'type': int}}

    train_transforms = Compose([LoadImaged(keys=['label']),
                        EnsureChannelFirstd(keys=["label"]),
                        EnsureTyped(keys=["label"]),
                        # TODO uncomment if not found a solution around 
                        #ResizeWithPadOrCropd( # In principle this is not need for the Brats2023 and BratsGOAT2024, however the Brats2024 glioma requires this (original shape 182, 218, 182)...
                        #    keys=['label'],
                        #    spatial_size=(240,240,155),
                        #    mode="constant",
                        #    value=0,
                        #    lazy=False,
                        #),
                        LABEL_TRANSFORM(keys="label"), # 3 channel label brats2023 and goat <-> 4 channel brats2024 glioma <-> 1 channel brats meningioma
                        CropLabel(keys=["label"]),
                        Resized(keys=["label_crop_pad"], spatial_size=(64,64,64)),
                        ToTensord(keys=['label_crop_pad'], dtype="float32")
                    ])

    train_CSVdataset = CSVDataset(src=CSV_PATH, col_names=col_names, col_types=col_types)

    warnings.warn(f"The data loader will load all labels to memory. In case it fails due to lack of memory, reduce the 'cache_rate' in the function 'create_train_loader()'.")
    
    train_ds = CacheDataset( 
        data=train_CSVdataset, 
        transform=train_transforms,
        cache_rate=1, 
        copy_cache=False,
        progress=True,
        num_workers=WORKERS,
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=WORKERS, drop_last=True, shuffle=True, collate_fn=pad_list_data_collate)
    print(f"Number of element in the Train loader: {len(train_loader)}")
    return(train_loader)

def rescale_array(arr: np.ndarray, minv: float = 0.0, maxv: float = 1.0): #monmai function adapted
    """
    Rescale the values of numpy array `arr` to be from `minv` to `maxv`.
    """

    mina = torch.min(arr)
    maxa = torch.max(arr)

    if mina == maxa:
        return arr * minv

    norm = (arr - mina) / (maxa - mina)  # normalize the array first
    return (norm * (maxv - minv)) + minv  # rescale by minv and maxv, which is the normalized array by default

def nets(args, LATENT_DIM):
    G = Generator(noise=LATENT_DIM, out_channels=int(args.out_channels))
    CD = Code_Discriminator(code_size=LATENT_DIM, num_units=4096)
    D = Discriminator(in_channels=int(args.in_channels))
    
    # __TO_GPU___
    G.cuda()   #|
    CD.cuda()  #|
    D.cuda()   #|
    #__________#|
    
    E = Encoder(out_class=LATENT_DIM, in_channels=int(args.in_channels))
    E.cuda()
    #______________________OPTIMIZERS______________________torch.optim.AdamW Try this optimizer (instead of Adam).
    g_optimizer = optim.AdamW(G.parameters(), lr=0.0002)
    d_optimizer = optim.AdamW(D.parameters(), lr=0.0002)
    e_optimizer = optim.AdamW(E.parameters(), lr = 0.0002)
    cd_optimizer = optim.AdamW(CD.parameters(), lr = 0.0002)

    return (G,CD,D,E,g_optimizer,cd_optimizer,d_optimizer,e_optimizer)

def calc_gradient_penalty(model, x, x_gen, _EPS):
    #__________________________ WGAN-GP gradient penalty __________________________
    #calc_gradient_penalty(model, real_data, generated_data)
    assert x.size()==x_gen.size(), "real and sampled sizes do not match"
    alpha_size = tuple((len(x), *(1,)*(x.dim()-1)))    
    device = 'cuda' if x.is_cuda else 'cpu'
    alpha = torch.empty(*alpha_size, device=device, dtype=torch.float32).uniform_() 
    x_hat = x.data*alpha + x_gen.data*(1-alpha)
    x_hat = Variable(x_hat, requires_grad=True)

    def eps_norm(x):
        x = x.view(len(x), -1)
        return (x*x+_EPS).sum(-1).sqrt()
    def bi_penalty(x):
        return (x-1)**2

    grad_xhat = torch.autograd.grad(model(x_hat).sum(), x_hat, create_graph=True, only_inputs=True)[0]

    penalty = bi_penalty(eps_norm(grad_xhat)).mean()
    return penalty

def gdloss(real,fake):
    #_____________________Gradient difference loss function_____________________
    '''
    https://github.com/Y-P-Zhang/3D-GANs-pytorch/blob/master/models/losses.py
    '''
    dx_real = real[:, :, :, 1:, :] - real[:, :, :, :-1, :] 
    dy_real = real[:, :, 1:, :, :] - real[:, :, :-1, :, :]
    dz_real = real[:, :, :, :, 1:] - real[:, :, :, :, :-1]
    dx_fake = fake[:, :, :, 1:, :] - fake[:, :, :, :-1, :]
    dy_fake = fake[:, :, 1:, :, :] - fake[:, :, :-1, :, :]
    dz_fake = fake[:, :, :, :, 1:] - fake[:, :, :, :, :-1]
    gd_loss = torch.sum(torch.pow(torch.abs(dx_real) - torch.abs(dx_fake),2),dim=(2,3,4)) + \
              torch.sum(torch.pow(torch.abs(dy_real) - torch.abs(dy_fake),2),dim=(2,3,4)) + \
              torch.sum(torch.pow(torch.abs(dz_real) - torch.abs(dz_fake),2),dim=(2,3,4))
    # torch.pow(value,2)==(value)**
    
    return torch.sum(gd_loss)

def networks_params(D, Dtruth, CD, CDtruth, E, Etruth, G, Gtruth):
    for p in D.parameters():  
        p.requires_grad = Dtruth
    for p in CD.parameters():  
        p.requires_grad = CDtruth
    for p in E.parameters():  
        p.requires_grad = Etruth
    for p in G.parameters():  
        p.requires_grad = Gtruth
        
    return(D, CD, E, G)

def see_information(
    iteration,
    loss2,
    loss1,
    loss3,
    mse_loss,
    gd_loss,
    Lloss1, 
    Lloss2, 
    Lloss3, 
    MSELossL, 
    Gd_LossL,
    TOTAL_ITER):
    """
    Print and save loss information every iteration
    """
    #Save losses in lists
    Lloss1.append(float(loss1.data.cpu().numpy()))
    Lloss2.append(float(loss2.data.cpu().numpy()))
    Lloss3.append(float(loss3.data.cpu().numpy()))
    MSELossL.append(float(mse_loss.data.cpu().numpy()))
    Gd_LossL.append(float(gd_loss))
    print('[{}/{}]'.format(iteration,TOTAL_ITER),
                  f'D: {loss2.data.cpu().numpy():<8.3}', 
                  f'En_Ge: {loss1.data.cpu().numpy():<8.3}',
                  f'Code: {loss3.data.cpu().numpy():<8.3}',
                  f'MSE_Loss: {mse_loss.data.cpu().numpy():<8.3}',
                  f'Gd_Loss: {gd_loss:<8.3}',
                  )
    
    return(Lloss1,Lloss2,Lloss3,MSELossL,Gd_LossL)

def visualization(image,reality):
        feat = np.squeeze((0.5*image[0]+0.5).data.cpu().numpy())
        feat = nib.Nifti1Image(feat,affine = np.eye(4))
        plotting.plot_img(feat,title=reality)
        plotting.show()
        print(torch.max(image[0]))
        print(torch.min(image[0]))

def load_dict(HOME_DIR,start,G,CD,D,E,g_optimizer,cd_optimizer,d_optimizer,e_optimizer):
    """
    Loading the state of the model, optimizer, and iteration
    """
    print(f"Loading models from {HOME_DIR}. Iteration: {start}")
    G_dict = torch.load(os.path.join(os.path.join(HOME_DIR, "label", "weights"), f"G_iter_{start}.pt"))
    G.load_state_dict(G_dict["state_dict"])
    G.iteration = G_dict["iteration"]
    g_optimizer.load_state_dict(G_dict["g_optimizer"])

    CD_dict = torch.load(os.path.join(os.path.join(HOME_DIR, "label", "weights"), f"CD_iter_{start}.pt"))
    CD.load_state_dict(CD_dict["state_dict"])
    CD.iteration = CD_dict["iteration"]
    cd_optimizer.load_state_dict(CD_dict["cd_optimizer"])

    E_dict = torch.load(os.path.join(os.path.join(HOME_DIR, "label", "weights"), f"E_iter_{start}.pt"))
    E.load_state_dict(E_dict["state_dict"])
    E.iteration = E_dict["iteration"]
    e_optimizer.load_state_dict(E_dict["e_optimizer"])

    D_dict = torch.load(os.path.join(os.path.join(HOME_DIR, "label", "weights"), f"D_iter_{start}.pt"))
    D.load_state_dict(D_dict["state_dict"])
    D.iteration = D_dict["iteration"]
    d_optimizer.load_state_dict(D_dict["d_optimizer"])

    # usar ADAMW
    return(G,CD,D,E,g_optimizer,cd_optimizer,d_optimizer,e_optimizer)

def save_ckp(state, checkpoint_dir):
    """
    To save the trained models
    """
    torch.save(state, checkpoint_dir)

def save_losses(loss_names, losses_lists, HOME_DIR):
    """
    To save all the loss values is txt files
    """
    HOME_DIR = os.path.join(HOME_DIR, "label", "loss_lists")
    for index, loss in enumerate(loss_names):
        b=list()
        if not os.path.exists(HOME_DIR+"/"+loss+".txt"):
            Path(HOME_DIR+"/"+loss+".txt").touch()
        if os.stat(HOME_DIR+"/"+loss+".txt").st_size != 0:
            with open(HOME_DIR+"/"+loss+".txt", "rb") as fp:   # Unpickling
                b = pickle.load(fp)
                b.append(losses_lists[index][-1])
                fp.close()
        with open(HOME_DIR+"/"+loss+".txt", "wb") as fp:   #Pickling
            pickle.dump(b, fp)
        fp.close()

def create_dirs(HOME_DIR):
    """ 
    Creating directories to save the weights, loss lists and generated scans 
    """
    # Creating HOME_DIR
    if not os.path.exists(HOME_DIR):
        os.makedirs(HOME_DIR)
        print(f"Directory {HOME_DIR} created")
    else:
        print(f"Directory {HOME_DIR} already exists")
    # Creating weights dir
    if not os.path.exists(f"{HOME_DIR}/label/weights"):
        os.makedirs(f"{HOME_DIR}/label/weights")
        print(f"Directory {HOME_DIR}/label/weights created")
    else:
        print(f"Directory {HOME_DIR}/label/weights already exists")
    # Creating loss_lists dir
    if not os.path.exists(f"{HOME_DIR}/label/loss_lists"):
        os.makedirs(f"{HOME_DIR}/label/loss_lists")
        print(f"Directory {HOME_DIR}/label/loss_lists created")
    else:
        print(f"Directory {HOME_DIR}/label/loss_lists already exists")
    # Creating weights checkpoint_scans
    if not os.path.exists(f"{HOME_DIR}/label/checkpoint_scans"):
        os.makedirs(f"{HOME_DIR}/label/checkpoint_scans")
        print(f"Directory {HOME_DIR}/label/checkpoint_scans created")
    else:
        print(f"Directory {HOME_DIR}/label/checkpoint_scans already exists")
    print("## ALL dirs set ##")

def save_sample(args, image, reality, iter_num, path, sum=False):
    """
    Save a sample to visualise. 
    Note: The output file does not have the correct values.
    """
    if sum:
        image = torch.sum(a=image, axis=0)
        image = image.type(torch.float)  
    feat = np.squeeze((image).data.cpu().numpy())
    feat = nib.Nifti1Image(feat,affine = np.eye(4))
    nib.save(feat, f"{path}/label/checkpoint_scans/{iter_num}_{reality}.nii.gz")

def draw_curve(flag, loss_list, losses, colour, file_name, HOME_DIR):
                """
                Creating the plot of loss values of the generator and discriminator
                """
                for idx, loss in enumerate(losses):
                    plt.plot(range(len(loss_list)), loss_list, f'{colour[idx]}', label=f'{loss}') 
                if flag:
                    plt.ylabel('Loss') 
                    plt.xlabel('Iter(*1000)')
                    plt.legend()
                #flag = False
                plt.savefig(os.path.join(HOME_DIR, f'{file_name}.jpg'))
                plt.clf()  
                plt.close() 
                #return flag

def training(
    args, 
    resume, 
    start, 
    HOME_DIR,
    TOTAL_ITER, 
    LATENT_DIM, 
    train_loader, 
    CD_Z_HAT_WEIGHT,
    D_WEIGHT, 
    MSE_WEIGHT, 
    GD_WEIGHT, 
    X_LOSS2_WEIGHT, 
    GP_D_WEIGHT,
    X_LOSS3_WEIGHT, 
    GP_CD_WEIGHT,
    _EPS):
    """
    Main training function (GAN game).
    """

    criterion_mse = nn.MSELoss()
    Lloss1=list()
    Lloss2=list()
    Lloss3=list()
    MSELossL=list()
    Gd_LossL=list()
    flag=True
    
    g_iter = 1
    d_iter = 1
    cd_iter =1
    G,CD,D,E,g_optimizer,cd_optimizer,d_optimizer,e_optimizer = nets(args=args, LATENT_DIM=LATENT_DIM)
    if resume:
        G,CD,D,E,g_optimizer,cd_optimizer,d_optimizer,e_optimizer = load_dict(HOME_DIR,start,G,CD,D,E,g_optimizer,cd_optimizer,d_optimizer,e_optimizer)

    for iteration in range(start,TOTAL_ITER+1):
    
        ###############################################
        # Train Encoder - Generator 
        ###############################################
        D, CD, E, G = networks_params(D=D, Dtruth=False, CD=CD, CDtruth=False, E=E, Etruth=True, G=G, Gtruth=True)
        
        for iters in range(g_iter):
                G.zero_grad()
                E.zero_grad()
                real_images = next(iter(train_loader)) #next image
                real_images = real_images['label_crop_pad']
                _batch_size = real_images.size(0)
                with torch.no_grad():
                    real_images = real_images.cuda(non_blocking=True)#next image into de GPU
                    z_rand = torch.randn((_batch_size,LATENT_DIM)).cuda(non_blocking=True) #random vector

                z_hat = E(real_images).view(_batch_size,-1) #Output vector of the Encoder
                x_hat = G(z_hat) #Generation of an image using the encoder's output vector
                x_rand = G(z_rand) #Generation of an image using a random vector
                
                #Code discriminator absolute value of the encoder's output vector 
                cd_z_hat_loss = CD(z_hat).mean() #(considering the random vector as real) 
                
                #calculation of the discriminator loss
                d_real_loss = D(x_hat).mean() #of the output vector of the Encoder
                d_fake_loss = D(x_rand).mean() #of the random vector
                d_loss = d_fake_loss+d_real_loss
                
       
                #_____________Mean_Squared_Error_____________
                mse_loss=criterion_mse(x_hat,real_images) 

                #_____________Gradient_Different_Loss_____________
                gd_loss=gdloss(real_images,x_hat).item() #considering axis -> x,y,z

                #__________Generator_Loss__________

                # this loss function is based on this
                loss1 = -cd_z_hat_loss*CD_Z_HAT_WEIGHT - d_loss*D_WEIGHT + mse_loss*MSE_WEIGHT + gd_loss*GD_WEIGHT #correct

                if iters<g_iter-1:
                    loss1.backward()
                else:
                    loss1.backward(retain_graph=True)
                e_optimizer.step()
                g_optimizer.step()
                g_optimizer.step()

        ###############################################
        # Train Discriminator
        ###############################################
        D, CD, E, G = networks_params(D=D, Dtruth=True, CD=CD, CDtruth=False, E=E, Etruth=False, G=G, Gtruth=False)
        
        for iters in range(d_iter):
            d_optimizer.zero_grad()
            real_images = next(iter(train_loader)) #next image
            real_images = real_images['label_crop_pad']
            _batch_size = real_images.size(0)
             
            with torch.no_grad():
                    real_images = real_images.cuda(non_blocking=True)#next image into de GPU
                    z_rand = torch.randn((_batch_size,LATENT_DIM)).cuda(non_blocking=True)
            
            z_hat = E(real_images).view(_batch_size,-1) #Output vector of the Encoder
            x_hat = G(z_hat) #Generation of an image using the encoder's output vector
            x_rand = G(z_rand) #Generation of an image using a random vector
            
            #calculation of the discriminator loss (if it can distinguish between real and fake)
            x_loss2 = -2*D(real_images).mean()+D(x_hat).mean()+D(x_rand).mean() 
            
            #calculation of the gradient penalty
            gradient_penalty_r = calc_gradient_penalty(D,real_images.data, x_rand.data, _EPS=_EPS)
            gradient_penalty_h = calc_gradient_penalty(D,real_images.data, x_hat.data, _EPS=_EPS)
            
            #__________Discriminator_loss__________
            loss2 = x_loss2*X_LOSS2_WEIGHT + (gradient_penalty_r+gradient_penalty_h)*GP_D_WEIGHT
            loss2.backward(retain_graph=True)
            d_optimizer.step()

        ###############################################
        # Train Code Discriminator
        ###############################################
        D, CD, E, G = networks_params(D=D, Dtruth=False, CD=CD, CDtruth=True, E=E, Etruth=False, G=G, Gtruth=False)    
        
        for iters in range(cd_iter):
            cd_optimizer.zero_grad()
            with torch.no_grad():
                    #random vector (considered as real here)
                    z_rand = torch.randn((_batch_size,LATENT_DIM)).cuda(non_blocking=True)
                    
            #Gradient Penalty between randon vector and encoder's output vector
            gradient_penalty_cd = calc_gradient_penalty(CD,z_hat.data, z_rand.data, _EPS=_EPS) 
            
            x_loss3=-CD(z_rand).mean() + CD(z_hat).mean()
            
            #___________Code_Discriminator_Loss___________
            loss3 = x_loss3*X_LOSS3_WEIGHT + gradient_penalty_cd * GP_CD_WEIGHT
            loss3.backward(retain_graph=True)
            cd_optimizer.step()

        ###############################################
        # Visualization
        ###############################################
        Lloss1,Lloss2,Lloss3,MSELossL,Gd_LossL=see_information(
            iteration=iteration,
            loss2=loss2,
            loss1=loss1,
            loss3=loss3,
            mse_loss=mse_loss,
            gd_loss=gd_loss,     
            Lloss1=Lloss1,
            Lloss2=Lloss2, 
            Lloss3=Lloss3, 
            MSELossL=MSELossL, 
            Gd_LossL=Gd_LossL,
            TOTAL_ITER=TOTAL_ITER)

        ###############################################
        # Model Save
        ###############################################
        if (iteration)%1000 ==0:
            checkpoint_G = {
                    "iteration": iteration,
                    "state_dict": G.state_dict(),
                    "g_optimizer": g_optimizer.state_dict(),
                }
            save_ckp(checkpoint_G, f"{HOME_DIR}/label/weights/G_iter_{iteration}.pt")

            checkpoint_D = {
                    "iteration": iteration,
                    "state_dict": D.state_dict(),
                    "d_optimizer": d_optimizer.state_dict(),
                }
            save_ckp(checkpoint_D, f"{HOME_DIR}/label/weights/D_iter_{iteration}.pt")

            checkpoint_E = {
                    "iteration": iteration,
                    "state_dict": E.state_dict(),
                    "e_optimizer": e_optimizer.state_dict(),
                }
            save_ckp(checkpoint_E, f"{HOME_DIR}/label/weights/E_iter_{iteration}.pt")

            checkpoint_CD = {
                    "iteration": iteration,
                    "state_dict": CD.state_dict(),
                    "cd_optimizer": cd_optimizer.state_dict(),
                }
            save_ckp(checkpoint_CD, f"{HOME_DIR}/label/weights/CD_iter_{iteration}.pt")
            print(f"Saved checkpoints in {HOME_DIR}/label/weights")

            #Save losses in lists and print instant loss value
            loss_names=["Lloss1","Lloss2","Lloss3","MSELossL","Gd_LossL"]
            losses_lists=[Lloss1,Lloss2,Lloss3,MSELossL,Gd_LossL]
            save_losses(loss_names, losses_lists, HOME_DIR)
            save_sample(args=args, image=real_images[0], reality="real", iter_num=iteration, path=HOME_DIR, sum=True)
            save_sample(args=args, image=x_rand[0], reality="x_rand", iter_num=iteration, path=HOME_DIR, sum=True)
            # To save a jpg file with the loss plot
            draw_curve(flag=flag, loss_list=Lloss1, losses=(['gen_enc']), colour=(['b-']), file_name="label_gen_enc_loss", HOME_DIR=HOME_DIR)
            draw_curve(flag=flag, loss_list=Lloss2, losses=(['disc']), colour=(['r-']), file_name="label_disc_loss", HOME_DIR=HOME_DIR)
            draw_curve(flag=flag, loss_list=Lloss3, losses=(['code_disc']), colour=(['b-']), file_name="label_code_disc_loss", HOME_DIR=HOME_DIR)
            draw_curve(flag=flag, loss_list=MSELossL, losses=(['mse']), colour=(['r-']), file_name="label_mse_loss", HOME_DIR=HOME_DIR)
            draw_curve(flag=flag, loss_list=Gd_LossL, losses=(['gd']), colour=(['b-']), file_name="label_gd_loss", HOME_DIR=HOME_DIR)
   
        resume = True


def __main__():
    parser = argparse.ArgumentParser(description="Label generator Training")
    parser.add_argument("--logdir", default="test", type=str, help="Directory to save the experiment")
    parser.add_argument("--batch_size", default=2, type=int, help="Batch size")
    parser.add_argument("--num_workers", default=2, type=int, help="Number of workers for the data loader. Choose a value less than or equal to the number of CPU cores.")
    parser.add_argument("--in_channels", default=3, type=int, help="Number of input channels (Default 3 -> Brats 2023| Brats2024 should have 4)")
    parser.add_argument("--out_channels", default=1, type=int, help="Number of output channels (Default 1 -> Scan with reconstructed tumour)")
    parser.add_argument("--total_iter", default=200000, type=int, help="Number of training iterations")
    parser.add_argument("--resume_iter", default=None, type=str, help="Iteration number to resume training")
    parser.add_argument("--csv_path", default="", type=str, help="Path to the CSV with all cases to use when training")
    parser.add_argument("--dataset", default="", type=str, help="What dataset and from what year. E.g. Brats_2023. Not prepared for versions before 2023 or other dataset")
    parser.add_argument("--cd_z_hat_weight", default=1, type=int, help="Weight of code discriminator loss in the generator and encoder")
    parser.add_argument("--d_weight", default=1, type=float, help="Weight of discriminator loss in the generator and encoder")
    parser.add_argument("--mse_weight", default=100, type=float, help="Weight of mse loss in the generator and encoder")
    parser.add_argument("--gd_weight", default=1/100, type=float, help="Weight of gradient difference loss in the generator and encoder")
    parser.add_argument("--x_loss2_weight", default=1, type=float, help="Weight of discriminator loss in the discriminator ")
    parser.add_argument("--gp_d_weight", default=100, type=float, help="Weight of gradient penalty in the discriminator")
    parser.add_argument("--x_loss3_weight", default=1, type=float, help="Weight of code discriminator in the code discriminator")
    parser.add_argument("--gp_cd_weight", default=100, type=float, help="Weight of gradient penalty in the code discriminator")
    parser.add_argument("--latent_dim", default=100, type=int, help="Size of the latend dim (random input vector of the label generator)")
    args = parser.parse_args()

    #_______________________________________________Constants_______________________________________________
    HOME_DIR = f"../../Checkpoint/{args.logdir}"
    create_dirs(HOME_DIR=HOME_DIR)

    #Neural net
    BATCH_SIZE = args.batch_size # batch_size must be a divisor of the data set number.
    WORKERS = args.num_workers
    _EPS = 1e-15 #for calc_gradient_penalty
    TOTAL_ITER = args.total_iter 
    try:
        START = int(args.resume_iter)
    except:
        START = 0

    #Train_Weights
    #loss1 = -cd_z_hat_loss*CD_Z_HAT_WEIGHT - d_loss*D_WEIGHT + mse_loss*MSE_WEIGHT + gd_loss*GD_WEIGHT
    CD_Z_HAT_WEIGHT = args.cd_z_hat_weight
    D_WEIGHT = args.d_weight
    MSE_WEIGHT = args.mse_weight
    GD_WEIGHT = args.gd_weight
    #----------------------
    #loss2 = x_loss2*X_LOSS2_WEIGHT + (gradient_penalty_r+gradient_penalty_h)*GP_D_WEIGHT
    X_LOSS2_WEIGHT= args.x_loss2_weight
    GP_D_WEIGHT = args.gp_d_weight
    #----------------------
    #loss3 = x_loss3*X_LOSS3_WEIGHT + gradient_penalty_cd * GP_CD_WEIGHT
    X_LOSS3_WEIGHT = args.x_loss3_weight
    GP_CD_WEIGHT = args.gp_cd_weight
    #setting latent variable sizes
    LATENT_DIM = int(args.latent_dim)

    if args.csv_path == "":
        for file_name in os.listdir(f"../../Checkpoint/{args.logdir}"):
            if file_name.endswith("csv"):
                CSV_PATH = os.path.join(f"../../Checkpoint/{args.logdir}", file_name)
    else:
        CSV_PATH = args.csv_path
    print(f"CSV_PATH: {CSV_PATH}")

    if ("2024" in args.dataset.lower()) and ("goat" in args.dataset.lower()) and ("brats" in args.dataset.lower()):
        args.dataset = "BRATS_GOAT_2024"
    elif ("brats" in  args.dataset.lower()) and ("2024" in  args.dataset.lower()) and ("goat" not in  args.dataset.lower()) and ("meningioma" not in  args.dataset.lower()):
        args.dataset = "BRATS_2024"
    elif ("brats" in  args.dataset.lower()) and ("2023" in  args.dataset.lower()) and ("goat" not in  args.dataset.lower()) and ("meningioma" not in  args.dataset.lower()):
        args.dataset = "BRATS_2023"
    elif  ("brats" in  args.dataset.lower()) and ("meningioma" in  args.dataset.lower()):
        args.dataset = "BRATS_2024_MENINGIOMA"
    else:
        raise ValueError("The dataset must be from BraTS: BRATS_GOAT_2024, BRATS_2024, BRATS_2023, BRATS_2024_MENINGIOMA")

    # Selecting the correct label transform (some have 1 value, others 3 values and ohers 4 values)
    if args.dataset=="BRATS_2023" or args.dataset=="BRATS_GOAT_2024":
        if args.dataset=="BRATS_2023":
            print(f"Using dataset: BRATS_2023")
        else:
             print(f"Using dataset: BRATS_GOAT_2024")
        from src.utils.convert_to_multi_channel_based_on_brats_classes import ConvertToMultiChannelBasedOnBratsGliomaClasses2023d as LABEL_TRANSFORM
        if int(args.in_channels)!=3:
            print("YOU WILL HAVE AN ERROR IN THE DATA LOADER. Change in_channels to 3")
    elif args.dataset=="BRATS_2024":
        print(f"Using dataset: BRATS_2024")
        from src.utils.convert_to_multi_channel_based_on_brats_classes import ConvertToMultiChannelBasedOnBratsGliomaPosTreatClasses2024d as LABEL_TRANSFORM
        if int(args.in_channels)!=4:
            print("YOU WILL HAVE AN ERROR IN THE DATA LOADER. Change in_channels to 4")
    elif args.dataset=="BRATS_2024_MENINGIOMA":
        print(f"Using dataset: BRATS_2024_MENINGIOMA")
        from src.utils.convert_to_multi_channel_based_on_brats_classes import ConvertToMultiChannelBasedOnBratsMeningiomaClasses2024d as LABEL_TRANSFORM
        if int(args.in_channels)!=1:
            print("YOU WILL HAVE AN ERROR IN THE DATA LOADER. Change in_channels to 1")
    else:
        raise ValueError("The dataset must be from BraTS: BRATS_GOAT_2024, BRATS_2024, BRATS_2023 or BRATS_2024_MENINGIOMA")


    G, CD, D, E, g_optimizer, cd_optimizer, d_optimizer, e_optimizer = nets(
        args=args, 
        LATENT_DIM=LATENT_DIM)

    train_loader = create_train_loader(
        args=args, 
        CSV_PATH=CSV_PATH, 
        BATCH_SIZE=BATCH_SIZE, 
        WORKERS=WORKERS, 
        LABEL_TRANSFORM=LABEL_TRANSFORM
        )

    if START > 0:
        RESUME = True
    else:
        RESUME = False

    training(
        args=args, 
        resume=RESUME, 
        start=START, 
        HOME_DIR=HOME_DIR,
        TOTAL_ITER=TOTAL_ITER, 
        LATENT_DIM=LATENT_DIM, 
        train_loader=train_loader,
        CD_Z_HAT_WEIGHT=CD_Z_HAT_WEIGHT,
        D_WEIGHT=D_WEIGHT,
        MSE_WEIGHT=MSE_WEIGHT,
        GD_WEIGHT=GD_WEIGHT,
        X_LOSS2_WEIGHT=X_LOSS2_WEIGHT,
        GP_D_WEIGHT=GP_D_WEIGHT,
        X_LOSS3_WEIGHT=X_LOSS3_WEIGHT,
        GP_CD_WEIGHT=GP_CD_WEIGHT,
        _EPS=_EPS)
    print(f"Finished training label GAN")

if __name__ == "__main__":
    __main__()


