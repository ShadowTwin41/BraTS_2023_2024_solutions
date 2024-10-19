import os
import argparse
import torch 
from torch.autograd import Variable
from monai.networks.nets import SwinUNETR
import pickle
from time import time
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path

# My imports
import sys
sys.path.append("./")
sys.path.append("../")
sys.path.append("../../")
#sys.path.insert(1, '/projects/brats2023_a_f/BRAINTUMOUR')
from src.utils.data_utils import get_loader


def save_ckp(state, checkpoint_dir):
    """
    To save the trained models
    """
    torch.save(state, checkpoint_dir)

def draw_curve(flag, list_iter, dic_loss, losses, colour, file_name, HOME_DIR):
    """
    Creating the plot of loss values of the generator
    """
    for idx, loss in enumerate(losses):
        plt.plot(list_iter, dic_loss[f'{loss}'], f'{colour[idx]}', label=f'{loss}') 
    if flag:
        plt.legend()
    flag = False
    plt.savefig(os.path.join(HOME_DIR, f'{file_name}.jpg'))
    return flag

def save_losses(args, loss_names, losses_lists, HOME_DIR):
    """
    To save all the loss values is txt files
    """
    HOME_DIR = os.path.join(HOME_DIR, f"baseline_{args.modality}", "loss_lists")
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

def create_dirs(args, HOME_DIR):
    """ 
    Creating directories to save teh weights, loss lists and generated scans 
    """
    # Creating HOME_DIR
    if not os.path.exists(HOME_DIR):
        os.makedirs(HOME_DIR)
        print(f"Directory {HOME_DIR} created")
    else:
        print(f"Directory {HOME_DIR} already exists")
    # Creating weights dir
    if not os.path.exists(f"{HOME_DIR}/baseline_{args.modality}/weights"):
        os.makedirs(f"{HOME_DIR}/baseline_{args.modality}/weights")
        print(f"Directory {HOME_DIR}/baseline_{args.modality}/weights created")
    else:
        print(f"Directory {HOME_DIR}/baseline_{args.modality}/weights already exists")
    # Creating loss_lists dir
    if not os.path.exists(f"{HOME_DIR}/baseline_{args.modality}/loss_lists"):
        os.makedirs(f"{HOME_DIR}/baseline_{args.modality}/loss_lists")
        print(f"Directory {HOME_DIR}/baseline_{args.modality}/loss_lists created")
    else:
        print(f"Directory {HOME_DIR}/baseline_{args.modality}/loss_lists already exists")
    # Creating weights checkpoint_scans
    if not os.path.exists(f"{HOME_DIR}/baseline_{args.modality}/checkpoint_scans"):
        os.makedirs(f"{HOME_DIR}/baseline_{args.modality}/checkpoint_scans")
        print(f"Directory {HOME_DIR}/baseline_{args.modality}/checkpoint_scans created")
    else:
        print(f"Directory {HOME_DIR}/baseline_{args.modality}/checkpoint_scans already exists")
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
    nib.save(feat, f"{path}/baseline_{args.modality}/checkpoint_scans/{iter_num}_{reality}.nii.gz")

def train(args, global_step, train_loader, generator, G_optimizer, recon_loss, HOME_DIR):
    """
    Training function 
    """
    generator.train()

    # Lists to save losses
    loss_gen_list = []
    loss_recons_list = []
    geral_generator = []

    for step, batch in enumerate(train_loader):
        t1 = time()
        # Get data from the batch
        x_crop_pad = batch["scan_t1ce_crop_pad"].cuda() # Scan cropped and with padding to 96x96x96
        scan_t1ce_noisy = batch["scan_t1ce_noisy"].cuda() # Scan cropped, with padding to 96x96x96 and noise
        y_crop_pad = batch["label_crop_pad"].cuda() # Label cropped and with padding to 96x96x96

        ################################
        #       Update G network       #
        ################################
        generator.zero_grad() 

        # Compute L1 loss
        input_noise = torch.cat([scan_t1ce_noisy, y_crop_pad], dim=1) 
        scan_recon = generator(input_noise) # Recontruction using the Generator
        loss_recons = recon_loss(scan_recon, x_crop_pad)
        loss_gen = loss_recons
        # Calculate gradients for the Generator
        loss_gen.backward()
        G_optimizer.step()

        print("Step:{}/{}, Loss_gen:{:.4f}, Loss_recons:{:.4f}, Time:{:.4f}"
            .format(global_step, args.num_steps, loss_gen.item(), loss_recons.item(), time() - t1))
        
        # Append Generator losses to lists 
        loss_gen_list.append(loss_gen.item())
        loss_recons_list.append(loss_recons.item())        
        geral_generator.append(loss_gen.item())

        if global_step % 1000 == 0:
            checkpoint_gen = {
                    "global_step": global_step,
                    "state_dict": generator.state_dict(),
                    "G_optimizer": G_optimizer.state_dict(),
                }
            save_ckp(checkpoint_gen, f"{HOME_DIR}/baseline_{args.modality}/weights/generator_{global_step}.pt")
            print(f"Saved in: {HOME_DIR}/generator_{global_step}.pt")
            
        global_step += 1 # Add global_step
        
 
    return global_step, generator, G_optimizer, loss_gen_list, loss_recons_list, geral_generator, x_crop_pad, y_crop_pad, scan_t1ce_noisy, scan_recon

def __main__():
    parser = argparse.ArgumentParser(description="Tumour generator baseline Training")
    parser.add_argument("--logdir", default="test", type=str, help="Directory to save the experiment")
    parser.add_argument("--batch_size", default=2, type=int, help="Batch size")
    parser.add_argument("--num_workers", default=2, type=int, help="Number of workers for the data loader. Choose a value less than or equal to the number of CPU cores.")
    parser.add_argument("--in_channels", default=4, type=int, help="Number of input channels (Default 4 -> 3 channel label + 1 channel scan with noise)")
    parser.add_argument("--out_channels", default=1, type=int, help="Number of output channels (Default 1 -> Scan with reconstructed tumour)")
    parser.add_argument("--feature_size", default=48, type=int, help="Feature size")
    parser.add_argument("--use_checkpoint", action="store_true", help="Use gradient checkpointing to save memory")
    parser.add_argument("--optim_lr", default=2e-4, type=float, help="Optimization learning rate (0.0002 recommended)")
    parser.add_argument("--reg_weight", default=1e-5, type=float, help="Regularization weight")
    parser.add_argument("--num_steps", default=100000, type=int, help="Number of training iterations")
    parser.add_argument("--resume_iter", default=None, type=str, help="Iteration number to resume training")
    parser.add_argument("--noise_type", default="saltnpepper", type=str, help="Type of noise to mask the tumour: saltnpepper or gaussian")
    parser.add_argument("--add_tumour_loss", default="False", type=str, help="Add tumour MSE loss")
    parser.add_argument("--tumour_size", default="all", type=str, help="What dataset to use. Selected dataset with tumour <64 (64), <96 (96), or all ")
    parser.add_argument("--modality", default="t1ce", type=str, help="What modality to train: t1, t2, t1ce, flair")
    parser.add_argument("--csv_path", default="", type=str, help="Path to the CSV with all cases to use when training")
    parser.add_argument("--dataset", type=str, help="What dataset and from what year. E.g. Brats_2023. Not prepared for versions before 2023 or other dataset")

    args = parser.parse_args()

    HOME_DIR = f"../../Checkpoint/{args.logdir}"
    create_dirs(args=args, HOME_DIR=HOME_DIR)

    if ("2024" in args.dataset.lower()) and ("goat" in args.dataset.lower()) and ("brats" in args.dataset.lower()):
        args.dataset = "BRATS_GOAT_2024"
    elif "2024" in args.dataset.lower() and "brats" in args.dataset.lower() and ("goat" not in args.dataset.lower()):
        args.dataset = "BRATS_2024"
    elif "2023" in args.dataset.lower() and "brats" in args.dataset.lower():
        args.dataset = "BRATS_2023"
    else:
        raise ValueError("The dataset must be from BraTS, BRATS_GOAT_2024, BRATS_2024, BRATS_2023")

    ####### Defining the networks and optimisers #######
    generator = SwinUNETR(
        img_size=(96, 96, 96),
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        feature_size=args.feature_size,
        use_checkpoint=args.use_checkpoint,
        )
    generator.cuda()
    G_optimizer = torch.optim.AdamW(params=generator.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight, betas=(0.5, 0.999))

    global_step = 0
    epoch = 0
    train_loader = get_loader(args=args)
    
    ####### For resuming the training #######
    if args.resume_iter != None:
        global_step = int(args.resume_iter)
        model_pth = f"../../Checkpoint/{args.logdir}" # Path to the checkpoint folder  # Path to the checkpoint folder 
        print(f"Loading models from {model_pth}")
        generator_dict = torch.load(os.path.join(os.path.join(model_pth, f"baseline_{args.modality}", "weights"), f"generator_{args.resume_iter}.pt"))
        
        generator.load_state_dict(generator_dict["state_dict"])
        generator.epoch = generator_dict["global_step"]
        generator.optimizer = generator_dict["G_optimizer"]

        print("Pre-trained weights loaded")

    # Reconstruction loss
    recon_loss = torch.nn.L1Loss().cuda()

    # Saving losses (in txt files) to build the plots later
    dic_loss = {}  
    dic_loss['gen'] = []
    dic_loss['recons'] = []
    dic_loss['geral_gen'] = [] # For every iteration
   
    list_iter = []
    flag = True # To call the creation of the legend in the loss graph

    while global_step < args.num_steps:
        # Start training!
        epoch+=1
        global_step, generator, G_optimizer, loss_gen_list, loss_recons_list, geral_generator, x_crop_pad, y_crop_pad, scan_t1ce_noisy, scan_recon = train(
            args=args, global_step=global_step, train_loader=train_loader, generator=generator, G_optimizer=G_optimizer, recon_loss=recon_loss, HOME_DIR=HOME_DIR)

        # Saving a reconstruction and respective groundtruth per epoch 
        save_sample(args=args, image=x_crop_pad[0], reality="x_crop_pad", iter_num=global_step, path=HOME_DIR)
        save_sample(args=args, image=y_crop_pad[0], reality="y_crop_pad", iter_num=global_step, path=HOME_DIR, sum=True)
        save_sample(args=args, image=scan_t1ce_noisy[0], reality="scan_t1ce_noisy", iter_num=global_step, path=HOME_DIR)
        save_sample(args=args, image=scan_recon[0], reality="scan_recon", iter_num=global_step, path=HOME_DIR)

        # Adding losses to respective dict
        dic_loss['gen'].append(np.mean(loss_gen_list))
        dic_loss['recons'].append(np.mean(loss_recons_list))
        dic_loss['geral_gen'].append(geral_generator)
        loss_names = ["loss_gen_list", "loss_recons_list", "geral_generator"]
        losses_lists =  [dic_loss['gen'], dic_loss['recons'], dic_loss['geral_gen']]
        save_losses(args=args, loss_names=loss_names, losses_lists=losses_lists , HOME_DIR=HOME_DIR)
        list_iter.append(epoch)
        draw_curve(flag=flag, list_iter=list_iter, dic_loss=dic_loss, losses=(['gen']), colour=(['b-']), file_name=f"{args.modality}_train_baseline", HOME_DIR=HOME_DIR)
        flag = False # To call OFF the creation of the legend in the loss graph

if __name__ == "__main__":
    __main__()
        
        
