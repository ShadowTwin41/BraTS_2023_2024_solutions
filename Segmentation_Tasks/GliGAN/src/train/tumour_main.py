import os
import argparse
import torch 
import pickle
from time import time
import warnings
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path
from monai.networks.nets import SwinUNETR
from monai.networks.nets import AttentionUnet
from monai.networks.nets import UNet
# My imports
import sys
sys.path.append("./")
sys.path.append("../")
sys.path.append("../../")
from src.utils.data_utils import get_loader
from src.networks.Discriminator import Discriminator

def save_ckp(state, checkpoint_dir):
    """
    To save the trained models
    """
    torch.save(state, checkpoint_dir)

def load_dict(args, generator, G_optimizer, discriminator, D_optimizer):
        """
        Loading the state of the model, optimizer, iteration, geral epoch and epoch for the second step training
        """
        if args.resume_iter == None:
            raise Exception("Please provide the start Iteration to load the models with: --resume_iter")

        model_pth = f"../../Checkpoint/{args.logdir}" # Path to the checkpoint folder 
        print(f"Loading models from {model_pth}")
        generator_dict = torch.load(os.path.join(os.path.join(model_pth, args.modality, "weights"), f"generator_{args.resume_iter}.pt"))
        discriminator_dict = torch.load(os.path.join(os.path.join(model_pth, args.modality, "weights"), f"discriminator_{args.resume_iter}.pt"))
        # Generator
        generator.load_state_dict(generator_dict["state_dict"])
        generator.global_step = generator_dict["global_step"]
        generator.optimizer = generator_dict["G_optimizer"]
        G_optimizer.load_state_dict(generator_dict["G_optimizer"])
        generator.epoch = generator_dict["epoch"]
        generator.second_step_epoch = generator_dict["second_step_epoch"]
        # Discriminator
        discriminator.load_state_dict(discriminator_dict["state_dict"])
        discriminator.global_step = discriminator_dict["global_step"]
        discriminator.optimizer = discriminator_dict["D_optimizer"]
        D_optimizer.load_state_dict(discriminator_dict["D_optimizer"])
        discriminator.epoch = discriminator_dict["epoch"]
        discriminator.second_step_epoch = discriminator_dict["second_step_epoch"]

        print("Pre-trained weights loaded")
        epoch = generator_dict["epoch"]
        second_step_epoch = generator_dict["second_step_epoch"]
        print(f"Resuming from epoch: {epoch}. Second step epoch: {second_step_epoch}. Starting global step: {discriminator.global_step}")
        return generator, G_optimizer, discriminator, D_optimizer, epoch, second_step_epoch
        
def get_nets(args):
        """
        Defining the networks and optimisers
        """
        if args.generator_type == "SwinUNETR":
            print("Using SwinUNETR Generator")
            generator = SwinUNETR(
                img_size=(96, 96, 96), # 96, 96, 96 originaly
                in_channels=args.in_channels,
                out_channels=args.out_channels,
                feature_size=args.feature_size,
                use_checkpoint=args.use_checkpoint,
                )

        elif args.generator_type == "AttentionUnet":
            print("Using AttentionUnet Generator")
            generator = AttentionUnet(
                spatial_dims=3, 
                in_channels=args.in_channels, 
                out_channels=args.out_channels, 
                #channels=(64, 128, 256),  
                channels=(48, 96, 192, 384, 768),
                strides=(2,2,2,2,1), 
                kernel_size=3, 
                up_kernel_size=3, 
                dropout=0.0)
            
        elif args.generator_type == "Unet":
            print("Using Unet Generator")
            generator = UNet(
                spatial_dims=3,
                in_channels=args.in_channels,
                out_channels=args.out_channels,
                channels=(48, 96, 192, 384, 768),
                strides=(2,2,2,1), 
            )
        else:
            raise Exception("Please choose a correct generator type: Transformers, AttentionUnet, Unet")
        generator.cuda()
        G_optimizer = torch.optim.AdamW(params=generator.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight, betas=(0.5, 0.999))

        print(f"Use Sigmoid:{args.use_sigmoid}")
        discriminator = Discriminator(args=args, use_sigmoid=args.use_sigmoid)
        discriminator.cuda()
        D_optimizer = torch.optim.AdamW(params=discriminator.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight, betas=(0.5, 0.999))

        return generator, G_optimizer, discriminator, D_optimizer

def draw_curve(flag, list_iter, dic_loss, losses, colour, file_name, HOME_DIR):
    """
    Creating the plot of loss values of the generator and discriminator
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
    HOME_DIR = os.path.join(HOME_DIR, args.modality, "loss_lists")
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
    if not os.path.exists(f"{HOME_DIR}/{args.modality}/weights"):
        os.makedirs(f"{HOME_DIR}/{args.modality}/weights")
        print(f"Directory {HOME_DIR}/{args.modality}/weights created")
    else:
        print(f"Directory {HOME_DIR}/{args.modality}/weights already exists")
    # Creating loss_lists dir
    if not os.path.exists(f"{HOME_DIR}/{args.modality}/loss_lists"):
        os.makedirs(f"{HOME_DIR}/{args.modality}/loss_lists")
        print(f"Directory {HOME_DIR}/{args.modality}/loss_lists created")
    else:
        print(f"Directory {HOME_DIR}/{args.modality}/loss_lists already exists")
    # Creating weights checkpoint_scans
    if not os.path.exists(f"{HOME_DIR}/{args.modality}/checkpoint_scans"):
        os.makedirs(f"{HOME_DIR}/{args.modality}/checkpoint_scans")
        print(f"Directory {HOME_DIR}/{args.modality}/checkpoint_scans created")
    else:
        print(f"Directory {HOME_DIR}/{args.modality}/checkpoint_scans already exists")
    print("## ALL dirs set ##")

def save_sample(args, image, reality, iter_num, path, label=False):
    """
    Save a sample to visualise. 
    Note: The output file does not have the correct values.
    """
    if label:
        try:
            image = image.float()
            new_image = torch.empty_like(image[0])
            TC = image[0]
            WT = image[1]
            ET = image[2]
            RC = image[3]
            NETC = TC - ET
            SNFH = WT - ET - NETC
            new_image[NETC>0] = 1
            new_image[SNFH>0] = 2
            new_image[image[3]>0] = 4 # RC
            new_image[image[2]>0] = 3 # ET
            image = new_image
        except:
            image = torch.sum(a=image, axis=0)
            image = image.type(torch.float)  
        
    feat = np.squeeze((image).data.cpu().numpy())
    feat = nib.Nifti1Image(feat,affine = np.eye(4))
    nib.save(feat, f"{path}/{args.modality}/checkpoint_scans/{iter_num}_{reality}.nii.gz")

def train(args, global_step, train_loader, generator, G_optimizer, recon_loss, w_progression, discriminator, D_optimizer, HOME_DIR, real_label=None, fake_label=None, criterion=None):
    """
    Training function 
    """
    generator.train()
    discriminator.train()

    # Lists to save losses
    loss_gen_list = []
    loss_recons_list = []
    loss_fake_G_list = [] 
    loss_dis_list = [] 
    loss_fake_list = []
    loss_real_list = []

    geral_generator = []
    geral_discriminator = []

    for step, batch in enumerate(train_loader):
        t1 = time()
    
        for D_n_i in range(args.D_n_update):
            #batch = next(iter(train_loader))
            # Get data from the batch
            x_crop_pad = batch["scan_t1ce_crop_pad"][0:args.batch_size].cuda() # Scan cropped and with padding to 96x96x96
            x_crop_noisy = batch["scan_t1ce_noisy"][0:args.batch_size].cuda() # Scan cropped, with padding to 96x96x96 and noise
            y_crop_pad = batch["label_crop_pad"][0:args.batch_size].cuda() # Label cropped and with padding to 96x96x96
            
            ############################
            #     Update D network     #
            ############################
            discriminator.zero_grad()

            ########### REAL ###########
            # Discriminator's input REAL
            input_real = torch.cat([x_crop_pad, y_crop_pad], dim=1)

            ########### Fake ###########
            # Discriminator's input FAKE
            input_noise = torch.cat([x_crop_noisy, y_crop_pad], dim=1) 
            scan_recon = generator(input_noise) # Recontruction using the Generator
            input_fake = torch.cat([scan_recon, y_crop_pad], dim=1)
            
            if args.not_abs_value_loss=="True":
                out_real = discriminator(input_real).view(-1)
                label_real = torch.full(size=(args.batch_size,), fill_value=real_label, dtype=torch.float).cuda()
                loss_real = criterion(out_real, label_real)
                ########### Fake ###########
                out_fake = discriminator(input_fake).view(-1) 
                label_fake = torch.full(size=(args.batch_size,), fill_value=fake_label, dtype=torch.float).cuda()
                loss_fake = criterion(out_fake, label_fake)
                loss_dis = loss_real + loss_fake 
            else:
                #print("Absolut values")
                loss_real = discriminator(input_real).mean()
                loss_fake = discriminator(input_fake).mean()
                loss_dis = loss_fake - loss_real
            
            # Update Discriminator
            loss_dis.backward()
            D_optimizer.step()

        for G_n_i in range(args.G_n_update):
            #batch = next(iter(train_loader))
            # Get data from the batch
            x_crop_pad = batch["scan_t1ce_crop_pad"][args.batch_size:].cuda() # Scan cropped and with padding to 96x96x96
            x_crop_noisy = batch["scan_t1ce_noisy"][args.batch_size:].cuda() # Scan cropped, with padding to 96x96x96 and noise
            y_crop_pad = batch["label_crop_pad"][args.batch_size:].cuda() # Label cropped and with padding to 96x96x96
            
            ################################
            #       Update G network       #
            ################################
            generator.zero_grad() 

            # Compute L1 loss
            input_noise = torch.cat([x_crop_noisy, y_crop_pad], dim=1) 
            scan_recon = generator(input_noise) # Recontruction using the Generator
            loss_recons = recon_loss(scan_recon, x_crop_pad)
            
            input_fake = torch.cat([scan_recon, y_crop_pad], dim=1)

            # The Discriminator was updated, so computing new loss 
            if args.not_abs_value_loss=="True":
                out_fake = discriminator(input_fake).view(-1) 
                label_real_2 = torch.full(size=(args.batch_size,), fill_value=real_label, dtype=torch.float).cuda()
                loss_fake_G = criterion(out_fake, label_real_2)
                loss_gen = loss_recons*w_progression + loss_fake_G/w_progression
            else:
                loss_fake_G = discriminator(input_fake).mean()
                loss_fake_G = (loss_fake_G)
                loss_gen = loss_recons*w_progression - loss_fake_G/w_progression

            # Calculate gradients for the Generator
            loss_gen.backward()
            G_optimizer.step()

        print("Step:{}/{}, Loss_gen:{:.4f}, Loss_recons:{:.4f}, Loss_fake_gen:{:.4f}, Loss_dis:{:.4f}, Loss_fake:{:.4f}, Loss_real:{:.4f}, Time:{:.4f}"
                .format(global_step, args.num_steps, loss_gen.item(), loss_recons.item(), loss_fake_G.item(), loss_dis.item(), loss_fake.item(), loss_real.item(), time() - t1))
        
        
        # Append Generator losses to lists 
        loss_gen_list.append(loss_gen.item())
        loss_recons_list.append(loss_recons.item())
        loss_fake_G_list.append(loss_fake_G.item()) 

        geral_generator.append(loss_gen.item())

        # Append Discriminator losses to lists
        loss_dis_list.append(loss_dis.item()) 
        loss_fake_list.append(loss_fake.item()) 
        loss_real_list.append(loss_real.item())
        geral_discriminator.append(loss_dis.item())
        
        if global_step == args.num_steps:
            break
        global_step += 1 # Add global_step
      

        
        
    return global_step, generator, G_optimizer, loss_gen_list, loss_recons_list, loss_fake_G_list, loss_dis_list, loss_fake_list, loss_real_list, geral_generator, geral_discriminator, x_crop_pad, y_crop_pad, x_crop_noisy, scan_recon

def __main__():
    parser = argparse.ArgumentParser(description="Tumour Generator Training")
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
    parser.add_argument("--not_abs_value_loss", default="False", type=str, help="Whether the loss is calculated using the BCE (True) or the absolute value (False). Note that the sigmoid must be used in the discriminator")
    parser.add_argument("--use_sigmoid", default="False", type=str, help="Add Sigmoid as last layer of the Discriminator")
    parser.add_argument("--use_gp", default="False", type=str, help="Using Gradient Penalty from WGAN-GP")
    parser.add_argument("--noise_type", default="gaussian_tumour", type=str, help="Type of noise to mask the tumour: gaussian_tumour, gaussian_extended, gaussian_tumour_v0.5, gaussian_tumour_effect")
    parser.add_argument("--add_tumour_loss", default="False", type=str, help="Add tumour MSE loss")
    parser.add_argument("--G_n_update", default=1, type=int, help="Number of times that the Generator is updated per iteration")
    parser.add_argument("--D_n_update", default=1, type=int, help="Number of times that the Discriminator is updated per iteration") 
    parser.add_argument("--generator_type", default="SwinUNETR", type=str, help="Generator type: SwinUNETR, AttentionUnetm Unet")
    parser.add_argument("--w_loss_recons", default=1, type=int, help="Reconstruction Loss Component Weight") 
    parser.add_argument("--l1_w_progressing", default="False", type=str, help="If progressively want to increase the weight of L1 loss in the loss function. Activates the second step of training to remove the noise around of the visible volume.")
    parser.add_argument("--modality", default="t1ce", type=str, help="What modality to train: t1, t2, t1ce, flair")
    parser.add_argument("--csv_path", default="", type=str, help="Path to the CSV with all cases to use when training")
    parser.add_argument("--dataset", type=str, help="What dataset and from what year. E.g. Brats_2023. Not prepared for versions before 2023 or other dataset")
    args = parser.parse_args()

    HOME_DIR = f"../../Checkpoint/{args.logdir}"
    create_dirs(args, HOME_DIR=HOME_DIR)

    print(f"The Generator is updated {args.G_n_update} times and the Discriminator {args.D_n_update} times per iteration")
    print(f"Reconstruction Loss Component Weight: {args.w_loss_recons}")

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
    
    global_step = 0 # Default value (it changes if resumed)
    epoch = 0 # Default value (it changes if resumed)
    second_step_epoch = 0 # Default value (it changes if resumed)

    ####### Defining the networks and optimisers #######
    generator, G_optimizer, discriminator, D_optimizer = get_nets(args)

    ####### For resuming the training #######
    if args.resume_iter is not None:
        global_step = int(args.resume_iter)
        generator, G_optimizer, discriminator, D_optimizer, epoch, second_step_epoch = load_dict(args, generator, G_optimizer, discriminator, D_optimizer)
    
    if second_step_epoch!=0:
        warnings.warn(f"Make sure you choose correctly the --w_loss_recons and --l1_w_progressing.\n In most cases, --w_loss_recons 100 and --l1_w_progressing True, so it corrects the noise created in the output (in the visible volume)")

    # Reconstruction loss
    recon_loss = torch.nn.L1Loss().cuda()

    if args.not_abs_value_loss=="True":
        print("NOT using absolute values")
        # Establish convention for real and fake labels during training
        real_label = 1.
        fake_label = 0.
        criterion = torch.nn.BCELoss()

    ###### Get train loader ######
    train_loader = get_loader(args=args)

    # Saving losses (in txt files) to build the plots later
    dic_loss = {}  
    dic_loss['gen'] = []
    dic_loss['recons'] = []
    dic_loss['fake_G'] = []
    dic_loss['dis'] = []
    dic_loss['fake'] = []
    dic_loss['real'] = []
    dic_loss['geral_gen'] = [] # For every iteration
    dic_loss['geral_dis'] = [] # For every iteration
   
    list_iter = []
    flag = True # To call the creation of the legend in the loss graph

    ########## Train loop ##########
    while global_step < args.num_steps:
        # Start training!
        epoch += 1
        # For the second step of training (correct the tissue already known)
        if args.l1_w_progressing=="True" and second_step_epoch<=1000: 
            second_step_epoch += 1
            w_progression = ((args.w_loss_recons-1)/1000)*second_step_epoch + 1 
            print(f"w_progression: {w_progression}")
            if w_progression > args.w_loss_recons:
                w_progression = args.w_loss_recons   
        else:
            w_progression = args.w_loss_recons

        if args.not_abs_value_loss=="True":
            pass
        else:
            real_label = None
            fake_label = None
            criterion = None
        
        global_step, generator, G_optimizer, loss_gen_list, loss_recons_list, loss_fake_G_list, loss_dis_list, loss_fake_list, loss_real_list, geral_generator, geral_discriminator, x_crop_pad, y_crop_pad, x_crop_noisy, scan_recon = train(
                args=args, global_step=global_step, train_loader=train_loader, generator=generator, G_optimizer=G_optimizer, recon_loss=recon_loss, w_progression=w_progression, discriminator=discriminator, D_optimizer=D_optimizer, real_label=real_label, fake_label=fake_label, criterion=criterion, HOME_DIR=HOME_DIR)
        # Saving a reconstruction and respective groundtruth per epoch 
        save_sample(args=args, image=x_crop_pad[0], reality="x_crop_pad", iter_num=epoch, path=HOME_DIR)
        save_sample(args=args, image=y_crop_pad[0], reality="y_crop_pad", iter_num=epoch, path=HOME_DIR, label=True)
        save_sample(args=args, image=x_crop_noisy[0], reality="scan_t1ce_noisy", iter_num=epoch, path=HOME_DIR)
        save_sample(args=args, image=scan_recon[0], reality="scan_recon", iter_num=epoch, path=HOME_DIR)

        # Adding losses to respective dict
        dic_loss['gen'].append(np.mean(loss_gen_list))
        dic_loss['recons'].append(np.mean(loss_recons_list))
        #
        dic_loss['fake_G'].append(np.mean(loss_fake_G_list))
        dic_loss['dis'].append(np.mean(loss_dis_list))
        dic_loss['fake'].append(np.mean(loss_fake_list))
        dic_loss['real'].append(np.mean(loss_real_list))
        #
        dic_loss['geral_gen'].append(geral_generator)
        dic_loss['geral_dis'].append(geral_discriminator)
        #
        losses_lists =  [dic_loss['gen'], dic_loss['recons'], dic_loss['fake_G'], dic_loss['dis'], dic_loss['fake'], dic_loss['real'], dic_loss['geral_gen'], dic_loss['geral_dis']]
        loss_names = ["loss_gen_list", "loss_recons_list", "loss_fake_G_list", "loss_dis_list", "loss_fake_list", "loss_real_list", "geral_generator", "geral_discriminator"]
        #
        save_losses(args=args, loss_names=loss_names, losses_lists=losses_lists , HOME_DIR=HOME_DIR)
        list_iter.append(epoch)
        draw_curve(flag=flag, list_iter=list_iter, dic_loss=dic_loss, losses=(['gen', 'dis']), colour=(['b-', 'r-']), file_name=f"{args.modality}_tumour_train_loss", HOME_DIR=HOME_DIR)
        flag=False # To call OFF the creation of the legend in the loss graph
        
        ### Saving checkpoint ###
        if (epoch%10==0) or (global_step+1>=args.num_steps): # TODO change to 1000
            if global_step+1>=args.num_steps: 
                print(f"LAST SAVE. global_step: {global_step}")
                print("Please run the second step of the training if you find the visible region too noisy.\nIn case the visible region is good, you don't need to run the second step.")
            # Save the model every 10 epoch and save in the last iter
            checkpoint_gen = {
                    "global_step": global_step,
                    "epoch": epoch,
                    "second_step_epoch":second_step_epoch,
                    "state_dict": generator.state_dict(),
                    "G_optimizer": G_optimizer.state_dict(),
                }
            save_ckp(checkpoint_gen, f"{HOME_DIR}/{args.modality}/weights/generator_{global_step}.pt")

            print(f"Saved in: {HOME_DIR}/{args.modality}/weights/generator_{global_step}.pt")
            checkpoint_dis = {
                        "global_step": global_step,
                        "epoch": epoch,
                        "second_step_epoch":second_step_epoch,
                        "state_dict": discriminator.state_dict(),
                        "D_optimizer": D_optimizer.state_dict(),
                    }
            save_ckp(checkpoint_dis, f"{HOME_DIR}/{args.modality}/weights/discriminator_{global_step}.pt")
            print(f"Saved in: {HOME_DIR}/{args.modality}/weights/discriminator_{global_step}.pt")

if __name__ == "__main__":
    __main__()
        
        
