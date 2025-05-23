"""
A script for training a diffusion model to unconditional image generation.
"""

import argparse
import numpy as np
import random
import sys
import torch as th
import datetime

sys.path.append(".")
sys.path.append("..")

from guided_diffusion import (dist_util,
                              logger)
from guided_diffusion.c_bratsloader import c_BraTSVolumes
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (model_and_diffusion_defaults,
                                          create_model_and_diffusion,
                                          args_to_dict,
                                          add_dict_to_argparser)
from guided_diffusion.train_util import TrainLoop
from torch.utils.tensorboard import SummaryWriter


def main():
    args = create_argparser().parse_args()
    seed = args.seed
    th.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    summary_writer = None
    if args.use_tensorboard:
        
        if args.tensorboard_path:
            logdir = args.tensorboard_path
        now = datetime.datetime.now()
        year = now.year
        month = now.month
        day = now.day
        hour = str(now.hour).zfill(2)
        minute = str(now.minute).zfill(2)
        second = str(now.second).zfill(2)
        logdir = f"./runs/{args.train_mode}_{day}_{month}_{year}_{hour}:{minute}:{second}"
        
        summary_writer = SummaryWriter(log_dir=logdir)
        summary_writer.add_text(
            'config',
            '\n'.join([f'--{k}={repr(v)} <br/>' for k, v in vars(args).items()])
        )
        logger.configure(dir=summary_writer.get_logdir())
    else:
        logger.configure()

    args.devices = [th.cuda.current_device()]
    dist_util.setup_dist(devices=args.devices)
    print(f"Devices: {args.devices}")

    logger.log("Creating model and diffusion...")
    arguments = args_to_dict(args, model_and_diffusion_defaults().keys())
    model, diffusion = create_model_and_diffusion(**arguments)

    # logger.log("Number of trainable parameters: {}".format(np.array([np.array(p.shape).prod() for p in model.parameters()]).sum()))
    model.to(dist_util.dev([0, 1]) if len(args.devices) > 1 else dist_util.dev())  # allow for 2 devices
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion,  maxt=1000)

    if args.dataset == 'brats':
        assert args.image_size in [128, 256], "We currently just support image sizes: 128, 256"
        datal, ds = c_BraTSVolumes(directory=args.data_dir, 
                          batch_size=args.batch_size,
                          num_workers=args.num_workers, 
                          mode='train',
                          img_size=args.image_size,
                          data_split_json=args.data_split_json).get_dl_ds()
    else:
        print("We currently just support the datasets: c_brats")

    logger.log("Start training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=datal,
        batch_size=args.batch_size,
        in_channels=args.in_channels,
        image_size=args.image_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        resume_step=args.resume_step,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        dataset=args.dataset,
        summary_writer=summary_writer,
        mode=args.train_mode,
        out_channels=args.out_channels,
        tumour_loss_weight=args.tumour_loss_weight,
    ).run_loop()


def create_argparser():
    defaults = dict(
        seed=0,
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,
        ema_rate="0.9999",
        log_interval=100,
        save_interval=5000,
        resume_checkpoint='',
        resume_step=0,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        dataset='brats',
        use_tensorboard=True,
        tensorboard_path='',  # set path to existing logdir for resuming
        devices=[0],
        dims=3,
        learn_sigma=False,
        num_groups=32,
        channel_mult="1,2,2,4,4",
        in_channels=8,
        out_channels=8,
        bottleneck_attention=False,
        num_workers=0,
        mode='default',
        renormalize=True,
        additive_skips=False,
        use_freq=False,
        data_split_json=None,
        train_mode=None,
        tumour_loss_weight=None,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
