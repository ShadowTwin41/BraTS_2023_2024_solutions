# general settings
SEED=42;                  # randomness seed for sampling
CHANNELS=64;              # number of model base channels (we use 64 for all experiments)
MODE='train';             # train vs sample
DATASET='brats';          # brats 
MODEL='ours_unet_256';    # 'ours_unet_256', 'ours_wnet_128', 'ours_wnet_256'
TRAIN_MODE='known_3_to_gen_1'; # Training modes available: default or known_all_time or known_3_to_gen_1(recommended)
echo "TRAIN_MODE: ${TRAIN_MODE}"

# settings for sampling/inference
ITERATIONS=2000;             # training iteration (as a multiple of 1k) checkpoint to use for sampling
SAMPLING_STEPS=3000;         # number of steps for accelerated sampling, 0 for the default 1000. We used 3000 (very slow).
RUN_DIR="runs/known_3_to_gen_1_27_7_2024_16:16:14/";               # tensorboard dir to be set for the evaluation -> runs/.../

IN_CHANNELS=32; # 4 modalities * 8 channels(from the wavalet transform) 
OUT_CHANNELS=32; # 4 modalities * 8 channels(from the wavalet transform) 

if [[ $TRAIN_MODE == 'known_all_time' ]]; then
  IN_CHANNELS=36; # 4 modalities * 8 channels(from the wavalet transform) + 4 channels to indicate what modality to generate
  OUT_CHANNELS=32; # 4 modalities * 8 channels(from the wavalet transform) 
  TUMOUR_LOSS_WEIGHT=1;
fi

if [[ $TRAIN_MODE == 'known_3_to_gen_1' ]]; then
  IN_CHANNELS=36; # 4 modalities * 8 channels(from the wavalet transform) + 4 channels to indicate what modality to generate
  OUT_CHANNELS=8; # 1 modality of interest * * 8 channels(from the wavalet transform)
  TUMOUR_LOSS_WEIGHT=1;
fi

echo "IN_CHANNELS: ${IN_CHANNELS}"
echo "OUT_CHANNELS: ${OUT_CHANNELS}"

DATA_DIR=../../DataSet/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData;
DATA_SPLIT_JSON=../utils/BraTS2023-Missing_modal_training_data_split.json;

# detailed settings (no need to change for reproducing)
if [[ $MODEL == 'ours_unet_128' ]]; then
  echo "MODEL: WDM (U-Net) 128 x 128 x 128";
  CHANNEL_MULT=1,2,2,4,4;
  IMAGE_SIZE=128;
  ADDITIVE_SKIP=True;
  USE_FREQ=False;
  BATCH_SIZE=10;
elif [[ $MODEL == 'ours_unet_256' ]]; then
  echo "MODEL: WDM (U-Net) 256 x 256 x 256";
  CHANNEL_MULT=1,2,2,4,4,4;
  IMAGE_SIZE=256;
  ADDITIVE_SKIP=True;
  USE_FREQ=False;
  BATCH_SIZE=1;
elif [[ $MODEL == 'ours_wnet_128' ]]; then
  echo "MODEL: WDM (WavU-Net) 128 x 128 x 128";
  CHANNEL_MULT=1,2,2,4,4;
  IMAGE_SIZE=128;
  ADDITIVE_SKIP=False;
  USE_FREQ=True;
  BATCH_SIZE=10;
elif [[ $MODEL == 'ours_wnet_256' ]]; then
  echo "MODEL: WDM (WavU-Net) 256 x 256 x 256";
  CHANNEL_MULT=1,2,2,4,4,4;
  IMAGE_SIZE=256;
  ADDITIVE_SKIP=False;
  USE_FREQ=True;
  BATCH_SIZE=1;
else
  echo "MODEL TYPE NOT FOUND -> Check the supported configurations again";
fi

# some information and overwriting batch size for sampling
# (overwrite in case you want to sample with a higher batch size)
# no need to change for reproducing
if [[ $MODE == 'sample' ]]; then
  echo "MODE: sample"
  BATCH_SIZE=1;
elif [[ $MODE == 'train' ]]; then
  if [[ $DATASET == 'brats' ]]; then
    echo "MODE: training";
    echo "DATASET: BRATS";
  else
    echo "DATASET NOT FOUND -> Check the supported datasets again";
  fi
fi

COMMON="
--dataset=${DATASET}
--num_channels=${CHANNELS}
--class_cond=False
--num_res_blocks=2
--num_heads=1
--learn_sigma=False
--use_scale_shift_norm=False
--attention_resolutions=
--channel_mult=${CHANNEL_MULT}
--diffusion_steps=1000
--noise_schedule=linear
--rescale_learned_sigmas=False
--rescale_timesteps=False
--dims=3
--batch_size=${BATCH_SIZE}
--num_groups=32
--in_channels=${IN_CHANNELS}
--out_channels=${OUT_CHANNELS}
--bottleneck_attention=False
--resample_2d=False
--renormalize=True
--additive_skips=${ADDITIVE_SKIP}
--use_freq=${USE_FREQ}
--predict_xstart=True
--data_split_json=${DATA_SPLIT_JSON}
--num_workers=8
"
# resume_checkpoint -> path to the weight, e.g., ./runs/known_3_to_gen_1_27_7_2024_16:16:14/checkpoints/brats_2000000.pt
# resume_step -> same as the weight, e.g., 2000000
TRAIN="
--data_dir=${DATA_DIR}
--resume_checkpoint=
--resume_step=0
--image_size=${IMAGE_SIZE}
--use_fp16=False
--lr=1e-5
--train_mode=${TRAIN_MODE}
--save_interval=5000
--tumour_loss_weight=${TUMOUR_LOSS_WEIGHT}
"
SAMPLE="
--data_dir=${DATA_DIR}
--data_mode=${DATA_MODE}
--seed=${SEED}
--image_size=${IMAGE_SIZE}
--use_fp16=False
--model_path=./${RUN_DIR}/checkpoints/${DATASET}_${ITERATIONS}000.pt
--output_dir=./results/${RUN_DIR}/${DATASET}_${MODEL}_${ITERATIONS}000/
--num_samples=1000
--use_ddim=False
--sampling_steps=${SAMPLING_STEPS}
--clip_denoised=True
--train_mode=${TRAIN_MODE}
--mode=${MODE}
"

# run the python scripts
if [[ $MODE == 'train' ]]; then
  python scripts/generation_train.py $TRAIN $COMMON;
else
  python scripts/generation_sample.py $SAMPLE $COMMON;
fi
