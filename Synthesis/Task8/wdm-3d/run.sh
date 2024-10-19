# general settings
#GPU=0;                    # gpu to use
SEED=42;                  # randomness seed for sampling
CHANNELS=64;              # number of model base channels (we use 64 for all experiments)
IN_CHANNELS=8;            # Number of channels in the input (8 after the wavelet transform)
MODE='c_train';             # c_train vs c_sample
TRAIN_MODE=Conditional_always_known_only_healthy; 
# Default (for the pre-training); 
# Conditional_always_known (for the conditional where the remaining volume does not have noise); 
# Conditional_default (trained as default+label_mse, and has the label as condition)
# Conditional_always_known_only_healthy (conditional, always known, without any unhealthy tissue, the region to generate has weight 10 and 1 for the overall. All losses are computed using full resolution scans.)
# Conditional_always_known_only_healthy_only_roi
# Conditional_always_known_only_healthy_stats_roi
DATASET='c_brats';          # c_brats
MODEL='ours_unet_256';    # 'ours_unet_256', 'ours_wnet_128', 'ours_wnet_256'

# settings for sampling/inference
ITERATIONS=3000;             # training iteration (as a multiple of 1k) checkpoint to use for sampling
SAMPLING_STEPS=5000;         # number of steps for accelerated sampling, 0 for the default TODO: 1000
NOISE_SCHEDULE=linear;       # linear(original) or cosine
STEPS_SCHEDULER=None; # None for default linear without changes, DPM_plus_plus_2M_Karras or cosine
RUN_DIR="runs/Conditional_always_known_only_healthy_31_7_2024_12:35:11";               # tensorboard dir to be set for the evaluation

# TRAIN
DATA_DIR=../../DataSet/ASNR-MICCAI-BraTS2023-Local-Synthesis-Challenge-Training;
DATA_SPLIT_JSON=./augmented_ASNR-MICCAI-BraTS2023-Local-Synthesis-Challenge-Training.json;

# SAMPLING
OUTPUT_DIR="results/Conditional_always_known_only_healthy_5000steps_linear_3000iter";
BEG=0;
END=end;

# VALIDATION
VALIDATION=False; 

if [[ $VALIDATION == 'True' ]]; then
  echo "DOING VALIDATION"
  BEG=$1; # 0 to start from the beginning of the list
  END=$2; # "end" to do everything until the end 
fi

USE_LABEL_COND=False; # True for conditional
USE_LABEL_COND_DILATED=False; # True for conditional 
USE_CONDITIONAL_MODEL=False; # True for conditional
LABEL_COND_WEIGHT=0; # 0 For Non conditional 

if [[ $TRAIN_MODE == 'Default' && $MODE == 'c_sample' ]]; then
  echo "Training_mode Default, mode c_sample";
  USE_LABEL_COND=True;
  USE_LABEL_COND_DILATED=False;
  USE_CONDITIONAL_MODEL=False;
fi

if [[ $TRAIN_MODE == 'Conditional_default' && $MODE == 'c_sample' ]]; then
  echo "Training_mode Conditional_default, mode c_sample";
  IN_CHANNELS=16;
  USE_LABEL_COND=True;
  USE_LABEL_COND_DILATED=False;
  USE_CONDITIONAL_MODEL=True;
fi

if [[ $TRAIN_MODE == 'Conditional_always_known' && $MODE == 'c_sample' ]]; then
  echo "Training_mode Conditional_always_known, mode c_sample";
  IN_CHANNELS=16;
  USE_LABEL_COND=True;
  USE_LABEL_COND_DILATED=False;
  USE_CONDITIONAL_MODEL=True;
fi

if [[ $TRAIN_MODE == 'Conditional_always_known_only_healthy' && $MODE == 'c_sample' ]]; then
  echo "Using Conditional_always_known_only_healthy";
  IN_CHANNELS=16;
  USE_LABEL_COND=True;
  USE_LABEL_COND_DILATED=False;
  USE_CONDITIONAL_MODEL=True;
fi

if [[ $TRAIN_MODE == 'Conditional_always_known' && $MODE == 'c_train' ]]; then
  echo "Using Conditional_always_known";
  IN_CHANNELS=16;
  LABEL_COND_WEIGHT=10;
  USE_LABEL_COND=True;
  USE_LABEL_COND_DILATED=True;
  USE_CONDITIONAL_MODEL=True;
fi

if [[ $TRAIN_MODE == 'Conditional_default' && $MODE == 'c_train' ]]; then
  echo "Using Conditional_default";
  IN_CHANNELS=16;
  LABEL_COND_WEIGHT=10;
  USE_LABEL_COND=True;
  USE_LABEL_COND_DILATED=False;
  USE_CONDITIONAL_MODEL=True;
fi

if [[ $TRAIN_MODE == 'Conditional_always_known_only_healthy' && $MODE == 'c_train' ]]; then
  echo "Using Conditional_always_known_only_healthy";
  IN_CHANNELS=16;
  LABEL_COND_WEIGHT=10;
  USE_LABEL_COND=True;
  USE_LABEL_COND_DILATED=False;
  USE_CONDITIONAL_MODEL=True;
fi

if [[ $TRAIN_MODE == 'Conditional_always_known_only_healthy_only_roi' && $MODE == 'c_train' ]]; then
  echo "Using Conditional_always_known_only_healthy_only_roi";
  IN_CHANNELS=16;
  LABEL_COND_WEIGHT=10;
  USE_LABEL_COND=True;
  USE_LABEL_COND_DILATED=False;
  USE_CONDITIONAL_MODEL=True;
fi

if [[ $TRAIN_MODE == 'Conditional_always_known_only_healthy_stats_roi' && $MODE == 'c_train' ]]; then
  echo "Using Conditional_always_known_only_healthy_stats_roi";
  IN_CHANNELS=16;
  LABEL_COND_WEIGHT=1;
  USE_LABEL_COND=True;
  USE_LABEL_COND_DILATED=False;
  USE_CONDITIONAL_MODEL=True;
fi

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
if [[ $MODE == 'c_sample' ]]; then
  echo "MODE: c_sample"
  BATCH_SIZE=1;
elif [[ $MODE == 'c_train' ]]; then
  if [[ $DATASET == 'c_brats' ]]; then
    echo "MODE: training";
    echo "DATASET: c_BRATS";
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
--rescale_learned_sigmas=False
--rescale_timesteps=False
--diffusion_steps=1000
--dims=3
--batch_size=${BATCH_SIZE}
--num_groups=32
--in_channels=${IN_CHANNELS}
--out_channels=8
--bottleneck_attention=False
--resample_2d=False
--renormalize=True
--additive_skips=${ADDITIVE_SKIP}
--use_freq=${USE_FREQ}
--predict_xstart=True
--num_workers=8
--image_size=${IMAGE_SIZE}
--data_split_json=${DATA_SPLIT_JSON}
--use_label_cond=${USE_LABEL_COND}
--use_label_cond_dilated=${USE_LABEL_COND_DILATED}
--use_conditional_model=${USE_CONDITIONAL_MODEL}
"

TRAIN="
--data_dir=${DATA_DIR}
--resume_checkpoint=
--resume_step=0
--use_fp16=False
--lr=1e-5
--diffusion_steps=1000
--noise_schedule=linear
--save_interval=5000
--label_cond_weight=${LABEL_COND_WEIGHT}
--mode=${MODE}
--train_mode=${TRAIN_MODE}
"
SAMPLE="
--data_dir=${DATA_DIR}
--data_mode=${DATA_MODE}
--seed=${SEED}
--use_fp16=False
--model_path=./${RUN_DIR}/checkpoints/${DATASET}_${ITERATIONS}000.pt
--output_dir=./results/${RUN_DIR}/${DATASET}_${MODEL}_${ITERATIONS}000/
--num_samples=1000
--use_ddim=False
--sampling_steps=${SAMPLING_STEPS}
--clip_denoised=True
--noise_schedule=${NOISE_SCHEDULE}
--steps_scheduler=${STEPS_SCHEDULER}
--mode=${MODE}
--train_mode=${TRAIN_MODE}
--validation=${VALIDATION}
--beg_case=${BEG}
--end_case=${END}
--output_dir=${OUTPUT_DIR}
"

# run the python scripts
if [[ $MODE == 'c_train' ]]; then
  python scripts/c_generation_train.py $TRAIN $COMMON;
else
  python scripts/c_generation_sample.py $SAMPLE $COMMON;
fi
