# 👋 Faking_it team! BraTS submissions 🎬

## :technologist: [BraTS 2024 (ISBI) - GoAT - Generalizability Across Tumors Challenge](https://www.synapse.org/Synapse:syn52939291/wiki/624518)

### 👨‍🎓 Introduction to the challenge

📚 This task aims to challenge participants to create a segmentation algorithm capable of adapting and generalizing to different scenarios with little prior information and/or data on the target class(es). The aim is to simulate the clinical scenario where we develop a segmentation tool agnostic to future clinical applications (i.e., a tool trained on a specific disease(s) that will be applied to new ones without access to additional training data).

📊 Specifically, the candidate algorithms should be able to generalize across:

* Lesion types (i.e., different number of lesions per scan, lesion sizes, and locations in the brain).
* Institutions (i.e., different MRI scanners, acquisition protocols).
* Demographics (i.e., different age, sex, etc.).

📚 Additionally, lesions will differ in image characteristics; for example, some will miss or have limited necrotic core, edema, or contrast enhancement. So, while the segmentation mask labels will be consistent across disease types (i.e., the necrotic core, edema, and contrast enhancement masks will always have a consistent value), the presence of each label will vary in the training and validation/test data.

### 💡 Solution - [Generalisation of Segmentation Using Generative Adversarial Networks](https://ieeexplore.ieee.org/document/10635839)

📖 Our solution utilises state-of-the-art conditional generative adversarial networks to generate realistic new cases and train a segmentation algorithm that takes advantage of the convolutions and attention mechanisms. Our solution achieved a lesion-wise DSC of 0.855, 0.863, 0.883 and a lesion-wise HD95 value of 24.83, 24.10 and 21.72 for the enhancing tumour, the tumour core and the whole tumour in the validation set, respectively.

#### 🤖🏥📑 Generative Adversarial Network - GliGAN

**📁 First, the folder structure should be as follows:**

1. GliGAN
   1. Checkpoint (**We provide our trained weights**)
      1. {args.logdir} (this directory ad sub directories will be created automatically)
         1. csv file (unless specified somewhere else).
         2. t1ce
         3. t1
         4. t2
         5. flair
         6. label
         7. debug
   2. DataSet (Optional - The dataset can be somewhere else. Set the correct path when creating the csv file `--datadir`)
      1. Dataset name
   3. src
      1. infer
      2. networks
      3. train
      4. utils

**🤖 Pipeline overview of the GliGAN:**

![alt text](imgs/GANs-train.png "Title")

🤖⚙️🏃‍♀️ To run the GliGAN training

1. Change to `GliGAN/src/train` directory.
2. **Create the csv file** - `python csv_creator.py --logdir brats_goat_2024 --dataset Brats_goat_2024 --datadir ../../DataSet/ISBI2024-BraTS-GoAT-TrainingData  --debug True`
3. **First step GliGAN training (Replace t1ce with t1, t2 or flair to train the other modalities) -** `python tumour_main.py --logdir brats_goat_2024 --batch_size 2 --num_workers 2 --in_channels 4 --out_channels 1 --optim_lr 0.0001 --num_steps 2000000 --reg_weight 0 --noise_type gaussian_extended --not_abs_value_loss True --use_sigmoid True --G_n_update 2 --D_n_update 1 --w_loss_recons 5 --modality t1ce --dataset Brats_goat_2024`
   1. **Resume training -** `python tumour_main.py --logdir brats_goat_2024 --batch_size 2 --num_workers 2 --in_channels 4 --out_channels 1 --optim_lr 0.0001 --num_steps 2000000 --reg_weight 0 --noise_type gaussian_extended --not_abs_value_loss True --use_sigmoid True --G_n_update 2 --D_n_update 1 --w_loss_recons 5 --modality t1ce --dataset Brats_goat_2024 --resume_iter 10`
4. **Second step GliGAN training -** `python tumour_main.py --logdir brats_goat_2024 --batch_size 2 --num_workers 2 --in_channels 4 --out_channels 1 --optim_lr 0.0001 --num_steps 2000000 --reg_weight 0 --noise_type gaussian_extended --not_abs_value_loss True --use_sigmoid True --G_n_update 2 --D_n_update 1 --modality t1ce --dataset Brats_goat_2024 --resume_iter 20 --w_loss_recons 100 --l1_w_progressing True`
   1. Replace `resume_iter` value with the desired checkpoint (recomended 200000).
5. **Label generator -** `python label_main.py --logdir brats_goat_2024 --batch_size 4 --num_worker 4 --in_channels 3 --out_channels 3 --total_iter 200000 --dataset Brats_goat_2024`
   1. **Resume training -** `python label_main.py --logdir brats_goat_2024 --batch_size 4 --num_worker 4 --in_channels 3 --out_channels 3 --total_iter 200000 --dataset Brats_goat_2024 --resume_iter 1000`
6. **GliGAN baseline (optional) -** `python tumour_main_baseline.py --logdir brats_goat_2024 --batch_size 2 --num_workers 2 --in_channels 4 --out_channels 1 --optim_lr 0.0001 --num_steps 2000000 --reg_weight 0 --noise_type gaussian_extended --modality t1ce --dataset Brats_goat_2024`

**🤔🩻 For inference:**

1. Change to `GliGAN/src/infer` directory.
2. `python main_random_label_random_dataset_generator_multiprocess.py --batch_size 1 --in_channels_tumour 4 --out_channels_tumour 1 --out_channels_label 3 --dataset brats_goat_2024 --g_t1ce_n 402000  --g_t1_n 402000 --g_t2_n 314000 --g_flair_n 320000 --g_label_n 100000 --latent_dim 100 --logdir brats_goat_2024 --num_process 1 --start_case 0 --end_case 100 --new_n 17`
   1. **Tip:** use `start_case` and `end_case` to split the inference process manually in distinct machines/nodes, by splitting the dataset. To use all dataset in same machine, don't set `--start_case` and `--end_case`.
   2. You can control how many cases are created per sample, by setting `--new_n`. The inference pipeline has a "patience limit", i.e. if it does not find a place for the tumour after several attempts, it moves on to the next case.

#### 🤖📈 Segmentation Networks

Each network was implemented in the version 2 of the nnU-Net to take advantage of the pre-processing and data augmentation pipeline provided by this framework. Feel free to use the newest version of the nnUNet and include the missing networks (Swin UNETR, and the 2021 winner version).

**💻 Create the env variables:**

* `export nnUNet_preprocessed="./nnUNet_preprocessed"`
* `export nnUNet_results="./nnUNet_results"`
* `export nnUNet_raw="./nnUNet_raw"`

**🤖⚙️🏃‍♀️ To use the same version as us:**

1. Go to the `nnUNet_install` and run `pip install -e .`
2. Convert all data to the nnUNet format:

   1. Change the `nnUNet/Dataset_conversion.ipynb` correspondingly, and run it.
   2. Create the json file after converting the dataset.
3. Change to the folder `nnUNet`.
4. `nnUNetv2_plan_and_preprocess -d 240 --verify_dataset_integrity`

   1. Don't forget to create the dataset.json file (see  `example/dataset_BraTS_ISBI_GoAT_2024_rGANs.json`)
   2. Copy the `example/nnUNetPlans_ISBI_2024_goat.json` to the postprocessing folder, rename it to `nnUNetPlans.json`, and change the "`dataset_name`" to the correct name given to the dataset, e.g., `Dataset240_BraTS_ISBI_GoAT_2024_rGANs`.
5. Create the `data_split.json` as you prefer (let the nnUNet do it automatically or use the `nnUNet/Data_splits.ipynb` (recomended))

**🤖⚙️🏃‍♀️ Run the training (only the fold "all", which uses the entire dataset to train, without validation):**

* nnUNet (3D full resolution) [nnUNet (3D full resolution)](https://github.com/MIC-DKFZ/nnUNet) - `nnUNetv2_train 240 3d_fullres all -device cuda --npz --c`
* [Swin UNETR](https://arxiv.org/pdf/2201.01266) - `nnUNetv2_train 240 3d_fullres_SwinUNETR all -device cuda --npz -tr nnUNetTrainer_SwinUNETR --c`
* [2021 winner](https://arxiv.org/pdf/2112.04653) - `nnUNetv2_train 240 3d_fullres_BN_BS5_RBT_DS_BD_PS all -device cuda --npz -tr nnUNetTrainerBN_BS5_RBT_DS_BD_PS --c`

Only 3 checkpoints trained for this solution.

#### 🤔🔍📈 Post-processing

Thresholding is used to reduce the number of FP. In previous editions, a threshold of 200 voxels was used by several winners, since several cases did not have enhancing tumour (ET label 3). However, this dataset will contain several differences between the training and the testing sets, therefore, the other regions, i.e., Whole Tumour (WT; labels 1, 2, 3), Tumour Core (TC; labels 1, 3) would benefit from this strategy. By knowing this and also due to the use of lesion-wise metrics, a threshold for each region is used to remove small structures that could be interpreted as FP. This is done in the segmentation inference pipeline.

#### 🤖⚙️🏃‍♀️📈 Segmentation inference

In our submission we use the ensemble of 3 checkpoints:

1. Synthetic data generated using synthetic labels (3 networks) -> nnUNet ID 238 (**We provide these weights**)

To control what models are used for inference, go to `BraTS_ISBI_GoAT_2024_inference/main.py` and choose the `ensemble_code`. By default, `ensemble_code='rGB_rGS_rGL'`. The post-processing by thresholding can also be changed in the same file, by changing `min_volume_threshold_WT = 250, min_volume_threshold_TC = 150, min_volume_threshold_ET = 100`.

* B - nnUNet
* S - Swin UNETR
* L - Large nnUNet
* rG - Real data + Synthetic data generated by the random label generator and GliGAN

**🤖⚙️🏃‍♀️📈 To run the segmentation inference:**

* Change to the directory `BraTS_ISBI_GoAT_2024_inference`
* `python main.py --data_path ./in_data --output_path ./output --nnUNet_results ../nnUNet/nnUNet_results `
* Tip: check the `perform_inference_step` functions in `infer_low_disk.py` to check if the nnUNet_results folders have the correct names.
* It will create two new directories:

  * `converted_dataset`
  * `inference`
* The final inference (post-processed), will be avaiable in the `--output_path`.

## 🏁 END
