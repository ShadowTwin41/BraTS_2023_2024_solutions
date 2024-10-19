# 👋 Faking_it team! BraTS submissions 🎬

## :technologist: [BraTS 2024 - Task 1 - Adult Glioma Post Treatment](https://www.synapse.org/Synapse:syn53708249/wiki/627500)

### 👨‍🎓 Introduction to the challenge

📚 The purpose of the 2024 BraTS subchallenge on post-treatment glioma is to develop an automated multi-compartment brain tumor segmentation algorithm for high and low grade diffuse gliomas on post-treatment MRI. Data and algorithms developed from this challenge can be used create a tools for objectively assessing residual tumor volume for treatment planning and outcome prediction.

💾 Multi-institutional routine post-treatment clinically-acquired multi-parametric MRI (mpMRI) scans of glioma, are used as the training, validation, and testing data for this year’s BraTS challenge. Data was contributed from seven different academic medical centers. All BraTS mpMRI scans are available as NIfTI files (.nii.gz) and describe a) native (T1) and b) post-contrast T1-weighted (T1Gd), c) T2-weighted (T2), and d) T2 Fluid Attenuated Inversion Recovery (FLAIR) volumes, and were acquired with different clinical protocols and various scanners from multiple data contributing institutions. The ground truth data was created after preprocessing, including co-registration to the same anatomical template, interpolation to the same resolution (1 mm ^3^ ), and skull stripping.

👩‍⚕ 👨‍⚕All the imaging datasets have been annotated manually, by one to four raters, following the same annotation protocol, and their annotations were approved by experienced neuroradiologists. Annotations comprise the enhancing tissue (ET — label 3), the surrounding non-enhancing FLAIR hyperintensity (SNFH) — label 2), the non-enhancing tumor core (NETC — label 1), and the resection cavity (RC - label 4) as described in the latest BraTS summarizing paper, except that the resection cavity has been incorporated subsequent to the paper's release.

**📊 The sub-regions considered for evaluation are the "enhancing tumor" (ET), the "tumor core" (TC), the "whole tumor" (WT), and "resection cavity" (RC)***.* **The submitted methods will be assessed using the Lesion-wise Dice Similarity Coefficient and the Lesion-wise Hausdorff distance (95%).**

### 💡 Solution - Improved Multi-Task Brain Tumour Segmentation with Synthetic Data Augmentation (to be published)

📖 Generative adversarial networks (GANs) is used to massively increase the amount of available samples for training three different deep learning models for brain tumour segmentation. The first model is the standard nnU-Net, the second is the Swin UNETR and the third is the MedNeXT. The entire pipeline is built on the nnU-Net implementation, except for the generation of the synthetic data. The use of convolutional algorithms and transformers is able to fill each other's knowledge gaps. Using the new metric, our best solution achieves the dice results 0.7557, 0.7868, 0.7053, 0.8704, 0.7500, 0.8734 and HD95: 34.59, 39.81, 56,97, 25.05; 35.61, 26.32 in the valdiation set for ET, NETC, RC, SNFH, TC, WT respectively.

#### 🤖🏥📑 Generative Adversarial Network - GliGAN

**📁 First, the folder structure should be as follows:**

1. GliGAN
   1. Checkpoint (**We provide these weights**)
      1. {args.logdir} (this directory ad sub directories will be created automatically)
         1. csv file (unless specified somewhere else).
         2. t1ce
         3. t1
         4. t2
         5. flair
         6. debug
   2. DataSet (Optional - The dataset can be somewhere else. Set the correct path when creating the csv file `--datadir`)
      1. Dataset name
   3. src
      1. infer
      2. networks
      3. train
      4. utils

**🤖 Pipeline overview of the GliGAN:**

![alt text](imgs/GANs-train.png "Title")

**🤖⚙️🏃‍♀️ To run the GliGAN training:**

1. Change to `GliGAN/src/train` directory.
2. **Create the csv file** - `python csv_creator.py --logdir brats2024 --dataset brats_2024 --datadir ../../DataSet/BraTS2024-BraTS-GLI-TrainingData --debug True`
3. **First step GliGAN training (Replace t1ce with t1, t2 or flair to train the other modalities)-** `python tumour_main.py --logdir brats2024 --batch_size 2 --num_workers 2 --in_channels 5 --out_channels 1 --optim_lr 0.0001 --num_steps 2000000 --reg_weight 0 --noise_type gaussian_extended --not_abs_value_loss True --use_sigmoid True --G_n_update 2 --D_n_update 1 --w_loss_recons 5 --modality t1ce --dataset brats_2024`
   1. **Resume training -** `python tumour_main.py --logdir brats2024 --batch_size 2 --num_workers 2 --in_channels 5 --out_channels 1 --optim_lr 0.0001 --num_steps 2000000 --reg_weight 0 --noise_type gaussian_extended --not_abs_value_loss True --use_sigmoid True --G_n_update 2 --D_n_update 1 --w_loss_recons 5 --modality t1ce --dataset Brats_2024 --resume_iter 10`
4. **Second step GliGAN training -** `python tumour_main.py --logdir brats2024 --batch_size 2 --num_workers 2 --in_channels 5 --out_channels 1 --optim_lr 0.0001 --num_steps 2000000 --reg_weight 0 --noise_type gaussian_extended --not_abs_value_loss True --use_sigmoid True --G_n_update 2 --D_n_update 1 --modality t1ce --dataset Brats_2024 --resume_iter 20 --w_loss_recons 100 --l1_w_progressing True`
5. **Label generator -** `python label_main.py --logdir brats2024 --batch_size 4 --num_worker 4 --in_channels 4 --out_channels 4 --total_iter 200000 --dataset Brats_2024`
   1. **Resume training -** `python label_main.py --logdir brats2024 --batch_size 4 --num_worker 4 --in_channels 4 --out_channels 4 --total_iter 200000 --dataset Brats_2024 --resume_iter 3`
6. **GliGAN baseline (optional) -** `python tumour_main_baseline.py --logdir brats2024 --batch_size 2 --num_workers 2 --in_channels 5 --out_channels 1 --optim_lr 0.0001 --num_steps 2000000 --reg_weight 0 --noise_type gaussian_extended --modality t1ce --dataset Brats_2024`

**🤔🩻 For inference:**

1. Change to `GliGAN/src/infer` directory.
2. `python main_random_label_random_dataset_generator_multiprocess.py --batch_size 1 --in_channels_tumour 5 --out_channels_tumour 1 --out_channels_label 4 --dataset brats2024 --g_t1ce_n 485300  --g_t1_n 457870 --g_t2_n 468420 --g_flair_n 462090 --g_label_n 200000 --latent_dim 100 --logdir brats2024 --num_process 1 --start_case 0 --end_case 100 --new_n 17`
   1. **Tip:** use `start_case` and `end_case` to split the inference process manually in distinct machines/nodes, by splitting the dataset. To use all dataset in same machine, don't set `--start_case` and `--end_case`.
   2. You can control how many cases are created per sample, by setting `--new_n`. The inference pipeline has a "patience limit", i.e. if it does not find a place for the tumour after several attempts, it moves on to the next case.

#### 🤖📈 Segmentation Networks

The MedNeXt is implemented in the nnUNet version 1 pipeline, therefore, `RESULTS_FOLDER` and `nnUNet_raw_data_base` need to be defined.

**💻 Create the env variables:**

* `export nnUNet_preprocessed="./nnUNet_preprocessed"`
* `export nnUNet_results="./nnUNet_results"`
* `export nnUNet_raw="./nnUNet_raw"`
* `export RESULTS_FOLDER="./nnUNet_results`"
* `export nnUNet_raw_data_base="./nnUNet_raw`"

**🤖⚙️🏃‍♀️ To use the same version as us:**

1. Go to the `nnUNet_install` and run `pip install -e .`
2. Go to the `mednext` and run `pip install -e .`

   1. mednext depends on the nnUNetv1 - `pip install nnunet`
3. Convert all data to the nnUNet format:

   1. Change the `nnUNet/Dataset_conversion.ipynb` correspondingly, and run it.
   2. Create the json file after converting the dataset.
4. Change to the folder `nnUNet`.
5. `nnUNetv2_plan_and_preprocess -d 242 --verify_dataset_integrity`

   1. Don't forget to create the dataset.json file (see  `example/dataset_2024_glioma.json`)
   2. Copy the `example/nnUNetPlans_2024_glioma.json` to the postprocessing folder, rename it to `nnUNetPlans.json`, and change the "`dataset_name`" to the correct name given to the dataset, e.g., `Dataset242_BraTS_2024_rGANs`.
6. `mednextv1_plan_and_preprocess -t 242 -pl3d ExperimentPlanner3D_v21_customTargetSpacing_1x1x1 -pl2d ExperimentPlanner2D_v21_customTargetSpacing_1x1x1`

   1. Don't forget to create the dataset.json file (see  `example/dataset_2024_glioma_mednext.json`)

**🤖⚙️🏃‍♀️ Run the training (run for all 5 folds):**

* [nnUNet (3D full resolution)](https://github.com/MIC-DKFZ/nnUNet) - `nnUNetv2_train 242 3d_fullres 0 -device cuda --npz --c`
* [Swin UNETR](https://arxiv.org/pdf/2201.01266) - `nnUNetv2_train 242 3d_fullres_SwinUNETR 0 -device cuda --npz -tr nnUNetTrainer_SwinUNETR --c`
  * **Warning**: It needs to be changed the network's number of channels of output:
    * Go to: `./nnUNet_install/nnunetv2/training/nnUNetTrainer/variants/network_architecture/nnUNetTrainer_SwinUNETR.py`
    * Change the `out_channels` to 4.
* [MedNeXt](https://github.com/MIC-DKFZ/MedNeXt) - `mednextv1_train 3d_fullres nnUNetTrainerV2_MedNeXt_L_kernel5 Task242_BraTS_2024_rGANs 0 -p nnUNetPlansv2.1_trgSp_1x1x1 [-c]`
  * -c to resume training

**Note:** The data split of the 5 folds was created randomly, but it was ensured that the validation set only contained real data. No synthetic data created using the case in the validation was in the training set (check `example/splits_final_2023_glioma.json` and `example/splits_final_2024_glioma_mednext.pkl`). ). `nnUNet/Data_splits.ipynb` contains the code necessary to make these changes for both MedNeXt and nnUNet.

In the end, 15 checkpoints should be trained (5 for each network).

#### 🤔🔍📈 Post-processing

Thresholding is used to reduce the number of FP. In previous editions, a threshold of 200 voxels was used by several winners, since several cases did not have enhancing tumour (ET label 3). However, this dataset will contain several differences between the training and the testing sets, therefore, the other regions, i.e., Whole Tumour (WT; labels 1, 2, 3), Tumour Core (TC; labels 1, 3) and Resection cavity (RC; label 4) would benefit from this strategy. By knowing this and also due to the use of lesion-wise metrics, a threshold for each region is used to remove small structures that could be interpreted as FP. This is done in the segmentation inference pipeline.

#### 🤖⚙️🏃‍♀️📈 Segmentation inference

In our submission we use the ensemble of 30 checkpoints:

1. Real data (3 networks * 5 folds) -> nnUNet ID 241
2. Real data +Synthetic data generated using synthetic labels (3 networks * 5 folds) -> nnUNet ID 242

To control what models are used for inference, go to `BraTS2024_inference/main.py` and choose the `ensemble_code` (`rGNg_rGSg_rGMg` by default, but in the original paper, we use `RMg_RNg_RSg_rGNg_rGSg_rGMg`). The post-processing by thresholding can also be changed in the same file, by changing `min_volume_threshold_WT = 50, min_volume_threshold_TC = 0, min_volume_threshold_ET = 0, min_volume_threshold_RC = 50`

* N - nnUNet (**We provide these weights**)
* S - Swin UNETR (**We provide these weights**)
* M - MedNeXt (**We provide these weights**)
* R - Real data (**We provide these weights**)
* rG - Real data + Synthetic data generated by the random label generator and GliGAN (**We provide these weights**)

**🤖⚙️🏃‍♀️📈 To run the segmentation inference:**

* Change to the directory `BraTS2024_inference`
* `python main.py --data_path ./in_data --output_path ./output --nnUNet_results ../nnUNet/nnUNet_results `
* Tip: check the `perform_inference_step` function in `infer_low_disk_glioma_mednext.py` to check if the nnUNet_results folders have the correct names.
* It will create two new directories:

  * `converted_dataset`
  * `inference`
* The final inference (post-processed), will be avaiable in the `--output_path`.

## 🏁 END
