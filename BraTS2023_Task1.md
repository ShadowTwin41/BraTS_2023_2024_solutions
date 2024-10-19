# 👋 Faking_it team! BraTS submissions 🎬

## :technologist: [BraTS 2023 - Task 1 - Adult Glioma Segmentation](https://www.synapse.org/Synapse:syn51156910/wiki/622351)

### 👨‍🎓 Introduction to the challenge

📚The **Brain Tumor Segmentation (BraTS) Adult Glioma Segmentation** seeks to identify the current, state-of-the-art segmentation algorithms for brain diffuse glioma patients and their sub-regions. Ample multi-institutional routine clinically-acquired multi-parametric MRI (mpMRI) scans of glioma, are used as the training, validation, and testing data for this year’s BraTS challenge.

💾 All BraTS mpMRI scans are available as NIfTI files (.nii.gz) and describe a) native (T1) and b) post-contrast T1-weighted (T1Gd), c) T2-weighted (T2), and d) T2 Fluid Attenuated Inversion Recovery (T2-FLAIR) volumes, and were acquired with different clinical protocols and various scanners from multiple data contributing institutions.

👩‍⚕ 👨‍⚕All the imaging datasets have been annotated manually, by one to four raters, following the same annotation protocol, and their annotations were approved by experienced neuro-radiologists. Annotations comprise the GD-enhancing tumor (ET — label 3), the peritumoral edematous/invaded tissue (ED — label 2), and the necrotic tumor core (NCR — label 1), as described in the latest BraTS summarizing paper. The ground truth data were created after their pre-processing, i.e., co-registered to the same anatomical template, interpolated to the same resolution (1 mm3) and skull-stripped.

**📊 The sub-regions considered for evaluation are the "enhancing tumor" (ET), the "tumor core" (TC), and the "whole tumor" (WT)****. The submitted methods will be assessed using the Lesion-wise Dice Similarity Coefficient and the Lesion-wise Hausdorff distance (95%).**

### 💡 Solution - [How we won BraTS 2023 Adult Glioma challenge? Just faking it! Enhanced Synthetic Data Augmentation and Model Ensemble for brain tumour segmentation.](https://arxiv.org/abs/2402.17317)

📖 Generative adversarial networks (GANs) and registration are used to massively increase the amount of available samples for training three different deep learning models for brain tumour segmentation. The first model is the standard nnU-Net, the second is the Swin UNETR and the third is the winning solution of the BraTS 2021 Challenge. The entire pipeline is built on the nnU-Net implementation, except for the generation of the synthetic data. The use of convolutional algorithms and transformers is able to fill each other's knowledge gaps. Using the new metric, our best solution achieves the dice results 0.9005, 0.8673, 0.8509 and HD95 14.940, 14.467, 17.699 (whole tumour, tumour core and enhancing tumour) in the validation set.

⚠️ On this page we will go through the generation of synthetic data using GANs. For the registration part, please have a look at the paper.

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

**🤖⚙️🏃‍♀️ To run the GliGAN training:**

1. Change to `GliGAN/src/train` directory.
2. **Create the csv file** - `python csv_creator.py --logdir brats2023 --dataset Brats2023 --datadir ../../DataSet/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData  --debug True`
3. **First step GliGAN training (Replace t1ce with t1, t2 or flair to train the other modalities) -** `python tumour_main.py --logdir brats2023 --batch_size 2 --num_workers 2 --in_channels 4 --out_channels 1 --optim_lr 0.0001 --num_steps 2000000 --reg_weight 0 --noise_type gaussian_extended --not_abs_value_loss True --use_sigmoid True --G_n_update 2 --D_n_update 1 --w_loss_recons 5 --modality t1ce --dataset Brats_2023`
   1. **Resume training -** `python tumour_main.py --logdir brats2023 --batch_size 2 --num_workers 2 --in_channels 4 --out_channels 1 --optim_lr 0.0001 --num_steps 2000000 --reg_weight 0 --noise_type gaussian_extended --not_abs_value_loss True --use_sigmoid True --G_n_update 2 --D_n_update 1 --w_loss_recons 5 --modality t1ce --dataset Brats_2023 --resume_iter 10`
4. **Second step GliGAN training -** `python tumour_main.py --logdir brats2023 --batch_size 2 --num_workers 2 --in_channels 4 --out_channels 1 --optim_lr 0.0001 --num_steps 2000000 --reg_weight 0 --noise_type gaussian_extended --not_abs_value_loss True --use_sigmoid True --G_n_update 2 --D_n_update 1 --modality t1ce --dataset Brats_2023 --resume_iter 20 --w_loss_recons 100 --l1_w_progressing True`
   1. Replace `resume_iter` value with the desired checkpoint (recomended 200000).
5. **Label generator -** `python label_main.py --logdir brats2023 --batch_size 4 --num_worker 4 --in_channels 3 --out_channels 3 --total_iter 200000 --dataset Brats_2023`
   1. **Resume training -** `python label_main.py --logdir brats2023 --batch_size 4 --num_worker 4 --in_channels 3 --out_channels 3 --total_iter 200000 --dataset Brats_2023 --resume_iter 1000`
6. **GliGAN baseline (optional) -** `python tumour_main_baseline.py --logdir brats2023 --batch_size 2 --num_workers 2 --in_channels 4 --out_channels 1 --optim_lr 0.0001 --num_steps 2000000 --reg_weight 0 --noise_type gaussian_extended --modality t1ce --dataset Brats_2023`

**🤔🩻For inference:**

1. Change to `GliGAN/src/infer` directory.
2. `python main_random_label_random_dataset_generator_multiprocess.py --batch_size 1 --in_channels_tumour 4 --out_channels_tumour 1 --out_channels_label 3 --dataset brats2023 --g_t1ce_n 400000  --g_t1_n 400000 --g_t2_n 400000 --g_flair_n 400000 --g_label_n 100000 --latent_dim 100 --logdir brats2023 --num_process 1 --start_case 0 --end_case 100 --new_n 17`
   1. **Tip:** use `start_case` and `end_case` to split the inference process manually in distinct machines/nodes, by splitting the dataset. To use all dataset in same machine, don't set `--start_case` and `--end_case`.
   2. You can control how many cases are created per sample, by setting `--new_n`. The inference pipeline has a "patience limit", i.e. if it does not find a place for the tumour after several attempts, it moves on to the next case.

#### 🤖📈Segmentation Networks

Each network was implemented in the version 2 of the nnU-Net to take advantage of the pre-processing and data augmentation pipeline provided by this framework. Feel free to use the newest version of the nnUNet and include the missing networks (Swin UNETR, and the 2021 winner version).

**💻 Create the env variables:**

* `export nnUNet_preprocessed="./nnUNet_preprocessed"`
* `export nnUNet_results="./nnUNet_results"`
* `export nnUNet_raw="./nnUNet_raw"`

**🤖⚙️🏃‍♀️To use the same version as us:**

1. Go to the `nnUNet_install` and run `pip install -e .`
2. Convert all data to the nnUNet format:

   1. Change the `nnUNet/Dataset_conversion.ipynb` correspondingly, and run it.
   2. Create the json file after converting the dataset.
3. Change to the folder `nnUNet`.
4. `nnUNetv2_plan_and_preprocess -d 232 --verify_dataset_integrity`

   1. Don't forget to create the dataset.json file (see example/dataset_2023_glioma.json).
   2. Copy the `nnUNetPlans_2023_glioma.json` to the postprocessing folder, rename it to `nnUNetPlans.json`, and change the "`dataset_name`" to the correct name given to the dataset, e.g., `Dataset242_BraTS_2024_rGANs`.
5. Create the `data_split.json` as you prefer (let the nnUNet do it automatically or use the `nnUNet/Data_splits.ipynb` (recomended))

**🤖⚙️🏃‍♀️Run the training (run for all 5 folds):**

* nnUNet (3D full resolution) [nnUNet (3D full resolution)](https://github.com/MIC-DKFZ/nnUNet) - `nnUNetv2_train 232 3d_fullres 0 -device cuda --npz --c`
* [Swin UNETR](https://arxiv.org/pdf/2201.01266) - `nnUNetv2_train 232 3d_fullres_SwinUNETR 0 -device cuda --npz -tr nnUNetTrainer_SwinUNETR --c`
* [2021 winner](https://arxiv.org/pdf/2112.04653) - `nnUNetv2_train 232 3d_fullres_BN_BS5_RBT_DS_BD_PS 0 -device cuda --npz -tr nnUNetTrainerBN_BS5_RBT_DS_BD_PS --c`

**Note:** The data split of the 5 folds was created randomly, but it was ensured that the validation set only contained real data. No synthetic data created using the case in the validation was in the training set (check `splits_final_2023_glioma.json`). `nnUNet/Data_splits.ipynb` contains the code necessary to make these changes.

#### 🤔🔍📈Post-processing

From 2023, lesion-wise DSC and lesion-wise HD95 are used, which evaluate each leasion individually, heavily penalising the existence of FP and FN. Therefore, post-processing based on a threshold is performed to remove some small tumours that are detected but are not actually tumours, reducing the number of FP. For this purpose, several values were tested for each region (WT, TC and ET), archieven the best results with thresholds of 250, 150, 100.

#### 📈 Segmentation inference

In our submission we use the ensemble of 45 checkpoints (9 solutions * 5 folds):

1. Real + Synthetic data generated using real labels (3 networks * 5 folds) -> nnUNet ID 236
2. Real + Registered cases (3 networks * 5 folds) -> nnUNet ID 233
3. Real + Synthetic data generated using synthetic labels (3 networks * 5 folds) -> nnUNet ID 232 (**We provide these weights**)

However, using only synthetic data generated with synthetic labels to train the 3 networks also shows promissing results, reducing the inference time by 2/3. To control which models are used for inference, go to `BraTS2023_inference/main.py` and choose the `ensemble_code`. By default `ensemble_code ='rGB_rGL_rGS'`, ensemble of all is  `'GB_GL_GS_RB_RL_RS_rGB_rGL_rGS'` .  The post-processing by thresholding can also be changed in the same file, by changing `min_volume_threshold_WT = 250, min_volume_threshold_TC = 150, min_volume_threshold_ET = 100`.

* B - nnUNet
* S - Swin UNETR
* L - Large nnUNet
* G - Real data + Synthetic data generated by the GliGAN
* rG - Real data + Synthetic data generated by the random label generator and GliGAN
* R - Synthetic data generated by registration

**🤖⚙️🏃‍♀️To run the segmentation inference:**

* Change to the directory `BraTS2023_inference`
* `python main.py --data_path ./in_data --output_path ./output --nnUNet_results ../nnUNet/nnUNet_results `
* Tip: check the `code_translator`  and `perform_inference_step` functions in `infer_low_disk.py` to check if the nnUNet_results folders have the correct names.
* It will create two new directories:
  * `converted_dataset`
  * `inference`
* The final inference (post-processed), will be avaiable in the `--output_path`.

## 🏁 END

---
