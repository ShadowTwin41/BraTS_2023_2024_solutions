# 👋 Faking_it team! BraTS submissions 🎬

## :technologist: [BraTS 2024 - Task 3 - Meningioma Radiotherapy](https://www.synapse.org/Synapse:syn53708249/wiki/627503)

### 👨‍🎓 Introduction to the challenge

📚 The purpose of the BraTS 2024 Meningioma challenge is to create a community benchmark for automated segmentation of meningioma GTV based on pre-radiation therapy planning brain MRI exams. This task, if successful, will provide an important tool for the objective delineation of meningioma GTV, which will be immediately relevant for radiotherapy planning. In addition, this algorithm will provide a starting point for future studies focused on distinguishing residual/recurrent meningioma from post-treatment changes and predicting risk of progression and future recurrence.

💾 👩‍⚕ 👨‍⚕In addition, image data will consist of 1) a single series (3D postcontrast T1-weighted spoiled gradient echo imaging) and 2) in native acquisition space, which mimics the data available for most radiotherapy planning scenarios, rather than 4 MRI scans co-registered to a canonical atlas space. Furthermore, previous BraTS challenges have utilized skull-stripping, whereas here we will preserve extracranial structures and instead use automated face removal algorithms to preserve patient anonymity (i.e., defacing).

📊 The goal of the BraTS 2024 Meningioma Radiotherapy Segmentation Challenge is to automatically segment GTV for cranial and/or facial meningiomas using radiotherapy planning MRI scans. Target labels will consist of a single tumor region (the GTV) in the native acquisition space.

### 💡 Solution - Improved Multi-Task Brain Tumour Segmentation with Synthetic Data Augmentation (to be published)

📖 Generative adversarial networks (GANs) is used to massively increase the amount of available samples for training three different deep learning models for brain tumour segmentation. The first model is the standard nnU-Net, the second is the Swin UNETR and the third is the MedNeXT. The entire pipeline is built on the nnU-Net implementation, except for the generation of the synthetic data. The use of convolutional algorithms and transformers is able to fill each other's knowledge gaps. Using the new metric, our best solution achieves the DSC 0.8214 and HD95 24.64 in the valdiation set.

#### 🤖🏥📑 Generative Adversarial Network - GliGAN

**📁 First, the folder structure should be as follows:**

1. GliGAN
   1. Checkpoint (**We provide our trained weights**)
      1. {args.logdir} (this directory ad sub directories will be created automatically)
         1. csv file (unless specified somewhere else).
         2. t1ce
         3. debug
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
2. **Create the csv file** - `python csv_creator.py --logdir brats2024_meningioma --dataset brats_2024_meningioma --datadir ../../DataSet/BraTS-MEN-RT-Train-v2 --seg_ending gtv.nii.gz --debug True`
3. **First step GliGAN training -** `python tumour_main.py --logdir brats2024_meningioma --batch_size 2 --num_workers 2 --in_channels 2 --out_channels 1 --optim_lr 0.0001 --num_steps 2000000 --reg_weight 0 --noise_type gaussian_extended --not_abs_value_loss True --use_sigmoid True --G_n_update 2 --D_n_update 1 --w_loss_recons 5 --modality t1ce --dataset brats_2024_meningioma`
   1. **Resume training -** `python tumour_main.py --logdir brats2024_meningioma --batch_size 2 --num_workers 2 --in_channels 2 --out_channels 1 --optim_lr 0.0001 --num_steps 2000000 --reg_weight 0 --noise_type gaussian_extended --not_abs_value_loss True --use_sigmoid True --G_n_update 2 --D_n_update 1 --w_loss_recons 5 --modality t1ce --dataset Brats_2024_meningioma --resume_iter 10`
4. **Second step GliGAN training -** `python tumour_main.py --logdir brats2024_meningioma --batch_size 2 --num_workers 2 --in_channels 2 --out_channels 1 --optim_lr 0.0001 --num_steps 2000000 --reg_weight 0 --noise_type gaussian_extended --not_abs_value_loss True --use_sigmoid True --G_n_update 2 --D_n_update 1 --modality t1ce --dataset Brats_2024_meningioma --resume_iter 20 --w_loss_recons 100 --l1_w_progressing True`
5. **Label generator -** `python label_main.py --logdir brats2024_meningioma --batch_size 4 --num_worker 4 --in_channels 1 --out_channels 1 --total_iter 200000 --dataset Brats_2024_meningioma`
   1. **Resume training -** `python label_main.py --logdir brats2024 --batch_size 4 --num_worker 4 --in_channels 1 --out_channels 1 --total_iter 200000 --dataset Brats_2024 --resume_iter 3`
6. **GliGAN baseline (optional) -** `python tumour_main_baseline.py --logdir brats2024_meningioma --batch_size 2 --num_workers 2 --in_channels 2 --out_channels 1 --optim_lr 0.0001 --num_steps 2000000 --reg_weight 0 --noise_type gaussian_extended --modality t1ce --dataset Brats_2024_meningioma`

**🤔🩻 For inference:**

1. Change to `GliGAN/src/infer` directory.
2. `python main_random_label_random_dataset_generator_multiprocess_meningioma.py --batch_size 1 --in_channels_tumour 2 --out_channels_tumour 1 --out_channels_label 1 --dataset brats2024_meningioma --g_t1ce_n 325080 --g_label_n 200000 --latent_dim 100 --logdir brats2024_meningioma --num_process 1 --start_case 0 --end_case 100 --new_n 17`
   1. **Tip:** use `start_case` and `end_case` to split the inference process manually in distinct machines/nodes, by splitting the dataset. To use all dataset in same machine, don't set `--start_case` and `--end_case`.
   2. You can control how many cases are created per sample, by setting `--new_n`. The inference pipeline has a "patience limit", i.e. if it does not find a place for the tumour after several attempts, it moves on to the next case.
3. **Improvements:** The inference pipeline does not consider the skull, i.e., some synthetic tumours might be placed over the skull. Please feel free to find a solution and share with us!

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
3. Convert all data to the nnUNet format:
   1. Change the nnUNet/Data_prepar.ipynb correspondingly
4. Change to the folder `nnUNet`.
5. `nnUNetv2_plan_and_preprocess -d 244 --verify_dataset_integrity`
   1. Don't forget to create the dataset.json file (see  `example/dataset_2024_meningioma.json`)
   2. Copy the `example/nnUNetPlans_2024_meningioma.json` to the postprocessing folder, rename it to `nnUNetPlans.json`, and change the "`dataset_name`" to the correct name given to the dataset, e.g., `Dataset244_BraTS_2024_meningioma_rGANs`.
6. `mednextv1_plan_and_preprocess -t 244 -pl3d ExperimentPlanner3D_v21_customTargetSpacing_1x1x1 -pl2d ExperimentPlanner2D_v21_customTargetSpacing_1x1x1`
   1. Don't forget to create the dataset.json file (see  `example/dataset_2024_meningioma_mednext.json`)

**🤖⚙️🏃‍♀️ Run all 5 folds:**

* [nnUNet (3D full resolution)](https://github.com/MIC-DKFZ/nnUNet) - `nnUNetv2_train 244 3d_fullres 0 -device cuda --npz --c`
* [Swin UNETR](https://arxiv.org/pdf/2201.01266) - `nnUNetv2_train 244 3d_fullres_SwinUNETR 0 -device cuda --npz -tr nnUNetTrainer_SwinUNETR --c`
  * **Warning**: It needs to be changed the network's number of channels of output:
    * Go to: `./nnUNet_install/nnunetv2/training/nnUNetTrainer/variants/network_architecture/nnUNetTrainer_SwinUNETR.py`
    * Change the `out_channels` to 1.
    * Warning: We couldn't train using the Swin UNETR. We didn't find why.
* [MedNeXt](https://github.com/MIC-DKFZ/MedNeXt) - `mednextv1_train 3d_fullres nnUNetTrainerV2_MedNeXt_L_kernel5 Task244_BraTS_2024_meningioma_rGANs 0 -p nnUNetPlansv2.1_trgSp_1x1x1 [-c]`
  * -c to resume training

**Note:** The data split of the 5 folds was created randomly, but it was ensured that the validation set only contained real data. No synthetic data created using the case in the validation was in the training set (check `example/splits_final_2023_glioma.json` and `example/splits_final_2024_glioma_mednext.pkl`). ). `nnUNet/Data_splits.ipynb` contains the code necessary to make these changes for both MedNeXt and nnUNet.

In the end, 15 checkpoints should be trained (5 for each network).

#### 🤔🔍📈 Post-processing

Thresholding is used to reduce the number of FP. However, since the False Positives are not heavly penalised, we decided to not use post-processing, but it's implemented in case it is necessary.

#### 🤖⚙️🏃‍♀️📈 Segmentation inference

In our submission we use the ensemble of 5 checkpoints:

1. Synthetic data generated using synthetic labels (MedNeXt network) -> nnUNet ID 244

   To control what models are used for inference, go to `BraTS2024_meningioma_inference/main.py` and choose the `ensemble_code` (`rGMm` by default). The post-processing by thresholding can also be changed in the same file, by changing `min_volume_threshold_GTV = 0`

* N - nnUNet
* S - Swin UNETR
* M - MedNeXt (**We provide these weights**)
* R - Real data
* rG - Real data + Synthetic data generated by the random label generator and GliGAN (**We provide these weights**)

**🤖⚙️🏃‍♀️📈 To run the segmentation inference:**

* Change to the directory `BraTS2024_meningioma_inference`
* `python main.py --data_path ./in_data --output_path ./output --nnUNet_results ../nnUNet/nnUNet_results `
* Tip: check the `perform_inference_step` function in `infer_low_disk_glioma_mednext.py` to check if the nnUNet_results folders have the correct names.
* It will create two new directories:

  * `converted_dataset`
  * `inference`
* The final inference (post-processed), will be avaiable in the `--output_path`.

## 🏁 END
