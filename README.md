# 👋 Faking_it team! BraTS submissions 🎬

![alt text](imgs/Logo.png "Title")

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 💡 Key Solutions (each subpage contains all the steps to reproduce the solutions):

- **🥇 BraTS 2023 Task 1:** [Adult Glioma Segmentation](BraTS2023_Task1.md)
- **🥇BraTS-ISBI 2024 GoAT:** [Generalizability Across Tumors Challenge](BraTS2024-ISBI_GoAT.md)
- **🥇BraTS 2024 Task 1:** [Adult Glioma Post Treatment](BraTS2024_Task1.md)
- **🥉BraTS 2024 Task 3:** [Meningioma Radiotherapy](BraTS2024_Task3.md)
- **🏅BraTS 2024 Task 7:** [Synthesis (Global) - Missing MRI ](./BraTS2024_Task7.md)-> [Check out poster! ](./imgs/MICCAI2024-Poster-Task7_8.pdf)
- **🥈BraTS 2024 Task 8:** [Synthesis (Local) - Inpainting](./BraTS2024_Task8.md) -> [Check out poster! ](./imgs/MICCAI2024-Poster-Task7_8.pdf)

✅ This repository contains the code and all the steps to reproduce the results of the submissions to BraTS 2023 Task 1, BraTS-ISBI 2024 GoAT, BraTS 2024 Tasks 1, 3, 7 and 8.

✅ Note that BraTS 2023 Task 1, BraTS-ISBI 2024 GoAT BraTS 2024 Tasks 1 and 3 are segmentation tasks and BraTS 2024 Tasks 7 and 8 are synthetic generation (using WDM 3D).

### :star_struck: We have released the trained weights!  :partying_face:

💾 You can download them at ... You just need to place them in the correct place 🤓

## Before running any experiments:

💻 For better experience, you should create a conda environment and have a machine with GPU.

### Segmentation tasks:

⚠️16GB of VRAM might be enough, however, we recomend using a GPU with 24GB. Be carefull with the amount of RAM you can use, as our code load the entire dataset to memory by default for faster training, but it might not be suitable for your machine. To reduce this, look into the data loaders.

**💻 To create the conda environment:**

1. conda create -n BraTS_solutions python=3.11.9
2. pip install:

   1. pip3 install torch torchvision torchaudio
   2. pip install monai
   3. pip install nilearn
   4. pip install nibabel
   5. pip install matplotlib
   6. pip install pathlib
   7. pip install einops
   8. pip install tqdm
   9. pip install SimpleITK
   10. pip install nnunet
   11. cd nnUNet_install
       1. pip install -e . (nnunet v2)
   12. cd mednext
       1. pip install -e . (mednext)

### Synthetesis tasks:

⚠️ 40GB of VRAM is enough. We have set the `cache_rate=0` in `CacheDatase` in `c_bratsloader.py` file. For faster processing you can increase this number, up to 1. Be carefull with the amount of RAM you can use.

💻 To create the conda environment:

1. conda create --name wdm_submit python=3.10.1
2. pip install:

   1. pip install nibabel
   2. pip install monai
   3. pip install blobfile
   4. pip install PyWavelets

🤞 After running all commands, all dependencies should be installed. We performed our final tests on the 15 of October of 2024. If you find difficulties matching the versions, try to install the versions avaiable at that time.

# If you find our work useful, please consider to ⭐️ **star this repository** and 📝 **cite our papers**:

**BraTS 2023 Task 1:** [Adult Glioma Segmentation](BraTS2023_Task1.md)

```
@article{ferreira2024we,
  title={How we won BraTS 2023 Adult Glioma challenge? Just faking it! Enhanced Synthetic Data Augmentation and Model Ensemble for brain tumour segmentation},
  author={Ferreira, Andr{\'e} and Solak, Naida and Li, Jianning and Dammann, Philipp and Kleesiek, Jens and Alves, Victor and Egger, Jan},
  journal={arXiv preprint arXiv:2402.17317},
  year={2024}
}
```

**BraTS-ISBI 2024 GoAT:** [Generalizability Across Tumors Challenge](BraTS2024-ISBI_GoAT.md)

```
@inproceedings{ferreira2024generalisation,
  title={Generalisation of Segmentation Using Generative Adversarial Networks},
  author={Ferreira, Andr{\'e} and Luijten, Gijs and Puladi, Behrus and Kleesiek, Jens and Alves, Victor and Egger, Jan},
  booktitle={2024 IEEE International Symposium on Biomedical Imaging (ISBI)},
  pages={1--4},
  year={2024},
  organization={IEEE}
}
```

![alt text](imgs/BraTS.png "Title")

---
