
## Cross-view image geo-localization with Panorama-BEV Co-Retrieval Network

[![arXiv](https://img.shields.io/badge/arXiv-2501.16764-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2408.05475)
[![Model](https://img.shields.io/badge/HF-Model-yellow)](https://huggingface.co/Yejy53/EB-BEV-CVACT)
[![Dataset](https://img.shields.io/badge/HF-Dataset-yellow)](https://huggingface.co/datasets/Yejy53/CVGlobal)


This repository contains the official implementation of the paper: Cross-view image geo-localization with Panorama-BEV Co-Retrieval Network. It is a very effective cross-view retrieval framework by adding an additional street view BEV retrieval branch. It achieves leading performance on multiple datasets, including VIGOR, CVUSA to CVACT retrieval.

![method](method.png)

## 📢 News
- **2025-6** We are very happy to announce that [where am I](https://arxiv.org/abs/2412.17007) was accepted by **ICCV 2025**
- **2024-12** We also published a new work on cross-view retrieval based on Natural Language text: Where am i ? CVG-Text [here](https://arxiv.org/abs/2412.17007)
- **2024-10** The code for Street View-BEV Co-retrieval inference is now available. If there is any missing code or abnormality, you can report it to me in the issue.
- **2024-09** The training and testing code for the BEV branch on CVACT has been released.
- **2024-08** Source code of BEV transformation is released（CVACT/CVUSA）.
- **2024-07-1** EP-BEV is accepted to **ECCV 2024**.

## Installation
Clone this repo to a local folder:
```bash
git clone https://github.com/yejy53/EP-BEV.git
cd EP-BEV
```

## Environment Setup

```bash
conda create -n EP-BEV python=3.9 -y
conda activate EP-BEV
pip install -r requirements.txt
```

If huggingface cannot download the weights successfully, you can add export HF_ENDPOINT="https://hf-mirror.com" at the end of .bashrc and reactivate it.


## Data Preparation
The publicly available datasets used in this paper can be obtained from the following sources: 

**Preparing CVUSA Dataset.**  The dataset can be downloaded [here](https://mvrl.cse.wustl.edu/datasets/cvusa). 

**Preparing CVACT Dataset.**  The dataset can be downloaded [here](https://github.com/Liumouliu/OriCNN). 

**Preparing VIGOR Dataset.**  The dataset can be downloaded [here](https://github.com/Jeff-Zilence/VIGOR/tree/main). 

**Preparing CVGlobal Dataset.**  The dataset can be downloaded [here](https://huggingface.co/datasets/Yejy53/CVGlobal). 

![ECCV2](https://github.com/user-attachments/assets/02252a74-a116-4829-80af-96f2426a326a)

## Data Structure:

```
├─ CVACT
  ├── ACT_data.mat
  ├── ANU_data_small/
    ├── bev/
    ├── satview_polish/ 
    ├── streetview/	
  └──ANU_data_test/

```

## Use our pre-trained model for retrieval 
1. You can download a pre-trained model (e.g. cvact) from [huggingface](https://huggingface.co/Yejy53/EB-BEV-CVACT/tree/main) and place it in ckpt folder.
2. You need to organize the generated BEV images into the above dataset format. You can download the generated BEV images directly from the following [huggingface](https://huggingface.co/datasets/Yejy53/CVACT-BEV) link to get consistent results, or generate BEV images yourself and then retrain.
3. When performing Street View-BEV Co-Retrieval, you only need to add the similarity of using a pure Street View image to the similarity of using a BEV image. The weights for using Street View search can be obtained from the following [huggingface](https://huggingface.co/Yejy53/CVACT-Street/tree/main) link. The method and weights for using Street View search also can be found in the [Sample4G](https://github.com/Skyy93/Sample4Geo).

## ❤️ Acknowledgements

Our code is built on top of [Sample4G](https://github.com/Skyy93/Sample4Geo) and [Boosting3DoF](https://github.com/YujiaoShi/Boosting3DoFAccuracy). We appreciate the previous open-source works.

If you have any questions, be free to contact with me! 

## BibTeX 

```
@inproceedings{ye2025cross,
  title={Cross-view image geo-localization with Panorama-BEV Co-Retrieval Network},
  author={Ye, Junyan and Lv, Zhutao and Li, Weijia and Yu, Jinhua and Yang, Haote and Zhong, Huaping and He, Conghui},
  booktitle={European Conference on Computer Vision},
  pages={74--90},
  year={2025},
  organization={Springer}
}
```
