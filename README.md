# PromptHDR: a low-light image enhancement algorithm based on prompt learning

[View the Paper](https://github.com/darkforest1908/PromptHDR/blob/main/PromptHDR.pdf)



The "PromptHDR: a low-light image enhancement algorithm based on prompt learning" we propose the PromptHDR algorithm. For the problem of detail distortion, our model firstly uses the Retinex theory to provide the lighting information of the image, and then designs the “U-shape” Transformer architecture to optimize the visual effect of the image through efficient feature extraction and image reconstruction to better capture the detail information, and thus improve the quality of the image. In order to improve the computational efficiency and enhance the generated features, we design a Prompt block for the Transformer module, adopt the Mamba algorithm to improve the efficiency of image processing, and design a powerful sampling module for image detail processing. 




## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [License](#license)
- [Authors](#authors)
- [Acknowledgments](#acknowledgments)

# Installation

Provide step-by-step series of examples and explanations about how to get a development env running.

This repository is built in PyTorch 1.11 and tested on Ubuntu 16.04 environment (Python3.7, CUDA10.2, cuDNN7.6).
Follow these intructions

## 1. Clone our repository

```shell
git clone git@github.com:darkforest1908/PromptHDR.git
cd PromptHDR
```

## 2. Make conda environment

```shell
conda create -n PromptHDR python=2.9
conda activate PromptHDR
```

## 3. Install dependencies

### Packages

```shell
pip install -r requirements.txt
```

## 4. Install basicsr

```shell
python setup.py develop
```

## 5. Download Dataset:
We use the following datasets:

Lol_v1  https://drive.google.com/drive/folders/1Y3vtVZASzfSOBDWZAtKo_ygPLQlksWk8?usp=drive_link

Lol_v2  https://drive.google.com/drive/folders/1Yk5dpkleSFNwHSWfxrftAoDPB4xvKmd0?usp=drive_link

## 6. Modify the configuration file
Please modify the parameters in `Options/PromptHDR_train.yml`.

# Usage

## 1. Train the model

```shell
python3 basicsr/train.py --opt Options/PromptHDR_train.yml
```

## 2. Test the model

```shell
# LOL-v1
python3 Enhancement/test_from_dataset.py --opt Options/PromptHDR_train.yml --weights pretrained_weights/LOL_v2_real.pth --dataset LOL_v1

# LOL-v2-real
python3 Enhancement/test_from_dataset.py --opt Options/PromptHDR_train.yml --weights pretrained_weights/LOL_v1.pth --dataset LOL_v2_real

# LOL-v2-syn
python3 Enhancement/test_from_dataset.py --opt Options/PromptHDR_train.yml --weights pretrained_weights/LOL_v2_syn.pth --dataset LOL_v2_synthetic
```

The image outcome is at results\data\LOLv1\Test folder



# License

This project is licensed under the MIT License


# Authors

Wenzhen Yan <kekefen402@gmail.com>

Fuming Qu  <qufuming@ustb.edu.cn>

Yingzhen Wang  <3273064476@qq.com>




# Acknowledgements

Inspiration from PromptIR <https://github.com/va1shn9v/PromptIR>, Retinexformer <https://github.com/caiyuanhao1998/Retinexformer> and basicSR <https://github.com/XPixelGroup/BasicSR>.

