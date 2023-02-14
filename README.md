# FAPM_official
This repository contains the implementation of FAPM(2022 arxiv).
This project 

https://arxiv.org/abs/2211.07381

The FAPM is proposed to find an anomaly industrial object. Our model is ranked #10 in MvTEC AD benchmark, and this result can be found in papers with code( https://paperswithcode.com/sota/anomaly-detection-on-mvtec-ad ). 

##Development setup

```sh
conda create -n FAPM
conda activate FAPM

conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install sklearn
pip install einops
pip install pytorch_lightning
pip install opencv-python
pip install scipy
pip install PIL
pip install tqdm

```
