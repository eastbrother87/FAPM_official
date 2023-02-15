# FAPM_official
This repository contains the implementation of FAPM (2022 arxiv).

https://arxiv.org/abs/2211.07381

![](architecture.png)

The FAPM is proposed to find an anomaly industrial object. Our model is ranked #10 in MvTEC AD benchmark, and this result can be found in papers with code (https://paperswithcode.com/sota/anomaly-detection-on-mvtec-ad). 

## Development setup

conda environment
```sh
conda create -n FAPM
conda activate FAPM

conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
```


pip environment
```sh
pip install sklearn
pip install einops
pip install pytorch_lightning
pip install opencv-python
pip install scipy
pip install PIL
pip install tqdm

```

## Future work

- [x] Inference Code
- [x] Pretrained Memory 
- [ ] Training Code 



## Usage

Inference Code
```sh
python test.py --category=capsule --n_neighbors=4
```
You can download [pretrained memory](https://drive.google.com/drive/folders/1z4dplHddceYLoYiKe29NY_SxeIJFstPu?usp=share_link) from this google link.
