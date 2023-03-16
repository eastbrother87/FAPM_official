# FAPM_official
This repository contains the implementation of FAPM (2023 ICASSP).

https://arxiv.org/abs/2211.07381

![](architecture.png)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/fapm-fast-adaptive-patch-memory-for-real-time/anomaly-detection-on-mvtec-ad)](https://paperswithcode.com/sota/anomaly-detection-on-mvtec-ad?p=fapm-fast-adaptive-patch-memory-for-real-time)

 We propose a new method called Fast Adaptive Patch Memory (FAPM) for real-time industrial anomaly detection. FAPM utilizes patch-wise and layer-wise memory banks that store the embedding features of images at the patch and layer level, respectively, which eliminates unnecessary repetitive computations. We also propose patch-wise adaptive coreset sampling for faster and more accurate detection. 

## Future work

- [x] Inference Code
- [x] Pretrained Memory 
- [ ] Training Code 

## Development setup

conda environment
```sh
conda create -n FAPM
conda activate FAPM
pip install -r requirements.txt

```
## Dataset
Please download the MVTec dataset from this [website](https://www.mvtec.com/company/research/datasets/mvtec-ad).



## Usage

Inference Code
```sh
python test.py --dataset_path=$DATASET_PATH --result_path=$RESULT_PATH --category=capsule --project_root_path=$PRETRAINED_MEMORY_DIRECTORY
```
You can download [pretrained memory](https://drive.google.com/drive/folders/1z4dplHddceYLoYiKe29NY_SxeIJFstPu?usp=share_link) from this google link.

## Citation
Cite as below if you find this repository is helpful to your project:
```sh
@article{kim2022fapm,
  title={FAPM: Fast Adaptive Patch Memory for Real-time Industrial Anomaly Detection},
  author={Kim, Donghyeong and Park, Chaewon and Cho, Suhwan and Lee, Sangyoun},
  journal={arXiv preprint arXiv:2211.07381},
  year={2022}
}
```

## Acknowledgement

Some code snippets are borrowed from [PatchCore_anomaly_detection](https://github.com/hcw-00/PatchCore_anomaly_detection) and [patchcore-inspection](https://github.com/amazon-science/patchcore-inspection). I'm really appreciate to your great projects!!

