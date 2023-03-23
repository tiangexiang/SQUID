# SQUID

We propose Space-aware memory QUeues for In-painting and Detecting anomalies from radiography images (abbreviated as SQUID).
Radiography imaging protocols focus on particular body regions, therefore producing images of great similarity and yielding recurrent anatomical structures across patients.
To exploit this structured information, SQUID consists of a new Memory Queue and a novel in-painting block in the feature space.

SQUID can taxonomize the ingrained anatomical structures into recurrent patterns; and in the inference, SQUID can identify anomalies (unseen/modified patterns) in the image.
SQUID surpasses the state of the art in unsupervised anomaly detection by over 5 points on two chest X-ray benchmark datasets.

<p align="center"><img width="100%" src="figures/fig_framework.pdf" /></p>

## Paper

This repository provides the official Pytorch implementation of SQUID in the following papers:

**Deep Feature In-painting for Unsupervised Anomaly Detection in X-ray Images** <br/>
[Tiange Xiang](https://tiangexiang.github.io/)<sup>1</sup>, [Yixiao Zhang](https://0-scholar-google-com.brum.beds.ac.uk/citations?user=lU3wroMAAAAJ&hl=fi)<sup>2</sup>, [Yongyi Lu](https://scholar.google.com/citations?user=rIJ99V4AAAAJ&hl=en&oi=ao)<sup>2</sup>, [Alan L. Yuille](https://www.cs.jhu.edu/~ayuille/)<sup>2</sup>, [Chaoyi Zhang](https://chaoyivision.github.io/)<sup>1</sup>, [Weidong Cai](https://weidong-tom-cai.github.io/)<sup>1</sup>, and [Zongwei Zhou](https://www.zongweiz.com)<sup>2</sup> <br/>
<sup>1</sup>University of Sydney,  <sup>2</sup>Johns Hopkins University <br/>
CVPR, 2023 <br/>
[paper](https://arxiv.org/pdf/2111.13495.pdf) | [code](https://github.com/tiangexiang/SQUID-private)

## Benchmarking SQUID on Chest X-rays

<p align="center"><img width="100%" src="figures/fig_ROC.png" /></p>

## Dependencies

Please clone our environment using the following command:

```
conda env create -f environment.yml
conda activate squid
```

## File Structures

* ```checkpoints/```: experiment folders, organized by unique exp identifier.
* ```dataloader/```: dataloaders for zhanglab, chexpert, and digitanotamy.
* ```models/```: models for SQUID, inpainting block, all kinds of memory, basic modules, and discriminator.
* ```configs/```: configure files for different experiments, based on the base configure class.


## Usage
### Data

**ZhangLab Chest X-ray**

Please download the offical training/testing and our validation splits from [google drive](https://drive.google.com/file/d/1kgYtvVvyfPnQnrPhhLt50ZK9SnxJpriC/view?usp=sharing), and unzip it to anywhere you like.

**Stanford ChexPert**

Please download the offical training/validation and our testing splits from [google drive](https://drive.google.com/file/d/14pEg9ch0fsice29O8HOjnyJ7Zg4GYNXM/view?usp=sharing), and unzip it to anywhere you like.

**DigitAnatomy**

Please unzip the files in ```data/digitanatomy.zip```, and place them to the data root as specified in the ```configs/base.py```.

### Configs

Different experiments are controlled by configure files, which are in ```configs/```. 

All configure files are inherited from the base configure file: ```configs/base.py```, we suggest you to take a look at this base file first, and **change the dataset root path in your machine**.

Then, you can inherite the base configure class and change settings as you want. 

We provide our default configures for ZhangLab: ```configs/zhang_best.py```, CheXpert: ```configs/chexpert_best.py``` and ```configs/digit_best.py```.

The path to a configure file needs to be passed to the program for training.

### Train

Train with a configure file and a unique experiment identifier:

``` 
python3 main.py --exp zhang_exp1 --config zhang_best
```

Alternatively, you can modify ```run.sh``` and simply run:

```
./run.sh
```

### Evaluation


Evaluate with an exp identifier (config file will be imported from the *exp folder* instead):
``` 
python3 eval.py --exp zhang_exp1
```

Alternatively, you can modify ```eval.sh``` and simply run:
``` 
./eval.sh
```
<!-- 
### Train MemAE

``` 
python3 main_baseline.py --exp memae1 --config baseline_zhang
```

### Evaluate MemAE

``` 
python3 eval_baseline.py --exp memae1
``` -->

## Citation
If you use this work for your research, please cite our paper:
```
@article{xiang2023painting,
  title={In-painting Radiography Images for Unsupervised Anomaly Detection},
  author={Xiang, Tiange and Liu, Yongyi and Yuille, Alan L and Zhang, Chaoyi and Cai, Weidong and Zhou, Zongwei},
  journal={IEEE/CVF Converence on Computer Vision and Pattern Recognition},
  year={2023}
}
```

## Acknowledgement
This work was supported by the Lustgarten Foundation for Pancreatic Cancer Research. We appreciate the constructive suggestions from Yingda Xia, Jessica Han, Yingwei Li, Bowen Li, Adam Kortylewski, Huiyu Wang, and Sonomi Oyagi.
