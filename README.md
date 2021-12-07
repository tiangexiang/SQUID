# SQUID: In-painting Radiography Images for Unsupervised Anomaly Detection

We propose Space-aware memory QUeues for In-painting and Detecting anomalies from radiography images (abbreviated as SQUID).
Radiography imaging protocols focus on particular body regions, therefore producing images of great similarity and yielding recurrent anatomical structures across patients.
To exploit this structured information, SQUID consists of a new Memory Queue and a novel in-painting block in the feature space.
We show that SQUID can taxonomize the ingrained anatomical structures into recurrent patterns; and in the inference, SQUID can identify anomalies (unseen/modified patterns) in the image.
SQUID surpasses the state of the art in unsupervised anomaly detection by over 5 points on two chest X-ray benchmark datasets.
Additionally, we have created a new dataset (DigitAnatomy), which synthesizes the spatial correlation and consistent shape in chest anatomy. We hope DigitAnatomy can prompt the development, evaluation, and interpretability of anomaly detection methods, particularly for radiography imaging.

## Paper

This repository provides the official Pytorch implementation of SQUID in the following papers:

**In-painting Radiography Images for Unsupervised Anomaly Detection** <br/>
[Tiange Xiang](https://scholar.google.com/citations?hl=en&user=sskixKkAAAAJ)<sup>1</sup>, [Yongyi Lu](https://scholar.google.com/citations?user=rIJ99V4AAAAJ&hl=en&oi=ao)<sup>2</sup>, [Alan L. Yuille](https://www.cs.jhu.edu/~ayuille/)<sup>2</sup>, [Chaoyi Zhang](https://chaoyivision.github.io/)<sup>1</sup>, [Weidong Cai](https://weidong-tom-cai.github.io/)<sup>1</sup>, and [Zongwei Zhou](https://www.zongweiz.com)<sup>1</sup> <br/>
<sup>1 </sup>University of Sydney,  <sup>2 </sup>Johns Hopkins University <br/>
Arxiv Preprint <br/>
[paper](https://arxiv.org/pdf/2111.13495.pdf) | [code](https://github.com/tiangexiang/SQUID-private)


## Dependencies

Please use our environment to reproduce the results through the following command:

```
conda env create -f environment.yml
conda activate chris
```

## File Structures

* ```checkpoints/```: experiment folders, organized by unique exp identifier.
* ```dataloader/```: dataloaders for zhanglab, chexpert, and digitanotamy.
* ```models/```: models for SQUID, inpainting block, all kinds of memory, basic modules, and discriminator.
* ```configs/```: configure files for different experiments, based on the base configure class.


## Quick Start with ZhangLab Chest X-ray
### Data

Please download the offical training/testing and our validation splits from [google drive](https://drive.google.com/file/d/1kgYtvVvyfPnQnrPhhLt50ZK9SnxJpriC/view?usp=sharing), and unzip it to anywhere you like.

### Configs

Different experiments are controlled by configure files, which are in ```configs/```. 

All configure files are inherited from the base configure file: ```configs/base.py```, we suggest you to take a look at this base file first, and **change the dataset root path in your machine**.

Then, you can inherite the base configure class and change settings as you want. 

We provide our default configures in ```configs/zhang_best.py```.

Configure file is passed to the program as a flag during training and evaluation.

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


Evaluate with an exp folder (config file from the exp folder will be used instead):
``` 
python3 eval.py --exp zhang_exp1
```

Alternatively, you can modify ```eval.sh``` and simply run:
``` 
./eval.sh
```
