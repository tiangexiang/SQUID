# SQUID-private

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
