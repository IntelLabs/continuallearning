# CLOC

# Code for ICCV 2021 paper: Online Continual Learning with Natural Distribution Shifts: An Empirical Study with Visual Data. 

Author: Zhipeng Cai, Ozan Sener, Vladlen Koltun

[[Paper link]](https://arxiv.org/pdf/2108.09020.pdf)

This repo contains the code for reproducing the experiments in the paper. 

Prerequisite
============
1. python 3

2. pytorch 1.7+

3. tensorboardX

4. numpy

We run the experiments mainly using 4 Quadro RTX 6000 GPUs.

Usage
=====
1. Clone this repo.

2. Download meta-data of CLOC from [the google drive link](https://drive.google.com/file/d/1UdIZe_9rEemO2QukHw7bf6aDFV-RjAfc/view?usp=sharing). Decompress it into "CLOC/data_preparation/release", via:

```
mv metadata.tar.gz CLOC/data_preparation/
tar -xvzf metadata.tar.gz
```

3. Download images (an important function for download_images.py is to simultaneously download different parts of the dataset, and resume from unexpected termination, please see the comments in the code for more details), via:

```
cd CLOC/data_preparation/download_images
python /download_images.py
```

4. Run the experiments:

Simply go to exp_<name_of_experiments> folders, and run the experiment that you want to replicate. "exp_best_model" refers to the code to train the proposed OCL model, i.e., using PoLRS, ADRep and small batch sizes.

In each exp_<name_of_experiments> folder, there will be pairs of scripts named as "train_xxx.sh" and "eval_xxx.sh". Both can be run by simply type in:

```
bash xxx_xxx.sh
```

"train_xxx.sh" trains the OCL model and plots the average online accuracy.

"eval_xxx.sh" produces the backward transfer curve.

The time axis for different plots may have a scaling effect, one can unify them by simply normalize the time axis into [0,1] and then mutiply individual time point by the maximum number of images or the maximum wall-clock time. 

5. use tensorboard to monitor the results:

Go to each exp_xxx folder, do:
```
tensorboard --logdir=./
```
The output folder of individual experiments can be found in the first line of the output logs.

------------------------
Contact
------------------------

Name: Zhipeng Cai

Homepage: https://zhipengcai.github.io/

Email: czptc2h@gmail.com

Do not hesitate to contact the authors if you have any question or find any bug :)
