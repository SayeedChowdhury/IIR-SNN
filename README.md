# IIR-SNN (Iteratively Initialized and Retrained SNN) 

This project implements a sequential training approach which enables SNN inference with just 1 timestep

Some trained SNN models are available at-
https://drive.google.com/drive/folders/1L6LIoqGcutb-Aso_8YlFW_JpeGT4xXwA?usp=sharing

The log files of training are available at-
https://drive.google.com/drive/folders/1ovMcSppx9jXTfRzyEkt1b-EWEfzpgogt?usp=sharing

Usage of codes for the paper- One Timestep is all you need: Training Spiking Neural Networks with Ultra Low Latency

first, an ANN is trained, codes for both training with and without batch-norm (bn)
during ANN training is attached. 

ANN with bn (cifar)- If ANN is to be trained with bn on cifar datasets, use ann_bn.py
file (user may give architecture choice, batch-size, cifar10/100, learning
rate etc) as parameters

Then use absorb_bn.py to fuse the bn params in the layerwise weights for ANN-SNN conversion.

ANN without bn (cifar)- use ann_withoutbn.py

ANN (imagenet)- use ann_imagenet.py

if bn is not used in ANN, no need to use the absorb_bn file

For SNN domain training-

SNN with ann_bn (cifar)- use snncifar_bnfused.py

SNN with ann without bn (cifar)- use snncifar_annwithoutbn.py

SNN_resnet_cifar (ann with bn)- use snn_resnet.py


SNN with imagenet- use snn_imagenet_try.py


Note, in SNNs, for training with conversion from ANN, the ANN file path has to
be given in pretrained_ann param, else for SNN training with a lower timestep, the higher timestep SNN
file path has to be given in pretrained_snn parameter

For subsequent snn training with lower timestep, the same SNN .py file has to
be run, just the file_path of the previous SNN trained with higher timestep
is to be given to pretrained_snn param.

For RL Atari experiments, use snn-atari-pong_try.py 


