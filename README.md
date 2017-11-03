# IAmelodyDNN

This repository contains a system based on an article published at ISMIR 2016. The main goal is to estimate the melodic line from a polyphonic piece. You can download it, run or adjust parameters to train the neural networks.

Basically, the system is composed by two deep neural networks: F0-detection, to estimate which fundamental notes are melody; and, Voice Activity Detection, to decide which frames are voiced/unvoiced.

## Usage

You can pre process data from a dataset with *preprocess.py*.

You can load the pre trained weights with *f0model.h5* and *f0model.json* or train the F0-estimation neural network using *f0dnn.py*.

You can load the pre trained weights with *SVDmodel.h5* and *SVDmodel.json* or train the **Voice Activity Detection**  neural network using *SVDdnn.py*.

The notebooks can be used to visualize and interact with the neural networks.

 - **Data pre-processing**: presents the steps to transform the audio and its labels as input for neural networks;
 - **DNN Model - F0 Estimation**: presents the training and evaluation steps for f0model;
 - **DNN Model - Voice Activity Detection**: presents the training and evaluation steps for SVDmodel;
 - **Making Predictions**: uses the pre-trained weights to make a melodic line estimation and a output evaluation, comparing with label data.
 - **Melody Transcription Using Deep Neural Networks**: is a summary of all process, from pre-process to prediction. 


This code is free and is open to improvements. Enjoy it ;)