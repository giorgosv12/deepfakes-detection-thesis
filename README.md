# Deepfakes Detection using 3D Biometric Characteristics
***

This repository contains methods used for deepfakes detection that utilizes 3D biometric 
characteristics extracted from the face. In order not to use original 3D data, 3DMMs were used. In the presented methods 
temporal features were taken into account. To achieve this, the frames of the videos that are used on the methods are 
firstly converted into parameters that express the 3DMMs. Features that help identify deepfakes are also extracted from 
facial landmarks. Such features hold information about how open or close the eyes and the mouth are, as well as specific
angles of the face.

The methods are divided into two categories, **One Class Classifiers** and **Binary Classifiers**.The first case 
includes VAEs and Gans,two for each category, and  the second case a deep convolutional network , 
[EfficientNetv2](https://github.com/hankyul2/EfficientNetV2-pytorch). 
It was observed that GANs show relatively good results in recognizing deepfakes, results
that outperform those of VAE. However, the Binary Classifier presented the best results. Also, the features proposed 
in this work improved the performance of the Binary Classifier and are, therefore, considered capable of helping to 
identify deepfakes.

Two Datasets were used to test and train the models, [VoxCeleb2](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html)
 and [FaceForensics++](https://github.com/ondyari/FaceForensics). The One Class Classifier models were trained on VoxCeleb2
and tested on FaceForesics++ while the EfficientNet was trained using both datasets.

The One Class Classifier models that are implemented are:
- Variational Autoencoder implemented using a Dense architecture and thus named DenseVAE
- [VQVAE](https://github.com/airalcorn2/vqvae-pytorch)
- GAN using convolutional layers
- [OCGAN](https://github.com/xiehousen/OCGAN-Pytorch)

The architecture of each model and the training method that were used are described in detail in the provided 
[deepfakes_detection_thesis.pdf](deepfakes_detection_thesis.pdf) file. 

## Overview
***

To transform the input videos into the features that are used to train the models, the steps described
in _Dataset_Preprocessing_ folder should be followed. The _One_Class_Classifiers_ folder includes the 4 One Class Classifier
models alongside the scripts used for their training and testing. File _test_trained_one_class_classifier_models.py_
can be used to test all One Class Classifier models. Moreover, _Binary_Classifier_ folder includes the
EfficientNetV2 model and the training/testing scripts.

The provided code is part of my Thesis 
[Detecting deepfakes by imprinting face’s 3D biometric characteristics](https://ikee.lib.auth.gr/record/343303/?ln=en) 
conducted during my final year studying at the Electrical & Computer Engineering School of Aristotle University of Thessaloniki.

