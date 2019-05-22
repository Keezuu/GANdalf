# MNIST_GAN [WORK IN PROGRESS]
A cGAN implementation for the MNIST dataset in PyTorch with convolutional and feed-forward neural network implementations in Keras.

## Purpose
The purpose of this implementation is to generate a dataset similar to the MNIST
dataset and check how well does:

a) Adding the generated dataset to the original MNIST dataset and training networks on that

b) Training the networks on the generated dataset

improve results of recognizing the images from MNIST dataset against training only on MNIST dataset.

## Dataset

The original dataset is reduced from 60 000 training samples and 10 000 test samples
to 8 000 training and 2 000 validation samples.
While training on condition a) 8 000 training and 2 000 validation samples from GAN generated images is added to the dataset and while training on condition b) 8 000 samples from GAN is used to train the network and 2 000 samples from MNIST dataset is used for validation.

## Results of the classification


## Conditional GAN
Conditional GAN algorithm is used instead of classic GAN to ensure balance
between classes of created images.
It is trained on 10 k of MNIST images to simulate "small" dataset.

Results of the last epoch:

![alt text](https://github.com/Jkeezuz/MNIST_GAN/raw/master/GAN_RESULTS/GAN_SAMPLES/samples/last_epoch.png "Logo Title Text 1")

## Training results

Accuracy throughout the training

![alt text](https://github.com/Jkeezuz/MNIST_GAN/raw/master/src/GAN/save/accuracy.png "Logo Title Text 1")

Loss throughout the training

![alt text](https://github.com/Jkeezuz/MNIST_GAN/raw/master/src/GAN/save/loss.png "Logo Title Text 1")
