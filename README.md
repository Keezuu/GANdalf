# MNIST_GAN [WORK IN PROGRESS]
A cGAN implementation for the MNIST dataset.

## Purpose
The purpose of this implementation is to generate a dataset similar to the MNIST
dataset and check how well does adding this dataset to the original one improve
the results of classification using convolutional neural network.

## Conditional GAN
Conditional GAN algorithm is used instead of classic GAN to ensure balance
between classes of created images.

Results of the last epoch:

![alt text](https://github.com/Jkeezuz/MNIST_GAN/raw/master/GAN/samples/last_epoch.png "Logo Title Text 1")

## Training results

Accuracy throughout the training

![alt text](https://github.com/Jkeezuz/MNIST_GAN/raw/master/GAN/save/accuracy.png "Logo Title Text 1")

Loss throughout the training

![alt text](https://github.com/Jkeezuz/MNIST_GAN/raw/master/GAN/save/loss.png "Logo Title Text 1")
