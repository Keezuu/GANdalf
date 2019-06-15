# DataAugmentationWithGAN
[WORK IN PROGRESS]
A cGAN implementation for the MNIST dataset in PyTorch with convolutional and feed-forward neural network implementations in Keras.

## TODO: 
a) Try to make the dataset even smaller (~1000 images) and optimize the GAN to achieve more impactful results.

b) Optimize the GAN to generate better images to achieve better results

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

## Conditional GAN
Conditional GAN algorithm is used instead of classic GAN to ensure balance
between classes of created images.
It is trained on 10 k of MNIST images to simulate "small" dataset.

Results of the last epoch:

![alt text](https://github.com/Jkeezuz/MNIST_GAN/raw/master/GAN_RESULTS/GAN_SAMPLES/samples/last_epoch.png "Logo Title Text 1")

## Training results

Accuracy throughout the training

![alt text](https://github.com/Jkeezuz/MNIST_GAN/raw/master/GAN_RESULTS/GAN_SAVES/save/accuracy.png  "Logo Title Text 1")

Loss throughout the training

![alt text](https://github.com/Jkeezuz/MNIST_GAN/raw/master/GAN_RESULTS/GAN_SAVES/save/loss.png  "Logo Title Text 1")

## Results of the classification
Accuracy of network while training only on MNIST dataset: 0.9795
![alt text](https://github.com/Jkeezuz/MNIST_GAN/raw/master/ALL_RESULTS/cnnnetacc.png "Logo Title Text 1")

Loss of network while training only on MNIST dataset: 0.126031
![alt text](https://github.com/Jkeezuz/MNIST_GAN/raw/master/ALL_RESULTS/cnnnetloss.png "Logo Title Text 1")

Accuracy of network while training on MNIST+GAN dataset: 0.9897
![alt text](https://github.com/Jkeezuz/MNIST_GAN/raw/master/ALL_RESULTS/cnn_mixednetacc.png "Logo Title Text 1")

Loss of network while training on MNIST+GAN dataset: 0.101791
![alt text](https://github.com/Jkeezuz/MNIST_GAN/raw/master/ALL_RESULTS/cnn_mixednetloss.png "Logo Title Text 1")

Accuracy of network while training only on GAN dataset: 0.8565
![alt text](https://github.com/Jkeezuz/MNIST_GAN/raw/master/ALL_RESULTS/cnn_gannetacc.png "Logo Title Text 1")

Loss of network while training only on GAN dataset: 2.291063
![alt text](https://github.com/Jkeezuz/MNIST_GAN/raw/master/ALL_RESULTS/cnn_gannetloss.png "Logo Title Text 1")
