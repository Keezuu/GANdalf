# GANdalf

Conditional DCGAN implementation for the MNIST dataset in PyTorch with classification using convolutional neural network implementation in Keras.

## Purpose

The purpose of this implementation is to train DCGAN on reduced number of MNIST images (currently 10k out of original 60k) to generate a dataset with similar data distribution as the original dataset and check how well does adding the generated dataset to the reduced MNIST dataset and training classification networks on that improve results of recognizing the images from original MNIST dataset against training only on MNIST dataset.

## Dataset

The original dataset is reduced from 60 000 training samples and 10 000 test samples
to 8 000 training and 2 000 validation samples.
While training 8 000 training and 2 000 validation samples from GAN generated images is added to the reduced MNIST dataset of 10K images which makes it 18k of training and 2k of validation data.

## Conditional Deep Convolution Generative Adversarial Network (conditional DCGAN)

Conditional DCGAN is used to learn the data distribution of reduced MNIST dataset and generate new images after being trained. 
The architecture still changes as I'm trying to improve the model.

<b>Current Generator Model</b>: https://github.com/Jkeezuz/GANdalf/blob/master/src/GAN/Generator.py
```
Generator(
  (label_embedding): Embedding(10, 10)
  (model): Sequential(
    (0): ConvTranspose2d(74, 512, kernel_size=(3, 3), stride=(1, 1), bias=False)
    (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace)
    (3): ConvTranspose2d(512, 256, kernel_size=(3, 3), stride=(2, 2), bias=False)
    (4): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): ReLU(inplace)
    (6): ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (7): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (8): ReLU(inplace)
    (9): ConvTranspose2d(128, 1, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (10): Tanh()
  )
)
```
<b>Current Discriminator Model</b>: https://github.com/Jkeezuz/GANdalf/blob/master/src/GAN/Discriminator.py
```
Discriminator(
  (label_embedding): Embedding(10, 784)
  (conv_layer): Sequential(
    (0): Conv2d(2, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (1): LeakyReLU(negative_slope=0.2, inplace)
    (2): Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (4): LeakyReLU(negative_slope=0.2, inplace)
    (5): Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (6): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (7): LeakyReLU(negative_slope=0.2, inplace)
  )
  (conv_layer_out): Sequential(
    (0): Conv2d(256, 1, kernel_size=(3, 3), stride=(1, 1), bias=False)
    (1): Sigmoid()
  )
)
```
There's separate layer for output because I was experimenting with having different layers as outputs (for example few full-connected layers).

The labels in discriminator are embedded as 28 x 28 vectors and appended as additional channel to the image. The labels in generator are embedded as 10x1 vectors and appended to the noise.

The model was trained on batches of size 256, for 60 epochs.

## Sample results after training on 10k MNIST images for 60 epochs

![alt text](https://github.com/Jkeezuz/GANdalf/raw/master/GAN_RESULTS/GAN_SAMPLES/samples/last_epochDCGAN.png "")

## Sample results after training on whole MNIST images for 8 epochs on 64 batch size and smaller amount of feature maps in layers

![missing image](https://github.com/Jkeezuz/GANdalf/raw/master/GAN_RESULTS/GAN_SAMPLES/samples/fullmnist.png "")

## Gif visualizing progress throughout training

![alt text](https://github.com/Jkeezuz/GANdalf/raw/master/GAN_RESULTS/GAN_SAMPLES/samples/resultgifDCGAN.gif "")


## Training results

### Accuracy throughout DCGAN training

![alt text](https://github.com/Jkeezuz/GANdalf/raw/master/GAN_RESULTS/GAN_SAVES/save/accuracyDCGAN_new.png "")

### Loss throughout DCGAN training

![alt text](https://github.com/Jkeezuz/GANdalf/raw/master/GAN_RESULTS/GAN_SAVES/save/lossDCGAN_new.png "")


## Results of the classification
Accuracy of network while training only on 10kMNIST dataset: 0.979</br>

![alt text](https://github.com/Jkeezuz/GANdalf/raw/master/ALL_RESULTS/DCGAN_classification/cnn/netacc.png "")

Loss of network while training only on 10kMNIST dataset: 0.09162351002503419</br>

![alt text](https://github.com/Jkeezuz/GANdalf/raw/master/ALL_RESULTS/DCGAN_classification/cnn/netloss.png "")

Accuracy of network while training on 10kMNIST+10kGAN dataset: 0.9782051282051282</br>
 (validation was done only on MNIST images not used for training!)

![alt text](https://github.com/Jkeezuz/GANdalf/raw/master/ALL_RESULTS/DCGAN_classification/cnn_mixed/netacc.png "")

Loss of network while training on 10kMNIST+10kGAN dataset: 0.18733093237085985</br>
(validation was done only on MNIST images not used for training!)

![alt text](https://github.com/Jkeezuz/GANdalf/raw/master/ALL_RESULTS/DCGAN_classification/cnn_mixed/netloss.png "")

Accuracy of network while training only on 10kGAN dataset: 0.811</br>
(validation was done only on MNIST images not used for training!)

![alt text](https://github.com/Jkeezuz/GANdalf/raw/master/ALL_RESULTS/DCGAN_classification/cnn_gan/netacc.png "")

Loss of network while training only on 10kGAN dataset: 3.025749439239502 </br>
(validation was done only on MNIST images not used for training!)

![alt text](https://github.com/Jkeezuz/GANdalf/raw/master/ALL_RESULTS/DCGAN_classification/cnn_gan/netloss.png "")

## Summary

The results are, of course, slightly dissapointing. Adding 10k of mnist images generated on <b>this</b> DCGAN model does not improve the results of classifying images on <b>these</b> CNN models. There's many things that affect those results. One possible explanation is that adding those generated mnist images introduces more diversity to the data distribution the CNNs have to learn therefore CNNs learn distributions which are not present in the original MNIST dataset, it could be verified by training CNN on GAN+MNIST like I did but validating also on both of these datasets. 

There's two things I can do now to improve the results.
First of all is just to generate more data, having a trained GAN we can generate far more than 10K, we can try 50K images for example, with the 10k from mnist it would give us 60K images, just like the original. This is the first thing I want to try while trying to improve those results. Second thing is tweaking the parameters of DCGAN and also trying out different number of layers and kernels etc. which should provide better results. 

# ----------------------------------------------

# OLD - CONDITIONAL GAN
## [ARCHITECTURE USED FOR BELOW RESULTS]
Below results were achieved on conditional GAN (not DCGAN) as a part of university project aiming at comparing accuracy obtained by feed-forward networks versus convolutional neural networks. The project was extended by trying to increase the limited MNIST dataset (number of training samples stated below) by generating new images with conditional Generative Adversarial Network. The network currently is being rebuild into conditional DCGAN as it is proven to provide better results for image generation. The dataset may later be swapped to more complex like CIFAR-10. The projects itself is kind of a sandbox for learning the mechanisms of Generative Adversarial Networks and the logic behind them.


## Conditional GAN
Conditional GAN algorithm is used instead of classic GAN to ensure balance
between classes of created images.
It is trained on 10 k of MNIST images to simulate "small" dataset.

Results of the last epoch:

![alt text](https://github.com/Jkeezuz/MNIST_GAN/raw/master/GAN_RESULTS/GAN_SAMPLES/samples/last_epoch.png "")

## Training results

Accuracy throughout the training

![alt text](https://github.com/Jkeezuz/MNIST_GAN/raw/master/GAN_RESULTS/GAN_SAVES/save/accuracy.png  "")

Loss throughout the training

![alt text](https://github.com/Jkeezuz/MNIST_GAN/raw/master/GAN_RESULTS/GAN_SAVES/save/loss.png  "")

## Results of the classification
Accuracy of network while training only on MNIST dataset: 0.9795
![alt text](https://github.com/Jkeezuz/MNIST_GAN/raw/master/ALL_RESULTS/cnnnetacc.png "")

Loss of network while training only on MNIST dataset: 0.126031
![alt text](https://github.com/Jkeezuz/MNIST_GAN/raw/master/ALL_RESULTS/cnnnetloss.png "")

Accuracy of network while training on MNIST+GAN dataset: 0.9897
![alt text](https://github.com/Jkeezuz/MNIST_GAN/raw/master/ALL_RESULTS/cnn_mixednetacc.png "")

Loss of network while training on MNIST+GAN dataset: 0.101791
![alt text](https://github.com/Jkeezuz/MNIST_GAN/raw/master/ALL_RESULTS/cnn_mixednetloss.png "")

Accuracy of network while training only on GAN dataset: 0.8565
![alt text](https://github.com/Jkeezuz/MNIST_GAN/raw/master/ALL_RESULTS/cnn_gannetacc.png "")

Loss of network while training only on GAN dataset: 2.291063
![alt text](https://github.com/Jkeezuz/MNIST_GAN/raw/master/ALL_RESULTS/cnn_gannetloss.png "")
