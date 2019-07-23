import torchvision
import torch.nn as nn
from torch.nn.functional import mse_loss

from src.resources.gan_utilities import *

from src.resources.utilities import *


# From pytorch DCGAN tutorial
# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# Create dataloader and return it
def get_data(transform):
    # Import MNIST dataset
    mnist = torchvision.datasets.MNIST(root=cnst.DATA_DIR,
                                       train=True,
                                       transform=transform,
                                       download=True)

    # Get only a part of the data if specified training size is less than the size of mnist dataset
    if cnst.GAN_MNIST_TRAINING_SIZE < len(mnist.data):
        subset_indices = [x for x in range(cnst.GAN_MNIST_TRAINING_SIZE)]
        mnist = torch.utils.data.Subset(mnist, subset_indices)
        # Check how many instances of each class there is

        cnt_labels = np.zeros(10)
        for i in subset_indices:
            cnt_labels[mnist[i][-1]] += 1
        print("Number of image from given class: ")

        for idx, val in enumerate(cnt_labels):
            print(str(idx) + ": " + str(val))

        # Create data loader with shuffling allowed
        data_loader = torch.utils.data.DataLoader(dataset=mnist,
                                                  batch_size=cnst.GAN_BATCH_SIZE,
                                                  shuffle=True)
        return data_loader
