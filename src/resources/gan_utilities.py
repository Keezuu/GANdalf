import torchvision
import torch.nn as nn
from torch.nn.functional import mse_loss

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


def batch_train_gan(G, D, G_opt, D_opt, loss,  batch_size, real_imgs, real_labels, valid, fake, device):
    """
    Train GAN on one batch.
    """
    # -----------------
    #  Train Generator
    # -----------------

    G_opt.zero_grad()

    # Sample noise as generator input
    z = torch.randn(batch_size, cnst.GAN_LATENT_SIZE, 1, 1, device=device)
    gen_labels = torch.cuda.LongTensor(np.random.randint(0, 10, batch_size))

    # Generate a batch of images
    gen_imgs = G(z, gen_labels)

    # Loss measures generator's ability to fool the discriminator
    validity = D(gen_imgs, gen_labels)

    # We try to maximize log(D(G(z))) as it doesn't have vanishing gradients
    # whereas trying to minimize log(1-D(G(z))) does
    # Goodfellow et. al (2014)
    g_loss = loss(validity, valid)

    g_loss.backward()
    G_opt.step()

    # ---------------------
    #  Train Discriminator
    # ---------------------

    D_opt.zero_grad()

    # Training on batch of fake and batch of real images separately
    # as proposed in tips to training gans: https://github.com/soumith/ganhacks
    # Loss for real images
    validity_real = D(real_imgs, real_labels)
    d_real_loss = loss(validity_real, valid)
    real_score = validity_real

    # Loss for fake images
    validity_fake = D(gen_imgs.detach(), gen_labels)
    d_fake_loss = loss(validity_fake, fake)
    fake_score = validity_fake

    # Total discriminator loss
    d_loss = (d_real_loss + d_fake_loss) / 2

    d_loss.backward()
    D_opt.step()

    return g_loss, d_loss, real_score, fake_score


def show_samples(G, epoch, device):
    # Show samples for debug purposes
    if epoch % 5 == 0:
        # Sample 4 noise and labels for debug and visualization purposes
        z = torch.randn(4, cnst.GAN_LATENT_SIZE, 1, 1, device=device)
        sample_labels = torch.cuda.LongTensor(np.random.randint(0, 10, 4))

        # Generate a batch of images
        sample_imgs = G(z, sample_labels)
        cpu_imgs = sample_imgs.detach().cpu().numpy()
        cpu_imgs = np.squeeze(cpu_imgs)
        for i, label in enumerate(sample_labels):
            plt.title('Label is {label}'.format(label=label))
            plt.imshow(cpu_imgs[i], cmap='gray')
            plt.show()
