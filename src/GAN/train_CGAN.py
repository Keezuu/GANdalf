import os
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import matplotlib.pyplot as plt
import pylab
import numpy as np
# Hyper-parameters
from torchvision.transforms import transforms

from src.GAN.Discriminator import Discriminator
from src.GAN.Generator import Generator
import src.resources.constants as cnst


# Create directories if they don't exist
if not os.path.exists(cnst.GAN_SAMPLES_DIR):
    os.makedirs(cnst.GAN_SAMPLES_DIR)

if not os.path.exists(cnst.GAN_SAVE_DIR):
    os.makedirs(cnst.GAN_SAVE_DIR)

# Image processing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])])

# Import MNIST dataset

mnist = torchvision.datasets.MNIST(root=cnst.DATA_DIR,
                                   train=True,
                                   transform=transform,
                                   download=True)

# Create data loader with shuffling allowed
data_loader = torch.utils.data.DataLoader(dataset=mnist,
                                          batch_size=cnst.GAN_BATCH_SIZE,
                                          shuffle=True)

# Create discriminator and generator and force them to use GPU
D = Discriminator(img_shape=(28, 28), n_classes=10).cuda()

G = Generator(img_shape=(28, 28), n_classes=10,latent_dim=cnst.GAN_LATENT_SIZE).cuda()

# Create MSE loss function
adv_loss = nn.MSELoss().cuda()

#Create optimizers
G_opt = torch.optim.Adam(G.parameters(), lr=0.0002)
D_opt = torch.optim.Adam(D.parameters(), lr=0.0002)

FloatTensor = torch.cuda.FloatTensor
LongTensor = torch.cuda.LongTensor


def sample_image(n_row, batches_done):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, cnst.GAN_LATENT_SIZE))))
    # Get labels ranging from 0 to n_classes for n rows
    labels = np.array([num for _ in range(n_row) for num in range(n_row)])
    labels = Variable(LongTensor(labels))
    gen_imgs = G(z, labels)
    save_image(gen_imgs.reshape(gen_imgs.shape[0], 1, gen_imgs.shape[1], gen_imgs.shape[2]).data, "samples/%d.png" % batches_done, nrow=n_row, normalize=True)


def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

# Statistics to be saved
d_losses = np.zeros(cnst.GAN_NUM_EPOCHS)
g_losses = np.zeros(cnst.GAN_NUM_EPOCHS)
real_scores = np.zeros(cnst.GAN_NUM_EPOCHS)
fake_scores = np.zeros(cnst.GAN_NUM_EPOCHS)


# Training
batches_done = 0
total_step = len(data_loader)
for epoch in range(cnst.GAN_NUM_EPOCHS):
    for i, (imgs, labels) in enumerate(data_loader):
        batch_size = imgs.shape[0]
        # Adversarial ground truths
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)
        # Configure input
        real_imgs = Variable(imgs.type(FloatTensor))
        labels = Variable(labels.type(LongTensor))
        # -----------------
        #  Train Generator
        # -----------------

        G_opt.zero_grad()

        # Sample noise and labels as generator input
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, cnst.GAN_LATENT_SIZE))))
        gen_labels = Variable(LongTensor(np.random.randint(0, 10, batch_size)))

        # Generate a batch of images
        gen_imgs = G(z, gen_labels)

        # Loss measures generator's ability to fool the discriminator
        validity = D(gen_imgs, gen_labels)
        g_loss = adv_loss(validity, valid)


        g_loss.backward()
        G_opt.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        D_opt.zero_grad()

        # Loss for real images
        validity_real = D(real_imgs, labels)
        d_real_loss = adv_loss(validity_real, valid)
        real_score = validity_real

        # Loss for fake images
        validity_fake = D(gen_imgs.detach(), gen_labels)
        d_fake_loss = adv_loss(validity_fake, fake)
        fake_score = validity_fake
        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2

        d_loss.backward()
        D_opt.step()

        #Update statistics
        d_losses[epoch] = d_losses[epoch] * (i / (i + 1.)) + d_loss.data * (1. / (i + 1.))
        g_losses[epoch] = g_losses[epoch] * (i / (i + 1.)) + g_loss.data * (1. / (i + 1.))
        real_scores[epoch] = real_scores[epoch] * (i / (i + 1.)) + real_score.mean().data * (1. / (i + 1.))
        fake_scores[epoch] = fake_scores[epoch] * (i / (i + 1.)) + fake_score.mean().data * (1. / (i + 1.))

        if (i + 1) % 20 == 0:
            print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}'
                  .format(epoch, cnst.GAN_NUM_EPOCHS, i + 1, total_step, d_loss.data, g_loss.data,
                          real_score.mean().data, fake_score.mean().data))

        batches_done = epoch * len(data_loader) + i

    #torch.save(G, os.path.join(save_dir, 'generative_model'+str(epoch)+".pth"))
    #torch.save(D, os.path.join(save_dir, 'discriminative_model'+str(epoch)+".pth"))
    # Save real images
    if (epoch + 1) == 1:
        images = imgs.view(imgs.size(0), 1, 28, 28)
        save_image(denorm(imgs.data), os.path.join(cnst.GAN_SAMPLES_DIR, 'real_images.png'))
    # Save sampled images
    sample_image(n_row=10, batches_done=batches_done)

    # Save and plot Statistics
    np.save(os.path.join(cnst.GAN_SAVE_DIR, 'd_losses.npy'), d_losses)
    np.save(os.path.join(cnst.GAN_SAVE_DIR, 'g_losses.npy'), g_losses)
    np.save(os.path.join(cnst.GAN_SAVE_DIR, 'fake_scores.npy'), fake_scores)
    np.save(os.path.join(cnst.GAN_SAVE_DIR, 'real_scores.npy'), real_scores)

    plt.figure()
    pylab.xlim(0, cnst.GAN_NUM_EPOCHS + 1)
    plt.plot(range(1, cnst.GAN_NUM_EPOCHS + 1), d_losses, label='d loss')
    plt.plot(range(1, cnst.GAN_NUM_EPOCHS + 1), g_losses, label='g loss')
    plt.legend()
    plt.savefig(os.path.join(cnst.GAN_SAVE_DIR, 'loss.pdf'))
    plt.close()

    plt.figure()
    pylab.xlim(0, cnst.GAN_NUM_EPOCHS + 1)
    pylab.ylim(0, 1)
    plt.plot(range(1, cnst.GAN_NUM_EPOCHS + 1), fake_scores, label='fake score')
    plt.plot(range(1, cnst.GAN_NUM_EPOCHS + 1), real_scores, label='real score')
    plt.legend()
    plt.savefig(os.path.join(cnst.GAN_SAVE_DIR, 'accuracy.pdf'))
    plt.close()
    # Save model at checkpoints
    if (epoch+1) % 50 == 0:
        torch.save(G.state_dict(), os.path.join(cnst.GAN_SAVE_DIR, 'G--{}.ckpt'.format(epoch+1)))
        torch.save(D.state_dict(), os.path.join(cnst.GAN_SAVE_DIR, 'D--{}.ckpt'.format(epoch+1)))

# Save the model checkpoints
torch.save(G.state_dict(), 'G.ckpt')
torch.save(D.state_dict(), 'D.ckpt')