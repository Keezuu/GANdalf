import datetime
import os
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import pylab
import numpy as np
import torch.utils.data
# Hyper-parameters
from torchvision.transforms import transforms

from src.GAN.Discriminator import Discriminator
from src.GAN.Generator import Generator
import src.resources.constants as cnst
from src.resources.utilities import sample_image, denorm, generate_gif, save_statistics, sample_same_label_image


# From pytorch DCGAN tutorial
# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# Create directories if they don't exist
if not os.path.exists(cnst.GAN_SAMPLES_DIR):
    os.makedirs(cnst.GAN_SAMPLES_DIR)

if not os.path.exists(cnst.GAN_SAVE_DIR):
    os.makedirs(cnst.GAN_SAVE_DIR)

# Get current date for naming folders

date = datetime.datetime.now().strftime("%m%d%H%M%S")

# Image processing
# Normalize the images to [-1, 1]
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])])

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

# Get GPU
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

# Create data loader with shuffling allowed
data_loader = torch.utils.data.DataLoader(dataset=mnist,
                                          batch_size=cnst.GAN_BATCH_SIZE,
                                          shuffle=True)

FloatTensor = torch.cuda.FloatTensor
LongTensor = torch.cuda.LongTensor

# Create discriminator and generator and force them to use GPU
D = Discriminator(img_shape=(28, 28), n_classes=10).cuda()
# Apply the weights init with value from a Normal distribution with mean=0, stdev=0.02.
# As stated in DCGAN paper
D.apply(weights_init)
# Create generator
G = Generator(img_shape=(28, 28), n_classes=10, latent_dim=cnst.GAN_LATENT_SIZE).cuda()
G.apply(weights_init)
print(G)
print(D)
# Create BCE loss function
bce_loss = nn.MSELoss().cuda()

# Create optimizers as specified in DCGAN paper
G_opt = torch.optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
D_opt = torch.optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

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
        valid = FloatTensor(batch_size, 1).fill_(1.0).cuda()
        fake = FloatTensor(batch_size, 1).fill_(0.0).cuda()

        # Configure input
        real_imgs = imgs.cuda()

        # -----------------
        #  Train Generator
        # -----------------

        G_opt.zero_grad()

        # Sample noise as generator input
        z = torch.randn(batch_size, cnst.GAN_LATENT_SIZE, 1, 1, device=device)
        gen_labels = np.random.randint(0, 10, batch_size)

        # Generate a batch of images
        gen_imgs = G(z)

        # Loss measures generator's ability to fool the discriminator
        validity = D(gen_imgs)

        g_loss = bce_loss(validity, valid)

        g_loss.backward()
        G_opt.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        D_opt.zero_grad()

        # Loss for real images
        validity_real = D(real_imgs)
        d_real_loss = bce_loss(validity_real, valid)
        real_score = validity_real

        # Loss for fake images
        validity_fake = D(gen_imgs.detach())
        d_fake_loss = bce_loss(validity_fake, fake)
        fake_score = validity_fake
        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2

        d_loss.backward()
        D_opt.step()

        # Update statistics
        d_losses[epoch] = d_losses[epoch] * (i / (i + 1.)) + d_loss.data * (1. / (i + 1.))
        g_losses[epoch] = g_losses[epoch] * (i / (i + 1.)) + g_loss.data * (1. / (i + 1.))
        real_scores[epoch] = real_scores[epoch] * (i / (i + 1.)) + real_score.mean().data * (1. / (i + 1.))
        fake_scores[epoch] = fake_scores[epoch] * (i / (i + 1.)) + fake_score.mean().data * (1. / (i + 1.))

        if i % 4 == 0:
            print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}'
                  .format(epoch, cnst.GAN_NUM_EPOCHS, i + 1, total_step, d_loss.data, g_loss.data,
                          real_score.mean().data, fake_score.mean().data))

        batches_done = epoch * len(data_loader) + i
        plt.clf()

    # Show samples for debug purposes
    # if epoch % 5 == 0:
    #     # Sample 4 noise and labels for debug and visualization purposes
    #     z = FloatTensor(np.random.normal(0, 1, (4, cnst.GAN_LATENT_SIZE)))
    #     sample_labels = np.random.randint(0, 10, 4)
    #
    #     # Generate a batch of images
    #     sample_imgs = G(z, sample_labels)
    #     for i, label in enumerate(sample_labels):
    #         plt.title('Label is {label}'.format(label=label))
    #         plt.imshow(sample_imgs.detach().cpu().numpy()[i], cmap='gray')
    #         plt.show()

    # Create save folder

    if not os.path.exists(os.path.join(cnst.GAN_SAVE_DIR, date)):
        os.makedirs(os.path.join(cnst.GAN_SAVE_DIR, date))
    if not os.path.exists(os.path.join(cnst.GAN_SAMPLES_DIR, date)):
        os.makedirs(os.path.join(cnst.GAN_SAMPLES_DIR, date))
    if not os.path.exists(os.path.join(cnst.GAN_MODEL_DIR, date)):
        os.makedirs(os.path.join(cnst.GAN_MODEL_DIR, date))

    # Save real images
    if (epoch + 1) == 1:
        images = imgs.view(imgs.size(0), 1, 28, 28)
        save_image(denorm(imgs.data), os.path.join(cnst.GAN_SAMPLES_DIR, date, 'real_images.png'))
    # Save sampled images
    if epoch % 5 == 0:
        sample_image(G, n_row=10, name=str(epoch).zfill(len(str(cnst.GAN_NUM_EPOCHS))),
                     path=os.path.join(cnst.GAN_SAMPLES_DIR, date))
    #   sample_same_label_image(G, available_labels=one_hot_labels, n_cols=10, name=str(epoch).zfill(len(str(cnst.GAN_NUM_EPOCHS))),
    #                path=os.path.join(cnst.GAN_SAMPLES_DIR+"-2", date))

    # Save and plot Statistics
    save_statistics(d_losses, g_losses, fake_scores, real_scores, os.path.join(cnst.GAN_SAVE_DIR, date))

    # Save model at checkpoints
    if (epoch + 1) % 50 == 0:
        torch.save(G.state_dict(), os.path.join(cnst.GAN_MODEL_DIR, date, 'G--{}.ckpt'.format(epoch + 1)))
        torch.save(D.state_dict(), os.path.join(cnst.GAN_MODEL_DIR, date, 'D--{}.ckpt'.format(epoch + 1)))

# Save the model checkpoints
torch.save(G.state_dict(), 'G.ckpt')
torch.save(D.state_dict(), 'D.ckpt')

# generate gif
filenames = os.listdir(os.path.join(cnst.GAN_SAMPLES_DIR, date, "img"))
generate_gif(filenames, save_path=os.path.join(cnst.GAN_SAVE_DIR, date),
             read_path=os.path.join(cnst.GAN_SAMPLES_DIR, date))
