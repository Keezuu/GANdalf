import os
import torch
import numpy as np
from torchvision.utils import save_image
from src.resources import constants as cnst
from src.GAN.Generator import Generator
import matplotlib.pyplot as plt
from torch import nn


# Create directories if they don't exist
if not os.path.exists(cnst.GAN_SAMPLES_DIR):
    os.makedirs(cnst.GAN_SAMPLES_DIR)

if not os.path.exists(cnst.GAN_SAVE_DIR):
    os.makedirs(cnst.GAN_SAVE_DIR)

n_classess = 10

# Import MNIST dataset
G = Generator(img_shape=(28, 28), n_classes=n_classess, latent_dim=cnst.GAN_LATENT_SIZE).cuda()
FloatTensor = torch.cuda.FloatTensor
LongTensor = torch.cuda.LongTensor

G.load_state_dict(torch.load('G.ckpt'))

num_of_images = 2
z = FloatTensor(np.random.normal(0, 1, (num_of_images, cnst.GAN_LATENT_SIZE)))
# Get labels ranging from 0 to n_classes for n rows
labels = np.array([1 for x in range(num_of_images)])

available_labels = [3]
single_label_imgs_num = 100
z = torch.cuda.FloatTensor(np.random.normal(0, 1,
                                            (len(available_labels)*single_label_imgs_num ,
                                             cnst.GAN_LATENT_SIZE)))
# Get labels ranging from 0 to n_classes for n rows
labels = np.array([label for label in available_labels for _ in range(single_label_imgs_num)])

one_hot_labels = np.arange(10)
G.train_one_hot(one_hot_labels)
gen_imgs = G(z, labels)

save_image(gen_imgs.reshape(gen_imgs.shape[0], 1, gen_imgs.shape[1], gen_imgs.shape[2]).data,
           os.path.join(cnst.GAN_SAMPLES_DIR, "generatedVIS3.png"), nrow=n_classess, normalize=True)
save_image(gen_imgs.reshape(gen_imgs.shape[0], 1, gen_imgs.shape[1], gen_imgs.shape[2]).data,
           os.path.join(cnst.GAN_SAMPLES_DIR, "generatedVIS4.png"), nrow=n_classess*10, normalize=True)
gen_imgs = gen_imgs.cpu().data.numpy()

# Move the data to CPU and then copy it to numpy array
np.save(file=os.path.join(cnst.GAN_DATA_DIR, "gan_images.npy"), arr=gen_imgs)
np.save(file=os.path.join(cnst.GAN_DATA_DIR, "gan_labels.npy"), arr=labels)



for i, label in enumerate(labels):
    plt.title('Label is {label}'.format(label=label))
    plt.imshow(gen_imgs[i], cmap='gray')
    plt.show()
    if i > 5:
        break
print("")