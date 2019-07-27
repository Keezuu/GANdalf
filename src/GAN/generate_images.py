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
G = Generator(n_classes=n_classess, latent_dim=cnst.GAN_LATENT_SIZE)
FloatTensor = torch.cuda.FloatTensor
LongTensor = torch.cuda.LongTensor

G.load_state_dict(torch.load('G.ckpt', map_location='cpu'))

num_of_images_per_class = 2
z = FloatTensor(np.random.normal(0, 1, (num_of_images_per_class, cnst.GAN_LATENT_SIZE)))
# Get labels ranging from 0 to n_classes for n rows
available_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
single_label_imgs_num = 1000
z = torch.randn(len(available_labels) * single_label_imgs_num, cnst.GAN_LATENT_SIZE, 1, 1)
# Get labels ranging from 0 to n_classes for n rows

labels = np.array([label for label in available_labels for _ in range(single_label_imgs_num)])
labels = torch.LongTensor(labels)

gen_imgs = G(z, labels)
G.eval()

save_image(gen_imgs.data, os.path.join(cnst.GAN_SAMPLES_DIR, "generatedVIS3.png"), nrow=n_classess, normalize=True)
save_image(gen_imgs.data, os.path.join(cnst.GAN_SAMPLES_DIR, "generatedVIS4.png"), nrow=n_classess * 10, normalize=True)
gen_imgs = gen_imgs.cpu().data.numpy()
labels = labels.cpu().data.numpy()

# Move the data to CPU and then copy it to numpy array
np.save(file=os.path.join(cnst.GAN_DATA_DIR, "gan_images.npy"), arr=gen_imgs)
np.save(file=os.path.join(cnst.GAN_DATA_DIR, "gan_labels.npy"), arr=labels)

cpu_imgs = np.squeeze(gen_imgs)
for i, label in enumerate(labels):
    plt.title('Label is {label}'.format(label=label))
    plt.imshow(cpu_imgs[i], cmap='gray')
    plt.show()
    if i > 5:
        break
print("")
