import os
import torch
from torch.autograd import Variable
import numpy as np
from torchvision.utils import save_image

from src.GAN.Generator import Generator
import matplotlib.pyplot as plt


latent_size = 64
hidden_size = 256
image_size = 28*28
batch_size = 64
sample_dir = 'samples'
chckpt_dir = 'save'
save_dir = "data/GAN"




# Create directories if they don't exist
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Import MNIST dataset
G = Generator(img_shape=(28, 28), n_classes=10, latent_dim=latent_size).cuda()
FloatTensor = torch.cuda.FloatTensor
LongTensor = torch.cuda.LongTensor

G.load_state_dict(torch.load('G.ckpt'))

n_class = 10
# Generate 1000 images from class "i" and save them in npy file
z = Variable(FloatTensor(np.random.normal(0, 1, (10000, latent_size))))
# Get labels ranging from 0 to n_classes for n rows
labels = np.array([num for _ in range(1000) for num in range(n_class)])

labels = Variable(LongTensor(labels))
gen_imgs = G(z, labels)
save_image(gen_imgs.reshape(gen_imgs.shape[0], 1, gen_imgs.shape[1], gen_imgs.shape[2]).data,
           "samples/generatedVIS.png", nrow=n_class, normalize=True)
save_image(gen_imgs.reshape(gen_imgs.shape[0], 1, gen_imgs.shape[1], gen_imgs.shape[2]).data,
           "samples/generatedVIS2.png", nrow=n_class*10, normalize=True)
gen_imgs = gen_imgs.cpu().data.numpy()
labels = labels.cpu().data.numpy()
# Move the data to CPU and then copy it to numpy array
np.save(file=os.path.join(save_dir, "gan_images.npy"), arr=gen_imgs)
np.save(file=os.path.join(save_dir, "gan_labels.npy"), arr=labels)

#visualize
img_array = np.load(os.path.join(save_dir, "gan_images.npy"))
for i in range(10):
    plt.title('Label is {label}'.format(label=i))
    plt.imshow(img_array[i], cmap='gray')
    plt.show()
print("")