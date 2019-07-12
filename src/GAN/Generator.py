from fastai.imports import torch
from torch import nn
import numpy as np
from src.resources import constants as cnst


# DCGAN architecture based on pytorch DCGAN tutorial
class Generator(nn.Module):
    def __init__(self, n_classes, latent_dim, img_shape):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 8 * cnst.GAN_GEN_FEATURE_MAPS,
                               kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(cnst.GAN_GEN_FEATURE_MAPS * 8),
            nn.ReLU(True),
            # state size. (cnst.GAN_GEN_FEATURE_MAPS*8) x 3 x 3
            nn.ConvTranspose2d(cnst.GAN_GEN_FEATURE_MAPS * 8, cnst.GAN_GEN_FEATURE_MAPS * 4,
                               3, 2, 0, bias=False),
            nn.BatchNorm2d(cnst.GAN_GEN_FEATURE_MAPS * 4),
            nn.ReLU(True),
            # state size. (cnst.GAN_GEN_FEATURE_MAPS*4) x 7 x 7
            nn.ConvTranspose2d(cnst.GAN_GEN_FEATURE_MAPS * 4, cnst.GAN_GEN_FEATURE_MAPS * 2,
                               4, 2, 1, bias=False),
            nn.BatchNorm2d(cnst.GAN_GEN_FEATURE_MAPS * 2),
            nn.ReLU(True),
            # state size. (cnst.GAN_GEN_FEATURE_MAPS*2) x 14 x 14
            nn.ConvTranspose2d(cnst.GAN_GEN_FEATURE_MAPS * 2, cnst.CHANNELS_NUM,
                               4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (cnst.CHANNELS_NUM) x 28 x 28
            # Output is number of channels, for BW images its 1
        )

    def forward(self, noise):
        # Concatenate label embedding and image to produce input
        # transformed = self.label_emb.transform(labels.reshape(-1, 1)).toarray()
        # embedded_tensor = torch.cuda.FloatTensor(transformed)
        #
        # gen_input = torch.cat((embedded_tensor, noise), 1)
        # img = self.model(gen_input)
        img = self.model(noise)
        # img = img.view(img.size(0), *self.img_shape)
        return img
