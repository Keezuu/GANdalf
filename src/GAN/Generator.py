from fastai.imports import torch
from torch import nn
import numpy as np
from src.resources import constants as cnst
from sklearn.preprocessing import OneHotEncoder

# DCGAN architecture based on pytorch DCGAN tutorial
class Generator(nn.Module):
    def __init__(self, n_classes, latent_dim, img_shape):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 8*cnst.GAN_GEN_FEATURE_MAPS,
                               kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(cnst.GAN_GEN_FEATURE_MAPS*8),
            nn.ReLU(True),
            # state size. (cnst.GAN_GEN_FEATURE_MAPS*8) x 4 x 4
            nn.ConvTranspose2d(cnst.GAN_GEN_FEATURE_MAPS * 8, cnst.GAN_GEN_FEATURE_MAPS * 4,
                               4, 2, 1, bias=False),
            nn.BatchNorm2d(cnst.GAN_GEN_FEATURE_MAPS * 4),
            nn.ReLU(True),
            # state size. (cnst.GAN_GEN_FEATURE_MAPS*4) x 8 x 8
            nn.ConvTranspose2d(cnst.GAN_GEN_FEATURE_MAPS * 4, cnst.GAN_GEN_FEATURE_MAPS * 2,
                               4, 2, 1, bias=False),
            nn.BatchNorm2d(cnst.GAN_GEN_FEATURE_MAPS * 2),
            nn.ReLU(True),
            # state size. (cnst.GAN_GEN_FEATURE_MAPS*2) x 16 x 16
            nn.ConvTranspose2d(cnst.GAN_GEN_FEATURE_MAPS*2, cnst.GAN_GEN_FEATURE_MAPS,
                               4, 2, 1, bias=False),
            nn.BatchNorm2d(cnst.GAN_GEN_FEATURE_MAPS),
            nn.ReLU(True),
            # state size. (cnst.GAN_GEN_FEATURE_MAPS) x 32 x 32
            # Output is number of channels, for BW images its 1
            nn.ConvTranspose2d(cnst.GAN_GEN_FEATURE_MAPS, cnst.CHANNELS_NUM,
                               4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (cnst.CHANNELS_NUM) x 64 x 64
        )


        # def block(in_chann, out_chann):
        #     layers = [nn.Linear(in_chann, out_chann),
        #               nn.BatchNorm1d(out_chann, 0.8),
        #               nn.LeakyReLU(0.2, inplace=True)]
        #     return layers
        #
        # self.model = nn.Sequential(
        #     *block(latent_dim + n_classes, 128),
        #     *block(128, 256),
        #     *block(256, 512),
        #     *block(512, 1024),
        #     nn.Linear(1024, int(np.prod(img_shape))),
        #     nn.Tanh()
        # )



    def forward(self, noise):
        # Concatenate label embedding and image to produce input
        # transformed = self.label_emb.transform(labels.reshape(-1, 1)).toarray()
        # embedded_tensor = torch.cuda.FloatTensor(transformed)
        #
        # gen_input = torch.cat((embedded_tensor, noise), 1)
        # img = self.model(gen_input)
        img = self.model(noise)
        #img = img.view(img.size(0), *self.img_shape)
        return img
