from fastai.imports import torch
from torch import nn
import numpy as np
from src.resources import constants as cnst


# DCGAN architecture based on pytorch DCGAN tutorial
class ResConvGenerator(nn.Module):
    def __init__(self, n_classes, latent_dim, img_shape):
        super(ResConvGenerator, self).__init__()

        self.label_embedding = nn.Embedding(n_classes, n_classes)

        self.model = nn.Sequential(
            # input size is latent_dim
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.ReflectionPad2d(1),
            nn.Conv2d(latent_dim + n_classes, 8 * cnst.GAN_GEN_FEATURE_MAPS,
                      kernel_size=2, stride=1, padding=0),
            nn.BatchNorm2d(cnst.GAN_GEN_FEATURE_MAPS * 8),
            nn.ReLU(True),
            # out size. (cnst.GAN_GEN_FEATURE_MAPS*8) x 3 x 3
            # ---------
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.ReflectionPad2d(1),
            nn.Conv2d(8 * cnst.GAN_GEN_FEATURE_MAPS, 4 * cnst.GAN_GEN_FEATURE_MAPS,
                      kernel_size=2, stride=1, padding=0),
            nn.BatchNorm2d(cnst.GAN_GEN_FEATURE_MAPS * 4),
            nn.ReLU(True),
            # out size. (cnst.GAN_GEN_FEATURE_MAPS*4) x 7 x 7
            # ---------
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.ReflectionPad2d(1),
            nn.Conv2d(4 * cnst.GAN_GEN_FEATURE_MAPS, 2 * cnst.GAN_GEN_FEATURE_MAPS,
                      kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(cnst.GAN_GEN_FEATURE_MAPS * 2),
            nn.ReLU(True),
            # out size. (cnst.GAN_GEN_FEATURE_MAPS*2) x 14 x 14
            # ---------
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.ReflectionPad2d(1),
            nn.Conv2d(2 * cnst.GAN_GEN_FEATURE_MAPS, cnst.CHANNELS_NUM,
                      kernel_size=3, stride=1, padding=0),
            nn.Tanh()
            # out size (cnst.CHANNELS_NUM) x 28 x 28
            # ---------
        )

    def forward(self, noise, labels):
        labels = self.label_embedding(labels)
        labels = labels.reshape(labels.size(0), labels.size(1), 1, 1)
        noise = torch.cat((labels, noise), 1)
        img = self.model(noise)
        return img
