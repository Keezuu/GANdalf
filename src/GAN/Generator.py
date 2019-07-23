from fastai.imports import torch
from torch import nn
from src.resources import constants as cnst


class Generator(nn.Module):
    def __init__(self, n_classes, latent_dim):
        super(Generator, self).__init__()

        self.label_embedding = nn.Embedding(n_classes, n_classes)

        self.model = nn.Sequential(
            # input size is latent_dim
            nn.ConvTranspose2d(latent_dim + n_classes, 8 * cnst.GAN_GEN_FEATURE_MAPS,
                               kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(cnst.GAN_GEN_FEATURE_MAPS * 8),
            nn.ReLU(True),
            # out size. (cnst.GAN_GEN_FEATURE_MAPS*8) x 3 x 3
            # ---------
            nn.ConvTranspose2d(cnst.GAN_GEN_FEATURE_MAPS * 8, cnst.GAN_GEN_FEATURE_MAPS * 4,
                               3, 2, 0, bias=False),
            nn.BatchNorm2d(cnst.GAN_GEN_FEATURE_MAPS * 4),
            nn.ReLU(True),
            # out size. (cnst.GAN_GEN_FEATURE_MAPS*4) x 7 x 7
            # ---------
            nn.ConvTranspose2d(cnst.GAN_GEN_FEATURE_MAPS * 4, cnst.GAN_GEN_FEATURE_MAPS * 2,
                               4, 2, 1, bias=False),
            nn.BatchNorm2d(cnst.GAN_GEN_FEATURE_MAPS * 2),
            nn.ReLU(True),
            # out size. (cnst.GAN_GEN_FEATURE_MAPS*2) x 14 x 14
            # ---------
            nn.ConvTranspose2d(cnst.GAN_GEN_FEATURE_MAPS * 2, cnst.CHANNELS_NUM,
                               4, 2, 1, bias=False),
            nn.Tanh()
            # out size (cnst.CHANNELS_NUM) x 28 x 28
            # ---------
        )

    def preprocess(self, noise, labels):
        n_labels = self.label_embedding(labels)
        n_labels = labels.reshape(labels.size(0), labels.size(1), 1, 1)
        n_noise = torch.cat((labels, noise), 1)
        return n_noise

    def forward(self, noise, labels):
        n_noise = self.preprocess(noise, labels)
        img = self.model(n_noise)
        return img
