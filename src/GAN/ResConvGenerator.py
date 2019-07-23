from fastai.imports import torch
from torch import nn
from src.resources import constants as cnst


class ResConvGenerator(nn.Module):
    def __init__(self, n_classes, latent_dim):
        super(ResConvGenerator, self).__init__()

        self.label_embedding = nn.Embedding(n_classes, n_classes)

        self.model = nn.Sequential(
            # input size is latent_dim
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(latent_dim + n_classes, 8 * cnst.GAN_GEN_FEATURE_MAPS,
                      kernel_size=2, stride=1, padding=0),
            nn.BatchNorm2d(cnst.GAN_GEN_FEATURE_MAPS * 8),
            nn.ReLU(True),
            # out size. (cnst.GAN_GEN_FEATURE_MAPS*8) x 3 x 3
            # ---------
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(8 * cnst.GAN_GEN_FEATURE_MAPS, 4 * cnst.GAN_GEN_FEATURE_MAPS,
                      kernel_size=2, stride=1, padding=0),
            nn.BatchNorm2d(cnst.GAN_GEN_FEATURE_MAPS * 4),
            nn.ReLU(True),
            # out size. (cnst.GAN_GEN_FEATURE_MAPS*4) x 7 x 7
            # ---------
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(4 * cnst.GAN_GEN_FEATURE_MAPS, 2 * cnst.GAN_GEN_FEATURE_MAPS,
                      kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(cnst.GAN_GEN_FEATURE_MAPS * 2),
            nn.ReLU(True),
            # out size. (cnst.GAN_GEN_FEATURE_MAPS*2) x 14 x 14
            # ---------
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(2 * cnst.GAN_GEN_FEATURE_MAPS, cnst.CHANNELS_NUM,
                      kernel_size=3, stride=1, padding=0),
            nn.Tanh()
            # out size (cnst.CHANNELS_NUM) x 28 x 28
            # ---------
        )

    def preprocess(self, noise, labels):
        n_labels = self.label_embedding(labels)
        n_labels = n_labels.reshape(n_labels.size(0), n_labels.size(1), 1, 1)
        n_noise = torch.cat((n_labels, noise), 1)
        return n_noise

    def forward(self, noise, labels):
        n_noise = self.preprocess(noise, labels)
        img = self.model(n_noise)
        return img
