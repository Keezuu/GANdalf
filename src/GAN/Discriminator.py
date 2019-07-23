from fastai.imports import torch
from torch import nn
import numpy as np
from src.resources import constants as cnst


# A discriminator class for predicting if image is fake or real


class Discriminator(nn.Module):
    def __init__(self, n_classes, img_shape):
        super(Discriminator, self).__init__()

        self.label_embedding = nn.Embedding(n_classes, 2*img_shape[0] * 2*img_shape[1])

        self.conv_layer = nn.Sequential(
            # input is (cnst.CHANNELS_NUM + labels channel)  x 56 x 56
            nn.Conv2d(cnst.CHANNELS_NUM + 1, cnst.GAN_DIS_FEATURE_MAPS,
                      kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # out is (cnst.GAN_DIS_FEATURE_MAPS) x 28 x 28
            # ---------
            nn.Conv2d(cnst.GAN_DIS_FEATURE_MAPS, cnst.GAN_DIS_FEATURE_MAPS * 2,
                      4, 2, 1, bias=False),
            nn.BatchNorm2d(cnst.GAN_DIS_FEATURE_MAPS * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # out is (cnst.GAN_DIS_FEATURE_MAPS*2) x 14 x 14
            # ---------
            nn.Conv2d(cnst.GAN_DIS_FEATURE_MAPS * 2, cnst.GAN_DIS_FEATURE_MAPS * 4,
                      4, 2, 1, bias=False),
            nn.BatchNorm2d(cnst.GAN_DIS_FEATURE_MAPS * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # out is (cnst.GAN_DIS_FEATURE_MAPS*4) x 7 x 7
            # ---------
            nn.Conv2d(cnst.GAN_DIS_FEATURE_MAPS * 4, cnst.GAN_DIS_FEATURE_MAPS * 8,
                      2, 2, 1, bias=False),
            nn.BatchNorm2d(cnst.GAN_DIS_FEATURE_MAPS * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # out is (cnst.GAN_DIS_FEATURE_MAPS*8) x 4 x 4
            # ---------
        )
        self.conv_layer_out = nn.Sequential(
            nn.Conv2d(cnst.GAN_DIS_FEATURE_MAPS * 8, 1,
                      4, 1, 0, bias=False),
            # out is 1x1x1
            nn.Sigmoid()
        )

    def preprocess(self, img, labels):
        n_labels = self.label_embedding(labels)
        n_img = nn.functional.interpolate(img, scale_factor=2, mode='bilinear')

        n_labels = n_labels.reshape(n_labels.size(0), 1, 2*28, 2*28)
        n_img = torch.cat((n_labels, n_img), 1)

        return n_img

    def forward(self, img, labels):
        n_img = self.preprocess(img, labels)
        res = self.conv_layer(n_img )
        final_res = self.conv_layer_out(res)

        return final_res
