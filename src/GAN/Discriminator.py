from fastai.imports import torch
from torch import nn
import numpy as np
from src.resources import constants as cnst
# A discriminator class for predicting if image is fake or real


class Discriminator(nn.Module):
    def __init__(self, n_classes, img_shape):
        super(Discriminator, self).__init__()

        self.conv_layer = nn.Sequential(
            # input is (cnst.CHANNELS_NUM) x 28 x 28
            nn.Conv2d(cnst.CHANNELS_NUM, cnst.GAN_DIS_FEATURE_MAPS,
                      kernel_size=3, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # out is (cnst.GAN_DIS_FEATURE_MAPS) x 14 x 14
            # ---------
            nn.Conv2d(cnst.GAN_DIS_FEATURE_MAPS, cnst.GAN_DIS_FEATURE_MAPS * 2,
                      3, 2, 1, bias=False),
            nn.BatchNorm2d(cnst.GAN_DIS_FEATURE_MAPS * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # out is (cnst.GAN_DIS_FEATURE_MAPS*2) x 7 x 7
            # ---------
            nn.Conv2d(cnst.GAN_DIS_FEATURE_MAPS * 2, cnst.GAN_DIS_FEATURE_MAPS * 4,
                      3, 2, 1, bias=False),
            nn.BatchNorm2d(cnst.GAN_DIS_FEATURE_MAPS * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # out is (cnst.GAN_DIS_FEATURE_MAPS*4) x 4 x 4
            # ---------
        )
        self.conv_layer_out = nn.Sequential(
            nn.Conv2d(cnst.GAN_DIS_FEATURE_MAPS * 4, 1,
                      4, 1, 0, bias=False),
            # out is 1x1x1
            nn.Sigmoid()
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(cnst.GAN_DIS_FEATURE_MAPS*4 * 4 * 4, 1),
            nn.Sigmoid()
        )



    def forward(self, img):
        # # Concatenate label embedding and image to produce input
        # transformed = self.label_emb.transform(labels.reshape(-1, 1)).toarray()
        # embedded_tensor = torch.cuda.FloatTensor(transformed)
        # transformed_img = img.view(img.size(0), -1)
        # d_in = torch.cat((transformed_img,  embedded_tensor), -1)
        res = self.conv_layer(img)
        #final_res = self.conv_layer_out(res)
        flat_res = res.reshape(res.size(0), -1)
        final_res = self.fc_layer(flat_res)
        return final_res
