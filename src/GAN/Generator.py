from fastai.imports import torch
from torch import nn
import numpy as np


class Generator(nn.Module):
    def __init__(self, n_classes, latent_dim, img_shape):
        super(Generator, self).__init__()
        self.img_shape = img_shape
        self.label_emb = nn.Embedding(n_classes, n_classes)

        def block(in_chann, out_chann, normalize=True):
            layers = [nn.Linear(in_chann, out_chann)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_chann, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim + n_classes, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # Concatenate label embedding and image to produce input
        embedded = self.label_emb(labels)
        gen_input = torch.cat((embedded, noise), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *self.img_shape)
        return img
