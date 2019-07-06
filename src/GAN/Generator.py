from fastai.imports import torch
from torch import nn
import numpy as np
from sklearn.preprocessing import OneHotEncoder


class Generator(nn.Module):
    def __init__(self, n_classes, latent_dim, img_shape):
        super(Generator, self).__init__()
        self.label_emb = OneHotEncoder(handle_unknown='ignore')


        def block(in_chann, out_chann):
            layers = [nn.Linear(in_chann, out_chann),
                      nn.BatchNorm1d(out_chann, 0.8),
                      nn.LeakyReLU(0.2, inplace=True)]
            return layers

        self.model = nn.Sequential(
            *block(latent_dim + n_classes, 128),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def train_one_hot(self, labels):
        self.label_emb = self.label_emb.fit(labels.reshape(-1, 1))

    def forward(self, noise, labels):
        # Concatenate label embedding and image to produce input
        transformed = self.label_emb.transform(labels.reshape(-1, 1)).toarray()
        embedded_tensor = torch.cuda.FloatTensor(transformed)

        gen_input = torch.cat((embedded_tensor, noise), 1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *self.img_shape)
        return img
