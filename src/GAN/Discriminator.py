from fastai.imports import torch
from sklearn.preprocessing import OneHotEncoder
from torch import nn
import numpy as np

# A discriminator class for predicting if image is fake or real


class Discriminator(nn.Module):
    def __init__(self, n_classes, img_shape):
        super(Discriminator, self).__init__()

        self.label_emb = OneHotEncoder(handle_unknown='ignore')

        self.model = nn.Sequential(
            nn.Linear(n_classes + int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
        )

    def train_one_hot(self, labels):
        self.label_emb = self.label_emb.fit(labels.reshape(1, -1))

    def forward(self, img, labels):
        # Concatenate label embedding and image to produce input
        transformed = self.label_emb.transform(labels.reshape(-1, 1)).toarray()
        embedded_tensor = torch.cuda.FloatTensor(transformed)
        transformed_img = img.view(img.size(0), -1)
        d_in = torch.cat((transformed_img,  embedded_tensor), -1)
        validity = self.model(d_in)
        return validity
