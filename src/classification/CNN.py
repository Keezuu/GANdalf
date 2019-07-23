from keras.constraints import max_norm, maxnorm
from keras.optimizers import Nadam, Adam, Adadelta
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense, SpatialDropout2D
from keras import backend, optimizers
from keras.regularizers import l2
import datetime
import numpy as np
import os
from src.classification.Network import Network

import src.resources.constants as cnst


class CNN(Network):

    # Simple feed forward neural classification

    def __init__(self):
        self.model = Sequential()

        self.model.add(Conv2D(32, (3, 3), padding='same', activation='relu',
                              input_shape=(cnst.IMG_SIZE, cnst.IMG_SIZE, 1)))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        self.model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        self.model.add(Flatten())

        self.model.add(Dense(1024, activation='relu'))
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dense(10, activation='softmax'))

        self.model.compile(loss="categorical_crossentropy",
                           optimizer=Adam(),
                           metrics=["accuracy"])
