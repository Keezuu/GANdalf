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
from src.network.Network import Network
import src.resources.constants as cnst


class FFN(Network):

    # Simple feed forward neural network

    def __init__(self):
        self.model = Sequential()

        # Create network with 4 hidden layers, input size is a flat array of pixels with
        # length equal to img width times img height
        self.model.add(Dense(1024, activation='relu', input_shape=(cnst.IMG_SIZE*cnst.IMG_SIZE,)))
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(10, activation='softmax'))

        self.model.compile(loss="categorical_crossentropy",
                           optimizer=Adam(),
                           metrics=["accuracy"])


