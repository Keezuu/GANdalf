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
from abc import ABC, abstractmethod
import src.resources.constants as cnst


class Network(ABC):

    @abstractmethod
    def __init__(self):
        pass

    def get_model(self):
        return self.model

    def train(self, x_train, y_train, x_test, y_test):

        if self.model is None:
            #   nw pozniej sie to lepiej zrobi
            return

        # Train the model. Epochs and batch size in constants.py
        history = self.model.fit(x=x_train, y=y_train, validation_data=(x_test, y_test),
                                 epochs=cnst.EPOCHS, batch_size=cnst.BATCH_SIZE)

        # Get current date and time for easier identyfication of models
        date = datetime.datetime.now().strftime("%m%d%H%M")
        # Save the weights in .h5 file

        if not os.path.exists(os.path.join(cnst.RES_DIR, cnst.MODELS_DIR)):
            os.makedirs(os.path.join(cnst.RES_DIR, cnst.MODELS_DIR))

        self.model.save_weights(os.path.join(cnst.RES_DIR, cnst.MODELS_DIR, date + "def" + ".h5"))

        # Evaluates how good does our network work on test images
        eval = self.model.evaluate(x_test, y_test, verbose=2)
        # Prints the result
        print("RESULTS ON THE TEST DATA: \n")
        print(eval)
        return history, eval

    def predict(self, data):
        # Predict the labels of test data
        predictions = self.model.predict(x=data, verbose=1)
        return predictions
