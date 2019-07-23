import datetime
import os

from src.classification.FFN import FFN
from src.classification.CNN import CNN
import src.resources.utilities as utils
import matplotlib.pyplot as plt
import src.resources.constants as cnst
import numpy as np
import sklearn.utils

def load_mixed_data():
    (x_train, y_train), (x_test, y_test) = utils.load_data(cnst.IMGS_AMOUNT + 3000, 0.3)
    # We only want to load train images
    (xg_train, yg_train), (xg_test, yg_test) = utils.load_GAN_data(10000, 0)
    x_train = np.concatenate((x_train, xg_train))
    y_train = np.concatenate((y_train, yg_train))
   # x_test = np.concatenate((x_test, xg_test))
    #y_test = np.concatenate((y_test, yg_test))
    # Shuffling data
    sklearn.utils.shuffle(x_train, y_train)
    sklearn.utils.shuffle(x_test, y_test)
    return (x_train, y_train), (x_test, y_test)


def load_mixed_data_flat():
    (x_train, y_train), (x_test, y_test) = utils.load_data_flat(cnst.IMGS_AMOUNT, cnst.VAL_SPLIT)
    (xg_train, yg_train), (xg_test, yg_test) = utils.load_GAN_data_flat(cnst.IMGS_AMOUNT, 0)
    x_train = np.concatenate((x_train, xg_train))
    y_train = np.concatenate((y_train, yg_train))
   # x_test = np.concatenate((x_test, xg_test))
    #y_test = np.concatenate((y_test, yg_test))
    # Shuffling data
    sklearn.utils.shuffle(x_train, y_train)
    sklearn.utils.shuffle(x_test, y_test)
    return (x_train, y_train), (x_test, y_test)


def load_gan_train_mnist_val_data():
    (_, _), (x_test, y_test) = utils.load_data(cnst.IMGS_AMOUNT, cnst.VAL_SPLIT)
    (xg_train, yg_train), (_, _) = utils.load_GAN_data(cnst.IMGS_AMOUNT, cnst.VAL_SPLIT)
    return (xg_train, yg_train), (x_test, y_test)


def load_gan_train_mnist_val_data_flat():
    (_, _), (x_test, y_test) = utils.load_data_flat(cnst.IMGS_AMOUNT, cnst.VAL_SPLIT)
    (xg_train, yg_train), (_, _) = utils.load_GAN_data_flat(cnst.IMGS_AMOUNT, cnst.VAL_SPLIT)
    return (xg_train, yg_train), (x_test, y_test)


# Starts the training and evaluation of convolutional neural classification on MNIST dataset
def run_cnn():
    cnn = CNN()
    #(x_train, y_train), (x_test, y_test) = utils.load_data(cnst.IMGS_AMOUNT, cnst.VAL_SPLIT)
    (x_train, y_train), (x_test, y_test) = utils.load_data(18000, 0.15)
    history = cnn.train(x_train, y_train, x_test, y_test)
    return history

# Starts the training and evaluation of convolutional neural classification on MNIST dataset mixed with GAN images
def run_cnn_with_gan_data():
    cnn = CNN()
    (x_train, y_train), (x_test, y_test) = load_mixed_data()
    history = cnn.train(x_train, y_train, x_test, y_test)
    return history

# Starts the training on GAN dataset and evaluate on MNIST
def run_cnn_only_on_gan_data():
    cnn = CNN()
    (x_train, y_train), (x_test, y_test) = load_gan_train_mnist_val_data()
    history = cnn.train(x_train, y_train, x_test, y_test)
    return history




def visualize_history(history, eval_sc,  name, res_dir):

    date = datetime.datetime.now().strftime("%m%d%H%M")
    if not os.path.exists(os.path.join(res_dir, date, name)):
        os.makedirs(os.path.join(res_dir, date, name))
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title(name+' model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(os.path.join(res_dir, date,  name, 'netacc.png'))
    plt.show()


    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(name+' model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(os.path.join(res_dir, date, name, 'netloss.png'))
    plt.show()

    with open(os.path.join(res_dir, date, name, 'eval.txt'), "w+") as f:
        for item in eval_sc:
            f.write("%s\n" % item)

if __name__=="__main__":

    # CNNs
    hist, eval_sc = run_cnn()
    visualize_history(hist, eval_sc, "cnn", os.path.join(cnst.RES_DIR, "ONLY_MNIST"))

   # hist, eval_sc = run_cnn_with_gan_data()
   # visualize_history(hist, eval_sc, "cnn_mixed", os.path.join(cnst.RES_DIR, "MIXED"))

  #  hist, eval_sc = run_cnn_only_on_gan_data()
  #  visualize_history(hist, eval_sc, "cnn_gan", os.path.join(cnst.RES_DIR, "ONLY_GAN"))
