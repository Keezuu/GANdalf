import os

from src.network.FFN import FFN
from src.network.CNN import CNN
import src.resources.utilities as utils
import matplotlib.pyplot as plt
import src.resources.constants as cnst
import numpy as np

def load_mixed_data():
    (x_train, y_train), (x_test, y_test) = utils.load_data(cnst.IMGS_AMOUNT, cnst.VAL_SPLIT)
    (xg_train, yg_train), (xg_test, yg_test) = utils.load_GAN_data(cnst.IMGS_AMOUNT, cnst.VAL_SPLIT)
    x_train = np.concatenate((x_train, xg_train))
    y_train = np.concatenate((y_train, yg_train))
    x_test = np.concatenate((x_test, xg_test))
    y_test = np.concatenate((y_test, yg_test))
    return (x_train, y_train), (x_test, y_test)


def load_mixed_data_flat():
    (x_train, y_train), (x_test, y_test) = utils.load_data_flat(cnst.IMGS_AMOUNT, cnst.VAL_SPLIT)
    (xg_train, yg_train), (xg_test, yg_test) = utils.load_GAN_data_flat(cnst.IMGS_AMOUNT, cnst.VAL_SPLIT)
    x_train = np.concatenate((x_train, xg_train))
    y_train = np.concatenate((y_train, yg_train))
    x_test = np.concatenate((x_test, xg_test))
    y_test = np.concatenate((y_test, yg_test))
    return (x_train, y_train), (x_test, y_test)


def load_gan_train_mnist_val_data():
    (_, _), (x_test, y_test) = utils.load_data(cnst.IMGS_AMOUNT, cnst.VAL_SPLIT)
    (xg_train, yg_train), (_, _) = utils.load_GAN_data(cnst.IMGS_AMOUNT, cnst.VAL_SPLIT)
    return (xg_train, yg_train), (x_test, y_test)


def load_gan_train_mnist_val_data_flat():
    (_, _), (x_test, y_test) = utils.load_data_flat(cnst.IMGS_AMOUNT, cnst.VAL_SPLIT)
    (xg_train, yg_train), (_, _) = utils.load_GAN_data_flat(cnst.IMGS_AMOUNT, cnst.VAL_SPLIT)
    return (xg_train, yg_train), (x_test, y_test)


# Starts the training and evaluation of feed forward neural network on MNIST dataset
def run_ffn():
    ffn = FFN()
    (x_train, y_train), (x_test, y_test) = utils.load_data_flat(cnst.IMGS_AMOUNT, cnst.VAL_SPLIT)
    history = ffn.train(x_train, y_train, x_test, y_test)
    return history


# Starts the training and evaluation of convolutional neural network on MNIST dataset
def run_cnn():
    cnn = CNN()
    (x_train, y_train), (x_test, y_test) = utils.load_data(cnst.IMGS_AMOUNT, cnst.VAL_SPLIT)
    history = cnn.train(x_train, y_train, x_test, y_test)
    return history




# Starts the training and evaluation of feed forward neural network on MNIST dataset mixed with GAN images
def run_ffn_with_GAN_data():
    ffn = FFN()
    (x_train, y_train), (x_test, y_test) = load_mixed_data_flat()
    history = ffn.train(x_train, y_train, x_test, y_test)
    return history


# Starts the training and evaluation of convolutional neural network on MNIST dataset mixed with GAN images
def run_cnn_with_GAN_data():
    cnn = CNN()
    (x_train, y_train), (x_test, y_test) = load_mixed_data()
    history = cnn.train(x_train, y_train, x_test, y_test)
    return history


# Starts the training on GAN dataset and evaluate on MNIST
def run_ffn_only_on_GAN_data():
    ffn = FFN()
    (x_train, y_train), (x_test, y_test) = load_gan_train_mnist_val_data_flat()
    history = ffn.train(x_train, y_train, x_test, y_test)
    return history


# Starts the training on GAN dataset and evaluate on MNIST
def run_cnn_only_on_GAN_data():
    cnn = CNN()
    (x_train, y_train), (x_test, y_test) = load_gan_train_mnist_val_data()
    history = cnn.train(x_train, y_train, x_test, y_test)
    return history




def visualize_history(history, eval_sc,  name, res_dir):
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title(name+' model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(os.path.join(res_dir, name+'netacc.png'))
    plt.show()


    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(name+' model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(os.path.join(res_dir, name+'netloss.png'))
    plt.show()

    with open(os.path.join(res_dir, name + 'eval.txt'), "w+") as f:
        for item in eval_sc:
            f.write("%s\n" % item)

if __name__=="__main__":

    hist, eval_sc = run_ffn()
    visualize_history(hist, eval_sc, "ffn", "RESULTS/NOGANRES")

    hist, eval_sc = run_ffn_with_GAN_data()
    visualize_history(hist, eval_sc, "ffn_mixed", "RESULTS/MNISTANDGANRES")

    hist, eval_sc = run_ffn_only_on_GAN_data()
    visualize_history(hist, eval_sc, "ffn_gan", "RESULTS/GANRES")

    # CNNs
    hist, eval_sc = run_cnn()
    visualize_history(hist, eval_sc, "cnn", "RESULTS/NOGANRES")

    hist, eval_sc = run_cnn_with_GAN_data()
    visualize_history(hist, eval_sc, "cnn_mixed", "RESULTS/MNISTANDGANRES")

    hist, eval_sc = run_cnn_only_on_GAN_data()
    visualize_history(hist, eval_sc, "cnn_gan", "RESULTS/GANRES")
