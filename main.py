import sys

from PyQt5.QtWidgets import QApplication

from src.network.FFN import FFN
from src.network.CNN import CNN
import src.resources.utilities as utils
from src.GUI.MainWindow import MainWindow
import matplotlib.pyplot as plt


def main():
    app = QApplication(sys.argv)
    ex = MainWindow()
    sys.exit(app.exec_())

    pass


if __name__ == "__main__":
    main()
