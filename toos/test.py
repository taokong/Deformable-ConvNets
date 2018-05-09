import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def iou_weight(x):
    mean = 0
    sigma = 0.12
    b = 0.5
    y = 1- b *np.exp(-np.abs(x-mean) / sigma)

    return y

if __name__ == '__main__':

    # x = np.linspace(0, 10, 100)
    # y = sigmoid(x)
    # y_w = iou_weight(x)
    #
    # plt.plot(x, y, 'r')
    # plt.plot(x, y_w, 'g')
    # plt.grid()
    # plt.show()
    gama = 6
    x = np.linspace(-1, 1, 100)
    y = gama * (-1 / (1 + np.exp(x*gama)))
    y1 = gama * (1 -1 / (1 + np.exp(x*gama)))

    plt.plot(x, y, 'r')
    plt.plot(x, y1, 'g')
    plt.grid()
    plt.show()




