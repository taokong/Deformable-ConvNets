import numpy as np
import matplotlib.pyplot as plt
import math

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def iou_weight(x):
    mean = 0
    sigma = 0.12
    b = 0.5
    y = 1- b *np.exp(-np.abs(x-mean) / sigma)

    return y

def test_rankout():

    overlaps = np.array([[1. ,   0.752, 0.723, 0.528, 0.4, 0.3]])
    scores = np.array([0.828, 0.803, 0.526, 0.828, 0.3, 0.2])
    rank_weights = np.zeros(6)
    for cls_i in range(6):
        lamda_ij = 0
        for cls_j in range(6):
            if cls_i == cls_j:
                continue

            if cls_j  < cls_i:
                lamda_ij -= -1 / (1 + math.exp( (scores[cls_j] - scores[cls_i])))
            else:
                lamda_ij += -1 / (1 + math.exp( (scores[cls_i] - scores[cls_j])))
            print lamda_ij

        rank_weights[cls_i] = lamda_ij

    print rank_weights



if __name__ == '__main__':

    # x = np.linspace(0, 10, 100)
    # y = sigmoid(x)
    # y_w = iou_weight(x)
    #
    # plt.plot(x, y, 'r')
    # plt.plot(x, y_w, 'g')
    # plt.grid()
    # plt.show()

    # test_rankout()
    #

    gama = 1
    x_b = np.linspace(0, 10, 100)
    x_a = np.linspace(-10, 0, 100)
    y_a = 1 * (1 / (1 + np.exp(x_b*gama)))
    y_b = -1 * (1 / (1 + np.exp(x_b*gama)))

    plt.plot(x_b, y_a, 'r')
    plt.plot(x_b, y_b, 'g')
    plt.grid()
    plt.show()




