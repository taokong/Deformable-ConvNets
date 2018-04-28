import numpy as np
import matplotlib.pyplot as plt
import cPickle

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def iou_weight(x):
    mean = 0.42
    sigma = 0.2
    b = 0.3
    y = 1 - b *np.exp(-np.abs(x-mean) / sigma)

    return y

def ce_loss(x):
    return  -np.log(x)

def get_iou_probs():

    file_path = './../my_models/original_faster_rcnn_resnet101/resnet_v1_101_voc0712_rcnn_end2end/2007_test/cal_probs_iou'
    with open(file_path, 'rb') as f:
        data = cPickle.load(f)

    probs = -data[:, 0]
    ious = data[:, 1]

    length = 10
    x = np.linspace(0, 1, length+1)

    prob_plot = x[:-1]+1.0/length/2
    iou_plot = np.zeros(length)
    iou_plot_std = np.zeros(length)
    iou_num = np.zeros(length)



    for i in range(length):
        inds = np.where((probs > x[i]) & (probs <= x[i+1]))[0]
        if len(inds) > 0:
            iou_this = ious[inds]
            iou_plot[i] = np.mean(iou_this)
            iou_plot_std[i] = np.std(iou_this)
            iou_num[i] = len(inds) / float(len(probs))

    # plt.figure()
    # type_mean, = plt.plot(prob_plot, iou_plot, 'g', label = 'real mean')
    # type_target, = plt.plot(prob_plot, prob_plot, 'r', label = 'target mean')
    # type_std, = plt.plot(prob_plot, iou_plot_std, 'b', label = 'real std')
    # first_legend = plt.legend(handles=[type_mean, type_target, type_std], loc=0)
    # ax = plt.gca().add_artist(first_legend)
    # plt.xlabel('probability of ground truth class')
    # plt.ylabel('overlaps with ground truth class')
    # plt.grid()
    #
    # plt.figure()
    # prob_plot_ce = ce_loss(prob_plot)
    # iou_plot_ce = ce_loss(iou_plot)
    # type_ce, = plt.plot(prob_plot, prob_plot_ce, 'r', label = 'ce loss')
    # type_iou_ce, = plt.plot(prob_plot, iou_plot_ce, 'g', label = 'iou ce loss')
    # first_legend = plt.legend(handles=[type_ce, type_iou_ce], loc=0)
    # ax = plt.gca().add_artist(first_legend)
    # plt.xlabel('probability of ground truth class')
    # plt.ylabel('ce loss')
    # plt.grid()

    plt.figure()
    iou_plot_ce = ce_loss(iou_plot)
    plot_plot_ce = ce_loss(prob_plot)
    type_ce, = plt.plot(iou_plot, iou_plot_ce, 'r', label = 'target ce loss')
    type_iou_ce, = plt.plot(iou_plot, plot_plot_ce, 'g', label = 'ce loss')
    type_ce_trans, = plt.plot(iou_plot, ce_loss(iou_plot) * iou_weight(iou_plot), 'b', label = 'trans')

    # type_iou_ce, = plt.plot(iou_plot, plot_plot_ce, 'g', label = 'prob ce loss')
    first_legend = plt.legend(handles=[type_ce, type_iou_ce,type_ce_trans], loc=0)
    ax = plt.gca().add_artist(first_legend)
    plt.xlabel('overlaps with ground truth class')
    plt.ylabel('ce loss')
    plt.grid()

    plt.show()



if __name__ == '__main__':

    get_iou_probs()

    # x = np.linspace(0, 1, 100)
    # # y = sigmoid(x)
    # y_w = ce_loss(x)
    #
    # # plt.plot(x, y, 'r')
    # plt.plot(x, y_w, 'g')
    # plt.grid()
    # plt.show()



