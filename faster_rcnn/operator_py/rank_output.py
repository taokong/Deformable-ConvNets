
import mxnet as mx
import math
import numpy as np
from scipy.spatial import distance_matrix

DEBUG = False

class RankOutputOperator(mx.operator.CustomOp):
    def __init__(self, num_classes, roi_per_img):
        super(RankOutputOperator, self).__init__()
        self.factor = 0.5
        self._num_classes = num_classes
        self._roi_per_img = roi_per_img

        self._min_overlap = 0.3
        self._gama = 1

    def forward(self, is_train, req, in_data, out_data, aux):
        if DEBUG:
            print is_train
            print len(in_data)

        probs, overlaps = in_data
        # probs = probs.asnumpy()
        # overlaps = overlaps.asnumpy()

        # sort the overlaps in decent way
        # sort_inds = np.argsort(-overlaps, axis = 0)
        losses = mx.nd.zeros(probs.shape)
        # for cls_i in range(self._num_classes):
        #     if cls_i < 1:
        #         continue
        #     for box_j in range(self._roi_per_img):
        #
        #         loss_ij = 0
        #         # get curent value
        #         overlap_j = overlaps[sort_inds[box_j, cls_i], cls_i]
        #         score_j = probs[sort_inds[box_j, cls_i], cls_i]
        #         if overlap_j < self._min_overlap:
        #             # skip this box since
        #             continue
        #
        #         n_samples = 0.0
        #         for box_i in range(self._roi_per_img):
        #             overlap_i = overlaps[sort_inds[box_i, cls_i], cls_i]
        #             score_i = probs[sort_inds[box_i, cls_i], cls_i]
        #             if overlap_i < self._min_overlap:
        #                 continue
        #
        #             n_samples += 1
        #             # compute lamda_i
        #             S_ji = 0
        #             if box_i < box_j:
        #                 S_ji = -1
        #             elif box_i > box_j:
        #                 S_ji = 1
        #             else:
        #                 S_ji = 0
        #
        #             loss_ij += 0.5 * (1-S_ji) * self._gama * (score_j - score_i) + math.log(1 + math.exp(-self._gama * (score_j - score_i)))
        #
        #         if n_samples > 0:
        #             losses[box_j, cls_i] = self.factor * loss_ij / n_samples

        self.assign(out_data[0], req[0], losses)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):

        assert len(in_data) == 2
        assert len(out_data) == 1
        probs, overlaps = in_data

        probs = probs.asnumpy()
        overlaps = overlaps.asnumpy()

        # sort the overlaps in decent way
        sort_inds = np.argsort(-overlaps, axis = 0)

        # print sort_inds
        lamdas = mx.nd.zeros(probs.shape)
        for cls_i in range(self._num_classes):
            if cls_i < 1:
                continue

            scores = probs[sort_inds[:, cls_i], cls_i].reshape((self._roi_per_img, 1))
            diff = distance_matrix(scores, scores, p = 1)
            diff = -1 / (1 + np.exp(self._gama * diff))
            for box_j in range(self._roi_per_img):
                lamda_ij = 0
                # get curent value
                overlap_j = overlaps[sort_inds[box_j, cls_i], cls_i]
                score_j = probs[sort_inds[box_j, cls_i], cls_i]

                if overlap_j < self._min_overlap:
                    # skip this box since
                    continue

                n_samples = 0.0
                # scores_i = probs[sort_inds[:, cls_i], cls_i]
                # logs = -1 / (1 + np.exp(self._gama * (np.abs(score_j - scores_i))))
                for box_i in range(self._roi_per_img):
                    overlap_i = overlaps[sort_inds[box_i, cls_i], cls_i]
                    if overlap_i < self._min_overlap:
                        continue

                    if box_j == box_i:
                        continue

                    n_samples += 1
                    # compute lamda_i
                    if box_i < box_j:
                        lamda_ij -= diff[box_j, box_i]
                    else:
                        lamda_ij += diff[box_j, box_i]

                    # print logs[box_i], diff[box_j, box_i]

                if n_samples > 0:
                    grad = self.factor * lamda_ij / n_samples
                    if (score_j < 0 and overlap_j > 0.5):
                        grad = -abs(grad)
                    elif (score_j > 0 and overlap_j < 0.5):
                        grad = abs(grad)

                    lamdas[sort_inds[box_j, cls_i], cls_i] = grad

        # find
        if DEBUG:
            max_overlap_cls_ind = np.argmax(overlaps[sort_inds[0, :], range(self._num_classes)])
            overlaps_this = overlaps[:, max_overlap_cls_ind]
            consider_inds = np.where(overlaps_this.flatten() > self._min_overlap)[0]
            overlaps_this = overlaps_this[consider_inds]
            probs_this = probs[consider_inds, max_overlap_cls_ind]
            lamda_this = lamdas.asnumpy()[consider_inds, max_overlap_cls_ind]

            decent_inds = np.argsort(-overlaps_this)
            overlaps_this = overlaps_this[decent_inds]
            probs_this = probs_this[decent_inds]
            lamda_this = lamda_this[decent_inds]
            print np.round(overlaps_this, 3)
            print np.round(probs_this, 3)
            print np.round(lamda_this, 3)



        # print probs, overlaps, lamdas.asnumpy()

        self.assign(in_grad[0], req[0], lamdas)


@mx.operator.register("RankOutput")
class RankOutputProp(mx.operator.CustomOpProp):
    def __init__(self, num_classes, roi_per_img):
        super(RankOutputProp, self).__init__(need_top_grad=False)
        self._num_classes = int(num_classes)
        self._roi_per_img = int(roi_per_img)

    def list_arguments(self):
        return ['prob', 'overlap']

    def list_outputs(self):
        return ['rank_output']

    def infer_shape(self, in_shape):
        prob_shape = in_shape[0]
        overlap_shape = in_shape[1]

        assert prob_shape[0] == overlap_shape[0], \
            'prob_shape[0] != overlap_shape_shape[0], {} vs {}'.format(prob_shape[0], overlap_shape[0])

        return in_shape, [overlap_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return RankOutputOperator(self._num_classes, self._roi_per_img)
