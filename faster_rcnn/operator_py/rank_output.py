
import mxnet as mx
import math
import numpy as np

DEBUG = False

class RankOutputOperator(mx.operator.CustomOp):
    def __init__(self, num_classes, roi_per_img):
        super(RankOutputOperator, self).__init__()
        self.factor = 1.0
        self._num_classes = num_classes
        self._roi_per_img = roi_per_img

        self._min_overlap = 0.4


    def forward(self, is_train, req, in_data, out_data, aux):
        if DEBUG:
            print is_train
            print len(in_data)
        #
        # probs, overlaps = in_data
        # probs = probs.asnumpy()
        # overlaps = overlaps.asnumpy()
        #
        # # sort the overlaps in decent way
        # # sort_inds = np.argsort(-overlaps, axis = 0)
        # losses = mx.nd.zeros(probs.shape)
        # for cls_i in range(self._num_classes):
        #     for box_j in range(self._roi_per_img):
        #
        #         loss_ij = 0
        #         # get curent value
        #         overlap_j = overlaps[box_j, cls_i]
        #         score_j = probs[box_j, cls_i]
        #         if overlap_j < self._min_overlap:
        #             # skip this box since
        #             continue
        #
        #         n_samples = 0
        #         for box_i in range(self._roi_per_img):
        #             overlap_i = overlaps[box_i, cls_i]
        #             score_i = probs[box_i, cls_i]
        #             if overlap_i < self._min_overlap:
        #                 continue
        #
        #             n_samples += 1
        #             # compute lamda_i
        #             if overlap_j > overlap_i:
        #                 # S_ij = 1
        #                 loss_ij += math.log(1 + math.exp(-(score_j - score_i)))
        #             elif overlap_j < overlap_i:
        #                 # S_ij = -1
        #                 loss_ij += math.log(1 + math.exp(-(score_i - score_j)))
        #             else:
        #                 loss_ij += 0.5 + math.log(1 + math.exp(-(score_j - score_i)))
        #
        #         if n_samples > 0:
        #             losses[box_j, cls_i] = loss_ij / n_samples

        self.assign(out_data[0], req[0], in_data[0])

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):

        assert len(in_data) == 2
        assert len(out_data) == 1
        probs, overlaps = in_data

        probs = probs.asnumpy()
        overlaps = overlaps.asnumpy()

        # sort the overlaps in decent way
        # sort_inds = np.argsort(-overlaps, axis = 0)
        lamdas = mx.nd.zeros(probs.shape)
        for cls_i in range(self._num_classes):
            for box_j in range(self._roi_per_img):
                lamda_ij = 0
                # get curent value
                overlap_j = overlaps[box_j, cls_i]
                score_j = probs[box_j, cls_i]

                if overlap_j < self._min_overlap:
                    # skip this box since
                    continue

                n_samples = 0
                scores_i = probs[:, cls_i]
                logs = -1 / (1 + np.exp(score_j - scores_i))
                for box_i in range(self._roi_per_img):
                    overlap_i = overlaps[box_i, cls_i]

                    if overlap_i < self._min_overlap:
                        continue

                    n_samples += 1
                    # compute lamda_i
                    if overlap_j > overlap_i:
                        # S_ij = 1
                        lamda_ij += logs[box_i]
                    elif overlap_j < overlap_i:
                        # S_ij = -1
                        lamda_ij += 1 + logs[box_i]
                    else:
                        lamda_ij += 0.5 + logs[box_i]

                if n_samples > 0:
                    lamdas[box_j, cls_i] = lamda_ij / n_samples

        # find
        if DEBUG:
            sort_inds = np.argsort(-overlaps, axis = 0)
            max_overlap_cls_ind = np.argmax(overlaps[sort_inds[0, :], range(self._num_classes)])
            overlaps_this = overlaps[:, max_overlap_cls_ind]
            probs_this = probs[:, max_overlap_cls_ind]
            lamda_this = lamdas.asnumpy()[:, max_overlap_cls_ind]
            inds_decent = np.argsort(-overlaps_this)
            print max_overlap_cls_ind
            print overlaps_this[inds_decent[10:26]]
            print probs_this[inds_decent[10:26]]
            print lamda_this[inds_decent[10:26]]



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
