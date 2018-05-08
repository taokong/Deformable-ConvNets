
import mxnet as mx
import math
DEBUG = False

class RankOutputOperator(mx.operator.CustomOp):
    def __init__(self, num_classes, roi_per_img):
        super(RankOutputOperator, self).__init__()
        self.factor = 1.0
        self._num_classes = num_classes
        self._roi_per_img = roi_per_img

    def forward(self, is_train, req, in_data, out_data, aux):
        if DEBUG:
            print is_train
            print len(in_data)

        self.assign(out_data[0], req[0], in_data[0])

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):

        assert len(in_data) == 2
        assert len(out_data) == 1
        probs, overlaps = in_data

        # sort the overlaps in decent way
        sort_inds = mx.nd.argsort(-overlaps, axis = 0)
        lamdas = mx.nd.zeros(probs.shape)
        for cls_i in range(self._num_classes):
            for box_j in range(self._roi_per_img):
                lamda_ij = 0
                # get curent value
                score_j = probs[sort_inds[box_j, cls_i], cls_i]
                for box_i in range(self._roi_per_img):
                    score_i = probs[sort_inds[box_i, cls_i], cls_i]
                    # compute lamda_i
                    if sort_inds[box_i, cls_i] > sort_inds[box_j, cls_i]:
                        # S_ij = 1
                        lamda_ij += -1 / (1 + math.exp(score_i - score_j))
                    elif sort_inds[box_i, cls_i] < sort_inds[box_j, cls_i]:
                        # S_ij = -1
                        lamda_ij += 1 - 1 / (1 + math.exp(score_i - score_j))

                lamdas[box_j, cls_i] = lamda_ij / self._roi_per_img

        self.assign(in_grad[0], req[0], mx.nd.array(lamdas))


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
