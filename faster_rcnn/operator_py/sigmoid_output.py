"""
Compute the instance segmentation output using the class-specific masks.
"""

import mxnet as mx

DEBUG = False

class SigmoidOutputOperator(mx.operator.CustomOp):
    def __init__(self, num_classes, roi_per_img):
        super(SigmoidOutputOperator, self).__init__()
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
        prob, label = in_data
        grad = self.factor*(prob - label)/float(self._roi_per_img) # only fg rois contribute to grad

        self.assign(in_grad[0], req[0], mx.nd.array(grad))


@mx.operator.register("SigmoidOutput")
class SigmoidOutputProp(mx.operator.CustomOpProp):
    def __init__(self, num_classes, roi_per_img):
        super(SigmoidOutputProp, self).__init__(need_top_grad=False)
        self._num_classes = int(num_classes)
        self._roi_per_img = int(roi_per_img)

    def list_arguments(self):
        return ['prob', 'label']

    def list_outputs(self):
        return ['output_prob']

    def infer_shape(self, in_shape):
        prob_shape = in_shape[0]
        label_shape = in_shape[1]

        assert prob_shape[0] == label_shape[0], \
            'prob_shape[0] != label_shape[0], {} vs {}'.format(prob_shape[0], label_shape[0])


        output_mask_shape = prob_shape
        return in_shape, [output_mask_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return SigmoidOutputOperator(self._num_classes, self._roi_per_img)
