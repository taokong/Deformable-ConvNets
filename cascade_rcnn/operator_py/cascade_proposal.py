# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2016 by Contributors
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Modified by Yuwen Xiong
# --------------------------------------------------------

"""
Proposal Operator.
"""

import mxnet as mx
import numpy as np
import cPickle
from bbox.bbox_transform import bbox_pred, clip_boxes

DEBUG = False

class CascadeProposalOperator(mx.operator.CustomOp):
    def __init__(self, batch_images, cfg, stage):
        super(CascadeProposalOperator, self).__init__()
        self._batch_images = batch_images
        self._cfg = cfg
        self._stage = stage

        if DEBUG:
            self._count = 0
            self._fg_num = 0
            self._bg_num = 0

    def forward(self, is_train, req, in_data, out_data, aux):
        assert self._batch_images == 1, \
            'batchimages {} must devide batch_rois {}'.format(self._batch_images, self._batch_rois)
        all_rois = in_data[0].asnumpy()
        all_rois_off = in_data[1].asnumpy()[0]
        im_info = in_data[2].asnumpy()[0, :]

        # print(np.shape(all_rois), np.shape(all_rois_off), np.shape(gt_boxes))
        # Include ground-truth boxes in the set of candidate rois
        # zeros = np.zeros((gt_boxes.shape[0], 1), dtype=gt_boxes.dtype)
        # all_rois = np.vstack((all_rois, np.hstack((zeros, gt_boxes[:, :-1]))))
        # Sanity check: single batch only
        assert np.all(all_rois[:, 0] == 0), 'Only single item batches are supported'

        if int(self._stage) == 3:
            num_reg_classes = (2 if self._cfg.CLASS_AGNOSTIC else self._cfg.dataset.NUM_CLASSES)
            stds_2nd = np.tile(
                np.array(self._cfg.TRAIN.BBOX_STDS_2nd), (num_reg_classes))
            means_2nd = np.tile(
                np.array(self._cfg.TRAIN.BBOX_MEANS), (num_reg_classes))

            all_rois_off *= stds_2nd
            all_rois_off += means_2nd

        # generate new bbox result
        all_rois_pred = bbox_pred(all_rois[:, 1:5], all_rois_off[:, 4:8])
        all_rois_pred = clip_boxes(all_rois_pred, im_info[:2])
        zeros = np.zeros((all_rois_pred.shape[0], 1), dtype=all_rois_pred.dtype)
        all_rois_pred = np.hstack((zeros, all_rois_pred))


        for ind, val in enumerate([all_rois_pred]):
            self.assign(out_data[ind], req[ind], val)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], 0)
        self.assign(in_grad[1], req[1], 0)


@mx.operator.register('cascade_proposal')
class CascadeProposalProp(mx.operator.CustomOpProp):
    def __init__(self, batch_images, cfg, stage=2):
        super(CascadeProposalProp, self).__init__(need_top_grad=False)
        self._batch_images = int(batch_images)
        self._cfg = cPickle.loads(cfg)
        self._stage = stage

    def list_arguments(self):
        return ['rois', 'bbox_offset', 'im_info']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        rpn_rois_shape = in_shape[0]
        bbox_offset_shape = in_shape[1]
        im_info_shape = in_shape[2]


        output_rois_shape = rpn_rois_shape

        return [rpn_rois_shape, bbox_offset_shape, im_info_shape], \
               [output_rois_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return CascadeProposalOperator(self._batch_images, self._cfg, self._stage)

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []
