# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2016 by Contributors
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Modified by Yuwen Xiong
# --------------------------------------------------------

"""
Proposal Target Operator selects foreground and background roi and assigns label, bbox_transform to them.
"""

import mxnet as mx
import numpy as np
from distutils.util import strtobool
from easydict import EasyDict as edict
import cPickle
import numpy.random as npr
from bbox.bbox_transform import bbox_overlaps, bbox_transform, nonlinear_pred
from bbox.bbox_regression import expand_bbox_regression_targets

DEBUG = False


def sample_rois(rois, fg_rois_per_image, rois_per_image, num_classes, cfg,
                labels=None, overlaps=None, bbox_targets=None, gt_boxes=None, stage=2):
    """
    generate random sample of ROIs comprising foreground and background examples
    :param rois: all_rois [n, 4]; e2e: [n, 5] with batch_index
    :param fg_rois_per_image: foreground roi number
    :param rois_per_image: total roi number
    :param num_classes: number of classes
    :param labels: maybe precomputed
    :param overlaps: maybe precomputed (max_overlaps)
    :param bbox_targets: maybe precomputed
    :param gt_boxes: optional for e2e [n, 5] (x1, y1, x2, y2, cls)
    :return: (labels, rois, bbox_targets, bbox_weights)
    """
    if labels is None:
        overlaps = bbox_overlaps(rois[:, 1:].astype(np.float), gt_boxes[:, :4].astype(np.float))
        gt_assignment = overlaps.argmax(axis=1)
        overlaps = overlaps.max(axis=1)
        labels = gt_boxes[gt_assignment, 4]


    if stage == 2:
        pos_thresh = 0.6
        bg_thresh_hi = 0.6
        bbox_std = cfg.TRAIN.BBOX_STDS_2nd
    else:
        pos_thresh = 0.7
        bg_thresh_hi = 0.7
        bbox_std = cfg.TRAIN.BBOX_STDS_3rd


    # foreground RoI with FG_THRESH overlap
    fg_indexes = np.where(overlaps >= pos_thresh)[0]
    bg_indexes = np.where((overlaps < bg_thresh_hi) & (overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]

    # indexes selected
    keep_indexes = np.append(fg_indexes, bg_indexes)

    # pad more to ensure a fixed minibatch size
    while keep_indexes.shape[0] < rois_per_image:
        gap = np.minimum(len(rois), rois_per_image - keep_indexes.shape[0])
        gap_indexes = npr.choice(range(len(rois)), size=gap, replace=False)
        keep_indexes = np.append(keep_indexes, gap_indexes)

    # select labels
    labels = labels[keep_indexes]
    # set labels of bg_rois to be 0
    overlaps = overlaps[keep_indexes]
    # set labels of bg_rois to be 0
    labels[(overlaps < bg_thresh_hi) & (overlaps >= cfg.TRAIN.BG_THRESH_LO)] = 0
    rois = rois[keep_indexes]

    # load or compute bbox_target
    if bbox_targets is not None:
        bbox_target_data = bbox_targets[keep_indexes, :]
    else:
        targets = bbox_transform(rois[:, 1:], gt_boxes[gt_assignment[keep_indexes], :4])
        if cfg.TRAIN.BBOX_NORMALIZATION_PRECOMPUTED:
            targets = ((targets - np.array(cfg.TRAIN.BBOX_MEANS))
                       / np.array(bbox_std))
        bbox_target_data = np.hstack((labels[:, np.newaxis], targets))

    bbox_targets, bbox_weights = \
        expand_bbox_regression_targets(bbox_target_data, num_classes, cfg)

    return rois, labels, bbox_targets, bbox_weights

class CascadeProposalTargetOperator(mx.operator.CustomOp):
    def __init__(self, num_classes, batch_images, batch_rois, batch_rois_off, cfg, fg_fraction, stage):
        super(CascadeProposalTargetOperator, self).__init__()
        self._num_classes = num_classes
        self._batch_images = batch_images
        self._batch_rois = batch_rois
        self._batch_rois_off = batch_rois_off
        self._cfg = cfg
        self._fg_fraction = fg_fraction
        self._stage = stage

        if DEBUG:
            self._count = 0
            self._fg_num = 0
            self._bg_num = 0

    def forward(self, is_train, req, in_data, out_data, aux):
        assert self._batch_rois == -1 or self._batch_rois % self._batch_images == 0, \
            'batchimages {} must devide batch_rois {}'.format(self._batch_images, self._batch_rois)
        all_rois = in_data[0].asnumpy()
        all_rois_off = in_data[1].asnumpy()
        gt_boxes = in_data[2].asnumpy()

        if self._batch_rois == -1:
            rois_per_image = all_rois.shape[0] + gt_boxes.shape[0]
            fg_rois_per_image = rois_per_image
        else:
            rois_per_image = self._batch_rois / self._batch_images
            fg_rois_per_image = np.round(self._fg_fraction * rois_per_image).astype(int)


        # Include ground-truth boxes in the set of candidate rois
        # zeros = np.zeros((gt_boxes.shape[0], 1), dtype=gt_boxes.dtype)
        # all_rois = np.vstack((all_rois, np.hstack((zeros, gt_boxes[:, :-1]))))
        # Sanity check: single batch only
        assert np.all(all_rois[:, 0] == 0), 'Only single item batches are supported'

        # generate new bbox result
        bbox_mean = self._cfg.TRAIN.BBOX_MEANS

        if self._stage == 2:
            bbox_std = self._cfg.TRAIN.BBOX_STDS_2nd
        else:
            bbox_std = self._cfg.TRAIN.BBOX_STDS_3rd

        stds = np.tile(
            np.array(bbox_std), (self._num_classes))
        means = np.tile(
            np.array(bbox_mean), (self._num_classes))

        all_rois_off *= stds
        all_rois_off += means
        all_rois_pred = nonlinear_pred(all_rois[:, 1:5], all_rois_off[:, 4:8])
        zeros = np.zeros((all_rois_pred.shape[0], 1), dtype=all_rois_pred.dtype)
        all_rois_pred = np.hstack((zeros, all_rois_pred))

        rois, labels, bbox_targets, bbox_weights = \
            sample_rois(all_rois_pred, fg_rois_per_image, rois_per_image, self._num_classes, self._cfg, gt_boxes=gt_boxes, stage = self._stage)

        if DEBUG:
            print "labels=", labels
            print 'num fg: {}'.format((labels > 0).sum())
            print 'num bg: {}'.format((labels == 0).sum())
            self._count += 1
            self._fg_num += (labels > 0).sum()
            self._bg_num += (labels == 0).sum()
            print "self._count=", self._count
            print 'num fg avg: {}'.format(self._fg_num / self._count)
            print 'num bg avg: {}'.format(self._bg_num / self._count)
            print 'ratio: {:.3f}'.format(float(self._fg_num) / float(self._bg_num))

        for ind, val in enumerate([rois, labels, bbox_targets, bbox_weights]):
            self.assign(out_data[ind], req[ind], val)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], 0)
        self.assign(in_grad[1], req[1], 0)


@mx.operator.register('cascade_proposal_target')
class CascadeProposalTargetProp(mx.operator.CustomOpProp):
    def __init__(self, num_classes, batch_images, batch_rois, batch_rois_off, cfg, fg_fraction='0.25', stage=2):
        super(CascadeProposalTargetProp, self).__init__(need_top_grad=False)
        self._num_classes = int(num_classes)
        self._batch_images = int(batch_images)
        self._batch_rois = int(batch_rois)
        self._batch_rois_off = batch_rois_off
        self._cfg = cPickle.loads(cfg)
        self._fg_fraction = float(fg_fraction)
        self._stage = stage

    def list_arguments(self):
        return ['rois', 'bbox_offset', 'gt_boxes']

    def list_outputs(self):
        return ['rois_output', 'label', 'bbox_target', 'bbox_weight']

    def infer_shape(self, in_shape):
        rpn_rois_shape = in_shape[0]
        bbox_offset_shape = in_shape[1]
        gt_boxes_shape = in_shape[2]

        rois = rpn_rois_shape[0] + gt_boxes_shape[0] if self._batch_rois == -1 else self._batch_rois

        output_rois_shape = (rois, 5)
        label_shape = (rois, )
        bbox_target_shape = (rois, self._num_classes * 4)
        bbox_weight_shape = (rois, self._num_classes * 4)

        return [rpn_rois_shape, bbox_offset_shape, gt_boxes_shape], \
               [output_rois_shape, label_shape, bbox_target_shape, bbox_weight_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return CascadeProposalTargetOperator(self._num_classes, self._batch_images, self._batch_rois, self._batch_rois_off, self._cfg, self._fg_fraction)

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []
