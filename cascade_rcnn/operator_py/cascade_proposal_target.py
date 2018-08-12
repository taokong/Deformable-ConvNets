# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2016 by Contributors
# Copyright (c) 2018 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Modified by Tao Kong
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
from bbox.bbox_transform import bbox_overlaps, bbox_transform, bbox_pred, clip_boxes
from bbox.bbox_regression import expand_bbox_regression_targets

DEBUG = False


def sample_rois(rois, rois_per_image, num_classes, cfg,
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


    if int(stage) == 2:
        bg_thresh_hi = 0.6
        bbox_std = cfg.TRAIN.BBOX_STDS_2nd
    else:
        bg_thresh_hi = 0.7
        bbox_std = cfg.TRAIN.BBOX_STDS_3rd


    # set labels of bg_rois to be 0
    bg_inds = np.where(overlaps < bg_thresh_hi)[0]
    labels[bg_inds] = 0

    # load or compute bbox_target
    if bbox_targets is not None:
        bbox_target_data = bbox_targets
    else:
        targets = bbox_transform(rois[:, 1:], gt_boxes[gt_assignment, :4])
        if cfg.TRAIN.BBOX_NORMALIZATION_PRECOMPUTED:
            targets = ((targets - np.array(cfg.TRAIN.BBOX_MEANS))
                       / np.array(bbox_std))
        bbox_target_data = np.hstack((labels[:, np.newaxis], targets))

    bbox_targets, bbox_weights = \
        expand_bbox_regression_targets(bbox_target_data, num_classes, cfg)

    return rois, labels, bbox_targets, bbox_weights

class CascadeProposalTargetOperator(mx.operator.CustomOp):
    def __init__(self, num_classes, batch_images, batch_rois, cfg, fg_fraction, stage):
        super(CascadeProposalTargetOperator, self).__init__()
        self._num_classes = num_classes
        self._batch_images = batch_images
        self._batch_rois = batch_rois
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
        all_rois_off = in_data[1].asnumpy()[0]
        gt_boxes = in_data[2].asnumpy()
        im_info = in_data[3].asnumpy()[0, :]

        # print(np.shape(all_rois), np.shape(all_rois_off), np.shape(gt_boxes))
        if self._batch_rois == -1:
            rois_per_image = all_rois.shape[0] + gt_boxes.shape[0]
        else:
            rois_per_image = self._batch_rois / self._batch_images


        # Include ground-truth boxes in the set of candidate rois
        # zeros = np.zeros((gt_boxes.shape[0], 1), dtype=gt_boxes.dtype)
        # all_rois = np.vstack((all_rois, np.hstack((zeros, gt_boxes[:, :-1]))))
        # Sanity check: single batch only
        assert np.all(all_rois[:, 0] == 0), 'Only single item batches are supported'

        # generate new bbox result
        bbox_mean = self._cfg.TRAIN.BBOX_MEANS

        if self._stage == 2:
            bbox_std = self._cfg.TRAIN.BBOX_STDS_1st
        else:
            bbox_std = self._cfg.TRAIN.BBOX_STDS_2nd

        stds = np.tile(
            np.array(bbox_std), (self._num_classes))
        means = np.tile(
            np.array(bbox_mean), (self._num_classes))

        all_rois_off *= stds
        all_rois_off += means
        all_rois_pred = bbox_pred(all_rois[:, 1:5], all_rois_off[:, 4:8])
        all_rois_pred = clip_boxes(all_rois_pred, im_info[:2])
        zeros = np.zeros((all_rois_pred.shape[0], 1), dtype=all_rois_pred.dtype)
        all_rois_pred = np.hstack((zeros, all_rois_pred))


        # avoid invalid bboxes
        all_rois_pred = np.nan_to_num(all_rois_pred)
        ws = all_rois_pred[:, 3] - all_rois_pred[:, 1]
        hs = all_rois_pred[:, 4] - all_rois_pred[:, 2]
        all_rois_pred[ws < 0, 3] = all_rois_pred[ws < 0, 1]
        all_rois_pred[hs < 0, 4] = all_rois_pred[hs < 0, 2]

        rois, labels, bbox_targets, bbox_weights = \
            sample_rois(all_rois_pred, rois_per_image, self._num_classes, self._cfg, gt_boxes=gt_boxes, stage = self._stage)

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
    def __init__(self, num_classes, batch_images, batch_rois, cfg, fg_fraction='0.25', stage=2):
        super(CascadeProposalTargetProp, self).__init__(need_top_grad=False)
        self._num_classes = int(num_classes)
        self._batch_images = int(batch_images)
        self._batch_rois = int(batch_rois)
        self._cfg = cPickle.loads(cfg)
        self._fg_fraction = float(fg_fraction)
        self._stage = stage

    def list_arguments(self):
        return ['rois', 'bbox_offset', 'gt_boxes', 'im_info']

    def list_outputs(self):
        return ['rois_output', 'label', 'bbox_target', 'bbox_weight']

    def infer_shape(self, in_shape):
        rpn_rois_shape = in_shape[0]
        bbox_offset_shape = in_shape[1]
        gt_boxes_shape = in_shape[2]
        im_info_shape = in_shape[3]


        rois = self._batch_rois
        output_rois_shape = (rois, 5)
        label_shape = (rois, )
        bbox_target_shape = (rois, self._num_classes * 4)
        bbox_weight_shape = (rois, self._num_classes * 4)

        return [rpn_rois_shape, bbox_offset_shape, gt_boxes_shape, im_info_shape], \
               [output_rois_shape, label_shape, bbox_target_shape, bbox_weight_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return CascadeProposalTargetOperator(self._num_classes, self._batch_images, self._batch_rois, self._cfg, self._fg_fraction, self._stage)

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []
