# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
from typing import Dict

import torch
import torch.nn.functional as F

import nnet_training.nnet_models.detr.box_ops as box_ops

from nnet_training.nnet_models.detr.segmentation import sigmoid_focal_loss
from nnet_training.nnet_models.detr.segmentation import dice_loss

from nnet_training.nnet_models.detr.matcher import HungarianMatcher

from nnet_training.nnet_models.detr.misc import accuracy
from nnet_training.nnet_models.detr.misc import nested_tensor_from_tensor_list
from nnet_training.nnet_models.detr.misc import interpolate
from nnet_training.nnet_models.detr.misc import is_dist_avail_and_initialized
from nnet_training.nnet_models.detr.misc import get_world_size

class DetrLoss(torch.nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, weight_dict, eos_coef, losses, weight=1.0, **kwargs):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative
                         weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = HungarianMatcher(**kwargs['matcher_cfg'])
        self.weight = weight
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, log=False):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'logits' in outputs
        src_logits = outputs['logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t[J] for t, (_, J) in zip(targets["labels"], indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @staticmethod
    @torch.no_grad()
    def loss_cardinality(outputs, targets, *_args):
        """ Compute the cardinality error, ie the absolute error in the number of predicted
            non-empty boxes.\n
            This is not really a loss, it is intended for logging purposes only, it doesn't
            propagate gradients.
        """
        pred_logits = outputs['logits']
        tgt_lengths = torch.as_tensor([len(v) for v in targets["labels"]],
                                      device=pred_logits.device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """ Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU
            loss targets dicts must contain the key "bboxes" containing a tensor of dim
            [nb_target_boxes, 4].\n
            The target boxes are expected in format (center_x, center_y, w, h),
            normalized by the image size.
        """
        assert 'bboxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['bboxes'][idx]
        target_boxes = torch.cat([t[i] for t, (_, i) in zip(targets['bboxes'], indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim
           [nb_target_boxes, h, w]
        """
        assert "masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["masks"]
        src_masks = src_masks[src_idx]

        target_masks = targets["masks"].to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    @staticmethod
    def _get_src_permutation_idx(indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    @staticmethod
    def _get_tgt_permutation_idx(indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see docs
        """
        detr_outputs = {k: v for k, v in predictions.items() if k in ['logits', 'bboxes', 'masks']}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(detr_outputs, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t) for t in targets["labels"])
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float,
                                    device=next(iter(predictions.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, predictions, targets, indices, num_boxes))

        # For auxiliary losses, repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in predictions:
            for i, aux_outputs in enumerate(predictions['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return self.weight * sum([losses[k] * self.weight_dict[k] for k in losses
                                  if k in self.weight_dict.keys()])
