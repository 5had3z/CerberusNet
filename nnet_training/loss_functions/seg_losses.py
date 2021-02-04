"""Various Segmentation losses."""

from typing import Dict

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

__all__ = ['MixSoftmaxCrossEntropyLoss', 'MixSoftmaxCrossEntropyOHEMLoss',
           'FocalLoss2D', 'SegCrossEntropy']

### MixSoftmaxCrossEntropyLoss etc from F-SCNN Repo
class MixSoftmaxCrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, aux=True, aux_weight=0.2, ignore_label=255, **kwargs):
        super(MixSoftmaxCrossEntropyLoss, self).__init__(ignore_index=ignore_label)
        self.aux = aux
        self.aux_weight = aux_weight

    def _aux_forward(self, *inputs, **kwargs):
        *preds, target = tuple(inputs)

        loss = super(MixSoftmaxCrossEntropyLoss, self).forward(preds[0], target)
        for i in range(1, len(preds)):
            aux_loss = super(MixSoftmaxCrossEntropyLoss, self).forward(preds[i], target)
            loss += self.aux_weight * aux_loss
        return loss

    def forward(self, *inputs, **kwargs):
        preds, target = tuple(inputs)
        inputs = tuple(list(preds) + [target])
        if self.aux:
            return self._aux_forward(*inputs)
        else:
            return super(MixSoftmaxCrossEntropyLoss, self).forward(*inputs)


class SoftmaxCrossEntropyOHEMLoss(nn.Module):
    def __init__(self, ignore_label=-1, thresh=0.7, min_kept=256, use_weight=True, **kwargs):
        super(SoftmaxCrossEntropyOHEMLoss, self).__init__()
        self.ignore_label = ignore_label
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        if use_weight:
            print("w/ class balance")
            weight = torch.FloatTensor([0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754,
                                        1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955,
                                        1.0865, 1.1529, 1.0507])
            self.criterion = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_label)
        else:
            print("w/o class balance")
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_label)

    def forward(self, predict, target, weight=None):
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))

        n, c, h, w = predict.size()
        input_label = target.data.cpu().numpy().ravel().astype(np.int32)
        x = np.rollaxis(predict.data.cpu().numpy(), 1).reshape((c, -1))
        input_prob = np.exp(x - x.max(axis=0).reshape((1, -1)))
        input_prob /= input_prob.sum(axis=0).reshape((1, -1))

        valid_flag = input_label != self.ignore_label
        valid_inds = np.where(valid_flag)[0]
        label = input_label[valid_flag]
        num_valid = valid_flag.sum()
        if self.min_kept >= num_valid:
            print('Labels: {}'.format(num_valid))
        elif num_valid > 0:
            prob = input_prob[:, valid_flag]
            pred = prob[label, np.arange(len(label), dtype=np.int32)]
            threshold = self.thresh
            if self.min_kept > 0:
                index = pred.argsort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if pred[threshold_index] > self.thresh:
                    threshold = pred[threshold_index]
            kept_flag = pred <= threshold
            valid_inds = valid_inds[kept_flag]

        label = input_label[valid_inds].copy()
        input_label.fill(self.ignore_label)
        input_label[valid_inds] = label
        valid_flag_new = input_label != self.ignore_label
        # print(np.sum(valid_flag_new))
        target = Variable(torch.from_numpy(input_label.reshape(target.size())).long().cuda())

        return self.criterion(predict, target)


class MixSoftmaxCrossEntropyOHEMLoss(SoftmaxCrossEntropyOHEMLoss):
    def __init__(self, aux=False, aux_weight=0.2, ignore_index=255, **kwargs):
        super(MixSoftmaxCrossEntropyOHEMLoss, self).__init__(ignore_label=ignore_index, **kwargs)
        self.aux = aux
        self.aux_weight = aux_weight

    def _aux_forward(self, *inputs, **kwargs):
        *preds, target = tuple(inputs)

        loss = super(MixSoftmaxCrossEntropyOHEMLoss, self).forward(preds[0], target)
        for i in range(1, len(preds)):
            aux_loss = super(MixSoftmaxCrossEntropyOHEMLoss, self).forward(preds[i], target)
            loss += self.aux_weight * aux_loss
        return loss

    def forward(self, *inputs, **kwargs):
        preds, target = tuple(inputs)
        inputs = tuple(list([preds]) + [target])
        if self.aux:
            return self._aux_forward(*inputs)
        else:
            return super(MixSoftmaxCrossEntropyOHEMLoss, self).forward(*inputs)

class FocalLoss2D(nn.Module):
    """
    Focal Loss for Imbalanced problems, also includes additonal weighting
    """
    def __init__(self, weight=1.0, gamma=2.0, ignore_index=255, dynamic_weights=False,
                 scale_factor=0.125, **kwargs):
        super().__init__()

        self.weight = weight
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.dynamic_weights = dynamic_weights
        self.scale_factor = scale_factor

    def forward(self, predictions: Dict[str, torch.Tensor],
                targets: torch.Tensor) -> torch.Tensor:
        '''
        Forward implementation that returns focal loss between prediciton and target
        '''
        assert all('seg' in dict_ for dict_ in [predictions.keys(), targets.keys()])
        seg_gt = targets['seg']

        weights = torch.ones(predictions['seg'].shape[1]).to(predictions['seg'].get_device())
        if self.dynamic_weights:
            class_ids, counts = seg_gt[seg_gt != self.ignore_index].unique(return_counts=True)
            weights[class_ids] = self.scale_factor / \
                    (self.scale_factor + counts / float(seg_gt.nelement()))

        # compute the negative likelyhood
        ce_loss = F.cross_entropy(
            predictions['seg'], seg_gt, ignore_index=self.ignore_index, weight=weights)

        # compute the loss
        focal_loss = torch.pow(1 - torch.exp(-ce_loss), self.gamma) * ce_loss

        # return the average
        return self.weight * focal_loss.mean()

class SegCrossEntropy(nn.Module):
    def __init__(self, weight=1.0, ignore_index=255, dynamic_weights=False,
                 scale_factor=0.125, **kwargs):
        super().__init__()

        self.weight = weight
        self.ignore_index = ignore_index
        self.dynamic_weights = dynamic_weights
        self.scale_factor = scale_factor

    def forward(self, predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        '''
        Forward implementation that returns cross entropy between prediciton and target
        '''
        assert all('seg' in dict_ for dict_ in [predictions.keys(), targets.keys()])
        seg_gt = targets['seg']

        weights = torch.ones(predictions['seg'].shape[1]).to(predictions['seg'].get_device())
        if self.dynamic_weights:
            class_ids, counts = seg_gt[seg_gt != self.ignore_index].unique(return_counts=True)
            weights[class_ids] = self.scale_factor / \
                    (self.scale_factor + counts / float(seg_gt.nelement()))

        return self.weight * F.cross_entropy(
            predictions['seg'], seg_gt, ignore_index=self.ignore_index, weight=weights)
