"""Custom losses."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

__all__ = ['MixSoftmaxCrossEntropyLoss', 'MixSoftmaxCrossEntropyOHEMLoss', 'FocalLoss2D',
    'DepthAwareLoss', 'ScaleInvariantError', 'InvHuberLoss']

### MixSoftmaxCrossEntropyLoss etc from F-SCNN Repo
class MixSoftmaxCrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, aux=True, aux_weight=0.2, ignore_label=-1, **kwargs):
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
    def __init__(self, aux=False, aux_weight=0.2, ignore_index=-1, **kwargs):
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
    https://github.com/doiken23/focal_segmentation/blob/master/focalloss2d.py
    OG Source but I've modified a bit
    """
    def __init__(self, gamma=0, weight=None, size_average=True, ignore_index=-100):
        super(FocalLoss2D, self).__init__()

        self.gamma = gamma
        self.weight = weight
        self.size_average = size_average
        self._ignore_index = ignore_index

    def forward(self, input, target):
        if input.dim()>2:
            input = input.contiguous().view(input.size(0), input.size(1), -1)
            input = input.transpose(1,2)
            input = input.contiguous().view(-1, input.size(2)).squeeze()
        if target.dim()==4:
            target = target.contiguous().view(target.size(0), target.size(1), -1)
            target = target.transpose(1,2)
            target = target.contiguous().view(-1, target.size(2)).squeeze()
        elif target.dim()==3:
            target = target.view(-1)
        else:
            target = target.view(-1, 1)

        # compute the negative likelyhood
        # weight = Variable(self.weight)
        logpt = -F.cross_entropy(input, target, ignore_index=self._ignore_index)
        pt = torch.exp(logpt)

        # compute the loss
        loss = -((1-pt)**self.gamma) * logpt

        # averaging (or not) loss
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

class DepthAwareLoss(nn.Module):
    def __init__(self, size_average=True, ignore_index=0):
        super(DepthAwareLoss, self).__init__()
        self.size_average = size_average
        self._ignore_index = ignore_index

    def forward(self, pred, target):
        pred = pred.squeeze(dim=1)
        regularization = 1 - torch.min(torch.log(pred), torch.log(target)) / torch.max(torch.log(pred), torch.log(target))
        l_loss = F.smooth_l1_loss(pred, target ,size_average=self.size_average)
        depth_aware_attention = target / torch.max(target)
        return ((depth_aware_attention + regularization)*l_loss).mean()

class ScaleInvariantError(nn.Module):
    def __init__(self, lmda=1, ignore_index=-1):
        super(ScaleInvariantError, self).__init__()
        self.lmda = lmda
        self._ignore_index = ignore_index

    def forward(self, pred, target):
        #   Number of pixels per image
        n_pixels = target.shape[1]*target.shape[2]
        #   Number of valid pixels in target image
        n_valid = (target != self._ignore_index).view(-1, n_pixels).float().sum(dim=1)

        #   Prevent infs and nans
        pred[pred<=0] = 0.00001
        target[target==self._ignore_index] = 0.00001
        target.unsqueeze_(dim=1)
        d = torch.log(pred) - torch.log(target)

        element_wise = torch.pow(d.view(-1, n_pixels),2).sum(dim=1)/n_valid
        scaled_error = self.lmda*(torch.pow(d.view(-1, n_pixels).sum(dim=1),2)/(2*(n_valid**2)))
        return (element_wise - scaled_error).sum()

class InvHuberLoss(nn.Module):
    def __init__(self, ignore_index=-1):
        super(InvHuberLoss, self).__init__()
        self.ignore_index = ignore_index
    
    def forward(self, pred, target):
        pred_relu = F.relu(pred.squeeze(dim=1)) # depth predictions must be >=0
        diff = pred_relu - target
        mask = target != self.ignore_index

        err = torch.abs(diff * mask.float())
        c = 0.2 * torch.max(err)
        err2 = (diff**2 + c**2) / (2. * c)
        mask_err = err <= c
        mask_err2 = err > c
        cost = torch.mean(err*mask_err.float() + err2*mask_err2.float())
        return cost

import PIL.Image as Image

if __name__ == '__main__':
    loss_fn = ScaleInvariantError()

    depth_map = Image.open('/media/bryce/4TB Seagate/Autonomous Vehicles Data/Cityscapes Data/disparity/test/berlin/berlin_000000_000019_disparity.png')
    disparity = np.array(depth_map).astype('float32')
    disparity[disparity > 1] = (0.209313 * 2262.52) / ((disparity[disparity > 1] - 1) / 256)
    disparity[disparity < 2] = -1 # Ignore value for loss functions
    uniform = torch.ones(disparity.shape).unsqueeze(0)
    ground_truth = torch.FloatTensor(disparity.astype('float32')).unsqueeze(0)

    loss = loss_fn(uniform, ground_truth)
    print(loss)