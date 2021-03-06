"""
This code is adapted from: https://github.com/ZJULearning/RMI

The implementation of the paper:
Region Mutual Information Loss for Semantic Segmentation.
"""

# python 2.X, 3.X compatibility
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rmi_utils import map_get_pairs, log_det_by_cholesky

_euler_num = 2.718281828        # euler number
_pi = 3.14159265		# pi
_ln_2_pi = 1.837877		# ln(2 * pi)
_CLIP_MIN = 1e-6        	# min clip value after softmax or sigmoid operations
_CLIP_MAX = 1.0    		# max clip value after softmax or sigmoid operations
_POS_ALPHA = 5e-4		# add this factor to ensure the AA^T is positive definite
_IS_SUM = 1			# sum the loss per channel


__all__ = ['RMILoss', 'RMILossAux']


class RMILoss(nn.Module):
    """
    region mutual information
    I(A, B) = H(A) + H(B) - H(A, B)
    This version need a lot of memory if do not downsample.
    """
    def __init__(self, num_classes=21, rmi_radius=3, rmi_pool_way=1, rmi_pool_size=4,
                 rmi_pool_stride=4, loss_weight_lambda=0.5, lambda_way=1, ignore_index=255):
        super().__init__()
        self.num_classes = num_classes
        # radius choices
        assert rmi_radius in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.rmi_radius = rmi_radius
        assert rmi_pool_way in [0, 1, 2, 3]
        self.rmi_pool_way = rmi_pool_way

        # set the pool_size = rmi_pool_stride
        assert rmi_pool_size == rmi_pool_stride
        self.rmi_pool_size = rmi_pool_size
        self.rmi_pool_stride = rmi_pool_stride
        self.weight_lambda = loss_weight_lambda
        self.lambda_way = lambda_way

        # dimension of the distribution
        self.half_d = self.rmi_radius * self.rmi_radius
        self.d = 2 * self.half_d
        self.kernel_padding = self.rmi_pool_size // 2
        # ignore class
        self.ignore_index = ignore_index

    def forward(self, logits_4D: torch.FloatTensor, labels_4D: torch.LongTensor,
                do_rmi=True) -> torch.Tensor:
        # explicitly disable fp16 mode because torch.cholesky and
        # torch.inverse aren't supported by half
        with torch.cuda.amp.autocast(enabled=False):
            loss = self.forward_sigmoid(
                logits_4D.float(), labels_4D.float(), do_rmi=do_rmi)
        return loss

    def forward_sigmoid(self, logits_4D, labels_4D, do_rmi=False):
        """
        Using the sigmiod operation both.
        Args:
                logits_4D 	:	[N, C, H, W], dtype=float32
                labels_4D 	:	[N, H, W], dtype=long
                do_rmi          :       bool
        """
        if len(labels_4D.shape) == 4:
            labels_4D = labels_4D.squeeze(1)

        # label mask -- [N, H, W, 1]
        label_mask_3d = labels_4D < self.num_classes

        # valid label
        valid_onehot_labels_4d = F.one_hot(labels_4D.long() * label_mask_3d.long(),
                                           num_classes=self.num_classes).float()
        label_mask_3d = label_mask_3d.float()
        label_mask_flat = label_mask_3d.view([-1, ])
        valid_onehot_labels_4d = valid_onehot_labels_4d * label_mask_3d.unsqueeze(dim=3)
        valid_onehot_labels_4d.requires_grad_(False)

        # PART I -- calculate the sigmoid binary cross entropy loss
        valid_onehot_label_flat = \
            valid_onehot_labels_4d.view([-1, self.num_classes]).requires_grad_(False)
        logits_flat = logits_4D.permute(0, 2, 3, 1).contiguous().view([-1, self.num_classes])

        # binary loss, multiplied by the not_ignore_mask
        valid_pixels = torch.sum(label_mask_flat)
        binary_loss = F.binary_cross_entropy_with_logits(
            logits_flat, target=valid_onehot_label_flat,
            weight=label_mask_flat.unsqueeze(dim=1), reduction='sum')

        bce_loss = torch.div(binary_loss, valid_pixels + 1.0)
        if not do_rmi:
            return bce_loss

        # PART II -- get rmi loss
        # onehot_labels_4D -- [N, C, H, W]
        probs_4d = logits_4D.sigmoid() * label_mask_3d.unsqueeze(dim=1) + _CLIP_MIN
        valid_onehot_labels_4d = valid_onehot_labels_4d.permute(0, 3, 1, 2).requires_grad_(False)

        # get region mutual information
        rmi_loss = self.rmi_lower_bound(valid_onehot_labels_4d, probs_4d)

        # add together
        if self.lambda_way:
            final_loss = self.weight_lambda * bce_loss + rmi_loss * (1 - self.weight_lambda)
        else:
            final_loss = bce_loss + rmi_loss * self.weight_lambda

        return final_loss

    def rmi_lower_bound(self, labels_4D, probs_4D):
        """
        calculate the lower bound of the region mutual information.
        Args:
                labels_4D 	:	[N, C, H, W], dtype=float32
                probs_4D 	:	[N, C, H, W], dtype=float32
        """
        assert labels_4D.size() == probs_4D.size()

        p, s = self.rmi_pool_size, self.rmi_pool_stride
        if self.rmi_pool_stride > 1:
            if self.rmi_pool_way == 0:
                labels_4D = F.max_pool2d(labels_4D, kernel_size=p, stride=s,
                                         padding=self.kernel_padding)
                probs_4D = F.max_pool2d(probs_4D, kernel_size=p, stride=s,
                                        padding=self.kernel_padding)
            elif self.rmi_pool_way == 1:
                labels_4D = F.avg_pool2d(labels_4D, kernel_size=p, stride=s,
                                         padding=self.kernel_padding)
                probs_4D = F.avg_pool2d(probs_4D, kernel_size=p, stride=s,
                                        padding=self.kernel_padding)
            elif self.rmi_pool_way == 2:
                # interpolation
                shape = labels_4D.size()
                new_h, new_w = shape[2] // s, shape[3] // s
                labels_4D = F.interpolate(labels_4D, size=(new_h, new_w), mode='nearest')
                probs_4D = F.interpolate(probs_4D, size=(new_h, new_w), mode='bilinear',
                                         align_corners=True)
            else:
                raise NotImplementedError("Pool way of RMI is not defined!")
        # we do not need the gradient of label.
        label_shape = labels_4D.size()
        n, c = label_shape[0], label_shape[1]

        # combine the high dimension points from label and probability map.
        # new shape [N, C, radius * radius, H, W]
        la_vectors, pr_vectors = map_get_pairs(
            labels_4D, probs_4D, radius=self.rmi_radius, is_combine=0)

        la_vectors = la_vectors.view([n, c, self.half_d, -1]).type(torch.cuda.DoubleTensor).requires_grad_(False)
        pr_vectors = pr_vectors.view([n, c, self.half_d, -1]).type(torch.cuda.DoubleTensor)

        # small diagonal matrix, shape = [1, 1, radius * radius, radius * radius]
        diag_matrix = torch.eye(self.half_d).unsqueeze(dim=0).unsqueeze(dim=0)

        # the mean and covariance of these high dimension points
        # Var(X) = E(X^2) - E(X) E(X), N * Var(X) = X^2 - X E(X)
        la_vectors = la_vectors - la_vectors.mean(dim=3, keepdim=True)
        la_cov = torch.matmul(la_vectors, la_vectors.transpose(2, 3))

        pr_vectors = pr_vectors - pr_vectors.mean(dim=3, keepdim=True)
        pr_cov = torch.matmul(pr_vectors, pr_vectors.transpose(2, 3))
        # https://github.com/pytorch/pytorch/issues/7500
        # waiting for batched torch.cholesky_inverse()
        pr_cov_inv = torch.inverse(pr_cov + diag_matrix.type_as(pr_cov) * _POS_ALPHA)
        # if the dimension of the point is less than 9, you can use the below function
        # to acceleration computational speed.
        #pr_cov_inv = utils.batch_cholesky_inverse(pr_cov + diag_matrix.type_as(pr_cov) * _POS_ALPHA)

        la_pr_cov = torch.matmul(la_vectors, pr_vectors.transpose(2, 3))
        # the approxiamation of the variance, det(c A) = c^n det(A), A is in n x n shape;
        # then log det(c A) = n log(c) + log det(A).
        # appro_var = appro_var / n_points, we do not divide the appro_var by number of points here,
        # and the purpose is to avoid underflow issue.
        # If A = A^T, A^-1 = (A^-1)^T.
        appro_var = la_cov - torch.matmul(la_pr_cov.matmul(pr_cov_inv), la_pr_cov.transpose(-2, -1))
        #appro_var = la_cov - torch.chain_matmul(la_pr_cov, pr_cov_inv, la_pr_cov.transpose(-2, -1))
        #appro_var = torch.div(appro_var, n_points.type_as(appro_var)) + diag_matrix.type_as(appro_var) * 1e-6

        # The lower bound. If A is nonsingular, ln( det(A) ) = Tr( ln(A) ).
        rmi_now = 0.5 * log_det_by_cholesky(appro_var + diag_matrix.type_as(appro_var) * _POS_ALPHA)
        #rmi_now = 0.5 * torch.logdet(appro_var + diag_matrix.type_as(appro_var) * _POS_ALPHA)

        # mean over N samples. sum over classes.
        rmi_per_class = rmi_now.view([-1, self.num_classes]).mean(dim=0).float()
        #is_half = False
        #if is_half:
        #	rmi_per_class = torch.div(rmi_per_class, float(self.half_d / 2.0))
        #else:
        rmi_per_class = torch.div(rmi_per_class, float(self.half_d))

        rmi_loss = torch.sum(rmi_per_class) if _IS_SUM else torch.mean(rmi_per_class)
        return rmi_loss

class MultiScaleRMILoss(nn.Module):
    def __init__(self, ocr_aux_rmi=False, supervised_mscale_wt=0.0, alpha=0.4, **kwargs):
        super().__init__()
        self.alpha = alpha
        self.ocr_aux_rmi = ocr_aux_rmi
        self.supervised_mscale_wt = supervised_mscale_wt
        self.rmi = RMILoss(**kwargs)

    def forward(self, seg_pred: Dict[str, torch.Tensor], seg_gt: torch.Tensor, **kwargs):
        aux_loss = self.rmi(seg_pred['aux'], seg_gt, do_rmi=self.ocr_aux_rmi)

        # Optionally turn off RMI loss for first epoch to try to work
        # around cholesky errors of singular matrix
        do_rmi_main = True  # cfg.EPOCH > 0
        main_loss = self.rmi(seg_pred['pred'], seg_gt, do_rmi=do_rmi_main)
        loss = self.alpha * aux_loss + main_loss

        # Optionally, apply supervision to the multi-scale predictions
        # directly. Turn off RMI to keep things lightweight
        if self.supervised_mscale_wt:
            scaled_pred_05x = F.interpolate(
                seg_pred['pred_05x'], size=tuple(seg_pred['pred_10x'][2:]),
                mode='bilinear', align_corners=True, recompute_scale_factor=True)

            loss_lo = self.rmi(scaled_pred_05x, seg_gt, do_rmi=False)
            loss_hi = self.rmi(seg_pred['pred_10x'], seg_gt, do_rmi=False)

            loss += self.supervised_mscale_wt * loss_lo
            loss += self.supervised_mscale_wt * loss_hi

        return loss

class RMILossAux(nn.Module):
    def __init__(self, weight=1., aux_weight=0.4, **kwargs):
        super().__init__()
        self.rmi = RMILoss(**kwargs)
        self.aux_weight = aux_weight
        self.weight = weight

    def forward(self, predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        assert all('seg' in dict_ for dict_ in [predictions.keys(), targets.keys()])
        seg_gt = targets['seg']

        seg_loss = self.rmi(predictions['seg'], seg_gt)

        if 'seg_aux' in predictions.keys():
            seg_loss += (self.aux_weight * self.rmi(predictions['seg_aux'], seg_gt))

        return self.weight * seg_loss
