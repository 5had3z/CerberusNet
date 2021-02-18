#!/usr/bin/env python3

__author__ = "Bryce Ferenczi"
__email__ = "bryce.ferenczi@monash.edu"

import sys
from typing import Dict

import torch
import numpy as np

from nnet_training.loss_functions.UnFlowLoss import flow_warp

from .base import MetricBase

class OpticFlowMetric(MetricBase):
    """
    Accuracy/Error and Loss Staticstics tracking for depth based networks.\n
    Tracks Flow rnd point error (EPE) and sum absolute difference between
    warped image sequence (SAD).
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._reset_metric()
        assert self.main_metric in self.metric_data.keys()

    @staticmethod
    def error_rate(epe_map, gt_flow, mask):
        """
        Calculates the outlier rate given a flow epe, ground truth and mask.
        """
        bad_pixels = torch.logical_and(
            epe_map * mask > 3,
            epe_map * mask / torch.maximum(
                torch.sqrt(torch.sum(torch.square(gt_flow), dim=1)),
                torch.tensor([1e-10], device=gt_flow.get_device())) > 0.05)
        return torch.sum(bad_pixels, dim=(1, 2)) / torch.sum(mask, dim=(1, 2))

    def add_sample(self, predictions: Dict[str, torch.Tensor],
                   targets: Dict[str, torch.Tensor], loss: int=0, **kwargs) -> None:
        """
        @input list of original, prediction and sequence images i.e. [left, right]
        """
        self.metric_data["Batch_Loss"].append(loss if loss is not None else 0)

        if isinstance(predictions['flow'], list):
            flow_pred = predictions['flow'][0]
        else:
            flow_pred = predictions['flow']

        if all(key in targets.keys() for key in ["flow", "flow_mask"]):
            targets["flow_mask"] = targets["flow_mask"].squeeze(1)
            n_valid = torch.sum(targets["flow_mask"], dim=(1, 2))

            diff = flow_pred - targets["flow"]
            norm_diff = ((diff[:, 0, :, :]**2 + diff[:, 1, :, :]**2)**0.5)
            masked_epe = torch.sum(norm_diff * targets["flow_mask"], dim=(1, 2))

            self.metric_data["Batch_EPE"].append(
                (masked_epe / n_valid).cpu().numpy())

            self.metric_data["Batch_Fl_all"].append(
                self.error_rate(norm_diff, targets["flow"], targets["flow_mask"]).cpu().numpy())
        else:
            self.metric_data["Batch_EPE"].append(np.zeros((flow_pred.shape[0], 1)))
            self.metric_data["Batch_Fl_all"].append(np.zeros((flow_pred.shape[0], 1)))

        self.metric_data["Batch_SAD"].append(
            (targets['l_img']-flow_warp(targets['l_seq'], flow_pred)) \
                .abs().mean(dim=(1, 2, 3)).cpu().numpy())

    def max_accuracy(self, main_metric=True):
        """
        Returns lowest end point error and sum absolute difference from per epoch summarised data.\n
        @param  main_metric, if true only returns scale invariant\n
        @output scale invariant, absolute relative, squared relative, rmse linear, rmse log
        """
        cost_func = {
            'Batch_Loss': [min, sys.float_info.max],
            'Batch_SAD': [min, sys.float_info.max],
            'Batch_EPE': [min, sys.float_info.max],
            'Batch_Fl_all': [min, sys.float_info.max]
        }

        if self._path is not None:
            ret_type = self.main_metric if main_metric else None
            return self._max_accuracy_lambda(cost_func, self._path, ret_type)

        print("No File Specified for Segmentation Metric Manager")
        return None

    def _reset_metric(self):
        self.metric_data = dict(
            Batch_Loss=[],
            Batch_SAD=[],
            Batch_Fl_all=[],
            Batch_EPE=[]
        )

if __name__ == "__main__":
    pass
