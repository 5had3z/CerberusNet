#!/usr/bin/env python3

__author__ = "Bryce Ferenczi"
__email__ = "bryce.ferenczi@monash.edu"

import sys
from typing import Dict

import torch

from .base import MetricBase

class DepthMetric(MetricBase):
    """
    Accuracy/Error and Loss Staticstics tracking for depth based networks.\n
    Tracks Invariant, RMSE Linear, RMSE Log, Squared Relative and Absolute Relative.
    """
    def __init__(self, **kwargs):
        super().__init__(savefile="depth_Data", **kwargs)
        self._reset_metric()
        assert self.main_metric in self.metric_data.keys()

    def add_sample(self, predictions: Dict[str, torch.Tensor],
                   targets: Dict[str, torch.Tensor], loss: int=0, **kwargs) -> None:
        assert 'depth' in predictions.keys() and 'disparity' in targets.keys()
        self.metric_data["Batch_Loss"].append(loss)

        gt_depth = targets['disparity']

        if isinstance(predictions['depth'], list):
            pred_depth = predictions['depth'][0]
        else:
            pred_depth = predictions['depth']

        pred_depth = pred_depth.squeeze(dim=1)
        pred_depth[pred_depth == 0] += 1e-7

        gt_mask = gt_depth < 80.
        gt_mask &= gt_depth > 0.
        n_valid = torch.sum(gt_mask, dim=(1, 2))
        gt_mask = ~gt_mask

        difference = pred_depth - gt_depth
        difference[gt_mask] = 0.
        squared_diff = difference.pow(2)

        log_diff = (torch.log(pred_depth) - torch.log(gt_depth))
        log_diff[gt_mask] = 0.
        sq_log_diff = torch.sum(log_diff.pow(2), dim=(1, 2)) / n_valid

        abs_rel = difference.abs() / gt_depth
        abs_rel[gt_mask] = 0.
        self.metric_data['Batch_Absolute_Relative'].append(
            (torch.sum(abs_rel, dim=(1, 2)) / n_valid).cpu().data.numpy())

        sqr_rel = squared_diff / gt_depth
        sqr_rel[gt_mask] = 0.
        self.metric_data['Batch_Squared_Relative'].append(
            (torch.sum(sqr_rel, dim=(1, 2)) / n_valid).cpu().data.numpy())

        self.metric_data['Batch_RMSE_Linear'].append(
            torch.sqrt(torch.sum(squared_diff, dim=(1, 2)) / n_valid).cpu().data.numpy())

        self.metric_data['Batch_RMSE_Log'].append(
            torch.sqrt(sq_log_diff).cpu().data.numpy())

        eqn1 = sq_log_diff
        eqn2 = torch.sum(log_diff.abs(), dim=(1, 2))**2 / n_valid**2
        self.metric_data['Batch_Invariant'].append((eqn1 - eqn2).cpu().data.numpy())

        threshold = torch.max(pred_depth / gt_depth, gt_depth / pred_depth)
        threshold[gt_mask] = 1.25 ** 3
        self.metric_data['Batch_a1'].append(
            (torch.sum(threshold < 1.25, dim=(1, 2)) / n_valid).cpu().numpy())
        self.metric_data['Batch_a2'].append(
            (torch.sum(threshold < 1.25 ** 2, dim=(1, 2)) / n_valid).cpu().numpy())
        self.metric_data['Batch_a3'].append(
            (torch.sum(threshold < 1.25 ** 3, dim=(1, 2)) / n_valid).cpu().numpy())

    def max_accuracy(self, main_metric=True):
        """
        Returns highest scale invariant, absolute relative, squared relative,
        rmse linear, rmse log Accuracy from per epoch summarised data.\n
        @param  main_metric, if true only returns scale invariant\n
        @output if main metric, will return its name and value
        @output scale invariant, absolute relative, squared relative, rmse linear, rmse log
        """
        cost_func = {
            'Batch_Loss': [min, sys.float_info.max],
            'Batch_Invariant': [min, sys.float_info.max],
            'Batch_RMSE_Log': [min, sys.float_info.max],
            'Batch_RMSE_Linear': [min, sys.float_info.max],
            'Batch_Squared_Relative': [min, sys.float_info.max],
            'Batch_Absolute_Relative': [min, sys.float_info.max],
            'Batch_a1': [max, 0.],
            'Batch_a2': [max, 0.],
            'Batch_a3': [max, 0.],
        }

        if self._path is not None:
            ret_type = self.main_metric if main_metric else None
            return self._max_accuracy_lambda(cost_func, self._path, ret_type)

        print("No File Specified for Depth Statistics Manager")
        return None

    def _reset_metric(self):
        self.metric_data = dict(
            Batch_Loss=[],
            Batch_Absolute_Relative=[],
            Batch_Squared_Relative=[],
            Batch_RMSE_Linear=[],
            Batch_RMSE_Log=[],
            Batch_Invariant=[],
            Batch_a1=[],
            Batch_a2=[],
            Batch_a3=[]
        )

if __name__ == "__main__":
    pass
