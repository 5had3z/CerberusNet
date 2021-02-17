#!/usr/bin/env python3

__author__ = "Bryce Ferenczi"
__email__ = "bryce.ferenczi@monash.edu"

import sys
from pathlib import Path
from typing import Dict

import torch
from scipy.optimize import linear_sum_assignment
from nnet_training.utilities.panoptic_post_processing import find_instance_center

from .base import MetricBase

class PanopticMetric(MetricBase):
    """
    Accuracy and Loss Staticstics tracking for panoptic segmentation networks.\n
    """
    def __init__(self, num_classes, savefile: str, base_dir: Path,
                 main_metric: str, mode='training'):
        super().__init__(
            savefile=savefile, base_dir=base_dir, main_metric=main_metric, mode=mode)
        self._n_classes = num_classes
        self._reset_metric()
        assert self.main_metric in self.metric_data.keys()

    def add_sample(self, predictions: Dict[str, torch.Tensor],
                   targets: Dict[str, torch.Tensor], loss: int=0, **kwargs) -> None:
        assert all(key in predictions for key in ['center', 'offset', 'seg']), \
            "Missing key in predictions required for panoptic caluclations."
        assert all(key in targets for key in ['center', 'offset', 'seg', 'center_points']), \
            "Missing key in targets required for panoptic caluclations."

        self.metric_data["Batch_Loss"].append(loss)

        # TODO Fix Batch PQ after fixing instance post processing
        self.metric_data["Batch_PQ"].append(0)

        self.metric_data["Batch_Offset_MSE"].append(
            torch.linalg.norm(predictions['offset']-targets['offset'], dim=1).mean().cpu().item()
        )

        total_mse = 0
        batch_size = predictions['center'].shape[0]
        for idx in range(predictions['center'].shape[0]):
            if not targets['center_points'][idx]:
                batch_size -= 1
                continue
            center_preds = find_instance_center(
                predictions['center'][idx], nms_kernel=7).type(torch.float32)
            center_gt = torch.as_tensor(
                targets['center_points'][idx], device=center_preds.device, dtype=torch.float32)
            cost_mat = torch.cdist(center_preds, center_gt, p=2).cpu()
            indicies = linear_sum_assignment(cost_mat)
            total_mse += cost_mat[indicies].mean()

        if batch_size > 0:
            self.metric_data["Batch_Center_MSE"].append(total_mse.item() / batch_size)

    def max_accuracy(self, main_metric=True):
        """
        Returns highest centeroid mse, offset map mse, panoptic quality
        from per epoch summarised data.\n
        @param  main_metric, if true only returns scale invariant\n
        @output if main metric, will return its name and value
        @output loss, centeroid mse, offset map mse, panoptic quality
        """
        cost_func = {
            'Batch_Loss': [min, sys.float_info.max],
            'Batch_Center_MSE': [min, sys.float_info.max],
            'Batch_Offset_MSE': [min, sys.float_info.max],
            'Batch_PQ': [max, sys.float_info.min],
        }

        if self._path is not None:
            ret_type = self.main_metric if main_metric else None
            return self._max_accuracy_lambda(cost_func, self._path, ret_type)

        print("No File Specified for Panoptic Statistics Manager")
        return None

    def _reset_metric(self):
        self.metric_data = dict(
            Batch_Loss=[],
            Batch_Center_MSE=[],
            Batch_Offset_MSE=[],
            Batch_PQ=[]
        )

if __name__ == "__main__":
    pass
