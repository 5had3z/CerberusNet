#!/usr/bin/env python3

__author__ = "Bryce Ferenczi"
__email__ = "bryce.ferenczi@monash.edu"

from pathlib import Path
from typing import Dict

import torch
from scipy.optimize import linear_sum_assignment

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

        offset_mse = torch.linalg.norm(predictions['offset'] - targets['offset'], dim=1)
        offset_mse = offset_mse.mean()
        self.metric_data["Batch_Offset_MSE"].append(offset_mse.cpu().numpy())

        # TODO Hungarian matcher for centeroids and find mse
        self.metric_data["Batch_Center_MSE"].append(0)


    def max_accuracy(self, main_metric=True):
        raise NotImplementedError

    def _reset_metric(self):
        self.metric_data = dict(
            Batch_Loss=[],
            Batch_Center_MSE=[],
            Batch_Offset_MSE=[],
            Batch_PQ=[]
        )

if __name__ == "__main__":
    pass
