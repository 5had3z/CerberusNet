#!/usr/bin/env python3

__author__ = "Bryce Ferenczi"
__email__ = "bryce.ferenczi@monash.edu"

from pathlib import Path
from typing import Dict

import torch

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
        raise NotImplementedError

    def max_accuracy(self, main_metric=True):
        raise NotImplementedError

    def _reset_metric(self):
        raise NotImplementedError

if __name__ == "__main__":
    pass
