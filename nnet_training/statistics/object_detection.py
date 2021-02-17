#!/usr/bin/env python3

__author__ = "Bryce Ferenczi"
__email__ = "bryce.ferenczi@monash.edu"

import sys
from pathlib import Path
from typing import Dict
from typing import Tuple
from typing import List

import torch
import numpy as np
from torchvision.ops.boxes import box_iou

from nnet_training.nnet_models.detr.matcher import HungarianMatcher
from nnet_training.nnet_models.detr.box_ops import box_cxcywh_to_xyxy

from .base import MetricBase

class BoundaryBoxMetric(MetricBase):
    """
    Accuracy/Error and Loss Staticstics tracking for nnets with boundary box output
    """
    def __init__(self, n_classes: int, savefile: str, base_dir: Path,
                 main_metric: str, mode='training'):
        super().__init__(
            savefile=savefile, base_dir=base_dir, main_metric=main_metric, mode=mode)
        self._n_classes = n_classes
        self._reset_metric()
        assert self.main_metric in self.metric_data.keys()
        self.matcher = HungarianMatcher()

    @staticmethod
    def _get_src_permutation_idx(indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    @staticmethod
    def _calculate_iou(predictions: Dict[str, torch.Tensor], targets: torch.Tensor,
                       indices: List[Tuple[torch.Tensor, torch.Tensor]]) -> np.ndarray:
        """
        Calulate the iou between predicted and ground truth boxes given an
        already matched set of indices.
        """
        idx = BoundaryBoxMetric._get_src_permutation_idx(indices)
        src_boxes = predictions[idx]
        target_boxes = torch.cat([t[i] for t, (_, i) in zip(targets, indices)], dim=0)

        return torch.diag(box_iou(
            box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes)))

    def add_sample(self, predictions: Dict[str, torch.Tensor],
                   targets: Dict[str, torch.Tensor], loss: int=0, **kwargs) -> None:
        assert all(key in predictions.keys() for key in ['logits', 'bboxes'])
        assert all(key in targets.keys() for key in ['labels', 'bboxes'])

        self.metric_data["Batch_Loss"].append(loss)

        #   Calculate IoU for each Box
        detr_outputs = {k: v for k, v in predictions.items() if k in ['logits', 'bboxes']}
        indices = self.matcher(detr_outputs, targets)

        conf_mat = torch.zeros((self._n_classes, self._n_classes), dtype=torch.int32).cuda()
        for idx, (src_idx, tgt_idx) in enumerate(indices):
            if tgt_idx.shape[0] > 0 and src_idx.shape[0] > 0:
                gt_lab = targets['labels'][idx][tgt_idx]
                pred_lab = predictions['logits'][idx][src_idx].argmax(dim=1)
                conf_mat += torch.bincount(
                    self._n_classes * gt_lab + pred_lab, minlength=self._n_classes**2
                    ).reshape(self._n_classes, self._n_classes)

        self.metric_data['Confusion_Mat'] += conf_mat.cpu()
        precision, recall = self._confmat_cls_pr_rc(conf_mat.cpu().numpy())
        self.metric_data['Batch_Precision'].append(precision)
        self.metric_data['Batch_Recall'].append(recall)

        iou_vector = self._calculate_iou(detr_outputs['bboxes'], targets['bboxes'], indices)
        target_idx = torch.cat([t[J] for t, (_, J) in zip(targets["labels"], indices)])
        iou_class = torch.zeros(self._n_classes, dtype=torch.float32).cuda()
        for cls_id in torch.unique(target_idx):
            iou_class[cls_id] = iou_vector[target_idx == cls_id].mean()

        self.metric_data['Batch_IoU'].append(iou_class.cpu().numpy())

    def get_current_statistics(self, main_only=True, return_loss=True):
        """
        Returns either all statistics or only main statistic\n
        @todo   get a specified epoch instead of only currently loaded one\n
        @param  main_metric, returns mIoU and not Pixel Accuracy\n
        @param  loss_metric, returns recorded loss\n
        """
        ret_mean = ()
        ret_var = ()
        precision, recall = self._confmat_cls_pr_rc(self.metric_data["Confusion_Mat"].numpy())
        if main_only:
            if self.main_metric == 'Batch_Precision':
                ret_mean += (np.nanmean(precision),)
                ret_var += (np.nanvar(
                    np.asarray(self.metric_data['Batch_Precision']).reshape(-1, self._n_classes),
                    axis=1).mean(),)
            elif self.main_metric == 'Batch_Recall':
                ret_mean += (np.nanmean(recall),)
                ret_var += (np.nanvar(
                    np.asarray(self.metric_data['Batch_Recall']).reshape(-1, self._n_classes),
                    axis=1).mean(),)
            else:
                data = np.asarray(self.metric_data[self.main_metric]).flatten()
                ret_mean += (data.mean(),)
                ret_var += (data.var(ddof=1),)
            if return_loss:
                ret_mean += (np.asarray(self.metric_data["Batch_Loss"]).mean(),)
                ret_var += (np.asarray(self.metric_data["Batch_Loss"]).var(ddof=1),)
        else:
            for key, data in sorted(self.metric_data.items(), key=lambda x: x[0]):
                if key == 'Batch_Precision':
                    ret_mean += (np.nanmean(precision),)
                    data = np.asarray(data).reshape(-1, self._n_classes)
                    ret_var += (np.nanvar(data, axis=1).mean(),)
                elif key == 'Batch_Recall':
                    ret_mean += (np.nanmean(recall),)
                    data = np.asarray(data).reshape(-1, self._n_classes)
                    ret_var += (np.nanvar(data, axis=1).mean(),)
                elif (key != 'Batch_Loss' or return_loss) and key.startswith('Batch'):
                    data = np.asarray(data).flatten()
                    ret_mean += (data.mean(),)
                    ret_var += (data.var(ddof=1),)

        return ret_mean, ret_var

    def max_accuracy(self, main_metric=True):
        """
        Returns highest average precision.\n
        @param  main_metric, if true only returns the main statistic\n
        @output average precision, mIoU
        """
        cost_func = {
            'Batch_Loss': [min, sys.float_info.max],
            'Batch_IoU': [max, sys.float_info.min],
            'Batch_Recall': [max, sys.float_info.min],
            'Batch_Precision': [max, sys.float_info.min]
        }

        if self._path is not None:
            ret_type = self.main_metric if main_metric else None
            return self._max_accuracy_lambda(cost_func, self._path, ret_type)

        print("No File Specified for Segmentation Metric Manager")
        return None

    def _reset_metric(self):
        self.metric_data = dict(
            Batch_Loss=[],
            Batch_IoU=[],
            Batch_Recall=[],
            Batch_Precision=[],
            Confusion_Mat=torch.zeros((self._n_classes, self._n_classes), dtype=torch.int32)
        )

class ClassificationMetric(MetricBase):
    """
    Accuracy/Error and Loss Staticstics tracking for classification problems
    """
    def __init__(self, savefile: str, base_dir: Path, main_metric: str, mode='training'):
        super().__init__(savefile=savefile, base_dir=base_dir,
                                                   main_metric=main_metric, mode=mode)
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
