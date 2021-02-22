#!/usr/bin/env python3

__author__ = "Bryce Ferenczi"
__email__ = "bryce.ferenczi@monash.edu"

import sys
from typing import Dict

import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from nnet_training.utilities.panoptic_post_processing import find_instance_center

from nnet_training.datasets.cityscapes_dataset import CityScapesDataset
from nnet_training.utilities.panoptic_post_processing import get_instance_segmentation
from .base import MetricBase

class PanopticMetric(MetricBase):
    """
    Accuracy and Loss Staticstics tracking for panoptic segmentation networks.\n
    """
    def __init__(self, num_classes, **kwargs):
        super().__init__(savefile="panoptic_data", **kwargs)
        self._n_classes = num_classes
        self._reset_metric()
        assert self.main_metric in self.metric_data.keys()

    @staticmethod
    def calculate_pq(pred_inst: np.ndarray, pred_seg: np.ndarray,
                     target_inst: np.ndarray, target_seg: np.ndarray) -> float:
        """
        Caclulates and returns Panoptic Quality statistic for prediction and target pair.
        """
        pred_inst_ids, pred_areas = np.unique(pred_inst, return_counts=True)
        tgt_inst_ids, tgt_areas = np.unique(target_inst, return_counts=True)

        if pred_inst_ids.size == 1 and tgt_inst_ids.size == 1:
            return np.nan # returns nan if no predictions or targets (only 0 label)

        correlated_instances = set()
        iou_sum = 0
        true_positives = 0
        for pred_id, pred_area in zip(pred_inst_ids, pred_areas):
            if not pred_id:
                continue # skip zero
            intersection_ids, intersection_area = \
                np.unique(target_inst[pred_inst == pred_id], return_counts=True)
            for intersect_id, intersect_area in zip(intersection_ids, intersection_area):
                union_area = pred_area + tgt_areas[np.where(tgt_inst_ids == intersect_id)][0] \
                             - intersect_area
                iou = intersect_area / union_area
                if iou > 0.5:
                    true_positives += 1
                    iou_sum += iou
                    correlated_instances.add((pred_id, intersect_id))

        corr_preds = set(inst[0] for inst in correlated_instances)
        false_positives = len(set(pred_inst_ids) - corr_preds) - 1 # remove zero

        corr_tgts = set(inst[1] for inst in correlated_instances)
        false_negatives = len(set(tgt_inst_ids) - corr_tgts) - 1 # remove zero

        # Add to false positives and take away from true positives if segmentation
        # prediction and target do not match. Choose the majority class of each instance.
        for pred_id, tgt_id in correlated_instances:
            pred_cls, pred_counts = np.unique(pred_seg[pred_inst == pred_id], return_counts=True)
            tgt_cls, tgt_counts = np.unique(target_seg[target_inst == tgt_id], return_counts=True)
            if pred_cls[np.argmax(pred_counts)] != tgt_cls[np.argmax(tgt_counts)]:
                true_positives -= 1
                false_positives += 1

        return iou_sum / (true_positives + 0.5 * false_positives + 0.5 * false_negatives)

    @staticmethod
    def batch_process_pq(predictions: Dict[str, torch.tensor],
                         targets: Dict[str, torch.tensor]) -> np.ndarray:
        """
        Process batched predictions and targets
        """
        batch_size = predictions['center'].shape[0]
        pq_results = []
        seg_pred = torch.argmax(predictions['seg'], dim=1)

        for idx in range(batch_size):
            pred_inst, _ = get_instance_segmentation(
                seg_pred[idx].unsqueeze(0),
                predictions['center'][idx].unsqueeze(0),
                predictions['offset'][idx].unsqueeze(0),
                CityScapesDataset.cityscapes_things, nms_kernel=7)

            tgt_inst, _ = get_instance_segmentation(
                targets['seg'][idx],
                targets['center'][idx].unsqueeze(0),
                targets['offset'][idx].unsqueeze(0),
                CityScapesDataset.cityscapes_things, nms_kernel=7)

            sample_pq = PanopticMetric.calculate_pq(
                pred_inst.cpu().numpy(), seg_pred[idx].unsqueeze(0).cpu().numpy(),
                tgt_inst.cpu().numpy(), targets['seg'][idx].cpu().numpy())

            if not np.isnan(sample_pq):
                pq_results.append(sample_pq)

        return np.asarray(pq_results)

    def add_sample(self, predictions: Dict[str, torch.Tensor],
                   targets: Dict[str, torch.Tensor], loss: int=0, **kwargs) -> None:
        assert all(key in predictions for key in ['center', 'offset', 'seg']), \
            "Missing key in predictions required for panoptic caluclations."
        assert all(key in targets for key in ['center', 'offset', 'seg', 'center_points']), \
            "Missing key in targets required for panoptic caluclations."

        self.metric_data["Batch_Loss"].append(loss)

        self.metric_data["Batch_PQ"].append(self.batch_process_pq(predictions, targets))

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
            if not center_preds.shape[0]:
                batch_size -= 1
                continue
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
