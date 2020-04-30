#!/usr/bin/env python3

__author__ = "Bryce Ferenczi"
__email__ = "bryce.ferenczi@monashmotorsport.com"

import threading
import numpy as np

__all__ = ['SegmentationMetric', 'BoundaryBoxMetric', 'ClassificationMetric']

class SegmentationMetric(object):
    def __init__(self, num_classes):
        self._n_classes = num_classes
        self.reset()

    def add_sample(self, preds, labels):
        """
        Update Accuracy Metrics
        """
        pxthread = threading.Thread(target=self._pixelwise, args=(preds, labels))
        iouthread = threading.Thread(target=self._iou, args=(preds, labels))
        pxthread.start()
        iouthread.start()
        pxthread.join()
        iouthread.join()

    def get_statistics(self):
        """
        Returns Accuracy Metrics [pixelwise, mIoU]
        """
        # np.true_divide(area_intersection, area_union, out=np.zeros_like(area_intersection, dtype=float), where=area_union!=0)
        pixAcc = 1.0 * self._sum_correct / (np.spacing(1) + self._sum_labeled)
        IoU = 1.0 * self._sum_inter / (np.spacing(1) + self._sum_union)
        # It has same result with np.nanmean() when all class exist
        mIoU = IoU.mean()
        return pixAcc, mIoU

    def _iou(self, prediction, target):
        # Remove classes from unlabeled pixels in gt image.
        # We should not penalize detections in unlabeled portions of the image.
        target = target.astype('int64') + 1
        prediction = prediction.astype('int64') + 1
        prediction = prediction * (target > 0).astype(prediction.dtype)

        # Compute area intersection:
        intersection = prediction * (prediction == target)
        area_intersection, _ = np.histogram(intersection, bins=self._n_classes, range=(1, self._n_classes))

        # Compute area union:
        area_pred, _ = np.histogram(prediction, bins=self._n_classes, range=(1, self._n_classes))
        area_lab, _ = np.histogram(target, bins=self._n_classes, range=(1, self._n_classes))
        area_union = area_pred + area_lab - area_intersection
        
        self._sum_inter += area_intersection
        self._sum_union += area_union
    
    def _pixelwise(self, prediction, target):
        # Remove classes from unlabeled pixels in gt image.
        # We should not penalize detections in unlabeled portions of the image.
        prediction = prediction.astype('int64') + 1
        target = target.astype('int64') + 1

        self._sum_labeled += np.sum(target >= 0)
        self._sum_correct += np.sum((prediction == target) * (target >= 0))
    
    def reset(self):
        self._sum_inter = 0
        self._sum_union = 0
        self._sum_correct = 0
        self._sum_labeled = 0

class BoundaryBoxMetric(object):
    def __init__(self):
        self.yes = True
        # @todo

class ClassificationMetric(object):
    def __init__(self):
        self.yes = True
        # @todo
