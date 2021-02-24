#!/usr/bin/env python3

__author__ = "Bryce Ferenczi"
__email__ = "bryce.ferenczi@monash.edu"

import sys
from pathlib import Path
from typing import Dict
from typing import overload

import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from scipy import stats
from cityscapesscripts.helpers.labels import trainId2label
import torch

from .base import MetricBase

class SegmentationMetric(MetricBase):
    """
    Accuracy and Loss Staticstics tracking for semantic segmentation networks.\n
    Tracks pixel wise accuracy (PixelAcc) and intersection over union (IoU).
    """
    def __init__(self, num_classes: int, **kwargs):
        super().__init__(savefile="seg_data", **kwargs)
        self._n_classes = num_classes
        self._reset_metric()
        assert self.main_metric in self.metric_data.keys()

    def add_sample(self, predictions: Dict[str, torch.Tensor],
                   targets: Dict[str, torch.Tensor], loss: int=0, **kwargs) -> None:
        """
        Update Accuracy (and Loss) Metrics
        """
        assert all('seg' in keys for keys in [predictions.keys(), targets.keys()])
        self.metric_data["Batch_Loss"].append(loss)

        labels = targets['seg'].type(torch.int32).cuda()
        preds = torch.argmax(predictions['seg'], dim=1, keepdim=True).type(torch.int32)

        batch_pix_acc = np.zeros((preds.shape[0], 1))
        batch_iou = np.zeros((preds.shape[0], self._n_classes))

        for idx in range(preds.shape[0]):
            conf_mat = self._gen_confusion_mat(preds[idx], labels[idx])
            self.metric_data["Confusion_Mat"] += conf_mat.cpu()
            pix_acc = torch.true_divide(torch.diag(conf_mat).sum(), conf_mat.sum())
            batch_pix_acc[idx, :] = pix_acc.cpu().data.numpy()
            batch_iou[idx, :] = self._confmat_cls_iou(conf_mat)

        self.metric_data["Batch_PixelAcc"].append(batch_pix_acc)

        self.metric_data["Batch_IoU"].append(batch_iou)

    def print_epoch_statistics(self):
        """
        Prints all the statistics
        """
        pixel_acc = torch.true_divide(torch.diag(self.metric_data["Confusion_Mat"]).sum(),
                                      self.metric_data["Confusion_Mat"].sum())
        pixel_acc = pixel_acc.cpu().data.numpy()

        miou = np.nanmean(self._confmat_cls_iou(self.metric_data["Confusion_Mat"]))

        loss = np.asarray(self.metric_data["Batch_Loss"]).mean()
        print(f"Pixel Accuracy: {pixel_acc:.4f}\tmIoU: {miou:.4f}\tLoss: {loss:.4f}")

    def get_current_statistics(self, main_only=True, return_loss=True):
        """
        Returns either all statistics or only main statistic\n
        @todo   get a specified epoch instead of only currently loaded one\n
        @param  main_metric, returns mIoU and not Pixel Accuracy\n
        @param  loss_metric, returns recorded loss\n
        """
        ret_mean = ()
        ret_var = ()
        if main_only:
            if self.main_metric == 'Batch_IoU':
                ret_mean += (np.nanmean(self._confmat_cls_iou(self.metric_data["Confusion_Mat"])),)
                data = np.asarray(self.metric_data['Batch_IoU']).reshape(-1, self._n_classes)
                ret_var += (np.nanvar(data, axis=1).mean(),)
            else:
                data = np.asarray(self.metric_data[self.main_metric]).flatten()
                ret_mean += (data.mean(),)
                ret_var += (data.var(ddof=1),)
            if return_loss:
                ret_mean += (np.asarray(self.metric_data["Batch_Loss"]).mean(),)
                ret_var += (np.asarray(self.metric_data["Batch_Loss"]).var(ddof=1),)
        else:
            for key, data in sorted(self.metric_data.items(), key=lambda x: x[0]):
                if key == 'Batch_IoU':
                    ret_mean += (np.nanmean(
                        self._confmat_cls_iou(self.metric_data["Confusion_Mat"])),)
                    data = np.asarray(data).reshape(-1, self._n_classes)
                    ret_var += (np.nanvar(data, axis=1).mean(),)
                elif (key != 'Batch_Loss' or return_loss) and key.startswith('Batch'):
                    data = np.asarray(data).flatten()
                    ret_mean += (data.mean(),)
                    ret_var += (data.var(ddof=1),)

        return ret_mean, ret_var

    def max_accuracy(self, main_metric=True):
        """
        Returns highest mIoU and Pixel-Wise Accuracy from per epoch summarised data.\n
        @param  main_metric, if true only returns the main metric specified\n
        @output Pixel-Wise Accuracy, mIoU
        """
        cost_func = {
            "Batch_IoU" : [max, 0.0],
            "Batch_PixelAcc" : [max, 0.0],
            "Batch_Loss" : [min, sys.float_info.max]
        }

        if self._path is not None:
            ret_type = self.main_metric if main_metric else None
            return self._max_accuracy_lambda(cost_func, self._path, ret_type)

        print("No File Specified for Segmentation Metric Manager")
        return None

    @overload
    @staticmethod
    def _confmat_cls_iou(conf_mat: torch.Tensor) -> np.ndarray:
        ...

    @overload
    @staticmethod
    def _confmat_cls_iou(conf_mat: np.ndarray) -> np.ndarray:
        ...

    @staticmethod
    def _confmat_cls_iou(conf_mat):
        if isinstance(conf_mat, torch.Tensor):
            divisor = conf_mat.sum(dim=1) + conf_mat.sum(dim=0) - torch.diag(conf_mat)
            return torch.true_divide(torch.diag(conf_mat), divisor).cpu().data.numpy()
        if isinstance(conf_mat, np.ndarray):
            divisor = conf_mat.sum(axis=1) + conf_mat.sum(axis=0) - np.diag(conf_mat)
            return np.true_divide(np.diag(conf_mat), divisor)
        raise NotImplementedError(type(conf_mat))

    @staticmethod
    def _confmat_cls_pr_rc(conf_mat: np.ndarray) -> np.ndarray:
        """
        Returns tuple of classwise average precision and recall
        The below calculations are made, but simplified in the actual code:
        cls_tp = np.diag(conf_mat)
        cls_fp = np.sum(conf_mat, axis=0) - cls_tp
        cls_fn = np.sum(conf_mat, axis=1) - cls_tp

        cls_precision = cls_tp / (cls_tp + cls_fp)
        cls_recall = cls_tp / (cls_tp + cls_fn)
        """
        cls_precision = np.diag(conf_mat) / np.sum(conf_mat, axis=0)
        cls_recall = np.diag(conf_mat) / np.sum(conf_mat, axis=1)
        return cls_precision, cls_recall

    def _basic_iou(self, prediction: torch.Tensor, target: torch.Tensor):
        # Remove classes from unlabeled pixels in gt image.
        # We should not penalize detections in unlabeled portions of the image.
        prediction = prediction * (target != 255)

        # Compute area intersection:
        intersection = prediction * (prediction == target)
        area_intersection = torch.histc(intersection, bins=self._n_classes,
                                        min=1, max=self._n_classes)

        # Compute area union:
        area_pred = torch.histc(prediction, bins=self._n_classes, min=1, max=self._n_classes)
        area_lab = torch.histc(target, bins=self._n_classes, min=1, max=self._n_classes)
        area_union = area_pred + area_lab - area_intersection

        iou = 1.0 * area_intersection / (np.spacing(1) + area_union)

        return iou.cpu().data.numpy()

    @staticmethod
    def _basic_pixelwise(prediction: torch.Tensor, target: torch.Tensor):
        # Remove classes from unlabeled pixels in gt image.
        # We should not penalize detections in unlabeled portions of the image.
        correct = 1.0 * ((prediction == target) * (target != 255)).sum()
        total_pixels = np.spacing(1) + (target != 255).sum()
        pix_acc = correct / total_pixels

        return pix_acc.cpu().data.numpy()

    def _gen_confusion_mat(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mask = target != 255
        conf_mat = torch.bincount(self._n_classes * target[mask] + prediction[mask],
                                  minlength=self._n_classes**2)
        return conf_mat.reshape(self._n_classes, self._n_classes)

    def get_last_batch(self, main_metric=True):
        """
        Return the data from the last batch
        """
        if main_metric:
            if self.main_metric == 'Batch_IoU':
                return np.nanmean(self.metric_data[self.main_metric][-1])
            return self.metric_data[self.main_metric][-1]

        ret_val = ()
        for key, data in sorted(self.metric_data.items(), key=lambda x: x[0]):
            if key not in ["Batch_Loss", 'Batch_IoU']:
                ret_val += (data[-1],)
            elif key == 'Batch_IoU':
                ret_val += (np.nanmean(data[-1]),)
        return ret_val

    def _reset_metric(self):
        self.metric_data = {
            'Batch_Loss': [],
            'Batch_PixelAcc': [],
            'Batch_IoU': [],
            'Confusion_Mat': torch.zeros((self._n_classes, self._n_classes), dtype=torch.int32)
        }

    def plot_classwise_iou(self):
        """
        This plots the iou of each class over all epochs
        """
        fig, axis = plt.subplots(3, 19//3+1, figsize=(18, 5))
        fig.suptitle(self._path.name + ' Summary Training and Validation Results')

        for dataset in ['training', 'validation']:
            with h5py.File(self._path, 'r') as hfile:
                data_mean = np.zeros((len(list(hfile[dataset])), self._n_classes))
                data_conf = np.zeros((len(list(hfile[dataset])), self._n_classes))

                sort_lmbda = lambda x: int(x[6:])
                for idx, epoch in enumerate(sorted(list(hfile[dataset]), key=sort_lmbda)):
                    epoch_data = hfile[f'{dataset}/{epoch}/Batch_IoU'][:]
                    n_samples = np.count_nonzero(~np.isnan(epoch_data), axis=0)
                    data_mean[idx] = np.nanmean(epoch_data, axis=0)
                    data_conf[idx] = stats.t.ppf(0.95, n_samples-1) * \
                        np.nanvar(epoch_data, axis=0, ddof=1) / np.sqrt(n_samples)

            for idx in range(self._n_classes):
                axis[idx%3][idx//3].plot(data_mean[:, idx], label=dataset)
                axis[idx%3][idx//3].fill_between(
                    np.arange(0, data_mean[:, idx].shape[0]),
                    data_mean[:, idx] - data_conf[:, idx],
                    data_mean[:, idx] + data_conf[:, idx],
                    alpha=0.2)

        for idx in range(self._n_classes):
            axis[idx%3][idx//3].set_title(f'{trainId2label[idx].name} over Epochs')

        plt.show(block=False)

    def conf_summary_data(self):
        """
        Generates summary data that can be infered from the confusion matrix.
        """
        test = {}
        train = {}

        with h5py.File(self._path, 'r') as hfile:
            # assert len(list(hfile['training'])) == len(list(hfile['validation']))
            num_epochs = len(list(hfile['validation']))
            training_data = np.zeros((num_epochs, self._n_classes, self._n_classes))
            testing_data = np.zeros((num_epochs, self._n_classes, self._n_classes))

            sort_lmbda = lambda x: int(x[6:])
            for idx, epoch in enumerate(sorted(list(hfile['training']), key=sort_lmbda)):
                training_data[idx] = hfile[f'training/{epoch}/Confusion_Mat'][:]

            for idx, epoch in enumerate(sorted(list(hfile['validation']), key=sort_lmbda)):
                testing_data[idx] = hfile[f'validation/{epoch}/Confusion_Mat'][:]

        test['iou'] = np.zeros((num_epochs, self._n_classes))
        train['iou'] = np.zeros((num_epochs, self._n_classes))

        test['precision'] = np.zeros((num_epochs, self._n_classes))
        train['precision'] = np.zeros((num_epochs, self._n_classes))

        test['recall'] = np.zeros((num_epochs, self._n_classes))
        train['recall'] = np.zeros((num_epochs, self._n_classes))

        for idx in range(num_epochs):
            test['iou'][idx] = self._confmat_cls_iou(testing_data[idx, :, :])
            train['iou'][idx] = self._confmat_cls_iou(training_data[idx, :, :])

            test['precision'][idx], test['recall'][idx] = \
                self._confmat_cls_pr_rc(testing_data[idx, :, :])

            train['precision'][idx], train['recall'][idx] = \
                self._confmat_cls_pr_rc(training_data[idx, :, :])

        return test, train

    def display_conf_mat(self, index=-1, dataset='validation'):
        """
        Plots the confusion matrix of an epoch.\n
        @param index: the index of the epoch you want to plot\n
        @param dataset: either training or validation data.
        """
        assert dataset in ['validation', 'training']
        with h5py.File(self._path, 'r') as hfile:
            n_epochs = len(list(hfile[dataset]))
            assert index < n_epochs, f"Index {index} exceeds number of epochs {n_epochs}"
            epoch_idx = index if index > 0 else n_epochs
            epoch_data = hfile[dataset][f'Epoch_{epoch_idx}/Confusion_Mat'][:]

        plt.figure(figsize=(18, 5))
        plt.suptitle("Class Confusion Matrix")
        labels = [trainId2label[i].name for i in range(19)]
        normalised_data = epoch_data / np.sum(epoch_data, axis=1, keepdims=True)
        conf_mat = pd.DataFrame(normalised_data, labels, labels)
        sn.set(font_scale=1)
        sn.heatmap(conf_mat, annot=True, annot_kws={"size":8})

        plt.show(block=False)

    def plot_classwise_data(self):
        """
        Plots the summary data from the confusion matrix.
        """
        test, train = self.conf_summary_data()
        plt.figure(figsize=(18, 5))
        plt.suptitle(f'{self._path.name} IoU')

        for idx in range(self._n_classes):
            plt.subplot(3, self._n_classes//3+1, idx+1)
            plt.plot(train['iou'][:, idx])
            plt.plot(test['iou'][:, idx])
            plt.legend(["Training", "Validation"])
            plt.title(f'{trainId2label[idx].name}')
            plt.xlabel('Epoch #')

        plt.tight_layout()
        plt.show(block=False)

        plt.figure(figsize=(18, 5))
        plt.suptitle(f'{self._path.name} Precision')

        for idx in range(self._n_classes):
            plt.subplot(3, self._n_classes//3+1, idx+1)
            plt.plot(train['precision'][:, idx])
            plt.plot(test['precision'][:, idx])
            plt.legend(["Training", "Validation"])
            plt.title(f'{trainId2label[idx].name}')
            plt.xlabel('Epoch #')

        plt.tight_layout()
        plt.show(block=False)

        plt.figure(figsize=(18, 5))
        plt.suptitle(f'{self._path.name} Recall')

        for idx in range(self._n_classes):
            plt.subplot(3, self._n_classes//3+1, idx+1)
            plt.plot(train['recall'][:, idx])
            plt.plot(test['recall'][:, idx])
            plt.legend(["Training", "Validation"])
            plt.title(f'{trainId2label[idx].name}')
            plt.xlabel('Epoch #')

        plt.tight_layout()
        plt.show(block=False)

if __name__ == "__main__":
    pass
