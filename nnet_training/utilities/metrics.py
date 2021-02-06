#!/usr/bin/env python3

__author__ = "Bryce Ferenczi"
__email__ = "bryce.ferenczi@monash.edu"

import os
import sys
from pathlib import Path
from typing import Dict, List, Callable, Union, Tuple, overload

import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from scipy import stats

import torch
from torchvision.ops.boxes import box_iou

from nnet_training.utilities.cityscapes_labels import trainId2name
from nnet_training.loss_functions.UnFlowLoss import flow_warp
from nnet_training.nnet_models.detr.matcher import HungarianMatcher
from nnet_training.nnet_models.detr.box_ops import generalized_box_iou, box_cxcywh_to_xyxy

__all__ = ['MetricBase', 'SegmentationMetric', 'DepthMetric',
           'BoundaryBoxMetric', 'ClassificationMetric']

class MetricBase():
    """
    Provides basic functionality for statistics tracking classes
    """
    def __init__(self, savefile: str, base_dir: Path, main_metric: str, mode='training'):
        assert mode in ['training', 'validation']
        self.mode = mode
        self.metric_data = {}
        if main_metric[:6] != "Batch_":
            main_metric = "Batch_"+main_metric
        self.main_metric = main_metric

        if savefile != "":
            if savefile[-5:] != '.hdf5':
                savefile = savefile + '.hdf5'
            self._path = base_dir / savefile
            if not os.path.isfile(self._path):
                with h5py.File(self._path, 'a') as hfile:
                    hfile.create_group('cache')
                    print("Training Statitsics created at ", self._path)
            else:
                # Clear any previously cached data
                with h5py.File(self._path, 'a') as hfile:
                    if 'cache' in list(hfile):
                        del hfile['cache']
        else:
            self._path = None

    def __len__(self):
        """
        Return Number of Epochs Recorded
        """
        with h5py.File(self._path, 'r') as hfile:
            n_epochs = 0
            if 'training' in list(hfile):
                n_epochs += len(list(hfile['training']))
            if 'cache' in list(hfile) and 'training' in list(hfile['cache']):
                n_epochs += len(list(hfile['cache/training']))
            return n_epochs

    def save_epoch(self):
        """
        Save Data to new dataset named by epoch name
        """
        if self._path is not None:
            with h5py.File(self._path, 'a') as hfile:
                # Clear any Cached data from previous first into main
                if 'cache' in list(hfile) and len(list(hfile['cache'])) > 0:
                    self._flush_to_main()

                if self.mode in list(hfile):
                    group_name = self.mode + '/Epoch_' + str(len(list(hfile[self.mode])) + 1)
                else:
                    group_name = self.mode + '/Epoch_1'

                top_group = hfile.create_group(group_name)
                for name, data in self.metric_data.items():
                    data = np.asarray(data)
                    if name[:6] == "Batch_" and len(data.shape) > 1:
                        data = data.reshape(data.shape[0] * data.shape[1], -1)
                    top_group.create_dataset(name, data=data)

                mean, variance = np.asarray(self.get_current_statistics(main_metric=False))
                top_group.create_dataset('Summary_Mean', data=mean)
                top_group.create_dataset('Summary_Variance', data=variance)

                # Flush current data as its now in long term
                # storage and we're ready for next dataset
                self._reset_metric()

        else:
            print("No File Specified for Segmentation Metric Manager")

    def load_statistics(self, epoch_idx, mode=None):
        """
        Load epoch statistics into metric manager and return statistics
        """
        if mode is None:
            mode = self.mode

        if self._path is not None:
            with h5py.File(self._path, 'r') as hfile:
                group_name = 'Epoch_' + str(epoch_idx)
                for metric in list(hfile[mode][group_name]):
                    self.metric_data[metric] = hfile[mode][group_name][metric][:]
            return self.metric_data

        print("No File Specified for Segmentation Metric Manager")
        return None

    def new_epoch(self, mode='training'):
        """
        Caches any data that hasn't been saved and resets metrics
        """
        assert mode in ['training', 'validation']

        # Cache data if it hasn't been saved yet (probably not a best epoch or
        # something, but the next might be, so we want to keep this data)
        if all(len(metric) > 0 for metric in self.metric_data.values()):
            self._cache_data()

        self.mode = mode
        self._reset_metric()

    def plot_epoch_data(self, epoch_idx):
        """
        This plots all the statistics for an epoch
        """
        plt.figure(figsize=(18, 5))
        plt.suptitle(self._path.name + ' Training and Validation Results Epoch' + str(epoch_idx))

        group_name = 'Epoch_' + str(epoch_idx)
        with h5py.File(self._path, 'r') as hfile:
            num_metrics = len(list(hfile['training'][group_name]))
            for idx, metric in enumerate(list(hfile['training'][group_name])):
                plt.subplot(1, num_metrics, idx+1)
                plt.plot(hfile['training'][group_name][metric][:])
                plt.plot(hfile['validation'][group_name][metric][:])
                plt.legend(["Training", "Validation"])
                plt.title('Batch ' + str(metric) + ' over Epochs')
                plt.ylabel(str(metric))
                plt.xlabel('Iter #')

            plt.show()

    def _cache_data(self):
        """
        Moves data to temporary location to be permanently saved or deleted later
        """
        if self._path is not None:
            with h5py.File(self._path, 'a') as hfile:
                n_cached = 1
                if 'cache' in list(hfile) and self.mode in list(hfile['cache']):
                    n_cached = len(list(hfile['cache/'+self.mode])) + 1

                if self.mode in list(hfile):
                    group_name = 'cache/' + self.mode + '/Epoch_' +\
                        str(len(list(hfile[self.mode])) + n_cached)
                else:
                    group_name = 'cache/' + self.mode + '/Epoch_' + str(n_cached)

                top_group = hfile.create_group(group_name)
                for name, data in self.metric_data.items():
                    data = np.asarray(data)
                    if name[:6] == "Batch_" and len(data.shape) > 1:
                        data = data.reshape(data.shape[0] * data.shape[1], -1)
                    top_group.create_dataset(name, data=data)

                mean, variance = np.asarray(self.get_current_statistics(main_metric=False))
                top_group.create_dataset('Summary_Mean', data=mean)
                top_group.create_dataset('Summary_Variance', data=variance)

        else:
            print("No File Specified for Metric Manager")

    def _flush_to_main(self):
        """
        Moves data from cache to main storage area
        """
        with h5py.File(self._path, 'a') as hfile:
            for mode in list(hfile['cache']):
                for epoch in list(hfile['cache/'+mode]):
                    hfile.copy('cache/'+mode+'/'+epoch, mode+'/'+epoch)
                    del hfile['cache/'+mode+'/'+epoch]

    def get_summary_data(self):
        """
        Returns dictionary with testing and validation statistics
        """
        with h5py.File(self._path, 'r') as hfile:
            metrics = []
            for metric in list(hfile['training/Epoch_1']):
                if metric[:5] == 'Batch':
                    metrics.append(metric)

            training_mean = np.zeros((len(list(hfile['training'])), len(metrics)))
            testing_mean = np.zeros((len(list(hfile['validation'])), len(metrics)))
            training_var = np.zeros((len(list(hfile['training'])), len(metrics)))
            testing_var = np.zeros((len(list(hfile['validation'])), len(metrics)))

            sort_func = lambda x: int(x[6:])
            for idx, epoch in enumerate(sorted(list(hfile['training']), key=sort_func)):
                if 'Summary' in list(hfile['training/'+epoch]):
                    training_mean[idx] = hfile['training/'+epoch+'/Summary'][:]
                else:
                    training_mean[idx] = hfile['training/'+epoch+'/Summary_Mean'][:]
                    training_var[idx] = hfile['training/'+epoch+'/Summary_Variance'][:]

            for idx, epoch in enumerate(sorted(list(hfile['validation']), key=sort_func)):
                if 'Summary' in list(hfile['validation/'+epoch]):
                    testing_mean[idx] = hfile['validation/'+epoch+'/Summary'][:]
                else:
                    testing_mean[idx] = hfile['validation/'+epoch+'/Summary_Mean'][:]
                    testing_var[idx] = hfile['validation/'+epoch+'/Summary_Variance'][:]

        ret_val = {}
        for idx, metric in enumerate(metrics):
            ret_val[metric] = {
                "Training_Mean" : training_mean[:, idx],
                "Validation_Mean" : testing_mean[:, idx],
                "Training_Variance" : training_var[:, idx],
                "Validation_Variance" : testing_var[:, idx],
            }

        return ret_val

    def get_n_samples(self, dataset='validation'):
        """
        Returns the number of samples in the validation set
        """
        assert dataset in ['validation', 'training']

        with h5py.File(self._path, 'r') as hfile:
            for metric in list(hfile[f'{dataset}/Epoch_1']):
                if metric[:5] == 'Batch':
                    return hfile[f'{dataset}/Epoch_1/{metric}'][:].shape[0]

        return 0

    def plot_summary_data(self):
        """
        This plots all the summary statistics over all epochs
        """
        plt.figure(figsize=(18, 5))
        plt.suptitle(self._path.name + ' Summary Training and Validation Results')

        n_training = self.get_n_samples('training')
        n_validation = self.get_n_samples('validation')
        summary_data = self.get_summary_data()

        for idx, metric in enumerate(summary_data):
            plt.subplot(1, len(summary_data), idx+1)

            data_mean = summary_data[metric]["Training_Mean"]
            data_conf = stats.t.ppf(0.95, n_training-1) * \
                summary_data[metric]["Training_Variance"] / np.sqrt(n_training)
            plt.plot(data_mean)
            plt.fill_between(
                np.arange(0, data_mean.shape[0]),
                data_mean - data_conf, data_mean + data_conf,
                alpha=0.2)

            data_mean = summary_data[metric]["Validation_Mean"]
            data_conf = stats.t.ppf(0.95, n_validation-1) * \
                summary_data[metric]["Validation_Variance"] / np.sqrt(n_validation)
            plt.plot(data_mean)
            plt.fill_between(
                np.arange(0, data_mean.shape[0]),
                data_mean - data_conf, data_mean + data_conf,
                alpha=0.2)

            plt.legend(["Training", "Validation"])
            plt.title(f'{metric.replace("Batch_", "")} over Epochs')
            plt.xlabel('Epoch #')

        plt.tight_layout()
        plt.show(block=False)

    def plot_iteration_data(self):
        """
        This plots all the statistics over all non-cached iterations
        """
        plt.figure(figsize=(18, 5))
        plt.suptitle(self._path.name + ' Iteration Training and Validation Results')

        with h5py.File(self._path, 'r') as hfile:
            training_metrics = {}
            validation_metrics = {}
            for metric in list(hfile['training/Epoch_1']):
                if metric[:5] == 'Batch':
                    training_metrics[metric] = np.zeros((1, 1))
                    validation_metrics[metric] = np.zeros((1, 1))

            for epoch in list(hfile['training']):
                for metric in list(hfile['training/'+epoch]):
                    if metric[:5] == 'Batch':
                        training_metrics[metric] = np.append(
                            training_metrics[metric],
                            hfile['training/'+epoch+'/'+metric][:])

            for epoch in list(hfile['validation']):
                for metric in list(hfile['validation/'+epoch]):
                    if metric[:5] == 'Batch':
                        validation_metrics[metric] = np.append(
                            validation_metrics[metric],
                            hfile['validation/'+epoch+'/'+metric][:])

            print("# Training, ", len(list(hfile['training'])),
                  "\t# Validation", len(list(hfile['validation'])))

        num_metrics = len(training_metrics.keys())
        for idx, metric in enumerate(training_metrics):
            plt.subplot(1, num_metrics, idx+1)
            plt.plot(training_metrics[metric])
            plt.plot(validation_metrics[metric])
            plt.legend(["Training", "Validation"])
            plt.title(metric + ' over Iterations')
            plt.xlabel('Iteration #')

        plt.show()

    def _reset_metric(self):
        raise NotImplementedError

    def get_current_statistics(self, main_metric=True, loss_metric=True):
        """
        Returns Accuracy and Loss Metrics from an Epoch\n
        @todo   get a specified epoch instead of only currently loaded one\n
        @param  main_metric, only main metric\n
        @param  loss_metric, returns recorded loss\n
        """
        ret_mean = ()
        ret_var = ()
        if main_metric:
            data = np.asarray(self.metric_data[self.main_metric]).flatten()
            ret_mean += (data.mean(),)
            ret_var += (data.var(ddof=1),)
            if loss_metric:
                ret_mean += (np.asarray(self.metric_data["Batch_Loss"]).mean(),)
                ret_var += (np.asarray(self.metric_data["Batch_Loss"]).var(ddof=1),)

        else:
            for key, data in sorted(self.metric_data.items(), key=lambda x: x[0]):
                if key != "Batch_Loss" or loss_metric:
                    data = np.asarray(data).flatten()
                    ret_mean += (data.mean(),)
                    ret_var += (data.var(ddof=1),)

        return ret_mean, ret_var

    def max_accuracy(self, main_metric=True):
        """
        Return the max accuracy from all epochs
        """
        raise NotImplementedError

    def add_sample(self, predictions: Dict[str, torch.Tensor],
                   targets: Dict[str, torch.Tensor], loss: int=0, **kwargs) -> None:
        """
        Add a sample of nnet outputs and ground truth to calculate statistics.
        """
        raise NotImplementedError

    @staticmethod
    def _max_accuracy_lambda(
            criterion_dict: Dict[str, List[Union[Callable[[float, float], bool], float]]],
            path: Path, specific=None):
        """
        Lambda function used for grabbing the best metrics, give a dictionary with the name of
        the metric and a list that contains a callable function for the comparitor [min|max]
        and its initialisation value.\n
        The dictionary should be structured as {metric_name: [compare_function, init_value]}
        """
        for criterion in criterion_dict:
            criterion_dict[criterion].append(0.)

        with h5py.File(path, 'a') as hfile:
            if 'validation' in list(hfile):
                for epoch in hfile['validation']:
                    if 'Summary' in list(hfile['validation/'+epoch]):
                        summary_mean = hfile['validation/'+epoch+'/Summary'][:]
                        summary_var = 0.
                    else:
                        summary_mean = hfile['validation/'+epoch+'/Summary_Mean'][:]
                        summary_var = hfile['validation/'+epoch+'/Summary_Variance'][:]

                    srt_fnc = lambda x: x[0]
                    for idx, (key, data) in enumerate(sorted(criterion_dict.items(), key=srt_fnc)):
                        if data[0](data[1], summary_mean[idx]) == summary_mean[idx]:
                            criterion_dict[key][1] = summary_mean[idx]
                            criterion_dict[key][2] = summary_var[idx]
            else:
                Warning(f"No validation in {str(path)}.")
                return None

        if specific is not None:
            return criterion_dict[specific]

        ret_val = {}
        for key, value in criterion_dict.items():
            if key != "Batch_Loss":
                ret_val[key] = value[1], value[2]
        return ret_val

    def get_last_batch(self, main_metric=True):
        """
        Return the data from the last batch
        """
        if main_metric:
            return self.metric_data[self.main_metric][-1].mean()

        ret_val = ()
        for key, data in sorted(self.metric_data.items(), key=lambda x: x[0]):
            if key != "Batch_Loss":
                ret_val += (data[-1],)
        return ret_val

    def print_epoch_statistics(self):
        """
        Prints all the statistics
        """
        for key, data in self.metric_data.items():
            stripped = key.replace("Batch_", "")
            mean_data = np.asarray(data).mean()
            print("%s: %.3f" %(stripped, mean_data))

    def get_epoch_data(self, epoch=-1, statistic=None, mode='validation'):
        """
        Returns epoch statistics form hdf5 file
        """
        assert mode in ['validation', 'training']
        with h5py.File(self._path, 'a') as hfile:
            if mode in list(hfile):
                if epoch == -1:
                    epoch = max(list(hfile[mode]), key=lambda x: int(x[6:]))
                else:
                    epoch = f'Epoch_{epoch}'

                if statistic is not None:
                    return hfile[f'{mode}/{epoch}/{statistic}'][:]

                data = {}
                for dataset in list(hfile[f'{mode}/{epoch}']):
                    data[dataset] = hfile[f'{mode}/{epoch}/{dataset}'][:]

        Warning(f"No validation in {str(self._path)}.")
        return None

    def get_epoch_statistic(self, statistic: str, epoch=-1, mode='validation'):
        """
        Returns the mean and variance of an epoch
        """
        epoch_data = self.get_epoch_data(epoch, statistic, mode)
        return np.nanmean(epoch_data), np.nanvar(epoch_data, ddof=1)

class SegmentationMetric(MetricBase):
    """
    Accuracy and Loss Staticstics tracking for semantic segmentation networks.\n
    Tracks pixel wise accuracy (PixelAcc) and intersection over union (IoU).
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
        """
        Update Accuracy (and Loss) Metrics
        """
        assert all('seg' in keys for keys in [predictions.keys(), targets.keys()])
        self.metric_data["Batch_Loss"].append(loss)

        labels = targets['seg'].type(torch.int32).cuda()
        preds = torch.argmax(
            predictions['seg'], dim=1,keepdim=True).type(torch.int32).squeeze(dim=1)

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
        print(f"Pixel Accuracy: {pixel_acc:.4f}\nmIoU: {miou:.4f}\nLoss: {loss:.4f}")

    def get_current_statistics(self, main_metric=True, loss_metric=True):
        """
        Returns Accuracy Metrics [pixelwise, mIoU, loss]\n
        @todo   get a specified epoch instead of only currently loaded one\n
        @param  main_metric, returns mIoU and not Pixel Accuracy\n
        @param  loss_metric, returns recorded loss\n
        """
        ret_mean = ()
        ret_var = ()
        if main_metric:
            if self.main_metric == 'Batch_IoU':
                ret_mean += (np.nanmean(self._confmat_cls_iou(self.metric_data["Confusion_Mat"])),)
                data = np.asarray(self.metric_data['Batch_IoU']).reshape(-1, self._n_classes)
                ret_var += (np.nanvar(data, axis=1).mean(),)
            else:
                data = np.asarray(self.metric_data[self.main_metric]).flatten()
                ret_mean += (data.mean(),)
                ret_var += (data.var(ddof=1),)
            if loss_metric:
                ret_mean += (np.asarray(self.metric_data["Batch_Loss"]).mean(),)
                ret_var += (np.asarray(self.metric_data["Batch_Loss"]).var(ddof=1),)
        else:
            for key, data in sorted(self.metric_data.items(), key=lambda x: x[0]):
                if key == 'Batch_IoU':
                    ret_mean += (np.nanmean(
                        self._confmat_cls_iou(self.metric_data["Confusion_Mat"])),)
                    data = np.asarray(data).reshape(-1, self._n_classes)
                    ret_var += (np.nanvar(data, axis=1).mean(),)
                elif (key != 'Batch_Loss' or loss_metric) and key[:5] == 'Batch':
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
            axis[idx%3][idx//3].set_title(f'{trainId2name[idx]} over Epochs')

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
                training_data[idx] = hfile['training/'+epoch+'/Confusion_Mat'][:]

            for idx, epoch in enumerate(sorted(list(hfile['validation']), key=sort_lmbda)):
                testing_data[idx] = hfile['validation/'+epoch+'/Confusion_Mat'][:]

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
            num_epochs = len(list(hfile[dataset]))
            assert index < num_epochs
            epoch_name = f'Epoch_{index}' if index > 0 else f'Epoch_{num_epochs}'
            epoch_data = hfile[dataset][epoch_name+'/Confusion_Mat'][:]

        plt.figure(figsize=(18, 5))
        plt.suptitle("Class Confusion Matrix")
        labels = [trainId2name[i] for i in range(19)]
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
        plt.suptitle(self._path.name + ' IoU')

        for idx in range(self._n_classes):
            plt.subplot(3, self._n_classes//3+1, idx+1)
            plt.plot(train['iou'][:, idx])
            plt.plot(test['iou'][:, idx])
            plt.legend(["Training", "Validation"])
            plt.title(f'{trainId2name[idx]}')
            plt.xlabel('Epoch #')

        plt.tight_layout()
        plt.show(block=False)

        plt.figure(figsize=(18, 5))
        plt.suptitle(self._path.name + ' Precision')

        for idx in range(self._n_classes):
            plt.subplot(3, self._n_classes//3+1, idx+1)
            plt.plot(train['precision'][:, idx])
            plt.plot(test['precision'][:, idx])
            plt.legend(["Training", "Validation"])
            plt.title(f'{trainId2name[idx]}')
            plt.xlabel('Epoch #')

        plt.tight_layout()
        plt.show(block=False)

        plt.figure(figsize=(18, 5))
        plt.suptitle(self._path.name + ' Recall')

        for idx in range(self._n_classes):
            plt.subplot(3, self._n_classes//3+1, idx+1)
            plt.plot(train['recall'][:, idx])
            plt.plot(test['recall'][:, idx])
            plt.legend(["Training", "Validation"])
            plt.title(f'{trainId2name[idx]}')
            plt.xlabel('Epoch #')

        plt.tight_layout()
        plt.show(block=False)

class DepthMetric(MetricBase):
    """
    Accuracy/Error and Loss Staticstics tracking for depth based networks.\n
    Tracks Invariant, RMSE Linear, RMSE Log, Squared Relative and Absolute Relative.
    """
    def __init__(self, savefile: str, base_dir: Path, main_metric: str, mode='training'):
        super().__init__(
            savefile=savefile, base_dir=base_dir, main_metric=main_metric, mode=mode)
        self._reset_metric()
        assert self.main_metric in self.metric_data.keys()

    def add_sample(self, predictions: Dict[str, torch.Tensor],
                   targets: Dict[str, torch.Tensor], loss: int=0, **kwargs) -> None:
        assert all('depth' in keys for keys in [predictions.keys(), targets.keys()])
        self.metric_data["Batch_Loss"].append(loss)

        gt_depth = targets['l_disp']

        if isinstance(pred_depth, list):
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

        print("No File Specified for Segmentation Metric Manager")
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

class OpticFlowMetric(MetricBase):
    """
    Accuracy/Error and Loss Staticstics tracking for depth based networks.\n
    Tracks Flow rnd point error (EPE) and sum absolute difference between
    warped image sequence (SAD).
    """
    def __init__(self, savefile: str, base_dir: Path, main_metric: str, mode='training'):
        super().__init__(
            savefile=savefile, base_dir=base_dir, main_metric=main_metric, mode=mode)
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

class BoundaryBoxMetric(MetricBase):
    """
    Accuracy/Error and Loss Staticstics tracking for nnets with boundary box output
    """
    def __init__(self, savefile: str, base_dir: Path, main_metric: str, mode='training'):
        super().__init__(
            savefile=savefile, base_dir=base_dir, main_metric=main_metric, mode=mode)
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
        iou = self._calculate_iou(detr_outputs['bboxes'], targets['bboxes'], indices)

        self.metric_data['Batch_IoU'].append(iou)


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
            Batch_Precision=[]
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

def get_loggers(logger_cfg: Dict[str, str], basepath: Path) -> Dict[str, MetricBase]:
    """
    Given a dictionary of [key, value] = [objective type, main metric] and
    basepath to save the file returns a dictionary that consists of performance
    metric trackers.
    """
    loggers = {}

    for logger_type, main_metric in logger_cfg.items():
        if logger_type == 'flow':
            loggers['flow'] = OpticFlowMetric(
                'flow_data', main_metric=main_metric, base_dir=basepath)
        elif logger_type == 'seg':
            loggers['seg'] = SegmentationMetric(
                19, 'seg_data', main_metric=main_metric, base_dir=basepath)
        elif logger_type == 'depth':
            loggers['depth'] = DepthMetric(
                'depth_data', main_metric=main_metric, base_dir=basepath)
        elif logger_type == 'bbox':
            loggers['bbox'] = BoundaryBoxMetric(
                'bbox_data', main_metric=main_metric, base_dir=basepath)
        else:
            raise NotImplementedError(logger_type)

    return loggers

if __name__ == "__main__":
    FILENAME = "MonoSF_SegNet3_FlwExt1_FlwEst1_CtxNet1_Adam_Fcl_Uflw_HRes_seg"
    TEST = MetricBase(savefile=FILENAME, main_metric="Batch_EPE",
                           base_dir=Path.cwd()/"torch_models")
    TEST.plot_summary_data()
    input("Press enter to leave")
