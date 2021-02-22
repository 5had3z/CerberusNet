#!/usr/bin/env python3

__author__ = "Bryce Ferenczi"
__email__ = "bryce.ferenczi@monash.edu"

import os
from pathlib import Path
from typing import Dict
from typing import List
from typing import Callable
from typing import Union

import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

__all__ = ['MetricBase']

class MetricBase():
    """
    Provides basic functionality for statistics tracking classes
    """
    def __init__(self, savefile: str, base_dir: Path, main_metric: str, mode='training'):
        assert mode in ['training', 'validation'], f'invalid mode: {mode}'
        self.mode = mode
        self.metric_data = {}
        if not main_metric.startswith("Batch_"):
            main_metric = "Batch_"+main_metric
        self.main_metric = main_metric

        if savefile != "":
            if not savefile.endswith('.hdf5'):
                savefile += '.hdf5'
            self._path = base_dir / savefile
            if not os.path.isfile(self._path):
                with h5py.File(self._path, 'a') as hfile:
                    hfile.create_group('cache')
                    print(f"Training Statitsics created at {self._path}")
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
                    group_name = f'{self.mode}/Epoch_{len(list(hfile[self.mode]))+1}'
                else:
                    group_name = f'{self.mode}/Epoch_1'

                top_group = hfile.create_group(group_name)
                for name, data in self.metric_data.items():
                    if name.startswith("Batch_") and isinstance(data[0], np.ndarray):
                        data = np.concatenate(data)
                    else:
                        data = np.asarray(data)
                    top_group.create_dataset(name, data=data)

                mean, variance = np.asarray(self.get_current_statistics(main_only=False))
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
                for metric in list(hfile[f'{mode}/Epoch_{epoch_idx}']):
                    self.metric_data[metric] = hfile[f'{mode}/Epoch_{epoch_idx}/{metric}'][:]
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
        plt.suptitle(f'{self._path.name} Training and Validation Results Epoch {epoch_idx}')

        with h5py.File(self._path, 'r') as hfile:
            num_metrics = len(list(hfile[f'training/Epoch_{epoch_idx}']))
            for idx, metric in enumerate(list(hfile[f'training/Epoch_{epoch_idx}'])):
                plt.subplot(1, num_metrics, idx+1)
                plt.plot(hfile[f'training/Epoch_{epoch_idx}/{metric}'][:])
                plt.plot(hfile[f'validation/Epoch_{epoch_idx}/{metric}'][:])
                plt.legend(["Training", "Validation"])
                plt.title(f'Batch {metric} over Epochs')
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
                    n_cached = len(list(hfile[f'cache/{self.mode}'])) + 1

                if self.mode in list(hfile):
                    group_name = f'cache/{self.mode}/Epoch_{len(list(hfile[self.mode]))+n_cached}'
                else:
                    group_name = f'cache/{self.mode}/Epoch_{n_cached}'

                top_group = hfile.create_group(group_name)
                for name, data in self.metric_data.items():
                    if name.startswith("Batch_") and isinstance(data[0], np.ndarray):
                        data = np.concatenate(data)
                    else:
                        data = np.asarray(data)
                    top_group.create_dataset(name, data=data)

                mean, variance = np.asarray(self.get_current_statistics(main_only=False))
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
                for epoch in list(hfile[f'cache/{mode}']):
                    hfile.copy(f'cache/{mode}/{epoch}', f'{mode}/{epoch}')
                    del hfile[f'cache/{mode}/{epoch}']

    def get_summary_data(self):
        """
        Returns dictionary with testing and validation statistics
        """
        with h5py.File(self._path, 'r') as hfile:
            metrics = []
            for metric in list(hfile['training/Epoch_1']):
                if metric.startswith('Batch'):
                    metrics.append(metric)

            training_mean = np.zeros((len(list(hfile['training'])), len(metrics)))
            testing_mean = np.zeros((len(list(hfile['validation'])), len(metrics)))
            training_var = np.zeros((len(list(hfile['training'])), len(metrics)))
            testing_var = np.zeros((len(list(hfile['validation'])), len(metrics)))

            sort_func = lambda x: int(x[6:])
            for idx, epoch in enumerate(sorted(list(hfile['training']), key=sort_func)):
                if 'Summary' in list(hfile[f'training/{epoch}']):
                    training_mean[idx] = hfile[f'training/{epoch}/Summary'][:]
                else:
                    training_mean[idx] = hfile[f'training/{epoch}/Summary_Mean'][:]
                    training_var[idx] = hfile[f'training/{epoch}/Summary_Variance'][:]

            for idx, epoch in enumerate(sorted(list(hfile['validation']), key=sort_func)):
                if 'Summary' in list(hfile[f'validation/{epoch}']):
                    testing_mean[idx] = hfile[f'validation/{epoch}/Summary'][:]
                else:
                    testing_mean[idx] = hfile[f'validation/{epoch}/Summary_Mean'][:]
                    testing_var[idx] = hfile[f'validation/{epoch}/Summary_Variance'][:]

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
        Returns the number of samples in the dataset type
        """
        assert dataset in ['validation', 'training']

        with h5py.File(self._path, 'r') as hfile:
            for metric in list(hfile[f'{dataset}/Epoch_1']):
                if metric.startswith('Batch'):
                    return hfile[f'{dataset}/Epoch_1/{metric}'][:].shape[0]

        return 0

    def plot_summary_data(self):
        """
        This plots all the summary statistics over all epochs
        """
        plt.figure(figsize=(18, 5))
        plt.suptitle(self._path.name + ' Summary Training and Validation Results')

        n_train = self.get_n_samples('training')
        n_val = self.get_n_samples('validation')
        summary_data = self.get_summary_data()

        conf_width = lambda var, n: stats.t.ppf(0.95, n-1) * var / np.sqrt(n)

        for idx, metric in enumerate(summary_data):
            plt.subplot(1, len(summary_data), idx+1)

            data_mean = summary_data[metric]["Training_Mean"]
            data_conf = conf_width(summary_data[metric]["Training_Variance"], n_train)
            plt.plot(data_mean)
            plt.fill_between(
                np.arange(0, data_mean.shape[0]),
                data_mean - data_conf, data_mean + data_conf,
                alpha=0.2)

            data_mean = summary_data[metric]["Validation_Mean"]
            data_conf = conf_width(summary_data[metric]["Validation_Variance"], n_val)
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
        plt.suptitle(f'{self._path.name} Iteration Training and Validation Results')

        with h5py.File(self._path, 'r') as hfile:
            training_metrics = {}
            validation_metrics = {}
            for metric in list(hfile['training/Epoch_1']):
                if metric.startswith('Batch'):
                    training_metrics[metric] = np.zeros((1, 1))
                    validation_metrics[metric] = np.zeros((1, 1))

            for epoch in list(hfile['training']):
                for metric in list(hfile[f'training/{epoch}']):
                    if metric.startswith('Batch'):
                        training_metrics[metric] = np.append(
                            training_metrics[metric], hfile[f'training/{epoch}/{metric}'][:])

            for epoch in list(hfile['validation']):
                for metric in list(hfile[f'validation/{epoch}']):
                    if metric.startswith('Batch'):
                        validation_metrics[metric] = np.append(
                            validation_metrics[metric], hfile[f'validation/{epoch}/{metric}'][:])

            print(f"# Training, {len(list(hfile['training']))}" \
                  f"\t# Validation, {len(list(hfile['validation']))}")

        num_metrics = len(training_metrics.keys())
        for idx, metric in enumerate(training_metrics):
            plt.subplot(1, num_metrics, idx+1)
            plt.plot(training_metrics[metric])
            plt.plot(validation_metrics[metric])
            plt.legend(["Training", "Validation"])
            plt.title(f'{metric} over Iterations')
            plt.xlabel('Iteration #')

        plt.show()

    def _reset_metric(self):
        raise NotImplementedError

    def get_current_statistics(self, main_only=True, return_loss=True):
        """
        Returns Accuracy and Loss Metrics from an Epoch\n
        @todo   get a specified epoch instead of only currently loaded one\n
        @param  main_metric, only main metric\n
        @param  loss_metric, returns recorded loss\n
        """
        ret_mean = ()
        ret_var = ()
        if main_only:
            data = np.asarray(self.metric_data[self.main_metric]).flatten()
            ret_mean += (data.mean(),)
            ret_var += (data.var(ddof=1),)
            if return_loss:
                ret_mean += (np.asarray(self.metric_data["Batch_Loss"]).mean(),)
                ret_var += (np.asarray(self.metric_data["Batch_Loss"]).var(ddof=1),)
        else:
            for key, data in sorted(self.metric_data.items(), key=lambda x: x[0]):
                if (key != "Batch_Loss" or return_loss) and key.startswith('Batch'):
                    if isinstance(data[0], np.ndarray):
                        data = np.concatenate(data)
                    else:
                        data = np.asarray(data).ravel()
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
                    if 'Summary' in list(hfile[f'validation/{epoch}']):
                        summary_mean = hfile[f'validation/{epoch}/Summary'][:]
                        summary_var = 0.
                    else:
                        summary_mean = hfile[f'validation/{epoch}/Summary_Mean'][:]
                        summary_var = hfile[f'validation/{epoch}/Summary_Variance'][:]

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
            return np.nanmean(self.metric_data[self.main_metric][-1])

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
            print(f"{key.replace('Batch_', '')}: {np.asarray(data).mean():.3f}")

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


if __name__ == "__main__":
    FILENAME = "MonoSF_SegNet3_FlwExt1_FlwEst1_CtxNet1_Adam_Fcl_Uflw_HRes_seg"
    TEST = MetricBase(savefile=FILENAME, main_metric="Batch_EPE",
                      base_dir=Path.cwd()/"torch_models")
    TEST.plot_summary_data()
    input("Press enter to leave")
