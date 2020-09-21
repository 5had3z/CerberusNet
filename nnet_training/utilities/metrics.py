#!/usr/bin/env python3

__author__ = "Bryce Ferenczi"
__email__ = "bryce.ferenczi@monashmotorsport.com"

import os
import sys
import multiprocessing
from pathlib import Path
from typing import Dict, List, Callable, Union

import h5py
import numpy as np
import matplotlib.pyplot as plt

from nnet_training.utilities.UnFlowLoss import flow_warp

__all__ = ['MetricBaseClass', 'SegmentationMetric', 'DepthMetric',
           'BoundaryBoxMetric', 'ClassificationMetric']

class MetricBaseClass(object):
    """
    Provides basic functionality for statistics tracking classes
    """
    def __init__(self, savefile: str, base_dir: Path, main_metric: str, mode='training'):
        assert mode in ['training', 'validation']
        self.mode = mode
        self.metric_data = dict()
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

    def __del__(self):
        with h5py.File(self._path, 'a') as hfile:
            if 'cache' in list(hfile):
                del hfile['cache']

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
                    top_group.create_dataset(name, data=np.asarray(data))

                summary_stats = np.asarray(self.get_epoch_statistics(main_metric=False))
                top_group.create_dataset('Summary', data=summary_stats)

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
        for metric in self.metric_data.values():
            if len(metric) > 0:
                self._cache_data()
                break

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
                for metric in self.metric_data:
                    top_group.create_dataset(metric, data=np.asarray(self.metric_data[metric]))

                summary_stats = np.asarray(self.get_epoch_statistics(main_metric=False))
                top_group.create_dataset('Summary', data=summary_stats)

        else:
            print("No File Specified for Segmentation Metric Manager")

    def _flush_to_main(self):
        """
        Moves data from cache to main storage area
        """
        with h5py.File(self._path, 'a') as hfile:
            for mode in list(hfile['cache']):
                for epoch in list(hfile['cache/'+mode]):
                    hfile.copy('cache/'+mode+'/'+epoch, mode+'/'+epoch)
                    del hfile['cache/'+mode+'/'+epoch]

    def plot_summary_data(self):
        """
        This plots all the summary statistics over all epochs
        """
        plt.figure(figsize=(18, 5))
        plt.suptitle(self._path.name + ' Summary Training and Validation Results')

        with h5py.File(self._path, 'r') as hfile:
            metrics = []
            for metric in list(hfile['training/Epoch_1']):
                if metric != 'Summary':
                    metrics.append(metric)

            training_data = np.zeros((len(list(hfile['training'])), len(metrics)))
            testing_data = np.zeros((len(list(hfile['validation'])), len(metrics)))

            sort_func = lambda x: int(x[6:])
            for idx, epoch in enumerate(sorted(list(hfile['training']), key=sort_func)):
                training_data[idx] = hfile['training/'+epoch+'/Summary'][:]

            for idx, epoch in enumerate(sorted(list(hfile['validation']), key=sort_func)):
                testing_data[idx] = hfile['validation/'+epoch+'/Summary'][:]

            print("# Training, ", len(list(hfile['training'])),
                  "\t# Validation", len(list(hfile['validation'])))

        for idx, metric in enumerate(metrics):
            plt.subplot(1, len(metrics), idx+1)
            plt.plot(training_data[:, idx])
            plt.plot(testing_data[:, idx])
            plt.legend(["Training", "Validation"])
            plt.title(metric + ' over Epochs')
            plt.xlabel('Epoch #')

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
                if metric != 'Summary':
                    training_metrics[metric] = np.zeros((1, 1))
                    validation_metrics[metric] = np.zeros((1, 1))

            for epoch in list(hfile['training']):
                for metric in list(hfile['training/'+epoch]):
                    if metric != 'Summary':
                        training_metrics[metric] = np.append(
                            training_metrics[metric],
                            hfile['training/'+epoch+'/'+metric][:])

            for epoch in list(hfile['validation']):
                for metric in list(hfile['validation/'+epoch]):
                    if metric != 'Summary':
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

    def get_epoch_statistics(self, main_metric=True, loss_metric=True):
        """
        Returns Accuracy and Loss Metrics from an Epoch\n
        @todo   get a specified epoch instead of only currently loaded one\n
        @param  main_metric, only main metric\n
        @param  loss_metric, returns recorded loss\n
        """
        ret_val = ()
        if main_metric:
            invariant = np.asarray(self.metric_data[self.main_metric]).mean()
            ret_val += (invariant,)
            if loss_metric:
                loss = np.asarray(self.metric_data["Batch_Loss"]).mean()
                ret_val += (loss,)

        else:
            for key, data in sorted(self.metric_data.items(), key=lambda x: x[0]):
                if key != "Batch_Loss" or loss_metric:
                    mean_data = np.asarray(data).mean()
                    ret_val += (mean_data,)

        return ret_val

    def max_accuracy(self, main_metric=True):
        """
        Return the max accuracy from all epochs
        """
        raise NotImplementedError

    @staticmethod
    def _max_accuracy_lambda(criterion_dict: Dict[str, List[Union[Callable[[int, int], bool], float]]],
                             path: Path, main_metric=None):
        """
        Lambda function used for grabbing the best metrics, give a dictionary with the name of
        the metric and a list that contains a callable function for the comparitor [min|max]
        and its initialisation value.\n
        The dictionary should be structured as {metric_name: [compare_function, init_value]}
        """
        with h5py.File(path, 'a') as hfile:
            if 'validation' in list(hfile):
                for epoch in hfile['validation']:
                    summary_data = hfile['validation/'+epoch+'/Summary'][:]

                    srt_fnc = lambda x: x[0]
                    for idx, (key, data) in enumerate(sorted(criterion_dict.items(), key=srt_fnc)):
                        criterion_dict[key][1] = data[0](data[1], summary_data[idx])

        if main_metric is not None:
            return main_metric, criterion_dict[main_metric]

        ret_val = ()
        for key, value in criterion_dict.items():
            if key != "Batch_Loss":
                ret_val += (value[1],)
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

class SegmentationMetric(MetricBaseClass):
    """
    Accuracy and Loss Staticstics tracking for semantic segmentation networks.\n
    Tracks pixel wise accuracy (PixelAcc) and intersection over union (IoU).
    """
    def __init__(self, num_classes, savefile: str, base_dir: Path,
                 main_metric: str, mode='training'):
        super(SegmentationMetric, self).__init__(savefile=savefile, base_dir=base_dir,
                                                 main_metric=main_metric, mode=mode)
        self._n_classes = num_classes
        self._reset_metric()
        assert self.main_metric in self.metric_data.keys()

    def add_sample(self, preds, labels, loss=None):
        """
        Update Accuracy (and Loss) Metrics
        """
        if loss is not None:
            self.metric_data["Batch_Loss"].append(loss)

        labels = labels.astype('int32') + 1
        preds = np.squeeze(preds.astype('int32') + 1, 1)

        rcv_px, send_px = multiprocessing.Pipe(False)
        pxthread = multiprocessing.Process(target=self._pixelwise, args=(preds, labels, send_px))
        rcv_iou, send_iou = multiprocessing.Pipe(False)
        iouthread = multiprocessing.Process(target=self._iou, args=(preds, labels, send_iou))
        pxthread.start()
        iouthread.start()
        pxthread.join()
        iouthread.join()
        self.metric_data["Batch_IoU"].append(rcv_iou.recv())
        self.metric_data["Batch_PixelAcc"].append(rcv_px.recv())

    def print_epoch_statistics(self):
        """
        Prints all the statistics
        """
        pixel_acc = np.asarray(self.metric_data["Batch_PixelAcc"]).mean()
        miou = np.asarray(self.metric_data["Batch_IoU"]).mean(axis=(0, 1))
        loss = np.asarray(self.metric_data["Batch_Loss"]).mean()
        print("Pixel Accuracy: %.4f\tmIoU: %.4f\tLoss: %.4f\n" % (pixel_acc, miou, loss))

    def get_epoch_statistics(self, main_metric=True, loss_metric=True):
        """
        Returns Accuracy Metrics [pixelwise, mIoU, loss]\n
        @todo   get a specified epoch instead of only currently loaded one\n
        @param  main_metric, returns mIoU and not Pixel Accuracy\n
        @param  loss_metric, returns recorded loss\n
        """
        ret_val = ()
        if main_metric:
            if self.main_metric == 'Batch_IoU':
                mean_data = np.asarray(self.metric_data[self.main_metric]).mean(axis=(0, 1))
            else:
                mean_data = np.asarray(self.metric_data[self.main_metric]).mean()
            ret_val += (mean_data,)
            if loss_metric:
                loss = np.asarray(self.metric_data["Batch_Loss"]).mean()
                ret_val += (loss,)
        else:
            for key, data in sorted(self.metric_data.items(), key=lambda x: x[0]):
                if key == 'Batch_IoU':
                    mean_data = np.asarray(data).mean(axis=(0, 1))
                elif key != 'Batch_Loss' or loss_metric:
                    mean_data = np.asarray(data).mean()
                ret_val += (mean_data,)

        return ret_val

    def max_accuracy(self, main_metric=True):
        """
        Returns highest mIoU and PixelWise Accuracy from per epoch summarised data.\n
        @param  main_metric, if true only returns mIoU\n
        @output PixelWise Accuracy, mIoU
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

    def _iou(self, prediction, target, ret_val):
        # Remove classes from unlabeled pixels in gt image.
        # We should not penalize detections in unlabeled portions of the image.
        prediction = prediction * (target > 0).astype(prediction.dtype)

        # Compute area intersection:
        intersection = prediction * (prediction == target)
        area_intersection, _ = np.histogram(intersection, bins=self._n_classes,
                                            range=(1, self._n_classes))

        # Compute area union:
        area_pred, _ = np.histogram(prediction, bins=self._n_classes, range=(1, self._n_classes))
        area_lab, _ = np.histogram(target, bins=self._n_classes, range=(1, self._n_classes))
        area_union = area_pred + area_lab - area_intersection

        iou = 1.0 * area_intersection / (np.spacing(1) + area_union)
        ret_val.send(iou)

    @staticmethod
    def _pixelwise(prediction, target, ret_val):
        # Remove classes from unlabeled pixels in gt image.
        # We should not penalize detections in unlabeled portions of the image.
        correct = 1.0 * np.sum((prediction == target) * (target > 0))
        total_pixels = np.spacing(1) + np.sum(target > 0)
        pix_acc = correct / total_pixels
        ret_val.send(pix_acc)

    def _reset_metric(self):
        self.metric_data = dict(
            Batch_Loss=[],
            Batch_PixelAcc=[],
            Batch_IoU=[]
        )

    def plot_classwise_iou(self):
        """
        This plots the iou of each class over all epochs
        """
        plt.figure(figsize=(18, 5))
        plt.suptitle(self._path.name + ' Summary Training and Validation Results')

        with h5py.File(self._path, 'r') as hfile:
            n_classes = hfile['training/Epoch_1/Batch_IoU'][:].shape[1]

            training_data = np.zeros((len(list(hfile['training'])), n_classes))
            testing_data = np.zeros((len(list(hfile['validation'])), n_classes))

            sort_lmbda = lambda x: int(x[6:])
            for idx, epoch in enumerate(sorted(list(hfile['training']), key=sort_lmbda)):
                training_data[idx] = hfile['training/'+epoch+'/Batch_IoU'][:].mean(axis=0)

            for idx, epoch in enumerate(sorted(list(hfile['validation']), key=sort_lmbda)):
                testing_data[idx] = hfile['validation/'+epoch+'/Batch_IoU'][:].mean(axis=0)

            print("# Training, ", len(list(hfile['training'])),
                  "\t# Validation", len(list(hfile['validation'])))

        for idx in range(n_classes):
            plt.subplot(3, (n_classes//3+1), idx+1)
            plt.plot(training_data[:, idx])
            plt.plot(testing_data[:, idx])
            plt.legend(["Training", "Validation"])
            plt.title('Class: '+str(idx)+' over Epochs')
            plt.xlabel('Epoch #')

        plt.show(block=False)

class DepthMetric(MetricBaseClass):
    """
    Accuracy/Error and Loss Staticstics tracking for depth based networks.\n
    Tracks Invariant, RMSE Linear, RMSE Log, Squared Relative and Absolute Relative.
    """
    def __init__(self, savefile: str, base_dir: Path, main_metric: str, mode='training'):
        super(DepthMetric, self).__init__(savefile=savefile, base_dir=base_dir,
                                          main_metric=main_metric, mode=mode)
        self._reset_metric()
        assert self.main_metric in self.metric_data.keys()

    def add_sample(self, pred_depth, gt_depth, loss=None):
        if loss is not None:
            self.metric_data["Batch_Loss"].append(loss)

        pred_depth = pred_depth.squeeze(axis=1)[gt_depth != 0]
        pred_depth[pred_depth == 0] += 0.00001
        gt_depth = gt_depth[gt_depth != 0]

        n_pixels = gt_depth.size
        difference = pred_depth - gt_depth
        squared_diff = np.square(difference)
        log_diff = np.log(pred_depth) - np.log(gt_depth)

        self.metric_data['Batch_Absolute_Relative'].append(
            np.mean(np.absolute(difference)/gt_depth))

        self.metric_data['Batch_Squared_Relative'].append(
            np.mean(squared_diff/gt_depth))

        self.metric_data['Batch_RMSE_Linear'].append(
            np.mean(np.sqrt(squared_diff)))

        self.metric_data['Batch_RMSE_Log'].append(
            np.mean(np.sqrt(np.square(log_diff))))

        eqn1 = np.mean(np.square(log_diff))
        eqn2 = np.square(np.sum(log_diff)) / n_pixels**2
        self.metric_data['Batch_Invariant'].append(eqn1 - eqn2)

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
            'Batch_Absolute_Relative': [min, sys.float_info.max]
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
            Batch_Invariant=[]
        )

class OpticFlowMetric(MetricBaseClass):
    """
    Accuracy/Error and Loss Staticstics tracking for depth based networks.\n
    Tracks Flow rnd point error (EPE) and sum absolute difference between
    warped image sequence (SAD).
    """
    def __init__(self, savefile: str, base_dir: Path, main_metric: str, mode='training'):
        super(OpticFlowMetric, self).__init__(savefile=savefile, base_dir=base_dir,
                                              main_metric=main_metric, mode=mode)
        self._reset_metric()
        assert self.main_metric in self.metric_data.keys()

    def add_sample(self, orig_img, seq_img, flow_pred, flow_target=None, loss=None):
        """
        @input list of original, prediction and sequence images i.e. [left, right]
        """
        if loss is not None:
            self.metric_data["Batch_Loss"].append(loss)

        if flow_target is not None:
            diff = flow_pred - flow_target["flow"]
            norm_diff = ((diff[:, 0, :, :]**2 + diff[:, 1, :, :]**2)**0.5).unsqueeze(1)
            masked_epe = (norm_diff * flow_target["flow_mask"]).sum()
            n_valid = flow_target["flow_mask"].sum()
            self.metric_data["Batch_EPE"].append(
                (masked_epe / n_valid).cpu().data.numpy())
        else:
            self.metric_data["Batch_EPE"].append(0)

        self.metric_data["Batch_SAD"].append(
            (flow_warp(orig_img, flow_pred)-seq_img).abs().mean().cpu().data.numpy())

    def max_accuracy(self, main_metric=True):
        """
        Returns lowest end point error and sum absolute difference from per epoch summarised data.\n
        @param  main_metric, if true only returns scale invariant\n
        @output scale invariant, absolute relative, squared relative, rmse linear, rmse log
        """
        cost_func = {
            'Batch_Loss': [min, sys.float_info.max],
            'Batch_SAD': [min, sys.float_info.max],
            'Batch_EPE': [min, sys.float_info.max]
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
            Batch_EPE=[]
        )

class BoundaryBoxMetric(MetricBaseClass):
    """
    Accuracy/Error and Loss Staticstics tracking for nnets with boundary box output
    """
    def __init__(self, savefile: str, base_dir: Path, main_metric: str, mode='training'):
        super(BoundaryBoxMetric, self).__init__(savefile=savefile, base_dir=base_dir,
                                                main_metric=main_metric, mode=mode)
        self._reset_metric()
        assert self.main_metric in self.metric_data.keys()
        raise NotImplementedError

    def add_sample(self, pred_depth, gt_depth, loss=None):
        raise NotImplementedError

    def max_accuracy(self, main_metric=True):
        raise NotImplementedError

    def _reset_metric(self):
        raise NotImplementedError

class ClassificationMetric(MetricBaseClass):
    """
    Accuracy/Error and Loss Staticstics tracking for classification problems
    """
    def __init__(self, savefile: str, base_dir: Path, main_metric: str, mode='training'):
        super(ClassificationMetric, self).__init__(savefile=savefile, base_dir=base_dir,
                                                   main_metric=main_metric, mode=mode)
        self._reset_metric()
        assert self.main_metric in self.metric_data.keys()

    def add_sample(self, pred_depth, gt_depth, loss=None):
        raise NotImplementedError

    def max_accuracy(self, main_metric=True):
        raise NotImplementedError

    def _reset_metric(self):
        raise NotImplementedError

if __name__ == "__main__":
    FILENAME = "MonoSF_SegNet3_FlwExt1_FlwEst1_CtxNet1_Adam_Fcl_Uflw_HRes_seg"
    TEST = MetricBaseClass(savefile=FILENAME, main_metric="Batch_EPE",
                           base_dir=Path.cwd()/"torch_models")
    TEST.plot_summary_data()
    input("Press enter to leave")
