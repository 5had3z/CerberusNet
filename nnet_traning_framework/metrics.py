#!/usr/bin/env python3

__author__ = "Bryce Ferenczi"
__email__ = "bryce.ferenczi@monashmotorsport.com"

import os
import h5py
import threading
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

__all__ = ['SegmentationMetric', 'BoundaryBoxMetric', 'ClassificationMetric']

class MetricBaseClass(object):
    def __init__(self, mode='training', filename=None):
        assert mode == 'training' or mode == 'validation'
        self.mode = mode
        self.metric_data = dict()
        self._reset_metric()

        if filename is not None:
            if filename[-5:] != '.hdf5':
                filename = filename + '.hdf5'
            self._path = Path.cwd() / "torch_models" / filename
            if not os.path.isfile(self._path):
                with h5py.File(self._path, 'a') as hf:
                    hf.create_group('cache')
                    print("Training Statitsics created at ", self._path)
        else:
            self._path = None
        
    def __len__(self):
        """
        Return Number of Epochs Recorded
        """
        with h5py.File(self._path, 'r') as hf:
            if 'training' in list(hf):
                n_epochs = len(list(hf['training']))
                if 'cache' in list(hf):
                    if 'training' in list(hf['cache']):
                        n_epochs += len(list(hf['cache/training']))
                return n_epochs
            else:
                return 0
    
    def __del__(self):
        with h5py.File(self._path, 'a') as hf:
            if 'cache' in list(hf):
                del hf['cache']

    def save_epoch(self):
        """
        Save Data to new dataset named by epoch name
        """             
        if self._path is not None:
            summary_stats = np.asarray(self._get_epoch_statistics())
            with h5py.File(self._path, 'a') as hf:
                # Clear any Cached data from previous first into main
                if len(list(hf['cache'])) > 0:
                    self._flush_to_main()
                
                if self.mode in list(hf):
                    group_name = self.mode + '/Epoch_' + str(len(list(hf[self.mode])) + 1)
                else:
                    group_name = self.mode + '/Epoch_1'

                top_group = hf.create_group(group_name)
                for metric in self.metric_data.keys():
                    top_group.create_dataset(metric, data=np.asarray(self.metric_data[metric]))
                top_group.create_dataset('Summary', data=summary_stats)

                #   Flush current data as its now in long term storage and we're ready for next dataset
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
            with h5py.File(self._path, 'r') as hf:
                group_name = 'Epoch_' + str(epoch_idx)
                for metric in list(hf[mode][group_name]):
                    self.metric_data[metric] = hf[mode][group_name][metric][:]
            return self.metric_data
        else:
            print("No File Specified for Segmentation Metric Manager")

    def new_epoch(self, mode='training'):
        assert mode == 'training' or mode == 'validation'

        # Cache data if it hasn't been saved yet (probably not a best epoch or
        # something, but the next might be, so we want to keep this data)
        for key in self.metric_data.keys():
            if len(self.metric_data[key]) > 0:
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
        with h5py.File(self._path, 'r') as hf:
            num_metrics = len(list(hf['training'][group_name]))
            for idx, metric in enumerate(list(hf['training'][group_name])):
                plt.subplot(1,num_metrics,idx+1)
                plt.plot(hf['training'][group_name][metric][:])
                plt.plot(hf['validation'][group_name][metric][:])
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
            summary_stats = np.asarray(self._get_epoch_statistics())
            with h5py.File(self._path, 'a') as hf:
                n_cached = 1
                if 'cache' in list(hf):
                    if self.mode in list(hf['cache']):
                        n_cached = len(list(hf['cache/'+self.mode])) + 1

                if self.mode in list(hf):
                    group_name = 'cache/' + self.mode + '/Epoch_' + str(len(list(hf[self.mode])) + n_cached)
                else:
                    group_name = 'cache/' + self.mode + '/Epoch_' + str(n_cached)

                top_group = hf.create_group(group_name)
                for metric in self.metric_data.keys():
                    top_group.create_dataset(metric, data=np.asarray(self.metric_data[metric]))
                top_group.create_dataset('Summary', data=summary_stats)

        else:
            print("No File Specified for Segmentation Metric Manager")

    def _flush_to_main(self):
        """
        Moves data from cache to main storage area
        """
        with h5py.File(self._path, 'a') as hf:
            for mode in list(hf['cache']):
                for epoch in list(hf['cache/'+mode]):
                    hf.copy('cache/'+mode+'/'+epoch, mode+'/'+epoch)
                    del hf['cache/'+mode+'/'+epoch]

    def plot_summary_data(self):
        """
        This plots all the summary statistics over all epochs
        """
        plt.figure(figsize=(18, 5))
        plt.suptitle(self._path.name + ' Summary Training and Validation Results')

        with h5py.File(self._path, 'r') as hf:
            
            metrics = []
            for metric in list(hf['training/Epoch_1']):
                if metric != 'Summary':
                    metrics.append(metric)
            
            training_data = np.zeros((len(list(hf['training'])), len(metrics)))
            testing_data = np.zeros((len(list(hf['validation'])), len(metrics)))

            for idx, epoch in enumerate(list(hf['training'])):
                training_data[idx] = hf['training/'+epoch+'/Summary'][:]

            for idx, epoch in enumerate(list(hf['validation'])):
                testing_data[idx] = hf['validation/'+epoch+'/Summary'][:]

            print("# Training, ", len(list(hf['training'])), "\t# Validation", len(list(hf['validation'])))
        
        for idx, metric in enumerate(metrics):
            plt.subplot(1,len(metrics),idx+1)
            plt.plot(training_data[:,idx])
            plt.plot(testing_data[:,idx])
            plt.legend(["Training", "Validation"])
            plt.title(metric + ' over Epochs')
            plt.xlabel('Epoch #')

        plt.show()

    def plot_iteration_data(self):
        """
        This plots all the statistics over all non-cached iterations
        """
        plt.figure(figsize=(18, 5))
        plt.suptitle(self._path.name + ' Iteration Training and Validation Results')

        with h5py.File(self._path, 'r') as hf:
            training_metrics = {}
            validation_metrics  = {}
            for metric in list(hf['training/Epoch_1']):
                if metric != 'Summary':
                    training_metrics[metric] = np.zeros((1,1))
                    validation_metrics[metric]  = np.zeros((1,1))

            for epoch in list(hf['training']):
                for metric in list(hf['training/'+epoch]):
                    if metric != 'Summary':
                        training_metrics[metric] = np.append(
                                training_metrics[metric],
                                hf['training/'+epoch+'/'+metric][:])

            for epoch in list(hf['validation']):
                for metric in list(hf['validation/'+epoch]):
                    if metric != 'Summary':
                        validation_metrics[metric] = np.append(
                                validation_metrics[metric],
                                hf['validation/'+epoch+'/'+metric][:])

            print("# Training, ", len(list(hf['training'])), "\t# Validation", len(list(hf['validation'])))
        
        num_metrics = len(training_metrics.keys())
        for idx, metric in enumerate(training_metrics):
            plt.subplot(1,num_metrics, idx+1)
            plt.plot(training_metrics[metric])
            plt.plot(validation_metrics[metric])
            plt.legend(["Training", "Validation"])
            plt.title(metric + ' over Iterations')
            plt.xlabel('Iteration #')

        plt.show()

    def _reset_metric(self):
        raise NotImplementedError

    def _get_epoch_statistics(self):
        raise NotImplementedError

    def max_accuracy(self):
        raise NotImplementedError

class SegmentationMetric(MetricBaseClass):
    def __init__(self, num_classes, mode='training', filename=None):
        super(SegmentationMetric, self).__init__(mode=mode, filename=filename)
        self._n_classes = num_classes

    def _add_sample(self, preds, labels, loss=None):
        """
        Update Accuracy (and Loss) Metrics
        """
        if loss is not None:
            self.metric_data["Batch_Loss"].append(loss)
        
        pxthread = threading.Thread(target=self._pixelwise, args=(preds, labels))
        iouthread = threading.Thread(target=self._iou, args=(preds, labels))
        pxthread.start()
        iouthread.start()
        pxthread.join()
        iouthread.join()
        
        return self.metric_data["Batch_PixelAcc"][-1], self.metric_data["Batch_mIoU"][-1]

    def _get_epoch_statistics(self, epoch_idx=-1, print_only=True):
        """
        Returns Accuracy Metrics [pixelwise, mIoU]
        """ 
        pixAcc = np.asarray(self.metric_data["Batch_PixelAcc"]).mean()
        mIoU = np.asarray(self.metric_data["Batch_mIoU"]).mean()
        loss = np.asarray(self.metric_data["Batch_Loss"]).mean()
        if print_only:
            print("Pixel Accuracy: %.4f\tmIoU: %.4f\tLoss: %.4f\n" % (pixAcc, mIoU, loss))
        return pixAcc, mIoU, loss

    def max_accuracy(self):
        """
        Returns highest mIoU and PixelWise Accuracy from per epoch summarised data
        @output mIoU, PixelWise Accuracy
        """
        mIoU = 0
        PixelAcc = 0
        if self._path is not None:
            with h5py.File(self._path, 'a') as hf:
                for epoch in hf['validation']:
                    summary_data = hf['validation/'+epoch+'/Summary'][:]
                    if summary_data[0] > PixelAcc:
                        PixelAcc = summary_data[0]
                    if summary_data[1] > mIoU:
                        mIoU = summary_data[1]
        else:
            print("No File Specified for Segmentation Metric Manager")
        return mIoU, PixelAcc

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
        
        mIoU = (1.0 * area_intersection / (np.spacing(1) + area_union)).mean()
        self.metric_data["Batch_mIoU"].append(mIoU)
    
    def _pixelwise(self, prediction, target):
        # Remove classes from unlabeled pixels in gt image.
        # We should not penalize detections in unlabeled portions of the image.
        prediction = np.squeeze(prediction.astype('int64') + 1, 1)
        target = target.astype('int64') + 1

        correct = 1.0 * np.sum((prediction == target) * (target > 0))
        total_pixels = np.spacing(1) + np.sum(target > 0)
        pixAcc = correct / total_pixels
        self.metric_data["Batch_PixelAcc"].append(pixAcc)
    
    def _reset_metric(self):
        self.metric_data = dict(
            Batch_Loss=[],
            Batch_PixelAcc=[],
            Batch_mIoU=[] )

class DepthMetric(MetricBaseClass):
    def __init__(self, mode='training', filename=None):
        super(DepthMetric, self).__init__(mode=mode, filename=filename)
        raise NotImplementedError

    def _add_sample(self, pred_depth, gt_depth, loss=None):
        raise NotImplementedError
    
    def _get_epoch_statistics(self, epoch_idx=-1, print_only=True):
        raise NotImplementedError

    def max_accuracy(self):
        raise NotImplementedError

    def _reset_metric(self):
        raise NotImplementedError


class BoundaryBoxMetric(MetricBaseClass):
    def __init__(self, mode='training', filename=None):
        super(BoundaryBoxMetric, self).__init__(mode=mode, filename=filename)
        raise NotImplementedError

    def _add_sample(self, pred_depth, gt_depth, loss=None):
        raise NotImplementedError
    
    def _get_epoch_statistics(self, epoch_idx=-1, print_only=True):
        raise NotImplementedError

    def max_accuracy(self):
        raise NotImplementedError

    def _reset_metric(self):
        raise NotImplementedError

class ClassificationMetric(MetricBaseClass):
    def __init__(self, mode='training', filename=None):
        super(ClassificationMetric, self).__init__(mode=mode, filename=filename)
        raise NotImplementedError

    def _add_sample(self, pred_depth, gt_depth, loss=None):
        raise NotImplementedError
    
    def _get_epoch_statistics(self, epoch_idx=-1, print_only=True):
        raise NotImplementedError

    def max_accuracy(self):
        raise NotImplementedError

    def _reset_metric(self):
        raise NotImplementedError

if __name__ == "__main__":
    # filename = "Stereo_Seg_Focal"
    filename = 'Focal_HalfSize'
    # filename = "Focal"
    metric = SegmentationMetric(19, filename=filename)
    metric.plot_iteration_data()
