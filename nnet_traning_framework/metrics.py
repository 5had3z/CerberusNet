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

class SegmentationMetric(object):
    def __init__(self, num_classes, mode='training', filename=None):
        assert mode == 'training' or mode == 'validation'
        self._n_classes = num_classes
        self.mode = mode
        self.metric_data = dict(
            Batch_Loss=[],
            Batch_PixelAcc=[],
            Batch_mIoU=[] )

        if filename is not None:
            if filename[-5:] != '.hdf5':
                filename = filename + '.hdf5'
            self._path = Path.cwd() / "torch_models" / filename
            if not os.path.isfile(self._path):
                with h5py.File(self._path, 'a') as hf:
                    tmp = hf.create_group('training')
                    tmp.create_dataset('tmp', data=5)
                    tmp = hf.create_group('validation')
                    tmp.create_dataset('tmp', data=5)
                    print("Training Statitsics created at ", self._path)
        else:
            self._path = None
        
    def __len__(self):
        """
        Return Number of Epochs Recorded
        """
        with h5py.File(self._path, 'r') as hf:
            n_epochs = len(list(hf['training']))-1
            if 'cache' in list(hf):
                if 'training' in list(hf['cache']):
                    n_epochs += len(list(hf['cache/training']))
            return  n_epochs
    
    def __del__(self):
        with h5py.File(self._path, 'a') as hf:
            if 'cache' in list(hf):
                del hf['cache']

    def add_sample(self, preds, labels, loss=None):
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

    def get_epoch_statistics(self, epoch_idx=-1, print_only=True):
        """
        Returns Accuracy Metrics [pixelwise, mIoU]
        """ 
        pixAcc = np.asarray(self.metric_data["Batch_PixelAcc"]).mean()
        mIoU = np.asarray(self.metric_data["Batch_mIoU"]).mean()
        loss = np.asarray(self.metric_data["Batch_Loss"]).mean()
        if print_only:
            print("Pixel Accuracy: %.4f\tmIoU: %.4f\tLoss: %.4f" % (pixAcc, mIoU, loss))
        return pixAcc, mIoU, loss

    def save_epoch(self):
        """
        Save Data to new dataset named by epoch name
        """             
        if self._path is not None:
            summary_stats = np.asarray(self.get_epoch_statistics())
            with h5py.File(self._path, 'a') as hf:
                # Clear any Cached data from previous first into main
                if len(list(hf['cache'])) > 0:
                    self._flush_to_main()
                
                group_name = self.mode + '/Epoch_' + str(len(list(hf[self.mode])) + 1)
                top_group = hf.create_group(group_name)
                top_group.create_dataset('PixelAcc', data=np.asarray(self.metric_data["Batch_PixelAcc"]))
                top_group.create_dataset('mIoU', data=np.asarray(self.metric_data["Batch_mIoU"]))
                top_group.create_dataset('Loss', data=np.asarray(self.metric_data["Batch_Loss"]))
                top_group.create_dataset('Summary', data=summary_stats)

                #   Flush current data as its now in long term storage and we're ready for next dataset
                self.metric_data = dict(
                        Batch_Loss=[],
                        Batch_PixelAcc=[],
                        Batch_mIoU=[] )

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
                self.metric_data["Batch_PixelAcc"] = hf[mode][group_name]['PixelAcc'][:]
                self.metric_data["Batch_mIoU"] = hf[mode][group_name]['mIoU'][:]
                self.metric_data["Batch_Loss"] = hf[mode][group_name]['Loss'][:]
            return self.metric_data
        else:
            print("No File Specified for Segmentation Metric Manager")

    def max_accuracy(self):
        """
        Returns highest mIoU and PixelWise Accuracy from per epoch summarised data
        @output mIoU, PixelWise Accuracy
        """
        mIoU = 0
        PixelAcc = 0
        if self._path is not None:
            with h5py.File(self._path, 'a') as hf:
                for data in hf['validation']:
                    if data != 'tmp':
                        summary_data = hf['validation/'+data+'/Summary'][:]
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
        prediction = prediction.astype('int64') + 1
        target = target.astype('int64') + 1
        
        pixAcc = 1.0 * np.sum((prediction == target) * (target >= 0)) / (np.spacing(1) + np.sum(target >= 0))
        self.metric_data["Batch_PixelAcc"].append(pixAcc)
    
    def new_epoch(self, mode='training'):
        assert mode == 'training' or mode == 'validation'

        # Cache data if it hasn't been saved yet (probably not a best epoch or
        # something, but the next might be, so we want to keep this data)
        if len(self.metric_data["Batch_PixelAcc"]) > 0 or len(self.metric_data["Batch_mIoU"]) > 0:
            self._cache_data()

        self.mode = mode
        self.metric_data = dict(
            Batch_Loss=[],
            Batch_PixelAcc=[],
            Batch_mIoU=[] )

    def plot_epoch_data(self, epoch_idx):
        """
        This plots all the statistics for an epoch
        """       
        plt.figure(figsize=(18, 5))
        plt.suptitle(self._path + ' Training and Validation Results Epoch' + str(epoch_idx))

        group_name = 'Epoch_' + str(epoch_idx)
        with h5py.File(self._path, 'r') as hf:
            plt.subplot(1,3,1)
            plt.plot(hf['training'][group_name]['PixelAcc'][:])
            plt.plot(hf['validation'][group_name]['PixelAcc'][:])
            plt.legend(["Training", "Validation"])
            plt.title('Batch Pixel Accuracy over Epochs')
            plt.ylabel('% Accuracy')
            plt.xlabel('Iter #')

            plt.subplot(1,3,2)
            plt.plot(hf['training'][group_name]['mIoU'][:])
            plt.plot(hf['validation'][group_name]['mIoU'][:])
            plt.legend(["Training", "Validation"])
            plt.title('Batch mIoU over Epochs')
            plt.ylabel('mIoU')
            plt.xlabel('Iter #')

            plt.subplot(1,3,3)
            plt.plot(hf['training'][group_name]['Loss'][:])
            plt.plot(hf['validation'][group_name]['Loss'][:])
            plt.legend(["Training", "Validation"])
            plt.title('Batch loss over Epochs')
            plt.ylabel('Loss')
            plt.xlabel('Iter #')
            plt.show()

    def _cache_data(self):
        """
        Moves data to temporary location to be permanently saved or deleted later
        """
        if self._path is not None:
            summary_stats = np.asarray(self.get_epoch_statistics())
            with h5py.File(self._path, 'a') as hf:
                n_cached = 0
                if 'cache' in list(hf):
                    if self.mode in list(hf['cache']):
                        n_cached = len(list(hf['cache/'+self.mode]))

                group_name = 'cache/' + self.mode + '/Epoch_' + str(len(list(hf[self.mode])) + n_cached)
                top_group = hf.create_group(group_name)
                top_group.create_dataset('PixelAcc', data=np.asarray(self.metric_data["Batch_PixelAcc"]))
                top_group.create_dataset('mIoU', data=np.asarray(self.metric_data["Batch_mIoU"]))
                top_group.create_dataset('Loss', data=np.asarray(self.metric_data["Batch_Loss"]))
                top_group.create_dataset('Summary', data=summary_stats)
        else:
            print("No File Specified for Segmentation Metric Manager")

    def _flush_to_main(self):
        """
        Moves data from cache to main storage area
        """
        with h5py.File(self._path, 'a') as hf:
            for mode in list(hf['cache']):
                for data in list(hf['cache/'+mode]):
                        hf.copy('cache/'+mode+'/'+data, mode+'/'+data)
                        del hf['cache/'+mode+'/'+data]

class BoundaryBoxMetric(object):
    def __init__(self):
        # @todo
        raise NotImplementedError

class ClassificationMetric(object):
    def __init__(self):
        # @todo
        raise NotImplementedError

if __name__ == "__main__":
    test = np.array([1,2,3,4,5])
    with h5py.File('test.hdf5', 'w') as hf:
        top_grp = hf.create_group('top1')
        subgrp1 = top_grp.create_group('sub1')
        subgrp2 = top_grp.create_dataset('sub2', data=test)
        top_grp['sub1'].create_dataset('subsub1', data=test)
        top_grp['sub1'].create_dataset('subsub2', data=test)

        top_grp = hf.create_group('top2')
        subgrp1 = top_grp.create_group('sub3')
        subgrp2 = top_grp.create_group('sub4')
        subgrp1 = test
        subgrp2 = test

    with h5py.File('test.hdf5', 'r') as hf:
        print(list(hf))
        for data in list(hf['top1']):
            nmpy = hf['top1'][data]
            print(nmpy)
        print(list(hf['top2']))

    with h5py.File('test.hdf5', 'a') as hf:
        for data in list(hf['top2']):
            hf.copy('top2/'+data,'top1/'+data)
            del hf['top2/'+data]
        print(list(hf['top1']))
        print(list(hf['top2']))
