#!/usr/bin/env python3

__author__ = "Bryce Ferenczi"
__email__ = "bryce.ferenczi@monashmotorsport.com"

import os
import sys
import h5py
import multiprocessing
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

__all__ = ['MetricBaseClass', 'SegmentationMetric', 'DepthMetric',
           'BoundaryBoxMetric', 'ClassificationMetric']

class MetricBaseClass(object):
    """
    Provides basic functionality for statistics tracking classes
    """
    def __init__(self, savefile: str, base_dir: Path, mode='training'):
        assert mode in ['training', 'validation']
        self.mode = mode
        self.metric_data = dict()

        if savefile != "":
            if savefile[-5:] != '.hdf5':
                savefile = savefile + '.hdf5'
            self._path = base_dir / savefile
            if not os.path.isfile(self._path):
                with h5py.File(self._path, 'a') as hf:
                    hf.create_group('cache')
                    print("Training Statitsics created at ", self._path)
            else:
                with h5py.File(self._path, 'a') as hf:
                    if 'cache' in list(hf):
                        del hf['cache']
        else:
            self._path = None

    def __len__(self):
        """
        Return Number of Epochs Recorded
        """
        with h5py.File(self._path, 'r') as hf:
            n_epochs = 0
            if 'training' in list(hf):
                n_epochs += len(list(hf['training']))
            if 'cache' in list(hf) and 'training' in list(hf['cache']):
                n_epochs += len(list(hf['cache/training']))
            return n_epochs

    def __del__(self):
        with h5py.File(self._path, 'a') as hf:
            if 'cache' in list(hf):
                del hf['cache']

    def save_epoch(self):
        """
        Save Data to new dataset named by epoch name
        """             
        if self._path is not None:
            with h5py.File(self._path, 'a') as hf:
                # Clear any Cached data from previous first into main
                if 'cache' in list(hf) and len(list(hf['cache'])) > 0:
                    self._flush_to_main()

                if self.mode in list(hf):
                    group_name = self.mode + '/Epoch_' + str(len(list(hf[self.mode])) + 1)
                else:
                    group_name = self.mode + '/Epoch_1'

                top_group = hf.create_group(group_name)
                for metric in self.metric_data.keys():
                    top_group.create_dataset(metric, data=np.asarray(self.metric_data[metric]))

                summary_stats = np.asarray(self._get_epoch_statistics(main_metric=False))
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
            with h5py.File(self._path, 'r') as hf:
                group_name = 'Epoch_' + str(epoch_idx)
                for metric in list(hf[mode][group_name]):
                    self.metric_data[metric] = hf[mode][group_name][metric][:]
            return self.metric_data
        else:
            print("No File Specified for Segmentation Metric Manager")
            return None

    def new_epoch(self, mode='training'):
        assert mode in ['training', 'validation']

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
                plt.subplot(1, num_metrics, idx+1)
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
            with h5py.File(self._path, 'a') as hf:
                n_cached = 1
                if 'cache' in list(hf) and self.mode in list(hf['cache']):
                    n_cached = len(list(hf['cache/'+self.mode])) + 1

                if self.mode in list(hf):
                    group_name = 'cache/' + self.mode + '/Epoch_' +\
                        str(len(list(hf[self.mode])) + n_cached)
                else:
                    group_name = 'cache/' + self.mode + '/Epoch_' + str(n_cached)

                top_group = hf.create_group(group_name)
                for metric in self.metric_data.keys():
                    top_group.create_dataset(metric, data=np.asarray(self.metric_data[metric]))

                summary_stats = np.asarray(self._get_epoch_statistics(main_metric=False))
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

            for idx, epoch in enumerate(sorted(list(hf['training']), key=lambda x: int(x[6:]))):
                training_data[idx] = hf['training/'+epoch+'/Summary'][:]

            for idx, epoch in enumerate(sorted(list(hf['validation']), key=lambda x: int(x[6:]))):
                testing_data[idx] = hf['validation/'+epoch+'/Summary'][:]

            print("# Training, ", len(list(hf['training'])),
                  "\t# Validation", len(list(hf['validation'])))
        
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

        with h5py.File(self._path, 'r') as hf:
            training_metrics = {}
            validation_metrics = {}
            for metric in list(hf['training/Epoch_1']):
                if metric != 'Summary':
                    training_metrics[metric] = np.zeros((1,1))
                    validation_metrics[metric] = np.zeros((1,1))

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

            print("# Training, ", len(list(hf['training'])),
                  "\t# Validation", len(list(hf['validation'])))

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

    def _get_epoch_statistics(self, print_only=False, main_metric=True, loss_metric=True):
        """
        Returns Accuracy and Loss Metrics from an Epoch\n
        @todo   get a specified epoch instead of only currently loaded one\n
        @param  main_metric, only main metric\n
        @param  loss_metric, returns recorded loss\n
        @param  print_only, prints stats and does not return values
        """
        raise NotImplementedError

    def max_accuracy(self):
        raise NotImplementedError

    def get_last_batch(self):
        raise NotImplementedError

class SegmentationMetric(MetricBaseClass):
    """
    Accuracy and Loss Staticstics tracking for semantic segmentation networks
    """
    def __init__(self, num_classes, savefile: str, base_dir: Path, mode='training'):
        super(SegmentationMetric, self).__init__(savefile=savefile, base_dir=base_dir, mode=mode)
        self._n_classes = num_classes
        self._reset_metric()

    def _add_sample(self, preds, labels, loss=None):
        """
        Update Accuracy (and Loss) Metrics
        """
        if loss is not None:
            self.metric_data["Batch_Loss"].append(loss)

        labels = labels.astype('int64') + 1
        preds = np.squeeze(preds.astype('int64') + 1, 1)
        
        rcv_px, send_px = multiprocessing.Pipe(False)
        pxthread = multiprocessing.Process(target=self._pixelwise, args=(preds, labels, send_px))
        rcv_iou, send_iou = multiprocessing.Pipe(False)
        iouthread = multiprocessing.Process(target=self._iou, args=(preds, labels, send_iou))
        pxthread.start()
        iouthread.start()
        pxthread.join()
        iouthread.join()
        self.metric_data["Batch_mIoU"].append(rcv_iou.recv())
        self.metric_data["Batch_PixelAcc"].append(rcv_px.recv())
        
    def _get_epoch_statistics(self, print_only=False, main_metric=True, loss_metric=True):
        """
        Returns Accuracy Metrics [pixelwise, mIoU, loss]\n
        @todo   get a specified epoch instead of only currently loaded one\n
        @param  main_metric, returns mIoU and not Pixel Accuracy\n
        @param  loss_metric, returns recorded loss\n
        @param  print_only, prints stats and does not return values
        """ 

        if print_only:
            PixelAcc = np.asarray(self.metric_data["Batch_PixelAcc"]).mean()
            mIoU = np.asarray(self.metric_data["Batch_mIoU"]).mean()
            loss = np.asarray(self.metric_data["Batch_Loss"]).mean()
            print("Pixel Accuracy: %.4f\tmIoU: %.4f\tLoss: %.4f\n" % (PixelAcc, mIoU, loss))
        else:
            ret_val = ()
            if main_metric:
                invariant = np.asarray(self.metric_data["Batch_mIoU"]).mean()
                ret_val += (invariant,)
                if loss_metric:
                    loss = np.asarray(self.metric_data["Batch_Loss"]).mean()
                    ret_val += (loss,)
            else:
                for metric in self.metric_data.keys():
                    mean_data = np.asarray(self.metric_data[metric]).mean()
                    ret_val += (mean_data,)
            return ret_val

    def max_accuracy(self, main_metric=True):
        """
        Returns highest mIoU and PixelWise Accuracy from per epoch summarised data.\n
        @param  main_metric, if true only returns mIoU\n
        @output PixelWise Accuracy, mIoU
        """
        mIoU = 0
        PixelAcc = 0
        if self._path is not None:
            with h5py.File(self._path, 'a') as hf:
                for epoch in hf['validation']:
                    summary_data = hf['validation/'+epoch+'/Summary'][:]
                    if summary_data[1] > mIoU:
                        mIoU = summary_data[1]
                    if summary_data[0] > PixelAcc:
                        PixelAcc = summary_data[0]
                    
        else:
            print("No File Specified for Segmentation Metric Manager")
        if main_metric:
            return mIoU
        else:
            return PixelAcc, mIoU

    def get_last_batch(self, main_metric=True):
        if main_metric:
            return self.metric_data["Batch_mIoU"][-1]
        else:
            return self.metric_data["Batch_mIoU"][-1], self.metric_data["Batch_PixelAcc"][-1]

    def _iou(self, prediction, target, ret_val):
        # Remove classes from unlabeled pixels in gt image.
        # We should not penalize detections in unlabeled portions of the image.
        prediction = prediction * (target > 0).astype(prediction.dtype)

        # Compute area intersection:
        intersection = prediction * (prediction == target)
        area_intersection, _ = np.histogram(intersection, bins=self._n_classes, range=(1, self._n_classes))

        # Compute area union:
        area_pred, _ = np.histogram(prediction, bins=self._n_classes, range=(1, self._n_classes))
        area_lab, _ = np.histogram(target, bins=self._n_classes, range=(1, self._n_classes))
        area_union = area_pred + area_lab - area_intersection
        
        mIoU = (1.0 * area_intersection / (np.spacing(1) + area_union)).mean()
        ret_val.send(mIoU)
    
    def _pixelwise(self, prediction, target, ret_val):
        # Remove classes from unlabeled pixels in gt image.
        # We should not penalize detections in unlabeled portions of the image.
        correct = 1.0 * np.sum((prediction == target) * (target > 0))
        total_pixels = np.spacing(1) + np.sum(target > 0)
        pixAcc = correct / total_pixels
        ret_val.send(pixAcc)
    
    def _reset_metric(self):
        self.metric_data = dict(
            Batch_Loss=[],
            Batch_PixelAcc=[],
            Batch_mIoU=[]
        )

class DepthMetric(MetricBaseClass):
    """
    Accuracy/Error and Loss Staticstics tracking for depth based networks
    """
    def __init__(self, savefile: str, base_dir: Path, mode='training'):
        super(DepthMetric, self).__init__(savefile=savefile, base_dir=base_dir, mode=mode)
        self._reset_metric()

    def _add_sample(self, pred_depth, gt_depth, loss=None):
        if loss is not None:
            self.metric_data["Batch_Loss"].append(loss)
        # pred_depth = pred_depth.squeeze(axis=1) * gt_depth[gt_depth > 0]
        pred_depth = pred_depth.squeeze(axis=1)[gt_depth != -1]
        pred_depth[pred_depth <= 0] = 0.00001
        gt_depth = gt_depth[gt_depth != -1]

        n_pixels = gt_depth.size
        difference = pred_depth-gt_depth
        squared_diff = np.square(difference)
        log_diff = np.log(pred_depth) - np.log(gt_depth)

        self.metric_data['Batch_Absolute_Relative'].append(np.mean(np.absolute(difference)/gt_depth))
        self.metric_data['Batch_Squared_Relative'].append(np.mean(squared_diff/gt_depth))
        self.metric_data['Batch_RMSE_Linear'].append(np.sqrt(np.mean(squared_diff)))
        self.metric_data['Batch_RMSE_Log'].append(np.sqrt(np.mean(np.square(log_diff))))

        eqn1 = np.mean(np.square(log_diff))
        eqn2 = np.square(np.sum(log_diff)) / n_pixels**2
        self.metric_data['Batch_Invariant'].append(eqn1 - eqn2)

    def _get_epoch_statistics(self, print_only=False, main_metric=True, loss_metric=True):
        """
        Returns Accuracy Metrics [scale invariant, absolute relative, squared relative,
        rmse linear, rmse log]\n
        @todo   get a specified epoch instead of only currently loaded one\n
        @param  main_metric, returns scale invariant\n
        @param  loss_metric, returns recorded loss\n
        @param  print_only, prints stats and does not return values
        """ 

        if print_only:
            abs_rel = np.asarray(self.metric_data["Batch_Absolute_Relative"]).mean()
            sqr_rel = np.asarray(self.metric_data["Batch_Squared_Relative"]).mean()
            rmse_lin = np.asarray(self.metric_data["Batch_RMSE_Linear"]).mean()
            rmse_log = np.asarray(self.metric_data["Batch_RMSE_Log"]).mean()
            invariant = np.asarray(self.metric_data["Batch_Invariant"]).mean()
            loss = np.asarray(self.metric_data["Batch_Loss"]).mean()
            print("Absolute Relative: %.4f\tSquared Relative: %.4f\tRMSE Linear: %.4f\t\
                  RMSE Log: %.4f\tScale Invariant: %.4f\tLoss: %.4f\n"
                  % (abs_rel, sqr_rel, rmse_lin, rmse_log, invariant, loss))
        else:
            ret_val = ()
            if main_metric:
                invariant = np.asarray(self.metric_data["Batch_Invariant"]).mean()
                ret_val += (invariant,)
                if loss_metric:
                    loss = np.asarray(self.metric_data["Batch_Loss"]).mean()
                    ret_val += (loss,)
            else:
                for metric in self.metric_data.keys():
                    mean_data = np.asarray(self.metric_data[metric]).mean()
                    ret_val += (mean_data,)
            return ret_val

    def max_accuracy(self, main_metric=True):
        """
        Returns highest scale invariant, absolute relative, squared relative, 
        rmse linear, rmse log Accuracy from per epoch summarised data.\n
        @param  main_metric, if true only returns scale invariant\n
        @output scale invariant, absolute relative, squared relative, rmse linear, rmse log
        """
        invariant = sys.float_info.max
        abs_rel = sys.float_info.max
        sqr_rel = sys.float_info.max
        rmse_lin = sys.float_info.max
        rmse_log = sys.float_info.max

        if self._path is not None:
            with h5py.File(self._path, 'a') as hf:
                for epoch in hf['validation']:
                    summary_data = hf['validation/'+epoch+'/Summary'][:]
                    if summary_data[0] < invariant:
                        invariant = summary_data[0]
                    if summary_data[1] < abs_rel:
                        abs_rel = summary_data[1]
                    if summary_data[2] < sqr_rel:
                        sqr_rel = summary_data[2]
                    if summary_data[3] < rmse_lin:
                        rmse_lin = summary_data[3]
                    if summary_data[4] < rmse_log:
                        rmse_log = summary_data[4]
        else:
            print("No File Specified for Segmentation Metric Manager")
        if main_metric:
            return invariant
        else:
            return invariant, abs_rel, sqr_rel, rmse_lin, rmse_log

    def get_last_batch(self, main_metric=True):
        if main_metric:
            return self.metric_data["Batch_Invariant"][-1]
        else:
            ret_val = ()
            for key in self.metric_data.keys():
                if key != "Batch_Loss":
                    ret_val += (self.metric_data[key][-1],)
            return ret_val
   
    def _reset_metric(self):
        self.metric_data = dict(
            Batch_Loss = [],
            Batch_Absolute_Relative=[],
            Batch_Squared_Relative=[],
            Batch_RMSE_Linear=[],
            Batch_RMSE_Log=[],
            Batch_Invariant=[]
        )

class OpticFlowMetric(MetricBaseClass):
    """
    Accuracy/Error and Loss Staticstics tracking for depth based networks
    """
    def __init__(self, savefile: str, base_dir: Path, mode='training'):
        super(OpticFlowMetric, self).__init__(savefile=savefile, base_dir=base_dir, mode=mode)
        self._reset_metric()

    def _add_sample(self, orig_img, seq_img, flow_pred, flow_target, loss=None):
        """
        @input list of original, prediction and sequence images i.e. [left, right]
        @todo write reconstruction error and endpoint error
        """
        if loss is not None:
            self.metric_data["Batch_Loss"].append(loss)

        self.metric_data["Batch_EPE"].append(1)
        self.metric_data["Batch_SAD"].append(1)
    
    def _get_epoch_statistics(self, print_only=False, main_metric=True, loss_metric=True):
        """
        Returns Accuracy Metrics [scale invariant, absolute relative, squared relative, rmse linear, rmse log]\n
        @todo   get a specified epoch instead of only currently loaded one\n
        @param  main_metric, returns sum abs diff\n
        @param  loss_metric, returns recorded loss\n
        @param  print_only, prints stats and does not return values
        """ 

        if print_only:
            epe = np.asarray(self.metric_data["Batch_EPE"]).mean()
            sad = np.asarray(self.metric_data["Batch_SAD"]).mean()
            loss = np.asarray(self.metric_data["Batch_Loss"]).mean()
            print("End Point Error: %.4f\tSum Aboslute Difference: %.4f\tLoss: %.4f" % (epe, sad, loss))
        else:
            ret_val = ()
            if main_metric:
                invariant = np.asarray(self.metric_data["Batch_SAD"]).mean()
                ret_val += (invariant,)
                if loss_metric:
                    loss = np.asarray(self.metric_data["Batch_Loss"]).mean()
                    ret_val += (loss,)
            else:
                for metric in self.metric_data.keys():
                    mean_data = np.asarray(self.metric_data[metric]).mean()
                    ret_val += (mean_data,)
            return ret_val

    def max_accuracy(self, main_metric=True):
        """
        Returns lowest end point error and sum absolute difference from per epoch summarised data.\n
        @param  main_metric, if true only returns scale invariant\n
        @output scale invariant, absolute relative, squared relative, rmse linear, rmse log
        """
        epe = sys.float_info.max
        sad = sys.float_info.max

        if self._path is not None:
            with h5py.File(self._path, 'a') as hf:
                for epoch in hf['validation']:
                    summary_data = hf['validation/'+epoch+'/Summary'][:]
                    if summary_data[0] < epe:
                        epe = summary_data[0]
                    if summary_data[1] < sad:
                        sad = summary_data[1]

        else:
            print("No File Specified for Segmentation Metric Manager")
        if main_metric:
            return sad
        else:
            return sad, epe

    def get_last_batch(self, main_metric=True):
        if main_metric:
            return self.metric_data["Batch_SAD"][-1]
        else:
            ret_val = ()
            for key in self.metric_data.keys():
                if key != "Batch_Loss":
                    ret_val += (self.metric_data[key][-1],)
            return ret_val
        
    def _reset_metric(self):
        self.metric_data = dict(
            Batch_Loss=[],
            Batch_SAD=[],
            Batch_EPE=[]
        )

class BoundaryBoxMetric(MetricBaseClass):
    def __init__(self, savefile: str, base_dir: Path, mode='training'):
        super(BoundaryBoxMetric, self).__init__(savefile=savefile, base_dir=base_dir, mode=mode)
        raise NotImplementedError

    def _add_sample(self, pred_depth, gt_depth, loss=None):
        raise NotImplementedError
    
    def _get_epoch_statistics(self, print_only=False, main_metric=True, loss_metric=True):
        raise NotImplementedError

    def max_accuracy(self, main_metric=True):
        raise NotImplementedError

    def _reset_metric(self):
        raise NotImplementedError

class ClassificationMetric(MetricBaseClass):
    def __init__(self, savefile: str, base_dir: Path, mode='training'):
        super(ClassificationMetric, self).__init__(savefile=savefile, base_dir=base_dir, mode=mode)
        raise NotImplementedError

    def _add_sample(self, pred_depth, gt_depth, loss=None):
        raise NotImplementedError

    def _get_epoch_statistics(self, print_only=False, main_metric=True, loss_metric=True):
        raise NotImplementedError

    def max_accuracy(self, main_metric=True):
        raise NotImplementedError

    def _reset_metric(self):
        raise NotImplementedError

if __name__ == "__main__":
    filename = "MonoSF_SegNet3_FlwExt1_FlwEst1_CtxNet1_Adam_Fcl_Uflw_HRes_flow"
    metric = MetricBaseClass(filename=filename)
    metric.plot_summary_data()
