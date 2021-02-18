#!/usr/bin/env python3

__author__ = "Bryce Ferenczi"
__email__ = "bryce.ferenczi@monash.edu"

import os
import time
import sys
from pathlib import Path
from typing import Dict
from typing import Union

import numpy as np
import torch
import torchvision

import matplotlib.pyplot as plt

from nnet_training.statistics import get_loggers
from nnet_training.utilities.lr_scheduler import LRScheduler
from nnet_training.utilities.visualisation import get_color_pallete
from nnet_training.utilities.visualisation import flow_to_image
from nnet_training.utilities.visualisation import apply_bboxes
from nnet_training.utilities.visualisation import get_panoptic_image
from nnet_training.utilities.panoptic_post_processing import get_panoptic_segmentation
from nnet_training.utilities.panoptic_post_processing import get_instance_segmentation
from nnet_training.utilities.panoptic_post_processing import compare_centers
from nnet_training.datasets.cityscapes_dataset import CityScapesDataset

__all__ = ['ModelTrainer']

MIN_DEPTH = 0.
MAX_DEPTH = 80.

class ModelTrainer():
    """
    Base class that various model trainers inherit from
    """
    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                 loss_fn: Dict[str, torch.nn.Module],
                 dataloaders: Dict[str, torch.utils.data.DataLoader],
                 lr_cfg: Dict[str, Union[str, float]], basepath: Path,
                 logger_cfg: Dict[str, str], checkpoints=True):
        '''
        Initialize the Model trainer giving it a nn.Model, nn.Optimizer and dataloaders as
        a dictionary with Training, Validation and Testing loaders
        '''
        self._dataloaders = dataloaders

        self.epoch = 0

        self.metric_loggers = get_loggers(logger_cfg, basepath)

        self._loss_fn = loss_fn
        self._scaler = torch.cuda.amp.GradScaler()

        self._model, self._optimizer = model.cuda(), optimizer

        self._lr_manager = LRScheduler(**lr_cfg)

        self._checkpoints = checkpoints

        if not os.path.isdir(basepath):
            os.makedirs(basepath)

        self._basepath = basepath

        if self._checkpoints:
            self.load_checkpoint(self._basepath / (self._model.modelname+"_latest.pth"))
        elif os.path.isfile(self._basepath / (self._model.modelname+"_latest.pth")):
            sys.stdout.write("\nWarning: Previous Checkpoint Exists and Checkpoints arent enabled!")
        else:
            sys.stdout.write("\nStarting From Scratch without Checkpoints!")

    def get_learning_rate(self) -> float:
        """
        Returns current learning rate of manager
        """
        return self._lr_manager.get_lr()

    def set_learning_rate(self, new_lr: float) -> None:
        """
        Sets new base learning rate for manager
        """
        self._lr_manager.base_lr = new_lr
        if self._lr_manager.mode == 'constant':
            self._lr_manager.target_lr = new_lr

    def load_checkpoint(self, path: Path):
        '''
        Loads previous progress of the model if available
        '''
        if os.path.isfile(path):
            #Load Checkpoint
            checkpoint = torch.load(path,
                map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            self._model.load_state_dict(checkpoint['model_state_dict'])
            self._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epoch = checkpoint['epochs']
            sys.stdout.write(f"\nCheckpoint loaded from {str(path)} "
                             f"starting from epoch: {self.epoch}\n")
        else:
            #Raise Error if it does not exist
            sys.stdout.write("\nCheckpoint Does Not Exist, Starting From Scratch!")

    def save_checkpoint(self, path: Path, metrics=False):
        '''
        Saves progress of the model
        '''
        sys.stdout.write("\nSaving Model")
        if metrics:
            for metric in self.metric_loggers.values():
                metric.save_epoch()

        torch.save({
            'model_state_dict'    : self._model.state_dict(),
            'optimizer_state_dict': self._optimizer.state_dict(),
            'epochs'              : self.epoch
        }, path)

    def write_summary(self):
        """
        Writes a brief summary of the current state of an experiment in a text file
        """
        with open(self._basepath / "Summary.txt", "w") as txt_file:
            txt_file.write(f"{self._model.modelname} Summary, # Epochs: {self.epoch}\n")
            for key, metric in self.metric_loggers.items():
                value = metric.max_accuracy(main_metric=True)[1]
                txt_file.write(f"Objective: {key}\tMetric: {metric.main_metric}"
                               f"\tValue: {value:.3f}\n")

    def train_model(self, n_epochs):
        """
        Train the model for a number of epochs
        """
        train_start_time = time.time()

        max_epoch = self.epoch + n_epochs

        self._lr_manager.set_epochs(
            nepochs=n_epochs, iters_per_epoch=len(self._dataloaders["Training"]))

        while self.epoch < max_epoch:
            self.epoch += 1
            epoch_start_time = time.time()

            # Calculate the training loss, training duration for each epoch, and validation accuracy
            for metric in self.metric_loggers.values():
                metric.new_epoch('training')

            torch.cuda.empty_cache()

            self._train_epoch(max_epoch)

            for metric in self.metric_loggers.values():
                metric.new_epoch('validation')

            torch.cuda.empty_cache()

            self._validate_model(max_epoch)

            epoch_duration = time.time() - epoch_start_time

            if self._checkpoints:
                for key, logger in self.metric_loggers.items():
                    epoch_acc, _ = logger.get_current_statistics(
                        main_only=True, return_loss=False)
                    prev_best = logger.max_accuracy(main_metric=True)
                    if prev_best is None or \
                        prev_best[0](epoch_acc[0], prev_best[1]) == epoch_acc[0]:
                        filename = f"{self._model.modelname}_{key}.pth"
                        self.save_checkpoint(self._basepath / filename, metrics=False)

                self.save_checkpoint(
                    self._basepath / f"{self._model.modelname}_latest.pth", metrics=True)

                self.write_summary()

            sys.stdout.write(f'\rEpoch {self.epoch} Finished, Time: {epoch_duration}s\n')
            sys.stdout.write("\033[K")
            sys.stdout.flush()

        train_end_time = time.time()

        print(f"\nTotal Traning Time: \t{train_end_time - train_start_time}")

    def _train_epoch(self, max_epoch):
        self._model.train()
        start_time = time.time()
        dataloader = self._dataloaders["Training"]

        for batch_idx, batch_data in enumerate(dataloader):
            cur_lr = self._lr_manager(batch_idx)
            for param_group in self._optimizer.param_groups:
                param_group['lr'] = cur_lr

            self._data_to_gpu(batch_data)

            # Computer loss, use the optimizer object to zero all of the gradients
            # Then backpropagate and step the optimizer
            forward = self._model(**batch_data)
            losses = self.calculate_losses(forward, batch_data)

            # Accumulate losses
            loss = 0
            for key in losses:
                loss += losses[key]

            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

            with torch.no_grad():
                for key in self.metric_loggers:
                    self.metric_loggers[key].add_sample(forward, batch_data, losses[key].item())

            if not batch_idx % 10:
                time_elapsed = time.time() - start_time
                time_remain = time_elapsed/(batch_idx+1)*(len(dataloader)-(batch_idx+1))

                sys.stdout.write(
                    f'\rTrain Epoch: [{self.epoch:2d}/{max_epoch:2d}] || '
                    f'Iter [{batch_idx+1:4d}/{len(dataloader):4d}] || '
                    f'lr: {self._lr_manager.get_lr():.2e} || Loss: {loss.item():.3f} || '
                    f'Time Elapsed: {time_elapsed:.1f} s || Remain: {time_remain:.1f} s\033[K')
                sys.stdout.flush()

    @torch.no_grad()
    def _validate_model(self, max_epoch):
        # TODO Fix bbox issues when using model.eval() for DETR model
        self._model.eval()
        start_time = time.time()
        dataloader = self._dataloaders["Validation"]

        for batch_idx, batch_data in enumerate(dataloader):
            # Put both image and target onto device
            self._data_to_gpu(batch_data)

            # Caculate the loss and accuracy for the predictions
            forward = self._model(**batch_data)
            losses = self.calculate_losses(forward, batch_data)

            for key in self.metric_loggers:
                self.metric_loggers[key].add_sample(forward, batch_data, losses[key].item())

            if not batch_idx % 10:
                sys.stdout.write(f'\rValidaton Epoch: [{self.epoch:2d}/{max_epoch:2d}] || '
                                 f'Iter: [{batch_idx+1:4d}/{len(dataloader):4d}]')

                for logger in self.metric_loggers.values():
                    sys.stdout.write(f" || {logger.main_metric}: {logger.get_last_batch():.3f}")

                time_elapsed = time.time() - start_time
                time_remain = time_elapsed/(batch_idx+1)*(len(dataloader)-batch_idx+1)
                sys.stdout.write(
                    f' || Time Elapsed: {time_elapsed:.1f} s || Remain: {time_remain:.1f} s\033[K')
                sys.stdout.flush()

    @staticmethod
    def _data_to_gpu(data):
        # Put both image and target onto device
        cuda_s = torch.cuda.Stream()
        with torch.cuda.stream(cuda_s):
            for key in data:
                if key in ['bboxes', 'labels']:
                    for i, elem in enumerate(data[key]):
                        data[key][i] = elem.cuda(non_blocking=True)
                elif isinstance(data[key], torch.Tensor):
                    data[key] = data[key].cuda(non_blocking=True)
        cuda_s.synchronize()

    def calculate_losses(self, nnet_outputs: Dict[str, torch.Tensor],
                         batch_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Calculates losses for different outputs and loss functions
        """
        losses = {}

        for loss_type, loss_func in self._loss_fn.items():
            losses[loss_type] = loss_func(predictions=nnet_outputs, targets=batch_data)

        return losses

    def plot_data(self):
        """
        Plots the main summary data for each of the metric loggers.\n
        You must overide this if you want some extra metric plots
        e.g. classwise segmentation iou\n
        (don't forget to call this after your override)
        """
        for metric in self.metric_loggers.values():
            metric.plot_summary_data()

        if 'seg' in self.metric_loggers.keys():
            self.metric_loggers['seg'].plot_classwise_iou()
            self.metric_loggers['seg'].display_conf_mat()

    @staticmethod
    def col_maj_2_row_maj(rows, cols, col_maj_idx):
        """
        Converts a column major index to a row major index for matplotlib.pyplot.subplot.
        Row = index // columns, Column = index % columns
        """
        crow = (col_maj_idx-1) % rows
        ccol = (col_maj_idx-1) // rows
        new_idx = crow * cols + ccol
        return rows, cols, new_idx + 1

    @staticmethod
    def show_segmentation(batch_data, nnet_outputs, batch_size, img_norm):
        """
        Displays Segmentation Prediction versus ground truth in window
        """
        plt.figure("Semantic Segmentation Estimation")
        seg_pred_cpu = torch.argmax(nnet_outputs['seg'], dim=1).cpu().numpy()

        for i in range(batch_size):
            plt.subplot(*ModelTrainer.col_maj_2_row_maj(3, batch_size, 3*i+1))
            plt.imshow(np.moveaxis(img_norm(batch_data['l_img'][i]).cpu().numpy(), 0, 2))
            plt.xlabel("Input Image")

            plt.subplot(*ModelTrainer.col_maj_2_row_maj(3, batch_size, 3*i+2))
            plt.imshow(get_color_pallete(batch_data['seg'].cpu().numpy()[i]))
            plt.xlabel("Ground Truth Segmentation")

            plt.subplot(*ModelTrainer.col_maj_2_row_maj(3, batch_size, 3*i+3))
            plt.imshow(get_color_pallete(seg_pred_cpu[i]))
            plt.xlabel("Predicted Segmentation")

        plt.suptitle("Segmentation Prediction versus Ground Truth")
        plt.show(block=False)

    @staticmethod
    def show_flow(batch_data, nnet_outputs, batch_size, img_norm):
        """
        Displays flow prediction versus ground truth in window
        """
        plt.figure("Optical Flow Estimation")
        if isinstance(nnet_outputs['flow'], list):
            np_flow_12 = batch_data['flow'][0].cpu().numpy()
        else:
            np_flow_12 = batch_data['flow'].cpu().numpy()

        plt_cols = batch_size * 2
        for i in range(plt_cols):
            plt.subplot(*ModelTrainer.col_maj_2_row_maj(3, batch_size, 3*i+1))
            plt.imshow(np.moveaxis(img_norm(batch_data['l_img'][i]).cpu().numpy(), 0, 2))
            plt.xlabel("Input Image @ t")

            plt.subplot(*ModelTrainer.col_maj_2_row_maj(3, batch_size, 3*i+1))
            plt.imshow(np.moveaxis(img_norm(batch_data['l_seq'][i]).cpu().numpy(), 0, 2))
            plt.xlabel("Input Image @ t + 1")

            if 'flow' in batch_data:
                plt.subplot(*ModelTrainer.col_maj_2_row_maj(3, batch_size, 3*i+2))
                plt.imshow(flow_to_image(
                        batch_data['flow'].cpu().numpy()[i].transpose([1, 2, 0])))
                plt.xlabel("Ground Truth Flow")

            plt.subplot(*ModelTrainer.col_maj_2_row_maj(3, batch_size, 3*i+3))
            plt.imshow(flow_to_image(np_flow_12[i].transpose([1, 2, 0])))
            plt.xlabel("Predicted Flow")

        plt.suptitle("Optical Flow predictions versus Ground Truth")
        plt.show(block=False)

    @staticmethod
    def show_depth(batch_data, nnet_outputs, batch_size, img_norm):
        """
        Displays depth prediction versus ground truth in window
        """
        plt.figure("Depth Estimation")
        if isinstance(nnet_outputs['depth'], list):
            depth_pred_cpu = nnet_outputs['depth'][0].cpu().numpy()
        else:
            depth_pred_cpu = nnet_outputs['depth'].cpu().numpy()

        depth_gt_cpu = batch_data['l_disp'].cpu().numpy()

        for i in range(batch_size):
            plt.subplot(*ModelTrainer.col_maj_2_row_maj(3, batch_size, 3*i+1))
            plt.imshow(np.moveaxis(img_norm(batch_data['l_img'][i]).cpu().numpy(), 0, 2))
            plt.xlabel("Input Image")

            plt.subplot(*ModelTrainer.col_maj_2_row_maj(3, batch_size, 3*i+2))
            plt.imshow(depth_gt_cpu[i], cmap='magma', vmin=MIN_DEPTH, vmax=MAX_DEPTH)
            plt.xlabel("Ground Truth Disparity")

            plt.subplot(*ModelTrainer.col_maj_2_row_maj(3, batch_size, 3*i+3))
            plt.imshow(depth_pred_cpu[i, 0], cmap='magma', vmin=MIN_DEPTH, vmax=MAX_DEPTH)
            plt.xlabel("Predicted Depth")

        plt.suptitle("Depth predictions versus Ground Truth")
        plt.show(block=False)

    @staticmethod
    def show_bboxes(batch_data, nnet_outputs, batch_size, img_norm):
        """
        Object detection versus ground truth in window
        """
        plt.figure("Object Detection Estimation")
        class_pred_cpu = torch.argmax(nnet_outputs['logits'], dim=2).cpu().numpy()

        for i in range(batch_size):
            plt.subplot(*ModelTrainer.col_maj_2_row_maj(2, batch_size, 2*i+1))
            base_img = np.moveaxis(img_norm(batch_data['l_img'][i]).cpu().numpy(), 0, 2)
            plt.imshow(apply_bboxes(base_img,
                batch_data['bboxes'][i].cpu().numpy(),
                batch_data['labels'][i].cpu().numpy()))
            plt.xlabel("Ground Truth Objects")

            plt.subplot(*ModelTrainer.col_maj_2_row_maj(2, batch_size, 2*i+2))
            pred_labels = class_pred_cpu[i]
            pred_boxes = nnet_outputs['bboxes'][i].cpu().numpy()
            plt.imshow(apply_bboxes(base_img, pred_boxes, pred_labels))
            plt.xlabel("Predicted Objects")

        plt.suptitle("Boundary Box Predictions versus Ground Truth")
        plt.show(block=False)

    @staticmethod
    def show_instance(batch_data, nnet_outputs, batch_size, img_norm):
        """
        Instance segmentation versus ground truth in window
        """
        plt.figure("Instance Prediction Estimation")
        seg_pred = torch.argmax(nnet_outputs['seg'], dim=1)

        for i in range(batch_size):
            plt.subplot(*ModelTrainer.col_maj_2_row_maj(3, batch_size, 3*i+1))
            plt.imshow(np.moveaxis(img_norm(batch_data['l_img'][i]).cpu().numpy(), 0, 2))
            plt.xlabel("Input Image")

            instance_gt, _ = get_instance_segmentation(
                batch_data['seg'][i],
                batch_data['center'][i].unsqueeze(0),
                batch_data['offset'][i].unsqueeze(0),
                CityScapesDataset.cityscapes_things, nms_kernel=7)

            plt.subplot(*ModelTrainer.col_maj_2_row_maj(3, batch_size, 3*i+2))
            plt.imshow(instance_gt.squeeze(0).cpu().numpy())
            plt.xlabel("Ground Truth Instances")

            instance_pred, _ = get_instance_segmentation(
                seg_pred[i].unsqueeze(0),
                nnet_outputs['center'][i].unsqueeze(0),
                nnet_outputs['offset'][i].unsqueeze(0),
                CityScapesDataset.cityscapes_things, nms_kernel=7)

            plt.subplot(*ModelTrainer.col_maj_2_row_maj(3, batch_size, 3*i+3))
            plt.imshow(instance_pred.squeeze(0).cpu().numpy())
            plt.xlabel("Predicted Instances")

        plt.suptitle("Instance Prediction versus Ground Truth")
        plt.show(block=False)

    @staticmethod
    def show_offsets(batch_data, nnet_outputs, batch_size, img_norm):
        """
        Center Offset Prediction versus ground truth in window
        """
        plt.figure("Center Offset Estimation")

        for i in range(batch_size):
            plt.subplot(*ModelTrainer.col_maj_2_row_maj(3, batch_size, 3*i+1))
            plt.imshow(np.moveaxis(img_norm(batch_data['l_img'][i]).cpu().numpy(), 0, 2))
            plt.xlabel("Input Image")

            plt.subplot(*ModelTrainer.col_maj_2_row_maj(3, batch_size, 3*i+2))
            plt.imshow(flow_to_image(batch_data['offset'][i].cpu().numpy().transpose([1, 2, 0])))
            plt.xlabel("Ground Truth Offsets")

            plt.subplot(*ModelTrainer.col_maj_2_row_maj(3, batch_size, 3*i+3))
            plt.imshow(flow_to_image(nnet_outputs['offset'][i].cpu().numpy().transpose([1, 2, 0])))
            plt.xlabel("Predicted Offsets")

        plt.suptitle("Instance Offset from Center versus Ground Truth")
        plt.show(block=False)

    @staticmethod
    def show_centers(batch_data, nnet_outputs, batch_size, img_norm):
        """
        Center Point Prediction versus ground truth in window
        """
        plt.figure("Center Point Estimation")

        for i in range(batch_size):
            plt.subplot(*ModelTrainer.col_maj_2_row_maj(3, batch_size, 3*i+1))
            plt.imshow(np.moveaxis(img_norm(batch_data['l_img'][i]).cpu().numpy(), 0, 2))
            plt.xlabel("Input Image")

            plt.subplot(*ModelTrainer.col_maj_2_row_maj(3, batch_size, 3*i+2))
            plt.imshow(batch_data['center'][i].squeeze(0).cpu().numpy())
            plt.xlabel("Ground Truth Centers")

            plt.subplot(*ModelTrainer.col_maj_2_row_maj(3, batch_size, 3*i+3))
            plt.imshow(nnet_outputs['center'][i].squeeze(0).cpu().numpy())
            plt.xlabel("Predicted Centers")

        plt.suptitle("Instance Center Prediction versus Ground Truth")
        plt.show(block=False)

    @staticmethod
    def show_panoptic(batch_data, nnet_outputs, batch_size, img_norm):
        """
        Panoptic Prediction versus ground truth in window
        """
        plt.figure("Panoptic Prediction Estimation")
        seg_pred = torch.argmax(nnet_outputs['seg'], dim=1)

        for i in range(batch_size):
            plt.subplot(*ModelTrainer.col_maj_2_row_maj(3, batch_size, 3*i+1))
            plt.imshow(np.moveaxis(img_norm(batch_data['l_img'][i]).cpu().numpy(), 0, 2))
            plt.xlabel("Input Image")

            panoptic_gt, _ = get_panoptic_segmentation(
                batch_data['seg'][i],
                batch_data['center'][i].unsqueeze(0),
                batch_data['offset'][i].unsqueeze(0),
                CityScapesDataset.cityscapes_things,
                1000, 2048, 1000*255, nms_kernel=7)

            plt.subplot(*ModelTrainer.col_maj_2_row_maj(3, batch_size, 3*i+2))
            plt.imshow(get_panoptic_image(panoptic_gt.squeeze(0).cpu().numpy(), 1000))
            plt.xlabel("Ground Truth Panoptic")

            panoptic_pred, _ = get_panoptic_segmentation(
                seg_pred[i].unsqueeze(0),
                nnet_outputs['center'][i].unsqueeze(0),
                nnet_outputs['offset'][i].unsqueeze(0),
                CityScapesDataset.cityscapes_things,
                1000, 2048, 1000*255, nms_kernel=7)

            plt.subplot(*ModelTrainer.col_maj_2_row_maj(3, batch_size, 3*i+3))
            plt.imshow(get_panoptic_image(panoptic_pred.squeeze(0).cpu().numpy(), 1000))
            plt.xlabel("Predicted Panoptic")

        plt.suptitle("Panoptic Prediction versus Ground Truth")
        plt.show(block=False)

    @torch.no_grad()
    def visualize_output(self):
        """
        Displays the outputs of the network
        """
        # TODO Fix bbox issues when using model.eval() for DETR model
        self._model.eval()

        dataloader = self._dataloaders["Validation"]
        batch_data = next(iter(dataloader))
        self._data_to_gpu(batch_data)

        start_time = time.time()
        forward = self._model(**batch_data)
        print(f"Propagation time {(time.time() - start_time) / dataloader.batch_size:.3f}")

        if 'img_normalize' in dataloader.dataset.augmentations:
            img_mean = dataloader.dataset.augmentations['img_normalize'].mean
            img_std = dataloader.dataset.augmentations['img_normalize'].std
            img_norm = torchvision.transforms.Normalize(
                [-mean / std for mean, std in zip(img_mean, img_std)],
                [1 / std for std in img_std])
        else:
            img_norm = torchvision.transforms.Normalize([0, 0, 0], [1, 1, 1])

        if 'depth' in forward and 'l_disp' in batch_data:
            self.show_depth(batch_data, forward, dataloader.batch_size, img_norm)

        if all('seg' in data for data in [forward, batch_data]):
            self.show_segmentation(batch_data, forward, dataloader.batch_size, img_norm)

        if 'flow' in forward:
            self.show_flow(batch_data, forward, dataloader.batch_size, img_norm)

        if all(key in batch_data for key in ['bboxes', 'labels']) and \
            all(key in forward for key in ['bboxes', 'logits']):
            self.show_bboxes(batch_data, forward, dataloader.batch_size, img_norm)

        if all(key in batch_data for key in ['center', 'offset', 'seg']):
            self.show_panoptic(batch_data, forward, dataloader.batch_size, img_norm)
            self.show_instance(batch_data, forward, dataloader.batch_size, img_norm)
            self.show_offsets(batch_data, forward, dataloader.batch_size, img_norm)
            self.show_centers(batch_data, forward, dataloader.batch_size, img_norm)
            compare_centers(batch_data['center_points'], batch_data['center'])

        plt.show(block=True)

if __name__ == "__main__":
    raise NotImplementedError
