#!/usr/bin/env python3

__author__ = "Bryce Ferenczi"
__email__ = "bryce.ferenczi@monash.edu"

import os
import time
import sys
from pathlib import Path
from typing import Dict, Union

import numpy as np
import torch
import torchvision

import matplotlib.pyplot as plt

from nnet_training.utilities.metrics import get_loggers
from nnet_training.utilities.lr_scheduler import LRScheduler
from nnet_training.utilities.visualisation import get_color_pallete, flow_to_image

__all__ = ['ModelTrainer']

MIN_DEPTH = 0.
MAX_DEPTH = 80.

class ModelTrainer(object):
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
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._training_loader = dataloaders["Training"]
        self._validation_loader = dataloaders["Validation"]

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
            checkpoint = torch.load(path, map_location=torch.device(self._device))
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

        self._lr_manager.set_epochs(nepochs=n_epochs, iters_per_epoch=len(self._training_loader))

        while self.epoch < max_epoch:
            self.epoch += 1
            epoch_start_time = time.time()

            # Calculate the training loss, training duration for each epoch, and validation accuracy
            for metric in self.metric_loggers.values():
                metric.new_epoch('training')

            torch.cuda.empty_cache()

            self._model.train()
            self._train_epoch(max_epoch)

            for metric in self.metric_loggers.values():
                metric.new_epoch('validation')

            torch.cuda.empty_cache()

            self._model.eval()
            self._validate_model(max_epoch)

            epoch_duration = time.time() - epoch_start_time

            if self._checkpoints:
                for key, logger in self.metric_loggers.items():
                    epoch_acc, _ = logger.get_current_statistics(
                        main_metric=True, loss_metric=False)
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
        start_time = time.time()

        for batch_idx, batch_data in enumerate(self._training_loader):
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

            self.log_output_performance(forward, batch_data, losses)

            if not batch_idx % 10:
                time_elapsed = time.time() - start_time
                time_remain = time_elapsed / (batch_idx + 1) * \
                    (len(self._training_loader) - (batch_idx + 1))

                sys.stdout.write(f'\rTrain Epoch: [{self.epoch:2d}/{max_epoch:2d}] || '
                                 f'Iter [{batch_idx + 1:4d}/{len(self._training_loader):4d}] || '
                                 f'lr: {self._lr_manager.get_lr():.2e} || '
                                 f'Loss: {loss.item():.4f} || '
                                 f'Time Elapsed: {time_elapsed:.2f} s '
                                 f'Remain: {time_remain:.2f} s')
                sys.stdout.write("\033[K")
                sys.stdout.flush()

    @torch.no_grad()
    def _validate_model(self, max_epoch):
        start_time = time.time()

        for batch_idx, batch_data in enumerate(self._validation_loader):
            # Put both image and target onto device
            self._data_to_gpu(batch_data)

            # Caculate the loss and accuracy for the predictions
            forward = self._model(**batch_data)
            losses = self.calculate_losses(forward, batch_data)

            self.log_output_performance(forward, batch_data, losses)

            if not batch_idx % 10:
                sys.stdout.write(f'\rValidaton Epoch: [{self.epoch:2d}/{max_epoch:2d}] || '
                                 f'Iter: [{batch_idx+1:4d}/{len(self._validation_loader):4d}]')

                for logger in self.metric_loggers.values():
                    sys.stdout.write(f" || {logger.main_metric}: {logger.get_last_batch():.4f}")

                time_elapsed = time.time() - start_time
                time_remain = time_elapsed/(batch_idx+1)*\
                    (len(self._validation_loader)-batch_idx+1)
                sys.stdout.write(f' || Time Elapsed: {time_elapsed:.1f} s'\
                                 f' Remain: {time_remain:.1f} s')
                sys.stdout.write("\033[K")
                sys.stdout.flush()

    @staticmethod
    def _data_to_gpu(data):
        # Put both image and target onto device
        cuda_s = torch.cuda.Stream()
        with torch.cuda.stream(cuda_s):
            for key in data:
                if key in ['l_img', 'l_seq', 'seg', 'l_disp', 'r_img', 'r_seq', 'r_disp']:
                    data[key] = data[key].cuda(non_blocking=True)

            if all(key in data.keys() for key in ["flow", "flow_mask"]):
                data['flow_gt'] = {"flow": data['flow'].cuda(non_blocking=True),
                                   "flow_mask": data['flow_mask'].cuda(non_blocking=True)}
                del data['flow'], data['flow_mask']
            else:
                data['flow_gt'] = None
        cuda_s.synchronize()

    @torch.no_grad()
    def log_output_performance(self, nnet_outputs: Dict[str, torch.Tensor],
                               batch_data: Dict[str, torch.Tensor],
                               losses: Dict[str, torch.Tensor]):
        """
        Calculates different metrics for each of the output types
        """
        if 'flow' in self.metric_loggers:
            self.metric_loggers['flow'].add_sample(
                batch_data['l_img'], batch_data['l_seq'], nnet_outputs['flow'][0],
                batch_data['flow_gt'], loss=losses['flow'].item()
            )

        if 'seg' in self.metric_loggers:
            self.metric_loggers['seg'].add_sample(
                torch.argmax(nnet_outputs['seg'], dim=1, keepdim=True),
                batch_data['seg'], loss=losses['seg'].item()
            )

        if 'depth' in self.metric_loggers:
            self.metric_loggers['depth'].add_sample(
                nnet_outputs['depth'], batch_data['l_disp'],
                loss=losses['depth'].item()
            )

    def calculate_losses(self, nnet_outputs: Dict[str, torch.Tensor],
                         batch_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Calculates losses for different outputs and loss functions
        """
        losses = {}

        for loss_type, loss_func in self._loss_fn.items():
            losses[loss_type] = loss_func(preds=nnet_outputs, targets=batch_data)

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

    @torch.no_grad()
    def visualize_output(self):
        """
        Displays the outputs of the network
        """
        self._model.eval()

        batch_data = next(iter(self._validation_loader))
        self._data_to_gpu(batch_data)

        start_time = time.time()
        forward = self._model(**batch_data)
        propagation_time = (time.time() - start_time)/self._validation_loader.batch_size

        if 'depth' in forward and 'l_disp' in batch_data:
            if isinstance(forward['depth'], list):
                forward['depth'] = forward['depth'][0]
            depth_pred_cpu = forward['depth'].type(torch.float32).detach().cpu().numpy()
            depth_gt_cpu = batch_data['l_disp'].cpu().numpy()

        if 'seg' in forward:
            seg_pred_cpu = torch.argmax(forward['seg'], dim=1).cpu().numpy()

        if 'flow' in forward:
            np_flow_12 = forward['flow'][0].detach().type(torch.float32).cpu().numpy()

        if hasattr(self._validation_loader.dataset, 'img_normalize'):
            img_mean = self._validation_loader.dataset.img_normalize.mean
            img_std = self._validation_loader.dataset.img_normalize.std
            inv_mean = [-mean / std for mean, std in zip(img_mean, img_std)]
            inv_std = [1 / std for std in img_std]
            img_norm = torchvision.transforms.Normalize(inv_mean, inv_std)
        else:
            img_norm = torchvision.transforms.Normalize([0, 0, 0], [1, 1, 1])

        for i in range(self._validation_loader.batch_size):
            plt.subplot(2, 4, 1)
            plt.imshow(np.moveaxis(img_norm(batch_data['l_img'][i]).cpu().numpy(), 0, 2))
            plt.xlabel("Base Image")

            if 'l_seq' in batch_data:
                plt.subplot(2, 4, 2)
                plt.imshow(np.moveaxis(img_norm(batch_data['l_seq'][i]).cpu().numpy(), 0, 2))
                plt.xlabel("Sequential Image")

            if 'seg' in forward and 'seg' in batch_data:
                plt.subplot(2, 4, 5)
                plt.imshow(get_color_pallete(batch_data['seg'].cpu().numpy()[i]))
                plt.xlabel("Ground Truth Segmentation")

                plt.subplot(2, 4, 6)
                plt.imshow(get_color_pallete(seg_pred_cpu[i]))
                plt.xlabel("Predicted Segmentation")

            if batch_data['flow_gt'] is not None:
                plt.subplot(2, 4, 3)
                plt.imshow(flow_to_image(
                    batch_data['flow_gt']['flow'].cpu().numpy()[i].transpose([1, 2, 0])))
                plt.xlabel("Ground Truth Flow")

            if 'flow' in forward:
                plt.subplot(2, 4, 7)
                plt.imshow(flow_to_image(np_flow_12[i].transpose([1, 2, 0])))
                plt.xlabel("Predicted Flow")

            if 'depth' in forward and 'l_disp' in batch_data:
                plt.subplot(2, 4, 4)
                plt.imshow(depth_gt_cpu[i], cmap='magma', vmin=MIN_DEPTH, vmax=MAX_DEPTH)
                plt.xlabel("Ground Truth Disparity")

                plt.subplot(2, 4, 8)
                plt.imshow(depth_pred_cpu[i, 0], cmap='magma', vmin=MIN_DEPTH, vmax=MAX_DEPTH)
                plt.xlabel("Predicted Depth")

            plt.suptitle("Propagation time: " + str(propagation_time))
            plt.show()

if __name__ == "__main__":
    raise NotImplementedError
