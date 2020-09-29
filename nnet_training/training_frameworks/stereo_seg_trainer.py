#!/usr/bin/env python3

__author__ = "Bryce Ferenczi"
__email__ = "bryce.ferenczi@monashmotorsport.com"

import sys
import time
from pathlib import Path
from typing import Dict, Union
import numpy as np

import torch
import matplotlib.pyplot as plt

from nnet_training.utilities.metrics import SegmentationMetric
from nnet_training.utilities.visualisation import get_color_pallete
from nnet_training.training_frameworks.trainer_base_class import ModelTrainer

__all__ = ['StereoSegTrainer']

class StereoSegTrainer(ModelTrainer):
    def __init__(self, model: torch.nn.Module, optim: torch.optim.Optimizer,
                 loss_fn: Dict[str, torch.nn.Module], lr_cfg: Dict[str, Union[str, float]],
                 dataldr: Dict[str, torch.utils.data.DataLoader],
                 modelpath: Path, checkpoints=True):
        '''
        Initialize the Model trainer giving it a nn.Model, nn.Optimizer and dataloaders as
        a dictionary with Training, Validation and Testing loaders
        '''
        self._loss_function = loss_fn['segmentation']
        self.metric_loggers = {
            'seg': SegmentationMetric(
                19, base_dir=modelpath, main_metric="IoU", savefile='segmentation_data')
        }

        super(StereoSegTrainer, self).__init__(model, optim, dataldr, lr_cfg,
                                               modelpath, checkpoints)

    def _train_epoch(self, max_epoch):

        start_time = time.time()

        for batch_idx, data in enumerate(self._training_loader):
            self._training_loader.dataset.resample_scale()
            cur_lr = self._lr_manager(batch_idx)
            for param_group in self._optimizer.param_groups:
                param_group['lr'] = cur_lr

            # Put both image and target onto device
            left = data['l_img'].to(self._device)
            right = data['r_img'].to(self._device)
            target = data['seg'].to(self._device)

            # Computer loss, use the optimizer object to zero all of the gradients
            # Then backpropagate and step the optimizer
            outputs = self._model(left, right)

            loss = self._loss_function(outputs, target)

            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

            self.metric_loggers['seg'].add_sample(
                torch.argmax(outputs, dim=1, keepdim=True).cpu().data.numpy(),
                target.cpu().data.numpy(),
                loss=loss.item()
            )

            if not batch_idx % 10:
                time_elapsed = time.time() - start_time
                time_remain = time_elapsed / (batch_idx + 1) * \
                    (len(self._training_loader) - (batch_idx + 1))
                sys.stdout.flush()
                sys.stdout.write(f'\rTrain Epoch: [{self.epoch:2d}/{max_epoch:2d}] || '
                                 f'Iter [{batch_idx + 1:4d}/{len(self._validation_loader):4d}] || '
                                 f'lr: {self._lr_manager.get_lr():.8f} || '
                                 f'Loss: {loss.item():.4f} || '
                                 f'Time Elapsed: {time_elapsed:.2f} sec || '
                                 f'Est Time Remain: {time_remain:.2f} sec')


    def _validate_model(self, max_epoch):
        with torch.no_grad():

            start_time = time.time()

            self._training_loader.dataset.resample_scale(True)
            for batch_idx, data in enumerate(self._validation_loader):
                # Put both image and target onto device
                left = data['l_img'].to(self._device)
                right = data['r_img'].to(self._device)
                target = data['seg'].to(self._device)

                outputs = self._model(left, right)

                # Caculate the loss and accuracy for the predictions
                loss = self._loss_function(outputs, target)

                self.metric_loggers['seg'].add_sample(
                    torch.argmax(outputs, dim=1, keepdim=True).cpu().data.numpy(),
                    target.cpu().numpy(),
                    loss=loss.item()
                )

                if not batch_idx % 10:
                    batch_acc = self.metric_loggers['seg'].get_last_batch()
                    time_elapsed = time.time() - start_time
                    time_remain = time_elapsed / (batch_idx + 1) * \
                        (len(self._validation_loader) - (batch_idx + 1))
                    sys.stdout.flush()
                    sys.stdout.write(f'\rValidaton Epoch: [{self.epoch:2d}/{max_epoch:2d}] || '
                                     f'Iter [{batch_idx+1:4d}/{len(self._validation_loader):4d}] ||'
                                     f' Accuracy: {batch_acc:.4f} || Loss: {loss.item():.4f} || '
                                     f'Time Elapsed: {time_elapsed:.2f} sec || '
                                     f'Est Time Remain: {time_remain:.2f} sec')

    def visualize_output(self):
        """
        Forward pass over a testing batch and displays the output
        """
        with torch.no_grad():
            self._model.eval()
            data = next(iter(self._validation_loader))
            left = data['l_img'].to(self._device)
            right = data['r_img'].to(self._device)
            seg_gt = data['seg']

            start_time = time.time()
            seg_pred = self._model(left, right)
            propagation_time = (time.time() - start_time)/self._validation_loader.batch_size

            pred_cpu = torch.argmax(seg_pred, dim=1, keepdim=True).cpu().numpy()

            for i in range(self._validation_loader.batch_size):
                plt.subplot(1, 3, 1)
                plt.imshow(np.moveaxis(left[i, 0:3, :, :].cpu().numpy(), 0, 2))
                plt.xlabel("Base Image")

                plt.subplot(1, 3, 2)
                plt.imshow(get_color_pallete(seg_gt.numpy()[i, :, :]))
                plt.xlabel("Ground Truth Segmentation")

                plt.subplot(1, 3, 3)
                plt.imshow(get_color_pallete(pred_cpu[i, 0, :, :]))
                plt.xlabel("Prediction")

                plt.suptitle(f"Propagation time: {propagation_time}")
                plt.show()

    def plot_data(self):
        super(StereoSegTrainer, self).plot_data()
        self.metric_loggers['seg'].plot_classwise_iou()

if __name__ == "__main__":
    raise NotImplementedError
