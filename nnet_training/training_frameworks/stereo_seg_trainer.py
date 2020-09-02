#!/usr/bin/env python3

__author__ = "Bryce Ferenczi"
__email__ = "bryce.ferenczi@monashmotorsport.com"

import os, sys, time, platform, multiprocessing
from pathlib import Path
from typing import Dict, TypeVar
T = TypeVar('T')
import numpy as np

import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from nnet_training.utilities.metrics import SegmentationMetric
from nnet_training.utilities.visualisation import get_color_pallete
from nnet_training.training_frameworks.trainer_base_class import ModelTrainer

__all__ = ['StereoSegTrainer']

class StereoSegTrainer(ModelTrainer):
    def __init__(self, model: torch.nn.Module, optim: torch.optim.Optimizer,
                 loss_fn: Dict[str, torch.nn.Module], lr_cfg: Dict[str, T],
                 dataldr: Dict[str, torch.utils.data.DataLoader],
                 modelpath: Path, checkpoints=True):
        '''
        Initialize the Model trainer giving it a nn.Model, nn.Optimizer and dataloaders as
        a dictionary with Training, Validation and Testing loaders
        '''
        super(StereoSegTrainer, self).__init__(model, optim, dataldr, lr_cfg,
                                               modelpath, checkpoints)

        self._loss_function = loss_fn
        self.metric_loggers['seg'] = SegmentationMetric(19, base_dir=modelpath, savefile='segmentation_data')

    def _train_epoch(self, max_epoch):
        self._model.train()

        start_time = time.time()

        for batch_idx, data in enumerate(self._training_loader):
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

            self.metric_loggers['seg']._add_sample(
                torch.argmax(outputs, dim=1, keepdim=True).cpu().data.numpy(),
                target.cpu().data.numpy(),
                loss=loss.item()
            )

            if not batch_idx % 10:
                time_elapsed = time.time() - start_time
                time_remain = time_elapsed / (batch_idx + 1) * (len(self._training_loader) - (batch_idx + 1))
                sys.stdout.flush()
                sys.stdout.write('\rTrain Epoch: [%2d/%2d] Iter [%4d/%4d] || lr: %.8f || Loss: %.4f || Time Elapsed: %.2f sec || Est Time Remain: %.2f sec' % (
                    self.epoch, max_epoch, batch_idx + 1, len(self._training_loader),
                    self._lr_manager.get_lr(), loss.item(), time_elapsed, time_remain))

    def _validate_model(self, max_epoch):
        with torch.no_grad():
            self._model.eval()

            start_time = time.time()

            for batch_idx, data in enumerate(self._validation_loader):
                # Put both image and target onto device
                left    = data['l_img'].to(self._device)
                right   = data['r_img'].to(self._device)
                target  = data['seg'].to(self._device)

                outputs = self._model(left, right)

                # Caculate the loss and accuracy for the predictions
                loss = self._loss_function(outputs, target)

                self.metric_loggers['seg']._add_sample(
                    torch.argmax(outputs, dim=1, keepdim=True).cpu().data.numpy(),
                    target.cpu().numpy(),
                    loss=loss.item()
                )

                if not batch_idx % 10:
                    batch_acc = self.metric_loggers['seg'].get_last_batch()
                    time_elapsed = time.time() - start_time
                    time_remain = time_elapsed / (batch_idx + 1) * (len(self._validation_loader) - (batch_idx + 1))
                    sys.stdout.flush()
                    sys.stdout.write('\rValidaton Epoch: [%2d/%2d] Iter [%4d/%4d] || Accuracy: %.4f || Loss: %.4f || Time Elapsed: %.2f sec || Est Time Remain: %.2f sec' % (
                        self.epoch, max_epoch, batch_idx + 1, len(self._validation_loader),
                        batch_acc, loss.item(), time_elapsed, time_remain))

    def visualize_output(self):
        """
        Forward pass over a testing batch and displays the output
        """
        with torch.no_grad():
            self._model.eval()
            data   = next(iter(self._validation_loader))
            left   = data['l_img'].to(self._device)
            right  = data['r_img'].to(self._device)
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

                plt.suptitle("Propagation time: " + str(propagation_time))
                plt.show()


if __name__ == "__main__":
    from nnet_training.utilities.loss_functions import FocalLoss2D
    from nnet_training.nnet_models.nnet_models import StereoSegmentaionSeparated
    from nnet_training.utilities.dataset import CityScapesDataset

    BATCH_SIZE = 8
    if platform.system() == 'Windows':
        n_workers = 0
    else:
        n_workers = min(multiprocessing.cpu_count(), BATCH_SIZE)

    base_dir = '/media/bryce/4TB Seagate/Autonomous Vehicles Data/Cityscapes Data/'

    training_dir = {
        'images'        : base_dir + 'leftImg8bit/train',
        'right_images'  : base_dir + 'rightImg8bit/train',
        'seg'           : base_dir + 'gtFine/train'
    }

    validation_dir = {
        'images'        : base_dir + 'leftImg8bit/val',
        'right_images'  : base_dir + 'rightImg8bit/val',
        'seg'           : base_dir + 'gtFine/val'
    }

    datasets = dict(
        Training    = CityScapesDataset(training_dir, crop_fraction=2),
        Validation  = CityScapesDataset(validation_dir, crop_fraction=2)
    )

    dataloaders=dict(
        Training    = DataLoader(datasets["Training"], batch_size=BATCH_SIZE,
                                 shuffle=True, num_workers=n_workers, drop_last=True),
        Validation  = DataLoader(datasets["Validation"], batch_size=BATCH_SIZE,
                                 shuffle=True, num_workers=n_workers, drop_last=True),
    )

    BASEPATH = Path.cwd() / "torch_models"
    Model = StereoSegmentaionSeparated()
    optimizer = torch.optim.SGD(Model.parameters(), lr=0.01, momentum=0.9)
    lossfn = FocalLoss2D(gamma=1, ignore_index=-1).to(torch.device("cuda"))

    lr_sched = {"lr": 0.01, "mode":"poly"}
    modeltrainer = StereoSegTrainer(model=Model, optim=optimizer, loss_fn=lossfn,
                                    dataldr=dataloaders, lr_cfg=lr_sched, modelpath=BASEPATH)
    modeltrainer.visualize_output()
    # modeltrainer.train_model(1)