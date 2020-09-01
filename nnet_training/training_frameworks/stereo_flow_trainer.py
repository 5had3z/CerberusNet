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
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms

from nnet_training.utilities.metrics import OpticFlowMetric
from nnet_training.utilities.dataset import CityScapesDataset
from nnet_training.training_frameworks.trainer_base_class import ModelTrainer

__all__ = ['StereoFlowTrainer']

class StereoFlowTrainer(ModelTrainer):
    '''
    Stereo Flow Training Class
    '''
    def __init__(self, model: torch.nn.Module, optim: torch.optim.Optimizer,
                 loss_fn: Dict[str, torch.nn.Module], lr_cfg: Dict[str, T],
                 dataldr: Dict[str, torch.utils.data.DataLoader],
                 modelpath: Path, checkpoints=True):
        '''
        Initialize the Model trainer giving it a nn.Model, nn.Optimizer and dataloaders as
        a dictionary with Training, Validation and Testing loaders
        '''
        super(StereoFlowTrainer, self).__init__(model, optim, dataldr,
                                                lr_cfg, modelpath, checkpoints)
        self._loss_function = loss_fn
        self._metric = OpticFlowMetric(base_dir=modelpath, savefile='flow_data')

    def save_checkpoint(self):
        super(StereoFlowTrainer, self).save_checkpoint()
        self._metric.save_epoch()

    def load_checkpoint(self):
        if os.path.isfile(self._path):
            self.epoch = len(self._metric)
        super(StereoFlowTrainer, self).load_checkpoint()

    def _train_epoch(self, max_epoch):
        self._model.train()

        self._metric.new_epoch('training')

        start_time = time.time()

        for batch_idx, data in enumerate(self._training_loader):
            cur_lr = self._lr_manager(batch_idx)
            for param_group in self._optimizer.param_groups:
                param_group['lr'] = cur_lr

            # Put both image and target onto device
            left      = data['l_img'].to(self._device)
            right     = data['r_img'].to(self._device)
            left_seq  = data['l_seq'].to(self._device)
            right_seq = data['r_seq'].to(self._device)

            # Computer loss, use the optimizer object to zero all of the gradients
            # Then backpropagate and step the optimizer
            output_left, output_right = self._model(left, right)

            loss =  self._loss_function(left, output_left, left_seq)
            loss += self._loss_function(right, output_right, right_seq)

            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

            self._metric._add_sample(
                [left.cpu().data.numpy(), right.cpu().data.numpy()],
                [left_seq.cpu().data.numpy(), right_seq.cpu().data.numpy()],
                [output_left.cpu().data.numpy(), output_right.cpu().data.numpy()],
                None,
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

            self._metric.new_epoch('validation')

            start_time = time.time()

            for batch_idx, data in enumerate(self._validation_loader):
                # Put both image and target onto device
                left      = data['l_img'].to(self._device)
                right     = data['r_img'].to(self._device)
                left_seq  = data['l_seq'].to(self._device)
                right_seq = data['r_seq'].to(self._device)

                output_left, output_right = self._model(left, right)

                # Caculate the loss and accuracy for the predictions
                loss =  self._loss_function(left, output_left, left_seq)
                loss += self._loss_function(right, output_right, right_seq)

                self._metric._add_sample(
                    [left.cpu().data.numpy(), right.cpu().data.numpy()],
                    [left_seq.cpu().data.numpy(), right_seq.cpu().data.numpy()],
                    [output_left.cpu().data.numpy(), output_right.cpu().data.numpy()],
                    None,
                    loss=loss.item()
                )

                if not batch_idx % 10:
                    batch_acc = self._metric.get_last_batch()
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
            data = next(iter(self._validation_loader))
            left      = data['l_img'].to(self._device)
            right     = data['r_img'].to(self._device)
            left_seq  = data['l_seq'].to(self._device)

            start_time = time.time()
            pred_l, _ = self._model(left, right)
            propagation_time = (time.time() - start_time)/self._validation_loader.batch_size

            for i in range(self._validation_loader.batch_size):
                plt.subplot(2, 2, 1)
                plt.imshow(np.moveaxis(left[i, 0:3, :, :].cpu().numpy(), 0, 2))
                plt.xlabel("Base Image")

                plt.subplot(2, 2, 2)
                plt.imshow(pred_l.cpu().numpy()[i, 0, :, :])
                plt.xlabel("Predicted Flow")

                recon = self.reconstruct_flow(left[i, 0:3, :, :].cpu().numpy(),
                                              pred_l.cpu().numpy()[i, 0, :, :])

                plt.subplot(2, 2, 3)
                plt.imshow(recon)
                plt.xlabel("Predicted Reconstruction")

                plt.subplot(2, 2, 4)
                plt.imshow(left_seq[i, :, :])
                plt.xlabel("Sequential Image")

                plt.suptitle("Propagation time: " + str(propagation_time))
                plt.show()

    def reconstruct_flow(self, image, flow):
        return image * flow


if __name__ == "__main__":
    from nnet_training.utilities.loss_functions import ReconstructionLossV1
    from nnet_training.nnet_models.nnet_models import StereoDepthSeparatedExp,\
                                                      StereoDepthSeparatedReLu

    if platform.system() == 'Windows':
        n_workers = 0
    else:
        n_workers = multiprocessing.cpu_count()

    base_dir = '/media/bryce/4TB Seagate/Autonomous Vehicles Data/Cityscapes Data/'
    training_dir = {
        'images'        : base_dir + 'leftImg8bit/train',
        'right_images'  : base_dir + 'rightImg8bit/train',
        'disparity'     : base_dir + 'disparity/train'
    }
    validation_dir = {
        'images'        : base_dir + 'leftImg8bit/val',
        'right_images'  : base_dir + 'rightImg8bit/val',
        'disparity'     : base_dir + 'disparity/val'
    }

    datasets = dict(
        Training    = CityScapesDataset(training_dir, crop_fraction=1, output_size=(1024, 512)),
        Validation  = CityScapesDataset(validation_dir, crop_fraction=1, output_size=(1024, 512))
    )

    dataloaders = dict(
        Training    = DataLoader(datasets["Training"], batch_size=8,
                                 shuffle=True, num_workers=n_workers, drop_last=True),
        Validation  = DataLoader(datasets["Validation"], batch_size=8,
                                 shuffle=True, num_workers=n_workers, drop_last=True),
    )

    filename = "ReLuModel_ScaleInv"
    disparityModel = StereoDepthSeparatedReLu()
    optimizer = torch.optim.SGD(disparityModel.parameters(), lr=0.01, momentum=0.9)
    # lossfn = DepthAwareLoss().to(torch.device("cuda"))
    lossfn = ReconstructionLossV1(img_b=8, img_w=1024, img_h=512).to(torch.device("cuda"))
    # lossfn = InvHuberLoss().to(torch.device("cuda"))

    lr_sched = { "mode":"poly", "lr":0.01 }
    modeltrainer = StereoFlowTrainer(disparityModel, optimizer, lossfn, dataloaders,
                                     lr_cfg=lr_sched, modelpath=filename)
    modeltrainer.visualize_output()
    # modeltrainer.train_model(5)
