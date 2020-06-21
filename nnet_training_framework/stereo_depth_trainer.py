#!/usr/bin/env python3

__author__ = "Bryce Ferenczi"
__email__ = "bryce.ferenczi@monashmotorsport.com"

import os, sys, time, platform, multiprocessing
import numpy as np
from pathlib import Path

import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms

from loss_functions import DepthAwareLoss, ScaleInvariantError, InvHuberLoss
from metrics import DepthMetric
from dataset import CityScapesDataset
from trainer_base_class import ModelTrainer

__all__ = ['StereoDisparityTrainer']

class StereoDisparityTrainer(ModelTrainer):
    def __init__(self, model, optimizer, loss_fn, dataloaders, learning_rate=1e-4, savefile=None, checkpoints=True):
        '''
        Initialize the Model trainer giving it a nn.Model, nn.Optimizer and dataloaders as
        a dictionary with Training, Validation and Testing loaders
        '''
        super(StereoDisparityTrainer, self).__init__(model, optimizer, dataloaders, learning_rate, savefile, checkpoints)
        self._loss_function = loss_fn
        self._metric = DepthMetric(filename=self._modelname)

    def save_checkpoint(self):
        super(StereoDisparityTrainer, self).save_checkpoint()
        self._metric.save_epoch()

    def load_checkpoint(self):
        if os.path.isfile(self._path):
            self.epoch = len(self._metric)
        super(StereoDisparityTrainer, self).load_checkpoint()

    def _train_epoch(self, max_epoch):
        self._model.train()

        self._metric.new_epoch('training')
        torch.autograd.set_detect_anomaly(True)

        start_time = time.time()

        for batch_idx, (data, target) in enumerate(self._training_loader):
            cur_lr = self._lr_manager(batch_idx)
            for param_group in self._optimizer.param_groups:
                param_group['lr'] = cur_lr
            
            # Put both image and target onto device
            left = data[0].to(self._device)
            right = data[1].to(self._device)
            target = target.to(self._device)
            
            # Computer loss, use the optimizer object to zero all of the gradients
            # Then backpropagate and step the optimizer
            outputs = self._model(left, right)

            loss = self._loss_function(outputs, target)

            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

            self._metric._add_sample(
                outputs.cpu().data.numpy(),
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

            self._metric.new_epoch('validation')

            start_time = time.time()

            for batch_idx, (data, target) in enumerate(self._validation_loader):
                # Put both image and target onto device
                left = data[0].to(self._device)
                right = data[1].to(self._device)
                target = target.to(self._device)

                outputs = self._model(left, right)
                
                # Caculate the loss and accuracy for the predictions
                loss = self._loss_function(outputs, target)

                self._metric._add_sample(
                    outputs.cpu().data.numpy(),
                    target.cpu().numpy(),
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
            image, disparity = next(iter(self._validation_loader))
            left = image[0].to(self._device)
            right = image[1].to(self._device)

            start_time = time.time()
            pred = self._model(left, right)
            propagation_time = (time.time() - start_time)/self._validation_loader.batch_size

            for i in range(self._validation_loader.batch_size):
                plt.subplot(1,3,1)
                plt.imshow(np.moveaxis(left[i,0:3,:,:].cpu().numpy(),0,2))
                plt.xlabel("Base Image")
        
                plt.subplot(1,3,2)
                plt.imshow(disparity[i,:,:])
                plt.xlabel("Ground Truth")
        
                plt.subplot(1,3,3)
                plt.imshow(pred.cpu().numpy()[i,0,:,:])
                # plt.imshow(torch.exp(pred).cpu().numpy()[i,0,:,:])
                plt.xlabel("Prediction")

                plt.suptitle("Propagation time: " + str(propagation_time))
                plt.show()

from StereoModels import StereoDepthSeparatedExp, StereoDepthSeparatedReLu

if __name__ == "__main__":
    print(Path.cwd())
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
        Training    = CityScapesDataset(training_dir, crop_fraction=1),
        Validation  = CityScapesDataset(validation_dir, crop_fraction=1)
    )

    dataloaders = dict(
        Training    = DataLoader(datasets["Training"], batch_size=8, shuffle=True, num_workers=n_workers, drop_last=True),
        Validation  = DataLoader(datasets["Validation"], batch_size=8, shuffle=True, num_workers=n_workers, drop_last=True),
    )

    filename = "ReLuModel_ScaleInv"
    disparityModel = StereoDepthSeparatedReLu()
    optimizer = torch.optim.SGD(disparityModel.parameters(), lr=0.01, momentum=0.9)
    # lossfn = DepthAwareLoss().to(torch.device("cuda"))
    lossfn = ScaleInvariantError().to(torch.device("cuda"))
    # lossfn = InvHuberLoss().to(torch.device("cuda"))

    modeltrainer = StereoDisparityTrainer(disparityModel, optimizer, lossfn, dataloaders, learning_rate=0.01, savefile=filename)
    modeltrainer.visualize_output()
    # modeltrainer.train_model(5)
