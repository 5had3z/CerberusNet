#!/usr/bin/env python3

__author__ = "Bryce Ferenczi"
__email__ = "bryce.ferenczi@monashmotorsport.com"

import os
import time
import platform
import multiprocessing
import numpy as np
from pathlib import Path

import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms

from loss_functions import DepthAwareLoss, ScaleInvariantError
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
        super().__init__(model, optimizer, loss_fn, dataloaders, learning_rate, savefile, checkpoints)
        self._metric = DepthMetric(filename=self._modelname)

    def visualize_output(self):
        """
        Forward pass over a testing batch and displays the output
        """
        with torch.no_grad():
            self._model.eval()
            image, disparity = next(iter(self._validation_loader))
            image = image.to(self._device)

            start_time = time.time()
            pred = self._model(image)
            propagation_time = (time.time() - start_time)/self._validation_loader.batch_size

            for i in range(self._validation_loader.batch_size):
                plt.subplot(1,3,1)
                plt.imshow(np.moveaxis(image[i,0:3,:,:].cpu().numpy(),0,2))
                plt.xlabel("Base Image")
        
                plt.subplot(1,3,2)
                plt.imshow(disparity[i,:,:])
                plt.xlabel("Ground Truth")
        
                plt.subplot(1,3,3)
                plt.imshow(pred.cpu().numpy()[i,0,:,:])
                plt.xlabel("Prediction")

                plt.suptitle("Propagation time: " + str(propagation_time))
                plt.show()

from chat_test_model import TestModel

if __name__ == "__main__":
    print(Path.cwd())
    if platform.system() == 'Windows':
        n_workers = 0
    else:
        n_workers = multiprocessing.cpu_count()

    training_dir = {
        'images': '/media/bryce/4TB Seagate/Autonomous Vehicles Data/Cityscapes Data/leftImg8bit/train',
        'right_images': '/media/bryce/4TB Seagate/Autonomous Vehicles Data/Cityscapes Data/rightImg8bit/train',
        'disparity': '/media/bryce/4TB Seagate/Autonomous Vehicles Data/Cityscapes Data/disparity/train'
    }

    validation_dir = {
        'images': '/media/bryce/4TB Seagate/Autonomous Vehicles Data/Cityscapes Data/leftImg8bit/val',
        'right_images': '/media/bryce/4TB Seagate/Autonomous Vehicles Data/Cityscapes Data/rightImg8bit/val',
        'disparity': '/media/bryce/4TB Seagate/Autonomous Vehicles Data/Cityscapes Data/disparity/val'
    }

    datasets = dict(
        Training=CityScapesDataset(training_dir),
        Validation=CityScapesDataset(validation_dir)
    )

    dataloaders=dict(
        Training=DataLoader(datasets["Training"], batch_size=12, shuffle=True, num_workers=n_workers, drop_last=True),
        Validation=DataLoader(datasets["Validation"], batch_size=12, shuffle=True, num_workers=n_workers, drop_last=True),
    )

    filename = "DisparityTest"
    disparityModel = TestModel()
    optimizer = torch.optim.SGD(disparityModel.parameters(), lr=0.01, momentum=0.9)
    lossfn = ScaleInvariantError().to(torch.device("cuda"))

    modeltrainer = StereoDisparityTrainer(disparityModel, optimizer, lossfn, dataloaders, learning_rate=0.01, savefile=filename)
    # modeltrainer.visualize_output()
    modeltrainer.train_model(1)
