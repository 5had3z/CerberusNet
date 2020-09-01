#!/usr/bin/env python3

__author__ = "Bryce Ferenczi"
__email__ = "bryce.ferenczi@monashmotorsport.com"

import os
import time
import sys
from pathlib import Path
from typing import Dict

import torch

from nnet_training.utilities.lr_scheduler import LRScheduler

__all__ = ['ModelTrainer']

class ModelTrainer(object):
    """
    Base class that various model trainers inherit from
    """
    def __init__(self, model: torch.nn.Module, optimizer: torch.nn.Optimizer,
                 dataloaders: Dict[torch.utils.data.DataLoader], lr_cfg: Dict,
                 basepath: Path, checkpoints=True):
        '''
        Initialize the Model trainer giving it a nn.Model, nn.Optimizer and dataloaders as
        a dictionary with Training, Validation and Testing loaders
        '''
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._training_loader = dataloaders["Training"]
        self._validation_loader = dataloaders["Validation"]

        self.epoch = 0

        self._model = model.to(self._device)
        self._optimizer = optimizer

        self._lr_cfg = lr_cfg

        self._checkpoints = checkpoints

        if not os.path.isdir(basepath):
            os.makedirs(basepath)

        self._path = basepath / (str(self._model) + ".pth")

        if self._checkpoints:
            self.load_checkpoint()
        else:
            if os.path.isfile(self._path):
                sys.stdout.write("\nWarning: Checkpoint Already Exists!")
            else:
                sys.stdout.write("\nStarting From Scratch!")

    def load_checkpoint(self):
        '''
        Loads previous progress of the model if available
        '''
        if os.path.isfile(self._path):
            #Load Checkpoint
            checkpoint = torch.load(self._path, map_location=torch.device(self._device))
            self._model.load_state_dict(checkpoint['model_state_dict'])
            self._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            sys.stdout.write("\nCheckpoint loaded from " + str(self._path)
                             + " starting from epoch:" + str(self.epoch) + "\n")
        else:
            #Raise Error if it does not exist
            sys.stdout.write("\nCheckpoint Does Not Exist\nStarting From Scratch!")

    def save_checkpoint(self):
        '''
        Saves progress of the model
        '''
        sys.stdout.write("\nSaving Model")
        torch.save({
            'model_state_dict':         self._model.state_dict(),
            'optimizer_state_dict':     self._optimizer.state_dict()
        }, self._path)

    def train_model(self, n_epochs):
        train_start_time = time.time()

        max_epoch = self.epoch + n_epochs

        self._lr_manager = LRScheduler(mode=self._lr_cfg['mode'], base_lr=self._lr_cfg['lr'],
                                       nepochs=n_epochs, iters_per_epoch=len(self._training_loader),
                                       power=0.9)

        while self.epoch < max_epoch:
            self.epoch += 1
            epoch_start_time = time.time()

            # Calculate the training loss, training duration for each epoch, and validation accuracy
            self._train_epoch(max_epoch)
            self._validate_model(max_epoch)

            epoch_end_time = time.time()

            if self._checkpoints:
                self.save_checkpoint()

            sys.stdout.flush()
            sys.stdout.write('\rEpoch '+ str(self.epoch) +
                             ' Finished, Time: '+ str(epoch_end_time - epoch_start_time)+ 's\n')

        train_end_time = time.time()

        print('\nTotal Traning Time:\t', train_end_time - train_start_time)

    def _train_epoch(self, max_epoch):
        raise NotImplementedError

    def _validate_model(self, max_epoch):
        raise NotImplementedError

    def _test_model(self):
        raise NotImplementedError
