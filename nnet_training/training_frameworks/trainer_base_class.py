#!/usr/bin/env python3

__author__ = "Bryce Ferenczi"
__email__ = "bryce.ferenczi@monashmotorsport.com"

import os
import time
import sys
from pathlib import Path
from typing import Dict, TypeVar
T = TypeVar('T')

import torch

from nnet_training.utilities.lr_scheduler import LRScheduler

__all__ = ['ModelTrainer', 'get_trainer']

class ModelTrainer(object):
    """
    Base class that various model trainers inherit from
    """
    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                 dataloaders: Dict[str, torch.utils.data.DataLoader], lr_cfg: Dict[str, T],
                 basepath: Path, checkpoints=True):
        '''
        Initialize the Model trainer giving it a nn.Model, nn.Optimizer and dataloaders as
        a dictionary with Training, Validation and Testing loaders
        '''
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._training_loader = dataloaders["Training"]
        self._validation_loader = dataloaders["Validation"]

        self.epoch = 0

        if not hasattr(self, 'metric_loggers'):
            self.metric_loggers = {}
            Warning("No metrics are initialised")

        self._model = model.to(self._device)
        self._optimizer = optimizer

        self._lr_manager = LRScheduler(**lr_cfg)

        self._checkpoints = checkpoints

        if not os.path.isdir(basepath):
            os.makedirs(basepath)

        self._best_path = basepath / (str(self._model) + "_best.pth")
        self._latest_path = basepath / (str(self._model) + "_latest.pth")

        if self._checkpoints:
            self.load_checkpoint(self._latest_path)
        else:
            if os.path.isfile(self._latest_path):
                sys.stdout.write("\nWarning: Checkpoint Already Exists!")
            else:
                sys.stdout.write("\nStarting From Scratch!")

    def get_learning_rate(self) -> float:
        return self._lr_manager.get_lr()

    def set_learning_rate(self, new_lr: float) -> None:
        self._lr_manager.base_lr = new_lr

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
            sys.stdout.write("\nCheckpoint loaded from " + str(path)
                             + " starting from epoch:" + str(self.epoch) + "\n")
        else:
            #Raise Error if it does not exist
            sys.stdout.write("\nCheckpoint Does Not Exist\nStarting From Scratch!")

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

            self._train_epoch(max_epoch)

            for metric in self.metric_loggers.values():
                metric.new_epoch('validation')

            self._validate_model(max_epoch)

            epoch_end_time = time.time()

            if self._checkpoints:
                self.save_checkpoint(self._latest_path, metrics=True)

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

    def plot_data(self):
        """
        Plots the main summary data for each of the metric loggers.\n
        You must overide this if you want some extra metric plots
        e.g. classwise segmentation iou\n
        (don't forget to call this after your override)
        """
        for metric in self.metric_loggers.values():
            metric.plot_summary_data()

def get_trainer(trainer_name: str) -> ModelTrainer:
    """
    Returns the corresponding network trainer given a string
    """
    from nnet_training.training_frameworks import MonoFlowTrainer,\
        MonoSegFlowTrainer, MonoSegmentationTrainer, StereoDisparityTrainer,\
        StereoFlowTrainer, StereoSegDepthTrainer, StereoSegTrainer

    if trainer_name == "MonoFlowTrainer":
        trainer = MonoFlowTrainer
    elif trainer_name == "MonoSegFlowTrainer":
        trainer = MonoSegFlowTrainer
    elif trainer_name == "MonoSegmentationTrainer":
        trainer = MonoSegmentationTrainer
    elif trainer_name == "StereoDisparityTrainer":
        trainer = StereoDisparityTrainer
    elif trainer_name == "StereoFlowTrainer":
        trainer = StereoFlowTrainer
    elif trainer_name == "StereoSegDepthTrainer":
        trainer = StereoSegDepthTrainer
    elif trainer_name == "StereoSegTrainer":
        trainer = StereoSegTrainer
    else:
        raise NotImplementedError(trainer_name)

    return trainer
