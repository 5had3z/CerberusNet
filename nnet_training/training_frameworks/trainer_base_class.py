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
from nnet_training.utilities.metrics import MetricBaseClass

__all__ = ['ModelTrainer']

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

        self._basepath = basepath

        if self._checkpoints:
            self.load_checkpoint(self._basepath / (str(self._model)+"_latest.pth"))
        else:
            if os.path.isfile(self._basepath / (str(self._model)+"_latest.pth")):
                sys.stdout.write("\nWarning: Checkpoint Already Exists!")
            else:
                sys.stdout.write("\nStarting From Scratch!")

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

    def write_summary(self):
        """
        Writes a brief summary of the current state of an experiment in a text file
        """
        with open(self._basepath / "Summary.txt", "w") as txt_file:
            txt_file.write("{} Summary, # Epochs: {}\n".format(str(self._model), self.epoch))
            for key, metric in self.metric_loggers.items():
                name, value = metric.max_accuracy(main_metric=True)
                txt_file.write("Objective: %s\tMetric: %s\tValue: %.3f\n"%(key, name, value[1]))

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

            self._model.train()
            self._train_epoch(max_epoch)

            for metric in self.metric_loggers.values():
                metric.new_epoch('validation')

            self._model.eval()
            self._validate_model(max_epoch)

            epoch_duration = time.time() - epoch_start_time

            if self._checkpoints:
                for key, logger in self.metric_loggers.items():
                    epoch_acc = logger.get_epoch_statistics(main_metric=True,
                                                            loss_metric=False)
                    _, prev_best = logger.max_accuracy(main_metric=True)
                    if prev_best[0](epoch_acc[0], prev_best[1]):
                        filename = "{}_{}.pth".format(str(self._model), key)
                        self.save_checkpoint(self._basepath / filename, metrics=False)

                self.save_checkpoint(
                    self._basepath / "{}_latest.pth".format(str(self._model)), metrics=True)

                self.write_summary()

            sys.stdout.flush()
            sys.stdout.write('\rEpoch '+str(self.epoch)+' Finished, Time: '+
                             str(epoch_duration)+'s\n')

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

if __name__ == "__main__":
    raise NotImplementedError
