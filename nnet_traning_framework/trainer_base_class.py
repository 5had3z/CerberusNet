#!/usr/bin/env python3

__author__ = "Bryce Ferenczi"
__email__ = "bryce.ferenczi@monashmotorsport.com"

from datetime import datetime
import torch
import os
import time
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torchvision.transforms
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

from metrics import SegmentationMetric

class ModelTrainer():
    def __init__(self, model, optimizer, loss_fn, dataloaders, learning_rate=1e-4, savefile=None, checkpoints=True):
        '''
        Initialize the Model trainer giving it a nn.Model, nn.Optimizer and dataloaders as
        a dictionary with Training, Validation and Testing loaders
        '''
        self._device = torch.device( "cuda" if torch.cuda.is_available() else "cpu" )

        self._model = model.to(self._device)
        self._optimizer = optimizer
        self._loss_function = loss_fn
        self._learning_rate = learning_rate
        self._modelname = savefile

        self._training_loader = dataloaders["Training"]
        self._validation_loader = dataloaders["Validation"]
        self._testing_loader = dataloaders["Testing"]
        self.epoch = 0
        self.best_acc = 0.0

        self._metric = SegmentationMetric(19)

        self._checkpoints = checkpoints

        self.training_data = dict(
            Loss=[],
            Accuracy=[]
            )
        self.validation_data = dict(
            Loss=[],
            Accuracy=[]
            )
        
        self._save_dir = Path.cwd() / "torch_models"
        if not os.path.isdir(self._save_dir):
            os.makedirs(self._save_dir)
        
        if savefile is not None:
            self._path = self._save_dir / savefile
        else:
            self._modelname = str(datetime.now()).replace(" ", "_")
            self._path = self._save_dir / self._modelname
            
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
            check_point = torch.load(self._path, map_location=torch.device(self._device))
            self._model.load_state_dict(check_point['model_state_dict'])
            self._optimizer.load_state_dict(check_point['optimizer_state_dict'])
            self.epoch = check_point['epoch']
            self.training_data = check_point['training_data']
            self.validation_data = check_point['validation_data']
            self.best_acc = max(self.validation_data["Accuracy"])
            sys.stdout.write("\nCheckpoint loaded, starting from epoch:" + str(self.epoch) + "\n")
        else:
            #Raise Error if it does not exist
            sys.stdout.write("\nCheckpoint Does Not Exist\nStarting From Scratch!")

    def save_checkpoint(self):
        '''
        Saves progress of the model
        '''
        sys.stdout.write("\nSaving Model")
        torch.save({
            'epoch':                    self.epoch,
            'model_state_dict':         self._model.state_dict(),
            'optimizer_state_dict':     self._optimizer.state_dict(), 
            'training_data':            self.training_data,
            'validation_data':          self.validation_data
        }, self._path)
        self.best_acc = max(self.validation_data["Accuracy"])

    def train_model(self, n_epochs):
        train_start_time = time.time()

        max_epoch = self.epoch + n_epochs
        while self.epoch < max_epoch:
            self.epoch += 1
            epoch_start_time = time.time()

            # Calculate the training loss, training duration for each epoch, and validation accuracy
            self._train_epoch(max_epoch)
            self._validate_model(max_epoch)

            epoch_end_time = time.time()
        
            if (self.best_acc < self.validation_data["Accuracy"][-1] or True) and self._checkpoints:
                self.save_checkpoint()
        
            sys.stdout.flush()
            sys.stdout.write('\rEpoch: '+ str(self.epoch)+
                    ' Training Loss: '+ str(self.training_data["Loss"][-1])+
                    ' Testing Accuracy:'+ str(self.training_data["Accuracy"][-1])+
                    ' Time: '+ str(epoch_end_time - epoch_start_time)+ 's')
    
        train_end_time = time.time()

        print('\nTotal Traning Time:\t', train_end_time - train_start_time)

    def _train_epoch(self, max_epoch):
        self._model.train()

        loss_epoch = 0.0

        start_time = time.time()

        for batch_idx, (data, target) in enumerate(self._training_loader):   
            # Put both image and target onto device
            data = data.to(self._device)
            target = target.to(self._device)
            
            # Computer loss, use the optimizer object to zero all of the gradients
            # Then backpropagate and step the optimizer
            outputs = self._model(data)

            loss = self._loss_function(outputs[0], target)

            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

            self._metric.add_sample(
                torch.argmax(outputs[0],dim=1,keepdim=True).cpu().data.numpy(),
                target.cpu().data.numpy()
            )

            loss_epoch = loss.item()/(batch_idx+1) + loss_epoch*batch_idx/(batch_idx+1)
            if batch_idx % 10 == 0:
                time_elapsed = time.time() - start_time
                time_remain = time_elapsed / (batch_idx + 1) * (len(self._training_loader) - (batch_idx + 1))
                print('Train Epoch: [%2d/%2d] Iter [%4d/%4d] || lr: %.8f || Loss: %.4f || Time Elapsed: %4.4f sec || Est Time Remain: %4.4f sec' % (
                        self.epoch, max_epoch, batch_idx + 1, len(self._training_loader),
                        self._learning_rate, loss.item(), time_elapsed, time_remain))
        
        _, accuracy = self._metric.get_statistics()
        self.training_data["Loss"].append(loss_epoch)
        self.training_data["Accuracy"].append(accuracy)

    def _validate_model(self, max_epoch):
        with torch.no_grad():
            self._model.eval()

            loss_epoch = 0.0

            start_time = time.time()

            for batch_idx, (data, target) in enumerate(self._validation_loader):
                # Put both image and target onto device
                data = data.to(self._device)
                target = target.to(self._device)

                outputs = self._model(data)
                
                # Caculate the loss and accuracy for the predictions
                loss = self._loss_function(outputs[0], target)
                loss_epoch = loss.item()/(batch_idx+1) + loss_epoch*batch_idx/(batch_idx+1)

                self._metric.add_sample(
                    torch.argmax(outputs[0],dim=1,keepdim=True).cpu().data.numpy(),
                    target.numpy()
                )

                if batch_idx % 10 == 0:
                    time_elapsed = time.time() - start_time
                    time_remain = time_elapsed / (batch_idx + 1) * (len(self._validation_loader) - (batch_idx + 1))
                    print('Validaton Epoch: [%2d/%2d] Iter [%4d/%4d] || Accuracy: %.4f || Loss: %.4f || Time Elapsed: %4.4f sec || Est Time Remain: %4.4f sec' % (
                            self.epoch, max_epoch, batch_idx + 1, len(self._validation_loader),
                            0.0, loss.item(), time_elapsed, time_remain))

        _, accuracy = self._metric.get_statistics()
        self.validation_data["Loss"].append(loss_epoch)
        self.validation_data["Accuracy"].append(accuracy)

    def _test_model(self):
        with torch.no_grad():
            self._model.eval()

            for batch_idx, (data, target) in enumerate(self._testing_loader):   
                # Caculate the loss and accuracy for the predictions
                data = data.to(self._device)
                target = target.to(self._device)

                outputs = self._model(data)

    def plot_data(self):
        """
        This plots all the training statistics
        """
        plt.figure(figsize=(18, 5))
        plt.suptitle(self._modelname + ' Training and Validation Results')

        plt.subplot(1,2,1)
        plt.plot(self.training_data["Loss"])
        plt.plot(self.validation_data["Loss"])
        plt.legend(["Training", "Validation"])
        plt.title('Loss Function over Epochs')
        plt.ylabel('Loss')
        plt.xlabel('Epoch #')

        plt.subplot(1,2,2)
        plt.plot(self.training_data["Accuracy"])
        plt.plot(self.validation_data["Accuracy"])
        plt.legend(["Training", "Validation"])
        plt.title('Accuracy over Epochs')
        plt.ylabel('% Accuracy')
        plt.xlabel('Epoch #')
        plt.show()

    def get_learning_rate(self):
        """
        Returns the current learning rate
        """
        return(self._learning_rate)

    def set_learning_rate(self, new_learning_rate):
        """
        Sets a new learning rate uniformly over the entire network
        """
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = new_learning_rate

