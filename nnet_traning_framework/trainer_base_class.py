#!/usr/bin/env python

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

class ModelTrainer():
    def __init__(self, model, optimizer, loss_fn, dataloaders, learning_rate=1e-4, savefile=None, checkpoints=True):
        '''
        Initialize the Model trainer giving it a nn.Model, nn.Optimizer and dataloaders as
        a dictionary with Training, Validation and Testing loaders
        '''
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

        self._checkpoints = checkpoints

        self.training_data = dict(
            Loss=[],
            Accuracy=[]
            )
        self.validation_data = dict(
            Loss=[],
            Accuracy=[]
            )
        self.testing_data = dict(
            Accuracy=[]
            )
        
        self._save_dir = Path.cwd() / "Models"
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
            self.testing_data = check_point['testing_data']
            self.best_acc = max(self.testing_data["Accuracy"])
            sys.stdout.write("\nCheckpoint loaded, starting from epoch:" + str(self.epoch))
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
            'validation_data':          self.validation_data,
            'testing_data':             self.testing_data,
        }, self._path)
        self.best_acc = max(self.testing_data["Accuracy"])

    def train_model(self, n_epochs):
        train_start_time = time.time()

        for epoch in range(self.epoch, self.epoch + n_epochs):

            epoch_start_time = time.time()
            self.epoch = epoch
            # Calculate the training loss, training duration for each epoch, and testing loss and accuracy
            self._train_epoch()
            self._validate_model()
            self._test_model()

            epoch_end_time = time.time()
        
            if (self.best_acc < self.testing_data["Accuracy"][-1] or True) and self._checkpoints:
                self.save_checkpoint()
        
            sys.stdout.flush()
            sys.stdout.write('\rEpoch: '+ str(epoch)+ ' Training Loss: '+ str(self.training_data["Loss"][-1])+
                  ' Testing Accuracy:'+ str(self.testing_data["Accuracy"][-1])+ ' Time: '+
                  str(epoch_end_time - epoch_start_time)+ 's')
            
    
        train_end_time = time.time()

        print('\nTotal Traning Time:\t', train_end_time - train_start_time)

    def _train_epoch(self):
        self._model.train()

        loss_data = 0.0
        accuracy_data = 0.0
    
        for batch_idx, (data, target) in enumerate(self._training_loader):   
            # Computer loss, use the optimizer object to zero all of the gradients
            # Then backpropagate and step the optimizer
            self._optimizer.zero_grad()
            data = data.to(self._device)
            target = target.to(self._device)
            
            outputs = self._model(data)

            accuracy_pass = self._calculate_accuracy(outputs[0], target)
            accuracy_data = accuracy_pass/(batch_idx+1) + accuracy_data*batch_idx/(batch_idx+1)

            loss = self._loss_function(outputs[0], target)
            loss_data = loss.item()/(batch_idx+1) + loss_data*batch_idx/(batch_idx+1)
            
            loss.backward()
            self._optimizer.step()    
        
        self.training_data["Loss"].append(loss_data)
        self.training_data["Accuracy"].append(accuracy_data)

    def _validate_model(self):
        with torch.no_grad():
            self._model.eval()

            loss_data = 0.0
            accuracy_data = 0.0

            for batch_idx, (data, target) in enumerate(self._validation_loader):   
                # Caculate the loss and accuracy for the predictions
                data = data.to(self._device)
                target = target.to(self._device)

                outputs = self._model(data)
                
                accuracy_pass = self._calculate_accuracy(outputs[0], target)
                accuracy_data = accuracy_pass/(batch_idx+1) + accuracy_data*batch_idx/(batch_idx+1)
                
                loss = self._loss_function(outputs[0], target)
                loss_data = loss.item()/(batch_idx+1) + loss_data*batch_idx/(batch_idx+1)

        self.validation_data["Loss"].append(loss_data)
        self.validation_data["Accuracy"].append(accuracy_data)

    def _test_model(self):
        with torch.no_grad():
            self._model.eval()

            accuracy = 0.0

            for batch_idx, (data, target) in enumerate(self._testing_loader):   
                # Caculate the loss and accuracy for the predictions
                data = data.to(self._device)
                target = target.to(self._device)

                outputs = self._model(data)
            
                accuracy_pass = self._calculate_accuracy(outputs[0], target)
                accuracy = accuracy_pass/(batch_idx+1) + accuracy*batch_idx/(batch_idx+1)

            self.testing_data["Accuracy"].append(accuracy)

    def plot_data(self):
        """
        This plots all the training statistics
        """
        plt.figure(figsize=(18, 5))
        plt.suptitle(self._modelname + ' Training, Validation and Testing Results')

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
        plt.plot(self.testing_data["Accuracy"])
        plt.legend(["Training", "Validation", "Testing"])
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
