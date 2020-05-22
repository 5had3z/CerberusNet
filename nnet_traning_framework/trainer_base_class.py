#!/usr/bin/env python3

__author__ = "Bryce Ferenczi"
__email__ = "bryce.ferenczi@monashmotorsport.com"

from datetime import datetime
import torch
import os
import time
import sys
from pathlib import Path

import torchvision.transforms
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

from metrics import SegmentationMetric
from lr_scheduler import LRScheduler

class ModelTrainer():
    def __init__(self, model, optimizer, loss_fn, dataloaders, learning_rate=1e-4, modelname=None, checkpoints=True):
        '''
        Initialize the Model trainer giving it a nn.Model, nn.Optimizer and dataloaders as
        a dictionary with Training, Validation and Testing loaders
        '''
        self._device = torch.device( "cuda" if torch.cuda.is_available() else "cpu" )

        self._training_loader = dataloaders["Training"]
        self._validation_loader = dataloaders["Validation"]

        self.epoch = 0
        self.best_acc = 0.0

        self._model = model.to(self._device)
        self._optimizer = optimizer
        self._loss_function = loss_fn

        self._checkpoints = checkpoints

        if not os.path.isdir(Path.cwd() / "torch_models"):
            os.makedirs(Path.cwd() / "torch_models")
        
        if modelname is not None:
            self._modelname = modelname 
            self._path = Path.cwd() / "torch_models" / str(modelname + ".pth")
        else:
            self._modelname = str(datetime.now()).replace(" ", "_")
            self._path = Path.cwd() / "torch_models" / str(self._modelname + ".pth")

        self._metric = SegmentationMetric(19, filename=self._modelname)
            
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
            self.epoch = len(self._metric)
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
            'model_state_dict':         self._model.state_dict(),
            'optimizer_state_dict':     self._optimizer.state_dict(), 
        }, self._path)
        self._metric.save_epoch()
        self.best_acc = self._metric.max_accuracy()[0]

    def train_model(self, n_epochs):
        train_start_time = time.time()

        max_epoch = self.epoch + n_epochs

        self._lr_manager = LRScheduler(mode='poly', base_lr=0.01, nepochs=n_epochs,
                                iters_per_epoch=len(self._training_loader), power=0.9)

        while self.epoch < max_epoch:
            self.epoch += 1
            epoch_start_time = time.time()

            # Calculate the training loss, training duration for each epoch, and validation accuracy
            self._train_epoch(max_epoch)
            self._validate_model(max_epoch)

            epoch_end_time = time.time()

            _, mIoU, loss = self._metric._get_epoch_statistics()

            if (self.best_acc < mIoU or True) and self._checkpoints:
                self.save_checkpoint()
        
            sys.stdout.flush()
            sys.stdout.write('\rEpoch: '+ str(self.epoch)+
                    ' Training Loss: '+ str(loss)+
                    ' Testing Accuracy:'+ str(mIoU)+
                    ' Time: '+ str(epoch_end_time - epoch_start_time)+ 's')
    
        train_end_time = time.time()

        print('\nTotal Traning Time:\t', train_end_time - train_start_time)

    def _train_epoch(self, max_epoch):
        self._model.train()

        self._metric.new_epoch('training')

        start_time = time.time()

        for batch_idx, (data, target) in enumerate(self._training_loader):
            cur_lr = self._lr_manager(batch_idx)
            for param_group in self._optimizer.param_groups:
                param_group['lr'] = cur_lr
            
            # Put both image and target onto device
            data = data.to(self._device)
            target = target.to(self._device)
            
            # Computer loss, use the optimizer object to zero all of the gradients
            # Then backpropagate and step the optimizer
            outputs = self._model(data)

            loss = self._loss_function(outputs, target)

            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

            self._metric._add_sample(
                torch.argmax(outputs,dim=1,keepdim=True).cpu().data.numpy(),
                target.cpu().data.numpy(),
                loss=loss.item()
            )

            if batch_idx % 10 == 0:
                time_elapsed = time.time() - start_time
                time_remain = time_elapsed / (batch_idx + 1) * (len(self._training_loader) - (batch_idx + 1))
                print('Train Epoch: [%2d/%2d] Iter [%4d/%4d] || lr: %.8f || Loss: %.4f || Time Elapsed: %4.4f sec || Est Time Remain: %4.4f sec' % (
                        self.epoch, max_epoch, batch_idx + 1, len(self._training_loader),
                        self._lr_manager.get_lr(), loss.item(), time_elapsed, time_remain))
        
    def _validate_model(self, max_epoch):
        with torch.no_grad():
            self._model.eval()

            self._metric.new_epoch('validation')

            start_time = time.time()

            for batch_idx, (data, target) in enumerate(self._validation_loader):
                # Put both image and target onto device
                data = data.to(self._device)
                target = target.to(self._device)

                outputs = self._model(data)
                
                # Caculate the loss and accuracy for the predictions
                loss = self._loss_function(outputs, target)

                _, miou = self._metric._add_sample(
                    torch.argmax(outputs,dim=1,keepdim=True).cpu().data.numpy(),
                    target.cpu().numpy(),
                    loss=loss.item()
                )
                
                if batch_idx % 10 == 0:
                    time_elapsed = time.time() - start_time
                    time_remain = time_elapsed / (batch_idx + 1) * (len(self._validation_loader) - (batch_idx + 1))
                    print('Validaton Epoch: [%2d/%2d] Iter [%4d/%4d] || Accuracy: %.4f || Loss: %.4f || Time Elapsed: %4.4f sec || Est Time Remain: %4.4f sec' % (
                            self.epoch, max_epoch, batch_idx + 1, len(self._validation_loader),
                            miou, loss.item(), time_elapsed, time_remain))

    def _test_model(self):
        raise NotImplementedError
        # with torch.no_grad():
        #     self._model.eval()

        #     for batch_idx, (data, target) in enumerate(self._testing_loader):   
        #         # Caculate the loss and accuracy for the predictions
        #         data = data.to(self._device)
        #         target = target.to(self._device)

        #         outputs = self._model(data)
