#!/usr/bin/env python3

__author__ = "Bryce Ferenczi"
__email__ = "bryce.ferenczi@monashmotorsport.com"

import os, sys, time, platform, multiprocessing
import numpy as np
from pathlib import Path

print(Path.cwd())

import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms

from .utilities.loss_functions import ReconstructionLossV1, ReconstructionLossV2
from .utilities.metrics import OpticFlowMetric
from .utilities.dataset import CityScapesDataset
from trainer_base_class import ModelTrainer

__all__ = ['MonoFlowTrainer']

class MonoFlowTrainer(ModelTrainer):
    def __init__(self, model, optimizer, loss_fn, dataloaders, learning_rate=1e-4, modelname=None, checkpoints=True):
        '''
        Initialize the Model trainer giving it a nn.Model, nn.Optimizer and dataloaders as
        a dictionary with Training, Validation and Testing loaders
        '''
        self._loss_function = loss_fn
        self._metric = OpticFlowMetric(filename=modelname)
        super(MonoFlowTrainer, self).__init__(model, optimizer, dataloaders, learning_rate, modelname, checkpoints)

    def save_checkpoint(self):
        super(MonoFlowTrainer, self).save_checkpoint()
        self._metric.save_epoch()

    def load_checkpoint(self):
        if os.path.isfile(self._path):
            self.epoch = len(self._metric)
        super(MonoFlowTrainer, self).load_checkpoint()

    def build_pyramid(self, image, lvl_stp=[0.5,0.5,0.5]):
        pyramid = [image]
        for level in lvl_stp:
            pyr_img = torch.nn.functional.interpolate(pyramid[-1], scale_factor=level, mode='bilinear', align_corners=True)
            pyramid.append(pyr_img)
        return pyramid

    def _train_epoch(self, max_epoch):
        self._model.train()

        self._metric.new_epoch('training')

        start_time = time.time()

        for batch_idx, data in enumerate(self._training_loader):
            cur_lr = self._lr_manager(batch_idx)
            for param_group in self._optimizer.param_groups:
                param_group['lr'] = cur_lr
            
            # Put both image and target onto device
            img     = data['l_img'].to(self._device)
            img_seq = data['l_seq'].to(self._device)
            
            # Computer loss, use the optimizer object to zero all of the gradients
            # Then backpropagate and step the optimizer
            pred_flow   = self._model(img, img_seq)

            loss        = self._loss_function(img, pred_flow, img_seq)

            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

            self._metric._add_sample(
                img.cpu().data.numpy(),
                pred_flow.cpu().data.numpy(),
                img_seq.cpu().data.numpy(),
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
                img         = data['l_img'].to(self._device)
                img_seq     = data['l_seq'].to(self._device)

                pred_flow   = self._model(img, img_seq)
                
                # Caculate the loss and accuracy for the predictions
                loss        = self._loss_function(img, pred_flow, img_seq)

                self._metric._add_sample(
                    img.cpu().data.numpy(),
                    pred_flow.cpu().data.numpy(),
                    img_seq.cpu().data.numpy(),
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
            image, seq_img = next(iter(self._validation_loader))
            left        = image['l_img'].to(self._device)
            seq_left    = seq_img['l_seq'].to(self._device)

            start_time = time.time()
            pred_l = self._model(left, seq_left)
            propagation_time = (time.time() - start_time)/self._validation_loader.batch_size

            for i in range(self._validation_loader.batch_size):
                plt.subplot(2,2,1)
                plt.imshow(np.moveaxis(left[i,0:3,:,:].cpu().numpy(),0,2))
                plt.xlabel("Base Image")

                plt.subplot(2,2,2)
                plt.imshow(pred_l.cpu().numpy()[i,0,:,:])
                plt.xlabel("Predicted Flow")

                recon = self.reconstruct_flow( left[i,0:3,:,:].cpu().numpy(),
                                pred_l.cpu().numpy()[i,0,:,:] )

                plt.subplot(2,2,3)
                plt.imshow(recon)
                plt.xlabel("Predicted Reconstruction")

                plt.subplot(2,2,4)
                plt.imshow(seq_left[i,:,:])
                plt.xlabel("Sequential Image")

                plt.suptitle("Propagation time: " + str(propagation_time))
                plt.show()

    def reconstruct_flow(self, image, flow):
        return image * flow


from ..nnet_models.nnet_models import MonoFlow1

if __name__ == "__main__":
    print(Path.cwd())
    if platform.system() == 'Windows':
        n_workers = 0
    else:
        n_workers = multiprocessing.cpu_count()

    base_dir = '/media/bryce/4TB Seagate/Autonomous Vehicles Data/Cityscapes Data/'
    training_dir = {
        'images'    : base_dir + 'leftImg8bit/train',
        'left_seq'  : base_dir + 'leftImg8bit_sequence/train',
        'cam'       : base_dir + 'camera/train'
    }
    validation_dir = {
        'images'    : base_dir + 'leftImg8bit/val',
        'left_seq'  : base_dir + 'leftImg8bit_sequence/val',
        'cam'       : base_dir + 'camera/val'
    }

    datasets = dict(
        Training    = CityScapesDataset(training_dir, crop_fraction=1, output_size=(1024,512)),
        Validation  = CityScapesDataset(validation_dir, crop_fraction=1, output_size=(1024,512))
    )

    dataloaders = dict(
        Training    = DataLoader(datasets["Training"], batch_size=8, shuffle=True, num_workers=n_workers, drop_last=True),
        Validation  = DataLoader(datasets["Validation"], batch_size=8, shuffle=True, num_workers=n_workers, drop_last=True),
    )

    Model = MonoFlow1()
    optimizer = torch.optim.SGD(Model.parameters(), lr=0.01, momentum=0.9)
    lossfn = ReconstructionLossV1(img_w=1024, img_h=512, device=torch.device("cuda")).to(torch.device("cuda"))
    filename = str(Model)+'_SGD_Recon'

    modeltrainer = MonoFlowTrainer(Model, optimizer, lossfn, dataloaders, learning_rate=0.01, modelname=filename)
    modeltrainer.visualize_output()
    # modeltrainer.train_model(1)
