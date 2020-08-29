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

from nnet_training.utilities.metrics import SegmentationMetric, DepthMetric
from nnet_training.utilities.dataset import CityScapesDataset
from trainer_base_class import ModelTrainer

__all__ = ['StereoSegDepthTrainer']

class StereoSegDepthTrainer(ModelTrainer):
    def __init__(self, model, optimizer, loss_fn, dataloaders, lr_cfg, modelname=None, checkpoints=True):
        '''
        Initialize the Model trainer giving it a nn.Model, nn.Optimizer and dataloaders as
        a dictionary with Training, Validation and Testing loaders
        '''
        self._seg_loss_fn = loss_fn['segmentation']
        self._depth_loss_fn = loss_fn['depth']
        self._seg_metric = SegmentationMetric(19, filename=modelname+'_seg')
        self._depth_metric = DepthMetric(filename=modelname+'_depth')

        super(StereoSegDepthTrainer, self).__init__(model, optimizer, dataloaders, lr_cfg, modelname, checkpoints)

    def save_checkpoint(self):
        super(StereoSegDepthTrainer, self).save_checkpoint()
        self._seg_metric.save_epoch()
        self._depth_metric.save_epoch()

    def load_checkpoint(self):
        if os.path.isfile(self._path):
            self.epoch = len(self._seg_metric)
        super(StereoSegDepthTrainer, self).load_checkpoint()

    def _train_epoch(self, max_epoch):
        self._model.train()

        self._seg_metric.new_epoch('training')
        self._depth_metric.new_epoch('training')

        start_time = time.time()

        for batch_idx, data in enumerate(self._training_loader):
            cur_lr = self._lr_manager(batch_idx)
            for param_group in self._optimizer.param_groups:
                param_group['lr'] = cur_lr
            
            # Put both image and target onto device
            left        = data['l_img'].to(self._device)
            right       = data['r_img'].to(self._device)
            seg_gt      = data['seg'].to(self._device)
            depth_gt    = data['disparity'].to(self._device)
            baseline    = data['cam']['baseline_T'].to(self._device)

            # Computer loss, use the optimizer object to zero all of the gradients
            # Then backpropagate and step the optimizer
            seg_pred, depth_pred = self._model(left, right)

            seg_loss = self._seg_loss_fn(seg_pred, seg_gt)
            l_depth_loss = self._depth_loss_fn(left, depth_pred, right, baseline, data['cam'])
            r_depth_loss = self._depth_loss_fn(right, depth_pred, left, -baseline, data['cam'])
            loss = seg_loss + l_depth_loss + r_depth_loss

            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

            self._seg_metric._add_sample(
                torch.argmax(seg_pred,dim=1,keepdim=True).cpu().data.numpy(),
                seg_gt.cpu().data.numpy(),
                loss=seg_loss.item()
            )

            self._depth_metric._add_sample(
                depth_pred.cpu().data.numpy(),
                depth_gt.cpu().data.numpy(),
                loss=l_depth_loss.item() + r_depth_loss.item()
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

            self._seg_metric.new_epoch('validation')
            self._depth_metric.new_epoch('validation')

            start_time = time.time()

            for batch_idx, data in enumerate(self._validation_loader):
                # Put both image and target onto device
                left        = data['l_img'].to(self._device)
                right       = data['r_img'].to(self._device)
                seg_gt      = data['seg'].to(self._device)
                depth_gt    = data['disparity'].to(self._device)
                baseline    = data['cam']['baseline_T'].to(self._device)
           
                # Caculate the loss and accuracy for the predictions
                seg_pred, depth_pred = self._model(left, right)

                seg_loss = self._seg_loss_fn(seg_pred, seg_gt)
                l_depth_loss = self._depth_loss_fn(left, depth_pred, right, baseline, data['cam'])
                r_depth_loss = self._depth_loss_fn(right, depth_pred, left, -baseline, data['cam'])
                loss = seg_loss + l_depth_loss + r_depth_loss

                self._seg_metric._add_sample(
                    torch.argmax(seg_pred,dim=1,keepdim=True).cpu().data.numpy(),
                    seg_gt.cpu().data.numpy(),
                    loss=seg_loss.item()
                )

                self._depth_metric._add_sample(
                    depth_pred.cpu().data.numpy(),
                    depth_gt.cpu().data.numpy(),
                    loss=l_depth_loss.item() + r_depth_loss.item()
                )
                
                if not batch_idx % 10:
                    seg_acc = self._seg_metric.get_last_batch()
                    depth_acc = self._depth_metric.get_last_batch()
                    time_elapsed = time.time() - start_time
                    time_remain = time_elapsed / (batch_idx + 1) * (len(self._validation_loader) - (batch_idx + 1))
                    sys.stdout.flush()
                    sys.stdout.write('\rValidaton Epoch: [%2d/%2d] Iter [%4d/%4d] || Depth Acc: %.4f || Seg Acc: %.4f || Loss: %.4f || Time Elapsed: %.2f sec || Est Time Remain: %.2f sec' % (
                            self.epoch, max_epoch, batch_idx + 1, len(self._validation_loader),
                            depth_acc, seg_acc, loss.item(), time_elapsed, time_remain))

    def visualize_output(self):
        """
        Forward pass over a testing batch and displays the output
        """
        with torch.no_grad():
            self._model.eval()
            data     = next(iter(self._validation_loader))
            left     = data['l_img'].to(self._device)
            right    = data['r_img'].to(self._device)
            seg_gt   = data['seg']
            depth_gt = data['disparity']

            start_time = time.time()

            seg_pred, depth_pred = self._model(left, right)
            seg_pred = torch.argmax(seg_pred,dim=1,keepdim=True)

            propagation_time = (time.time() - start_time)/self._validation_loader.batch_size

            for i in range(self._validation_loader.batch_size):
                plt.subplot(1,5,1)
                plt.imshow(np.moveaxis(left[i,0:3,:,:].cpu().numpy(),0,2))
                plt.xlabel("Base Image")
        
                plt.subplot(1,5,2)
                plt.imshow(seg_gt[i,:,:])
                plt.xlabel("Segmentation Ground Truth")
        
                plt.subplot(1,5,3)
                plt.imshow(seg_pred.cpu().numpy()[i,0,:,:])
                plt.xlabel("Segmentation Prediction")

                plt.subplot(1,5,4)
                plt.imshow(depth_gt[i,:,:])
                plt.xlabel("Depth Ground Truth")
        
                plt.subplot(1,5,5)
                plt.imshow(depth_pred.cpu().numpy()[i,0,:,:])
                plt.xlabel("Depth Prediction")

                plt.suptitle("Propagation time: " + str(propagation_time))
                plt.show()

from nnet_training.utilities.loss_functions import FocalLoss2D, InvHuberLoss, ReconstructionLossV2
from nnet_training.nnet_models.nnet_models import StereoDepthSegSeparated2, StereoDepthSegSeparated3

if __name__ == "__main__":
    # multiprocessing.set_start_method('spawn', True)
    print(Path.cwd())
    if platform.system() == 'Windows':
        n_workers = 0
    else:
        n_workers = int(multiprocessing.cpu_count()/2)

    base_dir = '/home/bpfer2/sh99/bpfer2/cityscapes_data/'

    training_dir = {
        'images'        : base_dir + 'leftImg8bit/train',
        'right_images'  : base_dir + 'rightImg8bit/train',
        'seg'           : base_dir + 'gtFine/train',
        'disparity'     : base_dir + 'disparity/train',
        'cam'           : base_dir + 'camera/train'
    }

    validation_dir = {
        'images'        : base_dir + 'leftImg8bit/val',
        'right_images'  : base_dir + 'rightImg8bit/val',
        'seg'           : base_dir + 'gtFine/val',
        'disparity'     : base_dir + 'disparity/val',
        'cam'           : base_dir + 'camera/val'
    }

    datasets = dict(
        Training    = CityScapesDataset(training_dir, crop_fraction=2),
        Validation  = CityScapesDataset(validation_dir, crop_fraction=1)
    )

    dataloaders=dict(
        Training    = DataLoader(datasets["Training"], batch_size=12, shuffle=True, num_workers=n_workers, drop_last=True),
        Validation  = DataLoader(datasets["Validation"], batch_size=12, shuffle=True, num_workers=n_workers, drop_last=True)
    )

    Model = StereoDepthSegSeparated2()
    optimizer = torch.optim.SGD(Model.parameters(), lr=0.01, momentum=0.9)
    lossfn = dict(
        segmentation   = FocalLoss2D(gamma=1,ignore_index=-1).to(torch.device("cuda")),
        # depth          = InvHuberLoss(ignore_index=-1).to(torch.device("cuda"))
        depth          = ReconstructionLossV2(batch_size=6, height=512, width=1024, pred_type="depth").to(torch.device("cuda"))
    )

    filename = str(Model)+'_SGD_Fcl_Recon2'
    print("Loading " + filename)

    lr_sched = { "lr": 0.01, "mode":"poly" }
    modeltrainer = StereoSegDepthTrainer(Model, optimizer, lossfn, dataloaders, lr_cfg=lr_sched, modelname=filename)
    # modeltrainer.visualize_output()
    modeltrainer.train_model(100)
