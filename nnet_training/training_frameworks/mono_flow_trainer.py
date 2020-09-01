#!/usr/bin/env python3

__author__ = "Bryce Ferenczi"
__email__ = "bryce.ferenczi@monashmotorsport.com"

import os, sys, time, platform, multiprocessing
from pathlib import Path
from typing import Dict, TypeVar
T = TypeVar('T')
import numpy as np

import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from nnet_training.utilities.metrics import OpticFlowMetric
from nnet_training.utilities.visualisation import flow_to_image
from nnet_training.training_frameworks.trainer_base_class import ModelTrainer

__all__ = ['MonoFlowTrainer', 'flow_to_image']

def build_pyramid(image, lvl_stp=[8, 4, 2, 1]):
    pyramid = []
    for level in lvl_stp:
        pyramid.append(torch.nn.functional.interpolate(
            image, scale_factor=1./level, mode='bilinear', align_corners=True))
    return pyramid

class MonoFlowTrainer(ModelTrainer):
    '''
    Monocular Flow Training Class
    '''
    def __init__(self, model: torch.nn.Module, optim: torch.optim.Optimizer,
                 loss_fn: Dict[str, torch.nn.Module], lr_cfg: Dict[str, T],
                 dataldr: Dict[str, torch.utils.data.DataLoader],
                 modelpath: Path, checkpoints=True):
        '''
        Initialize the Model trainer giving it a nn.Model, nn.Optimizer and dataloaders as
        a dictionary with Training, Validation and Testing loaders
        '''
        self._loss_function = loss_fn
        self._metric = OpticFlowMetric(base_dir=modelpath, savefile='flow_data')
        super(MonoFlowTrainer, self).__init__(model, optim, dataldr,
                                              lr_cfg, modelpath, checkpoints)

    def save_checkpoint(self):
        super(MonoFlowTrainer, self).save_checkpoint()
        self._metric.save_epoch()

    def load_checkpoint(self):
        if os.path.isfile(self._path):
            self.epoch = len(self._metric)
        super(MonoFlowTrainer, self).load_checkpoint()

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

            # img_pyr     = self.build_pyramid(img)
            # img_seq_pyr = self.build_pyramid(img_seq)

            # Computer loss, use the optimizer object to zero all of the gradients
            # Then backpropagate and step the optimizer
            pred_flow = self._model(img, img_seq)
            flows_12, flows_21 = pred_flow['flow_fw'], pred_flow['flow_bw']
            flows = [torch.cat([flo12, flo21], 1) for flo12, flo21 in
                     zip(flows_12, flows_21)]
            loss, _, _, _ = self._loss_function(flows, img, img_seq)

            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

            self._metric._add_sample(
                img.cpu().data.numpy(),
                pred_flow['flow_fw'][0].cpu().data.numpy(),
                img_seq.cpu().data.numpy(),
                None,
                loss=loss.item()
            )

            if not batch_idx % 10:
                time_elapsed = time.time() - start_time
                time_remain = time_elapsed / (batch_idx + 1) * (len(self._training_loader) - (batch_idx + 1))
                sys.stdout.flush()
                sys.stdout.write('\rTrain Epoch: [%2d/%2d] Iter [%4d/%4d] || lr: %.8f || Loss: %.4f \
                    || Time Elapsed: %.2f sec || Est Time Remain: %.2f sec' % (
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
                flows_12, flows_21 = pred_flow['flow_fw'], pred_flow['flow_bw']
                flows = [torch.cat([flo12, flo21], 1) for flo12, flo21 in
                        zip(flows_12, flows_21)]
                loss, _, _, _ = self._loss_function(flows, img, img_seq)

                self._metric._add_sample(
                    img.cpu().data.numpy(),
                    pred_flow['flow_fw'][0].cpu().data.numpy(),
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
            data = next(iter(self._validation_loader))
            left        = data['l_img'].to(self._device)
            seq_left    = data['l_seq'].to(self._device)

            start_time = time.time()
            flow_12 = self._model(left, seq_left)['flow_fw'][0]
            propagation_time = (time.time() - start_time)/self._validation_loader.batch_size

            np_flow_12 = flow_12.detach().cpu().numpy()

            for i in range(self._validation_loader.batch_size):
                plt.subplot(1, 3, 1)
                plt.imshow(np.moveaxis(left[i, 0:3, :, :].cpu().numpy(), 0, 2))
                plt.xlabel("Base Image")

                plt.subplot(1, 3, 2)
                plt.imshow(np.moveaxis(seq_left[i, :, :].cpu().numpy(), 0, 2))
                plt.xlabel("Sequential Image")

                vis_flow = self.flow_to_image(np_flow_12[i].transpose([1, 2, 0]))

                plt.subplot(1, 3, 3)
                plt.imshow(vis_flow)
                plt.xlabel("Predicted Flow")

                plt.suptitle("Propagation time: " + str(propagation_time))
                plt.show()

if __name__ == "__main__":
    from nnet_training.nnet_models.nnet_models import MonoFlow1
    from nnet_training.nnet_models.pwcnet import PWCNet
    from nnet_training.utilities.loss_functions import ReconstructionLossV1, ReconstructionLossV2
    from nnet_training.utilities.UnFlowLoss import unFlowLoss
    from nnet_training.utilities.dataset import CityScapesDataset

    BATCH_SIZE = 4
    if platform.system() == 'Windows':
        n_workers = 0
    else:
        n_workers = min(multiprocessing.cpu_count(), BATCH_SIZE)

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

    datasets = {
        'Training'   : CityScapesDataset(training_dir, crop_fraction=1, output_size=(1024, 512)),
        'Validation' : CityScapesDataset(validation_dir, crop_fraction=1, output_size=(1024, 512))
    }

    dataloaders = {
        'Training'   : DataLoader(datasets["Training"], batch_size=BATCH_SIZE,
                                 shuffle=True, num_workers=n_workers, drop_last=True),
        'Validation' : DataLoader(datasets["Validation"], batch_size=BATCH_SIZE,
                                 shuffle=True, num_workers=n_workers, drop_last=True),
    }

    Model = PWCNet()
    optimizer = torch.optim.Adam(Model.parameters(), betas=(0.9, 0.99), lr=1e-4, weight_decay=1e-6)
    photometric_weights = {"l1":0.15, "ssim":0.85}
    lossfn = unFlowLoss(photometric_weights).to(torch.device("cuda"))

    BASEPATH = Path.cwd() / "torch_models"

    lr_sched = {"lr": 1e-4, "mode":"constant"}
    modeltrainer = MonoFlowTrainer(model=Model, optim=optimizer, loss_fn=lossfn,
                                   dataldr=dataloaders, lr_cfg=lr_sched, modelpath=BASEPATH)
    modeltrainer.visualize_output()
    # modeltrainer.train_model(5)
