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

from nnet_training.utilities.metrics import OpticFlowMetric, SegmentationMetric
from nnet_training.utilities.dataset import CityScapesDataset
from nnet_training.utilities.visualisation import flow_to_image, get_color_pallete
from nnet_training.training_frameworks.trainer_base_class import ModelTrainer

__all__ = ['MonoSegFlowTrainer']

class MonoSegFlowTrainer(ModelTrainer):
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
        super(MonoSegFlowTrainer, self).__init__(model, optim, dataldr, lr_cfg,
                                                 modelpath, checkpoints)

        self._seg_loss_fn = loss_fn['segmentation']
        self._flow_loss_fn = loss_fn['flow']

        self.metric_loggers['seg'] = SegmentationMetric(19, base_dir=modelpath, savefile='seg_data')
        self.metric_loggers['flow'] = OpticFlowMetric(base_dir=modelpath, savefile='flow_data')

    def _train_epoch(self, max_epoch):
        self._model.train()

        start_time = time.time()

        for batch_idx, data in enumerate(self._training_loader):
            cur_lr = self._lr_manager(batch_idx)
            for param_group in self._optimizer.param_groups:
                param_group['lr'] = cur_lr

            # Put both image and target onto device
            img     = data['l_img'].to(self._device)
            img_seq = data['l_seq'].to(self._device)
            seg_gt  = data['seg'].to(self._device)

            # Computer loss, use the optimizer object to zero all of the gradients
            # Then backpropagate and step the optimizer
            pred_flow, seg_pred = self._model(img, img_seq)
            flows_12, flows_21 = pred_flow['flow_fw'], pred_flow['flow_bw']
            flows = [torch.cat([flo12, flo21], 1) for flo12, flo21 in
                     zip(flows_12, flows_21)]
            flow_loss, _, _, _ = self._flow_loss_fn(flows, img, img_seq)
            seg_loss = self._seg_loss_fn(seg_pred, seg_gt)

            loss = flow_loss + seg_loss

            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

            self.metric_loggers['flow']._add_sample(
                img.cpu().data.numpy(),
                pred_flow['flow_fw'][0].cpu().data.numpy(),
                img_seq.cpu().data.numpy(),
                None,
                loss=flow_loss.item()
            )

            self.metric_loggers['seg']._add_sample(
                torch.argmax(seg_pred, dim=1, keepdim=True).cpu().data.numpy(),
                seg_gt.cpu().data.numpy(),
                loss=seg_loss.item()
            )

            if not batch_idx % 10:
                time_elapsed = time.time() - start_time
                time_remain = time_elapsed/(batch_idx+1)*(len(self._training_loader)-(batch_idx+1))
                sys.stdout.flush()
                sys.stdout.write('\rTrain Epoch: [%2d/%2d] Iter [%4d/%4d] || lr: %.8f || Loss: %.4f || Time Elapsed: %.2f sec || Est Time Remain: %.2f sec' % (
                    self.epoch, max_epoch, batch_idx + 1, len(self._training_loader),
                    self._lr_manager.get_lr(), loss.item(), time_elapsed, time_remain))

    def _validate_model(self, max_epoch):
        with torch.no_grad():
            self._model.eval()

            start_time = time.time()

            for batch_idx, data in enumerate(self._validation_loader):
                # Put both image and target onto device
                img     = data['l_img'].to(self._device)
                img_seq = data['l_seq'].to(self._device)
                seg_gt  = data['seg'].to(self._device)

                # Caculate the loss and accuracy for the predictions
                pred_flow, seg_pred = self._model(img, img_seq)
                flows_12, flows_21 = pred_flow['flow_fw'], pred_flow['flow_bw']
                flows = [torch.cat([flo12, flo21], 1) for flo12, flo21 in
                         zip(flows_12, flows_21)]
                flow_loss, _, _, _ = self._flow_loss_fn(flows, img, img_seq)
                seg_loss = self._seg_loss_fn(seg_pred, seg_gt)

                self.metric_loggers['flow']._add_sample(
                    img.cpu().data.numpy(),
                    pred_flow['flow_fw'][0].cpu().data.numpy(),
                    img_seq.cpu().data.numpy(),
                    None,
                    loss=flow_loss.item()
                )

                self.metric_loggers['seg']._add_sample(
                    torch.argmax(seg_pred, dim=1, keepdim=True).cpu().data.numpy(),
                    seg_gt.cpu().data.numpy(),
                    loss=seg_loss.item()
                )

                if not batch_idx % 10:
                    loss = flow_loss + seg_loss
                    seg_acc = self.metric_loggers['seg'].get_last_batch()
                    time_elapsed = time.time() - start_time
                    time_remain = time_elapsed/(batch_idx+1)*(len(self._validation_loader)-(batch_idx+1))
                    sys.stdout.flush()
                    sys.stdout.write('\rValidaton Epoch: [%2d/%2d] Iter [%4d/%4d] || Seg mIoU: %.4f || Loss: %.4f || Time Elapsed: %.2f sec || Est Time Remain: %.2f sec' % (
                        self.epoch, max_epoch, batch_idx + 1, len(self._validation_loader),
                        seg_acc, loss.item(), time_elapsed, time_remain))

    def visualize_output(self):
        """
        Forward pass over a testing batch and displays the output
        """
        with torch.no_grad():
            self._model.eval()
            data = next(iter(self._validation_loader))
            left     = data['l_img'].to(self._device)
            seq_left = data['l_seq'].to(self._device)
            seg_gt   = data['seg']

            start_time = time.time()
            flow_12, seg_pred = self._model(left, seq_left)
            propagation_time = (time.time() - start_time)/self._validation_loader.batch_size

            np_flow_12 = flow_12['flow_fw'][0].detach().cpu().numpy()
            pred_cpu = torch.argmax(seg_pred, dim=1, keepdim=True).cpu().numpy()

            for i in range(self._validation_loader.batch_size):
                plt.subplot(2, 3, 1)
                plt.imshow(np.moveaxis(left[i, 0:3, :, :].cpu().numpy(), 0, 2))
                plt.xlabel("Base Image")

                plt.subplot(2, 3, 2)
                plt.imshow(np.moveaxis(seq_left[i, :, :].cpu().numpy(), 0, 2))
                plt.xlabel("Sequential Image")

                vis_flow = flow_to_image(np_flow_12[i].transpose([1, 2, 0]))

                plt.subplot(2, 3, 3)
                plt.imshow(vis_flow)
                plt.xlabel("Predicted Flow")

                plt.subplot(2, 3, 4)
                plt.imshow(get_color_pallete(seg_gt.numpy()[i, :, :]))
                plt.xlabel("Ground Truth Segmentation")

                plt.subplot(2, 3, 5)
                plt.imshow(get_color_pallete(pred_cpu[i, 0, :, :]))
                plt.xlabel("Predicted Segmentation")

                plt.suptitle("Propagation time: " + str(propagation_time))
                plt.show()

from nnet_training.nnet_models.mono_segflow import MonoSFNet
from nnet_training.utilities.UnFlowLoss import unFlowLoss
from nnet_training.utilities.loss_functions import FocalLoss2D

if __name__ == "__main__":
    BATCH_SIZE = 2
    if platform.system() == 'Windows':
        n_workers = 0
    else:
        n_workers = min(multiprocessing.cpu_count(), BATCH_SIZE)

    base_dir = '/media/bryce/4TB Seagate/Autonomous Vehicles Data/Cityscapes Data/'
    training_dir = {
        'images'    : base_dir + 'leftImg8bit/train',
        'left_seq'  : base_dir + 'leftImg8bit_sequence/train',
        'seg'       : base_dir + 'gtFine/train'
    }
    validation_dir = {
        'images'    : base_dir + 'leftImg8bit/val',
        'left_seq'  : base_dir + 'leftImg8bit_sequence/val',
        'seg'       : base_dir + 'gtFine/val',
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

    MODEL = MonoSFNet()
    OPTIM = torch.optim.Adam(MODEL.parameters(), betas=(0.9, 0.99), lr=1e-4, weight_decay=1e-6)
    PHOTOMETRIC_WEIGHTS = {"l1":0.15, "ssim":0.85}
    LOSS_FN = {
        "flow"          : unFlowLoss(PHOTOMETRIC_WEIGHTS).to(torch.device("cuda")),
        "segmentation"  : FocalLoss2D(gamma=2, ignore_index=-1).to(torch.device("cuda"))
    }
    BASEPATH = Path.cwd() / "torch_models" # str(MODEL)+'_Adam_Fcl_Uflw_HRes'

    LR_SCHED = {"lr": 1e-4, "mode":"constant"}
    MODELTRAINER = MonoSegFlowTrainer(model=MODEL, optim=OPTIM, loss_fn=LOSS_FN,
                                      dataldr=dataloaders, lr_cfg=LR_SCHED,
                                      modelpath=BASEPATH)

    # MODELTRAINER.visualize_output()
    MODELTRAINER.train_model(3)
