#!/usr/bin/env python3

__author__ = "Bryce Ferenczi"
__email__ = "bryce.ferenczi@monashmotorsport.com"

import sys
import time
from pathlib import Path
from typing import Dict, List, Union
import numpy as np

import torch
import apex.amp as amp
import matplotlib.pyplot as plt

from nnet_training.utilities.metrics import OpticFlowMetric
from nnet_training.utilities.visualisation import flow_to_image
from nnet_training.training_frameworks.trainer_base_class import ModelTrainer

__all__ = ['MonoFlowTrainer']

def build_pyramid(image: torch.Tensor, lvl_stp: List[int]) -> List[torch.Tensor]:
    """
    Given an base image and list of steps, creates a image pyramid\n
    For example (1920,1080), [4,2,1] will give a pyramid [(512,256),(1024,512),(2048,1024)]
    """
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
                 loss_fn: Dict[str, torch.nn.Module], lr_cfg: Dict[str, Union[str, float]],
                 dataldr: Dict[str, torch.utils.data.DataLoader],
                 modelpath: Path, amp_cfg="O0", checkpoints=True):
        '''
        Initialize the Model trainer giving it a nn.Model, nn.Optimizer and dataloaders as
        a dictionary with Training, Validation and Testing loaders
        '''
        self._loss_function = loss_fn['flow']
        self.metric_loggers = {
            'flow' : OpticFlowMetric(base_dir=modelpath, main_metric="SAD", savefile='flow_data')
        }

        super(MonoFlowTrainer, self).__init__(model, optim, dataldr, lr_cfg,
                                              modelpath, amp_cfg, checkpoints)

    def _train_epoch(self, max_epoch):

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

            if all(key in data.keys() for key in ["flow", "flow_mask"]):
                flow_gt = {"flow": data['flow'].to(self._device),
                           "flow_mask": data['flow_mask'].to(self._device)}
            else:
                flow_gt = None

            # Computer loss, use the optimizer object to zero all of the gradients
            # Then backpropagate and step the optimizer
            pred_flow = self._model(img, img_seq)
            loss, _, _, _ = self._loss_function(pred_flow, img, img_seq)

            self._optimizer.zero_grad()
            with amp.scale_loss(loss, self._optimizer) as scaled_loss:
                scaled_loss.backward()
            self._optimizer.step()

            with torch.no_grad():
                self.metric_loggers['flow'].add_sample(
                    img, img_seq, pred_flow['flow_fw'][0].detach(),
                    flow_gt, loss=loss.item()
                )

            if not batch_idx % 10:
                time_elapsed = time.time() - start_time
                time_remain = time_elapsed / (batch_idx + 1) * \
                    (len(self._training_loader) - (batch_idx + 1))
                sys.stdout.flush()
                sys.stdout.write(f'\rTrain Epoch: [{self.epoch:2d}/{max_epoch:2d}] || '
                                 f'Iter [{batch_idx + 1:4d}/{len(self._validation_loader):4d}] || '
                                 f'lr: {self._lr_manager.get_lr():.8f} || '
                                 f'Loss: {loss.item():.4f} || '
                                 f'Time Elapsed: {time_elapsed:.2f} sec || '
                                 f'Est Time Remain: {time_remain:.2f} sec')

    def _validate_model(self, max_epoch):
        with torch.no_grad():

            start_time = time.time()

            for batch_idx, data in enumerate(self._validation_loader):
                # Put both image and target onto device
                img = data['l_img'].to(self._device)
                img_seq = data['l_seq'].to(self._device)

                if all(key in data.keys() for key in ["flow", "flow_mask"]):
                    flow_gt = {"flow": data['flow'].to(self._device),
                               "flow_mask": data['flow_mask'].to(self._device)}
                else:
                    flow_gt = None

                pred_flow = self._model(img, img_seq)

                # Caculate the loss and accuracy for the predictions
                loss, _, _, _ = self._loss_function(pred_flow, img, img_seq)

                self.metric_loggers['flow'].add_sample(
                    img, img_seq, pred_flow['flow_fw'][0],
                    flow_gt, loss=loss.item()
                )

                if not batch_idx % 10:
                    batch_acc = self.metric_loggers['flow'].get_last_batch()
                    time_elapsed = time.time() - start_time
                    time_remain = time_elapsed / (batch_idx + 1) * \
                        (len(self._validation_loader) - (batch_idx + 1))
                    sys.stdout.flush()
                    sys.stdout.write(f'\rValidaton Epoch: [{self.epoch:2d}/{max_epoch:2d}] || '
                                     f'Iter [{batch_idx+1:4d}/{len(self._validation_loader):4d}] ||'
                                     f' Accuracy: {batch_acc:.4f} || Loss: {loss.item():.4f} || '
                                     f'Time Elapsed: {time_elapsed:.2f} sec || '
                                     f'Est Time Remain: {time_remain:.2f} sec')

    def visualize_output(self):
        """
        Forward pass over a testing batch and displays the output
        """
        with torch.no_grad():
            self._model.eval()
            data = next(iter(self._validation_loader))
            left  = data['l_img'].to(self._device)
            seq_left = data['l_seq'].to(self._device)

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

                vis_flow = flow_to_image(np_flow_12[i].transpose([1, 2, 0]))

                plt.subplot(1, 3, 3)
                plt.imshow(vis_flow)
                plt.xlabel("Predicted Flow")

                plt.suptitle(f"Propagation time: {propagation_time}")
                plt.show()

if __name__ == "__main__":
    raise NotImplementedError
