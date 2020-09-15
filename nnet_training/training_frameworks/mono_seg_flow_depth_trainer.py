#!/usr/bin/env python3

__author__ = "Bryce Ferenczi"
__email__ = "bryce.ferenczi@monashmotorsport.com"

import sys
import time
from pathlib import Path
from typing import Dict, TypeVar
T = TypeVar('T')
import numpy as np

import torch
import matplotlib.pyplot as plt

from nnet_training.utilities.metrics import OpticFlowMetric, SegmentationMetric, DepthMetric
from nnet_training.utilities.visualisation import flow_to_image, get_color_pallete
from nnet_training.training_frameworks.trainer_base_class import ModelTrainer

__all__ = ['MonoSegFlowDepthTrainer']

class MonoSegFlowDepthTrainer(ModelTrainer):
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
        self._seg_loss_fn = loss_fn['segmentation']
        self._flow_loss_fn = loss_fn['flow']
        self._depth_loss_fn = loss_fn['depth']

        self.metric_loggers = {
            'seg' : SegmentationMetric(19, base_dir=modelpath, savefile='seg_data'),
            'flow': OpticFlowMetric(base_dir=modelpath, savefile='flow_data'),
            'depth': DepthMetric(base_dir=modelpath, savefile='depth_data')
        }

        super(MonoSegFlowDepthTrainer, self).__init__(model, optim, dataldr, lr_cfg,
                                                 modelpath, checkpoints)

    def _train_epoch(self, max_epoch):
        self._model.train()

        start_time = time.time()

        for batch_idx, data in enumerate(self._training_loader):
            cur_lr = self._lr_manager(batch_idx)
            for param_group in self._optimizer.param_groups:
                param_group['lr'] = cur_lr

            # Put both image and target onto device
            img = data['l_img'].to(self._device)
            img_seq = data['l_seq'].to(self._device)
            seg_gt = data['seg'].to(self._device)
            depth_gt = data['disparity'].to(self._device)

            if all(key in data.keys() for key in ["flow", "flow_mask"]):
                flow_gt = {"flow": data['flow'].to(self._device),
                           "flow_mask": data['flow_mask'].to(self._device)}
            else:
                flow_gt = None

            # Computer loss, use the optimizer object to zero all of the gradients
            # Then backpropagate and step the optimizer
            pred_flow, depth_pred, seg_pred =\
                self._model(im1_rgb=img, im2_rgb=img_seq, seg_gt=seg_gt)

            flow_loss, _, _, _ = self._flow_loss_fn(
                pred_flow=pred_flow, im1_origin=img, im2_origin=img_seq)

            seg_loss = self._seg_loss_fn(
                seg_pred=seg_pred['seg_fw'], seg_gt=seg_gt)

            depth_loss = self._depth_loss_fn(
                disp_pred_pyr=depth_pred['depth_fw'], disp_gt=depth_gt)

            loss = flow_loss + seg_loss + depth_loss

            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

            with torch.no_grad():
                self.metric_loggers['flow']._add_sample(
                    img, img_seq, pred_flow['flow_fw'][0],
                    flow_gt, loss=flow_loss.item()
                )

                self.metric_loggers['seg']._add_sample(
                    torch.argmax(seg_pred['seg_fw'], dim=1, keepdim=True).cpu().data.numpy(),
                    seg_gt.cpu().data.numpy(),
                    loss=seg_loss.item()
                )

                self.metric_loggers['depth']._add_sample(
                    depth_pred['depth_fw'][0].cpu().data.numpy(),
                    data['disparity'].data.numpy(),
                    loss=depth_loss.item()
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
                img = data['l_img'].to(self._device)
                img_seq = data['l_seq'].to(self._device)
                seg_gt = data['seg'].to(self._device)
                depth_gt = data['disparity'].to(self._device)

                if all(key in data.keys() for key in ["flow", "flow_mask"]):
                    flow_gt = {"flow": data['flow'].to(self._device),
                               "flow_mask": data['flow_mask'].to(self._device)}
                else:
                    flow_gt = None

                # Caculate the loss and accuracy for the predictions
                pred_flow, depth_pred, seg_pred = self._model(im1_rgb=img, im2_rgb=img_seq)

                flow_loss, _, _, _ = self._flow_loss_fn(
                    pred_flow=pred_flow, im1_origin=img, im2_origin=img_seq)

                seg_loss = self._seg_loss_fn(
                    seg_pred=seg_pred['seg_fw'], seg_gt=seg_gt)

                depth_loss = self._depth_loss_fn(
                    disp_pred_pyr=depth_pred['depth_fw'], disp_gt=depth_gt)

                self.metric_loggers['flow']._add_sample(
                    img, img_seq, pred_flow['flow_fw'][0],
                    flow_gt, loss=flow_loss.item()
                )

                self.metric_loggers['seg']._add_sample(
                    torch.argmax(seg_pred['seg_fw'], dim=1, keepdim=True).cpu().data.numpy(),
                    data['seg'].data.numpy(),
                    loss=seg_loss.item()
                )

                self.metric_loggers['depth']._add_sample(
                    depth_pred['depth_fw'][0].cpu().data.numpy(),
                    data['disparity'].data.numpy(),
                    loss=depth_loss.item()
                )

                if not batch_idx % 10:
                    loss = flow_loss + seg_loss + depth_loss
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
            left = data['l_img'].to(self._device)
            seq_left = data['l_seq'].to(self._device)

            start_time = time.time()
            flow_12, depth_pred, seg_pred = self._model(left, seq_left)
            propagation_time = (time.time() - start_time)/self._validation_loader.batch_size

            np_flow_12 = flow_12['flow_fw'][0].detach().cpu().numpy()
            sed_pred_cpu = torch.argmax(seg_pred['seg_fw'], dim=1, keepdim=True).cpu().numpy()
            depth_pred_cpu = depth_pred['depth_fw'][0].detach().cpu().numpy()

            for i in range(self._validation_loader.batch_size):
                plt.subplot(2, 4, 1)
                plt.imshow(np.moveaxis(left[i, 0:3, :, :].cpu().numpy(), 0, 2))
                plt.xlabel("Base Image")

                plt.subplot(2, 4, 2)
                plt.imshow(np.moveaxis(seq_left[i, :, :].cpu().numpy(), 0, 2))
                plt.xlabel("Sequential Image")

                vis_flow = flow_to_image(np_flow_12[i].transpose([1, 2, 0]))

                plt.subplot(2, 4, 3)
                plt.imshow(vis_flow)
                plt.xlabel("Predicted Flow")

                plt.subplot(2, 4, 4)
                plt.imshow(data['disparity'][i, :, :])
                plt.xlabel("Ground Truth Disparity")

                plt.subplot(2, 4, 5)
                plt.imshow(get_color_pallete(data['seg'].numpy()[i, :, :]))
                plt.xlabel("Ground Truth Segmentation")

                plt.subplot(2, 4, 6)
                plt.imshow(get_color_pallete(sed_pred_cpu[i, 0, :, :]))
                plt.xlabel("Predicted Segmentation")

                if "flow" in data:
                    vis_flow = flow_to_image(data['flow'].numpy()[i].transpose([1, 2, 0]))
                    plt.subplot(2, 4, 7)
                    plt.imshow(vis_flow)
                    plt.xlabel("Ground Truth Flow")

                plt.subplot(2, 4, 8)
                plt.imshow(depth_pred_cpu[i, 0, :, :])
                plt.xlabel("Predicted Depth")

                plt.suptitle("Propagation time: " + str(propagation_time))
                plt.show()

    def plot_data(self):
        super(MonoSegFlowDepthTrainer, self).plot_data()
        self.metric_loggers['seg'].plot_classwise_iou()

if __name__ == "__main__":
    raise NotImplementedError
