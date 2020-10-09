#!/usr/bin/env python3

__author__ = "Bryce Ferenczi"
__email__ = "bryce.ferenczi@monashmotorsport.com"

import sys
import time
from pathlib import Path
from typing import Dict, Union

import numpy as np

import torch
import torchvision
import apex.amp as amp
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
                 loss_fn: Dict[str, torch.nn.Module], lr_cfg: Dict[str, Union[float, str]],
                 dataldr: Dict[str, torch.utils.data.DataLoader],
                 modelpath: Path, amp_cfg="O0", checkpoints=True):
        '''
        Initialize the Model trainer giving it a nn.Model, nn.Optimizer and dataloaders as
        a dictionary with Training, Validation and Testing loaders
        '''
        self._seg_loss_fn = loss_fn['segmentation']
        self._flow_loss_fn = loss_fn['flow']
        self._depth_loss_fn = loss_fn['depth']

        self.metric_loggers = {
            'seg' : SegmentationMetric(
                19, base_dir=modelpath, main_metric="IoU", savefile='seg_data'),
            'flow': OpticFlowMetric(base_dir=modelpath, main_metric="SAD", savefile='flow_data'),
            'depth': DepthMetric(base_dir=modelpath, main_metric="RMSE_Log", savefile='depth_data')
        }

        super(MonoSegFlowDepthTrainer, self).__init__(model, optim, dataldr, lr_cfg,
                                                      modelpath, amp_cfg, checkpoints)

    def _train_epoch(self, max_epoch):

        start_time = time.time()

        for batch_idx, data in enumerate(self._training_loader):
            cur_lr = self._lr_manager(batch_idx)
            for param_group in self._optimizer.param_groups:
                param_group['lr'] = cur_lr

            # Put both image and target onto device
            cuda_s = torch.cuda.Stream()
            with torch.cuda.stream(cuda_s):
                img = data['l_img'].cuda(non_blocking=True)
                img_seq = data['l_seq'].cuda(non_blocking=True)
                seg_gt = data['seg'].cuda(non_blocking=True)
                depth_gt = data['l_disp'].cuda(non_blocking=True)

                if all(key in data.keys() for key in ["flow", "flow_mask"]):
                    flow_gt = {"flow": data['flow'].cuda(non_blocking=True),
                               "flow_mask": data['flow_mask'].cuda(non_blocking=True)}
                else:
                    flow_gt = None
            cuda_s.synchronize()

            # Computer loss, use the optimizer object to zero all of the gradients
            # Then backpropagate and step the optimizer
            forward = self._model(im1_rgb=img, im2_rgb=img_seq)

            flow_loss, _, _, _ = self._flow_loss_fn(
                pred_flow_fw=forward['flow'],
                pred_flow_bw=forward['flow_b'],
                im1_origin=img, im2_origin=img_seq)

            seg_loss = self._seg_loss_fn(
                seg_pred=forward, seg_gt=seg_gt)

            depth_loss = self._depth_loss_fn(
                disp_pred=forward['depth'], disp_gt=depth_gt)

            loss = flow_loss + seg_loss + depth_loss

            self._optimizer.zero_grad()
            with amp.scale_loss(loss, self._optimizer) as scaled_loss:
                scaled_loss.backward()
            self._optimizer.step()

            with torch.no_grad():
                self.metric_loggers['flow'].add_sample(
                    img, img_seq, forward['flow'][0],
                    flow_gt, loss=flow_loss.item()
                )

                self.metric_loggers['seg'].add_sample(
                    torch.argmax(forward['seg'], dim=1, keepdim=True),
                    data['seg'], loss=seg_loss.item()
                )

                self.metric_loggers['depth'].add_sample(
                    forward['depth'], depth_gt,
                    loss=depth_loss.item()
                )

            if not batch_idx % 10:
                time_elapsed = time.time() - start_time
                time_remain = time_elapsed / (batch_idx + 1) * \
                    (len(self._training_loader) - (batch_idx + 1))
                sys.stdout.flush()
                sys.stdout.write(f'\rTrain Epoch: [{self.epoch:2d}/{max_epoch:2d}] || '
                                 f'Iter [{batch_idx + 1:4d}/{len(self._training_loader):4d}] || '
                                 f'lr: {self._lr_manager.get_lr():.8f} || '
                                 f'Loss: {loss.item():.4f} || '
                                 f'Time Elapsed: {time_elapsed:.2f} s || '
                                 f'Est Time Remain: {time_remain:.2f} s')

    def _validate_model(self, max_epoch):
        with torch.no_grad():

            start_time = time.time()

            for batch_idx, data in enumerate(self._validation_loader):
                # Put both image and target onto device
                cuda_s = torch.cuda.Stream()
                with torch.cuda.stream(cuda_s):
                    img = data['l_img'].cuda(non_blocking=True)
                    img_seq = data['l_seq'].cuda(non_blocking=True)
                    seg_gt = data['seg'].cuda(non_blocking=True)
                    depth_gt = data['l_disp'].cuda(non_blocking=True)

                    if all(key in data.keys() for key in ["flow", "flow_mask"]):
                        flow_gt = {"flow": data['flow'].cuda(non_blocking=True),
                                   "flow_mask": data['flow_mask'].cuda(non_blocking=True)}
                    else:
                        flow_gt = None
                cuda_s.synchronize()

                # Caculate the loss and accuracy for the predictions
                forward = self._model(im1_rgb=img, im2_rgb=img_seq)

                flow_loss, _, _, _ = self._flow_loss_fn(
                    pred_flow_fw=forward['flow'], pred_flow_bw=forward['flow_b'],
                    im1_origin=img, im2_origin=img_seq)

                seg_loss = self._seg_loss_fn(
                    seg_pred=forward, seg_gt=seg_gt)

                depth_loss = self._depth_loss_fn(
                    disp_pred=forward['depth'], disp_gt=depth_gt)

                self.metric_loggers['flow'].add_sample(
                    img, img_seq, forward['flow'][0],
                    flow_gt, loss=flow_loss.item()
                )

                self.metric_loggers['seg'].add_sample(
                    torch.argmax(forward['seg'], dim=1, keepdim=True),
                    data['seg'], loss=seg_loss.item()
                )

                self.metric_loggers['depth'].add_sample(
                    forward['depth'], depth_gt,
                    loss=depth_loss.item()
                )

                if not batch_idx % 10:
                    loss = flow_loss + seg_loss + depth_loss
                    seg_acc = self.metric_loggers['seg'].get_last_batch()
                    time_elapsed = time.time() - start_time
                    time_remain = time_elapsed / (batch_idx + 1) * \
                        (len(self._validation_loader) - (batch_idx + 1))
                    sys.stdout.flush()
                    sys.stdout.write(f'\rValidaton Epoch: [{self.epoch:2d}/{max_epoch:2d}] || '
                                     f'Iter [{batch_idx+1:4d}/{len(self._validation_loader):4d}] ||'
                                     f' Accuracy: {seg_acc:.4f} mIoU || Loss: {loss:.4f} || '
                                     f'Time Elapsed: {time_elapsed:.2f} sec || '
                                     f'Est Time Remain: {time_remain:.2f} sec')

    def visualize_output(self):
        """
        Forward pass over a testing batch and displays the output
        """
        MIN_DEPTH = 0.
        MAX_DEPTH = 80.
        with torch.no_grad():
            self._model.eval()
            data = next(iter(self._validation_loader))
            left = data['l_img'].to(self._device)
            seq_left = data['l_seq'].to(self._device)

            start_time = time.time()
            forward = self._model(left, seq_left)
            propagation_time = (time.time() - start_time)/self._validation_loader.batch_size

            np_flow_12 = forward['flow'][0].detach().cpu().numpy()
            sed_pred_cpu = torch.argmax(forward['seg'], dim=1, keepdim=True).cpu().numpy()
            depth_pred_cpu = forward['depth'].detach().cpu().numpy()

            if hasattr(self._validation_loader.dataset, 'img_normalize'):
                img_mean = self._validation_loader.dataset.img_normalize.mean
                img_std = self._validation_loader.dataset.img_normalize.std
                inv_mean = [-mean / std for mean, std in zip(img_mean, img_std)]
                inv_std = [1 / std for std in img_std]
                img_norm = torchvision.transforms.Normalize(inv_mean, inv_std)
            else:
                img_norm = torchvision.transforms.Normalize([0, 0, 0], [1, 1, 1])

            for i in range(self._validation_loader.batch_size):
                plt.subplot(2, 4, 1)
                plt.imshow(np.moveaxis(img_norm(left[i, :, :]).cpu().numpy(), 0, 2))
                plt.xlabel("Base Image")

                plt.subplot(2, 4, 2)
                plt.imshow(np.moveaxis(img_norm(seq_left[i, :, :]).cpu().numpy(), 0, 2))
                plt.xlabel("Sequential Image")

                if "flow" in data:
                    plt.subplot(2, 4, 3)
                    plt.imshow(flow_to_image(data['flow'].numpy()[i].transpose([1, 2, 0])))
                    plt.xlabel("Ground Truth Flow")

                plt.subplot(2, 4, 4)
                plt.imshow(data['l_disp'][i, :, :], cmap='magma', vmin=MIN_DEPTH, vmax=MAX_DEPTH)
                plt.xlabel("Ground Truth Disparity")

                plt.subplot(2, 4, 5)
                plt.imshow(get_color_pallete(data['seg'].numpy()[i, :, :]))
                plt.xlabel("Ground Truth Segmentation")

                plt.subplot(2, 4, 6)
                plt.imshow(get_color_pallete(sed_pred_cpu[i, 0, :, :]))
                plt.xlabel("Predicted Segmentation")

                plt.subplot(2, 4, 7)
                plt.imshow(flow_to_image(np_flow_12[i].transpose([1, 2, 0])))
                plt.xlabel("Predicted Flow")

                plt.subplot(2, 4, 8)
                plt.imshow(depth_pred_cpu[i, 0, :, :], cmap='magma', vmin=MIN_DEPTH, vmax=MAX_DEPTH)
                plt.xlabel("Predicted Depth")

                plt.suptitle(f"Propagation time: {propagation_time:.3f}")
                plt.show()

    def plot_data(self):
        super(MonoSegFlowDepthTrainer, self).plot_data()
        self.metric_loggers['seg'].plot_classwise_iou()

if __name__ == "__main__":
    raise NotImplementedError
