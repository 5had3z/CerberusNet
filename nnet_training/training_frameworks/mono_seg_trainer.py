#!/usr/bin/env python3

__author__ = "Bryce Ferenczi"
__email__ = "bryce.ferenczi@monashmotorsport.com"

import time
from pathlib import Path
from typing import Dict, Union
import numpy as np

import torch
import apex.amp as amp
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from nnet_training.utilities.metrics import SegmentationMetric
from nnet_training.utilities.visualisation import get_color_pallete

from .trainer_base_class import ModelTrainer

__all__ = ['MonoSegmentationTrainer']

class MonoSegmentationTrainer(ModelTrainer):
    def __init__(self, model: torch.nn.Module, optim: torch.optim.Optimizer,
                 loss_fn: Dict[str, torch.nn.Module], lr_cfg: Dict[str, Union[str, float]],
                 dataldr: Dict[str, torch.utils.data.DataLoader],
                 modelpath: Path, amp_cfg="O0", checkpoints=True):
        '''
        Initialize the Model trainer giving it a nn.Model, nn.Optimizer and dataloaders as
        a dictionary with Training, Validation and Testing loaders
        '''
        self.metric_loggers = {
            'seg': SegmentationMetric(19, base_dir=modelpath, main_metric="IoU",
                                      savefile='segmentation_data')
        }
        self._loss_function = loss_fn['segmentation']

        super(MonoSegmentationTrainer, self).__init__(model, optim, dataldr, lr_cfg,
                                                      modelpath, amp_cfg, checkpoints)

    def _train_epoch(self, max_epoch):

        start_time = time.time()

        for batch_idx, data in enumerate(self._training_loader):
            self._training_loader.dataset.resample_scale()
            cur_lr = self._lr_manager(batch_idx)
            for param_group in self._optimizer.param_groups:
                param_group['lr'] = cur_lr

            # Put both image and target onto device
            image   = data['l_img'].to(self._device)
            target  = data['seg'].to(self._device)

            # Computer loss, use the optimizer object to zero all of the gradients
            # Then backpropagate and step the optimizer
            outputs = self._model(image)

            loss = self._loss_function(outputs, target)

            self._optimizer.zero_grad()
            with amp.scale_loss(loss, self._optimizer) as scaled_loss:
                scaled_loss.backward()
            self._optimizer.step()

            self.metric_loggers['seg'].add_sample(
                torch.argmax(outputs, dim=1, keepdim=True).cpu().data.numpy(),
                target.cpu().data.numpy(),
                loss=loss.item()
            )

            if batch_idx % 10 == 0:
                time_elapsed = time.time() - start_time
                time_remain = time_elapsed / (batch_idx + 1) * (len(self._training_loader) - (batch_idx + 1))
                print('Train Epoch: [%2d/%2d] Iter [%4d/%4d] || lr: %.8f || Loss: %.4f || Time Elapsed: %.2f sec || Est Time Remain: %.2f sec' % (
                        self.epoch, max_epoch, batch_idx + 1, len(self._training_loader),
                        self._lr_manager.get_lr(), loss.item(), time_elapsed, time_remain))

    def _validate_model(self, max_epoch):
        with torch.no_grad():

            start_time = time.time()

            self._training_loader.dataset.resample_scale(True)
            for batch_idx, data in enumerate(self._validation_loader):
                # Put both image and target onto device
                image   = data['l_img'].to(self._device)
                target  = data['seg'].to(self._device)

                outputs = self._model(image)

                # Caculate the loss and accuracy for the predictions
                loss = self._loss_function(outputs, target)

                self.metric_loggers['seg'].add_sample(
                    torch.argmax(outputs, dim=1, keepdim=True).cpu().data.numpy(),
                    target.cpu().numpy(),
                    loss=loss.item()
                )
                
                if batch_idx % 10 == 0:
                    batch_acc = self.metric_loggers['seg'].get_last_batch()
                    time_elapsed = time.time() - start_time
                    time_remain = time_elapsed / (batch_idx + 1) * (len(self._validation_loader) - (batch_idx + 1))
                    print('Validaton Epoch: [%2d/%2d] Iter [%4d/%4d] || Accuracy: %.4f || Loss: %.4f || Time Elapsed: %.2f sec || Est Time Remain: %.2f sec' % (
                        self.epoch, max_epoch, batch_idx + 1, len(self._validation_loader),
                        batch_acc, loss.item(), time_elapsed, time_remain))

    def visualize_output(self):
        """
        Forward pass over a testing batch and displays the output
        """
        with torch.no_grad():
            self._model.eval()
            image, seg = next(iter(self._validation_loader))
            image = image.to(self._device)

            start_time = time.time()
            output = self._model(image)
            propagation_time = (time.time() - start_time)/self._validation_loader.batch_size

            pred = torch.argmax(output,dim=1,keepdim=True)
            for i in range(self._validation_loader.batch_size):
                plt.subplot(1,3,1)
                plt.imshow(np.moveaxis(image[i,0:3,:,:].cpu().numpy(), 0, 2))
                plt.xlabel("Base Image")

                plt.subplot(1, 3, 2)
                plt.imshow(get_color_pallete(seg[i, :, :]))
                plt.xlabel("Ground Truth")

                plt.subplot(1, 3, 3)
                plt.imshow(get_color_pallete(pred.cpu().numpy()[i, 0, :, :]))
                plt.xlabel("Prediction")

                plt.suptitle("Propagation time: " + str(propagation_time))
                plt.show()

    def custom_image(self, filename):
        """
        Forward Pass on a single image
        """
        with torch.no_grad():
            from PIL import Image

            self._model.eval()

            image = Image.open(filename)

            img_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.ToPILImage(),
                transforms.Resize([1024, 2048]),
                transforms.ToTensor()
            ])
            image = img_transform(image).unsqueeze(0)
            image = image.to(self._device)

            start_time = time.time()
            output = self._model(image)
            propagation_time = (time.time() - start_time)

            pred = torch.argmax(output[0], dim=1, keepdim=True)

            plt.subplot(1, 2, 1)
            plt.imshow(np.moveaxis(image[0, :, :, :].cpu().numpy(), 0, 2))
            plt.xlabel("Base Image")

            plt.subplot(1, 2, 2)
            plt.imshow(pred.cpu().numpy()[0, 0, :, :])
            plt.xlabel("Prediction")

            plt.suptitle("Propagation time: " + str(propagation_time))
            plt.show()

    def plot_data(self):
        super(MonoSegmentationTrainer, self).plot_data()
        self.metric_loggers['seg'].plot_classwise_iou()


if __name__ == "__main__":
    raise NotImplementedError
