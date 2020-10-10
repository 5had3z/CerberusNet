#!/usr/bin/env python3

__author__ = "Bryce Ferenczi"
__email__ = "bryce.ferenczi@monashmotorsport.com"

import os
import sys
import json
import time
import argparse
import hashlib
import platform
import multiprocessing

from pathlib import Path
from easydict import EasyDict
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
from nnet_training.nnet_models import get_model

from nnet_training.utilities.visualisation import flow_to_image, get_color_pallete
from nnet_training.utilities.KITTI import Kitti2015Dataset
from nnet_training.utilities.CityScapes import CityScapesDataset
from nnet_training.utilities.metrics import SegmentationMetric, DepthMetric, OpticFlowMetric

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MIN_DEPTH = 0.
MAX_DEPTH = 80.

def initialise_evaluation(config_json: EasyDict, experiment_path: Path)\
        -> Tuple[torch.nn.Module, torch.utils.data.DataLoader]:
    """
    Sets up the network and dataloader configurations
    Returns dataloader and model
    """
    if platform.system() == 'Windows':
        n_workers = 0
    else:
        n_workers = min(multiprocessing.cpu_count(), config_json.dataset.batch_size)

    if config_json.dataset.type == "Kitti":
        dataset = Kitti2015Dataset(
            config_json.dataset.rootdir, config_json.dataset.objectives,
            output_size=config_json.dataset.augmentations.output_size,
            disparity_out=config_json.dataset.augmentations.disparity_out,
            img_normalize=config_json.dataset.augmentations.img_normalize)
    elif config_json.dataset.type == "Cityscapes":
        validation_dirs = {}
        for subset in config_json.dataset.val_subdirs:
            validation_dirs[str(subset)] = config_json.dataset.rootdir +\
                                            config_json.dataset.val_subdirs[str(subset)]
        dataset = CityScapesDataset(
            validation_dirs, output_size=config_json.dataset.augmentations.output_size,
            disparity_out=config_json.dataset.augmentations.disparity_out,
            img_normalize=config_json.dataset.augmentations.img_normalize)
    else:
        raise NotImplementedError(config_json.dataset.type)

    dataloader = torch.utils.data.DataLoader(
        dataset, num_workers=n_workers,
        batch_size=config_json.dataset.batch_size,
        drop_last=config_json.dataset.drop_last,
        shuffle=config_json.dataset.shuffle
    )

    model = get_model(config_json.model).to(DEVICE)

    modelweights = experiment_path / (str(model)+"_latest.pth")
    checkpoint = torch.load(modelweights, map_location=torch.device(DEVICE))
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"Checkpoint loaded from {str(modelweights)}\n",
          f"Starting from epoch: {str(checkpoint['epochs'])}")

    return model, dataloader

def run_evaluation(model, dataloader, loggers):
    start_time = time.time()

    for batch_idx, data in enumerate(dataloader):
        # Put both image and target onto device
        img = data['l_img'].to(DEVICE)
        img_seq = data['l_seq'].to(DEVICE)

        forward = model(im1_rgb=img, im2_rgb=img_seq)

        if all(key in data.keys() for key in ["flow", "flow_mask"]):
            flow_gt = {"flow": data['flow'].to(DEVICE),
                       "flow_mask": data['flow_mask'].to(DEVICE)}
        else:
            flow_gt = None

        if 'flow' in forward.keys():
            loggers['flow'].add_sample(
                img, img_seq, forward['flow'][0], flow_gt
            )

        if 'seg' in forward.keys() and 'seg' in data.keys():
            loggers['seg'].add_sample(
                torch.argmax(forward['seg'], dim=1, keepdim=True), data['seg'].to(DEVICE)
            )

        if 'l_disp' in data.keys() and 'depth' in forward.keys():
            loggers['depth'].add_sample(
                forward['depth'], data['l_disp'].to(DEVICE)
            )

        if not batch_idx % 10:
            sys.stdout.write(f'\rValidaton Iter: [{batch_idx+1:4d}/{len(dataloader):4d}]')

            if 'seg' in forward.keys():
                sys.stdout.write(f" || {loggers['seg'].main_metric}: "\
                                    f"{loggers['seg'].get_last_batch():.4f}")
            if 'l_disp' in data.keys() and 'depth' in forward.keys():
                sys.stdout.write(f" || {loggers['depth'].main_metric}: "\
                                    f"{loggers['depth'].get_last_batch():.4f}")
            if 'flow' in forward.keys():
                sys.stdout.write(f" || {loggers['flow'].main_metric}: "\
                                    f"{loggers['flow'].get_last_batch():.4f}")

            time_elapsed = time.time() - start_time
            time_remain = time_elapsed/(batch_idx+1)*(len(dataloader)-(batch_idx+1))
            sys.stdout.write(f' || Time Elapsed: {time_elapsed:.1f} s'\
                                f' Remain: {time_remain:.1f} s')
            sys.stdout.flush()

def display_output(model, dataloader):
    data = next(iter(dataloader))

    start_time = time.time()
    forward = model(data['l_img'].to(DEVICE), data['l_seq'].to(DEVICE))
    propagation_time = (time.time() - start_time)/dataloader.batch_size

    np_flow_12 = forward['flow'][0].detach().cpu().numpy()
    sed_pred_cpu = torch.argmax(forward['seg'], dim=1, keepdim=True).cpu().numpy()

    if 'depth' in forward:
        depth_pred_cpu = forward['depth'].detach().cpu().numpy()

    if hasattr(dataloader.dataset, 'img_normalize'):
        img_mean = dataloader.dataset.img_normalize.mean
        img_std = dataloader.dataset.img_normalize.std
        inv_mean = [-mean / std for mean, std in zip(img_mean, img_std)]
        inv_std = [1 / std for std in img_std]
        img_norm = torchvision.transforms.Normalize(inv_mean, inv_std)
    else:
        img_norm = torchvision.transforms.Normalize([0, 0, 0], [1, 1, 1])

    for i in range(dataloader.batch_size):
        plt.subplot(2, 4, 1)
        plt.imshow(np.moveaxis(img_norm(data['l_img'][i, :, :]).numpy(), 0, 2))
        plt.xlabel("Base Image")

        plt.subplot(2, 4, 2)
        plt.imshow(np.moveaxis(img_norm(data['l_seq'][i, :, :]).numpy(), 0, 2))
        plt.xlabel("Sequential Image")

        if "flow" in data:
            plt.subplot(2, 4, 3)
            plt.imshow(flow_to_image(data['flow'].numpy()[i].transpose([1, 2, 0])))
            plt.xlabel("Ground Truth Flow")

        if 'l_disp' in data:
            plt.subplot(2, 4, 4)
            plt.imshow(data['l_disp'][i, :, :], cmap='magma',
                       vmin=MIN_DEPTH, vmax=MAX_DEPTH)
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

        if 'depth' in forward:
            plt.subplot(2, 4, 8)
            plt.imshow(depth_pred_cpu[i, 0, :, :], cmap='magma',
                       vmin=MIN_DEPTH, vmax=MAX_DEPTH)
            plt.xlabel("Predicted Depth")

        plt.suptitle("Propagation time: " + str(propagation_time))
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='configs/HRNetV2_sfd_kt.json')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = EasyDict(json.load(f))

    encoding = hashlib.md5(json.dumps(cfg).encode('utf-8'))
    training_path = Path.cwd() / "torch_models" / str(encoding.hexdigest())
    if not os.path.isdir(training_path):
        raise EnvironmentError("Existing not Found")

    print("Experiment # ", encoding.hexdigest())

    MODEL, DATALOADER = initialise_evaluation(cfg, training_path)

    LOGGERS = {
        'seg' : SegmentationMetric(19, base_dir=Path.cwd(), main_metric="IoU", savefile=''),
        'flow': OpticFlowMetric(base_dir=Path.cwd(), main_metric="EPE", savefile=''),
        'depth': DepthMetric(base_dir=Path.cwd(), main_metric="RMSE_Log", savefile='')
    }

    with torch.no_grad():
        MODEL.eval()
        run_evaluation(MODEL, DATALOADER, LOGGERS)

        for name, logger in LOGGERS.items():
            print(f'\n{name}')
            logger.print_epoch_statistics()

        while bool(input("Display Example? (Y): ")):
            torch.cuda.empty_cache()
            display_output(MODEL, DATALOADER)
