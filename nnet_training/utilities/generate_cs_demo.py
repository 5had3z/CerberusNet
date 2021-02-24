#!/usr/bin/env python3.8

"""
Generates a demonstration for qualitative analysis of nnet model performance in each task.
"""

import os
import re
import sys
import json
import hashlib
import argparse
import platform
import multiprocessing
from typing import List
from pathlib import Path
from easydict import EasyDict

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from cityscapesscripts.helpers.labels import labels

from nnet_training.loss_functions.UnFlowLoss import flow_warp
from nnet_training.evaluate_model import data_to_gpu
from nnet_training.nnet_models import get_model
from nnet_training.utilities.visualisation import flow_to_image, CITYSPALLETTE
from nnet_training.datasets.cityscapes_dataset import CityScapesDataset

IMG_EXT = '.png'
MIN_DEPTH = 0.
MAX_DEPTH = 80.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VIDEO_HZ = 17.0

class CityScapesDemo(CityScapesDataset):
    def __init__(self, directory: Path, output_size=(1024, 512), **kwargs):
        super(CityScapesDemo, self).__init__(directories={}, output_size=output_size)
        self.rand_flip = False
        self.l_img = []
        self.l_seq = []

        srt_lmbda = lambda x: int(re.split("_", x)[3])
        for filename in sorted(os.listdir(directory), key=srt_lmbda):
            if filename.endswith(IMG_EXT):
                frame_n = int(re.split("_", filename)[3])
                seq_name = filename.replace(
                    str(frame_n).zfill(6), str(frame_n+1).zfill(6))

                if os.path.isfile(directory / seq_name):
                    self.l_img.append(directory / filename)
                    self.l_seq.append(directory / seq_name)

def get_config_argparse(base_path_list: List[Path]):
    """
    Returns a model config and its savepath given a list of directorys to search for the model.\n
    Uses argparse for seraching for the model or config argument.
    """
    parser = argparse.ArgumentParser()
    # parser.add_argument('-c', '--config', default='configs/HRNetV2_sfd_kt.json')
    parser.add_argument('-e', '--experiment', default='8f23c8346c898db41c5bc7c13c36da66')
    model_path = None

    if 'config' in parser.parse_args():
        with open(parser.parse_args().config) as conf_f:
            model_cfg = EasyDict(json.load(conf_f))

        cfg_enc = hashlib.md5(json.dumps(model_cfg).encode('utf-8'))

        for base_path in base_path_list:
            model_path = base_path / str(cfg_enc.hexdigest())
            if os.path.isdir(model_path):
                break

        if not os.path.isdir(model_path):
            raise EnvironmentError("Existing not Found")

        print("Experiment # ", cfg_enc.hexdigest())

    elif 'experiment' in parser.parse_args():
        for base_path in base_path_list:
            model_path = base_path / parser.parse_args().experiment
            if os.path.isdir(model_path):
                break

        if not os.path.isdir(model_path):
            raise EnvironmentError("Existing not Found")

        for filename in os.listdir(model_path):
            if filename.endswith('.json'):
                with open(model_path / filename) as conf_f:
                    model_cfg = EasyDict(json.load(conf_f))
                break

    return model_cfg, model_path

def get_loader_and_model(model_cfg: EasyDict, model_path: Path, data_dir: str):
    """
    Sets up the network and dataloader configurations
    Returns dataloader and model
    """
    if platform.system() == 'Windows':
        n_workers = 0
    else:
        n_workers = min(multiprocessing.cpu_count(), model_cfg.dataset.batch_size * 5)

    dataset = CityScapesDemo(
        data_dir, output_size=model_cfg.dataset.augmentations.output_size,
        img_normalize=model_cfg.dataset.augmentations.img_normalize
    )

    dataloader = torch.utils.data.DataLoader(
        dataset, num_workers=n_workers,
        batch_size=model_cfg.dataset.batch_size * 5,
    )

    model = get_model(model_cfg.model).to(DEVICE)

    modelweights = model_path / (model.modelname+"_latest.pth")
    checkpoint = torch.load(modelweights, map_location=torch.device(DEVICE))
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"Checkpoint loaded from {str(modelweights)}\n",
          f"Starting from epoch: {str(checkpoint['epochs'])}")

    return dataloader, model

def generate_quiver_image(flow: np.ndarray, base_image: np.ndarray):
    """
    Returns the base image overlayed with a quiver plot from the flow field
    """
    quiver_image = np.zeros_like(base_image)
    for y_1 in range(25, flow.shape[0], 25):
        for x_1 in range(25, flow.shape[1], 25):
            y_2 = np.clip(int(y_1 + flow[y_1, x_1, 1]), 0, flow.shape[0])
            x_2 = np.clip(int(x_1 + flow[y_1, x_1, 0]), 0, flow.shape[1])
            quiver_image = cv2.arrowedLine(
                quiver_image, (x_1, y_1), (x_2, y_2), [0, 255, 0], 2, tipLength=0.2)
    quiver_image = cv2.add(base_image, quiver_image)
    return quiver_image

def generate_static_mask(segmentation: torch.Tensor):
    """
    Given a segmentation mask returns a mask tensor of flat, construction and object
    """
    mask = torch.zeros_like(segmentation)
    for label in labels:
        if label.category in ['flat', 'construction', 'object']:
            mask[segmentation == label.trainId] = 1

    return mask

@torch.no_grad()
def slam_testing(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, path: str):
    """
    Testing odometry concept
    """
    est_speed_history = []

    vid_writer = cv2.VideoWriter(
        str(path/"speed.avi"), cv2.VideoWriter_fourcc(*'XVID'),
        VIDEO_HZ, tuple(dataloader.dataset.output_shape))

    for idx, batch_data in enumerate(dataloader):
        data_to_gpu(batch_data)
        forward = model(**batch_data, slam=True)

        batch_depth = forward['depth'].detach()
        batch_depth[batch_depth < MIN_DEPTH] = MIN_DEPTH
        batch_depth[batch_depth > MAX_DEPTH] = MAX_DEPTH

        batch_depth_seq = forward['depth_b'].detach()
        batch_depth_seq[batch_depth_seq < MIN_DEPTH] = MIN_DEPTH
        batch_depth_seq[batch_depth_seq > MAX_DEPTH] = MAX_DEPTH

        batch_flow = forward['flow'][0].detach()

        batch_seg = torch.argmax(forward['seg'], dim=1)
        seq_depth = flow_warp(batch_depth_seq, batch_flow)

        for i in range(batch_seg.shape[0]):
            diff_depth = (batch_depth[i] - seq_depth[i]).cpu().numpy()

            # Plot histogram of estimates
            # plt.hist(diff_depth[diff_depth != 0], bins='auto', log=True)
            # plt.title("Motion Estimation with Depth(t) - FlowWarp(Depth(t+1))")
            # plt.xlabel("Difference (m)")
            # plt.ylabel("Numer of Pixels")
            # plt.show()

            diff_depth *= generate_static_mask(batch_seg[i]).cpu().numpy()

            hist, bin_edges = np.histogram(
                diff_depth[diff_depth != 0],
                bins=np.arange(diff_depth.min(), diff_depth.max(), 0.05))

            est_speed_history.append(17.*bin_edges[np.argmax(hist)]*3.6)

            image_frame = np.moveaxis(
                batch_data['l_img'][i].cpu().numpy() * 255, 0, 2).astype(np.uint8)
            image_frame = cv2.cvtColor(image_frame, cv2.COLOR_RGB2BGR)

            if len(est_speed_history) < 5:
                est_speed = np.asarray(est_speed_history).mean()
            else:
                est_speed = np.asarray(est_speed_history[-5:]).mean()

            cv2.putText(
                image_frame, f'{est_speed:.3f} km/h', (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            vid_writer.write(image_frame)

        sys.stdout.write(f'\rParsing Video: [{idx+1:3d}/{len(dataloader):3d}]')
        sys.stdout.flush()

    vid_writer.release()

    plt.plot(np.asarray(est_speed_history))
    plt.xlabel("Image Sequence Index")
    plt.ylabel("Speed (km/h)")
    plt.show()

@torch.no_grad()
def generate_video(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader,
                   path: str):
    """
    Sequentially steps through dataloader and uses opencv to write to a video
    """

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    resolution = tuple(dataloader.dataset.output_shape)
    x_res, y_res = resolution
    seg_vid = cv2.VideoWriter(str(path/"seg.avi"), fourcc, VIDEO_HZ, resolution)
    depth_vid = cv2.VideoWriter(str(path/"depth.avi"), fourcc, VIDEO_HZ, resolution)
    flow_vid = cv2.VideoWriter(str(path/"flow.avi"), fourcc, VIDEO_HZ, resolution)
    quiver_vid = cv2.VideoWriter(str(path/"quiver.avi"), fourcc, VIDEO_HZ, resolution)
    composed_vid = cv2.VideoWriter(
        str(path/"composed.avi"), fourcc, VIDEO_HZ, tuple(2*res for res in resolution))

    color_lut = np.zeros((256, 3), dtype=np.uint8)
    color_lut[:len(CITYSPALLETTE)//3, :] = np.asarray(
        [CITYSPALLETTE[i:i + 3] for i in range(0, len(CITYSPALLETTE), 3)],
        dtype=np.uint8)

    composed_frame = np.zeros((2*y_res, 2*x_res, 3), dtype=np.uint8)

    print('Beginning video writing')
    for idx, batch_data in enumerate(dataloader):
        data_to_gpu(batch_data)
        forward = model(**batch_data)

        batch_depth = forward['depth'].detach().cpu().numpy()
        batch_depth[batch_depth < MIN_DEPTH] = MIN_DEPTH
        batch_depth[batch_depth > MAX_DEPTH] = MAX_DEPTH

        batch_seg = torch.argmax(forward['seg'], dim=1).cpu().numpy()

        batch_flow = forward['flow'][0].detach().cpu().numpy()

        for i in range(batch_seg.shape[0]):
            seg_frame = np.empty(shape=(y_res, x_res, 3), dtype=np.uint8)
            for j in range(3):
                seg_frame[..., j] = cv2.LUT(batch_seg[i].astype(np.uint8), color_lut[:, j])
            cv2.cvtColor(seg_frame, cv2.COLOR_RGB2BGR, dst=seg_frame)

            flow_transpose = batch_flow[i].transpose([1, 2, 0])
            flow_frame = flow_to_image(flow_transpose)

            depth_frame = cv2.applyColorMap(
                (batch_depth[i, 0] / MAX_DEPTH * 255).astype(np.uint8),
                cv2.COLORMAP_MAGMA)

            image_frame = np.moveaxis(
                batch_data['l_img'][i].cpu().numpy() * 255, 0, 2).astype(np.uint8)
            image_frame = cv2.cvtColor(image_frame, cv2.COLOR_RGB2BGR)

            composed_frame[:y_res, :x_res] = image_frame
            composed_frame[:y_res, x_res:] = seg_frame
            composed_frame[y_res:, :x_res] = depth_frame
            composed_frame[y_res:, x_res:] = flow_frame

            seg_vid.write(seg_frame)
            depth_vid.write(depth_frame)
            flow_vid.write(flow_frame)
            composed_vid.write(composed_frame)

            quiver_vid.write(generate_quiver_image(flow_transpose, image_frame))

        sys.stdout.write(f'\rWriting Video: [{idx+1:3d}/{len(dataloader):3d}]')
        sys.stdout.flush()

    seg_vid.release()
    depth_vid.release()
    flow_vid.release()
    composed_vid.release()
    quiver_vid.release()

    print("Finished Writing Videos")

if __name__ == "__main__":
    DATA_DIR = Path('/media/bryce/4TB Seagate/Autonomous Vehicles Data/Cityscapes Data'+\
                    '/leftImg8bit_demoVideo/leftImg8bit/demoVideo/stuttgart_00')
    BASE_PATHS = [Path.cwd() / "torch_models"]

    MODEL_CFG, MODEL_PTH = get_config_argparse(BASE_PATHS)

    DATALOADER, MODEL = get_loader_and_model(MODEL_CFG, MODEL_PTH, DATA_DIR)

    slam_testing(MODEL, DATALOADER, MODEL_PTH)

    # generate_video(MODEL, DATALOADER, MODEL_PTH)
