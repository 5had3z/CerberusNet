#!/usr/bin/env python3.8

import os
import re
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

from nnet_training.nnet_models import get_model
from nnet_training.utilities.CityScapes import CityScapesDataset

IMG_EXT = '.png'
MIN_DEPTH = 0.
MAX_DEPTH = 80.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CityScapesDemo(CityScapesDataset):
    def __init__(self, directory: Path, output_size=(1024, 512), **kwargs):
        super(CityScapesDemo, self).__init__(directories={}, output_size=output_size)
        self.l_img = []
        self.l_seq = []

        for filename in os.listdir(directory):
            if filename.endswith(IMG_EXT):
                frame_n = int(re.split("_", filename)[2])
                seq_name = filename.replace(
                    str(frame_n).zfill(6), str(frame_n+1).zfill(6))

                if os.path.isfile(seq_name):
                    self.l_img.append(filename)
                    self.l_seq.append(seq_name)

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
        n_workers = min(multiprocessing.cpu_count(), model_cfg.dataset.batch_size)

    dataset = CityScapesDemo(
        data_dir, output_size=model_cfg.dataset.augmentations.output_size,
        img_normalize=model_cfg.dataset.augmentations.img_normalize
    )

    dataloader = torch.utils.data.DataLoader(
        dataset, num_workers=n_workers,
        batch_size=model_cfg.dataset.batch_size,
    )

    model = get_model(model_cfg.model).to(DEVICE)

    modelweights = model_path / (model.modelname+"_latest.pth")
    checkpoint = torch.load(modelweights, map_location=torch.device(DEVICE))
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"Checkpoint loaded from {str(modelweights)}\n",
          f"Starting from epoch: {str(checkpoint['epochs'])}")

    return dataloader, model

if __name__ == "__main__":
    DATA_DIR = '/media/bryce/4TB Seagate/Autonomous Vehicles Data/Cityscapes Data'+\
               '/leftImg8bit_demoVideo/leftImg8bit/demoVideo/stuttgart_00'
    BASE_PATHS = [Path.cwd() / "torch_models"]

    MODEL_CFG, MODEL_PTH = get_config_argparse(BASE_PATHS)

    DATALOADER, MODEL = get_loader_and_model(MODEL_CFG, MODEL_PTH, DATA_DIR)
