#!/usr/bin/env python3

__author__ = "Bryce Ferenczi"
__email__ = "bryce.ferenczi@monashmotorsport.com"

import platform
import os
import re
import random
import json
import multiprocessing
from shutil import copy
from pathlib import Path
from typing import Dict

import torch
import torchvision
from PIL import Image

import numpy as np

__all__ = ['KITTIFlowDataset', 'get_kitti_dataset']

IMG_EXT = '.png'

class Kitti2015Dataset(torch.utils.data.Dataset):
    """
    Cityscapes Dataset\n
    On initialization, give a dictionary to the respective paths with the corresponding keys:
        [(left_images or images), right_images, seg, disparity, left_seq, right_seq, cam, pose]
    Get Item will return the corresponding dictionary with keys:
        [l_img, r_img, l_seq", r_seq, cam, pose, disparity]
    """
    def __init__(self, directories: Path, output_size=(1024, 512), disparity_out=False, **kwargs):
        '''
        Initializer for KITTI 2015 dataset\n
        @input dictionary that contain paths of datasets as
        [l_img, r_img, seg, disparity, l_seq, r_seq, cam, pose]\n
        @param disparity_out determines whether the disparity is
        given or a direct depth estimation.\n
        @param crop_fraction determines if the image is randomly cropped to by a fraction
            e.g. 2 results in width/2 by height/2 random crop of original image\n
        @param rand_rotation randomly rotates an image a maximum number of degrees.\n
        @param rand_brightness, the maximum random increase or decrease as a percentage.
        '''
        raise NotImplementedError

def id_vec_generator(train_ratio, directory):
    """
    Generates the training and validation split of a monlitic dataset.\n
    Ensures that you are always using the same testing and training data for a given set and ratio.
    """
    num_images = 0
    for file in os.listdir(directory):
        num_images += file.endswith(IMG_EXT)

    print("Number of Images:\t", num_images)
    image_ids = list(range(num_images))
    random.shuffle(image_ids)

    n_train = int(num_images*train_ratio)

    train_ids = image_ids[0:n_train]
    val_ids = image_ids[n_train+1:-1]

    return train_ids, val_ids

def get_kitti_dataset(dataset_config) -> Dict[str, torch.utils.data.DataLoader]:
    """
    Input configuration json for kitti dataset
    Output dataloaders for training and validation
    """
    if platform.system() == 'Windows':
        n_workers = 0
    else:
        n_workers = min(multiprocessing.cpu_count(), dataset_config.batch_size)

    # Using the segmentaiton gt dir to count the number of images
    seg_dir = os.path.join(dataset_config.rootdir, "semantic")
    train_ids, val_ids = id_vec_generator(dataset_config.train_ratio, seg_dir)

    datasets = {
        'Training'   : Kitti2015Dataset(dataset_config.rootdir,
                                        **dataset_config.augmentations,
                                        id_vector=train_ids),
        'Validation' : Kitti2015Dataset(dataset_config.rootdir,
                                        **dataset_config.augmentations,
                                        id_vector=val_ids)
    }

    dataloaders = {
        'Training'   : torch.utils.data.DataLoader(
            datasets["Training"],
            batch_size=dataset_config.batch_size,
            shuffle=dataset_config.shuffle,
            num_workers=n_workers,
            drop_last=dataset_config.drop_last
        ),
        'Validation' : torch.utils.data.DataLoader(
            datasets["Validation"],
            batch_size=dataset_config.batch_size,
            shuffle=dataset_config.shuffle,
            num_workers=n_workers,
            drop_last=dataset_config.drop_last
        )
    }

    return dataloaders

def test_kitti_loading():
    """
    Get kitti dataset for testing
    """
    from easydict import EasyDict

    with open("configs/Kitti_test.json") as f:
        cfg = EasyDict(json.load(f))

    return get_kitti_dataset(cfg)

if __name__ == "__main__":
    test_kitti_loading()
