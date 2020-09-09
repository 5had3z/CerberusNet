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
from typing import Dict, List

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
    def __init__(self, directory: Path, objectives: List[str], id_vector=None,
                 output_size=(1274, 375), disparity_out=False, **kwargs):
        '''
        Initializer for KITTI 2015 dataset\n
        @input directory Path that contains the base directory of the KITTI Dataset\n
        @input objectives List that determines the datasets we need to use\n
        @param disparity_out determines whether the disparity is
        given or a direct depth estimation.\n
        @param crop_fraction determines if the image is randomly cropped to by a fraction
            e.g. 2 results in width/2 by height/2 random crop of original image\n
        @param rand_rotation randomly rotates an image a maximum number of degrees.\n
        @param rand_brightness, the maximum random increase or decrease as a percentage.
        '''
        self.l_img = []

        if "stereo" in objectives:
            self.r_img = []

        if "flow" in objectives:
            self.l_seq = []
            self.flow = []
            if "stereo" in objectives:
                self.r_seq = []

        if "seg" in objectives:
            self.seg = []

        # Get all file names
        for dir_name, _, file_list in os.walk(os.path.join(directory, 'image_2')):
            for filename in file_list:
                frame_n = int(re.split("_", filename)[1][:2])
                if filename.endswith(IMG_EXT) and frame_n == 10:
                    read_check = True
                    l_imgpath = os.path.join(dir_name, filename)

                    if hasattr(self, 'r_img'):
                        r_imgpath = os.path.join(directory, 'image_3', filename)
                        if not os.path.isfile(r_imgpath):
                            read_check = False
                            print("Error finding corresponding right image to ", l_imgpath)

                    if hasattr(self, 'seg'):
                        seg_path = os.path.join(directory, 'semantic', filename)
                        if not os.path.isfile(seg_path):
                            read_check = False
                            print("Error finding corresponding segmentation image to ", l_imgpath)

                    if hasattr(self, 'disp'):
                        raise NotImplementedError

                    if hasattr(self, 'l_seq'):
                        seq_name = filename.replace('10.png', '11.png')
                        left_seq_path = os.path.join(directory, 'image_2', seq_name)
                        if not os.path.isfile(left_seq_path):
                            read_check = False
                            print("Error finding corresponding left sequence image to ", l_imgpath)

                    if hasattr(self, 'r_seq'):
                        seq_name = filename.replace('10.png', '11.png')
                        right_seq_path = os.path.join(directory, 'image_3', seq_name)
                        if not os.path.isfile(right_seq_path):
                            read_check = False
                            print("Error finding corresponding right sequence image to ", l_imgpath)

                    if hasattr(self, 'flow'):
                        flow_path = os.path.join(directory, 'flow_noc', filename)
                        if not os.path.isfile(flow_path):
                            read_check = False
                            print("Error finding corresponding segmentation image to ", l_imgpath)

                    if read_check:
                        self.l_img.append(l_imgpath)
                        if hasattr(self, 'r_img'):
                            self.r_img.append(r_imgpath)
                        if hasattr(self, 'seg'):
                            self.seg.append(seg_path)
                        if hasattr(self, 'disp'):
                            raise NotImplementedError
                        if hasattr(self, 'l_seq'):
                            self.l_seq.append(left_seq_path)
                        if hasattr(self, 'r_seq'):
                            self.r_seq.append(right_seq_path)
                        if hasattr(self, 'flow'):
                            self.flow.append(flow_path)

        # Create dataset from specified ids if id_vector given else use all
        if 'id_vector' is not None:
            self.l_img = [self.l_img[i] for i in id_vector]
            if hasattr(self, 'r_img'):
                self.r_img = [self.r_img[i] for i in id_vector]
            if hasattr(self, 'seg'):
                self.seg = [self.seg[i] for i in id_vector]
            if hasattr(self, 'disp'):
                raise NotImplementedError
            if hasattr(self, 'l_seq'):
                self.l_seq = [self.l_seq[i] for i in id_vector]
            if hasattr(self, 'r_seq'):
                self.r_seq = [self.r_seq[i] for i in id_vector]
            if hasattr(self, 'flow'):
                self.flow = [self.flow[i] for i in id_vector]

        self.disparity_out = disparity_out
        self.output_size = output_size

        if 'crop_fraction' in kwargs:
            self.crop_fraction = kwargs['crop_fraction']
        if 'rand_rotation' in kwargs:
            self.rand_rot = kwargs['rand_rotation']
        if 'rand_brightness' in kwargs:
            self.brightness = kwargs['rand_brightness']

    def __len__(self):
        return len(self.l_img)

    def __getitem__(self, idx):
        '''
        Returns relevant training data as a dict
        @output l_img, r_img, seg, disparity, l_seq, r_seq, flow
        '''
        epoch_data = {}
        # Read image and labels
        epoch_data["l_img"] = Image.open(self.l_img[idx]).convert('RGB')

        if hasattr(self, 'r_img'):
            epoch_data["r_img"] = Image.open(self.r_img[idx]).convert('RGB')
        if hasattr(self, 'seg'):
            epoch_data["seg"] = Image.open(self.seg[idx])
        if hasattr(self, 'disp'):
            raise NotImplementedError
        if hasattr(self, 'l_seq'):
            epoch_data["l_seq"] = Image.open(self.l_seq[idx]).convert('RGB')
        if hasattr(self, 'r_seq'):
            epoch_data["r_seq"] = Image.open(self.r_seq[idx]).convert('RGB')
        if hasattr(self, 'flow'):
            epoch_data["flow"] = Image.open(self.flow[idx])

        self._sync_transform(epoch_data)

        #   Transform images to tensors
        for key in epoch_data.keys():
            if key in ["l_img", "r_img", "l_seq", "r_seq"]:
                epoch_data[key] = torchvision.transforms.functional.to_tensor(epoch_data[key])

        return epoch_data

    def _sync_transform(self, epoch_data):
        # random mirror
        if random.random() < 0.5:
            for key, data in epoch_data.items():
                epoch_data[key] = data.transpose(Image.FLIP_LEFT_RIGHT)

        if hasattr(self, 'crop_fraction'):
            # random crop
            crop_h = int(epoch_data["l_img"].size[0]/self.crop_fraction)
            crop_w = int(epoch_data["l_img"].size[1]/self.crop_fraction)
            crop_x = random.randint(0, epoch_data["l_img"].size[1] - crop_w)
            crop_y = random.randint(0, epoch_data["l_img"].size[0] - crop_h)

            for key, data in epoch_data.items():
                epoch_data[key] = data.crop((crop_y, crop_x, crop_y+crop_h, crop_x+crop_w))

        if hasattr(self, 'rand_rot'):
            angle = random.uniform(0, self.rand_rot)
            for key, data in epoch_data.items():
                if key in ["l_img", "r_img", "l_seq", "r_seq"]:
                    epoch_data[key] = torchvision.transforms.functional.rotate(
                        data, angle, resample=Image.BILINEAR)
                else:
                    epoch_data[key] = torchvision.transforms.functional.rotate(
                        data, angle, resample=Image.NEAREST, fill=-1)

        if hasattr(self, 'brightness'):
            brightness_scale = random.uniform(1-self.brightness/100, 1+self.brightness/100)
            for key, data in epoch_data.items():
                if key in ["l_img", "r_img", "l_seq", "r_seq"]:
                    epoch_data[key] = torchvision.transforms.functional.adjust_brightness(
                        data, brightness_scale)

        for key, data in epoch_data.items():
            if key in ["l_img", "r_img", "l_seq", "r_seq"]:                
                data = data.resize(self.output_size, Image.BILINEAR)
            elif key == "seg":
                data = data.resize(self.output_size, Image.NEAREST)
            elif key == "disparity":
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
                                        dataset_config.objectives,
                                        **dataset_config.augmentations,
                                        id_vector=train_ids),
        'Validation' : Kitti2015Dataset(dataset_config.rootdir,
                                        dataset_config.objectives,
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
