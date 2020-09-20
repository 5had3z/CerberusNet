#!/usr/bin/env python3

__author__ = "Bryce Ferenczi"
__email__ = "bryce.ferenczi@monashmotorsport.com"

import platform
import os
import re
import random
import json
import multiprocessing
from pathlib import Path
from typing import Dict, List
import cv2

import torch
import torchvision
from PIL import Image

import numpy as np

from nnet_training.utilities.visualisation import get_color_pallete, flow_to_image

__all__ = ['Kitti2015Dataset', 'get_kitti_dataset']

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

        if "disparity" in objectives:
            self.l_disp = []
            if "stereo" in objectives:
                self.r_disp = []

        # Get all file names
        for dir_name, _, file_list in os.walk(os.path.join(directory, 'image_2')):
            for filename in sorted(file_list):
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

                    if hasattr(self, 'l_disp'):
                        left_disp_path = os.path.join(directory, 'disp_noc_0', filename)
                        if not os.path.isfile(left_disp_path):
                            read_check = False
                            print("Error finding corresponding segmentation image to ", l_imgpath)

                    if hasattr(self, 'r_disp'):
                        right_disp_path = os.path.join(directory, 'disp_noc_1', filename)
                        if not os.path.isfile(right_disp_path):
                            read_check = False
                            print("Error finding corresponding segmentation image to ", l_imgpath)

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
                        if hasattr(self, 'l_disp'):
                            self.l_disp.append(left_disp_path)
                        if hasattr(self, 'r_disp'):
                            self.r_disp.append(right_disp_path)
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
            if hasattr(self, 'l_disp'):
                self.l_disp.append(self.l_disp[i] for i in id_vector)
            if hasattr(self, 'r_disp'):
                self.r_disp.append(self.r_disp[i] for i in id_vector)
            if hasattr(self, 'l_seq'):
                self.l_seq = [self.l_seq[i] for i in id_vector]
            if hasattr(self, 'r_seq'):
                self.r_seq = [self.r_seq[i] for i in id_vector]
            if hasattr(self, 'flow'):
                self.flow = [self.flow[i] for i in id_vector]

        self.disparity_out = disparity_out
        self.output_size = tuple(output_size)

        self.std_kitti_dims = (1274, 375)
        self.mirror_x = 1.0

        if 'crop_fraction' in kwargs:
            self.crop_fraction = kwargs['crop_fraction']
        if 'rand_rotation' in kwargs:
            self.rand_rot = kwargs['rand_rotation']
        if 'rand_brightness' in kwargs:
            self.brightness = kwargs['rand_brightness']

        self._key = np.array([255, 255, 255, 255, 255, 255,
                              255, 255, 0, 1, 255, 255,
                              2, 3, 4, 255, 255, 255,
                              5, 255, 6, 7, 8, 9,
                              10, 11, 12, 13, 14, 15,
                              255, 255, 16, 17, 18])
        self._mapping = np.array(range(-1, len(self._key) - 1)).astype('int32')

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
        if hasattr(self, 'l_disp'):
            epoch_data["l_disp"] = Image.open(self.l_disp[idx])
        if hasattr(self, 'r_disp'):
            epoch_data["r_disp"] = Image.open(self.r_disp[idx])
        if hasattr(self, 'l_seq'):
            epoch_data["l_seq"] = Image.open(self.l_seq[idx]).convert('RGB')
        if hasattr(self, 'r_seq'):
            epoch_data["r_seq"] = Image.open(self.r_seq[idx]).convert('RGB')
        if hasattr(self, 'flow'):
            raw_data = cv2.imread(self.flow[idx], cv2.IMREAD_UNCHANGED)
            epoch_data["flow_x"] = Image.fromarray(np.array(raw_data[:, :, 2]))
            epoch_data["flow_y"] = Image.fromarray(np.array(raw_data[:, :, 1]))
            epoch_data["flow_b"] = Image.fromarray(np.array(raw_data[:, :, 0]))

        self._sync_transform(epoch_data)

        return epoch_data

    def _sync_transform(self, epoch_data):
        # random mirror
        if random.random() < 0.5:
            self.mirror_x = -1.0
            for key, data in epoch_data.items():
                epoch_data[key] = data.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            self.mirror_x = 1.0

        if hasattr(self, 'rand_rot'):
            angle = random.uniform(0, self.rand_rot)
            for key, data in epoch_data.items():
                if key in ["l_img", "r_img", "l_seq", "r_seq"]:
                    epoch_data[key] = torchvision.transforms.functional.rotate(
                        data, angle, resample=Image.BILINEAR)
                elif key in ["flow_b", "l_disp", "r_disp"]:
                    epoch_data[key] = torchvision.transforms.functional.rotate(
                        data, angle, resample=Image.NEAREST, fill=0)
                else:
                    epoch_data[key] = torchvision.transforms.functional.rotate(
                        data, angle, resample=Image.NEAREST, fill=-1)

        if hasattr(self, 'crop_fraction'):
            # random crop
            crop_h = int(epoch_data["l_img"].size[0]/self.crop_fraction)
            crop_w = int(epoch_data["l_img"].size[1]/self.crop_fraction)
            crop_x = random.randint(0, epoch_data["l_img"].size[1] - crop_w)
            crop_y = random.randint(0, epoch_data["l_img"].size[0] - crop_h)

            for key, data in epoch_data.items():
                epoch_data[key] = data.crop((crop_y, crop_x, crop_y+crop_h, crop_x+crop_w))

        if hasattr(self, 'brightness'):
            brightness_scale = random.uniform(1-self.brightness/100, 1+self.brightness/100)
            for key, data in epoch_data.items():
                if key in ["l_img", "r_img", "l_seq", "r_seq"]:
                    epoch_data[key] = torchvision.transforms.functional.adjust_brightness(
                        data, brightness_scale)

        for key, data in epoch_data.items():
            if key in ["l_img", "r_img", "l_seq", "r_seq"]:
                data = data.resize(self.output_size, Image.BILINEAR)
                epoch_data[key] = torchvision.transforms.functional.to_tensor(data)
            elif key == "seg":
                data = data.resize(self.output_size, Image.NEAREST)
                epoch_data[key] = self._seg_transform(data)
            elif key in ["l_disp", "r_disp"]:
                data = data.resize(self.output_size, Image.NEAREST)
                epoch_data[key] = self._depth_transform(data)

        if all(key in epoch_data.keys() for key in ["flow_x", "flow_y", "flow_b"]):
            self._flow_transform(epoch_data)
        elif any(key in epoch_data.keys() for key in ["flow_x", "flow_y", "flow_b"]):
            raise UserWarning("Partially missing flow data, need x, y and bit mask")

    @staticmethod
    def _depth_transform(disparity):
        return torch.FloatTensor(np.array(disparity).astype('float32') / 256.0)

    def _flow_transform(self, epoch_data: Dict[str, Image.Image]):
        # Resize to appropriate size first
        bitmask = epoch_data['flow_b'].resize(self.output_size, Image.NEAREST)
        flow_x = epoch_data['flow_x'].resize(self.output_size, Image.NEAREST)
        flow_y = epoch_data['flow_y'].resize(self.output_size, Image.NEAREST)
        scale_x = self.output_size[0] / self.std_kitti_dims[0]
        scale_y = self.output_size[1] / self.std_kitti_dims[1]

        # Apply transform indicated by the devkit including ignore mask
        flow_out_x = self.mirror_x * scale_x * (np.array(flow_x).astype('float32') - 2**15) / 64.0
        flow_out_y = scale_y * (np.array(flow_y).astype('float32') - 2**15) / 64.0

        for key in ["flow_x", "flow_y", "flow_b"]:
            del epoch_data[key]

        epoch_data["flow"] = torch.stack([
            torch.FloatTensor(flow_out_x),
            torch.FloatTensor(flow_out_y)
        ])
        epoch_data["flow_mask"] = torchvision.transforms.functional.to_tensor(bitmask)

    def _class_to_index(self, seg):
        values = np.unique(seg)
        for value in values:
            assert value in self._mapping
        index = np.digitize(seg.ravel(), self._mapping, right=True)
        return self._key[index].reshape(seg.shape)

    def _seg_transform(self, segmentaiton):
        target = self._class_to_index(np.array(segmentaiton).astype('int32'))
        return torch.LongTensor(target.astype('int32'))

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
                                        output_size=dataset_config.augmentations.output_size,
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
    import matplotlib.pyplot as plt

    with open("configs/Kitti_test.json") as f:
        cfg = EasyDict(json.load(f))

    dataloaders = get_kitti_dataset(cfg['dataset'])

    data = next(iter(dataloaders['Validation']))
    left     = data['l_img']
    seq_left = data['l_seq']
    seg_gt   = data['seg']
    flow_gt  = data['flow']

    for i in range(dataloaders['Validation'].batch_size):
        plt.subplot(2, 2, 1)
        plt.imshow(np.moveaxis(left[i, 0:3, :, :].numpy(), 0, 2))
        plt.xlabel("Base Image")

        plt.subplot(2, 2, 2)
        plt.imshow(np.moveaxis(seq_left[i, :, :].numpy(), 0, 2))
        plt.xlabel("Sequential Image")

        plt.subplot(2, 2, 3)
        plt.imshow(get_color_pallete(seg_gt.numpy()[i, :, :]))
        plt.xlabel("Ground Truth Segmentation")

        vis_flow = flow_to_image(flow_gt[i].numpy().transpose([1, 2, 0]))

        plt.subplot(2, 2, 4)
        plt.imshow(vis_flow)
        plt.xlabel("Ground Truth Flow")

        plt.show()

if __name__ == "__main__":
    test_kitti_loading()
