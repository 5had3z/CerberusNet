#!/usr/bin/env python3

"""
Dataloader and misc utilities related to the cityscapes dataset
"""

__author__ = "Bryce Ferenczi"
__email__ = "bryce.ferenczi@monash.edu"

import platform
import os
import re
import random
import json
import multiprocessing
from shutil import copy
from pathlib import Path
from typing import Dict, List, Tuple, Union

import torch
import torchvision
from torch.utils.data import RandomSampler
from PIL import Image

import numpy as np

from nnet_training.utilities.custom_batch_sampler import BatchSamplerRandScale

__all__ = ['CityScapesDataset', 'get_cityscapse_dataset']

IMG_EXT = '.png'

class CityScapesDataset(torch.utils.data.Dataset):
    """
    Cityscapes Dataset\n
    On initialization, give a dictionary to the respective paths with the corresponding keys:
        [(left_images or images), right_images, seg, disparity, left_seq, right_seq, cam, pose]
    Get Item will return the corresponding dictionary with keys:
        [l_img, r_img, l_seq", r_seq, cam, pose, disparity]
    """
    def __init__(self, directories: Path, output_size=(1024, 512), disparity_out=False, **kwargs):
        '''
        Initializer for Cityscapes dataset\n
        @input dictionary that contain paths of datasets as
        [l_img, r_img, seg, disparity, l_seq, r_seq, cam, pose]\n
        @param disparity_out determines whether the disparity is
        given or a direct depth estimation.\n
        @param crop_fraction determines if the image is randomly cropped to by a fraction
            e.g. 2 results in width/2 by height/2 random crop of original image\n
        @param rand_rotation randomly rotates an image a maximum number of degrees.\n
        @param rand_brightness, the maximum random increase or decrease as a percentage.
        '''
        super().__init__()
        l_img_key = None

        for key in directories.keys():
            if key in ['left_images', 'images']:
                l_img_key = key
                self.l_img = []
            elif key == 'seg':
                self.seg = []
            elif key == 'right_images':
                self.r_img = []
            elif key == 'disparity':
                self.l_disp = []
            elif key == 'left_seq':
                self.l_seq = []
            elif key == 'right_seq':
                self.r_seq = []
            elif key == 'cam':
                self.cam = []
            elif key == 'pose':
                self.pose = []
            elif key == 'bbox':
                self.bbox = []

        if not any(hasattr(self, key_) for key_ in ['seg', 'l_disp', 'l_seq', 'bbox']):
            Warning("Neither Segmentation, Disparity, Img Sequence or Boundary Box Keys are used")

        if l_img_key is None:
            Warning("Empty dataset given to base cityscapes dataset")
        else:
            self._initialize_dataset(directories, l_img_key, **kwargs)

        self.disparity_out = disparity_out
        self.base_size = output_size # (width, height)
        self.output_shape = output_size
        self.scale_factor = 1

        if 'crop_fraction' in kwargs:
            self.crop_fraction = kwargs['crop_fraction']
        if 'rand_rotation' in kwargs:
            self.rand_rot = kwargs['rand_rotation']
        if 'rand_brightness' in kwargs:
            self.brightness = kwargs['rand_brightness']
        if 'rand_scale' in kwargs:
            assert len(kwargs['rand_scale']) == 2
            self.scale_range = kwargs['rand_scale']
        if 'img_normalize' in kwargs:
            # Typical normalisation parameters:
            # "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]
            self.img_normalize = torchvision.transforms.Normalize(
                kwargs['img_normalize']['mean'], kwargs['img_normalize']['std'])

        self.rand_flip = True

        # valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22,
        #                       23, 24, 25, 26, 27, 28, 31, 32, 33]
        self._key = np.array([255, 255, 255, 255, 255, 255,
                              255, 255, 0, 1, 255, 255,
                              2, 3, 4, 255, 255, 255,
                              5, 255, 6, 7, 8, 9,
                              10, 11, 12, 13, 14, 15,
                              255, 255, 16, 17, 18])
        self._mapping = np.array(range(-1, len(self._key) - 1)).astype('int32')

    def _initialize_dataset(self, directories, l_img_key, **kwargs):
        """
        Gets all the filenames
        """
        for dirpath, _, filenames in os.walk(directories[l_img_key]):
            for filename in filenames:
                if not filename.endswith(IMG_EXT):
                    continue

                read_check = True
                l_imgpath = os.path.join(dirpath, filename)
                foldername = os.path.basename(os.path.dirname(l_imgpath))
                frame_n = int(re.split("_", filename)[2])

                if hasattr(self, 'r_img'):
                    r_imgpath = os.path.join(
                        directories['right_images'], foldername,
                        filename.replace('leftImg8bit', 'rightImg8bit'))
                    if not os.path.isfile(r_imgpath):
                        read_check = False
                        print("Error finding corresponding right image to ", l_imgpath)

                if hasattr(self, 'seg'):
                    seg_path = os.path.join(
                        directories['seg'], foldername,
                        filename.replace('leftImg8bit', 'gtFine_labelIds'))
                    if not os.path.isfile(seg_path):
                        read_check = False
                        print("Error finding corresponding segmentation image to ", l_imgpath)

                if hasattr(self, 'l_disp'):
                    disp_path = os.path.join(
                        directories['disparity'], foldername,
                        filename.replace('leftImg8bit', 'disparity'))
                    if not os.path.isfile(disp_path):
                        read_check = False
                        print("Error finding corresponding disparity image to ", l_imgpath)

                if hasattr(self, 'l_seq'):
                    left_seq_path = os.path.join(
                        directories['left_seq'], foldername,
                        filename.replace(
                            str(frame_n).zfill(6), str(frame_n+1).zfill(6)))
                    if not os.path.isfile(left_seq_path):
                        read_check = False
                        print("Error finding corresponding left sequence image to ", l_imgpath)

                if hasattr(self, 'r_seq'):
                    right_seq_path = os.path.join(
                        directories['right_seq'], foldername,
                        filename.replace(
                            str(frame_n).zfill(6)+"_leftImg8bit",
                            str(frame_n+1).zfill(6)+"_rightImg8bit"))
                    if not os.path.isfile(right_seq_path):
                        read_check = False
                        print("Error finding corresponding right sequence image to ", l_imgpath)

                if hasattr(self, 'cam'):
                    cam_path = os.path.join(
                        directories['cam'], foldername,
                        filename.replace('leftImg8bit.png', 'camera.json'))
                    if not os.path.isfile(cam_path):
                        read_check = False
                        print("Error finding corresponding camera parameters to ", l_imgpath)

                if hasattr(self, 'pose'):
                    pose_path = os.path.join(
                        directories['pose'], foldername,
                        filename.replace('leftImg8bit.png', 'vehicle.json'))
                    if not os.path.isfile(pose_path):
                        read_check = False
                        print("Error finding corresponding GPS/Pose information to ", l_imgpath)

                if hasattr(self, 'bbox'):
                    bbox_path = os.path.join(
                        directories['bbox'], foldername,
                        filename.replace('leftImg8bit.png', 'gtFine_bbox.json'))
                    if not os.path.isfile(bbox_path):
                        read_check = False
                        print("Error finding corresponding bbox information to ", l_imgpath)

                if read_check:
                    self.l_img.append(l_imgpath)
                    if hasattr(self, 'r_img'):
                        self.r_img.append(r_imgpath)
                    if hasattr(self, 'seg'):
                        self.seg.append(seg_path)
                    if hasattr(self, 'l_disp'):
                        self.l_disp.append(disp_path)
                    if hasattr(self, 'l_seq'):
                        self.l_seq.append(left_seq_path)
                    if hasattr(self, 'r_seq'):
                        self.r_seq.append(right_seq_path)
                    if hasattr(self, 'cam'):
                        self.cam.append(cam_path)
                    if hasattr(self, 'pose'):
                        self.pose.append(pose_path)
                    if hasattr(self, 'bbox'):
                        self.pose.append(bbox_path)

        # Create dataset from specified ids if id_vector given else use all
        if 'id_vector' in kwargs:
            self.l_img = [self.l_img[i] for i in kwargs['id_vector']]
            if hasattr(self, 'r_img'):
                self.r_img = [self.r_img[i] for i in kwargs['id_vector']]
            if hasattr(self, 'seg'):
                self.seg = [self.seg[i] for i in kwargs['id_vector']]
            if hasattr(self, 'l_disp'):
                self.l_disp = [self.l_disp[i] for i in kwargs['id_vector']]
            if hasattr(self, 'l_seq'):
                self.l_seq = [self.l_seq[i] for i in kwargs['id_vector']]
            if hasattr(self, 'r_seq'):
                self.r_seq = [self.r_seq[i] for i in kwargs['id_vector']]
            if hasattr(self, 'cam'):
                self.cam = [self.cam[i] for i in kwargs['id_vector']]
            if hasattr(self, 'pose'):
                self.pose = [self.pose[i] for i in kwargs['id_vector']]
            if hasattr(self, 'bbox'):
                self.bbox = [self.bbox[i] for i in kwargs['id_vector']]

    def __len__(self):
        return len(self.l_img)

    def __getitem__(self, idx):
        '''
        Returns relevant training data as a dict
        @output l_img, r_img, seg, l_disp, l_seq, r_seq, cam, pose
        '''
        if isinstance(idx, tuple):
            idx, self.scale_factor = idx

        # Read image and labels
        epoch_data = {"l_img" : Image.open(self.l_img[idx]).convert('RGB')}

        if hasattr(self, 'r_img'):
            epoch_data["r_img"] = Image.open(self.r_img[idx]).convert('RGB')
        if hasattr(self, 'seg'):
            epoch_data["seg"] = Image.open(self.seg[idx])
        if hasattr(self, 'l_disp'):
            epoch_data["l_disp"] = Image.open(self.l_disp[idx])
        if hasattr(self, 'l_seq'):
            epoch_data["l_seq"] = Image.open(self.l_seq[idx]).convert('RGB')
        if hasattr(self, 'r_seq'):
            epoch_data["r_seq"] = Image.open(self.r_seq[idx]).convert('RGB')
        if hasattr(self, 'bbox'):
            epoch_data['labels'], epoch_data['bboxes'] = self.bbox_json_parse(self.bbox[idx])

        self._sync_transform(epoch_data)

        if hasattr(self, 'cam'):
            epoch_data["cam"] = self.intrinsics_json(self.cam[idx])

        # if hasattr(self, 'pose'):
        #     epoch_data["pose"] = self.pose_json(self.pose[idx])

        return epoch_data

    @staticmethod
    def _mirror_bbox(bbox_list: List[List[int]], img_dims: Tuple[int, int]) -> List[List[int]]:
        """
        Mirrors a boundary box [x1,y1,x2,y2] in an image of img_dims (width, height)
        """
        ret_val = []
        mirror_x = lambda x : img_dims[0] - (x - int(img_dims[0] / 2))
        for bbox in bbox_list:
            ret_val.append([mirror_x(bbox[0]), bbox[1], mirror_x(bbox[2]), bbox[3]])
        return ret_val

    @staticmethod
    def _crop_bbox(bbox_list: List[List[int]], img_dims: Tuple[int, int],
                   crop_dims: Tuple[int, int, int, int]) -> List[List[int]]:
        """
        Transforms a boundary box [x1,y1,x2,y2] according to an image crop (x, y, w, h)
        """
        ret_val = []
        for bbox in bbox_list:
            # Shift bbox top left, clip at 0
            b_x1 = max(bbox[0] - crop_dims[0], 0)
            b_y1 = max(bbox[1] - crop_dims[1], 0)
            # Shift bbox top left, clip at bottom right boundary
            # if image crop encroaches on transformed bbox.
            b_x2 = min(bbox[2] - crop_dims[0], img_dims[0] - crop_dims[2])
            b_x2 = min(bbox[3] - crop_dims[1], img_dims[1] - crop_dims[3])
            ret_val.append([b_x1, b_y1, b_x2, b_x2])
        return ret_val

    def _sync_transform(self, epoch_data: Dict[str, Union[Image, List]]) -> None:
        """
        Augments and formats all the batch data in a synchronised manner
        """
        scale_func = lambda x: int(self.scale_factor * x / 32.0) * 32
        self.output_shape = [scale_func(x) for x in self.base_size]

        # random mirror
        if random.random() < 0.5 and self.rand_flip:
            for key, data in epoch_data.items():
                if key not in ['labels', 'bboxes']:
                    epoch_data[key] = data.transpose(Image.FLIP_LEFT_RIGHT)
                elif key == 'bboxes':
                    epoch_data[key] = self._mirror_bbox(data, self.output_shape)

        if hasattr(self, 'brightness'):
            brightness_scale = random.uniform(1-self.brightness/100, 1+self.brightness/100)
            for key, data in epoch_data.items():
                if key in ["l_img", "r_img", "l_seq", "r_seq"]:
                    epoch_data[key] = torchvision.transforms.functional.adjust_brightness(
                        data, brightness_scale)

        if hasattr(self, 'rand_rot'):
            angle = random.uniform(0, self.rand_rot)
            for key, data in epoch_data.items():
                if key in ["l_img", "r_img", "l_seq", "r_seq"]:
                    epoch_data[key] = torchvision.transforms.functional.rotate(
                        data, angle, resample=Image.BILINEAR)
                else:
                    epoch_data[key] = torchvision.transforms.functional.rotate(
                        data, angle, resample=Image.NEAREST, fill=-1)

        for key, data in epoch_data.items():
            if key in ["l_img", "r_img", "l_seq", "r_seq"]:
                data = data.resize(self.output_shape, Image.BILINEAR)
                epoch_data[key] = self._img_transform(data)
            elif key == "seg":
                data = data.resize(self.output_shape, Image.NEAREST)
                epoch_data[key] = self._seg_transform(data)
            elif key == "l_disp":
                data = data.resize(self.output_shape, Image.NEAREST)
                epoch_data[key] = self._depth_transform(data)

        if hasattr(self, 'crop_fraction'):
            # random crop
            crop_h = int(epoch_data["l_img"].shape[1] / self.crop_fraction / 32.0) * 32
            crop_w = int(epoch_data["l_img"].shape[2] / self.crop_fraction / 32.0) * 32
            crop_x = random.randint(0, epoch_data["l_img"].shape[2] - crop_w)
            crop_y = random.randint(0, epoch_data["l_img"].shape[1] - crop_h)

            for key, data in epoch_data.items():
                if key in ["seg", "l_disp", "r_disp"]:
                    epoch_data[key] = data[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
                elif key == 'bboxes':
                    epoch_data[key] = self._crop_bbox(
                        data, self.output_shape, (crop_x, crop_y, crop_w, crop_h))
                else:
                    epoch_data[key] = data[:, crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]

    def _class_to_index(self, seg):
        values = np.unique(seg)
        for value in values:
            assert value in self._mapping
        index = np.digitize(seg.ravel(), self._mapping, right=True)
        return self._key[index].reshape(seg.shape)

    def _img_transform(self, img):
        img = torchvision.transforms.functional.to_tensor(img)
        if hasattr(self, 'img_normalize'):
            img = self.img_normalize(img)
        return img

    def _seg_transform(self, seg):
        target = self._class_to_index(np.array(seg).astype('int32'))
        return torch.LongTensor(target.astype('int32'))

    def _depth_transform(self, disparity):
        disparity = np.array(disparity).astype('float32')
        disparity[disparity > 0] = self.scale_factor * (disparity[disparity > 0] - 1) / 256.0
        if not self.disparity_out:
            disparity[disparity > 0] = (0.209313 * 2262.52) / disparity[disparity > 0]

            # Ignore sides and bottom of frame as these are patchy/glitchy
            side_clip = int(disparity.shape[1] / 20)
            bottom_clip = int(disparity.shape[0] / 10)
            disparity[-bottom_clip:-1, :] = 0.  #bottom
            disparity[:, :side_clip] = 0.       #lhs
            disparity[:, -side_clip:-1] = 0.    #rhs

        return torch.FloatTensor(disparity)

    @staticmethod
    def intrinsics_json(json_path):
        """
        Parses camera instrics json and returns intrinsics and baseline transform matricies.
        """
        with open(json_path) as json_file:
            #   Camera Intrinsic Matrix
            k_mat = np.eye(4, dtype=np.float32) #Idk why size 4? (To match translation?)
            json_data = json.load(json_file)
            k_mat[0, 0] = json_data["intrinsic"]["fx"]
            k_mat[1, 1] = json_data["intrinsic"]["fy"]
            k_mat[0, 2] = json_data["intrinsic"]["u0"]
            k_mat[1, 2] = json_data["intrinsic"]["v0"]

            #   Transformation Mat between cameras
            stereo_t = np.eye(4, dtype=np.float32)
            stereo_t[0, 3] = json_data["extrinsic"]["baseline"]

        return {"K":k_mat, "inv_K":np.linalg.pinv(k_mat), "baseline_T":stereo_t}

    @staticmethod
    def pose_json(json_path: str):
        """
        Parse json with gps pose information and return dictionary of information.
        """
        raise NotImplementedError

    @staticmethod
    def bbox_json_parse(json_path: str):
        """
        Parse json with bbox information and return list of information.
        """
        labels = []
        bboxes = []
        with open(json_path) as json_file:
            json_data = json.load(json_file)
            for bbox in json_data:
                labels.append(bbox['train_id'])
                bboxes.append(bbox['bbox'])

        return labels, bboxes

def get_cityscapse_dataset(dataset_config) -> Dict[str, torch.utils.data.DataLoader]:
    """
    Returns a cityscapes dataset given a config
    """
    if platform.system() == 'Windows':
        n_workers = 0
    else:
        n_workers = min(multiprocessing.cpu_count(), dataset_config.batch_size)

    training_dirs = {}
    for subset in dataset_config.train_subdirs:
        training_dirs[str(subset)] = dataset_config.rootdir +\
                                        dataset_config.train_subdirs[str(subset)]

    validation_dirs = {}
    for subset in dataset_config.val_subdirs:
        validation_dirs[str(subset)] = dataset_config.rootdir +\
                                        dataset_config.val_subdirs[str(subset)]

    aux_aug = {}
    if 'img_normalize' in dataset_config.augmentations:
        aux_aug['img_normalize'] = dataset_config.augmentations.img_normalize
    if 'disparity_out' in dataset_config.augmentations:
        aux_aug['disparity_out'] = dataset_config.augmentations.disparity_out

    datasets = {
        'Training'   : CityScapesDataset(training_dirs, **dataset_config.augmentations),
        'Validation' : CityScapesDataset(
            validation_dirs, output_size=dataset_config.augmentations.output_size, **aux_aug)
    }

    dataloaders = {
        'Validation' : torch.utils.data.DataLoader(
            datasets["Validation"],
            batch_size=dataset_config.batch_size,
            shuffle=dataset_config.shuffle,
            num_workers=n_workers,
            drop_last=dataset_config.drop_last,
            pin_memory=True
        )
    }

    if hasattr(dataset_config.augmentations, 'rand_scale'):
        dataloaders['Training'] = torch.utils.data.DataLoader(
            datasets["Training"], num_workers=n_workers, pin_memory=True,
            batch_sampler=BatchSamplerRandScale(
                sampler=RandomSampler(datasets["Training"]),
                batch_size=dataset_config.batch_size,
                drop_last=dataset_config.drop_last,
                scale_range=dataset_config.augmentations.rand_scale)
        )
    else:
        torch.backends.cudnn.benchmark = True

        dataloaders['Training'] = torch.utils.data.DataLoader(
            datasets["Training"],
            batch_size=dataset_config.batch_size,
            shuffle=dataset_config.shuffle,
            num_workers=n_workers,
            drop_last=dataset_config.drop_last,
            pin_memory=True
        )

    return dataloaders

def copy_cityscapes(src_dir: Path, datasets: Dict[str, Path], subsets: List[str], dst_dir: Path):
    """
    Tool for copying over a subset of data to a new place i.e. HDD mass storage to SSD\n
    Requires you at least have l_img in the dictionary to act as a base for checking
    correct corresponding datasets are being copied.
    """
    assert 'l_img' in datasets

    print("Copying", datasets.keys(), "from", src_dir, "to", dst_dir)

    regex_map = {
        'l_img' : ['leftImg8bit', 'leftImg8bit'],
        'r_img' : ['leftImg8bit', 'rightImg8bit'],
        'seg' : ['leftImg8bit', 'gtFine_labelIds'],
        'inst' : ['leftImg8bit', 'gtFine_instanceIds'],
        'bbox' : ['leftImg8bit.png', 'gtFine_bbox.json'],
        'disp' : ['leftImg8bit', 'disparity'],
        'l_seq' : ['leftImg8bit', 'leftImg8bit'],
        'r_seq' : ['leftImg8bit', 'rightImg8bit'],
        'cam' : ['leftImg8bit.png', 'camera.json'],
        'pose' : ['leftImg8bit.png', 'vehicle.json']
    }

    for subset in subsets:
        for dirpath, _, filenames in os.walk(os.path.join(src_dir, datasets['l_img'], subset)):
            for filename in filenames:
                if not filename.endswith(IMG_EXT):
                    continue

                l_imgpath = os.path.join(dirpath, filename)
                foldername = os.path.basename(os.path.dirname(l_imgpath))

                for datatype, directory in datasets.items():
                    if datatype in ['l_seq', 'r_seq']:
                        frame_n = int(re.split("_", filename)[2])
                        img_name = filename.replace(
                            str(frame_n).zfill(6)+"_"+regex_map[datatype][0],
                            str(frame_n+1).zfill(6)+"_"+regex_map[datatype][1])
                    else:
                        img_name = filename.replace(
                            regex_map[datatype][0], regex_map[datatype][1])

                    src_path = os.path.join(src_dir, directory, subset, foldername, img_name)
                    dst_path = os.path.join(dst_dir, directory, subset, foldername, img_name)

                    if not os.path.isfile(src_path):
                        print("Error finding corresponding data to ", l_imgpath)
                        continue
                    if os.path.isfile(dst_path):
                        continue

                    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                    copy(src_path, dst_path)

    print("success copying: ", datasets.keys())

def some_test_idk():
    import matplotlib.pyplot as plt

    print("Testing Folder Traversal and Image Extraction!")

    hdd_dir = '/media/bryce/4TB Seagate/Autonomous Vehicles Data/Cityscapes Data'
    # ssd_dir = '/home/bryce/Documents/Cityscapes Data'

    # mono_training_data = {
    #     'images': hdd_dir + 'leftImg8bit/train',
    #     'labels': hdd_dir + 'gtFine/train'
    # }

    full_training_data = {
        'left_images': hdd_dir + 'leftImg8bit/train',
        'right_images': hdd_dir + 'rightImg8bit/train',
        'seg': hdd_dir + 'gtFine/train',
        'disparity': hdd_dir + 'disparity/train',
        'cam': hdd_dir + 'camera/train',
        'left_seq': hdd_dir + 'leftImg8bit_sequence/train',
        'pose': hdd_dir + 'vehicle/train'
    }

    test_dset = CityScapesDataset(full_training_data, crop_fraction=1)

    print(len(test_dset.l_img))
    print(len(test_dset.seg))

    batch_size = 2
    test_loader = torch.utils.data.DataLoader(
        test_dset, batch_size=batch_size, shuffle=True, num_workers=0)

    data = next(iter(test_loader))

    image = data["l_img"].numpy()
    seg = data["seg"].numpy()

    for i in range(batch_size):
        plt.subplot(121)
        plt.imshow(np.moveaxis(image[i, 0:3, :, :], 0, 2))

        plt.subplot(122)
        plt.imshow(seg[i, :, :])

        plt.show()

    # classes = {}
    # import matplotlib.pyplot as plt
    # for i in range(batch_size):
    #     # # Get class for each individual pixel
    #     # for j in range(seg.shape[1]):
    #     #     for k in range(seg.shape[2]):
    #     #         class_id = seg[i,j,k]
    #     #         classes[class_id] = class_id

    #     plt.subplot(121)
    #     img_cpy = image[i,:,:,:]
    #     plt.imshow(np.moveaxis(img_cpy,0,2))
    #     plt.subplot(122)
    #     plt.imshow(seg[i,:,:])
    #     plt.show()

    # print(classes)

def move_hdd_to_ssd():
    """
    Copies cityscapes data from one directory to another
    """
    hdd_dir = '/media/bryce/4TB Seagate/Autonomous Vehicles Data/Cityscapes Data'
    ssd_dir = '/media/bryce/1TB Samsung/ml_datasets/cityscapes_data'

    datasets = {
        'l_img': 'leftImg8bit',
        'disp': 'disparity',
        'l_seq': 'leftImg8bit_sequence',
        'seg': 'gtFine',
        'inst': 'gtFine'
    }

    subsets = ['train', 'val']

    copy_cityscapes(hdd_dir, datasets, subsets, ssd_dir)

def regex_rep_test():
    """
    Testing incrementing correctly with regex
    """
    testsrc = 'frankfurt_000001_000099_leftImg8bit'
    testseq = 'frankfurt_000001_000100_leftImg8bit'
    testrseq = 'frankfurt_000001_000100_rightImg8bit'

    frame_no = int(re.split("_", testsrc)[2])
    testdst = testsrc.replace(str(frame_no).zfill(6), str(frame_no+1).zfill(6))
    print(testdst)
    assert testdst == testseq

    testdst = testsrc.replace(
        str(frame_no).zfill(6)+"_leftImg8bit",
        str(frame_no+1).zfill(6)+"_rightImg8bit")
    print(testrseq)
    assert testdst == testrseq

if __name__ == '__main__':
    move_hdd_to_ssd()
