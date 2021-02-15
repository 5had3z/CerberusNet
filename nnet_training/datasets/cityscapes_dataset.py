#!/usr/bin/env python3

"""
Dataloader and misc utilities related to the cityscapes dataset
"""

__author__ = "Bryce Ferenczi"
__email__ = "bryce.ferenczi@monash.edu"

import os
import re
import random
import json
from shutil import copy
from pathlib import Path
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import torch
import torchvision
from PIL import Image

import numpy as np

from nnet_training.nnet_models.detr.box_ops import box_xyxy_to_cxcywh
from nnet_training.nnet_models.detr.box_ops import normalize_boxes
from nnet_training.utilities.visualisation import get_color_pallete

from .cityscapes_panoptic_transform import PanopticTargetGenerator

__all__ = ['CityScapesDataset']

IMG_EXT = '.png'

class CityScapesDataset(torch.utils.data.Dataset):
    """
    Cityscapes Dataset\n
    On initialization, give a dictionary to the respective paths with the corresponding keys:
        [(left_images or images), right_images, seg, disparity, left_seq, right_seq, cam, pose]
    Get Item will return the corresponding dictionary with keys:
        [l_img, r_img, l_seq", r_seq, cam, pose, disparity]
    """
    sub_directories = {
        "l_img" : lambda root, folder, filename : \
            os.path.join(root, "leftImg8bit/", folder, filename),
        "r_img" : lambda root, folder, filename : \
            CityScapesDataset.replace_filename(
                os.path.join(root, "rightImg8bit/", folder),
                filename, 'leftImg8bit', 'rightImg8bit'),
        "seg" : lambda root, folder, filename : \
            CityScapesDataset.replace_filename(
                os.path.join(root, "gtFine/", folder),
                filename, 'leftImg8bit', 'gtFine_labelIds'),
        "panoptic" : lambda root, folder, filename : \
            CityScapesDataset.replace_filename(
                os.path.join(root, "gtFine/", folder),
                filename, 'leftImg8bit', 'gtFine_panoptic'),
        "bbox" : lambda root, folder, filename : \
            CityScapesDataset.replace_filename(
                os.path.join(root, "gtFine/", folder),
                filename, 'leftImg8bit.png', 'gtFine_bbox.json'),
        "disparity": lambda root, folder, filename : \
            CityScapesDataset.replace_filename(
                os.path.join(root, "disparity/", folder),
                filename, 'leftImg8bit', 'disparity'),
        "l_seq" : lambda root, folder, filename : \
            CityScapesDataset.replace_filename(
                os.path.join(root, "leftImg8bit_sequence/", folder), filename,
                f"{int(re.split('_', filename)[2]):03}_leftImg8bit",
                f"{int(re.split('_', filename)[2])+1:03}_leftImg8bit"),
        "r_seq" : lambda root, folder, filename : \
            CityScapesDataset.replace_filename(
                os.path.join(root, "rightImg8bit_sequence/", folder), filename,
                f"{int(re.split('_', filename)[2]):03}_leftImg8bit",
                f"{int(re.split('_', filename)[2])+1:03}_rightImg8bit"),
        "cam"   : lambda root, folder, filename : \
            CityScapesDataset.replace_filename(
                os.path.join(root, "camera/", folder),
                filename, 'leftImg8bit.png', 'camera.json'),
        "pose" : lambda root, folder, filename : \
            CityScapesDataset.replace_filename(
                os.path.join(root, "vehicle/", folder),
                filename, 'leftImg8bit.png', 'vehicle.json'),
    }

    base_size = [2048, 1024]

    # valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22,
    #                       23, 24, 25, 26, 27, 28, 31, 32, 33]
    train_ids = np.array([255, 255, 255, 255, 255, 255,
                          255, 255, 0, 1, 255, 255,
                          2, 3, 4, 255, 255, 255,
                          5, 255, 6, 7, 8, 9,
                          10, 11, 12, 13, 14, 15,
                          255, 255, 16, 17, 18])

    id_mapping = np.array(range(-1, len(train_ids) - 1)).astype('int32')

    valid_augmentations = set(
        ["crop_fraction", "rand_rotation", "rand_brightness", "img_normalize", "rand_flip"])

    cityscapes_things = [11, 12, 13, 14, 15, 16, 17, 18]

    @staticmethod
    def replace_filename(root_dir, filename, old_str: str, new_str: str) -> str:
        return os.path.join(root_dir, filename.replace(old_str, new_str))

    def __init__(self, directory: Path, subsets: List[str], split: str,
                 output_size=(1024, 512), disparity_out=False, **kwargs):
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

        assert all(subset in CityScapesDataset.sub_directories for subset in subsets), \
            f"Invalid subset found in given {subsets}"
        assert "l_img" in subsets, "Requires left images present"
        assert split in ["train", "test", "val"], f"Invalid Split given: {split}"

        self.subsets = subsets
        self.root_dir = directory
        self.split = split
        self.l_img = []
        self.panoptic_json = {}

        self._validate_dataset(directory, subsets, **kwargs)

        self.disparity_out = disparity_out
        self.output_size = output_size # (width, height)
        self.scale_factor = 1

        self.augmentations = {}
        for k, v in kwargs.items():
            if k == 'img_normalize':
                # Typical normalisation parameters:
                # "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]
                self.augmentations[k] = torchvision.transforms.Normalize(
                kwargs['img_normalize']['mean'], kwargs['img_normalize']['std'])
            elif k in CityScapesDataset.valid_augmentations:
                self.augmentations[k] = v
            else:
                Warning(f"Invalid Augmentation: {k}, {v}")

        if 'box_type' in kwargs:
            self.bbox_type = kwargs['box_type']
        if 'panoptic' in subsets:
            self.panoptic_gen = PanopticTargetGenerator(
                ignore_label=255, thing_list=CityScapesDataset.cityscapes_things)
            self.init_panoptic_json(
                f"{self.root_dir}gtFine/cityscapes_panoptic_{self.split}.json")

    def _validate_dataset(self, root_dir, subsets, **kwargs):
        """
        Gets all the filenames
        """
        for dirpath, _, filenames in os.walk(os.path.join(root_dir, 'leftImg8bit', self.split)):
            for filename in filenames:
                if not filename.endswith(IMG_EXT):
                    continue

                l_imgpath = os.path.join(dirpath, filename)
                foldername = os.path.join(self.split, os.path.basename(os.path.dirname(l_imgpath)))
                self.l_img.append(l_imgpath)

                for subset in subsets:
                    if subset == "panoptic":
                        target_pth = CityScapesDataset.sub_directories[subset](
                                root_dir, f"cityscapes_panoptic_{self.split}", filename)
                    else:
                        target_pth = CityScapesDataset.sub_directories[subset](
                                root_dir, foldername, filename)
                    assert os.path.isfile(target_pth), \
                        f"Error target {target_pth} corresponding to {l_imgpath}"

        # Create dataset from specified ids if id_vector given else use all
        if 'id_vector' in kwargs:
            self.l_img = [self.l_img[i] for i in kwargs['id_vector']]

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
        foldername = os.path.join(self.split, os.path.basename(os.path.dirname(self.l_img[idx])))
        filename = os.path.basename(self.l_img[idx])

        for subset in self.subsets:
            if subset in ["r_img", "l_seq", "r_seq"]:
                img_pth = CityScapesDataset.sub_directories[subset](
                    self.root_dir, foldername, filename)
                epoch_data[subset] = Image.open(img_pth).convert('RGB')
            elif subset in ['seg', 'l_disp', 'r_disp']:
                img_pth = CityScapesDataset.sub_directories[subset](
                    self.root_dir, foldername, filename)
                epoch_data[subset] = torchvision.transforms.functional.to_tensor(
                    np.asarray(Image.open(img_pth), dtype=np.int32))
            elif subset == "bbox":
                bbox_pth = CityScapesDataset.sub_directories[subset](
                    self.root_dir, foldername, filename)
                epoch_data['labels'], epoch_data['bboxes'] = self.bbox_json_parse(bbox_pth)
            elif subset == "panoptic":
                img_pth = CityScapesDataset.sub_directories[subset](
                    self.root_dir, f"cityscapes_panoptic_{self.split}", filename)
                panoptic_img = np.asarray(Image.open(img_pth))
                segments_info = self.panoptic_json[filename.replace("_leftImg8bit.png", '')]
                epoch_data.update(self.panoptic_gen(panoptic_img, segments_info))

        for key, data in epoch_data.items():
            if isinstance(data, Image.Image):
                epoch_data[key] = torchvision.transforms.functional.to_tensor(data)

        self._sync_transform(epoch_data)

        if 'panoptic' in self.subsets:
            for key in self.panoptic_gen.generated_items:
                epoch_data[key] = epoch_data[key].squeeze(0)

        if 'cam' in self.subsets:
            cam_pth = CityScapesDataset.sub_directories['cam'](
                self.root_dir, foldername, filename)
            epoch_data["cam"] = self.intrinsics_json(cam_pth)

        # if hasattr(self, 'pose'):
        #     epoch_data["pose"] = self.pose_json(self.pose[idx])

        if all(key in epoch_data.keys() for key in ['bboxes', 'labels']):
            epoch_data['bboxes'] = torch.as_tensor(epoch_data['bboxes'], dtype=torch.float32)
            epoch_data['labels'] = torch.as_tensor(epoch_data['labels'], dtype=torch.int64)
            if epoch_data['bboxes'].shape[0] != 0:
                if hasattr(self, 'bbox_type') and self.bbox_type == 'cxcywh':
                    epoch_data['bboxes'] = box_xyxy_to_cxcywh(epoch_data['bboxes'])
                epoch_data['bboxes'] = normalize_boxes(epoch_data['bboxes'],
                                                       epoch_data['l_img'].shape[-2:])

        return epoch_data

    @staticmethod
    def _mirror_bbox(bbox_list: List[List[int]], img_dims: Tuple[int, int]) -> List[List[int]]:
        """
        Mirrors a boundary box [x1,y1,x2,y2] in an image of img_dims (width, height)
        """
        ret_val = []
        mirror_x = lambda x : int(img_dims[0] / 2) - (x - int(img_dims[0] / 2))
        for bbox in bbox_list:
            ret_val.append([mirror_x(bbox[2]), bbox[1], mirror_x(bbox[0]), bbox[3]])
        return ret_val

    @staticmethod
    def _crop_bbox(batch_data: List[List[int]],
                   crop_dims: Tuple[int, int, int, int]) -> None:
        """
        Transforms a boundary box [x1,y1,x2,y2] according to an image crop (x, y, w, h)
        """
        for idx, bbox in enumerate(batch_data['bboxes']):
            # Shift bbox top left, clip at 0
            b_x1 = max(bbox[0] - crop_dims[0], 0)
            b_y1 = max(bbox[1] - crop_dims[1], 0)
            # Shift bbox top left, clip at bottom right boundary
            # if image crop encroaches on transformed bbox.
            b_x2 = min(bbox[2] - crop_dims[0], crop_dims[2])
            b_y2 = min(bbox[3] - crop_dims[1], crop_dims[3])

            if b_x1 > crop_dims[2] or b_y1 > crop_dims[3]:
                Warning(f"Box {idx}: {bbox} outside of crop {crop_dims}")
                del batch_data['labels'][idx]
                del batch_data['bboxes'][idx]
            else:
                batch_data['bboxes'][idx] = [b_x1, b_y1, b_x2, b_y2]

    @staticmethod
    def _rotate_bbox(batch_data: List[List[int]], angle: float) -> None:
        """
        Transforms a boundary box [x1,y1,x2,y2] according to an image
        rotation angle and ensures it stays within clipping.
        """
        rad_angle = np.deg2rad(-angle)
        rot_mat = np.asarray([[np.cos(rad_angle), -np.sin(rad_angle)],
                               [np.sin(rad_angle), np.cos(rad_angle)]])
        img_tr = np.asarray(CityScapesDataset.base_size) / 2

        clamp = lambda x, _min, _max: min(max(x, _min), _max)

        for idx, bbox in enumerate(batch_data['bboxes']):
            corners = [
                np.matmul(rot_mat, np.asarray(bbox[:2]) - img_tr) + img_tr,
                np.matmul(rot_mat, np.asarray([bbox[2], bbox[1]])- img_tr) + img_tr,
                np.matmul(rot_mat, np.asarray([bbox[0], bbox[3]]) - img_tr) + img_tr,
                np.matmul(rot_mat, np.asarray(bbox[2:]) - img_tr) + img_tr
            ]

            x_1 = clamp(min(corners, key=lambda x: x[0])[0], 0, CityScapesDataset.base_size[0])
            y_1 = clamp(min(corners, key=lambda x: x[1])[1], 0, CityScapesDataset.base_size[1])
            x_2 = clamp(max(corners, key=lambda x: x[0])[0], 0, CityScapesDataset.base_size[0])
            y_2 = clamp(max(corners, key=lambda x: x[1])[1], 0, CityScapesDataset.base_size[1])

            assert x_1 <= x_2, f"Angle {angle}: {x_1} > {x_2} for {bbox} -> {[x_1, y_1, x_2, y_2]}"
            assert y_1 <= y_2, f"Angle {angle}: {y_1} > {y_2} for {bbox} -> {[x_1, y_1, x_2, y_2]}"

            if x_1 == x_2 or y_1 == y_2:
                Warning(f"Box {idx}: {bbox} outside of rotation augmentation {angle}")
                del batch_data['labels'][idx]
                del batch_data['bboxes'][idx]
            else:
                batch_data['bboxes'][idx] = [x_1, y_1, x_2, y_2]

    @staticmethod
    def _scale_bbox(bbox_list: List[List[int]], dst_dims: Tuple[int, int]) -> List[List[int]]:
        """
        Scales bbox from scr dimensions to dst dimensions
        """
        ret_val = []
        [x_scale, y_scale] = [dst / src for dst, src in zip(dst_dims, CityScapesDataset.base_size)]
        for bbox in bbox_list:
            ret_val.append([bbox[0] * x_scale, bbox[1] * y_scale,
                            bbox[2] * x_scale, bbox[3] * y_scale])
        return ret_val

    @torch.no_grad()
    def _sync_transform(self, epoch_data: Dict[str, Union[Image.Image, List]]) -> None:
        """
        Augments and formats all the batch data in a synchronised manner
        """
        scale_func = lambda x: int(self.scale_factor * x / 32.0) * 32
        output_shape = [scale_func(x) for x in self.output_size]

        # random mirror
        if self.augmentations.get('rand_flip', False) and random.random() < 0.5:
            for key, data in epoch_data.items():
                if key not in ['labels', 'bboxes']:
                    epoch_data[key] = torchvision.transforms.functional.hflip(data)
                elif key == 'bboxes':
                    epoch_data[key] = self._mirror_bbox(data, CityScapesDataset.base_size)

        if 'rand_brightness' in self.augmentations:
            brightness_scale = random.uniform(
                1-self.augmentations['rand_brightness']/100,
                1+self.augmentations['rand_brightness']/100)
            for key, data in epoch_data.items():
                if key in ["l_img", "r_img", "l_seq", "r_seq"]:
                    epoch_data[key] = torchvision.transforms.functional.adjust_brightness(
                        data, brightness_scale)

        if 'rand_rot' in self.augmentations:
            # TODO Add mask for bboxes which is required for attention head
            angle = random.uniform(0, self.augmentations['rand_rot'])
            for key, data in epoch_data.items():
                if key in ["l_img", "r_img", "l_seq", "r_seq"]:
                    epoch_data[key] = torchvision.transforms.functional.rotate(
                        data, angle, resample=Image.BILINEAR)
                elif key in ['l_disp', 'r_disp', 'seg']:
                    epoch_data[key] = torchvision.transforms.functional.rotate(
                        data, angle, resample=Image.NEAREST, fill=-1)
                elif key == 'bboxes':
                    self._rotate_bbox(epoch_data, angle)

        for key, data in epoch_data.items():
            if key in ["l_img", "r_img", "l_seq", "r_seq", 'center', 'offset']:
                epoch_data[key] = torchvision.transforms.functional.resize(
                    data, tuple(output_shape[::-1]), Image.BILINEAR)
            elif key in ["l_disp", 'r_disp', "seg", 'foreground'] or key.endswith('mask'):
                epoch_data[key] = torchvision.transforms.functional.resize(
                    data, tuple(output_shape[::-1]), Image.NEAREST)
            elif key == "center_points":
                new_points = []
                rescale_func = lambda x: [tgt * new / old for (tgt, new, old) \
                                in zip(x, output_shape, CityScapesDataset.base_size)]
                for points in data:
                    new_points.append(rescale_func(points))
                epoch_data[key] = new_points
            elif key == "bboxes":
                epoch_data[key] = self._scale_bbox(data, output_shape)

        for key in ["l_img", "r_img", "l_seq", "r_seq"]:
            if key in epoch_data:
                epoch_data[key] = self._img_transform(epoch_data[key])

        for key in ["l_disp", 'r_disp']:
            if key in epoch_data:
                epoch_data[key] = self._depth_transform(epoch_data[key])

        if 'seg' in epoch_data:
            epoch_data['seg'] = self._seg_transform(epoch_data['seg'])


        if 'crop_fraction' in self.augmentations:
            # random crop
            crop_fraction = self.augmentations['crop_fraction']
            crop_h = int(epoch_data["l_img"].shape[1] / crop_fraction / 32.0) * 32
            crop_w = int(epoch_data["l_img"].shape[2] / crop_fraction / 32.0) * 32
            crop_x = random.randint(0, epoch_data["l_img"].shape[2] - crop_w)
            crop_y = random.randint(0, epoch_data["l_img"].shape[1] - crop_h)

            for key, data in epoch_data.items():
                if key in ["l_img", "r_img", "l_seq", "r_seq"]:
                    epoch_data[key] = data[:, crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
                elif key in ["seg", "l_disp", "r_disp"]:
                    epoch_data[key] = data[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
                elif key == 'bboxes':
                    self._crop_bbox(data, (crop_x, crop_y, crop_w, crop_h))

    @staticmethod
    def _class_to_index(seg):
        values = np.unique(seg)
        for value in values:
            assert value in CityScapesDataset.id_mapping
        index = np.digitize(seg.ravel(), CityScapesDataset.id_mapping, right=True)
        return CityScapesDataset.train_ids[index].reshape(seg.shape)

    def _img_transform(self, img):
        if 'img_normalize' in self.augmentations:
            img = self.augmentations['img_normalize'](img)
        return img

    def _seg_transform(self, seg):
        target = self._class_to_index(np.array(seg).astype('int32'))
        return torch.LongTensor(target.astype('int32'))

    def _depth_transform(self, disparity):
        disparity = np.array(disparity).astype('float32')
        disparity[disparity > 0] = self.scale_factor * (disparity[disparity > 0] - 1) / 256.
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
                # assert bbox['train_id'] < 255, f"invalid id {bbox['train_id']}"
                if bbox['train_id'] < 19:
                    labels.append(bbox['train_id'])
                    bboxes.append(bbox['bbox'])

        return labels, bboxes

    def init_panoptic_json(self, json_path: str):
        """
        Loads the annotation data into the panoptic_json dictonary for fast lookup during training.
        """
        with open(json_path) as json_file:
            json_data = json.load(json_file)
            for entry in json_data['annotations']:
                self.panoptic_json[entry['image_id']] = entry['segments_info']


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

    full_training_data = ['l_img', 'r_img', 'seg', 'disparity', 'cam', 'l_seq', 'pose']

    test_dset = CityScapesDataset(hdd_dir, full_training_data, 'train', crop_fraction=1)

    print(len(test_dset.l_img))

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
        plt.imshow(get_color_pallete(seg[i, :, :]))

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
    some_test_idk()
