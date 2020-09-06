#!/usr/bin/env python3

__author__ = "Bryce Ferenczi"
__email__ = "bryce.ferenczi@monashmotorsport.com"

import platform
import os
import random
import json
import multiprocessing
from pathlib import Path
from typing import Dict

import torch
import torchvision
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt

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
        given or a direct depth estimation\n
        @param crop_fraction determines if the image is randomly cropped to by a fraction
            e.g. 2 results in width/2 by height/2 random crop of original image\n
        '''
        l_img_key = None

        for key in directories.keys():
            if key in ['left_images', 'images']:
                l_img_key = key
                self.l_img = []
            elif key == 'seg':
                seg_dir_key = key
                self.seg = []
            elif key == 'right_images':
                self.r_img = []
            elif key == 'disparity':
                self.disp = []
            elif key == 'left_seq':
                self.l_seq = []
            elif key == 'right_seq':
                self.r_seq = []
            elif key == 'cam':
                self.cam = []
            elif key == 'pose':
                self.pose = []

        if not hasattr(self, 'seg') and not hasattr(self, 'disp') and \
                not (hasattr(self, 'l_seq') or hasattr(self, 'r_seq')):
            print("Neither Segmentation, Disparity or Img Sequence Keys are defined")

        if l_img_key is None:
            print("Left Image Key Error")

        #Get all file names
        for dir_name, _, file_list in os.walk(directories[l_img_key]):
            for filename in file_list:
                if filename.endswith(IMG_EXT):
                    read_check = True
                    l_imgpath = os.path.join(dir_name, filename)
                    foldername = os.path.basename(os.path.dirname(l_imgpath))

                    if hasattr(self, 'r_img'):
                        r_imgname = filename.replace('leftImg8bit', 'rightImg8bit')
                        r_imgpath = os.path.join(directories['right_images'], foldername, r_imgname)
                        if not os.path.isfile(r_imgpath):
                            read_check = False
                            print("Error finding corresponding right image to ", l_imgpath)

                    if hasattr(self, 'seg'):
                        seg_name = filename.replace('leftImg8bit', 'gtFine_labelIds')
                        seg_path = os.path.join(directories[seg_dir_key], foldername, seg_name)
                        if not os.path.isfile(seg_path):
                            read_check = False
                            print("Error finding corresponding segmentation image to ", l_imgpath)

                    if hasattr(self, 'disp'):
                        disp_name = filename.replace('leftImg8bit', 'disparity')
                        disp_path = os.path.join(directories['disparity'], foldername, disp_name)
                        if not os.path.isfile(disp_path):
                            read_check = False
                            print("Error finding corresponding disparity image to ", l_imgpath)

                    if hasattr(self, 'l_seq'):
                        left_seq_name = filename.replace('19_leftImg8bit', '20_leftImg8bit')
                        left_seq_path = os.path.join(directories['left_seq'], foldername, left_seq_name)
                        if not os.path.isfile(left_seq_path):
                            read_check = False
                            print("Error finding corresponding left sequence image to ", l_imgpath)

                    if hasattr(self, 'r_seq'):
                        right_seq_name = filename.replace('19_leftImg8bit', '20_rightImg8bit')
                        right_seq_path = os.path.join(directories['right_seq'], foldername, right_seq_name)
                        if not os.path.isfile(right_seq_path):
                            read_check = False
                            print("Error finding corresponding right sequence image to ", l_imgpath)

                    if hasattr(self, 'cam'):
                        cam_name = filename.replace('leftImg8bit.png', 'camera.json')
                        cam_path = os.path.join(directories['cam'], foldername, cam_name)
                        if not os.path.isfile(cam_path):
                            read_check = False
                            print("Error finding corresponding camera parameters for ", l_imgpath)

                    if hasattr(self, 'pose'):
                        pose_name = filename.replace('leftImg8bit.png', 'vehicle.json')
                        pose_path = os.path.join(directories['pose'], foldername, pose_name)
                        if not os.path.isfile(pose_path):
                            read_check = False
                            print("Error finding corresponding GPS/Pose information for ", l_imgpath)

                    if read_check:
                        self.l_img.append(l_imgpath)
                        if hasattr(self, 'r_img'):
                            self.r_img.append(r_imgpath)
                        if hasattr(self, 'seg'):
                            self.seg.append(seg_path)
                        if hasattr(self, 'disp'):
                            self.disp.append(disp_path)
                        if hasattr(self, 'l_seq'):
                            self.l_seq.append(left_seq_path)
                        if hasattr(self, 'r_seq'):
                            self.r_seq.append(right_seq_path)
                        if hasattr(self, 'cam'):
                            self.cam.append(cam_path)
                        if hasattr(self, 'pose'):
                            self.pose.append(pose_path)

        # Create dataset from specified ids if id_vector given else use all
        if 'id_vector' in kwargs:
            self.l_img = [self.l_img[i] for i in kwargs['id_vector']]
            if hasattr(self, 'r_img'):
                self.r_img = [self.r_img[i] for i in kwargs['id_vector']]
            if hasattr(self, 'seg'):
                self.seg = [self.seg[i] for i in kwargs['id_vector']]
            if hasattr(self, 'disp'):
                self.disp = [self.disp[i] for i in kwargs['id_vector']]
            if hasattr(self, 'l_seq'):
                self.l_seq = [self.l_seq[i] for i in kwargs['id_vector']]
            if hasattr(self, 'r_seq'):
                self.r_seq = [self.r_seq[i] for i in kwargs['id_vector']]
            if hasattr(self, 'cam'):
                self.cam = [self.cam[i] for i in kwargs['id_vector']]
            if hasattr(self, 'pose'):
                self.pose = [self.pose[i] for i in kwargs['id_vector']]

        self.disparity_out = disparity_out
        self.output_size = output_size

        if 'crop_fraction' in kwargs:
            self.crop_fraction = kwargs['crop_fraction']
        if 'rand_rotation' in kwargs:
            self.rand_rot = kwargs['rand_rotation']
        if 'rand_brightness' in kwargs:
            self.brightness = kwargs['rand_brightness']

        # valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22,
        #                       23, 24, 25, 26, 27, 28, 31, 32, 33]
        self._key = np.array([-1, -1, -1, -1, -1, -1,
                              -1, -1, 0, 1, -1, -1,
                              2, 3, 4, -1, -1, -1,
                              5, -1, 6, 7, 8, 9,
                              10, 11, 12, 13, 14, 15,
                              -1, -1, 16, 17, 18])
        self._mapping = np.array(range(-1, len(self._key) - 1)).astype('int32')

    def __len__(self):
        return len(self.l_img)

    def __getitem__(self, idx):
        '''
        Returns relevant training data as a dict
        @output l_img, r_img, seg, disparity, l_seq, r_seq, cam, pose
        '''
        epoch_data = {}
        # Read image and labels
        epoch_data["l_img"] = Image.open(self.l_img[idx]).convert('RGB')

        if hasattr(self, 'r_img'):
            epoch_data["r_img"] = Image.open(self.r_img[idx]).convert('RGB')
        if hasattr(self, 'seg'):
            epoch_data["seg"] = Image.open(self.seg[idx])
        if hasattr(self, 'disp'):
            epoch_data["disparity"] = Image.open(self.disp[idx])
        if hasattr(self, 'l_seq'):
            epoch_data["l_seq"] = Image.open(self.l_seq[idx]).convert('RGB')
        if hasattr(self, 'r_seq'):
            epoch_data["r_seq"] = Image.open(self.r_seq[idx]).convert('RGB')

        self._sync_transform(epoch_data)

        #   Transform images to tensors
        for key in epoch_data.keys():
            if key in ["l_img", "r_img", "l_seq", "r_seq"]:
                epoch_data[key] = torchvision.transforms.functional.to_tensor(epoch_data[key])

        if hasattr(self, 'cam'):
            epoch_data["cam"] = self.json_to_intrinsics(self.cam[idx])

        # if hasattr(self, 'pose'):
        #     epoch_data["pose"] = self.json_to_pose(self.pose[idx])

        return epoch_data

    def _sync_transform(self, epoch_data):
        # random mirror
        if random.random() < 0.5:
            for item in epoch_data.values():
                item = item.transpose(Image.FLIP_LEFT_RIGHT)

        if hasattr(self, 'crop_fraction'):
            # random crop
            crop_h = int(epoch_data["l_img"].size[0]/self.crop_fraction)
            crop_w = int(epoch_data["l_img"].size[1]/self.crop_fraction)
            crop_x = random.randint(0, epoch_data["l_img"].size[1] - crop_w)
            crop_y = random.randint(0, epoch_data["l_img"].size[0] - crop_h)

            for item in epoch_data.values():
                item = item.crop((crop_y, crop_x, crop_y+crop_h, crop_x+crop_w))

        if hasattr(self, 'rand_rot'):
            angle = random.uniform(0, self.rand_rot)
            for key in epoch_data.keys():
                if key in ["l_img", "r_img", "l_seq", "r_seq"]:
                    item = torchvision.transforms.functional.rotate(
                        epoch_data['key'], angle, resample=Image.BILINEAR)
                else:
                    item = torchvision.transforms.functional.rotate(
                        epoch_data['key'], angle, resample=Image.NEAREST, fill=-1)

        if hasattr(self, 'brightness'):
            brightness_scale = random.uniform(1-self.brightness/100, 1+self.brightness/100)
            for key in epoch_data.keys():
                if key in ["l_img", "r_img", "l_seq", "r_seq"]:
                    epoch_data[key] = torchvision.transforms.functional.adjust_brightness(
                        epoch_data[key], brightness_scale)

        for key in epoch_data.keys():
            if key in ["l_img", "r_img", "l_seq", "r_seq"]:                  
                epoch_data[key] = epoch_data[key].resize(self.output_size, Image.BILINEAR)
                epoch_data[key] = self._img_transform(epoch_data[key])
            elif key == "seg":
                epoch_data[key] = epoch_data[key].resize(self.output_size, Image.NEAREST)
                epoch_data[key] = self._seg_transform(epoch_data[key])
            elif key == "disparity":
                epoch_data[key] = epoch_data[key].resize(self.output_size, Image.NEAREST)
                epoch_data[key] = self._depth_transform(epoch_data[key])

    def _class_to_index(self, seg):
        values = np.unique(seg)
        for value in values:
            assert value in self._mapping
        index = np.digitize(seg.ravel(), self._mapping, right=True)
        return self._key[index].reshape(seg.shape)

    def _img_transform(self, img):
        return np.array(img)

    def _seg_transform(self, seg):
        target = self._class_to_index(np.array(seg).astype('int32'))
        return torch.LongTensor(np.array(target).astype('int32'))

    def _depth_transform(self, disparity):
        disparity = np.array(disparity).astype('float32')
        if not self.disparity_out:
            disparity[disparity > 1] = (0.209313 * 2262.52) / ((disparity[disparity > 1] - 1) / 256)
            disparity[disparity < 2] = -1 # Ignore value for loss functions
            # Ignore sides and bottom of frame as these are patchy/glitchy
            side_clip   = int(disparity.shape[1]/20)
            bottom_clip = int(disparity.shape[0]/10)
            disparity[-bottom_clip:-1,:]    = -1    #bottom
            disparity[:,:side_clip]         = -1    #lhs
            disparity[:,-side_clip:-1]      = -1    #rhs
        return  torch.FloatTensor(disparity.astype('float32'))

    def json_to_intrinsics(self, json_path):
        with open(json_path) as json_file:
            #   Camera Intrinsic Matrix
            K = np.eye(4, dtype=np.float32) #Idk why size 4? (To match translation?)
            json_data = json.load(json_file)
            K[0,0] = json_data["intrinsic"]["fx"]
            K[1,1] = json_data["intrinsic"]["fy"]
            K[0,2] = json_data["intrinsic"]["u0"]
            K[1,2] = json_data["intrinsic"]["v0"]

            #   Transformation Mat between cameras
            stereo_T = np.eye(4, dtype=np.float32)
            stereo_T[0,3] = json_data["extrinsic"]["baseline"]

        return {"K":K, "inv_K":np.linalg.pinv(K), "baseline_T":stereo_T}

    def json_to_pose(self, json_path):
        raise NotImplementedError

def id_vec_generator(train_ratio, val_ratio, test_ratio, directory):
    num_images = 0
    for file in os.listdir(directory):
        num_images += file.endswith(IMG_EXT)

    print("Number of Images:\t", num_images)
    image_ids = list(range(num_images))
    random.shuffle(image_ids)

    ratio_sum = train_ratio + val_ratio + test_ratio
    n_train = int(num_images*train_ratio/ratio_sum)
    n_val = int(num_images*val_ratio/ratio_sum)
    # n_test = int(num_images*test_ratio/ratio_sum) #This is implicit anyway

    train_ids = image_ids[0:n_train]
    val_ids = image_ids[n_train+1:n_train+n_val]
    test_ids = image_ids[n_train+n_val:-1]

    return train_ids, val_ids, test_ids

def get_cityscapse_dataset(dataset_config) -> Dict[str, torch.utils.data.DataLoader]:
    """
    Returns a cityscapes dataset given a config
    """
    if platform.system() == 'Windows':
        n_workers = 0
    else:
        n_workers = min(multiprocessing.cpu_count(), dataset_config.batch_size)

    crop_fraction = dataset_config.augmentations.crop_fraction
    output_size = dataset_config.augmentations.output_size

    training_dirs = {}
    for subset in dataset_config.train_subdirs:
        training_dirs[str(subset)] = dataset_config.rootdir +\
                                        dataset_config.train_subdirs[str(subset)]

    validation_dirs = {}
    for subset in dataset_config.val_subdirs:
        validation_dirs[str(subset)] = dataset_config.rootdir +\
                                        dataset_config.val_subdirs[str(subset)]

    datasets = {
        'Training'   : CityScapesDataset(training_dirs,
                                         crop_fraction=crop_fraction,
                                         output_size=output_size),
        'Validation' : CityScapesDataset(validation_dirs,
                                         crop_fraction=crop_fraction,
                                         output_size=output_size)
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
        ),
    }

    return dataloaders

if __name__ == '__main__':
    print("Testing Folder Traversal and Image Extraction!")

    mono_training_data = {
        'images': '/media/bryce/4TB Seagate/Autonomous Vehicles Data/Cityscapes Data/leftImg8bit/train',
        'labels': '/media/bryce/4TB Seagate/Autonomous Vehicles Data/Cityscapes Data/gtFine/train'
    }

    full_training_data = {
        'left_images': '/media/bryce/4TB Seagate/Autonomous Vehicles Data/Cityscapes Data/leftImg8bit/train',
        'right_images': '/media/bryce/4TB Seagate/Autonomous Vehicles Data/Cityscapes Data/rightImg8bit/train',
        'seg': '/media/bryce/4TB Seagate/Autonomous Vehicles Data/Cityscapes Data/gtFine/train',
        'disparity': '/media/bryce/4TB Seagate/Autonomous Vehicles Data/Cityscapes Data/disparity/train',
        'cam': '/media/bryce/4TB Seagate/Autonomous Vehicles Data/Cityscapes Data/camera/train',
        'left_seq': '/media/bryce/4TB Seagate/Autonomous Vehicles Data/Cityscapes Data/leftImg8bit_sequence/train',
        'pose': '/media/bryce/4TB Seagate/Autonomous Vehicles Data/Cityscapes Data/vehicle/train'
    }

    test_dset = CityScapesDataset(full_training_data, crop_fraction=1)
    
    print(len(test_dset.l_img))
    print(len(test_dset.seg))

    import multiprocessing
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader

    batch_size = 2
    testLoader = DataLoader(test_dset, batch_size=batch_size, shuffle=True, num_workers=0)

    for idx, data in enumerate(testLoader):
        print("yes")
    
    data = next(iter(testLoader))

    image = data["l_img"].numpy()
    seg = data["seg"].numpy()

    for i in range(batch_size):
        plt.subplot(121)
        plt.imshow(np.moveaxis(image[i,0:3,:,:],0,2))    

        plt.subplot(122)
        plt.imshow(seg[i,:,:])

        plt.show()
    
    # classes = {}
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
