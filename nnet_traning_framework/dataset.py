#!/usr/bin/env python3

__author__ = "Bryce Ferenczi"
__email__ = "bryce.ferenczi@monashmotorsport.com"

import torch
import torchvision
from PIL import Image

import os
import random
from pathlib import Path

import csv
import numpy as np
import matplotlib.pyplot as plt

imgextension = '.png'

class CityScapesDataset(torch.utils.data.Dataset):
    def __init__(self, directories, output_size=(1024,512), crop_fraction=2, id_vector=None, transform=torchvision.transforms.ToTensor()):
        
        l_img_key = None
        self.enable_right = False
        self.enable_seg = False
        self.enable_disparity = False

        for key in directories.keys():
            if key == 'left_images' or key == 'images':
                l_img = []
                l_img_key = key
            elif key == 'labels' or key == 'mask':
                mask = []
                seg_dir_key = key
                self.enable_seg = True
            elif key == 'right_images':
                r_img = []
                self.enable_right = True
            elif key == 'disparity':
                disp = []
                self.enable_disparity = True

        if not self.enable_seg and not self.enable_disparity:
            print("Neither Segmentation or Disparity Keys are defined")

        if l_img_key is None:
            print("Left Image Key Error")

        #Get all file names
        for dirName, _, fileList in os.walk(directories[l_img_key]):
            for filename in fileList:
                if filename.endswith(imgextension):
                    read_check = True
                    l_imgpath = os.path.join(dirName, filename)
                    foldername = os.path.basename(os.path.dirname(l_imgpath))

                    if self.enable_right:
                        r_imgname = filename.replace('leftImg8bit', 'rightImg8bit')
                        r_imgpath = os.path.join(directories['right_images'], foldername, r_imgname)
                        if not os.path.isfile(r_imgpath):
                            read_check = False
                            print("Error finding corresponding right image to ", l_imgpath)

                    if self.enable_seg:
                        mask_name = filename.replace('leftImg8bit', 'gtFine_labelIds')
                        mask_path = os.path.join(directories[seg_dir_key], foldername, mask_name)
                        if not os.path.isfile(mask_path):
                            read_check = False
                            print("Error finding corresponding segmentation image to ", l_imgpath)

                    if self.enable_disparity:
                        disp_name = filename.replace('leftImg8bit', 'disparity')
                        disp_path = os.path.join(directories['disparity'], foldername, disp_name)
                        if not os.path.isfile(disp_path):
                            read_check = False
                            print("Error finding corresponding disparity image to ", l_imgpath)

                    if read_check:
                        l_img.append(l_imgpath)

                        if self.enable_right:
                            r_img.append(r_imgpath)
                        if self.enable_seg:
                            mask.append(mask_path)
                        if self.enable_disparity:
                            disp.append(disp_path)
        
        #Create dataset from specified ids
        if id_vector is not None:
            self.l_img = [l_img[i] for i in id_vector]
            if self.enable_right:
                self.r_img = [r_img[i] for i in id_vector]
            if self.enable_seg:
                self.mask = [mask[i] for i in id_vector]
            if self.enable_disparity:
                self.disp = [disp[i] for i in id_vector]
        else:
            self.l_img = l_img
            if self.enable_right:
                self.r_img = r_img
            if self.enable_seg:
                self.mask = mask
            if self.enable_disparity:
                self.disp = disp

        self.transform = transform

        self.output_size = output_size
        self.crop_fraction = crop_fraction

        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22,
                              23, 24, 25, 26, 27, 28, 31, 32, 33]
        self._key = np.array([-1, -1, -1, -1, -1, -1,
                              -1, -1, 0, 1, -1, -1,
                              2, 3, 4, -1, -1, -1,
                              5, -1, 6, 7, 8, 9,
                              10, 11, 12, 13, 14, 15,
                              -1, -1, 16, 17, 18])
        self._mapping = np.array(range(-1, len(self._key) - 1)).astype('int32')

    def __getitem__(self, idx):
        '''
        Returns an Image and Label Pair
        '''
        # Read image and labels
        l_img = Image.open(self.l_img[idx]).convert('RGB')
        r_img = None
        mask = None
        disparity = None

        if self.enable_right:
            r_img = Image.open(self.r_img[idx]).convert('RGB')
        if self.enable_seg:
            mask = Image.open(self.mask[idx])
        if self.enable_disparity:
            disparity = Image.open(self.disp[idx])

        image, r_img, mask, disparity = self._sync_transform(l_img, r_img, mask, disparity)

        #   Apply Defined Transformations
        if self.transform is not None:
            image = self.transform(image)
            if self.enable_right:
                r_img = self.transform(r_img)

        if self.enable_right:
            image = image, r_img

        if self.enable_seg and self.enable_disparity:
            label = (mask, disparity)
        elif self.enable_seg:
            label = mask
        else:
            label = disparity

        return image, label

    def _sync_transform(self, l_img, r_img, mask, disparity):

        # random mirror
        if random.random() < 0.5:
            l_img = l_img.transpose(Image.FLIP_LEFT_RIGHT)
            if self.enable_right: 
                r_img = r_img.transpose(Image.FLIP_LEFT_RIGHT)
            if self.enable_seg:
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            if self.enable_disparity:
                disparity = disparity.transpose(Image.FLIP_LEFT_RIGHT)
        
        # random crop
        crop_h, crop_w = int(l_img.size[0]/self.crop_fraction), int(l_img.size[1]/self.crop_fraction)
        crop_x, crop_y = random.randint(0, l_img.size[1] - crop_w), random.randint(0, l_img.size[0] - crop_h)

        l_img = l_img.crop((crop_y, crop_x, crop_y+crop_h, crop_x+crop_w))

        if self.enable_right: 
            r_img = r_img.crop((crop_y, crop_x, crop_y+crop_h, crop_x+crop_w))
        if self.enable_seg:
            mask = mask.crop((crop_y, crop_x, crop_y+crop_h, crop_x+crop_w))
        if self.enable_disparity:
            disparity = disparity.crop((crop_y, crop_x, crop_y+crop_h, crop_x+crop_w))

        # resize to target
        l_img = l_img.resize(self.output_size, Image.BILINEAR)
        l_img = self._img_transform(l_img)

        if self.enable_right: 
            r_img = r_img.resize(self.output_size, Image.BILINEAR)
            r_img = self._img_transform(r_img)
        if self.enable_seg:
            mask = mask.resize(self.output_size, Image.NEAREST)
            mask = self._mask_transform(mask)
        if self.enable_disparity:
            disparity = disparity.resize(self.output_size, Image.NEAREST)
            disparity = self._depth_transform(disparity)
        
        return l_img, r_img, mask, disparity

    def _class_to_index(self, mask):
        values = np.unique(mask)
        for value in values:
            assert (value in self._mapping)
        index = np.digitize(mask.ravel(), self._mapping, right=True)
        return self._key[index].reshape(mask.shape)

    def _img_transform(self, img):
        return np.array(img)

    def _mask_transform(self, mask):
        target = self._class_to_index(np.array(mask).astype('int32'))
        return torch.LongTensor(np.array(target).astype('int32'))

    def _depth_transform(self, disparity):
        disparity = np.array(disparity).astype('float32')
        disparity[disparity > 1] = (0.209313 * 2262.52) / ((disparity[disparity > 1] - 1) / 256)
        disparity[disparity < 2] = -1 # Ignore value for loss functions
        # Ignore sides and bottom of frame as these are patchy/glitchy
        side_clip   = int(disparity.shape[1]/20)
        bottom_clip = int(disparity.shape[0]/10)
        disparity[-bottom_clip:-1,:]    = -1    #bottom
        disparity[:,:side_clip]         = -1    #lhs
        disparity[:,-side_clip:-1]      = -1    #rhs
        return  torch.FloatTensor(disparity.astype('float32'))

    def __len__(self):
        return len(self.l_img)

def id_vec_generator(train_ratio, val_ratio, test_ratio, directory):
    num_images = 0
    for file in os.listdir(directory): num_images += file.endswith(imgextension)

    print("Number of Images:\t", num_images)
    image_ids = list(range(num_images))
    random.shuffle(image_ids)

    ratio_sum = train_ratio + val_ratio + test_ratio
    n_train = int(num_images*train_ratio/ratio_sum)
    n_val = int(num_images*val_ratio/ratio_sum)
    # n_test = int(num_images*test_ratio/ratio_sum) #This is implicit anyway

    train_ids = image_ids[0:n_train]
    val_ids = image_ids[n_train+1:n_train+n_val]
    test_ids  = image_ids[n_train+n_val:-1]

    return train_ids, val_ids, test_ids

if __name__ == '__main__':
    print("Testing Folder Traversal and Image Extraction!")

    mono_training_data = {
        'images': '/media/bryce/4TB Seagate/Autonomous Vehicles Data/Cityscapes Data/leftImg8bit/train',
        'labels': '/media/bryce/4TB Seagate/Autonomous Vehicles Data/Cityscapes Data/gtFine/train'
    }

    full_training_data = {
        'left_images': '/media/bryce/4TB Seagate/Autonomous Vehicles Data/Cityscapes Data/leftImg8bit/train',
        'right_images': '/media/bryce/4TB Seagate/Autonomous Vehicles Data/Cityscapes Data/rightImg8bit/train',
        'mask': '/media/bryce/4TB Seagate/Autonomous Vehicles Data/Cityscapes Data/gtFine/train',
        'disparity': '/media/bryce/4TB Seagate/Autonomous Vehicles Data/Cityscapes Data/disparity/train'
    }

    test_dset = CityScapesDataset(full_training_data, crop_fraction=1)
    
    print(len(test_dset.l_img))
    print(len(test_dset.mask))

    import multiprocessing
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader

    batch_size = 10
    testLoader = DataLoader(test_dset, batch_size=batch_size, shuffle=True, num_workers=multiprocessing.cpu_count())
    image, mask = next(iter(testLoader))

    image = image[0].numpy()
    mask = mask[1].numpy()

    for i in range(batch_size):
        plt.subplot(121)
        plt.imshow(np.moveaxis(image[i,0:3,:,:],0,2))    

        plt.subplot(122)
        plt.imshow(mask[i,:,:])

        plt.show()
    
    # classes = {}
    # for i in range(batch_size):
    #     # # Get class for each individual pixel
    #     # for j in range(mask.shape[1]):
    #     #     for k in range(mask.shape[2]):
    #     #         class_id = mask[i,j,k]
    #     #         classes[class_id] = class_id

    #     plt.subplot(121)
    #     img_cpy = image[i,:,:,:]
    #     plt.imshow(np.moveaxis(img_cpy,0,2))
    #     plt.subplot(122)
    #     plt.imshow(mask[i,:,:])
    #     plt.show()

    # print(classes)
