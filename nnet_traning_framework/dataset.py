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

class MonoDataset(torch.utils.data.Dataset):
    def __init__(self, directories, output_size=(512,256), crop_fraction=2, id_vector=None, transform=torchvision.transforms.ToTensor()):
        images = []
        labels = []

        #Get all file names
        for dirName, _, fileList in os.walk(directories['images']):
            for filename in fileList:
                if filename.endswith(imgextension):
                    imgpath = os.path.join(dirName, filename)
                    foldername = os.path.basename(os.path.dirname(imgpath))
                    lblname = filename.replace('leftImg8bit', 'gtFine_labelIds')
                    lblpath = os.path.join(directories['labels'], foldername, lblname)
                    if os.path.isfile(lblpath):
                        images.append(imgpath)
                        labels.append(lblpath)
                    else:
                        print("Could not corresponding label to image: ", imgpath)
        
        #Create dataset from specified ids
        if id_vector is not None:
            self.images = [images[i] for i in id_vector]
            self.labels = [labels[i] for i in id_vector]
        else:
            self.images = images
            self.labels = labels

        self.img_dir = directories['images']
        self.lbl_dir = directories['labels']
        self.transform = transform

        self.output_size = output_size

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
        #Read image and labels
        image = Image.open(self.images[idx]).convert('RGB')
        mask = Image.open(self.labels[idx])

        image, mask = self._sync_transform(image, mask)
        #   Apply Defined Transformations
        if self.transform is not None:
            image = self.transform(image)

        return image, mask

    def _sync_transform(self, img, mask):
        # random mirror
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        
        crop_h, crop_w = mask.size[0]/2, mask.size[1]/2
        crop_x, crop_y = random.randint(0, mask.size[1] - crop_w), random.randint(0, mask.size[0] - crop_h)

        img = img.crop((crop_y, crop_x, crop_y+crop_h, crop_x+crop_w))
        mask = mask.crop((crop_y, crop_x, crop_y+crop_h, crop_x+crop_w))

        img = img.resize(self.output_size, Image.BILINEAR)
        mask = mask.resize(self.output_size, Image.NEAREST)
        
        return self._img_transform(img), self._mask_transform(mask)

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

    def __len__(self):
        return len(self.images)

class StereoDataset(torch.utils.data.Dataset):
    def __init__(self, directories, id_vector=None, transform=torchvision.transforms.ToTensor()):
        images = []
        labels = []

        #Get all file names
        for dirName, _, fileList in os.walk(directories['left_images']):
            for filename in fileList:
                if filename.endswith(imgextension):
                    images.append([dirName + '/' + filename])

        for idx, (dirName, _, fileList) in enumerate(os.walk(directories['right_images'])):
            for filename in fileList:
                if filename.endswith(imgextension):
                    images[idx].append(dirName + '/' + filename)
        
        for dirName, _, fileList in os.walk(directories['left_labels']):
            for filename in fileList:
                if filename.endswith('gtFine_labelIds' + imgextension):
                    labels.append([dirName + '/' + filename])

        for idx, (dirName, _, fileList) in enumerate(os.walk(directories['disparity'])):
            for filename in fileList:
                if filename.endswith(imgextension):
                    labels[idx].append(dirName + '/' + filename)
        
        #Create dataset from specified ids
        if id_vector is not None:
            self.images = [images[i] for i in id_vector]
            self.labels = [labels[i] for i in id_vector]
        else:
            self.images = images
            self.labels = labels
        
        self.directories = directories
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        '''
        Returns an Image and Label Pair
        '''
        #Read image and labels
        l_image = Image.open(self.directories['left_images'] / self.images[idx])
        l_mask = Image.open(self.directories['left_labels'] / self.labels[idx])
        r_image = Image.open(self.directories['right_images'] / self.images[idx])
        r_mask = Image.open(self.directories['right_labels'] / self.labels[idx])

        #   Apply Defined Transformation
        l_image = self.transform(l_image)
        r_image = self.transform(r_image)
        out_img = torch.cat((l_image, r_image), 0)

        l_mask = self.transform(l_mask)
        r_mask = self.transform(r_mask)
        out_mask = torch.cat((l_mask, r_mask), 0)

        return out_img, out_mask

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

    stereo_training_data = {
        'left_images': '/media/bryce/4TB Seagate/Autonomous Vehicles Data/Cityscapes Data/leftImg8bit/train',
        'right_images': '/media/bryce/4TB Seagate/Autonomous Vehicles Data/Cityscapes Data/rightImg8bit/train',
        'left_labels': '/media/bryce/4TB Seagate/Autonomous Vehicles Data/Cityscapes Data/gtFine/train',
        'disparity': '/media/bryce/4TB Seagate/Autonomous Vehicles Data/Cityscapes Data/disparity/train'
    }

    test_mono = MonoDataset(mono_training_data)
    test_stereo = StereoDataset(stereo_training_data)
    
    print(len(test_mono.images))
    print(len(test_mono.labels))
    print(len(test_stereo.images))
    print(len(test_stereo.labels))

    import multiprocessing
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader

    batch_size = 16
    testLoader = DataLoader(test_mono, batch_size=batch_size, shuffle=True, num_workers=multiprocessing.cpu_count())
    image, mask = next(iter(testLoader))

    image = image.numpy()
    mask = mask.numpy()

    for i in range(batch_size):
        classes = {}
        for j in range(mask.shape[1]):
            for k in range(mask.shape[2]):
                class_id = mask[i,j,k]
                classes[class_id] = class_id

        plt.subplot(121)
        img_cpy = image[i,:,:,:]
        plt.imshow(np.moveaxis(img_cpy,0,2))
        plt.subplot(122)
        plt.imshow(mask[i,:,:])
        plt.show()