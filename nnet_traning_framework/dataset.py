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

class MonoDataset():
    def __init__(self, directories, id_vector=None, transform=torchvision.transforms.ToTensor()):
        images = []
        labels = []

        #Get all file names
        for dirName, _, fileList in os.walk(directories['images']):
            for filename in fileList:
                if filename.endswith(imgextension):
                    images.append(dirName + '/' + filename)

        for dirName, _, fileList in os.walk(directories['labels']):
            for filename in fileList:
                if filename.endswith('gtFine_labelIds' + imgextension):
                    labels.append(dirName + '/' + filename)
        
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

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        '''
        Returns an Image and Label Pair
        '''
        #Read image and labels
        image = Image.open(self.images[idx])
        mask = Image.open(self.labels[idx])

        #   Apply Defined Transformations
        image = self.transform(image)
        mask = self.transform(mask)

        mask = (mask * 255).long().squeeze(0)

        return image, mask

class StereoDataset():
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
        'images': '/media/bryce/4TB Seagate/Autonomous Vehicles Data/Cityscapes Data/leftImg8bit_trainvaltest/leftImg8bit/test',
        'labels': '/media/bryce/4TB Seagate/Autonomous Vehicles Data/Cityscapes Data/gtFine_trainvaltest/gtFine/test'
    }

    stereo_training_data = {
        'left_images': '/media/bryce/4TB Seagate/Autonomous Vehicles Data/Cityscapes Data/leftImg8bit_trainvaltest/leftImg8bit/train',
        'right_images': '/media/bryce/4TB Seagate/Autonomous Vehicles Data/Cityscapes Data/rightImg8bit_trainvaltest/rightImg8bit/train',
        'left_labels': '/media/bryce/4TB Seagate/Autonomous Vehicles Data/Cityscapes Data/gtFine_trainvaltest/gtFine/train',
        'disparity': '/media/bryce/4TB Seagate/Autonomous Vehicles Data/Cityscapes Data/gtFine_trainvaltest/gtFine/train'
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

    testLoader = DataLoader(test_mono, batch_size=2, shuffle=False, num_workers=multiprocessing.cpu_count())
    image, mask = next(iter(testLoader))

    for i in range(2):
        plt.subplot(121)
        plt.imshow(np.moveaxis(image[i,:,:,:].numpy(),0,2))
        plt.subplot(122)
        plt.imshow(mask[i,:,:])
        plt.show()