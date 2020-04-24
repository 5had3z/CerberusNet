#!/usr/bin/env python

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
    def __init__(self, directories, id_vector, transform=torchvision.transforms.ToTensor()):
        images = []
        labels = []

        #Get all file names
        for image in os.listdir(directories['images']):
            if image.endswith(imgextension):
                images.append(image)

        for label in os.listdir(directories['labels']):
            if label.endswith(imgextension):
                labels.append(label)
        
        #Create dataset from specified ids
        self.images = [images[i] for i in id_vector]
        self.labels = [labels[i] for i in id_vector]

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
        image = Image.open(self.img_dir / self.images[idx])
        mask = Image.open(self.lbl_dir / self.labels[idx])

        #   Apply Defined Transformations
        image = self.transform(image)
        mask = self.transform(mask)

        mask = (mask*255).long().squeeze(0)

        return image, mask

class StereoDataset():
    def __init__(self, directories, id_vector=None, transform=torchvision.transforms.ToTensor()):
        images = []
        labels = []

        #Get all file names
        for image in os.listdir(directories['left_images']):
            if image.endswith(imgextension):
                images.append([image])

        for idx, image in enumerate(os.listdir(directories['right_images'])):
            if image.endswith(imgextension):
                images[idx].append(image)

        for label in os.listdir(directories['left_labels']):
            if label.endswith(imgextension):
                labels.append([label])
            
        for idx, label in enumerate(os.listdir(directories['right_labels'])):
            if label.endswith(imgextension):
                labels[idx].append(label)
        
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