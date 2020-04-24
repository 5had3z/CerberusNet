#!/usr/bin/env python

__author__ = "Bryce Ferenczi"
__email__ = "bryce.ferenczi@monashmotorsport.com"

from fast_scnn import FastSCNN, Classifer
from dataset import MonoDataset, id_vec_generator
from segmentation_trainer import MonoSegmentationTrainer

import platform
import os
import multiprocessing
from pathlib import Path

import torch
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
from lossfunctions import MixSoftmaxCrossEntropyLoss, MixSoftmaxCrossEntropyOHEMLoss, FocalLoss

def TrainingFSCNN(directory, device, n_workers):
    train_ids, val_ids, test_ids = id_vec_generator(3,1,1,directory)

    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.ToPILImage(),
        transforms.Resize([1024,2048]),
        transforms.ToTensor()
    ])

    datasets = dict(
        Training=MonoDataset(directory, train_ids, img_transform),
        Validation=MonoDataset(directory, val_ids, img_transform),
        Testing=MonoDataset(directory, test_ids, img_transform)
    )

    dataloaders=dict(
        Training=DataLoader(datasets["Training"], batch_size=3, shuffle=True, num_workers=n_workers),
        Validation=DataLoader(datasets["Validation"], batch_size=3, shuffle=True, num_workers=n_workers),
        Testing=DataLoader(datasets["Testing"], batch_size=3, shuffle=True, num_workers=n_workers)
    )

    model_type = int(input("1:\tFresh Model\n2:\tPretrained Model\n3:\tRun Completely OG\nModel Type: "))
    
    #   Determine what mode to run FSCNN in
    if model_type == 1:
        fastModel = FastSCNN(3)
        filename = "Fresh"
        model_params = fastModel.parameters()
    elif model_type == 2:
        filename = "Pretrained"
        fastModel = FastSCNN(19)
        fastModel.load_state_dict(torch.load(Path.cwd() / 'Models' / 'fast_scnn_citys.pth', map_location=torch.device(device)))
        fastModel.classifier = Classifer(128, 3)
        model_params = [
            {"params": fastModel.learning_to_downsample.parameters(), 'lr':1e-3},
            {"params": fastModel.global_feature_extractor.parameters(), 'lr':1e-3},
            {"params": fastModel.feature_fusion.parameters(), 'lr':1e-3},
            ]
    elif model_type == 3:
        filename = "OG"
        fastModel = FastSCNN(19)
        fastModel.load_state_dict(torch.load(Path.cwd() / 'Models' / 'fast_scnn_citys.pth'))
        model_params = fastModel.parameters()
    else:
        raise AssertionError("Wrong INPUT!")

    #   Determine the Optimizer Used
    str_optim = int(input("\n1:\tSGD\n2:\tADAM\nOptimizer: "))
    if (str_optim == 1):
        optimizer = optim.SGD(model_params, lr=1e-2, momentum=0.9, weight_decay=1e-4)
        filename+="SGD"
    elif (str_optim == 2):
        optimizer = optim.Adam(model_params, lr=0.004)
        filename+="ADAM"
    else:
        raise AssertionError("Wrong INPUT!")
    
    #   Determine the Criterion Used
    str_lossfn = int(input("\n1:\tCross Entropy Loss"+
        "\n2:\tWeighted Cross Entropy Loss"+
        "\n3:\tMix Softmax Cross Entropy OHEM Loss"+
        "\n4:\tFocal Loss Function"
        "\nLoss Function: "))
    if (str_lossfn == 1):
        lossfn = nn.CrossEntropyLoss()
        filename+="CrossEntropyLoss"
    elif (str_lossfn == 2):
        lossfn = nn.CrossEntropyLoss(weight=torch.Tensor([0.1,5.0,5.0]).to(torch.device(device)))
        filename+="WeightedCrossEntropyLoss"
    elif (str_lossfn == 3):
        lossfn = MixSoftmaxCrossEntropyOHEMLoss(aux=False, aux_weight=0.4,ignore_index=-1).to(torch.device(device))
        filename+="CustomLossFn"
    elif (str_lossfn == 4):
        lossfn = FocalLoss(3)
        filename+="FocalLoss"
    else:
        raise AssertionError("Wrong INPUT!")

    #   Create the Model Trainer management class
    modeltrainer = MonoSegmentationTrainer(fastModel, optimizer, lossfn, dataloaders, savefile=filename, checkpoints=True)

    return modeltrainer

if __name__ == "__main__":
    if platform.system() == 'Windows':
        n_workers = 0
    else:
        n_workers = multiprocessing.cpu_count()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    directory = '/media'

    usr_input = int(input("1: FSCNN\nUser Input:"))
    if (usr_input == 1):
        modeltrainer = TrainingFSCNN(directory, device, n_workers)

    #   User Input Training loop
    while_cond = 1
    while(while_cond == 1):
        n_epochs = int(input("Number of Training Epochs: "))
        modeltrainer.train_model(n_epochs)

        usr_input = int(input("Show Training Statistics? (1/0): "))
        if (usr_input == 1):
            modeltrainer.plot_data()
    
        usr_input = int(input("Show Example Output? (1/0): "))
        if (usr_input == 1):
            modeltrainer.visualize_output()

        usr_input = int(input("Pass Specific Image? (1/0): "))
        if (usr_input == 1):
            usr_input = str(input("Enter Name: "))
            modeltrainer.custom_image(directory / usr_input)
        
        print("Current Learning Rate: ", modeltrainer.get_learning_rate)
        
        usr_input = int(input("New Learning Rate? (1/0): "))
        if (usr_input == 1):
            usr_input = float(input("New Learning Rate Value: "))
            modeltrainer.set_learning_rate(usr_input)

        while_cond = int(input("Continue Training? (1/0): "))