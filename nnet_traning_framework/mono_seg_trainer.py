#!/usr/bin/env python3

__author__ = "Bryce Ferenczi"
__email__ = "bryce.ferenczi@monashmotorsport.com"

import os
import time
import platform
import multiprocessing
import numpy as np
from pathlib import Path

import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
from lossfunctions import MixSoftmaxCrossEntropyLoss, MixSoftmaxCrossEntropyOHEMLoss, FocalLoss

from fast_scnn import FastSCNN, Classifer
from dataset import MonoDataset, id_vec_generator
from trainer_base_class import ModelTrainer

__all__ = ['MonoSegmentationTrainer', 'Init_Training_MonoFSCNN']

class MonoSegmentationTrainer(ModelTrainer):
    def __init__(self, model, optimizer, loss_fn, dataloaders, learning_rate=1e-4, savefile=None, checkpoints=True):
        '''
        Initialize the Model trainer giving it a nn.Model, nn.Optimizer and dataloaders as
        a dictionary with Training, Validation and Testing loaders
        '''
        super().__init__(model, optimizer, loss_fn, dataloaders, learning_rate, savefile, checkpoints)

    def _calculate_accuracy(self, fx, y):
        preds = torch.argmax(fx,dim=1,keepdim=True).cpu().detach().long()
        target = y.cpu().long()
        return self._intersectionAndUnion(preds, target, 19)

    def _intersectionAndUnion(self, prediction, target, numClass):
        """
        This function takes the prediction and label of a single image,
        returns intersection and union areas for each class
        To compute over many images do:
        for i in range(Nimages):
            (area_intersection[:,i], area_union[:,i]) = intersectionAndUnion(imPred[i], imLab[i])
        IoU = 1.0 * np.sum(area_intersection, axis=1) / np.sum(np.spacing(1)+area_union, axis=1)
        """
        # Remove classes from unlabeled pixels in gt image.
        # We should not penalize detections in unlabeled portions of the image.
        prediction = prediction * (target > 0).type(prediction.dtype)

        # Compute area intersection:
        intersection = prediction * (prediction == target)
        (area_intersection, _) = np.histogram(intersection, bins=numClass, range=(1, numClass))

        # Compute area union:
        (area_pred, _) = np.histogram(prediction, bins=numClass, range=(1, numClass))
        (area_lab, _) = np.histogram(target, bins=numClass, range=(1, numClass))
        area_union = area_pred + area_lab - area_intersection
        acc = np.divide(area_intersection, area_union, out=np.zeros_like(area_intersection), where=area_union!=0)
        print(area_intersection)
        print(area_union)
        print(acc)
        return acc

    def visualize_output(self):
        """
        Forward pass over a testing batch and displays the output
        """
        with torch.no_grad():
            self._model.eval()
            image, mask = next(iter(self._validation_loader))
            image = image.to(self._device)

            start_time = time.time()
            output = self._model(image)
            propagation_time = (time.time() - start_time)/self._validation_loader.batch_size

            pred = torch.argmax(output[0],dim=1,keepdim=True)
            for i in range(self._validation_loader.batch_size):
                plt.subplot(1,3,1)
                plt.imshow(np.moveaxis(image[i,:,:,:].cpu().numpy(),0,2))
                plt.xlabel("Base Image")
        
                plt.subplot(1,3,2)
                plt.imshow(mask[i,:,:])
                plt.xlabel("Ground Truth")
        
                plt.subplot(1,3,3)
                plt.imshow(pred.cpu().numpy()[i,0,:,:])
                plt.xlabel("Prediction")

                plt.suptitle("Propagation time: " + str(propagation_time))
                plt.show()

    def custom_image(self, filename):
        """
        Forward Pass on a single image
        """
        with torch.no_grad():
            self._model.eval()

            image = Image.open(filename)

            img_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.ToPILImage(),
                transforms.Resize([1024,2048]),
                transforms.ToTensor()
            ])
            image = img_transform(image).unsqueeze(0)
            image = image.to(self._device)

            start_time = time.time()
            output = self._model(image)
            propagation_time = (time.time() - start_time)

            pred = torch.argmax(output[0],dim=1,keepdim=True)

            plt.subplot(1,2,1)
            plt.imshow(np.moveaxis(image[0,:,:,:].cpu().numpy(),0,2))
            plt.xlabel("Base Image")
        
            plt.subplot(1,2,2)
            plt.imshow(pred.cpu().numpy()[0,0,:,:])
            plt.xlabel("Prediction")

            plt.suptitle("Propagation time: " + str(propagation_time))
            plt.show()

def Init_Training_MonoFSCNN():

    if platform.system() == 'Windows':
        n_workers = 0
    else:
        n_workers = multiprocessing.cpu_count()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    training_dir = {
        'images': '/media/bryce/4TB Seagate/Autonomous Vehicles Data/Cityscapes Data/leftImg8bit_trainvaltest/leftImg8bit/train',
        'labels': '/media/bryce/4TB Seagate/Autonomous Vehicles Data/Cityscapes Data/gtFine_trainvaltest/gtFine/train'
    }

    validation_dir = {
        'images': '/media/bryce/4TB Seagate/Autonomous Vehicles Data/Cityscapes Data/leftImg8bit_trainvaltest/leftImg8bit/val',
        'labels': '/media/bryce/4TB Seagate/Autonomous Vehicles Data/Cityscapes Data/gtFine_trainvaltest/gtFine/val'
    }

    testing_dir = {
        'images': '/media/bryce/4TB Seagate/Autonomous Vehicles Data/Cityscapes Data/leftImg8bit_trainvaltest/leftImg8bit/test',
        'labels': '/media/bryce/4TB Seagate/Autonomous Vehicles Data/Cityscapes Data/gtFine_trainvaltest/gtFine/test'
    }

    datasets = dict(
        Training=MonoDataset(training_dir),
        Validation=MonoDataset(validation_dir),
        Testing=MonoDataset(testing_dir)
    )

    dataloaders=dict(
        Training=DataLoader(datasets["Training"], batch_size=2, shuffle=False, num_workers=n_workers),
        Validation=DataLoader(datasets["Validation"], batch_size=2, shuffle=False, num_workers=n_workers),
        Testing=DataLoader(datasets["Testing"], batch_size=2, shuffle=False, num_workers=n_workers)
    )

    model_type = int(input(
        "1:\tFresh Model\n"+
        "2:\tPretrained Model\n"+
        "Model Type: "))
    
    #   Determine what mode to run FSCNN in
    if model_type == 1:
        fastModel = FastSCNN(19)
        filename = "Untrained"
        model_params = fastModel.parameters()
    elif model_type == 2:
        filename = "Pretrained"
        fastModel = FastSCNN(19)
        fastModel.load_state_dict(torch.load('torch_models/fast_scnn_citys.pth'))
        model_params = [
            {"params": fastModel.learning_to_downsample.parameters(), 'lr':1e-3},
            {"params": fastModel.global_feature_extractor.parameters(), 'lr':1e-3},
            {"params": fastModel.feature_fusion.parameters(), 'lr':1e-3},
            ]
    else:
        raise AssertionError("Wrong INPUT!")

    #   Determine the Optimizer Used
    str_optim = int(input(
        "1:\tSGD\n"+
        "2:\tADAM\n"+
        "Optimizer: "))
    
    if (str_optim == 1):
        optimizer = torch.optim.SGD(model_params, lr=1e-2, momentum=0.9, weight_decay=1e-4)
        filename+="_SGD"
    elif (str_optim == 2):
        optimizer = torch.optim.Adam(model_params, lr=0.004)
        filename+="_ADAM"
    else:
        raise AssertionError("Wrong INPUT!")
    
    #   Determine the Criterion Used
    str_lossfn = int(input(
        "\n1:\tCross Entropy Loss"+
        "\n2:\tWeighted Cross Entropy Loss"+
        "\n3:\tMix Softmax Cross Entropy OHEM Loss"+
        "\n4:\tFocal Loss Function"
        "\nLoss Function: "))
    
    if (str_lossfn == 1):
        lossfn = torch.nn.CrossEntropyLoss()
        filename+="_CE"
    elif (str_lossfn == 2):
        lossfn = torch.nn.CrossEntropyLoss(weight=torch.Tensor([0.1,5.0,5.0]).to(torch.device(device)))
        filename+="_WeightedCE"
    elif (str_lossfn == 3):
        lossfn = MixSoftmaxCrossEntropyOHEMLoss(aux=False, aux_weight=0.4,ignore_index=-1).to(torch.device(device))
        filename+="_CustomFn"
    elif (str_lossfn == 4):
        lossfn = FocalLoss(3)
        filename+="_Focal"
    else:
        raise AssertionError("Wrong INPUT!")

    #   Create the Model Trainer management class
    modeltrainer = MonoSegmentationTrainer(fastModel, optimizer, lossfn, dataloaders, savefile=filename, checkpoints=True)

    return modeltrainer

from chat_test_model import TestModel

if __name__ == "__main__":
    if platform.system() == 'Windows':
        n_workers = 0
    else:
        n_workers = multiprocessing.cpu_count()

    training_dir = {
        'images': '/media/bryce/4TB Seagate/Autonomous Vehicles Data/Cityscapes Data/leftImg8bit/train',
        'labels': '/media/bryce/4TB Seagate/Autonomous Vehicles Data/Cityscapes Data/gtFine/train'
    }

    validation_dir = {
        'images': '/media/bryce/4TB Seagate/Autonomous Vehicles Data/Cityscapes Data/leftImg8bit/val',
        'labels': '/media/bryce/4TB Seagate/Autonomous Vehicles Data/Cityscapes Data/gtFine/val'
    }

    testing_dir = {
        'images': '/media/bryce/4TB Seagate/Autonomous Vehicles Data/Cityscapes Data/leftImg8bit/test',
        'labels': '/media/bryce/4TB Seagate/Autonomous Vehicles Data/Cityscapes Data/gtFine/test'
    }

    datasets = dict(
        Training=MonoDataset(training_dir),
        Validation=MonoDataset(validation_dir),
        Testing=MonoDataset(testing_dir)
    )

    dataloaders=dict(
        Training=DataLoader(datasets["Training"], batch_size=2, shuffle=False, num_workers=n_workers),
        Validation=DataLoader(datasets["Validation"], batch_size=2, shuffle=False, num_workers=n_workers),
        Testing=DataLoader(datasets["Testing"], batch_size=2, shuffle=False, num_workers=n_workers)
    )

    filename = "Pretrained"
    # fastModel = TestModel(19)
    fastModel = FastSCNN(19)
    # fastModel.load_state_dict(torch.load('torch_models/fast_scnn_citys.pth'))
    optimizer = torch.optim.SGD(fastModel.parameters(), lr=0.05)
    # lossfn = MixSoftmaxCrossEntropyOHEMLoss()
    lossfn = torch.nn.CrossEntropyLoss(ignore_index=-1)

    modeltrainer = MonoSegmentationTrainer(fastModel, optimizer, lossfn, dataloaders, savefile=filename, checkpoints=True)
    # modeltrainer.visualize_output()
    modeltrainer.train_model(5)