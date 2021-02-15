#!/usr/bin/env python3.8

"""
Main File for executing training loop
"""

__author__ = "Bryce Ferenczi"
__email__ = "bryce.ferenczi@monash.edu"

import os
import sys
import json
import argparse
import hashlib
from pathlib import Path
from shutil import copy
from easydict import EasyDict

import torch

from nnet_training.nnet_models import get_model
from nnet_training.datasets import get_dataset
from nnet_training.loss_functions import get_loss_function
from nnet_training.utilities.model_trainer import ModelTrainer

def initialise_training_network(config_json: EasyDict, train_path: Path) -> ModelTrainer:
    """
    Sets up the network and training configurations
    Returns initialised training framework class
    """
    datasets = get_dataset(config_json.dataset)

    model = get_model(config_json.model)

    loss_fns = get_loss_function(config_json.loss_functions)

    optim_map = {
        'Adam'  : torch.optim.Adam,
        'SGD'   : torch.optim.SGD,
        'AdamW' : torch.optim.AdamW
    }

    if optim_map.get(config_json.optimiser.type, False):
        optimiser = optim_map[config_json.optimiser.type](
            model.parameters(),
            **config_json.optimiser.args
        )
    else:
        raise NotImplementedError(config_json.optimiser.type)

    trainer = ModelTrainer(
        model=model, optimizer=optimiser, loss_fn=loss_fns,
        dataloaders=datasets, lr_cfg=config_json.lr_scheduler,
        basepath=train_path, logger_cfg=config_json.logger_cfg)

    return trainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='configs/DeeplabPanoptic_cs.json')
    parser.add_argument('-e', '--epochs', default=0)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = EasyDict(json.load(f))

    encoding = hashlib.md5(json.dumps(cfg).encode('utf-8'))
    training_path = Path.cwd() / "torch_models" / str(encoding.hexdigest())
    if not os.path.isdir(training_path):
        os.makedirs(training_path)

    print("Experiment # ", encoding.hexdigest())

    copy(args.config, training_path / os.path.basename(args.config))

    TRAINER = initialise_training_network(cfg, training_path)

    if args.epochs > 0:
        TRAINER.train_model(args.epochs)
        sys.exit()

    MAIN_MENU = {
        1 : lambda: TRAINER.train_model(int(input("Number of Training Epochs: "))),
        2 : TRAINER.plot_data,
        3 : TRAINER.visualize_output,
        4 : lambda: print("Current Base Learning Rate: " + str(TRAINER.get_learning_rate())),
        5 : lambda: TRAINER.set_learning_rate(float(input("New Learning Rate Value: "))),
        6 : sys.exit
    }

    #   User Input Training loop
    while True:
        USR_INPUT = int(input("Main Menu:\n1: Train Model\n2: Plot Statistics\n3: Visualise Output\
            \n4: Show LR\n5: Change LR\n6: Exit\nInput: "))

        FUNC = MAIN_MENU.get(USR_INPUT, None)

        if FUNC is not None:
            FUNC()
        else:
            print("Invalid Input")
