#!/usr/bin/env python3

__author__ = "Bryce Ferenczi"
__email__ = "bryce.ferenczi@monashmotorsport.com"

import json, os, argparse, hashlib
from pathlib import Path
from shutil import copy
from easydict import EasyDict

import torch
from nnet_training.nnet_models import get_model

from nnet_training.utilities.KITTI import get_kitti_dataset
from nnet_training.utilities.CityScapes import get_cityscapse_dataset
from nnet_training.utilities.loss_functions import get_loss_function
from nnet_training.training_frameworks.trainer_base_class import get_trainer, ModelTrainer

def initialise_training_network(config_json: EasyDict, train_path: Path) -> ModelTrainer:
    """
    Sets up the network and training configurations
    Returns initialised training framework class
    """

    if config_json.dataset.type == "Kitti":
        datasets = get_kitti_dataset(config_json.dataset)
    elif config_json.dataset.type == "Cityscapes":
        datasets = get_cityscapse_dataset(config_json.dataset)
    else:
        raise NotImplementedError(config_json.dataset.type)

    model = get_model(config_json.model)

    loss_fns = get_loss_function(config_json.loss_functions)

    if config_json.optimiser.type in ['adam', 'Adam']:
        optimiser = torch.optim.Adam(
            model.parameters(),
            **config_json.optimiser.args
        )
    elif config_json.optimiser.type in ['sgd', 'SGD']:
        optimiser = torch.optim.SGD(
            model.parameters(),
            **config_json.optimiser.args
        )
    else:
        raise NotImplementedError(config_json.optimiser.type)

    trainer = get_trainer(config_json.trainer)(
        model=model, optim=optimiser, loss_fn=loss_fns,
        dataldr=datasets, lr_cfg=config_json.lr_scheduler,
        modelpath=train_path)

    return trainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='configs/Kitti_test.json')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = EasyDict(json.load(f))

    encoding = hashlib.md5(json.dumps(cfg).encode('utf-8'))
    training_path = Path.cwd() / "torch_models" / str(encoding.hexdigest())
    if not os.path.isdir(training_path):
        os.makedirs(training_path)

    copy(args.config, training_path / os.path.basename(args.config))

    TRAINER = initialise_training_network(cfg, training_path)

    MAIN_MENU = {
        1 : lambda: TRAINER.train_model(int(input("Number of Training Epochs: "))),
        2 : TRAINER.plot_data,
        3 : TRAINER.visualize_output,
        4 : lambda: print("Current Base Learning Rate: " + str(TRAINER.get_learning_rate())),
        5 : lambda: TRAINER.set_learning_rate(float(input("New Learning Rate Value: "))),
        6 : quit
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
