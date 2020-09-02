#!/usr/bin/env python3

__author__ = "Bryce Ferenczi"
__email__ = "bryce.ferenczi@monashmotorsport.com"

import json, os, argparse, hashlib
from pathlib import Path
from shutil import copy
from easydict import EasyDict

import torch
from nnet_training.nnet_models import get_model

from nnet_training.utilities.dataset import get_cityscapse_dataset
from nnet_training.utilities.loss_functions import get_loss_function
from nnet_training.training_frameworks.trainer_base_class import get_trainer, ModelTrainer

def initialise_training_network(config_json: EasyDict, train_path: Path) -> ModelTrainer:
    """
    Sets up the network and training configurations
    Returns initialised training framework class
    """

    datasets = get_cityscapse_dataset(config_json.dataset)
    model = get_model(config_json.model)

    loss_fns = get_loss_function(config_json.loss_functions)

    if config_json.optimiser.type in ['adam', 'Adam']:
        optimiser = torch.optim.Adam(
            model.parameters(),
            betas=config_json.optimiser.args.betas,
            weight_decay=config_json.optimiser.args.weight_decay
        )
    elif config_json.optimiser.type in ['sgd', 'SGD']:
        optimiser = torch.optim.SGD(
            model.parameters(),
            momentum=config_json.optimiser.args.momentum,
            weight_decay=config_json.optimiser.args.weight_decay
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
    parser.add_argument('-c', '--config', default='configs/MonoSF_gauss.json')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = EasyDict(json.load(f))

    encoding = hashlib.md5(json.dumps(cfg).encode('utf-8'))
    training_path = Path.cwd() / "torch_models" / str(encoding.hexdigest())
    if not os.path.isdir(training_path):
        os.makedirs(training_path)

    copy(args.config, training_path / os.path.basename(args.config))

    TRAINER = initialise_training_network(cfg, training_path)

    #   User Input Training loop
    LOOP_COND = 1
    while LOOP_COND == 1:
        TRAINER.train_model(int(input("Number of Training Epochs: ")))

        if int(input("Show Training Statistics? (1/0): ")) == 1:
            # I should do this sometime
            TRAINER.plot_data()

        if int(input("Show Example Output? (1/0): ")) == 1:
            TRAINER.visualize_output()

        if int(input("Pass Specific Image? (1/0): ")) == 1:
            raise NotImplementedError
            # @todo but I don't really care
            # TRAINER.custom_image(str(input("Enter Path: ")))

        print("Current Base Learning Rate: ", TRAINER.get_learning_rate())

        if int(input("New Base Learning Rate? (1/0): ")) == 1:
            TRAINER.set_learning_rate(float(input("New Learning Rate Value: ")))

        LOOP_COND = int(input("Continue Training? (1/0): "))
