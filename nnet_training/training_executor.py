#!/usr/bin/env python3

__author__ = "Bryce Ferenczi"
__email__ = "bryce.ferenczi@monashmotorsport.com"

import json, os, argparse, hashlib
from pathlib import Path
from easydict import EasyDict

from nnet_training.utilities.dataset import get_cityscapse_dataset
from nnet_training.training_frameworks.trainer_base_class import get_trainer, ModelTrainer

def initialise_training_network(config_json) -> ModelTrainer:
    """
    Sets up the network and training configurations
    Returns initialised training framework class
    """
    encoding = hashlib.md5(json.dumps(config_json).encode('utf-8'))

    if not os.path.isdir(Path.cwd() / "torch_models" / str(encoding.hexdigest())):
        os.makedirs(Path.cwd() / "torch_models" / str(encoding.hexdigest()))

    trainer = get_trainer(config_json.trainer)

    datasets = get_cityscapse_dataset(config_json.dataset)

    return trainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='configs/MonoSF_gauss.json')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = EasyDict(json.load(f))

    TRAINER = initialise_training_network(cfg)

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
