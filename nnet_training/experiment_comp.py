#!/usr/bin/env python3.8

__author__ = "Bryce Ferenczi"
__email__ = "bryce.ferenczi@monashmotorsport.com"

import os
import json
import shutil
import hashlib
import argparse
from pathlib import Path
from typing import Dict, Union
from easydict import EasyDict

import matplotlib.pyplot as plt
from nnet_training.utilities.metrics import MetricBase, get_loggers

__all__ = ['compare_experiments']

def compare_experiments(experiment_dict: Dict[str, Union[EasyDict, MetricBase]]):
    """
    Given a list of experiments (Dictionary containing path), compares each of them to one another
    """
    # Gather each type of objective available (seg, depth, flow)
    objectives = set()
    for experiment in experiment_dict.values():
        for objective in experiment.keys():
            if objective not in ["root", "name", "config"]:
                objectives.add(objective)

    for objective in objectives:
        # Check to see if the objective is done and tracked
        # in an experiment and add it to the dictionary to be plotted
        experiment_data = {}
        for exper_hash, data in experiment_dict.items():
            if objective in data.keys():
                experiment_data[exper_hash] = data[objective].get_summary_data()

        sample_data = next(iter(experiment_data.values()))

        fig, axis = plt.subplots(1, len(sample_data), figsize=(18, 5))
        fig.suptitle(f"Comparision between {objective} Validation Results")

        # Plot each metric on a different subplot
        for idx, metric in enumerate(sample_data):
            for name, summary_dict in experiment_data.items():
                axis[idx].plot(summary_dict[metric]["Validation"], label=name)

            axis[idx].set_title(f'{metric}')
            axis[idx].set_xlabel('Epoch #')

        fig.legend(*fig.axes[-1].get_legend_handles_labels(), loc='lower center')
        plt.show(block=False)

    plt.show()

def segmentation_analysis(experiment_dict: Dict[str, Union[EasyDict, MetricBase]]):
    """
    Tool for analysing segmentation data
    """
    for data in experiment_dict.values():
        if 'seg' in data.keys():
            data['seg'].plot_classwise_data()

    plt.show()

def parse_expeiment_folder(root_dir: Path, experiment_dict: Dict[str, Union[EasyDict, MetricBase]]):
    """
    Parses each of the experiment subfolders and prints summaries.
    """
    for _, directories, _ in os.walk(root_dir):
        for directory in directories:
            experiment_dict[directory] = {}
            # Get the experiment config file
            for filename in os.listdir(root_dir / directory):
                if filename.endswith(".json"):
                    with open(root_dir / directory / filename) as json_file:
                        exper_config = EasyDict(json.load(json_file))
                        break

            # Ensure it is a valid new config file
            if 'logger_cfg' not in exper_config:
                directory = fix_experiment_cfg(
                    root_dir, directory, exper_config)

            # Get the experiment loggers
            loggers = get_loggers(exper_config['logger_cfg'], root_dir / directory)
            for exp_type, data in loggers.items():
                experiment_dict[directory][exp_type] = data

            # Set the config and root directory
            experiment_dict[directory]['config'] = exper_config
            experiment_dict[directory]['root'] = root_dir

def print_experiment_notes(experiment_dict: Dict[str, Union[EasyDict, MetricBase]]):
    """
    Prints the note for each of the experiments
    """
    for exper_hash, exper_data in experiment_dict.items():
        if 'note' in exper_data["config"]:
            print(f'{exper_hash}: {exper_data["config"]["note"]}')
        else:
            print(f'{exper_hash}: no note avaliable')

def print_experiment_perf(experiment_dict: Dict[str, Union[EasyDict, MetricBase]],
                          exper_type: str):
    """
    Prints basic performance of each experiment
    """
    for exper_hash, exper_data in experiment_dict.items():
        if exper_type in exper_data:
            metric_name, max_data = exper_data[exper_type].max_accuracy(main_metric=True)
            print(f'{exper_hash} {metric_name}: {max_data[1]}')

def fix_experiment_cfg(root_dir: Path, original_hash: str, config: EasyDict):
    """
    Fixes to newer version of config file
    """
    print(f"Fixing old config style from {original_hash}")

    # Parse all the files and add the objectives depending on the
    # performance tracking files that exist in the folder
    config['logger_cfg'] = {}
    metric_map = {
        'seg': lambda x: "Batch_IoU",
        'depth': lambda x: "Batch_RMSE_Linear",
        'flow': lambda x: "Batch_EPE" if x == 'Kitti' \
                        else ("Batch_SAD" if x == 'Cityscapes' \
                        else Warning(f"Invalid dataset {config.dataset.type}"))
    }
    for filename in os.listdir(root_dir / original_hash):
        if filename.endswith(".hdf5"):
            logger_type = filename.replace("_data.hdf5", "")
            assert logger_type in metric_map
            config['logger_cfg'][logger_type] = metric_map[logger_type](config.dataset.type)

        if filename.endswith(".json"):
            o_filename = filename

    # Delete any other depricated configs if they exist
    if 'amp_cfg' in config:
        del config['amp_cfg'], config.amp_cfg

    if 'trainer' in config:
        del config['trainer'], config.trainer

    new_hash = hashlib.md5(json.dumps(config).encode('utf-8')).hexdigest()

    # Create new folder if one doesn't already exist
    if not os.path.isdir(root_dir / new_hash):
        os.makedirs(root_dir / new_hash)
    else:
        Warning(f"path already exists while converting to new config"
                f"structure {original_hash} to {new_hash}")
        return original_hash

    # Write new config to new location
    with open(root_dir / new_hash / o_filename, 'w', encoding='utf-8') as json_file:
        json.dump(config, json_file, ensure_ascii=False, indent=4)

    # Move all the weights, statistics and any other files to new folder
    for filename in os.listdir(root_dir / original_hash):
        if not filename.endswith(".json"):
            shutil.move(root_dir / original_hash / filename, root_dir / new_hash / filename)

    # Now can remove old config file and folder
    os.remove(root_dir / original_hash / o_filename)
    os.rmdir(root_dir / original_hash)
    print(f"Moved contents from {original_hash} to {new_hash} complete!")

    return new_hash

def parse_experiment_list(experiment_dict: Dict[str, Union[EasyDict, MetricBase]]):
    """
    Requires the experiment root to already be populated
    """
    for exper_hash in experiment_dict.keys():
        assert 'root' in experiment_dict[exper_hash]
        exper_path = experiment_dict[exper_hash]["root"] / exper_hash

        # Get the config file for the experiment
        for filename in os.listdir(exper_path):
            if filename.endswith(".json"):
                with open(exper_path / filename) as json_file:
                    exper_config = EasyDict(json.load(json_file))
                    break

        # Ensure it is a valid new config file
        if 'logger_cfg' not in exper_config:
            exper_hash = fix_experiment_cfg(
                experiment_dict[exper_hash]["root"], exper_hash, exper_config)

        # Get the loggers for the experiment
        loggers = get_loggers(exper_config['logger_cfg'], exper_path)
        for exp_type, data in loggers.items():
            experiment_dict[exper_hash][exp_type] = data

        experiment_dict[exper_hash]['config'] = exper_config

if __name__ == "__main__":
    EXPER_DICTS = {}

    # ROOT_DIRS = [
    #     Path.cwd() / "torch_models",
    #     Path('/media/bryce/4TB Seagate/Autonomous Vehicles Data/Pytorch Models')
    # ]

    # for root_dir in ROOT_DIRS:
    #     parse_expeiment_folder(root_dir, EXPER_DICTS)

    # print_experiment_notes(EXPER_DICTS)
    # print_experiment_perf(EXPER_DICTS, 'flow')
    # print_experiment_perf(EXPER_DICTS, 'seg')

    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('-e', '--experiments', nargs='+',
                        default=['53b0d9580685d93958aa2468edb6c8d9',
                                 '6bb400efe627eec5e854a4a623550f5f',
                                 '69661449d4920dab67e521d8b173c43f',
                                 'fb04870fa815a7018a5a7b8cb8f8096d',
                                 '716f08f61521cc663b365e8dc62230e8'])

    for exper in PARSER.parse_args().experiments:
        EXPER_DICTS[exper] = {"root" : Path.cwd() / "torch_models"}

    parse_experiment_list(EXPER_DICTS)
    print_experiment_notes(EXPER_DICTS)

    compare_experiments(EXPER_DICTS)
    # segmentation_analysis(EXPER_DICTS)
