#!/usr/bin/env python3.8

__author__ = "Bryce Ferenczi"
__email__ = "bryce.ferenczi@monashmotorsport.com"

import os
import json
import hashlib
from pathlib import Path
import shutil
from typing import List, Dict, Union
from easydict import EasyDict

import matplotlib.pyplot as plt
from nnet_training.utilities.metrics import MetricBaseClass, SegmentationMetric, get_loggers

__all__ = ['compare_experiments']

def compare_experiments(experiment_list: Dict[str, Union[EasyDict, MetricBaseClass]]):
    """
    Given a list of experiments (Dictionary containing path), compares each of them to one another
    """
    for experiment in experiment_list:
        for filename in os.listdir(experiment["path"]):
            if filename.endswith(".hdf5"):
                m_type = filename.replace("_data.hdf5", "")
                experiment[m_type] = MetricBaseClass(
                    savefile=filename, base_dir=experiment["path"], main_metric="")

    # Gather each type of objective available (seg, depth, flow)
    objectives = set()
    for experiment in experiment_list:
        for objective in experiment.keys():
            if objective not in ["root", "name", "logger_cfg"]:
                objectives.add(objective)

    for objective in objectives:
        # Check to see if the objective is done and tracked
        # in an experiment and add it to the dictionary to be plotted
        experiment_data = {}
        for experiment in experiment_list:
            if objective in experiment.keys():
                experiment_data[experiment["name"]] = experiment[objective].get_summary_data()

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

def segmentation_analysis(experiment_list: Dict[str, Union[EasyDict, MetricBaseClass]]):
    """
    Tool for analysing segmentation data
    """
    for experiment in experiment_list:
        if "seg_data.hdf5" in os.listdir(experiment["path"]):
            experiment['data'] = SegmentationMetric(
                19, savefile="seg_data", base_dir=experiment["path"], main_metric="IoU")
        else:
            Warning(f'{experiment} does not have segmentation perf data')

    for experiment in experiment_list:
        experiment['data'].confusion_mat_summary()

    plt.show()

def parse_expeiments(root_dir: Path, experiment_dict: Dict[str, Union[EasyDict, MetricBaseClass]]):
    """
    Parses each of the experiment subfolders and prints summaries.
    """
    for _, directories, _ in os.walk(root_dir):
        for directory in directories:
            experiment_dict[directory] = {}
            for filename in os.listdir(root_dir / directory):
                if filename.endswith(".json"):
                    with open(root_dir / directory / filename) as json_file:
                        exper_config = EasyDict(json.load(json_file))
                        break

            if 'logger_cfg' not in exper_config:
                directory = fix_experiment_cfg(
                    root_dir, directory, exper_config)

            loggers = get_loggers(exper_config['logger_cfg'], root_dir / directory)
            for exp_type, data in loggers.items():
                experiment_dict[directory][exp_type] = data

            experiment_dict[directory]['config'] = exper_config
            experiment_dict[directory]['root'] = root_dir

def print_experiment_notes(experiment_dict: Dict[str, Union[EasyDict, MetricBaseClass]]):
    """
    Prints the note for each of the experiments
    """
    for exper_hash, exper_data in experiment_dict.items():
        if 'note' in exper_data["config"]:
            print(f'{exper_hash}: {exper_data["config"]["note"]}')
        else:
            print(f'{exper_hash}: no note avaliable')

def print_experiment_perf(experiment_dict: Dict[str, Union[EasyDict, MetricBaseClass]],
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
    for filename in os.listdir(root_dir / original_hash):
        if filename.endswith(".hdf5"):
            logger_type = filename.replace("_data.hdf5", "")
            if logger_type == 'seg':
                config['logger_cfg'][logger_type] = "Batch_IoU"
            elif logger_type == 'depth':
                config['logger_cfg'][logger_type] = "Batch_RMSE_Linear"
            elif logger_type == 'flow':
                if config.dataset.type == "Kitti":
                    config['logger_cfg'][logger_type] = "Batch_EPE"
                elif config.dataset.type == "Cityscapes":
                    config['logger_cfg'][logger_type] = "Batch_SAD"
                else:
                    Warning(f"Invalid dataset {config.dataset.type}")
            else:
                Warning(f"Invalid logger type detected {logger_type}")

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
        return

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

if __name__ == "__main__":
    EXPER_DICTS = {}

    ROOT_DIRS = [
        Path.cwd() / "torch_models",
        Path('/media/bryce/4TB Seagate/Autonomous Vehicles Data/Pytorch Models')
    ]

    for root_dir in ROOT_DIRS:
        parse_expeiments(root_dir, EXPER_DICTS)

    # print_experiment_notes(EXPER_DICTS)
    # print_experiment_perf(EXPER_DICTS, 'flow')
    print_experiment_perf(EXPER_DICTS, 'seg')

    # PARSER = argparse.ArgumentParser()
    # PARSER.add_argument('-e', '--experiments', nargs='+',
    #                     default=['53b0d9580685d93958aa2468edb6c8d9',
    #                              '6bb400efe627eec5e854a4a623550f5f',
    #                              '69661449d4920dab67e521d8b173c43f',
    #                              'fb04870fa815a7018a5a7b8cb8f8096d',
    #                              '716f08f61521cc663b365e8dc62230e8'])

    # EXPER_LIST = []
    # for exper in PARSER.parse_args().experiments:
    #     EXPER_LIST.append({
    #         "name" : exper,
    #         "path" : root
    #     })

    # compare_experiments(EXPER_LIST)
    # segmentation_analysis(EXPER_LIST)
