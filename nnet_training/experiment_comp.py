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

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from nnet_training.utilities.metrics import MetricBase, get_loggers
from nnet_training.utilities.cityscapes_labels import trainId2name

STATISTIC_2_TYPE = {
    "Batch_IoU": "seg",
    "Batch_RMSE_Linear": "depth",
    "Batch_RMSE_Log": "depth",
    "Batch_EPE": "flow",
    "Batch_SAD": "flow"
}

def epoch_summary_comparison(experiment_dict: Dict[str, Union[EasyDict, MetricBase]]):
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
                n_samples = data[objective].get_n_samples('validation')

        sample_data = next(iter(experiment_data.values()))

        fig, axis = plt.subplots(1, len(sample_data), figsize=(18, 5))
        fig.suptitle(f"Comparision between {objective} Validation Results over Epochs")

        # Plot each metric on a different subplot
        for idx, metric in enumerate(sample_data):
            for name, summary_dict in experiment_data.items():
                data_mean = summary_dict[metric]["Validation_Mean"]
                data_conf = stats.t.ppf(0.95, n_samples-1) * \
                    summary_dict[metric]["Validation_Variance"] / np.sqrt(n_samples)
                axis[idx].plot(data_mean, label=experiment_dict[name]['config'].note)
                axis[idx].fill_between(
                    np.arange(0, data_mean.shape[0]),
                    data_mean - data_conf,
                    data_mean + data_conf,
                    alpha=0.2)

            axis[idx].set_title(f'{metric}')

        fig.legend(*fig.axes[-1].get_legend_handles_labels(),
                   loc='lower center', ncol=min(len(experiment_data), 4))
        plt.tight_layout()

    plt.show()

def segmentation_comparison(experiment_dict: Dict[str, Union[EasyDict, MetricBase]]):
    """
    Tool for analysing segmentation data
    """
    experiment_data = {}
    for exper_hash, data in experiment_dict.items():
        if 'seg' in data.keys():
            experiment_data[exper_hash], _ = data['seg'].conf_summary_data()

    plots = {
        "iou" : plt.subplots(3, 19//3+1, figsize=(18, 5)),
        "recall" : plt.subplots(3, 19//3+1, figsize=(18, 5)),
        "precision" : plt.subplots(3, 19//3+1, figsize=(18, 5))
    }

    for statistic, (fig, axis) in plots.items():
        for exper_hash, summary_dict in experiment_data.items():
            for idx in range(19):
                axis[idx%3][idx//3].plot(
                    summary_dict[statistic][:, idx],
                    label=experiment_dict[exper_hash]['config'].note)

    for statistic, (fig, axis) in plots.items():
        for idx in range(19):
            axis[idx%3][idx//3].set_title(f'{trainId2name[idx]}')

        fig.legend(*fig.axes[0].get_legend_handles_labels(),
                   loc='lower center', ncol=min(len(experiment_data), 4))
        fig.suptitle(f"Class {statistic} validation comparision over Epochs")

    plt.tight_layout()
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

def print_experiment_perf_main(experiment_dict: Dict[str, Union[EasyDict, MetricBase]],
                               exper_type: str):
    """
    Prints expected performance of each experiment on an objective
    """
    for exper_hash, exper_data in experiment_dict.items():
        if exper_type in exper_data:
            max_data = exper_data[exper_type].max_accuracy(main_metric=True)
            print(f'{exper_hash} {exper_data[exper_type].main_metric}: {max_data[1]}')

def print_experiment_perf_all(experiment_dict: Dict[str, Union[EasyDict, MetricBase]],
                              exper_hash: str):
    """
    Prints all performance of an experiment with 95% confidence interval
    """
    print(f"\nPerformance in metrics for {experiment_dict[exper_hash]['config'].note}")
    for obj_type, obj_data in experiment_dict[exper_hash].items():
        if obj_type in ['seg', 'depth', 'flow']:
            print(f'-----------{obj_type}-----------')
            max_data = obj_data.max_accuracy(main_metric=False)
            n_samples = obj_data.get_n_samples('validation')
            for statistic, data in max_data.items():
                conf_95 = stats.t.ppf(0.95, n_samples-1) * data[1] / np.sqrt(n_samples)
                print(f'{statistic}: mean {data[0]:.3f} +/- {conf_95:.3f}, '
                      f'[{data[0] - conf_95:.3f}, {data[0] + conf_95:.3f}]')

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

def final_accuracy_comparison(experiment_dict: Dict[str, Union[EasyDict, MetricBase]],
                              statistic: str):
    """
    Barplot that depicts the accuracy against each other.
    """
    all_data = {}
    exper_type = STATISTIC_2_TYPE[statistic]
    for exper_hash, exper_data in experiment_dict.items():
        if exper_type in exper_data:
            max_data = exper_data[exper_type].max_accuracy(main_metric=False)
            if statistic in max_data and 0.0 < max_data[statistic][0] < 100.0:
                all_data[exper_hash] = max_data[statistic]

    all_data = dict(sorted(all_data.items(), key=lambda x: x[1]))
    plt.bar(list(all_data.keys()), all_data.values())
    for index, value in enumerate(all_data.values()):
        plt.text(value, index, str(value))

    plt.xticks(rotation=10)
    plt.show()

def significance_test(experiment_dict: Dict[str, Dict[str, Union[EasyDict, MetricBase]]],
                      null_hyp: str, statistic: str):
    """
    Prints a list of p-value test results for a base experiment and list of alternat experiments.
    """
    exper_type = STATISTIC_2_TYPE[statistic]
    null_data = experiment_dict[null_hyp][exper_type].get_epoch_data(statistic=statistic)

    print(f"\nNull Hypothesis: {experiment_dict[null_hyp]['config'].note}, Statistic {statistic}")
    for exper_hash, experiment_data in experiment_dict.items():
        if exper_hash != null_hyp and exper_type in experiment_data:
            comp_data = experiment_data[exper_type].get_epoch_data(statistic=statistic)
            _, p_value = stats.ttest_ind(
                null_data, comp_data, equal_var=False, nan_policy='omit')

            if p_value.shape[0] > 1:
                print(f"{experiment_data['config'].note}: average p-value {p_value.mean():.4f}")
            else:
                print(f"{experiment_data['config'].note}: p-value {p_value.mean():.4f}")

def arg_parse_list():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--experiments', nargs='+',
                        default=['9df8e9e62963906fd052a8ebdd611fbe',
                                 '0ad32ead4e944e1165ac1981f0c159f9'])

    experiment_dicts = {}

    ## Parsing args
    for exper in parser.parse_args().experiments:
        experiment_dicts[exper] = {"root" : Path.cwd() / "torch_models"}

    parse_experiment_list(experiment_dicts)

    return experiment_dicts

def parse_folders():
    experiment_dicts = {}

    ## Parsing Folders
    root_dirs = [
        Path.cwd() / "torch_models",
        # Path('/media/bryce/4TB Seagate/Autonomous Vehicles Data/Pytorch Models')
    ]

    for rdir in root_dirs:
        parse_expeiment_folder(rdir, experiment_dicts)

    return experiment_dicts

if __name__ == "__main__":
    # EXPER_DICTS = parse_folders()
    EXPER_DICTS = arg_parse_list()

    # print_experiment_perf_main(EXPER_DICTS, 'flow')
    # print_experiment_perf_main(EXPER_DICTS, 'seg')

    print_experiment_notes(EXPER_DICTS)

    # significance_test(
    #     EXPER_DICTS, 'fe33b0d75016b2c04f9309743fcb28e2', 'Batch_EPE')

    # significance_test(
    #     EXPER_DICTS, 'fe33b0d75016b2c04f9309743fcb28e2', 'Batch_IoU')

    # significance_test(
    #     EXPER_DICTS, 'fe33b0d75016b2c04f9309743fcb28e2', 'Batch_RMSE_Linear')

    # print_experiment_perf_all(EXPER_DICTS, 'fe33b0d75016b2c04f9309743fcb28e2')

    epoch_summary_comparison(EXPER_DICTS)
    # final_accuracy_comparison(EXPER_DICTS, 'flow', 'Batch_EPE')
    # segmentation_comparison(EXPER_DICTS)
