#!/usr/bin/env python3

__author__ = "Bryce Ferenczi"
__email__ = "bryce.ferenczi@monashmotorsport.com"

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict
from easydict import EasyDict

import matplotlib.pyplot as plt
from nnet_training.utilities.metrics import MetricBaseClass, SegmentationMetric

__all__ = ['compare_experiments']

def compare_experiments(experiment_list: List[Dict[str, Path]]):
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
            if objective not in ["path", "name"]:
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

def segmentation_analysis(experiment_list: List[Dict[str, Path]]):
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

def parse_expeiments(root_dir: Path):
    """
    Parses each of the experiment subfolders and prints summaries.
    """
    for _, directories, _ in os.walk(root_dir):
        for directory in directories:
            for filename in os.listdir(root_dir / directory):
                if filename.endswith(".json"):
                    with open(root_dir / directory / filename) as f:
                        experiment_cfg = EasyDict(json.load(f))
                        break
            print(f'{directory}: {experiment_cfg.note}')

if __name__ == "__main__":
    ROOT_DIR = Path.cwd() / "torch_models"
    parse_expeiments(ROOT_DIR)
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('-e', '--experiments', nargs='+',
                        default=['2c6adea0aa0fcf1efbd88b2bea79597e',
                                 'ad491472e80220edc277d297ce95b063',
                                 '62c31076a3cf8565cd29775b76f7abad'])

    EXPER_LIST = []
    for exper in PARSER.parse_args().experiments:
        EXPER_LIST.append({
            "name" : exper,
            "path" : ROOT_DIR / exper
        })

    compare_experiments(EXPER_LIST)
    # segmentation_analysis(EXPER_LIST)
