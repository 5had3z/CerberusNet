#!/usr/bin/env python3

__author__ = "Bryce Ferenczi"
__email__ = "bryce.ferenczi@monashmotorsport.com"

import os
import argparse
from pathlib import Path
from typing import List, Dict

import matplotlib.pyplot as plt
from nnet_training.utilities.metrics import MetricBaseClass

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

            axis[idx].set_title(f'{metric} over Epochs')
            axis[idx].set_xlabel('Epoch #')

        fig.legend(*fig.axes[-1].get_legend_handles_labels(), loc='lower center')
        plt.show(block=False)

    plt.show()

if __name__ == "__main__":
    ROOT_DIR = Path.cwd() / "torch_models"
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('-e', '--experiments', nargs='+',
                        default=['a8769a65bb701f77bce4d80053418b20',
                                 'cdf66a3f794e9bd78247e48b4595ce82'])

    EXPER_LIST = []
    for exper in PARSER.parse_args().experiments:
        EXPER_LIST.append({
            "name" : exper,
            "path" : ROOT_DIR / exper
        })

    EXPER_LIST.append({
        "name" : 'a69da6bf96cd342f101294c4d93de26b',
        "path" : ROOT_DIR / 'a69da6bf96cd342f101294c4d93de26b'
    })

    compare_experiments(EXPER_LIST)