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

    for metric in exper_1:
        if metric not in ["path", "name"] and metric in exper_2.keys():

            data_1 = exper_1[metric].get_summary_data()
            data_2 = exper_2[metric].get_summary_data()

            fig, ax = plt.subplots(1, len(data_1), figsize=(18, 5))
            fig.suptitle(f"Comparision between {metric} Validation Results")

            for idx, data in enumerate(data_1):
                ax[idx].plot(data_1[data]["Validation"], label=exper_1["name"])
                ax[idx].plot(data_2[data]["Validation"], label=exper_2["name"])
                ax[idx].set_title(f'{data} over Epochs')
                ax[idx].set_xlabel('Epoch #')

            fig.axes[-1].set_label((exper_1["name"], exper_2["name"]))
            handles, labels = fig.axes[-1].get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper right')
            plt.show(block=False)

if __name__ == "__main__":
    ROOT_DIR = Path.cwd() / "torch_models"
    parser = argparse.ArgumentParser()
    parser.add_argument('-e1', '--experiment1', default='a8769a65bb701f77bce4d80053418b20')
    parser.add_argument('-e2', '--experiment2', default='cdf66a3f794e9bd78247e48b4595ce82')
    args = parser.parse_args()

    exper_1 = {
        "name" : args.experiment1,
        "path" : ROOT_DIR / args.experiment1
    }
    exper_2 = {
        "name" : args.experiment2,
        "path" : ROOT_DIR / args.experiment2
    }

    expers = [exper_1, exper_2]

    compare_experiments(expers)
    input()
