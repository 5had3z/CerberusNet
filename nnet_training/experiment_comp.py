#!/usr/bin/env python3

__author__ = "Bryce Ferenczi"
__email__ = "bryce.ferenczi@monashmotorsport.com"

import os
import argparse
from pathlib import Path

from nnet_training.utilities.metrics import MetricBaseClass

if __name__ == "__main__":
    ROOT_DIR = Path.cwd() / "torch_models"
    parser = argparse.ArgumentParser()
    parser.add_argument('-e1', '--experiment1', default='1deaad09314bced71e32dd9c0a2fb481')
    parser.add_argument('-e2', '--experiment2', default='03c05be48eb98c6b202ed0f26c6fe1b4')
    args = parser.parse_args()

    exper_1 = {"path": ROOT_DIR / args.experiment1}
    exper_2 = {"path": ROOT_DIR / args.experiment2}

    expers = [exper_1, exper_2]
    for exper in expers:
        for filename in os.listdir(exper["path"]):
            if filename.endswith(".hdf5"):
                m_type = filename.strip("_data.hdf5")
                exper[m_type] = MetricBaseClass(
                    savefile=filename, base_dir=exper["path"], main_metric="")


    for metric in exper_1:
        if metric != "path" and metric in exper_2.keys():
            exper_1[metric].plot_summary_data()
            exper_2[metric].plot_summary_data()

    input()
