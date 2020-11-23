#!/usr/bin/env python3

__author__ = "Bryce Ferenczi"
__email__ = "bryce.ferenczi@monashmotorsport.com"

import os
import json
import hashlib
import argparse
from pathlib import Path
from easydict import EasyDict

import torch
from torch.onnx.symbolic_helper import parse_args

from nnet_training.nnet_models import get_model

@parse_args('v', 'v', 'i', 'i', 'i', 'i', 'i', 'i')
def correlation_op(g, input1, input2, pad_size, kernel_size,
                   max_displacement, stride1, stride2, corr_multiply):
    return g.op("cerberus::correlation", input1, input2, pad_size_i=pad_size,
                kernel_size_i=kernel_size, max_displacement_i=max_displacement,
                stride1_i=stride1, stride2_i=stride2, corr_multiply_i=corr_multiply)

@parse_args('v', 'v', 's', 's', 'b')
def grid_sample_op(g, input1, input2, mode, padding_mode, align_corners):
    return g.op("torch::grid_sampler", input1, input2, mode_s=mode,
                padding_mode_s=padding_mode, align_corners_i=align_corners)

def export_model(config: EasyDict, exp_path: Path) -> None:
    """
    Export model to ONNX format given config dictionary and path
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading Model")
    model = get_model(config.model).to(device)

    for filename in os.listdir(exp_path):
        if filename.endswith("_latest.pth"):
            modelweights = exp_path / filename

    model.load_state_dict(torch.load(modelweights, map_location=device)['model_state_dict'])

    torch.onnx.register_custom_op_symbolic('cerberus::correlation', correlation_op, 11)
    torch.onnx.register_custom_op_symbolic('::grid_sampler', grid_sample_op, 11)

    dim_h = config.dataset.augmentations.output_size[0]
    dim_w = config.dataset.augmentations.output_size[1]
    dummy_input_1 = torch.randn(1, 3, dim_h, dim_w, device=device)
    dummy_input_2 = torch.randn(1, 3, dim_h, dim_w, device=device)

    print("Exporting ONNX Engine")
    torch.onnx.export(
        model, (dummy_input_1, dummy_input_2),
        "onnx_models/export_test.onnx",
        opset_version=11)

if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('-c', '--config', default='configs/HRNetV2_sfd_kt.json')

    with open(PARSER.parse_args().config) as f:
        CONFIG = EasyDict(json.load(f))

    EXP_PATH = Path.cwd() / "torch_models" / \
        str(hashlib.md5(json.dumps(CONFIG).encode('utf-8')).hexdigest())

    export_model(CONFIG, EXP_PATH)

    print('Export Complete')
