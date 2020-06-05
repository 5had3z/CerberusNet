#!/usr/bin/env python3

__author__ = "Bryce Ferenczi"
__email__ = "bryce.ferenczi@monashmotorsport.com"

import torch
from fast_scnn import FastSCNN
from chat_test_model import StereoDepthSeparated

if __name__ == "__main__":
    print("Testing ONNX Export")

    print("Loading Model")
    device = torch.device( "cuda" if torch.cuda.is_available() else "cpu" )
    dummy_input_l = torch.randn(3, 3, 1024, 2048, device=device)
    dummy_input_r = torch.randn(3, 3, 1024, 2048, device=device)
    model = StereoDepthSeparated().to(torch.device('cuda'))
    model.load_state_dict(torch.load('torch_models/CustomModel_InvHuber.pth', map_location=device)['model_state_dict'])

    print("Exporting ONNX Engine")
    torch.onnx.export(model, (dummy_input_l,dummy_input_r), "onnx_models/custom_depth.onnx", opset_version=11)
    # torch.onnx.export(model, dummy_input, "onnx_models/custom_depth.onnx", opset_version=11,
    #     operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
