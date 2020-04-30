#!/usr/bin/env python3

__author__ = "Bryce Ferenczi"
__email__ = "bryce.ferenczi@monashmotorsport.com"

import torch
from fast_scnn import FastSCNN

if __name__ == "__main__":
    print("Testing ONNX Export of F-SCNN")

    print("Loading Model")
    device = torch.device( "cuda" if torch.cuda.is_available() else "cpu" )
    dummy_input = torch.randn(3, 3, 1024, 2048, device=device)
    model = FastSCNN(19).to(torch.device('cuda'))
    model.load_state_dict(torch.load('torch_models/fast_scnn_citys.pth', map_location=device))

    print("Exporting ONNX Engine")
    torch.onnx.export(model, dummy_input, "onnx_models/test_onnx_model.onnx", opset_version=11,
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)