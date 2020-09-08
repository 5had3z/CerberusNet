#!/usr/bin/env python3

__author__ = "Bryce Ferenczi"
__email__ = "bryce.ferenczi@monashmotorsport.com"

import torch
import nnet_training.nnet_models.nnet_models as nnet_models
from nnet_training.nnet_models.mono_segflow import MonoSFNet

if __name__ == "__main__":
    print("Testing ONNX Export")

    print("Loading Model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy_input_1 = torch.randn(1, 3, 256, 512, device=device)
    dummy_input_2 = torch.randn(1, 3, 256, 512, device=device)
    model = MonoSFNet().to(device)
    model.load_state_dict(torch.load('torch_models/MonoSF_FlwExt1_FlwEst1_CtxNet1_Adam_Focal_Uflw.pth', map_location=device)['model_state_dict'])

    print("Exporting ONNX Engine")
    # torch.onnx.export(model, (dummy_input_1, dummy_input_2), "onnx_models/segflow_test.onnx", opset_version=11)
    torch.onnx.export(model, (dummy_input_1, dummy_input_2), "onnx_models/custom_depth.onnx",
                      opset_version=11,
                      operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
