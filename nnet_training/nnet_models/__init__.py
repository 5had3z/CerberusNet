from .fast_scnn import FastSCNN
from .mono_segflow import MonoSFNet
from .mono_segflowdepth import MonoSFDNet
from .nnet_models import *
from .pwcnet import PWCNet
from .ocrnet import OCRNet, MscaleOCR

def get_model(model_args):
    """
    Returns pytorch model given dictionary pair of the model
    name and args/configuration
    """
    if model_args.name == "MonoSFNet":
        model = MonoSFNet(**model_args.args)
    elif model_args.name == "MonoSFDNet":
        model = MonoSFDNet(**model_args.args)
    elif model_args.name == "FastSCNN":
        model = FastSCNN(**model_args.args)
    elif model_args.name == "PWCNet":
        model = PWCNet(**model_args.args)
    elif model_args.name == "OCRNet":
        model = OCRNet(**model_args.args)
    elif model_args.name == "MscaleOCR":
        model = MscaleOCR(**model_args.args)
    else:
        raise NotImplementedError(model_args.name)

    return model
