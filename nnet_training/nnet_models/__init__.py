from .fast_scnn import FastSCNN
from .mono_segflow import MonoSFNet
from .nnet_models import *
from .pwcnet import PWCNet

def get_model(model_args):
    """
    Returns pytorch model given dictionary pair of the model
    name and args/configuration
    """
    if model_args.name == "MonoSFNet":
        model = MonoSFNet(**model_args.args)
    else:
        raise NotImplementedError(model_args.name)

    return model
