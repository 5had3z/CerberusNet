from .cerberus import CerberusBase
from .fast_scnn import FastSCNN
from .pwcnet_sfd import MonoSFDNet
from .nnet_models import *
from .pwcnet import PWCNet
from .ocrnet import OCRNet, MscaleOCR
from .ocrnet_sfd import OCRNetSFD
from .detr_sfd import DetrNetSFD

def get_model(model_args):
    """
    Returns pytorch model given dictionary pair of the model
    name and args/configuration
    """
    model_map = {
        "MonoSFDNet"    : MonoSFDNet,
        "FastSCNN"      : FastSCNN,
        "PWCNet"        : PWCNet,
        "OCRNet"        : OCRNet,
        "MscaleOCR"     : MscaleOCR,
        "OCRNetSFD"     : OCRNetSFD,
        "DetrNetSFD"    : DetrNetSFD,
        "CerberusBase"  : CerberusBase
    }

    if model_map.get(model_args.type, False):
        model = model_map[model_args.type](**model_args.args)
    else:
        raise NotImplementedError(model_args.type)

    return model
