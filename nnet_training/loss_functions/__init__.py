"""
Loss functions for several objectives such as segmentation, depth and flow
"""

from typing import Dict, List

import torch

from .depth_losses import InvHuberLoss, InvHuberLossPyr, DepthAwareLoss, ScaleInvariantError
from .UnFlowLoss import unFlowLoss
from .rmi import RMILoss, RMILossAux, MultiScaleRMILoss
from .seg_losses import FocalLoss2D, SegCrossEntropy
from .detr_loss import DetrLoss

def get_loss_function(loss_config) -> Dict[str, torch.nn.Module]:
    """
    Returns a dictionary of loss functions given a config
    """

    loss_fn_dict = {}
    for loss_fn in loss_config:
        if loss_fn['function'] == "FocalLoss2D":
            loss_fn_dict[loss_fn['type']] = FocalLoss2D(**loss_fn.args)
        elif loss_fn['function'] == "unFlowLoss":
            loss_fn_dict[loss_fn['type']] = unFlowLoss(**loss_fn.args)
        elif loss_fn['function'] == "InvHuberLoss":
            loss_fn_dict[loss_fn['type']] = InvHuberLoss(**loss_fn.args)
        elif loss_fn['function'] == "ScaleInvariantError":
            loss_fn_dict[loss_fn['type']] = ScaleInvariantError(**loss_fn.args)
        elif loss_fn['function'] == "DepthAwareLoss":
            loss_fn_dict[loss_fn['type']] = DepthAwareLoss(**loss_fn.args)
        elif loss_fn['function'] == "InvHuberLossPyr":
            loss_fn_dict[loss_fn['type']] = InvHuberLossPyr(**loss_fn.args)
        elif loss_fn['function'] == "RMILoss":
            loss_fn_dict[loss_fn['type']] = RMILoss(**loss_fn.args)
        elif loss_fn['function'] == "RMILossAux":
            loss_fn_dict[loss_fn['type']] = RMILossAux(**loss_fn.args)
        elif loss_fn['function'] == "MultiScaleRMILoss":
            loss_fn_dict[loss_fn['type']] = MultiScaleRMILoss(**loss_fn.args)
        elif loss_fn['function'] == "SegCrossEntropy":
            loss_fn_dict[loss_fn['type']] = SegCrossEntropy(**loss_fn.args)
        elif loss_fn['function'] == "DetrLoss":
            loss_fn_dict[loss_fn['type']] = DetrLoss(**loss_fn.args)
        else:
            raise NotImplementedError(loss_fn['function'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return {k:v.to(device) for k, v in loss_fn_dict.items()}
