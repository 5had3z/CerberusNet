"""
Loss functions for several objectives such as segmentation, depth and flow
"""

from typing import Dict

import torch

from .depth_losses import InvHuberLoss
from .depth_losses import InvHuberLossPyr
from .depth_losses import DepthAwareLoss
from .depth_losses import ScaleInvariantError

from .rmi import RMILoss
from .rmi import RMILossAux
from .rmi import MultiScaleRMILoss

from .seg_losses import FocalLoss2D
from .seg_losses import SegCrossEntropy

from .UnFlowLoss import unFlowLoss

from .detr_loss import DetrLoss

from .panoptic_loss import DeeplabPanopticLoss

def get_loss_function(loss_config) -> Dict[str, torch.nn.Module]:
    """
    Returns a dictionary of loss functions given a config
    """

    loss_map = {
        "FocalLoss2D"           : FocalLoss2D,
        "unFlowLoss"            : unFlowLoss,
        "InvHuberLoss"          : InvHuberLoss,
        "ScaleInvariantError"   : ScaleInvariantError,
        "DepthAwareLoss"        : DepthAwareLoss,
        "InvHuberLossPyr"       : InvHuberLossPyr,
        "RMILoss"               : RMILoss,
        "RMILossAux"            : RMILossAux,
        "MultiScaleRMILoss"     : MultiScaleRMILoss,
        "SegCrossEntropy"       : SegCrossEntropy,
        "DetrLoss"              : DetrLoss,
        "DeeplabPanopticLoss"   : DeeplabPanopticLoss
    }

    loss_fn_dict = {}
    for loss_fn in loss_config:
        if loss_map.get(loss_fn.function, False):
            loss_fn_dict[loss_fn.type] = loss_map[loss_fn.function](**loss_fn.args)
        else:
            raise NotImplementedError(loss_fn.function)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return {k:v.to(device) for k, v in loss_fn_dict.items()}
