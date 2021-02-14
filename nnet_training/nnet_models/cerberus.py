"""
Base class in which variants that need custom forwards and modules can inherit from.
"""
from typing import Dict
from pathlib import Path

import torch

from .pwcnet_sfd import PWCNetHead
from .ocrnet_sfd import OCRNetHead
from .detr_sfd import DetrSegmHead
from .pwcnet_sfd import DepthEstimator1
from .ocrnet_sfd import DepthHeadV1
from .hrnetv2 import HighResolutionNet
from .deeplab_panoptic import PanopticDeepLabDecoder
from .ocr_utils import scale_as

def get_backbone_enc(**kwargs):
    """
    Gets the backbone according to the name argument and accompanying configuration.
    Initialises any pretrained weights if given argument.
    """
    backb_map = {
        "HighResolutionNet" : HighResolutionNet
    }

    if kwargs['type'] in backb_map:
        backbone = backb_map[kwargs['type']](**kwargs['cfg'])
    else:
        raise NotImplementedError(f"{kwargs['type']} backbone does not exist")

    if kwargs.get('pretrained', False):
        backbone.init_weights(Path.cwd() / 'torch_models' / kwargs['pretrained'])

    return backbone

def get_segmentation_dec(in_channels, **kwargs):
    """
    Gets the segmentation decoder according to the name argument and accompanying configuration.\n
    Depth Networks must take number of channels in as the first argument.
    """
    seg_dec_map = {
        "OCRNetHead"                : OCRNetHead,
        "DetrSegmHead"              : DetrSegmHead,
        "PanopticDeepLabDecoder"    : PanopticDeepLabDecoder
    }

    if kwargs['type'] in seg_dec_map:
        seg_dec = seg_dec_map[kwargs['type']](in_channels, **kwargs['cfg'])
    else:
        raise NotImplementedError(f"{kwargs['type']} segmentation decoder type does not exist")

    return seg_dec

def get_depth_dec(in_channels, **kwargs):
    """
    Gets the depth decoder according to the name argument and accompanying configuration.\n
    Depth Networks must take number of channels in as the first argument.
    """
    depth_dec_map = {
        "DepthEstimator1"   : DepthEstimator1,
        "DepthHeadV1"       : DepthHeadV1
    }

    if kwargs['type'] in depth_dec_map:
        depth_dec = depth_dec_map[kwargs['type']](in_channels, **kwargs['cfg'])
    else:
        raise NotImplementedError(f"{kwargs['type']} depth decoder type does not exist")

    return depth_dec

def get_flow_dec(in_channels, **kwargs):
    """
    Gets the flow decoder according to the name argument and accompanying configuration.\n
    Flow Networks must take number of channels in as the first argument.
    """
    flow_dec_map = {
        "PWCNetHead" : PWCNetHead
    }

    if kwargs['type'] in flow_dec_map:
        flow_dec = flow_dec_map[kwargs['type']](in_channels, **kwargs['cfg'])
    else:
        raise NotImplementedError(f"{kwargs['type']} flow decoder type does not exist")

    return flow_dec

class CerberusBase(torch.nn.Module):
    """
    HRNetV2 with Segmentation + Optic Flow Output
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.modelname = kwargs['name']
        self.backbone = get_backbone_enc(**kwargs['backbone_config'])

        self.segmentation = get_segmentation_dec(
            self.backbone.output_ch, **kwargs['segmentation_config'])
        self.depth = get_depth_dec(self.backbone.output_ch, **kwargs['depth_config'])
        self.flow = get_flow_dec(self.backbone.output_ch, **kwargs['flow_config'])

    def forward(self, l_img: torch.Tensor, consistency=True, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Forward method for HRNetV2 with segmentation, flow and depth, returns dictionary \
        of outputs.\n During onnx export, consistency becomes the sequential image argument \
        because onnx export is not compatible with keyword aruments.
        """
        # Backbone Forward pass on first image
        enc_features = self.backbone(l_img)

        # Segmentation pass with first image
        forward = self.segmentation(enc_features)

        forward['seg'] = scale_as(forward['seg'], l_img)

        # Depth pass with first image
        forward['depth'] = self.depth(enc_features)

        # We must be ONNX exporting
        if isinstance(consistency, torch.Tensor):
            enc_features_bw = self.backbone(consistency)
            # Flow pass with image 1
            forward['flow'] = self.flow(enc_features)

        if kwargs.get('l_seq', False):
            # Backbone Forward pass on sequential image
            enc_features_bw = self.backbone(kwargs['l_seq'])

            # Flow pass with image 1 -> 2
            forward['flow'] = self.flow_forward(enc_features, enc_features_bw)

            if consistency:
                # Flow pass with image 2 -> 1
                forward['flow_b'] = self.flow_forward(enc_features_bw, enc_features)

            if kwargs.get('slam', False):
                # Estimate seg and depth
                forward['depth_b'] = self.depth(enc_features_bw)
                forward['seg_b'] = self.segmentation(enc_features_bw)

        return forward
