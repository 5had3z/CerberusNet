"""
Example Config Structure

"model" : {
    "name" : "DetrNetSFD",
    "args" : {
        "detr_config" : {
            "num_channels" : 512, "num_classes" : 19, "num_queries" : 100,
            "aux_loss" : False
        },
        "transformer_config" : {
            "dropout" : 0.1, "enc_layers" : 6, "dec_layers" : 6,
            "dim_feedforward" : 2048, "hidden_dim" : 256, "n_heads" : 8
        }
        "hrnetv2_config" : {
            "pretrained" : "hrnetv2_w48_imagenet_pretrained.pth",
            "STAGE1" : {
                "NUM_MODULES" : 1, "NUM_BRANCHES" : 1, "BLOCK": "BOTTLENECK",
                "NUM_BLOCKS" : [4], "NUM_CHANNELS" : [64],
                "FUSE_METHOD" : "SUM"
            },
            "STAGE2" : {
                "NUM_MODULES" : 1, "NUM_BRANCHES" : 2, "BLOCK": "BASIC",
                "NUM_BLOCKS" : [4, 4], "NUM_CHANNELS" : [48, 96],
                "FUSE_METHOD" : "SUM"
            },
            "STAGE3" : {
                "NUM_MODULES" : 4, "NUM_BRANCHES" : 3, "BLOCK": "BASIC",
                "NUM_BLOCKS" : [4, 4, 4], "NUM_CHANNELS" : [48, 96, 192],
                "FUSE_METHOD" : "SUM"
            },
            "STAGE4" : {
                "NUM_MODULES" : 3, "NUM_BRANCHES" : 4, "BLOCK": "BASIC",
                "NUM_BLOCKS" : [4, 4, 4, 4], "NUM_CHANNELS" : [48, 96, 192, 384],
                "FUSE_METHOD" : "SUM"
            }
        },
        "correlation_args" : {
            "pad_size" : 4, "max_displacement" : 4,
            "kernel_size" : 1, "stride1" : 1,
            "stride2" : 1, "corr_multiply" : 1
        },
        "flow_est_network" : {
            "type" : "FlowEstimatorLite",
            "args" : {}
        },
        "context_network" : {
            "type" : "ContextNetwork",
            "args" : {}
        },
        "1x1_conv_out" : 32,
        "depth_network" : {
            "type" : "DepthHeadV1",
            "args" : {"inter_ch" : [128, 32]}
        }
    }
}

"""

from typing import List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from nnet_training.loss_functions.UnFlowLoss import flow_warp
from nnet_training.correlation_package.correlation import Correlation

from .hrnetv2 import get_seg_model
from .pwcnet_modules import FlowEstimatorDense, FlowEstimatorLite, ContextNetwork, pwc_conv
from .ocrnet_sfd import DepthHeadV1, scale_as
from .detr.detr import DetrHead
from .detr.transformer import Transformer

class DetrNetSFD(nn.Module):
    """
    HRNetV2 with Segmentation + Optic Flow Output
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.modelname = "DetrNetSFD"

        self.backbone = get_seg_model(**kwargs['hrnetv2_config'])
        self.detr = DetrHead(
            transformer=Transformer(**kwargs['transformer_config']), **kwargs['detr_config'])

        if 'correlation_args' in kwargs:
            search_range = kwargs['correlation_args']['max_displacement']
            self.corr = Correlation(**kwargs['correlation_args'])
        else:
            search_range = 4
            self.corr = Correlation(pad_size=search_range, kernel_size=1,
                                    max_displacement=search_range, stride1=1,
                                    stride2=1, corr_multiply=1)

        out_1x1 = 32 if '1x1_conv_out' not in kwargs else kwargs['1x1_conv_out']
        self.conv_1x1 = nn.ModuleList()

        for channels in reversed(kwargs['hrnetv2_config']['STAGE4']['NUM_CHANNELS']):
            self.conv_1x1.append(
                pwc_conv(channels, out_1x1, kernel_size=1, stride=1, dilation=1)
            )

        dim_corr = (search_range * 2 + 1) ** 2
        num_ch_in = out_1x1 + dim_corr + 2 # 1x1 conv, correlation, previous flow
        if 'flow_est_network' in kwargs:
            if kwargs['flow_est_network']['type'] == 'FlowEstimatorDense':
                self.flow_estimator = FlowEstimatorDense(num_ch_in)
            elif kwargs['flow_est_network']['type'] == 'FlowEstimatorLite':
                self.flow_estimator = FlowEstimatorLite(num_ch_in)
            else:
                raise NotImplementedError(kwargs['flow_est_network']['type'])
        else:
            self.flow_estimator = FlowEstimatorDense(num_ch_in)

        if 'context_network' in kwargs:
            if kwargs['context_network']['type'] == 'ContextNetwork':
                self.context_networks = ContextNetwork(self.flow_estimator.feat_dim + 2)
            else:
                raise NotImplementedError(kwargs['context_network']['type'])
        else:
            self.context_networks = ContextNetwork(self.flow_estimator.feat_dim + 2)

        if 'depth_network' in kwargs:
            if kwargs['depth_network']['type'] == 'DepthHeadV1':
                self.depth_head = DepthHeadV1(self.backbone.high_level_ch,
                                              **kwargs['depth_network']['args'])
            else:
                raise NotImplementedError(kwargs['depth_network']['type'])
        else:
            self.depth_head = DepthHeadV1(self.backbone.high_level_ch, 32)

    def flow_forward(self, im1_pyr: List[torch.Tensor], im2_pyr: List[torch.Tensor],
                     final_scale: float):
        '''
        Auxillary forward method that does the flow prediction
        '''
        # output
        flows = []

        # init
        b_size, _, h_x1, w_x1, = im1_pyr[0].size()
        flow = im1_pyr[0].new_zeros((b_size, 2, h_x1, w_x1))

        for level, (im1, im2) in enumerate(zip(im1_pyr, im2_pyr)):
            # warping
            if level == 0:
                im2_warp = im2
            else:
                flow = F.interpolate(flow * 2, scale_factor=2,
                                     mode='bilinear', align_corners=True)
                im2_warp = flow_warp(im2, flow)

            # correlation
            out_corr = self.corr(im1, im2_warp)
            nn.functional.leaky_relu(out_corr, 0.1, inplace=True)

            # concat and estimate flow
            im1_1by1 = self.conv_1x1[level](im1)
            im1_intm, flow_res = self.flow_estimator(
                torch.cat([out_corr, im1_1by1, flow], dim=1))
            flow += flow_res

            flow_fine = self.context_networks(torch.cat([im1_intm, flow], dim=1))
            flow += flow_fine

            flows.append(flow)

        flows = [F.interpolate(flow * final_scale, scale_factor=final_scale,
                               mode='bilinear', align_corners=True) for flow in flows]

        return flows[::-1]

    def forward(self, l_img: torch.Tensor, consistency=True, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Forward method for HRNetV2 with segmentation, flow and depth, returns dictionary \
        of outputs.\n During onnx export, consistency becomes the sequential image argument \
        because onnx export is not compatible with keyword aruments.
        """
        forward = {}

        # Backbone Forward pass on image 1 and 2
        high_level_features, im1_pyr = self.backbone(l_img)

        # Segmentation pass with image 1
        forward['seg'], forward['seg_aux'], _ = self.detr(high_level_features)
        forward['seg_aux'] = scale_as(forward['seg_aux'], l_img)

        # Depth pass with image 1
        forward['depth'] = self.depth_head(high_level_features)

        # We must be ONNX exporting
        if isinstance(consistency, torch.Tensor):
            _, im2_pyr = self.backbone(consistency)
            # Flow pass with image 1
            scale_factor = l_img.size()[-1] // forward['seg'].size()[-1]
            forward['flow'] = self.flow_forward(im1_pyr, im2_pyr, scale_factor)[0]
            del forward['seg_aux']

        if 'l_seq' in kwargs:
            _, im2_pyr = self.backbone(kwargs['l_seq'])

            # Flow pass with image 1
            scale_factor = l_img.size()[-1] // forward['seg'].size()[-1]
            forward['flow'] = self.flow_forward(im1_pyr, im2_pyr, scale_factor)

            if consistency:
                # Flow pass with image 2
                forward['flow_b'] = self.flow_forward(im2_pyr, im1_pyr, scale_factor)

            if 'slam' in kwargs and kwargs['slam'] is True:
                # Another forward brah
                high_level_features, _ = self.backbone(kwargs['l_seq'])

                # Estimate seg and depth
                forward['depth_b'] = self.depth_head(high_level_features)
                forward['seg_b'], _, _ = self.ocr(high_level_features)

                # Rescale
                forward['seg_b'] = scale_as(forward['seg_b'], l_img)
                forward['depth_b'] = scale_as(forward['depth_b'], l_img)

        forward['seg'] = scale_as(forward['seg'], l_img)
        forward['depth'] = scale_as(forward['depth'], l_img)

        return forward
