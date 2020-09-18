
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from nnet_training.utilities.UnFlowLoss import flow_warp
from nnet_training.correlation_package.correlation import Correlation

from .hrnetv2 import get_seg_model
from .ocrnet import OCR_block, scale_as
from .pwcnet_modules import FlowEstimatorDense, ContextNetwork, pwc_conv

class OCRNetSF(nn.Module):
    """
    OCRNet with Segmentation + Optic Flow Output
    """
    def __init__(self, **kwargs):
        super(OCRNetSF, self).__init__()
        self.backbone = get_seg_model(**kwargs['hrnetv2_config'])
        self.ocr = OCR_block(self.backbone.high_level_ch, **kwargs['ocr_config'])

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

    def __str__(self):
        return "OCRNet"+str(self.backbone)+str(self.flow_estimator)+str(self.context_networks)

    def flow_forward(self, im1_pyr: List[torch.Tensor], im2_pyr: List[torch.Tensor],
                     final_scale: float):
        '''
        Auxillary forward method that does the flow prediction
        '''
        # output
        flows = []

        # init
        b_size, _, h_x1, w_x1, = im1_pyr[0].size()
        flow = im1_pyr[0].new_zeros((b_size, 2, h_x1, w_x1)).float()

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

    def forward(self, im1_rgb: torch.Tensor, im2_rgb: torch.Tensor, consistency=True):
        flows = {}
        segs = {}

        # Backbone Forward pass on image 1 and 2
        high_level_features, im1_pyr = self.backbone(im1_rgb)
        _, im2_pyr = self.backbone(im2_rgb)

        # Segmentation pass with image 1
        segs['fw'], aux_out, _ = self.ocr(high_level_features)

        # Flow pass with image 1
        scale_factor = im1_rgb.size()[-1] // segs['fw'].size()[-1]
        flows['fw'] = self.flow_forward(im1_pyr, im2_pyr, scale_factor)

        if consistency:
            # Flow pass with image 2
            flows['bw'] = self.flow_forward(im2_pyr, im1_pyr, scale_factor)

        aux_out = scale_as(aux_out, im1_rgb)
        segs['fw'] = scale_as(segs['fw'], im1_rgb)

        return flows, segs
