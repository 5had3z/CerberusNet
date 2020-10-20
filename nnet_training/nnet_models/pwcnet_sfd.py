"""
Example Config Structure

"model" : {
    "name" : "MonoSFDNet",
    "args" : {
        "feature_pyramid_extractor" : {
            "type" : "FeatureExtractor",
            "args" : { "channels" : [3, 16, 32, 64, 96, 128, 192] }
        },
        "segmentation_network": {
            "type" : "SegmentationNet1",
            "args" : {
                "classes" : 19, "interm_ch" : 48,
                "stride" : 1, "t" : 1, "g_noise" : 0.2
            }
        },
        "depth_est_network" : {
            "type" : "DepthEstimator1",
            "args" : { "pre_out_ch" : 32 }
        },
        "correlation_args" : {
            "pad_size" : 4, "max_displacement" : 4,
            "kernel_size" : 1, "stride1" : 1,
            "stride2" : 1, "corr_multiply" : 1
        },
        "flow_est_network" : {
            "type" : "FlowEstimatorDense",
            "args" : {}
        },
        "context_network" : {
            "type" : "ContextNetwork",
            "args" : {}
        },
        "1x1_conv_out" : 32
    }
}

"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from nnet_training.loss_functions.UnFlowLoss import flow_warp
from nnet_training.correlation_package.correlation import Correlation

from .pwcnet_modules import *
from .fast_scnn import Classifer
from .nnet_ops import LinearBottleneck, LinearBottleneckAGN

__all__ = ['MonoSFDNet']

class SegmentationNet1(nn.Module):
    '''
    Module for extracting semantic segmentation from encoding pyramid
    '''
    def __init__(self, input_ch: List[int], classes=19, interm_ch=32, **kwargs):
        super(SegmentationNet1, self).__init__()
        self.classifier = Classifer(interm_ch, classes)

        if 'g_noise' in kwargs and kwargs['g_noise'] != 0.0:
            bottleneck_module = LinearBottleneckAGN
        else:
            bottleneck_module = LinearBottleneck

        t = 1 if 't' not in kwargs else kwargs['t']
        stride = 1 if 'stride' not in kwargs else kwargs['stride']
        sigma = 0 if 'g_noise' not in kwargs else kwargs['g_noise']

        self.feature_fusion = nn.ModuleList(
            [bottleneck_module(input_ch[0], interm_ch, t=t, stride=stride, sigma=sigma)]
        )
        for ch_in in input_ch[1:]:
            self.feature_fusion.append(
                bottleneck_module(ch_in+interm_ch, interm_ch, t=t, stride=stride, sigma=sigma)
            )

        self.scale_factor = 2 * stride

    def forward(self, img_pyr: List[torch.Tensor]) -> torch.Tensor:
        interm = self.feature_fusion[0](img_pyr[0])
        interm = nn.functional.interpolate(interm, scale_factor=self.scale_factor)

        for level, img in enumerate(img_pyr[1:], start=1):
            interm = self.feature_fusion[level](torch.cat([interm, img], 1))
            interm = nn.functional.interpolate(interm, scale_factor=self.scale_factor)

        return self.classifier(interm)

class DepthEstimator1(nn.Module):
    """
    Simple Prototype Depth Estimation Module
    """
    def __init__(self, in_channels: int, pre_out_ch: 32, **kwargs):
        super(DepthEstimator1, self).__init__()
        in_int_mean = (in_channels + pre_out_ch) // 2
        self.network = nn.Sequential(
            nn.Conv2d(in_channels, in_int_mean, 3),
            nn.Conv2d(in_int_mean, pre_out_ch, 1),
            nn.Conv2d(pre_out_ch, 1, 1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor):
        out = self.network(x)
        out = F.interpolate(out, size=tuple(x.size()[2:]), mode='nearest')
        return out

class MonoSFDNet(nn.Module):
    '''
    Monocular image sequence to segmentation, depth and optic flow
    '''
    def __init__(self, upsample=True, **kwargs):
        super(MonoSFDNet, self).__init__()
        self.upsample = upsample
        self.output_level = 4
        self.modelname = "MonoSFDNet"

        if 'feature_pyramid_extractor' in kwargs:
            feat_pyr_cfg = kwargs['feature_pyramid_extractor']
            num_chs = feat_pyr_cfg['args']['channels']
            if feat_pyr_cfg['type'] == 'FeatureExtractor':
                self.feature_pyramid_extractor = FeatureExtractor(num_chs)
            else:
                raise NotImplementedError(feat_pyr_cfg['type'])
        else:
            num_chs = [3, 16, 32, 64, 96, 128, 192]
            self.feature_pyramid_extractor = FeatureExtractor(num_chs)

        if 'segmentation_network' in kwargs:
            seg_cfg = kwargs['segmentation_network']
            n_classes = kwargs['segmentation_network']['args']['classes']
            if seg_cfg['type'] == 'SegmentationNet1':
                self.segmentation_network = SegmentationNet1(num_chs[:0:-1], **seg_cfg['args'])
            else:
                raise NotImplementedError(seg_cfg['type'])
        else:
            n_classes = 19
            self.segmentation_network = SegmentationNet1(num_chs[:0:-1], 19)

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

        for lvl in range(self.output_level+1):
            self.conv_1x1.append(
                pwc_conv(num_chs[-(lvl+1)], out_1x1, kernel_size=1, stride=1, dilation=1)
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

        num_ch_in = out_1x1 + n_classes + 1 # 1x1 conv, segmentaiton, previous depth
        if 'depth_est_network' in kwargs:
            depth_cfg = kwargs['depth_est_network']
            if depth_cfg['type'] == 'DepthEstimator1':
                self.depth_estimator = DepthEstimator1(num_ch_in, **depth_cfg['args'])
            else:
                raise NotImplementedError(depth_cfg['type'])
        else:
            self.depth_estimator = DepthEstimator1(num_ch_in, 32)

        if 'context_network' in kwargs:
            if kwargs['context_network']['type'] == 'ContextNetwork':
                self.context_networks = ContextNetwork(self.flow_estimator.feat_dim + 2)
            else:
                raise NotImplementedError(kwargs['context_network']['type'])
        else:
            self.context_networks = ContextNetwork(self.flow_estimator.feat_dim + 2)

    def flow_forward(self, im1_pyr: List[torch.Tensor],
                     im2_pyr: List[torch.Tensor]) -> List[torch.Tensor]:
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
                im2_warp = flow_warp(im2, flow).type(im1.dtype)

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

            # upsampling or post-processing
            if level == self.output_level:
                break

        if self.upsample:
            flows = [F.interpolate(flow * 4, scale_factor=4, mode='bilinear',
                                   align_corners=True) for flow in flows]

        return flows[::-1]

    def depth_forward(self, feat_pyr, seg):
        depths = []

        # init
        b_size, _, h_x1, w_x1, = feat_pyr[0].size()
        depth = feat_pyr[0].new_zeros((b_size, 1, h_x1, w_x1)).float()

        for level, enc_feat in enumerate(feat_pyr):
            enc_1by1 = self.conv_1x1[level](enc_feat)

            # concat and estimate depth
            new_size = tuple(enc_feat.size()[2:])
            seg_resized = F.interpolate(seg, size=new_size, mode='nearest')
            depth = F.interpolate(depth, size=new_size, mode='nearest')

            depth = self.depth_estimator(
                torch.cat([seg_resized, enc_1by1, depth], dim=1))

            depths.append(depth)

            # upsampling or post-processing
            if level == self.output_level:
                break

        if self.upsample:
            depths = [F.interpolate(depth, scale_factor=4, mode='nearest') for depth in depths]

        return depths[::-1]

    def forward(self, l_img: torch.Tensor, consistency=True, **kwargs):
        '''
        Forward method that returns the flow prediction and segmentation
        '''
        # outputs
        preds = {}

        im1_pyr = self.feature_pyramid_extractor(l_img)
        preds['seg'] = self.segmentation_network(im1_pyr)

        # I'll revisit using GT, will have to make a
        # new tensor and cat for each class and give them
        # each a magnitude that is typical of output
        # if 'seg' not in kwargs or not self.training:
        #     seg_gt = preds['seg']
        # else:
        #     seg_gt = kwargs['seg']

        preds['depth'] = self.depth_forward(im1_pyr, preds['seg'])

        if 'l_seq' in kwargs:
            im2_pyr = self.feature_pyramid_extractor(kwargs['l_seq'])

            preds['seg_b'] = self.segmentation_network(im2_pyr)

            preds['flow'] = self.flow_forward(im1_pyr, im2_pyr)

            if consistency:
                preds['flow_b'] = self.flow_forward(im2_pyr, im1_pyr)

        return preds
