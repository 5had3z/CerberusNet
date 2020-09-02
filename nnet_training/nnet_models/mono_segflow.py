from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from nnet_training.utilities.UnFlowLoss import flow_warp
from nnet_training.correlation_package.correlation import Correlation

from .pwcnet_modules import *
from .fast_scnn import Classifer
from .nnet_ops import LinearBottleneck, LinearBottleneckAGN

__all__ = ['MonoSFNet']

class SegmentationNet1(nn.Module):
    '''
    Module for extracting semantic segmentation from encoding pyramid
    '''
    def __init__(self, input_ch: List[int], num_classes=19, interm_ch=32, **kwargs):
        super(SegmentationNet1, self).__init__()

        if kwargs:
            args = kwargs['kwargs']
            self.classifier = Classifer(args.interm_ch, args.classes)

            if args.g_noise:
                bottleneck_module = LinearBottleneckAGN
            else:
                bottleneck_module = LinearBottleneck

            self.feature_fusion = nn.ModuleList(
                [bottleneck_module(input_ch[0], args.interm_ch, t=args.t, stride=args.stride)]
            )
            for ch_in in input_ch[1:]:
                self.feature_fusion.append(
                    bottleneck_module(ch_in+args.interm_ch, args.interm_ch,
                                      t=args.t, stride=args.stride)
                )
            self.scale_factor = 2 * args.stride

        else:
            self.classifier = Classifer(interm_ch, num_classes)
            self.feature_fusion = nn.ModuleList(
                [LinearBottleneck(input_ch[0], interm_ch, t=1, stride=1)]
            )
            for ch_in in input_ch[1:]:
                self.feature_fusion.append(
                    LinearBottleneck(ch_in+interm_ch, interm_ch, t=1, stride=1)
                )
            self.scale_factor = 2

    def __str__(self):
        return "_SegNet1"

    def forward(self, img_pyr):
        interm = self.feature_fusion[0](img_pyr[0])
        interm = nn.functional.interpolate(interm, scale_factor=self.scale_factor)

        for level, img in enumerate(img_pyr[1:], start=1):
            interm = self.feature_fusion[level](torch.cat([interm, img], 1))
            interm = nn.functional.interpolate(interm, scale_factor=self.scale_factor)

        return self.classifier(interm)

class MonoSFNet(nn.Module):
    '''
    Monocular image sequence to segmentation and optic flow
    '''
    def __init__(self, upsample=True, **kwargs):
        super(MonoSFNet, self).__init__()
        self.upsample = upsample
        self.output_level = 4
        self.scale_levels = [8, 4, 2, 1]

        if kwargs:
            feat_pyr_cfg = kwargs['kwargs']['feature_pyramid_extractor']
            num_chs = feat_pyr_cfg['args']['channels']
            if feat_pyr_cfg['type'] == 'FeatureExtractor':
                self.feature_pyramid_extractor = FeatureExtractor(num_chs)
            else:
                raise NotImplementedError(feat_pyr_cfg['type'])

            seg_cfg = kwargs['kwargs']['segmentation_network']
            if seg_cfg['type'] == 'SegmentationNet1':
                self.segmentation_network = SegmentationNet1(num_chs[:0:-1], kwargs=seg_cfg['args'])
            else:
                raise NotImplementedError(seg_cfg['type'])

            corr_args = kwargs['kwargs']['correlation_args']
            self.corr = Correlation(pad_size=corr_args.pad_size,
                                    kernel_size=corr_args.kernel_size,
                                    max_displacement=corr_args.max_displacement,
                                    stride1=corr_args.stride1,
                                    stride2=corr_args.stride2,
                                    corr_multiply=corr_args.corr_multiply)

            dim_corr = (corr_args.max_displacement * 2 + 1) ** 2
            num_ch_in = 32 + (dim_corr + 2)

            flow_est_cfg = kwargs['kwargs']['flow_est_network']
            if flow_est_cfg['type'] == 'FlowEstimatorDense':
                self.flow_estimator = FlowEstimatorDense(num_ch_in)
            else:
                raise NotImplementedError(flow_est_cfg['type'])

            ctx_net_cfg = kwargs['kwargs']['context_network']
            if ctx_net_cfg['type'] == 'ContextNetwork':
                self.context_networks = ContextNetwork(self.flow_estimator.feat_dim + 2)
            else:
                raise NotImplementedError(ctx_net_cfg['type'])

            out_1x1 = kwargs['kwargs']['1x1_conv_out']
            self.conv_1x1 = nn.ModuleList([pwc_conv(192, out_1x1, kernel_size=1,
                                                    stride=1, dilation=1),
                                           pwc_conv(128, out_1x1, kernel_size=1,
                                                    stride=1, dilation=1),
                                           pwc_conv(96, out_1x1, kernel_size=1,
                                                    stride=1, dilation=1),
                                           pwc_conv(64, out_1x1, kernel_size=1,
                                                    stride=1, dilation=1),
                                           pwc_conv(32, out_1x1, kernel_size=1,
                                                    stride=1, dilation=1)])

        else:
            num_chs = [3, 16, 32, 64, 96, 128, 192]
            self.feature_pyramid_extractor = FeatureExtractor(num_chs)
            self.segmentation_network = SegmentationNet1(num_chs[:0:-1], 19)

            search_range = 4
            self.corr = Correlation(pad_size=search_range, kernel_size=1,
                                    max_displacement=search_range, stride1=1,
                                    stride2=1, corr_multiply=1)

            dim_corr = (search_range * 2 + 1) ** 2
            num_ch_in = 32 + (dim_corr + 2)

            self.flow_estimator = FlowEstimatorDense(num_ch_in)

            self.context_networks = ContextNetwork(self.flow_estimator.feat_dim + 2)

            self.conv_1x1 = nn.ModuleList([pwc_conv(192, 32, kernel_size=1, stride=1, dilation=1),
                                           pwc_conv(128, 32, kernel_size=1, stride=1, dilation=1),
                                           pwc_conv(96, 32, kernel_size=1, stride=1, dilation=1),
                                           pwc_conv(64, 32, kernel_size=1, stride=1, dilation=1),
                                           pwc_conv(32, 32, kernel_size=1, stride=1, dilation=1)])

    def __str__(self):
        return "MonoSF" + str(self.segmentation_network) + str(self.feature_pyramid_extractor)\
            + str(self.flow_estimator) + str(self.context_networks)

    def get_scales(self):
        '''
        Returns the subsampling scales
        '''
        return self.scale_levels

    def aux_forward(self, im1_pyr, im2_pyr, seg_gt):
        '''
        Auxillary forward method that does the flow prediction
        @todo incorperate segmentation
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

            # upsampling or post-processing
            if level == self.output_level:
                break

        if self.upsample:
            flows = [F.interpolate(flow * 4, scale_factor=4,
                                   mode='bilinear', align_corners=True) for flow in flows]
        return flows[::-1]

    def forward(self, im1_rgb, im2_rgb, seg_gt=None, consistency=True):
        '''
        Forward method that returns the flow prediction and segmentation
        '''
        # outputs
        flows = {}

        im1_pyr = self.feature_pyramid_extractor(im1_rgb)
        im2_pyr = self.feature_pyramid_extractor(im2_rgb)

        if seg_gt is None:
            seg_gt = self.segmentation_network(im1_pyr)

        flows['flow_fw'] = self.aux_forward(im1_pyr, im2_pyr, seg_gt)
        if consistency:
            flows['flow_bw'] = self.aux_forward(im2_pyr, im1_pyr, seg_gt)

        return flows, seg_gt