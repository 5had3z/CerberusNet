import torch
import torch.nn as nn
import torch.nn.functional as F

from nnet_training.utilities.UnFlowLoss import flow_warp
from nnet_training.correlation_package.correlation import Correlation

from .pwcnet_modules import *
from .fast_scnn import Classifer
from .nnet_ops import LinearBottleneck, LinearBottleneckAGN
from .mono_segflow import SegmentationNet1

__all__ = ['MonoSFDNet']

class DepthEstimator1(nn.Module):
    """
    Simple Prototype Depth Estimation Module
    """
    def __init__(self, in_channels: int, pre_out_ch: 32, **kwargs):
        super(DepthEstimator1, self).__init__()
        in_int_mean = (in_channels + pre_out_ch) // 2
        self.network = nn.Sequential(
            nn.Conv2d(in_channels, in_int_mean, 3),
            nn.Conv2d(in_int_mean, pre_out_ch, 3),
            nn.Conv2d(pre_out_ch, 1, 1),
            nn.ReLU(inplace=True)
        )

    def __str__(self):
        return "_DepthEst1"

    def forward(self, x: torch.Tensor):
        out = self.network(x)
        out = F.interpolate(out, size=x.size(), mode='nearest', align_corners=True)
        return out

class MonoSFDNet(nn.Module):
    '''
    Monocular image sequence to segmentation, depth and optic flow
    '''
    def __init__(self, upsample=True, **kwargs):
        super(MonoSFDNet, self).__init__()
        self.upsample = upsample
        self.output_level = 4
        self.scale_levels = [8, 4, 2, 1]

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
            corr_args = kwargs['correlation_args']
            search_range = kwargs['correlation_args']['max_displacement']
            self.corr = Correlation(**kwargs['correlation_args'])
        else:
            search_range = 4
            self.corr = Correlation(pad_size=search_range, kernel_size=1,
                                    max_displacement=search_range, stride1=1,
                                    stride2=1, corr_multiply=1)

        out_1x1 = 32 if '1x1_conv_out' not in kwargs else kwargs['1x1_conv_out']
        self.conv_1x1 = nn.ModuleList()

        for lvl in range(self.output_level):
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

    def __str__(self):
        return "MonoSF" + str(self.feature_pyramid_extractor) + str(self.segmentation_network)\
            + str(self.flow_estimator) + str(self.context_networks)

    def get_scales(self):
        '''
        Returns the subsampling scales
        '''
        return self.scale_levels

    def aux_forward(self, im1_pyr, im2_pyr, seg):
        '''
        Auxillary forward method that does the flow prediction
        @todo incorperate segmentation
        '''
        # output
        flows = []
        depths = []

        # init
        b_size, _, h_x1, w_x1, = im1_pyr[0].size()
        flow = im1_pyr[0].new_zeros((b_size, 2, h_x1, w_x1)).float()
        depth = im1_pyr[0].new_zeros((b_size, 1, h_x1, w_x1)).float()

        for level, (im1, im2) in enumerate(zip(im1_pyr, im2_pyr)):
            seg_resized = F.interpolate(seg.detach(), im1.size()[2:],
                                        mode='nearest', align_corners=True)
            # warping
            if level == 0:
                im2_warp = im2
            else:
                flow = F.interpolate(flow * 2, scale_factor=2,
                                     mode='bilinear', align_corners=True)
                depth = F.interpolate(depth, scale_factor=2,
                                      mode='nearest', align_corners=True)
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

            # concat and estimate depth
            depth_res = self.depth_estimator(
                torch.cat([seg_resized, im1_1by1, depth], dim=1))
            depth += depth_res

            depths.append(depth)

            # upsampling or post-processing
            if level == self.output_level:
                break

        if self.upsample:
            flows = [F.interpolate(flow * 4, scale_factor=4,
                                   mode='bilinear', align_corners=True) for flow in flows]

        return flows[::-1], depths[::-1]

    def forward(self, im1_rgb, im2_rgb, seg_gt=None, consistency=True):
        '''
        Forward method that returns the flow prediction and segmentation
        '''
        # outputs
        flows = {}
        depths = {}
        segs = {}

        im1_pyr = self.feature_pyramid_extractor(im1_rgb)
        im2_pyr = self.feature_pyramid_extractor(im2_rgb)

        segs['seg_fw'] = self.segmentation_network(im1_pyr)
        segs['seg_bw'] = self.segmentation_network(im2_pyr)

        if seg_gt is None:
            seg_gt = segs['seg_fw']

        flows['flow_fw'], depths['depth_fw'] = self.aux_forward(im1_pyr, im2_pyr, seg_gt)

        if consistency:
            flows['flow_bw'], depths['depth_bw'] = \
                self.aux_forward(im2_pyr, im1_pyr, segs['seg_bw'])

        return flows, depths, segs
