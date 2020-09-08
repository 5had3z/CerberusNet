import torch
import torch.nn as nn
import torch.nn.functional as F

from nnet_training.utilities.UnFlowLoss import flow_warp
from nnet_training.correlation_package.correlation import Correlation
from .pwcnet_modules import *

__all__ = ['PWCNet']

class PWCNet(nn.Module):
    '''
    PWCNet implementation from ARFlow-Net
    '''
    def __init__(self, lite=False, upsample=True):
        super(PWCNet, self).__init__()
        self.upsample = upsample
        self.output_level = 4
        self.scale_levels = [8, 4, 2, 1]

        num_chs = [3, 16, 32, 64, 96, 128, 192]
        self.feature_pyramid_extractor = FeatureExtractor(num_chs)

        search_range = 4
        self.corr = Correlation(pad_size=search_range, kernel_size=1,
                                max_displacement=search_range, stride1=1,
                                stride2=1, corr_multiply=1)

        dim_corr = (search_range * 2 + 1) ** 2
        num_ch_in = 32 + (dim_corr + 2)

        if lite:
            self.flow_estimator = FlowEstimatorLite(num_ch_in)
        else:
            self.flow_estimator = FlowEstimatorDense(num_ch_in)

        self.context_networks = ContextNetwork(self.flow_estimator.feat_dim + 2)

        self.conv_1x1 = nn.ModuleList([pwc_conv(192, 32, kernel_size=1, stride=1, dilation=1),
                                       pwc_conv(128, 32, kernel_size=1, stride=1, dilation=1),
                                       pwc_conv(96, 32, kernel_size=1, stride=1, dilation=1),
                                       pwc_conv(64, 32, kernel_size=1, stride=1, dilation=1),
                                       pwc_conv(32, 32, kernel_size=1, stride=1, dilation=1)])

    def __str__(self):
        return "PWCNet" + str(self.feature_pyramid_extractor)\
            + str(self.flow_estimator) + str(self.context_networks)

    def num_parameters(self):
        return sum(
            [p.data.nelement() if p.requires_grad else 0 for p in self.parameters()])

    def init_weights(self):
        for layer in self.named_modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

            elif isinstance(layer, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

    def get_scales(self):
        return self.scale_levels

    def aux_forward(self, im1_pyr, im2_pyr):
        # output
        flows = []
        
        # init
        b_size, _, h_x1, w_x1, = im1_pyr[0].size()
        flow = im1_pyr[0].new_zeros((b_size, 2, h_x1, w_x1)).float()

        for l, (im1, im2) in enumerate(zip(im1_pyr, im2_pyr)):
            # warping
            if l == 0:
                im2_warp = im2
            else:
                flow = F.interpolate(flow * 2, scale_factor=2,
                                        mode='bilinear', align_corners=True)
                im2_warp = flow_warp(im2, flow)

            # correlation
            out_corr = self.corr(im1, im2_warp)
            nn.functional.leaky_relu(out_corr, 0.1, inplace=True)

            # concat and estimate flow
            im1_1by1 = self.conv_1x1[l](im1)
            im1_intm, flow_res = self.flow_estimator(
                torch.cat([out_corr, im1_1by1, flow], dim=1))
            flow += flow_res

            flow_fine = self.context_networks(torch.cat([im1_intm, flow], dim=1))
            flow += flow_fine

            flows.append(flow)

            # upsampling or post-processing
            if l == self.output_level:
                break
        
        if self.upsample:
            flows = [F.interpolate(flow * 4, scale_factor=4,
                                   mode='bilinear', align_corners=True) for flow in flows]
        return flows[::-1]

    def forward(self, im1_rgb, im2_rgb, consistency=True):
        # outputs
        flows = {}

        im1_pyr = self.feature_pyramid_extractor(im1_rgb)
        im2_pyr = self.feature_pyramid_extractor(im2_rgb)

        flows['flow_fw'] = self.aux_forward(im1_pyr, im2_pyr)
        if consistency:
            flows['flow_bw'] = self.aux_forward(im2_pyr, im1_pyr)

        return flows
