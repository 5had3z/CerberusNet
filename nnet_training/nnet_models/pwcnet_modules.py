import torch
import torch.nn as nn
import torch.nn.functional as F

from nnet_training.loss_functions.UnFlowLoss import flow_warp
from nnet_training.correlation_package.correlation import Correlation

__all__ = ['pwc_conv', 'FeatureExtractor', 'FlowEstimatorDense', 'FlowEstimatorLite',
           'ContextNetwork']

def pwc_conv(in_planes, out_planes, kernel_size=3, stride=1, dilation=1, isReLU=True):
    if isReLU:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                      stride=stride, dilation=dilation,
                      padding=((kernel_size - 1) * dilation) // 2, bias=True),
            nn.LeakyReLU(0.1, inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                      stride=stride, dilation=dilation,
                      padding=((kernel_size - 1) * dilation) // 2, bias=True)
        )


class FeatureExtractor(nn.Module):
    def __init__(self, num_chs):
        super(FeatureExtractor, self).__init__()
        self.num_chs = num_chs
        self.convs = nn.ModuleList()

        for ch_in, ch_out in zip(num_chs[:-1], num_chs[1:]):
            layer = nn.Sequential(
                pwc_conv(ch_in, ch_out, stride=2),
                pwc_conv(ch_out, ch_out)
            )
            self.convs.append(layer)

    def forward(self, x):
        feature_pyramid = []
        for conv in self.convs:
            x = conv(x)
            feature_pyramid.append(x)

        return feature_pyramid[::-1]


class FlowEstimatorDense(nn.Module):
    def __init__(self, ch_in):
        super(FlowEstimatorDense, self).__init__()
        self.conv1 = pwc_conv(ch_in, 128)
        self.conv2 = pwc_conv(ch_in + 128, 128)
        self.conv3 = pwc_conv(ch_in + 256, 96)
        self.conv4 = pwc_conv(ch_in + 352, 64)
        self.conv5 = pwc_conv(ch_in + 416, 32)
        self.feat_dim = ch_in + 448
        self.conv_last = pwc_conv(ch_in + 448, 2, isReLU=False)

    def forward(self, x):
        x1 = torch.cat([self.conv1(x), x], dim=1)
        x2 = torch.cat([self.conv2(x1), x1], dim=1)
        x3 = torch.cat([self.conv3(x2), x2], dim=1)
        x4 = torch.cat([self.conv4(x3), x3], dim=1)
        x5 = torch.cat([self.conv5(x4), x4], dim=1)
        x_out = self.conv_last(x5)
        return x5, x_out


class FlowEstimatorLite(nn.Module):
    def __init__(self, ch_in):
        super(FlowEstimatorLite, self).__init__()
        self.conv1 = pwc_conv(ch_in, 128)
        self.conv2 = pwc_conv(128, 128)
        self.conv3 = pwc_conv(128 + 128, 96)
        self.conv4 = pwc_conv(128 + 96, 64)
        self.conv5 = pwc_conv(96 + 64, 32)
        self.feat_dim = 32
        self.predict_flow = pwc_conv(64 + 32, 2, isReLU=False)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(torch.cat([x1, x2], dim=1))
        x4 = self.conv4(torch.cat([x2, x3], dim=1))
        x5 = self.conv5(torch.cat([x3, x4], dim=1))
        flow = self.predict_flow(torch.cat([x4, x5], dim=1))
        return x5, flow


class ContextNetwork(nn.Module):
    def __init__(self, ch_in):
        super(ContextNetwork, self).__init__()

        self.convs = nn.Sequential(
            pwc_conv(ch_in, 128, 3, 1, 1),
            pwc_conv(128, 128, 3, 1, 2),
            pwc_conv(128, 128, 3, 1, 4),
            pwc_conv(128, 96, 3, 1, 8),
            pwc_conv(96, 64, 3, 1, 16),
            pwc_conv(64, 32, 3, 1, 1),
            pwc_conv(32, 2, isReLU=False)
        )

    def forward(self, x):
        return self.convs(x)
