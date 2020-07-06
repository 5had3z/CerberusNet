#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F

import itertools

__all__ = ['StereoDepthSeparatedReLu', 'StereoDepthSeparatedExp',
    'StereoSegmentaionSeparated', 'StereoDepthSegSeparated',
    'StereoDepthSegSeparated2']

from nnet_ops import _ConvBNReLU, _DSConv, _DWConv, LinearBottleneck

class StereoDepthSeparatedReLu(nn.Module):
    def __init__(self, aux=False, **kwargs):
        super(StereoDepthSeparatedReLu, self).__init__()
        self.left_ds        = SeparateDownsample()
        self.right_ds       = SeparateDownsample()
        self.ds_fusion      = DownsampleFusionModule(48)
        self.global_fusion  = GlobalFusionModule(48, 128, 48)
        self.upsample       = UpsampleDepthOutputReLu(48)

    def __str__(self):
        return "ReLuModel"

    def forward(self, left, right):
        left            = self.left_ds(left)
        right           = self.right_ds(right)
        stereo_fused    = self.ds_fusion(left, right)
        global_fused    = self.global_fusion(left, right, stereo_fused)
        out             = self.upsample(global_fused)
        return out

class StereoDepthSeparatedExp(nn.Module):
    def __init__(self, aux=False, **kwargs):
        super(StereoDepthSeparatedExp, self).__init__()
        self.left_ds        = SeparateDownsample()
        self.right_ds       = SeparateDownsample()
        self.ds_fusion      = DownsampleFusionModule(48)
        self.global_fusion  = GlobalFusionModule(48, 128, 48)
        self.upsample       = UpsampleDepthOutputReLu(48)

    def __str__(self):
        return "ExpModel"
    
    def forward(self, left, right):
        left            = self.left_ds(left)
        right           = self.right_ds(right)
        stereo_fused    = self.ds_fusion(left, right)
        global_fused    = self.global_fusion(left, right, stereo_fused)
        out             = self.upsample(global_fused)
        return out

class StereoSegmentaionSeparated(nn.Module):
    def __init__(self, classes=19, aux=False, **kwargs):
        super(StereoSegmentaionSeparated, self).__init__()
        self.left_ds        = SeparateDownsample()
        self.right_ds       = SeparateDownsample()
        self.ds_fusion      = DownsampleFusionModule(48)
        self.global_fusion  = GlobalFusionModule(48, 128, 48)
        self.upsample       = UpsampleSegmentation(48, classes=classes)

    def __str__(self):
        return "StereoSeg1.1"

    def forward(self, left, right):
        assert left.size() == right.size(), 'left and right shape mismatch'
        out_size = left.size()[2:]
        left            = self.left_ds(left)
        right           = self.right_ds(right)
        stereo_fused    = self.ds_fusion(left, right)
        global_fused    = self.global_fusion(left, right, stereo_fused)
        out             = self.upsample(global_fused)
        return F.interpolate(out, out_size, mode='bilinear', align_corners=True)

class StereoDepthSegSeparated(nn.Module):
    def __init__(self, classes=19, aux=False, **kwargs):
        super(StereoDepthSegSeparated, self).__init__()
        self.left_ds        = SeparateDownsample()
        self.right_ds       = SeparateDownsample()
        self.ds_fusion      = DownsampleFusionModule(48)
        self.global_fusion  = GlobalFusionModule(48, 128, 48)

        self.depth          = UpsampleDepthOutputReLu(48)
        self.segmentation   = UpsampleSegmentation(48, classes=classes)

    def __str__(self):
        return "StereoSegDepth1"

    def forward(self, left, right):
        assert left.size() == right.size(), 'left and right shape mismatch'
        out_size = left.size()[2:]
        left            = self.left_ds(left)
        right           = self.right_ds(right)
        stereo_fused    = self.ds_fusion(left, right)
        global_fused    = self.global_fusion(left, right, stereo_fused)

        #   Depth Only Branch
        depth_est       = self.depth(global_fused)
        depth_est       = F.interpolate(depth_est, out_size, mode='bilinear', align_corners=True)

        #   Segmentation Only Branch
        segmentation    = self.segmentation(global_fused)
        segmentation    = F.interpolate(segmentation, out_size, mode='bilinear', align_corners=True)

        return segmentation, depth_est

class StereoDepthSegSeparated2(nn.Module):
    def __init__(self, classes=19, aux=False, **kwargs):
        super(StereoDepthSegSeparated2, self).__init__()
        self.left_ds        = SeparateDownsample(dw_channels=16, out_channels=32)
        self.right_ds       = SeparateDownsample(dw_channels=16, out_channels=32)
        self.ds_fusion      = DownsampleFusionModule2(in_channels=32, block_channels=[48, 72, 96])
        self.global_fusion  = GlobalFusionModule(32, 96, 32, scale_factor=4)

        self.depth          = UpsampleDepthOutputReLu(32)
        self.segmentation   = UpsampleSegmentation(32, classes=classes)

    def __str__(self):
        return "StereoSD1.1"

    def forward(self, left, right):
        assert left.size() == right.size(), 'left and right shape mismatch'
        out_size = left.size()[2:]
        left            = self.left_ds(left)
        right           = self.right_ds(right)
        stereo_fused    = self.ds_fusion(left, right)
        global_fused    = self.global_fusion(left, right, stereo_fused)

        #   Depth Only Branch
        depth_est       = self.depth(global_fused)
        depth_est       = F.interpolate(depth_est, out_size, mode='bilinear', align_corners=True)

        #   Segmentation Only Branch
        segmentation    = self.segmentation(global_fused)
        segmentation    = F.interpolate(segmentation, out_size, mode='bilinear', align_corners=True)

        return segmentation, depth_est

class SeparateDownsample(nn.Module):
    """ Downsample Module for each Image """
    def __init__(self, in_ch = 3, dw_channels=32, out_channels=48, **kwargs):
        super(SeparateDownsample, self).__init__()
        self.conv   = _ConvBNReLU(in_ch, dw_channels, 3, 2)
        self.dsconv = _DSConv(dw_channels, out_channels, 2)

    def forward(self, x):
        x = self.conv(x)
        x = self.dsconv(x)
        return x

class DownsampleFusionModule(nn.Module):
    """Fusion of each downsampled stereo images"""
    def __init__(self, downsampled_channels, block_channels = [64, 96, 128], **kwargs):
        super(DownsampleFusionModule, self).__init__()
        self.dsconv1 = _DSConv(downsampled_channels*2, block_channels[0], 1)
        self.dsconv2 = _DSConv(block_channels[0], block_channels[1], 1)
        self.dsconv3 = _DSConv(block_channels[1], block_channels[2], 1)
        self.relu = nn.ReLU()

    def forward(self, left_downsample, right_downsample):
        x   = self.dsconv1(torch.cat((left_downsample, right_downsample),dim=1))
        x   = self.dsconv2(x)
        out = self.dsconv3(x)
        return self.relu(out)

class DownsampleFusionModule2(nn.Module):
    """Fusion of each downsampled stereo images"""
    def __init__(self, in_channels, block_channels = [64, 96, 128], **kwargs):
        super(DownsampleFusionModule2, self).__init__()
        nBlockIter = 2
        self.dsconv1 = self._make_layer(LinearBottleneck, in_channels*2, block_channels[0], nBlockIter, stride=2)
        self.dsconv2 = self._make_layer(LinearBottleneck, block_channels[0], block_channels[1], nBlockIter, stride=2)
        self.dsconv3 = self._make_layer(LinearBottleneck, block_channels[1], block_channels[2], nBlockIter, stride=1)
        self.relu = nn.ReLU()

    def _make_layer(self, block, inplanes, planes, blocks, t=6, stride=1):
        layers = []
        layers.append(block(inplanes, planes, t, stride))
        for _ in itertools.repeat(None, blocks):
            layers.append(block(planes, planes, t, 1))
        return nn.Sequential(*layers)

    def forward(self, left_downsample, right_downsample):
        x   = self.dsconv1(torch.cat((left_downsample, right_downsample),dim=1))
        x   = self.dsconv2(x)
        out = self.dsconv3(x)
        return self.relu(out)

class GlobalFusionModule(nn.Module):
    """Fusion of low res feature abstraction and both higher res stereo images"""
    def __init__(self, stereo_channels, fused_channels, out_channels, int_channels=96, scale_factor=1, **kwargs):
        super(GlobalFusionModule, self).__init__()
        self.scale_factor = scale_factor

        self.conv_lower_res = nn.Sequential(
            _DSConv(fused_channels, int_channels, 1),
            nn.BatchNorm2d(int_channels)
        )

        self.conv_fuse = nn.Sequential(
            nn.Conv2d(stereo_channels*2+int_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )

        self.relu = nn.ReLU()

    def forward(self, left_downsample, right_downsample, fused_model):
        fused_model = F.interpolate(fused_model, scale_factor=self.scale_factor, mode='bilinear', align_corners=True)
        fused_model = self.conv_lower_res(fused_model)

        stereo_fused = self.conv_fuse(torch.cat((fused_model, left_downsample, right_downsample),dim=1))
        return self.relu(stereo_fused)

class UpsampleDepthOutputReLu(nn.Module):
    """Fusion of each downsampled stereo images with ReLu Output"""
    def __init__(self, in_channels, scale_factor = 4,**kwargs):
        super(UpsampleDepthOutputReLu, self).__init__()
        self.scale_factor = scale_factor
        self.conv_fuse = nn.Conv2d(in_channels, 1, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        upsampled = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=True)
        upsampled = self.conv_fuse(upsampled)
        return self.relu(upsampled)

class UpsampleDepthOutputExp(nn.Module):
    """Fusion of each downsampled stereo images with Exponential Output"""
    def __init__(self, in_channels, scale_factor = 4,**kwargs):
        super(UpsampleDepthOutputExp, self).__init__()
        self.scale_factor = scale_factor
        self.conv_fuse = nn.Conv2d(in_channels, 1, 1)

    def forward(self, x):
        upsampled = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=True)
        upsampled = self.conv_fuse(upsampled)
        return torch.exp(upsampled)

class UpsampleSegmentation(nn.Module):
    """Fusion of each downsampled stereo images with Class Segmentation"""
    def __init__(self, in_channels, classes = 19, stride = 1,**kwargs):
        super(UpsampleSegmentation, self).__init__()
        self.dsconv = _DSConv(in_channels, in_channels, stride)
        self.conv_out = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv2d(in_channels, classes, 1)
        )

    def forward(self, x):
        x = self.dsconv(x)
        x = self.conv_out(x)
        return x