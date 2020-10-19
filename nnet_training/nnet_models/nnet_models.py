#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F

import itertools

__all__ = ['StereoDepthSeparatedReLu', 'StereoDepthSeparatedExp',
    'StereoSegmentaionSeparated', 'StereoDepthSegSeparated',
    'StereoDepthSegSeparated2']

from .nnet_modules import SeparateDownsample, DownsampleFusionModule, \
    DownsampleFusionModule2, GlobalFusionModule, UpsampleDepthOutputReLu, \
    UpsampleDepthOutputExp, UpsampleSegmentation, UpsampleFlowOutput

class StereoDepthSeparatedReLu(nn.Module):
    def __init__(self, aux=False, **kwargs):
        super(StereoDepthSeparatedReLu, self).__init__()
        self.left_ds        = SeparateDownsample()
        self.right_ds       = SeparateDownsample()
        self.ds_fusion      = DownsampleFusionModule(48)
        self.global_fusion  = GlobalFusionModule(48, 128, 48)
        self.upsample       = UpsampleDepthOutputReLu(48)

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

class StereoDepthSegSeparated3(nn.Module):
    def __init__(self, classes=19, aux=False, **kwargs):
        super(StereoDepthSegSeparated3, self).__init__()
        self.left_ds        = SeparateDownsample(dw_channels=16, out_channels=32)
        self.right_ds       = SeparateDownsample(dw_channels=16, out_channels=32)
        self.ds_fusion      = DownsampleFusionModule2(in_channels=32, block_depth=3, block_channels=[48, 72, 96])
        self.global_fusion  = GlobalFusionModule(32, 96, 32, scale_factor=4)

        self.depth          = UpsampleDepthOutputReLu(32)
        self.segmentation   = UpsampleSegmentation(32, classes=classes)

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

class MonoFlow1(nn.Module):
    def __init__(self, aux=False, **kwargs):
        super(MonoFlow1, self).__init__()
        self.left1_ds       = SeparateDownsample(dw_channels=16, out_channels=32)
        self.left2_ds       = SeparateDownsample(dw_channels=16, out_channels=32)
        self.ds_fusion      = DownsampleFusionModule2(in_channels=32, block_depth=3, block_channels=[48, 72, 96])
        self.global_fusion  = GlobalFusionModule(32, 96, 32, scale_factor=4)

        self.flow_out          = UpsampleFlowOutput(32)

    def forward(self, left, right):
        assert left.size() == right.size(), 'left and right shape mismatch'
        out_size = left.size()[2:]
        left            = self.left1_ds(left)
        right           = self.left2_ds(right)
        stereo_fused    = self.ds_fusion(left, right)
        global_fused    = self.global_fusion(left, right, stereo_fused)

        #   Depth Only Branch
        flow_est        = self.flow_out(global_fused)
        flow_est        = F.interpolate(flow_est, out_size, mode='bilinear', align_corners=True)

        return flow_est