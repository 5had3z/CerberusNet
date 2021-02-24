# ------------------------------------------------------------------------------
# Panoptic-DeepLab meta architecture.
# Written by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

from typing import List
from collections import OrderedDict
from functools import partial

import torch
from torch import nn
from torch.nn import functional as F

from .aspp import ASPP

__all__ = ["PanopticDeepLabDecoder"]

def basic_conv(in_planes, out_planes, kernel_size, stride=1, padding=1, groups=1,
               with_bn=True, with_relu=True):
    """convolution with bn and relu"""
    module = []
    has_bias = not with_bn
    module.append(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, groups=groups, bias=has_bias)
    )
    if with_bn:
        module.append(nn.BatchNorm2d(out_planes))
    if with_relu:
        module.append(nn.ReLU())
    return nn.Sequential(*module)

def depthwise_separable_conv(in_planes, out_planes, kernel_size, stride=1, padding=1, groups=1,
                             with_bn=True, with_relu=True):
    """depthwise separable convolution with bn and relu"""
    del groups

    module = []
    module.extend([
        basic_conv(in_planes, in_planes, kernel_size, stride, padding, groups=in_planes,
                   with_bn=True, with_relu=True),
        nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
    ])
    if with_bn:
        module.append(nn.BatchNorm2d(out_planes))
    if with_relu:
        module.append(nn.ReLU())
    return nn.Sequential(*module)

def stacked_conv(in_planes, out_planes, kernel_size, num_stack, stride=1, padding=1, groups=1,
                 with_bn=True, with_relu=True, conv_type='basic_conv'):
    """stacked convolution with bn and relu"""
    if num_stack < 1:
        assert ValueError('`num_stack` has to be a positive integer.')
    if conv_type == 'basic_conv':
        conv = partial(basic_conv, out_planes=out_planes, kernel_size=kernel_size, stride=stride,
                       padding=padding, groups=groups, with_bn=with_bn, with_relu=with_relu)
    elif conv_type == 'depthwise_separable_conv':
        conv = partial(depthwise_separable_conv, out_planes=out_planes, kernel_size=kernel_size,
            stride=stride, padding=padding, groups=1, with_bn=with_bn, with_relu=with_relu)
    else:
        raise ValueError('Unknown conv_type: {}'.format(conv_type))
    module = []
    module.append(conv(in_planes=in_planes))
    for _ in range(1, num_stack):
        module.append(conv(in_planes=out_planes))
    return nn.Sequential(*module)

class SinglePanopticDeepLabDecoder(nn.Module):
    def __init__(self, in_channels: List[int], low_level_channels_project: List[int],
                 decoder_channels: int, atrous_rates: List[int], aspp_channels:int=None):
        super().__init__()
        if aspp_channels is None:
            aspp_channels = decoder_channels
        self.aspp = ASPP(in_channels[-1], out_channels=aspp_channels, atrous_rates=atrous_rates)
        self.decoder_stage = len(in_channels)-1
        assert self.decoder_stage == len(low_level_channels_project), \
            f"Unequal decoder stage {in_channels[-1:]}: {low_level_channels_project}"
        fuse_conv = partial(stacked_conv, kernel_size=5, num_stack=1, padding=2,
                            conv_type='depthwise_separable_conv')

        # Transform low-level feature
        project = []
        # Fuse
        fuse = []
        # Top-down direction, i.e. starting from largest stride
        for i in range(self.decoder_stage):
            project.append(
                nn.Sequential(
                    nn.Conv2d(in_channels[-2-i], low_level_channels_project[i], 1, bias=False),
                    nn.BatchNorm2d(low_level_channels_project[i]),
                    nn.ReLU()
                )
            )
            if i == 0:
                fuse_in_channels = aspp_channels + low_level_channels_project[i]
            else:
                fuse_in_channels = decoder_channels + low_level_channels_project[i]
            fuse.append(fuse_conv(fuse_in_channels, decoder_channels,))
        self.project = nn.ModuleList(project)
        self.fuse = nn.ModuleList(fuse)

    def set_image_pooling(self, pool_size):
        self.aspp.set_image_pooling(pool_size)

    def forward(self, features):
        x = features[0]
        x = self.aspp(x)

        # build decoder
        for idx in range(self.decoder_stage):
            l = features[idx+1]
            l = self.project[idx](l)
            x = F.interpolate(x, size=l.size()[2:], mode='bilinear', align_corners=True)
            x = torch.cat((x, l), dim=1)
            x = self.fuse[idx](x)

        return x


class SinglePanopticDeepLabHead(nn.Module):
    def __init__(self, decoder_channels, head_channels, num_classes, class_key):
        super().__init__()
        fuse_conv = partial(stacked_conv, kernel_size=5, num_stack=1, padding=2,
                            conv_type='depthwise_separable_conv')

        self.num_head = len(num_classes)
        assert self.num_head == len(class_key)

        classifier = {}
        for i in range(self.num_head):
            classifier[class_key[i]] = nn.Sequential(
                fuse_conv(decoder_channels,head_channels,),
                nn.Conv2d(head_channels, num_classes[i], 1)
            )
        self.classifier = nn.ModuleDict(classifier)
        self.class_key = class_key

    def forward(self, x):
        pred = OrderedDict()
        # build classifier
        for key in self.class_key:
            pred[key] = self.classifier[key](x)

        return pred


class PanopticDeepLabDecoder(nn.Module):
    def __init__(self, in_channels: List[int], low_level_channels_project: List[int],
                 decoder_channels: int, atrous_rates: List[int], num_classes: int, **kwargs):
        super().__init__()
        # Build semantic decoder
        self.semantic_decoder = SinglePanopticDeepLabDecoder(in_channels,
            low_level_channels_project, decoder_channels, atrous_rates)
        self.semantic_head = SinglePanopticDeepLabHead(
            decoder_channels, decoder_channels, [num_classes], ['seg'])

        # Build instance decoder
        self.instance_decoder = None
        self.instance_head = None
        if kwargs.get('has_instance', False):
            instance_decoder_kwargs = dict(
                in_channels=in_channels,
                low_level_channels_project=kwargs['instance_low_level_channels_project'],
                decoder_channels=kwargs['instance_decoder_channels'],
                atrous_rates=atrous_rates,
                aspp_channels=kwargs['instance_aspp_channels']
            )
            self.instance_decoder = SinglePanopticDeepLabDecoder(**instance_decoder_kwargs)
            instance_head_kwargs = dict(
                decoder_channels=kwargs['instance_decoder_channels'],
                head_channels=kwargs['instance_head_channels'],
                num_classes=kwargs['instance_num_classes'],
                class_key=kwargs['instance_class_key']
            )
            self.instance_head = SinglePanopticDeepLabHead(**instance_head_kwargs)

    def set_image_pooling(self, pool_size):
        self.semantic_decoder.set_image_pooling(pool_size)
        if self.instance_decoder is not None:
            self.instance_decoder.set_image_pooling(pool_size)

    def forward(self, features):
        pred = OrderedDict()

        # Semantic branch
        semantic = self.semantic_decoder(features[1])
        semantic = self.semantic_head(semantic)
        for key in semantic.keys():
            pred[key] = semantic[key]

        # Instance branch
        if self.instance_decoder is not None:
            instance = self.instance_decoder(features[1])
            instance = self.instance_head(instance)
            for key in instance.keys():
                pred[key] = instance[key]

        return pred
