#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['TestModel']

from nnet_ops import _ConvBNReLU, _DSConv, _DWConv

class TestModel(nn.Module):
    def __init__(self, aux=False, **kwargs):
        super(TestModel, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(6, 16, 3),
            nn.ReLU(),
            nn.Conv2d(16, 32, 5),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3),
            nn.ReLU()
        )

    def forward(self, left, right):
        size = left.size()[2:]
        x = self.model(torch.cat((left, right),dim=1))
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        return x