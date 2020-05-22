#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['TestModel']

class TestModel(nn.Module):
    def __init__(self, aux=False, **kwargs):
        super(TestModel, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(6, 16, 3),
            nn.ReLU(),
            nn.Conv2d(16, 32, 5),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3),
            nn.Sigmoid()
        )

    def forward(self, x):
        size = x.size()[2:]
        x = self.model(x)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        return x