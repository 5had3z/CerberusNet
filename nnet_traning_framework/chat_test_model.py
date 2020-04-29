#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['TestModel']

class TestModel(nn.Module):
    def __init__(self, num_classes=19, aux=False, **kwargs):
        super(TestModel, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, 3),
            nn.ReLU(),
            nn.Conv2d(16, 64, 5),
            nn.ReLU(),
            nn.ConvTranspose2d(64, num_classes, 3),
            nn.Sigmoid()
        )

    def forward(self, x):
        size = x.size()[2:]
        x = self.model(x)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        return x