from typing import Dict

import torch
from torch import nn

class DeeplabPanopticLoss(nn.Module):
    def __init__(self, weight=1.0, ignore_index=255, **kwargs):
        super().__init__()
        self.weight = weight

        self.center_weight = kwargs.get('center_weight', 200.)
        self.offset_weight = kwargs.get('offset_weight', .1)

        self.semantic_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.center_loss = nn.MSELoss()
        self.offset_loss = nn.L1Loss()

    def forward(self, predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        loss = 0

        # Pixel-wise loss weight
        if 'center_mask' in targets.keys():
            center_loss_mask = targets['center_mask'][:, None, :, :] \
                .expand_as(predictions['center'])
            center_loss = self.center_loss(
                predictions['center'], targets['center']) * center_loss_mask
            # safe division
            if center_loss_mask.sum() > 0:
                loss += self.center_weight * center_loss.sum() / center_loss_mask.sum()
        else:
            loss += self.center_weight * self.center_loss(predictions['center'], targets['center'])

        # Pixel-wise loss weight
        if 'offset_mask' in targets.keys():
            offset_loss_mask = targets['offset_mask'][:, None, :, :] \
                .expand_as(predictions['offset'])
            offset_loss = self.offset_loss(
                predictions['offset'], targets['offset']) * offset_loss_mask
            # safe division
            if offset_loss_mask.sum() > 0:
                loss += self.offset_weight * offset_loss.sum() / offset_loss_mask.sum()
        else:
            loss += self.offset_weight * self.offset_loss(predictions['offset'], targets['offset'])

        return self.weight * loss
