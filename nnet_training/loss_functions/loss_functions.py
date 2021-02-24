"""Custom losses."""
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['FlowReconstructionLossV1', 'SSIM']

class FlowReconstructionLossV1(nn.Module):
    """
    Ultra basic reconstruction loss for flow
    """
    def __init__(self, img_b, img_h, img_w, device=torch.device("cpu")):
        super().__init__()
        self.img_h = img_h
        self.img_w = img_w

        base_x = torch.arange(0, img_w).repeat(img_b, img_h, 1)/float(img_w)*2.-1.
        base_y = torch.arange(0, img_h).repeat(img_b, img_w, 1).transpose(1, 2)/float(img_h)*2.-1.
        self.transf_base = torch.stack([base_x, base_y], 3).to(device)

    def forward(self, predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        assert 'flow' in predictions.keys() and \
            all(key in targets.keys() for key in ['l_img', 'l_seq'])

        pred_flow = predictions['flow'].reshape(-1, self.img_h, self.img_w, 2)
        pred_flow[:, :, :, 0] /= self.img_w
        pred_flow[:, :, :, 1] /= self.img_h
        pred_flow += self.transf_base
        im1_warp = F.grid_sample(targets['l_img'], pred_flow, mode='bilinear',
                                 padding_mode='zeros', align_corners=None)

        # debug disp
        # import matplotlib.pyplot as plt
        # plt.subplot(1,2,1)
        # plt.imshow(np.moveaxis(source[0,0:3,:,:].cpu().numpy(),0,2))
        # plt.subplot(1,2,2)
        # plt.imshow(np.moveaxis(pred[0,0:3,:,:].cpu().numpy(),0,2))
        # plt.show()

        diff = (targets['l_seq']-im1_warp).abs()
        loss = diff.view([1, -1]).sum(1).mean() / (self.img_w * self.img_h)
        return loss

class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self):
        super().__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        ssim_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        ssim_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - ssim_n / ssim_d) / 2, 0, 1)


if __name__ == '__main__':
    raise NotImplementedError
