"""Various Depth losses."""

from typing import Dict, List

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .loss_functions import SSIM

__all__ = ['DepthAwareLoss', 'ScaleInvariantError', 'InvHuberLoss',
           'DepthReconstructionLossV1']

class DepthAwareLoss(nn.Module):
    def __init__(self, size_average=True, ignore_index=-1, **kwargs):
        super(DepthAwareLoss, self).__init__()
        self.size_average = size_average
        self._ignore_index = ignore_index

    def forward(self, disp_pred: torch.Tensor, disp_gt: torch.Tensor, **kwargs) -> torch.Tensor:
        disp_pred = F.relu(disp_pred.squeeze(dim=1)) # depth predictions must be >=0
        disp_pred[disp_pred == 0] += 0.1 # prevent nans during log

        l_disp_pred = torch.log(disp_pred[disp_gt != 0])
        l_disp_gt = torch.log(disp_gt[disp_gt != 0])
        regularization = 1 - torch.min(l_disp_pred, l_disp_gt) / torch.max(l_disp_pred, l_disp_gt)

        l_loss = F.smooth_l1_loss(disp_pred, disp_gt, size_average=self.size_average)
        depth_aware_attention = disp_gt / torch.max(disp_gt)

        return ((depth_aware_attention + regularization) * l_loss).mean()

class ScaleInvariantError(nn.Module):
    def __init__(self, lmda=1, **kwargs):
        super(ScaleInvariantError, self).__init__()
        self.lmda = lmda

    def forward(self, disp_pred: torch.Tensor, disp_gt: torch.Tensor, **kwargs) -> torch.Tensor:
        disp_pred = F.relu(disp_pred.squeeze(dim=1)) # depth predictions must be >=0
        disp_pred[disp_pred == 0] += 0.1 # prevent nans during log

        #   Number of pixels per image
        n_pixels = disp_gt.shape[1]*disp_gt.shape[2]
        #   Number of valid pixels in target image
        n_valid = (disp_gt != 0).view(-1, n_pixels).float().sum(dim=1)

        log_diff = (torch.log(disp_pred[disp_gt != 0]) - torch.log(disp_gt[disp_gt != 0])).view(-1, n_pixels)

        element_wise = torch.pow(log_diff, 2).sum(dim=1) / n_valid
        scaled_error = self.lmda * (torch.pow(log_diff.sum(dim=1), 2) / (n_valid**2))
        return (element_wise - scaled_error).mean()

class InvHuberLoss(nn.Module):
    """
    Inverse Huber Loss for Depth/Disparity Training
    """
    def __init__(self, weight=1.0, **kwargs):
        super(InvHuberLoss, self).__init__()
        self.weight = weight

    def forward(self, disp_pred: torch.Tensor, disp_gt: torch.Tensor, **kwargs) -> torch.Tensor:
        pred_relu = F.relu(disp_pred.squeeze(dim=1)) # depth predictions must be >=0
        diff = pred_relu - disp_gt
        mask = disp_gt > 0

        err = (diff * mask.float()).abs()
        c = 0.2 * err.max()
        err2 = (diff**2 + c**2) / (2. * c)
        mask_err = err <= c
        mask_err2 = err > c
        cost = (err*mask_err.float() + err2*mask_err2.float()).mean()
        return self.weight * cost

class InvHuberLossPyr(InvHuberLoss):
    def __init__(self, lvl_weights: List[int], **kwargs):
        self.lvl_weights = lvl_weights
        super(InvHuberLossPyr, self).__init__(**kwargs)

    def forward(self, disp_pred_pyr: List[torch.Tensor],
                disp_gt: torch.Tensor, **kwargs) -> torch.Tensor:
        loss = 0

        for lvl, disp_pred in enumerate(disp_pred_pyr):
            disp_gt_scaled = F.interpolate(disp_gt.unsqueeze(1), tuple(disp_pred.size()[2:]),
                                           mode='nearest')
            lvl_loss = super(InvHuberLossPyr, self).forward(disp_pred, disp_gt_scaled)
            loss += (lvl_loss * self.lvl_weights[lvl])

        return loss

class BackprojectDepth(nn.Module):
    """Layer to transform a depth image into a point cloud
    """
    def __init__(self, batch_size, height, width):
        super(BackprojectDepth, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
                                      requires_grad=False)

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False)

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1),
                                       requires_grad=False)

    def forward(self, depth, inv_K):
        cam_points = torch.matmul(inv_K[:, :3, :3].cuda(), self.pix_coords)
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, self.ones], 1)

        return cam_points

class Project3D(nn.Module):
    """Layer which projects 3D points into a camera with intrinsics K and at position T
    """
    def __init__(self, batch_size, height, width, eps=1e-7):
        super(Project3D, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points, K, T):
        P = torch.matmul(K.cuda(), T.cuda())[:, :3, :]

        cam_points = torch.matmul(P, points)

        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)
        pix_coords = pix_coords.view(self.batch_size, 2, self.height, self.width)
        pix_coords = pix_coords.permute(0, 2, 3, 1)
        pix_coords[..., 0] /= self.width - 1
        pix_coords[..., 1] /= self.height - 1
        pix_coords = (pix_coords - 0.5) * 2
        return pix_coords

class DepthReconstructionLossV1(nn.Module):
    """Generate the warped (reprojected) color images for a minibatch.
    """
    def __init__(self, batch_size, height, width, pred_type="disparity", ssim=True):
        super(DepthReconstructionLossV1, self).__init__()
        self.pred_type = pred_type
        self.BackprojDepth = BackprojectDepth(batch_size, height, width)\
                                .to("cuda" if torch.cuda.is_available() else "cpu")
        self.Project3D = Project3D(batch_size, height, width)\
                                .to("cuda" if torch.cuda.is_available() else "cpu")
        if ssim:
            self.SSIM = SSIM().to("cuda" if torch.cuda.is_available() else "cpu")

    def depth_from_disparity(self, disparity):
        return (0.209313 * 2262.52) / ((disparity - 1) / 256)

    def forward(self, disp_pred, source_img, target_img, telemetry, camera):
        if self.pred_type == "depth":
            depth = disp_pred
        elif self.pred_type == "disparity":
            depth = self.depth_from_disparity(disp_pred)
        else:
            raise NotImplementedError(self.pred_type)

        cam_points = self.BackprojDepth(depth, camera["inv_K"])
        pix_coords = self.Project3D(cam_points, camera["K"], telemetry)

        source_img = F.grid_sample(source_img, pix_coords, padding_mode="border")

        abs_diff = (target_img - source_img).abs()
        if hasattr(self, 'SSIM'):
            loss = 0.15*abs_diff.mean(1, True) +\
                0.85*self.SSIM(source_img, target_img).mean(1, True)
        else:
            loss = abs_diff.mean(1, True)
        return loss.mean()

if __name__ == '__main__':
    import PIL.Image as Image
    import torchvision.transforms

    BASE_DIR = '/media/bryce/4TB Seagate/Autonomous Vehicles Data/Cityscapes Data/'
    transform = torchvision.transforms.ToTensor()

    img1 = Image.open(BASE_DIR+'leftImg8bit/test/berlin/berlin_000000_000019_leftImg8bit.png')
    img2 = Image.open(BASE_DIR+'leftImg8bit_sequence/test/berlin/berlin_000000_000020_leftImg8bit.png')

    loss_func = DepthReconstructionLossV1(1, img1.size[1], img1.size[0])
    uniform = torch.zeros([1, img1.size[1], img1.size[0], 2])

    print(loss_func(transform(img1).unsqueeze(0), uniform, transform(img1).unsqueeze(0)))