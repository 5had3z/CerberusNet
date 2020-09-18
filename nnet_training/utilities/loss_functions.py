"""Custom losses."""
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

__all__ = ['get_loss_function', 'MixSoftmaxCrossEntropyLoss', 'MixSoftmaxCrossEntropyOHEMLoss',
           'FocalLoss2D', 'DepthAwareLoss', 'ScaleInvariantError', 'InvHuberLoss',
           'DepthReconstructionLossV1', 'FlowReconstructionLossV1']

### MixSoftmaxCrossEntropyLoss etc from F-SCNN Repo
class MixSoftmaxCrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, aux=True, aux_weight=0.2, ignore_label=-1, **kwargs):
        super(MixSoftmaxCrossEntropyLoss, self).__init__(ignore_index=ignore_label)
        self.aux = aux
        self.aux_weight = aux_weight

    def _aux_forward(self, *inputs, **kwargs):
        *preds, target = tuple(inputs)

        loss = super(MixSoftmaxCrossEntropyLoss, self).forward(preds[0], target)
        for i in range(1, len(preds)):
            aux_loss = super(MixSoftmaxCrossEntropyLoss, self).forward(preds[i], target)
            loss += self.aux_weight * aux_loss
        return loss

    def forward(self, *inputs, **kwargs):
        preds, target = tuple(inputs)
        inputs = tuple(list(preds) + [target])
        if self.aux:
            return self._aux_forward(*inputs)
        else:
            return super(MixSoftmaxCrossEntropyLoss, self).forward(*inputs)


class SoftmaxCrossEntropyOHEMLoss(nn.Module):
    def __init__(self, ignore_label=-1, thresh=0.7, min_kept=256, use_weight=True, **kwargs):
        super(SoftmaxCrossEntropyOHEMLoss, self).__init__()
        self.ignore_label = ignore_label
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        if use_weight:
            print("w/ class balance")
            weight = torch.FloatTensor([0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754,
                                        1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955,
                                        1.0865, 1.1529, 1.0507])
            self.criterion = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_label)
        else:
            print("w/o class balance")
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_label)

    def forward(self, predict, target, weight=None):
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))

        n, c, h, w = predict.size()
        input_label = target.data.cpu().numpy().ravel().astype(np.int32)
        x = np.rollaxis(predict.data.cpu().numpy(), 1).reshape((c, -1))
        input_prob = np.exp(x - x.max(axis=0).reshape((1, -1)))
        input_prob /= input_prob.sum(axis=0).reshape((1, -1))

        valid_flag = input_label != self.ignore_label
        valid_inds = np.where(valid_flag)[0]
        label = input_label[valid_flag]
        num_valid = valid_flag.sum()
        if self.min_kept >= num_valid:
            print('Labels: {}'.format(num_valid))
        elif num_valid > 0:
            prob = input_prob[:, valid_flag]
            pred = prob[label, np.arange(len(label), dtype=np.int32)]
            threshold = self.thresh
            if self.min_kept > 0:
                index = pred.argsort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if pred[threshold_index] > self.thresh:
                    threshold = pred[threshold_index]
            kept_flag = pred <= threshold
            valid_inds = valid_inds[kept_flag]

        label = input_label[valid_inds].copy()
        input_label.fill(self.ignore_label)
        input_label[valid_inds] = label
        valid_flag_new = input_label != self.ignore_label
        # print(np.sum(valid_flag_new))
        target = Variable(torch.from_numpy(input_label.reshape(target.size())).long().cuda())

        return self.criterion(predict, target)


class MixSoftmaxCrossEntropyOHEMLoss(SoftmaxCrossEntropyOHEMLoss):
    def __init__(self, aux=False, aux_weight=0.2, ignore_index=-1, **kwargs):
        super(MixSoftmaxCrossEntropyOHEMLoss, self).__init__(ignore_label=ignore_index, **kwargs)
        self.aux = aux
        self.aux_weight = aux_weight

    def _aux_forward(self, *inputs, **kwargs):
        *preds, target = tuple(inputs)

        loss = super(MixSoftmaxCrossEntropyOHEMLoss, self).forward(preds[0], target)
        for i in range(1, len(preds)):
            aux_loss = super(MixSoftmaxCrossEntropyOHEMLoss, self).forward(preds[i], target)
            loss += self.aux_weight * aux_loss
        return loss

    def forward(self, *inputs, **kwargs):
        preds, target = tuple(inputs)
        inputs = tuple(list([preds]) + [target])
        if self.aux:
            return self._aux_forward(*inputs)
        else:
            return super(MixSoftmaxCrossEntropyOHEMLoss, self).forward(*inputs)

class FocalLoss2D(nn.Module):
    """
    Focal Loss for Imbalanced problems, also includes additonal weighting
    """
    def __init__(self, gamma=2, ignore_index=-100, dynamic_weights=False,
                 scale_factor=0.125, **kwargs):
        super(FocalLoss2D, self).__init__()

        self.gamma = gamma
        self.ignore_index = ignore_index
        self.dynamic_weights = dynamic_weights
        self.scale_factor = scale_factor

    def forward(self, seg_pred: torch.Tensor, seg_gt: torch.Tensor, **kwargs) -> torch.Tensor:
        '''
        Forward implementation that returns focal loss between prediciton and target
        '''
        weights = torch.ones(seg_pred.shape[1]).to(seg_pred.get_device())
        if self.dynamic_weights:
            class_ids, counts = seg_gt.unique(return_counts=True)
            weights[class_ids] = self.scale_factor / \
                    (self.scale_factor + counts/float(seg_gt.nelement()))

        # compute the negative likelyhood
        ce_loss = F.cross_entropy(seg_pred, seg_gt, ignore_index=self.ignore_index, weight=weights)

        # compute the loss
        focal_loss = torch.pow(1 - torch.exp(-ce_loss), self.gamma) * ce_loss

        # return the average
        return focal_loss.mean()


class DepthAwareLoss(nn.Module):
    def __init__(self, size_average=True, ignore_index=-1, **kwargs):
        super(DepthAwareLoss, self).__init__()
        self.size_average = size_average
        self._ignore_index = ignore_index

    def forward(self, disp_pred: torch.Tensor, disp_gt: torch.Tensor, **kwargs) -> torch.Tensor:
        disp_pred = F.relu(disp_pred.squeeze(dim=1)) # depth predictions must be >=0

        l_disp_pred = torch.log(disp_pred)
        l_disp_gt = torch.log(disp_gt)
        regularization = 1 - torch.min(l_disp_pred, l_disp_gt) / torch.max(l_disp_pred, l_disp_gt)

        l_loss = F.smooth_l1_loss(disp_pred, disp_gt, size_average=self.size_average)
        depth_aware_attention = disp_gt / torch.max(disp_gt)

        return ((depth_aware_attention + regularization) * l_loss).mean()

class ScaleInvariantError(nn.Module):
    def __init__(self, lmda=1, ignore_index=-1, **kwargs):
        super(ScaleInvariantError, self).__init__()
        self.lmda = lmda
        self._ignore_index = ignore_index

    def forward(self, disp_pred: torch.Tensor, disp_gt: torch.Tensor, **kwargs) -> torch.Tensor:
        #   Number of pixels per image
        n_pixels = disp_gt.shape[1]*disp_gt.shape[2]
        #   Number of valid pixels in target image
        n_valid = (disp_gt != self._ignore_index).view(-1, n_pixels).float().sum(dim=1)

        #   Prevent infs and nans
        disp_pred[disp_pred <= 0] = 0.00001
        disp_pred = disp_pred.squeeze(dim=1)
        disp_pred[disp_gt == self._ignore_index] = 0.00001
        disp_gt[disp_gt == self._ignore_index] = 0.00001

        log_diff = (torch.log(disp_pred) - torch.log(disp_gt)).view(-1, n_pixels)

        element_wise = torch.pow(log_diff, 2).sum(dim=1) / n_valid
        scaled_error = self.lmda * (torch.pow(log_diff.sum(dim=1), 2) / (n_valid**2))
        return (element_wise - scaled_error).mean()

class InvHuberLoss(nn.Module):
    """
    Inverse Huber Loss for Depth/Disparity Training
    """
    def __init__(self, ignore_index=-1, **kwargs):
        super(InvHuberLoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, disp_pred: torch.Tensor, disp_gt: torch.Tensor, **kwargs) -> torch.Tensor:
        pred_relu = F.relu(disp_pred.squeeze(dim=1)) # depth predictions must be >=0
        diff = pred_relu - disp_gt
        mask = disp_gt != self.ignore_index

        err = (diff * mask.float()).abs()
        c = 0.2 * err.max()
        err2 = (diff**2 + c**2) / (2. * c)
        mask_err = err <= c
        mask_err2 = err > c
        cost = (err*mask_err.float() + err2*mask_err2.float()).mean()
        return cost

class InvHuberLossPyr(InvHuberLoss):
    def __init__(self, lvl_weights: List[int], ignore_index=-1, **kwargs):
        self.lvl_weights = lvl_weights
        super(InvHuberLossPyr, self).__init__(ignore_index=-1, **kwargs)

    def forward(self, disp_pred_pyr: List[torch.Tensor],
                disp_gt: torch.Tensor, **kwargs) -> torch.Tensor:
        loss = 0

        for lvl, disp_pred in enumerate(disp_pred_pyr):
            disp_gt_scaled = F.interpolate(disp_gt.unsqueeze(1), tuple(disp_pred.size()[2:]), mode='nearest')
            lvl_loss = super(InvHuberLossPyr, self).forward(disp_pred, disp_gt_scaled)
            loss += (lvl_loss * self.lvl_weights[lvl])

        return loss

class FlowReconstructionLossV1(nn.Module):
    """
    Ultra basic reconstruction loss for flow
    """
    def __init__(self, img_b, img_h, img_w, device=torch.device("cpu")):
        super(FlowReconstructionLossV1, self).__init__()
        self.img_h = img_h
        self.img_w = img_w

        base_x = torch.arange(0, img_w).repeat(img_b, img_h, 1)/float(img_w)*2.-1.
        base_y = torch.arange(0, img_h).repeat(img_b, img_w, 1).transpose(1, 2)/float(img_h)*2.-1.
        self.transf_base = torch.stack([base_x, base_y], 3).to(device)

    def forward(self, pred_flow: torch.Tensor, im1_origin: torch.Tensor,
                im2_origin: torch.Tensor) -> torch.Tensor:
        pred_flow = pred_flow.reshape(-1, self.img_h, self.img_w, 2)
        pred_flow[:, :, :, 0] /= self.img_w
        pred_flow[:, :, :, 1] /= self.img_h
        pred_flow += self.transf_base
        im1_warp = F.grid_sample(im1_origin, pred_flow, mode='bilinear',
                                 padding_mode='zeros', align_corners=None)

        # debug disp
        # import matplotlib.pyplot as plt
        # plt.subplot(1,2,1)
        # plt.imshow(np.moveaxis(source[0,0:3,:,:].cpu().numpy(),0,2))
        # plt.subplot(1,2,2)
        # plt.imshow(np.moveaxis(pred[0,0:3,:,:].cpu().numpy(),0,2))
        # plt.show()

        diff = (im2_origin-im1_warp).abs()
        loss = diff.view([1, -1]).sum(1).mean() / (self.img_w * self.img_h)
        return loss

class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self):
        super(SSIM, self).__init__()
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

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)

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

def get_loss_function(loss_config) -> Dict[str, torch.nn.Module]:
    """
    Returns a dictionary of loss functions given a config
    """
    from nnet_training.utilities.UnFlowLoss import unFlowLoss
    from nnet_training.utilities.rmi import RMILoss

    loss_fn_dict = {}
    for loss_fn in loss_config:
        if loss_fn['function'] == "FocalLoss2D":
            loss_fn_dict[loss_fn['type']] = FocalLoss2D(**loss_fn.args)
        elif loss_fn['function'] == "unFlowLoss":
            loss_fn_dict[loss_fn['type']] = unFlowLoss(**loss_fn.args)
        elif loss_fn['function'] == "InvHuberLoss":
            loss_fn_dict[loss_fn['type']] = InvHuberLoss(**loss_fn.args)
        elif loss_fn['function'] == "ScaleInvariantError":
            loss_fn_dict[loss_fn['type']] = ScaleInvariantError(**loss_fn.args)
        elif loss_fn['function'] == "DepthAwareLoss":
            loss_fn_dict[loss_fn['type']] = DepthAwareLoss(**loss_fn.args)
        elif loss_fn['function'] == "InvHuberLossPyr":
            loss_fn_dict[loss_fn['type']] = InvHuberLossPyr(**loss_fn.args)
        elif loss_fn['function'] == "RMILoss":
            loss_fn_dict[loss_fn['type']] = RMILoss(**loss_fn.args)
        else:
            raise NotImplementedError(loss_fn['function'])

    return loss_fn_dict

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
