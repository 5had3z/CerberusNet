import torch
import torch.nn as nn
import torch.nn.functional as F
from nnet_training.utilities.loss_functions import SSIM

def mesh_grid(B, H, W):
    # mesh grid
    x_base = torch.arange(0, W).repeat(B, H, 1)  # BHW
    y_base = torch.arange(0, H).repeat(B, W, 1).transpose(1, 2)  # BHW

    base_grid = torch.stack([x_base, y_base], 1)  # B2HW
    return base_grid

def norm_grid(v_grid):
    _, _, H, W = v_grid.size()

    # scale grid to [-1,1]
    v_grid_norm = torch.zeros_like(v_grid)
    v_grid_norm[:, 0, :, :] = 2.0 * v_grid[:, 0, :, :] / (W - 1) - 1.0
    v_grid_norm[:, 1, :, :] = 2.0 * v_grid[:, 1, :, :] / (H - 1) - 1.0
    return v_grid_norm.permute(0, 2, 3, 1)  # BHW2

def get_corresponding_map(data):
    """
    :param data: unnormalized coordinates Bx2xHxW
    :return: Bx1xHxW
    """
    B, _, H, W = data.size()

    x = data[:, 0, :, :].view(B, -1)  # BxN (N=H*W)
    y = data[:, 1, :, :].view(B, -1)

    x1 = torch.floor(x)
    x_floor = x1.clamp(0, W - 1)
    y1 = torch.floor(y)
    y_floor = y1.clamp(0, H - 1)
    x0 = x1 + 1
    x_ceil = x0.clamp(0, W - 1)
    y0 = y1 + 1
    y_ceil = y0.clamp(0, H - 1)

    x_ceil_out = x0 != x_ceil
    y_ceil_out = y0 != y_ceil
    x_floor_out = x1 != x_floor
    y_floor_out = y1 != y_floor
    invalid = torch.cat([x_ceil_out | y_ceil_out,
                         x_ceil_out | y_floor_out,
                         x_floor_out | y_ceil_out,
                         x_floor_out | y_floor_out], dim=1)

    # encode coordinates, since the scatter function can only index along one axis
    corresponding_map = torch.zeros(B, H * W).type_as(data)
    indices = torch.cat([x_ceil + y_ceil * W,
                         x_ceil + y_floor * W,
                         x_floor + y_ceil * W,
                         x_floor + y_floor * W], 1).long()  # BxN   (N=4*H*W)
    values = torch.cat([(1 - torch.abs(x - x_ceil)) * (1 - torch.abs(y - y_ceil)),
                        (1 - torch.abs(x - x_ceil)) * (1 - torch.abs(y - y_floor)),
                        (1 - torch.abs(x - x_floor)) * (1 - torch.abs(y - y_ceil)),
                        (1 - torch.abs(x - x_floor)) * (1 - torch.abs(y - y_floor))],
                       1)
    # values = torch.ones_like(values)

    values[invalid] = 0

    corresponding_map.scatter_add_(1, indices, values)
    # decode coordinates
    corresponding_map = corresponding_map.view(B, H, W)

    return corresponding_map.unsqueeze(1)

def flow_warp(x, flow12, pad='border', mode='bilinear'):
    B, _, H, W = x.size()

    base_grid = mesh_grid(B, H, W).type_as(x)  # B2HW

    v_grid = norm_grid(base_grid + flow12)  # BHW2
    im1_recons = nn.functional.grid_sample(x, v_grid, mode=mode, padding_mode=pad)
    return im1_recons

def get_occu_mask_bidirection(flow12, flow21, scale=0.01, bias=0.5):
    flow21_warped = flow_warp(flow21, flow12, pad='zeros')
    flow12_diff = flow12 + flow21_warped
    mag = (flow12 * flow12).sum(1, keepdim=True) + \
          (flow21_warped * flow21_warped).sum(1, keepdim=True)
    occ_thresh = scale * mag + bias
    occ = (flow12_diff * flow12_diff).sum(1, keepdim=True) > occ_thresh
    return occ.float()

def get_occu_mask_backward(flow21, th=0.2):
    B, _, H, W = flow21.size()
    base_grid = mesh_grid(B, H, W).type_as(flow21)  # B2HW

    corr_map = get_corresponding_map(base_grid + flow21)  # BHW
    occu_mask = corr_map.clamp(min=0., max=1.) < th
    return occu_mask.float()

# Credit: https://github.com/simonmeister/UnFlow/blob/master/src/e2eflow/core/losses.py
def TernaryLoss(im, im_warp, max_distance=1):
    patch_size = 2 * max_distance + 1

    def _rgb_to_grayscale(image):
        grayscale = image[:, 0, :, :] * 0.2989 + \
                    image[:, 1, :, :] * 0.5870 + \
                    image[:, 2, :, :] * 0.1140
        return grayscale.unsqueeze(1)

    def _ternary_transform(image):
        intensities = _rgb_to_grayscale(image) * 255
        out_channels = patch_size * patch_size
        w = torch.eye(out_channels).view((out_channels, 1, patch_size, patch_size))
        weights = w.type_as(im)
        patches = F.conv2d(intensities, weights, padding=max_distance)
        transf = patches - intensities
        transf_norm = transf / torch.sqrt(0.81 + torch.pow(transf, 2))
        return transf_norm

    def _hamming_distance(t1, t2):
        dist = torch.pow(t1 - t2, 2)
        dist_norm = dist / (0.1 + dist)
        dist_mean = torch.mean(dist_norm, 1, keepdim=True)  # instead of sum
        return dist_mean

    def _valid_mask(t, padding):
        n, _, h, w = t.size()
        inner = torch.ones(n, 1, h - 2 * padding, w - 2 * padding).type_as(t)
        mask = F.pad(inner, [padding] * 4)
        return mask

    t1 = _ternary_transform(im)
    t2 = _ternary_transform(im_warp)
    dist = _hamming_distance(t1, t2)
    mask = _valid_mask(im, max_distance)

    return dist * mask

def gradient(data):
    D_dy = data[:, :, 1:] - data[:, :, :-1]
    D_dx = data[:, :, :, 1:] - data[:, :, :, :-1]
    return D_dx, D_dy

def smooth_grad_1st(flow, image, alpha):
    img_dx, img_dy = gradient(image)
    weights_x = torch.exp(-torch.mean(torch.abs(img_dx), 1, keepdim=True) * alpha)
    weights_y = torch.exp(-torch.mean(torch.abs(img_dy), 1, keepdim=True) * alpha)

    dx, dy = gradient(flow)

    loss_x = weights_x * dx.abs() / 2.
    loss_y = weights_y * dy.abs() / 2

    return loss_x.mean() / 2. + loss_y.mean() / 2.

def smooth_grad_2nd(flow, image, alpha):
    img_dx, img_dy = gradient(image)
    weights_x = torch.exp(-torch.mean(torch.abs(img_dx), 1, keepdim=True) * alpha)
    weights_y = torch.exp(-torch.mean(torch.abs(img_dy), 1, keepdim=True) * alpha)

    dx, dy = gradient(flow)
    dx2, _ = gradient(dx)
    _, dy2 = gradient(dy)

    loss_x = weights_x[:, :, :, 1:] * dx2.abs()
    loss_y = weights_y[:, :, 1:, :] * dy2.abs()

    return loss_x.mean()/2. + loss_y.mean() / 2.

class unFlowLoss(nn.modules.Module):
    def __init__(self, weights, smooth_deg=2, smooth_alpha=0.2, consistency=True, back_occ_only=False):
        super(unFlowLoss, self).__init__()

        if "l1" in weights:
            self.l1_weight = weights["l1"]
        if "ssim" in weights:
            self.ssim_weight = weights["ssim"]
            self.SSIM = SSIM().to("cuda" if torch.cuda.is_available() else "cpu")
        if "ternary" in weights:
            self.ternary_weight = weights["ternary"]
        
        self.smooth_deg     = smooth_deg
        self.smooth_alpha   = smooth_alpha
        self.smooth_w       = 75.0
        self.w_scales       = [1.0, 1.0, 1.0, 1.0, 0.0]
        self.w_sm_scales    = [1.0, 0.0, 0.0, 0.0, 0.0]
        self.consistency    = consistency
        self.back_occ_only  = back_occ_only

    def loss_photometric(self, im_orig, im_recons, occu_mask):
        loss = []

        if hasattr(self, 'l1_weight'):
            loss += [self.l1_weight * (im_orig - im_recons).abs()]

        if hasattr(self, 'ssim_weight'):
            loss += [self.ssim_weight * self.SSIM(im_recons,\
                                            im_orig)]

        if hasattr(self, 'ternary_weight'):
            loss += [self.ternary_weight * TernaryLoss(im_recons * occu_mask,\
                                                      im_orig * occu_mask)]

        return sum([l.mean() for l in loss]) #/ occu_mask.mean()

    def loss_smooth(self, flow, im_scaled):
        if self.smooth_deg == 2:
            func_smooth = smooth_grad_2nd
        elif self.smooth_deg == 1:
            func_smooth = smooth_grad_1st
        else:
            raise NotImplementedError

        loss = []
        loss += [func_smooth(flow, im_scaled, self.smooth_alpha)]
        return sum([l.mean() for l in loss])

    def forward(self, pyramid_flows, im1_origin, im2_origin):
        """
        :param output: Multi-scale forward/backward flows n * [B x 4 x h x w]
        :param target: image pairs Nx6xHxW
        :return:
        """

        pyramid_smooth_losses = []
        pyramid_warp_losses = []

        s = 1.
        for i, flow in enumerate(pyramid_flows):
            if self.w_scales[i] == 0:
                pyramid_warp_losses.append(0)
                pyramid_smooth_losses.append(0)
                continue

            _, _, h, w = flow.size()

            # resize images to match the size of layer
            im1_scaled = F.interpolate(im1_origin, (h, w), mode='area')
            im2_scaled = F.interpolate(im2_origin, (h, w), mode='area')

            im1_recons = flow_warp(im2_scaled, flow[:, :2], pad='border')
            im2_recons = flow_warp(im1_scaled, flow[:, 2:], pad='border')

            if i == 0:
                if self.back_occ_only:
                    occu_mask1 = 1 - get_occu_mask_backward(flow[:, 2:], th=0.2)
                    occu_mask2 = 1 - get_occu_mask_backward(flow[:, :2], th=0.2)
                else:
                    occu_mask1 = 1 - get_occu_mask_bidirection(flow[:, :2], flow[:, 2:])
                    occu_mask2 = 1 - get_occu_mask_bidirection(flow[:, 2:], flow[:, :2])
            else:
                occu_mask1 = F.interpolate(occu_mask1, (h, w), mode='nearest')
                occu_mask2 = F.interpolate(occu_mask2, (h, w), mode='nearest')

            loss_warp = self.loss_photometric(im1_scaled, im1_recons, occu_mask1)

            if i == 0:
                s = min(h, w)

            loss_smooth = self.loss_smooth(flow[:, :2] / s, im1_scaled)

            if self.consistency:
                loss_warp += self.loss_photometric(im2_scaled, im2_recons, occu_mask2)
                loss_smooth += self.loss_smooth(flow[:, 2:] / s, im2_scaled)

                loss_warp /= 2.
                loss_smooth /= 2.

            pyramid_warp_losses.append(loss_warp)
            pyramid_smooth_losses.append(loss_smooth)

        pyramid_warp_losses = [l * w for l, w in
                               zip(pyramid_warp_losses, self.w_scales)]
        pyramid_smooth_losses = [l * w for l, w in
                                 zip(pyramid_smooth_losses, self.w_sm_scales)]

        warp_loss = sum(pyramid_warp_losses)
        smooth_loss = self.smooth_w * sum(pyramid_smooth_losses)
        total_loss = warp_loss + smooth_loss

        return total_loss, warp_loss, smooth_loss, pyramid_flows[0].abs().mean()
