"""
Copyright 2020 Nvidia Corporation

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""
from collections import OrderedDict

from torch import nn

import hrnetv2

from .nnet_ops import initialize_weights
from .ocr_utils import BNReLU, SpatialGather_Module, SpatialOCR_Module, get_aspp

def scale_as(x, y):
    '''
    scale x to the same size as y
    '''
    y_size = y.size(2), y.size(3)
    x_scaled = nn.functional.interpolate(x, size=y_size, mode='bilinear',
                                         align_corners=True, recompute_scale_factor=True)
    return x_scaled

def init_attn(m):
    for module in m.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.zeros_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.5)
        elif isinstance(module, nn.BatchNorm2d):
            module.weight.data.fill_(1)
            module.bias.data.zero_()

def old_make_attn_head(in_ch, bot_ch, out_ch):
    attn = nn.Sequential(
        nn.Conv2d(in_ch, bot_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(bot_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(bot_ch, bot_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(bot_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(bot_ch, out_ch, kernel_size=out_ch, bias=False),
        nn.Sigmoid())

    init_attn(attn)
    return attn

def make_attn_head(in_ch, out_ch, **kwargs):
    bot_ch = 256 if 'segattn_bot_ch' not in kwargs else kwargs['segattn_bot_ch']

    if 'mscale_oldarch' in kwargs and kwargs['mscale_oldarch']:
        return old_make_attn_head(in_ch, bot_ch, out_ch)

    od = OrderedDict([('conv0', nn.Conv2d(in_ch, bot_ch, kernel_size=3,
                                          padding=1, bias=False)),
                      ('bn0', nn.BatchNorm2d(bot_ch)),
                      ('re0', nn.ReLU(inplace=True))])

    inner_3x3 = True if 'mscale_inner_3x3' not in kwargs else kwargs['mscale_inner_3x3']
    if inner_3x3:
        od['conv1'] = nn.Conv2d(bot_ch, bot_ch, kernel_size=3, padding=1,
                                bias=False)
        od['bn1'] = nn.BatchNorm2d(bot_ch)
        od['re1'] = nn.ReLU(inplace=True)

    if 'mscale_dropout' in kwargs and kwargs['mscale_dropout']:
        od['drop'] = nn.Dropout(0.5)

    od['conv2'] = nn.Conv2d(bot_ch, out_ch, kernel_size=1, bias=False)
    od['sig'] = nn.Sigmoid()

    attn_head = nn.Sequential(od)
    # init_attn(attn_head)
    return attn_head

class OCR_block(nn.Module):
    """
    Some of the code in this class is borrowed from:
    https://github.com/HRNet/HRNet-Semantic-Segmentation/tree/HRNet-OCR
    """
    def __init__(self, high_level_ch, **kwargs):
        super(OCR_block, self).__init__()

        ocr_mid_channels = kwargs['mid_channels']
        ocr_key_channels = kwargs['key_channels']
        num_classes = kwargs['classes']

        self.conv3x3_ocr = nn.Sequential(
            nn.Conv2d(high_level_ch, ocr_mid_channels,
                      kernel_size=3, stride=1, padding=1),
            BNReLU(ocr_mid_channels)
        )
        self.ocr_gather_head = SpatialGather_Module(num_classes)
        self.ocr_distri_head = SpatialOCR_Module(in_channels=ocr_mid_channels,
                                                 key_channels=ocr_key_channels,
                                                 out_channels=ocr_mid_channels,
                                                 scale=1,
                                                 dropout=0.05,
                                                 )
        self.cls_head = nn.Conv2d(
            ocr_mid_channels, num_classes, kernel_size=1, stride=1, padding=0,
            bias=True)

        self.aux_head = nn.Sequential(
            nn.Conv2d(high_level_ch, high_level_ch,
                      kernel_size=1, stride=1, padding=0),
            BNReLU(high_level_ch),
            nn.Conv2d(high_level_ch, num_classes,
                      kernel_size=1, stride=1, padding=0, bias=True)
        )

        if 'init_decoder' in kwargs:
            initialize_weights(self.conv3x3_ocr,
                               self.ocr_gather_head,
                               self.ocr_distri_head,
                               self.cls_head,
                               self.aux_head)

    def forward(self, high_level_features):
        feats = self.conv3x3_ocr(high_level_features)
        aux_out = self.aux_head(high_level_features)
        context = self.ocr_gather_head(feats, aux_out)
        ocr_feats = self.ocr_distri_head(feats, context)
        cls_out = self.cls_head(ocr_feats)
        return cls_out, aux_out, ocr_feats


class OCRNet(nn.Module):
    """
    OCR net
    """
    def __init__(self, num_classes, criterion=None, **kwargs):
        super(OCRNet, self).__init__()
        self.criterion = criterion
        self.backbone = hrnetv2.get_seg_model(**kwargs)
        self.ocr = OCR_block(self.backbone.high_level_ch, **kwargs)
        self.alpha = 0.4 if 'alpha' not in kwargs else kwargs['alpha']
        self.aux_rmi = False if 'aux_rmi' not in kwargs else kwargs['aux_rmi']

    def forward(self, inputs):
        assert 'images' in inputs
        x = inputs['images']

        _, _, high_level_features = self.backbone(x)
        cls_out, aux_out, _ = self.ocr(high_level_features)
        aux_out = scale_as(aux_out, x)
        cls_out = scale_as(cls_out, x)

        if self.training:
            gts = inputs['gts']
            aux_loss = self.criterion(aux_out, gts, do_rmi=self.aux_rmi)
            main_loss = self.criterion(cls_out, gts)
            loss = self.alpha * aux_loss + main_loss
            return loss
        else:
            output_dict = {'pred': cls_out}
            return output_dict


class OCRNetASPP(nn.Module):
    """
    OCR net
    """
    def __init__(self, num_classes, criterion=None, **kwargs):
        super(OCRNetASPP, self).__init__()
        self.criterion = criterion
        self.backbone = hrnetv2.get_seg_model(**kwargs)
        self.aspp, aspp_out_ch = get_aspp(self.backbone.high_level_ch,
                                          bottleneck_ch=256, output_stride=8)
        self.ocr = OCR_block(aspp_out_ch, **kwargs)
        self.alpha = 0.4 if 'alpha' not in kwargs else kwargs['alpha']

    def forward(self, inputs):
        assert 'images' in inputs
        x = inputs['images']

        _, _, high_level_features = self.backbone(x)
        aspp = self.aspp(high_level_features)
        cls_out, aux_out, _ = self.ocr(aspp)
        aux_out = scale_as(aux_out, x)
        cls_out = scale_as(cls_out, x)

        if self.training:
            gts = inputs['gts']
            loss = self.alpha * self.criterion(aux_out, gts) + \
                self.criterion(cls_out, gts)
            return loss

        output_dict = {'pred': cls_out}
        return output_dict


class MscaleOCR(nn.Module):
    """
    OCR net
    """
    def __init__(self, num_classes, criterion=None, **kwargs):
        super(MscaleOCR, self).__init__()
        self.criterion = criterion
        self.backbone = hrnetv2.get_seg_model(**kwargs)
        self.ocr = OCR_block(self.backbone.high_level_ch, **kwargs)

        mid_channels = 512 if "mid_channels" not in kwargs else kwargs['mid_channels']
        self.scale_attn = make_attn_head(in_ch=mid_channels, out_ch=1)

        self.alpha = 0.4 if 'alpha' not in kwargs else kwargs['alpha']
        self.n_scales = None if 'n_scales' not in kwargs else kwargs['n_scales']
        self.mscale_lo_scale = 0.5 if 'mscale_lo_scale' not in kwargs else kwargs['mscale_lo_scale']
        self.aux_rmi = False if 'aux_rmi' not in kwargs else kwargs['aux_rmi']
        self.supervised_mscale_wt = 0 if 'supervised_mscale_wt' not in kwargs\
            else kwargs['supervised_mscale_wt']

    @staticmethod
    def fmt_scale(prefix, scale):
        """
        format scale name

        :prefix: a string that is the beginning of the field name
        :scale: a scale value (0.25, 0.5, 1.0, 2.0)
        """

        scale_str = str(float(scale))
        scale_str.replace('.', '')
        return f'{prefix}_{scale_str}x'

    def _fwd(self, x):
        x_size = x.size()[2:]

        _, _, high_level_features = self.backbone(x)
        cls_out, aux_out, ocr_mid_feats = self.ocr(high_level_features)
        attn = self.scale_attn(ocr_mid_feats)

        aux_out = nn.functional.interpolate(aux_out, size=x_size)
        cls_out = nn.functional.interpolate(cls_out, size=x_size)
        attn = nn.functional.interpolate(attn, size=x_size)

        return {'cls_out': cls_out, 'aux_out': aux_out, 'logit_attn': attn}

    def nscale_forward(self, inputs, scales):
        """
        Hierarchical attention, primarily used for getting best inference
        results.

        We use attention at multiple scales, giving priority to the lower
        resolutions. For example, if we have 4 scales {0.5, 1.0, 1.5, 2.0},
        then evaluation is done as follows:

              p_joint = attn_1.5 * p_1.5 + (1 - attn_1.5) * down(p_2.0)
              p_joint = attn_1.0 * p_1.0 + (1 - attn_1.0) * down(p_joint)
              p_joint = up(attn_0.5 * p_0.5) * (1 - up(attn_0.5)) * p_joint

        The target scale is always 1.0, and 1.0 is expected to be part of the
        list of scales. When predictions are done at greater than 1.0 scale,
        the predictions are downsampled before combining with the next lower
        scale.

        Inputs:
          scales - a list of scales to evaluate
          inputs - dict containing 'images', the input, and 'gts', the ground
                   truth mask

        Output:
          If training, return loss, else return prediction + attention
        """
        x_1x = inputs['images']

        assert 1.0 in scales, 'expected 1.0 to be the target scale'
        # Lower resolution provides attention for higher rez predictions,
        # so we evaluate in order: high to low
        scales = sorted(scales, reverse=True)

        pred = None
        aux = None
        output_dict = {}

        for s in scales:
            x = nn.functional.interpolate(x_1x, scale_factor=s, mode='bilinear',
                                          align_corners=True, recompute_scale_factor=True)
            outs = self._fwd(x)
            cls_out = outs['cls_out']
            attn_out = outs['logit_attn']
            aux_out = outs['aux_out']

            output_dict[self.fmt_scale('pred', s)] = cls_out
            if s != 2.0:
                output_dict[self.fmt_scale('attn', s)] = attn_out

            if pred is None:
                pred = cls_out
                aux = aux_out
            elif s >= 1.0:
                # downscale previous
                pred = scale_as(pred, cls_out)
                pred = attn_out * cls_out + (1 - attn_out) * pred
                aux = scale_as(aux, cls_out)
                aux = attn_out * aux_out + (1 - attn_out) * aux
            else:
                # s < 1.0: upscale current
                cls_out = attn_out * cls_out
                aux_out = attn_out * aux_out

                cls_out = scale_as(cls_out, pred)
                aux_out = scale_as(aux_out, pred)
                attn_out = scale_as(attn_out, pred)

                pred = cls_out + (1 - attn_out) * pred
                aux = aux_out + (1 - attn_out) * aux

        if self.training:
            assert 'gts' in inputs
            gts = inputs['gts']
            loss = self.alpha * self.criterion(aux, gts) + self.criterion(pred, gts)
            return loss
        else:
            output_dict['pred'] = pred
            return output_dict

    def two_scale_forward(self, inputs):
        """
        Do we supervised both aux outputs, lo and high scale?
        Should attention be used to combine the aux output?
        Normally we only supervise the combined 1x output

        If we use attention to combine the aux outputs, then
        we can use normal weighting for aux vs. cls outputs
        """
        assert 'images' in inputs
        x_1x = inputs['images']

        x_lo = nn.functional.interpolate(x_1x, scale_factor=self.mscale_lo_scale, mode='bilinear',
                                         align_corners=True, recompute_scale_factor=True)
        lo_outs = self._fwd(x_lo)
        pred_05x = lo_outs['cls_out']
        p_lo = pred_05x
        aux_lo = lo_outs['aux_out']
        logit_attn = lo_outs['logit_attn']
        attn_05x = logit_attn

        hi_outs = self._fwd(x_1x)
        pred_10x = hi_outs['cls_out']
        p_1x = pred_10x
        aux_1x = hi_outs['aux_out']

        p_lo = logit_attn * p_lo
        aux_lo = logit_attn * aux_lo
        p_lo = scale_as(p_lo, p_1x)
        aux_lo = scale_as(aux_lo, p_1x)

        logit_attn = scale_as(logit_attn, p_1x)

        # combine lo and hi predictions with attention
        joint_pred = p_lo + (1 - logit_attn) * p_1x
        joint_aux = aux_lo + (1 - logit_attn) * aux_1x

        if self.training:
            gts = inputs['gts']
            aux_loss = self.criterion(joint_aux, gts, do_rmi=self.aux_rmi)

            # Optionally turn off RMI loss for first epoch to try to work
            # around cholesky errors of singular matrix
            do_rmi_main = True  # cfg.EPOCH > 0
            main_loss = self.criterion(joint_pred, gts, do_rmi=do_rmi_main)
            loss = self.alpha * aux_loss + main_loss

            # Optionally, apply supervision to the multi-scale predictions
            # directly. Turn off RMI to keep things lightweight
            if self.supervised_mscale_wt:
                scaled_pred_05x = scale_as(pred_05x, p_1x)
                loss_lo = self.criterion(scaled_pred_05x, gts, do_rmi=False)
                loss_hi = self.criterion(pred_10x, gts, do_rmi=False)
                loss += self.supervised_mscale_wt * loss_lo
                loss += self.supervised_mscale_wt * loss_hi
            return loss
        else:
            output_dict = {
                'pred': joint_pred,
                'pred_05x': pred_05x,
                'pred_10x': pred_10x,
                'attn_05x': attn_05x,
            }
            return output_dict

    def forward(self, inputs):
        if self.n_scales and not self.training:
            return self.nscale_forward(inputs, self.n_scales)
        return self.two_scale_forward(inputs)


def HRNet(num_classes, criterion, **kwargs):
    return OCRNet(num_classes, criterion=criterion, **kwargs)


def HRNet_Mscale(num_classes, criterion, **kwargs):
    return MscaleOCR(num_classes, criterion=criterion, **kwargs)
