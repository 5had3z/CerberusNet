"""
Example Config Structure

"model" : {
    "name" : "DetrNetSFD",
    "args" : {
        "detr_config" : {
            "num_channels" : 512, "num_classes" : 19, "num_queries" : 100,
            "aux_loss" : false
        },
        "transformer_config" : {
            "dropout" : 0.1, "enc_layers" : 6, "dec_layers" : 6,
            "dim_feedforward" : 2048, "hidden_dim" : 256, "n_heads" : 8
        },
        "position_embedding_config" : {
            "type" : "sine"
        },
        "hrnetv2_config" : {
            "pretrained" : "hrnetv2_w48_imagenet_pretrained.pth",
            "STAGE1" : {
                "NUM_MODULES" : 1, "NUM_BRANCHES" : 1, "BLOCK": "BOTTLENECK",
                "NUM_BLOCKS" : [4], "NUM_CHANNELS" : [64],
                "FUSE_METHOD" : "SUM"
            },
            "STAGE2" : {
                "NUM_MODULES" : 1, "NUM_BRANCHES" : 2, "BLOCK": "BASIC",
                "NUM_BLOCKS" : [4, 4], "NUM_CHANNELS" : [48, 96],
                "FUSE_METHOD" : "SUM"
            },
            "STAGE3" : {
                "NUM_MODULES" : 4, "NUM_BRANCHES" : 3, "BLOCK": "BASIC",
                "NUM_BLOCKS" : [4, 4, 4], "NUM_CHANNELS" : [48, 96, 192],
                "FUSE_METHOD" : "SUM"
            },
            "STAGE4" : {
                "NUM_MODULES" : 3, "NUM_BRANCHES" : 4, "BLOCK": "BASIC",
                "NUM_BLOCKS" : [4, 4, 4, 4], "NUM_CHANNELS" : [48, 96, 192, 384],
                "FUSE_METHOD" : "SUM"
            }
        },
        "correlation_args" : {
            "pad_size" : 4, "max_displacement" : 4,
            "kernel_size" : 1, "stride1" : 1,
            "stride2" : 1, "corr_multiply" : 1
        },
        "flow_est_network" : {
            "type" : "FlowEstimatorLite",
            "args" : {}
        },
        "context_network" : {
            "type" : "ContextNetwork",
            "args" : {}
        },
        "1x1_conv_out" : 32,
        "depth_network" : {
            "type" : "DepthHeadV1",
            "args" : {"inter_ch" : [128, 32]}
        }
    }
}

"""

from typing import List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from nnet_training.loss_functions.UnFlowLoss import flow_warp
from nnet_training.correlation_package.correlation import Correlation

from nnet_training.nnet_models.detr.detr import MLP
from nnet_training.nnet_models.detr.transformer import Transformer
from nnet_training.nnet_models.detr.segmentation import MaskHeadSmallConv
from nnet_training.nnet_models.detr.segmentation import MHAttentionMap
from nnet_training.nnet_models.detr.position_encoding import build_position_encoding

from .hrnetv2 import get_seg_model
from .pwcnet_modules import FlowEstimatorDense, FlowEstimatorLite, ContextNetwork, pwc_conv
from .ocrnet_sfd import DepthHeadV1, scale_as

class DetrSegmHead(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, num_channels, transformer, num_classes, num_queries,
                 aux_loss:bool=False, seg_enable:bool=False):
        """ Initializes the model.
        Parameters:
            num_channels: the number of channels from the backbone. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of
                         objects DETR can detect in a single image.
                         For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(num_channels, hidden_dim, kernel_size=1)
        self.aux_loss = aux_loss

        if seg_enable:
            self.bbox_attention = MHAttentionMap(
                hidden_dim, hidden_dim, self.transformer.nhead, dropout=0.0)
            self.mask_head = MaskHeadSmallConv(
                hidden_dim + self.transformer.nhead, [1024, 512, 256], hidden_dim)

    def forward(self, features: torch.Tensor, pos_embeddings: torch.Tensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor:    batched images, of shape [batch_size x 3 x H x W]
               - samples.mask:      a binary mask of shape [batch_size x H x W], containing 1 on
                                    padded pixels

            It returns a dict with the following elements:
               - "logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "bboxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in
                               [0, 1], relative to the size of each individual image (disregarding
                               possible padding).
                               See PostProcess for information on how to retrieve the unnormalized
                               bounding box.
               - "masks": Segmentation masks for the predicted bboxes
               - "detr_aux": Optional, only returned when auxilary losses are activated. It is a
                                list of dictionnaries containing the two above keys for each decoder
                                layer.
        """
        # Generating mask
        mask_size = list(features.shape)
        del mask_size[1]
        mask = torch.zeros(mask_size, dtype=torch.bool, device=features.device)

        src_proj = self.input_proj(features)
        hs, memory = self.transformer(src_proj, mask, self.query_embed.weight, pos_embeddings)

        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        out = {'logits': outputs_class[-1], 'bboxes': outputs_coord[-1]}
        if self.aux_loss:
            out['detr_aux'] = self._set_aux_loss(outputs_class, outputs_coord)

        if hasattr(self, 'bbox_attention') and hasattr(self, 'mask_head'):
            bbox_mask = self.bbox_attention(hs[-1], memory, mask=mask)
            seg_masks = self.mask_head(
                src_proj, bbox_mask, [features[2], features[1], features[0]])
            out["masks"] = seg_masks.view(
                features[-1].shape[0], self.detr.num_queries,
                seg_masks.shape[-2], seg_masks.shape[-1])

        return out

    @staticmethod
    @torch.jit.unused
    def _set_aux_loss(outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

class DetrNetSFD(nn.Module):
    """
    HRNetV2 with Segmentation + Optic Flow Output
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.modelname = "DetrNetSFD"

        self.backbone = get_seg_model(**kwargs['hrnetv2_config'])
        self.pos_embed = build_position_encoding(
            **kwargs['position_embedding_config'], **kwargs['transformer_config'])

        self.detr = DetrSegmHead(
            transformer=Transformer(**kwargs['transformer_config']), **kwargs['detr_config'])

        if 'correlation_args' in kwargs:
            search_range = kwargs['correlation_args']['max_displacement']
            self.corr = Correlation(**kwargs['correlation_args'])
        else:
            search_range = 4
            self.corr = Correlation(pad_size=search_range, kernel_size=1,
                                    max_displacement=search_range, stride1=1,
                                    stride2=1, corr_multiply=1)

        out_1x1 = 32 if '1x1_conv_out' not in kwargs else kwargs['1x1_conv_out']
        self.conv_1x1 = nn.ModuleList()

        for channels in reversed(kwargs['hrnetv2_config']['STAGE4']['NUM_CHANNELS']):
            self.conv_1x1.append(
                pwc_conv(channels, out_1x1, kernel_size=1, stride=1, dilation=1)
            )

        dim_corr = (search_range * 2 + 1) ** 2
        num_ch_in = out_1x1 + dim_corr + 2 # 1x1 conv, correlation, previous flow
        if 'flow_est_network' in kwargs:
            if kwargs['flow_est_network']['type'] == 'FlowEstimatorDense':
                self.flow_estimator = FlowEstimatorDense(num_ch_in)
            elif kwargs['flow_est_network']['type'] == 'FlowEstimatorLite':
                self.flow_estimator = FlowEstimatorLite(num_ch_in)
            else:
                raise NotImplementedError(kwargs['flow_est_network']['type'])
        else:
            self.flow_estimator = FlowEstimatorDense(num_ch_in)

        if 'context_network' in kwargs:
            if kwargs['context_network']['type'] == 'ContextNetwork':
                self.context_networks = ContextNetwork(self.flow_estimator.feat_dim + 2)
            else:
                raise NotImplementedError(kwargs['context_network']['type'])
        else:
            self.context_networks = ContextNetwork(self.flow_estimator.feat_dim + 2)

        if 'depth_network' in kwargs:
            if kwargs['depth_network']['type'] == 'DepthHeadV1':
                self.depth_head = DepthHeadV1(self.backbone.high_level_ch,
                                              **kwargs['depth_network']['args'])
            else:
                raise NotImplementedError(kwargs['depth_network']['type'])
        else:
            self.depth_head = DepthHeadV1(self.backbone.high_level_ch, 32)

    def flow_forward(self, im1_pyr: List[torch.Tensor], im2_pyr: List[torch.Tensor],
                     final_scale: float):
        '''
        Auxillary forward method that does the flow prediction
        '''
        # output
        flows = []

        # init
        b_size, _, h_x1, w_x1, = im1_pyr[0].size()
        flow = im1_pyr[0].new_zeros((b_size, 2, h_x1, w_x1))

        for level, (im1, im2) in enumerate(zip(im1_pyr, im2_pyr)):
            # warping
            if level == 0:
                im2_warp = im2
            else:
                flow = F.interpolate(flow * 2, scale_factor=2,
                                     mode='bilinear', align_corners=True)
                im2_warp = flow_warp(im2, flow)

            # correlation
            out_corr = self.corr(im1, im2_warp)
            nn.functional.leaky_relu(out_corr, 0.1, inplace=True)

            # concat and estimate flow
            im1_1by1 = self.conv_1x1[level](im1)
            im1_intm, flow_res = self.flow_estimator(
                torch.cat([out_corr, im1_1by1, flow], dim=1))
            flow += flow_res

            flow_fine = self.context_networks(torch.cat([im1_intm, flow], dim=1))
            flow += flow_fine

            flows.append(flow)

        flows = [F.interpolate(flow * final_scale, scale_factor=final_scale,
                               mode='bilinear', align_corners=True) for flow in flows]

        return flows[::-1]

    def forward(self, l_img: torch.Tensor, consistency=True, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Forward method for HRNetV2 with segmentation, flow and depth, returns dictionary \
        of outputs.\n During onnx export, consistency becomes the sequential image argument \
        because onnx export is not compatible with keyword aruments.
        """
        # Backbone Forward pass on image 1 and 2
        high_level_features, im1_pyr = self.backbone(l_img)
        positional_embeddings = self.pos_embed(high_level_features)

        # Segmentation pass with image 1
        forward = self.detr(high_level_features, positional_embeddings)

        # Depth pass with image 1
        forward['depth'] = self.depth_head(high_level_features)

        # We must be ONNX exporting
        if isinstance(consistency, torch.Tensor):
            _, im2_pyr = self.backbone(consistency)
            # Flow pass with image 1
            scale_factor = l_img.size()[-1] // forward['seg'].size()[-1]
            forward['flow'] = self.flow_forward(im1_pyr, im2_pyr, scale_factor)[0]
            del forward['seg_aux']

        if 'l_seq' in kwargs:
            _, im2_pyr = self.backbone(kwargs['l_seq'])

            # Flow pass with image 1
            scale_factor = l_img.size()[-1] // forward['seg'].size()[-1]
            forward['flow'] = self.flow_forward(im1_pyr, im2_pyr, scale_factor)

            if consistency:
                # Flow pass with image 2
                forward['flow_b'] = self.flow_forward(im2_pyr, im1_pyr, scale_factor)

            if 'slam' in kwargs and kwargs['slam'] is True:
                # Another forward brah
                high_level_features, _ = self.backbone(kwargs['l_seq'])

                # Estimate seg and depth
                forward['depth_b'] = self.depth_head(high_level_features)
                forward['seg_b'], _, _ = self.ocr(high_level_features)

                # Rescale
                forward['seg_b'] = scale_as(forward['seg_b'], l_img)
                forward['depth_b'] = scale_as(forward['depth_b'], l_img)

        if 'seg' in forward.keys():
            forward['seg'] = scale_as(forward['seg'], l_img)
        forward['depth'] = scale_as(forward['depth'], l_img)

        return forward
