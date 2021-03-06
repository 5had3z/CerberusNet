# ------------------------------------------------------------------------------
# Generates targets for Panoptic-DeepLab.
# Written by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

import numpy as np

import torch


class PanopticTargetGenerator():
    """
    Generates panoptic training target for Panoptic-DeepLab.
    Annotation is assumed to have Cityscapes format.
    Arguments:
    ----------
        ignore_label: Integer, the ignore label for semantic segmentation.
        rgb2id: Function, panoptic label is encoded in a colored image, this function convert
            color to the corresponding panoptic label.
        thing_list: List, a list of thing classes
        sigma: the sigma for Gaussian kernel.
        ignore_stuff_in_offset: Boolean, whether to ignore stuff region when training the offset
            branch.
        small_instance_area: Integer, indicates largest area for small instances.
        small_instance_weight: Integer, indicates semantic loss weights for small instances.
        ignore_crowd_in_semantic: Boolean, whether to ignore crowd region in semantic segmentation
            branch, crowd region is ignored in the original TensorFlow implementation.
    """
    generated_items = [
        'semantic', 'foreground', 'center', 'offset', 'semantic_mask', 'center_mask', 'offset_mask'
    ]

    def __init__(self, ignore_label, thing_list, sigma=8, ignore_stuff_in_offset=False,
                 small_instance_area=0, small_instance_weight=1, ignore_crowd_in_semantic=False):
        self.ignore_label = ignore_label
        self.thing_list = thing_list
        self.ignore_stuff_in_offset = ignore_stuff_in_offset
        self.small_instance_area = small_instance_area
        self.small_instance_weight = small_instance_weight
        self.ignore_crowd_in_semantic = ignore_crowd_in_semantic

        self.sigma = sigma
        size = 6 * sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3 * sigma + 1, 3 * sigma + 1
        self.g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    @staticmethod
    def rgb2id(color):
        """Converts the color to panoptic label.
        Color is created by `color = [segmentId % 256, segmentId // 256, segmentId // 256 // 256]`.
        Args:
            color: Ndarray or a tuple, color encoded image.
        Returns:
            Panoptic label.
        """
        if isinstance(color, np.ndarray) and len(color.shape) == 3:
            if color.dtype == np.uint8:
                color = color.astype(np.int32)
            return color[:, :, 0] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 2]
        return int(color[0] + 256 * color[1] + 256 * 256 * color[2])

    def __call__(self, panoptic, segments):
        """Generates the training target.
        reference: https://github.com/mcordts/cityscapesScripts/blob/master/\
                    cityscapesscripts/preparation/createPanopticImgs.py
        reference: https://github.com/facebookresearch/detectron2/blob/master/\
                    datasets/prepare_panoptic_fpn.py#L18
        Args:
            panoptic: numpy.array, colored image encoding panoptic label.
            segments: List, a list of dictionary containing information of segments with fields:
                - id: panoptic id, after decoding `panoptic`.
                - category_id: semantic class id.
                - area: segment area.
                - bbox: segment bounding box.
                - iscrowd: crowd region.
        Returns:
            A dictionary with fields:
                - semantic: Tensor, semantic label, shape=(H, W).
                - foreground: Tensor, foreground mask label, shape=(H, W).
                - center: Tensor, center heatmap, shape=(1, H, W).
                - center_points: List, center coordinates, with tuple (y-coord, x-coord).
                - offset: Tensor, offset, shape=(2, H, W), first dim is (offset_y, offset_x).
                - semantic_mask: Tensor, loss weight for semantic prediction, shape=(H, W).
                - center_mask: Tensor, ignore region of center prediction, shape=(H, W),
                    used as weights for center regression 0 is ignore, 1 is has instance.
                    Multiply this mask to loss.
                - offset_mask: Tensor, ignore region of offset prediction, shape=(H, W),
                    used as weights for offset regression 0 is ignore, 1 is has instance.
                    Multiply this mask to loss.
        """
        panoptic = self.rgb2id(panoptic)
        height, width = panoptic.shape[0], panoptic.shape[1]
        semantic = np.zeros_like(panoptic, dtype=np.uint8) + self.ignore_label
        foreground = np.zeros_like(panoptic, dtype=np.uint8)
        center = np.zeros((1, height, width), dtype=np.float32)
        center_pts = []
        offset = np.zeros((2, height, width), dtype=np.float32)
        y_coord = np.ones_like(panoptic, dtype=np.float32)
        x_coord = np.ones_like(panoptic, dtype=np.float32)
        y_coord = np.cumsum(y_coord, axis=0) - 1
        x_coord = np.cumsum(x_coord, axis=1) - 1
        # Generate pixel-wise loss weights
        semantic_mask = np.ones_like(panoptic, dtype=np.uint8)
        # 0: ignore, 1: has instance
        # three conditions for a region to be ignored for instance branches:
        # (1) It is labeled as `ignore_label`
        # (2) It is crowd region (iscrowd=1)
        # (3) (Optional) It is stuff region (for offset branch)
        center_mask = np.zeros_like(panoptic, dtype=np.uint8)
        offset_mask = np.zeros_like(panoptic, dtype=np.uint8)
        for seg in segments:
            cat_id = seg["category_id"]
            if self.ignore_crowd_in_semantic:
                if not seg['iscrowd']:
                    semantic[panoptic == seg["id"]] = cat_id
            else:
                semantic[panoptic == seg["id"]] = cat_id
            if cat_id in self.thing_list:
                foreground[panoptic == seg["id"]] = 1
            if not seg['iscrowd']:
                # Ignored regions are not in `segments`.
                # Handle crowd region.
                center_mask[panoptic == seg["id"]] = 1
                if self.ignore_stuff_in_offset:
                    # Handle stuff region.
                    if cat_id in self.thing_list:
                        offset_mask[panoptic == seg["id"]] = 1
                else:
                    offset_mask[panoptic == seg["id"]] = 1
            if cat_id in self.thing_list:
                # find instance center
                mask_index = np.where(panoptic == seg["id"])
                if len(mask_index[0]) == 0:
                    # the instance is completely cropped
                    continue

                # Find instance area
                ins_area = len(mask_index[0])
                if ins_area < self.small_instance_area:
                    semantic_mask[panoptic == seg["id"]] = self.small_instance_weight

                center_y, center_x = np.mean(mask_index[0]), np.mean(mask_index[1])
                center_pts.append([center_y, center_x])

                # generate center heatmap
                y, x = int(center_y), int(center_x)
                # outside image boundary
                if x < 0 or y < 0 or x >= width or y >= height:
                    continue
                sigma = self.sigma
                # upper left
                ul = int(np.round(x - 3 * sigma - 1)), int(np.round(y - 3 * sigma - 1))
                # bottom right
                br = int(np.round(x + 3 * sigma + 2)), int(np.round(y + 3 * sigma + 2))

                c, d = max(0, -ul[0]), min(br[0], width) - ul[0]
                a, b = max(0, -ul[1]), min(br[1], height) - ul[1]

                cc, dd = max(0, ul[0]), min(br[0], width)
                aa, bb = max(0, ul[1]), min(br[1], height)
                center[0, aa:bb, cc:dd] = np.maximum(
                    center[0, aa:bb, cc:dd], self.g[a:b, c:d])

                # generate offset (2, h, w) -> (y-dir, x-dir)
                offset_y_index = (np.zeros_like(mask_index[0]), mask_index[0], mask_index[1])
                offset_x_index = (np.ones_like(mask_index[0]), mask_index[0], mask_index[1])
                offset[offset_y_index] = center_y - y_coord[mask_index]
                offset[offset_x_index] = center_x - x_coord[mask_index]

        return dict(
            semantic=torch.as_tensor(semantic.astype('long')).unsqueeze(0),
            foreground=torch.as_tensor(foreground.astype('long')).unsqueeze(0),
            center=torch.as_tensor(center.astype(np.float32)).unsqueeze(0),
            center_points=center_pts,
            offset=torch.as_tensor(offset.astype(np.float32)).unsqueeze(0),
            semantic_mask=torch.as_tensor(semantic_mask.astype(np.float32)).unsqueeze(0),
            center_mask=torch.as_tensor(center_mask.astype(np.float32)).unsqueeze(0),
            offset_mask=torch.as_tensor(offset_mask.astype(np.float32)).unsqueeze(0)
        )
