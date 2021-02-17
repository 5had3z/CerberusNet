import numpy as np
from PIL import Image, ImageDraw
from matplotlib.colors import hsv_to_rgb
from cityscapesscripts.helpers import labels as cs_labels

__all__ = ['flow_to_image', 'get_color_pallete', ]

def flow_to_image(flow, max_flow=256):
    '''
    Converts optic flow to a hsv represenation and then rgb for display
    '''
    if max_flow is not None:
        max_flow = max(max_flow, 1.)
    else:
        max_flow = np.max(flow)

    n = 8
    u, v = flow[:, :, 0], flow[:, :, 1]
    mag = np.sqrt(np.square(u) + np.square(v))
    angle = np.arctan2(v, u)
    im_h = np.mod(angle / (2 * np.pi) + 1, 1)
    im_s = np.clip(mag * n / max_flow, a_min=0, a_max=1)
    im_v = np.clip(n - im_s, a_min=0, a_max=1)
    img = hsv_to_rgb(np.stack([im_h, im_s, im_v], 2))
    return (img * 255).astype(np.uint8)

def get_color_pallete(npimg):
    """Visualize image.

    Parameters
    ----------
    npimg : numpy.ndarray
        Single channel image with shape `H, W, 1`.
    Returns
    -------
    out_img : PIL.Image
        Image with color pallete
    """
    if len(npimg.shape) == 3:
        npimg = npimg[0]    # squeeze first dim
    out_img = Image.fromarray(npimg.astype('uint8'))
    out_img.putpalette(CITYSPALLETTE)
    return out_img

def apply_bboxes(npimg: np.ndarray, bboxes: np.ndarray, labels: np.ndarray) -> Image:
    """Adds colour coded boundary boxes with labels on top of image.

    Parameters
    ----------
    npimg : numpy.ndarray
        Normalised RGB image with shape C, H, W
    bboxes : numpy.ndarray
        List of boundary cxyxwh boxes in normalised form [0,1]
    labels : numpy.ndarray
        List of labels corresponding to the boxes.
    Returns
    -------
    img_out : PIL.Image
        RBG image with boxes applied ontop
    """
    img_out = Image.fromarray((npimg * 255).astype('uint8'))
    draw = ImageDraw.Draw(img_out)
    for bbox, label in zip(bboxes, labels):
        if label < 19:
            bbox_shape = [
                (bbox[0] - bbox[2] / 2) * img_out.width,
                (bbox[1] - bbox[3] / 2) * img_out.height,
                (bbox[0] + bbox[2] / 2) * img_out.width,
                (bbox[1] + bbox[3] / 2) * img_out.height]
            draw.rectangle(bbox_shape, outline=CITYSPALLETTE[label])
    return img_out

def get_panoptic_image(panoptic_img, label_divisor):
    colormap = [list(label.color) for label in cs_labels.labels if label.trainId != 255]

    # Add colormap to label.
    colored_label = np.zeros((panoptic_img.shape[0], panoptic_img.shape[1], 3), dtype=np.uint8)
    taken_colors = set([0, 0, 0])

    def _random_color(base, max_dist=30):
        new_color = base + np.random.randint(low=-max_dist,
                                             high=max_dist + 1,
                                             size=3)
        return tuple(np.maximum(0, np.minimum(255, new_color)))

    for lab in np.unique(panoptic_img):
        if lab // label_divisor == 255:
            continue
        mask = panoptic_img == lab
        base_color = colormap[lab // label_divisor]
        if tuple(base_color) not in taken_colors:
            taken_colors.add(tuple(base_color))
            color = base_color
        else:
            while True:
                color = _random_color(base_color)
                if color not in taken_colors:
                    taken_colors.add(color)
                    break
        colored_label[mask] = color

    return colored_label.astype(dtype=np.uint8)

CITYSPALLETTE = [
    128, 64, 128,
    244, 35, 232,
    70, 70, 70,
    102, 102, 156,
    190, 153, 153,
    153, 153, 153,
    250, 170, 30,
    220, 220, 0,
    107, 142, 35,
    152, 251, 152,
    0, 130, 180,
    220, 20, 60,
    255, 0, 0,
    0, 0, 142,
    0, 0, 70,
    0, 60, 100,
    0, 80, 100,
    0, 0, 230,
    119, 11, 32,
]
