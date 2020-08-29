import numpy as np
from PIL import Image
from matplotlib.colors import hsv_to_rgb

__all__ = ['flow_to_image', 'get_color_pallete']

def flow_to_image(flow, max_flow=256):
    '''
    Converts optic flow to a hsv represenation and then rgb for display
    Scaled by 40 just so I can see it at the moment
    '''
    if max_flow is not None:
        max_flow = max(max_flow, 1.)
    else:
        max_flow = np.max(flow)

    n = 8
    u, v = flow[:, :, 0] * 40, flow[:, :, 1] * 40
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

    out_img = Image.fromarray(npimg.astype('uint8'))
    out_img.putpalette(CITYSPALLETTE)
    return out_img

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
