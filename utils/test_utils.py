import numpy as np
import math
import skimage.measure


def compute_psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 2.0 # input -1~1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def compute_psnr_01(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0  # input -1~1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))