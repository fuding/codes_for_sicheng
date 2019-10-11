import os
import math
import pickle
import random
import numpy as np
import lmdb
import torch
import cv2
import logging


IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tif']




def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def _get_paths_from_lmdb(dataroot):
    env = lmdb.open(dataroot, readonly=True, lock=False, readahead=False, meminit=False)
    keys_cache_file = os.path.join(dataroot, '_keys_cache.p')
    logger = logging.getLogger('base')
    if os.path.isfile(keys_cache_file):
        logger.info('Read lmdb keys from cache: {}'.format(keys_cache_file))
        keys = pickle.load(open(keys_cache_file, "rb"))
    else:
        with env.begin(write=False) as txn:
            logger.info('Creating lmdb keys cache: {}'.format(keys_cache_file))
            keys = [key.decode('ascii') for key, _ in txn.cursor()]
        pickle.dump(keys, open(keys_cache_file, 'wb'))
    paths = sorted([key for key in keys if not key.endswith('.meta')])
    return env, paths



def _get_paths_from_images(path):
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return images


def get_image_paths(data_type, dataroot):
    env, paths = None, None
    if dataroot is not None:
        if data_type == 'lmdb':
            env, paths = _get_paths_from_lmdb(dataroot)
        elif data_type == 'img':
            paths = sorted(_get_paths_from_images(dataroot))
        else:
            raise NotImplementedError('data_type [{:s}] is not recognized.'.format(data_type))
    return env, paths



def read_img_2(env, path, dtype='uint16'):
    # read image by cv2 or from lmdb
    # return: Numpy float32, HWC, BGR, [0,1]
    assert dtype in ['uint8', 'uint10', 'uint16'], 'Invalid data type: {}.'.format(dtype)
    norm = 255. if dtype == 'uint8' else 65535.

    if env is None:  # img
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    else:
        img = _read_lmdb_img(env, path, dtype)

    # while type(img) is None:
        # img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    # img = cv2.cvtColord(img, cv2.COLOR_BGR2YCrCb)
    # img = img[:, :, [0, 2, 1]]
    img = img.astype(np.float32) / norm
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    # some images have 4 channels
    if img.shape[2] > 3:
        img = img[:, :, :3]
    return img


def _read_lmdb_img(env, path):
    with env.begin(write=False) as txn:
        buf = txn.get(path.encode('ascii'))
        buf_meta = txn.get((path + '.meta').encode('ascii')).decode('ascii')
    img_flat = np.frombuffer(buf, dtype=np.uint8)
    H, W, C = [int(s) for s in buf_meta.split(',')]
    img = img_flat.reshape(H, W, C)
    return img



def slice_gauss(img_list,
                precision=None,
                crop_size=None,
                random_size=True,
                ratio=None,
                seed=None):
    """Returns a cropped sample from an image array using :func:`index_gauss`"""
    index = index_gauss(img_list[0], precision, crop_size, random_size, ratio)
    return [img[index] for img in img_list]


def index_gauss(img,
                precision=None,
                crop_size=None,
                random_size=True,
                ratio=None,
                seed=None):
    """Returns indices (Numpy slice) of an image crop sampled spatially using a gaussian distribution.

    Args:
        img (Array): Image as a Numpy array (OpenCV view, hwc-BGR).
        precision (list or tuple, optional): Floats representing the precision
            of the Gaussians (default [1, 4])
        crop_size (list or tuple, optional): Ints representing the crop size
            (default [img_width/4, img_height/4]).
        random_size (bool, optional): If true, randomizes the crop size with
            a minimum of crop_size. It uses an exponential distribution such
            that smaller crops are more likely (default True).
        ratio (float, optional): Keep a constant crop ratio width/height (default None).
        seed (float, optional): Set a seed for np.random.seed() (default None)

    Note:
        - If `ratio` is None then the resulting ratio can be anything.
        - If `random_size` is False and `ratio` is not None, the largest dimension
          dictated by the ratio is adjusted accordingly:

                - `crop_size` is (w=100, h=10) and `ratio` = 9 ==> (w=90, h=10)
                - `crop_size` is (w=100, h=10) and `ratio` = 0.2 ==> (w=100, h=20)

    """
    np.random.seed(seed)
    dims = {"w": img.shape[1], "h": img.shape[0]}
    if precision is None:
        precision = {"w": 1, "h": 4}
    else:
        precision = {"w": precision[0], "h": precision[1]}

    if crop_size is None:
        crop_size = {key: int(dims[key] / 4) for key in dims}
    else:
        crop_size = {"w": crop_size[0], "h": crop_size[1]}

    if ratio is not None:
        ratio = max(ratio, 1e-4)
        if ratio > 1:
            if random_size:
                crop_size['h'] = int(
                    max(crop_size['h'], exponential_size(dims['h'])))
            crop_size['w'] = int(np.round(crop_size['h'] * ratio))
        else:
            if random_size:
                crop_size['w'] = int(
                    max(crop_size['w'], exponential_size(dims['w'])))
            crop_size['h'] = int(np.round(crop_size['w'] / ratio))
    else:
        if random_size:
            crop_size = {
                key: int(max(val, exponential_size(dims[key])))
                for key, val in crop_size.items()
            }

    centers = {
        key: int(
            clamped_gaussian(dim / 2, crop_size[key] / precision[key],
                             min(int(crop_size[key] / 2), dim),
                             max(int(dim - crop_size[key] / 2), 0)))
        for key, dim in dims.items()
    }
    starts = {
        key: max(center - int(crop_size[key] / 2), 0)
        for key, center in centers.items()
    }
    ends = {key: start + crop_size[key] for key, start in starts.items()}
    return np.s_[starts["h"]:ends["h"], starts["w"]:ends["w"], :]


def exponential_size(val):
    return val * (np.exp(-np.random.uniform())) / (np.exp(0) + 1)



def clamped_gaussian(mean, std, min_value, max_value):
    if max_value <= min_value:
        return mean
    factor = 0.99
    while True:
        ret = np.random.normal(mean, std)
        if ret > min_value and ret < max_value:
            break
        else:
            std = std * factor
            ret = np.random.normal(mean, std)

    return ret