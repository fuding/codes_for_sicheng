import os
import numpy as np
import glob
import cv2
import random
import torch
import logging
from datetime import datetime

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def mkdir_experiments(path):
    mkdir(path)
    mkdir(os.path.join(path, 'models'))
    mkdir(os.path.join(path, 'training_state'))


def radiance_writer(out_path, image):
    with open(out_path, "wb") as f:
        f.write(b"#?RADIANCE\n# Made with Python & Numpy\nFORMAT=32-bit_rle_rgbe\n\n")
        f.write(b"-Y %d +X %d\n" %(image.shape[0], image.shape[1]))

        brightest = np.maximum(np.maximum(image[...,0], image[...,1]), image[...,2])
        mantissa = np.zeros_like(brightest)
        exponent = np.zeros_like(brightest)
        np.frexp(brightest, mantissa, exponent)
        scaled_mantissa = mantissa * 255.0 / brightest
        rgbe = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
        rgbe[...,0:3] = np.around(image[...,0:3] * scaled_mantissa[...,None])
        rgbe[...,3] = np.around(exponent + 128)

        rgbe.flatten().tofile(f)


def get_image_paths(dataroot):
    paths = None

    if os.path.basename(dataroot) == 'ldr':
        paths = sorted(_get_paths_from_ldr_images(dataroot))
    elif os.path.basename(dataroot) == 'hdr':
        paths = sorted(_get_paths_from_hdr_images(dataroot))
    else:
        print('wrong!')

    return paths


def _get_paths_from_ldr_images(path):
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)

    img_path = glob.glob(os.path.join(path, '*_1.tif'))
    img_path = sorted(img_path)

    return img_path


def _get_paths_from_hdr_images(path):
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)

    img_path = glob.glob(os.path.join(path, '*.hdr'))
    img_path = sorted(img_path)

    return img_path


def read_img(path, need_resize=False, size=(720, 480)):
    # read image by cv2
    # return: Numpy float32, HWC, BGR, [-1,1]
    assert os.path.exists(path), '{} is not exists'.format(path)

    if path.endswith('hdr'):
        img = cv2.imread(path, -1).astype(np.float32)
    elif path.endswith('tif'):
        img = cv2.imread(path) / 255.
        img = img.astype(np.float32)
    elif path.endswith('jpg'):
        img = cv2.imread(path) / 255.
        img = img.astype(np.float32)
    else:
        raise NotImplementedError("!!!!!!!!!!!!!!!!!1")

    if need_resize:
        w, h = size
        img = cv2.resize(img, (w, h))

    img = img * 2. - 1
    return img.astype(np.float32)


def read_img_01(path, need_resize=False, size=(720, 480)):
    # read image by cv2
    # return: Numpy float32, HWC, BGR, [-1,1]
    assert os.path.exists(path), '{} is not exists'.format(path)

    if path.endswith('hdr'):
        img = cv2.imread(path, -1).astype(np.float32)
    elif path.endswith('tif'):
        img = cv2.imread(path) / 255.
        img = img.astype(np.float32)
    elif path.endswith('jpg'):
        img = cv2.imread(path) / 255.
        img = img.astype(np.float32)
    else:
        raise NotImplementedError("!!!!!!!!!!!!!!!!!1")

    if need_resize:
        w, h = size
        img = cv2.resize(img, (w, h))

    return img.astype(np.float32)


def data_augment(img_list, hflip=True, rot=True):

    # horizontal flip OR rotate
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip: img = img[:, ::-1, :]
        if vflip: img = img[::-1, :, :]
        if rot90: img = img.transpose(1, 0, 2)
        return img

    return [_augment(img) for img in img_list]


def LDR2HDR(img, expo):  # input/output -1~1
    GAMMA = 2.2
    return (((img + 1) / 2.) ** GAMMA / expo) * 2. - 1


def LDR2HDR_01(img, expo):
    GAMMA = 2.2
    return (img ** GAMMA / expo)


def tonemap(images):  # input/output -1~1
    MU = 5000.  # tunemapping parameter
    return torch.log(1 + MU * (images + 1) / 2.) / np.log(1 + MU) * 2. - 1


def tonemap_01(images):  # input/output 0~1
    MU = 5000.  # tunemapping parameter
    return torch.log(1 + MU * (images)) / np.log(1 + MU)


def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False):
    '''set up logger'''
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter(
        '%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s', datefmt='%y-%m-%d %H:%M:%S')
    log_file = os.path.join(root, phase + '_{}.log'.format(get_timestamp()))
    fh = logging.FileHandler(log_file, mode='w')
    fh.setFormatter(formatter)
    l.setLevel(level)
    l.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        l.addHandler(sh)

def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')


def tensor2img(tensor, min_max=(0, 1)):
    tensor = inverse_transform(tensor)
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp

    img_np = tensor.numpy()
    img_np = np.transpose(img_np, (1, 2, 0))  # HWC

    return img_np


def tensor2img_01(tensor, min_max=(0, 1)):

    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp

    img_np = tensor.numpy()
    img_np = np.transpose(img_np, (1, 2, 0))  # HWC

    return img_np

def tensor2img_2(tensor, min_max=(-1, 1)):
    # tensor = inverse_transform(tensor)
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp

    img_np = tensor.numpy()
    img_np = np.transpose(img_np, (1, 2, 0))  # HWC

    return img_np


def inverse_transform(images):
    return (images + 1.) / 2.


def save_results(img, img_path):
    # input 0~1
    radiance_writer(img_path, img)


def save_resluts_2(img, img_path):
    img = img * 255
    img = img[:, :, [2, 1, 0]]

    cv2.imwrite(img_path, img)