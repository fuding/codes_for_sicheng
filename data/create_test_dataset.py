import cv2
import os
import shutil
import numpy as np
import json



def center_crop(x, image_size):
    crop_h, crop_w = image_size
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return cv2.resize(x[max(0,j):min(h,j+crop_h), max(0,i):min(w,i+crop_w)], (crop_w, crop_h))

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

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

if __name__ == '__main__':
    image_dataset = '/home/sicheng/program/DeepHDR/dataset/test/'
    image_list = os.listdir(image_dataset)

    img_id = 1
    exp_dict = {}

    save_root = '/home/sicheng/data/hdr/multi_ldr_hdr_test'
    mkdir(save_root)

    image_size = [960, 1440]

    for image_fold in image_list:
        ldr_1_path = os.path.join(image_dataset, image_fold, 'input_1_aligned.tif')
        ldr_2_path = os.path.join(image_dataset, image_fold, 'input_2_aligned.tif')
        ldr_3_path = os.path.join(image_dataset, image_fold, 'input_3_aligned.tif')
        hdr_path = os.path.join(image_dataset, image_fold, 'ref_hdr_aligned.hdr')
        exp_path = os.path.join(image_dataset, image_fold, 'input_exp.txt')

        assert os.path.exists(ldr_1_path), '{} not exisits'.format(ldr_1_path)
        assert os.path.exists(ldr_2_path), '{} not exisits'.format(ldr_2_path)
        assert os.path.exists(ldr_3_path), '{} not exisits'.format(ldr_3_path)

        LDR_1 = cv2.imread(ldr_1_path).astype(np.uint8)
        LDR_2 = cv2.imread(ldr_2_path).astype(np.uint8)
        LDR_3 = cv2.imread(ldr_3_path).astype(np.uint8)
        HDR = cv2.imread(hdr_path, -1).astype(np.float32)  # read raw values

        LDR_1_crop = center_crop(LDR_1, image_size)
        LDR_2_crop = center_crop(LDR_2, image_size)
        LDR_3_crop = center_crop(LDR_3, image_size)
        HDR_crop = center_crop(HDR, image_size)
        HDR_save = HDR_crop[:, :, ::-1]

        in_exps = np.array(open(exp_path).read().split('\n')[:3]).astype(np.float32)
        exp_dict[img_id] = in_exps.tolist()

        h, w, c = LDR_1_crop.shape

        HDR_save_name = str(img_id) + '_.hdr'
        mkdir(os.path.join(save_root, 'hdr'))
        HDR_save_path = os.path.join(save_root, 'hdr', HDR_save_name)
        radiance_writer(HDR_save_path, HDR_save)

        LDR_1_save_name = str(img_id) + '_1' + '.tif'
        LDR_2_save_name = str(img_id) + '_2' + '.tif'
        LDR_3_save_name = str(img_id) + '_3' + '.tif'
        mkdir(os.path.join(save_root, 'ldr'))
        LDR_1_save_path = os.path.join(save_root, 'ldr', LDR_1_save_name)
        LDR_2_save_path = os.path.join(save_root, 'ldr', LDR_2_save_name)
        LDR_3_save_path = os.path.join(save_root, 'ldr', LDR_3_save_name)

        cv2.imwrite(LDR_1_save_path, LDR_1_crop)
        cv2.imwrite(LDR_2_save_path, LDR_2_crop)
        cv2.imwrite(LDR_3_save_path, LDR_3_crop)

        img_id += 1

    with open(os.path.join(save_root,'exp.json'),'w') as f:
        f.write(json.dumps(exp_dict))






