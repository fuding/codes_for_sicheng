import cv2
import json

from utils.utils import *



if __name__ == '__main__':

    image_dataset = '/home/sicheng/program/DeepHDR/dataset/train/'
    image_list = os.listdir(image_dataset)

    patch_size = 256
    patch_stride = 64
    img_id = 1
    exp_dict = {}

    save_root = '/home/sicheng/data/hdr/multi_ldr_hdr_patch'
    mkdir(save_root)
    for image_fold in image_list:
        ldr_1_path = os.path.join(image_dataset, image_fold, 'input_1_aligned.tif')
        ldr_2_path = os.path.join(image_dataset, image_fold, 'input_2_aligned.tif')
        ldr_3_path = os.path.join(image_dataset, image_fold, 'input_3_aligned.tif')
        hdr_path = os.path.join(image_dataset, image_fold, 'ref_hdr_aligned.hdr')
        exp_path = os.path.join(image_dataset, image_fold, 'input_exp.txt')

        assert os.path.exists(ldr_1_path), '{} not exisits'.format(ldr_1_path)
        assert os.path.exists(ldr_2_path), '{} not exisits'.format(ldr_2_path)
        assert os.path.exists(ldr_3_path), '{} not exisits'.format(ldr_3_path)

        patch_id = 1

        LDR_1 = cv2.imread(ldr_1_path).astype(np.uint8)
        LDR_2 = cv2.imread(ldr_2_path).astype(np.uint8)
        LDR_3 = cv2.imread(ldr_3_path).astype(np.uint8)
        HDR = cv2.imread(hdr_path, -1).astype(np.float32)  # read raw values

        in_exps = np.array(open(exp_path).read().split('\n')[:3]).astype(np.float32)
        exp_dict[img_id] = in_exps.tolist()

        h, w, c = LDR_1.shape
        def write_example(h1, h2, w1, w2, patch_id):
            LDR_1_patch = LDR_1[h1:h2, w1:w2, :]
            LDR_2_patch = LDR_2[h1:h2, w1:w2, :]
            LDR_3_patch = LDR_3[h1:h2, w1:w2, :]
            HDR_patch = HDR[h1:h2, w1:w2, ::-1]

            HDR_save_name = str(img_id)+'_'+str(patch_id).zfill(5)+'.hdr'
            mkdir(os.path.join(save_root, 'hdr'))
            HDR_save_path = os.path.join(save_root, 'hdr', HDR_save_name)
            radiance_writer(HDR_save_path, HDR_patch)

            LDR_1_save_name = str(img_id)+'_'+str(patch_id).zfill(5)+'_1'+'.tif'
            LDR_2_save_name = str(img_id)+'_'+str(patch_id).zfill(5)+'_2'+'.tif'
            LDR_3_save_name = str(img_id)+'_'+str(patch_id).zfill(5)+'_3'+'.tif'
            mkdir(os.path.join(save_root, 'ldr'))
            LDR_1_save_path = os.path.join(save_root, 'ldr', LDR_1_save_name)
            LDR_2_save_path = os.path.join(save_root, 'ldr', LDR_2_save_name)
            LDR_3_save_path = os.path.join(save_root, 'ldr', LDR_3_save_name)

            # h_patch, w_patch, _ = LDR_1_patch.shape
            # scale = 4
            # LDR_1_patch = cv2.resize(LDR_1_patch, (w_patch//scale, h_patch//scale), interpolation=cv2.INTER_LINEAR)
            # LDR_2_patch = cv2.resize(LDR_2_patch, (w_patch//scale, h_patch//scale), interpolation=cv2.INTER_LINEAR)
            # LDR_3_patch = cv2.resize(LDR_3_patch, (w_patch//scale, h_patch//scale), interpolation=cv2.INTER_LINEAR)

            cv2.imwrite(LDR_1_save_path, LDR_1_patch)
            cv2.imwrite(LDR_2_save_path, LDR_2_patch)
            cv2.imwrite(LDR_3_save_path, LDR_3_patch)




        # generate patches
        for h_ in range(0, h - patch_size + 1, patch_stride):
            for w_ in range(0, w - patch_size + 1, patch_stride):
                write_example(h_, h_ + patch_size, w_, w_ + patch_size, patch_id)
                patch_id += 1

        # deal with border patch
        if h % patch_size:
            for w_ in range(0, w - patch_size + 1, patch_stride):
                write_example(h - patch_size, h, w_, w_ + patch_size, patch_id)
                patch_id += 1

        if w % patch_size:
            for h_ in range(0, h - patch_size + 1, patch_stride):
                write_example(h_, h_ + patch_size, w - patch_size, w, patch_id)
                patch_id += 1

        if w % patch_size and h % patch_size:
            write_example(h - patch_size, h, w - patch_size, w, patch_id)
            patch_id += 1


        img_id += 1

        # HDR_save_path = os.path.join(save_root, 'hdr','1.hdr')
        # radiance_writer(HDR_save_path, HDR_save)
        # print(1)

    with open(os.path.join(save_root,'exp.json'),'w') as f:
        f.write(json.dumps(exp_dict))









