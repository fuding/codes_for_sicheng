import os.path
import random
import numpy as np
import cv2
import torch
import torch.utils.data as data
import utils.utils as utils
import utils.data_utils as data_utils



class FiveKDataset(data.Dataset):
    """
    * Read input yuv image (inp), interpolated yuv image (inp_interp) and reference image (ref).
    * The pair is ensured by 'sorted' function, so please check the name convention.
    * Guided filtering is performed to get inp_base and inp_detail
    """

    def __init__(self, conf):
        super(FiveKDataset, self).__init__()

        self.conf = conf
        self.paths_ref = None
        self.paths_inp = None
        self.ref_env = None  # environment for lmdb
        self.inp_env = None

        self.ref_env, self.paths_ref = data_utils.get_image_paths(conf.data_type, conf.dataroot_ref)
        self.inp_env, self.paths_inp = data_utils.get_image_paths(conf.data_type, conf.dataroot_inp)

        assert self.paths_ref, 'Error: reference path is empty.'
        assert self.paths_inp, 'Error: input yuv image path is empty.'
        # assert self.paths_inp_interp, 'Error: input interpolated yuv image path is empty.'

        assert len(self.paths_ref) == len(self.paths_inp), \
            'reference and input yuv image have different number of images - {}, {}.'.format( \
                len(self.paths_ref), len(self.paths_inp))

    def __getitem__(self, index):
        ref_path, inp_path, inp_interp_path = None, None, None


        # get reference image
        ref_path = self.paths_ref[index]
        img_ref = data_utils.read_img_2(self.ref_env, ref_path, dtype=self.conf.type_out)

        # modcrop in the validation / test phase
        # to make sure the size of reference image can be divided by scale

        # if self.opt['phase'] != 'train':
        #     img_H = util.modcrop(img_ref, scale)

        # get input image (inp) and interpolated image (inp_interp)
        inp_path = self.paths_inp[index]
        img_inp = data_utils.read_img_2(self.inp_env, inp_path, dtype=self.conf.type_in)

        # crop, flip, rotate

        H, W, C = img_inp.shape
        H_ref, W_ref, _ = img_ref.shape
        assert H_ref==H and W_ref==W, 'wrong size {}'.format(inp_path)

        if self.conf.preprocess == 'expandnet':
            img_inp, img_ref = data_utils.slice_gauss([img_inp, img_ref], crop_size=(384, 384), precision=(0.1, 1))
            img_inp = cv2.resize(img_inp, (256, 256))
            img_ref = cv2.resize(img_ref, (256, 256))
            img_inp = img_inp[:,:, (2, 1, 0)]
            img_ref = img_ref[:,:, (2, 1, 0)]
        else:
            raise NotImplementedError("!!!!!!")


        # HWC to CHW, numpy to tensor
        img_ref = torch.from_numpy(np.ascontiguousarray(np.transpose(img_ref, (2, 0, 1)))).float()
        img_inp = torch.from_numpy(np.ascontiguousarray(np.transpose(img_inp, (2, 0, 1)))).float()

        return {'LDR': img_inp, 'ref_HDR': img_ref, 'inp_path': inp_path, 'ref_path': ref_path}

    def __len__(self):
        return len(self.paths_inp)