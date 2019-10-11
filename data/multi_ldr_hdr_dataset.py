import torch
import torch.utils.data as data
import json

from utils.utils import *

class Multi_LDR_HDR_Dataset(data.Dataset):
    def __init__(self, conf):
        super(Multi_LDR_HDR_Dataset, self).__init__()

        self.conf = conf
        self.LDR_paths = get_image_paths(os.path.join(conf.dataset_dir, 'ldr'))
        self.HDR_paths = get_image_paths(os.path.join(conf.dataset_dir, 'hdr'))
        self.resize = False
        self.size = None

        exp_path = conf.exp_path
        with open(exp_path, 'r') as f:
            self.exp_dict = json.load(f)

        assert self.LDR_paths, 'Error: ldr path is empty.'
        assert self.HDR_paths, 'Error: hdr path is empty.'
        assert len(self.LDR_paths) == len(self.HDR_paths), \
            'HDR and LDR datasets have different number of images - {}, {}.'.format(
                len(self.HDR_paths), len(self.LDR_paths))

    def __getitem__(self, index):
        if not self.conf.is_train:
            self.resize = self.conf.need_resize
            self.size = self.conf.size

        #get HDR image
        HR_path = self.HDR_paths[index]

        ref_HR = read_img(HR_path, self.resize, self.size)

        # get LDR image
        LR_path = self.LDR_paths[index]
        in_LDR_1 = read_img(LR_path, self.resize, self.size)
        in_LDR_2 = read_img(LR_path.replace('1.', '2.'), self.resize, self.size)
        in_LDR_3 = read_img(LR_path.replace('1.', '3.'), self.resize, self.size)

        img_id = os.path.basename(LR_path).split('_')[0]
        exp_list = np.array(self.exp_dict[img_id]).astype(np.float32)
        in_HDR_1 = LDR2HDR(in_LDR_1, 2. ** exp_list[0])
        in_HDR_2 = LDR2HDR(in_LDR_2, 2. ** exp_list[1])
        in_HDR_3 = LDR2HDR(in_LDR_3, 2. ** exp_list[2])

        if self.conf.is_train:
            in_LDR_1, in_LDR_2, in_LDR_3, in_HDR_1, in_HDR_2, in_HDR_3, ref_HR = \
                data_augment([in_LDR_1, in_LDR_2, in_LDR_3, in_HDR_1, in_HDR_2, in_HDR_3, ref_HR])


        # BGR to RGB, HWC to CHW, numpy to tensor
        in_HDR_1 = in_HDR_1[:, :, [2, 1, 0]]
        in_HDR_2 = in_HDR_2[:, :, [2, 1, 0]]
        in_HDR_3 = in_HDR_3[:, :, [2, 1, 0]]
        in_LDR_1 = in_LDR_1[:, :, [2, 1, 0]]
        in_LDR_2 = in_LDR_2[:, :, [2, 1, 0]]
        in_LDR_3 = in_LDR_3[:, :, [2, 1, 0]]
        ref_HR = ref_HR[:, :, [2, 1, 0]]
        in_HDR_1 = torch.from_numpy(np.ascontiguousarray(np.transpose(in_HDR_1, (2, 0, 1)))).float()
        in_HDR_2 = torch.from_numpy(np.ascontiguousarray(np.transpose(in_HDR_2, (2, 0, 1)))).float()
        in_HDR_3 = torch.from_numpy(np.ascontiguousarray(np.transpose(in_HDR_3, (2, 0, 1)))).float()
        in_LDR_1 = torch.from_numpy(np.ascontiguousarray(np.transpose(in_LDR_1, (2, 0, 1)))).float()
        in_LDR_2 = torch.from_numpy(np.ascontiguousarray(np.transpose(in_LDR_2, (2, 0, 1)))).float()
        in_LDR_3 = torch.from_numpy(np.ascontiguousarray(np.transpose(in_LDR_3, (2, 0, 1)))).float()
        ref_HR = torch.from_numpy(np.ascontiguousarray(np.transpose(ref_HR, (2, 0, 1)))).float()


        return {'LDR_1': in_LDR_1, 'LDR_2': in_LDR_2, 'LDR_3': in_LDR_3,
                'HDR_1': in_HDR_1, 'HDR_2': in_HDR_2, 'HDR_3': in_HDR_3,
                'ref_HR': ref_HR, 'LR_path': LR_path, 'HR_path': HR_path}

    def __len__(self):
        return len(self.HDR_paths)