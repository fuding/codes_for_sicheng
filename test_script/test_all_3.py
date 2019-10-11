import time
import os
from collections import OrderedDict
import logging

from utils.test_utils import *
from data import create_dataloader, create_dataset
from models import create_model
from utils import utils

from conf.test.test_conf_awsrn_3 import get_config


########################################
#                  0~1(1440x960, 720x480) -1~1(1440x960,
#  deephdr 282000, 47.7041, 47.7112,        41.683484 dB
#  carn_m  200000, _______, 46.7985
#
#
#######################################


def test_all(save_results=False):
    conf = get_config()

    utils.mkdir(conf.results_dir)
    utils.setup_logger(None, conf.results_dir, 'test_all.log', level=logging.INFO, screen=True)
    logger = logging.getLogger('base')

    if conf.use_cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        print('use cpu')
    else:
        gpu_list = ','.join(str(x) for x in conf.gpu_ids)
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
        print('export CUDA_VISIBLE_DEVICES=' + gpu_list)

    test_set = create_dataset(conf)
    test_loader = create_dataloader(test_set, conf)

    psnr_dict = {}
    for i in range(conf.start_model, conf.end_model, 2000):
        pretained_name = str(i) + '_G.pth'
        prev_pretained_name = os.path.basename(conf.pretrained)
        conf.pretrained = conf.pretrained.replace(prev_pretained_name, pretained_name)
        logger.info('test: {}'.format(conf.pretrained))

        model = create_model(conf)

        test_results = OrderedDict()
        test_results['psnr_own'] = []

        test_start_time = time.time()
        for j, data in enumerate(test_loader):
            model.feed_data(data)
            img_path = data['LR_path'][0]
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            model.test()
            visuals = model.get_current_visuals()

            if save_results:
                output_hdr = visuals['fake']
                save_img_path = os.path.join(conf.results_dir, img_name + '.hdr')
                output_hdr = utils.tensor2img(output_hdr)
                utils.save_results(output_hdr, save_img_path)

            real_img_tm = utils.tensor2img_2(visuals['real_tm'])
            fake_img_tm = utils.tensor2img_2(visuals['fake_tm'])

            psnr_own = compute_psnr(real_img_tm, fake_img_tm)
            logger.info('{:20s} - PSNR: {:.6f} dB'.format(img_name, psnr_own))
            test_results['psnr_own'].append(psnr_own)

        ave_psnr_own = sum(test_results['psnr_own']) / len(test_results['psnr_own'])
        end_time = time.time()-test_start_time
        logger.info('Average PSNR_own: {:.6f} dB, total time: {:.4f}'.format(ave_psnr_own, end_time))
        psnr_dict[i] = ave_psnr_own

    max_psnr_ckp = max(psnr_dict, key=psnr_dict.get)
    max_psnr = psnr_dict[max_psnr_ckp]
    logger.info('max psnr:{}, {:.6}'.format(max_psnr_ckp, max_psnr))


if __name__ == '__main__':
    test_all()