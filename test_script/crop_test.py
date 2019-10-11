import time
import os
from collections import OrderedDict
import logging

from utils.test_utils import *
from data import create_dataloader, create_dataset
from conf.test.test_conf_deephdr import get_config
from models import create_model
from utils import utils

def test(save_results=True):
    conf = get_config()


    utils.mkdir(conf.results_dir)
    utils.setup_logger(None, conf.results_dir, 'test.log', level=logging.INFO, screen=True)
    logger = logging.getLogger('base')

    gpu_list = ','.join(str(x) for x in conf.gpu_ids)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
    print('export CUDA_VISIBLE_DEVICES=' + gpu_list)

    test_set = create_dataset(conf)
    test_loader = create_dataloader(test_set, conf)

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

        real_img_tm = utils.tensor2img(visuals['real_tm'])
        fake_img_tm = utils.tensor2img(visuals['fake_tm'])

        psnr_own = compute_psnr(real_img_tm, fake_img_tm)
        logger.info('{:20s} - PSNR: {:.6f} dB'.format(img_name, psnr_own))
        test_results['psnr_own'].append(psnr_own)

    ave_psnr_own = sum(test_results['psnr_own']) / len(test_results['psnr_own'])
    end_time = time.time()-test_start_time
    logger.info('Average PSNR_own: {:.6f} dB, total time: {:.4f}'.format(ave_psnr_own, end_time))


if __name__ == '__main__':
    test()