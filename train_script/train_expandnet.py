import time
import os
import torch
import math
import logging

from conf.train.train_conf_expandnet import get_config
from utils import utils
from models import create_model
from data import create_dataset, create_dataloader


def main():
    conf = get_config()
    utils.mkdir_experiments(conf.experiments_dir)
    utils.setup_logger(None, conf.experiments_dir, 'train', level=logging.INFO, screen=True)
    logger = logging.getLogger('base')

    # set gpu
    gpu_list = ','.join(str(x) for x in conf.gpu_ids)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
    print('export CUDA_VISIBLE_DEVICES=' + gpu_list)

    # set tensorboard
    if conf.use_tb_logger:
        from tensorboardX import SummaryWriter
        utils.mkdir(conf.log_dir)
        tb_logger = SummaryWriter(log_dir=conf.log_dir)

    torch.backends.cudnn.benckmark = True

    # set dataset
    train_dataset = create_dataset(conf)
    train_dataloader = create_dataloader(train_dataset, conf)

    train_size = int(math.ceil(len(train_dataset) / conf.batch_size))
    print('Number of train images: {:,d}, iters: {:,d} per epoch'.format(len(train_dataset), train_size))
    print('Total iters {:,d} for epochs: {:d} '.format(conf.epoch*train_size, conf.epoch))

    if conf.resume:
        resume_state = torch.load(conf.resume)
    else:
        resume_state = None

    # set model
    model = create_model(conf)

    if resume_state:
        start_epoch = resume_state['epoch']
        current_step = resume_state['iter']
        print('Resuming training from epoch: {}, iter: {}.'.format(
            resume_state['epoch'], resume_state['iter']))
        model.resume_training(resume_state)  # handle optimizers and schedulers
    else:
        current_step = 0
        start_epoch = 0


    print('Start training from epoch: {:d}, iter: {:d}'.format(start_epoch, current_step))
    iter_time = time.time()
    for epoch in range(start_epoch, conf.epoch):
        for _, train_data in enumerate(train_dataloader):
            current_step += 1

            model.update_learning_rate()

            # training
            model.feed_data(train_data)
            model.optimize_parameters(current_step)

            # log
            if current_step % conf.print_freq == 0:
                logs = model.get_current_log()
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(
                    epoch, current_step, model.get_current_learning_rate())
                for k, v in logs.items():
                    message += '{:s}: {:.4e} '.format(k, v)
                    # tensorboard logger
                    if conf.use_tb_logger:
                        tb_logger.add_scalar(k, v, current_step)
                message += '200 iters time: %4.4f' % (time.time() - iter_time)
                iter_time = time.time()
                logger.info(message)

            if current_step % conf.save_freq == 0:
                print('Saving models and training states.')
                model.save(current_step)
                model.save_training_state(epoch, current_step)

    logger.info('Saving the final model.')
    model.save('latest')
    logger.info('End of training.')



if __name__ == '__main__':
    main()