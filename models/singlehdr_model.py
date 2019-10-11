import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from collections import OrderedDict
import logging

from .basemodel import BaseModel
from models import def_network
from utils.utils import *
from .loss import StyleLoss

logger = logging.getLogger('base')



class SingleHDR_Model(BaseModel):
    def __init__(self, conf):
        super(SingleHDR_Model, self).__init__(conf)

        # define network and load pretrained models
        self.netG = def_network(conf).to(self.device)
        self.load()

        if self.is_train:
            self.netG.train()

            loss_type = conf.loss
            if loss_type == 'l1':
                self.loss = nn.L1Loss().to(self.device)
            elif loss_type == 'l2':
                self.loss = nn.MSELoss().to(self.device)
            elif loss_type == 'expandnetloss':
                from loss.expandnet_loss import ExpandNetLoss
                self.loss = ExpandNetLoss().to(self.device)
            else:
                raise NotImplementedError('Loss type [{:s}] is not recognized.'.format(loss_type))


            # optimizers
            self.optimizer_G = optim.Adam(self.netG.parameters(), lr=conf.learning_rate)
            self.optimizers.append(self.optimizer_G)

            # schedulers
            if conf.lr_scheme == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(lr_scheduler.MultiStepLR(optimizer, conf.lr_steps, conf.lr_gamma))
            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

            self.log_dict = OrderedDict()


        # self.print_network()


    def load(self):
        pretrained_model = self.conf.pretrained

        if pretrained_model is not None:
            print('Loading pretrained model for Network: [{:s}] ...'.format(pretrained_model))
            self.load_network(pretrained_model, self.netG)

    def feed_data(self, data):
        self.LDR = data['LDR'].to(self.device)

        self.ref_HDR = data['ref_HDR'].to(self.device)

    def optimize_parameters(self, step):
        self.optimizer_G.zero_grad()
        self.pred_HDR = self.netG(self.LDR)

        l_pix = self.loss(self.pred_HDR, self.ref_HDR)
        l_pix.backward()
        self.optimizer_G.step()

        self.log_dict['l_pix'] = l_pix.item()


    def save(self, iter_step):
        self.save_network(self.netG, 'G', iter_step)

    def get_current_log(self):
        return self.log_dict

    def test(self):
        self.netG.eval()
        with torch.no_grad():
            self.fake_H = self.netG([self.LDR_1, self.LDR_2, self.LDR_3], [self.HDR_1, self.HDR_2, self.HDR_3])
            self.fake_H_tm = tonemap_01(self.fake_H)
            self.ref_H_tm = tonemap_01(self.ref_HR)
        self.netG.train()

    def get_current_visuals(self):
        out_dict = OrderedDict()
        # out_dict['LR'] = self.var_L.detach()[0].float().cpu()
        out_dict['fake_tm'] = self.fake_H_tm.detach()[0].float().cpu()
        out_dict['fake'] = self.fake_H
        out_dict['real_tm'] = self.ref_H_tm.detach()[0].float().cpu()
        return out_dict