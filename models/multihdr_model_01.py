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



class MultiHDR_Model(BaseModel):
    def __init__(self, conf):
        super(MultiHDR_Model, self).__init__(conf)

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
            else:
                raise NotImplementedError('Loss type [{:s}] is not recognized.'.format(loss_type))
            if conf.loss2 == 'style_loss':
                self.loss2 = StyleLoss()
            else:
                self.loss2 = None

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
        self.LDR_1 = data['LDR_1'].to(self.device)
        self.LDR_2 = data['LDR_2'].to(self.device)
        self.LDR_3 = data['LDR_3'].to(self.device)
        self.HDR_1 = data['HDR_1'].to(self.device)
        self.HDR_2 = data['HDR_2'].to(self.device)
        self.HDR_3 = data['HDR_3'].to(self.device)

        self.ref_HR = data['ref_HR'].to(self.device)

    def optimize_parameters(self, step):
        self.optimizer_G.zero_grad()
        self.fake_H = self.netG([self.LDR_1, self.LDR_2, self.LDR_3], [self.HDR_1, self.HDR_2, self.HDR_3])
        self.fake_H_tm = tonemap_01(self.fake_H)
        self.ref_H_tm = tonemap_01(self.ref_HR)

        l_pix = self.loss(self.fake_H_tm, self.ref_H_tm)
        if self.loss2 is not None:
            l_style = self.loss2(self.fake_H_tm, self.ref_H_tm)
        else:
            l_style = 0
        w = 10
        l_total = l_pix + l_style * w
        l_total.backward()
        self.optimizer_G.step()

        self.log_dict['l_pix'] = l_pix.item()
        if l_style != 0:
            self.log_dict['l_style'] = l_style.item()
            self.log_dict['l_total'] = l_total.item()



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