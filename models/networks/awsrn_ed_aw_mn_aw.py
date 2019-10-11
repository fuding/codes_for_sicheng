import torch
import torch.nn as nn
import torch.nn.functional as F

from codes.utils.network_utils import *


#  AWSRN_SD with encoder-decoder architecture
#  when skip connecting, use adaptive weight

class AWSRN_ED_AW_MN_AW(nn.Module):
    def __init__(self, in_c=6, out_c=3, fc=64, n_feats=32, ks=3, block_feats=128, num_blocks=4, reduction=4):
        super(AWSRN_ED_AW_MN_AW, self).__init__()

        wn = lambda x: nn.utils.weight_norm(x)
        act = nn.ReLU(True)

        self.e1_1 = nn.Conv2d(in_c, fc, kernel_size=3, stride=2, padding=1)
        self.e1_2 = nn.Sequential(
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(fc, fc * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(fc * 2)
        )

        self.e2_1 = nn.Conv2d(in_c, fc, kernel_size=3, stride=2, padding=1)
        self.e2_2 = nn.Sequential(
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(fc, fc * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(fc * 2)
        )

        self.e3_1 = nn.Conv2d(in_c, fc, kernel_size=3, stride=2, padding=1)
        self.e3_2 = nn.Sequential(
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(fc, fc * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(fc * 2)
        )

        self.e_3 = nn.Sequential(
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(fc * 6, fc, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(fc)
        )

        self.dim_reduction = BasicBlock(fc, n_feats)

        lfb = []
        for i in range(num_blocks):
            lfb.append(LFB_aw(n_feats, wn, exp=6, reduction=reduction, attention_block=None))
        self.LFB = nn.Sequential(*lfb)

        self.AWMS = AWMS(n_feats, wn)

        self.d1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(n_feats, fc * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(fc * 2)
        )

        self.d2 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(fc * 2, fc, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(fc)
        )

        self.d3 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(fc, fc, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(fc)
        )

        self.out = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(fc, out_c, 3, 1, 1)
        )

        self.awms_scale = Scale(1)
        self.e_3_scale = Scale(1)
        self.d1_scale = Scale(1)
        self.e2_2_scale = Scale(1)
        self.d2_scale = Scale(1)
        self.e2_1_scale = Scale(1)

    def forward(self, in_LDRs, in_HDRs):
        in_1 = torch.cat((in_LDRs[0], in_HDRs[0]), dim=1)
        in_2 = torch.cat((in_LDRs[1], in_HDRs[1]), dim=1)
        in_3 = torch.cat((in_LDRs[2], in_HDRs[2]), dim=1)

        e1_1 = self.e1_1(in_1)
        e1_2 = self.e1_2(e1_1)

        e2_1 = self.e2_1(in_2)
        e2_2 = self.e2_2(e2_1)

        e3_1 = self.e3_1(in_3)
        e3_2 = self.e3_2(e3_1)

        e_2 = torch.cat((e1_2, e2_2, e3_2), 1)
        e_3 = self.e_3(e_2)
        e_3 = self.dim_reduction(e_3)

        LFB = self.LFB(e_3)
        AWMS = self.AWMS(LFB)

        d1 = self.awms_scale(AWMS) + self.e_3_scale(e_3)
        d1 = self.d1(d1)

        d2 = self.d1_scale(d1) + self.e2_2_scale(e2_2)
        d2 = self.d2(d2)

        d3 = self.d2_scale(d2) + self.e2_1_scale(e2_1)
        d3 = self.d3(d3)

        out = self.out(d3)

        return F.tanh(out)


class AWRU(nn.Module):
    def __init__(
        self, n_feats, kernel_size, block_feats, wn, res_scale=1, act=nn.ReLU(True)):
        super(AWRU, self).__init__()
        self.res_scale = Scale(1)
        self.x_scale = Scale(1)
        body = []
        body.append(
            wn(nn.Conv2d(n_feats, block_feats, kernel_size, padding=kernel_size//2)))
        body.append(act)
        body.append(
            wn(nn.Conv2d(block_feats, n_feats, kernel_size, padding=kernel_size//2)))

        self.body = nn.Sequential(*body)

    def forward(self, x):
        res = self.res_scale(self.body(x)) + self.x_scale(x)
        return res


class AWMS(nn.Module):
    def __init__(
        self, n_feats, wn):
        super(AWMS, self).__init__()
        out_feats = n_feats
        self.tail_k3 = wn(nn.Conv2d(n_feats, out_feats, 3, padding=3//2, dilation=1))
        self.tail_k5 = wn(nn.Conv2d(n_feats, out_feats, 5, padding=5//2, dilation=1))
        self.tail_k7 = wn(nn.Conv2d(n_feats, out_feats, 7, padding=7//2, dilation=1))
        self.tail_k9 = wn(nn.Conv2d(n_feats, out_feats, 9, padding=9//2, dilation=1))
        self.scale_k3 = Scale(0.25)
        self.scale_k5 = Scale(0.25)
        self.scale_k7 = Scale(0.25)
        self.scale_k9 = Scale(0.25)

    def forward(self, x):
        x0 = self.scale_k3(self.tail_k3(x))
        x1 = self.scale_k5(self.tail_k5(x))
        x2 = self.scale_k7(self.tail_k7(x))
        x3 = self.scale_k9(self.tail_k9(x))
        return x0+x1+x2+x3


