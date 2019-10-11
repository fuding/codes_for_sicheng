import torch
import torch.nn as nn
import torch.nn.functional as F

from codes.utils.network_utils import *


#  AWSRN_SD with encoder-decoder architecture

class AWSRN_SD_ED_C(nn.Module):
    def __init__(self, in_c=6, out_c=3, fc=64, n_feats=16, ks=3, block_feats=128):
        super(AWSRN_SD_ED_C, self).__init__()

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

        self.LFB_D = LFB_D(n_feats, ks, block_feats, wn, act)
        self.AWMS = AWMS(n_feats, wn)

        self.d1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(n_feats*2, fc * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(fc * 2)
        )

        self.d2 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(fc * 4, fc, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(fc)
        )

        self.d3 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(fc * 2, fc, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(fc)
        )

        self.out = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(fc, out_c, 3, 1, 1)
        )

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

        LFB_D = self.LFB_D(e_3)
        AWMS = self.AWMS(LFB_D)

        d1 = torch.cat((AWMS, e_3), 1)
        d1 = self.d1(d1)

        d2 = torch.cat((d1, e2_2), 1)
        d2 = self.d2(d2)

        d3 = torch.cat((d2, e2_1), 1)
        d3 = self.d3(d3)

        out = self.out(d3)

        return F.tanh(out)


class LFB_D(nn.Module):
    def __init__(
        self, n_feats, kernel_size, block_feats, wn, act=nn.ReLU(True)):
        super(LFB_D, self).__init__()
        self.b0 = AWRU(n_feats, kernel_size, block_feats, wn=wn, act=act)
        self.b1 = AWRU(n_feats, kernel_size, block_feats, wn=wn, act=act)
        self.b2 = AWRU(n_feats, kernel_size, block_feats, wn=wn, act=act)
        self.b3 = AWRU(n_feats, kernel_size, block_feats, wn=wn, act=act)
        self.b4 = AWRU(n_feats, kernel_size, block_feats, wn=wn, act=act)
        self.b5 = AWRU(n_feats, kernel_size, block_feats, wn=wn, act=act)
        self.b6 = AWRU(n_feats, kernel_size, block_feats, wn=wn, act=act)
        self.b7 = AWRU(n_feats, kernel_size, block_feats, wn=wn, act=act)
        self.reduction = wn(nn.Conv2d(n_feats*8, n_feats, 3, padding=3//2))
        self.res_scale = Scale(1)
        self.x_scale = Scale(1)

    def forward(self, x):
        x0 = self.b0(x)
        x1 = self.b1(x0)
        x2 = self.b2(x1)
        x3 = self.b3(x2)
        x4 = self.b4(x3)
        x5 = self.b5(x4)
        x6 = self.b6(x5)
        x7 = self.b7(x6)
        res = self.reduction(torch.cat((x0, x1, x2, x3, x4, x5, x6, x7),dim=1))
        return self.res_scale(res) + self.x_scale(x)


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


