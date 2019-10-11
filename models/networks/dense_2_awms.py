import torch
import torch.nn as nn
import torch.nn.functional as F

from codes.utils.network_utils import *


class Dense_2_AWMS(nn.Module):
    def __init__(self, in_c=6, out_c=3, fc=64, n_feats=32, num_blocks=4, reduction=4):
        super(Dense_2_AWMS, self).__init__()

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

        self.block1 = FirstBlock(n_feats, reduction=reduction)
        self.block2 = NormalBlock(n_feats, reduction=reduction)
        self.block3 = NormalBlock(n_feats, reduction=reduction)
        self.block4 = NormalBlock(n_feats, reduction=reduction)

        self.fusion = nn.Conv2d(n_feats*4, n_feats, 3, 1, 1, bias=False)

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

        x_list1 = self.block1(e_3)
        x_list2 = self.block2(x_list1)
        x_list3 = self.block3(x_list2)
        x_list4 = self.block4(x_list3)
        dense_out = torch.cat((x_list1[-1], x_list2[-1], x_list3[-1], x_list4[-1]), 1)
        dense_out = self.fusion(dense_out)

        awms = self.AWMS(dense_out)

        d1 = self.awms_scale(awms) + self.e_3_scale(e_3)
        d1 = self.d1(d1)

        d2 = self.d1_scale(d1) + self.e2_2_scale(e2_2)
        d2 = self.d2(d2)

        d3 = self.d2_scale(d2) + self.e2_1_scale(e2_1)
        d3 = self.d3(d3)

        out = self.out(d3)

        return F.tanh(out)


class FirstBlock(nn.Module):
    def __init__(self, n_feats, exp=6, reduction=4, attention_block=None):
        super(FirstBlock, self).__init__()

        self.b1 = InvertedResidual(n_feats, n_feats, expansion_factor=exp, reduction=reduction, attention_block=attention_block)
        self.b2 = InvertedResidual(n_feats, n_feats, expansion_factor=exp, reduction=reduction, attention_block=attention_block)
        self.b3 = InvertedResidual(n_feats, n_feats, expansion_factor=exp, reduction=reduction, attention_block=attention_block)
        self.b4 = InvertedResidual(n_feats, n_feats, expansion_factor=exp, reduction=reduction, attention_block=attention_block)

        self.c5 = BasicBlock(n_feats * 4, n_feats, 1, 1, 0)

    def forward(self, x):
        x_list = []

        b1 = self.b1(x)
        b2 = self.b2(b1)
        b3 = self.b3(b2)
        b4 = self.b4(b3)

        x_list.append(b1)
        x_list.append(b2)
        x_list.append(b3)
        x_list.append(b4)

        c5 = torch.cat((b1, b2, b3, b4), 1)
        c5 = self.c5(c5)
        x_list.append(c5)

        return x_list


class NormalBlock(nn.Module):
    def __init__(self, n_feats, exp=6, reduction=4, attention_block=None):
        super(NormalBlock, self).__init__()

        self.b1 = InvertedResidual(n_feats, n_feats, expansion_factor=exp, reduction=reduction,
                                   attention_block=attention_block)
        self.b2 = InvertedResidual(n_feats, n_feats, expansion_factor=exp, reduction=reduction,
                                   attention_block=attention_block)
        self.b3 = InvertedResidual(n_feats, n_feats, expansion_factor=exp, reduction=reduction,
                                   attention_block=attention_block)
        self.b4 = InvertedResidual(n_feats, n_feats, expansion_factor=exp, reduction=reduction,
                                   attention_block=attention_block)

        self.c1 = BasicBlock(n_feats * 2, n_feats, 1, 1, 0)
        self.c2 = BasicBlock(n_feats * 2, n_feats, 1, 1, 0)
        self.c3 = BasicBlock(n_feats * 2, n_feats, 1, 1, 0)
        self.c4 = BasicBlock(n_feats * 2, n_feats, 1, 1, 0)
        self.c5 = BasicBlock(n_feats * 4, n_feats, 1, 1, 0)

    def forward(self, x):
        x_list = []

        c1 = self.c1(torch.cat((x[0], x[-1]), 1))
        b1 = self.b1(c1)
        x_list.append(b1)

        c2 = self.c2(torch.cat((b1, x[1]), 1))
        b2 = self.b2(c2)
        x_list.append(b2)

        c3 = self.c3(torch.cat((b2, x[2]), 1))
        b3 = self.b3(c3)
        x_list.append(b3)

        c4 = self.c4(torch.cat((b3, x[3]), 1))
        b4 = self.b4(c4)
        x_list.append(b4)

        c5 = torch.cat((b1, b2, b3, b4), 1)
        c5 = self.c5(c5)
        x_list.append(c5)

        return x_list


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


