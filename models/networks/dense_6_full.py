import torch
import torch.nn as nn
import torch.nn.functional as F

from codes.utils.network_utils import *

class Dense_6_full(nn.Module):
    def __init__(self, in_c=6, out_c=3, fc=64, n_feats=16, group=4, reduction=4):
        super(Dense_6_full, self).__init__()

        self.e1_1 = nn.Sequential(
            nn.Conv2d(in_c, fc, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.e2_1 = nn.Sequential(
            nn.Conv2d(in_c, fc, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.e3_1 = nn.Sequential(
            nn.Conv2d(in_c, fc, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        self.e_2 = nn.Sequential(
            nn.Conv2d(fc*3, n_feats, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        self.block1 = FirstBlock(n_feats, reduction=reduction, attention_block=ChannelAttention)
        self.block2 = Block2(n_feats, reduction=reduction, attention_block=ChannelAttention)
        self.block3 = Block3(n_feats, reduction=reduction, attention_block=ChannelAttention)
        self.block4 = Block4(n_feats, reduction=reduction, attention_block=ChannelAttention)

        self.fusion = nn.Conv2d(n_feats * 4, fc, 3, 1, 1, bias=False)

        self.d = nn.Sequential(
            nn.Conv2d(fc, fc, 3, 1, 1),
            nn.ReLU(inplace=True)
        )

        self.out = nn.Conv2d(fc, out_c, 3, 1, 1)

    def forward(self, in_LDRs, in_HDRs):
        in_1 = torch.cat((in_LDRs[0], in_HDRs[0]), dim=1)
        in_2 = torch.cat((in_LDRs[1], in_HDRs[1]), dim=1)
        in_3 = torch.cat((in_LDRs[2], in_HDRs[2]), dim=1)

        e1_1 = self.e1_1(in_1)
        e2_1 = self.e2_1(in_2)
        e3_1 = self.e3_1(in_3)
        e_cat = torch.cat((e1_1, e2_1, e3_1), 1)
        e_2 = self.e_2(e_cat)

        x_list1 = self.block1(e_2)
        x_list2 = self.block2(x_list1)
        x_list3 = self.block3(x_list1, x_list2)
        x_list4 = self.block4(x_list1, x_list2, x_list3)
        dense_out = torch.cat((x_list1[-1], x_list2[-1], x_list3[-1], x_list4[-1]), 1)
        dense_out = self.fusion(dense_out)

        out = e2_1 + self.d(dense_out)
        out = self.out(out)

        return F.tanh(out)



class FirstBlock(nn.Module):
    def __init__(self, n_feats, exp=6, reduction=4, attention_block=None):
        super(FirstBlock, self).__init__()

        self.b1 = InvertedResidual(n_feats, n_feats, expansion_factor=exp, reduction=reduction, attention_block=attention_block)
        self.b2 = InvertedResidual(n_feats, n_feats, expansion_factor=exp, reduction=reduction, attention_block=attention_block)
        self.b3 = InvertedResidual(n_feats, n_feats, expansion_factor=exp, reduction=reduction, attention_block=attention_block)
        self.b4 = InvertedResidual(n_feats, n_feats, expansion_factor=exp, reduction=reduction, attention_block=attention_block)

        self.c2 = BasicBlock(n_feats * 2, n_feats, 1, 1, 0)
        self.c3 = BasicBlock(n_feats * 3, n_feats, 1, 1, 0)
        self.c4 = BasicBlock(n_feats * 4, n_feats, 1, 1, 0)
        self.c5 = BasicBlock(n_feats * 5, n_feats, 1, 1, 0)

    def forward(self, x):
        x_list = []

        b1 = self.b1(x)
        cat2 = torch.cat((x, b1), 1)
        c2 = self.c2(cat2)

        b2 = self.b2(c2)
        cat3 = torch.cat((cat2, b2), 1)
        c3 = self.c3(cat3)

        b3 = self.b3(c3)
        cat4 = torch.cat((cat3, b3), 1)
        c4 = self.c4(cat4)

        b4 = self.b4(c4)
        cat5 = torch.cat((cat4, b4), 1)
        c5 = self.c5(cat5)

        x_list.append(c2)
        x_list.append(c3)
        x_list.append(c4)
        x_list.append(c5)

        return x_list


class Block2(nn.Module):
    def __init__(self, n_feats, exp=6, reduction=4, attention_block=None):
        super(Block2, self).__init__()

        self.b1 = InvertedResidual(n_feats, n_feats, expansion_factor=exp, reduction=reduction,
                                   attention_block=attention_block)
        self.b2 = InvertedResidual(n_feats, n_feats, expansion_factor=exp, reduction=reduction,
                                   attention_block=attention_block)
        self.b3 = InvertedResidual(n_feats, n_feats, expansion_factor=exp, reduction=reduction,
                                   attention_block=attention_block)
        self.b4 = InvertedResidual(n_feats, n_feats, expansion_factor=exp, reduction=reduction,
                                   attention_block=attention_block)

        self.c1 = BasicBlock(n_feats * 2, n_feats, 1, 1, 0)
        self.c2 = BasicBlock(n_feats * 3, n_feats, 1, 1, 0)
        self.c3 = BasicBlock(n_feats * 4, n_feats, 1, 1, 0)
        self.c4 = BasicBlock(n_feats * 5, n_feats, 1, 1, 0)
        self.c5 = BasicBlock(n_feats * 5, n_feats, 1, 1, 0)

    def forward(self, x):
        x_list = []

        cat1 = torch.cat((x[0], x[-1]), 1)
        c1 = self.c1(cat1)
        b1 = self.b1(c1)

        cat2 = torch.cat((b1, x[1], c1), 1)
        c2 = self.c2(cat2)
        b2 = self.b2(c2)

        cat3 = torch.cat((b2, x[2], c1, b1), 1)
        c3 = self.c3(cat3)
        b3 = self.b3(c3)

        cat5 = torch.cat((b3, x[3], c1, b1, b2), 1)
        c4 = self.c4(cat5)
        b4 = self.b4(c4)

        c5 = torch.cat((b1, b2, b3, b4, c1), 1)
        c5 = self.c5(c5)

        x_list.append(c2)
        x_list.append(c3)
        x_list.append(c4)
        x_list.append(c5)

        return x_list



class Block3(nn.Module):
    def __init__(self, n_feats, exp=6, reduction=4, attention_block=None):
        super(Block3, self).__init__()

        self.b1 = InvertedResidual(n_feats, n_feats, expansion_factor=exp, reduction=reduction,
                                   attention_block=attention_block)
        self.b2 = InvertedResidual(n_feats, n_feats, expansion_factor=exp, reduction=reduction,
                                   attention_block=attention_block)
        self.b3 = InvertedResidual(n_feats, n_feats, expansion_factor=exp, reduction=reduction,
                                   attention_block=attention_block)
        self.b4 = InvertedResidual(n_feats, n_feats, expansion_factor=exp, reduction=reduction,
                                   attention_block=attention_block)

        self.c1 = BasicBlock(n_feats * 3, n_feats, 1, 1, 0)
        self.c2 = BasicBlock(n_feats * 4, n_feats, 1, 1, 0)
        self.c3 = BasicBlock(n_feats * 5, n_feats, 1, 1, 0)
        self.c4 = BasicBlock(n_feats * 6, n_feats, 1, 1, 0)
        self.c5 = BasicBlock(n_feats * 5, n_feats, 1, 1, 0)

    def forward(self, x1, x2):
        x_list = []

        cat1 = torch.cat((x2[0], x2[-1], x1[0]), 1)
        c1 = self.c1(cat1)
        b1 = self.b1(c1)

        cat2 = torch.cat((x2[1], x1[1], b1, c1), 1)
        c2 = self.c2(cat2)
        b2 = self.b2(c2)

        cat3 = torch.cat((x2[2], x1[1], c1, b1, b2), 1)
        c3 = self.c3(cat3)
        b3 = self.b3(c3)

        cat5 = torch.cat((x2[3], x1[3], c1, b1, b2, b3), 1)
        c4 = self.c4(cat5)
        b4 = self.b4(c4)

        c5 = torch.cat((b1, b2, b3, b4, c1), 1)
        c5 = self.c5(c5)

        x_list.append(c2)
        x_list.append(c3)
        x_list.append(c4)
        x_list.append(c5)

        return x_list


class Block4(nn.Module):
    def __init__(self, n_feats, exp=6, reduction=4, attention_block=None):
        super(Block4, self).__init__()

        self.b1 = InvertedResidual(n_feats, n_feats, expansion_factor=exp, reduction=reduction,
                                   attention_block=attention_block)
        self.b2 = InvertedResidual(n_feats, n_feats, expansion_factor=exp, reduction=reduction,
                                   attention_block=attention_block)
        self.b3 = InvertedResidual(n_feats, n_feats, expansion_factor=exp, reduction=reduction,
                                   attention_block=attention_block)
        self.b4 = InvertedResidual(n_feats, n_feats, expansion_factor=exp, reduction=reduction,
                                   attention_block=attention_block)

        self.c1 = BasicBlock(n_feats * 4, n_feats, 1, 1, 0)
        self.c2 = BasicBlock(n_feats * 5, n_feats, 1, 1, 0)
        self.c3 = BasicBlock(n_feats * 6, n_feats, 1, 1, 0)
        self.c4 = BasicBlock(n_feats * 7, n_feats, 1, 1, 0)
        self.c5 = BasicBlock(n_feats * 5, n_feats, 1, 1, 0)

    def forward(self, x1, x2, x3):
        x_list = []

        cat1 = torch.cat((x3[-1], x3[0], x2[0], x1[0]), 1)
        c1 = self.c1(cat1)
        b1 = self.b1(c1)

        cat2 = torch.cat((x3[1], x2[1], x1[1], b1, c1), 1)
        c2 = self.c2(cat2)
        b2 = self.b2(c2)

        cat3 = torch.cat((x3[2], x2[2], x1[1], c1, b1, b2), 1)
        c3 = self.c3(cat3)
        b3 = self.b3(c3)

        cat5 = torch.cat((x3[3], x2[3], x1[3], c1, b1, b2, b3), 1)
        c4 = self.c4(cat5)
        b4 = self.b4(c4)

        c5 = torch.cat((b1, b2, b3, b4, c1), 1)
        c5 = self.c5(c5)

        x_list.append(c2)
        x_list.append(c3)
        x_list.append(c4)
        x_list.append(c5)

        return x_list