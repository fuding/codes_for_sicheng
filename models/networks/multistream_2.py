import torch
import torch.nn as nn
import torch.nn.functional as F

from codes.utils.network_utils import *


class Net(nn.Module):
    def __init__(self, in_c=6, out_c=3, fc=64, n_feats=32, num_blocks=4, reduction=4):
        super(Net, self).__init__()

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

        self.dense_block = DenseBlock(n_feats, reduction, attention_blcok=ChannelAttention)

        # self.AWMS = AWMS(n_feats, wn)

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

        self.catattention1 = CatAttention(n_feats*2, n_feats)
        self.catattention2 = CatAttention(fc*4, fc*2)
        self.catattention3 = CatAttention(fc*2, fc)


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

        dense_out = self.dense_block(e_3)

        d1 = self.catattention1(dense_out, e_3)
        d1 = self.d1(d1)

        d2 = self.catattention2(d1, e2_2)
        d2 = self.d2(d2)

        d3 = self.catattention3(d2, e2_1)
        d3 = self.d3(d3)

        out = self.out(d3)

        return F.tanh(out)


class CatAttention(nn.Module):
    def __init__(self, in_c, out_c, reduction=4, mode='basic'):
        super(CatAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Linear(in_c, in_c // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_c // reduction, out_c, bias=False),
        )
        if mode == 'mobile':
            self.sigmoid = Hsigmoid()
        elif mode == 'basic':
            self.sigmoid = nn.Sigmoid()
        else:
            raise NotImplementedError('wrong mode in channel attention block!')

    def forward(self, x, x_cat):
        b, c, _, _ = x.size()
        cat_in = torch.cat((x, x_cat), 1)
        b, c2, _, _ = cat_in.size()
        y = self.avg_pool(cat_in).view(b, c2)
        y = self.se(y)
        y = self.sigmoid(y).view(b, c, 1, 1)

        return x * y.expand_as(x) + x_cat




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
        self.c2 = BasicBlock(n_feats * 2, n_feats, 1, 1, 0)
        self.c3 = BasicBlock(n_feats * 3, n_feats, 1, 1, 0)
        self.c4 = BasicBlock(n_feats * 4, n_feats, 1, 1, 0)
        self.c5 = BasicBlock(n_feats * 5, n_feats, 1, 1, 0)

    def forward(self, x_in, x):
        x_list = []

        cat1 = torch.cat((x_in, x[-1]), 1)
        c1 = self.c1(cat1)
        b1 = self.b1(c1)

        cat2 = torch.cat((c1, b1), 1)
        c2 = self.c2(cat2)
        b2 = self.b2(c2)

        cat3 = torch.cat((c1, b1, b2), 1)
        c3 = self.c3(cat3)
        b3 = self.b3(c3)

        cat5 = torch.cat((c1, b1, b2, b3), 1)
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

        self.c1 = BasicBlock(n_feats * 2, n_feats, 1, 1, 0)
        self.c2 = BasicBlock(n_feats * 2, n_feats, 1, 1, 0)
        self.c3 = BasicBlock(n_feats * 3, n_feats, 1, 1, 0)
        self.c4 = BasicBlock(n_feats * 4, n_feats, 1, 1, 0)
        self.c5 = BasicBlock(n_feats * 5, n_feats, 1, 1, 0)

    def forward(self, x_in, x2):
        x_list = []

        cat1 = torch.cat((x_in, x2[-1]), 1)
        c1 = self.c1(cat1)
        b1 = self.b1(c1)

        cat2 = torch.cat((b1, c1), 1)
        c2 = self.c2(cat2)
        b2 = self.b2(c2)

        cat3 = torch.cat((c1, b1, b2), 1)
        c3 = self.c3(cat3)
        b3 = self.b3(c3)

        cat5 = torch.cat((c1, b1, b2, b3), 1)
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

        self.c1 = BasicBlock(n_feats * 2, n_feats, 1, 1, 0)
        self.c2 = BasicBlock(n_feats * 2, n_feats, 1, 1, 0)
        self.c3 = BasicBlock(n_feats * 3, n_feats, 1, 1, 0)
        self.c4 = BasicBlock(n_feats * 4, n_feats, 1, 1, 0)
        self.c5 = BasicBlock(n_feats * 5, n_feats, 1, 1, 0)

    def forward(self, x_in, x3):
        x_list = []

        cat1 = torch.cat((x_in, x3[-1]), 1)
        c1 = self.c1(cat1)
        b1 = self.b1(c1)

        cat2 = torch.cat((b1, c1), 1)
        c2 = self.c2(cat2)
        b2 = self.b2(c2)

        cat3 = torch.cat((c1, b1, b2), 1)
        c3 = self.c3(cat3)
        b3 = self.b3(c3)

        cat5 = torch.cat((c1, b1, b2, b3), 1)
        c4 = self.c4(cat5)
        b4 = self.b4(c4)

        c5 = torch.cat((b1, b2, b3, b4, c1), 1)
        c5 = self.c5(c5)

        x_list.append(c2)
        x_list.append(c3)
        x_list.append(c4)
        x_list.append(c5)

        return x_list



class DenseBlock(nn.Module):
    def __init__(self, n_feats=32, reduction=4, attention_blcok=None):
        super(DenseBlock, self).__init__()
        self.block1 = FirstBlock(n_feats, reduction=reduction, attention_block=attention_blcok)
        self.block2 = Block2(n_feats, reduction=reduction, attention_block=attention_blcok)
        self.block3 = Block3(n_feats, reduction=reduction, attention_block=attention_blcok)
        self.block4 = Block4(n_feats, reduction=reduction, attention_block=attention_blcok)

        self.attention1 = ChannelAttention(n_feats, reduction=reduction, mode='basic')
        self.attention2 = ChannelAttention(n_feats, reduction=reduction, mode='basic')
        self.attention3 = ChannelAttention(n_feats, reduction=reduction, mode='basic')
        self.attention4 = ChannelAttention(n_feats, reduction=reduction, mode='basic')

        self.fusion = nn.Conv2d(n_feats * 4, n_feats, 3, 1, 1, bias=False)

    def forward(self, x):
        x1 = self.attention1(x)
        x2 = self.attention2(x)
        x3 = self.attention3(x)
        x4 = self.attention4(x)

        x_list1 = self.block1(x1)
        x_list2 = self.block2(x2, x_list1)
        x_list3 = self.block3(x3, x_list2)
        x_list4 = self.block4(x4, x_list3)
        dense_out = torch.cat((x_list1[-1], x_list2[-1], x_list3[-1], x_list4[-1]), 1)
        dense_out = self.fusion(dense_out)

        return dense_out



