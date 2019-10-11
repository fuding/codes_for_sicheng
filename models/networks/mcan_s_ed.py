import torch
import torch.nn as nn
import torch.nn.functional as F

from codes.utils.network_utils import *


#  AWSRN_SD with encoder-decoder architecture

class MCAN_S_ED(nn.Module):
    def __init__(self, in_c=6, out_c=3, fc=64, n_feats=16, ks=3, block_feats=128):
        super(MCAN_S_ED, self).__init__()

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

        self.cell1 = first_carn_cell(n_feats)
        self.cell2 = normal_carn_cell(n_feats)
        self.cell3 = normal_carn_cell(n_feats)
        self.fusion = nn.Conv2d(n_feats * 3, n_feats * 3, 3, 1, 1, bias=False)
        self.flatten = BasicBlock(n_feats * 3, n_feats)

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

        x_list1 = self.cell1(e_3)
        x_list2 = self.cell2(x_list1)
        x_list3 = self.cell3(x_list2)
        mcan_out = torch.cat([x_list1[-1], x_list2[-1], x_list3[-1]], dim=1)
        mcan_out = self.fusion(mcan_out)
        mcan_out = self.flatten(mcan_out)

        d1 = mcan_out + e_3
        d1 = self.d1(d1)

        d2 = d1 + e2_2
        d2 = self.d2(d2)

        d3 = d2 + e2_1
        d3 = self.d3(d3)

        out = self.out(d3)

        return F.tanh(out)



class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        self.shrink = nn.AdaptiveAvgPool2d(1)
        modules_body = [
            self.shrink,
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        ]
        self.body = nn.Sequential(*modules_body)

    def forward(self, x, y=None):
        attention = self.body(x)
        return x * attention


class CAResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=16):
        super(CAResidualBlock, self).__init__()
        modules_body = [
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            CALayer(out_channels, reduction)
        ]
        self.body = nn.Sequential(*modules_body)

    def forward(self, x, y=None, mode='origin'):
        out = self.body(x)
        if mode == 'origin':
            out = F.relu(out + x)
        elif mode == 'separate':
            out = F.relu(out + y)
        else:
            assert False, 'mode is wrong !'
        return out


class BaseBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BaseBlock, self).__init__()
        reduction=8
        self.b1 = CAResidualBlock(in_channels, out_channels, reduction=reduction)
        self.b2 = CAResidualBlock(in_channels, out_channels, reduction=reduction)
        self.b3 = CAResidualBlock(in_channels, out_channels, reduction=reduction)

    def forward(self, x):
        assert False, 'Need overwrite.'


class first_carn_cell(nn.Module):
    def __init__(self, filter):
        super(first_carn_cell, self).__init__()
        self.cell1 = first_block(filter, filter)
        self.cell2 = normal_block(filter, filter)
        self.cell3 = normal_block(filter, filter)

    def forward(self, x):
        result = list()
        c0 = o0 = x
        x_list1 = self.cell1(o0)
        x_list2 = self.cell2(x_list1)
        x_list3 = self.cell3(x_list2)
        result.append(x_list1[-1])
        result.append(x_list2[-1])
        result.append(x_list3[-1])
        return result


class normal_carn_cell(nn.Module):
    def __init__(self, filter):
        super(normal_carn_cell, self).__init__()
        self.cell1 = first_block(filter, filter)
        self.cell2 = normal_block(filter, filter)
        self.cell3 = normal_block(filter, filter)
        self.c0 = BasicBlock(filter * 1 + filter, filter, 1, 1, 0)

    def forward(self, x_list):
        result = list()
        c0 = torch.cat([x_list[-1], x_list[0]], dim=1)
        o0 = self.c0(c0)
        x_list1 = self.cell1(o0)
        x_list2 = self.cell2(x_list1, x_list[1])
        x_list3 = self.cell3(x_list2, x_list[2])
        result.append(x_list1[-1])
        result.append(x_list2[-1])
        result.append(x_list3[-1])
        return result


class first_block(BaseBlock):
    def __init__(self, in_channels, out_channels, group=1, mode='None'):
        super(first_block, self).__init__(in_channels, out_channels)
        self.c1 = BasicBlock(in_channels*2, out_channels, 1, 1, 0)
        self.c2 = BasicBlock(in_channels*3, out_channels, 1, 1, 0)
        self.c3 = BasicBlock(in_channels*4, out_channels, 1, 1, 0)

    def forward(self, x):
        result = list()
        c0 = o0 = x
        b1 = self.b1(o0)
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.c1(c1)
        result.append(o1)
        b2 = self.b2(o1)
        c2 = torch.cat([c0, b1, b2], dim=1)
        o2 = self.c2(c2)
        result.append(o2)
        b3 = self.b3(o2)
        c3 = torch.cat([c0, b1, b2, b3], dim=1)
        o3 = self.c3(c3)
        result.append(o3)
        return result


class normal_block(BaseBlock):
    def __init__(self, in_channels, out_channels):
        super(normal_block, self).__init__(in_channels, out_channels)
        self.c0 = BasicBlock(in_channels * 1 + in_channels, out_channels, 1, 1, 0)
        self.c0_fix = BasicBlock(in_channels * 2 + in_channels, out_channels, 1, 1, 0)
        self.c1 = BasicBlock(in_channels * 2 + in_channels, out_channels, 1, 1, 0)
        self.c2 = BasicBlock(in_channels * 3 + in_channels, out_channels, 1, 1, 0)
        self.c3 = BasicBlock(in_channels * 4, out_channels, 1, 1, 0)

    def forward(self, x_list, parallel_input=None):
        result = list()
        if parallel_input is None:
            c0 = torch.cat([x_list[-1], x_list[0]], dim=1)
            o0 = self.c0(c0)
        else:
            c0 = torch.cat([x_list[-1], x_list[0], parallel_input], dim=1)
            o0 = self.c0_fix(c0)
        b1 = self.b1(o0)
        c1 = torch.cat([o0, b1, x_list[1]], dim=1)
        o1 = self.c1(c1)
        result.append(o1)
        b2 = self.b2(o1)
        c2 = torch.cat([o0, b1, b2, x_list[2]], dim=1)
        o2 = self.c2(c2)
        result.append(o2)
        b3 = self.b3(o2)
        c3 = torch.cat([o0, b1, b2, b3], dim=1)
        o3 = self.c3(c3)
        result.append(o3)
        return result





