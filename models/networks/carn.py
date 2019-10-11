import torch
import torch.nn as nn
import torch.nn.functional as F


class CARN(nn.Module):
    def __init__(self, in_c=6, out_c=3, fc=64):
        super(CARN, self).__init__()

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
            nn.Conv2d(fc*3, fc, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        self.b1 = Block(64, 64)
        self.b2 = Block(64, 64)
        self.b3 = Block(64, 64)
        self.c1 = BasicBlock(64 * 2, 64, 1, 1, 0)
        self.c2 = BasicBlock(64 * 3, 64, 1, 1, 0)
        self.c3 = BasicBlock(64 * 4, 64, 1, 1, 0)

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

        c0 = o0 = e_2
        b1 = self.b1(o0)
        c1 = torch.cat((c0, b1), dim=1)
        o1 = self.c1(c1)

        b2 = self.b2(o1)
        c2 = torch.cat((c1, b2), dim=1)
        o2 = self.c2(c2)

        b3 = self.b3(o2)
        c3 = torch.cat((c2, b3), dim=1)
        o3 = self.c3(c3)

        out = e2_1 + self.d(o3)
        out = self.out(out)

        return F.tanh(out)


class BasicBlock(nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 ksize=3, stride=1, pad=1):
        super(BasicBlock, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, stride, pad),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.body(x)
        return out


class Block(nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 group=1):
        super(Block, self).__init__()

        self.b1 = ResidualBlock(64, 64)
        self.b2 = ResidualBlock(64, 64)
        self.b3 = ResidualBlock(64, 64)
        self.c1 = BasicBlock(64 * 2, 64, 1, 1, 0)
        self.c2 = BasicBlock(64 * 3, 64, 1, 1, 0)
        self.c3 = BasicBlock(64 * 4, 64, 1, 1, 0)

    def forward(self, x):
        c0 = o0 = x

        b1 = self.b1(o0)
        c1 = torch.cat((c0, b1), dim=1)
        o1 = self.c1(c1)

        b2 = self.b2(o1)
        c2 = torch.cat((c1, b2), dim=1)
        o2 = self.c2(c2)

        b3 = self.b3(o2)
        c3 = torch.cat((c2, b3), dim=1)
        o3 = self.c3(c3)

        return o3


class ResidualBlock(nn.Module):
    def __init__(self,
                 in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        )

    def forward(self, x):
        out = self.body(x)
        out = F.relu(out + x)
        return out


class EResidualBlock(nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 group=1):
        super(EResidualBlock, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, groups=group),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, groups=group),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1, 1, 0),
        )

    def forward(self, x):
        out = self.body(x)
        out = F.relu(out + x)
        return out