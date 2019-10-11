import torch
import torch.nn as nn
import torch.nn.functional as F


class AHDRNet(nn.Module):
    def __init__(self, in_c=6, out_c=3, fc=64, growth_rate=32):
        super(AHDRNet, self).__init__()

        self.z1 = BasicBlock(in_c, fc)
        self.z2 = BasicBlock(in_c, fc)
        self.z3 = BasicBlock(in_c, fc)

        self.a1 = AttentionModule(fc)
        self.a3 = AttentionModule(fc)

        self.f0 = BasicBlock(fc*3, fc)
        self.f1 = DRDB(fc, growth_rate)
        self.f2 = DRDB(fc, growth_rate)
        self.f3 = DRDB(fc, growth_rate)
        self.f5 = BasicBlock(fc*3, fc)
        self.f6 = BasicBlock(fc, fc)
        self.f7 = nn.Conv2d(fc, out_c, 1)

    def forward(self, in_LDRs, in_HDRs):
        in_1 = torch.cat((in_LDRs[0], in_HDRs[0]), 1)
        in_2 = torch.cat((in_LDRs[1], in_HDRs[1]), 1)
        in_3 = torch.cat((in_LDRs[2], in_HDRs[2]), 1)

        z1 = self.z1(in_1)
        z2 = self.z2(in_2)
        z3 = self.z3(in_3)

        a1 = self.a1(torch.cat((z1, z2), 1))
        a3 = self.a3(torch.cat((z3, z2), 1))

        z1_2 = z1 * a1
        z3_2 = z3 * a3
        zs = torch.cat((z1_2, z2, z3_2), 1)
        f0 = self.f0(zs)

        f1 = self.f1(f0)
        f2 = self.f2(f1)
        f3 = self.f3(f2)
        f4 = torch.cat((f1, f2, f3), 1)
        f5 = self.f5(f4)
        f6 = self.f6(f5+z2)
        f7 = self.f7(f6)

        return F.tanh(f7)


class BasicBlock(nn.Module):
    def __init__(self, in_c, out_c, ks=3, stride=1, padding=1):
        super(BasicBlock, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=ks, stride=stride, padding=padding),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv_block(x)
        return x


class AttentionModule(nn.Module):
    def __init__(self, fc=64, ks=3, stride=1, padding=1):
        super(AttentionModule, self).__init__()

        self.attention_block = nn.Sequential(
            nn.Conv2d(fc*2, fc, kernel_size=ks, stride=stride, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(fc, fc, kernel_size=ks, stride=stride, padding=padding),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.attention_block(x)
        return x


class DRDB(nn.Module):
    def __init__(self, in_c, fc=32, conv_num=6):
        super(DRDB, self).__init__()

        convs = []
        for i in range(conv_num):
            convs.append(DRDB_C(i*fc+in_c, fc))
        self.convs = nn.Sequential(*convs)

        # Local Feature Fusion
        self.LFF = nn.Sequential(
            nn.Conv2d(conv_num*fc+in_c, in_c, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.convs(x)
        out = self.LFF(out)
        return out + x


class DRDB_C(nn.Module):
    def __init__(self, in_c, out_c=32, dialation=2):
        super(DRDB_C, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=dialation, stride=1, dilation=dialation),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)
