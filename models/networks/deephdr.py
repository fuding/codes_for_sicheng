import torch
import torch.nn as nn
import torch.nn.functional as F



class DeepHDR(nn.Module):
    def __init__(self, in_c=6, out_c=3, fc=64, num_res_blocks=9):
        super(DeepHDR, self).__init__()
        self.e1_1 = nn.Conv2d(in_c, fc, kernel_size=5, stride=2, padding=2)
        self.e1_2 = nn.Sequential(
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(fc, fc*2, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(fc*2)
        )

        self.e2_1 = nn.Conv2d(in_c, fc, kernel_size=5, stride=2, padding=2)
        self.e2_2 = nn.Sequential(
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(fc, fc * 2, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(fc * 2)
        )

        self.e3_1 = nn.Conv2d(in_c, fc, kernel_size=5, stride=2, padding=2)
        self.e3_2 = nn.Sequential(
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(fc, fc * 2, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(fc * 2)
        )

        self.e_3 = nn.Sequential(
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(fc*6, fc*4, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(fc*4)
        )

        res_blocks = []
        for i in range(num_res_blocks):
            res_blocks.append(ResBlock(fc=fc*4, ks=3))
        self.res_blocks = nn.Sequential(*res_blocks)

        self.d1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(fc * 4 * 2, fc * 2, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(fc * 2)
        )

        self.d2 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(fc * 4 * 2, fc, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(fc)
        )

        self.d3 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(fc * 4, fc, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(fc)
        )

        self.out = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(fc, out_c, 1)
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

        d0 = torch.cat((self.res_blocks(e_3), e_3), 1)

        d1 = self.d1(d0)
        d1 = torch.cat((d1, e1_2, e2_2, e3_2), 1)

        d2 = self.d2(d1)
        d2 = torch.cat((d2, e1_1, e2_1, e3_1), 1)

        d3 = self.d3(d2)

        out = self.out(d3)

        return F.tanh(out)



class ResBlock(nn.Module):
    def __init__(self, fc, ks=3):
        super(ResBlock, self).__init__()

        self.act = nn.ReLU(inplace=True)
        self.resblock = nn.Sequential(
            nn.Conv2d(fc, fc, kernel_size=ks, stride=1, padding=1),
            nn.BatchNorm2d(fc),
            nn.ReLU(inplace=True),
            nn.Conv2d(fc, fc, kernel_size=ks, stride=1, padding=1),
            nn.BatchNorm2d(fc)
        )

    def forward(self, x):
        x = self.act(x)
        x = x + self.resblock(x)

        return x



