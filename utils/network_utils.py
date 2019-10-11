import torch
import torch.nn as nn
import torch.nn.functional as F

# conv -> relu
class BasicBlock(nn.Module):
    def __init__(self, in_c, out_c, ks=3, stride=1, padding=1, act_type='relu', norm=False, dilation=1, groups=1, bias=True):
        super(BasicBlock, self).__init__()

        conv_block = []
        conv_block.append(
            nn.Conv2d(in_c, out_c, kernel_size=ks, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        )

        if norm:
            norm = 'bn'
            if norm == 'in':
                conv_block.append(nn.InstanceNorm2d(out_c))
                print(11111111111111)
            elif norm == 'bn':
                conv_block.append(nn.BatchNorm2d(out_c))



        if act_type == 'relu':
            conv_block.append(nn.ReLU(inplace=True))
        elif act_type == 'sigmoid':
            conv_block.append(nn.Sigmoid())

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        x = self.conv_block(x)
        return x


class Scale(nn.Module):

    def __init__(self, init_value=1e-3):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale


class InputAttention(nn.Module):
    def __init__(self, in_c, out_c):
        super(InputAttention, self).__init__()

        self.ia = nn.Sequential(
            BasicBlock(in_c, out_c),
            BasicBlock(out_c, out_c, act_type='sigmoid'),
        )

    def forward(self, x):
        x = self.ia(x)
        return x


class DepthWiseBlock(nn.Module):
    def __init__(self, in_c, ks=3, stride=1, padding=1, bias=False, act_type='ReLU', norm=True):
        super(DepthWiseBlock, self).__init__()
        self.convdw = BasicBlock(in_c=in_c, out_c=in_c, ks=ks, stride=stride,
                                    padding=padding, groups=in_c, bias=bias, act_type=act_type, norm=norm)

    def forward(self, x):
        out = self.convdw(x)
        return out


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class InvertedResidual(nn.Module):
    # inverted residual with linear bottleneck
    # 1x1 3x3dw (se) 1x1
    def __init__(self, in_c, out_c, stride=1, expansion_factor=6, act_type='ReLU', norm=True, attention_block=None, bias=False, reduction=4):
        super(InvertedResidual, self).__init__()
        hidden_c = round(in_c)*expansion_factor


        self.exp_conv = BasicBlock(in_c, hidden_c, ks=1, norm=norm, act_type=act_type, padding=0, stride=1, bias=bias)

        self.dw_conv = DepthWiseBlock(hidden_c, bias=bias, norm=norm, act_type=act_type, stride=stride)

        self.pw_conv = BasicBlock(hidden_c, out_c, 1, norm=norm, act_type='None', padding=0, stride=1, bias=bias)

        if attention_block:
            self.ab = attention_block(hidden_c, reduction)
        else:
            self.ab = Identity()

    def forward(self, x):
        out = self.exp_conv(x)
        out = self.dw_conv(out)
        out = self.ab(out)
        out = self.pw_conv(out)
        out = out + x

        return out


class InvertedResidual_AW(nn.Module):
    # inverted residual with linear bottleneck and adaptive weight
    # 1x1 3x3dw (se) 1x1
    def __init__(self, in_c, out_c, stride=1, expansion_factor=1, act_type='ReLU', norm=True, attention_block=None, bias=False, reduction=4):
        super(InvertedResidual_AW, self).__init__()
        hidden_c = round(in_c)*expansion_factor

        self.exp_conv = BasicBlock(in_c, hidden_c, ks=1, norm=norm, act_type=act_type, padding=0, stride=1, bias=bias)

        self.dw_conv = DepthWiseBlock(hidden_c, bias=bias, norm=norm, act_type=act_type, stride=stride)

        self.pw_conv = BasicBlock(hidden_c, out_c, 1, norm=norm, act_type='None', padding=0, stride=1, bias=bias)

        if attention_block:
            self.ab = attention_block(hidden_c, reduction)
        else:
            self.ab = Identity()

        self.res_scale =Scale(1)
        self.x_scale = Scale(1)

    def forward(self, x):
        out = self.exp_conv(x)
        out = self.dw_conv(out)
        out = self.ab(out)
        out = self.pw_conv(out)

        out = self.res_scale(out) + self.x_scale(x)

        return out


class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.


class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 6.


class ChannelAttention(nn.Module):
    def __init__(self, in_c, reduction=4, mode='mobile'):
        super(ChannelAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Linear(in_c, in_c // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_c // reduction, in_c, bias=False),
        )
        if mode == 'mobile':
            self.sigmoid = Hsigmoid()
        elif mode == 'basic':
            self.sigmoid = nn.Sigmoid()
        else:
            raise NotImplementedError('wrong mode in channel attention block!')

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.se(y)
        y = self.sigmoid(y).view(b, c, 1, 1)

        return x * y.expand_as(x)


class scSE(nn.Module):
    def __init__(self, in_c, reduction=4, mode='mobile'):
        super(scSE, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        if mode == 'mobile':
            self.c_se = nn.Sequential(
                nn.Linear(in_c, in_c // reduction, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(in_c // reduction, in_c, bias=False),
                Hsigmoid()
            )
            self.s_se = nn.Sequential(
                nn.Conv2d(in_c, 1, kernel_size=1, bias=False),
                Hsigmoid()
            )
        elif mode == 'basic':
            self.c_se = nn.Sequential(
                nn.Linear(in_c, in_c // reduction, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(in_c // reduction, in_c, bias=False),
                nn.Sigmoid()
            )
            self.s_se = nn.Sequential(
                nn.Conv2d(in_c, 1, kernel_size=1, bias=False),
                nn.Sigmoid()
            )
        else:
            raise NotImplementedError('wrong mode in channel attention block!')

    def forward(self, x):
        b, c, _, _ = x.size()

        cSE = self.avg_pool(x).view(b, c)
        cSE = self.c_se(cSE).view(b, c, 1, 1)
        cSE = x * cSE.expand_as(x)

        sSE = self.s_se(x)
        sSE = x * sSE

        return cSE + sSE


class scSE_2(nn.Module):
    def __init__(self, in_c, reduction=4, mode='mobile'):
        super(scSE_2, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        if mode == 'mobile':
            self.c_se = nn.Sequential(
                nn.Linear(in_c, in_c // reduction, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(in_c // reduction, in_c, bias=False),
                Hsigmoid()
            )
            self.s_se = nn.Sequential(
                nn.Conv2d(in_c, 1, kernel_size=1, bias=False),
                Hsigmoid()
            )
        elif mode == 'basic':
            self.c_se = nn.Sequential(
                nn.Linear(in_c, in_c // reduction, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(in_c // reduction, in_c, bias=False),
                nn.Sigmoid()
            )
            self.s_se = nn.Sequential(
                nn.Conv2d(in_c, 1, kernel_size=1, bias=False),
                nn.Sigmoid()
            )
        else:
            raise NotImplementedError('wrong mode in channel attention block!')

    def forward(self, x):
        b, c, _, _ = x.size()

        cSE = self.avg_pool(x).view(b, c)
        cSE = self.c_se(cSE).view(b, c, 1, 1)
        cSE = x * cSE.expand_as(x)

        sSE = self.s_se(cSE)
        sSE = x * sSE

        return sSE


class CBAM(nn.Module):
    def __init__(self, in_c, reduction=4):
        super(CBAM, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(in_c, in_c // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_c // reduction, in_c, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

        self.sa = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        c_avg_out = self.ca(self.avg_pool(x))
        c_max_out = self.ca(self.max_pool(x))
        ca = c_avg_out + c_max_out
        ca = x * self.sigmoid(ca)

        s_avg_out = torch.mean(x, dim=1, keepdim=True)
        s_max_out, _ = torch.max(x, dim=1, keepdim=True)
        sa = torch.cat((s_avg_out, s_max_out), dim=1)
        sa = self.sa(sa)

        return ca*sa


class RAM(nn.Module):
    def __init__(self, in_c, reduction=4, mode='basic'):
        super(RAM, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Linear(in_c, in_c // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_c // reduction, in_c, bias=False),
        )

        self.sa = nn.Conv2d(in_c, in_c, kernel_size=3, padding=1, stride=1, groups=in_c, bias=False)

        if mode == 'mobile':
            self.sigmoid = Hsigmoid()
        elif mode == 'basic':
            self.sigmoid = nn.Sigmoid()
        else:
            raise NotImplementedError('wrong mode in attention block!')

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_pool = self.avg_pool(x).view(b, c)
        ca = self.ca(avg_pool).view(b, c, 1, 1)
        sa = self.sa(x)

        out = sa + ca.expand_as(sa)
        return x * self.sigmoid(out)


class LFB(nn.Module):
    def __init__(
        self, n_feats, wn, exp=6, reduction=4, attention_block=None):
        super(LFB, self).__init__()
        self.b0 = InvertedResidual(n_feats, n_feats, expansion_factor=exp, attention_block=attention_block, reduction=reduction)
        self.b1 = InvertedResidual(n_feats, n_feats, expansion_factor=exp, attention_block=attention_block, reduction=reduction)
        self.b2 = InvertedResidual(n_feats, n_feats, expansion_factor=exp, attention_block=attention_block, reduction=reduction)
        self.b3 = InvertedResidual(n_feats, n_feats, expansion_factor=exp, attention_block=attention_block, reduction=reduction)
        self.reduction = wn(nn.Conv2d(n_feats*4, n_feats, 3, padding=3//2))
        self.res_scale = Scale(1)
        self.x_scale = Scale(1)

    def forward(self, x):
        x0 = self.b0(x)
        x1 = self.b1(x0)
        x2 = self.b2(x1)
        x3 = self.b3(x2)
        res = self.reduction(torch.cat([x0, x1, x2, x3],dim=1))

        return self.res_scale(res) + self.x_scale(x)



class LFB_aw(nn.Module):
    def __init__(
        self, n_feats, wn, exp=6, reduction=4, attention_block=None):
        super(LFB_aw, self).__init__()
        self.b0 = InvertedResidual_AW(n_feats, n_feats, expansion_factor=exp, attention_block=attention_block, reduction=reduction)
        self.b1 = InvertedResidual_AW(n_feats, n_feats, expansion_factor=exp, attention_block=attention_block, reduction=reduction)
        self.b2 = InvertedResidual_AW(n_feats, n_feats, expansion_factor=exp, attention_block=attention_block, reduction=reduction)
        self.b3 = InvertedResidual_AW(n_feats, n_feats, expansion_factor=exp, attention_block=attention_block, reduction=reduction)
        self.reduction = wn(nn.Conv2d(n_feats*4, n_feats, 3, padding=3//2))
        self.res_scale = Scale(1)
        self.x_scale = Scale(1)

    def forward(self, x):
        x0 = self.b0(x)
        x1 = self.b1(x0)
        x2 = self.b2(x1)
        x3 = self.b3(x2)
        res = self.reduction(torch.cat([x0, x1, x2, x3],dim=1))

        return self.res_scale(res) + self.x_scale(x)


class SubPixel(nn.Module):
    def __init__(self, in_c, out_c, upscale_factor=2, kernel_size=3, stride=1, padding=1, bias=True, norm=False, act_type='relu'):
        super(SubPixel, self).__init__()

        subpixel = []
        conv = nn.Conv2d(in_c, out_c * (upscale_factor ** 2), kernel_size,
                         stride=stride, padding=padding, bias=bias)
        subpixel.append(conv)
        subpixel.append(nn.PixelShuffle(upscale_factor))

        if norm:
            subpixel.append(nn.BatchNorm2d(out_c))
        if act_type == 'relu':
            subpixel.append(nn.ReLU(inplace=True))

        self.subpixel = nn.Sequential(*subpixel)

    def forward(self, x):
        x = self.subpixel(x)

        return x


class UpCBlock(nn.Module):
    def __init__(self, in_c, out_c, upscale_factor=2, kernel_size=3, stride=1, padding=1, bias=True, norm=False,
                 act_type='relu', mode='nearest'):
        super(UpCBlock, self).__init__()

        self.upsample = nn.Upsample(scale_factor=upscale_factor, mode=mode, align_corners=True)
        self.conv = BasicBlock(in_c, out_c, kernel_size, stride=stride,
                               padding=padding, bias=bias, act_type=act_type, norm=norm)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)

        return x