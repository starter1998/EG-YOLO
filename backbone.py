import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBNReLU(nn.Sequential):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size=3,
        stride=1,
        groups=1,
        activation="ReLU",
    ):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size,
                stride,
                padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
        )


class AuxPath(nn.Module):
    def __init__(self, inchannels, out_channels):
        super(AuxPath, self).__init__()

        self.conv1 = ConvBNReLU(inchannels, out_channels, kernel_size=3,
                                stride=1, groups=out_channels)
        self.conv2 = ConvBNReLU(inchannels, out_channels, kernel_size=3,
                                stride=1, groups=out_channels)
        self.conv3 = ConvBNReLU(inchannels, out_channels, kernel_size=1,
                                stride=1, groups=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        return x


class P2PNL(nn.Module):
    def __init__(self, in_channels, patch=4, bn=True):
        super(P2PNL, self).__init__()

        self.in_channels = in_channels
        self.patch = patch

        self.conv_q = nn.Conv2d(in_channels=in_channels, out_channels=in_channels//2,
                                kernel_size=1, stride=1)
        self.conv_k = nn.Conv2d(in_channels=self.patch**2 * in_channels,
                                out_channels=in_channels//2, kernel_size=1, stride=1)
        self.conv_v = nn.Conv2d(in_channels=self.patch**2 * in_channels,
                                out_channels=in_channels//2, kernel_size=1, stride=1)

        if bn:
            self.conv = nn.Sequential(nn.Conv2d(in_channels=in_channels // 2, out_channels=in_channels,
                                    kernel_size=1, stride=1), nn.BatchNorm2d(self.in_channels), nn.ReLU())
            nn.init.constant_(self.conv[1].weight, 0)
            nn.init.constant_(self.conv[1].bias, 0)
        else:
            self.conv = nn.Conv2d(in_channels=in_channels // 2, out_channels=in_channels,
                                    kernel_size=1, stride=1)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        b, c, h, w = x.shape
        # assert h % self.patch == 0, 'h % patch !=0'
        # assert w % self.patch == 0, 'w % patch !=0'

        stacked_feat = torch.cat([x[...,i::self.patch, j::self.patch]
                                  for i in range(self.patch) for j in range(self.patch)], 1)
        q = self.conv_q(x)      # [b, c/2, h, w]
        k = self.conv_k(stacked_feat)      # [b, c/2, h, w]
        v = self.conv_v(stacked_feat)      # [b, c/2, h, w]

        q = q.reshape(b, self.in_channels // 2, -1)
        k = k.reshape(b, self.in_channels // 2, -1)
        v = v.reshape(b, self.in_channels // 2, -1)

        weight = self.softmax(q.permute(0, 2, 1) @ k)
        v = (weight @ v.permute(0, 2, 1)).permute(0, 2, 1)
        v = v.reshape(b, self.in_channels // 2, h, w)

        output = x + self.conv(v)               # [b, c, h, w]
        return output


class P2PNL_AUX(nn.Module):
    def __init__(self, inchannels, patch=4, bn=True):
        super(P2PNL_AUX, self).__init__()
        self.p2pnl = P2PNL(inchannels, patch)
        self.aux = AuxPath(inchannels, inchannels)

    def forward(self, x):
        x1 = self.p2pnl(x)
        x2 = self.aux(x)

        return x + x1 + x2

class P_CAM(nn.Module):
    '''
    单特征进行通道注意力加权,作用类似SE模块
    '''

    def __init__(self, channels=64, r=4):
        super(P_CAM, self).__init__()
        inter_channels = int(channels // r)

        # 局部注意力
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        # 全局注意力
        self.global_att = P2PNL(channels)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        xl = self.local_att(x)
        xg = self.global_att(x)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        return x * wei



def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k] 
    return p
    
class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=nn.LeakyReLU(0.1, inplace=True)):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv   = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn     = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        #act=False
        self.act    = nn.LeakyReLU(0.1, inplace=True) if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))
    
class Multi_Concat_Block(nn.Module):
    def __init__(self, c1, c2, c3, n=4, e=1, ids=[0]):
        super(Multi_Concat_Block, self).__init__()
        c_ = int(c2 * e)
        
        self.ids = ids
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = nn.ModuleList(
            [Conv(c_ if i ==0 else c2, c2, 3, 1) for i in range(n)]
        )
        self.cv4 = Conv(c_ * 2 + c2 * (len(ids) - 2), c3, 1, 1)

    def forward(self, x):
        x_1 = self.cv1(x)
        x_2 = self.cv2(x)
        
        x_all = [x_1, x_2]
        for i in range(len(self.cv3)):
            x_2 = self.cv3[i](x_2)
            x_all.append(x_2)
            
        out = self.cv4(torch.cat([x_all[id] for id in self.ids], 1))
        return out

class MP(nn.Module):
    def __init__(self, k=2):
        super(MP, self).__init__()
        self.m = nn.MaxPool2d(kernel_size=k, stride=k)

    def forward(self, x):
        return self.m(x)


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def hard_sigmoid(x, inplace: bool = False):
    if inplace:
        return x.add_(3.).clamp_(0., 6.).div_(6.)
    else:
        return F.relu6(x + 3.) / 6.


class SqueezeExcite(nn.Module):
    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None,
                 act_layer=nn.ReLU, gate_fn=hard_sigmoid, divisor=4, **_):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn
        reduced_chs = _make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        x_se = self.avg_pool(x)

        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)

        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x


class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]


class GhostBottleneck(nn.Module):
    def __init__(self, in_chs, mid_chs, out_chs, dw_kernel_size=3, stride=1, act_layer=nn.ReLU, se_ratio=0.):
        super(GhostBottleneck, self).__init__()
        has_se = se_ratio is not None and se_ratio > 0.
        self.stride = stride

        self.ghost1 = GhostModule(in_chs, mid_chs, relu=True)

        if self.stride > 1:
            self.conv_dw = nn.Conv2d(mid_chs, mid_chs, dw_kernel_size, stride=stride,
                                     padding=(dw_kernel_size - 1) // 2,
                                     groups=mid_chs, bias=False)
            self.bn_dw = nn.BatchNorm2d(mid_chs)

        if has_se:
            self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio)
        else:
            self.se = None

        self.ghost2 = GhostModule(mid_chs, out_chs, relu=False)

        if (in_chs == out_chs and self.stride == 1):
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chs, in_chs, dw_kernel_size, stride=stride,
                          padding=(dw_kernel_size - 1) // 2, groups=in_chs, bias=False),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(in_chs, out_chs, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_chs),
            )

    def forward(self, x):
        residual = x

        x = self.ghost1(x)

        if self.stride > 1:
            x = self.conv_dw(x)
            x = self.bn_dw(x)

        if self.se is not None:
            x = self.se(x)

        x = self.ghost2(x)

        x += self.shortcut(residual)
        return x



class Backbone(nn.Module):
    def __init__(self, transition_channels, block_channels, n, pretrained=False):
        super().__init__()
        #-----------------------------------------------#
        #   输入图片是640, 640, 3
        #-----------------------------------------------#
        ids = [-1, -2, -3, -4]
        #self.lglm1 = P2PNL_AUX(transition_channels * 8)
        #self.lglm2 = P2PNL_AUX(transition_channels * 16)
        #self.lglm3 = P2PNL_AUX(transition_channels * 32)
        self.stem = Conv(3, transition_channels * 2, 3, 2)
        self.dark2 = nn.Sequential(
            Conv(transition_channels * 2, transition_channels * 4, 3, 2),
            Multi_Concat_Block(transition_channels * 4, block_channels * 2, transition_channels * 4, n=n, ids=ids),
        )
        self.dark3 = nn.Sequential(
            MP(),
            Multi_Concat_Block(transition_channels * 4, block_channels * 4, transition_channels * 8, n=n, ids=ids),
        )
        self.dark4 = nn.Sequential(
            MP(),
            #Multi_Concat_Block(transition_channels * 8, block_channels * 8, transition_channels * 16, n=n, ids=ids),
            GhostBottleneck(transition_channels * 8, block_channels * 16, transition_channels * 16)
        )
        self.dark5 = nn.Sequential(
            MP(),
            #Multi_Concat_Block(transition_channels * 16, block_channels * 16, transition_channels * 32, n=n, ids=ids),
            GhostBottleneck(transition_channels * 16, block_channels * 16, transition_channels * 32)
        )
        
        if pretrained:
            url = 'https://github.com/bubbliiiing/yolov7-tiny-pytorch/releases/download/v1.0/yolov7_tiny_backbone_weights.pth'
            checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", model_dir="./model_data")
            self.load_state_dict(checkpoint, strict=False)
            print("Load weights from " + url.split('/')[-1])

    def forward(self, x):
        x = self.stem(x)
        x = self.dark2(x)
        #-----------------------------------------------#
        #   dark3的输出为80, 80, 256，是一个有效特征层
        #-----------------------------------------------#
        x = self.dark3(x)
        #x=self.lglm1(x)
        feat1 = x
        #-----------------------------------------------#
        #   dark4的输出为40, 40, 512，是一个有效特征层
        #-----------------------------------------------#
        x = self.dark4(x)
        #x=self.lglm2(x)
        feat2 = x
        #-----------------------------------------------#
        #   dark5的输出为20, 20, 1024，是一个有效特征层
        #-----------------------------------------------#
        x = self.dark5(x)
        #x=self.lglm3(x)
        feat3 = x
        return feat1, feat2, feat3
