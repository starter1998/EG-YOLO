import torch
import torch.nn as nn

from nets.backbone import Backbone, Multi_Concat_Block, Conv,P_CAM,P2PNL_AUX


class AFF(nn.Module):
    '''
    多特征融合 AFF
    '''

    def __init__(self, channels, r=4):
        super(AFF, self).__init__()
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
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)

        xo = x * wei + residual * (1 - wei)
        return xo


class CA_Block(nn.Module):
    def __init__(self, h, w, channel, reduction):
        super(CA_Block, self).__init__()
        self.h = h
        self.w = w
        self.x_avg_pool = nn.AdaptiveAvgPool2d((h, 1))
        self.y_avg_pool = nn.AdaptiveAvgPool2d((1, w))

        self.conv1=    nn.Conv2d(channel, channel // reduction, 1, bias=False)
        self.bn1=    nn.BatchNorm2d(channel // reduction)
        self.le=    nn.LeakyReLU(0.1)
        """self.conv2d_sigmod = nn.Sequential(
            nn.Conv2d(channel // reduction, channel, 1, bias=False),
            nn.Sigmoid()"""
        self.conv2=nn.Conv2d(channel // reduction, channel, 1, bias=False)
        self.sg=nn.Sigmoid()
    def forward(self, x1, x2):
        x1_h = self.x_avg_pool(x1).permute(0, 1, 3, 2)
        x1_w = self.y_avg_pool(x1)
        x2_h = self.x_avg_pool(x2).permute(0, 1, 3, 2)
        x2_w = self.y_avg_pool(x2)

        y1 = torch.cat([x1_h, x1_w], 3)
        #y1 = self.conv2d_bn_nonlinear(y1)
        y1=self.conv1(y1)
        y1=self.bn1(y1)
        y1=self.le(y1)
        y2 = torch.cat([x2_h, x2_w], 3)
        #y2 = self.conv2d_bn_nonlinear(y2)
        y2 = self.conv1(y2)
        y2 = self.bn1(y2)
        y2 = self.le(y2)
        x1_h, x1_w = y1.split([self.h, self.w], 3)
        #x1_h = self.conv2d_sigmod(x1_h)
        x1_h = self.conv2(x1_h)
        x1_h = self.sg(x1_h)
        #x1_w = self.conv2d_sigmod(x1_w)
        x1_w = self.conv2(x1_w)
        x1_w = self.sg(x1_w)
        x2_h, x2_w = y2.split([self.h, self.w], 3)
        #x2_h = self.conv2d_sigmod(x2_h)
        #x2_w = self.conv2d_sigmod(x2_w)
        x2_h = self.conv2(x2_h)
        x2_h = self.sg(x2_h)
        # x1_w = self.conv2d_sigmod(x1_w)
        x2_w = self.conv2(x2_w)
        x2_w = self.sg(x2_w)
        x1 = x1 * x1_h.expand_as(x1) * x1_w.expand_as(x1)
        x2 = x2 * x1_h.expand_as(x2) * x1_w.expand_as(x2)

        return x1, x2


class Channels_Attention(nn.Module):
    def __init__(self, channels, reduction):
        super(Channels_Attention, self).__init__()
        self.avepool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels * 2, channels // reduction),
            nn.LeakyReLU(0.1),
            nn.Linear(channels // reduction, channels * 2),
            nn.Sigmoid())

    def forward(self, x1, x2):
        b, c, _, _ = x1.size()
        x1 = self.avepool(x1).view(b, c, )
        x2 = self.avepool(x2).view(b, c, )
        x = torch.cat([x1, x2], dim=1)
        x = self.fc(x)
        x1, x2 = x.split([c, c], 1)
        x1 = x1.view(b, c, 1, 1)
        x2 = x2.view(b, c, 1, 1)

        return x1, x2


class Spatial_Attention(nn.Module):
    def __init__(self, channels, length):
        super(Spatial_Attention, self).__init__()
        self.conv_3x3 = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=3, stride=2, padding=3 // 2)
        self.resize_bilinear = nn.Upsample([length, length], mode='bilinear')
        self.sigmoid = nn.Sigmoid()
        self.conv_1x1 = nn.Conv2d(2, 2, 1, 1, 0)

    def forward(self, x1, x2):
        avgout1 = torch.mean(x1, dim=1, keepdim=True)
        avgout2 = torch.mean(x2, dim=1, keepdim=True)
        x = torch.cat([avgout1, avgout2], dim=1)
        x = self.conv_3x3(x)
        x = self.resize_bilinear(x)
        x = self.conv_1x1(x)
        x1, x2 = x.split([1, 1], 1)

        return x1, x2


class CSCAv2(nn.Module):
    def __init__(self, channels, length,reduction=1):
        super(CSCAv2, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.CA = CA_Block(length, length, channels, reduction)
        self.SCA = Spatial_Attention(channels, length)
        self.conv1 = nn.Conv2d(channels, channels, 1, 1, 0)
        self.conv2 = nn.Conv2d(1, channels, 1, 1, 0)
        #self.conv3=nn.Conv2d(channels,out_channels,kernel_size=1)

    def forward(self, x1, x2):
        out1, out2 = x1, x2
        c1, c2 = self.CA(x1, x2)
        s1, s2 = self.SCA(x1, x2)
        s1 = self.conv2(s1)
        s2 = self.conv2(s2)
        a1 = c1.expand_as(s1) * s1
        a1 = self.sigmoid(self.conv1(a1))
        a2 = c2.expand_as(s2) * s2
        a2 = self.sigmoid(self.conv1(a2))
        out1 = out1 * a1
        out2 = out2 * a2
        out = torch.cat([out1, out2], dim=1)
        #out=self.conv3(out)
        return out


class SPPCSPC(nn.Module):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(13, 9, 5)):
        super(SPPCSPC, self).__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cv3 = Conv(4 * c_, c_, 1, 1)
        self.cv4 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x):
        x1 = self.cv1(x)
        y1 = self.cv3(torch.cat([m(x1) for m in self.m] + [x1], 1))
        y2 = self.cv2(x)
        return self.cv4(torch.cat((y1, y2), dim=1))

class SPPFCSPC(nn.Module):

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(13, 9, 5)):
        super(SPPFCSPC, self).__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.m1 = nn.MaxPool2d(kernel_size=13, stride=1, padding=13 // 2)
        self.m2 = nn.MaxPool2d(kernel_size=9, stride=1, padding=9 // 2)
        self.m3 = nn.MaxPool2d(kernel_size=5, stride=1, padding=5 // 2)
        self.cv3 = Conv(4 * c_, c_, 1, 1)
        self.cv4 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.m1(x1)
        x3 = self.m2(x2)
        x4 = self.m3(x3)
        y1 = self.cv3(torch.cat((x1, x2, x3, x4), 1))
        y2 = self.cv2(x)
        return self.cv4(torch.cat((y1, y2), dim=1))

class SPPF(nn.Module):
    # SPP结构，5、9、13最大池化核的最大池化。
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_          = c1 // 2
        self.cv1    = Conv(c1, c_, 1, 1)
        self.cv2    = Conv(c_ * 4, c2, 1, 1)
        self.m      = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


def fuse_conv_and_bn(conv, bn):
    fusedconv = nn.Conv2d(conv.in_channels,
                          conv.out_channels,
                          kernel_size=conv.kernel_size,
                          stride=conv.stride,
                          padding=conv.padding,
                          groups=conv.groups,
                          bias=True).requires_grad_(False).to(conv.weight.device)

    w_conv  = conv.weight.clone().view(conv.out_channels, -1)
    w_bn    = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    # fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape).detach())

    b_conv  = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn    = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    # fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)
    fusedconv.bias.copy_((torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn).detach())
    return fusedconv

#---------------------------------------------------#
#   yolo_body
#---------------------------------------------------#
class YoloBody(nn.Module):
    def __init__(self, anchors_mask, num_classes, pretrained=False):
        super(YoloBody, self).__init__()
        #-----------------------------------------------#
        #   定义了不同yolov7-tiny的参数
        #-----------------------------------------------#
        transition_channels = 16
        self.csca1 = CSCAv2(transition_channels*8,40)
        self.csca2 = CSCAv2(transition_channels*4,80)
        self.csca3 = CSCAv2(transition_channels*8,40)
        self.csca4 = CSCAv2(transition_channels*16,20)
        self.aff1 = AFF(transition_channels * 8)
        self.aff2 = AFF(transition_channels * 4)
        self.aff3 = AFF(transition_channels * 8)
        self.aff4 = AFF(transition_channels * 8)
        #self.ss1  = SubSpace(transition_channels * 8)
        #self.ss2  = SubSpace(transition_channels * 16)
        #self.ss3  = SubSpace(transition_channels * 32)
        block_channels      = 16
        panet_channels      = 16
        e                   = 1
        n                   = 2
        ids                 = [-1, -2, -3, -4]
        #-----------------------------------------------#
        #   输入图片是640, 640, 3
        #-----------------------------------------------#

        #---------------------------------------------------#   
        #   生成主干模型
        #   获得三个有效特征层，他们的shape分别是：
        #   80, 80, 512
        #   40, 40, 1024
        #   20, 20, 1024
        #---------------------------------------------------#
        self.backbone   = Backbone(transition_channels, block_channels, n, pretrained=pretrained)
        #self.lfcmf = LMCFM(transition_channels * 8)
        self.upsample   = nn.Upsample(scale_factor=2, mode="nearest")
        #self.sppcspc                 = SPPCSPC(transition_channels * 32, transition_channels * 16)
        #self.sppfcspc                = SPPFCSPC(transition_channels * 32, transition_channels * 8)
        self.sppf                =SPPF(transition_channels * 32, transition_channels * 8, k=5)
        #self.conv_for_P5            = Conv(transition_channels * 16, transition_channels * 8)
        self.conv_for_feat2         = Conv(transition_channels * 16, transition_channels * 8)
        self.conv3_for_upsample1    = Multi_Concat_Block(transition_channels * 16, panet_channels * 4, transition_channels * 8, e=e, n=n, ids=ids)
        self.conv_for_P4            = Conv(transition_channels * 8, transition_channels * 4)
        self.conv_for_feat1         = Conv(transition_channels * 8, transition_channels * 4)
        self.conv3_for_upsample2    = Multi_Concat_Block(transition_channels * 8, panet_channels * 2, transition_channels * 4, e=e, n=n, ids=ids)
        #self.auxp2pnl=P2PNL_AUX(transition_channels * 4)
        self.down_sample1           = Conv(transition_channels * 4, transition_channels * 8, k=3, s=2)
        self.conv3_for_downsample1  = Multi_Concat_Block(transition_channels * 8, panet_channels * 4, transition_channels * 8, e=e, n=n, ids=ids)

        self.down_sample2           = Conv(transition_channels * 8, transition_channels * 8, k=3, s=2)
        self.conv3_for_downsample2  = Multi_Concat_Block(transition_channels * 8, panet_channels * 8, transition_channels * 16, e=e, n=n, ids=ids)
        #self.simam  = simam_module(channels=transition_channels * 8)
        #self.simam1 = simam_module(channels=transition_channels * 16)
        #self.simam2 = simam_module(channels=transition_channels * 32)
        self.rep_conv_1 = Conv(transition_channels * 4, transition_channels * 8, 3, 1)
        self.rep_conv_2 = Conv(transition_channels * 8, transition_channels * 16, 3, 1)
        self.rep_conv_3 = Conv(transition_channels * 16, transition_channels * 32, 3, 1)
        #self.pcam=P_CAM(transition_channels * 4)
        self.yolo_head_P3 = nn.Conv2d(transition_channels * 8, len(anchors_mask[2]) * (5 + num_classes), 1)
        self.yolo_head_P4 = nn.Conv2d(transition_channels * 16, len(anchors_mask[1]) * (5 + num_classes), 1)
        self.yolo_head_P5 = nn.Conv2d(transition_channels * 32, len(anchors_mask[0]) * (5 + num_classes), 1)
    def fuse(self):
        print('Fusing layers... ')
        for m in self.modules():
            if type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)
                delattr(m, 'bn')
                m.forward = m.fuseforward
        return self
    
    def forward(self, x):
        #  backbone
        feat1, feat2, feat3 = self.backbone.forward(x)
        #feat1 = self.lfcmf(feat1)
        P5          = self.sppf(feat3)
        #P5_conv     = self.conv_for_P5(P5)
        #P5_upsample = self.upsample(P5_conv)
        P5_upsample = self.upsample(P5)
        P4          = torch.cat([self.conv_for_feat2(feat2), P5_upsample], 1)
        #P4=self.csca1(self.conv_for_feat2(feat2), P5_upsample)
        #P4 = self.aff1(self.conv_for_feat2(feat2), P5_upsample)
        P4          = self.conv3_for_upsample1(P4)

        P4_conv     = self.conv_for_P4(P4)
        P4_upsample = self.upsample(P4_conv)
        P3          = torch.cat([self.conv_for_feat1(feat1), P4_upsample], 1)
        #P3=self.csca2(self.conv_for_feat1(feat1), P4_upsample)
        #P3 = self.aff2(self.conv_for_feat1(feat1), P4_upsample)
        P3          = self.conv3_for_upsample2(P3)
        #P3=self.auxp2pnl(P3)
        P3_downsample = self.down_sample1(P3)
        #P4 = torch.cat([P3_downsample, P4], 1)
        #P4=self.csca3(P3_downsample, P4)
        P4 = self.aff3(P3_downsample, P4)
        P4 = self.conv3_for_downsample1(P4)

        P4_downsample = self.down_sample2(P4)
        #P5 = torch.cat([P4_downsample, P5], 1)
        #P5=self.csca4(P4_downsample, P5)
        P5 = self.aff4(P4_downsample, P5)
        P5 = self.conv3_for_downsample2(P5)
        #P3 = self.pcam(P3)
        P3 = self.rep_conv_1(P3)
        P4 = self.rep_conv_2(P4)
        P5 = self.rep_conv_3(P5)
        #P3 = self.ss1(P3)
        #P4 = self.ss2(P4)
        #P5 = self.ss3(P5)
        #---------------------------------------------------#
        #   第三个特征层
        #   y3=(batch_size, 75, 80, 80)
        #---------------------------------------------------#
        out2 = self.yolo_head_P3(P3)
        #---------------------------------------------------#
        #   第二个特征层
        #   y2=(batch_size, 75, 40, 40)
        #---------------------------------------------------#
        out1 = self.yolo_head_P4(P4)
        #---------------------------------------------------#
        #   第一个特征层
        #   y1=(batch_size, 75, 20, 20)
        #---------------------------------------------------#
        out0 = self.yolo_head_P5(P5)

        return [out0, out1, out2]
