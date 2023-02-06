import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from inference.models.pico_det import CSPLayer
from inference.models.grasp_model import GraspModel, ResidualBlock
from inference.models.pp_lcnet import DepthwiseSeparable
from inference.models.attention import CoordAtt
from inference.models.duc import DenseUpsamplingConvolution


class sa_layer(nn.Module):
    """Constructs a Channel Spatial Group module.
    Args:
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, groups=64):
        super(sa_layer, self).__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.cweight = Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))
        self.cbias = Parameter(torch.ones(1, channel // (2 * groups), 1, 1))
        self.sweight = Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))
        self.sbias = Parameter(torch.ones(1, channel // (2 * groups), 1, 1))

        self.sigmoid = nn.Sigmoid()
        self.gn = nn.GroupNorm(channel // (2 * groups), channel // (2 * groups))

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape

        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        # flatten
        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x):
        b, c, h, w = x.shape
        print("shape x {}".format(x.shape))
        x = x.reshape(b * self.groups, -1, h, w)
        print("shape x {}".format(x.shape))
        x_0, x_1 = x.chunk(2, dim=1)

        # channel attention
        xn = self.avg_pool(x_0)
        xn = self.cweight * xn + self.cbias
        xn = x_0 * self.sigmoid(xn)

        # spatial attention
        xs = self.gn(x_1)
        xs = self.sweight * xs + self.sbias
        xs = x_1 * self.sigmoid(xs)

        # concatenate along channel axis
        out = torch.cat([xn, xs], dim=1)
        out = out.reshape(b, -1, h, w)

        out = self.channel_shuffle(out, 2)
        return out

def conv_3x3_bn(inp, oup, act, stride=1, group=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False, groups=group),
        nn.BatchNorm2d(oup),
        act(inplace=True)
    )


def conv_1x1_bn(inp, oup, act, group=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False, groups=group),
        nn.BatchNorm2d(oup),
        act(inplace=True)
    )


class down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels,use_se=False):
        super(down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            DepthwiseSeparable(num_channels= in_channels, num_filters=in_channels,stride=1,use_se=use_se),
            DepthwiseSeparable(num_channels= in_channels, num_filters=out_channels,stride=2,use_se=use_se)
        )
    def forward(self, x):
        return self.maxpool_conv(x)

class up(nn.Module):
    def __init__(self, in_ch, out_ch, upsample_type):
        super(up, self).__init__()
        self.upsample_type = upsample_type
        self.up = self._make_upconv(out_ch , out_ch, upscale_factor = 2)

        self.CSPconv = CSPLayer(in_ch, out_ch, kernel_size=3)
        
    def _make_upconv(self, in_channels, out_channels, upscale_factor = 2):
        if self.upsample_type == 'use_duc':
            print('duc')
            return DenseUpsamplingConvolution(in_channels, out_channels, upscale_factor = upscale_factor)
        elif self.upsample_type == 'use_convt':
            print('use_convt')
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size = upscale_factor, stride = upscale_factor, padding = 0, output_padding = 0),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True)
            )
        elif self.upsample_type == 'use_bilinear':
            print('use_bilinear')
            return nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else :
            print('upsample_type error , please check!!!!')

    def forward(self, x1, x2):
        
        x = torch.cat([x2, x1], dim=1)
        
        x = self.CSPconv(x)

        x = self.up(x)

        return x

class GenerativeResnet(GraspModel):

    def __init__(self, input_channels=4, output_channels=1, channel_size=32,upsamp='use_bilinear',dropout=False, prob=0.0):
        super(GenerativeResnet, self).__init__()
        print('upsamp {}'.format(upsamp))
        self.inconv = nn.Sequential(conv_3x3_bn(input_channels, input_channels, act=nn.ReLU, stride=1, group=input_channels),
                                    conv_1x1_bn(input_channels , channel_size , act=nn.Hardswish),
                                    DepthwiseSeparable(num_channels= channel_size, num_filters=channel_size,stride=1,use_se=True))

        self.downconv1 = down(in_channels= channel_size ,  out_channels= channel_size * 2,use_se=True)

        self.downconv2 = down(in_channels= channel_size * 2,  out_channels= channel_size * 4,use_se=True)

        self.bottonconv = nn.Sequential(
                                        DepthwiseSeparable(num_channels= channel_size * 4, num_filters=channel_size * 4,stride=1,use_se=True),
                                        DepthwiseSeparable(num_channels= channel_size * 4, num_filters=channel_size * 4,stride=1,use_se=True),
                                        DepthwiseSeparable(num_channels= channel_size * 4, num_filters=channel_size * 4,stride=1,use_se=True),
                                        DepthwiseSeparable(num_channels= channel_size * 4, num_filters=channel_size * 4,stride=1,use_se=True),
                                        DepthwiseSeparable(num_channels= channel_size * 4, num_filters=channel_size * 4,stride=1,use_se=True),
                                        CSPLayer(channel_size * 4, channel_size * 4, kernel_size=3)
        )
        self.up1 = up(channel_size * 8, channel_size * 2, upsamp)

        self.up2 = up(channel_size * 4, channel_size, upsamp)


        self.outconv = conv_1x1_bn(channel_size , channel_size , act=nn.ReLU)


        self.pos_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=1)
        self.cos_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=1)
        self.sin_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=1)
        self.width_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=1)

        self.dropout = dropout
        self.dropout_pos = nn.Dropout(p=prob)
        self.dropout_cos = nn.Dropout(p=prob)
        self.dropout_sin = nn.Dropout(p=prob)
        self.dropout_wid = nn.Dropout(p=prob)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight, gain=1)
        
    def forward(self, x_in):
        dbg=0
        x_in = self.inconv(x_in)
        if dbg == 1:
            print('x_in.shape  {}'.format(x_in.shape))
        d1 = self.downconv1(x_in)
        if dbg == 1:
            print('d1.shape  {}'.format(d1.shape))

        d2 = self.downconv2(d1)
        if dbg == 1:
            print('d2.shape  {}'.format(d2.shape))

        b1 = self.bottonconv(d2)
        if dbg == 1:
            print('b1.shape  {}'.format(b1.shape))

        u1 = self.up1(b1,d2)
        if dbg == 1:
            print('u1.shape  {}'.format(u1.shape))

        u2 = self.up2(u1,d1)
        if dbg == 1:
            print('u2.shape  {}'.format(u2.shape))

        x = self.outconv(u2)
        if dbg == 1:
            print('x.shape  {}'.format(x.shape))


        if self.dropout:
            pos_output = self.pos_output(self.dropout_pos(x))
            cos_output = self.cos_output(self.dropout_cos(x))
            sin_output = self.sin_output(self.dropout_sin(x))
            width_output = self.width_output(self.dropout_wid(x))
        else:
            pos_output = self.pos_output(x)
            cos_output = self.cos_output(x)
            sin_output = self.sin_output(x)
            width_output = self.width_output(x)
        return pos_output, cos_output, sin_output, width_output
if __name__ == "__main__":
    model = GenerativeResnet()
    model.eval()
    input = torch.rand(1, 4, 224, 224)
    output = model(input)
