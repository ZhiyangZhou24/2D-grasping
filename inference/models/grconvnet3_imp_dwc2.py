import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from inference.models.pico_det import CSPLayer
from inference.models.grasp_model import GraspModel, ResidualBlock
from inference.models.pp_lcnet import DepthwiseSeparable
from inference.models.attention import CoordAtt, eca_block , se_block,cbam_block
from inference.models.duc import DenseUpsamplingConvolution

from inference.models.RFB_Net_E_vgg import BasicRFB

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


class down1(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels,use_se=False):
        super(down1, self).__init__()

        self.cbr_4 = nn.Sequential(
            conv_3x3_bn(in_channels, out_channels, act=nn.ReLU, stride=1, group=1),
            conv_3x3_bn(out_channels, out_channels, act=nn.ReLU, stride=1, group=1),
            conv_3x3_bn(out_channels, out_channels, act=nn.ReLU, stride=1, group=1),
            conv_3x3_bn(out_channels, out_channels, act=nn.ReLU, stride=1, group=1)
        )
        
        self.mp = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x1 = self.cbr_4(x)
        x = self.mp(x1)
        return x, x1

class down2(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels,use_se=False):
        super(down2, self).__init__()

        self.cbr_2 = nn.Sequential(
            conv_3x3_bn(in_channels, out_channels, act=nn.ReLU, stride=1, group=1),
            conv_3x3_bn(out_channels, out_channels, act=nn.ReLU, stride=1, group=1)
        )
        
        self.mp = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x1 = self.cbr_2(x)
        x = self.mp(x1)
        return x, x1

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

class up_att(nn.Module):
    def __init__(self, in_ch, out_ch,att_type,reduc_ratio=16, upsample_type='use_bilinear'):
        super(up_att, self).__init__()
        self.upsample_type = upsample_type
        self.att_type = att_type

        self.att = self._make_att(in_ch, in_ch,reduc_ratio)

        self.up = self._make_upconv(in_ch , out_ch, upscale_factor = 2)
        
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

    def _make_att(self, in_channels, out_channels,reduc_ratio):
        if self.att_type == 'use_coora':
            print('use_coora reduc_ratio = {}'.format(reduc_ratio))
            return CoordAtt(in_channels,out_channels,reduc_ratio)
        elif self.att_type == 'use_eca':
            print('use_eca reduc_ratio = {}'.format(reduc_ratio))
            return eca_block(out_channels)
        elif self.att_type == 'use_se':
            print('use_se reduc_ratio = {}'.format(reduc_ratio))
            return se_block(channel=out_channels,ratio = reduc_ratio)
        elif self.att_type == 'use_cbam':
            print('use_cbam reduc_ratio = {}'.format(reduc_ratio))
            return cbam_block(out_channels,ratio=reduc_ratio)
        else :
            print('att type error , please check!!!!')

    def forward(self, x):
        # x = torch.cat([x1, x2], dim=1)
        # print(x.shape)
        x = self.att(x)

        x = self.up(x)

        return x

class GenerativeResnet(GraspModel):

    def __init__(self, input_channels=4, output_channels=1, channel_size=32, att = 'use_eca',upsamp='use_bilinear',dropout=False, prob=0.0):
        super(GenerativeResnet, self).__init__()
        print('Model is grc3_imp_dwc2')
        print('Model upsamp {}'.format(upsamp))
        print('Model att {}'.format(att))
        self.cbr_1 = nn.Sequential(  
            conv_3x3_bn(input_channels, channel_size, act=nn.ReLU, stride=1),
            conv_3x3_bn(channel_size, channel_size, act=nn.ReLU, stride=1),
            conv_3x3_bn(channel_size, channel_size, act=nn.ReLU, stride=1),
            conv_3x3_bn(channel_size, channel_size, act=nn.ReLU, stride=1)  # 400 32
        )
        self.mp1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.cbr_2 = nn.Sequential(
            conv_3x3_bn(channel_size, channel_size * 2, act=nn.ReLU, stride=1),
            conv_3x3_bn(channel_size * 2, channel_size * 2, act=nn.ReLU, stride=1)  # 200 64
        )
        
        self.mp2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.cbr_3 = nn.Sequential(
            conv_3x3_bn(channel_size * 2, channel_size * 4, act=nn.ReLU, stride=1),
            conv_3x3_bn(channel_size * 4, channel_size * 4, act=nn.ReLU, stride=1)  # 200 128
        )

        self.resblock = nn.Sequential(
            ResidualBlock(channel_size * 4, channel_size * 4),
            ResidualBlock(channel_size * 4, channel_size * 4),
            ResidualBlock(channel_size * 4, channel_size * 4)
        )
        
        self.rfb = BasicRFB(in_planes = channel_size * 4 ,out_planes = channel_size * 4)
        
        self.att1 = CoordAtt(inp=channel_size * 4,oup=channel_size * 4,reduction=16)

        self.up1 = up_att(in_ch=channel_size * 4 * 2,out_ch=channel_size * 2,att_type=att,upsample_type=upsamp)

        self.up2 = up_att(in_ch=channel_size * 2 * 2,out_ch=channel_size,att_type=att,upsample_type=upsamp)

        self.pos_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=3,padding=1)
        self.cos_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=3,padding=1)
        self.sin_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=3,padding=1)
        self.width_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=3,padding=1)

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
        cbr1_x = self.cbr_1(x_in)
        if dbg == 1:
            print('cbr1_x.shape  {}'.format(cbr1_x.shape))
        mp1_x = self.mp1(cbr1_x)
        if dbg == 1:
            print('mp1_x.shape  {}'.format(mp1_x.shape))

        cbr2_x = self.cbr_2(mp1_x)
        if dbg == 1:
            print('cbr2_x.shape  {}'.format(cbr2_x.shape))

        mp2_x = self.mp2(cbr2_x)
        if dbg == 1:
            print('mp2_x.shape  {}'.format(mp2_x.shape))

        cbr3_x = self.cbr_3(mp2_x)
        if dbg == 1:
            print('cbr3_x.shape  {}'.format(cbr3_x.shape))
        
        resx = self.resblock(cbr3_x)
        if dbg == 1:
            print('resx.shape  {}'.format(resx.shape))

        rfbx = self.rfb(resx)
        if dbg == 1:
            print('rfbx.shape  {}'.format(rfbx.shape))

        attx = self.att1(rfbx)
        if dbg == 1:
            print('attx.shape  {}'.format(attx.shape))

        up1x = self.up1(torch.cat((attx, cbr3_x), dim=1))  #att 128 100 cbr3 128 100
        if dbg == 1:
            print('up1x.shape  {}'.format(up1x.shape))

        up2x = self.up2(torch.cat((up1x, cbr2_x), dim=1))  #
        if dbg == 1:
            print('up2x.shape  {}'.format(up2x.shape))
        
        x = up2x
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
