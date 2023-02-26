import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import sys
sys.path.append('/home/lab/zzy/grasp/2D-grasping-my')
from inference.models.pico_det import CSPLayer, DepthwiseSeparable,ConvBNLayer,C2f
from inference.models.grasp_model import GraspModel
from inference.models.attention import CoordAtt
from inference.models.duc import DenseUpsamplingConvolution
from inference.models.pp_liteseg import SPPM,UAFM
from torchsummary import summary


class transition(nn.Module):

    def __init__(self, mid_c = 64,out = 32,act = 'mish'):
        super(transition, self).__init__()
        self.csp1 = CSPLayer(64,mid_c,act=act)
        self.csp2 = CSPLayer(128,mid_c,act=act)
        self.csp3 = CSPLayer(256,mid_c,act=act)
        self.csp4 = CSPLayer(512,mid_c,act=act)

        self.out_conv = nn.Sequential(
            ConvBNLayer(in_channel = mid_c*4,out_channel = out,kernel_size=3,act=act)
        ) 

    def forward(self, x1,x2,x3,x4):
        x1 = self.csp1(x1)
        x2 = self.csp2(x2)
        x3 = self.csp3(x3)
        x4 = self.csp4(x4)
        x4 = F.interpolate(x4, x3.size()[-2:],mode='bilinear', align_corners=True)
        x3 = torch.cat([x4, x3], dim=1)
        x3 = F.interpolate(x3, x2.size()[-2:],mode='bilinear', align_corners=True)
        x2 = torch.cat([x3, x2], dim=1)
        x2 = F.interpolate(x2, x1.size()[-2:],mode='bilinear', align_corners=True)
        x1 = torch.cat([x2, x1], dim=1)
        x1 = self.out_conv(x1)
        x1 = F.interpolate(x1,scale_factor=2,mode='bilinear', align_corners=True)
        return x1

class up(nn.Module):
    def __init__(self, in_ch, out_ch, upsample_type,num_blocks=2,act="leaky_relu"):
        super(up, self).__init__()
        self.upsample_type = upsample_type
        self.up = self._make_upconv(out_ch, out_ch, upscale_factor = 2)

        # self.conv = CSPLayer(in_ch, out_ch, kernel_size=3,act=act,use_depthwise=False)
        self.conv = ConvBNLayer(in_channel=in_ch,out_channel=out_ch,kernel_size=1,stride=1,groups=1,act=act)
        
    def _make_upconv(self, in_channels, out_channels, upscale_factor = 2):
        if self.upsample_type == 'use_duc':
            print('duc')
            return DenseUpsamplingConvolution(in_channels, out_channels, upscale_factor = upscale_factor)
        elif self.upsample_type == 'use_convt':
            print('use_convt')
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size = upscale_factor * 2 , stride = upscale_factor, padding = 1, output_padding = 0)
            )
        elif self.upsample_type == 'use_bilinear':
            print('use_bilinear')
            return nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        elif self.upsample_type == 'use_nearest':
            print('use_nearest')
            return nn.Upsample(scale_factor=2, mode='nearest', align_corners=True)
        else :
            print('upsample_type error , please check!!!!')

    def forward(self, x_high, x_low):
        x_high = self.conv(x_high)
        x_high = self.up(x_high)
        x_low += x_high
        return x_low

class GenerativeResnet(GraspModel):

    def __init__(self, input_channels=1, output_channels=1, channel_size=32,use_mish=True, att = 'use_eca',upsamp='use_bilinear',dropout=False, prob=0.0):
        super(GenerativeResnet, self).__init__()
        print('Model is grc3_imp_dwc1')
        print('GRCNN upsamp {}'.format(upsamp))
        print('USE mish {}'.format(use_mish))
        print('USE att {}'.format(att))
        self.att = att
        if use_mish :
            self.act = "mish"
        else:
            self.act = "hard_swish"
        
        self.stem = ConvBNLayer(in_channel=input_channels,out_channel=channel_size,kernel_size=7,stride=1,groups=1,act=self.act)#224 32

        self.dsc1 = nn.Sequential( #112 64
                    DepthwiseSeparable(num_channels= channel_size , num_filters=channel_size * 2,stride=2,att_type = self.att,act=self.act),
                    DepthwiseSeparable(num_channels= channel_size * 2, num_filters=channel_size * 2,stride=1,att_type = self.att,act=self.act)
        )

        self.dsc2 = nn.Sequential( #56 128
                    DepthwiseSeparable(num_channels= channel_size * 2, num_filters=channel_size * 4,stride=2,att_type = self.att,act=self.act),
                    DepthwiseSeparable(num_channels= channel_size * 4, num_filters=channel_size * 4,stride=1,att_type = self.att,act=self.act)
        )

        self.dsc3 = nn.Sequential( #28 256
                    DepthwiseSeparable(num_channels= channel_size * 4, num_filters=channel_size * 8,stride=2,att_type = self.att,act=self.act),
                    DepthwiseSeparable(num_channels= channel_size * 8, num_filters=channel_size * 8,stride=1,att_type = self.att,act=self.act)
        )

        self.dsc4 = nn.Sequential( #14 512
                    DepthwiseSeparable(num_channels= channel_size * 8, num_filters=channel_size * 16,dw_size = 5,stride=2,att_type = self.att,act=self.act),
                    DepthwiseSeparable(num_channels= channel_size * 16, num_filters=channel_size * 16,dw_size = 5,stride=1,att_type = self.att,act=self.act),
                    # DepthwiseSeparable(num_channels= channel_size * 16, num_filters=channel_size * 16,dw_size = 5,stride=1,att_type = self.att,act=self.act),
                    # CSPLayer(channel_size * 16,channel_size * 16)
                    # DepthwiseSeparable(num_channels= channel_size * 16, num_filters=channel_size * 32,dw_size = 5,stride=2,use_se=True),#7
                    # DepthwiseSeparable(num_channels= channel_size * 32, num_filters=channel_size * 32,dw_size = 5,stride=1,use_se=True),
        )

        self.up1 = up(channel_size * 16, channel_size * 8, upsamp,act=self.act) # 28 256

        self.up2 = up(channel_size * 8, channel_size * 4, upsamp,act=self.act) # 56 128

        self.up3 = up(channel_size * 4, channel_size * 2, upsamp,act=self.act)# 112 64

        self.trasition = transition()

        # self.sppm = SPPM(in_channels=channel_size * 16,inter_channels=channel_size * 16,out_channels=channel_size * 16, bin_sizes=[1, 2, 4])  #14 512

        # self.uafm1 = UAFM(low_chan=channel_size * 8,hight_chan=channel_size * 16,out_chan=channel_size * 8)  #28 256
        # self.uafm2 = UAFM(low_chan=channel_size * 4,hight_chan=channel_size * 8,out_chan=channel_size * 4)  #56 128
        # self.uafm3 = UAFM(low_chan=channel_size * 2,hight_chan=channel_size * 4,out_chan=channel_size * 2)  #112 64
        # self.uafm4 = UAFM(low_chan=channel_size * 1,hight_chan=channel_size * 2,out_chan=channel_size * 1)  #224 32

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
        dbg=1
        if dbg == 1:
            print('x_in.shape  {}'.format(x_in.shape))
        stem = self.stem(x_in) #224 32
        if dbg == 1:
            print('stem.shape  {}'.format(stem.shape))
        d1 = self.dsc1(stem) #112 64
        if dbg == 1:
            print('d1.shape  {}'.format(d1.shape))

        d2 = self.dsc2(d1) #56 128
        if dbg == 1:
            print('d2.shape  {}'.format(d2.shape))

        d3 = self.dsc3(d2) #28 256
        if dbg == 1:
            print('d3.shape  {}'.format(d3.shape))

        d4 = self.dsc4(d3) #14 512
        if dbg == 1:
            print('d4.shape  {}'.format(d4.shape))

        d3 = self.up1(d4,d3) #28 256
        if dbg == 1:
            print('u1.shape  {}'.format(d3.shape))

        d2 = self.up2(d3,d2)  #56 128
        if dbg == 1:
            print('u2.shape  {}'.format(d2.shape))

        d1 = self.up3(d2,d1)#112 64
        if dbg == 1:
            print('u3.shape  {}'.format(d1.shape))

        x = self.trasition(d1,d2,d3,d4)#112 32
        if dbg == 1:
            print('tra.shape  {}'.format(x.shape))
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
import sys
from torchstat import stat
sys.path.append('/home/lab/zzy/grasp/2D-grasping-my')
if __name__ == '__main__':
    model = GenerativeResnet(input_channels=1)
    model.eval()
    input = torch.rand(1, 1, 224, 224)
    summary(model, (1, 224, 224),device='cpu')
    sys.stdout = sys.__stdout__
    output = model(input)
    # stat(model,(1,224,224))
