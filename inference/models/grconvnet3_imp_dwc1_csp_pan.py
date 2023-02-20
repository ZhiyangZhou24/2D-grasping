import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import sys
sys.path.append('/home/lab/zzy/grasp/2D-grasping-my')
from inference.models.pico_det import CSPLayer, DepthwiseSeparable,ConvBNLayer,conv_1x1_bn,DPModule
from inference.models.grasp_model import GraspModel
from inference.models.attention import CoordAtt
from inference.models.duc import DenseUpsamplingConvolution
from torchsummary import summary


class down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels,act="hard_swish",use_se=False):
        super(down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            DepthwiseSeparable(num_channels= in_channels, num_filters=in_channels,stride=1,use_se=use_se,act=act),
            DepthwiseSeparable(num_channels= in_channels, num_filters=out_channels,stride=2,use_se=use_se,act=act)
        )
    def forward(self, x):
        return self.maxpool_conv(x)

class up(nn.Module):
    def __init__(self, in_ch, out_ch, upsample_type,num_blocks=2,act="leaky_relu"):
        super(up, self).__init__()
        self.upsample_type = upsample_type
        self.up = self._make_upconv(out_ch, out_ch, upscale_factor = 2)

        self.CSPconv = CSPLayer(in_ch, out_ch, kernel_size=3,num_blocks=num_blocks,act=act,use_depthwise=False)
        
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
        else :
            print('upsample_type error , please check!!!!')

    def forward(self, x1, x2):
        
        x = torch.cat([x2, x1], dim=1)
        
        x = self.CSPconv(x)

        x = self.up(x)

        return x

class GenerativeResnet(GraspModel):

    def __init__(self, input_channels=1, output_channels=1, channel_size=32,use_mish=True,use_depthwise = True, att = 'use_eca',upsamp='use_bilinear',dropout=False, prob=0.0):
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

        self.upsample_type = upsamp
        conv_func = DPModule if use_depthwise else ConvBNLayer
        
        self.stem = ConvBNLayer(in_channel=input_channels,out_channel=channel_size,kernel_size=3,stride=1,groups=1,act=self.act)

        self.dsc1 = nn.Sequential( #112  64
                    DepthwiseSeparable(num_channels= channel_size , num_filters=channel_size ,stride=1,att_type = self.att,act=self.act),
                    DepthwiseSeparable(num_channels= channel_size , num_filters=channel_size * 2,stride=2,att_type = self.att,act=self.act)
        )

        self.dsc2 = nn.Sequential( #56  128
                    DepthwiseSeparable(num_channels= channel_size * 2, num_filters=channel_size * 2,stride=1,att_type = self.att,act=self.act),
                    DepthwiseSeparable(num_channels= channel_size * 2, num_filters=channel_size * 4,stride=2,att_type = self.att,act=self.act)
        )

        self.dsc3 = nn.Sequential( #28  256
                    DepthwiseSeparable(num_channels= channel_size * 4, num_filters=channel_size * 4,stride=1,att_type = self.att,act=self.act),
                    DepthwiseSeparable(num_channels= channel_size * 4, num_filters=channel_size * 8,stride=2,att_type = self.att,act=self.act)
        )

        self.dscBottleNeck = nn.Sequential(
                    DepthwiseSeparable(num_channels= channel_size * 8, num_filters=channel_size * 8,dw_size = 5,stride=1,att_type = self.att,act=self.act),
                    DepthwiseSeparable(num_channels= channel_size * 8, num_filters=channel_size * 8,dw_size = 5,stride=1,att_type = self.att,act=self.act),
                    DepthwiseSeparable(num_channels= channel_size * 8, num_filters=channel_size * 8,dw_size = 5,stride=1,att_type = self.att,act=self.act),
                    DepthwiseSeparable(num_channels= channel_size * 8, num_filters=channel_size * 8,dw_size = 5,stride=1,att_type = self.att,act=self.act),
                    DepthwiseSeparable(num_channels= channel_size * 8, num_filters=channel_size * 8,dw_size = 5,stride=1,att_type = self.att,act=self.act)  #14

                    # DepthwiseSeparable(num_channels= channel_size * 16, num_filters=channel_size * 32,dw_size = 5,stride=2,use_se=True),#7
                    # DepthwiseSeparable(num_channels= channel_size * 32, num_filters=channel_size * 32,dw_size = 5,stride=1,use_se=True),
        )

        self.conv_1x1_1 = conv_1x1_bn(channel_size * 8, channel_size * 1,act=nn.Mish)  
        self.conv_1x1_2 = conv_1x1_bn(channel_size * 4, channel_size * 1,act=nn.Mish)
        self.conv_1x1_3 = conv_1x1_bn(channel_size * 2, channel_size * 1,act=nn.Mish)
        self.conv_1x1_4 = conv_1x1_bn(channel_size * 1, channel_size * 1,act=nn.Mish)

        self.upsample1 = self._make_upconv(in_channels=channel_size * 1,out_channels = channel_size * 1)#28 56
        self.upsample2 = self._make_upconv(in_channels=channel_size * 1,out_channels = channel_size * 1)#56 128
        self.upsample3 = self._make_upconv(in_channels=channel_size * 1,out_channels = channel_size * 1)#128 256

        self.csp1 = CSPLayer(channel_size * 2, channel_size, kernel_size=3,num_blocks=1,act=self.act,use_depthwise=False)
        self.csp2 = CSPLayer(channel_size * 2, channel_size, kernel_size=3,num_blocks=1,act=self.act,use_depthwise=False)
        self.csp3 = CSPLayer(channel_size * 2, channel_size, kernel_size=3,num_blocks=1,act=self.act,use_depthwise=False)

        self.csp4 = CSPLayer(channel_size * 2, channel_size, kernel_size=3,num_blocks=1,act=self.act,use_depthwise=False)
        self.csp5 = CSPLayer(channel_size * 2, channel_size, kernel_size=3,num_blocks=1,act=self.act,use_depthwise=False)
        self.csp6 = CSPLayer(channel_size * 2, channel_size, kernel_size=3,num_blocks=1,act=self.act,use_depthwise=False)

        self.down1 = conv_func(channel_size,channel_size,kernel_size=3,stride=2,act=self.act) # 112
        self.down2 = conv_func(channel_size,channel_size,kernel_size=3,stride=2,act=self.act) # 56
        self.down3 = conv_func(channel_size,channel_size,kernel_size=3,stride=2,act=self.act) # 28
        
        self.fuse = ConvBNLayer(channel_size * 3 , channel_size,kernel_size=3,act = self.act)

        self.up_final = self._make_upconv(in_channels=channel_size * 1,out_channels = channel_size * 1)#128 256
        
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
            return nn.Upsample(scale_factor=upscale_factor, mode='bilinear', align_corners=True)
        else :
            print('upsample_type error , please check!!!!')

    def forward(self, x_in):
        dbg=1
        if dbg == 1:
            print('x_in.shape  {}'.format(x_in.shape))
        stem = self.stem(x_in)
        if dbg == 1:
            print('stem.shape  {}'.format(stem.shape))
        d1 = self.dsc1(stem)
        if dbg == 1:
            print('d1.shape  {}'.format(d1.shape))

        d2 = self.dsc2(d1)
        if dbg == 1:
            print('d2.shape  {}'.format(d2.shape))

        d3 = self.dsc3(d2)
        if dbg == 1:
            print('d3.shape  {}'.format(d3.shape))

        d4 = self.dscBottleNeck(d3)
        if dbg == 1:
            print('d4.shape  {}'.format(d4.shape))

        d4 = self.conv_1x1_1(d4)

        d3 = self.conv_1x1_2(d2)

        d2 = self.conv_1x1_3(d1)

        c1 = self.csp1(torch.cat([d3, self.upsample1(d4)], dim=1)) 

        c2 = self.csp2(torch.cat([d2, self.upsample1(c1)], dim=1)) 

        c21 = self.csp4(torch.cat([c1, self.down1(c2)], dim=1))

        c22 = self.csp5(torch.cat([d4, self.down2(c21)], dim=1))

        c21 = F.interpolate(c21, c2.size()[-2:],mode='bilinear', align_corners=True)

        c22 = F.interpolate(c22, c2.size()[-2:],mode='bilinear', align_corners=True)

        x = self.fuse(torch.cat([c22,c21,c2],1))

        x = self.up_final(x)

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
sys.path.append('/home/lab/zzy/grasp/2D-grasping-my')
if __name__ == '__main__':
    model = GenerativeResnet()
    model.eval()
    input = torch.rand(1, 1, 224, 224)
    summary(model, (1, 224, 224),device='cpu')
    sys.stdout = sys.__stdout__
    output = model(input)
