import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('/home/lab/zzy/grasp/2D-grasping-my')
from inference.models.attention import CoordAtt, eca_block, se_block,cbam_block
from inference.models.grasp_model import GraspModel, Mish
from inference.models.duc import DenseUpsamplingConvolution
from inference.models.pp_lcnet import DepthwiseSeparable, ConvBNLayer,make_divisible
from inference.models.pico_det import CSPLayer
from torchsummary import summary

class conv_att(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch,use_mish=False,att_type = 'use_eca', reduc_ratio=3):
        super(conv_att, self).__init__()
        self.att_type = att_type

        self.att = self._make_att(in_ch, in_ch,reduc_ratio)
        if use_mish:
            self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            Mish()
        )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU()
            )
        
        self.channel_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_ch),
	        # nn.GroupNorm(32, out_ch),
        )
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
            print('att_type error , please check!!!!')

    def forward(self, x):
        residual = x
        x = self.conv(x)
        x = self.att(x)

        if residual.shape[1] != x.shape[1]:
            residual = self.channel_conv(residual)
        x += residual
        return x

class dsc_conv_att(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch,dsc_kernel=3,stride=2,use_mish=False,iden=False,att_type = 'use_eca', reduc_ratio=3):
        super(dsc_conv_att, self).__init__()
        self.att_type = att_type
        self.iden = iden
        if att_type == 'use_se':
            use_se = True
        else :
            use_se = False
        self.conv = nn.Sequential(
                DepthwiseSeparable(num_channels= in_ch, num_filters=in_ch,dw_size=dsc_kernel,stride=1,use_se=use_se),
                DepthwiseSeparable(num_channels= in_ch, num_filters=out_ch,dw_size=dsc_kernel,stride=stride,use_se=use_se)
        )

        self.channel_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_ch)
        )

    def forward(self, x):
        if self.iden:
            residual = x
        x = self.conv(x)
        if self.iden:
            if residual.shape[1] != x.shape[1]:
                residual = self.channel_conv(residual)
            x += residual
        return x

class inconv(nn.Module):
    def __init__(self, in_channels, out_channels,use_mish=False,att_type=None, reduc_ratio = 16):
        super(inconv, self).__init__()
        self.att = None
        if use_mish:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels // 2, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels // 2),
                Mish(),
                nn.Conv2d(out_channels // 2, out_channels, kernel_size= 3, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                Mish()
            )
        else:
            self.conv = nn.Sequential(
                ConvBNLayer(
                    num_channels=in_channels,
                    filter_size=3,
                    num_filters=out_channels // 2,
                    stride=1),
                ConvBNLayer(
                    num_channels=out_channels // 2,
                    filter_size=3,
                    num_filters=out_channels,
                    stride=2),
            )
            
        if att_type == 'use_coora':
            self.att = CoordAtt(out_channels,out_channels,reduc_ratio)
        elif att_type == 'use_eca':
            print('use_eca reduc_ratio = {}'.format(reduc_ratio))
            self.att = eca_block(out_channels)
        elif att_type == 'use_se':
            print('use_se reduc_ratio = {}'.format(reduc_ratio))
            self.att = se_block(channel=out_channels,ratio = reduc_ratio)
        elif att_type == 'use_cbam':
            print('use_cbam reduc_ratio = {}'.format(reduc_ratio))
            self.att = cbam_block(out_channels,ratio=reduc_ratio)
        elif att_type == None :
            print('inconv do not use attention')
        else :
            print('att_type error , please check!!!!')
    def forward(self, x):
        x = self.conv(x)
        if self.att != None:
            x = self.att(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch,dsc_kernel = 3,use_mish=False,att_type='use_se'):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            dsc_conv_att(in_ch, out_ch,stride=2,att_type=att_type,use_mish=use_mish,dsc_kernel=dsc_kernel)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x

class up(nn.Module):
    def __init__(self, in_ch, out_ch,use_mish=False,att_type='use_se'):
        super(up, self).__init__()
        self.att_type = att_type

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # not used
        self.cspconv = CSPLayer(in_ch,out_ch,kernel_size=3)

        self.conv = dsc_conv_att(in_ch, out_ch,stride=1,att_type=att_type,use_mish=use_mish)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class up_final(nn.Module):
    def __init__(self, in_ch, out_ch, use_mish=False):
        super(up_final, self).__init__()

        if use_mish:
            self.up = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 2,stride = 2),
            nn.BatchNorm2d(out_ch),
            Mish()
        )
        else:
            self.up = nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, 2,stride = 2),
                nn.BatchNorm2d(out_ch),
                nn.ReLU()
            )


    def forward(self, x):
        x = self.up(x) #64
        return x

class GenerativeResnet(GraspModel):

# DSC version of resunet
    def __init__(self, input_channels=4, output_channels=1, channel_size=32,use_mish=False,upsamp='use_bilinear', att = 'use_se', dropout=False, prob=0.0):
        super(GenerativeResnet, self).__init__()
        print('Model is resunet4')
        print('Model att {}'.format(att))

        self.inconv = inconv(input_channels,channel_size * 2,use_mish=use_mish,att_type=None)

        self.down1 = down(channel_size * 2, channel_size * 4, att_type=att,use_mish=use_mish)
        self.down2 = down(channel_size * 4, channel_size * 8, att_type=att,use_mish=use_mish)
        self.down3 = down(channel_size * 8, channel_size * 16, att_type=att,use_mish=use_mish)
        self.down4 = down(channel_size * 16, channel_size * 16, att_type=att,use_mish=use_mish)
        self.up1 = up(channel_size * 16 * 2, channel_size * 8, att_type=att,use_mish=use_mish)
        self.up2 = up(channel_size * 8 * 2, channel_size * 4, att_type=att,use_mish=use_mish)
        self.up3 = up(channel_size * 4 * 2, channel_size * 2, att_type=att,use_mish=use_mish)
        self.up4 = up(channel_size * 2 * 2, channel_size * 2, att_type=att,use_mish=use_mish)

        self.up_final = up_final(channel_size * 2, channel_size)

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
        dbg = 0
        x1 = self.inconv(x_in)
        if dbg == 1:
            print('x1.shape  {}'.format(x1.shape))

        x2 = self.down1(x1) 
        if dbg == 1:
            print('x2.shape  {}'.format(x2.shape))

        x3 = self.down2(x2) 
        if dbg == 1:
            print('x3.shape  {}'.format(x3.shape))

        x4 = self.down3(x3) 
        if dbg == 1:
            print('x4.shape  {}'.format(x4.shape))

        x5 = self.down4(x4) 
        if dbg == 1:
            print('x5.shape  {}'.format(x5.shape))

        x44 = self.up1(x5, x4) #512 + 512 256 14
        if dbg == 1:
            print('x44.shape  {}'.format(x44.shape))
        x33 = self.up2(x44,x3) #256 +256 > 128 28
        if dbg == 1:
            print('x33.shape  {}'.format(x33.shape))
        x22 = self.up3(x33,x2) # 128 + 128 > 64 56
        if dbg == 1:
            print('x22.shape  {}'.format(x22.shape))
        x11 = self.up4(x22,x1) #64 112
        if dbg == 1:
            print('x11.shape  {}'.format(x11.shape))

        # x = F.relu(self.bn5(self.conv5(self.up_out(x11)))) #32 224
        x = self.up_final(x11)
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
        if dbg == 1:
            print('width_output.shape  {}'.format(width_output.shape))
        return pos_output, cos_output, sin_output, width_output

if __name__ == '__main__':
    model = GenerativeResnet()
    model.eval()
    input = torch.rand(1, 4, 224, 224)
    summary(model, (4, 224, 224),device='cpu')
    sys.stdout = sys.__stdout__
    output = model(input)