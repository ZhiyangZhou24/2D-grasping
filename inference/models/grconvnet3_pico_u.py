import torch.nn as nn
import torch.nn.functional as F
import torch
import inference.models.pico_det as pico
from inference.models.grasp_model import GraspModel, ResidualBlock

class conv_block(nn.Module):
    """
    Convolution Block 
    """
    def __init__(self, input_channels, output_channels,kernel_size,stride,padding):
        super(conv_block, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(output_channels),
            nn.Hardswish(inplace=True))

    def forward(self, x):

        x = self.conv(x)
        return x

class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, input_channels, output_channels,kernel_size,stride,padding):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(output_channels),
            nn.Mish(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class GenerativeResnet(GraspModel):

    def __init__(self, input_channels=4, output_channels=1, channel_size=32, dropout=False, prob=0.0):
        super(GenerativeResnet, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, channel_size, kernel_size=3, stride=1, padding=1)  #32 224
        self.bn1 = nn.BatchNorm2d(channel_size)

        self.mid_c_1 = pico.make_divisible(
                    int(channel_size * 2),
                    divisor=8)
        self.conv2_pico = pico.InvertedResidualDS(channel_size, channel_size * 2 , channel_size * 2,stride = 2,act = 'hard_swish')  #64

        self.mid_c_2 = pico.make_divisible(
                    int(channel_size * 4),
                    divisor=8)
        self.conv3_pico = pico.InvertedResidualDS(channel_size * 2, channel_size * 4 , channel_size * 4,stride = 2,act = 'hard_swish')  #128


        self.res1 = ResidualBlock(channel_size * 4, channel_size * 4) #128 56
        self.res2 = ResidualBlock(channel_size * 4, channel_size * 4) #128 56
        self.res3 = ResidualBlock(channel_size * 4, channel_size * 4) #128 56
        self.res4 = ResidualBlock(channel_size * 4, channel_size * 4) #128 56
        self.res5 = ResidualBlock(channel_size * 4, channel_size * 4) #128 56

        self.csp_layer_1 = pico.CSPLayer(channel_size * 4 + channel_size * 4,channel_size * 4,kernel_size=3,act = 'hard_swish')
        self.conv4 = up_conv(channel_size * 4, channel_size * 2, kernel_size=4, stride=2, padding=1)

        self.csp_layer_2 = pico.CSPLayer(channel_size * 2 + channel_size * 2,channel_size*2,act = 'hard_swish')
        self.conv5 = up_conv(channel_size * 2, channel_size, kernel_size=6, stride=2, padding=2)

        self.csp_layer_3 = pico.CSPLayer(channel_size  + channel_size ,channel_size , act = 'hard_swish')
        self.conv6 = nn.ConvTranspose2d(channel_size, channel_size, kernel_size=3, stride=1, padding=1)

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
        x = F.hardswish(self.bn1(self.conv1(x_in))) #32
        x1 = self.conv2_pico(x) #64
        # print('x1.shape  {}'.format(x1.shape))
        x2 = self.conv3_pico(x1) #128
        # print('x2.shape  {}'.format(x2.shape))

        x3 = self.res1(x2) #128
        x3 = self.res2(x3)
        x3 = self.res3(x3)
        x3 = self.res4(x3)
        x3 = self.res5(x3)
        # print('x3.shape  {}'.format(x3.shape))

        x22 = self.csp_layer_1(torch.cat((x3, x2), dim=1))  #128+128 128
        # print('x22.shape  {}'.format(x22.shape))
        x22 = self.conv4(x22)  #64
        # print('x22.shape  {}'.format(x22.shape))

        x11 = self.csp_layer_2(torch.cat((x22, x1), dim=1))  #64+64 64
        # print('x11.shape  {}'.format(x11.shape))
        x11 = self.conv5(x11)
        # print('x11.shape  {}'.format(x11.shape))

        xx = self.csp_layer_3(torch.cat((x11,x), dim=1)) #32+32 32
        # print('xx.shape  {}'.format(xx.shape))
        xx = self.conv6(xx)
        # print('xx.shape  {}'.format(xx.shape))
        if self.dropout:
            pos_output = self.pos_output(self.dropout_pos(xx))
            cos_output = self.cos_output(self.dropout_cos(xx))
            sin_output = self.sin_output(self.dropout_sin(xx))
            width_output = self.width_output(self.dropout_wid(xx))
        else:
            pos_output = self.pos_output(xx)
            cos_output = self.cos_output(xx)
            sin_output = self.sin_output(xx)
            width_output = self.width_output(xx)

        return pos_output, cos_output, sin_output, width_output
