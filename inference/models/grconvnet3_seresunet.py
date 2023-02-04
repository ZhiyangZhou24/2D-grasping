import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from inference.models.grasp_model import GraspModel, ResidualBlock

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ECALayer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(ECALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)
import math
class eca_block(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(eca_block, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        print("eca kernel size is {}".format(kernel_size))
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch, reduction=16):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            # nn.GroupNorm(32, out_ch),
            
            # nn.ReLU(),
            # nn.Conv2d(out_ch, out_ch, 3, padding=1),
            # nn.BatchNorm2d(out_ch),
	        # nn.GroupNorm(32, out_ch),
	        
            nn.ReLU()
        )
        self.se = SELayer(out_ch, 3)
        self.eca = eca_block(out_ch)
        
        self.channel_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_ch),
	        # nn.GroupNorm(32, out_ch),
        )
    
    def forward(self, x):
        residual = x
        x = self.conv(x)
        x = self.eca(x)

        if residual.shape[1] != x.shape[1]:
            residual = self.channel_conv(residual)
        x += residual
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x
class down_final(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down_final, self).__init__()
        self.mpconv = nn.Sequential(
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False):
        super(up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

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
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up_final, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = nn.Sequential(  #32 > 32 
            nn.Conv2d(out_ch, out_ch * 2, kernel_size=1, stride=1, bias=False),# 224
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )  # 224 32

        self.conv1 = nn.Conv2d(out_ch * 2, out_ch, kernel_size=1, stride=1, bias=False)  # 64 > 32

    def forward(self, x1, x2):
        print('shape of x1{}'.format(x1.shape))
        x1 = self.up(x1) #64
        print('shape of x1{}'.format(x1.shape))
        x1 = self.conv1(x1) #32
        print('shape of x1{}'.format(x1.shape))
        print('shape of x2{}'.format(x2.shape))

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)  #32 >32
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=False, rate=0.1):
        super(outconv, self).__init__()
        self.dropout = dropout
        if dropout:
            print('dropout', rate)
            self.dp = nn.Dropout2d(rate)
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        if self.dropout:
            x = self.dp(x)
        x = self.conv(x)
        return x

class GenerativeResnet(GraspModel):

    def __init__(self, input_channels=4, output_channels=1, channel_size=32, dropout=False, prob=0.0):
        super(GenerativeResnet, self).__init__()

        #down 1
        self.conv1 = nn.Conv2d(input_channels, channel_size, kernel_size=3, stride=1, padding=1)#32 C
        self.bn1 = nn.BatchNorm2d(channel_size)

        #down 2
        self.conv2 = nn.Conv2d(channel_size, channel_size * 2, kernel_size=3, stride=2, padding=1)#64 C
        self.bn2 = nn.BatchNorm2d(channel_size * 2)

        self.down1 = down(channel_size * 2, channel_size * 4)
        self.down2 = down(channel_size * 4, channel_size * 8)
        self.down3 = down(channel_size * 8, channel_size * 16)
        self.down4 = down(channel_size * 16, channel_size * 16)
        self.up1 = up(channel_size * 16 * 2, channel_size * 8)
        self.up2 = up(channel_size * 8 * 2, channel_size * 4)
        self.up3 = up(channel_size * 4 * 2, channel_size * 2)
        self.up4 = up(channel_size * 2 * 2, channel_size * 2)

        # self.up_final = up_final(channel_size * 2 , channel_size )

        self.up_out = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv5 = nn.ConvTranspose2d(channel_size * 2, channel_size, kernel_size=3, stride=1, padding=1)

        self.bn5 = nn.BatchNorm2d(channel_size)

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
        # print('x_in shape{}'.format(x_in.shape))# 4 224
        x = F.relu(self.bn1(self.conv1(x_in))) 
        # print('x shape{}'.format(x.shape))# 32 224
        x1 = F.relu(self.bn2(self.conv2(x))) 
        # print('x shape{}'.format(x1.shape))# 64 112

        x2 = self.down1(x1) 
        # print('x2 shape{}'.format(x2.shape)) #128 56

        x3 = self.down2(x2) 
        # print('x3 shape{}'.format(x3.shape)) #256 28

        x4 = self.down3(x3) 
        # print('x4 shape{}'.format(x4.shape)) #512 14

        x5 = self.down4(x4) 
        # print('x5 shape{}'.format(x5.shape)) #512 7

        x44 = self.up1(x5, x4) #512 + 512 256 14
        # print('x44 shape{}'.format(x44.shape))
        x33 = self.up2(x44,x3) #256 +256 > 128 28
        # print('x33 shape{}'.format(x33.shape))
        x22 = self.up3(x33,x2) # 128 + 128 > 64 56
        # print('x22 shape{}'.format(x22.shape))
        x11 = self.up4(x22,x1) #64 112
        # print('x11 shape{}'.format(x11.shape)) 

        x = F.relu(self.bn5(self.conv5(self.up_out(x11)))) #32 224
        # print('x shape{}'.format(x.shape))

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
        # print('width_output shape{}'.format(width_output.shape))
        return pos_output, cos_output, sin_output, width_output


if __name__ == '__main__':

    def weights_init(m):
        classname = m.__class__.__name__
        # print(classname)
        if classname.find('Conv') != -1:
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)


    # model = SeResUNet(3, 3, deep_supervision=True).cuda()
    model = GenerativeResnet(3, 3, deep_supervision=True).cuda()
    model.apply(weights_init)

    x = torch.randn((1, 3, 256, 256)).cuda()

    for i in range(1000):
        y0, y1, y2, y3, y4 = model(x)
        print(y0.shape, y1.shape, y2.shape, y3.shape, y4.shape)