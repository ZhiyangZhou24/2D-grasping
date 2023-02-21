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
        self.se = SELayer(out_ch, reduction)
        
        self.channel_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_ch),
	        # nn.GroupNorm(32, out_ch),
        )
    
    def forward(self, x):
        residual = x
        x = self.conv(x)
        x = self.se(x)

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
    def __init__(self, in_ch, out_ch, bilinear=True):
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
        self.conv1 = nn.Conv2d(input_channels, channel_size, kernel_size=9, stride=1, padding=4)#32 C
        self.bn1 = nn.BatchNorm2d(channel_size)

        #down 2
        self.conv2 = nn.Conv2d(channel_size, channel_size * 2, kernel_size=4, stride=2, padding=1)#64 C
        self.bn2 = nn.BatchNorm2d(channel_size * 2)

        #down 3
        self.conv3 = nn.Conv2d(channel_size * 2, channel_size * 4, kernel_size=4, stride=2, padding=1) #128 C
        self.bn3 = nn.BatchNorm2d(channel_size * 4)
        
        #down 4
        self.down4_res = down(channel_size * 4, channel_size * 8) #256 C

        #down 5
        self.down5_res = down(channel_size * 8, channel_size * 16)  #512 C

        #down 5
        self.down6_res = down_final(channel_size * 16, channel_size * 16)  #512 C

        self.up1 = up(channel_size * 16 * 2, channel_size * 16)# 512 + 512> 256

        self.up2 = up(channel_size * 8 * 2, channel_size * 8)# 256 + 256> 128

        self.up3 = up(channel_size * 4 * 2, channel_size * 4)# 128 + 128 > 64   112


        self.up_final = up_final(channel_size * 2 , channel_size )
        self.up4 = nn.Sequential(  #64 > 32 
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), # 224
            nn.Conv2d(channel_size, channel_size, kernel_size=1, stride=1, bias=False),# 224
            nn.BatchNorm2d(channel_size),
            nn.ReLU()
        )  # 224 32

        self.conv4 = nn.ConvTranspose2d(channel_size * 4, channel_size * 2, kernel_size=4, stride=2, padding=1,
                                        output_padding=1)
        self.bn4 = nn.BatchNorm2d(channel_size * 2)

        self.conv5 = nn.ConvTranspose2d(channel_size * 2, channel_size, kernel_size=4, stride=2, padding=2,
                                        output_padding=1)
        self.bn5 = nn.BatchNorm2d(channel_size)

        self.conv6 = nn.ConvTranspose2d(channel_size, channel_size, kernel_size=9, stride=1, padding=4)

        self.pos_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)
        self.cos_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)
        self.sin_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)
        self.width_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)

        self.dropout = dropout
        self.dropout_pos = nn.Dropout(p=prob)
        self.dropout_cos = nn.Dropout(p=prob)
        self.dropout_sin = nn.Dropout(p=prob)
        self.dropout_wid = nn.Dropout(p=prob)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, x_in):
        x1 = F.relu(self.bn1(self.conv1(x_in))) 
        print('x1 shape{}'.format(x1.shape))# 32 224
        x2 = F.relu(self.bn2(self.conv2(x1))) 
        print('x2 shape{}'.format(x2.shape))# 64 112
        x3 = F.relu(self.bn3(self.conv3(x2))) 
        print('x3 shape{}'.format(x3.shape))# 128 56

        x4 = self.down4_res(x3) 
        print('x4 shape{}'.format(x4.shape)) #256 28

        x5 = self.down5_res(x4) #512 14
        print('x5 shape{}'.format(x5.shape))

        x6 = self.down6_res(x5) #512 > 512 14
        print('x6 shape{}'.format(x6.shape))

        x55 = self.up1(x6, x5) #512 +512 256
        print('x55 shape{}'.format(x55.shape))
        x44 = self.up2(x55,x4) #256 +256 > 128
        print('x44 shape{}'.format(x44.shape))
        x33 = self.up3(x44,x3) # 128 + 128 > 64
        print('x33 shape{}'.format(x33.shape))
        print('x2 shape{}'.format(x2.shape))

        x22 = self.up_final(x33,x1) # 64 + 64 > 32


        if self.dropout:
            pos_output = self.pos_output(self.dropout_pos(x22))
            cos_output = self.cos_output(self.dropout_cos(x22))
            sin_output = self.sin_output(self.dropout_sin(x22))
            width_output = self.width_output(self.dropout_wid(x22))
        else:
            pos_output = self.pos_output(x22)
            cos_output = self.cos_output(x22)
            sin_output = self.sin_output(x22)
            width_output = self.width_output(x22)

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


from time import time
import multiprocessing as mp
print(f"num of CPU: {mp.cpu_count()}")
for num_workers in range(6, mp.cpu_count(), 2):  
    train_data = torch.utils.data.DataLoader(
        val_dataset_novel,
        batch_size=1,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    start = time()
    for i in tqdm.tqdm(train_data):
        pass
    end = time()
    logging.info("Finish with:{} second, num_workers={}".format(end - start, num_workers))
sys.exit() 



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

    def __init__(self, input_channels=1, output_channels=1, channel_size=32,use_mish=True,use_depthwise = True,dbg=0, att = 'use_eca',upsamp='use_bilinear',dropout=False, prob=0.0):
        super(GenerativeResnet, self).__init__()
        print('Model is grc3_imp_dwc1')
        print('GRCNN upsamp {}'.format(upsamp))
        print('USE mish {}'.format(use_mish))
        print('USE att {}'.format(att))

        self.dbg=dbg
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
        if self.dbg == 1:
            print('x_in.shape  {}'.format(x_in.shape))
        stem = self.stem(x_in)
        if self.dbg == 1:
            print('stem.shape  {}'.format(stem.shape))
        d1 = self.dsc1(stem)
        if self.dbg == 1:
            print('d1.shape  {}'.format(d1.shape))

        d2 = self.dsc2(d1)
        if self.dbg == 1:
            print('d2.shape  {}'.format(d2.shape))

        d3 = self.dsc3(d2)
        if self.dbg == 1:
            print('d3.shape  {}'.format(d3.shape))

        d4 = self.dscBottleNeck(d3)
        if self.dbg == 1:
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
