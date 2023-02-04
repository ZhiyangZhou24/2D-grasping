import torch.nn as nn
import torch.nn.functional as F
import torch as t
from inference.models.grasp_model import GraspModel, ResidualBlock


class ConvBNLayer(nn.Module):
    def __init__(self,
                 in_channel=96,
                 out_channel=96,
                 kernel_size=3,
                 stride=1,
                 groups=1,
                 act='hard_swish'):
        super(ConvBNLayer, self).__init__()

        self.act = act
        assert self.act in ['leaky_relu', "hard_swish"]
        self.conv = nn.Conv2d(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=kernel_size,
            groups=groups,
            padding=(kernel_size - 1) // 2,
            stride=stride,
            bias=False)
        self.bn = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        x = self.bn(self.conv(x))
        if self.act == "leaky_relu":
            x = F.leaky_relu(x)
        elif self.act == "hard_swish":
            x = F.hardswish(x)
        return x


class DPModule(nn.Module):
    """
    Depth-wise and point-wise module.
     Args:
        in_channel (int): The input channels of this Module.
        out_channel (int): The output channels of this Module.
        kernel_size (int): The conv2d kernel size of this Module.
        stride (int): The conv2d's stride of this Module.
        act (str): The activation function of this Module,
                   Now support `leaky_relu` and `hard_swish`.
    """

    def __init__(self,
                 in_channel=96,
                 out_channel=96,
                 kernel_size=3,
                 stride=1,
                 act='hard_swish'):
        super(DPModule, self).__init__()
        self.act = act
        self.dwconv = nn.Conv2d(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=kernel_size,
            groups=out_channel,
            padding=(kernel_size - 1) // 2,
            stride=stride,
            bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.pwconv = nn.Conv2d(
            in_channels=out_channel,
            out_channels=out_channel,
            kernel_size=1,
            groups=1,
            padding=0,
            bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

    def act_func(self, x):
        if self.act == "leaky_relu":
            x = F.leaky_relu(x)
        elif self.act == "hard_swish":
            x = F.hardswish(x)
        return x

    def forward(self, x):
        x = self.act_func(self.bn1(self.dwconv(x)))
        x = self.act_func(self.bn2(self.pwconv(x)))
        return x


class DarknetBottleneck(nn.Module):
    """The basic bottleneck block used in Darknet.
    Each Block consists of two ConvModules and the input is added to the
    final output. Each ConvModule is composed of Conv, BN, and act.
    The first convLayer has filter size of 1x1 and the second one has the
    filter size of 3x3.
    Args:
        in_channels (int): The input channels of this Module.
        out_channels (int): The output channels of this Module.
        expansion (int): The kernel size of the convolution. Default: 0.5
        add_identity (bool): Whether to add identity to the out.
            Default: True
        use_depthwise (bool): Whether to use depthwise separable convolution.
            Default: False
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 expansion=0.5,
                 add_identity=True,
                 use_depthwise=False,
                 act="hard_swish"):
        super(DarknetBottleneck, self).__init__()
        hidden_channels = int(out_channels * expansion)
        conv_func = DPModule if use_depthwise else ConvBNLayer
        self.conv1 = ConvBNLayer(
            in_channel=in_channels,
            out_channel=hidden_channels,
            kernel_size=1,
            act=act)
        self.conv2 = conv_func(
            in_channel=hidden_channels,
            out_channel=out_channels,
            kernel_size=kernel_size,
            stride=1,
            act=act)
        self.add_identity = \
            add_identity and in_channels == out_channels

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)

        if self.add_identity:
            return out + identity
        else:
            return out


class CSPLayer(nn.Module):
    """Cross Stage Partial Layer.
    Args:
        in_channels (int): The input channels of the CSP layer.
        out_channels (int): The output channels of the CSP layer.
        expand_ratio (float): Ratio to adjust the number of channels of the
            hidden layer. Default: 0.5
        num_blocks (int): Number of blocks. Default: 1
        add_identity (bool): Whether to add identity in blocks.
            Default: True
        use_depthwise (bool): Whether to depthwise separable convolution in
            blocks. Default: False
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 expand_ratio=0.5,
                 num_blocks=1,
                 add_identity=True,
                 use_depthwise=False,
                 act="hard_swish"):
        super().__init__()
        mid_channels = int(out_channels * expand_ratio)
        self.main_conv = ConvBNLayer(in_channels, mid_channels, 1, act=act)
        self.short_conv = ConvBNLayer(in_channels, mid_channels, 1, act=act)
        self.final_conv = ConvBNLayer(
            2 * mid_channels, out_channels, 1, act=act)

        self.blocks = nn.Sequential(* [
            DarknetBottleneck(
                mid_channels,
                mid_channels,
                kernel_size,
                1.0,
                add_identity,
                use_depthwise,
                act=act) for _ in range(num_blocks)
        ])

    def forward(self, x):
        x_short = self.short_conv(x)

        x_main = self.main_conv(x)
        x_main = self.blocks(x_main)

        x_final = t.cat((x_main, x_short), dim=1)
        return self.final_conv(x_final)

def channel_shuffle(x, num_groups):
    batch_size, num_channels, height, width = x.size()
    assert num_channels % num_groups == 0
    x = x.view(batch_size, num_groups, num_channels // num_groups, height, width)
    x = x.permute(0, 2, 1, 3, 4)
    return x.contiguous().view(batch_size, num_channels, height, width)

class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channels=channel,
                               out_channels=channel // reduction,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)
        self.conv2 = nn.Conv2d(in_channels=channel // reduction,
                               out_channels=channel,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)

    def forward(self, inputs):
        outputs = self.avg_pool(inputs)
        outputs = F.relu(self.conv1(outputs))
        outputs = F.hardsigmoid(self.conv2(outputs))

        return inputs * outputs.expand_as(inputs)

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

class InvertedResidualDS(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, stride, act='hard_swish'):
        super(InvertedResidualDS, self).__init__()
        if act == 'relu':
            act = nn.ReLU
        elif act == 'relu6':
            act = nn.ReLU6
        elif act == 'leakyrelu':
            act = nn.LeakyReLU
        elif act == 'hard_swish':
            act = nn.Hardswish
        else:
            raise ValueError("the act is not available")

        self._conv_dw_1 = conv_3x3_bn(in_channels, in_channels, act=nn.ReLU, stride=stride, group=in_channels)
        self._conv_linear_1 = conv_1x1_bn(in_channels, out_channels//2, act=act)
        self._conv_pw_2 = conv_1x1_bn(in_channels, mid_channels//2, act=act)
        self._conv_dw_2 = conv_3x3_bn(mid_channels//2, mid_channels//2, stride=stride, group=mid_channels//2, act=nn.ReLU)
        self._se = SEModule(mid_channels//2)
        self._conv_linear_2 = conv_1x1_bn(mid_channels//2, out_channels//2, act=act)
        self._conv_dw_mv1 = conv_3x3_bn(out_channels, out_channels, group=out_channels, act=nn.Hardswish)
        self._conv_pw_mv1 = conv_1x1_bn(out_channels, out_channels, act=nn.Hardswish)

    def forward(self, inputs):
        x1 = self._conv_dw_1(inputs)
        x1 = self._conv_linear_1(x1)
        x2 = self._conv_pw_2(inputs)
        x2 = self._conv_dw_2(x2)
        x2 = self._se(x2)
        x2 = self._conv_linear_2(x2)
        out = t.cat([x1, x2], dim=1)
        out = self._conv_dw_mv1(out)
        out = self._conv_pw_mv1(out)

        return out
class InvertedResidual(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, stride, act='hard_swish'):
        super(InvertedResidual, self).__init__()
        if act == 'relu':
            act = nn.ReLU
        elif act == 'relu6':
            act = nn.ReLU6
        elif act == 'leakyrelu':
            act = nn.LeakyReLU
        elif act == 'hard_swish':
            act = nn.Hardswish
        else:
            raise ValueError("the act is not available")
        self._conv_pw = conv_1x1_bn(in_channels//2, mid_channels//2, act=act)
        self._conv_dw = conv_3x3_bn(mid_channels//2, mid_channels//2, act=nn.ReLU, group=mid_channels//2, stride=stride)
        self._se = SEModule(mid_channels)

        self._conv_linear = conv_1x1_bn(mid_channels, out_channels//2, act=act)

    def forward(self, inputs):
        x1, x2 = t.split(inputs, [inputs.shape[1]//2, inputs.shape[1]//2], dim=1)
        x2 = self._conv_pw(x2)
        x3 = self._conv_dw(x2)
        x3 = t.cat([x2, x3], dim=1)
        x3 = self._se(x3)
        x3 = self._conv_linear(x3)
        out = t.cat([x1, x3], dim=1)
        return channel_shuffle(out, 2)

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
        self.conv1 = conv_block(input_channels,channel_size,kernel_size=3, stride=1, padding=1)
        #self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)


        self.conv2 = InvertedResidualDS(in_channels= channel_size, mid_channels= channel_size * 2,  out_channels= channel_size * 2, stride=2)

        self.conv3 = InvertedResidualDS(in_channels= channel_size * 2, mid_channels= channel_size * 4, out_channels=channel_size * 4,stride=2)

        self.es1 = InvertedResidual(in_channels= channel_size * 4, mid_channels= channel_size * 4, out_channels=channel_size * 4,stride=1)
        self.es2 = InvertedResidual(in_channels= channel_size * 4, mid_channels= channel_size * 4, out_channels=channel_size * 4,stride=1)
        self.es3 = InvertedResidual(in_channels= channel_size * 4, mid_channels= channel_size * 4, out_channels=channel_size * 4,stride=1)
        self.es4 = InvertedResidual(in_channels= channel_size * 4, mid_channels= channel_size * 4, out_channels=channel_size * 4,stride=1)
        self.es5 = InvertedResidual(in_channels= channel_size * 4, mid_channels= channel_size * 4, out_channels=channel_size * 4,stride=1)

        self.up_res5 = CSPLayer(channel_size * 8, channel_size * 4, kernel_size=3)

        self.conv4 = up_conv(channel_size * 4, channel_size * 2, kernel_size=4, stride=2, padding=1)
        self.up_conv4 = CSPLayer(channel_size * 4, channel_size * 2, kernel_size=3)

        self.conv5 = up_conv(channel_size * 2, channel_size, kernel_size=6, stride=2, padding=2)
        self.up_conv5 = CSPLayer(channel_size * 2, channel_size, kernel_size=3)

        #self.conv6 = nn.ConvTranspose2d(channel_size, channel_size, kernel_size=9, stride=1, padding=4)
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
        e1 = self.conv1(x_in)

        e2 = self.conv2(e1)

        e3 = self.conv3(e2)

        es1 = self.es1(e3)
        es2 = self.es2(es1)
        es3 = self.es3(es2)
        es4 = self.es4(es3)
        es5 = self.es5(es4)

        es5 = t.cat((e3,es5),dim=1)
        es5 = self.up_res5(es5)

        d3 = self.conv4(es5)
        d3 = t.cat((e2,d3),dim=1)
        d3 = self.up_conv4(d3)

        d2 = self.conv5(d3)
        d2 = t.cat((e1,d2),dim=1)
        d2 = self.up_conv5(d2)

        d1 = self.conv6(d2)

        if self.dropout:
            pos_output = self.pos_output(self.dropout_pos(d1))
            cos_output = self.cos_output(self.dropout_cos(d1))
            sin_output = self.sin_output(self.dropout_sin(d1))
            width_output = self.width_output(self.dropout_wid(d1))
        else:
            pos_output = self.pos_output(d1)
            cos_output = self.cos_output(d1)
            sin_output = self.sin_output(d1)
            width_output = self.width_output(d1)

        return pos_output, cos_output, sin_output, width_output
