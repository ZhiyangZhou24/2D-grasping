import torch
import torch.nn as nn
import torch.nn.functional as F
from numbers import Integral
from inference.models.attention import CoordAtt, eca_block, se_block,cbam_block
import math

class Mish(nn.Module):
    def __init__(self):
        super().__init__()
        print("Mish activation loaded...")
    def forward(self,x):
        x = x * (torch.tanh(F.softplus(x)))
        return x

def mish(x):
    return x*(torch.tanh(F.softplus(x)))

def make_divisible(v, divisor=16, min_value=None):
    if min_value is None:
        min_value = divisor

    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


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


def channel_shuffle(x, num_groups):
    batch_size, num_channels, height, width = x.size()
    assert num_channels % num_groups == 0
    x = x.view(batch_size, num_groups, num_channels // num_groups, height, width)
    x = x.permute(0, 2, 1, 3, 4)
    return x.contiguous().view(batch_size, num_channels, height, width)

# this module is different from pp-lcnet
class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()
        print("SEModule loaded")
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
        x1, x2 = torch.split(inputs, [inputs.shape[1]//2, inputs.shape[1]//2], dim=1)
        x2 = self._conv_pw(x2)
        x3 = self._conv_dw(x2)
        x3 = torch.cat([x2, x3], dim=1)
        x3 = self._se(x3)
        x3 = self._conv_linear(x3)
        out = torch.cat([x1, x3], dim=1)
        return channel_shuffle(out, 2)


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
        out = torch.cat([x1, x2], dim=1)
        out = self._conv_dw_mv1(out)
        out = self._conv_pw_mv1(out)

        return out


class ESNet(nn.Module):
    def __init__(self, scale=1.0, act="hard_swish", feature_maps=[4, 11, 14],
                 channel_ratio=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]):
        super(ESNet, self).__init__()
        if act == 'relu':
            act_fn = nn.ReLU
        elif act == 'relu6':
            act_fn = nn.ReLU6
        elif act == 'leakyrelu':
            act_fn = nn.LeakyReLU
        elif act == 'hard_swish':
            act_fn = nn.Hardswish
        else:
            raise ValueError("the act is not available")
        self.scale = scale
        if isinstance(feature_maps, Integral):
            feature_maps = [feature_maps]
        self.feature_maps = feature_maps
        stage_repeats = [3, 7, 3]
        stage_out_channels = [
            -1, 24, make_divisible(128 * scale), make_divisible(256 * scale),
            make_divisible(512 * scale), 1024
        ]

        self._out_channels = []
        self._feature_idx = 0
        # 1. conv1

        self._conv1 = conv_3x3_bn(3, stage_out_channels[1], stride=2, act=act_fn)
        self._max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self._feature_idx += 1

        # 2. bottle sequences
        self._block_list = []
        arch_idx = 0
        for stage_id, num_repeat in enumerate(stage_repeats):
            for i in range(num_repeat):
                channels_scales = channel_ratio[arch_idx]
                mid_c = make_divisible(
                    int(stage_out_channels[stage_id + 2] * channels_scales),
                    divisor=8)
                if i == 0:
                    block = InvertedResidualDS(
                            in_channels=stage_out_channels[stage_id + 1],
                            mid_channels=mid_c,
                            out_channels=stage_out_channels[stage_id + 2],
                            stride=2,
                            act=act)
                else:
                    block = InvertedResidual(
                            in_channels=stage_out_channels[stage_id + 2],
                            mid_channels=mid_c,
                            out_channels=stage_out_channels[stage_id + 2],
                            stride=1,
                            act=act)
                self._block_list.append(block)
                arch_idx += 1
                self._feature_idx += 1
                self._update_out_channels(stage_out_channels[stage_id + 2],
                                          self._feature_idx, self.feature_maps)

    def _update_out_channels(self, channel, feature_idx, feature_maps):
        if feature_idx in feature_maps:
            self._out_channels.append(channel)

    def forward(self, inputs):
        y = self._conv1(inputs)
        y = self._max_pool(y)
        outs = []
        for i, inv in enumerate(self._block_list):
            y = inv(y)
            if i + 2 in self.feature_maps:
                outs.append(y)

        return outs


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
        assert self.act in ['leaky_relu', "hard_swish","mish"]
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
            x = F.leaky_relu(x,inplace=True)
        elif self.act == "hard_swish":
            x = F.hardswish(x,inplace=True)
        elif self.act == "mish":
            x = F.mish(x,inplace=True)
        return x

class DepthwiseSeparable(nn.Module):
    def __init__(self,
                 num_channels,
                 num_filters,
                 stride,
                 dw_size=3,
                 att_type = 'use_se',
                 use_identity = False,
                 act="hard_swish"):
        super().__init__()
        self.att_type = att_type
        self.use_identity = use_identity
        assert self.att_type in ['use_eca', 'use_se','use_coora',"use_cbam"]
        self.dw_conv = ConvBNLayer(
            in_channel=num_channels,
            out_channel=num_channels,
            kernel_size=dw_size,
            stride=stride,
            groups=num_channels,
            act=act)
        if self.att_type !=None:
            self.att = self._make_att(num_channels,num_channels)
        self.pw_conv = ConvBNLayer(
            in_channel=num_channels,
            kernel_size=1,
            out_channel=num_filters,
            stride=1,
            act=act)

        if self.use_identity:
            self.channel_conv = nn.Sequential(
                nn.Conv2d(num_channels, num_filters, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(num_filters),
            )
    def _make_att(self, in_channels, out_channels,reduc_ratio=32):
        if self.att_type == 'use_coora':
            print('use_coora reduc_ratio = {}'.format(reduc_ratio))
            return CoordAtt(in_channels,out_channels,reduc_ratio)
        elif self.att_type == 'use_eca':
            print('use_eca ')
            return eca_block(out_channels)
        elif self.att_type == 'use_se':
            print('use_se ')
            return SEModule(out_channels)
        elif self.att_type == 'use_cbam':
            print('use_cbam reduc_ratio = {}'.format(reduc_ratio))
            return cbam_block(out_channels,ratio=reduc_ratio)
        else :
            print('att_type error , please check!!!!')
    def forward(self, x):
        if self.use_identity:
            residual = x
        x = self.dw_conv(x)
        if self.att_type !=None:
            x = self.att(x)
        x = self.pw_conv(x)

        if self.use_identity:
            if residual.shape[1] != x.shape[1]:
                residual = self.channel_conv(residual)
            x += residual
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
                 act='leaky_relu'):
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
            x = F.leaky_relu(x,inplace=True)
        elif self.act == "hard_swish":
            x = F.hardswish(x,inplace=True)
        elif self.act == "mish":
            x = F.mish(x)
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
                 use_att=False,
                 act="leaky_relu"):
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
        
        self.use_att = use_att 
        self.att = eca_block(hidden_channels)

        self.add_identity = \
            add_identity and in_channels == out_channels

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        if self.use_att:
            out = self.att(out)
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
                 use_att=False,
                 act="leaky_relu"):
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
                use_att,
                act=act) for _ in range(num_blocks)
        ])

    def forward(self, x):
        x_short = self.short_conv(x)

        x_main = self.main_conv(x)
        x_main = self.blocks(x_main)

        x_final = torch.cat((x_main, x_short), dim=1)
        return self.final_conv(x_final)
    
class Channel_T(nn.Module):
    def __init__(self,
                 in_channels=[116, 232, 464],
                 out_channels=96,
                 act="leaky_relu"):
        super(Channel_T, self).__init__()
        self.convs = nn.ModuleList()
        for i in range(len(in_channels)):
            self.convs.append(
                ConvBNLayer(
                    in_channels[i], out_channels, 1, act=act))

    def forward(self, x):
        outs = [self.convs[i](x[i]) for i in range(len(x))]
        return outs


class CSPPAN(nn.Module):
    """Path Aggregation Network with CSP module.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        kernel_size (int): The conv2d kernel size of this Module.
        num_features (int): Number of output features of CSPPAN module.
        num_csp_blocks (int): Number of bottlenecks in CSPLayer. Default: 1
        use_depthwise (bool): Whether to depthwise separable convolution in
            blocks. Default: True
    """

    def __init__(self,
                 in_channels,
                 out_channels=128,
                 kernel_size=5,
                 num_features=4,
                 num_csp_blocks=1,
                 use_depthwise=True,
                 act='hard_swish',
                 spatial_scales=[0.125, 0.0625, 0.03125]):
        super(CSPPAN, self).__init__()
        self.conv_t = Channel_T(in_channels, out_channels, act=act)
        in_channels = [out_channels] * len(spatial_scales)  #384
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spatial_scales = spatial_scales
        self.num_features = num_features
        conv_func = DPModule if use_depthwise else ConvBNLayer

        if self.num_features == 4:
            self.first_top_conv = conv_func(
                in_channels[0], in_channels[0], kernel_size, stride=2, act=act)
            self.second_top_conv = conv_func(
                in_channels[0], in_channels[0], kernel_size, stride=2, act=act)
            self.spatial_scales.append(self.spatial_scales[-1] / 2)

        # build top-down blocks
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.top_down_blocks = nn.ModuleList()
        for idx in range(len(in_channels) - 1, 0, -1):
            self.top_down_blocks.append(
                CSPLayer(
                    in_channels[idx - 1] * 2,
                    in_channels[idx - 1],
                    kernel_size=kernel_size,
                    num_blocks=num_csp_blocks,
                    add_identity=False,
                    use_depthwise=use_depthwise,
                    act=act))

        # build bottom-up blocks
        self.downsamples = nn.ModuleList()
        self.bottom_up_blocks = nn.ModuleList()
        for idx in range(len(in_channels) - 1):
            self.downsamples.append(
                conv_func(
                    in_channels[idx],
                    in_channels[idx],
                    kernel_size=kernel_size,
                    stride=2,
                    act=act))
            self.bottom_up_blocks.append(
                CSPLayer(
                    in_channels[idx] * 2,
                    in_channels[idx + 1],
                    kernel_size=kernel_size,
                    num_blocks=num_csp_blocks,
                    add_identity=False,
                    use_depthwise=use_depthwise,
                    act=act))

    def forward(self, inputs):
        """
        Args:
            inputs (tuple[Tensor]): input features.

        Returns:
            tuple[Tensor]: CSPPAN features.
        """
        assert len(inputs) == len(self.in_channels)
        inputs = self.conv_t(inputs)

        # top-down path
        inner_outs = [inputs[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_heigh = inner_outs[0]
            feat_low = inputs[idx - 1]

            upsample_feat = self.upsample(feat_heigh)

            inner_out = self.top_down_blocks[len(self.in_channels) - 1 - idx](
                torch.cat([upsample_feat, feat_low], 1))
            inner_outs.insert(0, inner_out)

        # bottom-up path
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            downsample_feat = self.downsamples[idx](feat_low)
            out = self.bottom_up_blocks[idx](torch.cat(
                [downsample_feat, feat_height], 1))
            outs.append(out)

        top_features = None
        if self.num_features == 4:
            top_features = self.first_top_conv(inputs[-1])
            top_features = top_features + self.second_top_conv(outs[-1])
            outs.append(top_features)

        return tuple(outs)