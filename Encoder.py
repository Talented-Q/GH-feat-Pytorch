import torch.nn as nn
import numpy as np
from torch.nn import init
import torch
import math
import torch.nn.functional as F
from kornia.filters import filter2d

class Bottleneck(nn.Module):
    # 每个stage维度中扩展的倍数
    def __init__(self, inplanes, planes, stride, downsample=None):
        '''

        :param inplanes: 输入block的之前的通道数
        :param planes: 在block中间处理的时候的通道数
                planes*self.extention:输出的维度
        :param stride:
        :param downsample:
        '''
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)

        # self.relu = nn.ReLU()

        # 判断残差有没有卷积
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        # 参差数据
        residual = x

        # 卷积操作
        out = self.conv1(x)
        out = self.bn1(out)
        # out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        # out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        # out = self.relu(out)

        # 是否直连（如果Indentity blobk就是直连；如果Conv2 Block就需要对残差边就行卷积，改变通道数和size
        if self.downsample is not None:
            residual = self.downsample(x)

        # 将残差部分和卷积部分相加
        out += residual
        # out = self.relu(out)

        return out

class Blur(nn.Module):
    def __init__(self):
        super().__init__()
        f = torch.Tensor([1, 2, 1])
        self.register_buffer('f', f)
    def forward(self, x):
        f = self.f
        f = f[None, None, :] * f [None, :, None]
        '''
        f:tensor
        ([[[1., 2., 1.],
         [2., 4., 2.],
         [1., 2., 1.]]])
         相当于f的列向量乘以行向量得到一个(1,3,3)的矩阵，作为卷积核，其实就是高斯滤波器模板，用于抑制噪声，平滑图像
        '''
        return filter2d(x, f, normalized=True)

class EqualConv2d(nn.Module):
    def __init__(
        self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

    def forward(self, input):
        out = F.conv2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},"
            f" {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})"
        )

class ConvLayer(nn.Sequential):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        bias=True,
        activate=True,
    ):
        layers = []

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            layers.append(Blur())

            stride = 2
            self.padding = 0

        else:
            stride = 1
            self.padding = kernel_size // 2

        layers.append(
            EqualConv2d(
                in_channel,
                out_channel,
                kernel_size,
                padding=self.padding,
                stride=stride,
                bias=bias and not activate,
            )
        )

        super().__init__(*layers)

class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()

        self.conv1 = ConvLayer(in_channel, in_channel, 3)
        self.conv2 = ConvLayer(in_channel, out_channel, 1, downsample=True)

        self.skip = ConvLayer(
            in_channel, out_channel, 1, downsample=True, activate=False, bias=False
        )

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)
        skip = self.skip(input)
        out = (out + skip) / math.sqrt(2)

        return out

def downscale2d(x, factor=2):
    # 使用最大池化作为下采样
    assert isinstance(factor, int) and factor >= 1
    if factor == 1:
        return x
    else:
        return nn.AvgPool2d(2,2)(x)


def upscale2d(x, factor=2):
    # 使用reshape操作插值上采样
    assert isinstance(factor, int) and factor >= 1
    if factor == 1: return x
    else:
        return nn.Upsample(scale_factor=2)(x)


class Encoder(nn.Module):
    def __init__(self, block, layers, emb_dim=256, fout=512):
        # inplane=当前的fm的通道数
        self.inplane = 64
        super(Encoder, self).__init__()

        # 参数
        self.block = block
        self.layers = layers

        self.max_length = emb_dim * 2
        self.dsize = [self.max_length] * 8 + [self.max_length // 2] * 2 + [self.max_length // 4] * 2 + [self.max_length // 8] * 2

        # stem的网络层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplane)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # 64,128,256,512指的是扩大4倍之前的维度，即Identity Block中间的维度
        self.stage1 = self.make_layer(self.block, 256, layers[0], stride=1)
        self.stage2 = self.make_layer(self.block, 512, layers[1], stride=2)
        self.stage3 = self.make_layer(self.block, 1024, layers[2], stride=2)
        self.stage4 = self.make_layer(self.block, 2048, layers[3], stride=2)
        self.conv5 = ResBlock(2048, 2048)  # 512 128 128

        self.norm4 = nn.BatchNorm1d(sum(self.dsize[8:]))
        self.norm5 = nn.BatchNorm1d(sum(self.dsize[4:8]))
        self.norm6 = nn.BatchNorm1d(sum(self.dsize[:4]))

        self.conv11 = nn.Conv2d(1024,fout,kernel_size=3,stride=1, padding=1)
        self.conv12 = nn.Conv2d(fout, fout, kernel_size=3, stride=1, padding=1)
        self.conv13 = nn.Conv2d(fout, fout, kernel_size=3, stride=1, padding=1)

        self.conv21 = nn.Conv2d(2048, fout, kernel_size=3, stride=1, padding=1)
        self.conv22 = nn.Conv2d(fout, fout, kernel_size=3, stride=1, padding=1)
        self.conv23 = nn.Conv2d(fout, fout, kernel_size=3, stride=1, padding=1)

        self.conv31 = nn.Conv2d(2048, fout, kernel_size=3, stride=1, padding=1)
        self.conv32 = nn.Conv2d(fout, fout, kernel_size=3, stride=1, padding=1)
        self.conv33 = nn.Conv2d(fout, fout, kernel_size=3, stride=1, padding=1)

        self.convs1 = nn.ModuleList([self.conv11,self.conv21,self.conv31])
        self.convs2 = nn.ModuleList([self.conv12,self.conv22,self.conv32])
        self.convs3 = nn.ModuleList([self.conv13, self.conv23, self.conv33])

    def reshape(self,x):
        if len(x.shape) > 2:
            x = torch.reshape(x, [-1, np.prod([d for d in x.shape[1:]])])
            return x

    def fpn(self, inputs, start_level=3):
        # len(inputs) == 6 0,1,2,3,4,5
        laterals = []
        for i in range(start_level, len(inputs)):
            linput = inputs[i]
            linput = self.convs1[i-start_level](linput)
            laterals.append(linput)

        flevel = len(laterals)
        for i in range(flevel - 1, 0, -1):
            laterals[i - 1] += upscale2d(laterals[i])  # 对中间向量上采样

        outs = []
        for i in range(0, flevel):
            finput = laterals[i]
            finput = self.convs2[i](finput)
            outs.append(finput)
        return outs
    #
    def sam(self, inputs):
        # recurrent downsample
        for i in range(len(inputs) - 1):
            for j in range(0, len(inputs) - 1 - i):
                inputs[j] = downscale2d(inputs[j])

        for i in range(len(inputs)):
            inputs[i] = self.convs3[i](inputs[i])
        # latent_fusion
        for i in range(len(inputs) - 1):
            inputs[i] = inputs[i] + inputs[-1]

        return inputs

    def forward(self, x):
        # stem部分：conv+bn+maxpool
        out = self.conv1(x)
        out = self.bn1(out)
        x1 = self.relu(out)  # [10, 512, 256, 256]
        out = self.maxpool(x1)

        # block部分
        x2 = self.stage1(out)  # [10, 512, 128, 128]

        x3 = self.stage2(x2)  # [10, 512, 64, 64]  this result is res3

        x4 = self.stage3(x3)  # [10, 512, 32, 32]  this result is res4

        x5 = self.stage4(x4)  # [10, 512, 16, 16]  this result is res5

        x6 = self.conv5(x5)  # [10, 512, 8, 8]  this result is res6

        self.list = (x1,x2,x3,x4,x5,x6)
        out = self.fpn(self.list)
        out = self.sam(out)

        res4, res5, res6 = out

        res4 = self.reshape(res4)
        res4 = nn.Linear(res4.shape[1], sum(self.dsize[8:])).to('cuda')(res4)
        res4 = self.norm4(res4)
        res40 = torch.reshape(res4[:, :sum(self.dsize[8:10])], [-1, 2, 256])
        res40 = res40.repeat([1, 1, self.max_length // 256])
        res41 = torch.reshape(res4[:, sum(self.dsize[8:10]):sum(self.dsize[8:12])], [-1, 2, 128])
        res41 = res41.repeat([1, 1, self.max_length // 128])
        res42 = torch.reshape(res4[:, sum(self.dsize[8:12]):], [-1, 2, 64])
        res42 = res42.repeat([1, 1, self.max_length // 64])

        res5 = self.reshape(res5)
        res5 = nn.Linear(res5.shape[1], sum(self.dsize[4:8])).to('cuda')(res5)
        res5 = self.norm5(res5)
        res5 = torch.reshape(res5, [-1, 4, self.dsize[4]])

        res6 = self.reshape(res6)
        res6 = nn.Linear(res6.shape[1], sum(self.dsize[:4])).to('cuda')(res6)
        res6 = self.norm6(res6)
        res6 = torch.reshape(res6, [-1, 4, self.dsize[0]])

        styles = torch.cat([res6,res5,res40,res41,res42], dim=1)

        return styles

    def make_layer(self, block, plane, block_num, stride=1):
        '''
        :param block: block模板
        :param plane: 每个模块中间运算的维度，一般等于输出维度/4
        :param block_num: 重复次数
        :param stride: 步长
        :return:
        '''
        block_list = []
        # 先计算要不要加downsample
        downsample = None
        if (stride != 1 or self.inplane != plane):
            downsample = nn.Sequential(
                nn.Conv2d(self.inplane, plane, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(plane)
            )

        # Conv Block输入和输出的维度（通道数和size）是不一样的，所以不能连续串联，他的作用是改变网络的维度
        # Identity Block 输入维度和输出（通道数和size）相同，可以直接串联，用于加深网络
        # Conv_block
        conv_block = block(self.inplane, plane, stride=stride, downsample=downsample)
        block_list.append(conv_block)
        self.inplane = plane

        # Identity Block
        for i in range(1, block_num):
            block_list.append(block(self.inplane, plane, stride=1))

        return nn.Sequential(*block_list)


