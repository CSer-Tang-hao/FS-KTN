# -*-coding:utf-8-*-
import math

# 引入torch模块
import torch

# 引入torch.nn
import torch.nn as nn

# 卷积模块
import torch.nn.functional as F

# 卷积模块
class ConvBlock(nn.Module):
    # 初始化实例对象
    def __init__(self, in_planes, out_planes, userelu=True):
        # 对继承自父类的属性进行初始化
        super(ConvBlock, self).__init__()
        # 时序容器，快速搭建神经网络
        self.layers = nn.Sequential()
        # 卷积层，卷积核尺寸为3*3，步长为1，填充为1
        self.layers.add_module('Conv', nn.Conv2d(in_planes, out_planes,
            kernel_size=3, stride=1, padding=1, bias=False))

        # batchnorm层
        self.layers.add_module('BatchNorm', nn.BatchNorm2d(out_planes))

        # Relu层
        if userelu:
            self.layers.add_module('ReLU', nn.ReLU(inplace=True))

        # 最大池化层，尺寸2*2
        self.layers.add_module(
            'MaxPool', nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

    # 前向传播
    def forward(self, x):
        out = self.layers(x)
    # 获取输出
        return out

# 卷积网络
class ConvNet(nn.Module):
    # 初始化实例对象
    def __init__(self, in_planes, out_planes, num_stages,userelu=False):

        # 对继承自父类的属性进行初始化
        super(ConvNet, self).__init__()

        # 获取输入通道的大小
        self.in_planes  = in_planes

        # 获取输出通道的大小
        self.out_planes = out_planes

        # 获取网络层数
        self.num_stages = num_stages

        # 获取每层的输出通道数
        if type(self.out_planes) == int:
            self.out_planes = [self.out_planes for i in range(self.num_stages)]
        assert(type(self.out_planes)==list and len(self.out_planes)==self.num_stages)

        num_planes = [self.in_planes,] + self.out_planes



        conv_blocks = []
        # 定义4层卷积网络
        for i in range(self.num_stages):
            if i == (self.num_stages-1):
                conv_blocks.append(
                    ConvBlock(num_planes[i], num_planes[i+1], userelu=userelu))
            else:
                conv_blocks.append(
                    ConvBlock(num_planes[i], num_planes[i+1]))
        # 构建卷积网络
        self.conv_blocks = nn.Sequential(*conv_blocks)

        # 初始化卷积网络参数
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        # 进行前向传播
        out = self.conv_blocks(x)
        # 将输出转换为1维
        out = out.view(out.size(0),-1)
        # 获取输出
        return out

# 构建模型
# def create_model(opt):
#     return ConvNet(opt)
def Conv64():
    return  ConvNet(in_planes=3,out_planes=[64,64,64,64],num_stages=4,userelu=False)

def Conv128():
    return  ConvNet(in_planes=3,out_planes=[64,64,128,128],num_stages=4,userelu=False)
