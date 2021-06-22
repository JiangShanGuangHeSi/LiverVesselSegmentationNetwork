'''
修改自model_3dUNet_v2.py: 
基于nnunet的改进3d unet
XueZhimeng, 2020.7

新增深度监督
XueZhimeng, 2021.6
'''
from copy import deepcopy
from torch import nn
import torch
import numpy as np
import torch.nn.functional

from copy import deepcopy
from torch import nn
import torch
import numpy as np
import torch.nn.functional

class InitWeights_He(object):
    def __init__(self, neg_slope=1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.ConvTranspose3d):
            module.weight = nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)


class ConvNormNonlinBlock3d(nn.Module):
    '''3d卷积+归一化+非线性 模块'''
    def __init__(self, in_channels, out_channels):
        super(ConvNormNonlinBlock3d, self).__init__()

        self.conv = nn.Conv3d(in_channels, out_channels, [3, 3, 3], stride=1, padding=1)
        self.norm = nn.InstanceNorm3d(out_channels, affine=True)
        self.nonlin = nn.LeakyReLU(negative_slope=1e-2, inplace=True)

    def forward(self, x):
        return self.nonlin(self.norm(self.conv(x)))


class StackedConvBlocks3d(nn.Module):
    '''多个3d卷积模块'''
    def __init__(self, in_channels, out_channels, num_blocks=2, basic_block=ConvNormNonlinBlock3d):
        super(StackedConvBlocks3d, self).__init__()

        self.blocks = nn.Sequential(
            *([basic_block(in_channels, out_channels)] +
              [basic_block(out_channels, out_channels) for _ in range(num_blocks - 1)]))

    def forward(self, x):
        return self.blocks(x)


def ConvTranspose3d(in_channels, out_channels):
    return nn.ConvTranspose3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)


def MaxPool3d():
    return nn.MaxPool3d(kernel_size=2, stride=2, padding=0)


class UNetV2ds(nn.Module):
    '''
    深度监督UNet
    相对UNet，添加了num_ds参数(5 >= num_ds >=1)，表示深度监督的深度。
    调用forward返回每一层多尺度预测结果组成的列表。（没有经过softmax）
    当num_ds == 1，这个网络和UNet效果相同。（但是结果是保存在列表里的）
    '''
    def __init__(self, in_channels, out_channels, base_filters_num=32, num_ds=4):
        super(UNetV2ds, self).__init__()
        self.weightInitializer=InitWeights_He(1e-2)

        if base_filters_num == 32:
            features = [in_channels, 32, 64, 128, 256, 320, 320]
        elif base_filters_num == 64:
            features = [in_channels, 64, 128, 256, 512, 512, 512]
        self.down_0 = StackedConvBlocks3d(in_channels, features[1])
        self.pool_0 = MaxPool3d()
        self.down_1 = StackedConvBlocks3d(features[1], features[2])
        self.pool_1 = MaxPool3d()
        self.down_2 = StackedConvBlocks3d(features[2], features[3])
        self.pool_2 = MaxPool3d()
        self.down_3 = StackedConvBlocks3d(features[3], features[4])
        self.pool_3 = MaxPool3d()
        self.down_4 = StackedConvBlocks3d(features[4], features[5])
        self.pool_4 = MaxPool3d()

        self.bottleneck = StackedConvBlocks3d(features[5], features[6])

        self.trans_4 = ConvTranspose3d(features[6], features[5])
        self.up_4 = StackedConvBlocks3d(features[5]*2, features[5])
        self.trans_3 = ConvTranspose3d(features[5], features[4])
        self.up_3 = StackedConvBlocks3d(features[4]*2, features[4])
        self.trans_2 = ConvTranspose3d(features[4], features[3])
        self.up_2 = StackedConvBlocks3d(features[3]*2, features[3])
        self.trans_1 = ConvTranspose3d(features[3], features[2])
        self.up_1 = StackedConvBlocks3d(features[2]*2, features[2])
        self.trans_0 = ConvTranspose3d(features[2], features[1])
        self.up_0 = StackedConvBlocks3d(features[1]*2, features[1])

        #<添加深度监督输出层>
        self.do_ds = True
        self.num_ds = num_ds
        self.seg_output = []
        for ds in range(num_ds): # 第0, 1, 2, ..., num_ds-1层深度监督，其中第0层与普通unet输出scale是一样的
            self.seg_output.append(nn.Conv3d(features[ds+1], out_channels, kernel_size=1, bias=False))
        
        self.seg_output = nn.ModuleList(self.seg_output) # 普通列表类型无法被nn.Module的.cuda()放到GPU上，需要转换为nn.ModuleList
        #</添加深度监督输出层>

        # self.seg_output_0 = nn.Conv3d(features[1], out_channels, kernel_size=1, bias=False)

        self.apply(self.weightInitializer)

    def forward(self, x):
        down_0 = self.down_0(x)
        pool_0 = self.pool_0(down_0)

        down_1 = self.down_1(pool_0)
        pool_1 = self.pool_1(down_1)

        down_2 = self.down_2(pool_1)
        pool_2 = self.pool_2(down_2)

        down_3 = self.down_3(pool_2)
        pool_3 = self.pool_3(down_3)

        down_4 = self.down_4(pool_3)
        pool_4 = self.pool_4(down_4)

        bottleneck = self.bottleneck(pool_4)

        trans_4 = self.trans_4(bottleneck)
        up_4 = self.up_4(torch.cat((trans_4, down_4), dim=1))

        trans_3 = self.trans_3(up_4)
        up_3 = self.up_3(torch.cat((trans_3, down_3), dim=1))

        trans_2 = self.trans_2(up_3)
        up_2 = self.up_2(torch.cat((trans_2, down_2), dim=1))

        trans_1 = self.trans_1(up_2)
        up_1 = self.up_1(torch.cat((trans_1, down_1), dim=1))

        trans_0 = self.trans_0(up_1)
        up_0 = self.up_0(torch.cat((trans_0, down_0), dim=1))

        # seg_output_0 = self.seg_output_0(up_0)

        # return seg_output_0

        #<添加深度监督输出层>
        # 应该是原始尺寸为output[0]，1/2为output[1]，……
        seg_output = []
        if self.do_ds:
            # 执行深度监督（train）
            for ds in range(self.num_ds):
                seg_output.append(self.seg_output[ds](eval("up_%d"%ds))) # 相当seg_output[0] = self.seg_output[0](up_0)
        else:
            # 不执行深度监督（val和test）
            # seg_output.append(up_0)
            # seg_output.append(self.seg_output[0](up_0))
            seg_output = self.seg_output[0](up_0)

        return seg_output
        #<\添加深度监督输出层>