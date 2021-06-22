'''
An Attention-guided Deep Neural Network with
Multi-scale Feature Fusion for Liver Vessel
Segmentation 
复现尝试1 ... ACG模块

XueZhimeng, 2021.5
'''
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


class SpatialAttentionBlock(nn.Module):
    '''
    空间注意模块(Spatial Attention Block, SA)
    1×1×1卷积 + BN + ReLU → 1×1×1卷积 → sigmoid
    【ATTENTION】这里用的是instance norm而不是batch normal，不知道有没有影响
    也不知道outchannels应该是多少，反正输出通道数量一定是1
    '''
    def __init__(self, in_channels, out_channels):
        super(SpatialAttentionBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, [1, 1, 1], stride=1, padding=0)
        self.norm = nn.InstanceNorm3d(out_channels, affine=True)
        self.nonlin = nn.LeakyReLU(negative_slope=1e-2, inplace=True)
        self.conv2 = nn.Conv3d(in_channels, out_channels, [1, 1, 1], stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.conv2(self.nonlin(self.norm(self.conv1(x)))))


class AttentionGuidedConcatenationBlock(nn.Module):
    '''
    注意力引导连接模块(Attention-Guided Concatenation, AGC)
    高水平特征f_h反卷积→f'_h，
    → f'_h通过空间注意模块SA() → 0-1特征图A
    → A×f_l → 优化的低水平特征f'_l
    理论上输出通道数量与f_l一致
    '''
    def __init__(self, f_h_channels, f_l_channels):
        super(AttentionGuidedConcatenationBlock, self).__init__()

        # self.deconv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.deconv = nn.ConvTranspose3d(f_h_channels, f_l_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.SA = SpatialAttentionBlock(f_l_channels, f_l_channels)
    
    def forward(self, f_h, f_l):
        '''理论上输出通道数量与f_l一致'''
        f_h1 = self.deconv(f_h)
        A = self.SA(self.deconv(f_h))
        return self.SA(self.deconv(f_h)) * f_l


def softmax_helper(x):
    # copy from: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/utilities/nd_softmax.py
    rpt = [1 for _ in range(len(x.size()))]
    rpt[1] = x.size(1)
    x_max = x.max(1, keepdim=True)[0].repeat(*rpt)
    e_x = torch.exp(x - x_max)
    return e_x / e_x.sum(1, keepdim=True).repeat(*rpt)


class MultiScaleFeatureFusionConvBlock3d(nn.Module):
    '''
    多尺度特征融合模块+卷积模块

    多尺度特征融合模块(Multi-Scale Feature Fusion, MSFF)
    Split:
    将输入的特征通道通过1×1×1卷积，再分为4份f_1, f_2, f_3, f_4
    Addition:
    f'_1 = f_1
    f'_2 = Cov_3(f_1)
    f'_3 = Cov_3(f_3 + f'_2)
    f'_4 = Cov_3(f_4 + f'_3)
    Fusion:
    f''_i = f'_i + (f'_1 + f'_2 + f'_3 + f'_4)
    y = Cov_1(Concat(f''_1, f''_2, f''_3, f''_4))

    卷积模块同ConvNormNonlinBlock3d
    '''
    def __init__(self, in_channels, out_channels):
        super(MultiScaleFeatureFusionConvBlock3d, self).__init__()
        self.in_channels_split = in_channels // 4
        self.out_channels_split = out_channels // 4
        self.conv1_in = nn.Conv3d(in_channels, out_channels, [1, 1, 1], stride=1, padding=0)
        self.conv3_addition_2 = ConvNormNonlinBlock3d(self.out_channels_split, self.out_channels_split)
        self.conv3_addition_3 = ConvNormNonlinBlock3d(self.out_channels_split, self.out_channels_split)
        self.conv3_addition_4 = ConvNormNonlinBlock3d(self.out_channels_split, self.out_channels_split)
        self.conv1_out = nn.Conv3d(out_channels, out_channels, [1, 1, 1], stride=1, padding=0)

        self.conv3 = ConvNormNonlinBlock3d(out_channels, out_channels)
    
    def forward(self, x):
        f = self.conv1_in(x)
        f_1 = f[:, 0 : self.out_channels_split, :, :, :]
        f_2 = self.conv3_addition_2(f[:, self.out_channels_split : 2 * self.out_channels_split, :, :, :])
        f_3 = self.conv3_addition_3(f[:, 2 * self.out_channels_split : 3 * self.out_channels_split, :, :, :] + f_2)
        f_4 = self.conv3_addition_4(f[:, 3 * self.out_channels_split : 4 * self.out_channels_split, :, :, :] + f_3)
        fusion = f_1 + f_2 + f_3 + f_4
        f_1 = f_1 + fusion
        f_2 = f_2 + fusion
        f_3 = f_3 + fusion
        f_4 = f_4 + fusion
        return self.conv3(self.conv1_out(torch.cat((f_1, f_2, f_3, f_4), dim=1)))


class UNetMSFF(nn.Module):
    '''与UNetV2的唯一区别是用MultiScaleFeatureFusionConvBlock3d替换StackedConvBlocks3d'''
    def __init__(self, in_channels, out_channels, base_filters_num=32):
        super(UNetMSFF, self).__init__()
        self.weightInitializer=InitWeights_He(1e-2)

        if base_filters_num == 32:
            features = [in_channels, 32, 64, 128, 256, 320, 320]
        elif base_filters_num == 64:
            features = [in_channels, 64, 128, 256, 512, 512, 512]
        self.down_0 = StackedConvBlocks3d(in_channels, features[1]) # 第一层只有一个通道，不能分成4份
        self.pool_0 = MaxPool3d()
        self.down_1 = MultiScaleFeatureFusionConvBlock3d(features[1], features[2])
        self.pool_1 = MaxPool3d()
        self.down_2 = MultiScaleFeatureFusionConvBlock3d(features[2], features[3])
        self.pool_2 = MaxPool3d()
        self.down_3 = MultiScaleFeatureFusionConvBlock3d(features[3], features[4])
        self.pool_3 = MaxPool3d()
        self.down_4 = MultiScaleFeatureFusionConvBlock3d(features[4], features[5])
        self.pool_4 = MaxPool3d()

        self.bottleneck = MultiScaleFeatureFusionConvBlock3d(features[5], features[6])

        self.trans_4 = ConvTranspose3d(features[6], features[5])
        self.up_4 = MultiScaleFeatureFusionConvBlock3d(features[5]*2, features[5])
        self.trans_3 = ConvTranspose3d(features[5], features[4])
        self.up_3 = MultiScaleFeatureFusionConvBlock3d(features[4]*2, features[4])
        self.trans_2 = ConvTranspose3d(features[4], features[3])
        self.up_2 = MultiScaleFeatureFusionConvBlock3d(features[3]*2, features[3])
        self.trans_1 = ConvTranspose3d(features[3], features[2])
        self.up_1 = MultiScaleFeatureFusionConvBlock3d(features[2]*2, features[2])
        self.trans_0 = ConvTranspose3d(features[2], features[1])
        self.up_0 = MultiScaleFeatureFusionConvBlock3d(features[1]*2, features[1])

        self.seg_output_0 = nn.Conv3d(features[1], out_channels, kernel_size=1, bias=False)
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

        seg_output_0 = self.seg_output_0(up_0)

        return seg_output_0


class UNetAGC(nn.Module):
    '''唯一的区别是增加了AGC模块'''
    def __init__(self, in_channels, out_channels, base_filters_num=32):
        super(UNetAGC, self).__init__()
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
        
        self.AGC_4 = AttentionGuidedConcatenationBlock(features[6], features[5])
        self.trans_4 = ConvTranspose3d(features[6], features[5])
        self.up_4 = StackedConvBlocks3d(features[5]*2, features[5])
        self.AGC_3 = AttentionGuidedConcatenationBlock(features[5], features[4])
        self.trans_3 = ConvTranspose3d(features[5], features[4])
        self.up_3 = StackedConvBlocks3d(features[4]*2, features[4])
        self.AGC_2 = AttentionGuidedConcatenationBlock(features[4], features[3])
        self.trans_2 = ConvTranspose3d(features[4], features[3])
        self.up_2 = StackedConvBlocks3d(features[3]*2, features[3])
        self.AGC_1 = AttentionGuidedConcatenationBlock(features[3], features[2])
        self.trans_1 = ConvTranspose3d(features[3], features[2])
        self.up_1 = StackedConvBlocks3d(features[2]*2, features[2])
        self.AGC_0 = AttentionGuidedConcatenationBlock(features[2], features[1])
        self.trans_0 = ConvTranspose3d(features[2], features[1])
        self.up_0 = StackedConvBlocks3d(features[1]*2, features[1])

        self.seg_output_0 = nn.Conv3d(features[1], out_channels, kernel_size=1, bias=False)
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
        up_4 = self.up_4(torch.cat((trans_4, self.AGC_4(bottleneck, down_4)), dim=1))

        trans_3 = self.trans_3(up_4)
        up_3 = self.up_3(torch.cat((trans_3, self.AGC_3(trans_4, down_3)), dim=1))

        trans_2 = self.trans_2(up_3)
        up_2 = self.up_2(torch.cat((trans_2, self.AGC_2(trans_3, down_2)), dim=1))

        trans_1 = self.trans_1(up_2)
        up_1 = self.up_1(torch.cat((trans_1, self.AGC_1(trans_2, down_1)), dim=1))

        trans_0 = self.trans_0(up_1)
        up_0 = self.up_0(torch.cat((trans_0, self.AGC_0(trans_1, down_0)), dim=1))

        seg_output_0 = self.seg_output_0(up_0)

        return seg_output_0


class UNetAgcMsff(nn.Module):
    '''架构和UNetAGC一样，唯一的区别是将StackedConvBlocks3d替换为MSFF模块'''
    def __init__(self, in_channels, out_channels, base_filters_num=32):
        super(UNetAgcMsff, self).__init__()
        self.weightInitializer=InitWeights_He(1e-2)

        if base_filters_num == 32:
            features = [in_channels, 32, 64, 128, 256, 320, 320]
        elif base_filters_num == 64:
            features = [in_channels, 64, 128, 256, 512, 512, 512]
        
        self.down_0 = MultiScaleFeatureFusionConvBlock3d(in_channels, features[1])
        self.pool_0 = MaxPool3d()
        self.down_1 = MultiScaleFeatureFusionConvBlock3d(features[1], features[2])
        self.pool_1 = MaxPool3d()
        self.down_2 = MultiScaleFeatureFusionConvBlock3d(features[2], features[3])
        self.pool_2 = MaxPool3d()
        self.down_3 = MultiScaleFeatureFusionConvBlock3d(features[3], features[4])
        self.pool_3 = MaxPool3d()
        self.down_4 = MultiScaleFeatureFusionConvBlock3d(features[4], features[5])
        self.pool_4 = MaxPool3d()

        self.bottleneck = MultiScaleFeatureFusionConvBlock3d(features[5], features[6])
        
        self.AGC_4 = AttentionGuidedConcatenationBlock(features[6], features[5])
        self.trans_4 = ConvTranspose3d(features[6], features[5])
        self.up_4 = MultiScaleFeatureFusionConvBlock3d(features[5]*2, features[5])
        self.AGC_3 = AttentionGuidedConcatenationBlock(features[5], features[4])
        self.trans_3 = ConvTranspose3d(features[5], features[4])
        self.up_3 = MultiScaleFeatureFusionConvBlock3d(features[4]*2, features[4])
        self.AGC_2 = AttentionGuidedConcatenationBlock(features[4], features[3])
        self.trans_2 = ConvTranspose3d(features[4], features[3])
        self.up_2 = MultiScaleFeatureFusionConvBlock3d(features[3]*2, features[3])
        self.AGC_1 = AttentionGuidedConcatenationBlock(features[3], features[2])
        self.trans_1 = ConvTranspose3d(features[3], features[2])
        self.up_1 = MultiScaleFeatureFusionConvBlock3d(features[2]*2, features[2])
        self.AGC_0 = AttentionGuidedConcatenationBlock(features[2], features[1])
        self.trans_0 = ConvTranspose3d(features[2], features[1])
        self.up_0 = MultiScaleFeatureFusionConvBlock3d(features[1]*2, features[1])

        self.seg_output_0 = nn.Conv3d(features[1], out_channels, kernel_size=1, bias=False)
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
        up_4 = self.up_4(torch.cat((trans_4, self.AGC_4(bottleneck, down_4)), dim=1))

        trans_3 = self.trans_3(up_4)
        up_3 = self.up_3(torch.cat((trans_3, self.AGC_3(trans_4, down_3)), dim=1))

        trans_2 = self.trans_2(up_3)
        up_2 = self.up_2(torch.cat((trans_2, self.AGC_2(trans_3, down_2)), dim=1))

        trans_1 = self.trans_1(up_2)
        up_1 = self.up_1(torch.cat((trans_1, self.AGC_1(trans_2, down_1)), dim=1))

        trans_0 = self.trans_0(up_1)
        up_0 = self.up_0(torch.cat((trans_0, self.AGC_0(trans_1, down_0)), dim=1))

        seg_output_0 = self.seg_output_0(up_0)

        return seg_output_0


class UNetV2(nn.Module):
    def __init__(self, in_channels, out_channels, base_filters_num=32):
        super(UNetV2, self).__init__()
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

        self.seg_output_0 = nn.Conv3d(features[1], out_channels, kernel_size=1, bias=False)
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

        seg_output_0 = self.seg_output_0(up_0)

        return seg_output_0