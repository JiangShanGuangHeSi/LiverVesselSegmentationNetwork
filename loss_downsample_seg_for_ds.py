import torch
from torch.nn.functional import avg_pool2d, avg_pool3d
import numpy as np

# def downsample_seg_for_ds_transform3(seg, ds_scales=((1, 1, 1), (0.5, 0.5, 0.5), (0.25, 0.25, 0.25)), classes=None):
def downsample_seg_for_ds_transform3(seg, ds_scales=((1, 1, 1), (0.5, 0.5, 0.5), (0.25, 0.25, 0.25))):
    '''
    用于深度学习的label降采样（改进自nnunet\training\data_augmentation\downsampling.py同名函数）
    输入大小为(B, C, D, H, W)的torch（原本是三维np.array）
    返回不同尺度的label列表
    '''
    output = []
    # one_hot = torch.from_numpy(convert_seg_image_to_one_hot_encoding_batched(seg, classes)) # b, c,

    for s in ds_scales:
        if all([i == 1 for i in s]):
            # output.append(torch.from_numpy(seg))
            output.append(seg)
        else:
            kernel_size = tuple(int(1 / i) for i in s)
            stride = kernel_size
            pad = tuple((i-1) // 2 for i in kernel_size)

            # if len(s) == 2:
            #     pool_op = avg_pool2d
            # elif len(s) == 3:
            #     pool_op = avg_pool3d
            # else:
            #     raise RuntimeError()

            # pooled = pool_op(one_hot, kernel_size, stride, pad, count_include_pad=False, ceil_mode=False)
            pooled = avg_pool3d(seg, kernel_size, stride, pad, count_include_pad=False, ceil_mode=False)

            output.append(pooled)
    return output