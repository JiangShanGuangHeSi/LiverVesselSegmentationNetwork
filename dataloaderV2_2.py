'''
用于liver vessel CT数据
运行该程序之前，请成功运行preprocess产生数据集

训练过程完全随机采样，因此对于训练集，不需要提前准备batch
XueZhimeng, 2020.05

因为数据集中有血管1和肿瘤2两种标签，所以仅保留血管标签
各向异性数据，不旋转
XueZhimeng, 2020.7

用于ircadb数据，两种血管标签合并
增加了新的增强方法
XueZhimeng, 2020.8

原来test和val返回的是用clipdata函数切的块，现在改用nnunet的_compute_steps_for_sliding_window函数
XueZhimeng, 2020.8

添加了mask判断，如果存在mask则讲mask应用在image上
XueZhimeng, 2021.4
'''
import os
import struct
import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch import autograd, optim
import torch.utils.data as data
from torch.utils.data import DataLoader
from PIL import Image
from PIL import ImageDraw, ImageFont
# from torchvision.transforms import transforms
import random

import augmentation_space as ia
from augment_functions import augmentation

import matplotlib.image as mpimg # mpimg 用于读写图片

from typing import Union, Tuple, List

# TRAINSETINFO_FILE_PATH = "dataset/trainset_info.npy" # 数据集信息
# TESTSETINFO_FILE_PATH = "dataset/testset_info.npy"
# TRAINSETINFO_FILE_PATH = "dataset/trainset_info_spacing.npy" # 数据集信息
# TESTSETINFO_FILE_PATH = "dataset/testset_info_spacing.npy"

# 【对不同的数据集，记得要改！】
# TRAINSETINFO_FILE_PATH = "dataset/trainset_info_ircadb_spacing.npy" # 3dircadb数据集信息
# TESTSETINFO_FILE_PATH = "dataset/testset_info_ircadb_spacing.npy" 
# TRAINSETINFO_FILE_PATH = "dataset/trainset_info_seu_spacing.npy" # seu中大医院数据集信息
# TESTSETINFO_FILE_PATH = "dataset/testset_info_seu_spacing.npy" 
# TRAINSETINFO_FILE_PATH = "dataset/trainset_info_szrch_spacing.npy" # 深圳第二人民医院数据集信息
# TESTSETINFO_FILE_PATH = "dataset/testset_info_szrch_spacing.npy" 
TRAINSETINFO_FILE_PATH = "dataset/trainset_info_szrch_spacing.npy" # 我自己标的ZDYY和ircadb数据
TESTSETINFO_FILE_PATH = "dataset/testset_info_szrch_spacing.npy" 

DEBUG_ROOT = "pic_debug/"

# BATCH_SIZE = [96, 96, 96]
default_aug_params = {
    'aug_ratio': 0.64,
    
    'do_contrast': False,
    'contrast_range': (0.75, 1.25),

    'do_brightness_additive': False,
    'brightness_additive_mu': 0.0,
    'brightness_additive_sigma': 0.05,

    'do_gamma': False,
    'gamma_range': (0.5, 2),

    'do_gaussian_noise': False,
    'noise_variance': (0, 0.05),

    'do_flip':True,
    # 'flip_axis': [True, True, True],

    'do_rotate_and_zoom': False,
    'rotate_angle': [0, 0, 10], # 不是很确定是不是该旋转这个轴……
    'zoom_range':[1, 1.25],

    'do_deform_grid': False,
    'deform_grid_sigma': 5,
    'deform_grid_points': 3,
}

class DatasetV2_2(data.Dataset):         # 这个是用来生成数据集的
    def __init__(self, patch_size, dataset_category='train', channel=2, data_len=512, aug_params=default_aug_params): 
        self.data_size = patch_size
        self.dataset_category = dataset_category
        self.channel = channel
        self.isinterpolation = True

        # 2021-04-30补充：肝脏窗宽窗位
        self.WINDOW_LEVEL = 40.0
        self.WINDOW_WIDTH = 350.0

        # 加载数据集信息
        if dataset_category == 'train': # 训练集准备
            self.dataset_info = np.load(TRAINSETINFO_FILE_PATH, allow_pickle=True)
            self.dataset_info = self.dataset_info.tolist()    #将array转换为列表
            
            self.case_num = len(self.dataset_info)
            self.data_len = data_len # 全部随机，多少都无所谓了

            # 数据增强信息
            self.data_augment = augmentation(aug_params)

        elif dataset_category == 'test' or 'val': # 测试集/验证集准备
            self.dataset_info = np.load(TESTSETINFO_FILE_PATH, allow_pickle=True)
            # self.dataset_info = np.load(TRAINSETINFO_FILE_PATH, allow_pickle=True) # 测试，一定要注释掉！
            self.dataset_info = self.dataset_info.tolist()  
            if dataset_category == 'test':
                # self.dataset_info = self.dataset_info[0:8] # 一次装不下那么多1-8
                # self.dataset_info = self.dataset_info[8:15] # 一次装不下那么多9-15
                # self.dataset_info = [self.dataset_info[0]] # 测试
                # self.dataset_info = [self.dataset_info[30]] # 第30例数据有问题
                pass
            elif dataset_category == 'val':
                # self.dataset_info = self.dataset_info[::3] # 随便抽几个做验证，这里是每3个取一个
                # self.dataset_info = [self.dataset_info[0] ]
                pass

            self.testset_batches = []
            self.case_num = len(self.dataset_info)
            self.data_len = 0
            self.case_idxes = []
            case_idx = 0
            for case in self.dataset_info:
                if self.isinterpolation:
                    image = np.load(case['npy_image_interpolation_path'])
                    try:
                        label = np.load(case['npy_label_interpolation_path'])
                    except FileNotFoundError:
                        print('No such file:', case['npy_label_interpolation_path'], ', label will be set zeros.')
                        label = np.zeros(np.shape(image), np.float32)
                    box = np.squeeze(case['box_interpolation'])
                else:
                    image = np.load(case['npy_image_path'])
                    try:
                        label = np.load(case['npy_label_path'])
                    except FileNotFoundError:
                        print('No such file:', case['npy_label_path'], ', label will be set zeros.')
                        label = np.zeros(np.shape(image), np.float32)
                    box = np.squeeze(case['box'])

                box_shape = [box[3]-box[0], box[4]-box[1], box[5]-box[2]]
                cube_list, fold, overlap, weight = clipdata(box_shape, self.data_size, 1)
                steps, cube_list = _compute_steps_for_sliding_window(self.data_size, box_shape, 0.5)

                case['image'] = image
                case['label'] = label
                case['cube_list'] = cube_list
                case['weight'] = weight
                case['box_shape'] = box_shape
                case['batch_num'] = len(cube_list)
                case['steps'] = steps

                cube_list = np.array(cube_list)
                box_x = np.tile(box[0], (len(cube_list), 2))
                box_y = np.tile(box[1], (len(cube_list), 2))
                box_z = np.tile(box[2], (len(cube_list), 2))
                box_shape = np.concatenate((box_x, box_y, box_z), axis=1)
                cube_list = cube_list + box_shape
                cube_list = cube_list.tolist()

                for cube in cube_list:
                    batch = {}
                    batch['cube'] = cube
                    batch['case_idx'] = case_idx
                    self.testset_batches.append(batch)

                self.data_len += len(cube_list) # 计算预测数量
                self.case_idxes.append(case_idx)
                case_idx += 1 # 预测的图像序号

            pass
        

    def __getitem__(self, index):
        if self.dataset_category == 'train': # 训练集
            # 随机选择一个数据集
            data_idx = random.randint(0, self.case_num-1)
            case = self.dataset_info[data_idx]
            # 随机选择大脑mask里的一个位置
            if self.isinterpolation:
                box = np.squeeze(case['box_interpolation'])
            else:
                box = np.squeeze(case['box'])
                # 老子才不会说因为这个判断忘写了，白训了几个模型
            box_shape = [box[3]-box[0], box[4]-box[1], box[5]-box[2]]
            sample_range = np.array(box_shape) - self.data_size  #因为shape是list类型，不能直接减数字，必须转换为array
            #产生随机数，随机选择肝脏范围内的区域
            if sample_range[0] > 0:
                x = random.randint(0, sample_range[0]) + box[0]
            else:
                x = random.randint(sample_range[0], 0) + box[0]
            if sample_range[1] > 0:
                y = random.randint(0, sample_range[1]) + box[1]
            else:
                y = random.randint(sample_range[1], 0) + box[1]
            if sample_range[2] > 0:
                z = random.randint(0, sample_range[2]) + box[2]
            else:
                z = random.randint(sample_range[2], 0) + box[2]
            
            # 这引发了一个问题，就是数据坐标范围
            # data[x : x + self.data_size[0], y : y + self.data_size[1], z : z + self.data_size[2]]
            # 很可能超出实际的数据大小，因此取出来的数据比我们想要的小
            # 解决方法是用np.zeros初始化为data_size大小，然后将取出来的数据，无论大小，放在这个空间中间

            # 取出3d图像image
            if self.isinterpolation:                
                data = np.load(case['npy_image_interpolation_path'])
            else:    
                data = np.load(case['npy_image_path'])
            # data = data[x : x + self.data_size[0], y : y + self.data_size[1], z : z + self.data_size[2]]
            # 计算能够取到的数据范围上下限
            x_l = max(0, x)
            x_u = min(data.shape[0], x + self.data_size[0])
            y_l = max(0, y)
            y_u = min(data.shape[1], y + self.data_size[1])
            z_l = max(0, z)
            z_u = min(data.shape[2], z + self.data_size[2])
            # 初始化图像空间，保证返回的数据大小为data_size
            image = np.zeros(self.data_size, dtype=np.float32)
            # data = data[x_l:x_u, y_l:y_u, z_l:z_u]
            image[x_l-x:x_u-x, y_l-y:y_u-y, z_l-z:z_u-z] = data[x_l:x_u, y_l:y_u, z_l:z_u]
            image = (image - self.WINDOW_LEVEL) / self.WINDOW_WIDTH # 【2021.04.30 归一化修改】
            image = image.astype(np.float32)
            # 取出分类的label
            if self.isinterpolation:                
                data_label = np.load(case['npy_label_interpolation_path'])
            else:    
                data_label = np.load(case['npy_label_path'])
            # label = label[x : x + self.data_size[0], y : y + self.data_size[1], z : z + self.data_size[2]]
            # label = label.astype(np.float32)
            label = np.zeros(self.data_size)
            label[x_l-x:x_u-x, y_l-y:y_u-y, z_l-z:z_u-z] = data_label[x_l:x_u, y_l:y_u, z_l:z_u]
            
            # 【改动】取出分类的mask并且加在image里【注意test也一样改】
            if 'npy_mask_interpolation_path' in case:
                if self.isinterpolation:                
                    data_mask = np.load(case['npy_mask_interpolation_path'])
                else:    
                    data_mask = np.load(case['npy_mask_path'])
                mask = np.zeros(self.data_size)
                mask[x_l-x:x_u-x, y_l-y:y_u-y, z_l-z:z_u-z] = data_mask[x_l:x_u, y_l:y_u, z_l:z_u]
                mask[mask != 0] = 1 # 不是背景的都是mask
                # 图像取mask中的
                image = image * mask
            # 【改动结束】

            # # 存图用于检验
            # mpimg.imsave(DEBUG_ROOT+'%dimage.png'%index,np.max(image,axis=2))
            # mpimg.imsave(DEBUG_ROOT+'%dlabel.png'%index,np.max(label,axis=2))

            # 由于RuntimeWarning: invalid value encountered in true_divide image = image / np.max(image)
            # 【2021.04.30 该操作修改为(image - self.WINDOW_LEVEL) / self.WINDOW_WIDTH】
            # # 归一化 
            # image = image / np.max(image)

            # 数据增强
            image, label = self.data_augment.do_augment(image, label)
            # dataloader报错：Expected object of scalar type Float but got scalar type Double for sequence element 1 in sequence argument at position #1 'tensors'
            # 应该是增强过程中image被转换为double的缘故
            
            image = np.ascontiguousarray(image) # 为了解决不连续错误
            label = np.ascontiguousarray(label)
            image = torch.from_numpy(image)   # 从numpy数组转成tensor
            label = torch.from_numpy(label)
            image = image.float() # 解决上面错误

            # 图像增加维度
            image = image.unsqueeze(0)
            # label的one-hot编码
            if self.channel != 1:
                label = label > 0.5
                # mmp这个函数的编码有问题，只有一层怎么回事
                label = torch.nn.functional.one_hot(label.long(), num_classes=self.channel) # 用这个函数编码，新增的维度在最后
                label = label.permute(3,0,1,2) # 交换维度，把通道换到前面
                label = label.float()
                # 存图用于检验
                # pic0 = np.min(np.squeeze(label[0,:,:,:].numpy()),axis=2) #取最小值，不然啥都看不到！
                # pic1 = np.max(np.squeeze(label[1,:,:,:].numpy()),axis=2)
                # mpimg.imsave(DEBUG_ROOT+'%dlabel_onehot0.png'%index, pic0)
                # mpimg.imsave(DEBUG_ROOT+'%dlabel_onehot1.png'%index, pic1)
            else:
                # 如果是单通道，还需要额外增加通道↓
                label = label.unsqueeze(0)

            return image, label

        elif self.dataset_category == 'test' or 'val': # 测试集
            batch_info = self.testset_batches[index]
            case = self.dataset_info[batch_info['case_idx']]
            cube = batch_info['cube']

            data = case['image']
            # 计算能够取到的数据范围上下限
            x = cube[0]
            y = cube[2]
            z = cube[4]
            x_l = max(0, x)
            x_u = min(data.shape[0], x + self.data_size[0])
            y_l = max(0, y)
            y_u = min(data.shape[1], y + self.data_size[1])
            z_l = max(0, z)
            z_u = min(data.shape[2], z + self.data_size[2])
            # image = data[cube[0]:cube[1], cube[2]:cube[3], cube[4]:cube[5]]
            image = np.zeros(self.data_size, dtype=np.float32)
            image[x_l-x:x_u-x, y_l-y:y_u-y, z_l-z:z_u-z] = data[x_l:x_u, y_l:y_u, z_l:z_u]
            image = (image - self.WINDOW_LEVEL) / self.WINDOW_WIDTH # 【2021.04.30 归一化修改】
            image = image.astype(np.float32)

            data_label = case['label']
            # label = data[cube[0]:cube[1], cube[2]:cube[3], cube[4]:cube[5]]
            label = np.zeros(self.data_size)
            label[x_l-x:x_u-x, y_l-y:y_u-y, z_l-z:z_u-z] = data_label[x_l:x_u, y_l:y_u, z_l:z_u]

            # label[label != 1] = 0 # 只保留血管(1)标签，其他标签置零！！！
            label[label != 0] = 1 # 不是背景的都是血管
            label = label.astype(np.float32)
            
            # 【改动】取出分类的mask并且加在image里【注意test也一样改】
            if 'npy_mask_interpolation_path' in case:
                if self.isinterpolation:                
                    data_mask = np.load(case['npy_mask_interpolation_path'])
                else:    
                    data_mask = np.load(case['npy_mask_path'])
                mask = np.zeros(self.data_size)
                mask[x_l-x:x_u-x, y_l-y:y_u-y, z_l-z:z_u-z] = data_mask[x_l:x_u, y_l:y_u, z_l:z_u]
                mask[mask != 0] = 1 # 不是背景的都是mask
                # 图像取mask中的
                image = image * mask
            # 【改动结束】

            # 存图用于检验
            # mpimg.imsave(DEBUG_ROOT+'test_%dimage.png'%index,np.max(image,axis=2))
            # mpimg.imsave(DEBUG_ROOT+'test_%dlabel.png'%index,np.max(label,axis=2))
            
            # 由于RuntimeWarning: invalid value encountered in true_divide image = image / np.max(image)
            # 【2021.04.30 该操作修改为(image - self.WINDOW_LEVEL) / self.WINDOW_WIDTH】
            # # 归一化
            # image = image / np.max(image)

            image = np.ascontiguousarray(image) # 为了解决上面那个错误
            label = np.ascontiguousarray(label)

            image = torch.from_numpy(image)   # 从numpy数组转成tensor
            label = torch.from_numpy(label)
            image = image.float()

            if self.dataset_category == 'test':
                image = image.unsqueeze(0)
                label = label.unsqueeze(0)
                return image, label, batch_info
            elif self.dataset_category == 'val':
                # 图像增加维度
                image = image.unsqueeze(0)
                # label的one-hot编码
                if self.channel != 1:
                    label = label > 0.5
                    # mmp这个函数的编码有问题，只有一层怎么回事
                    label = torch.nn.functional.one_hot(label.long(), num_classes=self.channel) # 用这个函数编码，新增的维度在最后
                    label = label.permute(3,0,1,2) # 交换维度，把通道换到前面
                    label = label.float()
                    # 存图用于检验
                    pic0 = np.min(np.squeeze(label[0,:,:,:].numpy()),axis=2) #取最小值，不然啥都看不到！
                    pic1 = np.max(np.squeeze(label[1,:,:,:].numpy()),axis=2)
                    # mpimg.imsave(DEBUG_ROOT+'%dlabel_onehot0.png'%index, pic0)
                    # mpimg.imsave(DEBUG_ROOT+'%dlabel_onehot1.png'%index, pic1)
                else:
                    # 如果是单通道，还需要额外增加通道↓
                    label = label.unsqueeze(0)
                return image, label

    def __len__(self):
        return self.data_len

def rotate3d(image, label, p, angle_x, angle_y, angle_z):
    prop = np.random.uniform(0,1)    # 按照贪心策略来选择策略
    if prop < p:
        image_rot, label_rot = ia.random_transform(image, label, angle_x, angle_y, angle_z)
    
    else:
        image_rot = image
        label_rot = label
    
    return image_rot, label_rot

def flip_axis(image, label, p, axis):
    prop = np.random.uniform(0,1)    # 按照贪心策略来选择策略
    if prop < p:
        image_filp = np.asarray(image).swapaxes(axis, 0)
        image_filp = image_filp[::-1, ...]
        image_filp = image_filp.swapaxes(0, axis)
        
        label_filp = np.asarray(label).swapaxes(axis, 0)
        label_filp = label_filp[::-1, ...]
        label_filp = label_filp.swapaxes(0, axis)
    else:
        image_filp = image
        label_filp = label

    return image_filp, label_filp

def getfold(vol_dim,cube_size,ita):
    """#计算每一维度小体素块的数量。"""
    dim = np.asarray(vol_dim)
    # cube number and overlap along 3 dimensions
    fold = dim / cube_size + ita
    ovlap = np.ceil(np.true_divide((fold * cube_size - dim), (fold - 1)))
    ovlap = ovlap.astype('int')

    fold = np.ceil(np.true_divide((dim + (fold - 1)*ovlap), cube_size))
    fold = fold.astype('int')

    return fold, ovlap

def clipdata(vol_dim, cube_size, ita, filename=None):
    """##根据块的大小，和切割块大小，按照batchsize切块。###"""
    cube_list = []
    # get parameters for decompose
    # fold, ovlap = getfold(vol_data.shape, cube_size, ita)
    # dim = np.asarray(vol_data.shape)
    fold, ovlap = getfold(vol_dim, cube_size, ita)
    dim = np.asarray(vol_dim)
    cube_repetition = np.zeros(vol_dim, np.int8) #记录重复次数
    # decompose
    for R in range(0, fold[0]):
        r_s = R * cube_size[0] - R * ovlap[0]
        r_e = r_s + cube_size[0]
        if r_e >= dim[0]:
            r_s = dim[0] - cube_size[0]
            r_e = r_s + cube_size[0]
        for C in range(0, fold[1]):
            c_s = C * cube_size[1] - C * ovlap[1]
            c_e = c_s + cube_size[1]
            if c_e >= dim[1]:
                c_s = dim[1] - cube_size[1]
                c_e = c_s + cube_size[1]
            for H in range(0, fold[2]):
                h_s = H * cube_size[2] - H * ovlap[2]
                h_e = h_s + cube_size[2]
                if h_e >= dim[2]:
                    h_s = dim[2] - cube_size[2]
                    h_e = h_s + cube_size[2]
                # partition multiple channels
                # cube_temp = vol_data[r_s:r_e, c_s:c_e, h_s:h_e]
                # cube_batch = np.zeros([batch_size, cube_size, cube_size, cube_size, n_chn]).astype('float32')
                # cube_batch[0, :, :, :, 0] = copy.deepcopy(cube_temp)
                cube_idx = [r_s, r_e, c_s, c_e, h_s, h_e]
                cube_repetition[r_s:r_e, c_s:c_e, h_s:h_e] = cube_repetition[r_s:r_e, c_s:c_e, h_s:h_e] + 1
                # save
                # cube_list.append(cube_batch)
                if r_e < 0 or c_e < 0 or h_e < 0:
                    print('WARNING: cube is out of the box!') # 我也不知道为啥，但确实造成了问题！
                else:
                    cube_list.append(cube_idx)
    if filename != None:
        np.save(filename, cube_repetition)

    return cube_list, fold, ovlap, cube_repetition

def _compute_steps_for_sliding_window(patch_size: Tuple[int, ...], image_size: Tuple[int, ...], step_size: float):
    '''
    根据块的大小patch_size，图像大小image_size和块之间的间隔（按比例）step_size切块
    抄自nnunet.network_architecture.neural_network
    '''
    assert [i >= j for i, j in zip(image_size, patch_size)], "image size must be as large or larger than patch_size"
    assert 0 < step_size <= 1, 'step_size must be larger than 0 and smaller or equal to 1'

    # our step width is patch_size*step_size at most, but can be narrower. For example if we have image size of
    # 110, patch size of 32 and step_size of 0.5, then we want to make 4 steps starting at coordinate 0, 27, 55, 78
    target_step_sizes_in_voxels = [i * step_size for i in patch_size]

    num_steps = [int(np.ceil((i - k) / j)) + 1 for i, j, k in zip(image_size, target_step_sizes_in_voxels, patch_size)]

    steps = []
    for dim in range(len(patch_size)):
        # the highest step value for this dimension is
        max_step_value = image_size[dim] - patch_size[dim]
        if num_steps[dim] > 1:
            actual_step_size = max_step_value / (num_steps[dim] - 1)
        else:
            actual_step_size = 99999999999  # does not matter because there is only one step at 0

        steps_here = [int(np.round(actual_step_size * i)) for i in range(num_steps[dim])]

        steps.append(steps_here)
    
    cube_list = []
    for x in steps[0]:
        for y in steps[1]:
            for z in steps[2]:
                cube_list.append([x, x+patch_size[0], y, y+patch_size[1], z, z+patch_size[2]])
    return steps, cube_list


if __name__ == '__main__':
    epoch_num = 1
    batch_size = 2

    train_dataset = CTDataset(True)    # 这是训练集
    # train_dataset = CTDataset(False)    # 这是测试集
    print('训练集长度:', len(train_dataset))
    train_dataloaders = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    for epoch in range(epoch_num):
        for batch_image, batch_label in train_dataloaders:
            # 取出的是4维数组[batchsize, x, y, z]（xyz只是表示，是不是xyz我不知道）
            # image = batch_image[0,:].numpy()
            # label = batch_label[0,:].numpy()
            # image = image[:, np.newaxis, :, :, :]
            # label = label[np.newaxis, np.newaxis, :, :, :] # 加一个维度channel，鬼知道为啥取出来少了俩维度
            # image_rot, lable_rot = st.augment_spatial(image, label, [96, 96, 96])

            image = batch_image[0,:]
            label = batch_label[0,:]

            image = np.squeeze(image.numpy())
            label = np.squeeze(label.numpy())

            # vis.show_3d_array(image)
            # vis.show_3d_array(image_filp)
            # # vis.show_3d_array(image_rot)
            # # vis.show_3d_array2(label, label_rot)
            # vis.show_3d_array2(label, label_filp)
            pass