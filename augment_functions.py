'''
写一个数据增强函数试试
增强的功能包括：
大类1
亮度：
    对比度：image, contrast_range=(0.75, 1.25), preserve_range=True, per_channel=True，
    亮度：(data_sample, mu:float, sigma:float , per_channel:bool=True, p_per_channel:float=1.)，
    gamma校正：image, gamma_range=(0.5, 2), invert_image=False, epsilon=1e-7, per_channel=False, retain_stats=False；

噪声：
    添加高斯噪声：(image, noise_variance=(variance_min, variance_max))；

大类2
空间变换（除了翻转，都要在其他增强之后实现）：
    翻转image, label, p, axis，
    旋转image, label, p, angle_x, angle_y, angle_z，对于各向异性数据某些方向的旋转可能不适用
    形变image, label, sigma=5, points=3；
    缩放

所有数据中，有增强的比率为aug_ratio，也就是有任意三类增强中的一类的概率

假设三类增强的概率都是p，那么有
1 - aug_ratio = (1 - p)^3   →   p = 1 - pow(1 - r, 1 / 3)
比如aug_ratio = 0.657 （即1-0.7^3的值），那么有(1 - p)^3 = 1 - aug_ratio = 0.7^3，可得p = 0.3

同理，假设第i大类下有n个小类，则p_i需要满足
1 - p = (1 - p_i)^n   →   p_i = 1 - pow(1 - p, 1 / n)
如果p = 0.3，那么每一种亮度增强的概率p_1为0.112


XueZhimeng 2020.8
'''
import numpy as np 
import nibabel as nib
from functools import partial
from augmentation_space import random_transform, deform_grid, flip_axis
from augmentation_illumination import augment_contrast, augment_gamma, augment_brightness_additive
from augmentation_noise import augment_gaussian_noise

def augment_flip_axis(image, label):
    # 同时依概率翻转image和label的不同维度
    if np.random.random() < 0.5:
        image = flip_axis(image, 0)
        label = flip_axis(label, 0)

    if np.random.random() < 0.5:
        image = flip_axis(image, 1)
        label = flip_axis(label, 1)
        
    if np.random.random() < 0.5:
        image = flip_axis(image, 2)
        label = flip_axis(label, 2)

    return image, label


class augmentation():
    def __init__(self, aug_params):
        # 两个函数队列
        self.aug_function_queue1 = []
        self.aug_function_queue2 = []
        # 增强函数队列1准备：对比度，亮度，gamma校正，高斯噪声
        if aug_params['do_contrast']:
            self.aug_function_queue1.append(partial(augment_contrast, contrast_range=aug_params['contrast_range'], preserve_range=True, per_channel=False))
        if aug_params['do_brightness_additive']:
            self.aug_function_queue1.append(partial(augment_brightness_additive, mu=aug_params['brightness_additive_mu'], sigma=aug_params['brightness_additive_sigma'], per_channel=False, p_per_channel=1.))
        if aug_params['do_gamma']:
            self.aug_function_queue1.append(partial(augment_gamma, gamma_range=aug_params['gamma_range'], invert_image=False, epsilon=1e-7, per_channel=False, retain_stats=False))
        if aug_params['do_gaussian_noise']:
            self.aug_function_queue1.append(partial(augment_gaussian_noise, noise_variance=aug_params['noise_variance']))
        
        # 增强函数队列2准备：翻转，旋转，形变，
        # 缩放和旋转需要做到同一个函数里（用同一个转换矩阵），目前还不能单独
        if(aug_params['do_flip']):
            self.aug_function_queue2.append(partial(augment_flip_axis))
        if(aug_params['do_rotate_and_zoom']):
            self.aug_function_queue2.append(partial(random_transform, 
                                        rotation_range_alpha=aug_params['rotate_angle'][0],
                                        rotation_range_beta = aug_params['rotate_angle'][1],
                                        rotation_range_gamma = aug_params['rotate_angle'][2],
                                        zoom_range = aug_params['zoom_range']))
        if(aug_params['do_deform_grid']):
            self.aug_function_queue2.append(partial(deform_grid, sigma=aug_params['deform_grid_sigma'], points=aug_params['deform_grid_points']))

        # 两大类增强函数执行的概率
        aug_ratio = aug_params['aug_ratio']
        p = 1 - np.power(1 - aug_ratio, 1 / 2)
        self.p1 = 1 - np.power(1 - p, 1 / len(self.aug_function_queue1))
        self.p2 = 1 - np.power(1 - p, 1 / len(self.aug_function_queue2))
        pass

    def do_augment(self, image, label):
        for aug_function in self.aug_function_queue1:
            if np.random.uniform(0,1) < self.p1:
                image = aug_function(image)
        for aug_function in self.aug_function_queue2:
            if np.random.uniform(0,1) < self.p2:
                image, label = aug_function(image, label)
        return image, label

if __name__ == '__main__':
    aug_params = {
        'aug_ratio': 0.64,
        
        'do_contrast': True,
        'contrast_range': (0.75, 1.25),

        'do_brightness_additive': True,
        'brightness_additive_mu': 0.0,
        'brightness_additive_sigma': 0.05,

        'do_gamma': True,
        'gamma_range': (0.5, 2),

        'do_gaussian_noise': True,
        'noise_variance': (0, 0.05),

        'do_flip':True,
        # 'flip_axis': [True, True, True],

        'do_rotate_and_zoom': True,
        'rotate_angle': [0, 0, 10], # 不是很确定是不是该旋转这个轴……
        'zoom_range':[1, 1.25],

        'do_deform_grid': True,
        'deform_grid_sigma': 5,
        'deform_grid_points': 3,
    }
    trans = augmentation(aug_params)

    image_org_path = 'test_aug/case1_patient.nii.gz'
    label_org_path = 'test_aug/case1_port.nii.gz'
    image_aug_path_base = 'test_aug/case1_patient_aug'
    label_aug_path_base = 'test_aug/case1_portalvein_aug'

    image = nib.load(image_org_path)
    image_affine = image.get_affine()
    image = image.get_data()
    label = nib.load(label_org_path).get_data()

    for i in range(0,10):
        image_aug_path = image_aug_path_base + str(i) + '.nii.gz'
        label_aug_path = label_aug_path_base + str(i) + '.nii.gz'

        Imax = np.max(image)
        Imin = np.min(image)
        image_aug = image.astype('float64')
        image_aug = (image_aug - Imin) / (Imax - Imin)
        image_aug, label_aug = trans.do_augment(image_aug, label)
        image_aug = image_aug * (Imax - Imin) + Imin
        image_aug = image_aug.astype('i2')
        nib.Nifti1Image(image_aug,image_affine).to_filename(image_aug_path)
        nib.Nifti1Image(label_aug,image_affine).to_filename(label_aug_path)
        pass