'''
根据.csv文件，读取数据，得到roi区域，插值保存为npy文件，
并生成包含roi和插值信息的npy文件
DATASET_CSV_PATH = "data_path_info_mix.csv" # szrch数据集
XueZhimeng, 2021.4（可能吧）

添加了几例
DATASET_CSV_PATH = "data_path_info_mix20210607.csv" # szrch数据集

添加变量B_INTERP，
B_INTERP = TRUE 任何情况下都进行插值
B_INTERP = FALSE 检查之前是否有生成插值的结果，如果有就跳过，在添加数据集而不改变插值spacing的时候用
'''

import os
import re
import numpy as np
import pandas as pd
import nibabel as nib # 读取nii文件
from skimage.transform import resize # 插值
import random # 随机采样
import matplotlib.image as mpimg # mpimg 用于读取图片
import utils_file

def getside(binary_img):
    """获得label在体数据中占据的立方体区域
    返回值：（矢向坐标1，冠向坐标1，轴向坐标1，矢向坐标2，冠向坐标2，轴向坐标2）"""
    boxes=np.zeros([1, 6])
    slices_indicies = np.where(np.any(binary_img, axis=1))[0]
    horizontal_indicies = np.sort(np.where(np.any(binary_img, axis=0))[1])# 注意这边选出来的只是个对应坐标，x，y不能同时取大或者取小。
    vertical_indicies = np.where(np.any(binary_img, axis=0))[0]

    if horizontal_indicies.shape[0]:
        z1, z2 = slices_indicies[[0, -1]]
        x1, x2 = horizontal_indicies[[0, -1]]
        y1, y2 = vertical_indicies[[0, -1]]
    else:
        # No mask for this instance. Might happen due to
        # resizing or cropping. Set bbox to zeros
        x1, x2, y1, y2 ,z1, z2= 0, 0, 0, 0, 0, 0
    boxes[0] = np.array([z1, y1, x1, z2, y2, x2])
    return boxes.astype(np.int32)

# DATASET_CSV_PATH = "data_path_info_mix.csv" # szrch数据集
DATASET_CSV_PATH = "data_path_info_mix20210607.csv" # szrch数据集，但是训练集和测试集搞反了
DATASET_CSV_PATH = "data_path_info_mix20210610.csv" # 
STATIC_INFO_XLSX_PATH = "static_box_and_spacing.xlsx"

FOLDER_IMAGE_NPY = "dataset/image/"
FOLDER_MASK_NPY = "dataset/mask/"
FOLDER_LABEL_NPY = "dataset/label/"
FOLDER_IMAGE_INTERPOLATION_NPY = "dataset/image_interpolation/"
FOLDER_MASK_INTERPOLATION_NPY = "dataset/mask_interpolation/"
FOLDER_LABEL_INTERPOLATION_NPY = "dataset/label_interpolation/"

TRAINSETINFO_FILE_PATH = "dataset/trainset_info_szrch_spacing.npy" # 数据集信息
TESTSETINFO_FILE_PATH = "dataset/testset_info_szrch_spacing.npy"

# DEBUG_ROOT = "debug/preprocess/"
DEBUG_ROOT = "pic_debug/preprocess/"

def GetMedianSpacing(path_csv_dataset, path_xlsx_spacing_and_box):
    '''
    读取csv文件中的图像文件，统计它们的中值spacing，得到推荐spacing
    '''
    # 读取训练数据
    csv_data = pd.read_csv(path_csv_dataset)
    list_name_case = csv_data['name_case'].values  
    list_raw_image_path = csv_data['raw_image_path'].values
    list_raw_mask_path = csv_data['raw_mask_path'].values
    list_raw_label_path = csv_data['raw_label_path'].values
    list_train_or_test = csv_data['train_or_test'].values

    # 统计信息表准备
    shape_x = []
    shape_y = []
    shape_z = []
    box_x_s = []
    box_y_s = []
    box_z_s = []
    box_x_e = []
    box_y_e = []
    box_z_e = []
    spacing_x = []
    spacing_y = []
    spacing_z = []
    writer = pd.ExcelWriter(path_xlsx_spacing_and_box)

    list_spacing = []
    for row in range(csv_data.shape[0]):# csv_data.shape[0]
        case = {}
        name_case = list_name_case[row]
        path_image = list_raw_image_path[row]
        path_mask = list_raw_mask_path[row]
        path_label = list_raw_label_path[row]

        # 读取图片，获得spacing信息
        image = nib.load(path_image)
        affine = image.get_affine()
        image = image.get_data()
        mask = nib.load(path_mask).get_data()
        label = nib.load(path_label).get_data()
        # 转换spacing信息
        spacing = np.abs(np.array([affine[0,0], affine[1,1], affine[2,2]]))
        list_spacing.append(spacing)
        
        # 得到box
        box = np.squeeze(getside(mask))
        print('processing', name_case, '\tspacing', spacing, '\tbox', box)
        
        # 原始图像的spacing和box统计
        shape_x.append(np.shape(image)[0])
        shape_y.append(np.shape(image)[1])
        shape_z.append(np.shape(image)[2])
        box_x_s.append(box[0])
        box_y_s.append(box[1])
        box_z_s.append(box[2])
        box_x_e.append(box[3])
        box_y_e.append(box[4])
        box_z_e.append(box[5])
        spacing_x.append(spacing[0])
        spacing_y.append(spacing[1])
        spacing_z.append(spacing[2])
    
    # 保存统计excel文件
    df = pd.DataFrame()
    df['case'] = list_name_case
    df['shape x'] = shape_x
    df['shape y'] = shape_y
    df['shape z'] = shape_z
    df['box x s'] = box_x_s
    df['box x e'] = box_x_e
    df['box y s'] = box_y_s
    df['box y e'] = box_y_e
    df['box z s'] = box_z_s
    df['box z e'] = box_z_e
    df['spacing x'] = spacing_x
    df['spacing y'] = spacing_y
    df['spacing z'] = spacing_z
    
    df.to_excel(writer, index=False)
    writer.save()
    
    # 计算中值spacing
    spacing_median = np.median(np.array(list_spacing),0)
    return spacing_median

def DoBoxAndSpacing(path_csv_dataset, new_spacing, b_interp=True):
    # 读取训练数据
    csv_data = pd.read_csv(path_csv_dataset)
    list_name_case = csv_data['name_case'].values  
    list_raw_image_path = csv_data['raw_image_path'].values
    list_raw_mask_path = csv_data['raw_mask_path'].values
    list_raw_label_path = csv_data['raw_label_path'].values
    list_train_or_test = csv_data['train_or_test'].values
    
    TrainCaseList = []
    TestCaseList = []
    for row in range(csv_data.shape[0]):# csv_data.shape[0]
        case = {}
        # 读信息
        name_case = list_name_case[row]
        path_image = list_raw_image_path[row]
        path_mask = list_raw_mask_path[row]
        path_label = list_raw_label_path[row]
        # 准备存储路径
        case['npy_image_path'] = FOLDER_IMAGE_NPY + name_case + '.npy'
        case['npy_label_path'] = FOLDER_LABEL_NPY + name_case + '.npy'
        case['npy_mask_path'] = FOLDER_MASK_NPY + name_case + '.npy'
        # 插值图像保存路径
        case['npy_image_interpolation_path'] = FOLDER_IMAGE_INTERPOLATION_NPY + name_case + '.npy'
        case['npy_mask_interpolation_path'] = FOLDER_MASK_INTERPOLATION_NPY + name_case + '.npy'
        case['npy_label_interpolation_path'] = FOLDER_LABEL_INTERPOLATION_NPY + name_case + '.npy'
        
        b_files_exist = False
        if not b_interp:
            # 检查这些文件是否存在
            if (os.path.exists(case['npy_image_path']) \
                and os.path.exists(case['npy_label_path']) \
                and os.path.exists(case['npy_mask_path']) \
                and os.path.exists(case['npy_image_interpolation_path']) \
                and os.path.exists(case['npy_image_interpolation_path']) \
                and os.path.exists(case['npy_image_interpolation_path'])):
                b_files_exist = True
        
        # <debug> 调试
        if name_case == 'Ircadb_002':
            pass
        # <debug/>
        
        # 读取image, mask, label，获得spacing信息
        image = nib.load(path_image)
        affine = image.get_affine()
        image = image.get_data()
        mask = nib.load(path_mask).get_data()
        label = nib.load(path_label).get_data()
        # 转换spacing信息
        spacing = np.abs(np.array([affine[0,0], affine[1,1], affine[2,2]]))
        
        # 得到box
        box = np.squeeze(getside(mask))
        case['box'] = box # 用来训练的区域

        # 插值处理，计算放缩比例和新形状
        scale_rate = spacing / np.array(new_spacing) # 缩放比例
        new_shape = np.round(np.shape(image) * scale_rate).astype(np.int)
        # 插值，转换格式
        if not b_files_exist:
            image_resize = resize(image.astype(np.float32), new_shape, order=3) # 3阶
            image_resize = image_resize.astype(np.float32)
            label_resize = resize(label.astype(np.float32), new_shape, order=0) # 0阶
            label_resize = label_resize.astype(np.uint8)
        mask_resize = resize(mask.astype(np.float32), new_shape, order=0) # 0阶
        mask_resize = mask_resize.astype(np.uint8)
        # 计算新box
        new_box = getside(mask_resize) 
        # new_box = np.round(box * np.tile(scale_rate, 2)).astype(np.int)
        case['box_interpolation'] = new_box
        print('\n new shape:', new_shape, '\n new box', new_box)

        if not b_files_exist:
            # debug
            mpimg.imsave(DEBUG_ROOT+'%dimage_resize_x.png'%(row%10), np.max(image_resize, axis=0))
            mpimg.imsave(DEBUG_ROOT+'%dimage_resize_y.png'%(row%10), np.max(image_resize, axis=1))
            mpimg.imsave(DEBUG_ROOT+'%dimage_resize_z.png'%(row%10), np.max(image_resize, axis=2))
            mpimg.imsave(DEBUG_ROOT+'%dlabel_resize_x.png'%(row%10), np.max(label_resize, axis=0))
            mpimg.imsave(DEBUG_ROOT+'%dlabel_resize_y.png'%(row%10), np.max(label_resize, axis=1))
            mpimg.imsave(DEBUG_ROOT+'%dlabel_resize_z.png'%(row%10), np.max(label_resize, axis=2))
            mpimg.imsave(DEBUG_ROOT+'%dmask_resize_x.png'%(row%10), np.max(mask_resize, axis=0))
            mpimg.imsave(DEBUG_ROOT+'%dmask_resize_y.png'%(row%10), np.max(mask_resize, axis=1))
            mpimg.imsave(DEBUG_ROOT+'%dmask_resize_z.png'%(row%10), np.max(mask_resize, axis=2))

            # 保存插值前后文件【训练时不能存，之后记得取消注释！！！】
            np.save(case['npy_image_path'], image)
            np.save(case['npy_mask_path'], mask)
            np.save(case['npy_label_path'], label)
            np.save(case['npy_image_interpolation_path'], image_resize)
            np.save(case['npy_mask_interpolation_path'], mask_resize)
            np.save(case['npy_label_interpolation_path'], label_resize)
        
        # 分别保存到训练和测试集中
        if list_train_or_test[row] == 1:
            TrainCaseList.append(case)
        else:
            TestCaseList.append(case)
    
    # 保存数据表
    trainset_mat = np.array(TrainCaseList)
    np.save(TRAINSETINFO_FILE_PATH, trainset_mat)
    testset_mat = np.array(TestCaseList)
    np.save(TESTSETINFO_FILE_PATH, testset_mat)
    pass


if __name__ == '__main__':
    # 创建dataset/名下训练数据准备文件夹
    utils_file.mkdir(FOLDER_IMAGE_NPY)
    utils_file.mkdir(FOLDER_MASK_NPY)
    utils_file.mkdir(FOLDER_LABEL_NPY)
    utils_file.mkdir(FOLDER_IMAGE_INTERPOLATION_NPY)
    utils_file.mkdir(FOLDER_MASK_INTERPOLATION_NPY)
    utils_file.mkdir(FOLDER_LABEL_INTERPOLATION_NPY)
    utils_file.mkdir(DEBUG_ROOT)
    # 获得中位spacing
    spacing_median = GetMedianSpacing(DATASET_CSV_PATH, STATIC_INFO_XLSX_PATH)
    print('the median spacing of these case is', spacing_median)
    # 开始插值B_INTERP
    B_INTERP = False
    DoBoxAndSpacing(DATASET_CSV_PATH, spacing_median, B_INTERP)