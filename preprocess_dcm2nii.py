'''
参考D:/XueZhimeng/project/code/xzm_LiverVesselIrcadb/preprocess_dcm2nii.py
处理中大医院数据
将原始的dicom文件转换为nii文件

数据集位置在D:/XueZhimeng/project/DataSet/中大医院数据/ZDYYLiverVessel/
dcm文件夹名字存储在D:/XueZhimeng/project/DataSet/中大医院数据/data_info.xlsx

按照将对应的dcm文件夹转化为nnUNet规定的nii.gz文件格式，形如
ZDYY_abc_0000.nii.gz

XueZhimeng 2021.3
'''

import os
import SimpleITK as sitk 
import pandas as pd

def dcm_to_nii(dcm_dir, nii_filename):
    print(dcm_dir, '\t---→\t', nii_filename)
    # 读dcm文件夹下的文件
    reader = sitk.ImageSeriesReader()
    dcm_series = reader.GetGDCMSeriesFileNames(dcm_dir)
    reader.SetFileNames(dcm_series)
    img = reader.Execute()
    # 存为nii文件
    sitk.WriteImage(img,nii_filename)

if __name__ == '__main__':
    DATASET_FOLDER = 'D:/XueZhimeng/project/DataSet/ZDYY/ZDYYLiverVessel/'
    path_data_info = 'D:/XueZhimeng/project/DataSet/ZDYY/data_info.csv'
    TARGET_FOLDER = 'D:/XueZhimeng/project/DataSet/ZDYY/ZDYYLiverVesselNII/'
    
    csv_data_info = pd.read_csv(path_data_info)
    list_idx = csv_data_info['idx'].values
    list_dcm_folder = csv_data_info['dcm_folder'].values

    for i in range(12, len(list_idx)):
        idx = list_idx[i]
        dcm_folder = list_dcm_folder[i]

        dcm_folder_fullname = DATASET_FOLDER + dcm_folder
        nii_filename = TARGET_FOLDER + 'ZDYY_' + str(idx).zfill(3) + '_0000.nii.gz'

        dcm_to_nii(dcm_folder_fullname, nii_filename)
    pass