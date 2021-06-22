'''
根据3DIrcadb数据集的标注
D:/XueZhimeng/project/DataSet/ZDYY/ZDYYLiverVesselNIIlabel
和中大医院数据集的标注
D:/XueZhimeng/project/DataSet/3Dircadb/LabelX
文件夹中的有的label制作包括数据集信息的csv文件

3DIrcadb数据集的
数据文件形式为
D:/XueZhimeng/project/DataSet/3Dircadb/NiiFile/case序号/case序号_patient.nii.gz
肝脏标注文件形式为
D:/XueZhimeng/project/DataSet/3Dircadb/NiiFile/case序号/case序号_liver.nii.gz
血管标注文件形式为
D:/XueZhimeng/project/DataSet/3Dircadb/LabelX/case序号_livervessel.nii.gz

中大医院数据集的
数据文件形式为
D:/XueZhimeng/project/DataSet/ZDYY/ZDYYLiverVesselNII/ZDYY_三位数序号_0000.nii.gz
肝脏标注文件形式为
D:/XueZhimeng/project/DataSet/ZDYY/ZDYYLiverPredictNnunet20210401/ZDYY_三位数序号.nii.gz
血管标注文件形式为
D:/XueZhimeng/project/DataSet/ZDYY/ZDYYLiverVesselNIIlabel/ZDYY_三位数序号_label.nii.gz

因为数据集不够用，所以目前测试集和验证集设置为同一批数据

XueZhimeng, 2021.4

新增了几例数据集，之前是17例，现在是？？？
数据集重新划分，之前是对半划分，现在采用四折交叉验证
XueZhimeng, 2021.6
'''

import os
import re
import pandas as pd
import utils_file

FOLDER_IMAGE_IRCADB = 'D:/XueZhimeng/project/DataSet/3Dircadb/NiiFile/'
FOLDER_MASK_IRCADB = 'D:/XueZhimeng/project/DataSet/3Dircadb/NiiFile/'
FOLDER_LABEL_IRCADB = 'D:/XueZhimeng/project/DataSet/3Dircadb/LabelX/'
FOLDER_IMAGE_ZDYY = 'D:/XueZhimeng/project/DataSet/ZDYY/ZDYYLiverVesselNII/'
FOLDER_MASK_ZDYY = 'D:/XueZhimeng/project/DataSet/ZDYY/ZDYYLiverPredictNnunet20210401/'
FOLDER_LABEL_ZDYY = 'D:/XueZhimeng/project/DataSet/ZDYY/ZDYYLiverVesselNIIlabel/'

if __name__ == '__main__':
    # 读取2个数据集的label标号列表，得到数据的名字列表，数据集名字列表，数据image列表，肝脏mask列表，标注label列表
    count = 0 # 计数
    list_name_case = []
    list_name_dataset = []
    list_path_image = []
    list_path_mask = []
    list_path_label = []
    list_train_or_test = []
    # 读取Ircadb数据集文件夹
    for root, dirs, files in os.walk(FOLDER_LABEL_IRCADB):     # 分别代表根目录、文件夹、文件
        for file in files:                         # 遍历文件
            file_path = os.path.join(root, file)   # 获取文件绝对路径 
            idx_file = utils_file.get_filename_number(file, 0) # 获取文件序号
            name_case = 'Ircadb_%(idx_file)03d'%{"idx_file": idx_file} # 给这个case起个名字
            path_image = FOLDER_IMAGE_IRCADB + 'case%(idx_file)d/case%(idx_file)d_patient.nii.gz'%{"idx_file": idx_file}
            path_mask = FOLDER_MASK_IRCADB + 'case%(idx_file)d/case%(idx_file)d_liver.nii.gz'%{"idx_file": idx_file}
            
            list_name_case.append(name_case)
            list_name_dataset.append('Ircadb')
            list_path_image.append(path_image)
            list_path_mask.append(path_mask)
            list_path_label.append(file_path)
            # list_train_or_test.append(idx_file % 2)
            count = count + 1
            list_train_or_test.append(int(count % 4 != 0)) # 【改：修正为四折交叉验证】
    # 读取ZDYY数据集文件夹
    for root, dirs, files in os.walk(FOLDER_LABEL_ZDYY):     # 分别代表根目录、文件夹、文件
        for file in files:                         # 遍历文件
            file_path = os.path.join(root, file)   # 获取文件绝对路径 
            idx_file = utils_file.get_filename_number(file, 0) # 获取文件序号
            name_case = 'ZDYY_%(idx_file)03d'%{"idx_file": idx_file} # 给这个case起个名字
            path_image = FOLDER_IMAGE_ZDYY + 'ZDYY_%(idx_file)03d_0000.nii.gz'%{"idx_file": idx_file}
            path_mask = FOLDER_MASK_ZDYY + 'ZDYY_%(idx_file)03d.nii.gz'%{"idx_file": idx_file}
            
            list_name_case.append(name_case)
            list_name_dataset.append('ZDYY')
            list_path_image.append(path_image)
            list_path_mask.append(path_mask)
            list_path_label.append(file_path)
            # list_train_or_test.append(idx_file % 2)
            count = count + 1
            list_train_or_test.append(int(count % 4 != 0)) # 【改：修正为四折交叉验证】
    
    # 字典中的key值即为csv中列名
    dataframe = pd.DataFrame({ \
        'name_case': list_name_case, \
        'name_dataset': list_name_dataset, \
        'raw_image_path':list_path_image, \
        'raw_mask_path':list_path_mask, \
        'raw_label_path':list_path_label, \
        'train_or_test': list_train_or_test})
    #将DataFrame存储为csv,index表示是否显示行名，default=True
    # dataframe.to_csv("data_path_info_mix20210607.csv",index=False,sep=',') # 训练集测试集搞反了
    dataframe.to_csv("data_path_info_mix20210610.csv",index=False,sep=',')
    pass