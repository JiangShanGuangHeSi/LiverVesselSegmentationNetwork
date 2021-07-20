'''
处理3dircadb数据
将原始的dicom文件转换为nii文件
数据存储在D:/XueZhimeng/project/DataSet/3Dircadb/3Dircadb1
每个案例保存在3DircadbXX.YY中，其中XX为1,2，YY为1-20
图像保存在PATIENT_DICOM文件夹，标签保存在MASKS_DICOM文件夹下的文件夹
即形如
D:/XueZhimeng/project/DataSet/3Dircadb/3Dircadb1/3DircadbXX.YY/
    PATIENT_DICOM/
    MASKS_DICOM/
        liver/
        portalvein/
        venacava/
        ...
将数据转换为
target/caseXXYY/caseXXYY_名字.nii.gz文件

XueZhimeng 2020.8
'''
import os
import SimpleITK as sitk
from utils_file import mkdir

def dcm_to_nii(dcm_dir, nii_filename):
    print('turn', dcm_dir, 'to', nii_filename)
    # 读dcm文件夹下的文件
    reader = sitk.ImageSeriesReader()
    dcm_series = reader.GetGDCMSeriesFileNames(dcm_dir)
    reader.SetFileNames(dcm_series)
    img = reader.Execute()
    # 存为nii文件
    sitk.WriteImage(img,nii_filename)

if __name__ == '__main__':
    DATASET_FOLDER = 'D:/XueZhimeng/project/DataSet/3Dircadb/3Dircadb1/'
    TARGET_FOLDER = 'D:/XueZhimeng/project/DataSet/3Dircadb/NiiFile/'
    
    for idx in range(20, 23):
        dcm_folders_to_trans = []
        nii_file_names = []

        if idx <= 20:
            XX = 1
            YY = idx
        else:
            XX = 2
            YY = idx - 20
        case_dcm_folder = DATASET_FOLDER + '3Dircadb' + str(XX) + '.' + str(YY) + '/'
        case_nii_folder = TARGET_FOLDER + 'case%d'%idx + '/'

        case_image_dcm_folder = case_dcm_folder + 'PATIENT_DICOM/'
        dcm_folders_to_trans.append(case_image_dcm_folder) # 将图像加入列表
        nii_file_names.append(case_nii_folder + 'case%d_patient.nii.gz'%idx)

        case_masks_dcm_folder = case_dcm_folder + 'MASKS_DICOM/'
        # 读取masks文件夹列表
        for root,dirs,files in os.walk(case_masks_dcm_folder):
            # 读取每个文件夹
            for folder in dirs:
                case_label_dcm_folder = case_masks_dcm_folder + folder + '/'
                dcm_folders_to_trans.append(case_label_dcm_folder)
                nii_file_names.append(case_nii_folder + 'case%d_'%idx + folder + '.nii.gz')
                pass
        
        mkdir(case_nii_folder)
        pass

        for n in range(len(dcm_folders_to_trans)):
            dcm_to_nii(dcm_folders_to_trans[n], nii_file_names[n])
