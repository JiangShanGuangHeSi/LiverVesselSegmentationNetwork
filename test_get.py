'''
根据神经网络已经得到的npy结果预测。
按理说，预测结果保存在dataset\predice_...\文件夹里，并且有ZDYY_或者ircadb_标识数据集类型
训练所需要的所有信息都保存在image\, label\, mask\或者相应的
因此完全可以反查到
'''

from model_3D import UNet
from pathlib import Path
import numpy as np 
# import torch
import matplotlib.image as mpimg # mpimg 用于读写图片
import re # 提取文件名中的数字
import numpy as np 
import pandas as pd
import os
from skimage.transform import resize # 插值
# from losses import SoftDiceLoss, DICELoss
import nibabel as nib
from torch._C import ParameterDict
import utils_file as uf

# device = torch.device('cuda:0')  # 定义gpu位置

def myDICE(predict, label):
    intersection = np.sum(predict * label)
    den1 = np.sum(predict * predict)
    den2 = np.sum(label * label)
    eps = 1e-8
    dice = 2 * ((intersection + eps) / (den1 + den2 + eps))
    dice = np.sum(dice)
    return dice

def criterion(predict, label, thr):
    TP = np.sum(np.logical_and(predict > thr , label > thr))
    TN = np.sum(np.logical_and(predict < thr , label < thr))
    # FP = np.sum(np.logical_and(predict < thr , label > thr))
    # FN = np.sum(np.logical_and(predict > thr , label < thr))
    FP = np.sum(np.logical_and(predict > thr , label < thr))
    FN = np.sum(np.logical_and(predict < thr , label > thr))
    accuracy = (TP + TN) / np.size(label) # 精度
    sensitivity = TP / (TP + FN) # 敏感性
    specificity = TN / (TN + FP) # 特异性
    precision = TP / (TP + FP) # 查准率：(预测为1且正确预测的样本数)/(所有预测为1的样本数)
    recall = TP / (TP + FN) # 召回率：预测为1且正确预测的样本数)/(所有真实情况为1的样本数)
    # return acc, sens, spec
    return accuracy,sensitivity, specificity, precision, recall

if __name__=='__main__':
    # 原始数据位置，只需要image, label, mask中的一个来确定nii文件的affine
    # 因为label可能为多个不同的文件，故选择image文件位置
    PATH_RAW_IMAGE = {}
    PATH_RAW_IMAGE['ircadb'] = "D:/XueZhimeng/project/DataSet/3Dircadb/NiiFile/case{0:d}/case{0:d}_patient.nii.gz"
    PATH_RAW_IMAGE['ZDYY'] = "D:/XueZhimeng/project/DataSet/ZDYY/ZDYYLiverVesselNII/ZDYY_{0:03d}_0000.nii.gz"
    TESTSETINFO_FILE_PATH = "dataset/testset_info_szrch_spacing.npy" # 深圳第二人民医院数据集信息

    # 【保存的文件名字】
    # writer = pd.ExcelWriter("dataset/predict_szrch_flip.xlsx")
    # writer = pd.ExcelWriter("dataset/predict_szrch_flip2021-06-09-15-43-16.xlsx")

    # 【在下面写预测npy结果的保存路径】
    # UNet 2021-04-30-15-28-04
    # UNet+ACG 2021-05-24-15-45-46
    # UNet+MSFF 2021-05-25-14-47-11
    # UNet+AGC+MSFF 2021-05-28-17-22-37
    # PREDICT_FOLDERS = ["dataset/predict_2021-04-30-15-28-04_/", \
    #     "dataset/predict_2021-05-24-15-45-46_/", \
    #     "dataset/predict_2021-05-25-14-47-11_/", \
    #     "dataset/predict_2021-05-28-17-22-37_/"]
    # UNet 2021-06-07-15-53-49
    # UNet+MSFF 2021-06-08-11-04-51
    # PREDICT_FOLDERS = ["dataset/predict_2021-06-07-15-53-49_/", \
    #     "dataset/predict_2021-06-08-11-04-51_/"]
    # UNet 2021-06-09-15-43-16
    # PREDICT_FOLDERS = ["dataset/predict_2021-06-10-14-08-02_/"]
    # PREDICT_FOLDERS = ["dataset/predict_2021-06-10-16-44-34_/"]

    # 四折交叉验证的UNetDS4 2021-06-18-15-18-57和UNet 2021-06-19-11-58-30
    writer = pd.ExcelWriter("dataset/predict_szrch_flip20210621.xlsx")
    PREDICT_FOLDERS = ["dataset/predict_2021-06-18-15-18-57_/", "dataset/predict_2021-06-19-11-58-30_/"]

    # print(len(PATH_RAW_IMAGE['ircadb']), PATH_RAW_IMAGE['ircadb'].find('%', 60))
    # print(PATH_RAW_IMAGE['ircadb'].format(1))
    # print(PATH_RAW_IMAGE['ircadb'] % (1))

    for PREDICT_FOLDER in PREDICT_FOLDERS:
        PIC_PATH = PREDICT_FOLDER + "/predict_pic/" # 存图路径
        NII_PATH = PREDICT_FOLDER + "/predict_nii/" # 存图路径，保存为nii文件，affine和原来的一样
        uf.mkdir(PIC_PATH)
        uf.mkdir(NII_PATH)
        
        dataset_info = np.load(TESTSETINFO_FILE_PATH, allow_pickle=True).tolist() 

        case_names = []
        dices = []
        acces = []
        senses = []
        speces = []
        preces = []
        recalls = []
        case_idx = 0
        for case in dataset_info:
            # image = np.load(case['npy_data_path'])
            npy_label_path = case['npy_label_path'] # 不插值的label
            _, _, case_name, _ = uf.get_filename_info(npy_label_path)
            case_names.append(case_name)
            case_idx = uf.get_filename_number(case_name, 0)
            box = case['box']
            
            # --------loading prediction result (interpolated)--------
            predict = np.load(PREDICT_FOLDER + case_name + '.npy')
            
            # 保存用
            # --------loading NII raw image--------
            if(case_name.find('Ircadb') != -1):
                raw_image = nib.load(PATH_RAW_IMAGE['ircadb'].format(case_idx))
                # continue # 【debug】
            elif(case_name.find('ZDYY') != -1):
                raw_image = nib.load(PATH_RAW_IMAGE['ZDYY'].format(case_idx))

            # 测试用
            # --------loading label (not been interpolated)--------
            try:
                label = np.load('dataset/label/' + case_name + '.npy').astype(np.float32)
            except FileNotFoundError:
                print('label not exist, label set zeros.')
                label = np.zeros(raw_image.get_data().shape)
            else:
                # label[label != 1] = 0 # 【本数据集特色】
                label[label != 0] = 1 # 【本数据集特色】
            
            # --------loading mask (not been interpolated)--------
            try:
                mask = np.load('dataset/mask/' + case_name + '.npy')
            except: # 文件不存在or名字未定义错误
                print('mask may not exist, mask set ones.')
                mask = np.ones(raw_image.get_data().shape)
            else:
                mask[mask != 0] = 1
            
            # 反向插值
            # predict = resize(predict, label.shape, order=3)
            predict = resize(predict, label.shape, order=3)
            # 取阈值
            predict_bw = predict>0.5
            predict_bw = predict_bw.astype(np.float32)
            # 限制在mask里面
            predict_bw = predict_bw * mask
            label = label * mask

            predict_box = predict_bw[box[0]:box[3], box[1]:box[4], box[2]:box[5]]
            label_box = label[box[0]:box[3], box[1]:box[4], box[2]:box[5]]

            # dice = myDICE(predict, label)
            # acc, sens, spec, prec, recall = criterion(predict, label, 0.5)
            dice = myDICE(predict_box, label_box)
            acc, sens, spec, prec, recall = criterion(predict_box, label_box, 0.5)

            print(case_name, '\tDICE =', dice, '\tACC =', acc, '\tSENS =', sens, '\tSPEC =', spec, \
                '\tPREC =', prec, '\tRECALL =', recall)

            dices.append(dice)
            acces.append(acc)
            senses.append(sens)
            speces.append(spec)
            preces.append(prec)
            recalls.append(recall)

            mpimg.imsave(PIC_PATH + '%dpredict_x.png'%case_idx, np.max(predict_bw,axis=0))
            mpimg.imsave(PIC_PATH + '%dpredict_x_feature.png'%case_idx, np.max(predict,axis=0))
            mpimg.imsave(PIC_PATH + '%dlabel_x.png'%case_idx,np.max(label,axis=0))
            mpimg.imsave(PIC_PATH + '%dpredict_y.png'%case_idx,np.max(predict_bw,axis=1))
            mpimg.imsave(PIC_PATH + '%dpredict_y_feature.png'%case_idx,np.max(predict,axis=1))
            mpimg.imsave(PIC_PATH + '%dlabel_y.png'%case_idx,np.max(label,axis=1))
            mpimg.imsave(PIC_PATH + '%dpredict_z.png'%case_idx,np.max(predict_bw,axis=2))
            mpimg.imsave(PIC_PATH + '%dpredict_z_feature.png'%case_idx,np.max(predict,axis=2))
            mpimg.imsave(PIC_PATH + '%dlabel_z.png'%case_idx,np.max(label,axis=2))

            # 保存nii文件
            predict_bw = predict_bw.astype(np.uint8)
            nib.Nifti1Image(predict_bw,raw_image.affine).to_filename(NII_PATH + case_name + '.nii.gz')
            pass

        mean_dice = np.mean(np.array(dices))
        mean_acc = np.mean(np.array(acces))
        mean_sens = np.mean(np.array(senses))
        mean_spec = np.mean(np.array(speces))
        mean_prec = np.mean(np.array(preces))
        mean_recall = np.mean(np.array(recalls))

        case_names.append('mean')
        dices.append(mean_dice)
        acces.append(mean_acc)
        senses.append(mean_sens)
        speces.append(mean_spec)
        preces.append(mean_prec)
        recalls.append(mean_recall)
        print('\tmean dice =', mean_dice, '\tmean acc =', mean_acc, '\tmean sens =', mean_sens, '\tmean spec =', mean_spec)

        # 保存文件
        df = pd.DataFrame()
        df['case'] = case_names
        df['dice'] = dices
        df['accuracy'] = acces
        df['sensitivity'] = senses
        df['specificity'] = speces
        df['precision'] = preces
        df['recall'] = recalls
        df.to_excel(writer, sheet_name=PREDICT_FOLDER[-25:-2], index=False)
    
    writer.save()