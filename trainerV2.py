'''
训练器
根据参数，自动生成模型，读取数据，保存文件，打印信息
XueZhimeng, 2020.7

去掉了混合精度
XueZhimeng, 2021.4

改进模型名字为空训练后预测重新生成文件的问题
添加深度监督模型和对应的label处理
XueZhimeng, 2021.6
'''
import os
import struct
import numpy as np
import time
import json 

import torch.nn.functional as F
import torch
import torch.nn as nn
from torch import autograd, optim
import torch.utils.data as data
from torch.utils.data import DataLoader
from PIL import Image
from PIL import ImageDraw, ImageFont
import matplotlib.image as mpimg # mpimg 用于读取图片
# from torchvision.transforms import transforms
# from apex import amp

from model_3D import UNet
from models.networks.unet_CT_single_att_dsv_3D import unet_CT_single_att_dsv_3D
from model_3dUNet_v2 import UNetV2
from model_3dUNet_v2ds import UNetV2ds
from model_3dUNet_ACG import UNetAGC, UNetMSFF, UNetAgcMsff

# from losses import DICELoss
# from losses import SoftDiceLoss, DICELossMultiClass
# from losses import MulticlassDiceLoss, DiceLoss
from losses import DiceLoss
from loss_focal import FocalLoss
from loss_dice_and_ce import DC_and_CE_loss, SoftDiceLoss

# 深度监督损失
from loss_deep_supervision import MultipleOutputLoss2
from loss_downsample_seg_for_ds import downsample_seg_for_ds_transform3

import augmentation_space as ia

# from dataloader import CTDataset
# from dataloaderV2 import DatasetV2
# from dataloaderV2_1 import DatasetV2_1
from dataloaderV2_2 import DatasetV2_2

from utils_time import get_time_str
from utils_file import mkdir, get_filename_info

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class Trainer():
    def __init__(self, str_device='cuda:0', 
                str_model='UNet', in_dim=1, out_dim=2, num_filter=64, patch_size=[96, 96, 96],
                str_loss='dice', loss_op={},
                learning_rate=2e-4, batch_size=2, num_epoches=300, old_model_name=None, epoches_save=50,
                amp_opt='O0'):
        self.str_model = str_model
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_filter = num_filter
        self.patch_size = patch_size
        self.str_loss = str_loss
        self.lr = learning_rate
        self.batch_size = batch_size
        self.num_epoches = num_epoches
        self.old_model_name = old_model_name
        self.epoches_save = epoches_save
        self.amp_opt = amp_opt

        self.model_folder = None
        self.debug_pic_folder = None
        self.record_filename = None
        
        self.best_val_loss = 1000
        self.last_epoch = 0
        self.last_model_path = ''
        self.best_epoch = 0
        self.best_model_path = ''

        self.aug_params = {
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

        # 加载旧网络（如果有的话）
        if old_model_name:
            self.load_train_info_dict(old_model_name)

        # 指定设备
        self.device = torch.device(str_device)

        # 网络
        self.do_deep_supervision = False # 深度监督参数
        self.num_ds = 0
        if self.str_model == 'UNet':
            self.model = UNet(in_dim, out_dim, num_filter)
        elif self.str_model == 'UNetSAD':
            self.model = unet_CT_single_att_dsv_3D(feature_scale=64 // num_filter, n_classes=out_dim, is_deconv=True, in_channels=in_dim)
        elif self.str_model == 'UNetV2':
            self.model = UNetV2(1, 2, self.num_filter)
        elif self.str_model == 'UNetACG':
            self.model = UNetAGC(1, 2, self.num_filter)
        elif self.str_model == 'UNetMSFF':
            self.model = UNetMSFF(1, 2, self.num_filter)
        elif self.str_model == 'UNetAgcMsff':
            self.model = UNetAgcMsff(1, 2, self.num_filter)
        elif self.str_model == 'UNetV2DS4': # 深度监督网络
            self.do_deep_supervision = True
            self.num_ds = 4
            self.model = UNetV2ds(1, 2, self.num_filter, self.num_ds)
            # 计算降采样的尺度
            self.ds_scales = []
            for i in range(self.num_ds):
                self.ds_scales.append(list(np.power([1/2, 1/2, 1/2], i)))
        else:
            raise Exception('no proposed model')
        self.model = self.model.to(self.device)
        
        # 损失函数
        if self.str_loss == 'dice':
            # self.loss = MulticlassDiceLoss()
            loss = SoftDiceLoss()
        elif self.str_loss == 'bce':
            loss = nn.BCELoss()
        elif self.str_loss == 'focal':
            loss = FocalLoss(apply_nonlin=loss_op['apply_nolin'], alpha=loss_op['alpha'], \
                gamma=loss_op['gamma'], balance_index=loss_op['balance_index'])
        elif self.str_loss == 'DcAndCe':
            loss = DC_and_CE_loss()
        else:
            raise Exception('no proposed loss')
        
        self.loss_val = loss
        if self.do_deep_supervision: # 深度监督对损失函数改进
            ds_loss_weights = np.array([1 / (2 ** i) for i in range(self.num_ds)])
            ds_loss_weights = ds_loss_weights / ds_loss_weights.sum()
            self.loss_train = MultipleOutputLoss2(loss, ds_loss_weights)
        else:
            self.loss_train = loss

        # 优化器
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # # 混合精度
        # if self.amp_opt != '' and self.amp_opt != 'O0':
        #     # self.amp_opt = amp_opt
        #     self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level=amp_opt)
        # else:
        #     self.amp_opt = 'O0'

        # # 分布式训练
        # self.model = nn.DataParallel(self.model)

    def init_record_path(self):
        '''
        # 如果第一次训练则
        根据网络类型、通道数量、损失函数、精度、时间
        新建并命名模型保存文件夹，过程图像文件夹，命名记录文件

        记录网络信息到json文件

        # 如果继续训练则加载模型
        '''

        if self.old_model_name:
            if self.last_model_path != '':
                # self.str_time = self.old_model_name
                self.model.load_state_dict(torch.load(self.last_model_path, map_location=self.device))
            self.model_name = self.str_time + '_'
        else:
            str_filter = str(self.num_filter)
            str_ch = 'Ch' + str(self.out_dim)
            # if self.amp_opt:
            #     str_amp = 'amp' + self.amp_opt
            # else:
            #     str_amp = 'ampO0'
            str_amp = 'amp' + self.amp_opt
            self.str_time = get_time_str() # 获取时间（网络唯一标识）
            self.old_model_name = self.str_time # 【20210616修改】改进模型名字为空训练后预测重新生成文件的问题

            self.model_name = self.str_model + str_filter + '_' + str_ch + '_' + str_amp + '_' \
                        + self.str_loss + '_' + self.str_time
            self.model_folder = 'model/' + self.model_name + '/'
            self.debug_pic_folder = 'pic_debug/' + self.model_name + '/'
            self.record_filename = 'record/' + self.model_name + '.txt'
            global mkdir
            mkdir('model/') # UnboundLocalError: local variable 'mkdir' referenced before assignment
            mkdir('pic_debug/')
            mkdir('record/')
            mkdir('train_infomation/')
            self.model_name = self.str_time + '_'

            # 记录包含详细训练信息的json文件
            # self.train_info_dict = {}
            # self.train_info_dict['time'] = self.str_time
            # self.train_info_dict['model'] = self.str_model
            # self.train_info_dict['base filter'] = self.num_filter
            # self.train_info_dict['channel in'] = self.in_dim
            # self.train_info_dict['channel out'] = self.out_dim
            # self.train_info_dict['patch size'] = self.patch_size
            # self.train_info_dict['loss'] = self.str_loss
            # self.train_info_dict['learning rate'] = self.lr
            # # self.train_info_dict['amp opt'] = str_amp
            # self.train_info_dict['amp opt'] = self.amp_opt
            # self.train_info_dict['model folder'] = self.model_folder
            # self.train_info_dict['debug pic folder'] = self.debug_pic_folder
            # self.train_info_dict['record filename'] = self.record_filename
            # self.train_info_dict['last epoch'] = self.last_epoch # 为0
            # self.train_info_dict['last model'] = self.last_model_path # 为''
            # self.train_info_dict['description'] = '模型改进自nnunet，32通道，主要是对比nnunet的结构相对于旧的网络有没有提高，数据没有插值'
            # json_str = json.dumps(self.train_info_dict, indent=4)
            # with open('train_infomation/' + self.str_time + '.json', 'w') as json_file:
            #     json_file.write(json_str)
            self.save_train_info_dict()

            if os.path.exists(self.model_folder) or os.path.exists(self.debug_pic_folder):
                print('WARNING: MODEL_RECORD_FLODER与DEBUG_ROOT已经存在，请确定它们是空的，或者之前的内容已备份！')
            from utils_file import mkdir
            # 第一次训练该模型，初始化文件夹和文档
            mkdir(self.model_folder)
            mkdir(self.debug_pic_folder)
    
    def save_train_info_dict(self):
        # 记录包含详细网络，训练，存储路径信息的json文件
        self.train_info_dict = {}
        self.train_info_dict['time'] = self.str_time

        self.train_info_dict['model'] = self.str_model
        self.train_info_dict['base filter'] = self.num_filter
        self.train_info_dict['channel in'] = self.in_dim
        self.train_info_dict['channel out'] = self.out_dim
        self.train_info_dict['patch size'] = self.patch_size

        self.train_info_dict['loss'] = self.str_loss
        self.train_info_dict['learning rate'] = self.lr
        # self.train_info_dict['amp opt'] = str_amp
        self.train_info_dict['amp opt'] = self.amp_opt

        self.train_info_dict['model folder'] = self.model_folder
        self.train_info_dict['debug pic folder'] = self.debug_pic_folder
        self.train_info_dict['record filename'] = self.record_filename
        # self.train_info_dict['last epoch'] = self.last_epoch # 为0
        # Object of type int32 is not JSON serializable 
        # 报错是因为self.last_epoch的数据类型是np的int，而不是python的int
        self.train_info_dict['last epoch'] = int(self.last_epoch) # 为0
        self.train_info_dict['last model'] = self.last_model_path # 为''
        self.train_info_dict['best epoch'] = int(self.best_epoch)
        self.train_info_dict['best model'] = self.best_model_path
        self.train_info_dict['best validation loss'] = float(self.best_val_loss)

        self.train_info_dict['description'] = 'liver vessel'
        
        json_str = json.dumps(self.train_info_dict, indent=4)
        with open('train_infomation/' + self.str_time + '.json', 'w') as json_file:
            json_file.write(json_str)
    
    def load_train_info_dict(self, model_name):
        # 加载包含详细网络，训练，存储路径信息的json文件
        json_file = 'train_infomation/' + model_name + '.json'
        with open(json_file,'r') as load_f:
            self.train_info_dict = json.load(load_f)
        print(self.train_info_dict)
        self.str_time = self.train_info_dict['time']

        self.str_model = self.train_info_dict['model']
        self.num_filter = self.train_info_dict['base filter']
        self.in_dim = self.train_info_dict['channel in']
        self.out_dim = self.train_info_dict['channel out']
        self.patch_size = self.train_info_dict['patch size']

        self.str_loss = self.train_info_dict['loss']
        # self.lr = self.train_info_dict['learning rate'] # 【不读学习率，每次自己指定】
        self.amp_opt = self.train_info_dict['amp opt']

        self.model_folder = self.train_info_dict['model folder']
        self.debug_pic_folder = self.train_info_dict['debug pic folder']
        self.record_filename = self.train_info_dict['record filename']

        self.last_epoch = self.train_info_dict['last epoch']
        self.last_model_path = self.train_info_dict['last model']
        self.best_epoch = self.train_info_dict['best epoch']
        self.best_model_path = self.train_info_dict['best model']
        self.best_val_loss = self.train_info_dict['best validation loss']
    

    def train(self):
        
        self.init_record_path()

        # train_dataset = DatasetV2(self.patch_size, dataset_category='train', channel=self.out_dim, data_len=512)   # 这是训练集
        # train_dataloaders = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        # val_dataset = DatasetV2(self.patch_size, dataset_category='val', channel=self.out_dim) # 验证集
        # val_dataloaders = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

        # train_dataset = DatasetV2_1(self.patch_size, dataset_category='train', channel=self.out_dim, data_len=512, aug_params=self.aug_params)   # 这是训练集
        # train_dataloaders = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        # val_dataset = DatasetV2_1(self.patch_size, dataset_category='val', channel=self.out_dim, aug_params=self.aug_params) # 验证集
        # val_dataloaders = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        
        train_dataset = DatasetV2_2(self.patch_size, dataset_category='train', channel=self.out_dim, data_len=512, aug_params=self.aug_params)   # 这是训练集
        train_dataloaders = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        val_dataset = DatasetV2_2(self.patch_size, dataset_category='val', channel=self.out_dim, aug_params=self.aug_params) # 验证集
        val_dataloaders = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

        print('训练集长度:', len(train_dataset))
        print('验证集长度:', len(val_dataset))

        dt_size_train = len(train_dataloaders.dataset)
        dt_size_val = len(val_dataloaders.dataset)

        begin_epoch = self.last_epoch
        for epoch in np.arange(begin_epoch, self.num_epoches) + 1:   # range本身的范围是0-N-1，现在+1变成1-N
            # 训练
            if self.do_deep_supervision:# 设置深度监督网络的属性
                self.model.do_ds = True 

            self.model.train()
            print('Epoch {}/{}'.format(epoch, self.num_epoches))
            print('-' * 10)
            epoch_loss = 0   # 这个里面存的是每一轮里面的累计loss值
            # acc = 0
            step = 0
            for x, y in train_dataloaders:   # 这边一次取出一个batchsize的东西
                step += 1
                inputs = x.to(self.device)
                label = y.to(self.device)
                if self.do_deep_supervision: # 深度监督需要将label转换为各个尺度的list
                    label = downsample_seg_for_ds_transform3(label, self.ds_scales)

                self.optimizer.zero_grad()

                outputs = self.model(inputs)

                # outputs = F.softmax(outputs, 1) # 要特别注意，有些损失自带softmax，有些没有！
                if self.str_loss in ['dice', 'bce', 'focal']:
                    loss = self.loss_train(outputs, label)
                elif self.str_loss in ['DcAndCe']:
                    loss = self.loss_train(outputs, label[:, 1, :, :, :])

                # current_batchsize = outputs.size()[0] # 深度监督引发的问题：'list' object has no attribute 'size'
                current_batchsize = train_dataloaders.batch_size

                # 反向传播
                loss.backward()
                # 更新参数
                self.optimizer.step()
                epoch_loss += loss.item()*current_batchsize 

                # 存图检验
                if isinstance(outputs, list): # 深度监督引发的问题：list indices must be integers or slices, not tuple
                    pic0 = np.max(np.squeeze(outputs[0][0,1,:,:,:].detach().cpu().numpy()),axis=2)
                    pic1 = np.max(np.squeeze(label[0][0,1,:,:,:].detach().cpu().numpy()),axis=2)
                else: # 普通输出类型
                    pic0 = np.max(np.squeeze(outputs[0,1,:,:,:].detach().cpu().numpy()),axis=2)
                    pic1 = np.max(np.squeeze(label[0,1,:,:,:].detach().cpu().numpy()),axis=2)
                pic2 = np.max(np.squeeze(inputs[0,0,:,:,:].detach().cpu().numpy()),axis=2)
                mpimg.imsave(self.debug_pic_folder+'train_%doutput.png'%(step%10), pic0)
                mpimg.imsave(self.debug_pic_folder+'train_%dlabel.png'%(step%10), pic1)
                mpimg.imsave(self.debug_pic_folder+'train_%dinput.png'%(step%10), pic2)

                print("epoch:%d %d/%d,train_loss:%0.8f" % (epoch, step, (dt_size_train - 1) // train_dataloaders.batch_size + 1, loss.item()))   # 这个是输出每一步的loss值，step/一共要多少轮
            # 一轮训练结束之后
            # print("train_loss_%d"%epoch, epoch_loss)
            epochmean = epoch_loss/dt_size_train
            print("train_loss_mean_%d"%epoch, epochmean)

            # 在验证集上进行评估
            if self.do_deep_supervision:# 设置深度监督网络的属性
                self.model.do_ds = False 

            self.model.eval()
            with torch.no_grad():  # 不用记录梯度
                epoch_loss_val = 0
                step_val = 0
                for x, y in val_dataloaders:   # 这边一次取出一个batchsize的东西
                    step_val += 1
                    inputs = x.to(self.device)
                    label = y.to(self.device)
                    # if self.do_deep_supervision: # 深度监督需要将label转换为各个尺度的list，注意test和val的list长度为1
                    #     # label = downsample_seg_for_ds_transform3(label, self.ds_scales[0]) # TypeError: 'numpy.float64' object is not iterable
                    #     label = downsample_seg_for_ds_transform3(label, [self.ds_scales[0]])
                    
                    outputs = self.model(inputs)
                    
                    # outputs = F.softmax(outputs, 1) # 要特别注意，有些损失自带softmax，有些没有！
                    if self.str_loss in ['dice', 'bce', 'focal']:
                        loss = self.loss_val(outputs, label)
                    elif self.str_loss in ['DcAndCe']:
                        loss = self.loss_val(outputs, label[:, 1, :, :, :])

                    # current_batchsize = outputs.size()[0] # 深度监督引发的问题：'list' object has no attribute 'size'
                    current_batchsize = val_dataloaders.batch_size
                    epoch_loss_val += loss.item()*current_batchsize 
                    
                    # 存图检验
                    pic0 = np.max(np.squeeze(outputs[0,1,:,:,:].detach().cpu().numpy()),axis=2)
                    pic1 = np.max(np.squeeze(label[0,1,:,:,:].detach().cpu().numpy()),axis=2)
                    pic2 = np.max(np.squeeze(inputs[0,0,:,:,:].detach().cpu().numpy()),axis=2)
                    mpimg.imsave(self.debug_pic_folder+'val_%doutput.png'%(step_val%10), pic0)
                    mpimg.imsave(self.debug_pic_folder+'val_%dlabel.png'%(step_val%10), pic1)
                    mpimg.imsave(self.debug_pic_folder+'val_%dinput.png'%(step_val%10), pic2)
                    # # 存图检验
                    # if isinstance(outputs, list): # 深度监督引发的问题：list indices must be integers or slices, not tuple
                    #     pic0 = np.max(np.squeeze(outputs[0][0,1,:,:,:].detach().cpu().numpy()),axis=2)
                    #     pic1 = np.max(np.squeeze(label[0][0,1,:,:,:].detach().cpu().numpy()),axis=2)
                    # else: # 普通输出类型
                    #     pic0 = np.max(np.squeeze(outputs[0,1,:,:,:].detach().cpu().numpy()),axis=2)
                    #     pic1 = np.max(np.squeeze(label[0,1,:,:,:].detach().cpu().numpy()),axis=2)
                    # pic2 = np.max(np.squeeze(inputs[0,0,:,:,:].detach().cpu().numpy()),axis=2)
                    # mpimg.imsave(self.debug_pic_folder+'val_%doutput.png'%(step_val%10), pic0)
                    # mpimg.imsave(self.debug_pic_folder+'val_%dlabel.png'%(step_val%10), pic1)
                    # mpimg.imsave(self.debug_pic_folder+'val_%dinput.png'%(step_val%10), pic2)

                    print("epoch:%d %d/%d,val_loss:%0.8f" % (epoch, step_val, (dt_size_val - 1) // val_dataloaders.batch_size + 1, loss.item()))   # 这个是输出每一步的loss值，step/一共要多少轮
                # 一轮验证结束之后
                # print("val_loss_%d"%epoch, epoch_loss)
                epochmean_val = epoch_loss_val/dt_size_val
                print("val_loss_mean_%d"%epoch, epochmean_val)
                if epochmean_val < self.best_val_loss:
                    self.best_val_loss = epochmean_val
                    self.best_epoch = epoch
                    self.best_model_path = self.model_folder + self.model_name + '{}_best.pth'.format(epoch)
                    self.last_epoch = epoch
                    self.last_model_path = self.model_folder + self.model_name + '{}_best.pth'.format(epoch)
                    torch.save(self.model.state_dict(), self.last_model_path) # 存模型
                    self.save_train_info_dict()

            # 对结果进行存储
            self.saverecord(self.record_filename, epoch, epochmean, epochmean_val)
            torch.cuda.empty_cache()
            if epoch % self.epoches_save == 0:  # 每隔self.epoches_save轮存一下
                self.last_epoch = epoch
                self.last_model_path = self.model_folder + self.model_name + '{}.pth'.format(epoch)
                torch.save(self.model.state_dict(), self.last_model_path)  
                self.save_train_info_dict() # 存记录

    # 中间过程存储函数
    def saverecord(self, savepath, epoch, train_loss_mean, val_loss_mean):
        fo = open(savepath, 'a+')
        fo.write('train_loss_mean{}:{}'.format(epoch, train_loss_mean))
        fo.write('\n')
        fo.write('val_loss_mean{}:{}'.format(epoch, val_loss_mean))
        fo.write('\n')
        fo.close()
    
    def test(self, do_gaussian=True, do_mirroring=True):
        self.init_record_path()

        # 准备文件夹
        # predict_temp_folder = 'dataset/predict'
        predict_final_folder = 'dataset/predict_' + self.model_name + '/'
        # mkdir(predict_temp_folder)
        mkdir(predict_final_folder)

        test_dataset = DatasetV2_2(self.patch_size, dataset_category='test', channel=self.out_dim) # 测试集
        test_dataloaders = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)

        # 计算每个patch的高斯权重【待完成】
        gaussian_weight = _get_gaussian(self.patch_size, sigma_scale=1. / 4, do_gaussian=do_gaussian)
        gaussian_weight = torch.from_numpy(gaussian_weight)
        # 存图看看
        mpimg.imsave(self.debug_pic_folder+'gaussian_weight.png', np.squeeze(gaussian_weight[:,:,48]))

        predict_batches = [] # 存储预测结果
        with torch.no_grad():
            dt_size = len(test_dataloaders.dataset)
            self.model.eval()
            criterion = SoftDiceLoss() # 损失一律用dice
            eta = 0.0000001
            dice = 0
            dice_original = 0
            thres = 0.5  # 分割的阈值

            # softmax = F.softmax(1)

            # 获得单独batch的预测结果
            for x, y, batch_infos in test_dataloaders:   # 这边一次取出一个batchsize的东西
                inputs = x.to(self.device)
                labels = y.to(self.device)
                # 每个patch都进行flip预测，再平均【】
                # outputs = self.model(inputs)
                outputs = self.mirror_pred(inputs, mirror_axes=(0,1,2), do_mirroring=do_mirroring)
                # 这个时候，和单通道不同，是两个通道的
                outputs = F.softmax(outputs, 1)
                outputs = outputs[:,1,:,:,:] # 改成1个通道
                outputs = outputs.unsqueeze(1)
                outputs_binary = outputs.clone()
                outputs_binary[outputs > thres] = 1
                outputs_binary[outputs <= thres] = 0

                loss = criterion(outputs_binary, labels)

                # 进行存储
                # 原来batch_info是字典列表，结果取出来变成了列表字典……麻烦【待解决】
                case_idxes = batch_infos['case_idx']
                cubes = batch_infos['cube']
                for i in range(len(case_idxes)):
                    batch_predict = {}
                    batch_predict['case_idx'] = case_idxes[i]
                    batch_predict['cube'] = []
                    for j in range(6):
                        batch_predict['cube'].append(cubes[j][i])
                    # batch_predict['output'] = outputs.cpu().numpy()
                    batch_predict['output'] = torch.squeeze(outputs[i]).cpu() # 转移到cpu因为gpu存多了装不下
                    predict_batches.append(batch_predict)
                    pass

                print('case', case_idxes.cpu().numpy(), ', loss:', loss.cpu().numpy())
                # for batch in batch_infos:
                #     pass
                    # batch_infos['output'] = outputs.cpu().numpy()
                pass

            # 组合预测结果
            predict_labels = []
            weights = []
            labels = []
            case_idxes = test_dataset.case_idxes # 注意这里的序号和前面的不一样
            for case_idx in case_idxes: 
                # # 对每个case初始化空间
                # image_shape = test_dataset.dataset_info[case_idx]['image'].shape
                # predict_labels.append(torch.zeros(image_shape).to(device))
                # weights.append(torch.zeros(image_shape).to(device))
                # # 读出整体的label
                # label = test_dataset.dataset_info[case_idx]['label'].astype(np.float32)
                # labels.append(torch.from_numpy(label).to(device))
                # 对每个case初始化空间
                image_shape = test_dataset.dataset_info[case_idx]['image'].shape
                predict_labels.append(torch.zeros(image_shape))
                weights.append(torch.zeros(image_shape))
                # 读出整体的label
                label = test_dataset.dataset_info[case_idx]['label'].astype(np.float32)
                labels.append(torch.from_numpy(label))
            
            ifweight = True
            for predict_batch in predict_batches: # 累加结果
                case_idx = predict_batch['case_idx'] # 取出来的都是tensor，我也不知道为啥
                cube = predict_batch['cube']
                predict_label_batch = predict_batch['output']
                # 计算能够取到的数据范围上下限，这是为了防止选择的cube超出图像边界
                x = cube[0]
                y = cube[2]
                z = cube[4]
                x_l = max(0, x)
                x_u = min(predict_labels[case_idx].shape[0], x + self.patch_size[0])
                y_l = max(0, y)
                y_u = min(predict_labels[case_idx].shape[1], y + self.patch_size[1])
                z_l = max(0, z)
                z_u = min(predict_labels[case_idx].shape[2], z + self.patch_size[2])
                if ifweight: # 加权平均的方法
                    # predict_labels[case_idx][cube[0]:cube[1], cube[2]:cube[3], cube[4]:cube[5]] += \
                    #     predict_label_batch
                    # weights[case_idx][cube[0]:cube[1], cube[2]:cube[3], cube[4]:cube[5]] += \
                    #     1
                    predict_labels[case_idx][x_l:x_u, y_l:y_u, z_l:z_u] += predict_label_batch[x_l-x:x_u-x, y_l-y:y_u-y, z_l-z:z_u-z]
                    # weights[case_idx][x_l:x_u, y_l:y_u, z_l:z_u] += 1
                    weights[case_idx][x_l:x_u, y_l:y_u, z_l:z_u] += gaussian_weight[x_l-x:x_u-x, y_l-y:y_u-y, z_l-z:z_u-z] # 【高斯加权】add_(): argument 'other' (position 1) must be Tensor, not numpy.ndarray
                else: # 不平均，直接拼
                    # predict_labels[case_idx][cube[0]:cube[1], cube[2]:cube[3], cube[4]:cube[5]] = predict_label_batch
                    predict_labels[case_idx][x_l:x_u, y_l:y_u, z_l:z_u] = predict_label_batch[x_l-x:x_u-x, y_l-y:y_u-y, z_l-z:z_u-z]
            
            for case_idx in case_idxes: 
                if ifweight:
                    predict_labels[case_idx] /= (weights[case_idx] + eta) # 加权平均则除以权重

                # 存图检查
                predict_label_np = predict_labels[case_idx].cpu().numpy()
                label_np = labels[case_idx].cpu().numpy()
                # predict_npy_path = test_dataset.dataset_info[case_idx]['npy_predict_path']
                _, _, predict_npy_path, _ = get_filename_info(test_dataset.dataset_info[case_idx]['npy_image_path'])
                predict_npy_path = predict_final_folder + predict_npy_path + '.npy'
                np.save(predict_npy_path, predict_label_np)
                mpimg.imsave(self.debug_pic_folder+'test_%dpredict.png'%case_idx,np.max(predict_label_np,axis=2))
                mpimg.imsave(self.debug_pic_folder+'test_%dlabel.png'%case_idx,np.max(label_np,axis=2))

                # RuntimeError
                # invalid argument 2: view size is not compatible with input tensor's size and stride 
                # (at least one dimension spans across two contiguous subspaces). 
                # Call .contiguous() before .view(). 
                # 调用view时存储空间不连续
                predict_bw = predict_labels[case_idx]
                predict_bw[predict_bw > thres] = 1
                predict_bw[predict_bw <= thres] = 0
                # predict_bw = torch.where(predict_bw>0.5, torch.full_like(predict_bw, 1), torch.full_like(predict_bw, 0))
                loss = criterion(predict_bw.contiguous(), labels[case_idx].contiguous())
                print(loss)
            
            # # 修改文件夹名字
            # os.rename(predict_temp_folder, predict_final_folder)
    
    def mirror_pred(self, inputs, mirror_axes=(0,1,2), do_mirroring=True):
        '''每个轴都进行翻转预测，返回预测结果的平均值
        抄自nnunet.network_architecture.neural_network的_internal_maybe_mirror_and_pred_3D'''
        if do_mirroring:
            mirror_idx = 8
            num_results = 2 ** len(mirror_axes)
        else:
            mirror_idx = 1
            num_results = 1

        for m in range(mirror_idx):
            if m == 0:
                pred = self.model(inputs)
                result_torch = 1 / num_results * pred

            if m == 1 and (2 in mirror_axes):
                pred = self.model(torch.flip(inputs, (4, )))
                result_torch += 1 / num_results * torch.flip(pred, (4,))

            if m == 2 and (1 in mirror_axes):
                pred = self.model(torch.flip(inputs, (3, )))
                result_torch += 1 / num_results * torch.flip(pred, (3,))

            if m == 3 and (2 in mirror_axes) and (1 in mirror_axes):
                pred = self.model(torch.flip(inputs, (4, 3)))
                result_torch += 1 / num_results * torch.flip(pred, (4, 3))

            if m == 4 and (0 in mirror_axes):
                pred = self.model(torch.flip(inputs, (2, )))
                result_torch += 1 / num_results * torch.flip(pred, (2,))

            if m == 5 and (0 in mirror_axes) and (2 in mirror_axes):
                pred = self.model(torch.flip(inputs, (4, 2)))
                result_torch += 1 / num_results * torch.flip(pred, (4, 2))

            if m == 6 and (0 in mirror_axes) and (1 in mirror_axes):
                pred = self.model(torch.flip(inputs, (3, 2)))
                result_torch += 1 / num_results * torch.flip(pred, (3, 2))

            if m == 7 and (0 in mirror_axes) and (1 in mirror_axes) and (2 in mirror_axes):
                pred = self.model(torch.flip(inputs, (4, 3, 2)))
                result_torch += 1 / num_results * torch.flip(pred, (4, 3, 2))

        return result_torch

def _get_gaussian(patch_size, sigma_scale=1. / 8, do_gaussian=False) -> np.ndarray:
    '''得到一个patch的高斯加权的权值
    抄自nnunet.network_architecture.neural_network'''
    if do_gaussian:
        from scipy.ndimage.filters import gaussian_filter
        
        tmp = np.zeros(patch_size)
        center_coords = [i // 2 for i in patch_size]
        sigmas = [i * sigma_scale for i in patch_size]
        tmp[tuple(center_coords)] = 1
        gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)
        gaussian_importance_map = gaussian_importance_map / np.max(gaussian_importance_map) * 1
        gaussian_importance_map = gaussian_importance_map.astype(np.float32)

        # gaussian_importance_map cannot be 0, otherwise we may end up with nans!
        gaussian_importance_map[gaussian_importance_map == 0] = np.min(
            gaussian_importance_map[gaussian_importance_map != 0])
    else:
        gaussian_importance_map = np.ones(patch_size, dtype=np.float32)

    return gaussian_importance_map

if __name__ == '__main__':
    # mkdir('train_infomation/')
    # <Ircadb和ZDYY混合数据，增加了mask>
    # UNet2021-04-30-15-28-04 | 2021-06-07-15-53-49（只用了ZDYY数据集）|"2021-06-09-15-43-16"（训练集和测试集搞反了）
    # 
    # trainer = Trainer(str_device='cuda:0', 
    #             str_model='UNetV2', in_dim=1, out_dim=2, num_filter=32, patch_size=[128, 128, 64],
    #             str_loss='dice', loss_op={},
    #             learning_rate=1e-4, batch_size=2, num_epoches=200, old_model_name=None, epoches_save=25,
    #             amp_opt='O0')
    # trainer.train()
    # trainer.test(do_gaussian=False, do_mirroring=True)

    # # ACG模块测试2021-05-24-15-45-46
    # trainer = Trainer(str_device='cuda:0', 
    #             str_model='UNetACG', in_dim=1, out_dim=2, num_filter=32, patch_size=[128, 128, 64],
    #             str_loss='dice', loss_op={},
    #             learning_rate=1e-4, batch_size=2, num_epoches=300, old_model_name='2021-05-24-15-45-46', epoches_save=25,
    #             amp_opt='O0')
    # trainer.train()
    # trainer.test(do_gaussian=False, do_mirroring=True)

    # # MSFF模块测试2021-05-25-14-47-11
    # trainer = Trainer(str_device='cuda:0', 
    #             str_model='UNetMSFF', in_dim=1, out_dim=2, num_filter=32, patch_size=[128, 128, 64],
    #             str_loss='dice', loss_op={},
    #             learning_rate=1e-4, batch_size=2, num_epoches=200, old_model_name="2021-06-08-11-04-51", epoches_save=25,
    #             amp_opt='O0')
    # trainer.train()
    # trainer.test(do_gaussian=False, do_mirroring=True)

    # # AGC和MSFF混合测试2021-05-28-17-22-37
    # trainer = Trainer(str_device='cuda:0', 
    #             str_model='UNetAgcMsff', in_dim=1, out_dim=2, num_filter=32, patch_size=[128, 128, 64],
    #             str_loss='dice', loss_op={},
    #             learning_rate=1e-4, batch_size=2, num_epoches=300, old_model_name='2021-05-28-17-22-37', epoches_save=25,
    #             amp_opt='O0')
    # trainer.train()
    # trainer.test(do_gaussian=False, do_mirroring=True)

    # # 4层深度监督UNet（2021-06-18-15-18-57）
    # trainer = Trainer(str_device='cuda:0', 
    #             str_model='UNetV2DS4', in_dim=1, out_dim=2, num_filter=32, patch_size=[128, 128, 64],
    #             str_loss='dice', loss_op={},
    #             learning_rate=1e-4, batch_size=2, num_epoches=200, old_model_name="2021-06-18-15-18-57", epoches_save=25,
    #             amp_opt='O0')
    # trainer.train()
    # trainer.test(do_gaussian=False, do_mirroring=True)
    # pass

    # # UNet dice损失 2021-06-19-11-58-30
    # trainer = Trainer(str_device='cuda:0', 
    #             str_model='UNetV2', in_dim=1, out_dim=2, num_filter=32, patch_size=[128, 128, 64],
    #             str_loss='dice', loss_op={},
    #             learning_rate=1e-4, batch_size=2, num_epoches=200, old_model_name="2021-06-19-11-58-30", epoches_save=25,
    #             amp_opt='O0')
    # trainer.train()
    # trainer.test(do_gaussian=False, do_mirroring=True)

    # UNet 损失函数DcAndCe 2021-06-22-10-29-13
    trainer = Trainer(str_device='cuda:0', 
                str_model='UNetV2', in_dim=1, out_dim=2, num_filter=32, patch_size=[128, 128, 64],
                str_loss='DcAndCe', loss_op={},
                learning_rate=1e-4, batch_size=2, num_epoches=200, old_model_name="2021-06-22-10-29-13", epoches_save=25,
                amp_opt='O0')
    trainer.train()
    trainer.test(do_gaussian=False, do_mirroring=True)
    pass