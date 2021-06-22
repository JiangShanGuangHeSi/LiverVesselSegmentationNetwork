# LiverVesselSegmentationNetwork
## 调用结构：
```
preprocess_dcm2nii -> preprocess_nii2csv -> preprocess_csv2spacing ↘  
            augmentation_illumination     ↘                          dataloaderV2_2  
            augmentation_noise            ->     augment_functions ↗                  ↘  
            augmentation_spacing          ↗  
                                                                            loss_xxx   -> trainer  
                                                                            model_xxx  ↗  
```
## 1. 预处理
### preprocess_dcm2nii.py将dcm文件转换为nii.gz文件。
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

### preprocess_nii2csv.py 统计对应文件夹中的nii.gz文件的信息，分配训练集/测试集，生成csv文件。
根据.csv文件，读取数据，得到roi区域，插值保存为npy文件，
并生成包含roi和插值信息的npy文件  
DATASET_CSV_PATH = "data_path_info_mix.csv" # szrch数据集  
XueZhimeng, 2021.4（可能吧）  

添加了几例  
DATASET_CSV_PATH = "data_path_info_mix20210607.csv" # szrch数据集

添加变量B_INTERP，  
B_INTERP = TRUE 任何情况下都进行插值  
B_INTERP = FALSE 检查之前是否有生成插值的结果，如果有就跳过，在添加数据集而不改变插值spacing的时候用  

### preprocess_csv2spacing.py 根据DATASET_CSV_PATH的csv文件中的文件信息加载数据，统计中值spacing并插值，生成插值文件（npy格式）。
需要设置的路径参数为：
```
DATASET_CSV_PATH = "data_path_info_mix20210610.csv"
STATIC_INFO_XLSX_PATH = "static_box_and_spacing.xlsx"# 统计图像信息文件

FOLDER_IMAGE_NPY = "dataset/image/"
FOLDER_MASK_NPY = "dataset/mask/"
FOLDER_LABEL_NPY = "dataset/label/"
FOLDER_IMAGE_INTERPOLATION_NPY = "dataset/image_interpolation/"
FOLDER_MASK_INTERPOLATION_NPY = "dataset/mask_interpolation/"
FOLDER_LABEL_INTERPOLATION_NPY = "dataset/label_interpolation/"
```
输出训练集和测试集文件
```
TRAINSETINFO_FILE_PATH = "dataset/trainset_info_szrch_spacing.npy" # 数据集信息
TESTSETINFO_FILE_PATH = "dataset/testset_info_szrch_spacing.npy"
```

默认生成的插值文件目录为：
```
    dataset/
        image/ 
        image_interpolation/ 
        label/
        label_interpolation/
        mask/
        mask_interpolation/
```
以及TRAINSETINFO_FILE_PATH和TESTSETINFO_FILE_PATH路径下的训练集和数据集文件。可以根据这些文件检查程序正误。  

此外，程序将自动创建DEBUG_ROOT，存储插值后的image, label, mask信息。
```
DEBUG_ROOT = "pic_debug/preprocess/"
```

## 2. dataloader
### dataloaderV2_2.py 读取预处理数据，实现数据随机采样和数据增强，为trainer提供数据。
用于liver vessel CT数据   
运行该程序之前，请成功运行preprocess产生npy插值格式的数据集，并生成TRAINSETINFO_FILE_PATH和TESTSETINFO_FILE_PATH文件（见preprocess_csv2spacing.py）  

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


## 3. data augment数据增强
### augment_functions.py 实现图像和标注的数据增强
#### 增强的功能包括：
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

#### 数据增强的参数设置和初始化：
```
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
```

#### 数据增强的使用形如：
```
image_aug, label_aug = trans.do_augment(image_aug, label)
```

### augmentation_illumination.py 图像亮度增强

### augmentation_noise.py 随机噪声增强

### augmentation_spacing.py 需要坐标变化的增强类型，包括形变、旋转

## 4. trainer 训练器
### trainerV2.py 根据参数，自动生成模型，读取数据，保存文件，打印信息，生成测试结果（为npy格式文件）

Trainer初始化参数
```
str_device: 指定cuda字符串，如'cuda:0'
str_model: 模型名字，请在
    ['UNet', 'UNetSAD', 'UNetV2', 'UNetACG', 'UNetMSFF', 'UNetAgcMsff', 'UNetV2DS4']
    中选择
in_dim: 输入维度
out_dim: 输出维度（类型数量）
num_filter: 第一层滤波器数量，请在[32, 64]中选择
patch_size: 3d体块大小
str_loss: 损失函数，请在['dice', 'bce', 'focal', 'DcAndCe']中选择
loss_op: 损失函数选项，是否自带softmax等等
learning_rate: 学习率【固定学习率，我一直像修正这个选项】
batch_size:
num_epoches: 训练轮数
old_model_name: 如果为None，自动生成一个新的模型，模型信息保存在train_infomation，以时间字符串为标识。
如果为形如"2021-06-22-10-29-13"的时间字符串，则读取这个名字的模型最后保存的checkpoint继续训练。
amp_opt: 正如2021.4所注释，已经被弃用了
XueZhimeng, 2021.6

去掉了混合精度
XueZhimeng, 2021.4

改进模型名字为空训练后预测重新生成文件的问题
添加深度监督模型和对应的label处理
XueZhimeng, 2021.6
```

调用Trainer第一次训练的例子（old_model_name=None）：
```
    trainer = Trainer(str_device='cuda:0', 
                str_model='UNetV2', in_dim=1, out_dim=2, num_filter=32, patch_size=[128, 128, 64],
                str_loss='DcAndCe', loss_op={},
                learning_rate=1e-4, batch_size=2, num_epoches=200, old_model_name=None, epoches_save=25,
                amp_opt='O0')
    trainer.train()
    trainer.test(do_gaussian=False, do_mirroring=True)
```
调用Trainer训练之前训练的模型2021-06-22-10-29-13的例子：
```
    trainer = Trainer(str_device='cuda:0', 
                str_model='UNetV2', in_dim=1, out_dim=2, num_filter=32, patch_size=[128, 128, 64],
                str_loss='DcAndCe', loss_op={},
                learning_rate=1e-4, batch_size=2, num_epoches=200, old_model_name="2021-06-22-10-29-13", epoches_save=25,
                amp_opt='O0')
    trainer.train()
    trainer.test(do_gaussian=False, do_mirroring=True)
```
