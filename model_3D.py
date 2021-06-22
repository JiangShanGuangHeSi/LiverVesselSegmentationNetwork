'''
# 3D-UNet model.

MiaoJuzheng 2020.05
'''
# x: 128x128 resolution
import torch
import torch.nn as nn
# from losses import SoftDiceLoss
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def conv_block_3d(in_dim, out_dim, activation):   # 单个三维卷积
    return nn.Sequential(
        nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim),
        activation,)


def conv_trans_block_3d(in_dim, out_dim):
    return nn.Sequential(
        nn.ConvTranspose3d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),)


def max_pooling_3d():
    return nn.MaxPool3d(kernel_size=2, stride=2, padding=0)


def conv_block_2_3d(in_dim, out_dim, activation):   # 连续两次卷积
    return nn.Sequential(
        conv_block_3d(in_dim, out_dim//2, activation),
        conv_block_3d(out_dim//2, out_dim, activation),)

def conv_block_2_3d_up(in_dim, out_dim, activation):   # 连续两次卷积
    return nn.Sequential(
        conv_block_3d(in_dim, out_dim, activation),
        conv_block_3d(out_dim, out_dim, activation),)

class UNet(nn.Module):
    def __init__(self, in_dim, out_dim, num_filters):
        super(UNet, self).__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_filters = num_filters
        # activation = nn.LeakyReLU(0.2, inplace=True)
        activation = nn.ReLU(inplace=True)
        
        # Down sampling
        self.down_1 = conv_block_2_3d(self.in_dim, self.num_filters, activation)
        self.pool_1 = max_pooling_3d()
        self.down_2 = conv_block_2_3d(self.num_filters, self.num_filters * 2, activation)
        self.pool_2 = max_pooling_3d()
        self.down_3 = conv_block_2_3d(self.num_filters * 2, self.num_filters * 4, activation)
        self.pool_3 = max_pooling_3d()
        self.down_4 = conv_block_2_3d(self.num_filters * 4, self.num_filters * 8, activation)
        
        # Bridge
        # self.bridge = conv_block_2_3d(self.num_filters * 16, self.num_filters * 32, activation)
        
        # Up sampling
        self.trans_1 = conv_trans_block_3d(self.num_filters * 8, self.num_filters * 8)
        self.up_1 = conv_block_2_3d_up(self.num_filters * 12, self.num_filters * 4, activation)
        self.trans_2 = conv_trans_block_3d(self.num_filters * 4, self.num_filters * 4)
        self.up_2 = conv_block_2_3d_up(self.num_filters * 6, self.num_filters * 2, activation)
        self.trans_3 = conv_trans_block_3d(self.num_filters * 2, self.num_filters * 2)
        self.up_3 = conv_block_2_3d_up(self.num_filters * 3, self.num_filters, activation)        
        # Output
        self.out = nn.Conv3d(self.num_filters, out_dim, kernel_size=1, stride=1)
    
    def forward(self, x):
        # Down sampling
        down_1 = self.down_1(x) # -> [1, 64, 128, 128, 128]
        pool_1 = self.pool_1(down_1) # -> [1, 64, 64, 64, 64]
        
        down_2 = self.down_2(pool_1) # -> [1, 128, 64, 64, 64]
        pool_2 = self.pool_2(down_2) # -> [1, 128, 32, 32, 32]
        
        down_3 = self.down_3(pool_2) # -> [1, 256, 32, 32, 32]
        pool_3 = self.pool_3(down_3) # -> [1, 256, 16, 16, 16]
        
        down_4 = self.down_4(pool_3) # -> [1, 512, 16, 16, 16]
        
        # Up sampling
        trans_1 = self.trans_1(down_4) # -> [1, 512, 32, 32, 32]
        concat_1 = torch.cat([trans_1, down_3], dim=1) # -> [1, 768, 32, 32, 32]
        up_1 = self.up_1(concat_1) # -> [1, 256, 32, 32, 32]
        
        trans_2 = self.trans_2(up_1) # -> [1, 256, 64, 64, 64]
        concat_2 = torch.cat([trans_2, down_2], dim=1) # -> [1, 384, 64, 64, 64]
        up_2 = self.up_2(concat_2) # -> [1, 128, 64, 64, 64]
        
        trans_3 = self.trans_3(up_2) # -> [1, 128, 128, 128, 128]
        concat_3 = torch.cat([trans_3, down_1], dim=1) # -> [1, 192, 128, 128, 128]
        up_3 = self.up_3(concat_3) # -> [1, 64, 128, 128, 128]
        
        # Output
        out = self.out(up_3) # -> [1, 1, 128, 128, 128]
        # out = torch.softmax(out, dim=1)
        return out

# if __name__ == "__main__":
#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#     image_size = 96
#     x = torch.Tensor(2, 1, image_size, image_size, image_size)
#     x = x.to(device)
#     print("x size: {}".format(x.size()))
    
#     model = UNet(in_dim=1, out_dim=1, num_filters=64).to(device)
#     y = torch.randn(2, 1, image_size, image_size, image_size)
#     y = y.to(device)
#     criterion = SoftDiceLoss()
#     out = model(x)
#     loss = criterion(out, y)
#     loss.backward()
#     print("out size: {}".format(out.size()))