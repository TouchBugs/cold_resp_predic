import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch.nn import Linear
import torch.nn.functional as F

class bcolors:
    PURPLR = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    WHITE = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    UNDERLINE_Blue = '\033[4;34m'
    UNDERLINE_Yellow = '\033[4;33m'
    UNDERLINE_Purple = '\033[4;35m'
    

# print('\033[4;34m' + '下划线蓝色文本' + '\033[0m')
# print('\033[4;33m' + '下划线黄色文本' + '\033[0m')
# print('\033[4;35m' + '下划线紫色文本' + '\033[0m')


class CNN_structure(nn.Module):
    def __init__(self):
        super(CNN_structure, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=24, stride=12, padding=0)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=20, stride=10, padding=0)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=16, stride=8, padding=0)
        self.conv4 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=10, stride=5, padding=0)
        self.conv5 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=8, stride=4, padding=0)
        self.conv6 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=6, stride=3, padding=0)
        self.conv7 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0)
        self.conv8 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=2, stride=1, padding=0)
        self.dropout = nn.Dropout(p=0.2)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(64)
        self.bn5 = nn.BatchNorm1d(64)
        self.bn6 = nn.BatchNorm1d(64)
        self.bn7 = nn.BatchNorm1d(64)
        self.bn8 = nn.BatchNorm1d(64)

    def forward(self, x:torch.Tensor)->tuple:
        x1 = self.dropout(self.bn1(self.conv1(x)))
        x2 = self.dropout(self.bn2(self.conv2(x)))
        x3 = self.dropout(self.bn3(self.conv3(x)))
        x4 = self.dropout(self.bn4(self.conv4(x)))
        x5 = self.dropout(self.bn5(self.conv5(x)))
        x6 = self.dropout(self.bn6(self.conv6(x)))
        x7 = self.dropout(self.bn7(self.conv7(x)))
        x8 = self.dropout(self.bn8(self.conv8(x)))    
        # print('x1.shape', x1.shape, 'x2.shape', x2.shape, 'x3.shape', x3.shape, 'x4.shape', x4.shape, 'x5.shape', x5.shape, 'x6.shape', x6.shape, 'x7.shape', x7.shape, 'x8.shape', x8.shape)
        return x1, x2, x3, x4, x5, x6, x7, x8

class make_feature(nn.Module):
    def __init__(self):
        super(make_feature, self).__init__()
        self.cnn = CNN_structure()
        self.conv0 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=20, stride=1, padding=0)
        self.conv1 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=5, stride=2, padding=0)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=0)
        self.relu = nn.ReLU()
        self.SeLU = nn.SELU()
        # ELU激活函数
        self.ELU = nn.ELU(inplace=True)
        # Swish激活函数
        self.Swish = nn.Hardswish(inplace=True)
        # torch.dropout
        self.dropout = nn.Dropout(p=0.2)
        # tanh激活函数
        self.tanh = nn.Tanh()
        self.bn1 = nn.BatchNorm1d(8)
        self.bn2 = nn.BatchNorm1d(16)
        self.bn3 = nn.BatchNorm1d(32)

    def forward(self, x):
        x = self.SeLU(self.bn1(self.conv0(x)))
        # print('x0.shape', x.shape)
        # 最大池化
        x = F.max_pool1d(x, kernel_size=4, stride=5)
        # print('池化', x)
        # print('x0.1.shape', x.shape)
        x = self.SeLU(self.bn2(self.conv1(x)))
        # print('x0.2.shape', x.shape)
        x = F.max_pool1d(x, kernel_size=2, stride=2)
        # print('x1.shape', x.shape)
        x = self.Swish(self.bn3(self.conv2(x)))
        # print(x.shape)
        x = F.max_pool1d(x, kernel_size=2, stride=2)
        # print(x.shape)
        # print('x2.shape', x.shape)
        out = []
        for i in self.cnn(x):
            # print('i.shape', i.shape)
            pooled_tensors = []
            for num in range(i.size(1)):
                # 对当前通道进行全局最大池化操作
                pooled_tensor = F.adaptive_max_pool2d(i[:, num:num+1, :], (1, 1))
                # print('pooled_tensor.shape', pooled_tensor.shape)
                pooled_tensors.append(pooled_tensor)
            # 将池化后的结果沿着通道维度拼接在一起
            output_tensor = torch.cat(pooled_tensors, dim=1)
            out.append(output_tensor)
        output = self.tanh(torch.cat(out, dim=1)) # 我希望feature的值都在-1-1之间 tanh
        # print(output.shape)
        return output # torch.Size([32, 512, 1])

class GCN_MLP(nn.Module):
    def __init__(self, hidden_dim):
        super(GCN_MLP, self).__init__()
        
        self.fc2 = Linear(hidden_dim // 2, hidden_dim // 4)
        self.fc3 = Linear(hidden_dim // 4, 1)
        self.makeFeature = make_feature()
        self.SeLU = nn.SELU()
        # tanh激活函数
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=0.5)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(512)
        # sigmoid激活函数
        self.sigmoid = nn.Sigmoid()

    def forward(self, data_for_cnn):
        # print(data_for_cnn.shape) # torch.Size([32, 46398])
        data_for_cnn = data_for_cnn.unsqueeze(1) # ([32, 1, 46398])
        CNN_out = self.makeFeature(data_for_cnn).squeeze(2) # ([32, 512])
        
        x = CNN_out
        x = self.SeLU(x)
        
        x = self.dropout(self.fc2(x))
        x = self.SeLU(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x