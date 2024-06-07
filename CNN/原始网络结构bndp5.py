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

class CNN_structure(nn.Module):
    def __init__(self):
        super(CNN_structure, self).__init__()
        self.conv4 = nn.Conv1d(in_channels=8*4, out_channels=64, kernel_size=10, stride=5, padding=0)
        self.conv8 = nn.Conv1d(in_channels=8*4, out_channels=64, kernel_size=2, stride=1, padding=0)
        self.dropout = nn.Dropout(p=0.02)
        self.bn4 = nn.BatchNorm1d(64)
        self.bn8 = nn.BatchNorm1d(64)

    def forward(self, x:torch.Tensor)->tuple:

        x4 = self.dropout(self.bn4(self.conv4(x)))

        x8 = self.dropout(self.bn8(self.conv8(x)))    
        # print('x1.shape', x1.shape, 'x2.shape', x2.shape, 'x3.shape', x3.shape, 'x4.shape', x4.shape, 'x5.shape', x5.shape, 'x6.shape', x6.shape, 'x7.shape', x7.shape, 'x8.shape', x8.shape)
        # return x1, x2, x3, x4, x5, x6, x7, x8
        return x4, x8

class make_feature(nn.Module):
    def __init__(self):
        super(make_feature, self).__init__()
        self.cnn = CNN_structure()
        self.conv0 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=20, stride=1, padding=0)
        self.conv1 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=5, stride=2, padding=0)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=0)
        self.Relu = nn.ReLU()
        self.Tanh = nn.Tanh()
        # torch.dropout
        self.dropout = nn.Dropout(p=0.2)
        # tanh激活函数
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(16)
        self.bn3 = nn.BatchNorm1d(32)

    def forward(self, x):
        x = self.Relu(self.bn1(self.conv0(x)))

        out = []
        for i in self.cnn(x):
            # print('i.shape', i.shape)([32, 128, 46378])
            output_tensor = F.adaptive_avg_pool1d(i, 1)
            out.append(output_tensor)
        output = self.Tanh(torch.cat(out, dim=1)) # 我希望feature的值都在-1-1之间 tanh
        # print(output.shape)
        return output # torch.Size([32, 512, 1])

class GCN_MLP(nn.Module):
    def __init__(self):
        super(GCN_MLP, self).__init__()
        
        self.fc2 = Linear(128, 64)
        self.fc3 = Linear(64, 1)
        self.makeFeature = make_feature()
        self.SeLU = nn.SELU()
        # tanh激活函数

        self.dropout = nn.Dropout(p=0.01)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(512)
        # sigmoid激活函数
        self.sigmoid = nn.Sigmoid()

    def forward(self, data_for_cnn):
        # print(data_for_cnn.shape) # torch.Size([32, 46398])
        data_for_cnn = data_for_cnn.unsqueeze(1) # ([32, 1, 46398])
        CNN_out = self.makeFeature(data_for_cnn).squeeze(2) # ([32, 512])
        
        x = CNN_out
        x = self.dropout(self.fc2(x))
        x = self.SeLU(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x