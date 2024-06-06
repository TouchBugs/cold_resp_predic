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
        self.conv4 = nn.Conv1d(in_channels=8*4, out_channels=128, kernel_size=10, stride=5, padding=0)
        self.conv8 = nn.Conv1d(in_channels=8*4, out_channels=128, kernel_size=2, stride=1, padding=0)
        self.dropout = nn.Dropout(p=0.2)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn8 = nn.BatchNorm1d(128)

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

    def forward(self, x):
        x = self.SeLU(self.conv0(x))
        # print('x0.shape', x.shape)
        # 最大池化
        x = F.max_pool1d(x, kernel_size=4, stride=5)
        # print('池化', x)
        # print('x0.1.shape', x.shape)
        x = self.SeLU(self.conv1(x))
        # print('x0.2.shape', x.shape)
        x = F.max_pool1d(x, kernel_size=2, stride=2)
        # print('x1.shape', x.shape)
        x = self.Swish(self.conv2(x))
        x = F.max_pool1d(x, kernel_size=2, stride=2)

        # out = []
        # for i in self.cnn(x):
        #     # print('i.shape', i.shape)([32, 128, 46378])
        #     output_tensor = F.adaptive_avg_pool1d(i, 1)
        #     out.append(output_tensor)
        # output = self.Tanh(torch.cat(out, dim=1)) # 我希望feature的值都在-1-1之间 tanh
        # # print(output.shape)
        return x # torch.Size([32, 512, 1])

class GCN_MLP(nn.Module):
    def __init__(self):
        super(GCN_MLP, self).__init__()
        
        self.fc2 = Linear(256, 512)
        self.fc3 = Linear(512, 1)
        self.makeFeature = make_feature()
        self.SeLU = nn.SELU()
        # tanh激活函数

        self.dropout = nn.Dropout(p=0.1)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(512)
        # sigmoid激活函数
        self.sigmoid = nn.Sigmoid()

        # 定义一个embedding层
        self.embedding1 = nn.Embedding(5, 256)
        self.embedding2 = nn.Embedding(2, 256)
        self.transformer = nn.Transformer(d_model=256, batch_first=True)

    def forward(self, data_for_cnn, label):
        # print(data_for_cnn) # torch.Size([32, 46398])
        data_for_cnn = self.embedding1(data_for_cnn.int()) # torch.Size([32, 46398, 256])
        # print(label.shape) # torch.Size([32, 1]
        label = self.embedding2(label.int())

        # 准备transformer
        output = self.transformer(data_for_cnn, label)
        print(output.shape) # torch.Size([32, 46398, 256])
        exit()

        CNN_out = self.makeFeature(data_for_cnn).squeeze(2) # ([32, 512])
        
        x = CNN_out
        x = self.dropout(self.fc2(x))
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x
    

if __name__ == '__main__':
    model = GCN_MLP()
    x = torch.tensor(32*([-1, -2, 0, 1, 2]*9279+[0, 0 ,0])).reshape(32, 46398) + 2
    label = torch.tensor([0, 1]*16).reshape(32, 1)
    y = model(x, label)
    print(y.shape)
    print(y)