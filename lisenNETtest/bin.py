from os import name
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch.nn import Linear
import matplotlib.pyplot as plt
from tqdm import tqdm
from create_Data import create
import torch.nn.functional as F

class CNN_structure(nn.Module):
    def __init__(self):
        super(CNN_structure, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=10, stride=5, padding=0)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=20, stride=10, padding=0)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=6, stride=3, padding=0)
        self.conv4 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0)
        self.conv5 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=2, stride=1, padding=0)
        self.conv6 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=8, stride=4, padding=0)
        self.conv7 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=16, stride=8, padding=0)
        self.conv8 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=24, stride=12, padding=0)

    def forward(self, x:torch.Tensor)->tuple:
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        x5 = self.conv5(x)
        x6 = self.conv6(x)
        x7 = self.conv7(x)
        x8 = self.conv8(x)
        # print('x1.shape', x1.shape, 'x2.shape', x2.shape, 'x3.shape', x3.shape, 'x4.shape', x4.shape, 'x5.shape', x5.shape, 'x6.shape', x6.shape)
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
        self.dropout7 = nn.Dropout(p=0.7)

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
        output = self.ELU(torch.cat(out, dim=1))
        return output # torch.Size([32, 512, 1])

class GCN_MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GCN_MLP, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim // 2)
        self.fc1 = Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = Linear(hidden_dim // 2, hidden_dim // 4)
        self.fc3 = Linear(hidden_dim // 4, 1)
        self.makeFeature = make_feature()
        self.SeLU = nn.SELU()

    def forward(self, data, data_for_cnn):
        # print(data_for_cnn.shape) # torch.Size([32, 46398])
        data_for_cnn = data_for_cnn.unsqueeze(1) # ([32, 1, 46398])
        CNN_out = self.makeFeature(data_for_cnn).squeeze(2) # ([32, 512])
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        print('图卷积之后的形状', x.shape) # torch.Size([32, 512])
        # 把两个特征拼接在一起
        x = torch.cat((x, CNN_out), dim=1) # torch.Size([32, 1024])
        x = self.SeLU(x)
        x = self.fc1(x)
        x = self.SeLU(x)
        x = self.fc2(x)
        x = self.SeLU(x)
        x = self.fc3(x)
        x = torch.sigmoid(x)
        return x

def test(model, data):
    model.eval()
    out = model(data)
    output_binary = torch.round(out)
    num_same_elements = torch.sum(output_binary == data.y).item()
    acc = num_same_elements / int(len(data.y))
    return acc


if __name__ == '__main__':
    pass
    # from torchsummary import summary
    
    # make = GCN_MLP(46398, 512*2)
    # data = torch.randn([32, 1, 46398])
    # out = make(data)
    # print(out.shape)
    # summary(make, [1, 46398])
