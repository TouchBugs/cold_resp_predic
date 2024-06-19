from pandas import isna
from 原始网络结构bndp5 import bcolors
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch.nn import Linear
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


bcolors = bcolors

class SimpleGRU(nn.Module):
    def __init__(self, input_size=5, hidden_size1=128, hidden_size2=64):
        super(SimpleGRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size1, batch_first=True, num_layers=2, bidirectional=True)# 双向GRU
        self.fc1 = nn.Linear(hidden_size1, hidden_size2)
        self.fc2 = nn.Linear(hidden_size2, 1)
        self.BN128 = nn.BatchNorm1d(hidden_size1)
        self.BN64 = nn.BatchNorm1d(hidden_size2)

    def forward(self, packed_input):
        packed_output, hidden = self.gru(packed_input)
        # output = pad_packed_sequence(packed_output, batch_first=True)
        # print(hidden.size()) # torch.Size([4, 32, 128]) 32批次，4个方向，128个隐藏单元
        # 增加权重对四个输出进行选择
        weights = torch.nn.Parameter(torch.randn(4), requires_grad=True)
        attention_weights = F.softmax(weights, dim=0)
        # 扩展权重维度以便与输入张量相乘
        attention_weights_expanded = attention_weights.view(4, 1, 1)  # 变成 (4, 1, 1)
        # 加权相乘并求和得到最终的 (32, 128) 输出
        hidden = torch.sum(attention_weights_expanded * input, dim=0)  # 加权求和得到 (32, 128)

        hidden = self.BN128(hidden)
        hidden = self.fc1(hidden)
        hidden = torch.sigmoid(hidden)
        hidden = self.fc2(hidden)
        hidden = torch.sigmoid(hidden)
        
        return hidden

if __name__ == '__main__':
    
    device = torch.device("cuda:0")
    model = SimpleGRU().to(device)
    import pickle
    for i in range(133,134):
        with open(f"/Data4/gly_wkdir/coldgenepredict/raw_sec/S_italica/分好的数据集csv/二进制GRU/train/train_batch_{i}.pkl", 'rb') as f:
            dataloader = pickle.load(f)
            # print(dataloader[0])
            permuted_sequence = dataloader[0].to(device)
            labels = dataloader[2].unsqueeze(1).to(torch.float32).to(device)
            out = model(permuted_sequence)
            # print(out)
