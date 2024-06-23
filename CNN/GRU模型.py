from pandas import isna
from 原始网络结构bndp5 import bcolors
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch.nn import Linear
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


bcolors = bcolors

class SimpleGRU(nn.Module): # 128->128->64->1
    def __init__(self, input_size=5, hidden_size1=128, hidden_size2=128, hidden_size3=64, hidden_size4=32, output_size=1):
        super(SimpleGRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size1, batch_first=True, num_layers=2, bidirectional=True)# 双向GRU
        self.fc1 = nn.Linear(hidden_size1, hidden_size2)
        self.fc2 = nn.Linear(hidden_size2, hidden_size3)
        self.fc3 = nn.Linear(hidden_size3, output_size)

        self.BN1 = nn.BatchNorm1d(hidden_size1)
        self.BN2 = nn.BatchNorm1d(hidden_size2)
        self.BN3 = nn.BatchNorm1d(hidden_size3)
        self.BN4 = nn.BatchNorm1d(hidden_size4)

    def forward(self, packed_input):
        packed_output, hidden = self.gru(packed_input)
        # output = pad_packed_sequence(packed_output, batch_first=True)
        # print(hidden.size()) # torch.Size([4, 32, 128]) 32批次，4个方向，128个隐藏单元
        # 此处增加注意力机制会使得效果不好
        hidden_0 = F.adaptive_avg_pool1d(hidden.permute(1, 2, 0), 1).squeeze(2)

        hidden = self.fc1(hidden_0)
        hidden = self.BN2(hidden)  # 批量归一化层
        hidden = torch.relu(hidden)  # 将ReLU激活函数放在批量归一化之后
        hidden = self.fc2(hidden)

        # print(hidden.shape)
        hidden = torch.relu(hidden)
        hidden = self.BN3(hidden)  # 批量归一化层
        hidden = self.fc3(hidden)
        hidden = torch.sigmoid(hidden)
        # print(hidden.shape)
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
