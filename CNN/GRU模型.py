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
        self.gru = nn.GRU(input_size, hidden_size1, batch_first=True)
        self.fc1 = nn.Linear(hidden_size1, hidden_size2)
        self.fc2 = nn.Linear(hidden_size2, 1)
    
    def forward(self, packed_input):
        packed_output, hidden = self.gru(packed_input)
        # output = pad_packed_sequence(packed_output, batch_first=True)
        # print(hidden.size()) # torch.Size([1, 32, 128])
        hidden = hidden.squeeze(0)

        hidden = self.fc1(hidden)
        hidden = self.fc2(hidden)
        hidden = torch.sigmoid(hidden)
        return hidden
