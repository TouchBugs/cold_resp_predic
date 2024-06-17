from sympy import im
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np



# 定义简单的GRU模型
class SimpleGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleGRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)

    
    def forward(self, packed_input):
        packed_output, _ = self.gru(packed_input)
        output = pad_packed_sequence(packed_output, batch_first=True)

        return output, _


# 测试数据加载器和GRU模型
def main():
    # 参数设置
    file_path = '/Data4/gly_wkdir/coldgenepredict/raw_sec/S_italica/分好的数据集csv/二进制GRU/train/train_batch_0.pkl'  # 替换为你的文件路径
    max_len = 50  # 最大序列长度
    vocab_size = 5  # 独热编码的维度 (A, C, G, T)
    input_size = vocab_size  # 输入特征维度
    hidden_size = 128  # 隐藏层维度
    output_size = 2  # 输出类别数量 (假设二分类)
    batch_size = 32  # 批次大小


    # 创建数据集和数据加载器
    import pickle
    with open(file_path, 'rb') as f:
        dataloader = pickle.load(f)
        
        print(len(dataloader))
    model = SimpleGRU(input_size, hidden_size, output_size)
    packed_sequences, lengths, labels = dataloader
    print(lengths)
    print(labels)




if __name__ == "__main__":
    main()
