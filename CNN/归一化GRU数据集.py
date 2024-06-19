import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

# 独热编码函数
def one_hot_encode(sequence, max_len, vocab_size):
    encoding_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
    one_hot_encoded = np.zeros((max_len, vocab_size), dtype=int)

    for i, base in enumerate(sequence):
        if i >= max_len:
            break
        index = encoding_map.get(base, -1)
        if index != -1:
            one_hot_encoded[i, index] = 1

    return one_hot_encoded

# 自定义数据集类
class GeneDataset(Dataset):
    def __init__(self, file_path, max_len, vocab_size):
        self.data = pd.read_csv(file_path, header=None)
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.data['encoded_sequence'] = self.data[1].apply(lambda seq: one_hot_encode(seq, max_len, vocab_size)) # type: ignore
        self.data['length'] = self.data[1].apply(len)
        self.labels = torch.tensor(self.data[2].values, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        encoded_sequence = self.data.iloc[idx]['encoded_sequence']
        length = self.data.iloc[idx]['length']
        label = self.labels[idx]
        return torch.tensor(encoded_sequence, dtype=torch.float32), torch.tensor(length), label

# 生成数据加载器的批处理函数
def collate_fn(batch):
    sequences, lengths, labels = zip(*batch)

    # 转换为tensor
    sequences = [torch.tensor(seq) for seq in sequences]
    lengths = torch.tensor(lengths, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)

    # 填充序列
    padded_sequences = pad_sequence(sequences, batch_first=True)

    # 打包序列
    packed_sequences = pack_padded_sequence(padded_sequences, lengths, batch_first=True, enforce_sorted=False)
    return packed_sequences, lengths, labels


# 测试数据加载器和GRU模型
def main():
    # 参数设置
    file_path = '/Data4/gly_wkdir/coldgenepredict/raw_sec/S_italica/分好的数据集csv/sorted_valid.csv'  
    max_len = 46398  # 最大序列长度
    vocab_size = 5  # 独热编码的维度 (A, C, G, T, N)
 
    batch_size = 32  # 批次大小


    # 创建数据集和数据加载器
    dataset = GeneDataset(file_path, max_len, vocab_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # 把dataloaders里的每个批次的数据存起来
    import pickle
    for i, batch in enumerate(dataloader):
        with open(f'/Data4/gly_wkdir/coldgenepredict/raw_sec/S_italica/分好的数据集csv/二进制GRU/排序好/val/val_batch_{i}.pkl', 'wb') as f:
            pickle.dump(batch, f)



if __name__ == "__main__":
    main()
