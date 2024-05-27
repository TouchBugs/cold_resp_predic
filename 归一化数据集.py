import numpy as np
import torch
"""
这个代码是用来归一化数据集的,定义了一个Dataset类,用于读取数据集,并将其转换为one-hot编码。
然后,使用DataLoader来创建一个数据加载器,用于加载数据集,并将其划分为批次,把每个批次的序列和标签保存下来。
最后,使用pickle模块来保存每个批次的数据。修改第51行和68行
"""
csv_path = r'/Data4/gly_wkdir/coldgenepredict/raw_sec/S_italica/分好的数据集csv/new/'

# 定义Dataset类
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file):
        self.total_length = 46398
        self.data = self.get_sequence_lable(csv_file)
        
    def __getitem__(self, key):
        return self.data[key]
    
    def __len__(self):
        return len(self.data)

    def get_sequence_lable(self, csv_file)->list:
        with open(csv_file, 'r') as f:
            out = []
            for line in f:
                tmp = []
                # 如果line是空的，就跳过
                if line.strip() == '':
                    continue
                seq = self.encode_sequence(line.strip().split(',')[1])
                tmp.append(seq)
                tmp.append(int(line.strip().split(',')[2]))
                out.append(tmp)
            return out

    # 定义一个函数,用于将序列转换为one-hot编码
    def encode_sequence(self, sequence:str)->torch.Tensor:
        # 把sequence: AGCT...用K填充到46398这么长
        # sequence = sequence + 'K' * (self.total_length - len(sequence))
        encoding = torch.tensor([
            [-2] if base == 'A' else
            [-1] if base == 'C' else
            [1] if base == 'G' else
            [2] if base == 'T' else
            [0] if base == 'N' else
            [0] for base in sequence
        ],dtype=torch.float32)
        # 把encoding用[0, 0, 0, 0, 0]填充到46398这么长
        encoding = torch.cat([encoding, torch.zeros(self.total_length - len(encoding), 1, dtype=torch.float32)])
        return encoding
        
from torch.utils.data import DataLoader
# 读取zero_rows_val.csv 调代码的假的数据集.csv
data_t = MyDataset(csv_path + 'train_data.csv')
data_v = MyDataset(csv_path + 'val_data.csv')
# data = MyDataset(csv_path + '调代码的假的数据集.csv')

# print(data[0])
# 创建DataLoader
dataloader_t = DataLoader(data_t, batch_size=32, shuffle=False)
dataloader_v = DataLoader(data_v, batch_size=32, shuffle=False)
# 打印第一个batch的数据
for i in  enumerate(dataloader_t):
    print('batch: ',i[0]+1) 
    print(i[1][0].shape)
    print(i[1][1].shape)
    print(i[1][0])
    print(i[1][1])
    break
for i in  enumerate(dataloader_v):
    print('batch: ',i[0]+1) 
    print(i[1][0].shape)
    print(i[1][1].shape)
    print(i[1][0])
    print(i[1][1])
    break
#由于归一化数据集的时间太长,我想直接把它保存下来直接读取
import pickle
# 把每个批次的数据单独保存
for i, batch in enumerate(dataloader_t):
    with open(f'/Data4/gly_wkdir/coldgenepredict/raw_sec/S_italica/分好的数据集csv/new/二进制一位分批数据集/train/train_batch_{i}.pkl', 'wb') as f:
        pickle.dump(batch, f)
for i, batch in enumerate(dataloader_v):
    with open(f'/Data4/gly_wkdir/coldgenepredict/raw_sec/S_italica/分好的数据集csv/new/二进制一位分批数据集/val/val_batch_{i}.pkl', 'wb') as f:
        pickle.dump(batch, f)
# # 读取每个批次的数据
# for i in range(len(dataloader)):
#     with open(f'/Data4/gly_wkdir/coldgenepredict/raw_sec/S_italica/分好的数据集csv/test_batch_{i}.pkl', 'rb') as f:
#         batch = pickle.load(f)
#         # 测试
#         print('batch[0].shape',batch[0].shape)
#         print('batch[0].shape',batch[1].shape)




