
import pickle
import torch
from torch_geometric.data import Data
import torch_geometric.data
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# 参数定义
torch.cuda.set_device(0)

databasename = '二进制一位分批数据集'
# 保存图片的名字
# 给一个二进制数据集路径
train_data_dir = f'/Data4/gly_wkdir/coldgenepredict/raw_sec/S_italica/分好的数据集csv/new/二进制一位分批数据集/train/'
val_data_dir = f'/Data4/gly_wkdir/coldgenepredict/raw_sec/S_italica/分好的数据集csv/new/二进制一位分批数据集/val/'
# 给一个二进制Data数据集保存路径
train_save_dir = f'/Data4/gly_wkdir/coldgenepredict/raw_sec/S_italica/分好的数据集csv/new/二进制一位分批数据集/Data/32/train/'
val_save_dir = f'/Data4/gly_wkdir/coldgenepredict/raw_sec/S_italica/分好的数据集csv/new/二进制一位分批数据集/Data/32/val/'

def get_graph_data(sequence, q):
    sequences = np.array(permuted_sequence)
    # 计算基因序列之间的相似度矩阵
    # print(f'我正在计算余弦相似度-{q}')
    # similarity_matrix = cosine_similarity(sequences, sequences)
    similarity_matrix = np.zeros((len(sequences), len(sequences)))
    for i in range(len(sequences)):
        for j in range(i + 1, len(sequences)):
            seq1 = sequences[i]
            seq2 = sequences[j]
            # 直接计算余弦相似度
            
            similarity = cosine_similarity([seq1], [seq2])[0][0] # type: ignore 反正能跑
            similarity_matrix[i][j] = similarity
            similarity_matrix[j][i] = similarity
    # print('余弦相似度算完了')
    # 根据阈值确定边的存在
    threshold = 0.001  # 设定一个阈值-------------------------------------超参数
    edge_index = []
    for i in range(len(sequences)):
        for j in range(i + 1, len(sequences)):
            if similarity_matrix[i][j] >= threshold:
                edge_index.append([i, j])

    edge_index = np.array(edge_index).T  # 转换为二维数组
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    # 创建PyG的Data对象
    data = Data(x=sequence, y=label, edge_index=edge_index) # type: ignore
    return data

for epoch in range(1):

    epoch_loss = 0
    right_num = 0
    num1s = 0

    # 用于合并sequence和label的list
    sequence_list = []
    label_list = []

    for i in range(129):
        with open(train_data_dir+'train_batch_'+str(i)+'.pkl', 'rb') as f:
            batch = pickle.load(f)
            sequence = batch[0].to(torch.float32)
            label = batch[1].to(torch.float32)
            
            # print('sequence: ', sequence.shape) # torch.Size([64, 46398, 1])
            # print('label: ', label.shape) # torch.Size([64])
            sequence = sequence.squeeze(2) # torch.Size([64, 46398])
            label = label.unsqueeze(1) # torch.Size([64, 1])
            num1s += (label == 1).sum().item()
            # print('sequence: ', sequence.shape)
            # print('label: ', label.shape)
            size_0_sequence = sequence.size(0)
            # size_0_label = label.size(0)

            # 生成相同的随机排列索引
            permuted_index = torch.randperm(size_0_sequence)
            # print('train_permuted_index: ', permuted_index)
            # 使用相同的索引对两个张量进行重新排列
            permuted_sequence = sequence[permuted_index]
            permuted_label = label[permuted_index]

            # --------------------------------------------------------------------------------------
            data = get_graph_data(permuted_sequence, i)
            # --------------------------------------------------------------------------------------
            # 将Data对象保存到文件中
            with open(train_save_dir+'train_batch_'+str(i)+'.pkl', 'wb') as f:
                pickle.dump(data, f)

    for i in range(92):
        with open(val_data_dir+'val_batch_'+str(i)+'.pkl', 'rb') as f:
            batch = pickle.load(f)
            valsequence = batch[0].to(torch.float32)
            label = batch[1].to(torch.float32)

            size_0_sequence = valsequence.size(0)
            # size_0_label = label.size(0)

            # 生成相同的随机排列索引
            permuted_index = torch.randperm(size_0_sequence)
            # print('eval_permuted_index: ', permuted_index)
            # 使用相同的索引对两个张量进行重新排列
            permuted_sequence = valsequence[permuted_index]
            permuted_label = label[permuted_index]
            permuted_sequence = permuted_sequence.squeeze(2) # torch.Size([64, 46398])
            permuted_label = permuted_label.unsqueeze(1) # torch.Size([64, 1])
            # --------------------------------------------------------------------------------------
            data = get_graph_data(permuted_sequence, i)
            # --------------------------------------------------------------------------------------
            with open(val_save_dir+'val_batch_'+str(i)+'.pkl', 'wb') as f:
                pickle.dump(data, f)

