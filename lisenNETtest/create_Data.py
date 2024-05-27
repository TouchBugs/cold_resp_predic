import torch
from torch_geometric.data import Data
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def create(df,max_length)->tuple[Data, torch.Tensor]:
    # 定义基因序列编码字典
    base_to_index = {'A': -2, 'C': -1, 'G': 1, 'T': 2}


    # 定义函数将基因序列编码为所需形式并填充至最长长度
    def encode_sequence(sequence, max_length):
        encoded_seq = []
        for base in sequence:
            if base in base_to_index:
                encoded_seq.append(base_to_index[base])
            else:
                # 如果字符不在字典中，则将其编码为0
                encoded_seq.append(0)
        padding_length = max_length - len(encoded_seq)
        encoded_seq += [0] * padding_length
        return encoded_seq



    # 编码基因序列并填充至最长长度
    df['encoded_sequence'] = df['sequence'].apply(lambda x: encode_sequence(x, max_length))

    # 将编码后的序列转换为PyTorch Tensor
    x = torch.tensor(df['encoded_sequence'].tolist(), dtype=torch.float)
    # print('xshape', x.shape)xshape torch.Size([32, 46398])
    # 将标签转换为PyTorch Tensor
    y_raw = torch.tensor(df['label'].values, dtype=torch.long)
    y = y_raw.unsqueeze(dim=1)
    # print('yshape', y.shape)yshape torch.Size([32, 1])
    # 将填充后的基因序列转换为 numpy 数组
    sequences = np.array(df['encoded_sequence'].values)

    # 计算基因序列之间的相似度矩阵
    print('我正在计算余弦相似度')
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
    print('余弦相似度算完了')
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
    data = Data(x=x, y=y, edge_index=edge_index)

    return data, x