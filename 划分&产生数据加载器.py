# 加载数据集
import numpy as np
from sklearn.model_selection import StratifiedKFold
import torch
from torch.utils.data import TensorDataset, DataLoader
import threading

# 划分数据集，采用StratifiedKFold，将数据集划分为训练集和验证集，训练集和验证集的两类标签分布应该都大概为1:1
# 创建一个StratifiedKFold对象，设置折叠数为5（训练集和验证集）,种子为1
skf = StratifiedKFold(n_splits=6, shuffle=True, random_state=1)
print('start loading data')
data = np.load('data.npy')
print('loading data done')
labels = np.load('labels.npy')
print('loading labels done')
from GRU import Classifier

# 创建一个模型实例
model = Classifier(46398, 2000)

# 定义损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epoches = 10
print('开始划分数据集')
# 划分数据集
import pickle
for fold, (train_index, val_index) in enumerate(skf.split(data, labels)):
    print("Fold:", fold)
    X_train, X_val = data[train_index], data[val_index]
    y_train, y_val = labels[train_index], labels[val_index]
    
    print("训练集大小:", len(X_train))
    print("验证集大小:", len(X_val))
    print("训练集标签分布:", np.bincount(y_train))
    print("验证集标签分布:", np.bincount(y_val))
    
    # 转换为PyTorch张量
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
    
    # 创建数据集
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # 保存数据加载器
    with open(f"train_loader_fold_{fold}.pkl", "wb") as f:
        pickle.dump(train_loader, f)
    with open(f"val_loader_fold_{fold}.pkl", "wb") as f:
        pickle.dump(val_loader, f)
    
    print('start training')




    