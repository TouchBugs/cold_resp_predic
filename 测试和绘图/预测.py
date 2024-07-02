import pickle
from sympy import sequence
import torch
from zmq import device
from GRU模型 import SimpleGRU
import numpy as np
from sklearn.metrics import roc_curve, auc, f1_score
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib


model = SimpleGRU(hidden_size2=128, hidden_size3=64)
model.load_state_dict(torch.load('/Data4/gly_wkdir/coldgenepredict/raw_sec/S_italica/测试和绘图/Si_model(2_100-0.8047-0.8318-0.7558-0.7877-0.005-0.0001-2024-06-25-10:58:23--1排序128-0.5).pth', map_location='cpu'))
model.to('cuda:0')
sequence = 'AGCTAGAGCTAGAGCTAGAGCTAGAGCTAGAGCTAGAGCTAGAGCTAGAGCTAGAGCTAGAGCTAGAGCTAGAGCTAGAGCTAGAGCTAGAGCTAGAGCTAGAGCTAGAGCTAGAGCTAGAGCTAGAGCTAGAGCTAGAGCTAGAGCTAGAGCTAGAGCTAGAGCTAGAGCTAGAGCTAGAGCTAGAGCTAGAGCTAGAGCTAGAGCTAGAGCTAG'
# 把sequence，onehot编码
# A[1,0,0,0,0]
# G[0,1,0,0,0]
# C[0,0,1,0,0]
# T[0,0,0,1,0]
# N[0,0,0,0,1]
# 一共5个维度
permuted_sequence = []
for i in sequence:
    if i == 'A':
        permuted_sequence.append([1,0,0,0,0])
    elif i == 'G':
        permuted_sequence.append([0,1,0,0,0])
    elif i == 'C':
        permuted_sequence.append([0,0,1,0,0])
    elif i == 'T':
        permuted_sequence.append([0,0,0,1,0])
    elif i == 'N':
        permuted_sequence.append([0,0,0,0,1])

permuted_sequence = torch.tensor(permuted_sequence, dtype=torch.float32).unsqueeze(0).to('cuda:0')
print(permuted_sequence.shape)
outputs = model(permuted_sequence).cpu()
print(outputs)
