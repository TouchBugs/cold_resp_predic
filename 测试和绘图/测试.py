import pickle
import torch
from GRU模型 import SimpleGRU
import numpy as np
from sklearn.metrics import roc_curve, auc, f1_score
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

def preprocess(f):
    batch = pickle.load(f)
    sequence = batch[0]
    label = batch[2].to(torch.float32)
    label = label.unsqueeze(1)
    return sequence, label

# 定义物种和数据路径
species_data_paths = {
    '高粱': '/Data4/gly_wkdir/coldgenepredict/raw_sec/S_italica/分好的数据集csv/二进制GRU/排序好/sbicolor/',
    '拟南芥': '/Data4/gly_wkdir/coldgenepredict/raw_sec/S_italica/分好的数据集csv/二进制GRU/排序好/At/',
    '大豆': '/Data4/gly_wkdir/coldgenepredict/raw_sec/S_italica/分好的数据集csv/二进制GRU/排序好/Gm',
    '水稻': '/Data4/gly_wkdir/coldgenepredict/raw_sec/S_italica/分好的数据集csv/二进制GRU/排序好/Os'
}
# 定义每个物种的数据批次范围
species_batch_ranges = {
    '高粱': (0, 155),
    '拟南芥': (0, 203),
    '大豆': (0, 225),
    '水稻': (0, 228)
}
f1_scores = {}
roc_data = {}

for species, data_path in species_data_paths.items():
    device = torch.device("cpu")
    model = SimpleGRU(hidden_size2=128, hidden_size3=64).to(device)
    model.load_state_dict(torch.load('/Data4/gly_wkdir/coldgenepredict/raw_sec/S_italica/测试和绘图/model(2_100-0.8047-0.8318-0.7558-0.7877-0.005-0.0001-2024-06-25-10:58:23--1排序128-0.5).pth'))
    criterion = torch.nn.BCELoss()
    preds = []
    labels = []

    with torch.no_grad():
        # 循环处理每个物种
        for species, data_path in species_data_paths.items():
            start_batch, end_batch = species_batch_ranges[species]
            for i in range(start_batch, end_batch):
                with open(data_path + 'data_batch_' + str(i) + '.pkl', 'rb') as f:
                    permuted_sequence, permuted_label = preprocess(f)
                    permuted_sequence, permuted_label = permuted_sequence.to(device), permuted_label.to(device)

                    outputs = model(permuted_sequence)
                    if outputs.isnan().any():
                        continue
                    preds.extend(outputs.squeeze().cpu().numpy())
                    labels.extend(permuted_label.squeeze().cpu().numpy())
                break  # 如果是测试，保留这个break；如果处理全部数据，删除这个break

    binary_preds = [1 if x >= 0.5 else 0 for x in preds]
    f1 = f1_score(labels, binary_preds)
    f1_scores[species] = f1

    fpr, tpr, thresholds = roc_curve(labels, preds)
    roc_auc = auc(fpr, tpr)
    roc_data[species] = (fpr, tpr, roc_auc)

# 绘制ROC曲线
plt.figure(figsize=(10, 8))
for species, (fpr, tpr, roc_auc) in roc_data.items():
    plt.plot(fpr, tpr, lw=2, label=f'{species} ROC curve (area = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('/Data4/gly_wkdir/coldgenepredict/raw_sec/S_italica/测试和绘图/ROC曲线.png')

# 绘制F1精度直方图
plt.figure(figsize=(8, 6))
species = list(f1_scores.keys())
f1_values = list(f1_scores.values())
plt.bar(species, f1_values, color='skyblue')
plt.xlabel('Species')
plt.ylabel('F1 Score')
plt.title('F1 Scores by Species')
plt.savefig('/Data4/gly_wkdir/coldgenepredict/raw_sec/S_italica/测试和绘图/F1精度直方图.png')