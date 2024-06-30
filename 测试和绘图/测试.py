import pickle
import torch
from GRU模型 import bcolors, SimpleGRU
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

data_root = '/Data4/gly_wkdir/coldgenepredict/raw_sec/S_italica/分好的数据集csv/二进制GRU/排序好'
val_data_dir = data_root + '/sbicolor/'
# 设置GPU
device = torch.device("cuda:0")
print('创建模型实例')

model = SimpleGRU(hidden_size2=128, hidden_size3=64).to(device)
model.load_state_dict(torch.load('/Data4/gly_wkdir/coldgenepredict/raw_sec/S_italica/测试和绘图/model(2_100-0.8047-0.8318-0.7558-0.7877-0.005-0.0001-2024-06-25-10:58:23--1排序128-0.5).pth'))
criterion = torch.nn.BCELoss()
# 假设preds和labels分别存储了所有的预测概率和真实标签
preds = []
labels = []

with torch.no_grad():
    for i in range(135,290):
        with open(val_data_dir + 'train_batch_' + str(i) + '.pkl', 'rb') as f:
            permuted_sequence, permuted_label= preprocess(f)
            permuted_sequence, permuted_label = permuted_sequence.to(device), permuted_label.to(device)

            outputs = model(permuted_sequence)
            if outputs.isnan().any():
                print('验证时模型输出存在NaN, 模型觉得很nan！')
                continue
            preds.extend(outputs.squeeze().cpu().numpy())
            labels.extend(permuted_label.squeeze().cpu().numpy())
        break

# 计算F1精度
binary_preds = [1 if x >= 0.5 else 0 for x in preds]
f1 = f1_score(labels, binary_preds)
print(f'F1 Score: {f1}')

# 计算ROC曲线和AUC
fpr, tpr, thresholds = roc_curve(labels, preds)
roc_auc = auc(fpr, tpr)


# 指定字体路径
font_path = '/Data4/gly_wkdir/coldgenepredict/raw_sec/S_italica/测试和绘图/NotoSansMonoCJKsc-Regular.otf'
font_prop = FontProperties(fname=font_path)

fpr, tpr, thresholds = roc_curve(labels, preds)
roc_auc = auc(fpr, tpr)
# font_prop = {'family': 'sans-serif', 'size': 12}
# 绘制ROC曲线
plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel(f'False Positive Rate', fontproperties=font_prop)
plt.text(0.5, -0.15, '高粱', ha='center', color='red', fontproperties=font_prop)
# 调整图表边界，确保文本可见
plt.subplots_adjust(bottom=0.15)
# plt.xlabel('高粱', color='red', fontproperties=font_prop)  # 使用指定字体显示中文
plt.ylabel('True Positive Rate', fontproperties=font_prop)
plt.title('Receiver Operating Characteristic', fontproperties=font_prop)
plt.legend(loc="lower right", prop=font_prop)
# 保存ROC曲线
plt.savefig('/Data4/gly_wkdir/coldgenepredict/raw_sec/S_italica/测试和绘图/ROC曲线.png')