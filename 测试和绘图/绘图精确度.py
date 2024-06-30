import pickle
import torch
from zmq import device
from GRU模型 import SimpleGRU
import numpy as np
from sklearn.metrics import roc_curve, auc, f1_score
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib

# def preprocess(f):
#     batch = pickle.load(f)
#     sequence = batch[0]
#     label = batch[2].to(torch.float32)
#     label = label.unsqueeze(1)
#     return sequence, label

# # 定义物种和数据路径
# species_data_paths = {
#     '高粱': '/Data4/gly_wkdir/coldgenepredict/raw_sec/S_italica/分好的数据集csv/二进制GRU/排序好/sbicolor/',
#     '拟南芥': '/Data4/gly_wkdir/coldgenepredict/raw_sec/S_italica/分好的数据集csv/二进制GRU/排序好/At/',
#     '大豆': '/Data4/gly_wkdir/coldgenepredict/raw_sec/S_italica/分好的数据集csv/二进制GRU/排序好/Gm/',
#     '水稻': '/Data4/gly_wkdir/coldgenepredict/raw_sec/S_italica/分好的数据集csv/二进制GRU/排序好/Os/',
#     '小米-train': '/Data4/gly_wkdir/coldgenepredict/raw_sec/S_italica/分好的数据集csv/二进制GRU/排序好/train/',
#     '小米-val': '/Data4/gly_wkdir/coldgenepredict/raw_sec/S_italica/分好的数据集csv/二进制GRU/排序好/val/'
# }
# # 定义每个物种的数据批次范围
# species_batch_ranges = {
#     '高粱': (0, 155),
#     '拟南芥': (0, 203),
#     '大豆': (0, 225),
#     '水稻': (0, 228),
#     '小米-train': (0, 135),
#     '小米-val': (0, 27)
# }

# from sklearn.metrics import accuracy_score

# accuracy_scores = {}  # 用于存储每个物种的准确率

# for species, data_path in species_data_paths.items():
#     print(f"正在处理物种：{species}")
#     model = SimpleGRU(hidden_size2=128, hidden_size3=64)
#     model.load_state_dict(torch.load('/Data4/gly_wkdir/coldgenepredict/raw_sec/S_italica/测试和绘图/model(2_100-0.8047-0.8318-0.7558-0.7877-0.005-0.0001-2024-06-25-10:58:23--1排序128-0.5).pth', map_location='cpu'))
#     model.to('cuda:0')
#     preds = []
#     labels = []

#     with torch.no_grad():
#         start_batch, end_batch = species_batch_ranges[species]
#         for i in range(start_batch, end_batch):
#             print(f"正在处理{species}的数据批次：{i}")
#             a = 'data_batch_'
#             if species == '小米-train':
#                 a = 'train_batch_'
#             elif species == '小米-val':
#                 a = 'val_batch_'
#             with open(data_path + a + str(i) + '.pkl', 'rb') as f:
#                 permuted_sequence, permuted_label = preprocess(f)
#                 permuted_sequence, permuted_label = permuted_sequence.to('cuda:0'), permuted_label.to('cpu')

#                 outputs = model(permuted_sequence)
#                 if outputs.isnan().any():
#                     continue
#                 preds.extend(outputs.squeeze().cpu().numpy())
#                 labels.extend(permuted_label.squeeze().cpu().numpy())

#     binary_preds = [1 if x >= 0.5 else 0 for x in preds]
#     accuracy = accuracy_score(labels, binary_preds)
#     accuracy_scores[species] = accuracy

# # 打印每个物种的准确率
# for species, accuracy in accuracy_scores.items():
#     print(f"{species}的准确率为：{accuracy:.4f}")

# # 保存准确率数据到文件
# with open('/Data4/gly_wkdir/coldgenepredict/raw_sec/S_italica/测试和绘图/accuracy_scores.pkl', 'wb') as f:
#     pickle.dump(accuracy_scores, f)

# 读取准确率数据
with open('/Data4/gly_wkdir/coldgenepredict/raw_sec/S_italica/测试和绘图/accuracy_scores.pkl', 'rb') as f:
    accuracy_scores = pickle.load(f)
# 指定字体路径
font_path = '/Data4/gly_wkdir/coldgenepredict/raw_sec/S_italica/测试和绘图/NotoSansMonoCJKsc-Regular.otf'
font_prop = FontProperties(fname=font_path)


plt.figure(figsize=(8, 6), dpi=300)
species = list(accuracy_scores.keys())
f1_values = list(accuracy_scores.values())

al = 1
# 自定义每个条形的颜色
colors = [
    (0,194/255,163/255, al),  
    (133/255,159/255,205/255, al),  
    (255/255,139/255,196/255, al), 
    (140/255,216/255,66/255, al), 
    (255/255,217/255,0, al), 
    (239/255,194/255,146/255, al), 
    (0.75, 0.75, 0.75, al),
    (0.96, 0.96, 0.96, al) 
]
# 确保颜色列表长度与条形数量相等
assert len(colors) >= len(species), "颜色列表长度小于条形数量，请添加更多颜色。"

# 绘制直方图，设置间距为0，进行黑色描边
plt.bar(species, f1_values, color=colors[:len(species)], edgecolor='black', width=1.0)  # width=1.0表示条形之间没有间距

# 设置标签和标题
plt.xlabel('Species', fontproperties=font_prop)
plt.ylabel('F1 Score', fontproperties=font_prop)
plt.title('F1 Scores by Species', fontproperties=font_prop)

# 设置x轴标签旋转和字体
plt.xticks(rotation=1, fontproperties=font_prop)

# 增加网格线
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 保存图像
plt.savefig('/Data4/gly_wkdir/coldgenepredict/raw_sec/S_italica/测试和绘图/准确度直方图.png', dpi=300, bbox_inches='tight')

print("所有处理和绘图任务完成。")


