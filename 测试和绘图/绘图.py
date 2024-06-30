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
#     # '高粱': '/Data4/gly_wkdir/coldgenepredict/raw_sec/S_italica/分好的数据集csv/二进制GRU/排序好/sbicolor/',
#     # '拟南芥': '/Data4/gly_wkdir/coldgenepredict/raw_sec/S_italica/分好的数据集csv/二进制GRU/排序好/At/',
#     # '大豆': '/Data4/gly_wkdir/coldgenepredict/raw_sec/S_italica/分好的数据集csv/二进制GRU/排序好/Gm/',
#     # '水稻': '/Data4/gly_wkdir/coldgenepredict/raw_sec/S_italica/分好的数据集csv/二进制GRU/排序好/Os/'
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
# f1_scores = {}
# roc_data = {}

# print("开始处理数据...")

# for species, data_path in species_data_paths.items():
#     print(f"正在处理物种：{species}")
#     model = SimpleGRU(hidden_size2=128, hidden_size3=64)
#     model.load_state_dict(torch.load('/Data4/gly_wkdir/coldgenepredict/raw_sec/S_italica/测试和绘图/model(2_100-0.8047-0.8318-0.7558-0.7877-0.005-0.0001-2024-06-25-10:58:23--1排序128-0.5).pth', map_location='cpu'))
#     model.to('cuda:0')
#     preds = []
#     labels = []

#     with torch.no_grad():
#         # 循环处理每个物种
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

#                 outputs = model(permuted_sequence).cpu()
#                 if outputs.isnan().any():
#                     continue
#                 preds.extend(outputs.squeeze().cpu().numpy())
#                 labels.extend(permuted_label.squeeze().cpu().numpy())
#             # break  # 如果是测试，保留这个break；如果处理全部数据，删除这个break

#     binary_preds = [1 if x >= 0.5 else 0 for x in preds]
#     f1 = f1_score(labels, binary_preds)
#     f1_scores[species] = f1

#     fpr, tpr, thresholds = roc_curve(labels, preds)
#     roc_auc = auc(fpr, tpr)
#     roc_data[species] = (fpr, tpr, roc_auc)
# # 把计算出来的ROC曲线和F1精度保存到文件中
# with open('/Data4/gly_wkdir/coldgenepredict/raw_sec/S_italica/测试和绘图/roc_data小米.pkl', 'wb') as f:
#     pickle.dump(roc_data, f)
# with open('/Data4/gly_wkdir/coldgenepredict/raw_sec/S_italica/测试和绘图/f1_scores小米.pkl', 'wb') as f:
#     pickle.dump(f1_scores, f)
# exit()
# 读取保存的ROC曲线和F1精度
with open('/Data4/gly_wkdir/coldgenepredict/raw_sec/S_italica/测试和绘图/roc_data.pkl', 'rb') as f:
    roc_data = pickle.load(f)
with open('/Data4/gly_wkdir/coldgenepredict/raw_sec/S_italica/测试和绘图/f1_scores.pkl', 'rb') as f:
    f1_scores = pickle.load(f)
with open('/Data4/gly_wkdir/coldgenepredict/raw_sec/S_italica/测试和绘图/roc_data小米.pkl', 'rb') as f:
    roc_data_add = pickle.load(f)
with open('/Data4/gly_wkdir/coldgenepredict/raw_sec/S_italica/测试和绘图/f1_scores小米.pkl', 'rb') as f:
    f1_scores_add = pickle.load(f)
roc_data['小米-train'] = roc_data_add['小米-train']
roc_data['小米-val'] = roc_data_add['小米-val']
f1_scores['小米-train'] = f1_scores_add['小米-train']
f1_scores['小米-val'] = f1_scores_add['小米-val']


print("数据处理完成，开始绘制ROC曲线...")
# 指定字体路径
font_path = '/Data4/gly_wkdir/coldgenepredict/raw_sec/S_italica/测试和绘图/NotoSansMonoCJKsc-Regular.otf'
font_prop = FontProperties(fname=font_path)

# # 绘制ROC曲线
# plt.figure(figsize=(10, 8))
# for species, (fpr, tpr, roc_auc) in roc_data.items():
#     plt.plot(fpr, tpr, lw=2, label=f'{species} ROC curve (area = {roc_auc:.2f})')
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate', fontproperties=font_prop)
# plt.ylabel('True Positive Rate', fontproperties=font_prop)
# plt.title('Receiver Operating Characteristic', fontproperties=font_prop)
# # 只在这里调用plt.legend()，确保所有设置都在一个地方统一
# plt.legend(loc="lower right", prop=font_prop, fontsize=24)  # 使用prop设置字体属性，同时直接指定fontsize覆盖字体大小
# plt.savefig('/Data4/gly_wkdir/coldgenepredict/raw_sec/S_italica/测试和绘图/ROC曲线.png', dpi=300)
# print("ROC曲线绘制完成，开始绘制F1精度直方图...")

#plt.figure(figsize=(8, 6), dpi=300)
species = list(f1_scores.keys())
f1_values = list(f1_scores.values())

al = 1
# 自定义每个条形的颜色
colors = [
    (0,194/255,163/255, al),  # lightcoral，透明度50%
    (133/255,159/255,205/255, al),  # lightgreen，透明度50%
    (255/255,139/255,196/255, al), # lightskyblue，透明度50%
    (140/255,216/255,66/255, al),  # lightcyan，透明度50%
    (255/255,217/255,0, al), # violet，透明度50%
    (239/255,194/255,146/255, al), # lightyellow，透明度50%
    (0.75, 0.75, 0.75, al), # grey，透明度50%
    (0.96, 0.96, 0.96, al)  # whitesmoke，透明度50%
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
plt.savefig('/Data4/gly_wkdir/coldgenepredict/raw_sec/S_italica/测试和绘图/F1精度直方图.png', dpi=300, bbox_inches='tight')

print("所有处理和绘图任务完成。")