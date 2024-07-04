import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import pickle

# 加载数据
# 指定字体路径
font_path = '/Data4/gly_wkdir/coldgenepredict/raw_sec/S_italica/测试和绘图/NotoSansMonoCJKsc-Regular.otf'
font_prop = FontProperties(fname=font_path)

roc_data_paths = [
    '/Data4/gly_wkdir/coldgenepredict/raw_sec/S_italica/测试和绘图/roc_dataS_I.pkl',
    '/Data4/gly_wkdir/coldgenepredict/raw_sec/S_italica/测试和绘图/roc_dataAt.pkl',
    '/Data4/gly_wkdir/coldgenepredict/raw_sec/S_italica/测试和绘图/roc_dataGm.pkl',
    '/Data4/gly_wkdir/coldgenepredict/raw_sec/S_italica/测试和绘图/roc_dataOs.pkl',
    '/Data4/gly_wkdir/coldgenepredict/raw_sec/S_italica/测试和绘图/roc_dataSb.pkl'
]
f1_scores_paths = [
    '/Data4/gly_wkdir/coldgenepredict/raw_sec/S_italica/测试和绘图/f1_scoresS_I.pkl',
    '/Data4/gly_wkdir/coldgenepredict/raw_sec/S_italica/测试和绘图/f1_scoresAt.pkl',
    '/Data4/gly_wkdir/coldgenepredict/raw_sec/S_italica/测试和绘图/f1_scoresGm.pkl',
    '/Data4/gly_wkdir/coldgenepredict/raw_sec/S_italica/测试和绘图/f1_scoresOs.pkl',
    '/Data4/gly_wkdir/coldgenepredict/raw_sec/S_italica/测试和绘图/f1_scoresSb.pkl'
]
species_names = ['小米', '拟南芥', '大豆', '水稻', '高粱']

# 绘制ROC曲线
plt.figure(figsize=(15, 10))
linestyles = ['-', '--', '-.', ':', (0, (3, 5, 1, 5))]  # 最后一个是自定义的点划线
plt.figure(figsize=(15, 10))
for i, path in enumerate(roc_data_paths):
    with open(path, 'rb') as f:
        roc_data = pickle.load(f)
    plt.subplot(2, 3, i+1)
    for j, (species, (fpr, tpr, roc_auc)) in enumerate(roc_data.items()):
        plt.plot(fpr, tpr, lw=2, label=f'{species} (AUC = {roc_auc:.2f})', linestyle=linestyles[j % len(linestyles)])
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.title(species_names[i], fontproperties=font_prop)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.legend(loc="lower right", prop=font_prop, fontsize=18)
plt.tight_layout()
plt.savefig('/Data4/gly_wkdir/coldgenepredict/raw_sec/S_italica/测试和绘图/ROC曲线合并虚线.png', dpi=300, bbox_inches='tight')

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
# 绘制F1分数直方图
plt.figure(figsize=(15, 4))
for i, path in enumerate(f1_scores_paths):
    with open(path, 'rb') as f:
        f1_scores = pickle.load(f)
    plt.subplot(1, 5, i+1)
    species = list(f1_scores.keys())
    f1_values = list(f1_scores.values())
    plt.bar(species, f1_values, color=colors[:len(species)], edgecolor='black', width=1.0)
    plt.title(species_names[i], fontproperties=font_prop)
    plt.ylim([0.0, 1.0])
    plt.xlabel('Species', fontproperties=font_prop)
    plt.ylabel('F1 Score', fontproperties=font_prop)
    # 增加网格线
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=1, fontproperties=font_prop)

plt.tight_layout()
plt.savefig('/Data4/gly_wkdir/coldgenepredict/raw_sec/S_italica/测试和绘图/F1分数直方图合并.png', dpi=300, bbox_inches='tight')