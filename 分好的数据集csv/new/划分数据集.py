import pandas as pd
from sklearn.model_selection import train_test_split

# 读取数据
data = pd.read_csv('/Data4/gly_wkdir/coldgenepredict/raw_sec/S_italica/分好的数据集csv/backup/all_dic.csv', header=None)
data.columns = ['id', 'data', 'label', 'nothing']

# 分别提取正类和负类样本
positive_samples = data[data['label'] == 1]
negative_samples = data[data['label'] == 0]
# 查看positive_samples有多少行
print(positive_samples.shape) # (2580, 4) 取2064 516*2=1032 435随机取
print(negative_samples.shape) # (3531, 4) 取2064 1467
# 划分训练集和验证集，保持正负样本的平衡
positive_train, positive_val = train_test_split(positive_samples, train_size=2064, random_state=42)
negative_train, negative_val = train_test_split(negative_samples, train_size=2064, random_state=42)
# 替换函数
def replace_chars(element, r):
    element = str(element)
    start = int(len(element) * r)
    end = start + 10
    return element[:start] + 'N' * (end - start) + element[end:]

# 复制一份posotive_val, 把它的data列的序列的随机几个长度为10的位点修改为N
positive_val_5 = positive_val.copy()
import random
r = random.random()
positive_val_5['data'] = positive_val_5['data'].apply(lambda element: replace_chars(element, r))
# 把positive_val_5合并到positive_val中
positive_val_b = pd.concat([positive_val, positive_val_5])
# 在positive_val_5中随机抽取435个行
positive_val_435 = positive_val.sample(n=435)
positive_val_435['data'] = positive_val_435['data'].apply(lambda element: replace_chars(element, r))
# 把positive_val_435合并到positive_val_b中
positive_val = pd.concat([positive_val_b, positive_val_435])
print(positive_val.shape)
print(negative_val.shape)
print(positive_train.shape)
print(negative_train.shape)
# 合并训练集和验证集
train_data = pd.concat([positive_train, negative_train]).sample(frac=1, random_state=42).reset_index(drop=True)
val_data = pd.concat([positive_val, negative_val]).sample(frac=1, random_state=42).reset_index(drop=True)

# 保存结果到新的csv文件
train_data.to_csv('/Data4/gly_wkdir/coldgenepredict/raw_sec/S_italica/分好的数据集csv/new/train_data.csv', index=False, header=False)
val_data.to_csv('/Data4/gly_wkdir/coldgenepredict/raw_sec/S_italica/分好的数据集csv/new/val_data.csv', index=False, header=False)
