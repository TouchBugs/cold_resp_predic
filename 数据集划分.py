import pandas as pd
import numpy as np

# 读取CSV文件
df = pd.read_csv(r'/Data4/gly_wkdir/coldgenepredict/raw_sec/S_italica/all_dic.csv',sep=',', header=None)
value_counts = df.iloc[:, 2].value_counts()
print(value_counts[0])
# 提取第三列为0的行
zero_rows = df[df.iloc[:, 2] == 0].sample(n=3531, random_state=42)

# 提取第三列为1的行
one_rows = df.drop(zero_rows.index)

# 存储提取的数据
zero_rows.to_csv('zero_rows.csv', index=False, header=False)
one_rows.to_csv('one_rows.csv', index=False, header=False)

zero_rows_train = zero_rows.sample(n=2942, random_state=42)
# 删除选中的行，得到剩下的行
zero_rows_val = zero_rows.drop(zero_rows_train.index)
one_rows_train = one_rows.sample(n=2150, random_state=42)
one_rows_val = one_rows.drop(one_rows_train.index)

# 存储训练集和验证集
zero_rows_train.to_csv('zero_rows_train.csv', index=False, header=False)
zero_rows_val.to_csv('zero_rows_val.csv', index=False, header=False)
one_rows_train.to_csv('one_rows_train.csv', index=False, header=False)
one_rows_val.to_csv('one_rows_val.csv', index=False, header=False)

# 划分完了
# 训练集2942:2150
# 验证集589:430

# 在one_rows_train.csv中随机选792个行出来，加在它后边
# 在one_rows_val.csv中随机选159个行出来，加在它后边
train_append = one_rows_train.sample(n=792, random_state=42)
one_rows_train = pd.concat([one_rows_train, train_append], ignore_index=True)
one_rows_train.to_csv('one_rows_train_app.csv', index=False, header=False)
# 把train_append加在one_rows_train.csv后边
val_append = one_rows_val.sample(n=159, random_state=42)
one_rows_val = pd.concat([one_rows_val, val_append], ignore_index=True)
one_rows_val.to_csv('one_rows_val_app.csv', index=False, header=False)

# 训练集 zero_rows_train.csv one_rows_train_app.csv 
# 验证集 zero_rows_val.csv one_rows_val_app.csv

