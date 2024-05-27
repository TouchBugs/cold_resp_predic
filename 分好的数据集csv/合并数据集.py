import pandas as pd

# 读取两个CSV文件
df1 = pd.read_csv('/Data4/gly_wkdir/coldgenepredict/raw_sec/S_italica/分好的数据集csv/one_rows_train_app.csv', header=None)
df2 = pd.read_csv('/Data4/gly_wkdir/coldgenepredict/raw_sec/S_italica/分好的数据集csv/zero_rows_train.csv', header=None)

# 使用zip函数将两个数据框的行逐行交叉合并
merged_rows = []
for row_a, row_b in zip(df1.values, df2.values):
    merged_rows.append(row_a)
    merged_rows.append(row_b)

# 创建新的DataFrame
merged_df = pd.DataFrame(merged_rows, columns=df1.columns)

# 将合并后的结果保存为新的CSV文件
merged_df.to_csv('/Data4/gly_wkdir/coldgenepredict/raw_sec/S_italica/分好的数据集csv/train.csv', index=False, header=False)
