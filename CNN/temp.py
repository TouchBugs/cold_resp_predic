# 导入所需的库
import csv

# 定义文件名列表
filenames = ['wrong_sequence1.csv', 'wrong_sequence2.csv', 'wrong_sequence3.csv', 'wrong_sequence4.csv']

# 使用字典来存储每个文件的第一列数据，以及一个计数器来跟踪每个数据出现的次数
first_column_data = {}

# 遍历每个文件
for filename in filenames:
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        line_count = 0
        for row in reader:
            # 只处理前100行
            if line_count < 100:
                # 只考虑第一列数据
                first_column_value = row[0]
                if first_column_value in first_column_data:
                    first_column_data[first_column_value] += 1
                else:
                    first_column_data[first_column_value] = 1
                line_count += 1
            else:
                break

# 找出在所有文件中都出现的第一列数据
common_first_column_values = [value for value, count in first_column_data.items() if count == len(filenames)]

# 输出这些共有的第一列数据
for value in common_first_column_values:
    print(value)