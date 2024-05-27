import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch.nn import Linear
import matplotlib.pyplot as plt
from tqdm import tqdm
from create_Data import create
from bin import GCN_MLP, test

# 读取CSV文件
print('正在读取CSV文件...')
df_raw_train = pd.read_csv('/Data4/gly_wkdir/coldgenepredict/raw_sec/S_italica/分好的数据集csv/train.csv', header=None)
df_raw_val=pd.read_csv('/Data4/gly_wkdir/coldgenepredict/raw_sec/S_italica/分好的数据集csv/val.csv', header=None)
df_raw_train.columns = ['geneid', 'sequence', 'label', 'lens']
df_raw_val.columns = ['geneid', 'sequence', 'label', 'lens']
df_raw_train['lens'] = 0
df_raw_val['lens'] = 0
torch.cuda.set_device(0)
device = torch.device('cuda:0')
sample_size = 64 # 选择的行数--------------------------------------
test_acc_list = [] # 存储测试集上的准确率
train_acc_list = [] # 存储训练集上的准确率
print('读取csv文件完成, 计算最长基因序列长度')
# 计算最长基因序列长度
max_length_train = max(df_raw_train['sequence'].apply(len))
max_length_test = max(df_raw_val['sequence'].apply(len))
if max_length_train >= max_length_test:
    max_length = max_length_train
else:
    max_length = max_length_test
# max_length 是所有数据中最长的长度46398
print('最长基因序列长度为：', max_length)
print('把test数据放到device上')
# df_test=create(df_raw_val,max_length).to(device) # -----------------------
print('把test数据放到device上...完成')

print('定义GCN模型')
model_train=GCN_MLP(max_length,512*2).to(device)
print('模型定义完成')
print('定义优化器和损失函数')
optimizer = torch.optim.Adam(model_train.parameters(), lr=0.01, weight_decay=1e-4)
loss_function = torch.nn.BCELoss().to(device)
print('开始训练')
# from torchsummary import summary
# summary(model_train, (46398, 5))

for i in tqdm(range(1500)):
    print(f'对train数据按照sample_size={sample_size}采样')
    df_train_csv = df_raw_train.sample(n=sample_size)
    print('把train数据放到device上')
    df_train, train_cnn=create(df_train_csv,max_length)  # -----------------------
    df_train = df_train.to(device) # type: ignore
    train_cnn = train_cnn.to(device)
    print('训练数据输入模型...', end='..')
    output = model_train(df_train, train_cnn)
    print('完成前向传播')
    out_train=torch.round(output)
    num=torch.sum(out_train==df_train.y).item()
    acc_train=num/int(sample_size)
    print('训练精度: ',acc_train)
    train_acc_list.append(acc_train)

    print('计算loss')
    loss = loss_function(output.float(), df_train.y.float()) # type: ignore
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print('loss: ', loss.item())
    # validation

    # test_acc = test(model_train, df_test)
    # print('验证精度: ',test_acc)
    # test_acc_list.append(test_acc)

    # if test_acc >0.9:
    #     # 保存模型参数
    #     torch.save(model_train.state_dict(),'model.pth')
    #     print("模型参数已保存")

plt.plot(train_acc_list, label='Train Accuracy')
plt.plot(test_acc_list, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()  # 添加图例
plt.savefig('accuracy_plot.png')