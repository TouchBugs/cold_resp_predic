import pickle
from numpy import float32
from DANET import mymodel, bcolors
import torch

# 设置随机种子
# seed = 3407 # https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/2109.08203.pdf
# torch.manual_seed(seed)

# 参数定义
# 保存图片的名字
loss_png = 'loss32.png'
acc_png = 'acc32.png'
# 学习率和L2正则化参数
lr=0.001
weight_decay=1e-4
epochs = 300

# 定义一个最好的精度，val最好精度的模型被保存，没0.8你就别存了
best_acc = 0.8
# 给一个二进制数据集路径
train_data_dir = '/Data4/gly_wkdir/coldgenepredict/raw_sec/S_italica/分好的数据集csv/二进制分批数据集/train/'
val_data_dir = '/Data4/gly_wkdir/coldgenepredict/raw_sec/S_italica/分好的数据集csv/二进制分批数据集/val/'

# sequence-shape torch.Size([32, 46398, 5])
# label-shape torch.Size([32])

# 创建模型实例
print('创建模型实例')
# model = GRU.Classifier_1(input_size=46398)
model = mymodel()
print('模型实例创建完成')

# 定义损失函数和优化器
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
device = torch.device("cuda:2")
cpu = torch.device("cpu")
# 记录train_loss train_acc val_loss val_acc
train_losses = []
train_accs = []
val_losses = []
val_accs = []
# device = torch.device("cpu")

# 训练模型
print('训练模型')

for epoch in range(epochs):
    model.train()
    model = model.to(device)
    epoch_loss = 0
    right_num = 0
    num1s = 0
    for i in range(184):
        with open(train_data_dir+'train_batch_'+str(i)+'.pkl', 'rb') as f:
            batch = pickle.load(f)
            sequence = batch[0].to(torch.float32).to(device)
            label = batch[1].to(torch.float32).to(device)
            num1s += (label == 1).sum().item()
            # print('sequence: ', sequence.shape)
            # print('label: ', label.shape)
            size_0_sequence = sequence.size(0)
            # size_0_label = label.size(0)

            # 生成相同的随机排列索引
            permuted_index = torch.randperm(size_0_sequence)
            # print('train_permuted_index: ', permuted_index)
            # 使用相同的索引对两个张量进行重新排列
            permuted_sequence = sequence[permuted_index]
            permuted_label = label[permuted_index]

            outputs = model(permuted_sequence)
            # print('outputs: ', outputs.shape)
            # print('realoutputs', outputs)
            loss = criterion(outputs, permuted_label)
            epoch_loss += loss.item()

            # outputs = torch.where(outputs > 0, torch.tensor(torch.float32(1)).to(device), torch.tensor(torch.float32(0)).to(device))
            outputs10 = torch.where(outputs > 0, torch.tensor(1.0, dtype=torch.float32).to(device), torch.tensor(0.0, dtype=torch.float32).to(device))

            right_num += (outputs10 == permuted_label).sum().item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    accuracy = right_num / (184 * 32)
    print('标签里1的个数：', num1s)
    print('标签里0的个数：', (184 * 32 - 4) - num1s)
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}')
    print(f'Train Accuracy: {bcolors.FAIL}{accuracy:.4f}{bcolors.ENDC}')
    train_losses.append(epoch_loss)
    train_accs.append(accuracy)
    print('最后一个批次的输出，不满batchsize是正常的:\n',outputs)

    model.eval()
    val_loss = 0
    right_num = 0
    num1s = 0
    for i in range(36):
        with open(val_data_dir+'val_batch_'+str(i)+'.pkl', 'rb') as f:
            batch = pickle.load(f)
            valsequence = batch[0].to(torch.float32).to(device)
            label = batch[1].to(torch.float32).to(device)

            size_0_sequence = valsequence.size(0)
            # size_0_label = label.size(0)

            # 生成相同的随机排列索引
            permuted_index = torch.randperm(size_0_sequence)
            # print('eval_permuted_index: ', permuted_index)
            # 使用相同的索引对两个张量进行重新排列
            permuted_sequence = valsequence[permuted_index]
            permuted_label = label[permuted_index]

            outputs = model(permuted_sequence)
            loss = criterion(outputs, permuted_label)
            val_loss += loss.item()

            # outputs = torch.where(outputs > 0, torch.tensor(torch.float32(1)).to(device), torch.tensor(torch.float32(0)).to(device))
            outputs = torch.where(outputs > 0, torch.tensor(1.0, dtype=torch.float32).to(device), torch.tensor(0.0, dtype=torch.float32).to(device))
            # 计算标签为1的个数
            num1s += (permuted_label == 1).sum().item()
            right_num += (outputs == permuted_label).sum().item()
            
    print('标签里1的个数：', num1s)
    print('标签里0的个数：', (36 * 32) - num1s)
    print(f'Validation Loss: {loss.item():.4f}')
    accuracy = right_num / (36 * 32)
    print(f"Validation Accuracy: {bcolors.OKGREEN}{accuracy:.4f}{bcolors.ENDC}")
    val_losses.append(val_loss)
    val_accs.append(accuracy)

    if accuracy > best_acc:
        best_acc = accuracy
        torch.save(model.state_dict(), f'best_model({epoch}:{accuracy:.4f}).pth')
# 保存模型
# torch.save(model.state_dict(), 'model.pth')

# 根据训练结果绘制loss和acc曲线，分别保存
import matplotlib.pyplot as plt

plt.plot(train_losses, label='train_loss')
plt.plot(val_losses, label='val_loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.savefig(loss_png)
# 清空plot
plt.clf()
plt.plot(train_accs, label='train_acc')
plt.plot(val_accs, label='val_acc')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.savefig(acc_png)

print('trian_loss: ', train_losses)
print('val_loss: ', val_losses)
print('train_acc: ', train_accs)
print('val_acc: ', val_accs)









