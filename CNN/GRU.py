from ast import arg
import csv
import pickle
import re
import argparse
from sre_constants import GROUPREF_UNI_IGNORE
from ipykernel import write_connection_file
from requests import get
import yaml
from GRU模型 import bcolors, SimpleGRU
import torch
import time
import matplotlib.pyplot as plt
# 创建 ArgumentParser 对象
parser = argparse.ArgumentParser(description='Process some integers.')

TheTime = str(time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()))
print(TheTime)


# 添加参数 -lr 、weight_decay、 freeze_GRU、threathhold, hidden_size2, hidden_size3和 -epoch
# 参数分别指的是学习率、权重衰减、是否冻结GRU层、阈值、隐藏层2的大小、隐藏层3的大小和训练的轮数
parser.add_argument('-lr', type=float, default=0.003, help='learning rate')
parser.add_argument('-weight_decay', type=float, default=1e-5, help='weight decay')
parser.add_argument('-freeze_GRU', type=int, default=0, help='freeze GRU layer or not')
parser.add_argument('-threathhold', type=float, default=0.4, help='threathhold')
parser.add_argument('-hidden_size2', type=int, default=128, help='hidden size 2: 128->Hidden_size2->Hidden_size3->1')
parser.add_argument('-hidden_size3', type=int, default=64, help='hidden size 3: 128->Hidden_size2->Hidden_size3->1')
parser.add_argument('-epoch', type=int, default=100, help='number of epochs')

# 设置随机种子
# seed = 3407
# torch.manual_seed(seed)

# 超参数
# 解析参数
args = parser.parse_args()
# =============================================
# 是否冻结 GRU 层的所有权重: 1冻结，0不冻结
freeze_GRU = args.freeze_GRU # 冻结的效果更好
lr = args.lr
weight_decay = args.weight_decay
epochs = args.epoch
hidden_size2 = args.hidden_size2
hidden_size3 = args.hidden_size3
threathhold = args.threathhold

# =============================================

Thetarget = f'{freeze_GRU}排序128-{threathhold}'
print(Thetarget)

best_acc = 0.5
root_dir = '/Data4/gly_wkdir/coldgenepredict/raw_sec/S_italica/CNN/'
loss_png = root_dir + f'loss-lr{lr}-wd{weight_decay}-ep{epochs}-{TheTime}-{Thetarget}.png'
acc_png = root_dir + f'acc-lr{lr}-wd{weight_decay}-ep{epochs}-{TheTime}-{Thetarget}.png'
precision_png = root_dir + f'precision-lr{lr}-wd{weight_decay}-ep{epochs}-{TheTime}-{Thetarget}.png'
recall_png = root_dir + f'recall-lr{lr}-wd{weight_decay}-ep{epochs}-{TheTime}-{Thetarget}.png'
f1_png = root_dir + f'f1-lr{lr}-wd{weight_decay}-ep{epochs}-{TheTime}-{Thetarget}.png'
roc_png = root_dir + f'roc-lr{lr}-wd{weight_decay}-ep{epochs}-{TheTime}-{Thetarget}.png'
data_root = '/Data4/gly_wkdir/coldgenepredict/raw_sec/S_italica/分好的数据集csv/二进制GRU/排序好/Os/'
train_data_dir = data_root
val_data_dir = data_root

# device = torch.cuda.set_device(0)
# 设置GPU
device = torch.device("cuda:0")
# device = torch.device()
# device = torch.device("cpu")
print('创建模型实例')
model = SimpleGRU(hidden_size2=hidden_size2, hidden_size3=hidden_size3).to(device)
print('模型实例创建完成')

# 只对GRU加载预训练参数
print('加载预训练参数')
model.load_state_dict(torch.load("/Data4/gly_wkdir/coldgenepredict/raw_sec/S_italica/CNN/model(87_100-0.7218-0.9164-0.9592-0.9322-0.001-1e-05-2024-06-30-19:53:07--0排序128-0.4).pth"))

want_save_gru = 0
if want_save_gru:
    # 加载权重
    saved_model_path = '/Data4/gly_wkdir/coldgenepredict/raw_sec/S_italica/CNN/比较好的权重/model(94_100-0.8070-0.8332-0.7655-0.7935-0.005-0.001-2024-06-25-22:32:45--1排序128-0.4).pth'
    saved_state_dict = torch.load(saved_model_path)
    # 保存 GRU 的参数
    model.load_state_dict(saved_state_dict)
    torch.save(model.gru.state_dict(), root_dir + 'gru_weight.pth')
    exit()

if freeze_GRU:
    for param in model.gru.parameters():
        param.requires_grad = False
print('预训练参数加载完成')

criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, eps=1e-10)
train_losses = []
train_accs = []
val_losses = []
val_accs = []
train_precisions = []
train_recalls = []
train_f1s = []
val_precisions = []
val_recalls = []
val_f1s = []
train_rocs = []
val_rocs = []

from sklearn.metrics import roc_curve, auc
import numpy as np

def compute_roc(outputs, labels):
    """
    计算ROC曲线的数据。
    :param outputs: 模型的输出，形状为[N, 1]，其中N是样本数量。
    :param labels: 真实标签，形状与outputs相同。
    :return: 返回FPR, TPR和对应的AUC值。
    """
    # 将输出和标签转换为一维numpy数组
    outputs = outputs.squeeze().detach().cpu().numpy()
    labels = labels.squeeze().detach().cpu().numpy()
    
    # 计算FPR, TPR和阈值
    fpr, tpr, thresholds = roc_curve(labels, outputs)
    # 计算AUC
    roc_auc = auc(fpr, tpr)
    
    return fpr, tpr, roc_auc

#定义一个字典，记录不正确序列的批次i以及其在output中的位置，及其错误次数{(i, locat): count}
wrong_sequence = {}

def get_wrong_sequence(outputs, labels, i):
    # print(outputs.shape) # ([32, 1])
    # print(labels.shape)
    outputs = outputs.squeeze(1)
    labels = labels.squeeze(1)

    for j in range(outputs.shape[0]):
        if (outputs[j] < threathhold and labels[j] == 1) or (outputs[j] >= threathhold and labels[j] == 0):
            if (i, j) in wrong_sequence:
                wrong_sequence[(i, j)] += 1
            else:
                wrong_sequence[(i, j)] = 1
    # 打印出wrong_sequence的最后一个元素
    # print(list(wrong_sequence.items())[-1])


def preprocess(num1s, num0s, f):
    batch = pickle.load(f)
    sequence = batch[0]
    label = batch[2].to(torch.float32)

    label = label.unsqueeze(1)
    num1s += (label == 1).sum().item()
    num0s += (label == 0).sum().item()


    return sequence, label, num1s, num0s

def calculate_metrics(outputs, labels):
    predicted = torch.where(outputs >= threathhold, torch.tensor(1.0, dtype=torch.float32).cpu(), torch.tensor(0.0, dtype=torch.float32).cpu())

    true_positive = ((predicted == 1) & (labels == 1)).sum().item()
    false_positive = ((predicted == 1) & (labels == 0)).sum().item()
    false_negative = ((predicted == 0) & (labels == 1)).sum().item()

    precision = true_positive / (true_positive + false_positive + 1e-10)
    recall = true_positive / (true_positive + false_negative + 1e-10)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)

    return precision, recall, f1

def record_results(epoch, epoch_loss, right_num, num1s, num0s, outputs, metrics, roc, train=True):
    accuracy = right_num / (num1s + num0s)
    precision, recall, f1 = metrics
    if train:
        train_losses.append(epoch_loss)
        train_accs.append(accuracy)
        train_precisions.append(precision)
        train_recalls.append(recall)
        train_f1s.append(f1)
        train_rocs.append(roc)
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}')
        print(f'train_Accuracy: {bcolors.RED}{accuracy:.4f}{bcolors.WHITE}')
        print(f'train_Precision: {bcolors.UNDERLINE_Blue}{precision:.4f}{bcolors.WHITE}, \
              train_Recall: {bcolors.UNDERLINE_Yellow}{recall:.4f}{bcolors.WHITE}, \
                train_F1 Score: {bcolors.UNDERLINE_Purple}{f1:.4f}{bcolors.WHITE}')
        print('标签里1的个数: ', num1s)
        print('标签里0的个数: ', num0s)
        print('总数: ', num1s + num0s)
        print('正确的个数: ', right_num)
        print('最后一个批次的输出, 不满batchsize是正常的:\n', outputs)
        print(f'Epoch [{epoch+1}/{epochs}], T_AUC: {bcolors.UNDERLINE}{roc:.4f}{bcolors.WHITE}')
    else:
        val_losses.append(epoch_loss)
        val_accs.append(accuracy)
        val_precisions.append(precision)
        val_recalls.append(recall)
        val_f1s.append(f1)
        val_rocs.append(roc)
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}')
        print(f'val_Accuracy: {bcolors.GREEN}{accuracy:.4f}{bcolors.WHITE}')
        print(f'val_Precision: {bcolors.UNDERLINE_Blue}{precision:.4f}{bcolors.WHITE}, \
              val_Recall: {bcolors.UNDERLINE_Yellow}{recall:.4f}{bcolors.WHITE}, \
                val_F1 Score: {bcolors.UNDERLINE_Purple}{f1:.4f}{bcolors.WHITE}')
        print('标签里1的个数: ', num1s)
        print('标签里0的个数: ', num0s)
        print('总数: ', num1s + num0s)
        print('正确的个数: ', right_num)
        print(f'Epoch [{epoch+1}/{epochs}], V_AUC: {bcolors.UNDERLINE}{roc:.4f}{bcolors.WHITE}')

# 记录最近4个epoch的训练损失
recent_train_losses = []
recent_val_losses = []
recent_val_accs = []
recent_train_accs = []
lr_increase_factor = 1.1
lr_decrease_factor = 0.9  # 学习率减小的因子
tolerance = 0.01  # 容忍度

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    right_num = 0
    num1s = 0
    num0s = 0
    train_precision = 0
    train_recall = 0
    train_f1 = 0
    trian_roc = 0

    for i in range(180):
        with open(train_data_dir + 'data_batch_' + str(i) + '.pkl', 'rb') as f:
            permuted_sequence, permuted_label, num1s, num0s = preprocess(num1s, num0s, f)
            permuted_sequence, permuted_label = permuted_sequence.to(device), permuted_label.to(device)

            outputs = model(permuted_sequence)
            if outputs.isnan().any():
                print('hidden:', outputs)
                # raise ValueError('训练时模型输出存在NaN, 模型觉得很nan！')
                print('训练时模型输出存在NaN, 模型觉得很nan！')
                continue
            loss = criterion(outputs, permuted_label)
            # print(loss)
            epoch_loss += loss.item()

            outputs10 = torch.where(outputs >= threathhold, torch.tensor(1.0, dtype=torch.float32).cpu(), torch.tensor(0.0, dtype=torch.float32).cpu())
            right_num += (outputs10 == permuted_label).sum().item()
            outputs10 = outputs10.cpu()
            outputs = outputs.cpu()
            permuted_label = permuted_label.cpu()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            
            precision, recall, f1 = calculate_metrics(outputs, permuted_label)
            train_precision += precision
            train_recall += recall
            train_f1 += f1
            fpr, tpr, roc_auc = compute_roc(outputs10, permuted_label)
            trian_roc += roc_auc
            # 把没用的内存释放掉，把不用的变量删除
            #del permuted_sequence, permuted_label, outputs, outputs10, loss, precision, recall, f1, fpr, tpr, roc_auc

    train_precision /= 180
    train_recall /= 180
    train_f1 /= 180
    trian_roc /= 180

    record_results(epoch, epoch_loss, right_num, num1s, num0s, outputs, (train_precision, train_recall, train_f1), trian_roc, train=True)
    recent_train_losses.append(epoch_loss)
    recent_train_accs.append(right_num / (num1s + num0s))

    model.eval()
    val_loss = 0
    right_num = 0
    num1s = 0
    num0s = 0
    val_precision = 0
    val_recall = 0
    val_f1 = 0
    val_roc = 0

    with torch.no_grad():
        for i in range(180,227):
            with open(val_data_dir + 'data_batch_' + str(i) + '.pkl', 'rb') as f:
                permuted_sequence, permuted_label, num1s, num0s = preprocess(num1s, num0s, f)
                permuted_sequence, permuted_label = permuted_sequence.to(device), permuted_label.to(device)

                outputs = model(permuted_sequence)
                if outputs.isnan().any():
                    print('hidden:', outputs)
                    # raise ValueError('验证时模型输出存在NaN, 模型觉得很nan！')
                    print('验证时模型输出存在NaN, 模型觉得很nan！')
                    continue
                loss = criterion(outputs, permuted_label)
                val_loss += loss.item()

                outputs10 = torch.where(outputs >= threathhold, torch.tensor(1.0, dtype=torch.float32).cpu(), torch.tensor(0.0, dtype=torch.float32).cpu())
                outputs10 = outputs10.cpu()
                outputs = outputs.cpu()
                permuted_label = permuted_label.cpu()
                right_num += (outputs10 == permuted_label).sum().item()

                precision, recall, f1 = calculate_metrics(outputs, permuted_label)
                # 写一个函数记录不正确的序列
                get_wrong_sequence(outputs, permuted_label, i)

                # 在验证阶段计算ROC
                fpr, tpr, roc_auc = compute_roc(outputs10, permuted_label)
                
                val_precision += precision
                val_recall += recall
                val_f1 += f1
                val_roc += roc_auc

    val_precision /= 227-180
    val_recall /= 227-180
    val_f1 /= 227-180
    val_roc /= 227-180

    record_results(epoch, val_loss, right_num, num1s, num0s, outputs, (val_precision, val_recall, val_f1), val_roc, train=False)
    recent_val_losses.append(val_loss)
    recent_val_accs.append(right_num / (num1s + num0s))
    
    # # 增大和减小学习率的判断
    # if len(recent_train_losses) > 4 and len(recent_val_losses) > 4 and len(recent_val_accs) > 4:
    #     recent_train_losses.pop(0)
    #     recent_val_losses.pop(0)
    #     recent_val_accs.pop(0)

    
    #     if (recent_train_losses[-1] > recent_train_losses[0] * (1 - tolerance)) and \
    #        (recent_val_losses[-1] > recent_val_losses[0] * (1 - tolerance)) and \
    #        (recent_val_accs[-1] < recent_val_accs[0] * (1 + tolerance) and \
    #         recent_train_accs[-1] <= 0.85):
    #         lr *= lr_increase_factor # lr_increase_factor = 1.05
    #         for param_group in optimizer.param_groups:
    #             param_group['lr'] = lr
    #         print(f"{bcolors.UNDERLINE}Learning rate {bcolors.RED}increased{bcolors.UNDERLINE} to {lr:.6f}{bcolors.WHITE}")

    #     # 减小学习率的条件: 每十个周期就减小一次学习率
    #     if (epoch + 1) % 10 == 0:
    #         lr *= lr_decrease_factor # lr_decrease_factor = 0.9
    #         for param_group in optimizer.param_groups:
    #             param_group['lr'] = lr
    #         print(f"{bcolors.UNDERLINE}Learning rate {bcolors.GREEN}decreased{bcolors.UNDERLINE} to {lr:.6f}{bcolors.WHITE}")

    if val_accs[-1] > best_acc:
        best_acc = val_accs[-1]
        torch.save(model.state_dict(), root_dir + f'Os-model({epoch}_{epochs}-{best_acc:.4f}-{val_precision:.4f}-{val_recall:.4f}-{val_f1:.4f}-{lr}-{weight_decay}-{TheTime}--{Thetarget}).pth')


def plot_metric(train_metric, val_metric, metric_name, y_label, save_path):
    plt.plot(train_metric, label=f'train_{metric_name}')
    plt.plot(val_metric, label=f'val_{metric_name}')
    plt.xlabel('epoch')
    plt.ylabel(y_label)
    plt.legend()
    plt.savefig(save_path)
    plt.clf()

plot_metric(train_losses, val_losses, 'loss', 'loss', loss_png)
plot_metric(train_accs, val_accs, 'accuracy', 'accuracy', acc_png)
plot_metric(train_precisions, val_precisions, 'precision', 'precision', precision_png)
plot_metric(train_recalls, val_recalls, 'recall', 'recall', recall_png)
plot_metric(train_f1s, val_f1s, 'f1', 'F1-score', f1_png)
plot_metric(train_rocs, val_rocs, 'roc', 'AUC', roc_png)

# 把wrong_sequence按照count大小排序,大的在前面
wrong_sequence = dict(sorted(wrong_sequence.items(), key=lambda x: x[1], reverse=True))
# 把wrong_sequence（{[i, locat]: count}）写入文件csv，格式为：第i批第locat个, count
with open(root_dir + 'wrong_sequence.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['i', 'locat', 'count'])
    for key, value in wrong_sequence.items():
        writer.writerow([key[0], key[1], value])
# 取出csv前五条数据, 格式为：第{i}批第{locat}个, {count}
with open(root_dir + 'wrong_sequence.csv', 'r') as f:
    reader = csv.reader(f)
    mail = ''
    for i, row in enumerate(reader):
        if i == 0:
            continue
        if i > 5:
            break
        print(f'第{row[0]}批第{row[1]}个, {row[2]}')
        mail += f'第{row[0]}批第{row[1]}个, {row[2]}\n'

import yagmail
# 把超参数都写入邮件
mail += '\n'
mail += f'lr: {lr}\nweight_decay: {weight_decay}\nfreeze_GRU: {freeze_GRU}\nthreathhold: {threathhold}\nhidden_size2: {hidden_size2}\nhidden_size3: {hidden_size3}\nepoch: {epochs}\n'
def send_email(subject, body):
    # 1439389719
    qq = 2196692208
    if len(str(qq))!=10:
        raise ValueError('qq号码长度不对')
    receiver = str(qq) +'@qq.com'  # 接收方邮箱
    yag = yagmail.SMTP(user='2196692208@qq.com', host='smtp.qq.com', port=465, smtp_ssl=True) 
    yag.send(to=receiver, subject=subject, contents=[body, yagmail.inline(loss_png), yagmail.inline(acc_png), yagmail.inline(precision_png), yagmail.inline(recall_png), yagmail.inline(f1_png), yagmail.inline(roc_png)])
    print('send email successfully')
send_email('程序跑完了', '模型训练完了.\n' + TheTime + '\n' + Thetarget + mail)
