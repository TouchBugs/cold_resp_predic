import pickle
import re
from GRU模型 import bcolors, SimpleGRU
import torch
import time
import matplotlib.pyplot as plt

TheTime = str(time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()))
print(TheTime)

# 设置随机种子
# seed = 3407
# torch.manual_seed(seed)
# =============================================
Thetarget = '冻结GRU排序好的数据注意力'
# =============================================
torch.cuda.set_device(0)
device = torch.device("cuda:0")
# device = torch.device("cpu")

lr = 0.001
weight_decay = 1e-5
epochs = 100

best_acc = 0.7
root_dir = '/Data4/gly_wkdir/coldgenepredict/raw_sec/S_italica/CNN/'
loss_png = root_dir + f'loss-lr{lr}-wd{weight_decay}-ep{epochs}-{TheTime}-{Thetarget}.png'
acc_png = root_dir + f'acc-lr{lr}-wd{weight_decay}-ep{epochs}-{TheTime}-{Thetarget}.png'
precision_png = root_dir + f'precision-lr{lr}-wd{weight_decay}-ep{epochs}-{TheTime}-{Thetarget}.png'
recall_png = root_dir + f'recall-lr{lr}-wd{weight_decay}-ep{epochs}-{TheTime}-{Thetarget}.png'
f1_png = root_dir + f'f1-lr{lr}-wd{weight_decay}-ep{epochs}-{TheTime}-{Thetarget}.png'
data_root = '/Data4/gly_wkdir/coldgenepredict/raw_sec/S_italica/分好的数据集csv/二进制GRU/排序好'
train_data_dir = data_root + '/train/'
val_data_dir = data_root + '/val/'

print('创建模型实例')
model = SimpleGRU().to(device)
print('模型实例创建完成')

# 只对GRU加载预训练参数
print('加载预训练参数')
model.gru.load_state_dict(torch.load(root_dir + 'gru_weight.pth'))

want_save_gru = 0
if want_save_gru:
    # 加载权重
    saved_model_path = '/Data4/gly_wkdir/coldgenepredict/raw_sec/S_italica/CNN/model(1_100-0.8012-0.8157-0.7710-0.7877-0.001-1e-05-2024-06-20-12:04:04--不冻结GRU排序好的数据注意力).pth'
    saved_state_dict = torch.load(saved_model_path)
    # 保存 GRU 的参数
    model.load_state_dict(saved_state_dict)
    torch.save(model.gru.state_dict(), root_dir + 'gru_weight.pth')
    exit()
# 是否冻结 GRU 层的所有权重: 1冻结，0不冻结
freeze_GRU = 1
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

def preprocess(num1s, num0s, f):
    batch = pickle.load(f)
    sequence = batch[0]
    label = batch[2].to(torch.float32)

    label = label.unsqueeze(1)
    num1s += (label == 1).sum().item()
    num0s += (label == 0).sum().item()

    
    return sequence, label, num1s, num0s

def calculate_metrics(outputs, labels):
    predicted = torch.where(outputs >= 0.5, torch.tensor(1.0, dtype=torch.float32).to(device), torch.tensor(0.0, dtype=torch.float32).to(device))

    true_positive = ((predicted == 1) & (labels == 1)).sum().item()
    false_positive = ((predicted == 1) & (labels == 0)).sum().item()
    false_negative = ((predicted == 0) & (labels == 1)).sum().item()

    precision = true_positive / (true_positive + false_positive + 1e-10)
    recall = true_positive / (true_positive + false_negative + 1e-10)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)

    return precision, recall, f1

def record_results(epoch, epoch_loss, right_num, num1s, num0s, outputs, metrics, train=True):
    accuracy = right_num / (num1s + num0s)
    precision, recall, f1 = metrics
    if train:
        train_losses.append(epoch_loss)
        train_accs.append(accuracy)
        train_precisions.append(precision)
        train_recalls.append(recall)
        train_f1s.append(f1)
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
    else:
        val_losses.append(epoch_loss)
        val_accs.append(accuracy)
        val_precisions.append(precision)
        val_recalls.append(recall)
        val_f1s.append(f1)
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}')
        print(f'val_Accuracy: {bcolors.GREEN}{accuracy:.4f}{bcolors.WHITE}')
        print(f'val_Precision: {bcolors.UNDERLINE_Blue}{precision:.4f}{bcolors.WHITE}, \
              val_Recall: {bcolors.UNDERLINE_Yellow}{recall:.4f}{bcolors.WHITE}, \
                val_F1 Score: {bcolors.UNDERLINE_Purple}{f1:.4f}{bcolors.WHITE}')
        print('标签里1的个数: ', num1s)
        print('标签里0的个数: ', num0s)
        print('总数: ', num1s + num0s)
        print('正确的个数: ', right_num)

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

    for i in range(135):
        with open(train_data_dir + 'train_batch_' + str(i) + '.pkl', 'rb') as f:
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

            outputs10 = torch.where(outputs >= 0.5, torch.tensor(1.0, dtype=torch.float32).to(device), torch.tensor(0.0, dtype=torch.float32).to(device))
            right_num += (outputs10 == permuted_label).sum().item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            precision, recall, f1 = calculate_metrics(outputs, permuted_label)
            train_precision += precision
            train_recall += recall
            train_f1 += f1

    train_precision /= 135
    train_recall /= 135
    train_f1 /= 135

    record_results(epoch, epoch_loss, right_num, num1s, num0s, outputs, (train_precision, train_recall, train_f1), train=True)
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

    with torch.no_grad():
        for i in range(27):
            with open(val_data_dir + 'val_batch_' + str(i) + '.pkl', 'rb') as f:
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

                outputs10 = torch.where(outputs >= 0.5, torch.tensor(1.0, dtype=torch.float32).to(device), torch.tensor(0.0, dtype=torch.float32).to(device))
                right_num += (outputs10 == permuted_label).sum().item()

                precision, recall, f1 = calculate_metrics(outputs, permuted_label)
                val_precision += precision
                val_recall += recall
                val_f1 += f1

    val_precision /= 27
    val_recall /= 27
    val_f1 /= 27

    record_results(epoch, val_loss, right_num, num1s, num0s, outputs, (val_precision, val_recall, val_f1), train=False)
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
        torch.save(model.state_dict(), root_dir + f'model({epoch}_{epochs}-{best_acc:.4f}-{val_precision:.4f}-{val_recall:.4f}-{val_f1:.4f}-{lr}-{weight_decay}-{TheTime}--{Thetarget}).pth')


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
