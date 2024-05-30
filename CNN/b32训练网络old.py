import pickle
from 原始网络结构bndp5 import GCN_MLP, bcolors
import torch
import time
import matplotlib.pyplot as plt

TheTime = str(time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()))
print(TheTime)

# 设置随机种子
# seed = 3407
# torch.manual_seed(seed)
# =============================================
Thetarget = '无GCN'
# =============================================
torch.cuda.set_device(0)
device = torch.device("cuda:0")

lr = 0.001
weight_decay = 1e-4
epochs = 50

best_acc = 0.8
root_dir = '/Data4/gly_wkdir/coldgenepredict/raw_sec/S_italica/CNN/'
loss_png = root_dir + f'loss-lr{lr}-wd{weight_decay}-ep{epochs}-{TheTime}-{Thetarget}.png'
acc_png = root_dir + f'acc-lr{lr}-wd{weight_decay}-ep{epochs}-{TheTime}-{Thetarget}.png'
precision_png = root_dir + f'precision-lr{lr}-wd{weight_decay}-ep{epochs}-{TheTime}-{Thetarget}.png'
recall_png = root_dir + f'recall-lr{lr}-wd{weight_decay}-ep{epochs}-{TheTime}-{Thetarget}.png'
f1_png = root_dir + f'f1-lr{lr}-wd{weight_decay}-ep{epochs}-{TheTime}-{Thetarget}.png'
data_root = '/Data4/gly_wkdir/coldgenepredict/raw_sec/S_italica/分好的数据集csv/二进制'
train_data_dir = data_root + '/train/'
val_data_dir = data_root + '/val/'

print('创建模型实例')
model = GCN_MLP(128*2).to(device)
print('模型实例创建完成')

criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
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
    sequence = batch[0].to(torch.float32)
    label = batch[1].to(torch.float32)

    sequence = sequence.squeeze(2)
    label = label.unsqueeze(1)
    num1s += (label == 1).sum().item()
    num0s += (label == 0).sum().item()

    size_0_sequence = sequence.size(0)
    permuted_index = torch.randperm(size_0_sequence)
    permuted_sequence = sequence[permuted_index]
    permuted_label = label[permuted_index]
    return permuted_sequence, permuted_label, num1s, num0s

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
        # print('最后一个批次的输出, 不满batchsize是正常的:\n', outputs)

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
            loss = criterion(outputs, permuted_label)
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

    if val_accs[-1] > best_acc:
        best_acc = val_accs[-1]
        torch.save(model.state_dict(), root_dir + f'model({epoch}-{best_acc:.4f}-{lr}-{weight_decay}-{epochs}-{TheTime}--{Thetarget}).pth')


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
