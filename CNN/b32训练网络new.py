import pickle
from 原始网络结构bndp5 import GCN_MLP, bcolors
import torch
import time
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

def train_or_evaluate(model, data_dir, data_type, criterion, optimizer=None, train=True):
    num_batches = 135 if data_type == 'train' else 27
    epoch_loss = 0
    right_num = 0
    num1s = 0
    num0s = 0
    all_preds = []
    all_labels = []

    for i in range(num_batches):
        with open(f'{data_dir}{data_type}_batch_{i}.pkl', 'rb') as f:
            batch = pickle.load(f)
            sequence = batch[0].to(torch.float32)
            label = batch[1].to(torch.float32)

            sequence = sequence.squeeze(2)  # torch.Size([64, 46398])
            label = label.unsqueeze(1)  # torch.Size([64, 1])
            num1s += (label == 1).sum().item()
            num0s += (label == 0).sum().item()

            size_0_sequence = sequence.size(0)
            permuted_index = torch.randperm(size_0_sequence)
            permuted_sequence = sequence[permuted_index]
            permuted_label = label[permuted_index]

            permuted_sequence = permuted_sequence.to(device)
            permuted_label = permuted_label.to(device)
            outputs = model(permuted_sequence)

            loss = criterion(outputs, permuted_label)
            epoch_loss += loss.item()

            outputs = torch.sigmoid(outputs)
            outputs = torch.where(outputs >= 0.5, torch.tensor(1.0, dtype=torch.float32).to(device), torch.tensor(0.0, dtype=torch.float32).to(device))

            right_num += (outputs == permuted_label).sum().item()
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(permuted_label.cpu().numpy())

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    _all = num1s + num0s
    accuracy = right_num / _all
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    return epoch_loss, accuracy, precision, recall, f1

def save_metrics_plot(train_metrics, val_metrics, metric_name, file_name):
    plt.plot(train_metrics, label=f'train_{metric_name}')
    plt.plot(val_metrics, label=f'val_{metric_name}')
    plt.xlabel('epoch')
    plt.ylabel(metric_name)
    plt.legend()
    plt.savefig(file_name)
    plt.clf()

TheTime = str(time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()))
print(TheTime)
# 参数定义
torch.cuda.set_device(0)
device = torch.device("cuda:0")
cpu = torch.device("cpu")
lr = 0.001
weight_decay = 1e-4
epochs = 100
best_acc = 0.8

root_dir = '/Data4/gly_wkdir/coldgenepredict/raw_sec/S_italica/CNN/'
loss_png = root_dir + f'loss64-lr{lr}-wd{weight_decay}-ep{epochs}-{TheTime}.png'
acc_png = root_dir + f'acc64-lr{lr}-wd{weight_decay}-ep{epochs}-{TheTime}.png'
precision_png = root_dir + f'precision64-lr{lr}-wd{weight_decay}-ep{epochs}-{TheTime}.png'
recall_png = root_dir + f'recall64-lr{lr}-wd{weight_decay}-ep{epochs}-{TheTime}.png'
f1_png = root_dir + f'f1_score64-lr{lr}-wd{weight_decay}-ep{epochs}-{TheTime}.png'
train_data_dir = '/Data4/gly_wkdir/coldgenepredict/raw_sec/S_italica/分好的数据集csv/二进制/train/'
val_data_dir = '/Data4/gly_wkdir/coldgenepredict/raw_sec/S_italica/分好的数据集csv/二进制/val/'

print('创建模型实例')
model = GCN_MLP(512*2).to(device)
print('模型实例创建完成')

criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

train_losses, train_accs = [], []
val_losses, val_accs = [], []
train_precisions, train_recalls, train_f1s = [], [], []
val_precisions, val_recalls, val_f1s = [], [], []

for epoch in range(epochs):
    model.train()
    train_loss, train_acc, train_precision, train_recall, train_f1 = train_or_evaluate(
        model, train_data_dir, 'train', criterion, optimizer, train=True
    )

    print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.4f}')
    print(f'Train Accuracy: {bcolors.FAIL}{train_acc:.4f}{bcolors.ENDC}')
    print(f'Train Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1 Score: {train_f1:.4f}')
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    train_precisions.append(train_precision)
    train_recalls.append(train_recall)
    train_f1s.append(train_f1)

    model.eval()
    with torch.no_grad():
        val_loss, val_acc, val_precision, val_recall, val_f1 = train_or_evaluate(
            model, val_data_dir, 'val', criterion, train=False
        )

    print(f'Validation Loss: {val_loss:.4f}')
    print(f'Validation Accuracy: {bcolors.OKGREEN}{val_acc:.4f}{bcolors.ENDC}')
    print(f'Validation Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1 Score: {val_f1:.4f}')
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    val_precisions.append(val_precision)
    val_recalls.append(val_recall)
    val_f1s.append(val_f1)

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), root_dir + f'model({epoch}-{val_acc:.4f}-{lr}-{weight_decay}-{epochs}-{TheTime}).pth')

save_metrics_plot(train_losses, val_losses, 'loss', loss_png)
save_metrics_plot(train_accs, val_accs, 'accuracy', acc_png)
save_metrics_plot(train_precisions, val_precisions, 'precision', precision_png)
save_metrics_plot(train_recalls, val_recalls, 'recall', recall_png)
save_metrics_plot(train_f1s, val_f1s, 'f1_score', f1_png)

print('Train Losses: ', train_losses)
print('Validation Losses: ', val_losses)
print('Train Accuracies: ', train_accs)
print('Validation Accuracies: ', val_accs)
print('Train Precisions: ', train_precisions)
print('Validation Precisions: ', val_precisions)
print('Train Recalls: ', train_recalls)
print('Validation Recalls: ', val_recalls)
print('Train F1 Scores: ', train_f1s)
print('Validation F1 Scores: ', val_f1s)
