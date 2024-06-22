#定义一个字典，记录不正确序列及其错误次数{sequence: count}
import pickle
import torch
def preprocess(num1s, num0s, f):
    batch = pickle.load(f)
    sequence = batch[0]
    label = batch[2].to(torch.float32)

    label = label.unsqueeze(1)
    num1s += (label == 1).sum().item()
    num0s += (label == 0).sum().item()

    return sequence, label, num1s, num0s
device = torch.device("cuda:0")
wrong_sequence = {}
def get_wrong_sequence(outputs, labels, sequence):
    outputs = torch.where(outputs >= 0.5, torch.tensor(1.0, dtype=torch.float32).to(device), torch.tensor(0.0, dtype=torch.float32).to(device))
    print(outputs)
    for i in range(len(outputs)):
        if outputs[i] != labels[i]:
            seq = sequence[i]
            seq = seq.cpu().detach().numpy()
            seq = ''.join([str(x) for x in seq])
            if seq in wrong_sequence:
                wrong_sequence[seq] += 1
            else:
                wrong_sequence[seq] = 1

                
data_root = '/Data4/gly_wkdir/coldgenepredict/raw_sec/S_italica/分好的数据集csv/二进制GRU/排序好'
train_data_dir = data_root + '/train/'
for i in range(1):
    with open(train_data_dir + 'train_batch_' + str(i) + '.pkl', 'rb') as f:
        permuted_sequence, permuted_label, num1s, num0s = preprocess(num1s, num0s, f)
        outputs = torch.randn(32, 1).to(device)
        get_wrong_sequence(outputs, permuted_label, permuted_sequence)