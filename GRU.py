import torch
import torch.nn as nn


class AttentionLayer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AttentionLayer, self).__init__()
        print('创建注意力机制')
        self.attn = nn.MultiheadAttention(embed_dim=input_size, num_heads=1)
        print('创建全连接层')
        self.fc = nn.Linear(input_size, hidden_size)
        print('创建激活函数')
        self.relu = nn.ReLU()

    def forward(self, gru_outputs):
        attn_output, _ = self.attn(gru_outputs, gru_outputs, gru_outputs)
        attn_output = self.fc(attn_output)
        attn_output = self.relu(attn_output)
        return attn_output


class Classifier(nn.Module):
    def __init__(self, input_size:int, hidden_size:int):
        """
        这是一个包含注意力机制的GRU模型。
        Attributes:
            input_size (int): 输入序列的长度,46398...。
            hidden_size (int): 隐藏状态的大小。
        """
        super(Classifier, self).__init__()
        print('创建GRU')
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        # print('创建注意力机制模块')
        # self.attention = AttentionLayer(input_size, hidden_size)
        print('创建全连接层')
        self.fc1 = nn.Linear(hidden_size, 50)
        print('创建输出层')
        self.fc2 = nn.Linear(50, 2)
        print('创建激活函数')
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.flatten = nn.Flatten()
        self.fc3 = nn.Linear(10, 1)
        self.Relu = nn.ReLU()

    def forward(self, inputs):
        # print('1', inputs.shape)torch.Size([32, 46398, 5])
        inputs = inputs.permute(0, 2, 1)  # 1 torch.Size([32, 5, 46398])
        gru_outputs, _ = self.gru(inputs)  # 2 torch.Size([32, 4, 20])
        # print('2', gru_outputs.shape)
        # attention_output = self.attention(inputs)  # 3 torch.Size([32, 4, 20])
        # print('3', attention_output.shape)
        # 把注意力层的输出和GRU层的输出按位相乘起来
        # concat_output = gru_outputs * attention_output  # 3.5 torch.Size([32, 4, 20])
        # print('3.5', concat_output.shape)
        concat_output = gru_outputs
        fc1_output = self.fc1(concat_output)  # 4 torch.Size([32, 4, 50])
        # print('4', fc1_output.shape)
        fc1_output = self.relu(fc1_output)  # 5 torch.Size([32, 4, 50])
        # print('5', fc1_output.shape)
        fc2_output = self.fc2(fc1_output)  # 6 torch.Size([32, 4, 2])
        # print('6', fc2_output.shape)
        output = self.sigmoid(fc2_output)  # 7 torch.Size([32, 4, 2])
        # print('7', output.shape)
        output = self.flatten(output)  # 8 torch.Size([32, 8])
        # print('8', output.shape)
        output = self.sigmoid(output)  # 9 torch.Size([32, 8])
        # print('9', output.shape)
        output = self.fc3(output)
        output = self.sigmoid(output)  # 10 torch.Size([32, 1])
        # print('10', output.shape)
        return output


class Classifier_1(nn.Module):
    def __init__(self, input_size:int):
        """
        卷积神经网络模型
        Attributes:
            input_size (int): 输入序列的长度,46398...。
        """
        super(Classifier_1, self).__init__()
        print('创建卷积层')
        # 输入通道数为 5，输出通道数为 64，卷积核大小为 3x5
        self.conv1 = nn.Conv2d(in_channels=5, out_channels=64, kernel_size=(10, 1), stride=10)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=640, kernel_size=(10, 1), stride=10)
        self.conv3 = nn.Conv2d(in_channels=640, out_channels=1280, kernel_size=(10, 1), stride=10)
        self.weight1 = nn.Parameter(torch.randn(1, 1, 46, 46))
        self.conv4 = nn.Conv2d(in_channels=1280, out_channels=1000, kernel_size=(3, 3), stride=1)
        self.conv5 = nn.Conv2d(in_channels=1000, out_channels=500, kernel_size=(4, 4), stride=4)
        self.conv6 = nn.Conv2d(in_channels=500, out_channels=300, kernel_size=(2, 2), stride=1)
        self.conv7 = nn.Conv2d(in_channels=300, out_channels=100, kernel_size=(1, 1), stride=1)
        self.conv8 = nn.Conv2d(in_channels=100, out_channels=50, kernel_size=(1, 1), stride=1)
        self.conv9 = nn.Conv2d(in_channels=50, out_channels=25, kernel_size=(1, 1), stride=1)
        self.conv10 = nn.Conv2d(in_channels=25, out_channels=8, kernel_size=(1, 1), stride=1)
        self.conv11 = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=(1, 1), stride=1)
        self.conv12 = nn.Conv2d(in_channels=4, out_channels=2, kernel_size=(1, 1), stride=1)
        self.weight2 = nn.Parameter(torch.randn(1, 1, 10, 1))
        self.weight3 = nn.Parameter(torch.randn(2, 1))
        self.lstm = nn.LSTM(10, 1, batch_first=True)
    def forward(self, inputs):
        # 输入数据形状为 (batch_size, channels, height, width)
        print('inputs.shape', inputs.shape)
        inputs = inputs.permute(0, 2, 1, 3)  # 1 torch.Size([32, 5, 46398, 1])
        print('inputs.shape', inputs.shape)
        # 应用卷积操作
        output = torch.tanh(self.conv1(inputs))
        print(output.shape)
        output = torch.relu(self.conv2(output))
        print(output.shape)
        output = torch.tanh(self.conv3(output))
        print(output.shape)
        output = output * self.weight1 # torch.Size([32, 1280, 46, 46])
        print(output.shape)
        output = torch.relu(self.conv4(output))
        print(output.shape)
        output = torch.tanh(self.conv5(output))
        print(output.shape)
        output = torch.relu(self.conv6(output))
        print(output.shape)
        output = torch.tanh(self.conv7(output))
        print(output.shape)
        output = torch.relu(self.conv8(output))
        print(output.shape)
        output = torch.tanh(self.conv9(output))
        print(output.shape)
        output = torch.relu(self.conv10(output))
        print(output.shape)
        output = torch.tanh(self.conv11(output))
        print(output.shape)
        output = torch.relu(self.conv12(output))
        print(output.shape)
        # print('output', output)
        output = torch.matmul(output, self.weight2) # torch.Size([32, 1, 10, 1])
        print(output.shape)
        output = output.squeeze(3)  # 移除维度为 1 的维度
        print(output.shape)
        output, _ = self.lstm(output)
        print(output.shape)
        output = torch.relu(output)
        output = output.squeeze(2)  # 移除维度为 1 的维度
        print(output.shape)
        output = torch.matmul(output, self.weight3)  # torch.Size([32, 1])
        print(output.shape)
        output = torch.sigmoid(output)
        print('output', output)
        # 输出形状
        # print('1', output.shape)
        return output



class Classifier_jjppgg(nn.Module):
    def __init__(self):
        """
        把输入当图片处理, 图片是五个通道, 216*216
        """
        super(Classifier_jjppgg, self).__init__()
        print('创建卷积层')


