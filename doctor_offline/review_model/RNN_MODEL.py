import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        # input_size: 输入张量最后一个维度的大小
        # hidden_size: 隐藏层张量最后一个维度的大小
        # output_size: 输出张量最后一个维度的大小
        super(RNN, self).__init__()

        # 将隐藏层的大小写成类的内部变量
        self.hidden_size = hidden_size

        # 构建第一个线性层, 输入尺寸是input_size + hidden_size，因为真正进入全连接层的张量是X(t) + h(t-1)
        # 输出尺寸是hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)

        # 构建第二个线性层, 输入尺寸是input_size + hidden_size
        # 输出尺寸是output_size
        self.i2o = nn.Linear(input_size + hidden_size, output_size)

        # 定义最终输出的softmax处理层
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input1, hidden1):
        # 首先要进行输入张量的拼接, 将X(t)和h(t-1)拼接在一起
        combined = torch.cat((input1, hidden1), 1)

        # 让输入经过隐藏层获得hidden
        hidden = self.i2h(combined)

        # 让输入经过输出层获得output
        output = self.i2o(combined)

        # 让output经过softmax层
        output = self.softmax(output)

        # 返回两个张量,output, hidden
        return output, hidden

    def initHidden(self):
        # 将隐藏层初始化为一个[1, hidden_size]的全0张量
        return torch.zeros(1, self.hidden_size)

