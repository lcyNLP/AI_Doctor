import torch
import torch.nn as nn
from RNN_MODEL import RNN
import time
import math


input_size = 768
hidden_size = 128
n_categories = 2

input1 = torch.rand(1, input_size)
hidden1 = torch.rand(1, hidden_size)

rnn = RNN(input_size, hidden_size, n_categories)
output, hidden = rnn(input1, hidden1)
# print(output)
# print(output.shape)

def timeSince(since):
    # 功能:获取每次打印的时间消耗, since是训练开始的时间
    # 获取当前的时间
    now = time.time()

    # 获取时间差, 就是时间消耗
    s = now - since

    # 获取时间差的分钟数
    m = math.floor(s/60)

    # 获取时间差的秒数
    s -= m*60

    return '%dm %ds' % (m, s)

since = time.time() - 10*60

period = timeSince(since)
print(period)

