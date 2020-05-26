# 导入若干包
import os
import torch
import torch.nn as nn

# 导入RNN类
from RNN_MODEL import RNN

# 导入bert预训练模型的编码函数
from bert_chinese_encode import get_bert_encode_for_single

# 设定预加载的模型路径
MODEL_PATH = './BERT_RNN.pth'

# 设定若干参数, 注意：这些参数一定要和训练的时候保持完全一致
n_hidden = 128
input_size = 768
n_categories = 2

# 实例化RNN模型，并加载保存的模型参数
rnn = RNN(input_size, n_hidden, n_categories)
rnn.load_state_dict(torch.load(MODEL_PATH))


# 编写测试函数
def _test(line_tensor):
    # 功能：本函数为预测函数服务，用于调用RNN模型并返回结果
    # line_tensor: 代表输入中文文本的张量标识
    # 初始化隐藏层
    hidden = rnn.initHidden()

    # 遍历输入文本中的每一个字符张量
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i].unsqueeze(0), hidden)

    # 返回RNN模型的最终输出
    return output


# 编写预测函数
def predict(input_line):
    # 功能：完成模型的预测
    # input_line: 代表需要预测的中文文本信息
    # 注意: 所有的预测必须保证不自动求解梯度
    with torch.no_grad():
        # 将input_line使用bert模型进行编码，然后将张量传输给_test()函数
        output = _test(get_bert_encode_for_single(input_line))

        # 从output中取出最大值对应的索引，比较的维度是1
        _, topi = output.topk(1, 1)
        return topi.item()


# 编写批量预测的函数
def batch_predict(input_path, output_path):
    # 功能: 批量预测函数
    # input_path: 以原始文本的输入路径(等待进行命名实体审核的文件)
    # output_path: 预测后的输出文件路径(经过命名实体审核通过的所有数据)
    csv_list = os.listdir(input_path)

    # 遍历每一个csv文件
    for csv in csv_list:
        # 要以读的方式打开每一个csv文件
        with open(os.path.join(input_path, csv), "r") as fr:
            # 要以写的方式打开输出路径下的同名csv文件
            with open(os.path.join(output_path, csv), "w") as fw:
                # 读取csv文件的每一行
                input_line = fr.readline()
                # 调用预测函数，利用RNN模型进行审核
                res = predict(input_line)
                if res:
                    # 如果res==1, 说明通过了审核
                    fw.write(input_line + "\n")
                else:
                    pass



if __name__ == '__main__':
    # input_line = "点淤样尖针性发多"
    # result = predict(input_line)
    # print("result:", result)
    input_path = "/data/doctor_offline/structured/noreview/"
    output_path = "/data/doctor_offline/structured/reviewed/"
    batch_predict(input_path, output_path)

