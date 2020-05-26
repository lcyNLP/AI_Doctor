import json
import numpy as np

# 创建训练数据集, 从原始训练文件中将中文字符进行数字化编码, 同时也将标签进行数字化的编码
def create_train_data(train_data_file, result_file, json_file, tag2id, max_length=100):
    '''
    train_data_file: 原始训练文件
    result_file: 处理后的结果文件
    json_file: 中文字符向id的映射表, 也是一个文件char_to_id.json
    tag2id: 标签向id的映射表, 提前已经写好了
    '''
    # 导入json格式的中文字符向id的映射表
    char2id = json.load(open(json_file, mode='r', encoding='utf-8'))

    char_data, tag_data = [], []

    # 打开原始训练文件
    with open(train_data_file, mode='r', encoding='utf-8') as f:
        # 初始化一条语句数字化编码后的列表
        char_ids = [0] * max_length
        tag_ids = [0] * max_length
        idx = 0
        # 遍历文件中的每一行
        for line in f.readlines():
            # char \t tag
            line = line.strip('\n').strip()
            # 如果不是空行, 并且当前语句的长度没有超过max_length，则进行字符到id的映射
            if line and len(line) > 0 and idx < max_length:
                ch, tag = line.split('\t')
                # 如果当前字符在映射表中,则直接映射为对应的id值
                if char2id.get(ch):
                    char_ids[idx] = char2id[ch]
                # 否则直接用"UNK"的id值进行赋值, 代表的是未知的字符
                else:
                    char_ids[idx] = char2id['UNK']
                # 将标签对应的id值进行数字化编码映射
                tag_ids[idx] = tag2id[tag]
                idx += 1
            # 如果是空行, 或者当前语句的长度超过了max_length
            else:
                # 如果当前语句的长度超过了max_length，直接将[0: max_length]的部分直接进行结果赋值
                if idx <= max_length:
                    char_data.append(char_ids)
                    tag_data.append(tag_ids)
                # 遇到空行, 说明当前一条完整的语句已经结束了, 需要将初始化列表进行清零操作, 为了下一个句子的迭代做准备
                char_ids = [0] * max_length
                tag_ids = [0] * max_length
                idx = 0

    # 将数字化编码后的数据封装成numpy的数组类型, 数字化编码采用int32
    x_data = np.array(char_data, dtype=np.int32)
    y_data = np.array(tag_data, dtype=np.int32)

    # 直接利用np.savez()将数据存储成.npz类型的文件
    np.savez(result_file, x_data=x_data, y_data=y_data)
    print("create_train_data Finished!".center(100, "-"))

json_file = './data/char_to_id.json'

# 参数2:标签码表对照字典
tag2id = {"O": 0, "B-dis": 1, "I-dis": 2, "B-sym": 3, "I-sym": 4, "<START>": 5, "<STOP>": 6}

# 参数3:训练数据文件路径
train_data_file = './data/total.txt'

# 参数4:创建的npz文件保路径(训练数据)
result_file = './data/total.npz'


if __name__ == '__main__':
    create_train_data(train_data_file, result_file, json_file, tag2id)

