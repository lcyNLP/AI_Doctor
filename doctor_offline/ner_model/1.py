import json
import numpy as np

# 创建训练数据集, 从原始训练文件中将中文字符进行数字化编码, 同时也将标签进行数字化的编码
def create_train_data(train_data_file, validate_data_file, result_file, json_file, tag2id, max_length=100):
    '''
    train_data_file: 原始训练文件
    result_file: 处理后的结果文件
    json_file: 中文字符向id的映射表, 也是一个文件char_to_id.json
    tag2id: 标签向id的映射表, 提前已经写好了
    '''
    # 导入json格式的中文字符向id的映射表
    char_to_id = json.load(open(json_file, mode='r', encoding='utf-8'))

    train_data_list, label_data_list = [], []

    # 打开原始训练文件
    with open(train_data_file, mode='r', encoding='utf-8') as f:
        # 初始化一条语句数字化编码后的列表
        for line in f.readlines():
            line = line.strip('\n').strip()
            data = json.loads(line)
            feature = data['text']
            label = data['label']
            char_ids = [0] * max_length
            tag_ids = [0] * max_length
            for idx, char in enumerate(feature):
                if idx >= max_length:
                    break
                if char_to_id.get(char):
                    char_ids[idx] = char_to_id[char]
                else:
                    char_ids[idx] = char_to_id["UNK"]

            train_data_list.append(char_ids)

            for idx, tag in enumerate(label):
                if idx >= max_length:
                    break
                tag_ids[idx] = tag2id[tag]

            label_data_list.append(tag_ids)

    with open(validate_data_file, mode='r', encoding='utf-8') as f:
        # 初始化一条语句数字化编码后的列表
        for line in f.readlines():
            line = line.strip('\n').strip()
            data = json.loads(line)
            feature = data['text']
            label = data['label']
            char_ids = [0] * max_length
            tag_ids = [0] * max_length
            for idx, char in enumerate(feature):
                if idx >= max_length:
                    break
                if char_to_id.get(char):
                    char_ids[idx] = char_to_id[char]
                else:
                    char_ids[idx] = char_to_id["UNK"]

            train_data_list.append(char_ids)

            for idx, tag in enumerate(label):
                if idx >= max_length:
                    break
                tag_ids[idx] = tag2id[tag]

            label_data_list.append(tag_ids)

    # 将数字化编码后的数据封装成numpy的数组类型, 数字化编码采用int32
    x_data = np.array(train_data_list, dtype=np.int32)
    y_data = np.array(label_data_list, dtype=np.int32)

    # 直接利用np.savez()将数据存储成.npz类型的文件
    np.savez(result_file, x_data=x_data, y_data=y_data)
    print("create_train_data Finished!".center(100, "-"))

json_file = './data/char_to_id.json'

# 参数2:标签码表对照字典
tag2id = {"O": 0, "B-dis": 1, "I-dis": 2, "B-sym": 3, "I-sym": 4, "<START>": 5, "<STOP>": 6}

# 参数3:训练数据文件路径
# train_data_file = './data/train.txt'
train_data_file = './data/train.txt'
validate_data_file = './data/validate.txt'

# 参数4:创建的npz文件保路径(训练数据)
# result_file = './data/train.npz'
result_file = './data/train.npz'


if __name__ == '__main__':
    create_train_data(train_data_file, validate_data_file, result_file, json_file, tag2id)

