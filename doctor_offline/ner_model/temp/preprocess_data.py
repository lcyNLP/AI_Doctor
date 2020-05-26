import json
import numpy as np


def create_train_data(train_data_file, result_file, json_file, tag2id, max_length=20):
    char2id = json.load(open(json_file, mode='r', encoding='utf-8'))

    char_data, tag_data = [], []

    with open(train_data_file, mode='r', encoding='utf-8') as f:
        char_ids = [0] * max_length
        tag_ids = [0] * max_length
        idx = 0
        for line in f.readlines():
            line = line.strip('\n').strip()
            if len(line) > 0 and line and idx < max_length:
                ch, tag = line.split('\t')
                if char2id.get(ch):
                    char_ids[idx] = char2id[ch]
                else:
                    char_ids[idx] = char2id['UNK']
                tag_ids[idx] = tag2id[tag]
                idx += 1
            else:
                if idx <= max_length:
                    char_data.append(char_ids)
                    tag_data.append(tag_ids)
                char_ids = [0] * max_length
                tag_ids = [0] * max_length
                idx = 0

    x_data = np.array(char_data, dtype=np.int32)
    y_data = np.array(tag_data, dtype=np.int32)

    np.savez(result_file, x_data=x_data, y_data=y_data)
    print("create_train_data Finished!".center(100, "-"))


json_file = './data/char_to_id.json'

tag2id = {"O": 0, "B-dis": 1, "I-dis": 2, "B-sym": 3, "I-sym": 4, "<START>": 5, "<STOP>": 6}

train_data_file = './data/train.txt'

result_file = './data/train.npz'


if __name__ == '__main__':
    create_train_data(train_data_file, result_file, json_file, tag2id)

