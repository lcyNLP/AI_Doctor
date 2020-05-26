import os
import torch
import json
from bilstm_crf import BiLSTM

def singel_predict(model_path, content, char_to_id_json_path, batch_size, embedding_dim,
                   hidden_dim, num_layers, sentence_length, offset, target_type_list, tag2id):

    char_to_id = json.load(open(char_to_id_json_path, mode="r", encoding="utf-8"))
    # 将字符串转为码表id列表
    char_ids = content_to_id(content, char_to_id)
    # 处理成 batch_size * sentence_length 的 tensor 数据
    # 定义模型输入列表
    model_inputs_list, model_input_map_list = build_model_input_list(content,
                                                                     char_ids,
                                                                     batch_size,
                                                                     sentence_length,
                                                                     offset)
    # 加载模型
    model = BiLSTM(vocab_size=len(char_to_id),
                       tag_to_ix=tag2id,
                       embedding_dim=embedding_dim,
                       hidden_dim=hidden_dim,
                       batch_size=batch_size,
                       num_layers=num_layers,
                       sequence_length=sentence_length)
    # 加载模型字典
    model.load_state_dict(torch.load(model_path))

    tag_id_dict = {v: k for k, v in tag_to_id.items() if k[2:] in target_type_list}
    # 定义返回实体列表
    entities = []
    with torch.no_grad():
        for step, model_inputs in enumerate(model_inputs_list):
            prediction_value = model(model_inputs)
            # 获取每一行预测结果
            for line_no, line_value in enumerate(prediction_value):
                # 定义将要识别的实体
                entity = None
                # 获取当前行每个字的预测结果
                for char_idx, tag_id in enumerate(line_value):
                    # 若预测结果 tag_id 属于目标字典数据 key 中
                    if tag_id in tag_id_dict:
                        # 取符合匹配字典id的第一个字符，即B, I
                        tag_index = tag_id_dict[tag_id][0]
                        # 计算当前字符确切的下标位置
                        current_char = model_input_map_list[step][line_no][char_idx]
                        # 若当前字标签起始为 B, 则设置为实体开始
                        if tag_index == "B":
                            entity = current_char
                        # 若当前字标签起始为 I, 则进行字符串追加
                        elif tag_index == "I" and entity:
                            entity += current_char
                    # 当实体不为空且当前标签类型为 O 时，加入实体列表
                    if tag_id == tag_to_id["O"] and entity:
                        # 满足当前字符为O，上一个字符为目标提取实体结尾时，将其加入实体列表
                        entities.append(entity)
                        # 重置实体
                        entity = None
    return entities


def content_to_id(content, char_to_id):
    # 定义字符串对应的码表 id 列表
    char_ids = []
    for char in list(content):
        # 判断若字符不在码表对应字典中，则取 NUK 的编码（即 unknown），否则取对应的字符编码
        if char_to_id.get(char):
            char_ids.append(char_to_id[char])
        else:
            char_ids.append(char_to_id["UNK"])
    return char_ids


def build_model_input_list(content, char_ids, batch_size, sentence_length, offset):
    # 定义模型输入数据列表
    model_input_list = []
    # 定义每个批次句子 id 数据
    batch_sentence_list = []
    # 将文本内容转为列表
    content_list = list(content)
    # 定义与模型 char_id 对照的文字
    model_input_map_list = []
    #  定义每个批次句子字符数据
    batch_sentence_char_list = []
    # 判断是否需要 padding
    if len(char_ids) % sentence_length > 0:
        # 将不足 batch_size * sentence_length 的部分填充0
        padding_length = (batch_size * sentence_length
                          - len(char_ids) % batch_size * sentence_length
                          - len(char_ids) % sentence_length)
        char_ids.extend([0] * padding_length)
        content_list.extend(["#"] * padding_length)
    # 迭代字符 id 列表
    # 数据满足 batch_size * sentence_length 将加入 model_input_list
    for step, idx in enumerate(range(0, len(char_ids) + 1, sentence_length)):
        # 起始下标，从第一句开始增加 offset 个字的偏移
        start_idx = 0 if idx == 0 else idx - step * offset
        # 获取长度为 sentence_length 的字符 id 数据集
        sub_list = char_ids[start_idx:start_idx + sentence_length]
        # 获取长度为 sentence_length 的字符数据集
        sub_char_list = content_list[start_idx:start_idx + sentence_length]
        # 加入批次数据集中
        batch_sentence_list.append(sub_list)
        # 批量句子包含字符列表
        batch_sentence_char_list.append(sub_char_list)
        # 每当批次长度达到 batch_size 时候，将其加入 model_input_list
        if len(batch_sentence_list) == batch_size:
            # 将数据格式转为 tensor 格式，大小为 batch_size * sentence_length
            model_input_list.append(torch.tensor(batch_sentence_list))
            # 重置 batch_sentence_list
            batch_sentence_list = []
            # 将 char_id 对应的字符加入映射表中
            model_input_map_list.append(batch_sentence_char_list)
            # 重置批字符串内容
            batch_sentence_char_list = []
    # 返回模型输入列表
    return model_input_list, model_input_map_list


# 参数1:待识别文本
content = "本病是由DNA病毒的单纯疱疹病毒所致。人类单纯疱疹病毒分为两型，" \
"即单纯疱疹病毒Ⅰ型（HSV-Ⅰ）和单纯疱疹病毒Ⅱ型（HSV-Ⅱ）。" \
"Ⅰ型主要引起生殖器以外的皮肤黏膜（口腔黏膜）和器官（脑）的感染。" \
"Ⅱ型主要引起生殖器部位皮肤黏膜感染。" \
"病毒经呼吸道、口腔、生殖器黏膜以及破损皮肤进入体内，" \
"潜居于人体正常黏膜、血液、唾液及感觉神经节细胞内。" \
"当机体抵抗力下降时，如发热胃肠功能紊乱、月经、疲劳等时，" \
"体内潜伏的HSV被激活而发病。"
# 参数2:模型保存文件路径
model_path = "./model/bilstm_crf_state_dict_20200129_210417.pt"
# 参数3:批次大小
BATCH_SIZE = 8
# 参数4:字向量维度
EMBEDDING_DIM = 300
# 参数5:隐层维度
HIDDEN_DIM = 128
NUM_LAYERS = 1
# 参数6:句子长度
SENTENCE_LENGTH = 100
# 参数7:偏移量
OFFSET = 10
# 参数8:标签码表对照字典
tag_to_id = {"O": 0, "B-dis": 1, "I-dis": 2, "B-sym": 3, "I-sym": 4, "<START>": 5, "<STOP>": 6}
# 参数9:字符码表文件路径
char_to_id_json_path = "./data/char_to_id.json"
# 参数10:预测结果存储路径
prediction_result_path = "prediction_result"
# 参数11:待匹配标签类型
target_type_list = ["sym"]


entities = singel_predict(model_path,
                          content,
                          char_to_id_json_path,
                          BATCH_SIZE,
                          EMBEDDING_DIM,
                          HIDDEN_DIM,
                          NUM_LAYERS,
                          SENTENCE_LENGTH,
                          OFFSET,
                          target_type_list,
                          tag_to_id)

print("entities:\n", entities)
