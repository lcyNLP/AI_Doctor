import torch
import torch.nn as nn

# 评估模型的准确率, 召回率, F1等指标
def evaluate(sentence_list, true_tag, predict_tag, id2char, id2tag):
    '''
    sentence_list: 文本向量化后的句子张量
    true_tag: 真实的标签
    predict_tag: 预测的标签
    id2tag: id值到中文字符的映射表
    id2tag: id值到tag标签的映射表
    '''
    # 初始化真实的命名实体, 预测的命名实体, 接下来比较两者的异同来评估指标
    true_entities, true_entity = [], []
    predict_entities, predict_entity = [], []

    # 逐条的遍历批次中的所有语句
    for line_num, sentence in enumerate(sentence_list):
        # 遍历一条样本语句中的每一个字符编码(这里面都是数字化之后的编码)
        for char_num in range(len(sentence)):
            # 如果编码等于0, 表示PAD, 说明后续全部都是填充的0, 可以跳出当前for循环
            if sentence[char_num] == 0:
                break

            # 依次提取真实的语句字符, 真实的样本标签, 预测的样本标签
            char_text = id2char[sentence[char_num]]
            true_tag_type = id2tag[true_tag[line_num][char_num]]
            predict_tag_type = id2tag[predict_tag[line_num][char_num]]

            # 先对真实的标签进行命名实体的匹配
            # 如果第一个字符是"B", 表示一个实体的开始, 将"字符/标签"的格式添加进实体列表中
            if true_tag_type[0] == "B":
                true_entity = [char_text + "/" + true_tag_type]
            # 如果第一个字符是"I", 表示处于一个实体的中间
            # 如果真实的命名实体列表非空, 并且最后一个添加进去的标签类型和当前的标签类型一样, 则继续添加
            # 意思就是比如true_entity = ["中/B-Person", "国/I-Person"], 此时"人/I-Person"就可以进行添加
            elif true_tag_type[0] == "I" and len(true_entity) != 0 and true_entity[-1].split("/")[1][1:] == true_tag_type[1:]:
                true_entity.append(char_text + "/" + true_tag_type)
            # 如果第一个字符是"O", 并且true_entity非空, 表示一个命名实体已经匹配结束
            elif true_tag_type[0] == "O" and len(true_entity) != 0:
                true_entity.append(str(line_num) + "_" + str(char_num))
                # 将匹配结束的一个命名实体加入到最终的真实实体列表中
                true_entities.append(true_entity)
                # 清空true_entity，为了下一个命名实体的匹配做准备
                true_entity = []
            # 除了上述3种情况, 说明当前没有匹配出任何的实体, 则清空true_entity， 继续下一轮匹配
            else:
                true_entity = []

            # 对预测的标签进行命名实体的匹配
            # 如果第一个字符是"B", 表示一个实体的开始, 将"字符/标签"的格式添加进实体列表中
            if predict_tag_type[0] == "B":
                predict_entity = [char_text + "/" + predict_tag_type]
            # 如果第一个字符是"I", 表示处于一个实体的中间
            # 如果预测命名实体列表非空, 并且最后一个添加进去的标签类型和当前的标签类型一样, 则继续添加
            elif predict_tag_type[0] == "I" and len(predict_entity) != 0 and predict_entity[-1].split("/")[1][1:] == predict_tag_type[1:]:
                predict_entity.append(char_text + "/" + predict_tag_type)
            # 如果第一个字符是"O", 并且predict_entity非空, 表示一个完整的命名实体已经匹配结束了
            elif predict_tag_type[0] == "O" and len(predict_entity) != 0:
                predict_entity.append(str(line_num) + "_" + str(char_num))
                # 将这个匹配结束的预测命名实体添加到最终的预测实体列表中
                predict_entities.append(predict_entity)
                # 清空predict_entity, 为下一个命名实体的匹配做准备
                predict_entity = []
            # 除了上述3种情况, 说明当前没有匹配出任何的实体, 则清空predict_entity, 继续下一轮的匹配
            else:
                predict_entity = []

    # 遍历所有预测出来的实体列表, 只有那些在真实命名实体列表中的实体才是正确的预测
    acc_entities = [entity for entity in predict_entities if entity in true_entities]

    # 计算正确实体的个数, 预测实体的个数, 真实实体的个数
    acc_entities_length = len(acc_entities)
    predict_entities_length = len(predict_entities)
    true_entities_length = len(true_entities)

    # 至少争取预测了一个实体的情况下, 才计算准确率, 召回率, F1值
    if acc_entities_length > 0:
        accuracy = float(acc_entities_length / predict_entities_length)
        recall = float(acc_entities_length / true_entities_length)
        f1_score = 2.0 * accuracy * recall / (accuracy + recall)
        return accuracy, recall, f1_score, acc_entities_length, predict_entities_length, true_entities_length
    else:
        return 0, 0, 0, acc_entities_length, predict_entities_length, true_entities_length


# 真实标签数据
tag_list = [
    [0, 0, 3, 4, 0, 3, 4, 0, 0, 0, 0, 0, 0, 0, 3, 4, 0, 0, 0, 0],
    [0, 0, 3, 4, 0, 3, 4, 0, 0, 0, 0, 0, 0, 0, 3, 4, 0, 0, 0, 0],
    [0, 0, 3, 4, 0, 3, 4, 0, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [3, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 2, 0, 0, 0, 0],
    [0, 0, 1, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [3, 4, 0, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 3, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
]

# 预测标签数据
predict_tag_list = [
    [0, 0, 0, 0, 0, 3, 4, 0, 0, 0, 0, 0, 0, 0, 3, 4, 0, 0, 0, 0],
    [0, 0, 3, 4, 0, 3, 4, 0, 0, 0, 0, 0, 0, 0, 3, 4, 0, 0, 0, 0],
    [0, 0, 3, 4, 0, 3, 4, 0, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [3, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 2, 0, 0, 0, 0],
    [0, 0, 1, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 3, 4, 0, 0, 0, 0, 0],
    [3, 4, 0, 3, 4, 0, 0, 1, 2, 0, 0, 3, 4, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 3, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
]

# 编码与字符对照字典
id2char = {0: '<PAD>', 1: '确', 2: '诊', 3: '弥', 4: '漫', 5: '大', 6: 'b', 7: '细', 8: '胞', 9: '淋', 10: '巴', 11: '瘤', 12: '1', 13: '年', 14: '反', 15: '复', 16: '咳', 17: '嗽', 18: '、', 19: '痰', 20: '4', 21: '0', 22: ',', 23: '再', 24: '发', 25: '伴', 26: '气', 27: '促', 28: '5', 29: '天', 30: '。', 31: '生', 32: '长', 33: '育', 34: '迟', 35: '缓', 36: '9', 37: '右', 38: '侧', 39: '小', 40: '肺', 41: '癌', 42: '第', 43: '三', 44: '次', 45: '化', 46: '疗', 47: '入', 48: '院', 49: '心', 50: '悸', 51: '加', 52: '重', 53: '胸', 54: '痛', 55: '3', 56: '闷', 57: '2', 58: '多', 59: '月', 60: '余', 61: ' ', 62: '周', 63: '上', 64: '肢', 65: '无', 66: '力', 67: '肌', 68: '肉', 69: '萎', 70: '缩', 71: '半'}

# 编码与标签对照字典
id2tag = {0: 'O', 1: 'B-dis', 2: 'I-dis', 3: 'B-sym', 4: 'I-sym'}

# 输入的数字化sentences_sequence, 由下面的sentence_list经过映射函数sentence_map()转化后得到
sentence_list = [
    "确诊弥漫大b细胞淋巴瘤1年",
    "反复咳嗽、咳痰40年,再发伴气促5天。",
    "生长发育迟缓9年。",
    "右侧小细胞肺癌第三次化疗入院",
    "反复气促、心悸10年,加重伴胸痛3天。",
    "反复胸闷、心悸、气促2多月,加重3天",
    "咳嗽、胸闷1月余, 加重1周",
    "右上肢无力3年, 加重伴肌肉萎缩半年"
]


# 添加中文字符的数字化编码函数
def sentence_map(sentence_list, char_to_id, max_length=100):
    sentence_list.sort(key=lambda x: len(x), reverse=True)
    sentence_map_list = []
    for sentence in sentence_list:
        sentence_id_list = [char_to_id[c] for c in sentence]
        padding_list = [0] * (max_length - len(sentence))
        sentence_id_list.extend(padding_list)
        sentence_map_list.append(sentence_id_list)
    return torch.tensor(sentence_map_list, dtype=torch.long)

char_to_id = {"<PAD>": 0}

SENTENCE_LENGTH = 20

for sentence in sentence_list:
    for c in sentence:
        if c not in char_to_id:
            char_to_id[c] = len(char_to_id)


if __name__ == '__main__':
    sentence_sequence = sentence_map(sentence_list, char_to_id, SENTENCE_LENGTH)
    accuracy, recall, f1_score, acc_entities_length, predict_entities_length, true_entities_length = evaluate(sentence_sequence.tolist(), tag_list, predict_tag_list, id2char, id2tag)

    print("accuracy:",          accuracy,
          "\nrecall:",          recall,
          "\nf1_score:",        f1_score,
          "\nacc_entities_length:", acc_entities_length,
          "\npredict_entities_length:", predict_entities_length,
          "\ntrue_entities_length:", true_entities_length)

