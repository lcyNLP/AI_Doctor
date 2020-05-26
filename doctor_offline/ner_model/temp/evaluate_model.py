import torch
import torch.nn as nn


def sentence_map(sentence_list, char_to_id, max_length):
    sentence_list.sort(key=lambda c:len(c), reverse=True)
    sentence_map_list = []
    for sentence in sentence_list:
        sentence_id_list = [char_to_id[c] for c in sentence]
        padding_list = [0] * (max_length-len(sentence))
        sentence_id_list.extend(padding_list)
        sentence_map_list.append(sentence_id_list)
    return torch.tensor(sentence_map_list, dtype=torch.long)


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

char_to_id = {"<PAD>":0}

SENTENCE_LENGTH = 20

for sentence in sentence_list:
    for _char in sentence:
        if _char not in char_to_id:
            char_to_id[_char] = len(char_to_id)

sentences_sequence = sentence_map(sentence_list, char_to_id, SENTENCE_LENGTH)

tag_list = [
    [0, 0, 3, 4, 0, 3, 4, 0, 0, 0, 0, 0, 0, 0, 3, 4, 0, 0, 0, 0],
    [0, 0, 3, 4, 0, 3, 4, 0, 0, 0, 0, 0, 0, 0, 3, 4, 0, 0, 0, 0],
    [0, 0, 3, 4, 0, 3, 4, 0, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [3, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 2, 0, 0, 0, 0],
    [0, 0, 1, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [3, 4, 0, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 3, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

predict_tag_list = [
    [0, 0, 0, 0, 0, 3, 4, 0, 0, 0, 0, 0, 0, 0, 3, 4, 0, 0, 0, 0],
    [0, 0, 3, 4, 0, 3, 4, 0, 0, 0, 0, 0, 0, 0, 3, 4, 0, 0, 0, 0],
    [0, 0, 3, 4, 0, 3, 4, 0, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [3, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 2, 0, 0, 0, 0],
    [0, 0, 1, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 3, 4, 0, 0, 0, 0, 0],
    [3, 4, 0, 3, 4, 0, 0, 1, 2, 0, 0, 3, 4, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 3, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]


id2char = {0: '<PAD>', 1: '确', 2: '诊', 3: '弥', 4: '漫', 5: '大', 6: 'b', 7: '细', 8: '胞', 9: '淋', 10: '巴', 11: '瘤', 12: '1', 13: '年', 14: '反', 15: '复', 16: '咳', 17: '嗽', 18: '、', 19: '痰', 20: '4', 21: '0', 22: ',', 23: '再', 24: '发', 25: '伴', 26: '气', 27: '促', 28: '5', 29: '天', 30: '。', 31: '生', 32: '长', 33: '育', 34: '迟', 35: '缓', 36: '9', 37: '右', 38: '侧', 39: '小', 40: '肺', 41: '癌', 42: '第', 43: '三', 44: '次', 45: '化', 46: '疗', 47: '入', 48: '院', 49: '心', 50: '悸', 51: '加', 52: '重', 53: '胸', 54: '痛', 55: '3', 56: '闷', 57: '2', 58: '多', 59: '月', 60: '余', 61: ' ', 62: '周', 63: '上', 64: '肢', 65: '无', 66: '力', 67: '肌', 68: '肉', 69: '萎', 70: '缩', 71: '半'}


id2tag = {0: 'O', 1: 'B-dis', 2: 'I-dis', 3: 'B-sym', 4: 'I-sym'}


def evaluate(sentence_list, true_tag, predict_tag, id2char, id2tag):
    true_entities, true_entity = [], []
    predict_entities, predict_entity = [], []
    for line_num, sentence in enumerate(sentence_list):
        for char_num in range(len(sentence)):
            if sentence[char_num]==0:
                break
            char_text = id2char[sentence[char_num]]
            true_tag_type = id2tag[true_tag[line_num][char_num]]
            predict_tag_type = id2tag[predict_tag[line_num][char_num]]
            if true_tag_type[0] == "B":
                true_entity = [char_text + "/" + true_tag_type]
            elif true_tag_type[0] == "I" and len(true_entity) != 0 and true_entity[-1].split("/")[1][1:] == true_tag_type[1:]:
                true_entity.append(char_text + "/" + true_tag_type)
            elif true_tag_type[0] == "O" and len(true_entity) != 0 :
                true_entity.append(str(line_num) + "_" + str(char_num))
                true_entities.append(true_entity)
                true_entity=[]
            else:
                true_entity=[]

            if predict_tag_type[0] == "B":
                predict_entity = [char_text + "/" + predict_tag_type]
            elif predict_tag_type[0] == "I" and len(predict_entity) != 0 and predict_entity[-1].split("/")[1][1:] == predict_tag_type[1:]:
                predict_entity.append(char_text + "/" + predict_tag_type)
            elif predict_tag_type[0] == "O" and len(predict_entity) != 0:
                predict_entity.append(str(line_num) + "_" + str(char_num))
                predict_entities.append(predict_entity)
                predict_entity = []
            else:
                predict_entity = []
    acc_entities = [entity for entity in predict_entities if entity in true_entities]
    acc_entities_length = len(acc_entities)
    predict_entities_length = len(predict_entities)
    true_entities_length = len(true_entities)
    if acc_entities_length > 0:
        step_acc = float(acc_entities_length / predict_entities_length)
        step_recall = float(acc_entities_length / true_entities_length)
        f1_score = 2 * step_acc * step_recall / (step_acc + step_recall)
        return step_acc, step_recall, f1_score, acc_entities_length, predict_entities_length, true_entities_length
    else:
        return 0, 0, 0, acc_entities_length, predict_entities_length, true_entities_length    

if __name__ == '__main__':
    step_acc, step_recall, f1_score, acc_entities_length, predict_entities_length, true_entities_length = evaluate(sentences_sequence.tolist(), tag_list, predict_tag_list, id2char, id2tag)

    print("step_acc:",               step_acc,
      "\nstep_recall:",              step_recall,
      "\nf1_score:",                 f1_score,
      "\nacc_entities_length:",      acc_entities_length,
      "\npredict_entities_length:",  predict_entities_length,
      "\ntrue_entities_length:",     true_entities_length)

