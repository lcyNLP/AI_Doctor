import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim


START_TAG = "<START>"
STOP_TAG = "<STOP>"
tag_to_ix = {"O": 0, "B-dis": 1, "I-dis": 2, "B-sym": 3, "I-sym": 4, START_TAG: 5, STOP_TAG: 6}
EMBEDDING_DIM = 200
HIDDEN_DIM = 100
BATCH_SIZE = 8
SENTENCE_LENGTH = 20
NUM_LAYERS = 1

def to_scalar(var): #var是Variable,维度是１
    # returns a python float
    return var.view(-1).data.tolist()[0]
 
def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)

def log_sum_exp(vec): #vec是1*7, type是Variable
    max_score = vec[0, argmax(vec)]
    #max_score维度是１，　max_score.view(1,-1)维度是１＊１，max_score.view(1, -1).expand(1, vec.size()[1])的维度1 * 7
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1]) # vec.size()维度是1 * 7
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))



class BiLSTM(nn.Module):
    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim,
                 num_layers, batch_size, sequence_length):
        super(BiLSTM, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=self.num_layers, bidirectional=True)

        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))

        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2 * self.num_layers, self.batch_size, self.hidden_dim // 2), 
                torch.randn(2 * self.num_layers, self.batch_size, self.hidden_dim // 2))

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        # LSTM的输入要求形状为 [sequence_length, batch_size, embedding_dim]
        # LSTM的隐藏层h0要求形状为 [num_layers * direction, batch_size, hidden_dim]
        embeds = self.word_embeds(sentence).view(self.sequence_length, self.batch_size, -1)
        
        print(embeds.shape)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        
        lstm_out = lstm_out.view(self.sequence_length, self.batch_size, self.hidden_dim)
        
        lstm_feats = self.hidden2tag(lstm_out)
        
        return lstm_feats

    def _forward_alg(self, feats):
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        forward_var = init_alphas
        feats = feats.transpose(1, 0)
        # Iterate through the sentence
        # feats: [8, 20, 7]是一个3维矩阵, 最外层代表8个句子, 内层代表每个句子有20个字符, 
        result = torch.zeros((1, self.batch_size))
        idx = 0
        for feat_line in feats:
            for feat in feat_line:
                alphas_t = []  # The forward tensors at this timestep
                for next_tag in range(self.tagset_size):
                    emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size)

                    trans_score = self.transitions[next_tag].view(1, -1)

                    next_tag_var = forward_var + trans_score + emit_score

                    alphas_t.append(log_sum_exp(next_tag_var).view(1))

                forward_var = torch.cat(alphas_t).view(1, -1)

            terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
            alpha = log_sum_exp(terminal_var)
            result[0][idx] = alpha
            idx += 1
        return result

    def _score_sentence(self, feats, tags):
        # feats: [20, 8, 7] , tags: [8, 20]
        # Gives the score of a provided tag sequence
        score = torch.zeros(1)
        tags = torch.cat((torch.tensor(torch.full((self.batch_size, 1), \
                          self.tag_to_ix[START_TAG]), dtype=torch.long), tags), dim=1)

        feats = feats.transpose(1, 0)
        # feats: [8, 20, 7]
        idx = 0
        result = torch.zeros((1, self.batch_size))
        for feat_line in feats:
            for i, feat in enumerate(feat_line):
                score = score + self.transitions[tags[idx][i + 1], tags[idx][i]] + feat[tags[idx][i + 1]]
            score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[idx][-1]]
            result[0][idx] = score
            idx += 1
        return result

    def _viterbi_decode(self, feats):
        result_best_path = []
        feats = feats.transpose(1, 0)
        
        for feat_line in feats:
            backpointers = []

            # Initialize the viterbi variables in log space
            init_vvars = torch.full((1, self.tagset_size), -10000.)
            init_vvars[0][self.tag_to_ix[START_TAG]] = 0

            # forward_var at step i holds the viterbi variables for step i-1
            forward_var = init_vvars
            for feat in feat_line:
                bptrs_t = []  # holds the backpointers for this step
                viterbivars_t = []  # holds the viterbi variables for this step

                for next_tag in range(self.tagset_size):
                    # next_tag_var[i] holds the viterbi variable for tag i at the previous step, plus the score of transitioning from tag i to next_tag.
                    # We don't include the emission scores here because the max does not depend on them (we add them in below)
                    # 注意此处没有加发射矩阵分数, 因为求最大值不需要发射矩阵
                    next_tag_var = forward_var + self.transitions[next_tag]
                
                    best_tag_id = argmax(next_tag_var)
                    bptrs_t.append(best_tag_id)
                    viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            
                # Now add in the emission scores, and assign forward_var to the set of viterbi variables we just computed
                # 此处再将发射矩阵分数加上
                forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)  # forward_var : [1, 5]
                backpointers.append(bptrs_t)   # bptrs_t : [3, 3, 3, 3, 3], backpointers : [[3, 3, 3, 3, 3]]
            
            # Transition to STOP_TAG
            terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
            best_tag_id = argmax(terminal_var)
            path_score = terminal_var[0][best_tag_id]

            best_path = [best_tag_id]
            for bptrs_t in reversed(backpointers):
                best_tag_id = bptrs_t[best_tag_id]
                best_path.append(best_tag_id)
            start = best_path.pop()
            assert start == self.tag_to_ix[START_TAG]  # Sanity check
            best_path.reverse()
            result_best_path.append(best_path)

        return result_best_path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        # feats : [20, 8, 7] 代表每一个word映射到5个标签的概率, 发射矩阵
        # forward_score 代表公式推导中损失函数loss的第一项
        forward_score = self._forward_alg(feats)
        # print(forward_score)
        
        # gold_score 代表公式推导中损失函数loss的第二项
        gold_score = self._score_sentence(feats, tags)
        # print(gold_score)
        # 按行求和, 在torch.sum()函数值中, 需要设置dim=1 ; 同理, dim=0代表按列求和
        return torch.sum(forward_score - gold_score, dim=1)

    def forward(self, sentence):  # don't confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        tag_seq = self._viterbi_decode(lstm_feats)
        return tag_seq



# char_to_id = {"双": 0, "肺": 1, "见": 2, "多": 3, "发": 4, "斑": 5, "片": 6,
#              "状": 7, "稍": 8, "高": 9, "密": 10, "度": 11, "影": 12, "。": 13}

'''
model = BiLSTM(vocab_size=len(char_to_id),
               tag_to_ix=tag_to_ix,
               embedding_dim=EMBEDDING_DIM,
               hidden_dim=HIDDEN_DIM,
               num_layers=NUM_LAYERS,
	       batch_size=BATCH_SIZE,
	       sequence_length=SENTENCE_LENGTH)
'''

# print(model)


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

tag_list = [
    [0, 0, 3, 4, 0, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 3, 4, 0, 0, 0],
    [0, 0, 3, 4, 0, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 3, 4, 0, 0, 0],
    [0, 0, 3, 4, 0, 3, 4, 0, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [3, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 1, 2, 2, 2, 0, 0, 0, 0, 0],
    [0, 0, 1, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [3, 4, 0, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 3, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
]

tags = torch.tensor(tag_list, dtype=torch.long)


if __name__ == '__main__':
    for sentence in sentence_list:
        for _char in sentence:
            if _char not in char_to_id:
                char_to_id[_char] = len(char_to_id)
    sentence_sequence = sentence_map(sentence_list, char_to_id, SENTENCE_LENGTH)
    # print("sentence_sequence:\n", sentence_sequence)
    model = BiLSTM(vocab_size=len(char_to_id), tag_to_ix=tag_to_ix, embedding_dim=EMBEDDING_DIM, \
                   hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, batch_size=BATCH_SIZE, \
                   sequence_length=SENTENCE_LENGTH)

    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)
    # sentence_features = model._get_lstm_features(sentence_sequence)
    
    for epoch in range(1):
        model.zero_grad()

        feats = model._get_lstm_features(sentence_sequence)

        forward_score = model._forward_alg(feats)
        print(forward_score)

        # gold_score = model._score_sentence(feats, tags)
        # print(gold_score)

        # result_tags = model._viterbi_decode(feats)
        # print(result_tags)

        # loss = model.neg_log_likelihood(sentence_sequence, tags)
        # print(loss)

        # loss.backward()
        # optimizer.step()    

        # result = model(sentence_sequence)
        # print(result)

