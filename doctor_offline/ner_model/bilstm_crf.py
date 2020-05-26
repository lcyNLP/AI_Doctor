import torch
import torch.nn as nn
import torch.optim as optim


# 添加几个辅助函数, 为log_sum_exp()服务
def to_scalar(var):
    # 返回一个python float类型的值
    return var.view(-1).data.tolist()[0]


def argmax(vec):
    # 返回列的维度上最大值的下标, 而且下标是一个标量float类型
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)


def log_sum_exp(vec):
    # 求向量中的最大值
    max_score = vec[0, argmax(vec)]
    # 构造一个最大值的广播变量
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    # 先减去最大值, 再求解log_sum_exp, 最终的返回值上再加上max_score
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))



# 函数sentence_map()完成中文文本信息的数字编码, 将中文语句变成数字化张量
def sentence_map(sentence_list, char_to_id, max_length):
    # 首先对一个批次的所有语句按照句子的长短进行排序, 这个操作并非必须
    sentence_list.sort(key=lambda x: len(x), reverse=True)
    # 定义一个最终存储结果特征张量的空列表
    sentence_map_list = []
    # 循环遍历一个批次内所有的语句
    for sentence in sentence_list:
        # 采用列表生成式来完成中文字符到id值的映射
        sentence_id_list = [char_to_id[c] for c in sentence]
        # 长度不够max_length的部分用0填充
        padding_list = [0] * (max_length - len(sentence))
        # 将每一个语句扩充为相同长度的张量
        sentence_id_list.extend(padding_list)
        # 追加进最终存储结果的列表中
        sentence_map_list.append(sentence_id_list)

    # 返回一个标量类型的张量
    return torch.tensor(sentence_map_list, dtype=torch.long)


class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim,
                       num_layers, batch_size, sequence_length):
        '''
        vocab_size:   单词总数量
        tag_to_ix:    标签到id的映射字典
        embedding_dim:  词嵌入的维度
        hidden_dim:    隐藏层的维度
        num_layers:    堆叠的LSTM层数
        batch_size:    批次的大小
        sequence_length:  语句的最大长度
        '''

        # 继承函数的初始化
        super(BiLSTM_CRF, self).__init__()
        # 设置单词的总数量
        self.vocab_size = vocab_size
        # 设置标签到id的映射字典
        self.tag_to_ix = tag_to_ix
        # 设置标签的总数
        self.tagset_size = len(tag_to_ix)
        # 设置词嵌入的维度
        self.embedding_dim = embedding_dim
        # 设置隐藏层的维度
        self.hidden_dim = hidden_dim
        # 设置LSTM层数
        self.num_layers = num_layers
        # 设置批次的大小
        self.batch_size = batch_size
        # 设置语句的长度
        self.sequence_length = sequence_length

        # 构建词嵌入层, 两个参数分别单词总数量, 词嵌入维度
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)

        # 构建双向LSTM层, 输入参数包括词嵌入维度, 隐藏层大小, LSTM层数, 是否双向标志
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=self.num_layers, bidirectional=True)

        # 构建全连线性层, 一端对接BiLSTM, 另一端对接输出层, 注意输出层维度是tagset_size
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # 初始化转移矩阵, 注意转移矩阵的维度[tagset_size, tagset_size]
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))

        # 任何合法的句子不会转移到"START_TAG"，设置为-10000
        # 任何合法的句子不会从"STOP_TAG"继续转移, 设置为-10000
        self.transitions.data[tag_to_ix["<START>"], :] = -10000
        self.transitions.data[:, tag_to_ix["<STOP>"]] = -10000

        # 初始化隐藏层, 利用类中的函数init_hidden()来完成
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # 为了符合LSTM的要求, 返回h0, c0, 这两个张量拥有相同的shape
        # shape: [2 * num_layers, batch_size, hidden_dim // 2]
        return (torch.randn(2 * self.num_layers, self.batch_size, self.hidden_dim // 2),
                torch.randn(2 * self.num_layers, self.batch_size, self.hidden_dim //2))

    # 在类中将文本信息经过词嵌入层, BiLSTM层, 线性层的处理, 最终输出句子的张量
    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()

        # 让sentence经历词嵌入层
        embeds = self.word_embeds(sentence).view(self.sequence_length, self.batch_size, -1)

        # 将词嵌入层的输出, 进入BiLSTM层, LSTM输入的两个参数: 词嵌入后的张量, 随机初始化的隐藏层张量
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)

        # 保证输出张量的形状:[sequence_length, batch_size, hidden_dim]
        lstm_out = lstm_out.view(self.sequence_length, self.batch_size, self.hidden_dim)

        # 最后经过线性层的处理, 得到最后输出张量的shape: [sequence_length, batch_size, tagset_size]
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats


    def _forward_alg(self, feats):
        # 初始化一个alphas张量, 代表转移矩阵的起始位置
        init_alphas = torch.full((1, self.tagset_size), -10000)
        # 仅仅将"START_TAG"赋值为0, 代表着接下来的矩阵转移只能从START_TAG开始
        init_alphas[0][self.tag_to_ix["<START>"]] = 0

        # 将初始化的init_alphas赋值为前向计算变量, 为了后续在反向传播求导的时候可以自动更新参数
        forward_var = init_alphas

        # 输入进来的feats - shape:[20, 8, 7], 为了后续按句子为单位进行计算, 需要将batch_size放在第一个维度上
        feats = feats.transpose(1, 0)

        # 初始化一个最终的结果张量
        result = torch.zeros((1, self.batch_size))
        idx = 0

        # 遍历每一行文本, 总共循环batch_size次
        for feat_line in feats:
            # feats: [8, 20, 7], feat_line: [20, 7]
            # 遍历每一行, 每一个feat代表一个time_step
            for feat in feat_line:
                # 当前的time_step，初始化一个前向计算张量
                alphas_t = []
                # 每一个时间步, 遍历所有可能的转移标签, 进行累加计算
                for next_tag in range(self.tagset_size):
                    # 构造发射分数的广播张量
                    emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size)

                    # 当前时间步, 转移到next_tag标签的转移分数
                    trans_score = self.transitions[next_tag].view(1, -1)

                    # 将前向计算矩阵, 发射矩阵, 转移矩阵累加
                    next_tag_var = forward_var + trans_score + emit_score

                    # 计算log_sum_exp()的值, 并添加进alphas_t列表中
                    alphas_t.append(log_sum_exp(next_tag_var).view(1))

                # 将列表张量转换为二维张量
                forward_var = torch.cat(alphas_t).view(1, -1)

            # 添加最后一步转移到"STOP_TAG"的分数, 就完成了整条语句的分数计算
            terminal_var = forward_var + self.transitions[self.tag_to_ix["<STOP>"]]

            # 将terminal_var放进log_sum_exp()中进行计算, 得到一条样本语句最终的分数
            alpha = log_sum_exp(terminal_var)
            # 将得分添加进最终的结果列表中, 作为整个函数的返回结果
            result[0][idx] = alpha
            idx += 1
        return result


    def _score_sentence(self, feats, tags):
        '''
        feats: [20, 8, 7], 经历了_get_lstm_features()处理后的特征张量
        tags: [8, 20], 代表的是训练语句真实的标签矩阵
        '''
        # 初始化一个0值的tensor，为后续的累加做准备
        score = torch.zeros(1)
        # 要在tags矩阵的第一列添加,这一列全部都是START_TAG
        temp = torch.tensor(torch.full((self.batch_size, 1), self.tag_to_ix["<START>"]), dtype=torch.long)
        tags = torch.cat((temp, tags), dim=1)

        # 将传入的feats形状转变为[batch_size, sequence_length, tagset_size]
        feats = feats.transpose(1, 0)

        # 初始化最终的结果分数张量, 每一个句子得到一个分数
        result = torch.zeros((1, self.batch_size))
        idx = 0
        # 遍历所有的语句特征向量
        for feat_line in feats:
            # 此处feat_line: [20, 7]
            # 遍历每一个时间步, 注意: 最重要的区别在于这里是在真实标签tags的指导下进行的转移矩阵和发射矩阵的累加分数求和
            for i, feat in enumerate(feat_line):
                score = score + self.transitions[tags[idx][i+1], tags[idx][i]] + feat[tags[idx][i+1]]
            # 遍历完当前语句所有的时间步之后, 最后添加上"STOP_TAG"的转移分数
            score = score + self.transitions[self.tag_to_ix["<STOP>"], tags[idx][-1]]
            # 将该条语句的最终得分添加进结果列表中
            result[0][idx] = score
            idx += 1
        return result


    def _viterbi_decode(self, feats):
        # 根据传入的语句特征feats，推断出标签序列
        # 初始化一个最佳路径结果的存放列表
        result_best_path = []
        # 将输入的张量形状变为 [batch_size, sequence_length, tagset_size]
        feats = feats.transpose(1, 0)

        # 对批次中的每一个语句进行遍历, 每个语句产生一个最优的标注序列
        for feat_line in feats:
            backpointers = []

            # 初始化前向传播的张量, 同时设置START_TAG等于0, 约束了合法的序列只能从START_TAG开始
            init_vvars = torch.full((1, self.tagset_size), -10000)
            init_vvars[0][self.tag_to_ix["<START>"]] = 0

            # 将初始化的变量赋值给forward_var, 在第i个time_step中, 张量forward_var保存的是第i-1个time_step的viterbi张量
            forward_var = init_vvars

            # 遍历从i=0, 到序列最后一个time_step, 每一个时间步
            for feat in feat_line:
                # 初始化保存当前time_step的回溯指针
                bptrs_t = []
                # 初始化保存当前tme_step的viterbi变量
                viterbivars_t = []

                # 遍历所有可能的转移标签
                for next_tag in range(self.tagset_size):
                    # next_tag_var[i]保存了tag_i在前一个time_step的viterbi变量
                    # 通过前向传播张量forward_var加上从tag_i转移到next_tag的转移分数, 赋值给next_tag_var
                    # 注意: 在这里不去加发射矩阵的分数, 因为发射矩阵分数一致, 不影响求最大值下标
                    next_tag_var = forward_var + self.transitions[next_tag]

                    # 将最大的标签所对应的id加入到当前time_step的回溯列表中
                    best_tag_id = argmax(next_tag_var)
                    bptrs_t.append(best_tag_id)
                    viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))

                # 此处再将发射矩阵的分数feat添加上来, 继续赋值给forward_var, 作为下一个time_step的前向传播变量
                forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)

                # 将当前time_step的回溯指针添加进当前样本行的总体回溯指针中
                backpointers.append(bptrs_t)

            # 最后加上转移到STOP_TAG的分数
            terminal_var = forward_var + self.transitions[self.tag_to_ix["<STOP>"]]
            best_tag_id = argmax(terminal_var)
            
            # 根据回溯指针, 解码最佳路径
            best_path = [best_tag_id]
            # 从后向前回溯最佳路径
            for bptrs_t in reversed(backpointers):
                # 通过第i个time_step得到的最佳id, 找到第i-1个time_step的最佳id
                best_tag_id = bptrs_t[best_tag_id]
                best_path.append(best_tag_id)

            # 将START_TAG去除掉
            start = best_path.pop()
            # print(start)
            # 确认一下最佳路径的第一个标签是START_TAG
            # if start != self.tag_to_ix["<START>"]:
            #     print(start)
            assert start == self.tag_to_ix["<START>"]

            # 因为是从后向前进行回溯, 所以在此对列表进行逆序操作得到从前向后的真实路径
            best_path.reverse()
            # 将当前这一行的样本结果添加到最终的结果列表中
            result_best_path.append(best_path)

        return result_best_path


    # 对数似然函数, 输入两个参数: 数字化编码后的张量, 和真实的标签
    # 注意: 这个函数是未来真实训练中要用到的损失函数, 虚拟化的forward()
    def neg_log_likelihood(self, sentence, tags):
        # 第一步先得到BiLSTM层的输出特征张量
        feats = self._get_lstm_features(sentence)

        # feats: [20, 8, 7]代表一个批次8个样本, 每个样本长度20, 每一个字符映射成7个标签
        # feats本质上就是发射矩阵
        # forward_score, 代表损失函数中的第一项
        forward_score = self._forward_alg(feats)

        # gold_score, 代表损失函数中的第二项
        gold_score = self._score_sentence(feats, tags)

        # 注意: 未来通过forward_score和gold_score的差值作为loss，进行梯度下降的优化求解
        # 按行求和的时候, 在torch.sum()函数中, 需要设置dim=1；同理, 如果要按列求和, 需要设置dim=0
        return torch.sum(forward_score - gold_score, dim=1)


    # 编写正式的forward()函数, 注意应用场景是在预测的时候, 模型训练的时候并没有用到forward()函数
    def forward(self, sentence):
        # 首先获取BiLSTM层的输出特征, 得到发射矩阵
        lstm_feats = self._get_lstm_features(sentence)

        # 通过维特比算法直接解码出最优路径
        result_sequence = self._viterbi_decode(lstm_feats)
        return result_sequence



# 开始字符和结束字符
START_TAG = "<START>"
STOP_TAG = "<STOP>"
# 标签和序号的对应码表
tag_to_ix = {"O": 0, "B-dis": 1, "I-dis": 2, "B-sym": 3, "I-sym": 4, START_TAG: 5, STOP_TAG: 6}
# 词嵌入的维度
EMBEDDING_DIM = 200
# 隐藏层神经元的数量
HIDDEN_DIM = 100
# 批次的大小
BATCH_SIZE = 8
# 设置最大语句限制长度
SENTENCE_LENGTH = 100
# 默认神经网络的层数
NUM_LAYERS = 1
# 初始化的字符和序号的对应码表
# char_to_id = {"双": 0, "肺": 1, "见": 2, "多": 3, "发": 4, "斑": 5, "片": 6,
#               "状": 7, "稍": 8, "高": 9, "密": 10, "度": 11, "影": 12, "。": 13}

'''
model = BiLSTM_CRF(vocab_size=len(char_to_id), tag_to_ix=tag_to_ix, embedding_dim=EMBEDDING_DIM,
                   hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, batch_size=BATCH_SIZE, sequence_length=SENTENCE_LENGTH)

print(model)
'''

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


# 真实标签数据, 对应为tag_to_ix中的数字标签
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
# 将标签转为标量tags
tags = torch.tensor(tag_list, dtype=torch.long)


char_to_id = {"<PAD>": 0}

if __name__ == '__main__':
    for sentence in sentence_list:
        for c in sentence:
            # 如果当前字符不在映射字典中, 追加进字典
            if c not in char_to_id:
                char_to_id[c] = len(char_to_id)

    # 首先利用char_to_id完成中文文本的数字化编码
    sentence_sequence = sentence_map(sentence_list, char_to_id, SENTENCE_LENGTH)
    # print("sentence_sequence:\n", sentence_sequence)

    # 构建类的实例, 去得到语句的特征张量
    model = BiLSTM_CRF(vocab_size=len(char_to_id), tag_to_ix=tag_to_ix, embedding_dim=EMBEDDING_DIM,
                       hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, batch_size=BATCH_SIZE,
                       sequence_length=SENTENCE_LENGTH)

    # 调用类内部的_get_lstm_features()函数, 得到特征张量
    # sentence_features = model._get_lstm_features(sentence_sequence)
    # print("sentence_features:\n", sentence_features)

    # 定义优化器
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

    for epoch in range(1):
        model.zero_grad()

        # feats = model._get_lstm_features(sentence_sequence)

        # forward_score = model._forward_alg(feats)
        # print(forward_score)

        # gold_score = model._score_sentence(feats, tags)
        # print(gold_score)

        # result_tag = model._viterbi_decode(feats)
        # print(result_tag)

        loss = model.neg_log_likelihood(sentence_sequence, tags)
        print(loss)

        loss.backward()
        optimizer.step()

        result = model(sentence_sequence)
        print(result)

