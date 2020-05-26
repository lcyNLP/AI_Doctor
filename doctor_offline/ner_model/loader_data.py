# 导入包
import numpy as np
import torch
import torch.utils.data as Data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建生成批量训练数据的函数
def load_dataset(data_file, batch_size):
    '''
    data_file: 代表待处理的文件
    batch_size: 代表每一个批次样本的数量
    '''
    # 将train.npz文件带入到内存中
    data = np.load(data_file)

    # 分别提取data中的特征和标签
    x_data = data['x_data']
    y_data = data['y_data']

    # 将数据封装成Tensor张量
    x = torch.tensor(x_data, dtype=torch.long).to(device)
    y = torch.tensor(y_data, dtype=torch.long).to(device)

    # 将数据再次封装
    dataset = Data.TensorDataset(x, y).to(device)

    # 求解一下数据的总量
    total_length = len(dataset)

    # 确认一下将80%的数据作为训练集, 剩下的20%的数据作为测试集
    train_length = int(total_length * 0.8)
    validation_length = total_length - train_length

    # 利用Data.random_split()直接切分数据集, 按照80%, 20%的比例进行切分
    train_dataset, validation_dataset = Data.random_split(dataset=dataset, lengths=[train_length, validation_length])

    # 将训练数据集进行DataLoader封装
    # dataset: 代表训练数据集
    # batch_size: 代表一个批次样本的数量, 若数据集的总样本数无法被batch_size整除, 则最后一批数据的大小为余数, 
    #             若设置另一个参数drop_last=True, 则自动忽略最后不能被整除的数量
    # shuffle: 是否每隔批次为随机抽取, 若设置为True, 代表每个批次的数据样本都是从数据集中随机抽取的
    # num_workers: 设置有多少子进程负责数据加载, 默认为0, 即数据将被加载到主进程中
    # drop_last: 是否把最后一个批次的数据(指那些无法被batch_size整除的余数数据)忽略掉
    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                   shuffle=True, num_workers=2, drop_last=False).to(device)

    validation_loader = Data.DataLoader(dataset=validation_dataset, batch_size=batch_size,
                                        shuffle=True, num_workers=2, drop_last=False).to(device)

    # 将两个数据生成器封装成一个字典类型
    data_loaders = {'train': train_loader, 'validation': validation_loader}

    # 将两个数据集的长度也封装成一个字典类型
    data_size = {'train': train_length, 'validation': validation_length}

    return data_loaders, data_size


# 批次的大小
BATCH_SIZE = 32

# 训练数据集的文件路径
DATA_FILE = './data/total.npz'

if __name__ == '__main__':
    data_loader, data_size = load_dataset(DATA_FILE, BATCH_SIZE)
    print('data_loader:', data_loader, '\ndata_size:', data_size)

