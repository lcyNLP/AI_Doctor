import numpy as np
import torch
import torch.utils.data as Data


def load_dataset(data_file, batch_size):
    data = np.load(data_file)

    x_data = data['x_data']
    y_data = data['y_data']

    x = torch.tensor(x_data, dtype=torch.long)
    y = torch.tensor(y_data, dtype=torch.long)

    dataset = Data.TensorDataset(x, y)

    total_length = len(dataset)
    print(total_length)

    train_length = int(total_length * 0.8)
    validation_length = total_length - train_length

    train_dataset, validation_dataset = Data.random_split(dataset=dataset, 
                                        lengths=[train_length, validation_length])

    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                   shuffle=True, num_workers=4, drop_last=True)

    validation_loader = Data.DataLoader(dataset=validation_dataset, batch_size=batch_size,
                                        shuffle=True, num_workers=4, drop_last=True)

    data_loaders = {'train': train_loader, 'validation': validation_loader}

    data_size = {'train': train_length, 'validation': validation_length}

    return data_loaders, data_size


BATCH_SIZE = 16

DATA_FILE = './data/train.npz'

if __name__ == '__main__':
    data_loader, data_size = load_dataset(DATA_FILE, BATCH_SIZE)

    print('data_loader:', data_loader, '\ndata_size:', data_size)

