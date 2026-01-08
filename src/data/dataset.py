import torch

def get_batch(split, device, train_data, val_data, block_size, batch_size):
    data = train_data if split == 'train' else val_data
    idx = torch.randint(len(data) - block_size, (batch_size, ))
    x = torch.stack([data[i : i + block_size] for i in idx])
    y = torch.stack([data[i + 1: i + block_size + 1] for i in idx])
    x = x.to(device)
    y = y.to(device)
    return x, y