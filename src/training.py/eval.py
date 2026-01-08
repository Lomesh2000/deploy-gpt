import torch
from data.dataset import get_batch

@torch.no_grad()
def estimate_loss(model, device, train_data, val_data, block_size, batch_size, eval_iters):
    out = {}
    model.eval()

    for split, data in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, device, train_data, val_data, block_size, batch_size)
            _, loss = model(X, Y)
            losses[k] = loss.item()

        out[split] = losses.mean().item()
    
    model.train()
    return out

