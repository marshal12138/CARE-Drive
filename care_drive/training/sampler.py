"""Full-epoch sharding with explicit masks for the last padded batch."""
import torch

def epoch_windows(size, seed, epoch, effective_batch, rank, world):
    if size < 1 or effective_batch < 1 or world < 1 or effective_batch % world or not 0 <= rank < world:
        raise ValueError('Invalid full-epoch sampler dimensions')
    order = torch.randperm(size, generator=torch.Generator().manual_seed(seed + epoch)).tolist()
    windows = []
    for offset in range(0, size, effective_batch):
        real = order[offset:offset + effective_batch]
        padded = [(index, 1) for index in real]
        padded.extend([(real[0], 0)] * (-len(real) % world))
        windows.append((len(real), padded[rank::world]))
    return windows

