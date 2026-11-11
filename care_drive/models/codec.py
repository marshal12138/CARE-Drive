"""Bounded metric XY residuals and reproducible group sampling."""
import torch

def position_labels(base, gt, ranges=(5.,3.)):
    residual=gt[...,:2].detach()-base[...,:2].detach()
    return (torch.round((residual/residual.new_tensor(ranges)).clamp(-1,1)*200).long()+200).flatten(-2)


def position_decode(base, ids, ranges=(5.,3.)):
    assert ids.shape[-1]==16
    return base[...,:2].detach().float()+(ids.reshape(*ids.shape[:-1],8,2).float()-200)/200*base.new_tensor(ranges)


def uniforms(seeds, length, candidates=8):
    # Same per-scene inverse-CDF draws as original G07, extended to16 for XY.
    values=[]
    for seed in seeds:
        generator=torch.Generator().manual_seed(seed)
        values.append(torch.stack([torch.rand((candidates,1),generator=generator) for _ in range(length)]))
    return torch.stack(values,1).flatten(1,2)

