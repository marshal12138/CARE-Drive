"""The three pre-existing recipes, each using the same real-candidate masks."""
import torch
import torch.distributed as dist

def local_terms(rewards,valid,mode,floor=.01,reference_rewards=None,greedy_reward=None):
    r=rewards.detach().float();valid=valid.detach().float();official=r if reference_rewards is None else reference_rewards.detach().float()
    centered=r-r.mean(1,keepdim=True)
    target=(official-official.mean(1,keepdim=True))/official.std(1,unbiased=False,keepdim=True).clamp_min(floor)
    if mode=='global_rms_matched':raw=centered
    elif mode=='scene_std':raw=centered/r.std(1,unbiased=False,keepdim=True).clamp_min(floor)
    elif mode=='greedy_baseline_rms_matched':
        assert greedy_reward is not None
        raw=(r-greedy_reward.detach().float()[:,None])/r.std(1,unbiased=False,keepdim=True).clamp_min(floor)
    else:raise ValueError(mode)
    raw=raw*valid[:,None];target=target*valid[:,None];centered=centered*valid[:,None]
    totals=torch.stack([centered.square().sum(),target.square().sum(),valid.sum()*r.shape[1],raw.square().sum()])
    return raw,totals

def finish(raw,totals,mode):
    if mode=='scene_std':scale=totals.new_ones(())
    else:scale=torch.where(totals[3]>0,(totals[1]/totals[3].clamp_min(1e-30)).sqrt(),totals.new_zeros(()))
    return raw*scale,dict(scale=scale,totals=totals)

@torch.no_grad()
def build_advantages(rewards,valid,mode,floor=.01,reference_rewards=None,greedy_reward=None,collective_device=None):
    raw,totals=local_terms(rewards,valid,mode,floor,reference_rewards,greedy_reward)
    reduced=totals.to(collective_device) if collective_device is not None else totals
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(reduced)
    return finish(raw,reduced.to(raw.device),mode)
