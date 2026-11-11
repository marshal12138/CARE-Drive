"""FP32 inverse-CDF categorical sampling."""
import torch

def choose(logits,greedy,generator=None):
    logits=logits.float()
    if greedy:return logits.argmax(-1)
    # Inverse CDF gives deterministic CPU-generated uniforms on any device;
    # sampling probabilities and replay both use FP32 normalized typed logits.
    u=torch.rand((len(logits),1),generator=generator,device='cpu').to(logits.device)
    cdf=logits.softmax(-1).cumsum(-1)
    return (u>cdf).sum(-1).clamp_max(logits.shape[-1]-1)

