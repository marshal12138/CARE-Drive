"""Categorical adjacent-heading expert; original XY and old heading are not learned."""
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from .sampling import choose

CLASSES=401
BOS=401
HORIZON=8
MAX_PARAMETERS=300_000_000

class Attention(nn.Module):
    def __init__(self,width,heads):
        super().__init__();self.width=width;self.heads=heads
        self.q=nn.Linear(width,width);self.k=nn.Linear(width,width);self.v=nn.Linear(width,width);self.out=nn.Linear(width,width)
    def split(self,x):return x.reshape(x.shape[0],x.shape[1],self.heads,self.width//self.heads).transpose(1,2)
    def kv(self,x):return self.split(self.k(x)),self.split(self.v(x))
    def attend(self,q,kv,causal=False):
        h=F.scaled_dot_product_attention(self.split(self.q(q)),*kv,dropout_p=0.,is_causal=causal)
        return self.out(h.transpose(1,2).reshape(q.shape[0],q.shape[1],self.width))

class Block(nn.Module):
    def __init__(self,width,heads,ffn):
        super().__init__();self.self_attention=Attention(width,heads);self.cross_attention=Attention(width,heads)
        self.norm1=nn.LayerNorm(width);self.norm2=nn.LayerNorm(width);self.norm3=nn.LayerNorm(width)
        self.ff=nn.Sequential(nn.Linear(width,ffn),nn.GELU(),nn.Linear(ffn,width))
    def full(self,x,memory):
        h=self.norm1(x);x=x+self.self_attention.attend(h,self.self_attention.kv(h),True)
        x=x+self.cross_attention.attend(self.norm2(x),self.cross_attention.kv(memory))
        return x+self.ff(self.norm3(x))
    def step(self,x,old,cross):
        h=self.norm1(x);k,v=self.self_attention.kv(h)
        if old is not None:k,v=torch.cat([old[0],k],2),torch.cat([old[1],v],2)
        x=x+self.self_attention.attend(h,(k,v));x=x+self.cross_attention.attend(self.norm2(x),cross)
        return x+self.ff(self.norm3(x)),(k,v)

def parameter_count(architecture):
    d=architecture['width'];f=architecture['ffn'];n=architecture['layers']
    state=architecture.get('heading_state_features',False)
    if type(state) is not bool:raise ValueError('heading_state_features must be boolean')
    return n*(8*d*d+2*d*f+15*d+f)+d*d+2881*d+401+(2*d if state else 0)

def heading_state_features(increment_units):
    """Prefix sum in tenths of a degree, derived only from preceding tokens."""
    radians=increment_units.float()*(math.pi/1800.)
    return torch.stack([torch.sin(radians),torch.cos(radians)-1.],-1)

class HeadingExpert(nn.Module):
    def __init__(self,architecture):
        super().__init__();self.architecture=dict(architecture)
        d=architecture['width'];heads=architecture['heads'];f=architecture['ffn'];n=architecture['layers']
        assert d>0 and n>0 and d%heads==0
        if parameter_count(architecture)>MAX_PARAMETERS:raise ValueError('Expert parameter cap0.3B exceeded')
        self.scene_projection=nn.Linear(2048,d)
        self.xy_projection=nn.Sequential(nn.Linear(5,d),nn.GELU(),nn.Linear(d,d))
        self.memory_type=nn.Embedding(2,d);self.memory_position=nn.Embedding(HORIZON,d)
        self.memory_norm=nn.LayerNorm(d);self.token_embedding=nn.Embedding(CLASSES+1,d)
        self.position_embedding=nn.Embedding(HORIZON,d)
        self.heading_state_projection=nn.Linear(2,d,bias=False) if architecture.get('heading_state_features',False) else None
        if self.heading_state_projection is not None:nn.init.zeros_(self.heading_state_projection.weight)
        self.blocks=nn.ModuleList([Block(d,heads,f) for _ in range(n)])
        self.output_norm=nn.LayerNorm(d);self.output=nn.Linear(d,CLASSES)
        assert sum(p.numel() for p in self.parameters())==parameter_count(architecture)<=MAX_PARAMETERS

    def memory(self,scene,xy):
        assert scene.shape[1:]==(8,2048) and xy.shape[1:]==(8,2)
        # Detach is an additional barrier; the frozen baseline is not in this module.
        scene=scene.detach().float();xy=xy.detach().float()
        delta=xy-torch.cat([torch.zeros_like(xy[:,:1]),xy[:,:-1]],1)
        t=torch.arange(1,9,device=xy.device,dtype=xy.dtype).reshape(1,8,1).expand(len(xy),-1,-1)/8
        values=torch.cat([xy/xy.new_tensor([50.,20.]),delta/10.,t],-1)
        positions=self.memory_position(torch.arange(8,device=xy.device))[None]
        a=self.scene_projection(scene)+positions+self.memory_type.weight[0]
        p=self.xy_projection(values)+positions+self.memory_type.weight[1]
        return self.memory_norm(torch.cat([a,p],1))

    def full(self,inputs,memory):
        x=self.token_embedding(inputs)+self.position_embedding(torch.arange(inputs.shape[1],device=inputs.device))[None]
        if self.heading_state_projection is not None:
            units=torch.where(inputs==BOS,torch.zeros_like(inputs),inputs-200).cumsum(-1)
            x=x+self.heading_state_projection(heading_state_features(units))
        for block in self.blocks:x=block.full(x,memory)
        return self.output(self.output_norm(x)).float()

    def forward(self,scene,xy,targets):
        assert targets.shape[1]==HORIZON
        inputs=torch.cat([torch.full_like(targets[:,:1],BOS),targets[:,:-1]],1)
        return self.full(inputs,self.memory(scene,xy))

    @torch.no_grad()
    def generate(self,scene,xy,greedy=True,generator=None,return_logp=False):
        memory=self.memory(scene,xy);cross=[b.cross_attention.kv(memory) for b in self.blocks]
        cache=[None]*len(self.blocks);previous=torch.full((len(scene),),BOS,device=scene.device,dtype=torch.long)
        cumulative_units=torch.zeros_like(previous)
        ids=[];scores=[]
        for t in range(HORIZON):
            x=self.token_embedding(previous)[:,None]+self.position_embedding.weight[t][None,None]
            if self.heading_state_projection is not None:
                x=x+self.heading_state_projection(heading_state_features(cumulative_units))[:,None]
            for k,block in enumerate(self.blocks):x,cache[k]=block.step(x,cache[k],cross[k])
            logits=self.output(self.output_norm(x[:,0])).float();previous=choose(logits,greedy,generator)
            ids.append(previous);scores.append(F.log_softmax(logits,-1).gather(1,previous[:,None])[:,0])
            cumulative_units=cumulative_units+previous-200
        tokens=torch.stack(ids,1)
        return (tokens,torch.stack(scores,1)) if return_logp else tokens

def statistics(logits,targets):
    logdist=F.log_softmax(logits.float(),-1)
    logp=logdist.gather(-1,targets[...,None])[...,0]
    entropy=-(logdist.exp()*logdist).sum(-1)
    return logp,entropy,logdist

def grpo_objective(logits,reference,targets,old_logp,advantage,epsilon=.2,beta=.04):
    logp,entropy,logdist=statistics(logits,targets);ratio=(logp-old_logp).exp();a=advantage[:,None]
    surrogate=torch.minimum(ratio*a,ratio.clamp(1-epsilon,1+epsilon)*a)
    ref=F.log_softmax(reference.detach().float(),-1);kl=(logdist.exp()*(logdist-ref)).sum(-1)
    return (-surrogate+beta*kl).mean(1),dict(kl=kl.mean(),token_kl=kl.detach(),entropy=entropy.mean(),ratio_error=(ratio-1).abs().max())

@torch.no_grad()
def update_statistics(old_logdist,new_logits,targets,old_logp,epsilon):
    """Actual policy movement after an optimizer step, at the sampled prefixes."""
    logp,_,new_logdist=statistics(new_logits,targets)
    ratio=(logp-old_logp).exp()
    old_kl=(old_logdist.exp()*(old_logdist-new_logdist)).sum(-1)
    return dict(old_policy_KL=old_kl.mean(1),
                clipped_token_fraction=((ratio<1-epsilon)|(ratio>1+epsilon)).float().mean(1),
                mean_absolute_log_ratio=(logp-old_logp).abs().mean(1))

def encode_headings(headings):
    h=np.asarray(headings,dtype=np.float64);previous=np.concatenate([np.zeros((*h.shape[:-1],1)),h[...,:-1]],-1)
    delta=np.rad2deg(np.arctan2(np.sin(h-previous),np.cos(h-previous)))
    return np.rint(np.clip(delta,-20,20)*10).astype(np.int64)+200,np.abs(delta)>20

def decode(xy,ids):
    points=xy.detach().float().cpu().numpy() if torch.is_tensor(xy) else np.asarray(xy,dtype=np.float32)
    tokens=ids.detach().cpu().numpy() if torch.is_tensor(ids) else np.asarray(ids)
    assert points.shape[-2:]==(8,2) and tokens.shape==points.shape[:-1]
    assert np.issubdtype(tokens.dtype,np.integer) and np.all((tokens>=0)&(tokens<CLASSES))
    h=np.deg2rad((tokens.astype(np.float64)-200)/10).cumsum(-1)
    h=np.arctan2(np.sin(h),np.cos(h)).astype(np.float32)
    output=np.concatenate([points,h[...,None]],-1)
    assert np.array_equal(output[...,:2],points)
    return output
