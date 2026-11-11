"""One-query residual readout of the student's final 96 future hidden states."""
import torch
from torch import nn
import torch.nn.functional as F


def normalize_readout_config(config):
    model = config['model']
    raw = model.get('future_readout', {})
    if not isinstance(raw, dict) or set(raw) - {'enabled', 'dim', 'num_heads'}:
        raise ValueError('Invalid future_readout configuration')
    result = {'enabled': False, 'dim': 512, 'num_heads': 4, **raw}
    if type(result['enabled']) is not bool:
        raise ValueError('future_readout.enabled must be bool')
    if any(type(result[k]) is not int or result[k] <= 0 for k in ('dim', 'num_heads')):
        raise ValueError('Readout dimensions must be positive integers')
    if result['dim'] % result['num_heads']:
        raise ValueError('Readout dimension must divide evenly into heads')
    if result['enabled'] and (model.get('future_vision', True) is not True or model['num_jepa_tokens'] != 96):
        raise ValueError('Future readout requires 96 future positions')
    return result


class FutureReadout(nn.Module):
    def __init__(self, hidden_dim=2048, dim=512, num_heads=4):
        super().__init__()
        if any(type(v) is not int or v <= 0 for v in (hidden_dim, dim, num_heads)) or dim % num_heads:
            raise ValueError('Invalid readout dimensions')
        self.hidden_dim, self.dim, self.num_heads = hidden_dim, dim, num_heads
        self.query_norm = nn.LayerNorm(hidden_dim, eps=1e-5)
        self.memory_norm = nn.LayerNorm(hidden_dim, eps=1e-5)
        self.q_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, hidden_dim, bias=False)
        for layer in (self.q_proj, self.k_proj, self.v_proj):
            nn.init.xavier_uniform_(layer.weight)
        nn.init.zeros_(self.out_proj.weight)

    def forward(self, trajectory_hidden, future_hidden):
        if (trajectory_hidden.ndim != 2 or future_hidden.ndim != 3
                or trajectory_hidden.shape[1] != self.hidden_dim
                or future_hidden.shape != (trajectory_hidden.shape[0], 96, self.hidden_dim)):
            raise ValueError('Readout expects [B,H] and [B,96,H]')
        batch = trajectory_hidden.shape[0]
        width = self.dim // self.num_heads
        query = self.q_proj(self.query_norm(trajectory_hidden))
        memory = self.memory_norm(future_hidden)
        query = query.reshape(batch, 1, self.num_heads, width).transpose(1, 2)
        key = self.k_proj(memory).reshape(batch, 96, self.num_heads, width).transpose(1, 2)
        value = self.v_proj(memory).reshape(batch, 96, self.num_heads, width).transpose(1, 2)
        context = F.scaled_dot_product_attention(query, key, value, attn_mask=None,
                                                dropout_p=0.0, is_causal=False)
        return self.out_proj(context.transpose(1, 2).reshape(batch, self.dim))
