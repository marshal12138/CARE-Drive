"""Waypoint residual decoder used by the CARE-Drive position expert."""
import torch
from torch import nn
from torch.nn import functional as F
from .heading import Block
CLASSES, BOS, ZERO = 401, 401, 200

class ResidualExpert(nn.Module):
    def __init__(self, architecture, kind):
        super().__init__()
        assert kind in ('position', 'heading')
        self.kind = kind
        self.length = 16 if kind == 'position' else 8
        d, heads, f, n = (architecture[k] for k in ('width', 'heads', 'ffn', 'layers'))
        self.scene_projection = nn.Linear(2048, d)
        self.base_projection = nn.Sequential(nn.Linear(7, d), nn.GELU(), nn.Linear(d, d))
        self.corrected_projection = nn.Sequential(nn.Linear(5, d), nn.GELU(), nn.Linear(d, d)) if kind == 'heading' else None
        self.memory_type = nn.Embedding(3 if kind == 'heading' else 2, d)
        self.memory_position = nn.Embedding(8, d)
        self.memory_norm = nn.LayerNorm(d)
        self.token_embedding = nn.Embedding(CLASSES + 1, d)
        self.time_embedding = nn.Embedding(8, d)
        self.axis_embedding = nn.Embedding(2 if kind == 'position' else 1, d)
        self.blocks = nn.ModuleList([Block(d, heads, f) for _ in range(n)])
        self.output_norm = nn.LayerNorm(d)
        self.output = nn.Linear(d, CLASSES)

    def memory(self, scene, base, corrected_xy=None):
        assert scene.shape[1:] == (8, 2048) and base.shape[1:] == (8, 3)
        scene, base = scene.detach().float(), base.detach().float()
        xy = base[..., :2]
        delta = xy - torch.cat([torch.zeros_like(xy[:, :1]), xy[:, :-1]], 1)
        time = torch.arange(1, 9, device=xy.device).reshape(1, 8, 1).expand(len(xy), -1, -1) / 8
        positions = self.memory_position.weight[None]
        angle = base[..., 2:3]
        values = torch.cat([xy / xy.new_tensor([50., 20.]), delta / 10., angle.sin(), angle.cos() - 1, time], -1)
        memory = [self.scene_projection(scene) + positions + self.memory_type.weight[0],
                  self.base_projection(values) + positions + self.memory_type.weight[1]]
        if self.kind == 'heading':
            assert corrected_xy is not None and corrected_xy.shape == xy.shape
            xy = corrected_xy.detach().float()
            delta = xy - torch.cat([torch.zeros_like(xy[:, :1]), xy[:, :-1]], 1)
            values = torch.cat([xy / xy.new_tensor([50., 20.]), delta / 10., time], -1)
            memory.append(self.corrected_projection(values) + positions + self.memory_type.weight[2])
        return self.memory_norm(torch.cat(memory, 1))

    def positions(self, device):
        index = torch.arange(self.length, device=device)
        t = index // 2 if self.kind == 'position' else index
        axis = index % 2 if self.kind == 'position' else torch.zeros_like(index)
        return self.time_embedding(t) + self.axis_embedding(axis)

    def forward(self, scene, base, targets, corrected_xy=None):
        assert targets.shape[1] == self.length
        inputs = torch.cat([torch.full_like(targets[:, :1], BOS), targets[:, :-1]], 1)
        x = self.token_embedding(inputs) + self.positions(inputs.device)[None]
        memory = self.memory(scene, base, corrected_xy)
        for block in self.blocks:
            x = block.full(x, memory)
        return self.output(self.output_norm(x)).float()

    @torch.no_grad()
    def generate(self, scene, base, corrected_xy=None, uniforms=None):
        memory = self.memory(scene, base, corrected_xy)
        cross = [b.cross_attention.kv(memory) for b in self.blocks]
        cache = [None] * len(self.blocks)
        previous = torch.full((len(scene),), BOS, device=scene.device, dtype=torch.long)
        positions = self.positions(scene.device)
        ids, logp = [], []
        if uniforms is not None:
            assert uniforms.shape == (self.length, len(scene), 1)
        for t in range(self.length):
            x = self.token_embedding(previous)[:, None] + positions[t][None, None]
            for k, block in enumerate(self.blocks):
                x, cache[k] = block.step(x, cache[k], cross[k])
            logits = self.output(self.output_norm(x[:, 0])).float()
            previous = logits.argmax(-1) if uniforms is None else (uniforms[t] > logits.softmax(-1).cumsum(-1)).sum(-1).clamp_max(CLASSES - 1)
            ids.append(previous)
            logp.append(F.log_softmax(logits, -1).gather(1, previous[:, None])[:, 0])
        return torch.stack(ids, 1), torch.stack(logp, 1)

