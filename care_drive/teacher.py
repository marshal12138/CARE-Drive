"""Frozen official V-JEPA2.1 video targets, following CoWorld's JEPA branch."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F

from .io import event, sha256


class JepaTeacher(nn.Module):
    def __init__(self, config, device='cpu'):
        super().__init__()
        sys.path.insert(0, str(Path(config['model']['teacher_source']).resolve()))
        from app.vjepa_2_1.models.vision_transformer import vit_gigantic_xformers
        cfg = config['model']
        checkpoint_path = Path(cfg['teacher_path'])
        expected_hash = cfg.get('teacher_sha256')
        if expected_hash and sha256(checkpoint_path) != expected_hash:
            raise ValueError('Teacher checkpoint SHA256 differs from configuration')
        self.pool = tuple(cfg['teacher_pool'])
        with torch.device('meta'):
            self.encoder = vit_gigantic_xformers(
                img_size=(384, 384), patch_size=16, num_frames=8, tubelet_size=2,
                use_sdpa=True, use_SiLU=False, wide_SiLU=True, uniform_power=False,
                use_rope=True, img_temporal_dim_size=1, interpolate_rope=True,
                n_output_distillation=1,
            )
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True, mmap=True)
        state = {}
        for name, value in checkpoint['encoder'].items():
            name = name.removeprefix('module.').removeprefix('backbone.')
            state[name] = value
        self.encoder.load_state_dict(state, strict=True, assign=True)
        self.encoder.requires_grad_(False).eval()
        self.encoder.to(device=device, dtype=torch.bfloat16)
        self.hidden_dim = self.encoder.embed_dim
        if self.hidden_dim != cfg['teacher_dim']:
            raise ValueError(f'Teacher width mismatch: {self.hidden_dim}')
        del checkpoint, state
        self.eval()
        event('teacher_loaded', dim=self.hidden_dim, dtype='bfloat16', strict=True, device=str(device))

    def train(self, mode=True):
        super().train(False)
        return self

    @staticmethod
    def prepare_images(paths, current_path=None):
        if len(paths) != 8:
            raise ValueError('Teacher requires eight images')
        if current_path is not None:
            paths = [current_path]
        frames = []
        for path in paths:
            with Image.open(path) as image:
                rgb = np.asarray(image.convert('RGB'), dtype=np.float32).copy() / 255.0
            frames.append(torch.from_numpy(rgb).permute(2, 0, 1))
        # Use CoWorld's final teacher resolution and ImageNet normalization.
        x = torch.stack(frames)
        x = F.interpolate(x, size=(384, 384), mode='bilinear', align_corners=False, antialias=True)
        if current_path is not None:
            x = x.expand(8, -1, -1, -1)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        return ((x - mean) / std).permute(1, 0, 2, 3).contiguous()

    @torch.inference_mode()
    def forward(self, clips):
        parameter = next(self.encoder.parameters())
        if clips.shape[1:] != (3, 8, 384, 384):
            raise ValueError(f'Invalid teacher clip: {clips.shape}')
        dense = self.encoder(clips.to(device=parameter.device, dtype=parameter.dtype), training=False)
        if dense.shape[1:] != (4 * 24 * 24, self.hidden_dim):
            raise ValueError(f'Unexpected dense teacher shape: {dense.shape}')
        grid = dense.reshape(-1, 4, 24, 24, self.hidden_dim).permute(0, 1, 4, 2, 3)
        pooled = F.adaptive_avg_pool2d(grid.flatten(0, 1).float(), self.pool[1:])
        result = pooled.reshape(-1, 4, self.hidden_dim, 4, 6).permute(0, 1, 3, 4, 2).reshape(-1, 96, self.hidden_dim)
        if not torch.isfinite(result).all():
            raise ValueError('Nonfinite teacher features')
        return result.to(torch.bfloat16)
