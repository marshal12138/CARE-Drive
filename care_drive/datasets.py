"""Manifest and frozen-feature inputs for backbone and expert training."""
import json
from pathlib import Path
import numpy as np
import torch
from .data import validate_observations
from .models.heading import encode_headings


def read_manifest(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    ids = [row["sample_id"] for row in rows]
    if not rows or len(ids) != len(set(ids)):
        raise ValueError("A manifest must contain unique, nonempty scene IDs")
    for row in rows:
        validate_observations(row["observations"])
        if "targets" in row:
            target = np.asarray(row["targets"]["trajectory"])
            if target.shape != (8, 3) or not np.isfinite(target).all():
                raise ValueError("Expected finite target trajectories [8,3]")
    return rows


class BackboneScenes:
    def __init__(self, manifest, sensor_root, teacher_root, config):
        from .backbone import ObservationProcessor
        self.rows = read_manifest(manifest)
        self.sensor_root, self.teacher_root = Path(sensor_root), Path(teacher_root)
        self.processor = ObservationProcessor(config)

    def __len__(self):
        return len(self.rows)

    def batch(self, indices, device):
        from safetensors.torch import load_file
        selected = [self.rows[i] for i in indices]
        inputs = self.processor([row["observations"] for row in selected], [self.sensor_root] * len(selected))
        features = []
        for row in selected:
            token = row["sample_id"]
            if not token.isalnum():
                raise ValueError("Scene IDs used in cache filenames must be alphanumeric")
            value = load_file(str(self.teacher_root / token[:2] / (token + ".safetensors")))["teacher_features"]
            if value.shape != (96, 1664) or not torch.isfinite(value).all():
                raise ValueError("Invalid future teacher cache")
            features.append(value)
        targets = {"trajectory": torch.tensor([row["targets"]["trajectory"] for row in selected], dtype=torch.float32, device=device),
                   "teacher_features": torch.stack(features).to(device)}
        return {key: value.to(device) for key, value in inputs.items()}, targets


class ExpertScenes:
    def __init__(self, path):
        value = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        self.tokens = value["tokens"]
        n = len(self.tokens)
        if n == 0 or len(set(self.tokens)) != n:
            raise ValueError("Expected a nonempty unique feature-cache scene list")
        self.scene, self.anchor, self.gt = value["scene"], value["anchor"], value["gt"]
        for tensor, shape in [(self.scene, (n, 8, 2048)), (self.anchor, (n, 8, 3)), (self.gt, (n, 8, 3))]:
            if tensor.shape != shape or not torch.isfinite(tensor).all():
                raise ValueError("Invalid expert cache tensor")
        self.metric_paths = value.get("metric_cache_paths", [None] * n)
        if len(self.metric_paths) != n:
            raise ValueError("Metric-cache paths must align with scene IDs")
        self.headings = torch.from_numpy(encode_headings(self.gt[..., 2].numpy())[0])
        self.fixed_xy = value.get("fixed_xy")
        self.metadata = value.get("metadata", {})

    def __len__(self):
        return len(self.tokens)
