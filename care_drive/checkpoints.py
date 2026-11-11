"""Portable checkpoints; original state-dict parameter names are retained."""
import os
from pathlib import Path
import torch


def save(path, model, config, *, optimizer=None, scheduler=None, epoch=0, step=0):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    value = {"model": model.state_dict(), "config": config,
             "epoch": epoch, "step": step}
    if hasattr(model, "tokenizer_identity"):
        value["contract"] = {"tokenizer": model.tokenizer_identity()}
    if optimizer is not None:
        value["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        value["scheduler"] = scheduler.state_dict()
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def load_backbone(path, config, device):
    from .training.losses import L1Model
    model = L1Model(config)
    value = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    model.load_state_dict(value["model"], strict=True)
    if "tokenizer" in value.get("contract", {}):
        if model.tokenizer_identity() != value["contract"]["tokenizer"]:
            raise ValueError("Checkpoint tokenizer identity differs from the configured backbone")
    return model.to(device).eval().requires_grad_(False)
