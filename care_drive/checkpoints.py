"""Portable checkpoints; original state-dict parameter names are retained."""
import os
from copy import deepcopy
from pathlib import Path
import torch
from .assets import backbone_assets, save_backbone_assets


def save(path, model, config, *, optimizer=None, scheduler=None, epoch=0, step=0):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    value = {"model": model.state_dict(), "config": config,
             "epoch": epoch, "step": step}
    if hasattr(model, "tokenizer_identity"):
        value["contract"] = {"tokenizer": model.tokenizer_identity()}
        save_backbone_assets(model, path.parent / "assets")
    if optimizer is not None:
        value["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        value["scheduler"] = scheduler.state_dict()
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def load_backbone(path, config, device):
    from accelerate import init_empty_weights
    from .training.losses import L1Model
    assets = backbone_assets(path)
    config = deepcopy(config)
    config["model"]["vlm_path"] = str(assets)
    config["model"]["local_files_only"] = True
    # Keep rotary and other nonpersistent buffers real; all parameters come from
    # the CARE-Drive checkpoint without allocating a second initial weight set.
    with init_empty_weights(include_buffers=False):
        model = L1Model(config, load_pretrained=False)
    value = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    model.load_state_dict(value["model"], strict=True, assign=True)
    # Assignment materializes parameters independently; restore Qwen's declared
    # embedding/output sharing after leaving the empty-weight context.
    model.vlm.model.tie_weights()
    if "tokenizer" in value.get("contract", {}):
        if model.tokenizer_identity() != value["contract"]["tokenizer"]:
            raise ValueError("Checkpoint tokenizer identity differs from the configured backbone")
    return model.to(device).eval().requires_grad_(False)
