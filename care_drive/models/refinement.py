"""Position residual correction followed by position-conditioned headings."""
import math
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from .position import ResidualExpert
from .heading import HeadingExpert, heading_state_features, decode
from .codec import position_decode


@torch.no_grad()
def sample_heading(model, scene, xy, uniforms=None):
    """Cached decoding with externally supplied inverse-CDF draws [8,B,1]."""
    if uniforms is None:
        return model.generate(scene, xy, greedy=True, return_logp=True)
    if uniforms.shape != (8, len(scene), 1):
        raise ValueError("Expected heading uniforms [8,B,1]")
    memory = model.memory(scene, xy)
    cross = [block.cross_attention.kv(memory) for block in model.blocks]
    cache = [None] * len(model.blocks)
    previous = torch.full((len(scene),), 401, device=scene.device, dtype=torch.long)
    cumulative = torch.zeros_like(previous)
    tokens, scores = [], []
    for t in range(8):
        x = model.token_embedding(previous)[:, None] + model.position_embedding.weight[t][None, None]
        if model.heading_state_projection is not None:
            x = x + model.heading_state_projection(heading_state_features(cumulative))[:, None]
        for k, block in enumerate(model.blocks):
            x, cache[k] = block.step(x, cache[k], cross[k])
        logits = model.output(model.output_norm(x[:, 0])).float()
        previous = (uniforms[t] > logits.softmax(-1).cumsum(-1)).sum(-1).clamp_max(400)
        tokens.append(previous)
        scores.append(F.log_softmax(logits, -1).gather(1, previous[:, None])[:, 0])
        cumulative = cumulative + previous - 200
    return torch.stack(tokens, 1), torch.stack(scores, 1)


class ExpertPolicy(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.component = config["component"]
        self.ranges = config["quantizer"]["xy_range_m"]
        self.length = 16 if self.component == "position" else 8
        if self.component == "position":
            self.expert = ResidualExpert(config["architecture"], "position")
        elif self.component == "heading":
            self.expert = HeadingExpert(config["architecture"])
        else:
            raise ValueError("Unknown expert component")

    def forward(self, scene, base, targets, fixed_xy=None):
        if self.component == "position":
            return self.expert(scene, base, targets)
        if fixed_xy is None:
            raise ValueError("Heading training requires its conditioning XY")
        return self.expert(scene, fixed_xy.detach(), targets)

    @torch.no_grad()
    def generate(self, scene, base, fixed_xy=None, random_uniforms=None, heading_helper=None):
        if self.component == "position":
            if heading_helper is None:
                raise ValueError("Position rewards require a frozen conditional heading expert")
            if heading_helper.training or any(p.requires_grad for p in heading_helper.parameters()):
                raise ValueError("The conditional heading expert must be frozen and in eval mode")
            ids, logp = self.expert.generate(scene, base, uniforms=random_uniforms)
            xy = position_decode(base, ids, self.ranges)
            heading_ids = heading_helper.generate(scene, xy, greedy=True)
        else:
            if fixed_xy is None:
                raise ValueError("Heading decoding requires conditioning XY")
            xy = fixed_xy.detach()
            ids, logp = sample_heading(self.expert, scene, xy, random_uniforms)
            heading_ids = ids
        return {"token_ids": ids, "old_logp": logp, "heading_ids": heading_ids,
                "xy": xy, "trajectory": decode(xy, heading_ids)}


def load_expert(path, config, device="cpu"):
    model = ExpertPolicy(config)
    state = torch.load(Path(path), map_location="cpu", weights_only=True, mmap=True)
    saved_config = state.get("config", state.get("contract", {}))
    if saved_config.get("component", config["component"]) != config["component"]:
        raise ValueError("Checkpoint component mismatch")
    if "architecture" in saved_config and saved_config["architecture"] != config["architecture"]:
        raise ValueError("Checkpoint architecture mismatch")
    model.load_state_dict(state.get("model", state), strict=True)
    return model.to(device).eval().requires_grad_(False)


class CAREDriveRefiner(nn.Module):
    def __init__(self, position, heading):
        super().__init__()
        if position.component != "position" or heading.component != "heading":
            raise ValueError("Expected a position policy and a heading policy")
        self.position, self.heading = position, heading

    @torch.no_grad()
    def forward(self, scene_features, base_trajectory):
        self.eval()
        return self.position.generate(scene_features, base_trajectory,
                                      heading_helper=self.heading.expert)
