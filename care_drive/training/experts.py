"""One-component SFT/GRPO with frozen conditioning experts."""
from contextlib import nullcontext
import copy
import hashlib
import math
from pathlib import Path
import time
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from ..checkpoints import save
from ..config import stage_checkpoint
from ..datasets import ExpertScenes
from ..io import write_json
from ..models.codec import position_labels, position_decode, uniforms
from ..models.heading import statistics, grpo_objective
from ..models.refinement import ExpertPolicy, load_expert
from ..runtime import make_scheduler, initialize, seed_everything, barrier, shutdown
from ..performance_gate import require_preflight, describe, training_parameters, bounded_probe
from .advantages import build_advantages
from .rewards import training_reward
from .sampler import epoch_windows


def scene_seed(seed, epoch, step, token):
    return int.from_bytes(hashlib.sha256(f"{seed}/{epoch}/{step}/{token}".encode()).digest()[:8], "little") % (2**63 - 1)


def train(config, stage_name, stages, *, probe_steps=0):
    g = dict(stages[stage_name])
    rank, world, device = initialize(config.get("device", "npu"))
    seed_everything(g["seed"], device)
    probe = bounded_probe(probe_steps)
    effective, micro = g["effective_batch"], g.get("microbatch", 8)
    context_config = {"device": device.type, "world_size": world, "expert": g,
                      "feature_cache": config["expert_cache"], "performance_preflight": config.get("performance_preflight")}
    parameters = training_parameters(context_config, microbatch=micro, effective_batch=effective,
                                     checkpointing=False, workers=0, precision="fp32-forward/fp32-state")
    workload = "care-drive-" + g["component"] + "-" + g["task"]
    if not probe:
        require_preflight("training", context_config, parameters, workload=workload)
    measured_context = describe("training", context_config, parameters, workload=workload)
    cache = ExpertScenes(config["expert_cache"])
    output = Path(config["output_dir"]) / stage_name / ("probe" if probe else "")
    output.mkdir(parents=True, exist_ok=True)
    if g.get("source"):
        model = load_expert(stage_checkpoint(config["output_dir"], g["source"]), g, device)
    else:
        model = ExpertPolicy(g).to(device)
    model.requires_grad_(True).train()
    reference = copy.deepcopy(model).eval().requires_grad_(False) if g["task"] == "GRPO" else None
    helper, position = None, None
    if g["component"] == "position":
        name = g["heading_source"]
        helper = load_expert(stage_checkpoint(config["output_dir"], name), stages[name], device).expert
    elif g.get("position_source"):
        name = g["position_source"]
        position = load_expert(stage_checkpoint(config["output_dir"], name), stages[name], device)
    ddp = DDP(model, device_ids=[device.index] if device.type != "cpu" else None,
              broadcast_buffers=False) if world > 1 else model
    optimizer = torch.optim.AdamW(model.parameters(), lr=g["lr"], betas=(0.9, 0.999),
                                 eps=1e-8, weight_decay=g["weight_decay"], foreach=False)
    total_steps = math.ceil(len(cache) / effective) * g["epochs"]
    scheduler = (make_scheduler(optimizer, total_steps, 0.05) if g["task"] == "SFT" else
                 torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0))
    scorer = None
    if reference is not None:
        from ..integrations.navsim import OfficialScorer
        scorer = OfficialScorer()
    step, started, records = 0, time.monotonic(), []
    for epoch in range(g["epochs"]):
        for count, entries in epoch_windows(len(cache), g["seed"], epoch, effective, rank, world):
            ix = [i for i, _ in entries]
            valid = torch.tensor([v for _, v in entries], device=device, dtype=torch.float32)
            scene, base = cache.scene[ix].to(device).float(), cache.anchor[ix].to(device).float()
            xy = None
            if g["component"] == "heading":
                if position is None:
                    xy = base[..., :2]
                else:
                    with torch.no_grad():
                        ids, _ = position.expert.generate(scene, base)
                        xy = position_decode(base, ids, position.ranges)
            targets = (cache.headings[ix].to(device) if g["component"] == "heading"
                       else position_labels(base, cache.gt[ix].to(device), model.ranges))
            candidates, advantages = None, None
            group = g["group_size"]
            if scorer is not None:
                with torch.no_grad():
                    seeds = [scene_seed(g["seed"], epoch, step, cache.tokens[i]) for i in ix]
                    candidates = model.generate(scene.repeat_interleave(group, 0), base.repeat_interleave(group, 0),
                                                None if xy is None else xy.repeat_interleave(group, 0),
                                                uniforms(seeds, model.length, group).to(device), heading_helper=helper)
                    trajectories = candidates["trajectory"].reshape(len(ix), group, 8, 3)
                    greedy = None
                    if g["advantage_mode"] == "greedy_baseline_rms_matched":
                        greedy = model.generate(scene, base, xy)
                        trajectories = np.concatenate([trajectories, greedy["trajectory"][:, None]], axis=1)
                    active = [k for k, (_, v) in enumerate(entries) if v]
                    metrics = torch.zeros((len(ix), trajectories.shape[1], 7), device=device)
                    if active:
                        measured = scorer.score([cache.metric_paths[ix[k]] for k in active], trajectories[active])
                        metrics[active] = torch.as_tensor(measured, device=device)
                    candidate_metrics = metrics[:, :group]
                    rewards = training_reward(candidate_metrics, g)
                    advantages, _ = build_advantages(rewards, valid, g["advantage_mode"], g["advantage_std_floor"],
                                                     reference_rewards=candidate_metrics[..., 0],
                                                     greedy_reward=None if greedy is None else metrics[:, group, 0])
            optimizer.zero_grad(set_to_none=True)
            local_sum = torch.zeros((), device=device)
            for start in range(0, len(ix), micro):
                stop = min(start + micro, len(ix))
                sync = ddp.no_sync() if world > 1 and stop < len(ix) else nullcontext()
                with sync:
                    s, b = scene[start:stop], base[start:stop]
                    condition = None if xy is None else xy[start:stop]
                    if candidates is None:
                        logits = ddp(s, b, targets[start:stop], condition)
                        losses = -statistics(logits, targets[start:stop])[0].mean(1)
                    else:
                        sl = slice(start * group, stop * group)
                        s, b = s.repeat_interleave(group, 0), b.repeat_interleave(group, 0)
                        if condition is not None:
                            condition = condition.repeat_interleave(group, 0)
                        tokens, old = candidates["token_ids"][sl], candidates["old_logp"][sl]
                        with torch.no_grad():
                            ref_logits = reference(s, b, tokens, condition)
                        logits = ddp(s, b, tokens, condition)
                        loss, diagnostic = grpo_objective(logits, ref_logits, tokens, old,
                                                         advantages[start:stop].reshape(-1), g["epsilon"], g["beta"])
                        if float(diagnostic["ratio_error"].detach()) > 1e-3:
                            raise RuntimeError("Sampled-prefix replay differs from the rollout policy")
                        losses = loss.reshape(-1, group).mean(1)
                    weighted = (losses * valid[start:stop]).sum()
                    (weighted * world / count).backward()
                    local_sum += weighted.detach()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            if helper is not None and any(p.grad is not None for p in helper.parameters()):
                raise RuntimeError("Frozen conditional heading expert received gradients")
            optimizer.step(); scheduler.step(); step += 1
            if world > 1:
                dist.all_reduce(local_sum)
            record = {"epoch": epoch + 1, "step": step, "loss": float(local_sum / count),
                      "gradient_norm": float(norm), "elapsed_seconds": time.monotonic() - started}
            records.append(record)
            if rank == 0:
                print(record, flush=True)
            if probe and step >= probe_steps:
                break
        if probe and step >= probe_steps:
            break
    if rank == 0:
        path = output / ("probe.pt" if probe else f"epoch_{g['epochs']:03d}.pt")
        save(path, model, g, optimizer=optimizer, scheduler=scheduler, epoch=epoch + 1, step=step)
    if probe:
        write_json(output / f"probe.rank{rank:03d}.json", {"status": "PASS", "context": measured_context,
                   "parameters": parameters, "completed_steps": step, "records": records}, replace=True)
    barrier(); shutdown()
