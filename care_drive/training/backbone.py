"""Five-epoch L1/future-feature training with previous-endpoint PDMS weights."""
from contextlib import nullcontext
import math
from pathlib import Path
import time
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from ..checkpoints import save
from ..datasets import BackboneScenes
from ..io import write_json
from ..runtime import make_scheduler, initialize, seed_everything, amp, barrier, shutdown
from ..performance_gate import bounded_probe, training_parameters, require_preflight, describe
from .losses import L1Model, per_scene_losses
from .sampler import epoch_windows
from .weighting import build_weights, weighted_terms


@torch.no_grad()
def refresh_weights(model, data, config, rank, world, device):
    from ..integrations.navsim import OfficialScorer
    scorer = OfficialScorer()
    model.eval()
    scores, ground_truth = {}, {}
    for i in range(rank, len(data), world):
        row = data.rows[i]
        cache = row.get("evaluation", {}).get("metric_cache_path")
        if not cache:
            continue
        encoded = model.encode_observations([row["observations"]], [data.sensor_root])
        with amp(device):
            result = model({k: v.to(device) for k, v in encoded.items()})
        candidates = torch.stack([result["trajectory"][0].float().cpu(),
                                  torch.tensor(row["targets"]["trajectory"], dtype=torch.float32)])
        values = scorer.score([cache], candidates.numpy()[None])[0]
        scores[row["sample_id"]], ground_truth[row["sample_id"]] = float(values[0, 0]), float(values[1, 0])
    if world > 1:
        gathered = [None] * world
        dist.all_gather_object(gathered, (scores, ground_truth))
        scores, ground_truth = {}, {}
        for left, right in gathered:
            if set(left) & set(scores):
                raise ValueError("Duplicate scored scenes across ranks")
            scores.update(left); ground_truth.update(right)
    model.train()
    return build_weights([row["sample_id"] for row in data.rows], scores, ground_truth)


def train(config, *, probe_steps=0):
    tr = config["training"]
    rank, world, device = initialize(config.get("device", "npu"))
    seed_everything(config["seed"], device)
    probe = bounded_probe(probe_steps)
    parameters = training_parameters(config, workers=0)
    if not probe:
        require_preflight("training", config, parameters, workload="care-drive-backbone-l1")
    context = describe("training", config, parameters, workload="care-drive-backbone-l1")
    data = BackboneScenes(config["train_manifest"], config["sensor_root"], config["teacher_cache"], config)
    model = L1Model(config).to(device).train()
    ddp = DDP(model, device_ids=[device.index] if device.type != "cpu" else None,
              broadcast_buffers=False, find_unused_parameters=False) if world > 1 else model
    optimizer = torch.optim.AdamW(model.parameter_groups(), betas=(0.9, 0.95), eps=1e-8,
                                 weight_decay=tr["weight_decay"], foreach=False)
    effective, micro = tr["effective_batch_size"], tr["microbatch_size"]
    total = math.ceil(len(data) / effective) * tr["epochs"]
    scheduler = make_scheduler(optimizer, total, tr["warmup_ratio"])
    output = Path(config["output_dir"]) / "A0-DYNAMIC-L1" / ("probe" if probe else "")
    output.mkdir(parents=True, exist_ok=True)
    weights = dict.fromkeys((row["sample_id"] for row in data.rows), 1.0)
    step, records, started = 0, [], time.monotonic()
    for epoch in range(tr["epochs"]):
        if epoch and tr.get("sample_weighting") == "previous_endpoint_pdms_vs_gt_binary_mean1":
            weights, summary = refresh_weights(model, data, config, rank, world, device)
            if rank == 0:
                write_json(output / f"weights_epoch_{epoch + 1:03d}.json", {"weights": weights, "summary": summary}, replace=True)
        for count, entries in epoch_windows(len(data), config["seed"], epoch, effective, rank, world):
            optimizer.zero_grad(set_to_none=True)
            local_loss = torch.zeros((), device=device)
            for start in range(0, len(entries), micro):
                batch = entries[start:start + micro]
                ix = [i for i, _ in batch]
                valid = torch.tensor([v for _, v in batch], device=device, dtype=torch.float32)
                w = torch.tensor([weights[data.rows[i]["sample_id"]] for i in ix], device=device)
                inputs, targets = data.batch(ix, device)
                sync = ddp.no_sync() if world > 1 and start + micro < len(entries) else nullcontext()
                with sync, amp(device):
                    result = ddp(inputs)
                    trajectory, future = per_scene_losses(result, targets)
                    loss = weighted_terms(trajectory, future, w, valid)[0]
                    (loss * world / count).backward()
                local_loss += loss.detach()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                          tr["gradient_clip"], error_if_nonfinite=True)
            optimizer.step(); scheduler.step(); step += 1
            if world > 1:
                dist.all_reduce(local_loss)
            record = {"epoch": epoch + 1, "step": step, "loss": float(local_loss / count),
                      "elapsed_seconds": time.monotonic() - started}
            records.append(record)
            if rank == 0:
                print(record, flush=True)
            if probe and step >= probe_steps:
                break
        if probe and step >= probe_steps:
            break
    if rank == 0:
        path = output / ("probe.pt" if probe else f"epoch_{tr['epochs']:03d}.pt")
        save(path, model, config, optimizer=optimizer, scheduler=scheduler, epoch=epoch + 1, step=step)
    if probe:
        write_json(output / f"probe.rank{rank:03d}.json", {"status": "PASS", "context": context,
                   "parameters": parameters, "completed_steps": step, "records": records}, replace=True)
    barrier(); shutdown()
