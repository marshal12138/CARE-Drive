"""Device setup shared by the command-line entry points."""
from contextlib import nullcontext
import os
import math
import random
import numpy as np
import torch
import torch.distributed as dist


def initialize(kind="npu"):
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    if kind == "npu":
        import torch_npu  # noqa: F401
        torch.npu.set_device(local)
        device, backend = torch.device("npu", local), "hccl"
    elif kind == "cuda":
        torch.cuda.set_device(local)
        device, backend = torch.device("cuda", local), "nccl"
    elif kind == "cpu":
        device, backend = torch.device("cpu"), "gloo"
    else:
        raise ValueError("device must be cpu, cuda, or npu")
    if world > 1 and not dist.is_initialized():
        dist.init_process_group(backend=backend)
    return rank, world, device


def seed_everything(seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "npu":
        torch.npu.manual_seed_all(seed)
    elif device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def synchronize(device):
    if device.type == "npu":
        torch.npu.synchronize(device)
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def amp(device):
    return (torch.autocast(device.type, dtype=torch.bfloat16)
            if device.type in ("npu", "cuda") else nullcontext())


def barrier():
    if dist.is_initialized():
        dist.barrier()


def shutdown():
    if dist.is_initialized():
        dist.destroy_process_group()


def make_scheduler(optimizer, total_steps, warmup_ratio):
    warmup = max(1, int(total_steps * warmup_ratio))
    def schedule(step):
        if step < warmup:
            return float(step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
