"""Greedy raw-observation or frozen-feature inference."""
import argparse
from pathlib import Path
import torch
from .config import load, recipes
from .datasets import read_manifest, ExpertScenes
from .checkpoints import load_backbone
from .models.refinement import load_expert, CAREDriveRefiner
from .runtime import initialize, amp, shutdown
from .io import write_jsonl
from .performance_gate import bounded_probe, inference_parameters, require_preflight


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--position", required=True)
    parser.add_argument("--heading", required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--manifest")
    inputs.add_argument("--feature-cache")
    parser.add_argument("--backbone")
    parser.add_argument("--sensor-root")
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--probe-steps", type=int, default=0)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    config = load(args.config)
    if args.manifest and (not args.backbone or not args.sensor_root):
        parser.error("raw observation inference requires --backbone and --sensor-root")
    probe = bounded_probe(args.probe_steps)
    context_config = dict(config, inference_inputs={
        "manifest": args.manifest, "feature_cache": args.feature_cache,
        "sensor_root": args.sensor_root, "backbone": args.backbone,
        "position": args.position, "heading": args.heading})
    if not probe:
        require_preflight("inference", context_config, inference_parameters(args.batch_size, workers=0, precision="bf16-backbone/fp32-experts"),
                          workload="care-drive-greedy-inference")
    rank, world, device = initialize(config.get("device", "npu"))
    stages = recipes(config["stage_config"])
    refiner = CAREDriveRefiner(load_expert(args.position, stages["C3-XY-EP1"], device),
                              load_expert(args.heading, stages["D1-H-ADAPT"], device))
    backbone = load_backbone(args.backbone, config, device) if args.manifest else None
    data = read_manifest(args.manifest) if args.manifest else ExpertScenes(args.feature_cache)
    indices = list(range(rank, len(data), world))
    if probe:
        indices = indices[:args.probe_steps * args.batch_size]
    results = []
    with torch.inference_mode():
        for start in range(0, len(indices), args.batch_size):
            ix = indices[start:start + args.batch_size]
            if backbone is not None:
                rows = [data[i] for i in ix]
                encoded = backbone.encode_observations([row["observations"] for row in rows], [args.sensor_root] * len(ix))
                with amp(device):
                    result = backbone({k: v.to(device) for k, v in encoded.items()})
                scene, base = result["scene_features"].float(), result["trajectory"].float()
                tokens = [row["sample_id"] for row in rows]
            else:
                scene, base = data.scene[ix].to(device).float(), data.anchor[ix].to(device).float()
                tokens = [data.tokens[i] for i in ix]
            result = refiner(scene, base)
            results.extend({"sample_id": token, "trajectory": trajectory.tolist()}
                           for token, trajectory in zip(tokens, result["trajectory"]))
    output = Path(args.output)
    if world > 1:
        output = output.with_name(output.stem + f".rank{rank:03d}" + output.suffix)
    write_jsonl(output, results)
    shutdown()


if __name__ == "__main__":
    main()
