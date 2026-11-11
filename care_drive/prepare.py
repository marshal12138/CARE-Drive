"""Export NAVSIM manifests and build training-only teacher / frozen feature caches."""
import argparse
from pathlib import Path
import os
import numpy as np
import torch
from .config import load
from .data import CAMERA_SENSORS, relative_image, validate_observations
from .datasets import read_manifest
from .io import write_json, write_jsonl, sha256


def manifest(args):
    from navsim.common.dataloader import SceneLoader, MetricCacheLoader
    from navsim.common.dataclasses import SceneFilter, SensorConfig
    cfg = load(args.scene_filter)
    kwargs = {k: v for k, v in cfg.items() if not k.startswith('_')}
    if kwargs.get('num_history_frames') != 4 or kwargs.get('num_future_frames', 0) < 8:
        raise ValueError('Use a NAVSIM split with four history frames and at least eight future frames')
    loader = SceneLoader(Path(args.logs), Path(args.sensor_root), SceneFilter(**kwargs), SensorConfig.build_no_sensors())
    metric_paths = MetricCacheLoader(Path(args.metric_cache)).metric_cache_paths if args.metric_cache else {}
    rows = []
    for token in sorted(loader.tokens):
        raw = loader.scene_frames_dicts[token]
        scene = loader.get_scene_from_token(token)
        ego = scene.frames[3].ego_status
        command = np.asarray(ego.driving_command)
        if command.shape != (4,) or command.sum() != 1 or not np.isin(command, [0, 1]).all():
            raise ValueError('Expected a four-class one-hot navigation command')
        navigation = int(command.argmax())
        views = {view: [relative_image(frame['cams'][sensor]['data_path']) for frame in raw[:4]]
                 for view, sensor in CAMERA_SENSORS.items()}
        obs = {'history_image_paths': views, 'current_image_path': views['front'][-1],
               'history_trajectory': scene.get_history_trajectory(4).poses.tolist(),
               'history_time_offsets_seconds': [(f['timestamp'] - raw[3]['timestamp']) / 1e6 for f in raw[:4]],
               'velocity': ego.ego_velocity.tolist(), 'acceleration': ego.ego_acceleration.tolist(),
               'navigation_command': navigation}
        validate_observations(obs)
        row = {'sample_id': token, 'observations': obs}
        if not args.observations_only:
            future_times = [(f['timestamp'] - raw[3]['timestamp']) / 1e6 for f in raw[4:12]]
            if np.max(np.abs(np.asarray(future_times) - np.arange(1, 9) * .5)) > .05:
                raise ValueError('Future frames do not match the nominal 2 Hz timeline')
            row['targets'] = {'trajectory': scene.get_future_trajectory(8).poses.tolist(),
                              'future_image_paths': [relative_image(f['cams']['CAM_F0']['data_path']) for f in raw[4:12]]}
        if token in metric_paths:
            row['evaluation'] = {'metric_cache_path': str(Path(metric_paths[token]).resolve())}
        rows.append(row)
    if not rows:
        raise ValueError('The selected split is empty')
    write_jsonl(Path(args.output), rows)
    write_json(Path(args.output).with_suffix('.summary.json'),
               {'scenes': len(rows), 'scoreable_scenes': sum('evaluation' in r for r in rows),
                'scene_filter_sha256': sha256(Path(args.scene_filter)), 'observations_only': args.observations_only})


def cache(args):
    from .runtime import initialize, amp, shutdown
    from .performance_gate import bounded_probe, require_preflight, inference_parameters
    config = load(args.config)
    rows = read_manifest(args.manifest)
    probe = bounded_probe(args.probe_steps)
    context_config = dict(config, cache_manifest=str(Path(args.manifest).resolve()), cache_kind=args.kind,
                          backbone_checkpoint=args.backbone, cache_sensor_root=args.sensor_root)
    if not probe:
        require_preflight('inference', context_config, inference_parameters(args.batch_size, workers=0),
                          workload='care-drive-cache-' + args.kind)
    rank, world, device = initialize(config.get('device', 'npu'))
    indices = list(range(rank, len(rows), world))
    if probe:
        indices = indices[:args.probe_steps * args.batch_size]
    if args.kind == 'teacher':
        from safetensors.torch import save_file
        from .teacher import JepaTeacher
        model = JepaTeacher(config, device)
    else:
        from .checkpoints import load_backbone
        if not args.backbone:
            raise ValueError('--backbone is required for frozen expert features')
        model = load_backbone(args.backbone, config, device)
    tokens, scenes, bases, targets, paths = [], [], [], [], []
    selected_rows = [rows[i] for i in indices]
    if args.kind == 'features':
        # Reward training uses the scoreable subset. Record, rather than silently hide, the selection.
        selected_rows = [r for r in selected_rows if r.get('evaluation', {}).get('metric_cache_path')]
        if not selected_rows:
            raise ValueError('Expert training features require scenes with official metric caches')
    for start in range(0, len(selected_rows), args.batch_size):
        batch = selected_rows[start:start + args.batch_size]
        with torch.inference_mode():
            if args.kind == 'teacher':
                clips = torch.stack([model.prepare_images([Path(args.sensor_root) / p for p in r['targets']['future_image_paths']]) for r in batch])
                features = model(clips).cpu().contiguous()
                for row, value in zip(batch, features):
                    token = row['sample_id']
                    if not token.isalnum():
                        raise ValueError('Teacher cache scene IDs must be alphanumeric')
                    output = Path(args.output) / token[:2] / (token + '.safetensors')
                    output.parent.mkdir(parents=True, exist_ok=True)
                    if output.exists():
                        raise FileExistsError(output)
                    save_file({'teacher_features': value}, str(output), metadata={'sample_id': token})
            else:
                encoded = model.encode_observations([r['observations'] for r in batch], [args.sensor_root] * len(batch))
                with amp(device):
                    result = model({k: v.to(device) for k, v in encoded.items()})
                scenes.append(result['scene_features'].detach().float().cpu())
                bases.append(result['trajectory'].detach().float().cpu())
                targets.extend(r['targets']['trajectory'] for r in batch)
                paths.extend(r['evaluation']['metric_cache_path'] for r in batch)
                tokens.extend(r['sample_id'] for r in batch)
    if args.kind == 'features':
        output = Path(args.output)
        if world > 1:
            output = output.with_name(output.stem + f'.rank{rank:03d}' + output.suffix)
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            raise FileExistsError(output)
        torch.save({'tokens': tokens, 'scene': torch.cat(scenes), 'anchor': torch.cat(bases),
                    'gt': torch.tensor(targets, dtype=torch.float32), 'metric_cache_paths': paths,
                    'metadata': {'manifest_sha256': sha256(Path(args.manifest)),
                                 'backbone_sha256': sha256(Path(args.backbone)),
                                 'rank': rank, 'world_size': world, 'probe': probe,
                                 'selected': len(tokens), 'input_scenes_on_rank': len(indices)}}, output)
    shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('manifest')
    for key in ('logs', 'sensor-root', 'scene-filter', 'output'):
        p.add_argument('--' + key, required=True)
    p.add_argument('--metric-cache')
    p.add_argument('--observations-only', action='store_true')
    p.set_defaults(function=manifest)
    p = sub.add_parser('cache')
    p.add_argument('kind', choices=['teacher', 'features'])
    for key in ('config', 'manifest', 'sensor-root', 'output'):
        p.add_argument('--' + key, required=True)
    p.add_argument('--backbone')
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--probe-steps', type=int, default=0)
    p.set_defaults(function=cache)
    args = parser.parse_args()
    if hasattr(args, 'batch_size') and args.batch_size < 1:
        parser.error('batch-size must be positive')
    args.function(args)


if __name__ == '__main__':
    main()
