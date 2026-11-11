"""Controlled image/video inputs for the MLP visual-input factorial experiment."""
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from .data import CAMERA_ORDER, validate_observations

SCHEMA = 'visual_input_ablation_v1'


def enabled(config):
    schema = config['model'].get('visual_prompt_schema')
    if schema not in (None, SCHEMA):
        raise ValueError(f'Unknown visual prompt schema: {schema}')
    if schema is None and config['model'].get('visual_input_mode') is not None:
        raise ValueError('Explicit visual input mode requires its shared prompt schema')
    return schema == SCHEMA


def identity(config):
    from .backbone import SYSTEM_PROMPT
    cfg = config['model']
    mode, width, height = cfg['visual_input_mode'], cfg['frame_width'], cfg['frame_height']
    if mode not in ('image', 'video') or min(width, height) <= 0 or width % 32 or height % 32:
        raise ValueError('Expected image/video mode and fixed dimensions divisible by 32')
    return {'system_prompt': SYSTEM_PROMPT.replace('historical videos', 'historical visual observations'),
            'observation_layout': {'schema': SCHEMA, 'camera_order': list(CAMERA_ORDER),
                'frames_per_view': 4, 'fps': 2, 'history_window_seconds': 2,
                'media_type': mode, 'frame_width': width, 'frame_height': height,
                'resize': 'PIL RGB BILINEAR, no crop; processor resize disabled',
                'visual_tokens': (12 if mode == 'image' else 6) * (width // 32) * (height // 32)}}


def messages(processor, observations):
    from .backbone import driving_prompt
    validate_observations(observations)
    spec = identity(processor.config)
    mode = spec['observation_layout']['media_type']
    times = ', '.join(f'{float(t):.3f}' for t in observations['history_time_offsets_seconds'])
    content = [{'type': 'text', 'text': (
        'The following visual observations show the front, left, and right cameras over a 2-second history. '
        'Each camera has four real frames from the nominal 2 Hz timeline, ordered oldest to newest '
        f'and ending at the current time. Their ego-relative times are {times} seconds. '
        'Video timestamps, when present, are relative to the first frame of each clip.')}]
    for view in CAMERA_ORDER:
        content.append({'type': 'text', 'text': f'{view.capitalize()} camera history:'})
        content.extend([{'type': mode} for _ in range(4 if mode == 'image' else 1)])
    content.append({'type': 'text', 'text': driving_prompt(
        observations, processor.future_vision, processor.action_token_mode)})
    return [{'role': 'system', 'content': [{'type': 'text', 'text': spec['system_prompt']}]},
            {'role': 'user', 'content': content}]


def encode(processor, observations, roots):
    from .backbone import START_ACTION_TOKEN, END_ACTION_TOKEN, action_strings, get_all_jepa_action_token_strings
    spec = identity(processor.config)['observation_layout']
    mode, width, height = spec['media_type'], spec['frame_width'], spec['frame_height']
    media, metadata, texts = [], [], []
    for obs, root in zip(observations, roots, strict=True):
        validate_observations(obs)
        for view in CAMERA_ORDER:
            frames = []
            for path in obs['history_image_paths'][view]:
                with Image.open(Path(root) / path) as image:
                    frames.append(np.asarray(image.convert('RGB').resize(
                        (width, height), Image.Resampling.BILINEAR)))
            if mode == 'image':
                media.extend(frames)
            else:
                media.append(np.stack(frames))
                times = np.asarray(obs['history_time_offsets_seconds'])
                indices = np.rint((times - times[0]) * 2).astype(int).tolist()
                metadata.append({'fps': 2.0, 'total_num_frames': indices[-1] + 1,
                                 'frames_indices': indices})
        texts.append(processor.processor.apply_chat_template(
            messages(processor, obs), tokenize=False, add_generation_prompt=False))
    kwargs = ({'images': media, 'images_kwargs': {'do_resize': False}} if mode == 'image' else
              {'videos': media, 'videos_kwargs': {'video_metadata': metadata,
                                                 'do_sample_frames': False, 'do_resize': False}})
    encoded = processor.processor(text=texts, padding=True, truncation=False, return_tensors='pt', **kwargs)
    grid_key, other = ('image_grid_thw', 'video_grid_thw') if mode == 'image' else ('video_grid_thw', 'image_grid_thw')
    expected = torch.tensor([[1 if mode == 'image' else 2, height // 16, width // 16]]
                            * ((12 if mode == 'image' else 3) * len(observations)))
    if other in encoded or not torch.equal(encoded[grid_key], expected):
        raise ValueError('Processor changed the controlled visual input contract')
    visual_id = processor.processor.tokenizer.convert_tokens_to_ids(f'<|{mode}_pad|>')
    for row in encoded['input_ids']:
        if int((row == visual_id).sum()) != spec['visual_tokens']:
            raise ValueError('Unexpected visual-token count or truncation')
        counts = {START_ACTION_TOKEN: 1, END_ACTION_TOKEN: 1,
                  **Counter(action_strings(processor.action_token_mode))}
        counts.update({s: 24 if processor.future_vision else 0 for s in get_all_jepa_action_token_strings(4)})
        for name, count in counts.items():
            if int((row == processor.token_ids[name]).sum()) != count:
                raise ValueError(f'Latent token count mismatch: {name}')
    if 'labels' in encoded:
        raise ValueError('Text labels are forbidden')
    return dict(encoded)
