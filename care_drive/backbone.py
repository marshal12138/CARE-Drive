"""CoWorld future-token and MLP trajectory branches, without text supervision."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoProcessor

from .models.mlp import MLPActionHead
from .models.driving_prompt import (
    END_ACTION_TOKEN, START_ACTION_TOKEN, TRAJ_ACTION_TOKEN,
    build_action_token_string, format_trajectory_number,
    get_all_jepa_action_token_strings, NAVIGATION_COMMANDS,
    VLM_DRIVING_SYSTEM_MESSAGE,
)
from .models.qwen3_vl import Qwen3VLWrapper
from .data import CAMERA_ORDER, validate_observations
from .io import digest
from .supervision import future_supervision
from .future_readout import FutureReadout, normalize_readout_config
from .action_tokens import (action_mode, action_strings, register_independent,
                            initialize_independent, extract_independent)
from .flow_action import head_kind

SYSTEM_PROMPT = VLM_DRIVING_SYSTEM_MESSAGE.strip().replace(
    'multi-view images from 8 cameras, ego vehicle states (position)',
    'historical videos from three cameras (front, left, and right), with four '
    'frames per camera from the nominal 2 Hz timeline, ego pose history, current velocity and acceleration',
) + '\nPositions are in meters in the current ego rear-axle coordinate frame; headings are in radians.'


def token_block(future_vision=True, action_token_mode='shared'):
    block = build_action_token_string(num_jepa_action_tokens=96 if future_vision else 0, num_jepa_token_steps=4,
                                     num_wan_action_tokens=0, num_vggt_action_tokens=0,
                                     num_traj_action_tokens=8)
    return block.replace(TRAJ_ACTION_TOKEN * 8, ''.join(action_strings(action_token_mode)))


def register_tokens(processor):
    strings = [START_ACTION_TOKEN, END_ACTION_TOKEN, TRAJ_ACTION_TOKEN] + get_all_jepa_action_token_strings(4)
    processor.tokenizer.add_tokens(strings, special_tokens=True)
    processor.tokenizer.padding_side = 'left'
    for text in strings:
        if len(processor.tokenizer.encode(text, add_special_tokens=False)) != 1:
            raise ValueError(f'Latent marker is not a single token: {text}')
    return {s: processor.tokenizer.convert_tokens_to_ids(s) for s in strings}


def driving_prompt(observations, future_vision=True, action_token_mode='shared'):
    validate_observations(observations)
    nav = observations['navigation_command']
    if nav not in (0, 1, 2, 3):
        raise ValueError('Invalid navigation command')
    lines = ['Ego state for the 2-second history:', 'Historical poses (x, y, heading):']
    for t, pose in zip(observations['history_time_offsets_seconds'], observations['history_trajectory']):
        values = ', '.join(format_trajectory_number(float(x)) for x in pose)
        lines.append(f't={float(t):.3f}s: ({values})')
    lines.append(f'Navigation: {NAVIGATION_COMMANDS[nav]}')
    for key in ('velocity', 'acceleration'):
        values = ', '.join(format_trajectory_number(float(x)) for x in observations[key])
        lines.append(f'{key}: ({values})')
    lines.append('Predict the ego vehicle trajectory for the next 4 seconds.')
    lines.append(token_block(future_vision, action_token_mode))
    return '\n'.join(lines)


class ObservationProcessor:
    def __init__(self, config):
        self.config = config
        self.future_vision = config['model'].get('future_vision', True)
        self.processor = AutoProcessor.from_pretrained(
            config['model']['vlm_path'], local_files_only=config['model'].get('local_files_only', True),
            fix_mistral_regex=False)
        self.token_ids = register_tokens(self.processor)
        self.action_token_mode = action_mode(config)
        if self.action_token_mode == 'independent':
            self.token_ids.update(register_independent(self.processor))

    def messages(self, observations):
        from .visual_inputs import enabled, messages
        if enabled(self.config):
            return messages(self, observations)
        validate_observations(observations)
        times = ', '.join(f'{float(t):.3f}' for t in observations['history_time_offsets_seconds'])
        content = [{'type': 'text', 'text': (
            'The following three videos show the front, left, and right cameras over a 2-second history. '
            'Each video contains four real frames from the nominal 2 Hz timeline, ordered oldest to newest '
            f'and ending at the current time. Their ego-relative times are {times} seconds. '
            'Video timestamps are relative to the first frame of each clip.'
        )}]
        for view in CAMERA_ORDER:
            content.extend([{'type': 'text', 'text': f'{view.capitalize()} camera history:'},
                            {'type': 'video'}])
        content.append({'type': 'text', 'text': driving_prompt(observations, self.future_vision, self.action_token_mode)})
        return [
            {'role': 'system', 'content': [{'type': 'text', 'text': SYSTEM_PROMPT}]},
            {'role': 'user', 'content': content},
        ]

    def __call__(self, observations, roots):
        from .visual_inputs import enabled, encode
        if enabled(self.config):
            return encode(self, observations, roots)
        videos, metadata, texts = [], [], []
        width, height = self.config['model']['frame_width'], self.config['model']['frame_height']
        for obs, root in zip(observations, roots, strict=True):
            validate_observations(obs)
            for view in CAMERA_ORDER:
                frames = []
                for path in obs['history_image_paths'][view]:
                    with Image.open(Path(root) / path) as image:
                        frames.append(np.asarray(image.convert('RGB').resize((width, height), Image.Resampling.BILINEAR)))
                videos.append(np.stack(frames))
                times = np.asarray(obs['history_time_offsets_seconds'])
                indices = np.rint((times - times[0]) * 2).astype(int).tolist()
                metadata.append({'fps': 2.0, 'total_num_frames': indices[-1] + 1, 'frames_indices': indices})
            texts.append(self.processor.apply_chat_template(self.messages(obs), tokenize=False, add_generation_prompt=False))
        encoded = self.processor(
            text=texts, videos=videos, padding=True, return_tensors='pt',
            videos_kwargs={'video_metadata': metadata, 'do_sample_frames': False, 'do_resize': False},
        )
        expected_grid = torch.tensor([[2, height // 16, width // 16]] * (3 * len(observations)))
        if not torch.equal(encoded['video_grid_thw'], expected_grid):
            raise ValueError('Video processor changed the three-view/four-frame input contract')
        if 'pixel_values' in encoded or 'image_grid_thw' in encoded:
            raise ValueError('History must be represented as three videos, not separate still images')
        for row in encoded['input_ids']:
            from collections import Counter
            expected = {START_ACTION_TOKEN: 1, END_ACTION_TOKEN: 1,
                        **Counter(action_strings(self.action_token_mode))}
            expected.update({s: 24 if self.future_vision else 0 for s in get_all_jepa_action_token_strings(4)})
            for name, count in expected.items():
                actual = int((row == self.token_ids[name]).sum())
                if actual != count:
                    raise ValueError(f'Token count mismatch: {name} {actual} != {count}')
        if 'labels' in encoded:
            raise ValueError('Text labels are forbidden')
        return dict(encoded)


def representation_loss(predicted, teacher, cosine_weight=0.5):
    if predicted.shape != teacher.shape:
        raise ValueError(f'Teacher/student shape mismatch: {predicted.shape} vs {teacher.shape}')
    predicted, teacher = predicted.float(), teacher.detach().float()
    l1 = F.smooth_l1_loss(predicted, teacher, beta=1.0)
    cosine = 1 - F.cosine_similarity(predicted, teacher, dim=-1).mean()
    return l1 + cosine_weight * cosine, l1, cosine


class FutureVisionModel(nn.Module):
    def __init__(self, config, *, load_pretrained=True):
        super().__init__()
        self.config = config
        cfg = config['model']
        self.trajectory_head = head_kind(config)
        if self.trajectory_head != 'mlp':
            raise ValueError('CARE-Drive uses the MLP backbone trajectory head')
        self.future_vision = cfg.get('future_vision', True)
        self.future_supervision = future_supervision(config)
        if cfg['num_jepa_tokens'] != (96 if self.future_vision else 0):
            raise ValueError('Future-token count conflicts with the selected model variant')
        if cfg['use_vlm_autoreg_text_trajectory'] or cfg['eval_traj_mode'] != 'traj_token':
            raise ValueError('Only non-text latent-token training is supported')
        self.vlm = Qwen3VLWrapper(
            model_path=cfg['vlm_path'], num_jepa_action_tokens=96, num_jepa_token_steps=4,
            num_vggt_action_tokens=0, num_wan_action_tokens=0, num_traj_action_tokens=8,
            max_action_token_slots=4, torch_dtype=torch.bfloat16,
            gradient_checkpointing=cfg['gradient_checkpointing'], attn_implementation=cfg['attention'],
            load_pretrained=load_pretrained, local_files_only=cfg.get('local_files_only', True),
        )
        for parameter in self.vlm.parameters():
            parameter.requires_grad_(False)
        # FP32 trainable parameters give AdamW FP32 moments; forward uses BF16 autocast.
        self.vlm.model.model.language_model.float().requires_grad_(True)
        self.vlm.model.model.visual.eval().requires_grad_(False)
        hidden = self.vlm.hidden_size
        self.jepa_token_norm = nn.LayerNorm(hidden)
        self.jepa_head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, cfg['teacher_dim']))
        self.action_head = MLPActionHead({'input_feature_dim': hidden, 'num_points': 8, 'point_dim': 3})
        if not self.future_vision:
            # Preserve the reference RNG draws before action-head initialization, then remove the branch.
            del self.jepa_head, self.jepa_token_norm
        self.observation_processor = ObservationProcessor(config)
        self.action_token_mode = action_mode(config)
        if self.action_token_mode == 'independent':
            if load_pretrained:
                initialize_independent(self.vlm)
            else:
                # The checkpoint supplies every embedding row, including action queries.
                register_independent(self.vlm.processor)
                self.vlm.model.resize_token_embeddings(len(self.vlm.processor.tokenizer), mean_resizing=False)
        actual_ids = {s: self.vlm.processor.tokenizer.convert_tokens_to_ids(s) for s in self.observation_processor.token_ids}
        if actual_ids != self.observation_processor.token_ids:
            raise ValueError('Processor and model tokenizer identities differ')
        readout = normalize_readout_config(config)
        self.future_readout = None
        if readout['enabled']:
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(config['seed'])
                self.future_readout = FutureReadout(hidden, readout['dim'], readout['num_heads']).float()
        self.trainable_report = self.parameter_report() if load_pretrained else None

    def train(self, mode=True):
        super().train(mode)
        self.vlm.model.model.visual.eval()
        return self

    def parameter_report(self):
        groups = {'language': 0, 'jepa': 0, 'trajectory': 0, 'frozen': 0}
        names = []
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                groups['frozen'] += parameter.numel()
                continue
            names.append(name)
            if name.startswith('vlm.model.model.language_model.'):
                group = 'language'
            elif name.startswith(('jepa_head.', 'jepa_token_norm.')):
                group = 'jepa'
            elif name.startswith(('action_head.', 'future_readout.')):
                group = 'trajectory'
            else:
                raise ValueError(f'Unexpected trainable parameter: {name}')
            if parameter.dtype != torch.float32:
                raise ValueError(f'Trainable parameter must be FP32: {name}')
            groups[group] += parameter.numel()
        report = {'counts': groups, 'trainable_names': names, 'visual_frozen': True}
        if self.future_readout is not None:
            report['future_readout_parameters'] = sum(p.numel() for p in self.future_readout.parameters())
        return report

    def parameter_groups(self):
        cfg = self.config['training']
        groups = [
            {'name': 'language', 'params': list(self.vlm.model.model.language_model.parameters()), 'lr': cfg['language_lr']},
            {'name': 'trajectory', 'params': list(self.action_head.parameters()), 'lr': cfg['trajectory_lr']},
        ]
        if self.future_vision:
            groups.insert(1, {'name': 'jepa', 'params': list(self.jepa_token_norm.parameters()) + list(self.jepa_head.parameters()), 'lr': cfg['jepa_lr']})
        if self.future_readout is not None:
            groups[-1]['params'].extend(self.future_readout.parameters())
        return groups

    def encode_observations(self, observations, roots):
        return self.observation_processor(observations, roots)

    def forward(self, inputs, targets=None, *, variant=None, initial_noise=None, profile=False):
        if variant is None:
            variant = 'C' if self.future_supervision else 'A'
        if variant not in ('A', 'B', 'C'):
            raise ValueError(variant)
        supervised = self.future_supervision and variant != 'A'
        if not self.future_supervision and variant != 'A':
            raise ValueError('Forward variant conflicts with disabled future supervision')
        if 'labels' in inputs:
            raise ValueError('Text labels are forbidden')
        if profile:
            import time
            from .runtime import synchronize
            device = inputs['input_ids'].device
            synchronize(device)
            vlm_started = time.perf_counter()
        # The multimodal base preserves DeepStack vision processing without computing LM logits.
        output = self.vlm.model.model(**inputs, output_hidden_states=False, use_cache=False, return_dict=True)
        if profile:
            synchronize(device)
            vlm_seconds = time.perf_counter() - vlm_started
        hidden = output.last_hidden_state
        ids = inputs['input_ids']
        traj = (extract_independent(hidden, ids, self.observation_processor.token_ids)
                if getattr(self, 'action_token_mode', action_mode(self.config)) == 'independent'
                else self.vlm.extract_tokens(hidden, ids, 'traj_action_tokens'))
        prediction = None
        if self.future_vision:
            future = self.vlm.extract_tokens(hidden, ids, 'jepa_action_tokens')
            prediction = self.jepa_head(self.jepa_token_norm(future))
        trajectory_hidden = traj[:, -1]
        if self.future_readout is not None:
            trajectory_hidden = trajectory_hidden + self.future_readout(trajectory_hidden, future)
        trajectory = self.action_head(trajectory_hidden)
        result = {'future_features': prediction if supervised else None, 'trajectory': trajectory, 'scene_features': traj}
        if targets is not None:
            gt = targets['trajectory'].float()
            if gt.shape != trajectory.shape:
                raise ValueError('Trajectory target shape mismatch')
            trajectory_loss = F.mse_loss(trajectory.float(), gt)
            # Keep the projection branch in DDP's graph in the action-only control.
            if not supervised and 'teacher_features' in targets:
                raise ValueError('Trajectory-only training must not load teacher targets')
            future_loss = prediction.sum() * 0 if prediction is not None else trajectory_loss.new_zeros(())
            l1, cosine = future_loss.detach(), future_loss.detach()
            if supervised:
                future_loss, l1, cosine = representation_loss(prediction, targets['teacher_features'], self.config['training']['cosine_weight'])
            result.update(loss=trajectory_loss + future_loss, metrics={
                'trajectory_loss': trajectory_loss.detach(), 'future_loss': future_loss.detach() if supervised else None,
                'future_smooth_l1': l1.detach() if supervised else None,
                'future_cosine_error': cosine.detach() if supervised else None,
                'ade': (trajectory.float()[..., :2] - gt[..., :2]).norm(dim=-1).mean().detach(),
                'fde': (trajectory.float()[:, -1, :2] - gt[:, -1, :2]).norm(dim=-1).mean().detach(),
            })
        return result


    @torch.no_grad()
    def predict(self, observations, roots, *, sample_ids=None):
        device = next(self.parameters()).device
        inputs = {key: value.to(device) for key, value in self.encode_observations(observations, roots).items()}
        modes = {module: module.training for module in self.modules()}
        try:
            self.eval()
            noise = None
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type in ('npu', 'cuda')):
                return self(inputs, initial_noise=noise)
        finally:
            for module, mode in modes.items():
                module.training = mode

    def tokenizer_identity(self):
        tokenizer = self.vlm.processor.tokenizer
        result = {'vocab_sha256': digest(tokenizer.get_vocab()), 'token_ids': self.observation_processor.token_ids,
                'token_count': len(tokenizer), 'system_prompt': SYSTEM_PROMPT,
                'token_block': token_block(self.future_vision, self.action_token_mode),
                'observation_layout': {'camera_order': list(CAMERA_ORDER), 'frames_per_view': 4,
                                       'fps': 2, 'history_window_seconds': 2, 'media_type': 'video'}}
        from .visual_inputs import enabled, identity
        if enabled(self.config):
            result.update(identity(self.config))
        return result
