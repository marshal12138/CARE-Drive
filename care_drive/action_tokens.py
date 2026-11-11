"""Untie eight trajectory inputs while preserving the original function at step zero."""
import torch

from .models.driving_prompt import TRAJ_ACTION_TOKEN

EXTRA_ACTION_TOKENS = tuple(f'<|traj_action_{i}|>' for i in range(7))
INDEPENDENT_ACTION_TOKENS = EXTRA_ACTION_TOKENS + (TRAJ_ACTION_TOKEN,)


def action_mode(config):
    mode = config['model'].get('action_token_mode', 'shared')
    if mode not in ('shared', 'independent'):
        raise ValueError(f'Unknown action token mode: {mode}')
    return mode


def action_strings(mode):
    if mode == 'shared':
        return (TRAJ_ACTION_TOKEN,) * 8
    if mode == 'independent':
        return INDEPENDENT_ACTION_TOKENS
    raise ValueError(mode)


def register_independent(processor):
    processor.tokenizer.add_tokens(list(EXTRA_ACTION_TOKENS), special_tokens=True)
    ids = [processor.tokenizer.convert_tokens_to_ids(s) for s in INDEPENDENT_ACTION_TOKENS]
    if len(set(ids)) != 8 or any(len(processor.tokenizer.encode(s, add_special_tokens=False)) != 1
                                for s in INDEPENDENT_ACTION_TOKENS):
        raise ValueError('Independent action slots require eight distinct single-token IDs')
    return dict(zip(INDEPENDENT_ACTION_TOKENS, ids))


def initialize_independent(vlm):
    """Call after all common modules initialize; preserve CPU RNG and old vocabulary rows."""
    old = vlm.model.get_input_embeddings().weight
    old_size = old.shape[0]
    source = old[vlm.traj_action_token_id].detach().clone()
    with torch.random.fork_rng(devices=[]):
        ids = register_independent(vlm.processor)
        vlm.model.resize_token_embeddings(len(vlm.processor.tokenizer), mean_resizing=False)
        weight = vlm.model.get_input_embeddings().weight
        if weight.shape[0] != old_size + 7:
            raise ValueError('Expected exactly seven appended vocabulary rows')
        with torch.no_grad():
            for name in EXTRA_ACTION_TOKENS:
                if ids[name] < old_size:
                    raise ValueError('New action ID overwrote an existing vocabulary row')
                weight[ids[name]].copy_(source)
    weight.requires_grad_(True)
    return ids


def extract_independent(hidden, input_ids, token_ids):
    ids = torch.tensor([token_ids[s] for s in INDEPENDENT_ACTION_TOKENS], device=input_ids.device)
    matches = input_ids[..., None] == ids
    if not bool((matches.sum(1) == 1).all()):
        raise ValueError('Each independent action ID must occur exactly once')
    positions = matches.long().argmax(1)
    if not bool((positions[:, 1:] > positions[:, :-1]).all()):
        raise ValueError('Independent action slots are not in their registered order')
    return hidden.gather(1, positions[..., None].expand(-1, -1, hidden.shape[-1]))
