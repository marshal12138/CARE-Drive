"""Backbone head selection."""
def head_kind(config):
    kind = config['model'].get('trajectory_head', 'mlp')
    if kind not in ('mlp', 'dit'):
        raise ValueError(f'Unknown trajectory head: {kind}')
    return kind
