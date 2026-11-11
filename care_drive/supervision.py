"""Separate future positions from future-target supervision."""


def future_supervision(config):
    model = config['model']
    enabled = model.get('future_supervision', model.get('future_vision', True))
    if not isinstance(enabled, bool):
        raise ValueError('future_supervision must be boolean')
    if enabled and not model.get('future_vision', True):
        raise ValueError('Future supervision requires future positions')
    return enabled
