"""Training-only safe EP bonus. Official PDMS evaluation is unchanged."""
import torch
PDM_METRICS = ('score', 'no_at_fault_collisions', 'drivable_area_compliance', 'ego_progress', 'time_to_collision_within_bound', 'comfort', 'driving_direction_compliance')

EXPECTED=['score','no_at_fault_collisions','drivable_area_compliance','ego_progress','time_to_collision_within_bound','comfort','driving_direction_compliance']
assert list(PDM_METRICS)==EXPECTED


def fully_safe(metrics):
    assert metrics.shape[-1]==7
    return (metrics[...,[1,2,4,5]]==1).all(-1)


def training_reward(metrics,config):
    official=metrics[...,0]
    mode=config['reward_mode']
    if mode=='official_pdms':return official
    if mode!='safe_ep_bonus':raise ValueError(mode)
    ep=metrics[...,3]
    return official+float(config['safe_ep_coefficient'])*fully_safe(metrics).to(ep.dtype)*ep*(2-ep)
