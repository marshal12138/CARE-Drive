"""Frozen two-tier PDMS weighting, independent of model and scoring runtime."""
import numpy as np

RULE = dict(threshold_normalized=.01, numerical_epsilon=1e-12, raw_hard=2., raw_easy=1.,
            normalize='mean1_over_scored_population', unscorable_final_weight=1., term='trajectory_L1_only')

def build_weights(all_ids, model_scores, gt_scores):
    if len(all_ids) != len(set(all_ids)) or not all_ids:
        raise ValueError('Nonunique or empty training population')
    if not model_scores or set(model_scores) != set(gt_scores) or not set(model_scores) <= set(all_ids):
        raise ValueError('Scores must cover the identical declared scorable population')
    tokens=sorted(model_scores)
    model=np.asarray([model_scores[t] for t in tokens],dtype=np.float64)
    gt=np.asarray([gt_scores[t] for t in tokens],dtype=np.float64)
    if not np.isfinite(model).all() or not np.isfinite(gt).all() or np.any(model<0) or np.any(model>1) or np.any(gt<0) or np.any(gt>1):
        raise ValueError('Invalid PDMS scores; scorer errors are not neutral samples')
    hard=gt-model > RULE['threshold_normalized']+RULE['numerical_epsilon']
    raw=np.where(hard,2.,1.);denominator=float(raw.mean());values=raw/denominator
    weights=dict.fromkeys(all_ids,1.);weights.update(zip(tokens,values.tolist()))
    summary=dict(rule=RULE,training_scenes=len(all_ids),scored_scenes=len(tokens),neutral_scenes=len(all_ids)-len(tokens),
                 hard_scenes=int(hard.sum()),easy_scenes=int((~hard).sum()),raw_weight_mean=denominator,
                 hard_final_weight=2./denominator,easy_final_weight=1./denominator,
                 full_weight_mean=float(np.mean(list(weights.values()))),scored_weight_mean=float(values.mean()))
    return weights,summary

def weighted_terms(trajectory, auxiliary, weights, valid):
    """Return sums before DDP global real-scene normalization; ignore padding."""
    if trajectory.shape != auxiliary.shape or trajectory.shape != weights.shape or weights.shape != valid.shape:
        raise ValueError('One weight and validity flag required per real training scene')
    return ((trajectory*weights+auxiliary)*valid).sum(),(trajectory*weights*valid).sum(),(auxiliary*valid).sum()
