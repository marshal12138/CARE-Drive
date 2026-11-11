"""Paper backbone objective: physical trajectory L1 plus future-feature loss."""
import torch
import torch.nn.functional as F
from ..backbone import FutureVisionModel

def per_scene_losses(result, targets):
    trajectory = (result['trajectory'].float() - targets['trajectory'].float()).abs().mean((-2, -1))
    future = result['future_features'].float()
    teacher = targets['teacher_features'].detach().float()
    auxiliary = F.smooth_l1_loss(future, teacher, reduction='none').mean((-2, -1))
    auxiliary = auxiliary + .5 * (1 - F.cosine_similarity(future, teacher, dim=-1).mean(-1))
    return trajectory, auxiliary


class L1Model(FutureVisionModel):
    def forward(self, inputs, targets=None, **kwargs):
        result = super().forward(inputs, **kwargs)
        if targets is not None:
            trajectory, auxiliary = per_scene_losses(result, targets)
            distance = torch.linalg.vector_norm(result['trajectory'].float()[..., :2] - targets['trajectory'].float()[..., :2], dim=-1)
            result.update(loss=(trajectory + auxiliary).mean(), metrics=dict(
                trajectory_loss=trajectory.mean().detach(), future_loss=auxiliary.mean().detach(),
                ade=distance.mean().detach(), fde=distance[:, -1].mean().detach()))
        return result

