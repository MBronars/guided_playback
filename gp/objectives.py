"""Guidance objectives on a rollout pos [B, T, P, 3] (metres): per-member scalars. All terms are differentiable."""
import torch


def centroid(pos, mask):
    w = mask.to(pos.dtype)[..., None]
    return (pos * w).sum(-2) / w.sum(-2).clamp(min=1)


class Objective:
    """objective = ||centroid_obj(last frame) - target|| + box_weight * (mean + worst clearance shortfall to every box)
    + floor_weight * mean depth below z = 0, over all valid particles and generated frames. Boxes: (centre, half) metres."""
    def __init__(self, target, obj_mask, valid, boxes=(), clearance_m=0.005, box_weight=5.0, floor_weight=0.0):
        self.target, self.obj_mask, self.valid = target, obj_mask, valid
        self.boxes, self.clearance, self.box_weight, self.floor_weight = list(boxes), clearance_m, box_weight, floor_weight

    def terms(self, pos):
        B = pos.shape[0]
        goal = (centroid(pos[:, -1], self.obj_mask) - self.target).norm(dim=-1)
        P = pos[:, 1:]; sel = self.valid[:, None].expand(P.shape[:3])
        box = pos.new_zeros(B); floor = pos.new_zeros(B)
        for c, h in self.boxes:
            q = (P - c).abs() - h; sd = q.clamp(min=0).norm(dim=-1) + q.amax(-1).clamp(max=0)
            short = (self.clearance - sd).clamp(min=0) * sel
            box = box + short.sum((1, 2)) / sel.sum((1, 2)) + short.amax((1, 2))
        if self.floor_weight > 0:
            floor = ((-P[..., 2]).clamp(min=0) * sel).sum((1, 2)) / sel.sum((1, 2))
        return goal, box, floor

    def __call__(self, pos):
        goal, box, floor = self.terms(pos)
        return goal + self.box_weight * box + self.floor_weight * floor, dict(goal=goal, box=box, floor=floor)


def wall_between(c0, offset, height=0.10, thickness=0.02, width=0.30):
    """A box at the midpoint between the object's start c0 and c0 + offset, across the push direction; (centre, half)."""
    along = offset.clone(); along[2] = 0
    size = torch.tensor([thickness, width, height], device=c0.device)
    if abs(float(along[1])) > abs(float(along[0])):
        size = size[[1, 0, 2]]
    centre = torch.stack([c0[0] + along[0] / 2, c0[1] + along[1] / 2, torch.tensor(height / 2, device=c0.device)])
    return centre, size / 2
