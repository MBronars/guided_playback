"""Fail-closed checks and the one scoring vocabulary shared by evaluators.

Every evaluator that scores predicted particle trajectories reports the same keys,
computed the same way (TrajectoryScores). Coverage is counted in clips (iter_clips):
one clip is one scored trajectory, however the shards package them. Paper-protocol
evaluators keep their published grouping and units and do not use these scores.
"""
import json
import math
from collections import Counter

import torch

RAD2DEG = 180.0 / math.pi


def validate_checkpoint_fields(requested, trained, fields, *, prefix="data."):
    """Explicit eval selectors cannot reinterpret checkpoint units/representation.

    Normalization is validated separately using the resolved numeric snapshot.
    """
    for key in fields:
        if key not in requested:
            continue
        actual, expected = requested[key], trained.get(key)
        if actual != expected:
            raise ValueError(f"{prefix}{key}={actual!r} disagrees with checkpoint {expected!r}")


def validate_request(budget, context_len, future_len):
    """budget: None (the whole split) or a positive count of clips, files or episodes."""
    if budget is not None and (isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0):
        raise ValueError("evaluation budget must be None (the whole split) or a positive integer")
    if context_len <= 0 or (future_len is not None and future_len <= 0):
        raise ValueError("context_len and an explicit future_len must be positive")


def parse_clips(value):
    """The --clips flag: 'all' reads the whole split, otherwise a positive clip count."""
    text = str(value).strip().lower()
    if text == "all":
        return None
    if not text.isdigit() or int(text) <= 0:
        raise ValueError(f"--clips must be 'all' or a positive integer, got {value!r}")
    return int(text)


def validate_prediction(pred, gt, valid):
    if (pred.shape != gt.shape or pred.ndim != 4 or pred.shape[-1] != 3
            or pred.shape[0] == 0 or pred.shape[1] == 0):
        raise ValueError("Prediction must match a nonempty [B, F, N, 3] target")
    if valid.shape != (pred.shape[0], pred.shape[2]) or not bool(valid.any(1).all()):
        raise ValueError("Every evaluated episode must contain valid particles")
    mask = valid[:, None, :].expand(pred.shape[:3])
    if not torch.isfinite(pred[mask]).all() or not torch.isfinite(gt[mask]).all():
        raise ValueError("Non-finite prediction or ground truth on valid particles")


def finish_metrics(result):
    if result["n_episodes"] <= 0:
        raise ValueError("No episodes evaluated; check the split, exclusions and sample errors")
    try:
        json.dumps(result, allow_nan=False)
    except ValueError as error:
        raise ValueError("Non-finite evaluation metrics") from error
    return result


def iter_clips(dataset, max_clips):
    """(batch, source) pairs until exactly max_clips clips; None reads the whole split.

    A record that crosses the limit is trimmed, so the count is exact and no record
    past it is read. `source` is the shard the loader tagged the record with ('' when
    the dataset does not tag its records)."""
    remaining = max_clips
    if remaining is not None and remaining <= 0:
        return
    for batch in dataset:
        source = getattr(batch, "source_shard", "")
        if remaining is not None:
            if batch.batch_size > remaining:
                batch = batch.slice(remaining)
            remaining -= batch.batch_size
        yield batch, source
        if remaining is not None and remaining <= 0:
            return


def read_provenance(sources, declared_shards=()):
    """Shards the scored clips came from, next to the shard count the split declares."""
    read = {name: int(count) for name, count in sorted(sources.items()) if name}
    if not read:
        return {}
    return {"shards_declared": len(declared_shards), "shards_read": len(read),
            "clips_per_shard_read": read}


def rigid_pose_errors(batch, pred, gt_poses):
    """Per-frame pose error of every rigid body in every clip.

    pred [B, H, N, 3] predicted positions; gt_poses [B, H, K, 4, 4]. Each body's
    template is fitted (Kabsch) onto its predicted rows and that pose is compared with
    ground truth, so particle-only and pose-native checkpoints are scored identically.
    Returns [(clip, body, actuated, degrees [H], millimetres [H])]."""
    from utils.pose_ops import rotation_geodesic_error
    from utils.rigid_ops import kabsch_batched
    if batch.body_type is None:
        return []
    H = pred.shape[1]
    valid = batch.get_valid_mask()
    errors = []
    for k in range(batch.body_type.shape[1]):
        for e in range(pred.shape[0]):
            if int(batch.body_type[e, k]) != 1:
                continue
            sel = valid[e] & (batch.particle_body[e] == k)
            template = batch.body_templates[e, k, :int(batch.body_template_mask[e, k].sum())]
            R, t = kabsch_batched(template[None].expand(H, -1, -1), pred[e][:, sel])
            gt = gt_poses[e, :, k]
            errors.append((e, k, bool(batch.action_mask[e, sel].any()),
                           rotation_geodesic_error(R, gt[:, :3, :3]) * RAD2DEG,
                           (t - gt[:, :3, 3]).norm(dim=-1) * 1000.0))
    return errors


def chamfer_per_frame(pred, gt, budget=4e7):
    """Symmetric Chamfer distance per frame, in the inputs' unit, between the point sets pred[h] and gt[h]
    ([H, n, 3] each): the sum of the two directional mean nearest-neighbour distances. Frames are taken in chunks
    that keep at most `budget` pair distances in memory."""
    step = max(1, int(budget // max(pred.shape[1] * gt.shape[1], 1)))
    out = []
    for lo in range(0, pred.shape[0], step):
        dist = torch.cdist(pred[lo:lo + step], gt[lo:lo + step])                     # [h, n, n]
        out.append(dist.min(dim=2).values.mean(dim=1) + dist.min(dim=1).values.mean(dim=1))
    return torch.cat(out)


class TrajectoryScores:
    """Episode-weighted errors of predicted trajectories: the evaluation vocabulary.

    add() takes predicted and ground-truth positions over the scored frames of each
    clip ([B, H, N, 3], metres; frames C..C+H of the batch). Keys:
      particle_mean_mm / particle_final_mm / particle_at{m}_mm  Euclidean mm over a
          clip's valid particles, averaged over its frames (or at its last / m-th
          frame), then averaged over clips
      particle_per_frame_mm  the per-frame curve (per_frame=True): at frame h the mean over the clips that
          reach h, with particle_per_frame_n[h] the number of such clips, so clips of different lengths
          share one curve. per_frame=True adds the same pair for the deformable rows
          (deform_per_frame_*) and for the passive bodies' pose (pose_passive_per_frame_{mm,deg,n}), and
          particle_mse_mm2 / particle_mse_per_frame_mm2: the squared Euclidean distance in mm^2, averaged the same way
      chamfer_mean_mm / chamfer_final_mm / chamfer_per_frame_mm  (chamfer=True) symmetric Chamfer distance of the
          scored rows at every frame, mm: the sum of the two directional mean nearest-neighbour distances. It needs
          no point correspondence, so it compares with methods that predict unordered point sets
      deform_mean_mm / deform_final_mm  the same on deformable rows (body type 0)
      chamfer_cm_at{m} / chamfer_n_at{m}  symmetric Chamfer on deformable rows: the sum
          of the two directional mean nearest-neighbour distances, cm, at frame m
      pose_{passive,actuated}_{deg,mm}  rigid_pose_errors averaged over frames, then
          over (clip, body) pairs, split by whether the body is actuated
      horizon_frames_min / horizon_frames_max  frames scored per clip
    `rows` restricts the particle and deformable metrics to a particle subset.
    """

    def __init__(self, *, marks=(16, 32), chamfer_marks=(), per_frame=False, chamfer=False):
        self.marks, self.chamfer_marks = tuple(marks), tuple(chamfer_marks or ())
        self.per_frame, self.chamfer = per_frame, chamfer
        self.n, self.horizons, self.curves = 0, [], {}
        self.sums, self.counts = Counter(), Counter()
        self.last_curves = {}                             # the curves of the last clip added (a one-clip batch: that clip)

    def _add(self, key, value):
        self.sums[key] += float(value)
        self.counts[key] += 1

    def _add_curve(self, key, curve):
        """Per-frame sums and clip counts of one clip's [H] curve; a longer clip extends the curve."""
        curve = curve.double().cpu()
        sums, counts = self.curves.get(key, (torch.zeros(0, dtype=torch.float64), torch.zeros(0, dtype=torch.long)))
        if curve.numel() > sums.numel():
            grow = curve.numel() - sums.numel()
            sums = torch.cat([sums, torch.zeros(grow, dtype=torch.float64)])
            counts = torch.cat([counts, torch.zeros(grow, dtype=torch.long)])
        sums[:curve.numel()] += curve
        counts[:curve.numel()] += 1
        self.curves[key] = (sums, counts)
        self.last_curves[key] = [round(float(v), 3) for v in curve]

    def add(self, batch, pred, gt, *, rows=None, poses=True):
        valid = batch.get_valid_mask()
        validate_prediction(pred, gt, valid)
        rows = valid if rows is None else valid & rows.to(valid.device)
        H = pred.shape[1]
        self.last_curves = {}
        deform = (None if batch.body_type is None
                  else rows & (batch.body_type.gather(1, batch.particle_body) == 0))
        for e in range(pred.shape[0]):
            if not rows[e].any():
                continue
            err = (pred[e][:, rows[e]] - gt[e][:, rows[e]]).norm(dim=-1) * 1000.0
            self._add("particle_mean_mm", err.mean())
            self._add("particle_final_mm", err[-1].mean())
            for m in self.marks:
                if m <= H:
                    self._add(f"particle_at{m}_mm", err[m - 1].mean())
            if self.per_frame:
                self._add_curve("particle", err.mean(dim=1))
                self._add("particle_mse_mm2", err.square().mean())
                self._add_curve("particle_mse", err.square().mean(dim=1))
            if self.chamfer:
                distance = chamfer_per_frame(pred[e][:, rows[e]], gt[e][:, rows[e]]) * 1000.0
                self._add("chamfer_mean_mm", distance.mean())
                self._add("chamfer_final_mm", distance[-1])
                self._add_curve("chamfer", distance)
            self.n += 1
            self.horizons.append(H)
            if deform is not None and deform[e].any():
                d = deform[e]
                derr = (pred[e][:, d] - gt[e][:, d]).norm(dim=-1) * 1000.0
                self._add("deform_mean_mm", derr.mean())
                self._add("deform_final_mm", derr[-1].mean())
                if self.per_frame:
                    self._add_curve("deform", derr.mean(dim=1))
                for m in self.chamfer_marks:
                    if m <= H:
                        dist = torch.cdist(pred[e][m - 1, d], gt[e][m - 1, d])
                        self._add(f"chamfer_cm_at{m}",
                                  (dist.min(dim=1).values.mean() + dist.min(dim=0).values.mean()) * 100.0)
        if poses and batch.body_type is not None and batch.body_poses is not None:
            C = batch.context_len
            passive = []
            for _, _, actuated, deg, mm in rigid_pose_errors(batch, pred, batch.body_poses[:, C:C + H]):
                kind = "actuated" if actuated else "passive"
                self._add(f"pose_{kind}_deg", deg.mean())
                self._add(f"pose_{kind}_mm", mm.mean())
                if self.per_frame and not actuated:
                    self._add_curve("pose_passive_mm", mm)
                    self._add_curve("pose_passive_deg", deg)
                    passive.append((mm, deg))
            if passive:                                   # the clip's curve: the mean over its passive bodies
                for i, key in enumerate(("pose_passive_mm", "pose_passive_deg")):
                    self.last_curves[key] = [round(float(v), 3) for v in torch.stack([p[i] for p in passive]).mean(dim=0)]

    def result(self):
        mean = lambda key: round(self.sums[key] / self.counts[key], 3)
        out = {"n_episodes": self.n}
        if self.horizons:
            out["horizon_frames_min"], out["horizon_frames_max"] = min(self.horizons), max(self.horizons)
        for key in ("particle_mean_mm", "particle_final_mm", *(f"particle_at{m}_mm" for m in self.marks)):
            if self.counts[key]:
                out[key] = mean(key)
        curve = lambda key: [round(float(s) / int(c), 3) for s, c in zip(*self.curves[key])]
        if "particle" in self.curves:
            out["particle_per_frame_mm"] = curve("particle")
            out["particle_per_frame_n"] = [int(c) for c in self.curves["particle"][1]]
            out["particle_mse_mm2"] = mean("particle_mse_mm2")
            out["particle_mse_per_frame_mm2"] = curve("particle_mse")
        if "chamfer" in self.curves:
            out["chamfer_mean_mm"], out["chamfer_final_mm"] = mean("chamfer_mean_mm"), mean("chamfer_final_mm")
            out["chamfer_per_frame_mm"] = curve("chamfer")
        for key in ("deform_mean_mm", "deform_final_mm"):
            if self.counts[key]:
                out[key] = mean(key)
        if "deform" in self.curves:
            out["deform_per_frame_mm"] = curve("deform")
            out["deform_per_frame_n"] = [int(c) for c in self.curves["deform"][1]]
        for m in self.chamfer_marks:
            if self.counts[f"chamfer_cm_at{m}"]:
                out[f"chamfer_cm_at{m}"] = mean(f"chamfer_cm_at{m}")
                out[f"chamfer_n_at{m}"] = self.counts[f"chamfer_cm_at{m}"]
        for key in ("pose_passive_deg", "pose_passive_mm", "pose_actuated_deg", "pose_actuated_mm"):
            if self.counts[key]:
                out[key] = mean(key)
        if "pose_passive_mm" in self.curves:             # per frame over the (clip, body) pairs that reach it
            out["pose_passive_per_frame_mm"] = curve("pose_passive_mm")
            out["pose_passive_per_frame_deg"] = curve("pose_passive_deg")
            out["pose_passive_per_frame_n"] = [int(c) for c in self.curves["pose_passive_mm"][1]]
        return out
