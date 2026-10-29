"""Frozen evaluation cases: the windows and rollouts a dataset's test folder is scored on.

A case manifest (data/manifests/eval_cases/<dataset>_v1.json, written by
scripts/unified/freeze_eval_cases.py) names, per case set, every scored clip: its episode, first
frame and horizon, with the error of the two trivial predictions on it. docs/EVAL_PROTOCOLS.md is
the protocol these lists implement. This module holds what the freezer and an evaluator share:
the row sets a clip is scored on, the trivial predictions, the selection rules and the reader.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

KIND, VERSION = "pwm_eval_cases", 1
MOVING_MM = 5.0                                   # a predicted row is "moving" when it leaves its anchor by this much
QUANTILE_LEVELS = (0.125, 0.375, 0.625, 0.875)    # within-episode motion quantiles of the windows_motion set
FIELDS = ("id", "start", "horizon", "stratum", "tag", "rows_all", "rows_predicted", "rows_moving",
          "motion_mm_per_frame", "zero_all_mm", "zero_predicted_mm", "zero_predicted_final_mm",
          "zero_moving_mm", "hold_predicted_mm", "hold_predicted_final_mm",
          "bodies_passive", "zero_pose_passive_mm", "zero_pose_passive_deg")


def predicted_rows(batch):
    """[B, N] rows a model predicts: valid particles minus rigid actuators and fixtures, whose motion is an
    input. Commanded deformable particles stay: a drag command does not prescribe the achieved trajectory.
    The batch must carry its rigid-body sidecar (the evaluator's _prepare turns it on); the rule is the
    overfit harness's primary_mask."""
    valid = batch.get_valid_mask()
    return valid if batch.body_type is None else valid & ~batch.action_mask


def scored_rows(batch, positions=None):
    """Named [B, N] row sets of clips whose scored frames are positions[:, C:] (default: the batch's own).

    all        every valid particle: the evaluator's particle_mean_mm
    predicted  predicted_rows
    moving     predicted rows that leave their last context position by MOVING_MM or more at some scored frame
    still      the other predicted rows: error here is motion the model invented
    """
    positions = batch.positions if positions is None else positions
    C = batch.context_len
    predicted = predicted_rows(batch)
    reach = (positions[:, C:] - positions[:, C - 1:C]).norm(dim=-1).amax(dim=1)          # [B, N] metres
    moving = predicted & (reach >= MOVING_MM / 1000.0)
    return {"all": batch.get_valid_mask(), "predicted": predicted, "moving": moving, "still": predicted & ~moving}


def trivial_predictions(positions, context_len):
    """(zero motion, hold velocity) predictions [B, H, N, 3] for the scored frames positions[:, C:]."""
    C = context_len
    H = positions.shape[1] - C
    anchor = positions[:, C - 1:C]
    steps = torch.arange(1, H + 1, dtype=positions.dtype, device=positions.device).view(1, H, 1, 1)
    return anchor.expand(-1, H, -1, -1), anchor + steps * (anchor - positions[:, C - 2:C - 1])


def _mm(pred, gt, rows):
    """(mean over scored frames and rows, mean at the last frame) in mm for one clip; None without rows.
    The per-clip averaging of utils.evaluation.TrajectoryScores."""
    if not bool(rows.any()):
        return None, None
    err = (pred[:, rows] - gt[:, rows]).norm(dim=-1) * 1000.0
    return float(err.mean()), float(err[-1].mean())


def zero_motion_pose_errors(batch, lo=0, frames=None):
    """Per clip (passive rigid bodies, mm, degrees): the scorer's pose_passive_* for predicting no motion over the
    span batch frames lo..lo+frames-1, averaged over scored frames and then over the clip's passive bodies. With
    rigid particles a body fitted to its anchor-frame points has the anchor pose, so this is the pose change from
    the last context frame; it does not depend on which frame the body templates were taken from."""
    from utils.pose_ops import rotation_geodesic_error
    B = batch.positions.shape[0]
    if batch.body_type is None or batch.body_poses is None:
        return [(0, None, None)] * B
    C = batch.context_len
    poses = batch.body_poses[:, lo:None if frames is None else lo + frames]
    valid, out = batch.get_valid_mask(), []
    for e in range(B):
        mm, deg = [], []
        for k in range(batch.body_type.shape[1]):
            rows = valid[e] & (batch.particle_body[e] == k)
            if int(batch.body_type[e, k]) != 1 or not bool(rows.any()) or bool(batch.action_mask[e, rows].any()):
                continue
            anchor, future = poses[e, C - 1:C, k], poses[e, C:, k]
            mm.append(float((anchor[:, :3, 3] - future[:, :3, 3]).norm(dim=-1).mean()) * 1000.0)
            deg.append(float(torch.rad2deg(rotation_geodesic_error(anchor[:, :3, :3].expand_as(future[:, :3, :3]),
                                                                    future[:, :3, :3])).mean()))
        out.append((len(mm), float(np.mean(mm)), float(np.mean(deg))) if mm else (0, None, None))
    return out


def case_numbers(batch, lo=0, frames=None):
    """Per clip, the FIELDS from rows_all on, for the span batch frames lo..lo+frames-1 (default: the whole
    batch): row counts, mean step of the predicted rows over the scored transitions, the trivial predictions'
    particle errors and the zero-motion pose error of the passive bodies."""
    positions = batch.positions[:, lo:None if frames is None else lo + frames]
    C = batch.context_len
    if positions.shape[1] <= C or C < 2:
        raise ValueError("A case needs at least two context frames and one scored frame")
    rows = scored_rows(batch, positions)
    zero, hold = trivial_predictions(positions, C)
    gt = positions[:, C:]
    pose = zero_motion_pose_errors(batch, lo, frames)
    out = []
    for e in range(positions.shape[0]):
        predicted = rows["predicted"][e]
        if not bool(predicted.any()):
            raise ValueError("A case without predicted rows cannot be scored")
        step = (positions[e, C:, predicted] - positions[e, C - 1:-1, predicted]).norm(dim=-1)
        zero_pred, hold_pred = _mm(zero[e], gt[e], predicted), _mm(hold[e], gt[e], predicted)
        values = [int(rows["all"][e].sum()), int(predicted.sum()), int(rows["moving"][e].sum()),
                  float(step.mean()) * 1000.0, _mm(zero[e], gt[e], rows["all"][e])[0], zero_pred[0], zero_pred[1],
                  _mm(zero[e], gt[e], rows["moving"][e])[0], hold_pred[0], hold_pred[1], *pose[e]]
        out.append([None if v is None else (v if isinstance(v, int) else round(v, 4)) for v in values])
    return out


def motion_profile(batch):
    """Mean step of the predicted rows per transition [T-1] in metres, for a one-clip batch."""
    if batch.positions.shape[0] != 1:
        raise ValueError("motion_profile reads one clip")
    rows = predicted_rows(batch)[0]
    return (batch.positions[0, 1:, rows] - batch.positions[0, :-1, rows]).norm(dim=-1).mean(dim=1).double().numpy()


def span_scores(profile, first, context_len, horizon):
    """(starts, mean step over each start's `horizon` scored transitions). `profile[i]` is the transition
    from frame first + i; starts are source frames, the rule of IndexedEpisodeDataset._start_weights."""
    frames = first + len(profile) + 1
    starts = np.arange(first, frames - context_len - horizon + 1)
    if not len(starts):
        raise ValueError("The episode is too short for this horizon")
    total = np.concatenate([[0.0], np.cumsum(profile)])
    rel = starts - first
    score = (total[rel + context_len + horizon - 1] - total[rel + context_len - 1]) / horizon
    return starts, np.round(score, 7)               # 0.1 micrometre: ties do not hang on float32 rounding noise


def quantile_starts(profile, first, context_len, future_len, levels=QUANTILE_LEVELS):
    """Window starts at the given within-episode quantiles of the window motion score (ties: earliest)."""
    starts, score = span_scores(profile, first, context_len, future_len)
    order = np.lexsort((starts, score))
    return [int(starts[order[int(round(level * (len(order) - 1)))]]) for level in levels]


def active_start(profile, first, context_len, horizon):
    """The start whose `horizon` scored frames carry the most motion (ties: earliest)."""
    starts, score = span_scores(profile, first, context_len, horizon)
    return int(starts[int(np.argmax(score))])


def hash_order(names, salt=""):
    """Names in a fixed pseudo-random order that needs no generator state."""
    return sorted(names, key=lambda name: hashlib.sha256(f"{salt}{name}".encode()).digest())


def load_cases(path, case_set=None):
    """A validated manifest; with `case_set`, that set's cases as dicts keyed by its fields."""
    value = json.loads(Path(path).read_text())
    if value.get("kind") != KIND or value.get("version") != VERSION or not isinstance(value.get("sets"), dict):
        raise ValueError(f"Not a {KIND} v{VERSION} manifest: {path}")
    for name, entry in value["sets"].items():
        fields, cases = entry.get("fields"), entry.get("cases")
        if not isinstance(fields, list) or not isinstance(cases, list) or not cases or fields[:3] != list(FIELDS[:3]):
            raise ValueError(f"Case set {name} needs fields starting with {FIELDS[:3]} and a nonempty case list")
        if any(not isinstance(case, list) or len(case) != len(fields) for case in cases):
            raise ValueError(f"Case set {name} has a row that does not match its fields")
        keys = [(case[0], case[1], case[2]) for case in cases]
        if len(set(keys)) != len(keys):
            raise ValueError(f"Case set {name} repeats a case")
    if case_set is None:
        return value
    if case_set not in value["sets"]:
        raise ValueError(f"{path} has no case set {case_set!r}; it has {sorted(value['sets'])}")
    entry = value["sets"][case_set]
    return [dict(zip(entry["fields"], case)) for case in entry["cases"]]


class CaseDataset:
    """The clips of one frozen case set, read through the dataset the evaluator built for the checkpoint
    (scripts/unified/eval_unified.evaluation_dataset). Iterating yields one TrajectoryBatch per case, frames
    start .. start + C + horizon - 1, in the manifest's order; eval_domain / rollout_domain take it as `dataset=`.

    Everything else is the inner dataset's (sidecar and articulation switches, scales, provenance). `future_len`
    is accepted and ignored: a case names its own horizon, whatever the evaluator's rollout convention sets.
    `tag` (a motion quantile q1..q4, `beyond`) and `stratum` (a fold action, a yikai population) keep a subset."""

    _OWN = ("inner", "cases", "manifest", "case_set", "future_len")

    def __init__(self, inner, path, case_set, *, tag=None, stratum=None):
        manifest = load_cases(path)
        cases = [case for case in load_cases(path, case_set)
                 if (tag is None or tag in case["tag"].split(",")) and (stratum is None or case["stratum"] == str(stratum))]
        if not cases:
            raise ValueError(f"No case of {case_set} has tag {tag!r} and stratum {stratum!r}")
        for name, value in (("inner", inner), ("manifest", manifest), ("case_set", case_set),
                            ("cases", cases), ("future_len", None)):
            object.__setattr__(self, name, value)
        if manifest.get("partial"):
            raise ValueError(f"{path} is a partial (debugging) manifest")
        if manifest["domain"] != inner.domain:
            raise ValueError(f"{path} holds {manifest['domain']} cases, the dataset is {inner.domain}")
        if int(manifest["window"]["context_len"]) != int(inner.context_len):
            raise ValueError("The cases were frozen with another context length")
        frozen = manifest["inputs"].get("episode_index_sha256")
        if frozen is not None and frozen != inner.index["index_sha256"]:
            raise ValueError("The cases were frozen on another test index than the one this checkpoint validates on")
        listed = manifest["inputs"].get("folder_index_sha256")
        if listed:
            from utils.reproducibility import file_sha256
            for root in inner.roots:
                index = Path(root) / "INDEX.json"
                if listed.get(str(index)) != file_sha256(index):
                    raise ValueError(f"The cases were frozen on another test folder than {root}")

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def __setattr__(self, name, value):
        if name in self._OWN:
            object.__setattr__(self, name, value)
        else:
            setattr(self.inner, name, value)

    def __len__(self):
        return len(self.cases)

    def __iter__(self):
        inner = self.inner
        reader = self._indexed if hasattr(inner, "records") else self._rl_push if inner.domain == "rl_push" else self._pushing
        yield from reader()

    def _indexed(self):
        inner, C = self.inner, self.inner.context_len
        records = {record["episode_id"]: record for record in inner.records}
        kept = inner.future_len
        try:
            for case in self.cases:
                inner.future_len = int(case["horizon"])
                record = records[case["id"]]
                batch = inner.window(record, int(case["start"]))
                if batch.positions.shape[1] != C + case["horizon"]:
                    raise ValueError(f"Case {case['id']}@{case['start']} does not fit its episode")
                batch.source_shard, batch.source_episode_id = record["file"], record["episode_id"]
                batch.eval_case = case
                yield batch
        finally:
            inner.future_len = kept

    def _rl_push(self):
        from data.rl_push import window_points
        inner, C = self.inner, self.inner.context_len
        objects = {obj.file: obj for obj in inner.objects}
        usable = {(inner.objects[i].file, e) for i, e in inner.episodes}
        for case in self.cases:
            file, episode = case["id"].rsplit(":", 1)
            if (file, int(episode)) not in usable:
                raise ValueError(f"Case {case['id']} is excluded or missing in this release")
            obj = objects[file]
            window = window_points(obj, int(episode), int(case["start"]), C + int(case["horizon"]), inner.links,
                                   inner._object_rows(obj))
            batch = inner._tagged([window], obj)
            batch.eval_case = case
            yield batch

    def _pushing(self):
        """Stream each needed shard once in the loader's order; a record is cut to its cases' horizon by the
        adapter's own prefix window, then split into its rows."""
        inner = self.inner
        wanted = {}
        for case in self.cases:
            shard, key, row = case["id"].rsplit(":", 2)
            if int(case["start"]) != 0:
                raise ValueError("pushing cases start at frame 0")
            rows = wanted.setdefault((shard, key), {"horizon": int(case["horizon"]), "rows": [], "cases": []})
            if rows["horizon"] != int(case["horizon"]):
                raise ValueError(f"{shard}:{key} mixes horizons in one case set")
            rows["rows"].append(int(row))
            rows["cases"].append(case)
        kept, shards = inner.future_len, inner.shards
        try:
            inner.shards = [s for s in shards if Path(s).name in {shard for shard, _ in wanted}]
            for sample in inner._create_pipeline():
                name = (Path(str(sample["__url__"])).name, str(sample["__key__"]))
                if name not in wanted:
                    continue
                entry = wanted.pop(name)
                inner.future_len = entry["horizon"]
                whole = inner._process_sample(sample)
                if whole.positions.shape[1] != inner.context_len + entry["horizon"]:
                    raise ValueError(f"{name[0]}:{name[1]} is shorter than its cases")
                for row, case in zip(entry["rows"], entry["cases"]):
                    clip = whole._map_tensors(lambda value: value[row:row + 1])
                    clip.source_shard = str(sample["__url__"])
                    clip.eval_case = case
                    yield clip
                if not wanted:
                    break
        finally:
            inner.future_len, inner.shards = kept, shards
        if wanted:
            raise ValueError(f"Cases not found in the test folder: {sorted(wanted)[:3]}")

