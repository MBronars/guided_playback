"""Unified evaluation: one checkpoint, any of its domains, one vocabulary.

Protocol per domain: the checkpoint's validation split (its saved data.roots, else the
registered corpus), read in stratified order (task roots and object-count directories
interleaved, data/dataset.py), with the checkpoint's own normalization snapshot,
context/future lengths, sampler settings and EMA weights
(--schedule overrides the sampler schedule). Coverage is explicit: --clips all scores the whole split, --clips N the first N
clips of the stratified order.

  fixed window  one model window per clip from ground-truth context
  --rollout     inference.windowed_rollout: overlap-commit autoregression to
                --rollout_frames (capped by each clip's length); after the first
                window the context is the model's own output

Scores are utils.evaluation.TrajectoryScores: particle mm, deformable mm and Chamfer
cm, rigid-body pose deg/mm split passive/actuated, all averaged within a clip and
then across clips. Each batch is seeded from --seed and its position, so a domain's
numbers do not depend on which domains ran before it. Receipts record the requested
and achieved horizon, the clips requested and scored, and the shards they came from.
docs/EVALUATION.md maps each evaluation question to its command.

Frozen case sets (docs/EVAL_PROTOCOLS.md): --cases <manifest> --case_set <name> scores exactly
the clips a data/manifests/eval_cases/ manifest pins, windows or rollouts, and adds the
particle error on the predicted, moving and still rows next to the usual keys.

Reserved benchmarks (pickup 142-asset holdout, MPM unseen/OOD populations, paper
protocols) have their own entrypoints.

Usage:
  PYTHONPATH=. python scripts/unified/eval_unified.py --checkpoint <run>/checkpoints/final.pt \
      --domains pickup --rollout --rollout_frames 36 --clips all --seed 0 --out eval.json
"""
from __future__ import annotations

import argparse
import collections
import json
import os
from pathlib import Path

import torch
from omegaconf import OmegaConf

from data.unified_dataset import DOMAIN_FUTURE_LEN, build_domain_dataset
from data.normalization import resolve_normalization
from inference import load_model_and_diffusion, run_diffusion_inference_df, windowed_rollout
from utils.reproducibility import code_revision, file_sha256, seed_for_device
from utils.evaluation import (
    TrajectoryScores,
    finish_metrics as _finish_metrics,
    iter_clips,
    parse_clips,
    read_provenance,
    validate_prediction as _validate_prediction,
    validate_request as _validate_request,
)
from utils.pose_native import clamp_actuated_pose, materialize_pose_native  # overfit_prod imports it from here
from utils.conditioning import hide_actions

# Per-batch seeding base, distinct from train.validate_sampler's 54321 so the
# in-training curve and this benchmark never share a noise draw.
EVAL_SEED_BASE = 700_001


def _batch_seed(seed: int, index: int) -> int:
    """Noise for batch `index` depends only on --seed and the batch's position.

    The sampler draws from the global generator (inference.run_diffusion_inference_df),
    so a single process-level manual_seed made every domain's number depend on which
    domains were evaluated BEFORE it in the same command: the same checkpoint, clips,
    horizon and seed read 16.038 mm in a 4-domain run and 15.694 mm alone.
    """
    return EVAL_SEED_BASE + 1_000_003 * int(seed) + int(index)


def hides_given_actions(cfg) -> bool:
    """A checkpoint trained with every action hidden (train.action_dropout=1) generates the
    actuator, so the sampler gets each window with its future actions hidden, exactly as in
    training and validation. Every body is still scored against ground truth."""
    return float(cfg.get("train", {}).get("action_dropout", 0.0)) >= 1.0

def _input_provenance(dataset, normalization, *, command_line=False):
    """A receipt's `inputs` for one domain, read from the dataset that was scored."""
    from data.blocklists import manifest_path
    source = ("command line" if command_line else
              "checkpoint data.roots" if getattr(dataset, "roots_override", False) else "registered")
    roots = getattr(dataset, "roots", None) or [dataset.root]       # water reads one directory
    result = {"split": "val", "roots": list(roots), "roots_source": source,
              "normalization_sha256": normalization["sha256"],
              "blocklists_sha256": {p: file_sha256(manifest_path(p))
                                    for p in getattr(dataset, "blocklist_paths", ())}}
    if getattr(dataset, "index", None) is not None:
        result["episode_index_sha256"] = dataset.index["index_sha256"]
        result["episodes_in_index"] = len(dataset.records)
        result["index_split"] = dataset.index["split"]        # a release's test index may be the validation root
    return result


def evaluation_dataset(cfg, domain, context_len, future_len, roots_override=None, test_folders=None):
    """The validation split a checkpoint is scored on.

    A domain named in the checkpoint's saved data.roots reads that release with its own
    exclusion lists, as training validated it; other domains read the registered corpus.
    roots_override (--mpm_val_root, --pickup_roots) re-points a registered domain for this
    evaluation only and never mutates training/global roots.
    test_folders (--test_folder) re-points an indexed dataset of the checkpoint's own data.roots at another test
    folder of the same dataset, a release's report folder (pwm-data-v1/mpm_fresh/test_seen): the folder replaces the
    validation root for this evaluation only and passes the same checks (index domain and split, disjoint from train)."""
    from data import unified_dataset as data
    spec = cfg.get("data", {}).get("roots")
    spec = OmegaConf.to_container(spec, resolve=True) if OmegaConf.is_config(spec) else spec
    if test_folders and domain in test_folders:
        if not spec or domain not in spec or domain not in data.INDEXED_DOMAINS:
            raise ValueError(f"--test_folder re-points an indexed dataset of the checkpoint's own data.roots; {domain} is not one")
        spec = {**spec, domain: {**spec[domain], "val": test_folders[domain]}}
    saved = data.resolve_data_roots(spec)
    if not roots_override or domain not in roots_override:
        return build_domain_dataset(domain, split="val", training=False,
                                    future_len=future_len, context_len=context_len,
                                    normalization=cfg.normalization, roots=saved)
    if domain in saved:
        raise ValueError(f"The checkpoint trained {domain} on its own data.roots; --mpm_val_root and "
                         "--pickup_roots re-point only the registered corpus")
    roots = roots_override[domain]
    common = dict(context_len=context_len, future_len=future_len, training=False,
                  shuffle=False, normalization=cfg.normalization)
    if domain == "mpm":
        return data.UnifiedMPMDataset(wds_location=roots[0], **common)
    if domain == "pickup":
        return data.UnifiedPickupDataset(roots, split="val", **common,
                                         blocklist_paths=[data.HOLDOUT_BLOCKLIST])
    raise ValueError(f"Unsupported explicit evaluation roots for {domain}")


def _prepare(ds, model, domain):
    """Ground-truth poses for metrics, the checkpoint's articulation, fail-fast pose scales."""
    ds.emit_sidecar = True          # metrics need GT poses; particle-only checkpoints ignore them
    ds.articulated = bool(getattr(model.cfg, "articulated_bodies", False))
    if domain != "mpm":
        ds._pose_stats()            # fail fast on missing pose-delta stats


def row_scorers(row_sets, per_frame):
    """One scorer per named row set; with per-frame curves the predicted rows also get the Chamfer distance."""
    return {name: TrajectoryScores(per_frame=per_frame, chamfer=per_frame and name == "predicted") for name in row_sets}


class CaseExtras:
    """What a frozen case set adds to a receipt (--cases): every key also per frame, the two trivial predictions
    scored by the same scorer on the same clips (`reference`: zero = no motion, hold = the last context velocity),
    and one record per case with its per-frame curves (`records`, the receipt's sidecar), so any horizon of a
    rollout is read without scoring again."""

    def __init__(self, row_sets, cases=(), save_dir=None):
        self.reference = {name: (TrajectoryScores(per_frame=True), row_scorers(row_sets, True)) for name in ("zero", "hold")}
        self.records = []
        self.position = {(case["id"], int(case["start"]), int(case["horizon"])): k for k, case in enumerate(cases)}
        self.save_dir = Path(save_dir) if save_dir else None
        if self.save_dir is not None:
            self.save_dir.mkdir(parents=True, exist_ok=True)

    def save_prediction(self, batch, case, pred):
        """--save_predictions: <dir>/<position in the case set>.npz holds the predicted positions of every scored frame
        (metres, float32, [H, rows, 3]) on the clip's valid rows, `rows` (their indices in the clip's particle order)
        and `predicted` (which of them the model predicts; the others follow given actions). Later visualization and
        the scoring of another method's predictions read this layout."""
        if self.save_dir is None or case is None:
            return
        import numpy as np
        from utils.eval_cases import predicted_rows
        valid = batch.get_valid_mask()[0]
        k = self.position[(case["id"], int(case["start"]), int(case["horizon"]))]
        np.savez(self.save_dir / f"{k:05d}.npz", pred=pred[0][:, valid].float().cpu().numpy(),
                 rows=valid.nonzero(as_tuple=True)[0].cpu().numpy().astype("int32"),
                 predicted=predicted_rows(batch)[0][valid].cpu().numpy(),
                 id=np.array(case["id"]), start=np.array(int(case["start"])), horizon=np.array(int(case["horizon"])))

    def add(self, batch, case, gt, positions, scores, by_rows):
        """After the model's scorers took this clip: score the trivial predictions on it and keep the case's curves."""
        from utils.eval_cases import trivial_predictions
        for name, trivial in zip(("zero", "hold"), trivial_predictions(positions, batch.context_len)):
            reference, reference_rows = self.reference[name]
            reference.add(batch, trivial, gt, poses=name == "zero")  # a held velocity does not keep a body rigid
            _add_row_sets(reference_rows, batch, trivial, gt, positions)
        if case is None:
            return
        if gt.shape[0] != 1:
            raise ValueError("A case is one clip")
        predicted, zero, hold = (rows["predicted"].last_curves for rows in
                                 (by_rows, self.reference["zero"][1], self.reference["hold"][1]))
        self.records.append({
            "id": case["id"], "start": int(case["start"]), "horizon": int(case["horizon"]),
            "stratum": case.get("stratum", ""), "tag": case.get("tag", ""),
            "predicted_mm": predicted.get("particle"), "predicted_mse_mm2": predicted.get("particle_mse"),
            "predicted_chamfer_mm": predicted.get("chamfer"), "moving_mm": by_rows["moving"].last_curves.get("particle"),
            "all_mm": scores.last_curves.get("particle"), "pose_passive_mm": scores.last_curves.get("pose_passive_mm"),
            "pose_passive_deg": scores.last_curves.get("pose_passive_deg"),
            "zero_predicted_mm": zero.get("particle"), "zero_predicted_chamfer_mm": zero.get("chamfer"),
            "hold_predicted_mm": hold.get("particle")})

    def result(self):
        return {"reference": {name: {**scores.result(), **_row_results(by_rows)}
                              for name, (scores, by_rows) in self.reference.items()}}


@torch.no_grad()
def eval_domain(model, diffusion, cfg, domain: str, max_clips, device,
                future_len=None, context_len=2, dataset=None, clamp_actuated: bool = False,
                seed: int = 0, hide_given_actions: bool = False, schedule: str | None = None, row_sets=(),
                case_extras: CaseExtras | None = None):
    """One model window per clip from ground-truth context; max_clips None = the whole split,
    schedule None = the checkpoint's df_schedule. row_sets (utils.eval_cases.scored_rows names) adds the
    particle keys on those rows under `rows`; case_extras adds per-frame curves and the trivial references."""
    _validate_request(max_clips, context_len, future_len)
    ds = dataset if dataset is not None else evaluation_dataset(cfg, domain, context_len, future_len)
    _check_explicit_dataset(dataset, cfg, context_len)
    _prepare(ds, model, domain)
    steps = int(cfg.flow_matching.inference_steps)
    per_frame = case_extras is not None
    scores, sources = TrajectoryScores(per_frame=per_frame), collections.Counter()
    by_rows = row_scorers(row_sets, per_frame)
    for i, (batch, source) in enumerate(iter_clips(ds, max_clips)):
        sources[source] += batch.batch_size
        case = getattr(batch, "eval_case", None)
        batch = batch.to(device)
        seed_for_device(_batch_seed(seed, i), device)
        _validate_prediction(batch.future, batch.future, batch.get_valid_mask())
        x0 = run_diffusion_inference_df(
            model, hide_actions(batch, 1.0) if hide_given_actions else batch, diffusion,
            num_inference_steps=steps,
            schedule=schedule or cfg.get("df_schedule", "pyramid"),
            uncertainty_scale=float(cfg.get("df_uncertainty_scale", 1.0)))
        if isinstance(x0, tuple):                             # pose-native checkpoint
            if clamp_actuated:
                x0 = (x0[0], clamp_actuated_pose(batch, x0[1]))
            pred_pos = materialize_pose_native(batch, *x0)[0]
        else:
            pred_pos = batch.predictions_to_positions(
                batch.prediction_stats.denormalize(x0))       # [B, F, N, 3]
        scores.add(batch, pred_pos, batch.future)
        _add_row_sets(by_rows, batch, pred_pos, batch.future, batch.positions)
        if case_extras is not None:
            case_extras.add(batch, case, batch.future, batch.positions, scores, by_rows)
            case_extras.save_prediction(batch, case, pred_pos)
    return _finish_metrics({**scores.result(), **_row_results(by_rows),
                            **(case_extras.result() if case_extras is not None else {}),
                            **read_provenance(sources, getattr(ds, "shards", ()))})


def _add_row_sets(by_rows, batch, pred, gt, positions):
    """Particle keys on named row sets of the scored clip (positions = its context and scored frames)."""
    if by_rows:
        from utils.eval_cases import scored_rows
        rows = scored_rows(batch, positions)
        for name, scorer in by_rows.items():
            scorer.add(batch, pred, gt, rows=rows[name], poses=False)


def _without_curves(metrics):
    """The metrics without their per-frame lists, for the printed line; the receipt keeps everything."""
    return {key: _without_curves(value) if isinstance(value, dict) else value
            for key, value in metrics.items() if "_per_frame_" not in key}


def _row_results(by_rows):
    """{"rows": {name: particle keys and the clips that have such rows}}; a set no clip has is reported empty."""
    return {"rows": {name: scorer.result() for name, scorer in by_rows.items()}} if by_rows else {}


def _check_explicit_dataset(dataset, cfg, context_len):
    if dataset is None:
        return
    if getattr(dataset, "raw", False):
        raise ValueError("A raw calibration dataset cannot be scored with checkpoint normalization")
    if int(dataset.context_len) != context_len:
        raise ValueError("Explicit rollout dataset context_len must match the request")
    if getattr(dataset, "normalization", None) is not None:
        if resolve_normalization(dataset.normalization)["sha256"] != resolve_normalization(cfg.normalization)["sha256"]:
            raise ValueError("Explicit evaluation dataset normalization differs from checkpoint")


@torch.no_grad()
def rollout_domain(model, diffusion, cfg, domain: str, max_clips, device,
                   horizon: int, dump_dir: str | None = None,
                   dump_pick: str = "first", future_len=None, context_len=2,
                   dump_k: int = 1,
                   chamfer_marks: list[int] | None = None, dataset=None, clamp_actuated: bool = False,
                   seed: int = 0, hide_given_actions: bool = False, schedule: str | None = None, row_sets=(),
                   case_extras: CaseExtras | None = None):
    """inference.windowed_rollout of every clip to `horizon` frames, scored against ground
    truth; GT enters only as the actuator's actions. max_clips None = the whole split and
    schedule None = the checkpoint's df_schedule. An explicit dataset supports reserved
    protocols without mutating global corpus roots or exclusion lists. case_extras adds per-frame
    curves and the trivial references (see CaseExtras)."""
    _validate_request(max_clips, context_len, future_len)
    if horizon <= 0 or dump_k < 1 or dump_pick not in ("first", "max_motion"):
        raise ValueError("horizon/dump_k must be positive and dump_pick must be first or max_motion")
    if chamfer_marks and any(mark <= 0 or mark > horizon for mark in chamfer_marks):
        raise ValueError("Chamfer marks must fall within the requested positive horizon")
    F_train = future_len if future_len else DOMAIN_FUTURE_LEN[domain]
    ds = dataset if dataset is not None else evaluation_dataset(cfg, domain, context_len, future_len)
    _check_explicit_dataset(dataset, cfg, context_len)
    _prepare(ds, model, domain)
    ds.future_len = 10_000            # _window keeps the full episode when T <= C+F
    steps = int(cfg.flow_matching.inference_steps)
    schedule = schedule or cfg.get("df_schedule", "pyramid")
    uscale = float(cfg.get("df_uncertainty_scale", 1.0))
    per_frame = case_extras is not None
    scores, sources = TrajectoryScores(chamfer_marks=chamfer_marks, per_frame=per_frame), collections.Counter()
    by_rows = row_scorers(row_sets, per_frame)
    dump_best: dict = {"motion": -1.0}   # episode picked for the viewer dump
    dump_list: list = []                 # first dump_k val episodes (multi-episode dumps)

    for i, (batch, source) in enumerate(iter_clips(ds, max_clips)):
        sources[source] += batch.batch_size
        case = getattr(batch, "eval_case", None)
        batch = batch.to(device)
        seed_for_device(_batch_seed(seed, i), device)
        _validate_prediction(batch.future, batch.future, batch.get_valid_mask())
        C = batch.context_len
        rollout = windowed_rollout(model, diffusion, batch, horizon=horizon, future_len=F_train,
                                   num_inference_steps=steps, schedule=schedule,
                                   uncertainty_scale=uscale, clamp_actuated=clamp_actuated,
                                   hide_given_actions=hide_given_actions)
        H = rollout["horizon"]
        target_len = C + H
        pred_full = rollout["positions"][:, C:]                 # [B, H, N, 3]
        gt_full = batch.positions[:, C:target_len]
        B = pred_full.shape[0]
        valid = batch.get_valid_mask()
        if dump_dir is not None:
            # Include the anchor->first-future transition; H=1 is a valid rollout.
            motion = torch.diff(batch.positions[:, C - 1:target_len], dim=1).norm(dim=-1)
            for e in range(B) if dump_pick == "max_motion" else ([0] if i == 0 else []):
                v = valid[e]
                # GT motion score: PEAK per-frame mean displacement — a burst of
                # folding registers even when most of the horizon is static.
                mot = motion[e, :, v].mean(dim=1).max().item()
                if mot > dump_best["motion"]:
                    dump_best.update(
                        motion=mot,
                        pred=pred_full[e].cpu(), gt=gt_full[e].cpu(),
                        context=batch.positions[e, :C].cpu(),
                        valid=v.cpu(), object_ids=batch.object_ids[e].cpu(),
                        action_mask=batch.action_mask[e].cpu())
        if dump_dir is not None and dump_k > 1:
            for e in range(B):
                if len(dump_list) >= dump_k:
                    break
                v = valid[e]
                mot = motion[e, :, v].mean(dim=1).max().item()
                dump_list.append(dict(
                    motion=mot, batch=i, elem=e,
                    pred=pred_full[e].cpu(), gt=gt_full[e].cpu(),
                    context=batch.positions[e, :C].cpu(), valid=v.cpu(),
                    object_ids=batch.object_ids[e].cpu(),
                    action_mask=batch.action_mask[e].cpu()))
        scores.add(batch, pred_full, gt_full)
        _add_row_sets(by_rows, batch, pred_full, gt_full, batch.positions[:, :target_len])
        if case_extras is not None:
            case_extras.add(batch, case, gt_full, batch.positions[:, :target_len], scores, by_rows)
            case_extras.save_prediction(batch, case, pred_full)

    if dump_dir is not None and dump_list:
        import numpy as np
        os.makedirs(dump_dir, exist_ok=True)
        index = []
        for j, d in enumerate(dump_list):
            idx = d["valid"].nonzero(as_tuple=True)[0]
            if idx.numel() > 4000:                          # viewer decimation
                g = torch.Generator().manual_seed(0)
                idx = idx[torch.randperm(idx.numel(), generator=g)[:4000]].sort().values
            np.savez_compressed(
                os.path.join(dump_dir, f"{domain}_{j:02d}.npz"),
                pred=d["pred"][:, idx].numpy().astype("float32"),
                gt=d["gt"][:, idx].numpy().astype("float32"),
                context=d["context"][:, idx].numpy().astype("float32"),
                object_ids=d["object_ids"][idx].numpy().astype("uint8"),
                action_mask=d["action_mask"][idx].numpy())
            index.append({"file": f"{domain}_{j:02d}.npz", "batch": d["batch"], "elem": d["elem"],
                          "gt_peak_motion_mm_per_frame": d["motion"] * 1000.0,
                          "n_points": int(idx.numel()), "horizon": int(d["pred"].shape[0])})
        json.dump(index, open(os.path.join(dump_dir, f"{domain}_index.json"), "w"), indent=1)
        print(f"  [{domain}] dumped {len(dump_list)} episodes -> {dump_dir}")
    if dump_dir is not None and dump_best["motion"] >= 0:
        import numpy as np
        os.makedirs(dump_dir, exist_ok=True)
        idx = dump_best["valid"].nonzero(as_tuple=True)[0]
        if idx.numel() > 4000:                              # viewer decimation
            g = torch.Generator().manual_seed(0)
            idx = idx[torch.randperm(idx.numel(), generator=g)[:4000]].sort().values
        np.savez_compressed(
            os.path.join(dump_dir, f"{domain}.npz"),
            pred=dump_best["pred"][:, idx].numpy().astype("float32"),
            gt=dump_best["gt"][:, idx].numpy().astype("float32"),
            context=dump_best["context"][:, idx].numpy().astype("float32"),
            object_ids=dump_best["object_ids"][idx].numpy().astype("uint8"),
            action_mask=dump_best["action_mask"][idx].numpy())
        print(f"  [{domain}] dump episode GT motion {dump_best['motion'] * 1000:.2f} mm/frame "
              f"(pick={dump_pick})")

    return _finish_metrics({**scores.result(), **_row_results(by_rows),
                            **(case_extras.result() if case_extras is not None else {}),
                            **read_provenance(sources, getattr(ds, "shards", ()))})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--domains", default=None,
                   help="default: checkpoint training domains; explicit comma-separated list enables cross-domain evaluation")
    p.add_argument("--clips", required=True,
                   help="'all' scores the whole validation split (reported numbers); N scores the "
                        "first N clips of its stratified order (a labelled quick check)")
    p.add_argument("--max_batches", default=None, help=argparse.SUPPRESS)
    p.add_argument("--rollout", action="store_true",
                   help="overlap-commit autoregression (inference.windowed_rollout) instead of "
                        "one model window per clip")
    p.add_argument("--rollout_frames", type=int, default=48,
                   help="requested rollout horizon in frames; each clip is capped at its own length "
                        "(pickup clips allow 36, cloth episodes reach 421)")
    p.add_argument("--schedule", default=None, choices=["pyramid", "full_sequence"],
                   help="sampler schedule for every window; default = the checkpoint's df_schedule "
                        "(pyramid for current checkpoints). full_sequence samples the whole window at "
                        "one shared noise level, still committed window by window")
    p.add_argument("--future_len", type=int, default=None,
                   help="window override; default = the checkpoint's own data.future_len "
                        "(per-domain lengths if the checkpoint has no fixed length)")
    p.add_argument("--context_len", type=int, default=None,
                   help="default = the checkpoint's own data.context_len")
    p.add_argument("--dump_pick", default="first", choices=["first", "max_motion"],
                   help="which episode the viewer dump shows: the first val episode "
                        "(legacy) or the highest-GT-motion episode among evaluated batches")
    p.add_argument("--chamfer_marks", default=None,
                   help="rollout only: comma-separated FRAME marks at which to report "
                        "paper-protocol symmetric Chamfer (cm) on deformable rows "
                        "(e.g. 5,25,50,250 = native t 10/50/100/500 on stride-2 cloth)")
    p.add_argument("--dump_k", type=int, default=1,
                   help="also dump the first K val episodes per domain as <domain>_NN.npz "
                        "(+ <domain>_index.json); 1 = legacy single <domain>.npz only")
    p.add_argument("--dump_dir", default=None,
                   help="rollout only: save episode-0 pred/GT point clouds per domain")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--clamp_actuated", action="store_true",
                   help="materialize ACTUATED rigid bodies exactly from their given pose_actions "
                        "(grippers, zero-motion fixtures) instead of the network's prediction; "
                        "default off = the historical protocol")
    p.add_argument("--mpm_val_root", default=None,
                   help="override the mpm val root (e.g. the ORIGINAL paper-era "
                        "robert_mpm/webdataset_test) — same shift/stats/windows")
    p.add_argument("--pickup_roots", default=None,
                   help="comma-separated override of the pickup roots (e.g. ONE task root "
                        "for a per-task stratified eval); the val split/blocklist rules "
                        "are unchanged")
    p.add_argument("--test_folder", action="append", default=[], metavar="DOMAIN=FOLDER",
                   help="score another test folder of an indexed dataset the checkpoint trained on its own data.roots: a "
                        "release's report folder, e.g. mpm_fresh=<release>/mpm_fresh/test_seen. The folder replaces the "
                        "validation folder for this evaluation and must be disjoint from the train folder. Repeatable.")
    p.add_argument("--cases", default=None,
                   help="a frozen case manifest (data/manifests/eval_cases/<dataset>_v1.json, docs/EVAL_PROTOCOLS.md): "
                        "score exactly the clips of --case_set instead of the loader's order; needs --clips all and "
                        "one domain. Rollout sets need --rollout with --rollout_frames at least the set's horizon")
    p.add_argument("--case_set", default=None, help="windows | windows_motion | rollouts_first | rollouts_active")
    p.add_argument("--case_tag", default=None, help="with --cases: only cases carrying this tag (q1..q4, beyond)")
    p.add_argument("--case_stratum", default=None, help="with --cases: only cases of this stratum (fold action, yikai population)")
    p.add_argument("--save_predictions", default=None, metavar="DIR",
                   help="with --cases: also write every case's predicted positions, one <position in the set>.npz per case")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    if (args.cases is None) != (args.case_set is None):
        p.error("--cases and --case_set go together")
    if args.cases is None and (args.case_tag is not None or args.case_stratum is not None or args.save_predictions):
        p.error("--case_tag, --case_stratum and --save_predictions go with a --cases set")
    domains = [d.strip() for d in args.domains.split(",")] if args.domains is not None else None
    if domains is not None and (not domains or len(set(domains)) != len(domains) or set(domains) - set(DOMAIN_FUTURE_LEN)):
        p.error("domains must be distinct supported domain names")
    if args.max_batches is not None:
        raise ValueError("--max_batches is retired: it counted shard records of 1 to 4 clips each. "
                         "Use --clips N or --clips all")
    max_clips = parse_clips(args.clips)
    if args.cases is not None and (max_clips is not None or domains is None or len(domains) != 1):
        p.error("--cases scores a whole case set of one dataset: pass --clips all and one --domains name")
    _validate_request(max_clips, args.context_len if args.context_len is not None else 2,
                      args.future_len)
    test_folders = {}
    for item in args.test_folder:
        name, _, folder = item.partition("=")
        if not name or not folder or name in test_folders:
            p.error("--test_folder takes DOMAIN=FOLDER, once per dataset")
        test_folders[name] = folder
    roots_override = {}
    if args.mpm_val_root:
        roots_override["mpm"] = [args.mpm_val_root]
    if args.pickup_roots:
        roots_override["pickup"] = [r for r in args.pickup_roots.split(",") if r]
        if not roots_override["pickup"]:
            p.error("pickup_roots must contain at least one path")

    # Ambient seed for anything outside the per-batch draws; the sampler noise
    # itself is reseeded per batch (see _batch_seed) so a domain's number does not
    # depend on which domains ran before it.
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, diffusion, cfg, _steps = load_model_and_diffusion(args.checkpoint, device)
    model.eval()
    hide_given = hides_given_actions(cfg)
    if domains is None:
        from data.unified_dataset import selected_domains
        domains = list(selected_domains(cfg.data.get("domains"), cfg.data.get("domain_weights"))[0])

    # Evaluate with the windows the checkpoint was TRAINED on unless explicitly overridden.
    # 2026-08-30: the old hard default of context_len=2 silently ran the C=4 unified UNets on
    # 2 context frames, and pickup on 38-frame windows instead of the trained 16, across the
    # whole 17-cell normalization comparison (pickup_pick final error 32.1 -> 19.6 mm once
    # matched). The model never checks; only this default does.
    _tc = int(cfg.data.get("context_len", 2))
    _tf = int(cfg.data.get("future_len", -1) or -1)
    if args.context_len is None:
        args.context_len = _tc
    if args.future_len is None:
        args.future_len = _tf if _tf > 0 else None
    print(f"[eval_unified] windows: context_len={args.context_len} "
          f"future_len={args.future_len or 'per-domain'} "
          f"(checkpoint trained with C={_tc}, F={_tf if _tf > 0 else 'per-domain'})", flush=True)
    if args.context_len != _tc or (args.future_len or None) != (_tf if _tf > 0 else None):
        print("[eval_unified] WARNING: evaluating with windows that differ from training", flush=True)
    checkpoint_schedule = cfg.get("df_schedule", "pyramid")
    schedule = args.schedule or checkpoint_schedule
    print(f"[eval_unified] schedule: {schedule} (checkpoint {checkpoint_schedule})", flush=True)
    if schedule != checkpoint_schedule:
        print("[eval_unified] WARNING: sampling with a schedule that differs from the checkpoint's", flush=True)

    normalization = resolve_normalization(cfg.normalization)
    print(f"[eval_unified] normalization: {normalization['profile']} {normalization['sha256']}")
    results = {"checkpoint": args.checkpoint, "checkpoint_sha256": file_sha256(args.checkpoint),
               "clips_requested": "all" if max_clips is None else max_clips,
               "seed": args.seed, "rollout": args.rollout,
               "clamp_actuated": bool(args.clamp_actuated),
               "actions": "hidden" if hide_given else "given",
               "normalization": normalization, "context_len": args.context_len,
               "future_len": args.future_len, "inference_steps": int(cfg.flow_matching.inference_steps),
               "df_schedule": schedule, "checkpoint_df_schedule": checkpoint_schedule,
               "df_uncertainty_scale": float(cfg.get("df_uncertainty_scale", 1.0)),
               "torch_version": str(torch.__version__), **code_revision(),
               "inputs": {},
               "domains": domains, "explicit_cross_domain_selection": args.domains is not None}
    if args.rollout:
        results["rollout_frames_requested"] = args.rollout_frames
    if set(test_folders) - set(domains):
        raise ValueError(f"--test_folder names datasets this command does not score: {sorted(set(test_folders) - set(domains))}")
    if test_folders:
        results["test_folders"] = dict(test_folders)      # the scored folder is not the checkpoint's validation folder
    for d in domains:
        print(f"=== {d} ===", flush=True)
        dataset = evaluation_dataset(cfg, d, args.context_len, args.future_len, roots_override, test_folders)
        results["inputs"][d] = _input_provenance(dataset, normalization, command_line=d in roots_override or d in test_folders)
        row_sets, case_extras = (), None
        if args.cases is not None:
            from utils.eval_cases import CaseDataset
            dataset = CaseDataset(dataset, args.cases, args.case_set, tag=args.case_tag, stratum=args.case_stratum)
            horizons = {int(case["horizon"]) for case in dataset.cases}
            if args.rollout != (args.case_set.startswith("rollouts")):
                p.error("rollout case sets need --rollout; window case sets must not have it")
            if args.rollout and args.rollout_frames < max(horizons):
                p.error(f"--rollout_frames {args.rollout_frames} is shorter than the set's horizon {max(horizons)}")
            if not args.rollout and horizons != {args.future_len or DOMAIN_FUTURE_LEN[d]}:
                p.error("the window cases were frozen with another future length than this evaluation's")
            row_sets = ("predicted", "moving", "still")
            case_extras = CaseExtras(row_sets, dataset.cases, args.save_predictions)
            per_case = Path(args.out).with_suffix(".cases.json")
            results["cases"] = {"manifest": args.cases, "manifest_sha256": file_sha256(args.cases), "case_set": args.case_set,
                                "case_tag": args.case_tag, "case_stratum": args.case_stratum,
                                "cases": len(dataset), "dataset": dataset.manifest["dataset"],
                                "moving_mm": dataset.manifest["moving_mm"], "per_case": per_case.name,
                                "predictions": str(Path(args.save_predictions).resolve()) if args.save_predictions else None}
        if args.rollout:
            results[d] = rollout_domain(model, diffusion, cfg, d, max_clips,
                                        device, horizon=args.rollout_frames, seed=args.seed,
                                        hide_given_actions=hide_given, schedule=schedule,
                                        dump_dir=args.dump_dir,
                                        dump_pick=args.dump_pick,
                                        clamp_actuated=args.clamp_actuated,
                                        dump_k=args.dump_k,
                                        future_len=args.future_len,
                                        context_len=args.context_len,
                                        dataset=dataset, row_sets=row_sets, case_extras=case_extras,
                                        chamfer_marks=[int(x) for x in
                                                       args.chamfer_marks.split(",")]
                                        if args.chamfer_marks else None)
        else:
            results[d] = eval_domain(model, diffusion, cfg, d, max_clips, device,
                                     future_len=args.future_len, seed=args.seed, hide_given_actions=hide_given,
                                     schedule=schedule, row_sets=row_sets, case_extras=case_extras,
                                     context_len=args.context_len, dataset=dataset, clamp_actuated=args.clamp_actuated)
        print(f"  {_without_curves(results[d])}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=1, allow_nan=False) + "\n")
    print(f"wrote {args.out}")
    if args.cases is not None:                            # one record per case, with its per-frame curves
        per_case.write_text(json.dumps({"receipt": Path(args.out).name, "case_set": args.case_set, "curves": "mm per scored "
                                        "frame (pose_passive_deg in degrees); zero = no motion, hold = the last context velocity",
                                        "cases": case_extras.records}, allow_nan=False, separators=(",", ":")) + "\n")
        print(f"wrote {per_case}")


if __name__ == "__main__":
    main()
