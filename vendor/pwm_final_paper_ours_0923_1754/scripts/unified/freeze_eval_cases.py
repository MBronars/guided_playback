"""Freeze the evaluation cases of one dataset's test folder (protocol: docs/EVAL_PROTOCOLS.md).

The test folder is read through a training recipe exactly as eval_unified.py reads it
(evaluation_dataset plus the rigid-body sidecar), so a case is the clip the evaluator scores.
Output: data/manifests/eval_cases/<name>_v1.json with these case sets

  windows          one window per test episode: the loader's own evaluation window, which
                   `eval_unified.py --clips all` scores and training-time validation reads
  windows_motion   long episodes only: four windows per episode at the within-episode motion
                   quantiles 0.125 / 0.375 / 0.625 / 0.875 of the window's mean predicted step
  rollouts_first   a stratified subset of episodes (--rollouts, 0 = every episode) rolled out from
                   the first allowed frame for --rollout_frames (or to the episode's end)
  rollouts_active  long episodes only: the same episodes from the span with the most motion
  rollouts_full    long-recording datasets with --full_rollouts: the same episodes from the first allowed
                   frame to the episode's last frame (the other datasets' rollouts_first already end there)

Every case carries the mm error of predicting no motion and of holding the last context velocity
(utils/eval_cases.py), so a model number always has its trivial references on the same clip.

Usage:
  PYTHONPATH=. python scripts/unified/freeze_eval_cases.py --recipe <recipe.yaml> --domain yikai \
      --rollouts 256 --rollout_frames 128 --extent_tags data/manifests/yikai_extent_tags_20260918.json \
      --out data/manifests/eval_cases/yikai_v1.json
"""
from __future__ import annotations

import argparse
import datetime
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from utils.eval_cases import (FIELDS, KIND, MOVING_MM, QUANTILE_LEVELS, VERSION, active_start, case_numbers,
                              hash_order, load_cases, motion_profile, quantile_starts)

STRATUM_FIELD = {"folding": "action", "dex": "teacher", "yikai": "population",
                 "mpm_fresh": "evaluation_category", "mpm_drag": "evaluation_category"}   # the MPM suite's record field
LONG_EPISODES = ("folding", "yikai", "dex")             # recordings far longer than a window: motion-selected sets too
# Dex points are computed from link poses by a batched product whose float32 rounding depends on the number of frames
# (up to 6e-8 m between a window and the slice of its episode, measured 2026-09-18). Its case numbers therefore come from
# the adapter's own window, the clip the evaluator's reader yields; stored-position datasets slice the episode, bit for bit.
COMPUTED_POSITIONS = ("dex",)
PUSHING_ROLLOUT_SHARDS, PUSHING_ROLLOUT_CLIPS = 32, 8


def _command(case_set, rollout_frames=None):
    """The evaluator words that score a case set of the manifest being written."""
    rollout = f" --rollout --rollout_frames {rollout_frames}" if rollout_frames else ""
    return f"eval_unified.py --cases <this manifest> --case_set {case_set}{rollout} --clips all"


def name_test_folder(sets, domain, folder):
    """A report folder (pwm-data-v1/mpm_fresh/test_seen) is scored through --test_folder: say so in every set's command."""
    words = f"eval_unified.py --test_folder {domain}={Path(folder).resolve()} "
    for entry in sets.values():
        assert "eval_unified.py " in entry["evaluator"]
        entry["evaluator"] = entry["evaluator"].replace("eval_unified.py ", words)
    return sets


def _overlaps(ranges, start, frames):
    return any(lo <= start + frames - 1 and start <= hi for lo, hi in ranges)


def freeze_indexed(ds, domain, args, beyond):
    """Indexed episodes (folding, yikai, dex, mpm_*): one pass per episode over its whole recording."""
    from data.indexed_episodes import evaluation_records
    C, F, first = ds.context_len, ds.future_len, ds.first_start
    order = evaluation_records(ds.records, domain)
    if args.limit:
        order = order[:args.limit]
    rollout_ids = {r["episode_id"] for r in (order[:args.rollouts] if args.rollouts else order)}
    stream = iter(ds)                                    # the evaluator's own reading, for the first episodes
    sets = {name: [] for name in ("windows", "windows_motion", "rollouts_first", "rollouts_active", "rollouts_full")}
    started = time.time()
    for position, record in enumerate(order):
        ds.future_len = F
        seeded = ds._draw_starts(record, 1)[0]
        if position < args.verify_stream:
            next(stream)
            read = ds.last_windows[0]
            if (read["episode_id"], read["start"]) != (record["episode_id"], seeded):
                raise ValueError(f"The loader read {read['episode_id']}@{read['start']}, the freezer "
                                 f"{record['episode_id']}@{seeded}")
        ds.future_len = -1
        whole = ds.window(record, first)                 # frames first..end of the episode
        ds.future_len = F
        if whole.body_type is None and domain not in ("mpm_fresh", "mpm_drag"):
            raise ValueError("Rigid scenes need the body sidecar to tell predicted rows from actuators")
        profile = motion_profile(whole)
        # Motion-selected cases only where an episode is long enough to offer a choice of spans.
        long_episode = domain in LONG_EPISODES and record["frames"] - first >= 4 * (C + F)
        stratum = str(record.get(STRATUM_FIELD.get(domain), "all"))
        ranges = beyond.get(record["episode_id"], ())

        def case(start, horizon, tag=""):
            if domain in COMPUTED_POSITIONS:
                ds.future_len = int(horizon)
                numbers = case_numbers(ds.window(record, int(start)))[0]
                ds.future_len = F
            else:
                numbers = case_numbers(whole, start - first, C + horizon)[0]
            labels = [tag] if tag else []
            if _overlaps(ranges, start, C + horizon):
                labels.append("beyond")
            return [record["episode_id"], int(start), int(horizon), stratum, ",".join(labels), *numbers]

        sets["windows"].append(case(seeded, F))
        if domain not in COMPUTED_POSITIONS and position % args.verify_every == 0:   # a sliced case is the adapter's window
            window = ds.window(record, seeded)
            lo = seeded - first
            if (not torch.equal(window.positions, whole.positions[:, lo:lo + C + F])
                    or not torch.equal(window.action_mask, whole.action_mask)
                    or not torch.equal(window.get_valid_mask(), whole.get_valid_mask())):
                raise ValueError(f"{record['episode_id']}: a window differs from the slice of its episode")
            # Pose references come from the episode's poses; the window's own poses must give the same numbers.
            ours, theirs = sets["windows"][-1][5:], case_numbers(window)[0]
            if any((a is None) != (b is None) or (a is not None and abs(a - b) > 2e-3) for a, b in zip(ours, theirs)):
                raise ValueError(f"{record['episode_id']}: case numbers differ between the window and its episode: "
                                 f"{ours} vs {theirs}")
        if long_episode:
            for k, start in enumerate(quantile_starts(profile, first, C, F)):
                sets["windows_motion"].append(case(start, F, f"q{k + 1}"))
        if record["episode_id"] in rollout_ids:
            horizon = min(args.rollout_frames, record["frames"] - first - C)
            sets["rollouts_first"].append(case(first, horizon))
            if long_episode:
                sets["rollouts_active"].append(case(active_start(profile, first, C, horizon), horizon))
            if getattr(args, "full_rollouts", False) and domain in LONG_EPISODES:
                sets["rollouts_full"].append(case(first, record["frames"] - first - C))
        if position % 25 == 0:
            print(f"  [{domain}] {position + 1}/{len(order)} episodes, {time.time() - started:.0f}s", flush=True)
    rollout_clips = f"--clips {args.rollouts}" if args.rollouts and args.rollouts < len(ds.records) else "--clips all"
    notes = {
        "windows": ("one window per test episode in the loader's stratified order, start drawn from the episode id "
                    "(IndexedEpisodeDataset._draw_starts)",
                    f"eval_unified.py --clips all (the loader's own order, no row sets), or {_command('windows')}"),
        "windows_motion": ("four windows per episode at the within-episode quantiles "
                           f"{list(QUANTILE_LEVELS)} of the window's mean predicted step (ties: earliest start)",
                           _command("windows_motion")),
        "rollouts_first": (f"the first {len(rollout_ids)} episodes of the stratified order, from frame {first} for "
                           f"{args.rollout_frames} frames (capped by the episode)",
                           f"eval_unified.py --rollout --rollout_frames {args.rollout_frames} {rollout_clips} (the loader's own "
                           f"order, no row sets), or {_command('rollouts_first', args.rollout_frames)}"),
        "rollouts_active": ("the same episodes from the start whose scored frames carry the most predicted motion "
                            "(ties: earliest start)", _command("rollouts_active", args.rollout_frames)),
        "rollouts_full": (f"the same episodes from frame {first} to the episode's last frame",
                          _command("rollouts_full", max((c[2] for c in sets["rollouts_full"]), default=0))),
    }
    return {name: dict(selection=notes[name][0], evaluator=notes[name][1], cases=cases)
            for name, cases in sets.items() if cases}


def freeze_rl_push(ds, args):
    """rl_push: the loader's seeded window of every usable episode; rollouts to the episode's end, spread
    over the objects in the loader's round-robin order, episodes within an object in hash order."""
    from data.rl_push import window_points
    C, F, first = ds.context_len, ds.future_len, ds.first_start
    by_object = {}
    for i, e in ds.episodes:
        by_object.setdefault(i, []).append(e)
    order = ds._eval_order()[:args.limit or None]
    windows, eligible = [], {}
    for i in order:
        obj, rows = ds.objects[i], ds._object_rows(ds.objects[i])
        items = ds._eval_windows(i, by_object.get(i, ()))
        if not items:
            continue
        batch = ds._tagged([window_points(obj, e, start, L, ds.links, rows) for L, e, start in items], obj)
        if batch.body_type is None:
            raise ValueError("rl_push cases need the body sidecar to tell the object from the gripper")
        for (L, e, start), numbers in zip(items, case_numbers(batch)):
            windows.append([f"{obj.file}:{e}", int(start), int(L - C), obj.category, "", *numbers])
        eligible[i] = [int(name) for name in hash_order([str(e) for _, e, _ in items], f"{obj.asset_key}:")]
    rollouts, depth = [], 0
    wanted = args.rollouts or sum(len(v) for v in eligible.values())
    while len(rollouts) < wanted and any(depth < len(eligible[i]) for i in eligible):
        for i in order:
            if i in eligible and depth < len(eligible[i]) and len(rollouts) < wanted:
                obj, e = ds.objects[i], eligible[i][depth]
                L = int(obj.lengths[e]) - first
                batch = ds._tagged([window_points(obj, e, first, L, ds.links, ds._object_rows(obj))], obj)
                rollouts.append([f"{obj.file}:{e}", first, L - C, obj.category, "", *case_numbers(batch)[0]])
        depth += 1
    return {
        "windows": dict(selection="one window per usable episode (C + F frames from frame 3 on), objects in the loader's "
                                  "round-robin order, start drawn from (subset seed, object, episode) "
                                  "(UnifiedRLPushDataset._eval_windows)",
                        evaluator=f"eval_unified.py --clips all (the loader's own order, no row sets), or {_command('windows')}",
                        cases=windows),
        "rollouts_first": dict(selection=f"{len(rollouts)} episodes taken round-robin over the objects in the loader's "
                                         "order, each object's usable episodes in sha256(object:episode) order, from "
                                         f"frame {first} to the episode's end",
                               evaluator=_command("rollouts_first", max(case[2] for case in rollouts)) +
                                         " (a --clips prefix would cover a few objects only)",
                               cases=rollouts),
    }


def freeze_pushing(ds, args):
    """pushing: the prefix window of every test episode; whole-clip rollouts on the first clips of
    hash-chosen shards. A record is a pre-batch of episodes; a case id is shard:key:row."""
    C, F = ds.context_len, ds.future_len
    ds.future_len = 10_000                               # the evaluator's rollout convention: keep every frame
    chosen = set(hash_order(sorted(Path(s).name for s in ds.shards), "pushing-rollouts:")[:PUSHING_ROLLOUT_SHARDS])
    windows, rollouts, taken, skipped = [], [], Counter(), []
    for count, sample in enumerate(ds._create_pipeline()):
        if args.limit and count >= args.limit:
            break
        shard, key = Path(str(sample["__url__"])).name, str(sample["__key__"])
        batch = ds._process_sample(sample)
        if not torch.isfinite(batch.positions).all():    # the loader skips such a record
            skipped.append(f"{shard}:{key}")
            continue
        if batch.body_type is None:
            raise ValueError("pushing cases need the body sidecar to tell the block from the rod")
        T = batch.positions.shape[1]
        if T < C + F:
            raise ValueError(f"{shard}:{key} holds {T} frames, a window needs {C + F}")
        prefix = case_numbers(batch, 0, C + F)
        whole = case_numbers(batch) if shard in chosen and taken[shard] < PUSHING_ROLLOUT_CLIPS else None
        for row in range(batch.positions.shape[0]):
            windows.append([f"{shard}:{key}:{row}", 0, F, "all", "", *prefix[row]])
            if whole is not None and taken[shard] < PUSHING_ROLLOUT_CLIPS:
                rollouts.append([f"{shard}:{key}:{row}", 0, T - C, "all", "", *whole[row]])
                taken[shard] += 1
        if count % 250 == 0:
            print(f"  [pushing] {count + 1} records", flush=True)
    ds.future_len = F
    out = {
        "windows": dict(selection="the prefix window (frames 0..C+F-1) of every test episode, shards in sorted order",
                        evaluator=f"eval_unified.py --clips all (the loader's own order, no row sets), or {_command('windows')}",
                        cases=windows),
        "rollouts_first": dict(selection=f"whole clips from frame 0: the first {PUSHING_ROLLOUT_CLIPS} episodes of the "
                                         f"{PUSHING_ROLLOUT_SHARDS} shards first in sha256('pushing-rollouts:' + name) order",
                               evaluator=_command("rollouts_first", max(case[2] for case in rollouts)) +
                                         " (a --clips prefix would read one shard)",
                               cases=rollouts),
    }
    if skipped:
        out["windows"]["skipped_records"] = skipped
    return out


def freeze_pickup(ds, args):
    """pickup (pwm-data-v1): the prefix window of every test episode, and every test episode rolled out whole (40-frame clips, so
    36 predicted frames after 4 context frames). A record is a pre-batch of episodes and a case id is shard:key:row, read back by
    the evaluator's shard reader. The stratum is the collection (place, convex, pick, upright), which is the test shard's name."""
    C, F = ds.context_len, ds.future_len
    ds.future_len = 10_000                               # the evaluator's rollout convention: keep every frame
    windows, rollouts, skipped = [], [], []
    for count, sample in enumerate(ds._create_pipeline()):
        if args.limit and count >= args.limit:
            break
        shard, key = Path(str(sample["__url__"])).name, str(sample["__key__"])
        batch = ds._process_sample(sample)
        if not torch.isfinite(batch.positions).all():    # the loader skips such a record
            skipped.append(f"{shard}:{key}")
            continue
        if batch.body_type is None:
            raise ValueError("pickup cases need the body sidecar to tell the object from the gripper")
        T = batch.positions.shape[1]
        if T < C + F:
            raise ValueError(f"{shard}:{key} holds {T} frames, a window needs {C + F}")
        prefix, whole = case_numbers(batch, 0, C + F), case_numbers(batch)
        stratum = Path(shard).stem
        for row in range(batch.positions.shape[0]):
            windows.append([f"{shard}:{key}:{row}", 0, F, stratum, "", *prefix[row]])
            rollouts.append([f"{shard}:{key}:{row}", 0, T - C, stratum, "", *whole[row]])
    ds.future_len = F
    out = {
        "windows": dict(selection="the prefix window (frames 0..C+F-1) of every test episode, shards in sorted order",
                        evaluator=f"eval_unified.py --clips all (the loader's own order, no row sets), or {_command('windows')}",
                        cases=windows),
        "rollouts_first": dict(selection="every test episode rolled out whole from frame 0",
                               evaluator=_command("rollouts_first", max(case[2] for case in rollouts)),
                               cases=rollouts),
    }
    if skipped:
        out["windows"]["skipped_records"] = skipped
    return out


def summarize(cases):
    """Episode-weighted means of a case set, overall and per stratum."""
    index = {name: FIELDS.index(name) for name in FIELDS}

    def block(rows):
        column = lambda name: [r[index[name]] for r in rows if r[index[name]] is not None]
        mean = lambda name: round(float(np.mean(column(name))), 3) if column(name) else None
        # The scorer averages pose errors over (clip, body) pairs: weight each case by its passive bodies.
        weights = [r[index["bodies_passive"]] for r in rows]
        pairs = lambda name: (round(float(np.average([r[index[name]] for r, w in zip(rows, weights) if w],
                                                     weights=[w for w in weights if w])), 3)
                              if any(weights) else None)
        return {"cases": len(rows), "episodes": len({r[0] for r in rows}),
                "horizon_min": min(r[2] for r in rows), "horizon_max": max(r[2] for r in rows),
                "cases_with_moving_rows": sum(1 for r in rows if r[index["rows_moving"]] > 0),
                **{name: mean(name) for name in FIELDS[8:] if name != "bodies_passive"},
                "zero_pose_passive_pairs_mm": pairs("zero_pose_passive_mm"),
                "zero_pose_passive_pairs_deg": pairs("zero_pose_passive_deg")}
    out = block(cases)
    strata = sorted({r[3] for r in cases})
    if 1 < len(strata) <= 12:
        out["by_stratum"] = {s: block([r for r in cases if r[3] == s]) for s in strata}
    tags = sorted({label for r in cases for label in r[4].split(",") if label})
    if tags:
        out["by_tag"] = {t: block([r for r in cases if t in r[4].split(",")]) for t in tags}
    return out


def build_manifest(sets, *, dataset, domain, inputs, window, code, recipe, extent_tags=None, partial=False):
    """The manifest of one dataset: provenance, the row and number conventions, and every case set with its summary."""
    return {
        "kind": KIND, "version": VERSION, "dataset": dataset, "domain": domain,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "protocol": "docs/EVAL_PROTOCOLS.md", "code": code, "recipe": recipe, "inputs": inputs, "window": window,
        "moving_mm": MOVING_MM,
        "rows": "all = valid particles; predicted = all minus rigid actuators and fixtures; moving = predicted rows that "
                f"leave their last context position by {MOVING_MM} mm or more within the scored frames",
        "numbers": "mm; zero = predict no motion, hold = hold the last context velocity; mean over scored frames and rows "
                   "of one clip (the per-clip averaging of TrajectoryScores), _final = at the last scored frame; "
                   "motion_mm_per_frame = mean step of the predicted rows over the scored transitions",
        **({"extent_tags": extent_tags} if extent_tags else {}),
        **({"partial": True} if partial else {}),
        "sets": {name: {"selection": entry["selection"], "evaluator": entry["evaluator"],
                        "summary": summarize(entry["cases"]),
                        **({"skipped_records": entry["skipped_records"]} if "skipped_records" in entry else {}),
                        "fields": list(FIELDS), "cases": entry["cases"]}
                 for name, entry in sets.items()},
    }


def write_manifest(path, manifest):
    """One case per line; the written file must read back as a valid manifest."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(manifest, allow_nan=False, separators=(",", ":"))
    path.write_text(text.replace('"cases":[[', '"cases":[\n[').replace("],[", "],\n[") + "\n")
    if load_cases(path) != json.loads(text):
        raise ValueError(f"{path} does not read back as written")


def main():
    from omegaconf import OmegaConf
    from scripts.unified.eval_unified import _input_provenance, evaluation_dataset
    from utils.config import load_config
    from utils.reproducibility import code_revision, file_sha256
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--recipe", required=True, help="a collection recipe whose data.roots name the release folders")
    p.add_argument("--domain", required=True)
    p.add_argument("--name", default=None, help="dataset folder name when it differs from the domain (folding_8mm)")
    p.add_argument("--test_folder", default=None, help="freeze on this folder of the same dataset instead of the recipe's validation "
                   "folder: a release's report folder (mpm_fresh/test_seen), which no recipe reads")
    p.add_argument("--rollouts", type=int, default=256, help="episodes in the rollout sets; 0 = every episode")
    p.add_argument("--rollout_frames", type=int, default=128, help="indexed datasets: rollout horizon (capped per episode)")
    p.add_argument("--full_rollouts", action="store_true", help="folding, yikai, dex: add the rollouts_full set, the rollout "
                   "episodes from their first allowed frame to their last frame (manifests frozen before 2026-09-19 have none)")
    p.add_argument("--extent_tags", default=None, help="yikai: the extent tag manifest; overlapping cases get the tag `beyond`")
    p.add_argument("--verify_every", type=int, default=25, help="indexed: compare every n-th window with its episode slice")
    p.add_argument("--verify_stream", type=int, default=3, help="indexed: first episodes also read through the loader's iterator")
    p.add_argument("--limit", type=int, default=0, help="debugging: first episodes/objects/records only; marks the manifest partial")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    if min(args.rollouts, args.rollout_frames - 1, args.verify_every - 1, args.verify_stream, args.limit) < 0:
        raise ValueError("Counts must be nonnegative and rollout_frames / verify_every positive")

    cfg = load_config(args.recipe, [])
    C, F = int(cfg.data.context_len), int(cfg.data.future_len)
    ds = evaluation_dataset(cfg, args.domain, C, F, test_folders={args.domain: args.test_folder} if args.test_folder else None)
    ds.emit_sidecar = True                               # as eval_unified._prepare: poses for metrics, body types for rows
    beyond = {}
    if args.extent_tags:
        tags = json.loads(Path(args.extent_tags).read_text())
        if tags.get("kind") != "yikai_extent_tags":
            raise ValueError("--extent_tags must be a yikai_extent_tags manifest")
        beyond = {row["episode_id"]: row["frames_beyond"] for row in tags["episodes"] if row["split"] == "test"}
    torch.manual_seed(0)
    if hasattr(ds, "records"):
        sets = freeze_indexed(ds, args.domain, args, beyond)
    elif args.domain == "rl_push":
        sets = freeze_rl_push(ds, args)
    elif args.domain == "pushing":
        sets = freeze_pushing(ds, args)
    elif args.domain == "pickup":
        sets = freeze_pickup(ds, args)
    else:
        raise ValueError(f"No case rules for {args.domain}")
    if args.test_folder:
        name_test_folder(sets, args.domain, args.test_folder)
    inputs = _input_provenance(ds, OmegaConf.to_container(cfg.normalization, resolve=True), command_line=bool(args.test_folder))
    if "episode_index_sha256" not in inputs:             # shard datasets: the release's per-folder index
        inputs["folder_index_sha256"] = {str(Path(root) / "INDEX.json"): file_sha256(Path(root) / "INDEX.json")
                                         for root in inputs["roots"] if (Path(root) / "INDEX.json").is_file()}
    manifest = build_manifest(sets, dataset=args.name or args.domain, domain=args.domain, inputs=inputs,
                              window={"context_len": C, "future_len": F, "first_start": int(getattr(ds, "first_start", 0))},
                              code=code_revision(), partial=bool(args.limit),
                              recipe={"path": str(Path(args.recipe).resolve()), "sha256": file_sha256(args.recipe)},
                              extent_tags=({"path": args.extent_tags, "sha256": file_sha256(args.extent_tags)}
                                           if args.extent_tags else None))
    write_manifest(args.out, manifest)
    for name, entry in manifest["sets"].items():
        print(f"[{manifest['dataset']}] {name}: {json.dumps({k: v for k, v in entry['summary'].items() if not k.startswith('by_')})}")
    print(f"FREEZE_OK {args.out} sha256 {file_sha256(args.out)[:12]}")


if __name__ == "__main__":
    main()
