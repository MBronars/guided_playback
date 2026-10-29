#!/usr/bin/env python3
"""Write the rl_push validation object list and the outlier episode list (JSON manifests read by data/rl_push.py).

  PYTHONPATH=. python scripts/dev/rl_push_manifests.py val --out data/manifests/rl_push_val_assets.json
  PYTHONPATH=. python scripts/dev/rl_push_manifests.py exclusions --stage1 <stage1_episodes.npz> \
      --stage2 <stage2_rows.json> --stage2_summary <stage2.json> --stage2_script <check_contacts.py> \
      --launches <launches.json> --out <new_exclusions.json>

val: in every (root, category) group of the release with at least --min_objects objects that hold episodes, a
seeded random --fraction of those objects (at least one) is held out for validation; smaller groups stay in
training. Object files without episodes are left out, as the loader skips them.

exclusions: episodes whose audit shows the object leaving the table by more than --airborne_mm, sinking more
than --sink_mm into it, the gripper more than --penetration_mm inside the object, or motion no contact can explain:
the push starting before the gripper can reach the object, the object gaining more than --gain_mps of horizontal
speed in one frame while contact is impossible after the spawn settle, a kick of that kind during the spawn settle
at event frame --kick_from_frame or later (the first frame a window may start at; event frame f compares the
transitions f-2 -> f-1 and f-1 -> f), or a contact launch: the object's horizontal centre leaving a contact more
than --launch_ratio times faster than the fastest gripper point. The audit receipts are the per-episode outputs of
the acceptance checks (stage 1 covers every episode; stage 2, with the contact metrics, and the launch check must
too). Episodes are keyed by asset and collection provenance, which survives re-exports. Both files are exclusive
creations unless --replace; checkpoints hash them (utils.checkpoint.corpus_identity).

Stage 2 already filters events. --gain_mps must equal its recorded detector cutoff:
settle-event gains are absent, so neither lowering nor raising that cutoff can be
implemented from these receipts. Regenerate stage 2 to change it. New manifests
also hash the threshold summary and detector reference script; old output-row
hashes alone do not establish the generating parameters or code.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import sys
import time
import zlib
from collections import Counter
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data.rl_push import EXCLUSIONS_KIND, RL_PUSH_FIRST_START, VAL_ASSETS_KIND, asset_key_of
from data.unified_dataset import RL_PUSH_ROOT
from utils.reproducibility import code_revision


def _sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def _write(path, value, replace):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w" if replace else "x") as f:
        json.dump(value, f, indent=1)
        f.write("\n")


def _episodes(root, rel):
    with np.load(os.path.join(root, rel), allow_pickle=False) as z:
        return len(z["episode_len"])


def val_assets(root, fraction, seed, min_objects):
    files = sorted(os.path.relpath(p, root) for p in glob.glob(os.path.join(root, "objects", "**", "*.npz"), recursive=True))
    empty = [rel for rel in files if not _episodes(root, rel)]              # the loader skips these as well
    groups = {}
    for key in map(asset_key_of, sorted(set(files) - set(empty))):
        groups.setdefault("/".join(key.split("/")[:2]), []).append(key)
    held, summary = [], {}
    for group, keys in sorted(groups.items()):
        n = len(keys)
        k = max(1, round(fraction * n)) if n >= min_objects else 0
        chosen = [keys[j] for j in np.random.default_rng([seed, zlib.crc32(group.encode())]).permutation(n)[:k]]
        held += chosen
        summary[group] = {"objects": n, "held_out": k}
    return sorted(held), summary, len(files), empty


def _contact_thresholds(summary):
    # The original row-only receipts support only the frozen 0.10 m/s recipe.
    thresholds = ({"accel_mps_per_frame": 0.10} if summary is None else
                  json.loads(Path(summary).read_text())["thresholds"])
    cutoff = thresholds["accel_mps_per_frame"]
    if not isinstance(cutoff, (int, float)) or isinstance(cutoff, bool) or not math.isfinite(cutoff) or cutoff <= 0:
        raise ValueError("stage-2 accel_mps_per_frame must be positive and finite")
    return thresholds


def exclusions(stage1, stage2, launches, airborne_mm, sink_mm, penetration_mm, gain_mps, kick_from_frame, launch_ratio,
               *, stage2_summary=None):
    cutoff = _contact_thresholds(stage2_summary)["accel_mps_per_frame"]
    if gain_mps != cutoff:
        raise ValueError(f"gain_mps must equal the stage-2 detector cutoff {cutoff}; "
                         "per-event settle gains and discarded events are unavailable; regenerate stage 2")
    if any(not math.isfinite(v) or v < 0 for v in (airborne_mm, sink_mm, penetration_mm, gain_mps, launch_ratio)):
        raise ValueError("exclusion thresholds must be finite and nonnegative")
    if not isinstance(kick_from_frame, int) or isinstance(kick_from_frame, bool) or kick_from_frame < 0:
        raise ValueError("kick_from_frame must be a nonnegative integer")
    s1 = dict(np.load(stage1, allow_pickle=False))                          # one read per array, not per episode
    rows = json.load(open(stage2))
    prov = ("source_run", "source_shard", "source_episode_index", "source_env")
    key = lambda asset, get: (asset, str(get("source_run")), str(get("source_shard")),
                              int(get("source_episode_index")), int(get("source_env")))
    def indexed(records, asset, numeric, label):
        out = {}
        for r in records:
            k = key(asset(r), r.get)
            if k in out:
                raise ValueError(f"duplicate {label} episode: {k}")
            if any(not math.isfinite(float(r[name])) for name in numeric):
                raise ValueError(f"nonfinite {label} metrics: {k}")
            out[k] = r
        return out
    contact = indexed(rows, lambda r: asset_key_of(str(r["file"])),
                      ("max_penetration_mm", "unexplained_later_max_gain_mps"), "stage 2")
    launch = indexed(json.load(open(launches))["rows"], lambda r: r["asset_key"],
                     ("max_ratio",), "launch check")
    found, missing = {}, Counter()
    seen = set()
    for j in range(len(s1["object_file"])):
        k = key(asset_key_of(str(s1["object_file"][j])), lambda name: s1[name][j])
        if k in seen:
            raise ValueError(f"duplicate stage 1 episode: {k}")
        seen.add(k)
        values = {"airborne_max_mm": float(s1["airborne_max_mm"][j]), "bottom_min_mm": float(s1["bottom_min_mm"][j])}
        if any(not math.isfinite(v) for v in values.values()):
            raise ValueError(f"nonfinite stage 1 metrics: {k}")
        reasons = []
        if values["airborne_max_mm"] > airborne_mm:
            reasons.append(f"object leaves the table by more than {airborne_mm} mm")
        if values["bottom_min_mm"] < -sink_mm:
            reasons.append(f"object sinks more than {sink_mm} mm into the table")
        if k in contact:
            c = contact[k]
            values["gripper_penetration_max_mm"] = float(c["max_penetration_mm"])
            if c["max_penetration_mm"] > penetration_mm:
                reasons.append(f"gripper more than {penetration_mm} mm inside the object")
            if c["onset_before_contact_possible"]:
                values["push_onset_frame"] = int(c["push_onset_frame"])
                reasons.append("object starts moving before the gripper can reach it")
            if c["unexplained_later_frames"] and c["unexplained_later_max_gain_mps"] > gain_mps:
                values["unexplained_gain_mps"] = float(c["unexplained_later_max_gain_mps"])
                reasons.append(f"object gains more than {gain_mps} m/s in one frame while contact is impossible")
            kicks = [int(f) for f in c["unexplained_settle_frames"] if f >= kick_from_frame]
            if kicks:
                values["settle_kick_frames"] = kicks
                reasons.append(f"object kicked with no contact possible at frame {kick_from_frame} or later")
        else:
            missing["stage 2"] += 1
        if k in launch:
            r = launch[k]
            if r["max_ratio"] > launch_ratio:
                if any(not math.isfinite(float(r[name])) for name in ("frame", "object_mps", "gripper_mps")):
                    raise ValueError(f"nonfinite launch check metrics: {k}")
                values.update(launch_ratio=float(r["max_ratio"]), launch_frame=int(r["frame"]),
                              launch_object_mps=float(r["object_mps"]), launch_gripper_mps=float(r["gripper_mps"]))
                reasons.append(f"object leaves a contact more than {launch_ratio}x faster than the gripper")
        else:
            missing["launch check"] += 1
        if reasons:
            found[k] = {"asset_key": k[0], **dict(zip(prov, k[1:])), "reasons": reasons, "values": values}
    if missing:
        raise ValueError(f"stage-1 episodes without a row in {dict(missing)}; run stage 2 and the launch check "
                         "on every episode")
    return [found[k] for k in sorted(found)], len(s1["object_file"])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    v = sub.add_parser("val")
    v.add_argument("--root", default=RL_PUSH_ROOT)
    v.add_argument("--fraction", type=float, default=0.10)
    v.add_argument("--min_objects", type=int, default=5)
    v.add_argument("--seed", type=int, default=0)
    x = sub.add_parser("exclusions")
    x.add_argument("--root", default=RL_PUSH_ROOT)
    x.add_argument("--stage1", required=True)
    x.add_argument("--stage2", required=True)
    x.add_argument("--stage2_summary", required=True, help="stage-2 summary containing detector thresholds")
    x.add_argument("--stage2_script", required=True, help="detector reference code to hash with the receipts")
    x.add_argument("--launches", required=True)
    x.add_argument("--airborne_mm", type=float, default=20.0)
    x.add_argument("--sink_mm", type=float, default=5.0)
    x.add_argument("--penetration_mm", type=float, default=5.0)
    x.add_argument("--gain_mps", type=float, default=0.10, help="must equal stage-2 detector cutoff; regenerate to retune")
    x.add_argument("--kick_from_frame", type=int, default=RL_PUSH_FIRST_START)
    x.add_argument("--launch_ratio", type=float, default=2.0)
    for p in (v, x):
        p.add_argument("--out", required=True)
        p.add_argument("--replace", action="store_true")
    a = ap.parse_args(argv)
    made = {"made": time.strftime("%Y-%m-%d %H:%M"), "script": "scripts/dev/rl_push_manifests.py", **code_revision()}
    if a.command == "val":
        assets, groups, n, empty = val_assets(a.root, a.fraction, a.seed, a.min_objects)
        value = {"kind": VAL_ASSETS_KIND, "release": os.path.realpath(a.root), "objects_in_release": n,
                 "objects_without_episodes": empty,
                 "rule": {"group": "asset root and category, objects with episodes", "fraction": a.fraction,
                          "min_objects": a.min_objects, "seed": a.seed,
                          "selection": "numpy default_rng([seed, crc32(group)]).permutation"},
                 "groups": groups, "held_out_objects": len(assets), **made, "assets": assets}
    else:
        episodes, scanned = exclusions(a.stage1, a.stage2, a.launches, a.airborne_mm, a.sink_mm, a.penetration_mm,
                                       a.gain_mps, a.kick_from_frame, a.launch_ratio, stage2_summary=a.stage2_summary)
        value = {"kind": EXCLUSIONS_KIND, "release": os.path.realpath(a.root), "episodes_scanned": scanned,
                 "criteria": {"airborne_mm": a.airborne_mm, "sink_mm": a.sink_mm, "penetration_mm": a.penetration_mm,
                              "moves_before_contact_is_possible": True, "unexplained_gain_mps": a.gain_mps,
                              "settle_kick_from_frame": a.kick_from_frame, "launch_ratio": a.launch_ratio},
                 "contact_audit": {"thresholds": _contact_thresholds(a.stage2_summary),
                                   "gain_policy": "both settle and later gains use the recorded detector cutoff",
                                   "provenance_scope": "summary and detector reference captured at manifest creation; "
                                                       "this does not prove historical receipt generation"},
                 "receipts": {os.path.realpath(p): _sha256(p) for p in
                              (a.stage1, a.stage2, a.launches, a.stage2_summary, a.stage2_script)},
                 "excluded_episodes": len(episodes),
                 "episodes_per_reason": dict(Counter(r for e in episodes for r in e["reasons"])),
                 **made, "episodes": episodes}
    _write(a.out, value, a.replace)
    print(json.dumps({k: value[k] for k in value if k not in ("assets", "episodes", "groups")}, indent=1))


if __name__ == "__main__":
    main()
