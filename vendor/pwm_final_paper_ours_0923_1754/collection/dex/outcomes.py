"""Per-episode outcome labels for dex-hand shards, from each task's termination rules and the recorded object state.

The generator's `terminated_early` counts every termination before the rollout ended, which for in-palm rotation is
mostly success (15 consecutive goals reached), not failure. Shards that record which termination fired
(`episode_end_reason`, names in the manifest's `done_reasons`) are labelled from it; older shards (v1) by the rules
below. Labels go to <teacher>_outcomes.npz/.json next to each shard. Shards are never modified, existing labels are left unchanged, and a shard without its manifest (still being
written) is skipped.

    python -m collection.dex.outcomes <out_dir>
"""
import argparse
import glob
import json
import os

import numpy as np

from .geometry import quat_to_R

# termination rules, from the task configs (tactile-genesis dexterous-hands/src/tasks/*/config.py)
DROP_Z = {"in_palm_rotate": 0.45, "in_hand_repose": 0.46}      # object below OBJECT_POS z - 0.05 / - 0.04
HAS_SUCCESS_TERMINATION = {"in_palm_rotate"}                     # reached_max_consecutive_successes (15)
OUTCOMES = {0: "ran the whole rollout", 1: "dropped", 2: "ended on a success streak", 3: "other early termination",
            4: "solver error"}
# recorded termination -> outcome; other task terms (screwdriver nut_stagnation) are "other early termination".
# solver_error: Eden resets envs the Genesis rigid solver flags (contact-buffer overflow, NaN); the frames before it
# are valid, the ending is not a physical event
REASON_OUTCOME = {"none": 0, "time_out": 0, "obj_dropped": 1, "handle_dropped": 1,
                  "reached_max_consecutive_successes": 2, "solver_error": 4}


def label(z, task, done_reasons=None):
    """{outcome [E] int8, end_object_z [E], and tilt_end_deg [E] for the screwdriver} of one shard.

    done_reasons: the manifest's list for shards that record episode_end_reason; None infers the outcome (v1)."""
    L = z["episode_length"]
    obj_pos = z["obj_pos"]
    E, T = len(L), obj_pos.shape[0]
    idx = np.arange(E)
    early = L < T
    end_z = obj_pos[L - 1, idx, 2]
    outcome = np.zeros(E, np.int8)
    if done_reasons is not None:
        outcome[:] = [REASON_OUTCOME.get(done_reasons[r], 3) for r in z["episode_end_reason"]]
    elif task in HAS_SUCCESS_TERMINATION:
        # drop vs success streak: the pose is recorded before the step whose result terminates, so a drop can end a
        # few cm above the threshold; a held object rests ~7 cm above it, so split halfway
        dropped = early & (end_z < DROP_Z[task] + 0.03)
        outcome[dropped] = 1
        outcome[early & ~dropped] = 2
    elif task in DROP_Z:
        outcome[early] = 1                                        # the task's only failure termination is the drop
    else:                                                         # screwdriver: every early end is a failure
        outcome[early] = 3
    out = {"outcome": outcome, "end_object_z": end_z}
    if task == "screwdriver":                                     # tilt of the long axis since the start
        oq = z["obj_quat"]
        q0, q1 = oq[0, idx], oq[L - 1, idx]
        R0 = quat_to_R(q0 / np.linalg.norm(q0, axis=-1, keepdims=True))
        R1 = quat_to_R(q1 / np.linalg.norm(q1, axis=-1, keepdims=True))
        a = np.array([0.0, 1.0, 0.0])
        cos = np.clip(np.einsum("ei,ei->e", R0 @ a, R1 @ a), -1, 1)
        out["tilt_end_deg"] = np.degrees(np.arccos(cos)).astype(np.float32)
    return out


def relabel(npz):
    """Write the labels of one finished shard; returns its summary, or None if labels already exist."""
    name = os.path.basename(npz)[:-len(".npz")]
    base = os.path.join(os.path.dirname(npz), f"{name}_outcomes")
    if os.path.exists(base + ".npz") or os.path.exists(base + ".json"):
        return None
    z = np.load(npz)
    reasons = json.load(open(npz[:-len(".npz")] + ".json")).get("done_reasons")
    out = label(z, name.partition("@")[0].rsplit("-", 1)[0], reasons)      # <teacher>[@<part>]
    np.savez(base + ".npz", **out)
    pert = z["episode_perturbed"]
    summary = {"shard": name, "episodes": len(out["outcome"]), "outcome_codes": OUTCOMES,
               "outcome_source": "recorded termination" if reasons else "inferred from the object state"}
    for group, m in (("clean", ~pert), ("perturbed", pert)):
        summary[group] = {OUTCOMES[k]: int(((out["outcome"] == k) & m).sum()) for k in OUTCOMES}
        summary[group]["episodes"] = int(m.sum())
    if "tilt_end_deg" in out:
        early = out["outcome"] > 0
        summary["tilt_end_deg_of_early_endings_p50"] = float(np.median(out["tilt_end_deg"][early])) if early.any() else None
    json.dump(summary, open(base + ".json", "w"), indent=2)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir")
    a = ap.parse_args()
    for npz in sorted(glob.glob(os.path.join(a.out_dir, "*.npz"))):
        if npz.endswith("_outcomes.npz") or not os.path.exists(npz[:-len(".npz")] + ".json"):
            continue
        s = relabel(npz)
        if s is None:
            print(f"{os.path.basename(npz)}: labels exist, left unchanged")
            continue
        c, p = s["clean"], s["perturbed"]
        print(f"{s['shard']:24s} clean {c['episodes']}: full {c[OUTCOMES[0]]}, dropped {c[OUTCOMES[1]]}, "
              f"success-streak {c[OUTCOMES[2]]}, other {c[OUTCOMES[3]]}, solver error {c[OUTCOMES[4]]} | "
              f"perturbed {p['episodes']}: dropped {p[OUTCOMES[1]]}, other {p[OUTCOMES[3]]}")


if __name__ == "__main__":
    main()
