"""Train-only calibration, real-data contract audits and pinned overfit fixtures.

All outputs are new files. Training/test separation is inherited from frozen
indices; official test episodes never enter calibration or fitting fixtures.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from data.batch import TrajectoryBatch
from data.indexed_episodes import IndexedEpisodeDataset
from data.native_episodes import reconstruction_error
from data.normalization import load_normalization_artifact, resolve_normalization
from data.stats_fitting import PriorityReservoir, _half_ranges, raw_motion_rows
from utils.pose_native import materialize_pose_native
from utils.reproducibility import file_sha256


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def audit_batch(batch):
    batch._validate_shapes()
    for field in ("positions", "actions", "E", "nu", "rho"):
        if not torch.isfinite(getattr(batch, field)).all():
            raise ValueError(f"Nonfinite {field}")
    valid = batch.get_valid_mask()
    gt = batch.future
    dp = batch.prediction_x0
    norm = batch.prediction_stats.normalize(dp, clip=False)
    rebuilt = batch.predictions_to_positions(batch.prediction_stats.denormalize(norm))
    particle_error = float((rebuilt - gt).norm(dim=-1).max())
    rigid_error = pose_error = 0.
    if batch.body_type is not None:
        rigid_error = reconstruction_error(batch)
        ridx = (batch.body_type[0] == 1).nonzero(as_tuple=True)[0]
        pose_norm = batch.pose_delta_stats.normalize(batch.pose_deltas_gt[:, :, ridx], clip=False)
        pred = materialize_pose_native(batch, norm, pose_norm)[0]
        pose_error = float((pred - gt).norm(dim=-1).max())
        torch.testing.assert_close(batch.pose_actions,
            batch.pose_deltas_gt * batch.pose_action_mask[:, None, :, None], atol=0, rtol=0)
    if max(particle_error, rigid_error, pose_error) > 1e-5:
        raise ValueError(f"Ground-truth round trip exceeds 0.01 mm: particle={particle_error}, rigid={rigid_error}, pose={pose_error}")
    realized = batch.future_deltas * batch.action_mask[:, None, :, None]
    command_gap = float((batch.actions - realized).abs().max())
    if batch.action_active is None:
        torch.testing.assert_close(batch.actions, realized, atol=0, rtol=0)
    primary = valid if batch.body_type is None else valid & ~batch.action_mask
    motion = (gt - batch.positions[:, batch.context_len - 1:batch.context_len]).norm(dim=-1)
    return dict(shape=list(batch.positions.shape), particle_roundtrip_max_mm=1000 * particle_error,
        rigid_reconstruction_max_mm=1000 * rigid_error, joint_target_roundtrip_max_mm=1000 * pose_error,
        command_vs_achieved_max_mm=1000 * command_gap, actuated_particles=int(batch.action_mask.sum()),
        primary_zero_motion_mm=float(motion[primary[:, None].expand_as(motion)].mean() * 1000),
        position_max_abs_m=float(batch.positions.abs().max()), normalized_target_max_abs=float(norm.abs().max()))


def selection(records, count, seed):
    # Same deterministic episode-uniform selection across free/jointed passes.
    return sorted(records, key=lambda r: hashlib.sha256(f"{seed}:{r['episode_id']}".encode()).digest())[:count]


def calibrate(root, domain, out, episodes=128, windows=4, seed=916, capacity=65536, particle_rows="all"):
    # particle_rows="supervised": the particle scale describes the rows the particle loss covers (data/stats_fitting.py).
    source_paths = [Path(p) for p in ("data/recording_adapters.py", "data/indexed_episodes.py",
        "data/npz_arrays.py", "data/native_episodes.py", "data/stats_fitting.py",
        "data/unified_dataset.py", "data/normalization.py", "utils/pose_native.py",
        "utils/pose_ops.py", "scripts/unified/verify_collections.py")]
    source_versions = {str(p): file_sha256(p) for p in source_paths}
    ds = IndexedEpisodeDataset(Path(root) / domain / "train", domain=domain, raw=True)
    chosen = selection(ds.records, episodes, seed)
    particle = PriorityReservoir(capacity, 3, seed)
    free = PriorityReservoir(capacity, 6, seed + 1)
    jointed = PriorityReservoir(capacity, 6, seed + 2)
    ledger, audits = [], []
    rng = np.random.default_rng(seed)
    for i, record in enumerate(chosen):
        # Uniform over the starts training may draw: from the domain's first allowed frame (yikai skips 10).
        starts = (ds.first_start + rng.integers(record["frames"] - 20 + 1 - ds.first_start, size=windows)).tolist()
        for start in starts:
            ds.articulated = False
            b = ds.window(record, start)
            metadata = dict(valid_particle_mask=b.get_valid_mask()[:, None].expand(-1, b.num_frames, -1))
            dp, dq = raw_motion_rows(b, metadata, particle_rows=particle_rows)
            particle.add(dp)
            if len(dq):
                free.add(dq)
                ds.articulated = True
                b = ds.window(record, start)
                _, dq = raw_motion_rows(b, metadata)
                jointed.add(dq)
            audits.append(audit_batch(b))
            ledger.append(dict(episode_id=record["episode_id"], file=record["file"], start=start))
        if (i + 1) % 16 == 0:
            print(f"[calibration] {domain} {i + 1}/{len(chosen)} episodes", flush=True)
    ps = _half_ranges(particle.rows)
    qs = None if not len(free.rows) else np.maximum(_half_ranges(free.rows, True), _half_ranges(jointed.rows, True))
    if source_versions != {str(p): file_sha256(p) for p in source_paths}:
        raise RuntimeError("Calibration source changed during the fit; repeat on a stable checkout")
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    cache = out / f"{domain}.npz"
    if cache.exists():
        raise FileExistsError(cache)
    np.savez_compressed(cache, particle=particle.rows, pose_free=free.rows, pose_jointed=jointed.rows)
    report = dict(domain=domain, train_index_sha256=ds.index["index_sha256"], episodes=len(chosen), windows=len(ledger),
        seed=seed, capacity=capacity, particle_rows=particle_rows, particle_rows_seen=particle.seen, pose_rows_seen=free.seen,
        particle_scales=ps.tolist(), pose_scales=None if qs is None else qs.tolist(),
        cache_sha256=file_sha256(cache), ledger=ledger,
        audit_maxima={k: max(a[k] for a in audits) for k in audits[0] if k != "shape"},
        policy="episode-uniform selection, uniform starts, uniform valid row priority reservoirs; max free/jointed pose p99 bounds",
        source_versions=source_versions, source_stable_during_fit=True)
    write_json(out / f"{domain}.json", report)
    return report


def assemble_artifact(reports, output, base="stats/normalization_rl_push.json", version="collections-xy-p99-2026-09-16-v2"):
    artifact = json.loads(Path(base).read_text())
    artifact["artifact_version"] = version
    scales = artifact["profiles"]["xy_p99_per_domain"]
    for report in reports:
        scales["particle"][report["domain"]] = report["particle_scales"]
        if report["pose_scales"] is not None:
            scales["pose"][report["domain"]] = report["pose_scales"]
    artifact["provenance"] = dict(base_artifact=str(base), base_sha256=file_sha256(base),
        reference_values="all existing entries and global profile unchanged",
        fit={r["domain"]: {k: v for k, v in r.items() if k != "ledger"} for r in reports},
        script="scripts/unified/verify_collections.py", created_utc=datetime.now(timezone.utc).isoformat())
    write_json(output, artifact)
    load_normalization_artifact(output)


def save_fixture(batch, name, out, normalization, source):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.pt"
    if path.exists():
        raise FileExistsError(path)
    audit = audit_batch(batch)
    torch.save(batch, path)
    write_json(path.with_suffix(".json"), dict(domain=batch.domain,
        stats_set=normalization["profile"], normalization=normalization, normalization_sha256=normalization["sha256"],
        sha256=file_sha256(path), shape=list(batch.positions.shape), C=batch.context_len, F=batch.num_future_frames,
        B=batch.batch_size, N=batch.num_particles, source=source, audit=audit,
        drawn=datetime.now(timezone.utc).isoformat()))
    print(f"[fixture] {name}: {audit}", flush=True)


def indexed_fixtures(root, domain, out, normalization, seed=916, batch_size=2, teacher=None, suffix=""):
    for split in ("train", "val"):
        ds = IndexedEpisodeDataset(Path(root) / domain / split, domain=domain, normalization=normalization)
        ds.articulated = True
        candidates = [r for r in ds.records if teacher is None or r.get("teacher") == teacher]
        if domain == "yikai":
            # Fit a reviewed ordinary interaction; retain all valid episodes in
            # the actual training corpus and calibration distribution.
            candidates = [r for r in candidates if not r.get("audit", {}).get("src_body", {}).get("flags")]
        best = None
        for record in selection(candidates, 6, seed):
            starts = sorted(set(np.linspace(0, record["frames"] - 20, 12).astype(int).tolist()))
            windows = []
            for start in starts:
                b = ds.window(record, start)
                primary = b.get_valid_mask() if b.body_type is None else b.get_valid_mask() & ~b.action_mask
                motion = (b.future - b.positions[:, 3:4]).norm(dim=-1)
                score = float(motion[primary[:, None].expand_as(motion)].mean())
                windows.append((score, start))
            selected = sorted(windows, reverse=True)[:batch_size]
            score = sum(w[0] for w in selected) / len(selected)
            if len(selected) == batch_size and (best is None or score > best[0]):
                best = score, record, selected
        if best is None or best[0] < .001:
            raise ValueError(f"No informative moving fixture for {domain}/{split}; best={best}")
        _, record, selected = best
        batch = TrajectoryBatch.collate([ds.window(record, s) for _, s in selected])
        save_fixture(batch, domain + suffix + ("_hold" if split == "val" else ""), out, normalization,
            dict(index_sha256=ds.index["index_sha256"], split=split, record=record, starts=[s for _, s in selected],
                 selection="top-motion windows of best of six deterministic episode candidates; engineering stress fixture"))


def rl_push_fixtures(root, out, normalization, batch_size=2):
    from data.rl_push import UnifiedRLPushDataset
    from data.unified_dataset import RL_PUSH_EXCLUSIONS, RL_PUSH_VAL_ASSETS
    from utils.rl_push_release import verify_rl_push_release
    release_hash = verify_rl_push_release(root)
    for split in ("train", "val"):
        ds = UnifiedRLPushDataset(root, split=split, training=False, context_len=4, future_len=16,
            manifests=[RL_PUSH_EXCLUSIONS, RL_PUSH_VAL_ASSETS], normalization=normalization)
        ds.emit_sidecar = ds.articulated = True
        ds.batch_size = batch_size
        batch = next(iter(ds))
        save_fixture(batch, "rl_push" + ("_hold" if split == "val" else ""), out, normalization,
                     dict(root=str(root), release_sha256=release_hash, split=split,
                          source_shard=getattr(batch, "source_shard", None), selection="first deterministic production batch"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["calibrate", "assemble", "fixtures", "rl-push-fixtures"])
    p.add_argument("--root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--domain")
    p.add_argument("--artifact", default="stats/normalization_rl_push.json")
    p.add_argument("--episodes", type=int, default=128)
    p.add_argument("--particle-rows", choices=["all", "supervised"], default="all",
                   help="supervised = fit the particle scale on the rows the particle loss covers (deformable rows of scenes with rigid bodies)")
    p.add_argument("--teacher")
    p.add_argument("--suffix", default="")
    a = p.parse_args()
    torch.set_num_threads(1)
    if a.mode == "calibrate":
        calibrate(a.root, a.domain, a.out, episodes=a.episodes, particle_rows=a.particle_rows)
    elif a.mode == "assemble":
        reports = [json.loads(p.read_text()) for p in sorted(Path(a.root).glob("*.json"))]
        assemble_artifact(reports, a.out, a.artifact)
    else:
        normalization = resolve_normalization(dict(profile="xy_p99_per_domain", artifact=a.artifact))
        if a.mode == "fixtures":
            indexed_fixtures(a.root, a.domain, a.out, normalization, teacher=a.teacher, suffix=a.suffix)
        else:
            rl_push_fixtures(a.root, a.out, normalization)


if __name__ == "__main__":
    main()
