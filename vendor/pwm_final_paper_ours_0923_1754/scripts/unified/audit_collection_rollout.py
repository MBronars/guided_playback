"""Check real-recording rollout plumbing with a labelled ground-truth oracle.

This is a geometry/indexing check, not learned rollout performance. The sampler
is replaced only inside this process; the production windowing, pose integration,
actuator handling and committed context remain unchanged.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from data.indexed_episodes import IndexedEpisodeDataset
from data.normalization import resolve_normalization
from inference import windowed_rollout
from scripts.unified.verify_collections import write_json
from utils.reproducibility import file_sha256, working_source_versions


def audit_fixture_record(fixture, indices, normalization):
    receipt = json.loads(Path(fixture).read_text())
    domain, source = receipt["domain"], receipt["source"]
    ds = IndexedEpisodeDataset(Path(indices) / domain / source["split"], domain=domain,
                              context_len=4, future_len=-1, training=False,
                              normalization=normalization)
    ds.articulated = True
    record = next(r for r in ds.records if r["episode_id"] == source["record"]["episode_id"])
    batch = ds.window(record, 0)
    calls, offset = [], batch.context_len

    def oracle(model, win, diffusion, **kwargs):
        nonlocal offset
        length, C = win.num_future_frames, win.context_len
        torch.testing.assert_close(win.future, batch.positions[:, offset:offset + length], rtol=0, atol=0)
        torch.testing.assert_close(win.actions, batch.actions[:, offset-C:offset-C+length], rtol=0, atol=0)
        if win.action_active is not None:
            gates = torch.cat([batch.context_action_active, batch.action_active], 1)
            torch.testing.assert_close(win.context_action_active, gates[:, offset-C:offset-1])
            torch.testing.assert_close(win.action_active, gates[:, offset-1:offset+length-1])
        calls.append(dict(start=offset, length=length,
            context_max_error_mm=float((win.context - batch.positions[:, offset-C:offset]).norm(dim=-1).max() * 1000)))
        offset += length
        particle = win.prediction_stats.normalize(win.prediction_x0, clip=False)
        if win.body_type is None:
            return particle
        ridx = (win.body_type[0] == 1).nonzero(as_tuple=True)[0]
        pose = win.pose_delta_stats.normalize(win.pose_deltas_gt[:, :, ridx], clip=False)
        return particle, pose

    model = SimpleNamespace(cfg=SimpleNamespace(pose_native=True, predict_body_poses=False))
    with patch("inference.run_diffusion_inference_df", side_effect=oracle):
        result = windowed_rollout(model, None, batch, horizon=batch.num_future_frames,
                                  future_len=16, clamp_actuated=True)
    error = float((result["positions"] - batch.positions).norm(dim=-1).max() * 1000)
    if error > .02:
        raise ValueError(f"{domain}: oracle rollout reconstruction exceeds 0.02 mm: {error}")
    if len(calls) < 2:
        raise ValueError("Choose a multi-window recording for the rollout audit")
    phases = None
    if batch.action_active is not None:
        active = batch.action_active.any(dim=-1)
        moving = batch.actions.ne(0).any(dim=(-1, -2))
        phases = dict(commanded=int(moving.sum()), active_zero_command=int((active & ~moving).sum()),
                      released=int((~active).sum()))
    return dict(domain=domain, fixture=str(fixture), fixture_receipt_sha256=file_sha256(fixture),
                episode_id=record["episode_id"], index_sha256=ds.index["index_sha256"],
                frames=batch.num_frames, particles=batch.num_particles,
                max_error_mm=error, windows=calls, command_phases=phases)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--indices", required=True)
    p.add_argument("--normalization", required=True)
    p.add_argument("--fixtures", nargs="+", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    torch.set_num_threads(1)
    normalization = resolve_normalization(dict(profile="xy_p99_per_domain", artifact=a.normalization))
    sources = working_source_versions()
    rows = []
    for fixture in a.fixtures:
        row = audit_fixture_record(fixture, a.indices, normalization)
        rows.append(row)
        print(f"{row['domain']}: {row['frames']} states, {len(row['windows'])} windows, "
              f"{row['max_error_mm']:.6f} mm maximum oracle error", flush=True)
    if sources != working_source_versions():
        raise RuntimeError("Source changed during rollout audit")
    write_json(a.out, dict(protocol="recorded-target oracle; not learned performance",
                          normalization_sha256=normalization["sha256"], source_versions=sources, results=rows))


if __name__ == "__main__":
    main()
