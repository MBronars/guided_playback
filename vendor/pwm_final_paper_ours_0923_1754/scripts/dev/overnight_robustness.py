"""Bounded, resumable paired observation-error evaluation on pinned real-corpus clips.

Run inside the assigned single visible CUDA device. No training or model ranking.
Results are diagnostics of the named checkpoint and source manifest.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import time
import traceback
from pathlib import Path

import torch

from scripts.dev.observation_probe_common import (
    Predictor, checkpoint_batch, corrupt_observation, geometry_residual_mm,
    observed_input, save_preview, score, single_clip, write_json, write_frozen_run,
)
from utils.reproducibility import code_revision, config_for_json, evaluation_seed, file_sha256


DOMAINS = ("mpm", "pickup", "pushing", "cloth")


def families(domain):
    result = [("history_jitter", (1, 5, 10)), ("history_drift", (1, 5, 10))]
    if domain != "mpm":
        result += [("pose_translation", (1, 5, 10)), ("pose_rotation", (0.5, 2, 5))]
    if domain == "mpm":
        result += [("stiffness", (0.5, 2)), ("density", (0.8, 1.2))]
    return result


def verified_fixture(entry):
    """A manifest fixture, loaded only while its file still matches the recorded sha256: a tensor
    replaced at the same path must not reuse or extend an existing receipt."""
    if file_sha256(entry["fixture"]) != entry["fixture_sha256"]:
        raise ValueError(f"Fixture {entry['fixture']} changed since its manifest recorded sha256 "
                         f"{entry['fixture_sha256']}; rebuild the manifest and use a fresh output directory")
    return torch.load(entry["fixture"], map_location="cpu", weights_only=False)


def fixtures(root, cfg, cloth_manifest=None):
    provenance = json.loads((root / "geometry_results.json").read_text())
    receipts = {x["receipt"]["name"]: x["receipt"] for x in provenance["windows"]}
    batches, manifest = {}, []
    for domain in DOMAINS:
        for index in range(4):
            name = f"{domain}_{index}"
            path = root / (name + ".pt")
            b = checkpoint_batch(torch.load(path, map_location="cpu", weights_only=False), cfg, domain)
            digest = file_sha256(path)
            for row in range(b.batch_size):
                key = f"{name}_row{row}"
                batches[key] = single_clip(b, row)
                manifest.append({"key": key, "domain": domain, "fixture": str(path.resolve()),
                                 "fixture_sha256": digest, "fixture_row": row,
                                 "particles": b.num_particles, "source": receipts[name]})
    if cloth_manifest:
        entries = json.loads(Path(cloth_manifest).read_text())["clips"]
        manifest = [x for x in manifest if x["domain"] != "cloth"]
        for entry in entries:
            batches[entry["key"]] = checkpoint_batch(verified_fixture(entry), cfg, "cloth")
            manifest.append(entry)
    return batches, manifest


def check_contracts(batches):
    checks = []
    for key in ("mpm_0_row0", "pickup_0_row0", "pushing_0_row0", "cloth_0_row0"):
        clean = batches[key]
        snapshot = clean.positions.clone()
        pose_snapshot = None if clean.body_poses is None else clean.body_poses.clone()
        baseline_residual = geometry_residual_mm(clean)
        for family, levels in families(clean.domain):
            zero = corrupt_observation(clean, family, 1 if family in ("stiffness", "density") else 0, 91)
            base = observed_input(clean)
            for field in dataclasses.fields(clean):
                z, b = getattr(zero, field.name), getattr(base, field.name)
                if torch.is_tensor(z):
                    assert torch.equal(z, b), (key, family, field.name, "zero corruption")
            for level in levels:
                perturbed = corrupt_observation(clean, family, level, 91)
                residual = geometry_residual_mm(perturbed)
                assert abs(residual - baseline_residual) < 0.002, (key, family, residual, baseline_residual)
                assert torch.equal(perturbed.actions, clean.actions)
                if clean.pose_actions is not None:
                    assert torch.equal(perturbed.pose_actions, clean.pose_actions)
                # Future target channels are erased; references and commands survive unchanged.
                assert torch.equal(perturbed.future, perturbed.anchor[:, None].expand_as(perturbed.future))
                if perturbed.pose_deltas_gt is not None:
                    assert torch.count_nonzero(perturbed.pose_deltas_gt) == 0
                if clean.domain == "mpm" and family.startswith("history"):
                    d = perturbed.anchor - clean.anchor
                    measured = float(d.square().sum(-1).mean().sqrt()) * 1000
                    assert abs(measured - level) < 0.002, (key, family, measured, level)
                checks.append({"clip": key, "family": family, "severity": level,
                               "geometry_residual_mm": residual})
        assert torch.equal(clean.positions, snapshot), "reference modified"
        if pose_snapshot is not None:
            assert torch.equal(clean.body_poses, pose_snapshot), "reference poses modified"
    return {"passed": True, "conditions_checked": len(checks), "checks": checks}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fixtures", default="validation/architecture_review_2026-09-12/deep_dive")
    ap.add_argument("--hours", type=float, default=6)
    ap.add_argument("--cloth-manifest")
    ap.add_argument("--pilot-only", action="store_true")
    ap.add_argument("--cpu-check", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(8)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    begun = time.time()
    if a.cpu_check:
        from utils.checkpoint import checkpoint_config, read_checkpoint
        cfg = checkpoint_config(read_checkpoint(a.checkpoint))
        batches, _ = fixtures(Path(a.fixtures), cfg, a.cloth_manifest)
        result = check_contracts(batches)
        write_json(out / "cpu_contracts.json", result)
        print(json.dumps({"passed": True, "conditions": result["conditions_checked"]}), flush=True)
        return
    predictor = Predictor(a.checkpoint)
    batches, manifest = fixtures(Path(a.fixtures), predictor.cfg, a.cloth_manifest)
    receipt = {"start_unix": begun, "checkpoint": str(Path(a.checkpoint).resolve()),
               "checkpoint_sha256": file_sha256(a.checkpoint), "code": code_revision(),
               "config": config_for_json(predictor.cfg), "precision": predictor.precision,
               "gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__),
               "protocol": "C4/F16 fixed window; paired input sensitivity; full valid particle metrics",
               "evidence": "Diagnostic validation clips; historical checkpoint training exclusions unaudited",
               "action_assumption": "Known world-frame action deltas; shapes, object IDs and materials known except the tested family",
               "sampling_seeds": [710001, 710002], "corruption_seeds": [9201, 9202],
               "hours_limit": a.hours, "fixtures": manifest}
    write_frozen_run(out / "run.json", receipt,
                     ('fixtures', 'precision', 'protocol', 'sampling_seeds', 'corruption_seeds'))
    write_json(out / "contracts.json", check_contracts(batches))
    pilot = {}
    for key in ("pushing_0_row0", "pickup_0_row0", "mpm_3_row0", "cloth_3_row0"):
        pred, timing = predictor(observed_input(batches[key]), 710001)
        pilot[batches[key].domain] = {"key": key, **timing,
                                    "metrics": score(batches[key], observed_input(batches[key]), pred)}
        print("PILOT", key, json.dumps(pilot[batches[key].domain]), flush=True)
        write_json(out / "pilot.json", pilot)
    # Integration test: erasing future geometry/pose targets cannot alter inference.
    reference = batches["pushing_0_row0"]
    raw_pred, _ = predictor(reference, 799991)
    scrubbed_pred, _ = predictor(observed_input(reference), 799991)
    discrepancy = float((raw_pred - scrubbed_pred).abs().max())
    assert discrepancy < 1e-6, ("future-state leakage", discrepancy)
    repeat, _ = predictor(corrupt_observation(reference, "pose_translation", 0, 91), 799991)
    assert torch.equal(scrubbed_pred, repeat), "zero corruption not reproducible"
    write_json(out / "sampler_contracts.json", {"passed": True, "future_erasure_max_m": discrepancy,
                                               "zero_corruption_bitwise_equal": True})
    if a.pilot_only:
        return
    # A conservative worst-observed per-domain timing determines a COMPLETE core sweep.
    # Prefer more clips, then more independent corruption draws if time permits.
    budget = max(0, a.hours * 3600 - (time.time() - begun) - 120)
    chosen = None
    saved_manifest = out / 'manifest.json'
    if saved_manifest.exists():
        frozen = json.loads(saved_manifest.read_text())
        chosen = (frozen['clips'], len(frozen['corruption_seeds']),
                  frozen['estimated_seconds_with_50pct_margin'])
    for max_row in (() if chosen is not None else (4, 3, 2, 1)):
        subset = [x for x in manifest if x["fixture_row"] < max_row]
        for n_corruption in (2, 1):
            cost = sum(pilot[x["domain"]]["seconds"] * 1.5 * 2 *
                       (1 + n_corruption * sum(len(v) for _, v in families(x["domain"]))) for x in subset)
            if cost <= budget:
                chosen = subset, n_corruption, cost
                break
        if chosen:
            break
    if chosen is None:
        raise RuntimeError("Pilot leaves insufficient time for even the 16-clip core sweep")
    subset, n_corruption, estimated = chosen
    cases = []
    # Round robin across domains/fixtures; retain coverage if an external interruption occurs.
    subset.sort(key=lambda x: (x["fixture_row"], x["key"].split('_', 1)[1], DOMAINS.index(x["domain"])))
    for sampler in (710001, 710002):
        for x in subset:
            cases.append((x, "clean", 0, sampler, 9201))
        for corruption in (9201, 9202)[:n_corruption]:
            for family in ("history_jitter", "history_drift", "pose_translation", "pose_rotation", "stiffness", "density"):
                for li in range(3):
                    for x in subset:
                        available = dict(families(x["domain"]))
                        if family in available and li < len(available[family]):
                            cases.append((x, family, available[family][li], sampler, corruption))
    write_json(out / "manifest.json", {"clips": subset, "cases": len(cases),
                                       "estimated_seconds_with_50pct_margin": estimated,
                                       "sampler_seeds": [710001, 710002],
                                       "corruption_seeds": [9201, 9202][:n_corruption]})
    case_dir = out / "cases"; case_dir.mkdir(exist_ok=True)
    previews = out / "previews"; previews.mkdir(exist_ok=True)
    completed = failed = 0
    for x, family, severity, sampler, corruption in cases:
        if time.time() - begun >= a.hours * 3600:
            break
        key = f'{x["key"]}__{family}_{severity:g}__s{sampler}__c{corruption}'
        target = case_dir / (key + ".json")
        if target.exists():
            saved = json.loads(target.read_text())
            completed += int(saved["status"] == "ok"); failed += int(saved["status"] != "ok")
            continue
        clean = batches[x["key"]]
        # Family-specific draw, independent of sampler seed; repeated across severity.
        noise_seed = evaluation_seed(corruption, 0, x["key"] + family)
        result = {"case": key, "clip": x["key"], "domain": x["domain"], "family": family,
                  "severity": severity, "sampler_seed": sampler, "corruption_seed": corruption,
                  "actual_corruption_seed": noise_seed, "status": "running"}
        try:
            observed = corrupt_observation(clean, family, severity, noise_seed)
            pred, timing = predictor(observed, sampler)
            result.update(status="ok", timing=timing, metrics=score(clean, observed, pred))
            if sampler == 710001 and corruption == 9201:
                save_preview(previews / (key + ".npz"), clean, observed, pred)
                result["preview"] = str(Path("previews") / (key + ".npz"))
            completed += 1
        except Exception:
            result.update(status="failed", traceback=traceback.format_exc())
            failed += 1
            torch.cuda.empty_cache()
        write_json(target, result)
        write_json(out / "status.json", {"state": "running", "completed": completed, "failed": failed,
                                         "planned": len(cases), "last_case": key,
                                         "elapsed_seconds": time.time() - begun})
        print("CASE", key, result["status"], result.get("timing", {}).get("seconds"), flush=True)
        if failed >= 5 and completed == 0:
            break
    write_json(out / "status.json", {"state": "sampling_finished", "completed": completed,
                                     "failed": failed, "planned": len(cases),
                                     "all_planned_cases_attempted": completed + failed == len(cases),
                                     "elapsed_seconds": time.time() - begun})


if __name__ == "__main__":
    main()
