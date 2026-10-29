"""Bounded passive-MPM candidate: prepare, inspect, collect, export, verify.

No training-promotion command. Uses the extracted numerical backend and explicit
versioned scene recipes. Run `python -m collection.mpm --help` from the repository root.
"""
import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

import numpy as np

from collection.common import canonical, check_hashes, claim, digest, fsync_dir, relative_path, write_once
from . import runtime as ref
from .recipes import VERSIONS, INTERACTION_VERSIONS


def now():
    return datetime.now(timezone.utc).isoformat()


def save_npz(path, arrays):
    with Path(path).open("xb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    fsync_dir(Path(path).parent)


def prepare(root, recipe_version='legacy-v1', *, seed_start=2026090900, seed_count=2,
            bank=None, pool=None, bank_role='train', split_manifest=None,
            dataset_split='train', evaluation='train', cycles=1, cycle_start=0,
            specification_factory=None):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Preparation must explicitly hide CUDA")
    from . import clock as coverage, contracts
    from .recipes import load_bank, scene, specifications
    fresh = recipe_version == 'fresh-shapes-v1'
    if specification_factory is not None and not fresh:
        raise ValueError('Configured specifications require the fresh-shape contract')
    coverage_specs = None if fresh else specifications(recipe_version, seed_start=seed_start, seed_count=seed_count)
    if (bank is None) != (pool is None):
        raise ValueError('An explicit object bank requires explicit training/holdout pool files too')
    bank = Path(bank).resolve(strict=True) if bank is not None else ref.reference_root() / 'collection_variability/run-02/bank'
    pool = Path(pool).resolve(strict=True) if pool is not None else ref.reference_root() / 'source-snapshot/source/configs'
    metadata, objects = load_bank(bank, pool, role=bank_role, split_manifest=split_manifest)
    if fresh:
        from .protocol import specifications as fresh_specs, coverage as shape_coverage
        coverage_specs = fresh_specs(metadata, split=dataset_split, evaluation=evaluation,
                                     cycles=cycles, cycle_start=cycle_start)
        if specification_factory is not None:
            coverage_specs = specification_factory(metadata)
            if (not coverage_specs or any(s['split']!=dataset_split or s['evaluation']!=evaluation for s in coverage_specs)
                    or len({s['seed_namespace'] for s in coverage_specs})!=len(coverage_specs)):
                raise ValueError('Configured specifications must match the split and have unique sequence keys')
    bank_sources = {str(bank / 'MANIFEST.json'): digest(bank / 'MANIFEST.json')}
    bank_sources.update({str(relative_path(bank, r['file'])): r['sha256'] for r in metadata['objects']})
    bank_sources.update({str(pool / name): digest(pool / name) for name in
                        ('objaverse_uid_train.json', 'objaverse_uid_holdout.json')})
    if split_manifest is not None:
        bank_sources[str(Path(split_manifest).resolve(strict=True))] = digest(split_manifest)
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    source_hashes = ref.sources()
    # Full text of the executed local closure, not a misleading clean HEAD.
    for name in source_hashes:
        write_once(root / "source" / (name + ".json"),
                   dict(path=name, sha256=source_hashes[name], text=(ref.ROOT / name).read_text()))
    cases = []

    def add(label, initial, m, provenance, kind):
        folder = root / "cases" / label
        folder.mkdir(parents=True)
        save_npz(folder / "initial.npz", initial)
        payload = dict(label=label, kind=kind, initial_sha256=digest(folder / "initial.npz"),
                       template=m, provenance=provenance)
        case = dict(release_id="wall-free-" + recipe_version + "-journal-v1", case_index=len(cases),
                    seed_namespace=str(provenance.get("seed_namespace", provenance.get("seed", label))), asset_ids=m["case"]["asset_ids"],
                    config_sha256=hashlib.sha256(canonical(dict(
                        initial_sha256=payload["initial_sha256"], solver=m["solver_config"], clock=m["clock"],
                        scenario=m.get("scenario"), kind=kind, sources=source_hashes)).encode()).hexdigest(),
                    task="passive")
        manifest = dict(m, case=case, episode_id=contracts.identity(case)[0],
                        source_sha256={str(ref.ROOT / p): h for p, h in source_hashes.items()},
                        accepted_for_training=False)
        write_once(folder / "MANIFEST.json", manifest)
        payload["manifest_sha256"] = digest(folder / "MANIFEST.json")
        cases.append(payload)

    # First two runs test the exact previously reviewed positive/negative inputs.
    old_plan = json.loads((ref.reference_root() / "wall_filter/run-02/PLAN.json").read_text())
    for old in old_plan["cases"]:
        if old["provenance"]["kind"] != "matched_control":
            continue
        folder = ref.reference_root() / "wall_filter/run-02" / old["label"]
        if digest(folder / "initial.npz") != old["initial_sha256"]:
            raise ValueError("Historical reference initial changed")
        with np.load(folder / "initial.npz", allow_pickle=False) as data:
            initial = {k: data[k].copy() for k in data.files}
        old_result = json.loads((folder / "RESULT.json").read_text())
        provenance = dict(old["provenance"], reference_raw=old_result["raw_path"],
                          reference_raw_sha256=old_result["raw_sha256"])
        add(old["label"], initial, old["template_manifest"], provenance, "matched_control")
    if len(cases) != 2:
        raise ValueError("Exactly two pinned functionality controls required")
    for spec in coverage_specs:
        initial, geometry = scene(spec, metadata, objects)
        clock, wave = coverage.choose_clock(initial)
        config = dict(n_grid=100, grid_lim_m=100/64, floor_m=.1, floor_contact=True, friction=.3,
                      floor_tolerance_m=.002, observation_emergency_depth_m=.06,
                      gravity_m_s2=[0., 0., -9.8], dt_s=clock.substep_dt_s,
                      precision="float64-master", softening=.1, plasticine_policy="archived-fcr-log-return")
        m = dict(task="passive", n_objects=spec["count"], clock=asdict(clock),
                 scenario=spec["scenario"], floor_m=.1, grid_lim_m=100/64, n_grid=100,
                 volume_policy=metadata["volume_policy"], solver_config=config,
                 units=dict(position="m", velocity="m/s", time="s", volume="m^3", density="kg/m^3", E="Pa"),
                 population=f"predeclared-{len(coverage_specs)}-case-{bank_role}-asset-diagnostic", split="diagnostic",
                 object_cohort=bank_role, split_manifest_sha256=digest(split_manifest) if split_manifest is not None else None,
                 material_names=[("elastic", "plasticine", "sand")[v] for v in initial["mat_type"]],
                 case=dict(asset_ids=geometry["asset_ids"]), geometry_provenance=geometry)
        if fresh:
            from .collapse import CollapsePolicy
            m.update(split=dataset_split, evaluation=evaluation, collapse_policy=asdict(CollapsePolicy()),
                     population='fresh-shapes-v1-' + evaluation)
        add(f"case-{spec['index']:02d}-{spec['count']}obj-{spec['scenario']}", initial, m,
            dict(spec, **geometry, wave_budget=wave, bank_manifest_sha256=bank_sources[str(bank / 'MANIFEST.json')]), "coverage")
    versions = {name: importlib.metadata.version(name) for name in ("numpy", "warp-lang", "h5py")}
    if fresh:
        write_once(root / 'COVERAGE.json', shape_coverage(coverage_specs, metadata))
        write_once(root / 'PROTOCOL.json', dict(version=recipe_version, split=dataset_split,
                   evaluation=evaluation, object_cohort=bank_role, cycles=cycles, cycle_start=cycle_start,
                   maximum_full_extent_m=[.12, .20], orientation='yaw only',
                   volume_policy=metadata['volume_policy'], proxy_volume_qualified=False,
                   material_types=['elastic', 'plasticine', 'sand'], E_Pa=[1e4, 1e5, 1e6],
                   nu=.2, density_kg_m3=1000, plasticine_yield_Pa=10000,
                   collapse_policy=asdict(CollapsePolicy()),
                   floor_observation_tolerance_m=.002, floor_export_tolerance_m=.012,
                   floor_emergency_depth_m=.06, timestep='material-dependent fixed before scene',
                   saved_states=49, saved_interval_s=.0417, duration_s=2.0016,
                   active_drag=False, accepted_for_training=False))
    ref.verify_sources(source_hashes)
    if any(digest(p) != sha for p, sha in bank_sources.items()):
        raise ValueError('Bank or split manifest changed during preparation')
    write_once(root / "PLAN.json", dict(schema="bounded-wall-free-pilot-v1", recipe_version=recipe_version, created_utc=now(),
               seed_start=seed_start, seed_count=seed_count, bank_source_sha256=bank_sources,
               object_cohort=bank_role, split_manifest_sha256=digest(split_manifest) if split_manifest is not None else None,
               cases=cases, source_sha256=source_hashes, runtime_versions=versions,
               planned=len(cases), training_approval=False, backend="extracted-passive-mpm-diagnostic",
               material_note="Reviewed elastic candidate + archived plasticine law/softening + FP64 master; no new material policy",
               scenario_note="New explicit geometry/velocity recipes, not historical sampler yield; no outcome top-ups"))
    return dict(prepared=len(cases), placement_rejected=sum(bool(c["provenance"].get("placement_reasons")) for c in cases),
                source_files=len(source_hashes), out=str(root))


def inspect(root):
    root = Path(root).resolve()
    plan = json.loads((root / "PLAN.json").read_text())
    ref.verify_sources(plan["source_sha256"])
    for path, expected in plan.get('bank_source_sha256', {}).items():
        if digest(path) != expected:
            raise ValueError('Prepared bank or object split changed: ' + path)
    seen = set()
    for case in plan["cases"]:
        folder = relative_path(root, "cases/" + case["label"])
        if digest(folder / "initial.npz") != case["initial_sha256"] or digest(folder / "MANIFEST.json") != case["manifest_sha256"]:
            raise ValueError("Prepared case changed: " + case["label"])
        m = json.loads((folder / "MANIFEST.json").read_text())
        if m["episode_id"] in seen:
            raise ValueError("Duplicate episode")
        seen.add(m["episode_id"])
    if len(seen) != plan["planned"]:
        raise ValueError("Incomplete planned cases")
    return plan


def gpu_status(uuid, *, allow_own=False, memory_limit_mib=4096):
    inventory = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,memory.used,utilization.gpu",
                                         "--format=csv,noheader,nounits"], text=True, timeout=10)
    rows = [line.strip().split(",") for line in inventory.splitlines() if line.strip()]
    matches = [row for row in rows if row[0].strip() == uuid]
    if len(matches) != 1:
        raise RuntimeError("Selected GPU missing/ambiguous")
    output = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
                                      "--format=csv,noheader,nounits"], text=True, timeout=10)
    processes = [r.strip().split(",") for r in output.splitlines() if r.strip()]
    active = [r for r in processes if r[0].strip() == uuid]
    if any(not allow_own or r[1].strip() != str(os.getpid()) for r in active):
        raise RuntimeError("Foreign process on selected GPU; owned worker stops")
    used = int(matches[0][1].strip())
    # Driver bookkeeping can occupy a few MiB with no compute process. Foreign
    # processes are rejected above regardless of their memory or utilization.
    if used > memory_limit_mib or (not allow_own and (used > 32 or int(matches[0][2].strip()))):
        raise RuntimeError("GPU idle/memory guard failed")
    return dict(utc=now(), uuid=uuid, device_memory_mib=used,
                process_ids=[int(r[1].strip()) for r in active])


class Recording:
    def __init__(self, observed, deadline, uuid, telemetry, journal=None):
        self.observed, self.deadline, self.uuid, self.telemetry = observed, deadline, uuid, telemetry
        self.last_check = 0.
        self.journal = journal

    def check(self):
        if time.monotonic() > self.deadline:
            raise TimeoutError("case_or_worker_wall_budget")
        if self.uuid and time.monotonic() - self.last_check >= 2:
            self.telemetry.append(gpu_status(self.uuid, allow_own=True))
            self.last_check = time.monotonic()

    def step(self, *args):
        self.check()
        self.observed.step(*args)

    def read(self):
        self.check()
        value = self.observed.read()
        if self.journal is not None:
            self.journal.append(self.observed.steps, dict(value, **self.observed.wall_snapshot()))
        return value


def verify_attempt(folder):
    result = json.loads((folder / "RESULT.json").read_text())
    if result.get("raw_sha256") and digest(folder / "raw.h5") != result["raw_sha256"]:
        raise ValueError("Existing attempt raw changed")
    if digest(folder.parent.parent / "initial.npz") != result["initial_sha256"]:
        raise ValueError("Attempt input changed")
    if 'journal' in result:
        from .journal import verify_journal
        observed = verify_journal(folder / 'frames', initial_sha256=result['initial_sha256'],
            manifest_sha256=digest(folder.parent.parent / 'MANIFEST.json'), complete=result['status'] == 'complete',
            raw_path=folder / 'raw.h5' if result['status'] == 'complete' else None)
        if observed != result['journal']:
            raise ValueError('Journal receipt changed')
    return result


def collect(args):
    from collection.lease import gpu_lease
    with gpu_lease(args.uuid):
        return _collect(args)


def _collect(args):
    root = args.root.resolve()
    plan = inspect(root)
    if not 0 <= args.start < len(plan["cases"]) or args.limit < 1 or args.attempt < 0:
        raise ValueError("Invalid bounded range/attempt")
    if not 0 < args.wall_seconds <= 7200 or not 0 < args.case_seconds <= 1200:
        raise ValueError("Explicit bounded wall times required")
    if not args.uuid or os.environ.get("CUDA_VISIBLE_DEVICES") != args.uuid:
        raise ValueError("Exactly the selected full GPU UUID must be visible")
    if not args.execution.replace("-", "").isalnum():
        raise ValueError("Simple execution identifier required")
    if args.start + args.limit > len(plan["cases"]):
        raise ValueError("Requested range exceeds plan")
    reuse_own_context = getattr(args, 'reuse_own_context', False)
    gpu_status(args.uuid, allow_own=reuse_own_context)
    # Two independent preflight readings; short sleep is inside bounded worker.
    time.sleep(2)
    gpu_status(args.uuid, allow_own=reuse_own_context)
    api, _, _, _, _, _ = ref.load()
    import warp as wp
    ref.verify_sources(plan["source_sha256"])
    wp.config.kernel_cache_dir = str(root / "kernel-cache")
    wp.init()
    if wp.get_cuda_device_count() != 1 or wp.get_device("cuda:0").uuid != args.uuid:
        raise RuntimeError("Warp device visibility differs from selected GPU")
    start = time.monotonic()
    deadline = start + args.wall_seconds
    rows = []
    def stop(signum, frame):
        raise TimeoutError("owned_worker_signal_" + str(signum))
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGALRM):
        signal.signal(sig, stop)
    execution = dict(pid=os.getpid(), started_utc=now(), uuid=args.uuid, source_sha256=plan["source_sha256"],
                     start=args.start, limit=args.limit, attempt=args.attempt, wall_seconds=args.wall_seconds,
                     case_seconds=args.case_seconds, accepted_for_training=False)
    with claim(root / ("EXECUTION-" + args.execution + ".json"), execution):
        signal.setitimer(signal.ITIMER_REAL, args.wall_seconds)
        try:
            for case in plan["cases"][args.start:args.start + args.limit]:
                if time.monotonic() >= deadline - 10:
                    break
                folder = root / "cases" / case["label"]
                attempt = folder / "attempts" / f"attempt-{args.attempt:05d}"
                if (attempt / "RESULT.json").is_file():
                    rows.append(dict(verify_attempt(attempt), resumed=True))
                    continue
                if attempt.exists():
                    raise RuntimeError("Interrupted attempt retained; choose a new explicit attempt index")
                if sum(p.stat().st_size for p in root.rglob('*') if p.suffix in ('.h5', '.npz')) > 8 * 1024**3:
                    raise RuntimeError("Bounded raw-data disk budget exceeded")
                attempt.mkdir(parents=True)
                write_once(attempt / "CLAIM.json", dict(execution, label=case["label"], initial_sha256=case["initial_sha256"]))
                result = dict(label=case["label"], initial_sha256=case["initial_sha256"],
                              kind=case["kind"], accepted_for_training=False)
                begin = time.monotonic()
                observed = None
                recording = None
                telemetry = []
                try:
                    reasons = case["provenance"].get("placement_reasons", [])
                    if reasons:
                        result.update(status="placement_rejected", reasons=reasons)
                    else:
                        ref.verify_sources(plan["source_sha256"])
                        if digest(folder / "initial.npz") != case["initial_sha256"]:
                            raise ValueError("Initial changed before collection")
                        with np.load(folder / "initial.npz", allow_pickle=False) as arrays:
                            initial = {k: arrays[k].copy() for k in arrays.files}
                        m = json.loads((folder / "MANIFEST.json").read_text())
                        observed = ref.backend(initial, m, "cuda:0")
                        for key in ("x", "v"):
                            np.testing.assert_array_equal(observed.read()[key], initial[key])
                        from .journal import FrameJournal
                        journal = FrameJournal(attempt / 'frames', initial_sha256=case['initial_sha256'],
                            manifest_sha256=case['manifest_sha256'], clock=observed.clock)
                        recording = Recording(observed, min(deadline, begin + args.case_seconds), args.uuid, telemetry, journal)
                        api.collect_passive(recording, observed, m, {k: initial[k] for k in ref.STATIC}, attempt / "raw.h5")
                        quality = api.validate_wall_free(attempt / "raw.h5")
                        ref.verify_sources(plan["source_sha256"])
                        result.update(status="complete", quality=quality, raw_sha256=digest(attempt / "raw.h5"),
                                      raw_path=str(attempt / "raw.h5"))
                except Exception as exc:
                    result.update(status="failed", reasons=[type(exc).__name__ + ":" + str(exc)])
                finally:
                    result.update(wall_seconds=time.monotonic() - begin, telemetry=telemetry)
                    if (attempt / 'frames/HEADER.json').is_file():
                        from .journal import verify_journal
                        try:
                            result['journal'] = verify_journal(attempt / 'frames', initial_sha256=case['initial_sha256'],
                                manifest_sha256=case['manifest_sha256'], complete=result['status'] == 'complete')
                        except Exception as error:
                            result.update(status='failed', reasons=result.get('reasons', []) + ['journal:' + str(error)])
                    if observed is not None:
                        # Preserve a failed finite/nonfinite state separately; never export it.
                        if result["status"] != "complete":
                            state = observed.solver.mpm_state
                            save_npz(attempt / "failed-state.npz", {k: getattr(state, "particle_" + k).numpy()
                                     for k in ("x", "v", "F_trial", "C")})
                        result["observed_steps"] = observed.steps
                    write_once(attempt / "RESULT.json", result)
                    rows.append(result)
                    print(canonical({k: result[k] for k in ("label", "status", "wall_seconds")}), flush=True)
                    observed = None
                    recording = None
                    gc.collect()
                if result["status"] == "failed" and any("Foreign process" in r or "wall_budget" in r or "signal_" in r or "GPU idle/memory" in r for r in result["reasons"]):
                    break
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            write_once(root / ("RESULT-" + args.execution + ".json"),
                       dict(execution=execution, rows=rows, finished=len(rows), requested=args.limit,
                            status_counts=dict(Counter(r["status"] for r in rows)),
                            wall_seconds=time.monotonic() - start, accepted_for_training=False))
    return dict(finished=len(rows), requested=args.limit)


def export(root, output, attempt_index=0):
    plan = inspect(root)
    from . import export as api
    rows, raws = [], []
    for case in plan["cases"]:
        folder = Path(root) / "cases" / case["label"] / "attempts" / f"attempt-{attempt_index:05d}"
        if not (folder / "RESULT.json").exists():
            rows.append(dict(label=case["label"], status="incomplete" if folder.exists() else "unrun", kind=case["kind"]))
            continue
        result = verify_attempt(folder)
        if (plan.get('recipe_version') in INTERACTION_VERSIONS and case['kind'] == 'coverage'
                and result['status'] == 'complete'):
            from .interactions import analyze
            from .validate import read_raw
            manifest, arrays = read_raw(folder / 'raw.h5', api.POLICY)
            result = dict(result, interaction=analyze(manifest, arrays))
        rows.append(result)
        if (result["status"] == "complete" and case["kind"] == "coverage"
                and result.get('interaction', {}).get('semantic_eligible', True)):
            raws.append(folder / "raw.h5")
    # Controls never enter the prospective training-data export.
    output = Path(output)
    request = dict(plan_sha256=digest(Path(root) / "PLAN.json"), attempt=attempt_index,
                   outcomes=[dict(label=r['label'], status=r['status'], raw_sha256=r.get('raw_sha256')) for r in rows])
    request_hash = hashlib.sha256(canonical(request).encode()).hexdigest()
    if output.is_symlink():
        raise ValueError("Export destination is a symlink")
    if output.exists():
        complete = json.loads((output / "COMPLETE.json").read_text())
        if complete['request_sha256'] != request_hash or complete['accepted_for_training'] is not False:
            raise ValueError("Existing export request changed")
        check_hashes(output, complete['artifacts'])
        return dict(complete, resumed=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with claim(output.with_name(output.name + ".request.json"), request):
        stage = Path(tempfile.mkdtemp(prefix='.' + output.name + '.partial-', dir=output.parent))
        if raws:
            published = api.export_wall_free(raws, stage / 'filtered')
        else:
            published = dict(retained=0)
        write_once(stage / "ATTEMPTS.json", dict(cases=rows, status_counts=dict(Counter(r["status"] for r in rows)),
                                                request=request, accepted_for_training=False))
        artifacts = {str(p.relative_to(stage)): digest(p) for p in sorted(stage.rglob('*')) if p.is_file()}
        complete = dict(schema='bounded-mpm-export-v1', request_sha256=request_hash, artifacts=artifacts,
                        retained=published['retained'], accepted_for_training=False)
        write_once(stage / 'COMPLETE.json', complete)
        fsync_dir(stage)
        if output.exists():
            raise FileExistsError(output)
        os.rename(stage, output)
        fsync_dir(output.parent)
    return complete


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("prepare", "inspect", "collect", "export"):
        s = sub.add_parser(name)
        s.add_argument("--root", type=Path, required=True)
        if name == 'prepare':
            s.add_argument('--recipe-version', choices=VERSIONS, default='legacy-v1')
            s.add_argument('--seed-start', type=int, default=2026090900)
            s.add_argument('--seed-count', type=int, default=2)
            s.add_argument('--bank', type=Path, help='Explicit compatible source-point/weight bank with MANIFEST.json')
            s.add_argument('--pool', type=Path, help='Explicit directory containing pinned train/holdout asset JSONs')
            s.add_argument('--bank-role', choices=('train', 'validation', 'test'), default='train')
            s.add_argument('--split-manifest', type=Path, help='Frozen object-level cohort assignment; mandatory for evaluation banks')
            s.add_argument('--dataset-split', choices=('train', 'validation', 'test'), default='train')
            s.add_argument('--evaluation', choices=('train', 'seen', 'unseen', 'count_seen', 'count_unseen'), default='train')
            s.add_argument('--cycles', type=int, default=1)
            s.add_argument('--cycle-start', type=int, default=0)
        if name == "collect":
            s.add_argument("--uuid", required=True)
            s.add_argument("--execution", required=True)
            s.add_argument("--start", type=int, default=0)
            s.add_argument("--limit", type=int, required=True)
            s.add_argument("--attempt", type=int, default=0)
            s.add_argument("--wall-seconds", type=int, default=7200)
            s.add_argument("--case-seconds", type=int, default=600)
        if name == "export":
            s.add_argument("--out", type=Path, required=True)
            s.add_argument("--attempt", type=int, default=0)
    a = p.parse_args()
    if a.command == "prepare": result = prepare(a.root, a.recipe_version, seed_start=a.seed_start,
        seed_count=a.seed_count, bank=a.bank, pool=a.pool, bank_role=a.bank_role, split_manifest=a.split_manifest,
        dataset_split=a.dataset_split, evaluation=a.evaluation, cycles=a.cycles, cycle_start=a.cycle_start)
    elif a.command == "inspect": result = dict(planned=inspect(a.root)["planned"], source_and_inputs_verified=True)
    elif a.command == "collect": result = collect(a)
    else: result = export(a.root, a.out, a.attempt)
    print(canonical(result), flush=True)


if __name__ == "__main__":
    main()
