"""Fail-closed MPM raw validator and atomic diagnostic exporter.

This first milestone deliberately has NO production-release promotion command.
Integrity-valid data remains pending physics approval. New diagnostic exports
preserve the existing prebatched numeric consumer fields plus a JSON row receipt.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import pickle
import tarfile
import tempfile

import h5py
import numpy as np

from .contracts import Clock, SettlingMonitor, SettlingPolicy, canonical_json, identity
from collection.common import digest as sha256


SCHEMA = "mpm-candidate-raw-v1"
PHYSICS_BLOCKERS = ["volume_policy_unvalidated", "elastic_constitutive_review_pending",
                    "timestep_convergence_unresolved", "representative_gpu_pilot_pending",
                    "quality_threshold_calibration_pending"]


@dataclass(frozen=True)
class Policy:
    floor_tolerance_m: float = 0.002
    max_array_bytes: int = 128 << 20
    name: str = "diagnostic-integrity-v1-not-physics-approval"

    def __post_init__(self):
        if not np.isfinite(self.floor_tolerance_m) or self.floor_tolerance_m < 0:
            raise ValueError("Invalid floor tolerance")
        if not isinstance(self.max_array_bytes, int) or self.max_array_bytes <= 0 or not self.name:
            raise ValueError("Invalid policy")


def _receipt(status, reasons, metrics, path, digest, policy, integrity_pass=False, manifest=None):
    manifest = manifest if isinstance(manifest, dict) else {}
    return {"schema": "mpm-validation-v1", "status": status, "reasons": sorted(set(reasons)),
            "metrics": metrics, "raw_sha256": digest, "raw_path": str(path),
            "validator_sha256": sha256(__file__), "contracts_sha256": sha256(Path(__file__).with_name("contracts.py")),
            "policy": asdict(policy), "policy_sha256": hashlib.sha256(canonical_json(asdict(policy)).encode()).hexdigest(),
            "integrity_pass": integrity_pass, "accepted_for_training": False,
            "episode_id": (manifest or {}).get("episode_id"),
            "strata": {key: (manifest or {}).get(key) for key in ("task", "population", "n_objects", "material_names")}}


def read_raw(path, policy):
    """Bound each array before materializing it; only new explicit schema allowed."""
    arrays = {}
    total_bytes = 0
    with h5py.File(path, "r") as f:
        if f.attrs.get("schema") != SCHEMA:
            raise ValueError("unsupported_raw_schema")
        if f.attrs.get("complete") != True:
            raise ValueError("raw_incomplete")
        manifest = json.loads(f.attrs["manifest_json"])
        if not isinstance(manifest, dict):
            raise ValueError("manifest_must_be_object")
        for name in f:
            ds = f[name]
            if not isinstance(ds, h5py.Dataset) or ds.dtype.kind not in "biuf":
                raise ValueError("unsupported_dataset:" + name)
            if ds.size * ds.dtype.itemsize > policy.max_array_bytes:
                raise ValueError("array_budget_exceeded:" + name)
            total_bytes += ds.size * ds.dtype.itemsize
            if total_bytes > 4 * policy.max_array_bytes:
                raise ValueError("total_array_budget_exceeded")
            arrays[name] = ds[()]
    return manifest, arrays


def validate_raw(path, policy=None):
    policy = policy or Policy()
    path = Path(path)
    digest, metrics, manifest = None, {}, None
    try:
        if not path.is_file() or path.is_symlink():
            raise ValueError("raw_missing_or_symlink")
        digest = sha256(path)
        manifest, a = read_raw(path, policy)
        if sha256(path) != digest:
            raise ValueError("raw_changed_during_validation")
        required_manifest = {"case", "episode_id", "task", "n_objects", "clock", "floor_m", "grid_lim_m",
                             "n_grid", "volume_policy", "source_sha256", "units", "population", "split"}
        if not required_manifest <= manifest.keys():
            raise ValueError("manifest_missing_required_fields")
        if manifest.get('geometry_provenance', {}).get('recipe_version') in ('fresh-shapes-v1','drag-fresh-shapes-v1'):
            from .collapse import CollapsePolicy
            from .fresh_banks import VOLUME_POLICY
            from .protocol import cohort
            if manifest.get('collapse_policy') != asdict(CollapsePolicy()):
                raise ValueError('Fresh protocol requires pinned collapse checks')
            evaluation = manifest.get('evaluation')
            if (manifest.get('object_cohort') != cohort(manifest['split'], evaluation)
                    or manifest['volume_policy'] != VOLUME_POLICY
                    or manifest['n_objects'] not in ((5, 6, 7) if evaluation.startswith('count_') else (1, 2, 3, 4))):
                raise ValueError('Fresh protocol split/count/volume mismatch')
        if manifest["units"] != {"position": "m", "velocity": "m/s", "time": "s", "volume": "m^3", "density": "kg/m^3", "E": "Pa"}:
            raise ValueError("unsupported_units")
        episode_id, _ = identity(manifest["case"])
        if episode_id != manifest["episode_id"] or manifest["task"] != manifest["case"]["task"]:
            raise ValueError("episode_identity_mismatch")
        clock = Clock(**manifest["clock"])
        t, n = a["x"].shape[:2]
        k = manifest["n_objects"]
        if not isinstance(k, int) or not 1 <= k <= n or len(manifest["case"]["asset_ids"]) != k:
            raise ValueError("object_count_invalid")
        shapes = {"x": (t, n, 3), "v": (t, n, 3), "F_trial": (t, n, 3, 3), "C": (t, n, 3, 3),
                  "state_time_s": (t,), "object_ids": (n,), "particle_volume": (n,),
                  "selected": (n,), "command_velocity": (n, 3), "transition_active": (t - 1,),
                  "active_substeps": (t - 1,), "E": (k,), "nu": (k,), "density": (k,), "mat_type": (k,), "yield_stress": (k,)}
        scheduled = manifest.get('action_schema') in (
            'mpm-drag-command-schedule-v1','mpm-drag-command-sequence-v2','mpm-drag-command-sequence-v3','mpm-drag-command-sequence-v4')
        if manifest.get('action_schema') is not None and not scheduled:
            raise ValueError('Unknown action schema')
        if scheduled:
            if manifest['task']!='drag':raise ValueError('Scheduled commands require drag task')
            from .collapse import CollapsePolicy
            if manifest.get('collapse_policy')!=asdict(CollapsePolicy()):
                raise ValueError('Scheduled drag requires pinned collapse checks')
            shapes['command_velocity']=(t-1,n,3)
        if manifest.get('action_schema')=='mpm-drag-command-sequence-v4' and 'fragmentation_policy' not in manifest:
            raise ValueError('Gentle dragging requires fragmentation screening')
        if 'fragmentation_policy' in manifest:
            from .drag.fragmentation import FragmentationPolicy
            if not scheduled or manifest['fragmentation_policy'] != asdict(FragmentationPolicy()):
                raise ValueError('Fragmentation screening requires scheduled drag and pinned policy')
        for name, shape in shapes.items():
            if name not in a or a[name].shape != shape:
                raise ValueError("array_shape:" + name)
        for name in ("x", "v", "F_trial", "C", "command_velocity"):
            if a[name].dtype != np.dtype("float32"):
                raise ValueError("array_dtype_float32_required:" + name)
        if a["state_time_s"].dtype != np.dtype("float64"):
            raise ValueError("time_dtype_float64_required")
        if t != clock.transitions + 1 or n < 1:
            raise ValueError("recording_length_mismatch")
        if not isinstance(manifest["n_grid"], int) or manifest["n_grid"] < 4:
            raise ValueError("invalid_grid")
        scalar_values = [manifest["floor_m"], manifest["grid_lim_m"]]
        if not np.isfinite(scalar_values).all() or not 0 <= scalar_values[0] < scalar_values[1]:
            raise ValueError("invalid_floor_or_grid")
        if not manifest["volume_policy"] or not manifest["source_sha256"]:
            raise ValueError("source_or_volume_provenance_missing")
        if (not np.issubdtype(a["object_ids"].dtype, np.integer)
                or not np.array_equal(np.unique(a["object_ids"]), np.arange(k))):
            raise ValueError("object_membership_invalid")
        ids = a["object_ids"]
        reasons = ["nonfinite:" + name for name, value in a.items() if not np.isfinite(value).all()]
        if reasons:
            return _receipt("rejected", reasons, metrics, path, digest, policy, manifest=manifest)
        if not np.array_equal(a["state_time_s"], clock.state_time_s):
            reasons.append("state_time_contract_mismatch")
        for name in ("selected", "transition_active"):
            if not np.isin(a[name], [0, 1]).all():
                reasons.append("nonbinary:" + name)
        if not np.issubdtype(a["active_substeps"].dtype, np.integer):
            reasons.append("active_substeps_not_integer")
        if not np.array_equal(a["active_substeps"], a["transition_active"] * clock.substeps_per_transition):
            reasons.append("action_substep_count_mismatch")
        selected = a["selected"] != 0
        if np.any(a["command_velocity"][...,~selected,:] != 0):
            reasons.append("command_outside_selected")
        if scheduled:
            from .drag.protocol import check_schedule, patch_mask, declared_schedule, SETTLING, SEQUENCE_SCHEMAS
            schema=manifest['action_schema']
            check_schedule(a['command_velocity'],a['selected'],a['transition_active'],transitions=t-1,schema=schema)
            if not np.array_equal(clock.state_time_s,Clock(1e-4,417,t-1).state_time_s):
                raise ValueError('Changed scheduled drag observation clock')
            if manifest.get('solver_config',{}).get('dt_s')!=clock.substep_dt_s:
                raise ValueError('Drag solver timestep differs from observation clock')
            if manifest.get('settling_policy')!=asdict(SETTLING):raise ValueError('Changed drag settling policy')
            if manifest.get('settling_mode')!='group-quiet-tail-v1' or manifest.get('settling_transitions')!=96:
                raise ValueError('Changed drag settling protocol')
            action=manifest['action']
            expected=patch_mask(a['x'][0],ids,action['object_id'],action['fraction'])
            command,active=declared_schedule(expected,action,schema=schema,transitions=t-1)
            if schema in SEQUENCE_SCHEMAS:
                if (not np.isin(a['mat_type'],[0,1]).all()
                        or manifest.get('drag_material_policy')!='elastic-plasticine-only-v1'
                        or manifest.get('wall_selection_policy')!='strict-wall-free-v1'):
                    raise ValueError('Sequence drag requires no sand and strict walls')
                from .walls import validate_evidence
                if not validate_evidence(manifest,a)['wall_free']:reasons.append('drag_sequence_wall_contact')
            if not all(np.array_equal(left,right) for left,right in (
                    (expected,a['selected']),(command,a['command_velocity']),(active,a['transition_active']))):
                raise ValueError('Drag action differs from declared recipe')
        if manifest["task"] == "passive":
            if selected.any() or a["transition_active"].any() or a["command_velocity"].any():
                reasons.append("passive_contains_action")
        else:
            intended = manifest.get("drag_object_ids", [])
            if not selected.any() or not intended or set(ids[selected]) != set(intended):
                reasons.append("drag_object_selection_mismatch")
            if not a["transition_active"].any():
                reasons.append("drag_has_no_active_transition")
            if "settling_policy" not in manifest or "settle_x" not in a or "settle_v" not in a:
                reasons.append("drag_missing_settling_evidence")
            else:
                monitor = SettlingMonitor(ids, SettlingPolicy(**manifest["settling_policy"]))
                sx, sv = a["settle_x"], a["settle_v"]
                if sx.ndim != 3 or sx.shape[1:] != (n, 3) or sv.shape != sx.shape or len(sx) < 2:
                    raise ValueError("settle_array_shape")
                if scheduled and len(sx)!=97:raise ValueError('Incomplete scheduled settling evidence')
                state = "pending"
                for i in range(1, len(sx)):
                    if state != "pending" and not (scheduled and manifest.get('settling_mode')=='group-quiet-tail-v1'):
                        raise ValueError("settling_record_continues_past_terminal_outcome")
                    state = monitor.observe(sx[i - 1], sx[i], sv[i])
                if state != "settled":
                    reasons.append("initial_settling_not_certified")
                if len(sx)-1 > monitor.policy.max_transitions:reasons.append('settling_budget_exceeded')
                if not np.array_equal(sx[-1], a["x"][0]) or not np.array_equal(sv[-1], a["v"][0]):
                    reasons.append("settled_to_recorded_state_mismatch")
                metrics["settling_transitions"] = len(sx) - 1
                metrics["settling_last"] = monitor.history[-1]
        if np.any(a["particle_volume"] <= 0) or np.any(a["density"] <= 0):
            reasons.append("nonpositive_volume_or_density")
        if np.any(a["E"] < 0) or np.any(a["nu"] <= -1) or np.any(a["nu"] >= .5):
            reasons.append("invalid_elastic_parameters")
        if not np.isin(a["mat_type"], [0, 1, 2]).all() or np.any(a["yield_stress"] < 0):
            reasons.append("invalid_material_metadata")
        points = [a["x"]]
        if "settle_x" in a:
            points.append(a["settle_x"])
        min_z = min(float(x[..., 2].min()) for x in points)
        metrics.update({"states": t, "particles": n, "object_counts": np.bincount(ids).tolist(),
                        "max_floor_penetration_m": max(0., manifest["floor_m"] - min_z),
                        "mean_final_speed_m_s": float(np.linalg.norm(a["v"][-1], axis=-1).mean()),
                        "final_transition_active": bool(a["transition_active"][-1]),
                        "object_mass_kg": [float(np.sum(a["particle_volume"][ids == oid]) * a["density"][oid]) for oid in range(k)]})
        if metrics["max_floor_penetration_m"] > policy.floor_tolerance_m:
            reasons.append("floor_penetration")
        dx = manifest["grid_lim_m"] / manifest["n_grid"]
        for x in points:
            base = np.trunc(x / dx - .5)
            if np.any(x < 0) or np.any(base < 0) or np.any(base + 2 >= manifest["n_grid"]):
                reasons.append("unsafe_transfer_stencil")
        j = np.linalg.det(a["F_trial"])
        metrics["minimum_det_F_trial"] = float(j.min())
        if np.any(j <= 0):
            reasons.append("nonpositive_det_F_trial")
        # Opt-in, versioned policy for fresh geometry releases. Historical
        # controls keep their original policy; never reinterpret old receipts.
        if 'collapse_policy' in manifest:
            from .collapse import CollapsePolicy, analyze
            collapse = analyze(a['x'], ids, a['F_trial'], CollapsePolicy(**manifest['collapse_policy']))
            metrics['collapse'] = collapse
            reasons.extend(collapse['reasons'])
            if scheduled:
                from .walls import FIELDS,validate_evidence
                sf=a.get('settle_F_trial')
                if sf is None:raise ValueError('Missing settling deformation history')
                settling_collapse=analyze(a['settle_x'],ids,sf,CollapsePolicy(**manifest['collapse_policy']))
                metrics['settling_collapse']=settling_collapse
                reasons.extend('settling:'+r for r in settling_collapse['reasons'])
                evidence=validate_evidence(dict(manifest,clock=asdict(Clock(clock.substep_dt_s,
                    clock.substeps_per_transition,96))),dict(x=a['settle_x'],**{k:a['settle_'+k] for k in FIELDS}))
                metrics['settling_wall_evidence']=evidence
                # Initial qualification is strict for ALL settling materials.
                # Sand-only motion contact remains separately attributed.
                if not evidence['wall_free']:reasons.append('settling_wall_contact')
                if not np.array_equal(sf[-1],a['F_trial'][0]):reasons.append('settling_deformation_discontinuity')
        if 'fragmentation_policy' in manifest:
            from .drag.fragmentation import analyze as fragmentation
            # New manifests opt in explicitly; historical receipts stay intact.
            # Quarantine is a dataset inclusion rule, not a fracture-law claim.
            for prefix in ('', 'settle_'):
                report = fragmentation(a[prefix+'x'], ids, dx_m=dx,
                                       selected=a['selected'] if not prefix else None)
                metrics[prefix+'fragmentation'] = report
                reasons.extend(prefix+reason for reason in report['reasons'])
        if reasons:
            return _receipt("rejected", reasons, metrics, path, digest, policy, manifest=manifest)
        return _receipt("pending", PHYSICS_BLOCKERS, metrics, path, digest, policy, True, manifest)
    except Exception as error:
        return _receipt("failed", [str(error) or type(error).__name__], metrics, path, digest, policy, manifest=manifest)


def accounting(receipts):
    counts = Counter(row["status"] for row in receipts)
    reasons = Counter(reason for row in receipts for reason in set(row["reasons"]))
    strata = defaultdict(Counter)
    for row in receipts:
        strata[canonical_json(row["strata"])][row["status"]] += 1
    return {"attempts": len(receipts), "states": {s: counts[s] for s in ("accepted", "rejected", "failed", "pending")},
            "overlapping_reason_counts": dict(sorted(reasons.items())),
            "strata": {k: dict(v) for k, v in sorted(strata.items())}}


def _fsync_file(path):
    with Path(path).open("rb") as stream:
        os.fsync(stream.fileno())


def _fsync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def check_tar(path):
    """Read every payload, require terminal blocks and nonempty complete samples."""
    path = Path(path)
    if path.stat().st_size % 512 or path.stat().st_size < 1024:
        raise ValueError("Invalid TAR length")
    with path.open("rb") as f:
        f.seek(-1024, os.SEEK_END)
        if f.read() != bytes(1024):
            raise ValueError("Missing TAR terminal blocks")
    members = set()
    with tarfile.open(path, "r:") as tar:
        for member in tar:
            if not member.isfile() or member.name in members:
                raise ValueError("Invalid/duplicate TAR member")
            payload = tar.extractfile(member)
            if payload is None or len(payload.read()) != member.size:
                raise ValueError("Incomplete TAR payload")
            members.add(member.name)
    if not members:
        raise ValueError("Empty TAR")
    return len(members)


def _batch_payload(rows):
    a = [row[1] for row in rows]
    properties = []
    for arrays in a:
        props = np.column_stack([arrays[key] for key in ("E", "nu", "density", "mat_type", "yield_stress")])
        properties.append(props[arrays["object_ids"]])
    # Terminal state has no outgoing transition; compatibility T-gate ends false.
    gate = np.stack([np.r_[x["transition_active"], False] for x in a])
    values = {"particle_pos": np.stack([x["x"] for x in a]).astype(np.float32),
              "particle_vel": np.stack([x["v"] for x in a]).astype(np.float32),
              "initial_vel": np.stack([x["v"][0] for x in a]).astype(np.float32),
              "material_properties": np.stack(properties).astype(np.float32),
              "object_ids": np.stack([x["object_ids"] for x in a]).astype(np.int32),
              "valid_particles": np.ones((len(a), *a[0]["x"].shape[:2]), bool),
              "n_objects": np.array([row[0]["n_objects"] for row in rows], np.int32),
              "grab_velocity": np.stack([x["command_velocity"] for x in a]).astype(np.float32),
              "is_grabbed": np.stack([x["selected"] for x in a]).astype(np.int32),
              "drag_active": gate.astype(bool)}
    return values


def validate_export_cohorts(paths):
    """Keep legacy diagnostics compatible, but never mix declared object roles."""
    cohorts = set()
    for path in paths:
        with h5py.File(path, 'r') as f:
            manifest = json.loads(f.attrs['manifest_json'])
        role, split_hash = manifest.get('object_cohort'), manifest.get('split_manifest_sha256')
        if role is not None and role not in ('train', 'validation', 'test'):
            raise ValueError('Unknown object cohort')
        if role is None and split_hash is not None:
            raise ValueError('Split hash without an object cohort')
        if role in ('validation', 'test') and (not isinstance(split_hash, str) or len(split_hash) != 64
                or any(c not in '0123456789abcdef' for c in split_hash)):
            raise ValueError('Evaluation cohort requires a frozen split hash')
        cohorts.add((role, split_hash, manifest.get('split'), manifest.get('evaluation'),manifest.get('action_schema')))
    if len(cohorts) != 1:
        raise ValueError('Mixed object cohorts or split versions cannot share an export')


def export_diagnostic(paths, output, policy=None, batch_size=4, batches_per_shard=8, selection_policy=None):
    """Atomic no-clobber diagnostic transaction. Never marks training acceptance.

    A preexisting matching complete result is an idempotent resume, after all
    output hashes are checked. Process crashes leave a conspicuous lock/staging
    directory requiring reviewed recovery, never a partial published release.
    """
    policy = policy or Policy()
    if not all(isinstance(x, int) and not isinstance(x, bool) and x > 0 for x in (batch_size, batches_per_shard)):
        raise ValueError("Positive integer batch/shard sizes required")
    paths = sorted(map(Path, paths))
    if not paths or len(set(map(lambda p: p.resolve(), paths))) != len(paths):
        raise ValueError("Nonempty unique raw paths required")
    receipts = [validate_raw(path, policy) for path in paths]
    if not all(row["integrity_pass"] and row["status"] == "pending" for row in receipts):
        raise ValueError("Mandatory validation blocked export: " + canonical_json(accounting(receipts)))
    episode_ids = [r["episode_id"] for r in receipts]
    if len(set(episode_ids)) != len(episode_ids):
        raise ValueError("Duplicate episode identity")
    validate_export_cohorts(paths)
    # Worker allocation and physical basename must not determine export order.
    ordered = sorted(zip(paths, receipts), key=lambda pair: pair[1]["episode_id"])
    paths, receipts = map(list, zip(*ordered))
    request = {"raw": sorted((r["episode_id"], r["raw_sha256"]) for r in receipts),
               "policy": asdict(policy), "validator_sha256": sha256(__file__), "batch_size": batch_size,
               "contracts_sha256": sha256(Path(__file__).with_name("contracts.py")),
               "batches_per_shard": batches_per_shard, "export_kind": "diagnostic_only"}
    if selection_policy is not None:
        request['selection_policy'] = selection_policy
    request_hash = hashlib.sha256(canonical_json(request).encode()).hexdigest()
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.parent / ("." + output.name + ".lock")
    lock_fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    stage = None
    try:
        os.write(lock_fd, canonical_json({"pid": os.getpid(), "request_sha256": request_hash}).encode())
        os.fsync(lock_fd)
        if output.exists():
            complete = json.loads((output / "COMPLETE.json").read_text())
            if complete["request_sha256"] != request_hash or complete["accepted_for_training"] is not False:
                raise FileExistsError("Destination is not this exact diagnostic transaction")
            for item in complete["artifacts"]:
                path = output / item["path"]
                if sha256(path) != item["sha256"]:
                    raise ValueError("Existing output hash mismatch")
                if path.suffix == ".tar":
                    check_tar(path)
            return dict(complete, resumed=True)
        stage = Path(tempfile.mkdtemp(prefix="." + output.name + ".staging-", dir=output.parent))
        # Streaming bounded by one shard; the small pending list contains paths,
        # never the whole collection's particle arrays.
        groups = defaultdict(list)
        for path, receipt in zip(paths, receipts):
            with h5py.File(path, "r") as f:
                meta = json.loads(f.attrs["manifest_json"])
                shape = f["x"].shape
            groups[(meta["task"], meta["n_objects"], shape)].append((path, receipt))
        artifacts, shard_count, episode_rows = [], 0, []
        for group, sources in sorted(groups.items()):
            per_shard = batch_size * batches_per_shard
            for offset in range(0, len(sources), per_shard):
                shard_name = f"{shard_count:05d}.tar"
                shard_count += 1
                with tarfile.open(stage / shard_name, "x:", format=tarfile.USTAR_FORMAT) as tar:
                    chunk = sources[offset:offset + per_shard]
                    for begin in range(0, len(chunk), batch_size):
                        subset = chunk[begin:begin + batch_size]
                        rows = []
                        metadata = []
                        key = f"batch_{begin // batch_size:08d}"
                        for row_index, (path, receipt) in enumerate(subset):
                            if sha256(path) != receipt["raw_sha256"]:
                                raise ValueError("Raw changed after validation")
                            manifest, arrays = read_raw(path, policy)
                            if sha256(path) != receipt["raw_sha256"]:
                                raise ValueError("Raw changed during export")
                            rows.append((manifest, arrays))
                            provenance = {"episode_id": receipt["episode_id"], "raw_sha256": receipt["raw_sha256"],
                                          "state_indices": list(range(len(arrays["x"]))),
                                          "state_time_s": arrays["state_time_s"].tolist(),
                                          "task": manifest["task"], "split": manifest["split"],
                                          "population": manifest["population"], "asset_ids": manifest["case"]["asset_ids"],
                                          "floor_m": manifest["floor_m"], "volume_policy": manifest["volume_policy"],
                                          "accepted_for_training": False}
                            if 'object_cohort' in manifest:
                                provenance['object_cohort'] = manifest['object_cohort']
                                provenance['split_manifest_sha256'] = manifest.get('split_manifest_sha256')
                            if 'evaluation' in manifest:
                                provenance['evaluation'] = manifest['evaluation']
                            if selection_policy is not None:
                                provenance['selection_policy'] = selection_policy
                            if 'collapse_policy' in manifest:
                                provenance['collapse_policy'] = manifest['collapse_policy']
                            if 'fragmentation_policy' in manifest:
                                provenance['fragmentation_policy'] = manifest['fragmentation_policy']
                            if 'action_schema' in manifest:
                                provenance['action_schema']=manifest['action_schema']
                                provenance['command_semantics']=manifest['command_semantics']
                                if manifest['action_schema'] in ('mpm-drag-command-sequence-v2','mpm-drag-command-sequence-v3','mpm-drag-command-sequence-v4'):
                                    provenance['action']=manifest['action']
                                    provenance['drag_material_policy']=manifest['drag_material_policy']
                                    provenance['wall_selection_policy']=manifest['wall_selection_policy']
                            metadata.append(provenance)
                            episode_rows.append(dict(provenance, shard=shard_name, sample_key=key, batch_row=row_index))
                        payloads = {name + ".pyd": pickle.dumps(value, protocol=4) for name, value in _batch_payload(rows).items()}
                        payloads["provenance.json"] = canonical_json(metadata).encode()
                        for suffix, payload in payloads.items():
                            member = tarfile.TarInfo(key + "." + suffix)
                            member.size = len(payload)
                            tar.addfile(member, io.BytesIO(payload))
                check_tar(stage / shard_name)
                _fsync_file(stage / shard_name)
                artifacts.append({"path": shard_name, "sha256": sha256(stage / shard_name)})
        (stage / "validation.json").write_text(canonical_json({"receipts": receipts, "accounting": accounting(receipts)}) + "\n")
        _fsync_file(stage / "validation.json")
        artifacts.append({"path": "validation.json", "sha256": sha256(stage / "validation.json")})
        complete = {"schema": "mpm-diagnostic-export-v1", "request_sha256": request_hash,
                    "accepted_for_training": False, "physics_blockers": PHYSICS_BLOCKERS,
                    "episodes": episode_rows, "shards": shard_count, "artifacts": artifacts}
        (stage / "COMPLETE.json").write_text(canonical_json(complete) + "\n")
        _fsync_file(stage / "COMPLETE.json")
        _fsync_dir(stage)
        if output.exists():
            raise FileExistsError("Destination appeared during export")
        os.rename(stage, output)
        _fsync_dir(output.parent)
        return dict(complete, resumed=False)
    finally:
        os.close(lock_fd)
        # Normal error paths retain failed staging evidence, but unlock retries.
        # A killed process retains its lock for deliberate recovery.
        lock.unlink()
        _fsync_dir(output.parent)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path, nargs="+")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--diagnostic-export", type=Path)
    args = parser.parse_args()
    rows = [validate_raw(path) for path in args.raw]
    with args.report.open("x") as f:
        json.dump({"receipts": rows, "accounting": accounting(rows)}, f, indent=2, allow_nan=False)
    if args.diagnostic_export:
        export_diagnostic(args.raw, args.diagnostic_export)
    print(canonical_json(accounting(rows)))
    # 3 = structurally valid but not approved; 2 = rejected/technical failure.
    raise SystemExit(3 if all(row["integrity_pass"] for row in rows) else 2)
