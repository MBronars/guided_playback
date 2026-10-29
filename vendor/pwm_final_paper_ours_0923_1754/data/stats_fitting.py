"""Bounded raw-quantity calibration, deterministic fitting and physical-unit audits.

No legacy statistics are read. The fitting population is valid particle-frames
and rigid-body-frames, NOT uniformly weighted episodes. A separate reservoir is
maintained per domain and window policy; fitting takes the maximum of each
policy's p99 bounds, not a silently reweighted pool of those policies.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from data.normalization import (NORMALIZATION_DOMAINS, NORMALIZATION_PROFILES,
                                RIGID_DOMAINS, domain_normalizers, resolve_normalization)

COLLECTOR_VERSION = "raw-reservoir-v1"
_ARRAY_NAMES = ("particle", "pose", "particle_rows", "pose_rows")


def _implementation_fingerprint():
    """Quantity extraction and geometry changes invalidate automatic cache reuse.

    The adapter manifest separately fingerprints the readers, exclusions and
    calibration traversal. Include the downstream raw masking/reservoir code,
    batch accessors and rigid-geometry kernels here as well.
    """
    root = Path(__file__).resolve().parents[1]
    names = ("data/stats_fitting.py", "data/batch.py", "data/normalization.py",
             "utils/pose_ops.py", "utils/rigid_ops.py")
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _array_digest(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256(str(value.dtype).encode() + str(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def write_json_new(path, value):
    """Exclusive creation: never overwrite frozen stats or an existing candidate."""
    path = Path(path)
    encoded = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(encoded)


class PriorityReservoir:
    """Exact fixed-capacity uniform row sample via independent random priorities.

    Retaining the smallest priorities is equivalent to sampling without
    replacement. Ties use source row index. Memory is O(capacity + chunk_rows),
    and results are invariant to how callers chunk the same ordered row stream.
    """

    def __init__(self, capacity, channels, seed, chunk_rows=65536):
        for name, value in (("capacity", capacity), ("channels", channels),
                            ("seed", seed), ("chunk_rows", chunk_rows)):
            if type(value) is not int or value < (0 if name == "seed" else 1):
                raise ValueError(f"{name} must be a {'nonnegative' if name == 'seed' else 'positive'} integer")
        self.capacity, self.channels, self.chunk_rows = capacity, channels, chunk_rows
        self.rng = np.random.default_rng(seed)
        self.rows = np.empty((0, channels), dtype=np.float32)
        self.indices = np.empty(0, dtype=np.int64)
        self.priorities = np.empty(0, dtype=np.float64)
        self.seen = 0

    def add(self, rows):
        rows = np.asarray(rows, dtype=np.float32)
        if rows.ndim != 2 or rows.shape[1] != self.channels or not np.isfinite(rows).all():
            raise ValueError(f"reservoir expects finite [N,{self.channels}] rows")
        for start in range(0, len(rows), self.chunk_rows):
            block = rows[start:start + self.chunk_rows]
            priorities = np.concatenate((self.priorities, self.rng.random(len(block))))
            indices = np.concatenate((self.indices, np.arange(self.seen, self.seen + len(block))))
            combined = np.concatenate((self.rows, block))
            keep = np.lexsort((indices, priorities))[:self.capacity]
            self.rows, self.indices, self.priorities = combined[keep], indices[keep], priorities[keep]
            self.seen += len(block)


def calibration_identity(source_manifest, *, capacity):
    if type(capacity) is not int or capacity < 1:
        raise ValueError("reservoir capacity must be a positive integer")
    identity = dict(
        schema_version=1, collector_version=COLLECTOR_VERSION,
        implementation_sha256=_implementation_fingerprint(),
        runtime_versions=dict(numpy=np.__version__, torch=str(torch.__version__)),
        source_manifest=source_manifest, capacity_per_representation=capacity,
        weighting="uniform valid particle-frame / complete rigid-body-frame, not episode-uniform",
        particle_transitions="all adjacent stored frames in the C+F window; both endpoints valid",
        pose_transitions="future F only; rigid body with all linked points valid at both endpoints",
        sampling="smallest iid PCG64 priorities; ties broken by source row index",
    )
    return {**identity, "sha256": json_digest(identity)}


def raw_motion_rows(batch, metadata, particle_rows="all"):
    """Extract physical targets with temporal particle and per-element body masks.

    particle_rows="supervised" keeps, per element, only the particles the objective supervises through the
    particle channel: the deformable rows of a scene that also has rigid bodies, whose motion has its own pose
    rows. An element without deformable rows, and every passive domain, keeps all valid particles. "all" is the
    historical population; on folding it lets the commanded gripper points set the cloth scale."""
    if particle_rows not in ("all", "supervised"):
        raise ValueError("particle_rows must be all or supervised")
    pos = batch.positions.detach().cpu().float()
    B, T, N, _ = pos.shape
    valid = torch.as_tensor(metadata["valid_particle_mask"], dtype=torch.bool).cpu()
    if tuple(valid.shape) != (B, T, N):
        raise ValueError("source valid_particle_mask must have shape [B,T,N]")
    prefix = batch.get_valid_mask().detach().cpu()[:, None, :]
    if bool((valid & ~prefix).any()):
        raise ValueError("Unsupported source validity outside the batch's representable prefix; "
                         "non-prefix or newly appearing particles must not be silently dropped during calibration")
    valid = valid & prefix
    pair_valid = valid[:, 1:] & valid[:, :-1]
    particle_valid = pair_valid
    if (particle_rows == "supervised" and getattr(batch, "body_type", None) is not None
            and getattr(batch, "particle_body", None) is not None):
        deform = (batch.body_type.gather(1, batch.particle_body) == 0).cpu() & prefix[:, 0]      # [B, N]
        keep = torch.where(deform.any(dim=1, keepdim=True), deform, prefix[:, 0])
        particle_valid = pair_valid & keep[:, None, :]
    particle = (pos[:, 1:] - pos[:, :-1])[particle_valid].numpy()
    if not np.isfinite(particle).all():
        raise ValueError("nonfinite valid particle displacements")
    pose = np.empty((0, 6), dtype=np.float32)
    from data.normalization import PASSIVE_DOMAINS
    if batch.domain not in PASSIVE_DOMAINS:
        if batch.pose_deltas_gt is None or batch.body_type is None or batch.particle_body is None:
            raise ValueError("rigid calibration requires pose deltas, body_type and particle_body")
        deltas = batch.pose_deltas_gt.detach().cpu().float()
        btype, pbody = batch.body_type.cpu(), batch.particle_body.cpu()
        F, K = T - batch.context_len, btype.shape[1]
        if tuple(deltas.shape) != (B, F, K, 6):
            raise ValueError("pose calibration shape does not match C/F and body count")
        future_valid = pair_valid[:, batch.context_len - 1:]
        body_valid = torch.zeros((B, F, K), dtype=torch.bool)
        for k in range(K):
            members = (pbody == k) & batch.get_valid_mask().cpu()
            present = members.any(dim=-1) & (btype[:, k] == 1)
            complete = (future_valid | ~members[:, None, :]).all(dim=-1)
            if batch.body_template_mask is not None:
                present &= batch.body_template_mask.cpu()[:, k].any(dim=-1)
            body_valid[:, :, k] = present[:, None] & complete
        pose = deltas[body_valid].numpy()
        if not np.isfinite(pose).all():
            raise ValueError("nonfinite valid rigid-pose displacements")
    return particle, pose


def collect_rows(iterator, identity):
    """Collect from a finite raw iterator; preserve exact selected-window ledger."""
    source = identity["source_manifest"]
    seed = int(source["seed"])
    domain = source["domain"]
    capacity = identity["capacity_per_representation"]
    particle, pose = PriorityReservoir(capacity, 3, seed), PriorityReservoir(capacity, 6, seed + 1)
    ledger = []
    for batch, metadata in iterator:
        if batch.domain != domain or metadata["domain"] != domain:
            raise ValueError("mixed-domain calibration stream")
        if metadata["split"] != source["split"]:
            raise ValueError("calibration source/record split mismatch")
        if (metadata["context_len"], metadata["future_len"], metadata["window_policy"], metadata["seed"]) != (
                source["context_len"], source["future_len"], source["window_policy"], seed):
            raise ValueError("calibration record window/seed does not match source identity")
        dp, dr = raw_motion_rows(batch, metadata)
        row = {k: v for k, v in metadata.items() if k != "valid_particle_mask"}
        row.update(particle_row_start=particle.seen, particle_rows=len(dp),
                   pose_row_start=pose.seen, pose_rows=len(dr),
                   valid_mask_sha256=_array_digest(metadata["valid_particle_mask"].cpu().numpy()))
        ledger.append(row)
        particle.add(dp)
        pose.add(dr)
    if not len(particle.rows) or (domain in RIGID_DOMAINS and not len(pose.rows)):
        raise ValueError(f"{domain}: empty valid particle/rigid calibration population")
    arrays = dict(particle=particle.rows, pose=pose.rows,
                  particle_rows=particle.indices, pose_rows=pose.indices)
    metadata = dict(identity=identity, ledger=ledger,
                    seen=dict(particle=particle.seen, pose=pose.seen),
                    retained=dict(particle=len(particle.rows), pose=len(pose.rows)),
                    array_sha256={name: _array_digest(value) for name, value in arrays.items()})
    metadata["sha256"] = json_digest(metadata)
    return arrays, metadata


def save_collection_cache(path, arrays, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.savez_compressed(handle, **arrays, metadata_json=np.array(json.dumps(metadata, sort_keys=True,
                                                                              allow_nan=False)))


def load_collection_cache(path, expected_identity=None):
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {*_ARRAY_NAMES, "metadata_json"}:
            raise ValueError("calibration cache has unsupported fields")
        arrays = {name: archive[name] for name in _ARRAY_NAMES}
        metadata = json.loads(str(archive["metadata_json"]))
    stored_hash = metadata.pop("sha256", None)
    if json_digest(metadata) != stored_hash:
        raise ValueError("calibration cache metadata checksum mismatch")
    metadata["sha256"] = stored_hash
    identity = metadata["identity"]
    payload = {k: v for k, v in identity.items() if k != "sha256"}
    if json_digest(payload) != identity.get("sha256"):
        raise ValueError("calibration cache identity checksum mismatch")
    if identity["collector_version"] != COLLECTOR_VERSION:
        raise ValueError("calibration cache collector version is stale")
    if expected_identity is not None and identity != expected_identity:
        raise ValueError("calibration cache does not match requested sources/windows/settings")
    for name, values in arrays.items():
        if _array_digest(values) != metadata["array_sha256"][name]:
            raise ValueError(f"calibration cache array checksum mismatch: {name}")
        width = 3 if name == "particle" else 6
        if name in ("particle", "pose"):
            if values.dtype != np.float32 or values.ndim != 2 or values.shape[1] != width or not np.isfinite(values).all():
                raise ValueError(f"invalid calibration array {name}")
        elif values.dtype != np.int64 or values.ndim != 1:
            raise ValueError(f"invalid reservoir source-row indices {name}")
    return arrays, metadata


def _half_ranges(rows, pose=False):
    if len(rows) == 0:
        raise ValueError("Cannot fit an empty calibration array")
    q = np.percentile(np.abs(rows), 99, axis=0, method="linear")
    result = [max(q[0], q[1]), q[2]]
    if pose:
        result.extend((max(q[3], q[4]), q[5]))
    result = np.asarray(result, dtype=np.float32)
    if not np.isfinite(result).all() or np.any(result <= 0):
        raise ValueError("Zero/nonfinite p99 scale: inspect data; no automatic scale floor is applied")
    return result


def collection_coverage(metadata):
    """Summarize recorded source coverage without changing historical caches.

    A record can hold several episode windows. The ledger contains successful
    records and their skipped-before entries, not every excluded/read attempt.
    Missing optional historical ledger details remain unknown, not zero.
    """
    source = metadata["identity"]["source_manifest"]
    ledger = metadata["ledger"]
    available = {row["path"] for row in source["sources"]}
    contributing = {row["source"] for row in ledger}
    skipped = [entry for row in ledger for entry in row.get("skipped_before", ())]
    visited = contributing | {row["source"] for row in skipped if "source" in row}
    windows_known = all("window_starts" in row for row in ledger)
    skips_known = all("skipped_before" in row for row in ledger)
    limit = source.get("max_records")
    return dict(
        records=len(ledger),
        episode_windows=sum(len(row["window_starts"]) for row in ledger) if windows_known else None,
        record_limit=limit, record_limit_reached=len(ledger) >= limit if limit is not None else None,
        shards=dict(contributing=len(contributing), visited_recorded=len(visited), available=len(available),
                    visited_recorded_fraction=len(visited & available) / len(available) if available else None),
        skipped_records_recorded=len(skipped) if skips_known else None,
        meaning="Per-cache/window-policy coverage, not a corpus episode-coverage estimate. "
                "Recorded visited shards include successful and logged skipped-before records only. "
                "Blocklist-filtered records/shards and trailing skips after the last emitted record "
                "are not counted; recorded visits/skips can be lower bounds. Different policies "
                "can revisit the same sources and must not be summed as unique corpus coverage.",
    )


def fit_artifact(caches, *, artifact_version):
    """Fit both profiles using train-only rows from the fixed all-domain reference."""
    if not isinstance(artifact_version, str) or not artifact_version.strip() or artifact_version.startswith("frozen-"):
        raise ValueError("A distinct nonempty candidate artifact_version is required (not frozen-*)")
    particles, poses, summaries = {}, {}, []
    seen = set()
    policies = {}
    reference_contract = None
    for arrays, metadata in caches:
        source = metadata["identity"]["source_manifest"]
        domain, policy = source["domain"], source["window_policy"]
        if source["split"] != "train":
            raise ValueError("Fitting is restricted to training episodes; validation is audit-only")
        if domain not in NORMALIZATION_DOMAINS or (domain, policy) in seen:
            raise ValueError("Fit requires supported, unique domain/window-policy caches")
        contract = (source["context_len"], source["future_len"], source["seed"])
        if reference_contract is not None and contract != reference_contract:
            raise ValueError("Fit caches must share context/horizon/seed")
        reference_contract = contract
        seen.add((domain, policy))
        policies.setdefault(domain, set()).add(policy)
        p = _half_ranges(arrays["particle"])
        particles[domain] = np.maximum(particles.get(domain, p), p)
        if domain in RIGID_DOMAINS:
            p = _half_ranges(arrays["pose"], pose=True)
            poses[domain] = np.maximum(poses.get(domain, p), p)
        summaries.append(dict(domain=domain, window_policy=policy, cache_sha256=metadata["sha256"],
                              identity_sha256=metadata["identity"]["sha256"],
                              source_manifest_sha256=source["sha256"],
                              context_len=source["context_len"], future_len=source["future_len"],
                              native_frame_stride=source.get("native_frame_stride"), seed=source["seed"],
                              seen=metadata["seen"], retained=metadata["retained"],
                              coverage=collection_coverage(metadata)))
    if set(particles) != set(NORMALIZATION_DOMAINS) or set(poses) != set(RIGID_DOMAINS):
        raise ValueError("Both profiles require the fixed four-domain calibration reference; missing domains")
    if len({tuple(sorted(p)) for p in policies.values()}) != 1:
        raise ValueError("Every domain must use the same explicit window-policy set")
    translation = np.maximum(np.max(list(particles.values()), axis=0),
                             np.max([v[:2] for v in poses.values()], axis=0))
    rotation = np.max([v[2:] for v in poses.values()], axis=0)
    return dict(
        schema_version=1, artifact_version=artifact_version,
        units=dict(position="m", translation_delta="m/frame", rotation_delta="rad/frame"),
        position_half_range_m=0.8,
        profiles={
            "xy_p99_per_domain": dict(particle={d: particles[d].tolist() for d in NORMALIZATION_DOMAINS},
                                       pose={d: poses[d].tolist() for d in RIGID_DOMAINS}),
            "xy_p99_global": dict(translation=translation.tolist(), rotation=rotation.tolist()),
        },
        provenance=dict(kind="audited_raw_candidate", collector_version=COLLECTOR_VERSION,
                        reference_domains=list(NORMALIZATION_DOMAINS), percentile=99,
                        quantile_method="linear", xy_group="max(p99(abs(x)), p99(abs(y)))",
                        policy_aggregation="componentwise maximum of per-policy percentiles",
                        global_aggregation="componentwise maxima over fixed all-domain particle and pose bounds",
                        caches=sorted(summaries, key=lambda row: (row["domain"], row["window_policy"])),
                        limitations="Source tar identity is path/size/mtime, not full-content hashing. "
                        "Explicit fitting may use verified historical caches with their recorded code identity; "
                        "automatic collect reuse requires today's matching implementation fingerprint. "
                        "Reservoir estimates have finite-sample uncertainty. These diagnostics do not select "
                        "the best profile without matched trained physical-unit rollouts."),
    )


def _channel_report(rows, normalizer):
    if not len(rows):
        return None
    x = torch.from_numpy(rows)
    normalized = normalizer.normalize(x, clip=False)
    reconstruction = normalizer.denormalize(normalized.clamp(-1, 1))
    error = reconstruction - x
    quantiles = (50, 90, 99, 99.9, 100)
    return dict(
        rows=len(rows),
        abs_physical_percentiles={str(q): np.percentile(np.abs(rows), q, axis=0).tolist() for q in quantiles},
        normalized_rms_per_channel=normalized.square().mean(0).sqrt().tolist(),
        normalized_abs_max_per_channel=normalized.abs().amax(0).tolist(),
        clipped_fraction_per_channel=(normalized.abs() > 1).float().mean(0).tolist(),
        clipped_fraction_any_channel=float((normalized.abs() > 1).any(-1).float().mean()),
        clipped_target_reconstruction_rmse_per_channel=error.square().mean(0).sqrt().tolist(),
        clipped_target_reconstruction_max_abs_per_channel=error.abs().amax(0).tolist(),
    )


def audit_caches(caches, artifact=None):
    """Audit only; never fit or mutate the supplied normalization specification.

    Reservoir rows are not continuous trajectories. Reconstruction here is
    single-transition error, not an integrated rollout or a paper metric.
    """
    # Freeze the chosen bounds before iterating the caches. Include the complete
    # numerical contracts so an audit remains attributable if its artifact path
    # is later moved or removed; filenames alone are not experimental provenance.
    snapshots = {
        profile: resolve_normalization({"profile": profile,
                    **({"artifact": str(artifact)} if artifact is not None else {})})
        for profile in NORMALIZATION_PROFILES
    }
    if len({snapshot["artifact_version"] for snapshot in snapshots.values()}) != 1:
        raise ValueError("Normalization artifact changed while resolving audit profiles")
    reports = []
    for arrays, metadata in caches:
        source = metadata["identity"]["source_manifest"]
        entry = dict(domain=source["domain"], split=source["split"], window_policy=source["window_policy"],
                     cache_sha256=metadata["sha256"], coverage=collection_coverage(metadata), profiles={})
        for profile, snapshot in snapshots.items():
            particle, _, pose = domain_normalizers(snapshot, entry["domain"])
            entry["profiles"][profile] = dict(particle=_channel_report(arrays["particle"], particle),
                                              pose=_channel_report(arrays["pose"], pose) if pose else None)
        reports.append(entry)
    return dict(schema_version=1, normalization=snapshots, reports=reports,
                meaning="Physical units are m/frame and rad/frame. Clipped-target reconstruction is "
                "per-transition on reservoir samples, NOT an integrated rollout. Validation is audit-only; "
                "training caches are calibration diagnostics, not independent validation.")


def collect_domain_cache(domain, *, output_dir, context_len, future_len, seed, max_records,
                         capacity, window_policy, split="train"):
    """Resolve actual raw source identity before deciding whether a cache is reusable."""
    from data.calibration import calibration_source_manifest, iter_calibration_batches
    from data.unified_dataset import build_domain_dataset

    ds = build_domain_dataset(domain, split, context_len=context_len, future_len=future_len, raw=True)
    ds.emit_sidecar = domain != "mpm"
    if window_policy == "uniform" and hasattr(ds, "motion_alpha"):
        ds.motion_alpha = 0.0
    request = dict(seed=seed, window_policy=window_policy, max_records=max_records)
    manifest = calibration_source_manifest(ds, **request)
    identity = calibration_identity(manifest, capacity=capacity)
    path = Path(output_dir) / f"{domain}-{window_policy}-{identity['sha256']}.npz"
    if path.exists():
        arrays, metadata = load_collection_cache(path, expected_identity=identity)
        return path, metadata, True
    iterator = iter_calibration_batches(domain, context_len=context_len, future_len=future_len,
                                        emit_sidecar=ds.emit_sidecar, split=split, dataset=ds, **request)
    arrays, metadata = collect_rows(iterator, identity)
    after = calibration_source_manifest(ds, **request)
    if after != manifest:
        raise ValueError("Calibration sources/settings changed during collection; cache was not written")
    save_collection_cache(path, arrays, metadata)
    return path, metadata, False


def load_collection_index(path):
    path = Path(path)
    index = json.loads(path.read_text())
    if set(index) != {"schema_version", "caches", "sha256"} or index["schema_version"] != 1:
        raise ValueError("Unsupported collection index schema")
    if json_digest({k: v for k, v in index.items() if k != "sha256"}) != index["sha256"]:
        raise ValueError("Collection index checksum mismatch")
    caches = []
    for row in index["caches"]:
        arrays, metadata = load_collection_cache(path.parent / row["path"])
        if metadata["sha256"] != row["sha256"]:
            raise ValueError("Collection index/cache checksum mismatch")
        caches.append((arrays, metadata))
    if not caches:
        raise ValueError("Collection index is empty")
    return caches
