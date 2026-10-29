"""Physical-unit min/max normalization; arithmetic preserves checkpoint behavior.

Particle deltas use metres/frame. Pose deltas use metres/frame and radians/frame.
The normalizer changes coordinates, not temporal units. Clipping is a caller policy.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch


NORMALIZATION_DOMAINS = ("mpm", "pickup", "cloth", "pushing")   # the fixed reference; every artifact has them
RIGID_DOMAINS = ("pickup", "cloth", "pushing")
# Opt-in domains: an artifact MAY carry their particle scales (and, for the rigid ones, pose scales);
# they never enter the global aggregation, so adding one leaves the reference entries and both
# profiles' values unchanged.
OPTIONAL_DOMAINS = ("water", "rl_push", "mpm_fresh", "mpm_drag", "folding", "yikai", "dex", "mould")
OPTIONAL_RIGID_DOMAINS = ("rl_push", "folding", "yikai", "dex", "mould")   # mould: the paddle is rigid
# Historical name: these domains have no rigid sidecar/pose normalizer. Dragging
# does have commanded particle actions; this is not an actuation classification.
PASSIVE_DOMAINS = ("mpm", "water", "mpm_fresh", "mpm_drag")
NORMALIZATION_PROFILES = ("xy_p99_per_domain", "xy_p99_global")
DEFAULT_PROFILE = NORMALIZATION_PROFILES[0]
DEFAULT_ARTIFACT = Path(__file__).resolve().parents[1] / "stats" / "normalization.json"
_UNITS = {"position": "m", "translation_delta": "m/frame", "rotation_delta": "rad/frame"}
_SNAPSHOT_KEYS = {"schema_version", "artifact_version", "profile", "units",
                  "position_half_range_m", "scales", "provenance", "sha256"}
_ARTIFACT_KEYS = {"schema_version", "artifact_version", "units",
                 "position_half_range_m", "profiles", "provenance"}


def normalization_profile(name: str | None = None) -> str:
    """Resolve only explicit supported aliases; the old baseline is not xy-p99."""
    if name is None:
        return DEFAULT_PROFILE
    if not isinstance(name, str):
        raise ValueError("normalization profile must be a string")
    name = name.removeprefix("unified://")
    if name == "xy_p99":
        return DEFAULT_PROFILE
    if name in NORMALIZATION_PROFILES:
        return name
    if name in {"per-domain", "iso_p9999", "global_iso_p9999", "unified_asinh", "pd_asinh"}:
        raise ValueError(
            f"Retired unified stats profile {name!r}; use commit 55c386c to reproduce "
            "historical normalization. The old 'per-domain' baseline is NOT "
            "xy_p99_per_domain; no automatic substitution is safe.")
    raise ValueError(f"Unknown normalization profile {name!r}; choose {NORMALIZATION_PROFILES}")


def _keys(value, expected, where):
    if not isinstance(value, Mapping) or set(value) != set(expected):
        actual = sorted(map(str, value)) if isinstance(value, Mapping) else type(value).__name__
        raise ValueError(f"{where} must contain exactly {sorted(expected)}; got {actual}")


def _canonical_json(value) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("normalization must contain finite JSON-serializable values") from exc


def _json_copy(value):
    """Return detached built-in containers, not mutable caller-owned config nodes."""
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("normalization JSON object keys must be strings")
        return {key: _json_copy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_copy(item) for item in value]
    # OmegaConf list nodes implement Sequence, but strings must remain scalar.
    from collections.abc import Sequence
    if isinstance(value, Sequence) and not isinstance(value, str):
        return [_json_copy(item) for item in value]
    if value is None or type(value) in (str, bool, int, float):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("normalization must contain finite JSON-serializable values")
        return value
    raise ValueError(f"normalization contains a non-JSON value: {type(value).__name__}")


def _positive_float32(value, where):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{where} must be finite and strictly positive")
    with np.errstate(over="ignore", under="ignore"):
        rounded = float(np.float32(value))
    if not math.isfinite(rounded) or rounded <= 0:
        raise ValueError(f"{where} must be representable as a positive float32")
    return rounded


def _vector(value, length, where):
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{where} must be an array of {length} positive scales")
    return [_positive_float32(x, f"{where}[{i}]") for i, x in enumerate(value)]


def _validate_scales(profile, scales):
    if profile == "xy_p99_global":
        _keys(scales, {"translation", "rotation"}, "global scales")
        return {name: _vector(scales[name], 2, name) for name in ("translation", "rotation")}
    _keys(scales, {"particle", "pose"}, "per-domain scales")
    optional = [d for d in OPTIONAL_DOMAINS if isinstance(scales["particle"], Mapping) and d in scales["particle"]]
    _keys(scales["particle"], NORMALIZATION_DOMAINS + tuple(optional), "particle domains")
    rigid = RIGID_DOMAINS + tuple(d for d in optional if d in OPTIONAL_RIGID_DOMAINS)
    _keys(scales["pose"], rigid, "pose domains")        # an opt-in rigid entry needs both scales
    return {
        "particle": {d: _vector(scales["particle"][d], 2, f"particle.{d}")
                     for d in NORMALIZATION_DOMAINS + tuple(optional)},
        "pose": {d: _vector(scales["pose"][d], 4, f"pose.{d}") for d in rigid},
    }


def _validate_common(value):
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise ValueError("Unsupported normalization schema_version; expected integer 1")
    if not isinstance(value["artifact_version"], str) or not value["artifact_version"].strip():
        raise ValueError("normalization artifact_version must be a nonempty string")
    if value["units"] != _UNITS:
        raise ValueError(f"normalization units must be exactly {_UNITS}")
    if _positive_float32(value["position_half_range_m"], "position_half_range_m") != float(np.float32(0.8)):
        raise ValueError("Unified position_half_range_m must remain 0.8 metres")
    if not isinstance(value["provenance"], dict) or not value["provenance"]:
        raise ValueError("normalization provenance must be a nonempty JSON object")


def _no_duplicate_keys(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"Duplicate normalization JSON key {key!r}")
        value[key] = item
    return value


def load_normalization_artifact(path=DEFAULT_ARTIFACT) -> dict:
    """Load and validate both profiles, including their fixed all-domain aggregation.

    There is no data access, NPZ lookup, fitting or dependence on selected training
    domains. Candidate artifacts use this same schema with a distinct version.
    """
    with Path(path).open(encoding="utf-8") as handle:
        value = _json_copy(json.load(handle, object_pairs_hook=_no_duplicate_keys))
    _keys(value, _ARTIFACT_KEYS, "normalization artifact")
    _validate_common(value)
    _keys(value["profiles"], NORMALIZATION_PROFILES, "normalization profiles")
    value["profiles"] = {p: _validate_scales(p, value["profiles"][p])
                         for p in NORMALIZATION_PROFILES}
    per_domain = value["profiles"]["xy_p99_per_domain"]
    reference = [per_domain["particle"][d] for d in NORMALIZATION_DOMAINS]
    reference_pose = [per_domain["pose"][d] for d in RIGID_DOMAINS]
    expected_global = {
        "translation": [max([v[i] for v in reference] + [v[i] for v in reference_pose]) for i in (0, 1)],
        "rotation": [max(v[i] for v in reference_pose) for i in (2, 3)],
    }
    if value["profiles"]["xy_p99_global"] != expected_global:
        raise ValueError("xy_p99_global must be the maxima over the fixed four-domain "
                         "reference, sharing particle and pose translation")
    return value


def resolve_normalization(spec: str | Mapping | None = None) -> dict:
    """Return a validated, self-contained, checksummed normalization snapshot.

    Accepts a profile name/legacy xy-p99 alias, a selector
    ``{'profile': ..., 'artifact': optional_path}``, or an already resolved snapshot.
    Snapshots load without reading any files, so checkpoint numerical behavior does
    not depend on today's bundle or historical absolute filesystem paths.
    """
    if isinstance(spec, Mapping):
        spec = _json_copy(spec)
        if "schema_version" in spec or "sha256" in spec:
            _keys(spec, _SNAPSHOT_KEYS, "normalization snapshot")
            supplied_hash = spec.pop("sha256")
            calculated_hash = hashlib.sha256(_canonical_json(spec).encode()).hexdigest()
            if supplied_hash != calculated_hash:
                raise ValueError("normalization snapshot checksum mismatch: corrupted or modified values")
            _validate_common(spec)
            profile = normalization_profile(spec["profile"])
            if profile != spec["profile"]:
                raise ValueError("normalization snapshots require canonical profile identities")
            canonical = _validate_scales(profile, spec["scales"])
            if canonical != spec["scales"]:
                raise ValueError("normalization snapshot scales must contain resolved float32 values")
            return {**spec, "sha256": supplied_hash}
        if not set(spec) <= {"profile", "artifact"}:
            raise ValueError("normalization selector accepts only profile and optional artifact")
        profile = normalization_profile(spec.get("profile"))
        path = spec.get("artifact", DEFAULT_ARTIFACT)
        if path is not DEFAULT_ARTIFACT and (not isinstance(path, str) or not path):
            raise ValueError("normalization artifact must be a nonempty JSON path")
    else:
        profile, path = normalization_profile(spec), DEFAULT_ARTIFACT
    artifact = load_normalization_artifact(path)
    snapshot = {key: artifact[key] for key in _ARTIFACT_KEYS - {"profiles"}}
    snapshot.update(profile=profile, scales=artifact["profiles"][profile])
    snapshot["sha256"] = hashlib.sha256(_canonical_json(snapshot).encode()).hexdigest()
    return snapshot


def domain_normalizers(spec, domain: str):
    """Construct float32 particle/position/pose normalizers for one domain.

    The pose result is None for passive domains (MPM, water). Bounds are symmetric, but
    the historical min/max formula (including its epsilon and caller-controlled clip) is
    unchanged. An optional domain needs an artifact that carries its entry under the
    per-domain profile; under the global profile it shares the reference scales.
    """
    if domain not in NORMALIZATION_DOMAINS + OPTIONAL_DOMAINS:
        raise ValueError(f"Unknown normalization domain {domain!r}; choose {NORMALIZATION_DOMAINS + OPTIONAL_DOMAINS}")
    resolved = resolve_normalization(spec)
    scales = resolved["scales"]
    if resolved["profile"] == "xy_p99_global":
        particle = scales["translation"]
        pose = particle + scales["rotation"]
    else:
        if domain not in scales["particle"]:
            raise ValueError(f"Normalization artifact {resolved['artifact_version']!r} has no {domain!r} entry; "
                             f"select an artifact that carries one (normalization.artifact=<path>)")
        particle = scales["particle"][domain]
        pose = scales["pose"].get(domain)

    def symmetric(values):
        maximum = torch.tensor(values, dtype=torch.float32)
        return NormalizationStats(-maximum, maximum)

    delta_stats = symmetric([particle[0], particle[0], particle[1]])
    position_stats = symmetric([resolved["position_half_range_m"]] * 3)
    pose_stats = None if domain in PASSIVE_DOMAINS else symmetric([pose[0], pose[0], pose[1],
                                                        pose[2], pose[2], pose[3]])
    return delta_stats, position_stats, pose_stats


def fixture_normalization_label(meta):
    """Read a fixture's stated coordinates without supplying a missing identity.

    Old receipts name a profile; new receipts embed a complete snapshot. Raw
    calibration fixtures explicitly state that no learned scales were loaded.
    """
    label = meta.get("stats_set")
    source = meta.get("normalization")
    if label == "raw":
        if (source is not None or meta.get("normalization_sha256") is not None or
                meta.get("requires_explicit_rebind_normalization") is not True or
                meta.get("normalization_loaded_during_selection") is not False):
            raise ValueError("Raw fixture requires an explicit, consistent no-normalization receipt")
        return "raw"
    if source is not None:
        if not isinstance(source, Mapping) or "sha256" not in source:
            raise ValueError("Fixture normalization must be a complete resolved snapshot")
        source = resolve_normalization(source)
        if label is not None and normalization_profile(label) != source["profile"]:
            raise ValueError("Fixture stats_set contradicts its normalization snapshot")
        if meta.get("normalization_sha256", source["sha256"]) != source["sha256"]:
            raise ValueError("Fixture normalization_sha256 contradicts its snapshot")
        return source["profile"]
    if label is None:
        raise ValueError("Fixture receipt lacks a normalization identity (stats_set or snapshot)")
    return normalization_profile(label)


def _fixture_bounds(batch, domain, expected=None, *, contract="recipe"):
    """Validate every actual normalizer and return JSON-native source bounds."""
    bounds = {}
    for i, (name, actual, size) in enumerate((
            ("delta", batch.delta_stats, 3), ("position", batch.position_stats, 3),
            ("pose", batch.pose_delta_stats, 0 if domain in PASSIVE_DOMAINS else 6))):
        if (actual is None) != (size == 0):
            raise ValueError(f"Saved fixture {name} normalizer presence differs from the {contract}")
        bounds[name] = None
        if actual is None:
            continue
        values = {}
        for key in ("min_val", "max_val"):
            value = getattr(actual, key, None)
            if (not isinstance(value, torch.Tensor) or value.dtype != torch.float32 or
                    value.shape != (size,) or not torch.isfinite(value).all()):
                raise ValueError(f"Saved fixture {name}.{key} differs from the {contract}: "
                                 "expected finite float32 bounds of the domain's exact shape")
            value = value.detach().cpu()
            if expected is not None and not torch.equal(value, getattr(expected[i], key).cpu()):
                raise ValueError(f"Saved fixture {name}.{key} differs from the {contract}")
            values[key] = value.tolist()
        if not (actual.max_val > actual.min_val).all():
            raise ValueError(f"Saved fixture {name} bounds must be strictly increasing")
        bounds[name] = values
    return bounds


def _validate_fixture_raw_contract(batch, domain):
    """Both strict reuse and rebinding require the same physical action fields."""
    if batch.prediction_target != "delta":
        raise ValueError("Only raw unified-delta fixtures are supported")
    for name in ("body_poses", "body_templates", "body_template_mask", "body_type",
                 "particle_body", "pose_deltas_gt", "pose_actions", "pose_action_mask"):
        value = getattr(batch, name)
        if ((domain in RIGID_DOMAINS + OPTIONAL_RIGID_DOMAINS and not isinstance(value, torch.Tensor)) or
                (domain in PASSIVE_DOMAINS and value is not None)):
            raise ValueError(f"Pinned fixture {name} violates the domain's native-pose contract")
    if batch.action_active is None:
        torch.testing.assert_close(batch.actions,
            batch.future_deltas * batch.action_mask[:, None, :, None], rtol=0, atol=0)
    elif domain not in ("mpm", "mpm_drag"):
        raise ValueError("Only MPM dragging has commanded particle actions")
    else:
        # Commands deliberately differ from achieved deltas; Batch validates
        # their temporal gates and zeros outside active patch transitions.
        batch._validate_shapes()
        if not torch.isfinite(batch.actions).all():
            raise ValueError("Non-finite commanded particle actions")
    if batch.pose_actions is not None:
        torch.testing.assert_close(batch.pose_actions,
            batch.pose_deltas_gt * batch.pose_action_mask[:, None, :, None], rtol=0, atol=0)


def bind_batch_normalization(batch, domain, snapshot):
    """Explicit ablation: replace normalizers, preserving every raw tensor.

    Unified particle and pose actions are physical quantities. Validate that
    contract before rebinding; no inverse affine/epsilon round trip is performed.
    Use prepare_fixture when loading saved fixtures to validate their receipts too.
    """
    delta, position, pose = domain_normalizers(snapshot, domain)
    _validate_fixture_raw_contract(batch, domain)
    return replace(batch, delta_stats=delta, position_stats=position, pose_delta_stats=pose)


def prepare_fixture(batch, meta, domain, snapshot, batch_size, *, rebind=False,
                    context_len=None, future_len=None):
    """Check saved coordinates/windows and record a deterministic prefix subset.

    Callers supply checkpoint/recipe C/F; rebinding never permits a window change.
    The original file digest remains untouched. A legacy profile-only receipt
    cannot establish an entire historical artifact: strict use checks its actual
    bounds against the recipe; explicit rebind records the original bounds instead.
    Complete source snapshots and raw placeholders are checked even when rebinding.
    """
    snapshot = resolve_normalization(snapshot)
    if domain not in NORMALIZATION_DOMAINS + OPTIONAL_DOMAINS:
        raise ValueError(f"Unknown normalization domain {domain!r}")
    identities = [getattr(batch, "domain", None), meta.get("domain"), meta.get("holdout_for")]
    identities = [value for value in identities if value is not None]
    if not identities or any(value != domain for value in identities):
        raise ValueError(f"Fixture domain identity {identities!r} does not establish requested {domain!r}")
    if type(batch_size) is not int or batch_size <= 0 or batch.batch_size < batch_size:
        raise ValueError("Fixture must contain the requested positive batch_size")
    for name, actual, requested, alias in (
            ("context_len", batch.context_len, context_len, "C"),
            ("future_len", batch.num_future_frames, future_len, "F")):
        if requested is not None and (type(requested) is not int or requested <= 0 or actual != requested):
            raise ValueError(f"Saved fixture C/F {name}={actual} differs from the recipe ({requested})")
        for key in (name, alias):
            if key in meta and (type(meta[key]) is not int or meta[key] != actual):
                raise ValueError(f"Fixture receipt {key} disagrees with actual C/F")
    if "shape" in meta and list(meta["shape"]) != list(batch.positions.shape):
        raise ValueError("Fixture receipt shape disagrees with actual tensors")

    label = fixture_normalization_label(meta)
    expected = domain_normalizers(snapshot, domain)
    source = meta.get("normalization")
    if label == "raw":
        if not rebind:
            raise ValueError("Raw fixture requires --rebind_normalization explicitly")
        # Raw adapters retain the common position range, with unit motion bounds.
        def unit(size):
            maximum = torch.ones(size, dtype=torch.float32)
            return NormalizationStats(-maximum, maximum)
        original_bounds = _fixture_bounds(batch, domain,
            (unit(3), expected[1], None if domain in PASSIVE_DOMAINS else unit(6)), contract="raw placeholder contract")
    elif source is not None:
        original_bounds = _fixture_bounds(batch, domain, domain_normalizers(source, domain),
                                           contract="source snapshot")
    else:
        original_bounds = _fixture_bounds(batch, domain)
    if not rebind:
        if label != snapshot["profile"]:
            raise ValueError("Saved fixture has different coordinates; use --rebind_normalization explicitly")
        source_hash = source["sha256"] if source is not None else meta.get("normalization_sha256")
        if source_hash is not None and source_hash != snapshot["sha256"]:
            raise ValueError("Saved fixture normalization snapshot differs from the recipe; "
                             "use --rebind_normalization explicitly")
        _fixture_bounds(batch, domain, expected)

    original_size = batch.batch_size
    batch = batch.slice(batch_size)
    meta = {**meta, "original_batch_size": original_size,
            "selected_rows": list(range(batch_size)), "domain": domain,
            "source_normalizer_bounds": original_bounds,
            "effective_normalization": snapshot,
            "effective_normalization_sha256": snapshot["sha256"]}
    if rebind:
        batch = bind_batch_normalization(batch, domain, snapshot)
        meta.update(rebound_normalization_sha256=snapshot["sha256"],
                    rebound_profile=snapshot["profile"])
    else:
        _validate_fixture_raw_contract(batch, domain)
    return batch, meta


@dataclass
class NormalizationStats:
    """
    Unified normalization statistics for min-max scaling to [-1, 1].

    Uses a single set of stats for all particles (gripper and object).
    """
    min_val: torch.Tensor  # [3] for XYZ
    max_val: torch.Tensor  # [3] for XYZ

    @staticmethod
    def _check_kind(data, prefix=""):
        kind = str(data.get(prefix + "norm_kind", "linear"))
        if kind != "linear":
            raise ValueError(
                f"Retired normalization kind {kind!r}; use commit 55c386c "
                "to reproduce historical nonlinear-normalization experiments.")

    @property
    def scale(self) -> torch.Tensor:
        """max - min"""
        return self.max_val - self.min_val

    @classmethod
    def load_all(cls, path: str) -> tuple:
        """
        Load delta, position, and offset stats from a single .npz file.

        Returns:
            (delta_stats, position_stats_or_None, offset_stats_or_None)
        """
        data = np.load(path)
        cls._check_kind(data, "delta_")
        delta_stats = cls(
            min_val=torch.from_numpy(data["min_delta"]).float(),
            max_val=torch.from_numpy(data["max_delta"]).float(),
        )
        position_stats = None
        if "min_position" in data and "max_position" in data:
            position_stats = cls(
                min_val=torch.from_numpy(data["min_position"]).float(),
                max_val=torch.from_numpy(data["max_position"]).float(),
            )
        offset_stats = None
        if "min_offset" in data and "max_offset" in data:
            offset_stats = cls(
                min_val=torch.from_numpy(data["min_offset"]).float(),
                max_val=torch.from_numpy(data["max_offset"]).float(),
            )
        return delta_stats, position_stats, offset_stats

    def normalize(self, x: torch.Tensor, clip: bool = False) -> torch.Tensor:
        """
        Normalize to [-1, 1].

        Args:
            x: [..., 3] tensor
            clip: If True, clip output to [-1, 1]

        Returns:
            Normalized tensor with same shape
        """
        min_val = self.min_val.to(x.device)
        scale = self.scale.to(x.device)
        normalized = 2 * (x - min_val) / (scale + 1e-8) - 1
        if clip:
            normalized = torch.clamp(normalized, -1, 1)
        return normalized

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Denormalize from [-1, 1] back to original scale.

        Args:
            x: [..., 3] normalized tensor

        Returns:
            Denormalized tensor with same shape
        """
        min_val = self.min_val.to(x.device)
        scale = self.scale.to(x.device)
        return (x + 1) / 2 * scale + min_val

    def __repr__(self) -> str:
        return (
            f"NormalizationStats(\n"
            f"  min_val={self.min_val.tolist()},\n"
            f"  max_val={self.max_val.tolist()},\n"
            f"  scale={self.scale.tolist()}\n"
            f")"
        )
