"""Small MPM-specific generation contracts; no production source mutation.

SI units throughout. Command index k acts on state k -> k+1, before P2G.
The helpers reject invalid inputs instead of silently changing geometry or time.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def identity(case):
    """Caller supplies logical release/asset/case/config identity, never paths."""
    required = {"release_id", "case_index", "seed_namespace", "asset_ids", "config_sha256", "task"}
    if not required <= case.keys() or case["task"] not in {"passive", "drag"}:
        raise ValueError("Incomplete generation identity")
    if not isinstance(case["case_index"], int) or isinstance(case["case_index"], bool) or case["case_index"] < 0:
        raise ValueError("case_index must be a nonnegative integer")
    for key in ("release_id", "seed_namespace", "config_sha256"):
        if not isinstance(case[key], str) or not case[key]:
            raise ValueError(f"Invalid {key}")
    if not case["asset_ids"] or any(not isinstance(x, str) or not x for x in case["asset_ids"]):
        raise ValueError("Explicit nonempty asset identities required")
    if any(key in case for key in ("path", "root", "worker", "gpu", "stats")):
        raise ValueError("Storage/worker/normalization does not define an episode")
    digest = hashlib.sha256(canonical_json(case).encode()).hexdigest()
    return "mpm-" + digest, int(digest[:16], 16)


def finite_array(value, shape=None, name="array"):
    value = np.asarray(value)
    if not np.issubdtype(value.dtype, np.number) and value.dtype != np.bool_:
        raise ValueError(f"{name}: numeric array required")
    if shape is not None and value.shape != shape:
        raise ValueError(f"{name}: expected shape {shape}, got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"{name}: NaN/Inf")
    return value


def bounds_placement(template, xy, floor_m, clearance_m, grid_lim_m, stencil_margin_m=0.):
    """Translate the actual rotated/downsampled template; never rescale it."""
    p = finite_array(template, name="template").astype(np.float64)
    xy = finite_array(xy, (2,), "xy").astype(np.float64)
    scalars = finite_array([floor_m, clearance_m, grid_lim_m, stencil_margin_m])
    if p.ndim != 2 or p.shape[1] != 3 or not len(p):
        raise ValueError("Nonempty Nx3 template required")
    if clearance_m < 0 or stencil_margin_m < 0 or not 0 <= floor_m < grid_lim_m:
        raise ValueError("Invalid floor/clearance/grid")
    shift = np.r_[xy, floor_m + clearance_m - p[:, 2].min()]
    placed = p + shift
    if placed.min() < stencil_margin_m or placed.max() >= grid_lim_m - stencil_margin_m:
        raise ValueError("Placement violates domain/stencil bounds")
    return placed, shift


def object_scoped_mask(spatial_mask, object_ids, intended_object):
    mask = finite_array(spatial_mask, name="spatial_mask")
    ids = finite_array(object_ids, mask.shape, "object_ids")
    if mask.ndim != 1 or not np.isin(mask, [0, 1]).all():
        raise ValueError("Binary particle mask required")
    if not np.issubdtype(ids.dtype, np.integer) or not isinstance(intended_object, (int, np.integer)):
        raise ValueError("Integer object IDs required")
    result = (mask.astype(bool) & (ids == intended_object)).astype(np.int32)
    if not result.any():
        raise ValueError("Empty intended-object grab: reject, do not skip")
    return result


@dataclass(frozen=True)
class Clock:
    substep_dt_s: float
    substeps_per_transition: int
    transitions: int

    def __post_init__(self):
        if not np.isfinite(self.substep_dt_s) or self.substep_dt_s <= 0:
            raise ValueError("Positive finite substep dt required")
        for value in (self.substeps_per_transition, self.transitions):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError("Explicit positive integer step counts required")

    @property
    def state_time_s(self):
        return np.arange(self.transitions + 1, dtype=np.float64) * self.substeps_per_transition * self.substep_dt_s

    def active(self, window):
        if (len(window) != 2 or any(not isinstance(x, (int, np.integer)) for x in window)
                or not 0 <= window[0] <= window[1] <= self.transitions):
            raise ValueError("Invalid [start,stop) transition window")
        gate = np.zeros(self.transitions, dtype=bool)
        gate[window[0]:window[1]] = True
        return gate


def future_commands(command_velocity_m_s, selected, active, state_time_s, context):
    """Pure command displacement, not achieved future motion; supports active hold."""
    times = finite_array(state_time_s, name="state_time_s")
    selected = finite_array(selected, name="selected")
    commands = finite_array(command_velocity_m_s, (len(selected), 3), "command_velocity")
    active = finite_array(active, (len(times) - 1,), "active")
    if selected.ndim != 1 or not np.isin(selected, [0, 1]).all() or not np.isin(active, [0, 1]).all():
        raise ValueError("Binary selection/transition gates required")
    if not isinstance(context, int) or not 1 <= context < len(times) or not np.all(np.diff(times) > 0):
        raise ValueError("Invalid context/time grid")
    if np.any(commands[selected == 0] != 0):
        raise ValueError("Command outside selected particles")
    gate = active[context - 1:, None] * selected[None, :]
    delta = commands[None, :, :] * gate[:, :, None] * np.diff(times)[context - 1:, None, None]
    return delta.astype(np.float32), gate.astype(bool)


def require_passive_consumer_input(sample):
    """Staged fail-closed guard, not an active-MPM model adapter.

    A zero-velocity hold is still an active capability. All three historical
    command fields must agree they are passive; absence is allowed for old
    passive-only schemas, but a partial command schema is never guessed.
    """
    keys = {"grab_velocity.pyd", "is_grabbed.pyd", "drag_active.pyd"}
    present = keys.intersection(sample)
    if not present:
        return
    if present != keys:
        raise ValueError("Incomplete MPM command schema")
    for key in sorted(keys):
        value = finite_array(sample[key], name=key)
        if value.any():
            raise ValueError("Active MPM is unsupported by the passive consumer: " + key)


@dataclass(frozen=True)
class SettlingPolicy:
    """Diagnostic thresholds require empirical calibration before any release."""
    consecutive_transitions: int
    max_transitions: int
    max_object_rms_speed_m_s: float
    max_particle_speed_m_s: float
    max_particle_displacement_m: float

    def __post_init__(self):
        if (not isinstance(self.consecutive_transitions, int) or not isinstance(self.max_transitions, int)
                or not 1 <= self.consecutive_transitions <= self.max_transitions):
            raise ValueError("Invalid settling window/budget")
        thresholds = [self.max_object_rms_speed_m_s, self.max_particle_speed_m_s,
                      self.max_particle_displacement_m]
        if not np.isfinite(thresholds).all() or min(thresholds) <= 0:
            raise ValueError("Explicit positive finite settling thresholds required")


class SettlingMonitor:
    def __init__(self, object_ids, policy):
        self.ids = finite_array(object_ids, name="object_ids")
        if self.ids.ndim != 1 or not len(self.ids) or not np.issubdtype(self.ids.dtype, np.integer):
            raise ValueError("Integer object membership required")
        self.policy = policy
        self.history = []
        self.consecutive = 0

    def observe(self, previous_x, x, v):
        shape = (len(self.ids), 3)
        previous_x = finite_array(previous_x, shape, "previous_x")
        x, v = finite_array(x, shape, "x"), finite_array(v, shape, "v")
        speeds = np.linalg.norm(v, axis=-1)
        row = {"max_particle_speed_m_s": float(speeds.max()),
               "max_particle_displacement_m": float(np.linalg.norm(x - previous_x, axis=-1).max()),
               "max_object_rms_speed_m_s": float(max(np.sqrt(np.mean(speeds[self.ids == oid] ** 2))
                                                       for oid in np.unique(self.ids)))}
        passed = all(row[key] <= getattr(self.policy, key) for key in row)
        self.consecutive = self.consecutive + 1 if passed else 0
        self.history.append(dict(row, threshold_pass=passed))
        if self.consecutive >= self.policy.consecutive_transitions:
            return "settled"
        if len(self.history) >= self.policy.max_transitions:
            return "settling_timeout"
        return "pending"
