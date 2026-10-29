"""Unified dataset v1 — adapters that present four corpora as canonical TrajectoryBatch.

Spec: docs/unified_dataset_spec.md (every constant below is measured, see
reports/unified/audit_report.json). Corpora: MPM dropping (passive), pickup x4 success,
cloth (materialized WDS, see scripts/unified/materialize_cloth_wds.py), pushing_actions.

Canonical contract recap:
  - positions are shifted by a FIXED per-domain translation so the support plane is z=0
    everywhere, then normalized by the shared metric scale S (position_stats = +/-S, which
    makes min-max normalization compute exactly p/S).
  - object_ids are OPAQUE scene-local grouping labels (stratified encoding only). All
    semantics live in mat_type (material) and action_mask (actuation) — never branch on ids.
  - actions[f] = achieved displacement INTO future frame f (pos[C+f] - pos[C+f-1]) on
    particles of actuated bodies, zero elsewhere. The explicit MPM drag adapter instead
    supplies commanded velocity × recorded duration (not achieved future displacement).
    Pushing derives actions ONLY from rod
    particle motion — its absolute_targets are future GT for the block too (label leak).
  - particle/pose deltas stay in physical units in the batch. The explicit normalization
    snapshot selects frozen per-domain or global coordinates when the model consumes them.
"""
from __future__ import annotations

import glob
import math
import os
from collections.abc import Mapping
from typing import Iterator, List, Optional, Sequence

import numpy as np
import torch

from data.batch import TrajectoryBatch
from data.dataset import (BaseGraspDataset,  # noqa: F401
                          MPMMultiObjectDataset, NormalizationStats, SkipSample)

# --------------------------------------------------------------------- canonical constants
UNIFIED_SCALE = 0.8  # metres; shared metric position scale S

CANONICAL_TRANSLATION = {           # fixed per-domain rigid shift (metres); NEVER per-scene
    "mpm":     (-0.75, -0.75, -0.10),
    "pickup":  (-0.45,  0.00, -0.10),
    "pushing": ( 0.00,  0.00,  0.00),
    "cloth":   (-0.65,  0.00, -0.80),
    # The RL push release is in the robot base frame like pickup, with the table top already at z = 0.
    "rl_push": (-0.45,  0.00,  0.00),
    # Water-3D positions are first rotated from the file's y-up frame to z-up,
    # (x, y, z) -> (x, -z, y); the shift then centres the unit box (1 box unit = 1 m)
    # in xy and puts its floor at z = 0 (data/water3d.py:to_model_frame).
    "water":   (-0.50,  0.50, -0.125),
    "mpm_fresh": (-0.75, -0.75, -0.10),
    "mpm_drag":  (-0.75, -0.75, -0.10),
    "folding":   (0.00, 0.00, 0.00),  # retained exports are already canonical
    "yikai":     (-0.45, 0.00, 0.00), # fixed robot frame; table height remains observed
    "dex":       (0.00, 0.00, 0.00),  # recorded world frame, no per-env recentering
    "mould":    (0.00, 0.00, 0.00),  # genesis moulding scene: +-0.10 m in xy around the origin, floor at z = 0
}

# mat_type vocabulary: 0..2 = MPM's raw material-type column (passed through verbatim),
# appended types for the other corpora.
MAT_MPM_BASE = 0        # mpm raw col-3 values {0,1,2} used as-is
MAT_RIGID = 3
MAT_GRIPPER = 4
MAT_CLOTH = 5
NUM_MAT_TYPES = 6       # the four-domain vocabulary (default model.num_mat_types)
MAT_WATER = 6           # opt-in water domain; needs model.num_mat_types >= 7
MAT_DOUGH = 7           # opt-in mould domain (elasto-plastic dough); needs model.num_mat_types >= 8
# Dough constants of the moulding collector (Genesis/examples/collect_deformable_mould_demos.py); the
# shards store rho only. Same values as the hybrid_unified tree, so the material means the same thing.
DOUGH_FILL_E, DOUGH_FILL_NU, DOUGH_FILL_RHO = 2.0e4, 0.30, 500.0

# Fill constants for corpora without recorded values (disambiguated by mat_type).
RIGID_FILL_E, RIGID_FILL_NU = 1.0e9, 0.30
CLOTH_FILL_E, CLOTH_FILL_NU, CLOTH_FILL_RHO = 1.0e5, 0.30, 300.0

DOMAIN_DT = {"mpm": 1.0 / 24, "pickup": 0.05, "cloth": 1.0 / 30, "pushing": None,
             "water": 8 * 0.005,      # every 8th Water-3D file frame (metadata dt 0.005 s): 25 fps
             "rl_push": 0.05, "mpm_fresh": .0417, "mpm_drag": .0417,
             "folding": 1 / 30, "yikai": .05, "dex": .05,
             "mould": 0.05}           # collector step 2 ms, the converter keeps every 25th frame


def unified_position_stats() -> NormalizationStats:
    """min=-S / max=+S on all axes => normalize() computes exactly p/S."""
    return NormalizationStats(
        min_val=torch.full((3,), -UNIFIED_SCALE),
        max_val=torch.full((3,), UNIFIED_SCALE),
    )


def _shift(domain: str) -> torch.Tensor:
    return torch.tensor(CANONICAL_TRANSLATION[domain], dtype=torch.float32)


def _identity_delta_stats() -> NormalizationStats:
    """Bootstrap stats (used ONLY by the stats-computation pass, never for training)."""
    return NormalizationStats(min_val=torch.full((3,), -1.0), max_val=torch.full((3,), 1.0))


def _validate_window(context_len, future_len):
    if isinstance(context_len, bool) or int(context_len) != context_len or context_len <= 0:
        raise ValueError("context_len must be a positive integer")
    if future_len is not None and (isinstance(future_len, bool) or int(future_len) != future_len
                                   or (future_len != -1 and future_len <= 0)):
        raise ValueError("future_len must be a positive integer, -1, or None")


def _split_shards(shards: List[str], split: str, val_fraction: float = 0.10) -> List[str]:
    """Deterministic last-10%-of-sorted split (the pickup v9 convention)."""
    n_val = max(1, int(round(len(shards) * val_fraction)))
    if split == "train":
        return shards[:-n_val]
    if split == "val":
        return shards[-n_val:]
    return shards


# Compatibility re-exports for evaluation helpers and historical tests.
from data.blocklists import (load_blocklists as _load_blocklists,
                             filter_blocklisted as _blocklist_filter,
                             canonical_shard as _canonical_shard,
                             manifest_path as _manifest_path)


def _to_t(x) -> torch.Tensor:
    return torch.from_numpy(x) if isinstance(x, np.ndarray) else x


def _rigid_sidecar_from_particles(positions: torch.Tensor, body_index: torch.Tensor,
                                  n_bodies: int, rigid_bodies=None) -> dict:
    """Per-body Kabsch poses vs frame-0-centered templates.

    positions [B, T, N, 3]; body_index [B, N] in 0..n_bodies-1. Returns the
    TrajectoryBatch sidecar fields (poses, padded templates + mask, body_type).
    rigid_bodies: iterable of body indices to fit (default all). Bodies outside it
    are deformable: identity pose, empty template, body_type 0.
    Assumes each body's particle set is identical across the batch dim per sample
    (true for domain-pure adapters: counts are fixed per corpus).
    """
    from utils.rigid_ops import kabsch_batched
    B, T, N, _ = positions.shape
    rigid = set(range(n_bodies)) if rigid_bodies is None else set(rigid_bodies)
    counts = [int((body_index[0] == k).sum()) if k in rigid else 0 for k in range(n_bodies)]
    P_max = max(counts)
    poses = torch.zeros(B, T, n_bodies, 4, 4, dtype=positions.dtype)
    poses[..., 0, 0] = poses[..., 1, 1] = poses[..., 2, 2] = poses[..., 3, 3] = 1.0
    tmpl = torch.zeros(B, n_bodies, P_max, 3, dtype=positions.dtype)
    tmask = torch.zeros(B, n_bodies, P_max, dtype=torch.bool)
    btype = torch.zeros(B, n_bodies, dtype=torch.long)
    for k in sorted(rigid):
        sel = body_index[0] == k
        pts = positions[:, :, sel]                       # [B, T, Pk, 3]
        centered = pts[:, 0] - pts[:, 0].mean(dim=1, keepdim=True)   # [B, Pk, 3]
        R, t = kabsch_batched(centered[:, None].expand_as(pts).reshape(B * T, -1, 3),
                              pts.reshape(B * T, -1, 3))
        poses[:, :, k, :3, :3] = R.reshape(B, T, 3, 3)
        poses[:, :, k, :3, 3] = t.reshape(B, T, 3)
        tmpl[:, k, :counts[k]] = centered
        tmask[:, k, :counts[k]] = True
        btype[:, k] = 1
    return {"body_poses": poses, "body_templates": tmpl, "body_template_mask": tmask,
            "body_type": btype}


def _augment_pose_channels(sidecar: dict, actuated: torch.Tensor, context_len: int,
                           pose_stats) -> dict:
    """Tier-2 v2 (docs/tier2_v2_design.md): first-class pose channels on the batch.

    pose_deltas_gt [B, F, K, 6]: raw world-frame [dt, so3_log(dR)] over the future
    window from the C-1 anchor — the SAME math the v1 aux used as GT, promoted to a
    training target. pose_actions: the deltas of ACTUATED rigid bodies (achieved ==
    commanded, Phase-0 audit), zeros for passive; pose_action_mask [B, K].
    """
    if not sidecar:
        return sidecar
    from utils.pose_native import layout_from, pose_sequence_deltas
    poses = sidecar["body_poses"].float()               # [B, T, K, 4, 4]
    B, _, K = poses.shape[:3]
    layout = None
    if sidecar.get("joint_type") is not None:          # jointed bodies carry their joint coordinate
        layout = layout_from(sidecar["body_parent"][0], sidecar["joint_type"][0],
                             sidecar["joint_axis"], sidecar["joint_pivot"], torch.arange(K, device=poses.device))
    d6 = pose_sequence_deltas(poses[:, context_len - 1:], layout)
    pb = sidecar["particle_body"]                       # [B, N]
    act_body = torch.zeros(B, K, dtype=torch.bool)
    for k in range(K):
        sel = pb[0] == k
        if bool(sel.any()):
            act_body[:, k] = actuated[:, sel].any(dim=1)
    act_body &= sidecar["body_type"] == 1
    sidecar["pose_deltas_gt"] = d6
    sidecar["pose_actions"] = d6 * act_body[:, None, :, None]
    sidecar["pose_action_mask"] = act_body
    sidecar["pose_delta_stats"] = pose_stats
    return sidecar


def _articulate_fingers(sidecar: dict, pairs, axes: torch.Tensor) -> dict:
    """Each finger becomes a prismatic child of its hand (the corpus grippers are parallel-jaw).

    pairs: [(hand, finger)] body indices; axes [B, len(pairs), 3]: each finger's opening direction
    in its hand's body frame. Finger poses are rebuilt from the hand pose and the finger's slide
    along its axis (relative rotation held at the window's first frame), so the pose channels
    reproduce body_poses exactly; the finger particles follow that line to the per-link Kabsch
    residual (<= 0.003 mm on pickup and cloth, 2026-09-11)."""
    from utils.pose_native import JOINT_FREE, JOINT_PRISMATIC
    poses = sidecar["body_poses"].clone()
    B, T, K = poses.shape[:3]
    parent = torch.full((B, K), -1, dtype=torch.long)
    jtype = torch.full((B, K), JOINT_FREE, dtype=torch.long)
    axis = torch.zeros(B, K, 3, dtype=poses.dtype)
    for i, (h, f) in enumerate(pairs):
        a = (axes[:, i] / axes[:, i].norm(dim=-1, keepdim=True)).to(poses.dtype)       # [B, 3]
        Rh, th = poses[:, :, h, :3, :3], poses[:, :, h, :3, 3]
        t_rel = (Rh.transpose(-1, -2) @ (poses[:, :, f, :3, 3] - th)[..., None])[..., 0]   # [B, T, 3]
        q = ((t_rel - t_rel[:, :1]) * a[:, None]).sum(-1, keepdim=True)                  # slide since frame 0
        R_rel = Rh[:, 0].transpose(-1, -2) @ poses[:, 0, f, :3, :3]
        poses[:, :, f, :3, :3] = Rh @ R_rel[:, None]
        poses[:, :, f, :3, 3] = th + (Rh @ (t_rel[:, :1] + q * a[:, None])[..., None])[..., 0]
        parent[:, f], jtype[:, f], axis[:, f] = h, JOINT_PRISMATIC, a
    sidecar.update(body_poses=poses, body_parent=parent, joint_type=jtype, joint_axis=axis,
                   joint_pivot=torch.zeros_like(axis))
    return sidecar


class _UnifiedWDSBase(BaseGraspDataset):
    """BaseGraspDataset with: multi-root shards, deterministic split, optional blocklist,
    canonical shift + unified position stats. Subclasses implement _process_sample."""

    domain: str = ""
    # Joint-coupled grippers (fingers as prismatic children of their hand) for articulated
    # models; set by the consumer from model.articulated_bodies, like emit_sidecar.
    articulated: bool = False

    def __init__(
        self,
        roots: Sequence[str],
        *,
        split: str = "train",
        context_len: int = 2,
        future_len: int = -1,
        training: bool = True,
        shuffle: bool = True,
        blocklist_paths: Optional[Sequence[str]] = None,
        val_fraction: float = 0.10,
        split_per_root: bool = True,
        normalization=None,
        raw: bool = False,
    ):
        # NOTE: deliberately NOT calling super().__init__ — it assumes a single root.
        _validate_window(context_len, future_len)
        self.roots = [r.rstrip("/") for r in roots]
        if not self.roots:
            raise ValueError("Unified dataset roots must not be empty")
        self.wds_location = self.roots[0]
        self.context_len = context_len
        self.future_len = future_len
        self.training = training
        self.shuffle = shuffle
        self.num_collate = 1
        self.prediction_target = "delta"
        self.raw = bool(raw)
        self.normalization = None
        self._pose_stats_cache = None
        if raw:
            self.delta_stats = _identity_delta_stats()
            self.position_stats = unified_position_stats()
        else:
            from data.normalization import resolve_normalization, domain_normalizers
            self.normalization = resolve_normalization(normalization)
            self.delta_stats, self.position_stats, self._pose_stats_cache = domain_normalizers(
                self.normalization, self.domain)

        shards: List[str] = []
        for r in self.roots:
            rs = sorted(dict.fromkeys(
                glob.glob(f"{r}/*.tar") + glob.glob(f"{r}/**/*.tar", recursive=True)))
            rs = [t for t in rs if "_quarantine" not in t]
            if not rs:
                raise ValueError(f"no shards under {r}")
            shards += _split_shards(rs, split, val_fraction) if split_per_root else rs
        if not split_per_root:
            shards = _split_shards(sorted(shards), split, val_fraction)
        self.shards = shards
        print(f"[Unified/{self.domain}] split={split}: {len(self.shards)} shards "
              f"across {len(self.roots)} root(s)")

        self.blocklist_paths = tuple(blocklist_paths or ())
        self.blocklist = _load_blocklists(self.blocklist_paths) if self.blocklist_paths else None
        if self.blocklist is not None:
            print(f"[Unified/{self.domain}] blocklist: {len(self.blocklist)} (shard,key) entries "
                  f"from {len(list(blocklist_paths))} file(s)")
        self._shift_t = _shift(self.domain)

    def _pose_stats(self):
        """Resolved pose scales, or identity placeholders for raw calibration."""
        if self._pose_stats_cache is None:
            if self.raw:
                self._pose_stats_cache = NormalizationStats(-torch.ones(6), torch.ones(6))
            else:
                from data.normalization import domain_normalizers
                _, _, self._pose_stats_cache = domain_normalizers(
                    self.normalization, self.domain)
        return self._pose_stats_cache

    def _create_pipeline(self):
        pipe = super()._create_pipeline()
        if self.blocklist is None:
            return pipe
        return _blocklist_filter(pipe, self.blocklist, self.roots)

    # shared helpers -------------------------------------------------------------
    def _window(self, positions: torch.Tensor):
        """Random sub-window (training) / prefix window (eval) to context+future frames."""
        T = positions.shape[1]
        self._window_starts = [0] * positions.shape[0]
        if self.future_len > 0:
            w = self.context_len + self.future_len
            if T > w:
                t0 = (int(torch.randint(0, T - w + 1, (1,)).item())
                      if self.training and not getattr(self, "first_window", False) else 0)
                positions = positions[:, t0:t0 + w]
                self._window_starts = [t0] * positions.shape[0]
        return positions

    def _achieved_actions(self, positions: torch.Tensor, actuated: torch.Tensor):
        """actions[f] = pos[C+f] - pos[C+f-1], masked to actuated particles.

        positions [B,T,N,3]; actuated [B,N] bool -> [B, F, N, 3]."""
        d = positions[:, 1:] - positions[:, :-1]              # [B, T-1, N, 3]
        actions = d[:, self.context_len - 1:]                 # [B, F, N, 3]
        return actions * actuated[:, None, :, None].to(actions.dtype)


class UnifiedPickupDataset(_UnifiedWDSBase):
    """Pickup x4 success -> canonical TrajectoryBatch.

    Raw obj_ids: 0=object, 1=gripper (kept verbatim as opaque labels). Actions = achieved
    gripper motion (verified == absolute_targets semantics, Phase-0). Materials real from
    sidecars (E=1e9); mat_type RIGID/GRIPPER. Train split additionally excludes the 142
    held-out assets + 299 phantom entries via blocklists.
    """
    domain = "pickup"
    emit_sidecar = False
    # Particle layout (verified on data): gripper FIRST as [palma_hand, leftfinger,
    # rightfinger] segments sized by link_local_npts[8:11], then the 512 object points.
    EE_LINK_ROWS = (8, 9, 10)
    # Panda finger joints slide along the hand link's +y (left) / -y (right), as the robot
    # description has it; on the corpus each finger origin sits at (0, +-q, 58.4 mm) in the hand
    # frame to 0.02 mm (2026-09-11). Opening = a positive joint change for both fingers.
    PANDA_FINGER_AXES = ((0.0, 1.0, 0.0), (0.0, -1.0, 0.0))

    def _finger_axes(self, sample: dict) -> torch.Tensor:
        """[B, 2, 3] left / right finger opening directions in the hand's body frame (world axes at
        the window's first frame): the recorded hand link rotation there applied to the joint axes."""
        from utils.pose_ops import quat_wxyz_to_matrix
        quat = _to_t(sample["link_quat.pyd"]).float()                    # [B, T, 14, 4] wxyz
        R = quat_wxyz_to_matrix(quat[:, self._window_starts[0], self.EE_LINK_ROWS[0]])
        return torch.einsum("bij,fj->bfi", R, torch.tensor(self.PANDA_FINGER_AXES))

    def _process_sample(self, sample: dict) -> TrajectoryBatch:
        positions = _to_t(sample["particle_pos.pyd"]).float()        # [B, T, N, 3]
        oid = _to_t(sample["obj_ids.pyd"]).long()                    # [B, N]
        mats = _to_t(sample["material_properties.pyd"]).float()      # [B, N, 4]
        valid = _to_t(sample["valid_particles.pyd"]).bool() if "valid_particles.pyd" in sample \
            else torch.ones(positions.shape[0], positions.shape[1], positions.shape[2], dtype=torch.bool)

        positions = self._window(positions + self._shift_t)
        actuated = oid == 1
        actions = self._achieved_actions(positions, actuated)
        mat_type = torch.where(actuated, MAT_GRIPPER, MAT_RIGID).long()

        sidecar = {}
        if self.emit_sidecar:
            # Bodies at link granularity, all via the same verified Kabsch path:
            # 0=object, 1=hand, 2=left finger, 3=right finger. Segment boundaries
            # from this sample's link_local_npts; a wrong boundary mixes points
            # across an articulating joint and the recon check blows up.
            npts = _to_t(sample["link_local_npts.pyd"]).long().reshape(-1)
            seg = [int(npts[r]) for r in self.EE_LINK_ROWS]
            B, _, N, _ = positions.shape
            n_grip = int((oid[0] == 1).sum())
            assert sum(seg) == n_grip, f"EE segments {seg} != gripper count {n_grip}"
            pb = torch.zeros(N, dtype=torch.long)
            pb[:seg[0]] = 1
            pb[seg[0]:seg[0] + seg[1]] = 2
            pb[seg[0] + seg[1]:n_grip] = 3
            particle_body = pb[None].expand(B, N)
            sidecar = _rigid_sidecar_from_particles(positions, particle_body, n_bodies=4)
            sidecar["particle_body"] = particle_body.clone()
            if self.articulated:
                sidecar = _articulate_fingers(sidecar, [(1, 2), (1, 3)], self._finger_axes(sample))
            sidecar = _augment_pose_channels(sidecar, actuated, self.context_len,
                                             self._pose_stats())

        return TrajectoryBatch(
            positions=positions, actions=actions, action_mask=actuated,
            object_ids=oid, valid_lengths=valid[:, 0].sum(1).long(),
            E=mats[..., 0], nu=mats[..., 1], rho=mats[..., 2], mat_type=mat_type,
            context_len=self.context_len, delta_stats=self.delta_stats,
            position_stats=self.position_stats,
            prediction_target=self.prediction_target, **sidecar,
        )


class UnifiedPushingDataset(_UnifiedWDSBase):
    """pushing_actions -> canonical TrajectoryBatch.

    ⚠ absolute_targets is NOT read at all: it is non-NaN for the passive block and equals
    the block's future GT (Phase-0) — deriving actions from it would leak labels. Actions
    come from ROD particle motion only (raw obj_ids: 0=block, 1=rod). Materials: only rho
    is recorded; E/nu filled with rigid constants, disambiguated by mat_type=RIGID.

    emit_sidecar=True (Tier-2): per-body Kabsch poses vs frame-0 templates for both
    rigid bodies. Off by default — Tier-1 batches stay lean.
    """
    domain = "pushing"
    emit_sidecar = False

    def _process_sample(self, sample: dict) -> TrajectoryBatch:
        positions = _to_t(sample["particle_pos.pyd"]).float()
        oid = _to_t(sample["obj_ids.pyd"]).long()
        mats = _to_t(sample["material_properties.pyd"]).float()
        if positions.dim() == 3:                                     # [T,N,3] -> [1,T,N,3]
            positions, oid, mats = positions[None], oid[None], mats[None]

        positions = self._window(positions + self._shift_t)
        B, T, N, _ = positions.shape
        actuated = oid == 1
        actions = self._achieved_actions(positions, actuated)
        assert bool((actions[~actuated[:, None, :].expand(B, T - self.context_len, N)] == 0).all()), \
            "non-actuated particles must carry zero action (pushing label-leak guard)"

        rho = torch.nan_to_num(mats[..., 2], nan=750.0)
        E = torch.full_like(rho, RIGID_FILL_E)
        nu = torch.full_like(rho, RIGID_FILL_NU)
        mat_type = torch.full_like(oid, MAT_RIGID)

        sidecar = {}
        if self.emit_sidecar:
            sidecar = _rigid_sidecar_from_particles(positions, oid, n_bodies=2)
            sidecar["particle_body"] = oid.clone()
            sidecar = _augment_pose_channels(sidecar, actuated, self.context_len,
                                             self._pose_stats())

        return TrajectoryBatch(
            positions=positions, actions=actions, action_mask=actuated,
            object_ids=oid, valid_lengths=torch.full((B,), N, dtype=torch.long),
            E=E, nu=nu, rho=rho, mat_type=mat_type,
            context_len=self.context_len, delta_stats=self.delta_stats,
            position_stats=self.position_stats,
            prediction_target=self.prediction_target, **sidecar,
        )


class UnifiedMouldDataset(_UnifiedWDSBase):
    """deformable_mould_new_6k_wds -> canonical TrajectoryBatch (opt-in domain `mould`).

    1,280 particles in every episode: raw obj_ids 0 = dough (1,024 surface particles of an elasto-plastic
    MPM blob), 1 = the rigid paddle (256 points, moved by x, y, z and yaw only). The shards carry no
    per-frame actions, so, as in pushing, actions are the ACTUATED body's own achieved motion. Materials:
    only rho is stored (500 dough / 1000 tool) -> the dough gets the collector's E / nu, the tool the
    rigid constants. mat_type: dough MAT_DOUGH, tool MAT_RIGID.

    emit_sidecar=True: a Kabsch pose for the paddle (body 1); the dough (body 0) is deformable (identity
    pose, no template). Ported from the hybrid_unified tree, minus its in-adapter yaw augmentation, which
    this tree's adapters do not do."""
    domain = "mould"
    emit_sidecar = False

    def _process_sample(self, sample: dict) -> TrajectoryBatch:
        positions = _to_t(sample["particle_pos.pyd"]).float()
        oid = _to_t(sample["obj_ids.pyd"]).long()
        mats = _to_t(sample["material_properties.pyd"]).float()
        if positions.dim() == 3:                                     # [T,N,3] -> [1,T,N,3]
            positions, oid, mats = positions[None], oid[None], mats[None]

        positions = self._window(positions + self._shift_t)
        B, T, N, _ = positions.shape
        actuated = oid == 1
        actions = self._achieved_actions(positions, actuated)

        rho = torch.nan_to_num(mats[..., 2], nan=DOUGH_FILL_RHO)
        E = torch.where(actuated, torch.full_like(rho, RIGID_FILL_E), torch.full_like(rho, DOUGH_FILL_E))
        nu = torch.where(actuated, torch.full_like(rho, RIGID_FILL_NU), torch.full_like(rho, DOUGH_FILL_NU))
        mat_type = torch.where(actuated, MAT_RIGID, MAT_DOUGH).long()

        sidecar = {}
        if self.emit_sidecar:
            sidecar = _rigid_sidecar_from_particles(positions, oid, n_bodies=2, rigid_bodies=[1])
            sidecar["particle_body"] = oid.clone()
            sidecar = _augment_pose_channels(sidecar, actuated, self.context_len, self._pose_stats())

        return TrajectoryBatch(
            positions=positions, actions=actions, action_mask=actuated,
            object_ids=oid, valid_lengths=torch.full((B,), N, dtype=torch.long),
            E=E, nu=nu, rho=rho, mat_type=mat_type,
            context_len=self.context_len, delta_stats=self.delta_stats,
            position_stats=self.position_stats,
            prediction_target=self.prediction_target, **sidecar,
        )


class UnifiedMPMDataset(MPMMultiObjectDataset):
    """MPM dropping -> canonical frame. The parent already emits mat_type from raw col 3
    ({0,1,2} = the verbatim MPM vocabulary slots); col 4 (yield) is dropped in v1 (sidecar
    concern). This wrapper only shifts positions and swaps in the unified position stats.
    Splits: train = webdataset_full49 (all); val = robert_mpm_v2_test_seen (held-out
    scenes, seen objects — preserves the specialist's exact train set)."""
    domain = "mpm"

    def __init__(self, wds_location: str, *, normalization=None, raw: bool = False,
                 blocklist_paths: Optional[Sequence[str]] = None, **kwargs):
        retired = {"stats_location", "stats_override", "stats_set"}.intersection(kwargs)
        if retired:
            raise TypeError(f"Retired unified options {sorted(retired)}; use normalization= with a profile, "
                            "artifact selector, or resolved snapshot")
        # The native reader retains an external-NPZ fallback; this canonical
        # adapter deliberately has only the self-contained resolver path.
        from data.normalization import resolve_normalization
        snapshot = None if raw else resolve_normalization(normalization)
        super().__init__(wds_location=wds_location, stats_location=None,
                         normalization=snapshot, raw=raw, **kwargs)
        self.position_stats = unified_position_stats()
        self._shift_t = _shift("mpm")
        # Episode blocklist (2026-09-03 corpus audit): raw samples are batches of 4
        # episodes, so blocked (shard, key, elem) rows are dropped from the sample and
        # the batch shrinks; nothing on disk changes. Same file format + matching as
        # the pickup blocklists (_load_blocklists / _blocklist_filter).
        self.blocklist_paths = tuple(blocklist_paths or ())
        self.blocklist = _load_blocklists(self.blocklist_paths) if self.blocklist_paths else None
        self.roots = [str(wds_location)]
        if self.blocklist is not None:
            print(f"[Unified/mpm] blocklist: {len(self.blocklist)} (shard,key) entries (absolute + root-relative "
                  f"forms of the same episodes) from {len(list(blocklist_paths))} file(s)")

    def _create_pipeline(self):
        pipe = super()._create_pipeline()
        if self.blocklist is not None:
            pipe.append(lambda src: _blocklist_filter(src, self.blocklist, self.roots))
        return pipe

    def _process_sample(self, sample: dict) -> TrajectoryBatch:
        # Parent draws its sub-window with torch.randint regardless of eval mode;
        # for val/eval, pin the draw to the episode key so the window is
        # deterministic per episode — independent of domain order, batch count,
        # or any RNG consumed earlier in the process.
        if not self.training:
            import zlib
            seed = zlib.crc32(str(sample.get("__key__", "")).encode()) & 0x7FFFFFFF
            with torch.random.fork_rng(devices=[]):
                # Only the CPU window draw belongs to this local seed. Seeding
                # all devices here would reset the evaluator's CUDA noise stream
                # while fork_rng(devices=[]) restores only CPU state.
                torch.random.default_generator.manual_seed(seed)
                b = super()._process_sample(sample)
        else:
            b = super()._process_sample(sample)
        b.positions = b.positions + self._shift_t.to(b.positions.dtype)
        b.position_stats = self.position_stats
        return b


class UnifiedClothDataset(_UnifiedWDSBase):
    """Materialized cloth WDS (scripts/unified/materialize_cloth_wds.py) -> TrajectoryBatch.

    The materializer already applied stride 2 and the canonical shift, padded cloth to a
    fixed N with valid masks, and split train/val by the 38 held-out garments — so this
    adapter is thin: window slice, EE-delta actions, material fills.
    obj_ids: 0=cloth, 1=left EE, 2=right EE (opaque labels).
    """
    domain = "cloth"
    emit_sidecar = False
    strip_padding = True      # drop the 14000-row tail padding (native ragged N)
    # episode -> ee_link_ids (GENESIS constant is defined below this class)
    SIDECAR_IDS = "/data/group_data/katefgroup/VLA/3DPCWM/unified_v1/cloth_sidecar.npz"
    # Both SIM1 grippers are one parallel-jaw geometry whose points come in the same order in every
    # episode, but the WDS records particles only: each finger's opening direction comes from
    # registering the hand's points to a reference hand (data/embodiments/sim1_gripper.npz, made
    # by scripts/dev/make_sim1_gripper_reference.py from a validation episode whose fingers move).
    GRIPPER_REFERENCE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "embodiments",
                                     "sim1_gripper.npz")

    def _finger_axes(self, sc: dict) -> torch.Tensor:
        """[B, 4, 3] opening directions of fingers 2, 3 (left gripper) and 5, 6 (right gripper) in
        their hand's body frame."""
        from utils.rigid_ops import kabsch_batched
        if not hasattr(self, "_gripper_ref"):
            z = np.load(self.GRIPPER_REFERENCE)
            self._gripper_ref = (torch.from_numpy(z["hand_points"]).float(),
                                 torch.from_numpy(z["finger_axes"]).float())
        ref_pts, ref_axes = self._gripper_ref
        axes = []
        for h in (1, 4):
            P = sc["body_templates"][:, h, :ref_pts.shape[0]].float()          # [B, n, 3] centred
            ref = ref_pts.expand_as(P)
            R, _ = kabsch_batched(ref, P)
            resid = float((torch.einsum("bij,bpj->bpi", R, ref) - P).norm(dim=-1).max())
            if resid > 1e-3:
                raise ValueError(f"cloth gripper body {h}: hand points do not match the reference "
                                 f"gripper ({resid * 1e3:.2f} mm)")
            axes.append(torch.einsum("bij,fj->bfi", R, ref_axes))
        return torch.cat(axes, dim=1)
    # Motion-aware window sampling: t0 ~ p(t0) ∝ (mean cloth motion in the window's
    # FUTURE frames + floor)^alpha. Cloth motion is spatially and temporally sparse
    # (79% of particle-frames < 0.5 mm); uniform windows spend most steps on
    # near-static targets. alpha=0 recovers uniform. Training only — eval windows
    # stay the deterministic prefix.
    motion_alpha = 1.0
    motion_floor_m = 2e-5     # 0.02 mm/frame — keeps every window reachable

    def _motion_window(self, positions, oid, valid):
        T = positions.shape[1]
        self._window_starts = [0] * positions.shape[0]
        w = self.context_len + self.future_len
        if self.future_len <= 0 or T <= w:
            return positions
        if not self.training or self.motion_alpha <= 0 or getattr(self, "first_window", False):
            return self._window(positions)
        cloth = (oid[0] == 0) & valid[0]
        m = (positions[0, 1:, cloth] - positions[0, :-1, cloth]).norm(dim=-1).mean(1)  # [T-1]
        # score[t0] = mean motion over the window's future deltas [t0+C-1, t0+w-1)
        cs = torch.cat([torch.zeros(1), m.cumsum(0)])
        F = self.future_len
        starts = torch.arange(0, T - w + 1)
        score = (cs[starts + w - 1] - cs[starts + self.context_len - 1]) / F
        p = (score + self.motion_floor_m) ** self.motion_alpha
        t0 = int(starts[torch.multinomial(p / p.sum(), 1)].item())
        self._window_starts = [t0] * positions.shape[0]
        return positions[:, t0:t0 + w]

    def _process_sample(self, sample: dict) -> TrajectoryBatch:
        positions = _to_t(sample["particle_pos.pyd"]).float()        # [T, N, 3] canonical
        oid = _to_t(sample["obj_ids.pyd"]).long()                    # [N]
        valid = _to_t(sample["valid_particles.pyd"]).bool()          # [N]
        positions, oid, valid = positions[None], oid[None], valid[None]

        # Strip the materializer's TAIL padding: the unet consumes ragged native counts,
        # and the base loader buckets batches by N, so same-N garments collate
        # together. Padding is a tail block (valid is a prefix mask, asserted).
        if self.strip_padding:
            n_valid = int(valid[0].sum())
            assert bool(valid[0, :n_valid].all()), "cloth valid mask is not a prefix"
            positions, oid, valid = positions[:, :, :n_valid], oid[:, :n_valid], valid[:, :n_valid]

        positions = self._motion_window(positions, oid, valid)       # shift already applied
        actuated = oid >= 1
        actions = self._achieved_actions(positions, actuated)
        # padding rows must not contribute deltas
        actions = actions * valid[:, None, :, None].to(actions.dtype)

        mat_type = torch.where(actuated, MAT_GRIPPER, MAT_CLOTH).long()
        E = torch.where(actuated, RIGID_FILL_E, CLOTH_FILL_E).float()
        nu = torch.full_like(E, CLOTH_FILL_NU)
        rho = torch.full_like(E, CLOTH_FILL_RHO)

        sidecar = {}
        if self.emit_sidecar:
            sidecar = self._cloth_sidecar(sample, positions, oid, valid)
            sidecar = _augment_pose_channels(sidecar, actuated & valid, self.context_len,
                                             self._pose_stats())

        return TrajectoryBatch(
            positions=positions, actions=actions, action_mask=actuated & valid,
            object_ids=oid, valid_lengths=valid.sum(1).long(),
            E=E, nu=nu, rho=rho, mat_type=mat_type,
            context_len=self.context_len, delta_stats=self.delta_stats,
            position_stats=self.position_stats,
            prediction_target=self.prediction_target, **sidecar,
        )

    def _cloth_sidecar(self, sample: dict, positions, oid, valid) -> dict:
        """K=7 bodies: 0=cloth (deformable) + per-link left (1-3) / right (4-6) grippers.

        Link membership from cloth_sidecar.npz (source ee_link_ids; contiguous
        hand/lf/rf segments, uniformly (200, 401, 399) on usable episodes). 10 of
        1,223 episodes carry a stale ids array that cannot map their 1000 EE points
        (v2_*_side/half collection artifact) — those are SKIPPED for the sidecar
        path only (Tier-1 never reads ids). Padding rows get body 0 but are invalid;
        consumers must AND particle_body with the valid mask.
        """
        if not hasattr(self, "_ee_ids"):
            z = np.load(self.SIDECAR_IDS)
            self._ee_ids = {k: z[k] for k in z.files}
        key = sample.get("__key__", "")
        ids = self._ee_ids.get(key)
        n_cloth = int(_to_t(sample["n_cloth.pyd"]).item())
        n_ee = 1000
        if ids is None or len(ids) != n_ee:
            raise SkipSample(f"sidecar-skip: unusable ee_link_ids for {key} "
                             f"(len {None if ids is None else len(ids)})")
        seg = np.bincount(ids, minlength=3).tolist()
        N = positions.shape[2]
        pb = torch.zeros(N, dtype=torch.long)
        off = n_cloth
        for j, s in enumerate(seg):                       # left gripper links -> 1..3
            pb[off:off + s] = 1 + j
            off += s
        for j, s in enumerate(seg):                       # right gripper links -> 4..6
            pb[off:off + s] = 4 + j
            off += s
        particle_body = pb[None].expand(positions.shape[0], N)
        sc = _rigid_sidecar_from_particles(positions, particle_body, n_bodies=7,
                                           rigid_bodies=range(1, 7))
        sc["particle_body"] = particle_body.clone()
        if self.articulated:
            sc = _articulate_fingers(sc, [(1, 2), (1, 3), (4, 5), (4, 6)], self._finger_axes(sc))
        return sc


# --------------------------------------------------------------------------- factory
GROUP = "/data/group_data/katefgroup/VLA/3DPCWM"
# The robertre/Genesis/3DPCWM_Data tree is a symlink farm onto GROUP (verified
# 2026-08-26: every unified root resolves there) — and /data/user_data/robertre is
# mode 700, so the personal path was unreadable to everyone else. Identical bytes,
# group-readable path.
GENESIS = GROUP

PICKUP_ROOTS = [f"{GENESIS}/robert_pickup_data_{v}_success"
                for v in ("place", "convex", "pick", "upright")]
# Asset holdout (142 of 946 pickup assets, seed 0): blocks the HELD-OUT assets, so it is
# applied to BOTH splits — train never sees them, and val stays asset-matched to train.
HOLDOUT_BLOCKLIST = "/data/group_data/katefgroup/VLA/3DPCWM/stats/unified_v1/blocklists/holdout_TRAIN_block.json"

PICKUP_BLOCKLISTS = [
    "./data/manifests/robert_pickup_data_4success_combined.json",
    HOLDOUT_BLOCKLIST,
]
# MPM episode blocklist (2026-09-03 corpus audit, reports/unified/data_audit_0903): the 64
# training episodes (0.97%) with an object that collapses below 30% of its initial radius
# of gyration -- all purely elastic material at E < 1e5 Pa with low Poisson ratio.
# TRAIN split only; the 7 collapsed eval episodes stay in the eval sets (comparability)
# and are listed in mpm_blocklist_eval.json.
MPM_BLOCKLISTS = [
    "./data/manifests/mpm_blocklist_train.json",
]
# Cloth episode blocklists (same audit; user decision 2026-09-03): garments whose front and
# back layers are COINCIDENT at frame 0 (z-extent < 0.75 particle spacings: the sewing
# pattern laid flat, not a stitched garment) -- 415 train + 57 val episodes (39% of the
# corpus) -- plus the mis-assembled garment v2_549 (3 train episodes). Applied to BOTH
# splits so val is not scored on a regime train never sees; earlier cloth val/eval numbers
# were on the unfiltered 144-episode set.
CLOTH_BLOCKLISTS = {
    "train": ["./data/manifests/cloth_blocklist_train.json"],
    "val": ["./data/manifests/cloth_blocklist_val.json"],
}
MPM_TRAIN_ROOT = f"{GENESIS}/robert_mpm_v2/webdataset_full49"
MPM_VAL_ROOT = f"{GENESIS}/robert_mpm_v2_test_seen"
PUSHING_ROOT = f"{GROUP}/pushing_data_actions"
MOULD_ROOT = f"{GROUP}/deformable_mould_new_6k_wds"     # {train,test}/ (304 / 32 shards); test is the val split
CLOTH_WDS_ROOT = f"{GENESIS}/unified_v1/cloth"          # materializer output {train,val}
# DeepMind Water-3D as materialized by scripts/unified/materialize_water3d.py: {train,valid,test}/
# (valid is the water val split; test stays held out).
WATER_ROOT = "/data/group_data/katefgroup/datasets/deepmind_lts/Water-3D/materialized_v1"
# Low-dim RL push release read by data/rl_push.py (one npz per object, points built at load time). Its
# validation objects and outlier episodes are manifests; both splits apply both files.
RL_PUSH_ROOT = f"{GROUP}/rl_push_data_09_14"
RL_PUSH_EXCLUSIONS = "./data/manifests/rl_push_exclusions.json"
RL_PUSH_VAL_ASSETS = "./data/manifests/rl_push_val_assets.json"

# Native horizons are explicit adapter defaults, not hidden stats-fitting windows.
DOMAIN_FUTURE_LEN = {"mould": 16, "mpm": 16, "pickup": 38, "cloth": 16, "pushing": 16, "water": 16, "rl_push": 16,
                     "mpm_fresh": 16, "mpm_drag": 16, "folding": 16, "yikai": 16, "dex": 16}
DOMAIN_ORDER = ("mpm", "pickup", "cloth", "pushing")     # the default domain set, canonical order
INDEXED_DOMAINS = ("mpm_fresh", "mpm_drag", "folding", "yikai", "dex")
CROSS_EPISODE_DOMAINS = ("mpm_fresh", "mpm_drag")   # equal point counts per scene, no articulation: batches may mix scenes
OPTIONAL_DOMAINS = ("water", "rl_push", "mould") + INDEXED_DOMAINS  # opt-in: never selected by default
SUPPORTED_DOMAINS = DOMAIN_ORDER + OPTIONAL_DOMAINS


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, (str, bytes, os.PathLike)):
        return [value]
    return list(value)


def _canonical_root(path, name):
    if not isinstance(path, (str, os.PathLike)) or not os.fspath(path):
        raise ValueError(f"{name} must be an absolute directory path, got {path!r}")
    path = os.fspath(path)
    if not os.path.isabs(path):
        raise ValueError(f"{name} must be an absolute directory path, got {path!r}")
    canonical = _canonical_shard(path).rstrip("/")
    if not os.path.isdir(canonical):
        raise ValueError(f"{name} does not exist: {path}")
    return canonical


def resolve_data_roots(spec):
    """Validate an explicit `data.roots` mapping before any shard is opened.

    {domain: {"train": path | [paths], "val": path | [paths] (optional),
              "exclusions": {"train": [files], "val": [files]} (optional)}}
    Domains that are absent keep the registered unified://v1 roots. An overridden
    domain applies ONLY the exclusion files it names: the registered v1 lists
    belong to the v1 corpus. Without "val", validation is the deterministic
    last-10%-of-sorted-shards split of the training roots (per root for pickup,
    pooled otherwise); with "val" both splits are read whole and the val roots
    must not overlap the train roots. MPM's native reader takes one directory per
    split, so MPM requires both roots. Paths are canonicalized (root aliases,
    realpath) and must exist.
    """
    if spec is None:
        return {}
    if not isinstance(spec, Mapping):
        raise ValueError("data.roots must be a mapping of domain -> {train, val, exclusions}")
    resolved = {}
    for domain, entry in spec.items():
        if domain not in DOMAIN_FUTURE_LEN:
            raise ValueError(f"data.roots names an unsupported domain: {domain!r} (supported: {list(SUPPORTED_DOMAINS)})")
        if not isinstance(entry, Mapping):
            raise ValueError(f"data.roots.{domain} must be a mapping with train, val and exclusions")
        unknown = set(entry) - {"train", "val", "exclusions", "task", "shift_m"}
        if unknown:
            raise ValueError(f"data.roots.{domain} has unknown keys {sorted(unknown)}; allowed: train, val, exclusions "
                             "(and shift_m for pickup)")
        task=entry.get('task','passive')
        if task not in ('passive','drag') or ('task' in entry and domain!='mpm'):
            raise ValueError('data.roots task is only supported for mpm: passive or drag')
        if not entry.get("train"):
            raise ValueError(f"data.roots.{domain}.train is required")
        train = [_canonical_root(p, f"data.roots.{domain}.train") for p in _as_list(entry["train"])]
        val = [_canonical_root(p, f"data.roots.{domain}.val") for p in _as_list(entry.get("val"))]
        if len(set(train)) != len(train) or len(set(val)) != len(val):
            raise ValueError(f"data.roots.{domain} lists the same root twice")
        if domain in ("mpm", "water") and (len(train) != 1 or len(val) != 1):
            raise ValueError(f"data.roots.{domain} requires both one train root and one val root "
                             "(its reader reads one directory per split)")
        if domain == "rl_push" and (len(train) != 1 or len(val) > 1):
            raise ValueError("data.roots.rl_push takes one release folder as train and at most one as val")
        if domain in INDEXED_DOMAINS and (len(train) != 1 or len(val) != 1):
            raise ValueError(f"data.roots.{domain} requires separate indexed train and val directories")
        for v in val:
            for t in train:
                if v == t or v.startswith(t + "/") or t.startswith(v + "/"):
                    raise ValueError(f"data.roots.{domain}: val root {v} overlaps train root {t}; "
                                     "validation must be held out")
        exclusions_spec = entry.get("exclusions") or {}
        if not isinstance(exclusions_spec, Mapping) or set(exclusions_spec) - {"train", "val"}:
            raise ValueError(f"data.roots.{domain}.exclusions must be a mapping with train and/or val lists")
        exclusions = {}
        for split in ("train", "val"):
            files = [os.fspath(f) for f in _as_list(exclusions_spec.get(split))]
            for f in files:
                if not _manifest_path(f).is_file():
                    raise ValueError(f"data.roots.{domain}.exclusions.{split} file not found: {f}")
            exclusions[split] = files
        resolved[domain] = {"train": train, "val": val, "exclusions": exclusions}
        if entry.get("shift_m") is not None:
            # A release folder may state its own canonical translation. Only pickup: its registered shift
            # (-0.45, 0, -0.10) leaves the table top 10 mm above z = 0, because the collection's table top is at
            # 0.110 m (pwm-data-v1/pickup, 2026-09-19). The value is part of the resolved roots, so a resumed run
            # and an evaluation read the same frame as the training run.
            if domain != "pickup":
                raise ValueError(f"data.roots.{domain}.shift_m is only supported for pickup")
            try:
                shift = [float(v) for v in entry["shift_m"]]
            except (TypeError, ValueError):
                shift = []
            if len(shift) != 3 or not all(math.isfinite(v) for v in shift) or any(isinstance(v, bool) for v in entry["shift_m"]):
                raise ValueError(f"data.roots.{domain}.shift_m must be three finite numbers (metres), got {entry['shift_m']!r}")
            resolved[domain]["shift_m"] = shift
        if domain in INDEXED_DOMAINS:
            if any(exclusions.values()):
                raise ValueError("Indexed datasets pin exclusions in their index; do not add row blocklists")
            from data.indexed_episodes import validate_index_pair
            validate_index_pair(train[0], val[0], domain)
        if task=='drag':
            if any(exclusions.values()):raise ValueError('Drag roots must use prefiltered exports, not row blocklists')
            resolved[domain]['task']='drag'
    return resolved


def exclusion_files(domain, split, spec=None):
    """The exclusion lists one domain split applies, as declared: an explicit release's own
    (`spec`, its resolve_data_roots entry), else the registered v1 lists."""
    if spec is not None:
        return list(spec["exclusions"].get(split) or ())
    registered = {"pickup": PICKUP_BLOCKLISTS if split == "train" else [HOLDOUT_BLOCKLIST],
                  "cloth": CLOTH_BLOCKLISTS.get(split),
                  "mpm": MPM_BLOCKLISTS if split == "train" else None,
                  "rl_push": [RL_PUSH_EXCLUSIONS, RL_PUSH_VAL_ASSETS]}
    return list(registered.get(domain) or ())


def _build_override_dataset(domain, split, spec, common):
    """Explicit roots resolved by resolve_data_roots; same adapters, same windows."""
    train, val = spec["train"], spec["val"]
    blocklists = exclusion_files(domain, split, spec) or None
    if domain in INDEXED_DOMAINS:
        from data.indexed_episodes import IndexedEpisodeDataset
        ds = IndexedEpisodeDataset(train[0] if split == "train" else val[0], domain=domain, **common)
    elif domain == "mpm":
        cls=UnifiedMPMDataset
        if spec.get('task')=='drag':
            from data.mpm_drag import UnifiedMPMDragDataset
            cls=UnifiedMPMDragDataset
        ds = cls(wds_location=train[0] if split == "train" else val[0],
                               **common, blocklist_paths=blocklists)
    elif domain == "water":
        if blocklists:
            raise ValueError("data.roots.water takes no exclusion lists")
        from data.water3d import UnifiedWaterDataset
        ds = UnifiedWaterDataset(train[0] if split == "train" else val[0], **common)
    elif domain == "rl_push":
        # One release folder split by a validation object list, or two folders read whole. Only the
        # manifests the entry names apply, so name the object list when there is no val folder.
        from data.rl_push import UnifiedRLPushDataset
        if val:
            ds = UnifiedRLPushDataset(train[0] if split == "train" else val[0], split="all",
                                      manifests=blocklists, **common)
        else:
            ds = UnifiedRLPushDataset(train[0], split=split, manifests=blocklists, **common)
    else:
        cls = {"pickup": UnifiedPickupDataset, "pushing": UnifiedPushingDataset,
               "cloth": UnifiedClothDataset, "mould": UnifiedMouldDataset}[domain]
        if val:
            ds = cls(train if split == "train" else val, split="all", **common,
                     blocklist_paths=blocklists)
        else:
            ds = cls(train, split=split, split_per_root=(domain != "pushing"), **common,
                     blocklist_paths=blocklists)
    if spec.get("shift_m") is not None:
        ds._shift_t = torch.tensor(spec["shift_m"], dtype=torch.float32)
    ds.source_split = split
    ds.roots_override = True
    return ds


def selected_domains(domains=None, weights=None):
    """Validate before I/O; canonical order makes rank-wise domain draws identical.

    Explicit zero weights disable domains for BOTH train and default validation.
    Weight keys must match the requested domain set, so misspellings cannot hide.
    """
    if domains is None:
        domains = DOMAIN_ORDER
    if isinstance(domains, (str, bytes)):
        raise ValueError("domains must be a sequence of domain names, not a string")
    domains = tuple(domains)
    if not domains or len(set(domains)) != len(domains) or set(domains) - set(SUPPORTED_DOMAINS):
        raise ValueError(f"Expected distinct supported domains, got {domains}")
    ordered = tuple(d for d in SUPPORTED_DOMAINS if d in domains)
    if weights is None:
        return ordered, {d: 1.0 for d in ordered}
    if set(weights) != set(domains):
        raise ValueError(f"Domain weight keys must match domains {domains}, got {list(weights)}")
    from numbers import Real
    if any(isinstance(weights[d], (bool, np.bool_)) or not isinstance(weights[d], Real)
           for d in ordered):
        raise ValueError("Domain weights must be real numbers, not booleans or strings")
    values = np.asarray([weights[d] for d in ordered], dtype=np.float64)
    if not np.isfinite(values).all() or (values < 0).any() or values.max() <= 0:
        raise ValueError("Domain weights must be finite, nonnegative, and not all zero")
    active = tuple(d for d, v in zip(ordered, values) if v > 0)
    return active, {d: float(weights[d]) for d in active}


def build_domain_dataset(domain: str, split: str = "train", *, training: bool = True,
                         future_len: Optional[int] = None,
                         context_len: int = 2,
                         normalization=None, raw: bool = False, roots=None):
    """One canonical adapter for training, evaluation, and stats-free calibration.

    `roots` is the output of resolve_data_roots (explicit per-domain locations);
    domains it does not name use the registered unified://v1 corpus below."""
    if domain not in DOMAIN_FUTURE_LEN or split not in ("train", "val"):
        raise ValueError(f"Unsupported unified domain/split: {domain}/{split}")
    _validate_window(context_len, future_len)
    fl = future_len if future_len is not None else DOMAIN_FUTURE_LEN[domain]
    common = dict(future_len=fl, context_len=context_len, training=training,
                  shuffle=training, normalization=normalization, raw=raw)
    if roots and domain in roots:
        return _build_override_dataset(domain, split, roots[domain], common)
    if domain in INDEXED_DOMAINS:
        raise ValueError(f"{domain} requires explicit frozen data.roots.{domain}.train and val indices")
    blocklists = exclusion_files(domain, split) or None
    if domain == "pickup":
        ds = UnifiedPickupDataset(PICKUP_ROOTS, split=split, **common, blocklist_paths=blocklists)
    elif domain == "pushing":
        ds = UnifiedPushingDataset(
            [PUSHING_ROOT], split=split, split_per_root=False, **common)
    elif domain == "cloth":
        ds = UnifiedClothDataset(
            [f"{CLOTH_WDS_ROOT}/{split}"], split="all", **common, blocklist_paths=blocklists)
    elif domain == "water":
        from data.water3d import UnifiedWaterDataset
        ds = UnifiedWaterDataset(f"{WATER_ROOT}/{'train' if split == 'train' else 'valid'}", **common)
    elif domain == "mould":
        ds = UnifiedMouldDataset([f"{MOULD_ROOT}/{'train' if split == 'train' else 'test'}"], split="all", **common)
    elif domain == "rl_push":
        from data.rl_push import UnifiedRLPushDataset
        ds = UnifiedRLPushDataset(RL_PUSH_ROOT, split=split, manifests=blocklists, **common)
    # Native MPM reader is shared with paper/evaluation populations. Only this
    # canonical wrapper applies the unified coordinate shift.
    else:
        ds = UnifiedMPMDataset(
            wds_location=MPM_TRAIN_ROOT if split == "train" else MPM_VAL_ROOT,
            **common, blocklist_paths=blocklists)
    ds.source_split = split
    return ds
def validate_window_sampling(window_motion=None, batch_across_episodes=None):
    """Training window policy of the indexed domains (recipe keys data.window_motion and
    data.batch_across_episodes): p(start) ∝ (mean future step of the predicted particles + floor_mm)^alpha
    for the domains that have a motion profile, and cross-episode batches for the equal-point-count MPM
    domains. Returns (weighting {domain: {alpha, floor_mm}}, [cross-episode domains])."""
    from data.recording_adapters import MOTION
    weighting = {}
    for domain, spec in (window_motion or {}).items():
        if domain not in MOTION:
            raise ValueError(f"data.window_motion.{domain}: no motion profile for this domain (supported: {sorted(MOTION)})")
        if not isinstance(spec, Mapping) or set(spec) != {"alpha", "floor_mm"}:
            raise ValueError(f"data.window_motion.{domain} must be a mapping with alpha and floor_mm")
        alpha, floor = spec["alpha"], spec["floor_mm"]
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in (alpha, floor)) \
                or not (np.isfinite(alpha) and alpha >= 0 and np.isfinite(floor) and floor > 0):
            raise ValueError(f"data.window_motion.{domain}: alpha must be a number >= 0 and floor_mm a number > 0")
        weighting[domain] = dict(alpha=float(alpha), floor_mm=float(floor))
    across = list(batch_across_episodes or ())
    if len(set(across)) != len(across) or set(across) - set(CROSS_EPISODE_DOMAINS):
        raise ValueError(f"data.batch_across_episodes accepts {list(CROSS_EPISODE_DOMAINS)}, each at most once")
    return weighting, across


class UnifiedTrainStream:
    """Infinite iterable over the weighted domain mix — drop-in for train.py's
    `for b in train_loader` loop. The domain-draw seed is FIXED (not rank-dependent):
    under DDP every rank then draws the same domain each step, so per-step batch
    shapes agree across ranks and no rank straggles on a cloth batch while another
    runs pushing. Data still differs per rank via wds split_by_node."""

    def __init__(self, weights=None, batch_size=4, num_workers=2, seed=0,
                 emit_sidecar=False, future_len=None, context_len=2,
                 bucket_max_items=0,
                 first_window=False, no_shard_split=False, shuffle=True,
                 domains=None, normalization=None, roots=None, articulated=False,
                 window_motion=None, batch_across_episodes=None):
        domains, weights = selected_domains(domains, weights)
        _validate_window(context_len, future_len)
        validate_window_sampling(window_motion, batch_across_episodes)
        self.kw = dict(domains=domains, weights=weights, batch_size=batch_size,
                       num_workers=num_workers, seed=seed, emit_sidecar=emit_sidecar,
                       articulated=articulated,
                       future_len=future_len, context_len=context_len,
                       bucket_max_items=bucket_max_items,
                       first_window=first_window, no_shard_split=no_shard_split, shuffle=shuffle,
                       normalization=normalization, roots=roots,
                       window_motion=window_motion, batch_across_episodes=batch_across_episodes)

    def __iter__(self):
        return iter(create_unified_dataloader(split="train", **self.kw))


class UnifiedValRoundRobin:
    """Deterministic strict domain rotation over selected FINITE val splits —
    mpm, pickup, cloth, pushing, mpm, ... — so validate(num_batches=4k) sees
    exactly k batches per domain (balanced val by construction). Fresh iterators
    per __iter__ so repeated validation passes are identical."""

    DOMAINS = DOMAIN_ORDER

    def __init__(self, batch_size: int = 4, num_workers: int = 0, emit_sidecar: bool = False,
                 future_len: Optional[int] = None, context_len: int = 2,
                 bucket_max_items: int = 0,
                 first_window: bool = False,
                 domains=None, normalization=None, roots=None, articulated=False):
        from torch.utils.data import DataLoader
        self.DOMAINS, _ = selected_domains(domains)
        _validate_window(context_len, future_len)
        self.loaders = {}
        for d in self.DOMAINS:
            ds = build_domain_dataset(d, split="val", training=False,
                                      future_len=future_len, context_len=context_len,
                                      normalization=normalization, roots=roots)
            ds.batch_size = batch_size
            ds.first_window = first_window
            ds.bucket_max_items = int(bucket_max_items or 0)
            ds.emit_sidecar = emit_sidecar          # no-op for mpm (pure deformable)
            ds.articulated = articulated
            if emit_sidecar and d != "mpm":
                ds._pose_stats()                    # fail-fast: missing stats would
                                                    # otherwise skip every sample silently
            # Every rank sees EVERY val shard: ranks then agree on domain counts, so
            # the per-domain accelerator.reduce calls in validate() align across
            # ranks (rank-split val shares could exhaust a domain on one rank only
            # -> reduce-count mismatch -> DDP deadlock). Val is 40 batches; the
            # redundant compute is negligible.
            ds.no_shard_split = True
            self.loaders[d] = DataLoader(ds, batch_size=None, num_workers=num_workers)

    def __iter__(self):
        its = {d: iter(l) for d, l in self.loaders.items()}
        alive = list(self.DOMAINS)
        while alive:
            for d in list(alive):
                try:
                    b = next(its[d])
                except StopIteration:
                    alive.remove(d)
                    continue
                b.domain = d
                yield b


def create_unified_dataloader(domains=None,
                              split: str = "train", weights: Optional[dict] = None,
                              batch_size: int = 4, num_workers: int = 2, seed: int = 0,
                              emit_sidecar: bool = False,
                              future_len: Optional[int] = None, context_len: int = 2,
                              bucket_max_items: int = 0,
                              first_window=False,
                              no_shard_split=False, shuffle=True, normalization=None,
                              roots=None, articulated=False,
                              window_motion=None, batch_across_episodes=None):
    """Weighted interleave of domain-pure loaders. Each yielded TrajectoryBatch carries
    `batch.domain` metadata and its own per-domain stats.
    future_len=None keeps the per-domain DOMAIN_FUTURE_LEN; an int makes it uniform.
    window_motion / batch_across_episodes apply to the training split only (validation
    windows and per-episode validation batches are unchanged)."""
    from torch.utils.data import DataLoader
    domains, weights = selected_domains(domains, weights)
    _validate_window(context_len, future_len)
    weighting, across = validate_window_sampling(window_motion, batch_across_episodes)
    if split not in ("train", "val"):
        raise ValueError(f"Unsupported unified split: {split}")
    loaders = {}
    for d in domains:
        ds = build_domain_dataset(d, split=split, training=(split == "train"),
                                  future_len=future_len, context_len=context_len,
                                  normalization=normalization, roots=roots)
        ds.batch_size = batch_size
        ds.first_window, ds.no_shard_split = first_window, no_shard_split
        ds.shuffle = bool(split == "train" and shuffle)
        ds.bucket_max_items = int(bucket_max_items or 0)
        ds.emit_sidecar = emit_sidecar              # no-op for mpm (pure deformable)
        ds.articulated = articulated
        if split == "train" and d in weighting:
            ds.motion_alpha, ds.motion_floor_m = weighting[d]["alpha"], weighting[d]["floor_mm"] / 1000.0
        if split == "train" and d in across:
            ds.cross_episode = True
        if emit_sidecar and d != "mpm" and getattr(ds, "_pose_stats_cache", None) is None:
            ds._pose_stats()                        # fail-fast (see round-robin note)
        loaders[d] = DataLoader(ds, batch_size=None, num_workers=num_workers,
                                persistent_workers=num_workers > 0)

    def _gen() -> Iterator[TrajectoryBatch]:
        rng = np.random.default_rng(seed)
        its = {d: iter(l) for d, l in loaders.items()}
        names = list(its)
        p = np.array([weights[d] for d in names], dtype=np.float64)
        p /= p.max()  # normalize without overflowing on large finite weights
        p /= p.sum()
        while names:
            d = str(rng.choice(names, p=p))
            try:
                b = next(its[d])
            except StopIteration:
                if split == "train":       # train shards are resampled; shouldn't happen
                    its[d] = iter(loaders[d])
                    try:
                        b = next(its[d])
                    except StopIteration as error:
                        raise ValueError(f"Empty training stream for {d}; check shards and exclusions") from error
                else:
                    i = names.index(d)
                    names.pop(i)
                    p = np.delete(p, i)
                    if p.size:
                        p /= p.sum()
                    continue
            b.domain = d
            yield b

    return _gen()
