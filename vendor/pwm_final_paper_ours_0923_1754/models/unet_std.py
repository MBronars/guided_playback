"""Hierarchical UNet for the unified particle/pose contract.

The supported architecture and geometry contracts are documented in
docs/CODE_WALKTHROUGH.md. The historical module filename remains stable for
checkpoint compatibility; fresh configurations select architecture="unet".

What "std" means, one design per axis:

  spatial mixing    ONE flash self-attention over every token in the frame
                    (whole scene, cross-object), banded 3D RoPE from
                    current-frame positions, RMS qk-norm.
                    STRATIFIED L0: at level 0
                    the attention runs WITHIN each object — one exact-size
                    SDPA per object, no mask, no padding; levels >= 1 stay
                    scene-wide.
  block order       temporal attention -> spatial attention -> GEGLU FFN —
                    the trunk convention (JointTransformerBlock). Each sub-op
                    pre-LN + per-frame 3-mod adaLN, zero-init => exact
                    identity at construction.
  seam families     the ONE variant axis in this file (ported from the unet
                    branch 2026-08-26): pool_mode / unpool_mode select "attn"
                    (all-to-all attentive reduce down / FP trunk + gated
                    cross-attn read up — the default) or "pointnet"
                    (PointNet++ set abstraction down / FP unit up). Independent
                    knobs; the seams' adaLN grammar, gates and identity-at-init
                    are untouched by either.
  long skips        HunyuanDiT-style block-level skips: every non-bottleneck
                    encoder block's output re-enters at its mirrored decoder
                    block (enc block 0 <-> LAST dec block of the level) as
                    concat -> LayerNorm -> Linear(2w -> w).
  DF-on-context     the wrapper accepts per-frame context timesteps and
                    stashes context-token predictions for train_step's
                    context loss (train.supervise_context).

Token layout (docs/CODE_WALKTHROUGH.md). Objects are CONTIGUOUS
blocks of particles in canonical order 0..O-1; the per-object counts may
DIFFER (ragged). Two ladders:

  classic     l1_by_mat unset: every object has `points` particles and the
              per-object ladder is points / strides[0] / strides[1] / ...
              (the pnseams configs, unchanged).
  stratified  l1_by_mat set: L0 = whatever each object has; the per-object
              L1 count n1 is looked up by the object's MATERIAL category;
              strides[0] is the P -> n1 seam marker (not a stride) and
              strides[1:] divide n1 down the ladder. Levels >= 1 are ragged
              across objects but static per (category, level).

Both are one code path: the GraphProvider carries per-level per-object
segments (offset, length) and every per-object op loops over segments.

Organizing structure:

  GraphProvider   Stage 0 — ALL geometry for one chunk, built eagerly once
                  per forward: per-object FPS prefix anchor ladder (one
                  ragged FPS call), per-level positions/ids/segments, the
                  per-frame noisy fine->coarse interp table per seam, and
                  (pointnet pool only) the coarse->fine group table.
  SpatialAttention  the one spatial mixer (scene-wide, or per-segment at L0).
  Block           the ONE processing block, identical at every level.
  Pool / Unpool   the seams between levels — the only ops that change the
                  token set, and the only place widths change.
  UNet            stem -> encoder levels/pools -> bottleneck -> unpools /
                  decoder levels -> per-particle head.
  UNetWrapper     train.py's joint forward contract + the 10-channel
                  featurization (SHIFT convention, detached geometry).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import math

import torch
import torch.nn as nn
from diffusers.models.attention import Attention, FeedForward
from pytorch3d.ops import knn_points, sample_farthest_points

from models.components.attention import RotaryAttnProcessor
from models.components.embeddings import (RotaryPositionEncoding,
                                          RotaryPositionEncoding3D,
                                          SinusoidalTimeEmbed)

# Banded 3D RoPE calibration shared by every spatial attention in this file
# (production values for stats-normalized coordinates: f_min 0.5 / f_max 64
# rad/unit, 75% of head_dim rotated, rest identity pass-through).
_ROPE_FRACTION, _ROPE_F_MIN, _ROPE_F_MAX = 0.75, 0.5, 64.0


def _banded_rope(head_dim: int, f_min: float = _ROPE_F_MIN) -> RotaryPositionEncoding3D:
    # f_min sets the coarsest band = the code's UNIQUE RANGE (2*pi/f_min * r).
    # 0.5 -> 1.67m: 2x headroom over the CLEAN scene spread (max pair 0.80m,
    # measured 2026-08-05) — right for delta-objective arms. f0off arms need
    # 0.15 (5.6m): their high-t noisy addresses spread to p99 3.8m and 53% of
    # pairs ALIAS at 0.5 (attended-pair census, same date).
    return RotaryPositionEncoding3D(
        dim=head_dim, rotary_dim=int(round(head_dim * _ROPE_FRACTION)),
        f_min=f_min, f_max=_ROPE_F_MAX)


def _ladder(n1: int, sub_strides) -> list:
    """[n1, n1/s1, n1/s1/s2, ...] — the per-object counts at levels 1..L."""
    out, n = [n1], n1
    for s in sub_strides:
        n //= s
        out.append(n)
    return out


@dataclass
class UNetConfig:
    """The full config surface."""
    points: int                      # P, particles per object (classic ladder; 0 in stratified mode)
    strides: tuple                   # classic: per-object level sizes P, P/s0, ...; stratified: s0 is the P->n1 seam marker
    widths: tuple                    # width per level, len == len(strides) + 1
    enc_depths: tuple                # blocks per encoder level incl. bottleneck (len strides + 1)
    dec_depths: tuple                # blocks per decoder level (len strides)
    in_dim: int                      # raw per-particle channels (deltas | f0 offset | z | physics)
    num_materials: int = 8
    head_dim: int = 64               # per-head width (spatial & temporal heads = w // head_dim)
    expansion: float = 2.67          # GEGLU FFN ratio (iso-param vs GELU @ 4x)
    t_dim: int = 256                 # noise-level embedding width
    dropout: float = 0.0
    fps_random_start: bool = True    # resample the FPS ladder start each forward
    causal_context: bool = False     # True: temporal attention fully causal (flash
                                     # fast path); False: block-causal trunk
                                     # convention via explicit float mask.
    bidirectional_temporal: bool = False  # True: no temporal mask at all -- every
                                     # frame (context and future) attends to every
                                     # frame; per-frame noise levels carry the
                                     # ordering (DFoT convention). Exclusive with
                                     # causal_context.
    drop_offset_channel: bool = False  # delta path: drop the pos-pos0 feature channel -> in_dim 7
    rope_f_min: float = 0.5          # coarsest RoPE band (rad per r-unit); 0.15 for f0off arms
    rope_f_max: float = 64.0         # finest RoPE band (rad per r-unit); 64 = 13.1 mm at geo_r 133 mm (tube_joint reads it)
    ckpt_ingraph: bool = False       # checkpoint INSIDE Block.forward (dynamo HOP)
    ckpt_from_level: int = -1        # gradient-checkpoint enc/dec blocks at levels >= this; -1 = off
    ckpt_seams: bool = False         # ALSO checkpoint the seams (pool/unpool at levels >=
                                     # ckpt_from_level) and the stem/head. These run at
                                     # full particle resolution outside the blocks and
                                     # hold ~9 GiB on a 9.4k-particle cloth batch (B=4,
                                     # measured 2026-08-26: 31.1 -> 22.3 GiB, +2% step).
    serial_levels: int = 0           # PTv3-style curve-patch attention at levels l < this (classic only)
    serial_patch: int = 1024         # tokens per curve patch (serialized levels)
    dual_ffn: bool = False           # a GEGLU FFN after EACH attention stage
    pose_action_mode: str = 'skip'   # 'skip' (action vector) | 'clean' (given body unnoised, t = 0)
    token_adaln: bool = False        # per-OBJECT adaLN in the blocks: a zero-init projection of each
                                     # object's material + physics vector, summed into the block's
                                     # modulations on top of the per-frame noise level. Per object, not
                                     # per token, because both inputs are constant within an object
                                     # (GraphProvider already asserts it for the material) -- identical
                                     # values, with the projection run on O rows instead of N.
    register_from_level: int = -1    # learned register tokens at levels >= this (-1 = off)
    register_objs: int = 1           # registers = this many objects' worth of the level's per-object tokens
    register_count: int = 0          # register rows per set (per_level / tube_joint); 0 = register_objs x
                                     # the level's per-object count. per_block rows ignore it.
    # HOW register rows live, the classic core's counterpart of unet_tube_register_mode:
    #   per_block  the historical scratch rows — every block appends its OWN learned set and drops it
    #              again at exit, so nothing a register computes survives the block.
    #   per_level  one learned set per level, appended at the start of each RUN of that level's blocks
    #              (its encoder run, its decoder run), carried from block to block, and stripped where
    #              the run ends. Registers pass through the blocks, never through a seam.
    register_mode: str = "per_block"
    # -- seam families
    pool_mode: str = "attn"          # "attn" | "pointnet"
    unpool_mode: str = "attn"        # "attn" | "pointnet"
    pool_k: int = 16                 # pointnet pool: fine tokens grouped per anchor (per-frame kNN, own object)
    interp_k: int = 3                # unpool trunk: coarse anchors per fine particle (inverse-distance interp)
    # -- stratified L0
    l1_by_mat: dict = field(default_factory=dict)   # material id -> per-object L1 count
    l1_default: int = 0              # L1 count for materials absent from the table (0 = error)
    # -- rule-based L1 budget (opt-in alternative to the material table, 2026-08-28):
    # "pow2_quarter": n1 = clamp(pow2_round(n0 / 4), l1_rule_min, l1_rule_max) from the
    # object's OWN particle count n0 — no per-material constant to maintain. When a table
    # is ALSO given it wins for the materials it lists (the rule fills the rest).
    # "capped_quarter" additionally respects available particles and an optional
    # scene cap. It is a separate policy so historical checkpoints keep their ladder.
    l1_rule: str = ""                # "" | "pow2_quarter" | "capped_quarter"
    l1_rule_min: int = 32
    l1_rule_max: int = 1024
    l1_scene_budget: int = 0         # L1 PARTICLE tokens only; 0 = no scene cap
    l1_rigid_max: int = 0            # ceiling for wholly rigid objects; 0 = disabled
    # -- native pose tokens (docs/CODE_WALKTHROUGH.md):
    # Kr rigid-body pose TOKENS ride the stream at every level as trailing
    # rows [particles | pose]; seams pass them through a width-change linear.
    pose_native: bool = False
    articulated_bodies: bool = False   # joint description on pose tokens (TrajectoryBatch articulation)

    def __post_init__(self):
        assert len(self.widths) == len(self.strides) + 1
        assert self.pose_action_mode in ("skip", "clean"), \
            f"model.unet_pose_action_mode must be skip or clean, got {self.pose_action_mode!r}"
        assert len(self.enc_depths) == len(self.strides) + 1
        assert len(self.dec_depths) == len(self.strides)
        assert self.register_mode in ("per_block", "per_level"), self.register_mode
        assert self.pool_mode in ("attn", "pointnet")
        assert self.unpool_mode in ("attn", "pointnet")
        assert self.pool_k >= 1 and self.interp_k >= 1
        # Matched encoder/decoder depths at every non-bottleneck level: the
        # HunyuanDiT long skips pair blocks one-to-one (enc block i <-> dec
        # block D-1-i), so every decoder block carries a skip merge.
        assert tuple(self.enc_depths[:-1]) == tuple(self.dec_depths), (
            f"enc_depths[:-1] {self.enc_depths[:-1]} must equal dec_depths "
            f"{self.dec_depths} (one-to-one long-skip pairing)")
        self.l1_by_mat = {int(k): int(v) for k, v in dict(self.l1_by_mat).items()}
        assert self.l1_rule in ("", "pow2_quarter", "capped_quarter"), f"unknown l1_rule {self.l1_rule!r}"
        assert type(self.l1_scene_budget) is int and self.l1_scene_budget >= 0, (
            "l1_scene_budget must be a nonnegative integer (0 disables the cap)")
        assert not self.l1_scene_budget or self.l1_rule == "capped_quarter", (
            "l1_scene_budget requires l1_rule='capped_quarter'")
        assert type(self.l1_rigid_max) is int and self.l1_rigid_max >= 0, (
            "l1_rigid_max must be a nonnegative integer (0 disables the ceiling)")
        if self.l1_rigid_max:
            assert self.l1_rule == "capped_quarter", "l1_rigid_max requires l1_rule='capped_quarter'"
            assert self.l1_rigid_max <= self.l1_rule_max and self.l1_rigid_max % math.prod(self.strides[1:]) == 0, (
                "l1_rigid_max must not exceed l1_rule_max and must be divisible by the sub-stride product")
        if self.l1_rule == "capped_quarter":
            assert not self.l1_by_mat and self.l1_default == 0, (
                "capped_quarter requires an empty l1_by_mat and l1_default=0; "
                "material overrides would bypass the count-based policy")
        if self.l1_rule:
            lo, hi = self.l1_rule_min, self.l1_rule_max
            assert type(lo) is int and type(hi) is int and lo > 0 and hi >= lo and (lo & (lo - 1)) == 0 and (hi & (hi - 1)) == 0, (
                f"l1_rule bounds must be powers of two with min <= max (got {lo}, {hi})")
            sub = 1
            for s_ in self.strides[1:]:
                sub *= s_
            assert lo % sub == 0, (
                f"l1_rule_min {lo} must be divisible by the sub-stride product {sub} "
                f"(every power of two >= min then divides too)")
        if self.stratified:
            assert self.serial_levels == 0, "stratified L0: serialization unsupported"
            assert self.register_from_level != 0, (
                "stratified L0: register counts are static but the L0 size is "
                "dynamic — registers allowed at levels >= 1 only")
            for n1 in self._l1_candidates():
                assert n1 > 0, f"L1 count must be positive (got {n1})"
                n = n1
                for i, s in enumerate(self.strides[1:], start=1):
                    assert n % s == 0, (
                        f"L1 count {n1}: stride {s} at level {i} does not divide "
                        f"the running per-object count {n}")
                    n //= s
        else:
            assert self.points > 0, "classic ladder needs points > 0"
            n = self.points
            for i, s in enumerate(self.strides):
                assert n % s == 0, (
                    f"level {i}: stride {s} does not divide the running per-object "
                    f"count {n}. Membership pooling has no tail mechanism — pick "
                    f"strides that divide P exactly.")
                n //= s

    @property
    def stratified(self) -> bool:
        return bool(self.l1_by_mat) or self.l1_default > 0 or bool(self.l1_rule)

    def _l1_candidates(self):
        """Reference L1 counts (register sizing + divisibility checks). Table values
        (+ default) when a table is given; under the bare rule its upper bound."""
        c = set(self.l1_by_mat.values())
        if self.l1_default > 0:
            c.add(self.l1_default)
        if self.l1_rule and not c:
            c.add(self.l1_rule_max)
        return sorted(c)

    def l1_for_object(self, mat: int, n0: int, *, rigid=None) -> int:
        """Per-object L1 count from (material, own particle count). Table first
        (when it lists the material), else the rule, else l1_default."""
        if mat in self.l1_by_mat or not self.l1_rule:
            return self.l1_for(mat)
        if self.l1_rule == "capped_quarter":
            assert type(n0) is int and n0 > 0, "object particle count must be a positive integer"
        n = 2 ** int(round(math.log2(max(n0 / 4.0, 1.0))))
        n = int(min(self.l1_rule_max, max(self.l1_rule_min, n)))
        if self.l1_rule == "capped_quarter":
            n = min(n, self._l1_capacity(n0))
        if self.l1_rigid_max:
            assert type(rigid) is bool, "l1_rigid_max requires explicit object rigidity"
            if rigid:
                n = min(n, self.l1_rigid_max)
        return n

    def _l1_capacity(self, n0: int) -> int:
        """Available particles rounded down to a complete hierarchy; tiny objects
        keep their particles and use a one-token floor at deeper levels."""
        quantum = math.prod(self.strides[1:])
        return n0 if n0 < quantum else (n0 // quantum) * quantum

    def l1_for_scene(self, mats, counts, *, rigid=None) -> list:
        """Allocate L1 particles, preserving each object's feasible minimum.

        Under a binding scene cap, distribute the remaining budget in proportion
        to each object's requested extras. Round DOWN to complete hierarchies:
        this is an upper bound, not a demand to fill every last slot. The policy
        is deterministic and independent of object order, positions and noise.
        Pose tokens and learned registers are outside this budget.
        """
        assert len(mats) == len(counts) and counts, "need one material and count per object"
        if self.l1_rigid_max:
            assert rigid is not None and len(rigid) == len(counts), (
                "l1_rigid_max requires one rigidity flag per object")
        flags = rigid if rigid is not None else [None] * len(counts)
        assert len(flags) == len(counts), "need one rigidity flag per object"
        requested = [self.l1_for_object(m, n, rigid=r) for m, n, r in zip(mats, counts, flags)]
        budget = self.l1_scene_budget
        if not budget or sum(requested) <= budget:
            return requested
        # A rigid ceiling may be lower than the preferred minimum. It also
        # limits the feasible floor, so redistribution never exceeds the request.
        floors = [min(self.l1_rule_min, want) for want in requested]
        assert sum(floors) <= budget, (
            f"l1_scene_budget={budget} cannot preserve the feasible per-object "
            f"minimums {floors} (sum={sum(floors)}); increase the scene budget "
            "or lower l1_rule_min")
        remaining, extras = budget - sum(floors), sum(requested) - sum(floors)
        quantum = math.prod(self.strides[1:])
        # Objects below the hierarchy quantum are already at capacity (no extras).
        return [floor + quantum * ((want - floor) * remaining // (extras * quantum))
                for floor, want in zip(floors, requested)]

    def l1_for(self, mat: int) -> int:
        """Per-object L1 count for a material id (stratified mode)."""
        if mat in self.l1_by_mat:
            return self.l1_by_mat[mat]
        assert self.l1_default > 0, (
            f"material {mat} has no L1 count in l1_by_mat and l1_default is 0")
        return self.l1_default

    def ladder(self, n1: int) -> list:
        """Per-object counts at levels 1..L from an L1 count."""
        if self.l1_rule == "capped_quarter":
            out = [n1]
            for stride in self.strides[1:]:
                out.append(max(1, out[-1] // stride))
            return out
        return _ladder(n1, self.strides[1:])

    @property
    def level_sizes(self):
        """REFERENCE per-object token count at every level, fine -> coarse
        (incl. L0). Classic: the actual ladder. Stratified: L0 is batch-dynamic
        (-1 sentinel) and levels >= 1 follow the LARGEST L1 count in the table
        (used for register sizing)."""
        if self.stratified:
            return [-1] + self.ladder(max(self._l1_candidates()))
        return [self.points] + self.ladder(self.points // self.strides[0])


# ---------------------------------------------------------------------------
# gather helpers
# ---------------------------------------------------------------------------

def gather_tokens(x, idx):
    """[B,T,N,C] gathered along the token dim by [B,M] -> [B,T,M,C]."""
    B, T, _, C = x.shape
    return x.gather(2, idx[:, None, :, None].expand(B, T, idx.shape[1], C))


def gather_neighbors(x, idx):
    """[B,T,N,C] -> [B,T,M,k,C] by neighbor indices.

    idx [B,M,k]   : one static table shared by every frame
    idx [B,T,M,k] : a per-frame table (the noisy k-NN interp)
    """
    B, T, _, C = x.shape
    if idx.ndim == 3:
        M, k = idx.shape[1:]
        flat = idx.reshape(B, 1, M * k, 1).expand(B, T, M * k, C)
    else:
        M, k = idx.shape[2:]
        flat = idx.reshape(B, T, M * k, 1).expand(B, T, M * k, C)
    return x.gather(2, flat).reshape(B, T, M, k, C)


def _segments(sizes) -> list:
    """Per-object counts -> [(offset, length), ...]."""
    out, off = [], 0
    for n in sizes:
        out.append((off, int(n)))
        off += int(n)
    return out


def _heads(z, h, hd):
    """[B,T,n,w] -> [B*T, h, n, hd]."""
    B, T, n, _ = z.shape
    return z.reshape(B * T, n, h, hd).transpose(1, 2)


def _unheads(z, B, T, w):
    """[B*T, h, n, hd] -> [B,T,n,w]."""
    n = z.shape[2]
    return z.transpose(1, 2).reshape(B, T, n, w)


def _segment_attention(q, k, v, q_segs, kv_segs):
    """Cross-attention restricted to matching segments: query segment i
    attends ONLY key/value segment i. q [BT,h,Nq,hd], k/v [BT,h,Nk,hd];
    segs are [(offset, length)] lists in token order. One exact-size SDPA per
    segment (flash kernel, no mask, no padding); results concatenated in
    query order."""
    # Consecutive segments with identical (q_len, kv_len) are folded into one
    # batched SDPA (the classic equal-block case becomes a single call).
    BT, h, _, hd = q.shape
    outs, i, n = [], 0, len(q_segs)
    while i < n:
        (qo, ql), (ko, kl) = q_segs[i], kv_segs[i]
        j = i + 1
        while j < n and q_segs[j][1] == ql and kv_segs[j][1] == kl:
            j += 1
        m = j - i                                       # segments in this run
        if m == 1:
            outs.append(nn.functional.scaled_dot_product_attention(
                q[:, :, qo:qo + ql], k[:, :, ko:ko + kl], v[:, :, ko:ko + kl]))
        else:
            def fold(z, off, ln):                       # [BT,h,m*ln,hd] -> [BT*m,h,ln,hd]
                return (z[:, :, off:off + m * ln].reshape(BT, h, m, ln, hd)
                        .permute(0, 2, 1, 3, 4).reshape(BT * m, h, ln, hd))
            o = nn.functional.scaled_dot_product_attention(
                fold(q, qo, ql), fold(k, ko, kl), fold(v, ko, kl))
            outs.append(o.reshape(BT, m, h, ql, hd).permute(0, 2, 1, 3, 4)
                        .reshape(BT, h, m * ql, hd))
        i = j
    return torch.cat(outs, dim=2) if len(outs) > 1 else outs[0]


def _spread_bits3(v):
    """int64 with 10-bit values -> bits spread to every 3rd position (Morton)."""
    v = (v | (v << 16)) & 0x30000FF
    v = (v | (v << 8)) & 0x300F00F
    v = (v | (v << 4)) & 0x30C30C3
    v = (v | (v << 2)) & 0x9249249
    return v


def _morton3(u):
    """u [..., 3] in [0,1) -> 30-bit Z-order codes [...] (long)."""
    q = (u.clamp(0.0, 1.0 - 1e-6) * 1024).long()
    return (_spread_bits3(q[..., 0])
            | (_spread_bits3(q[..., 1]) << 1)
            | (_spread_bits3(q[..., 2]) << 2))


# ---------------------------------------------------------------------------
# Stage 0 — GraphProvider
# ---------------------------------------------------------------------------

def particle_rigidity(batch):
    """Known rigid membership from body metadata, independent of material/actions.

    Sidecar-free particle streams (MPM/water) have no known rigid members. Partial
    sidecars are errors. Allocation below caps only objects whose EVERY particle
    belongs to a rigid body; mixed objects retain their deformable capacity.
    """
    if batch.body_type is None and batch.particle_body is None:
        return torch.zeros_like(batch.object_ids, dtype=torch.bool)
    assert batch.body_type is not None and batch.particle_body is not None, (
        "rigid allocation requires both body_type and particle_body")
    assert batch.particle_body.shape == batch.object_ids.shape, "particle_body must match object_ids"
    assert batch.body_type.ndim == 2 and batch.body_type.shape[0] == batch.object_ids.shape[0], (
        "body_type must have shape [batch, bodies]")
    assert bool(((batch.body_type == 0) | (batch.body_type == 1)).all()), "unknown body_type"
    return batch.body_type.gather(1, batch.particle_body) == 1


class GraphProvider:
    """All geometry for one chunk, computed once per forward — the only place
    geometry is ever built. Carries exactly: the FPS anchor ladder, per-level
    positions/ids/segments, one noisy fine->coarse interp table per seam, and
    (pointnet pool only) one noisy coarse->fine group table per seam. No
    membership maps, no member tables.

    Layout invariants (asserted): objects are CONTIGUOUS blocks in canonical
    order 0..O-1, identical layout across the batch; per-object counts may
    differ. At every level, tokens stay grouped by object and each object's
    tokens are a PREFIX of its one FPS ordering — so the level-(l+1) anchors
    occupy the first n_{l+1,o} slots of object o's level-l segment
    (`anchor_slots`), and pooling never needs a gather to find its anchors.
    """

    def __init__(self, cfg: UNetConfig, pos, pos0, obj_id, mat_id=None, particle_rigid=None):
        """pos [B,T,N,3] noisy integrated per-frame; pos0 [B,N,3] clean frame-0;
        obj_id [B,N] long (canonical contiguous blocks); mat_id [B,N] long
        (stratified mode: material per particle, uniform within an object).
        particle_rigid [B,N] bool: known rigid membership when the ceiling is on."""
        B, T, N, _ = pos.shape
        dev = pos.device
        oid0 = obj_id[0]
        assert bool((obj_id == oid0[None]).all()), \
            "object layout must be identical across the batch (bucket batches by shape)"
        change = torch.nonzero(oid0[1:] != oid0[:-1]).flatten() + 1
        starts = [0] + change.tolist()
        ends = starts[1:] + [N]
        O = len(starts)
        assert oid0[starts].tolist() == list(range(O)), (
            "particles must be contiguous per-object blocks in canonical order "
            "0..O-1 (scene layout [obj0 x P0, obj1 x P1, ...])")
        P = [e - s for s, e in zip(starts, ends)]

        if cfg.stratified:
            assert mat_id is not None, "stratified L0 needs per-particle material ids"
            mats = []
            for s, e in zip(starts, ends):
                m = int(mat_id[0, s])
                assert bool((mat_id[0, s:e] == m).all()), \
                    f"material must be uniform within an object (object at {s}:{e})"
                mats.append(m)
            rigid = None
            if cfg.l1_rigid_max:
                assert particle_rigid is not None and particle_rigid.shape == obj_id.shape and particle_rigid.dtype == torch.bool, (
                    "l1_rigid_max requires particle_rigid with shape [B,N] and boolean dtype")
                flags = torch.stack([particle_rigid[:, s:e].all(dim=1) for s, e in zip(starts, ends)], dim=1)
                assert bool((flags == flags[:1]).all()), (
                    "object rigidity must match across the batch (bucket batches by rigidity)")
                rigid = flags[0].tolist()
            self.rigid_objects = rigid
            n1 = cfg.l1_for_scene(mats, P, rigid=rigid)
            for o in range(O):
                assert P[o] >= n1[o], (
                    f"object {o} (material {mats[o]}) has {P[o]} particles < its "
                    f"L1 count {n1[o]} (docs/CODE_WALKTHROUGH.md"
                    f"{'; l1_rule=' + cfg.l1_rule if cfg.l1_rule else ''})")
        else:
            assert all(p == cfg.points for p in P), (
                f"classic ladder: every object needs exactly {cfg.points} particles, "
                f"got {P}")
            n1 = [cfg.points // cfg.strides[0]] * O
        lad = [cfg.ladder(n) for n in n1]                   # per object, levels 1..L
        L = len(cfg.strides) + 1
        # sizes[l][o]: per-object token count at level l
        self.sizes = [list(P)] + [[lad[o][l - 1] for o in range(O)] for l in range(1, L)]
        self.segs = [_segments(sz) for sz in self.sizes]    # per level [(off, len)]
        self.cfg, self.B, self.T, self.O, self.N = cfg, B, T, O, N

        # -- one FPS ordering per object on clean frame-0 positions (ONE ragged
        # call: objects padded to P_max with `lengths`, per-object K = n1);
        # every coarser level is a prefix of it.
        Pmax = max(P)
        packed = pos0.new_zeros(B, O, Pmax, 3)
        for o, (s, e) in enumerate(zip(starts, ends)):
            packed[:, o, :P[o]] = pos0[:, s:e]
        lengths = torch.tensor(P, device=dev).repeat(B)               # [B*O]
        K = torch.tensor(n1, device=dev).repeat(B)                    # [B*O]
        _, order = sample_farthest_points(
            packed.reshape(B * O, Pmax, 3), lengths=lengths, K=K,
            random_start_point=cfg.fps_random_start)                 # [B*O, Kmax] (-1 padded)
        order = order.reshape(B, O, -1)

        # scene-level token ids per level (indices into the N particles).
        self.ids = [torch.arange(N, device=dev)[None].expand(B, N)]
        for l in range(1, L):
            parts = [order[:, o, :self.sizes[l][o]] + starts[o] for o in range(O)]
            self.ids.append(torch.cat(parts, dim=1))
        self.pos = [pos] + [gather_tokens(pos, self.ids[l]) for l in range(1, L)]
        self.pos0 = [pos0] + [pos0.gather(1, self.ids[l][..., None].expand(-1, -1, 3))
                              for l in range(1, L)]

        # -- anchor slots: position of level-(l+1) tokens inside level l's list,
        # [B, N_{l+1}]. Levels >= 1 are stored in FPS order, so the coarser set
        # occupies each object's segment PREFIX. Level 0 is the RAW particle
        # order, so the level-1 anchors sit at their particle ids.
        self.anchor_slots = []
        for l in range(L - 1):
            if l == 0:
                self.anchor_slots.append(self.ids[1])
            else:
                parts = [torch.arange(self.sizes[l + 1][o], device=dev) + off
                         for o, (off, _) in enumerate(self.segs[l])]
                self.anchor_slots.append(torch.cat(parts)[None].expand(B, -1))

        # -- unpool interp: per-frame same-object k-NN from each fine token to
        # its object's coarse anchors over the NOISY (SHIFT-convention)
        # positions. Indices only; consumers compute weights from current-
        # frame positions (house rule). [B,T,N_l,K] per seam.
        self.interp = [self._build_interp(l) for l in range(L - 1)]

        # -- pointnet pool grouping (cfg.pool_mode): built only when the
        # pointnet seam needs it — the attn pool reads nothing from here.
        self.groups = [self._build_groups(l) for l in range(L - 1)] \
            if cfg.pool_mode == "pointnet" else [None] * (L - 1)

        # -- serialization orders for the dense levels (classic only).
        self.serial = [self._build_serial(l) if l < cfg.serial_levels else None
                       for l in range(L)]

    def _build_serial(self, l):
        """Two (order, inverse) serialization pairs for level l, [B, n] long."""
        pos = self.pos0[l]                                             # [B, n, 3]
        mn = pos.amin(dim=1, keepdim=True)
        mx = pos.amax(dim=1, keepdim=True)
        u = (pos - mn) / (mx - mn + 1e-6)
        out = []
        for perm in ((0, 1, 2), (1, 2, 0)):
            order = _morton3(u[..., perm]).argsort(dim=1)              # [B, n]
            out.append((order, order.argsort(dim=1)))
        return out

    def N_l(self, l):
        return sum(self.sizes[l])

    def _knn_same_object(self, l_q, l_kv, k_max):
        """Per-frame k-NN from every token of level l_q to the tokens of the
        SAME object at level l_kv, over noisy positions. K is uniform across
        objects (clamped to the smallest kv segment) so the table is one
        rectangular [B, T, N_{l_q}, K] of scene-level indices."""
        B, T = self.B, self.T
        K = min(k_max, min(self.sizes[l_kv]))
        parts = []
        for (qo, ql), (ko, kl) in zip(self.segs[l_q], self.segs[l_kv]):
            q = self.pos[l_q][:, :, qo:qo + ql].reshape(B * T, ql, 3)
            kv = self.pos[l_kv][:, :, ko:ko + kl].reshape(B * T, kl, 3)
            _, idx, _ = knn_points(q, kv, K=K)                         # [B*T, ql, K]
            parts.append(idx.reshape(B, T, ql, K) + ko)
        return torch.cat(parts, dim=2)

    def _build_interp(self, l):
        """Fine level l -> coarse l+1 same-object interp_k-NN, [B,T,n_f,K]."""
        return self._knn_same_object(l, l + 1, self.cfg.interp_k)

    def _build_groups(self, l):
        """Coarse level l+1 ANCHOR -> the level-l tokens it aggregates
        (pool_k-NN over the noisy positions, same object). Groups OVERLAP and
        are not a partition. [B,T,n_c,K]."""
        return self._knn_same_object(l + 1, l, self.cfg.pool_k)


# ---------------------------------------------------------------------------
# the spatial mixer — scene-wide flash attention with banded 3D RoPE
# ---------------------------------------------------------------------------

class SpatialAttention(nn.Module):
    """ONE flash self-attention over every token in the frame (whole scene,
    cross-object) per block — or, when `segments` is given (stratified L0),
    one exact-size self-attention PER OBJECT. Geometry enters via banded 3D
    RoPE on q/k from CURRENT-frame positions. RMS qk-norm: without it, sharp
    overfits grow attention logits unboundedly -> bf16 spike-collapse."""

    def __init__(self, w: int, head_dim: int, f_min: float = _ROPE_F_MIN):
        super().__init__()
        assert w % head_dim == 0, f"width {w} not divisible by head_dim {head_dim}"
        self.h, self.hd = w // head_dim, head_dim
        self.rope = _banded_rope(head_dim, f_min=f_min)
        self.to_q = nn.Linear(w, w, bias=False)
        self.to_k = nn.Linear(w, w, bias=False)
        self.to_v = nn.Linear(w, w, bias=False)
        self.to_out = nn.Linear(w, w)
        self.q_norm = nn.RMSNorm(self.hd)
        self.k_norm = nn.RMSNorm(self.hd)

    def forward(self, x, pos_t, serial=None, n_reg=0, segments=None, groups=None):
        """serial: None => one flash attention over the whole scene.
        (order, inv, patch) => PTv3-style curve patches.
        segments: [(offset, length)] => attention WITHIN each segment only
        (stratified L0); incompatible with registers and serialization.
        groups: [LongTensor] => like segments but each group is an explicit
        (non-overlapping) token index set — stratified L0 with pose rows
        (object particles + the pose tokens of the bodies inside it).

        n_reg: trailing register tokens in x (beyond pos_t's tokens). They are
        spatially UNPOSITIONED: identity RoPE (cos=1, sin=0) leaves their q/k
        unrotated, so register affinities are pure content."""
        B, T, n, w = x.shape
        if segments is not None or groups is not None:
            assert n_reg == 0 and serial is None, \
                "within-object attention: no registers / serialization"
        if serial is not None:
            assert n_reg == 0, "registers unsupported with serialized attention"
            order, inv, patch = serial
            x = gather_tokens(x, order)
            pos_t = gather_tokens(pos_t, order)
            assert n % patch == 0, f"n={n} not divisible by serial_patch={patch}"
        cos, sin = self.rope(pos_t.reshape(B * T, n - n_reg, 3))  # [B*T, n_real, hd]
        cos, sin = cos.to(x.dtype), sin.to(x.dtype)
        if n_reg:
            cos = torch.cat([cos, torch.ones(B * T, n_reg, self.hd,
                                             device=x.device, dtype=x.dtype)], dim=1)
            sin = torch.cat([sin, torch.zeros(B * T, n_reg, self.hd,
                                              device=x.device, dtype=x.dtype)], dim=1)
        cos, sin = cos[:, None], sin[:, None]

        q = self.rope.apply_rotary(self.q_norm(_heads(self.to_q(x), self.h, self.hd)), cos, sin)
        k = self.rope.apply_rotary(self.k_norm(_heads(self.to_k(x), self.h, self.hd)), cos, sin)
        v = _heads(self.to_v(x), self.h, self.hd)
        if serial is not None:
            m = n // patch                        # patches per frame

            def to_patches(z):                    # [B*T, h, n, hd] -> [B*T*m, h, P, hd]
                return (z.reshape(B * T, self.h, m, patch, self.hd)
                        .permute(0, 2, 1, 3, 4)
                        .reshape(B * T * m, self.h, patch, self.hd))

            out = nn.functional.scaled_dot_product_attention(
                to_patches(q), to_patches(k), to_patches(v))
            out = (out.reshape(B * T, m, self.h, patch, self.hd)
                   .permute(0, 2, 1, 3, 4).reshape(B * T, self.h, n, self.hd))
        elif groups is not None:
            # allocate in SDPA's OUTPUT dtype (bf16 under autocast while q may
            # be fp32 after the RMS qk-norm) — index_put requires a match.
            out = None
            for g in groups:
                o = nn.functional.scaled_dot_product_attention(
                    q[:, :, g], k[:, :, g], v[:, :, g])
                if out is None:
                    out = q.new_empty(q.shape, dtype=o.dtype)
                out[:, :, g] = o
        elif segments is not None:
            out = _segment_attention(q, k, v, segments, segments)
        else:
            out = nn.functional.scaled_dot_product_attention(q, k, v)
        out = self.to_out(_unheads(out, B, T, w))
        if serial is not None:
            out = gather_tokens(out, inv)         # back to raw token order
        return out


# ---------------------------------------------------------------------------
# the Block — identical at every level, encoder and decoder
# ---------------------------------------------------------------------------

class Block(nn.Module):
    """temporal attention -> spatial attention -> GEGLU FFN — the trunk stage
    order (JointTransformerBlock). Each sub-op: pre-LN residual, per-frame
    3-mod adaLN (shift/scale/gate) from the frame's OWN noise-level embedding,
    adaLN Linear zero-init => the whole block is an exact identity at
    construction. Temporal attention is dot-product + RoPE on the frame index
    with a block-causal mask."""

    def __init__(self, cfg: UNetConfig, w: int, skip: bool = False,
                 n_registers: int = 0):
        super().__init__()
        # registers: per-block learned scratch tokens (trunk convention:
        # randn * 0.02, one object's worth at this block's level).
        self.registers = (nn.Parameter(torch.randn(n_registers, w) * 0.02)
                          if n_registers > 0 else None)
        # HunyuanDiT-style long-skip merge (decoder blocks only): the paired
        # encoder block's output re-enters here as concat -> LayerNorm ->
        # linear down to w. The merged result REPLACES the stream (faithful to
        # the reference; not residual, not zero-init).
        if skip:
            self.skip_norm = nn.LayerNorm(2 * w)
            self.skip_linear = nn.Linear(2 * w, w)
        else:
            self.skip_norm = self.skip_linear = None
        self.norm1 = nn.LayerNorm(w, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(w, elementwise_affine=False, eps=1e-6)
        self.norm3 = nn.LayerNorm(w, elementwise_affine=False, eps=1e-6)
        # dual_ffn: 4 sub-ops (temporal, FFN, spatial, FFN) => 12 adaLN mods
        # and an extra norm+FFN; single-FFN blocks keep the 9-mod layout.
        self.dual_ffn = cfg.dual_ffn
        n_mods = 12 if cfg.dual_ffn else 9
        if cfg.dual_ffn:
            self.norm1b = nn.LayerNorm(w, elementwise_affine=False, eps=1e-6)
            self.ffn1 = FeedForward(w, inner_dim=int(round(w * cfg.expansion)),
                                    dropout=cfg.dropout, activation_fn="geglu")
        self.adaln = nn.Linear(cfg.t_dim, n_mods * w)
        nn.init.zeros_(self.adaln.weight)
        nn.init.zeros_(self.adaln.bias)
        # cfg.token_adaln: the per-object term. No bias, so a token with no object (a register,
        # whose conditioning row is zero) contributes exactly nothing. Zero-init = identity at step 0.
        self.adaln_obj = None
        if cfg.token_adaln:
            self.adaln_obj = nn.Linear(cfg.t_dim, n_mods * w, bias=False)
            nn.init.zeros_(self.adaln_obj.weight)
        hd = min(cfg.head_dim, w)
        assert w % hd == 0, f"width {w} not divisible by head_dim {hd}"
        self.is_causal = cfg.causal_context
        self.mixer = SpatialAttention(w, hd, f_min=cfg.rope_f_min)
        self.t_attn = Attention(query_dim=w, heads=w // hd, dim_head=hd,
                                dropout=cfg.dropout, bias=False, qk_norm="rms_norm")
        self.rotary = RotaryPositionEncoding(dim=hd)
        self.processor = RotaryAttnProcessor()
        self.ffn = FeedForward(w, inner_dim=int(round(w * cfg.expansion)),
                               dropout=cfg.dropout, activation_fn="geglu")

    def forward(self, x, pos_t, t_emb, causal_mask, rope=None, skip=None,
                serial=None, segments=None, groups=None, n_carry=0, obj_cond=None, obj_index=None,
                obj_given=None, t_emb0=None):
        # IN-GRAPH activation checkpointing (cfg.ckpt_ingraph): the checkpoint
        # call lives INSIDE the compiled forward, so dynamo traces it as a
        # higher-order op and emits a COMPILED recompute subgraph.
        # `grad_ckpt` is set per block by UNet.__init__ from ckpt_from_level.
        if getattr(self, "grad_ckpt", False) and self.training \
                and torch.is_grad_enabled():
            return torch.utils.checkpoint.checkpoint(
                self._forward_impl, x, pos_t, t_emb, causal_mask, rope, skip,
                serial, segments, groups, n_carry, obj_cond, obj_index, obj_given, t_emb0,
                use_reentrant=False)
        return self._forward_impl(x, pos_t, t_emb, causal_mask, rope, skip,
                                  serial, segments, groups, n_carry, obj_cond, obj_index, obj_given, t_emb0)

    def _forward_impl(self, x, pos_t, t_emb, causal_mask, rope=None, skip=None,
                      serial=None, segments=None, groups=None, n_carry=0,
                      obj_cond=None, obj_index=None, obj_given=None, t_emb0=None):
        """Three gated-residual sub-ops on the token stream x [B,T,n,w]:

            1. TEMPORAL attention (within a particle, across frames)
            2. SPATIAL attention  (within a frame, across particles; within
               the particle's OBJECT when `segments` is given)
            3. GEGLU feed-forward

        pos_t [B,T,n,3]: current-frame particle positions (SHIFT convention).
        causal_mask [T,T] or None; rope: temporal cos/sin hoisted by
        UNet.forward. skip [B,T,n,w]: the paired encoder block's output (long
        skip, decoder skip-blocks only) — merged FIRST, before the sub-ops."""
        if self.skip_linear is not None:
            assert skip is not None, "skip-block called without its skip tensor"
            x = self.skip_linear(self.skip_norm(torch.cat([x, skip], dim=-1)))
        else:
            assert skip is None, "skip tensor passed to a non-skip block"
        # -- registers: append the block's learned scratch tokens (dropped at
        # exit). They ride the temporal attention as extra "particles" and the
        # spatial attention as unpositioned tokens (identity RoPE).
        # n_carry: trailing rows of the LEVEL's carried set (cfg.register_mode=per_level),
        # already in x. They are unpositioned like the block's own rows, but stay in the
        # stream when the block returns — UNet.forward strips them where the run ends.
        n_own = 0
        if self.registers is not None:
            n_own = self.registers.shape[0]
            x = torch.cat([x, self.registers[None, None].to(x.dtype)
                           .expand(x.shape[0], x.shape[1], -1, -1)], dim=2)
        n_reg = n_own + int(n_carry)
        B, T, n, w = x.shape
        n_mods = 12 if self.dual_ffn else 9
        mods = self.adaln(t_emb).reshape(B, T, 1, n_mods, w)
        if obj_index is not None:
            # obj_index [B, n'] the object slot of each token ([particles | pose rows]); registers were
            # appended above and have no object, so they take slot O -- never given, zero conditioning.
            O = (obj_cond if obj_cond is not None else obj_given).shape[1]
            idx = obj_index
            if n_reg:
                idx = torch.cat([idx, idx.new_full((B, n_reg), O)], dim=1)
            if obj_given is not None and t_emb0 is not None:
                # pose_action_mode=clean: a GIVEN object's tokens are modulated at t = 0 in every frame,
                # in place of their frame's level. The material term below still applies to them.
                gv = torch.cat([obj_given, obj_given.new_zeros(B, 1)], dim=1).gather(1, idx)   # [B, n]
                m0 = self.adaln(t_emb0).reshape(B, 1, 1, n_mods, w)
                mods = torch.where(gv[:, None, :, None, None], m0, mods)          # [B,T,n,M,w]
            if self.adaln_obj is not None and obj_cond is not None:
                # obj_cond [B, O, t_dim]: one row per object, constant within it; slot O is zeros,
                # which adaln_obj (no bias) maps to zero.
                cond = torch.cat([obj_cond, obj_cond.new_zeros(B, 1, obj_cond.shape[-1])], dim=1)
                om = self.adaln_obj(cond).reshape(B, 1, O + 1, n_mods, w)      # [B,1,O+1,M,w]
                mods = mods + om.gather(2, idx[:, None, :, None, None].expand(B, 1, n, n_mods, w))
        # flag off: mods stays [B,T,1,M,w] and the modulations broadcast exactly as before.
        mods = mods.unbind(dim=3)
        if self.dual_ffn:
            s1, c1, g1, s1b, c1b, g1b, s2, c2, g2, s3, c3, g3 = mods
        else:
            s1, c1, g1, s2, c2, g2, s3, c3, g3 = mods

        # -- 1. temporal attention: fold particles into the batch dim so each
        # particle attends over its own T frames ([B,T,n,w] -> [B*n,T,w]),
        # block-causal mask (context bidirectional, future causal) unless
        # is_causal; None under bidirectional_temporal (no mask, is_causal False).
        h = self.norm1(x) * (1 + c1) + s1
        h = h.permute(0, 2, 1, 3).reshape(B * n, T, w)
        if rope is None:
            _c, _s = self.rotary(torch.arange(T, device=x.device).float()[None])
            rope = (_c[0], _s[0])    # 2-D [T,hd]: apply_rotary_emb's broadcast branch
        att = self.processor(self.t_attn, h, attention_mask=causal_mask,
                             rotary_emb=rope, is_causal=self.is_causal)
        x = x + g1 * att.reshape(B, n, T, w).permute(0, 2, 1, 3)

        # -- 1b. (dual_ffn) feed-forward after the temporal stage.
        if self.dual_ffn:
            h = self.norm1b(x) * (1 + c1b) + s1b
            x = x + g1b * self.ffn1(h)

        # -- 2. spatial attention (whole scene, curve patches, or per object).
        h = self.norm2(x) * (1 + c2) + s2
        x = x + g2 * self.mixer(h, pos_t, serial, n_reg=n_reg, segments=segments,
                                groups=groups)

        # -- 3. feed-forward.
        h = self.norm3(x) * (1 + c3) + s3
        x = x + g3 * self.ffn(h)
        return x[:, :, :n - n_own] if n_own else x    # drop the block's OWN registers


# ---------------------------------------------------------------------------
# the seams — Pool and Unpool (the only ops that change the token set)
# ---------------------------------------------------------------------------

class Pool(nn.Module):
    """Per-object attentive reduce in the HOUSE attention grammar, lift-first
    form (pool_mode="attn"): each anchor cross-attends to EVERY token of its
    object (Perceiver-style data-positioned latents) — locality is learned
    via RoPE, not imposed by a membership table. pool_mode="pointnet": the
    PointNet++ set-abstraction read over the anchor's per-frame kNN group
    (group -> M1 -> SiLU -> per-channel max -> post).

    Seam rule (shared with Unpool): ONE rectangular linear at seam entry —
    `up: w_in -> w_out` on the whole fine stream — then all math at the
    destination width. The anchor trunk is the anchor rows of that lift.

    Identity-at-init: adaLN is zero-init, so the gate is zero and
    out = up(x)[anchors] — pooling begins as pure strided subsampling of the
    (linearly lifted) fabric. The residual base is the UNNORMED lift; norms
    touch scoring inputs only.
    """

    def __init__(self, cfg: UNetConfig, w_in: int, w_out: int):
        super().__init__()
        self.w_out = w_out
        self.mode = cfg.pool_mode
        self.up = nn.Linear(w_in, w_out)         # the seam's ONE rectangular linear
        hd = min(cfg.head_dim, w_out)
        assert w_out % hd == 0
        self.h, self.hd = w_out // hd, hd
        self.norm = nn.LayerNorm(w_out, elementwise_affine=False, eps=1e-6)
        # zero-init adaLN: [shift, scale, gate] for the read + [shift, scale,
        # gate] for the seam FFN, each w_out. Same shape in both families —
        # in pointnet mode the read's shift/scale modulate the M1 input.
        self.adaln = nn.Linear(cfg.t_dim, 6 * w_out)
        nn.init.zeros_(self.adaln.weight)
        nn.init.zeros_(self.adaln.bias)
        if self.mode == "pointnet":
            # M1 over [h_j ‖ h_j - h_anchor ‖ dp] -> SiLU -> per-channel max.
            # Stored as ONE Linear on the concatenation; forward splits the
            # weight to apply it factorized.
            self.m1 = nn.Linear(2 * w_out + 3, w_out)
            self.post = nn.Linear(w_out, w_out)
        else:
            self.to_q = nn.Linear(w_out, w_out, bias=False)
            self.to_k = nn.Linear(w_out, w_out, bias=False)
            self.to_v = nn.Linear(w_out, w_out, bias=False)
            self.to_out = nn.Linear(w_out, w_out)
            self.q_norm = nn.RMSNorm(self.hd)
            self.k_norm = nn.RMSNorm(self.hd)
            self.rope = _banded_rope(self.hd, f_min=cfg.rope_f_min)
        # seam FFN: the blocks' third sub-op, mirrored onto the seam.
        self.norm_f = nn.LayerNorm(w_out, elementwise_affine=False, eps=1e-6)
        self.ffn = FeedForward(w_out, inner_dim=int(round(w_out * cfg.expansion)),
                               dropout=cfg.dropout, activation_fn="geglu")

    def forward(self, x, pos_t, provider: GraphProvider, l: int, t_emb):
        """x [B,T,n_f,w_in] at level l -> [B,T,n_c,w_out] at level l+1.
        t_emb [B,T,t_dim]: per-frame noise embedding (adaLN conditioning)."""
        B, T, n_f, _ = x.shape
        w_out = self.w_out
        lifted = self.up(x)                                            # [B,T,n_f,w_out]
        slots = provider.anchor_slots[l]                               # [B, n_c]
        anchor = gather_tokens(lifted, slots)                          # the branched trunk
        a_pos = gather_tokens(pos_t, slots)

        # -- per-frame adaLN on the ONE scoring stream (trunk stays raw).
        shift, scale, gate, shift_f, scale_f, gate_f = (
            self.adaln(t_emb).reshape(B, T, 1, 6, w_out).unbind(dim=3))
        h = self.norm(lifted) * (1 + scale) + shift                    # [B,T,n_f,w_out]
        h_a = gather_tokens(h, slots)                                  # anchor rows of h

        if self.mode == "pointnet":
            out = anchor + gate * self._read_pointnet(h, h_a, pos_t, a_pos, provider, l)
        else:
            def rope_tables(pos):
                n = pos.shape[2]
                cos, sin = self.rope(pos.reshape(B * T, n, 3))
                return cos.to(x.dtype)[:, None], sin.to(x.dtype)[:, None]

            cos_f, sin_f = rope_tables(pos_t)                          # all fine tokens
            cos_a, sin_a = rope_tables(a_pos)                          # anchor positions
            # -- per-object dense cross-attention: anchors query ALL their
            # object's tokens. One exact-size SDPA per object.
            q = self.rope.apply_rotary(self.q_norm(_heads(self.to_q(h_a), self.h, self.hd)), cos_a, sin_a)
            k = self.rope.apply_rotary(self.k_norm(_heads(self.to_k(h), self.h, self.hd)), cos_f, sin_f)
            v = _heads(self.to_v(h), self.h, self.hd)
            read = _segment_attention(q, k, v, provider.segs[l + 1], provider.segs[l])
            out = anchor + gate * self.to_out(_unheads(read, B, T, w_out))

        # -- seam FFN (house grammar third sub-op; zero-init gate => the seam
        # stays exact identity-at-init).
        h_f = self.norm_f(out) * (1 + scale_f) + shift_f
        return out + gate_f * self.ffn(h_f)

    def _read_pointnet(self, h, h_a, pos_t, a_pos, provider: GraphProvider, l):
        """PointNet++ set-abstraction read: group -> M1 -> SiLU -> per-channel
        max -> post. h [B,T,n_f,w] is the normed+modulated fine stream, h_a its
        anchor rows; positions are current-frame. Returns [B,T,n_c,w].

        FACTORIZED M1. The layer is one Linear on [h_j ‖ h_j - h_a ‖ dp], i.e.
            M1 = (W_self + W_rel)·h_j - W_rel·h_a + W_pos·dp + b
        so the two w x w matmuls run at FINE and ANCHOR resolution and only
        the cheap 3 -> w offset term touches the grouped [B,T,n_c,K,w] tensor.
        Same function, same parameters, same gradients as gathering first.
        """
        grp = provider.groups[l]                                       # [B,T,n_c,K]
        assert grp is not None, "pool_mode='pointnet' needs provider.groups"
        w_self, w_rel, w_pos = self.m1.weight.split(
            [self.w_out, self.w_out, 3], dim=1)
        a_full = nn.functional.linear(h, w_self + w_rel, self.m1.bias)  # [B,T,n_f,w]
        a_anch = nn.functional.linear(h_a, w_rel)                      # [B,T,n_c,w]

        mem = gather_neighbors(a_full, grp)                            # [B,T,n_c,K,w]
        dp = gather_neighbors(pos_t, grp) - a_pos.unsqueeze(3)         # [B,T,n_c,K,3]
        e = mem - a_anch.unsqueeze(3) + nn.functional.linear(dp, w_pos)

        # THE REDUCE RUNS IN FP32, DELIBERATELY. Under bf16 autocast +
        # torch.compile the amax BACKWARD produces NaN (bisected 2026-08-10 on
        # the unet branch; job 9933294 died this way). Upcast ONLY from the
        # low-precision dtypes so fp64 tests stay fp64.
        # ... AND THROUGH max(dim).values, NOT amax (2026-08-28, torch 2.11+cu128 /
        # A100): the fp32 upcast alone no longer suffices — the compiled amax backward
        # (an equality-mask divided by the match count) still returned NaN from the
        # deepest Pool seam, poisoning every upstream parameter. max(dim) carries its
        # argmax indices, so its backward is a plain scatter. Same forward values.
        _rd = torch.float32 if e.dtype in (torch.bfloat16, torch.float16) else e.dtype
        red = nn.functional.silu(e.to(_rd)).max(dim=3).values          # [B,T,n_c,w]
        return self.post(red.to(mem.dtype))


class Unpool(nn.Module):
    """Trunk-residual unpool seam in the HOUSE attention grammar — the mirror
    of Pool's lift-first form. ONE rectangular linear at seam entry —
    `down: w_coarse -> w_fine` on the whole coarse stream.

    TRUNK. The residual base is a SMOOTH FIELD interpolation of the decoded
    hierarchy: inverse-distance weights over each fine particle's interp_k
    nearest same-object coarse anchors (PointNet++ feature propagation), with
    indices from the provider's per-frame NOISY k-NN and weights from
    current-frame distances.

    READ (unpool_mode="attn"). The encoder skip supplies the QUERIES — each
    fine particle interrogates its object's coarse tokens — pre-LN -> adaLN
    -> square q/k/v -> RMS qk-norm -> banded 3D RoPE -> SDPA over the
    object's coarse tokens -> to_out -> zero-init gate onto the trunk.
    unpool_mode="pointnet": the PointNet++ FP unit — interpolate, concat the
    skip, one gated pointwise MLP. No read, no queries, no RoPE.

    Identity-at-init: the gate is zero, so out is exactly the interpolated
    (linearly downed) coarse field."""

    def __init__(self, cfg: UNetConfig, w_coarse: int, w_fine: int):
        super().__init__()
        self.mode = cfg.unpool_mode
        self.down = nn.Linear(w_coarse, w_fine)  # the seam's ONE rectangular linear
        hd = min(cfg.head_dim, w_fine)
        assert w_fine % hd == 0
        self.h, self.hd = w_fine // hd, hd
        # norm_c = the trunk/coarse stream, norm_s = the encoder skip.
        self.norm_s = nn.LayerNorm(w_fine, elementwise_affine=False, eps=1e-6)
        self.norm_c = nn.LayerNorm(w_fine, elementwise_affine=False, eps=1e-6)
        if self.mode == "pointnet":
            # ONE sub-op: [shift_c, scale_c, shift_s, scale_s, gate]. The FP
            # MLP is the seam's only nonlinearity, so there is no separate
            # seam-FFN modulation triple here.
            self.adaln = nn.Linear(cfg.t_dim, 5 * w_fine)
            self.ffn = FeedForward(2 * w_fine, dim_out=w_fine,
                                   inner_dim=int(round(w_fine * cfg.expansion)),
                                   dropout=cfg.dropout, activation_fn="geglu")
        else:
            # zero-init adaLN: [shift_s, scale_s, shift_c, scale_c, gate] for
            # the read + [shift_f, scale_f, gate_f] for the seam FFN.
            self.adaln = nn.Linear(cfg.t_dim, 8 * w_fine)
            self.norm_f = nn.LayerNorm(w_fine, elementwise_affine=False, eps=1e-6)
            self.ffn = FeedForward(w_fine, inner_dim=int(round(w_fine * cfg.expansion)),
                                   dropout=cfg.dropout, activation_fn="geglu")
            self.to_q = nn.Linear(w_fine, w_fine, bias=False)
            self.to_k = nn.Linear(w_fine, w_fine, bias=False)
            self.to_v = nn.Linear(w_fine, w_fine, bias=False)
            self.to_out = nn.Linear(w_fine, w_fine)
            self.q_norm = nn.RMSNorm(self.hd)
            self.k_norm = nn.RMSNorm(self.hd)
            self.rope = _banded_rope(self.hd, f_min=cfg.rope_f_min)
        nn.init.zeros_(self.adaln.weight)
        nn.init.zeros_(self.adaln.bias)

    def forward(self, skip, coarse, provider: GraphProvider, l: int, t_emb):
        """skip [B,T,n_f,w_fine] (encoder stash — queries only), coarse
        [B,T,n_c,w_coarse], t_emb [B,T,t_dim] -> [B,T,n_f,w_fine]."""
        B, T, n_f, w_f = skip.shape
        coarse_f = self.down(coarse)                                   # [B,T,n_c,w_f]

        # -- trunk: inverse-distance interpolation over the per-frame noisy
        # k-NN (indices from the provider; weights from CURRENT-frame
        # distances — indices and weights speak one currency).
        idx = provider.interp[l]                                       # [B,T,n_f,K]
        nbr = gather_neighbors(coarse_f, idx)                          # [B,T,n_f,K,w_f]
        npos = gather_neighbors(provider.pos[l + 1], idx)              # [B,T,n_f,K,3]
        d = (provider.pos[l].unsqueeze(3) - npos).norm(dim=-1)         # [B,T,n_f,K]
        iw = 1.0 / (d + 1e-8)                                          # numerical floor
        iw = (iw / iw.sum(dim=-1, keepdim=True)).unsqueeze(-1)
        trunk = (iw * nbr).sum(dim=3)                                  # [B,T,n_f,w_f]

        if self.mode == "pointnet":
            # PointNet++ FP unit: interpolate (above) -> concat the encoder
            # skip -> one gated pointwise MLP.
            shift_c, scale_c, shift_s, scale_s, gate = \
                self.adaln(t_emb).reshape(B, T, 1, 5, w_f).unbind(dim=3)
            h = torch.cat([self.norm_c(trunk) * (1 + scale_c) + shift_c,
                           self.norm_s(skip) * (1 + scale_s) + shift_s], dim=-1)
            return trunk + gate * self.ffn(h)

        # -- per-frame adaLN on the SCORING inputs only (trunk stays raw).
        shift_s, scale_s, shift_c, scale_c, gate, shift_f, scale_f, gate_f = \
            self.adaln(t_emb).reshape(B, T, 1, 8, w_f).unbind(dim=3)
        q_src = self.norm_s(skip) * (1 + scale_s) + shift_s            # skip side (queries)
        k_src = self.norm_c(coarse_f) * (1 + scale_c) + shift_c        # coarse side (k/v)

        def rope_tables(pos):
            n = pos.shape[2]
            cos, sin = self.rope(pos.reshape(B * T, n, 3))
            return cos.to(skip.dtype)[:, None], sin.to(skip.dtype)[:, None]

        cos_fine, sin_fine = rope_tables(provider.pos[l])          # fine-particle positions
        cos_coarse, sin_coarse = rope_tables(provider.pos[l + 1])  # coarse-anchor positions

        q = self.rope.apply_rotary(self.q_norm(_heads(self.to_q(q_src), self.h, self.hd)), cos_fine, sin_fine)
        k = self.rope.apply_rotary(self.k_norm(_heads(self.to_k(k_src), self.h, self.hd)), cos_coarse, sin_coarse)
        v = _heads(self.to_v(k_src), self.h, self.hd)
        read = _segment_attention(q, k, v, provider.segs[l], provider.segs[l + 1])
        out = trunk + gate * self.to_out(_unheads(read, B, T, w_f))

        # -- seam FFN (house grammar third sub-op; zero-init gate => the seam
        # stays exact identity-at-init).
        h_f = self.norm_f(out) * (1 + scale_f) + shift_f
        return out + gate_f * self.ffn(h_f)


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------

class UNet(nn.Module):
    """stem -> [enc level: blocks, pool] x L -> bottleneck blocks ->
    [unpool, dec level: blocks] x L -> head. See module docstring."""

    def __init__(self, cfg: UNetConfig):
        super().__init__()
        self.cfg = cfg
        W, L = cfg.widths, len(cfg.strides)
        self.stem = nn.Sequential(nn.Linear(cfg.in_dim, W[0]), nn.SiLU(),
                                  nn.Linear(W[0], W[0]))
        self.mat_emb = nn.Embedding(cfg.num_materials, W[0])
        # cfg.token_adaln: one conditioning vector per object = material embedding + physics MLP.
        self.obj_mat_emb = self.obj_phys = None
        if cfg.token_adaln:
            self.obj_mat_emb = nn.Embedding(cfg.num_materials, cfg.t_dim)
            self.obj_phys = nn.Sequential(nn.Linear(3, cfg.t_dim), nn.SiLU(),
                                          nn.Linear(cfg.t_dim, cfg.t_dim))
        self.t_embed = SinusoidalTimeEmbed(cfg.t_dim)
        self.t_mlp = nn.Sequential(nn.Linear(cfg.t_dim, cfg.t_dim), nn.SiLU(),
                                   nn.Linear(cfg.t_dim, cfg.t_dim))
        self.enc = nn.ModuleList(
            nn.ModuleList(Block(cfg, W[l], n_registers=self._n_reg(cfg, l))
                for _ in range(cfg.enc_depths[l]))
            for l in range(L + 1))
        self.pools = nn.ModuleList(Pool(cfg, W[l], W[l + 1]) for l in range(L))
        self.unpools = nn.ModuleList(Unpool(cfg, W[l + 1], W[l]) for l in range(L))
        # Block-level long skips (HunyuanDiT style), paired one-to-one per
        # level and anchored at the OUTER ends: encoder block 0 <-> the LAST
        # decoder block, ... bottleneck blocks pair with nothing.
        self.dec = nn.ModuleList(
            nn.ModuleList(Block(cfg, W[l], skip=True,
                                n_registers=self._n_reg(cfg, l))
                          for _ in range(cfg.dec_depths[l]))
            for l in range(L))
        self.head = nn.Sequential(nn.LayerNorm(W[0]), nn.Linear(W[0], W[0]),
                                  nn.SiLU(), nn.Linear(W[0], 3))
        # cfg.register_mode=per_level: one learned set per level, re-born at the start of
        # each run of that level's blocks (encoder run, decoder run) and stripped at its end.
        self.n_level_reg = {l: self._n_level_reg(cfg, l) for l in range(L + 1)}
        if any(self.n_level_reg.values()):
            assert cfg.register_from_level >= 1, (
                "register_mode=per_level: registers are static rows, so they need a level "
                "with a static token count (>= 1; the stratified L0 is batch-dynamic)")
            self.reg_init = nn.ParameterDict(
                {str(l): nn.Parameter(torch.randn(n, W[l]) * 0.02)
                 for l, n in self.n_level_reg.items() if n > 0})
        if cfg.pose_native:
            # Pose tokens (Tier-2 v2, v9 pattern): embed(normalized noisy Δ6)
            # + template-geometry summary + token-type embedding; pose-space
            # action skip on actuated bodies, learned no_action token else.
            self.pose_token_in = nn.Sequential(nn.Linear(6, W[0]), nn.SiLU(),
                                               nn.Linear(W[0], W[0]))
            self.pose_geom_embed = nn.Sequential(nn.Linear(3, 64), nn.SiLU(),
                                                 nn.Linear(64, W[0]))
            self.pose_action_skip = self.pose_no_action_token = None
            if cfg.pose_action_mode != "clean":           # clean mode reads no action vector at all
                self.pose_action_skip = nn.Linear(6, W[0])
                self.pose_no_action_token = nn.Parameter(torch.randn(W[0]) * 0.02)
            self.pose_token_type = nn.Parameter(torch.randn(W[0]) * 0.02)
            if cfg.articulated_bodies:
                # each body's joint description (free / fixed / prismatic / revolute, axis and
                # pivot in the parent frame), so the model reads a gripper's structure
                self.pose_joint_type = nn.Parameter(torch.randn(4, W[0]) * 0.02)
                self.pose_joint_geom = nn.Sequential(nn.Linear(6, 64), nn.SiLU(), nn.Linear(64, W[0]))
            # seams: pose rows do not pool/unpool — one width-change linear
            # per seam, each direction.
            self.pose_up = nn.ModuleList(nn.Linear(W[l], W[l + 1]) for l in range(L))
            self.pose_down = nn.ModuleList(nn.Linear(W[l + 1], W[l]) for l in range(L))
            # readout: zero-init linear head -> [.., Kr, 6] (v-prediction)
            self.pose_norm_out = nn.LayerNorm(W[0])
            self.pose_v_head = nn.Linear(W[0], 6)
            nn.init.zeros_(self.pose_v_head.weight)
            nn.init.zeros_(self.pose_v_head.bias)
        # In-graph checkpointing: mark blocks at levels >= ckpt_from_level to
        # self-checkpoint inside their (compiled) forward.
        if cfg.ckpt_ingraph and cfg.ckpt_from_level >= 0:
            for l in range(cfg.ckpt_from_level, L + 1):
                for blk in self.enc[l]:
                    blk.grad_ckpt = True
                if l < L:
                    for blk in self.dec[l]:
                        blk.grad_ckpt = True

    @staticmethod
    def _n_reg(cfg, l):
        """A block's OWN register rows at level l (register_mode=per_block): `register_objs`
        objects' worth of the level's (reference) per-object token count, coarse levels only.
        per_level keeps its rows in the stream between blocks, so the blocks own none."""
        if cfg.register_mode == "per_level":
            return 0
        if cfg.register_from_level >= 0 and l >= cfg.register_from_level:
            return cfg.register_objs * cfg.level_sizes[l]
        return 0

    @staticmethod
    def _n_level_reg(cfg, l):
        """Rows in level l's carried register set (register_mode=per_level): an explicit
        register_count at every level, else `register_objs` objects' worth of that level's
        per-object token count, which scales down the ladder. 0 above the birth level."""
        if cfg.register_mode != "per_level" or cfg.register_from_level < 0 \
                or l < cfg.register_from_level:
            return 0
        return int(cfg.register_count) or cfg.register_objs * cfg.level_sizes[l]

    @staticmethod
    @torch.compiler.disable
    def _build_provider(cfg, pos, pos0, obj_id, mat_id=None, particle_rigid=None):
        """Stage 0 runs EAGER even under torch.compile (custom CUDA ops,
        tensor asserts, data-dependent shapes). One graph break here; the
        network region below it compiles clean."""
        return GraphProvider(cfg, pos, pos0, obj_id, mat_id, particle_rigid)

    def _serial_for(self, provider, l, i):
        """Curve serialization for block index i at level l, or None."""
        if l < len(provider.serial) and provider.serial[l] is not None:
            order, inv = provider.serial[l][i % 2]
            return (order, inv, self.cfg.serial_patch)
        return None

    @staticmethod
    @torch.compiler.disable
    def _l0_groups(provider, body_obj, n_pose):
        """Stratified L0 token groups with pose rows: object o's particles plus
        the pose tokens of the bodies inside it (token ids N0 + k). Objects
        holding no rigid body get their particle range alone. Eager."""
        dev = provider.pos[0].device
        N0 = provider.N
        groups = []
        for o, (off, ln) in enumerate(provider.segs[0]):
            ids = [torch.arange(off, off + ln, device=dev)]
            mine = [N0 + k for k, bo in enumerate(body_obj) if bo == o]
            if mine:
                ids.append(torch.tensor(mine, device=dev))
            groups.append(torch.cat(ids))
        assert sum(int(g.numel()) for g in groups) == N0 + n_pose, \
            "every pose token must belong to exactly one object"
        return groups

    def _pose_tokens(self, pose, T, n_context):
        """[B, T', Kr, W0] pose token stream from the wrapper's `pose` pack."""
        feats, geom, gmask = pose["feats"], pose["geom"], pose["geom_mask"]
        B, Tp, Kr, _ = feats.shape
        g = self.pose_geom_embed(geom.to(feats.dtype))                # [B, Kr, P, W0]
        g = g.masked_fill(~gmask[..., None], float("-inf")).amax(dim=2)   # [B, Kr, W0]
        if "joint_type" in pose:                                      # each body's joint description
            g = g + self.pose_joint_type[pose["joint_type"]] + self.pose_joint_geom(pose["joint_geom"].to(g.dtype))
        tok = self.pose_token_in(feats) + g[:, None] + self.pose_token_type
        if self.pose_action_skip is None:
            # pose_action_mode=clean: no action value enters. A given body is conveyed by its pose rows
            # arriving unnoised and its tokens being told t = 0 (UNet.forward -> Block), nothing else.
            return tok
        # pose-space action skip on the FUTURE rows (commanded Δ6 of actuated
        # bodies; passive bodies read the learned no_action token).
        a = self.pose_action_skip(pose["actions"].to(feats.dtype))     # [B, F, Kr, W0]
        a = torch.where(pose["action_mask"][:, None, :, None], a,
                        self.pose_no_action_token.to(a.dtype))
        tok = torch.cat([tok[:, :n_context], tok[:, n_context:] + a], dim=1)
        return tok

    def _causal_mask(self, T, n_context, device, dtype):
        """Block-causal: context bidirectional, future causal (trunk convention)."""
        i = torch.arange(T, device=device)
        allowed = (i[None, :] < n_context) | (i[None, :] <= i[:, None])
        mask = torch.zeros(T, T, device=device, dtype=dtype)
        return mask.masked_fill(~allowed, float("-inf"))

    def forward(self, feats, pos, pos0, obj_id, mat_id, t, n_context: int,
                pose=None, particle_rigid=None, phys=None):
        """The U pass:

            stem -> [L0 enc blocks] -> pool -> [L1 enc] -> pool -> ...
                 -> [bottleneck blocks]
                 -> unpool -> [dec blocks] -> ... -> head

        Inputs: feats [B,T,N,c_in] raw per-particle channels (assembled by
        UNetWrapper); pos [B,T,N,3] noisy INTEGRATED positions (shift
        convention); pos0 [B,N,3] clean frame-0 positions; obj_id/mat_id
        [B,N]; t [B,T] per-frame noise levels (diffusion forcing).
        pose (pose_native): dict with feats [B,T,Kr,6] (normalized Δ6: clean
        context rows, noisy future rows), pos [B,T,Kr,3] body centres in the
        geometry currency, geom/geom_mask [B,Kr,P,3]/[B,Kr,P] templates,
        actions [B,F,Kr,6] normalized commanded Δ6, action_mask [B,Kr],
        body_obj [Kr] canonical object index per body, has_deform bool.
        Returns per-particle predictions [B,T,N,3] (None under pose_native
        when the batch has no deformable rows) — and, when `pose` is given,
        the tuple (particles, pose_v [B,T,Kr,6])."""
        B, T, N, _ = feats.shape
        provider = self._build_provider(self.cfg, pos, pos0, obj_id, mat_id, particle_rigid)
        n_pose = 0
        if pose is not None:
            assert self.cfg.pose_native, "pose tokens need cfg.pose_native"
            n_pose = int(pose["feats"].shape[2])
        t_emb = self.t_mlp(self.t_embed(t.reshape(-1)).reshape(B, T, -1))
        # cfg.token_adaln: ONE conditioning row per object, read at each object's first particle.
        # Material and physics are constant within an object (GraphProvider asserts the material on
        # the stratified ladder), so this carries the same values a per-token vector would, with the
        # projection on O rows. obj_index[l] maps that level's tokens to their object slot; the
        # provider lays every level out as contiguous per-object blocks.
        obj_cond, obj_index, obj_given, t_emb0 = None, None, None, None
        # cfg.pose_action_mode=clean: the given bodies (the pose pack's action_mask, i.e. after
        # train.action_dropout) are told t = 0. Per OBJECT, like the adaLN term: any given body makes
        # its object given, and every token of that object -- particles and pose rows -- sits at t = 0.
        _clean = self.cfg.pose_action_mode == "clean" and n_pose > 0
        if self.obj_mat_emb is not None or _clean:
            obj_index = [torch.repeat_interleave(
                torch.arange(len(provider.segs[l]), device=feats.device),
                torch.tensor([c for _, c in provider.segs[l]], device=feats.device)
            )[None].expand(B, -1) for l in range(len(provider.segs))]
            # the block stream is [particles | pose rows | registers]: each pose row takes its OWN
            # body's object slot, so a rigid body's token is modulated by the same material as its
            # particles. Registers are padded inside the block.
            if n_pose:
                bo = torch.as_tensor(pose["body_obj"], device=feats.device).long()[None].expand(B, -1)
                obj_index = [torch.cat([oi, bo], dim=1) for oi in obj_index]
        if self.obj_mat_emb is not None:
            assert phys is not None, "cfg.token_adaln: the wrapper must pass phys [B, N, 3]"
            first = torch.tensor([off for off, _ in provider.segs[0]], device=feats.device)
            oc_mat = self.obj_mat_emb(mat_id.gather(1, first[None].expand(B, -1)))
            oc_phys = self.obj_phys(phys.gather(1, first[None, :, None].expand(B, -1, 3))
                                    .to(self.obj_phys[0].weight.dtype))
            obj_cond = nn.functional.silu(oc_mat + oc_phys)                    # [B, O, t_dim]
        if _clean:
            O = len(provider.segs[0])
            bo = torch.as_tensor(pose["body_obj"], device=feats.device).long()[None].expand(B, -1)
            obj_given = torch.zeros(B, O, dtype=torch.long, device=feats.device).scatter_reduce(
                1, bo, pose["action_mask"].long(), reduce="amax").bool()          # [B, O]
            t_emb0 = self.t_mlp(self.t_embed(torch.zeros(B, device=t.device, dtype=t.dtype))).reshape(B, 1, -1)
        _oi = lambda l: None if obj_index is None else obj_index[l]
        assert not (self.cfg.causal_context and self.cfg.bidirectional_temporal), \
            "causal_context and bidirectional_temporal are exclusive"
        mask = None if (self.cfg.causal_context or self.cfg.bidirectional_temporal) \
            else self._causal_mask(T, n_context, feats.device, feats.dtype)
        L = len(self.cfg.strides)
        # stratified L0: spatial attention within each object's segment —
        # with pose rows, explicit index groups (object particles + its bodies'
        # pose tokens); without, contiguous segments.
        segs0 = groups0 = None
        if self.cfg.stratified:
            if n_pose:
                groups0 = self._l0_groups(provider, pose["body_obj"], n_pose)
            else:
                segs0 = provider.segs[0]
        # per-level positions for the mixer: [particles | pose] (registers are
        # appended inside the block and are unpositioned)
        if n_pose:
            lvl_pos = [torch.cat([provider.pos[l], pose["pos"]], dim=2)
                       for l in range(len(provider.pos))]
        else:
            lvl_pos = provider.pos
        # hoisted temporal rope: one (cos, sin) per head_dim per forward.
        ropes = {}

        def rope_for(blk):
            hd = blk.rotary.dim
            if hd not in ropes:
                cos, sin = blk.rotary(
                    torch.arange(T, device=feats.device).float()[None])
                ropes[hd] = (cos[0], sin[0])
            return ropes[hd]

        _ck = torch.utils.checkpoint.checkpoint
        _ck_seams = (self.cfg.ckpt_seams and self.cfg.ckpt_from_level >= 0
                     and self.training and torch.is_grad_enabled())

        def seam_level_ckpt(l):
            return _ck_seams and l >= self.cfg.ckpt_from_level

        if _ck_seams and self.cfg.ckpt_from_level == 0:
            x = _ck(self.stem, feats, use_reentrant=False) + self.mat_emb(mat_id)[:, None]
        else:
            x = self.stem(feats) + self.mat_emb(mat_id)[:, None]
        if n_pose:
            x = torch.cat([x, self._pose_tokens(pose, T, n_context)], dim=2)

        def split(z, n_part):                 # [particles | pose] -> both
            return (z[:, :, :n_part], z[:, :, n_part:]) if n_pose else (z, None)

        def born(z, l):                       # level l's carried register rows join the stream
            nr = self.n_level_reg[l]
            return (torch.cat([z, self.reg_init[str(l)][None, None].to(z.dtype)
                               .expand(z.shape[0], z.shape[1], -1, -1)], dim=2), nr) if nr else (z, 0)

        skips = []            # per-level seam skip (feeds the unpool)
        block_outs = []       # per-level list of every enc block's output (long skips)
        for l in range(L + 1):
            segs = segs0 if l == 0 else None
            grps = groups0 if l == 0 else None
            ckpt = (self.cfg.ckpt_from_level >= 0
                    and l >= self.cfg.ckpt_from_level and self.training
                    and not self.cfg.ckpt_ingraph)
            x, nr = born(x, l)
            outs = []
            for i, blk in enumerate(self.enc[l]):
                s = self._serial_for(provider, l, i)
                if ckpt:
                    x = torch.utils.checkpoint.checkpoint(
                        blk, x, lvl_pos[l], t_emb, mask, rope_for(blk),
                        None, s, segs, grps, nr, obj_cond, _oi(l), obj_given, t_emb0, use_reentrant=False)
                else:
                    x = blk(x, lvl_pos[l], t_emb, mask, rope_for(blk),
                            serial=s, segments=segs, groups=grps, n_carry=nr,
                            obj_cond=obj_cond, obj_index=_oi(l), obj_given=obj_given, t_emb0=t_emb0)
                outs.append(x)
            if nr:
                x = x[:, :, :x.shape[2] - nr]     # the run ends: registers do not cross the seam
            if l < L:
                skips.append(x)
                # the long skips KEEP their register rows: a decoder block merges them into a
                # stream that carries this level's own set in exactly those trailing slots.
                block_outs.append(outs)
                xp, xq = split(x, provider.N_l(l))
                if seam_level_ckpt(l):
                    xp = _ck(lambda a, t_, _l=l: self.pools[_l](a, provider.pos[_l], provider, _l, t_),
                             xp, t_emb, use_reentrant=False)
                else:
                    xp = self.pools[l](xp, provider.pos[l], provider, l, t_emb)
                x = torch.cat([xp, self.pose_up[l](xq)], dim=2) if n_pose else xp
        for l in range(L - 1, -1, -1):
            segs = segs0 if l == 0 else None
            grps = groups0 if l == 0 else None
            skp, _ = split(skips[l], provider.N_l(l))
            xp, xq = split(x, provider.N_l(l + 1))
            if seam_level_ckpt(l):
                xp = _ck(lambda s_, c_, t_, _l=l: self.unpools[_l](s_, c_, provider, _l, t_),
                         skp, xp, t_emb, use_reentrant=False)
            else:
                xp = self.unpools[l](skp, xp, provider, l, t_emb)
            x = torch.cat([xp, self.pose_down[l](xq)], dim=2) if n_pose else xp
            D = len(self.dec[l])
            ckpt = (self.cfg.ckpt_from_level >= 0
                    and l >= self.cfg.ckpt_from_level and self.training
                    and not self.cfg.ckpt_ingraph)
            x, nr = born(x, l)
            for i, blk in enumerate(self.dec[l]):
                # mirrored pairing: dec block D-1 merges enc block 0's output,
                # D-2 merges enc block 1's, ...
                s = self._serial_for(provider, l, self.cfg.enc_depths[l] + i)
                sk = block_outs[l][D - 1 - i]
                if ckpt:
                    x = torch.utils.checkpoint.checkpoint(
                        blk, x, lvl_pos[l], t_emb, mask, rope_for(blk),
                        sk, s, segs, grps, nr, obj_cond, _oi(l), obj_given, t_emb0, use_reentrant=False)
                else:
                    x = blk(x, lvl_pos[l], t_emb, mask, rope_for(blk),
                            skip=sk, serial=s, segments=segs, groups=grps, n_carry=nr,
                            obj_cond=obj_cond, obj_index=_oi(l), obj_given=obj_given, t_emb0=t_emb0)
            if nr:
                x = x[:, :, :x.shape[2] - nr]
        head = (lambda z: _ck(self.head, z, use_reentrant=False)) \
            if (_ck_seams and self.cfg.ckpt_from_level == 0) else self.head
        if not n_pose:
            return head(x)
        xp, xq = split(x, N)
        pose_v = self.pose_v_head(self.pose_norm_out(xq))            # [B,T,Kr,6]
        part = head(xp) if pose.get("has_deform", True) else None
        return part, pose_v


# ---------------------------------------------------------------------------
# train.py integration
# ---------------------------------------------------------------------------

def parse_l1_by_mat(spec) -> dict:
    """'0:256,1:256,5:512' (or a dict) -> {0: 256, 1: 256, 5: 512}."""
    if not spec:
        return {}
    if isinstance(spec, dict):
        return {int(k): int(v) for k, v in spec.items()}
    out = {}
    for item in str(spec).replace(" ", "").split(","):
        if not item:
            continue
        k, v = item.split(":")
        out[int(k)] = int(v)
    return out


def build_unet_config(cfg):
    """Resolve and validate the UNet ladder without allocating learned weights."""
    assert cfg.all_relative, "unet consumes delta streams (all_relative only)"
    split = lambda s: tuple(int(v) for v in str(s).replace(" ", "").split(",") if v)
    drop_off = bool(getattr(cfg, "unet_drop_offset_channel", False))
    l1_by_mat = parse_l1_by_mat(getattr(cfg, "unet_l1_by_mat", ""))
    l1_default = int(getattr(cfg, "unet_l1_default", 0) or 0)
    l1_rule = str(getattr(cfg, "unet_l1_rule", "") or "")
    if not (l1_by_mat or l1_default > 0 or l1_rule):
        assert cfg.unet_points > 0, "model.unet_points is required (classic ladder)"
    # UNIFIED (ported from robert/unified-unet-sidepose, 2026-08-28): 3 of the 4
    # domains are action-conditioned; the MPM lineage had no per-particle action
    # channels. +4 = [commanded delta (3), actuated flag (1)] on every token.
    actions = bool(getattr(cfg, "unet_actions", False))
    return UNetConfig(
        points=int(cfg.unet_points),
        strides=split(cfg.unet_strides),
        widths=split(cfg.unet_widths),
        enc_depths=split(cfg.unet_enc_depths),
        dec_depths=split(cfg.unet_dec_depths),
        in_dim=(7 if drop_off else UNetWrapper.IN_DIM) + (4 if actions else 0),
        num_materials=int(cfg.num_mat_types),
        head_dim=int(cfg.unet_head_dim),
        expansion=float(cfg.unet_expansion),
        t_dim=int(cfg.unet_t_dim),
        dropout=float(cfg.dropout),
        fps_random_start=bool(getattr(cfg, "unet_fps_random_start", True)),
        causal_context=bool(getattr(cfg, "unet_causal_context", False)),
        bidirectional_temporal=bool(getattr(cfg, "unet_bidirectional_temporal", False)),
        ckpt_from_level=int(getattr(cfg, "unet_ckpt_from_level", -1)),
        ckpt_ingraph=bool(getattr(cfg, "unet_ckpt_ingraph", False)),
        ckpt_seams=bool(getattr(cfg, "unet_ckpt_seams", False)),
        rope_f_min=float(getattr(cfg, "unet_rope_f_min", 0.5)),
        rope_f_max=float(getattr(cfg, "unet_rope_f_max", 64.0)),
        drop_offset_channel=drop_off,
        serial_levels=int(getattr(cfg, "unet_serial_levels", 0)),
        serial_patch=int(getattr(cfg, "unet_serial_patch", 1024)),
        dual_ffn=bool(getattr(cfg, "unet_dual_ffn", False)),
        token_adaln=bool(getattr(cfg, "unet_token_adaln", False)),
        pose_action_mode=str(getattr(cfg, "unet_pose_action_mode", "skip") or "skip"),
        register_from_level=int(getattr(cfg, "unet_register_from_level", -1)),
        register_objs=int(getattr(cfg, "unet_register_objs", 2)),
        register_count=int(getattr(cfg, "unet_register_count", 0) or 0),
        register_mode=str(getattr(cfg, "unet_register_mode", "per_block") or "per_block"),
        pool_mode=str(getattr(cfg, "unet_pool_mode", "attn")),
        unpool_mode=str(getattr(cfg, "unet_unpool_mode", "attn")),
        pool_k=int(getattr(cfg, "unet_pool_k", 16)),
        interp_k=int(getattr(cfg, "unet_interp_k", 3)),
        l1_by_mat=l1_by_mat,
        l1_default=l1_default,
        l1_rule=l1_rule,
        l1_rule_min=int(getattr(cfg, "unet_l1_rule_min", 32)),
        l1_rule_max=int(getattr(cfg, "unet_l1_rule_max", 1024)),
        l1_scene_budget=getattr(cfg, "unet_l1_scene_budget", 0),
        l1_rigid_max=getattr(cfg, "unet_l1_rigid_max", 0),
        pose_native=bool(getattr(cfg, "pose_native", False)),
        articulated_bodies=bool(getattr(cfg, "articulated_bodies", False)),
    )


class UNetWrapper(nn.Module):
    """Adapter that gives the UNet the joint model's train.py contract:

        pred = wrapper(batch, noisy_future, timestep,
                       noisy_context=..., context_timesteps=...)  -> [B,F,N,3]

    so the standard training loop — noising, context noising, DF sampling,
    val, gifs, logging — runs UNCHANGED. The featurization is assembled here,
    in the open:

      tokens     the (C-1)+F delta frames; context channel = `noisy_context`
                 when given (DF-on-context or light noise in training, clean
                 normalized context at inference), else the clean normalized
                 batch deltas.
      positions  integrated frame0 + cumsum with the SHIFT convention ALWAYS
                 ON: each token's geometry is the START of its transition, so
                 no frame's graphs, offsets, or coordinate channels include
                 its OWN noise draw.
      currencies TWO position currencies leave this wrapper. GEOMETRY (RoPE,
                 FPS, kNN, interp weights) gets pos_meters / r with r the
                 geometry radius: `unet_geo_r_m` when set (fixed, metric —
                 docs/CODE_WALKTHROUGH.md), else the isotropic delta
                 half-range delta_scale/2 (checkpoint compatibility). FEATURE
                 channels (frame-0 offset, z) stay position-stats-normalized.
      detach     geometry positions are DETACHED (house rule: x_t-derived
                 coordinates act as geometry only; gradient flows through
                 the delta channels).
      channels   [delta(3) | frame-0 offset(3) | z(1) | physics(3)], in_dim=10.
      layout     objects are contiguous blocks in canonical order; per-object
                 counts may differ (stratified mode). No padded rows.

      DF-on-context (train.noise_context + supervise_context): 2-D per-frame
      context timesteps are accepted and enter t_tok; when
      cfg.unet_supervise_context is on, the context-token predictions are
      stashed on `self._last_ctx_pred` for train_step's context loss.

    Rejected by hard assert: per-object noise ([B,F,N] timesteps), subset
    supervision, non-delta targets.
    """

    IN_DIM = 10

    def __init__(self, cfg):
        """cfg: ModelConfig with the unet_* fields (models/transformer.py)."""
        super().__init__()
        self.cfg = cfg
        self._drop_off = bool(getattr(cfg, "unet_drop_offset_channel", False))
        self._geo_r = float(getattr(cfg, "unet_geo_r_m", 0.0) or 0.0)
        self._actions = bool(getattr(cfg, "unet_actions", False))
        self.unet_cfg = build_unet_config(cfg)
        # model.unet_core selects the ladder's attention. "unet" (default) is the factorized production
        # UNet -- every knob below is then unread and the model is exactly what it has always been.
        core = str(getattr(cfg, "unet_core", "unet") or "unet")
        self._unet_core = core
        self._token_adaln = bool(getattr(cfg, "unet_token_adaln", False))
        if core == "tube_joint":
            from models.tube_joint import TubeJointCore
            _list = lambda v: [int(x) for x in str(v or "").replace(" ", "").split(",") if x]
            _floats = lambda v: [float(x) for x in str(v).replace(" ", "").split(",") if x]
            budget = _list(getattr(cfg, "unet_rope_budget", ""))
            assert not budget or len(budget) == 4, \
                f"model.unet_rope_budget is 'spatial,row,identity,pass' in head dims, got {budget!r}"
            head_dims = _list(getattr(cfg, "unet_head_dims", ""))
            self.model = TubeJointCore(
                self.unet_cfg,
                tube_sizes=_list(cfg.unet_tube_sizes),
                curve_orders=int(getattr(cfg, "unet_curve_orders", 2)),
                grouping=str(getattr(cfg, "unet_tube_grouping", "curves") or "curves"),
                group_draws=int(getattr(cfg, "unet_tube_group_draws", 2)),
                within_object_levels=_list(getattr(cfg, "unet_tube_within_object", "")),
                long_skip=bool(getattr(cfg, "unet_tube_long_skip", False)),
                dec_mirror=bool(getattr(cfg, "unet_tube_dec_mirror", False)),
                register_mode=str(getattr(cfg, "unet_tube_register_mode", "carried") or "carried"),
                register_objs=float(getattr(cfg, "unet_tube_register_objs", 0.0) or 0.0),
                token_adaln=self._token_adaln,
                head_dims=head_dims or None,
                rope_budget_dims=budget or None,
                rope_fraction=float(getattr(cfg, "unet_rope_fraction", 0.75) or 0.75),
                rope_t_dims=int(getattr(cfg, "unet_rope_t_dims", 8)),
                rope_t_max_period=float(getattr(cfg, "unet_rope_t_max_period", 64.0)),
                rope_t_min_period=float(getattr(cfg, "unet_rope_t_min_period", 2.0)),
                rope_t_base=float(getattr(cfg, "unet_rope_t_base", 0.0) or 0.0),
                rope_identity=str(getattr(cfg, "unet_rope_identity", "curve_rank")),
                rope_identity_max_period=float(getattr(cfg, "unet_rope_identity_max_period", 2048.0)),
                rope_identity_periods_m=_floats(getattr(cfg, "unet_rope_identity_periods_m", "0.012,0.4")),
                fp32_head=bool(getattr(cfg, "unet_fp32_head", True)),
                geo_r=float(getattr(cfg, "unet_geo_r_m", 0.133155) or 0.133155))
        else:
            assert core == "unet", f"unknown model.unet_core {core!r}"
            self.model = UNet(self.unet_cfg)
        if self._actions:
            # Zero-init the stem columns of the 4 action channels: at step 0 the model is
            # bit-identical to the action-free one (same seed), and the columns still get
            # gradient (outer(dL/dh, action inputs)) so the channels are learned.
            with torch.no_grad():
                self.model.stem[0].weight[:, -4:].zero_()

    def _physics(self, batch):
        """Same channel recipe as the trunk's _prepare_physics_features,
        honoring the same normalize_physics_features flag."""
        f = torch.stack([torch.log10(batch.E + 1e-8), batch.nu,
                         batch.rho / 1000.0], dim=-1)
        if getattr(self.cfg, "normalize_physics_features", False):
            f = torch.stack([(f[..., 0] - 5.5) / 1.5,
                             (f[..., 1] - 0.25) / 0.2,
                             (f[..., 2] - 1.75) / 1.25], dim=-1)
        return f

    @torch.compiler.disable
    def _object_ids(self, batch):
        """Validate the contiguous-block layout; return canonical 0..O-1 ids.
        Runs eager under compile (tensor-content asserts would graph-break).
        Classic mode additionally requires every block to be `unet_points`."""
        B, _, N, _ = batch.positions.shape
        oid = batch.object_ids
        assert bool((oid == oid[:1]).all()), \
            "object layout must be identical across the batch (bucket batches by shape)"
        o0 = oid[0]
        assert bool((batch.valid_lengths == N).all()), "padded batches unsupported"
        # Contiguous runs of equal id, in ANY order (pickup stores the gripper
        # block first with id 1, then the object with id 0). Each id must form
        # exactly one run; the UNet gets canonical ids in RUN order. The
        # batch's own object_ids are untouched (the loss path keeps them).
        change = torch.nonzero(o0[1:] != o0[:-1]).flatten() + 1
        starts = torch.cat([torch.zeros(1, dtype=torch.long, device=o0.device), change])
        run_ids = o0[starts]
        assert run_ids.unique().numel() == run_ids.numel(), \
            "object_ids must be contiguous blocks (an id appears in more than one run)"
        ends = torch.cat([change, torch.tensor([N], device=o0.device)])
        counts = ends - starts
        if not self.unet_cfg.stratified:
            P = self.unet_cfg.points
            assert bool((counts == P).all()), (
                f"classic ladder needs equal blocks of unet_points={P}; got "
                f"{counts.tolist()} (set model.unet_l1_by_mat for ragged objects)")
        return torch.arange(counts.numel(), device=batch.device).repeat_interleave(
            counts)[None].expand(B, N)

    def _geo_radius(self, batch, device, dtype):
        """Geometry radius r (metres) for the RoPE/kNN/interp currency."""
        if self._geo_r > 0:
            return torch.tensor(self._geo_r, device=device, dtype=dtype)
        _dscale = torch.as_tensor(batch.delta_stats.scale, device=device, dtype=dtype)
        assert torch.allclose(_dscale, _dscale[:1]), (
            "UNet geometry requires isotropic delta bounds "
            "or a positive fixed model.unet_geo_r_m")
        return _dscale[0] / 2                                 # isotropic half-range, metres

    def forward(self, batch, noisy_future, timestep, noisy_context=None,
                context_timesteps=None, noisy_pose=None, noisy_ctx_pose=None):
        """Returns [B,F,N,3]; under pose_native with `noisy_pose` [B,F,Kr,6]
        given, the tuple (particle_pred | None, pose_v [B,F,Kr,6]) — Robert's
        Tier-2 v2 contract (train.py / inference.py call it this way)."""
        # context_timesteps: full DF-on-context mode (train.noise_context +
        # supervise_context). Per-frame [B, C-1] noise levels for the context
        # transitions; None => clean context (t=0).
        if context_timesteps is not None:
            assert context_timesteps.ndim == 2, \
                "unet: per-object context timesteps unsupported"
        assert batch.prediction_target == "delta", "Unified UNet supports delta targets only"
        B, F, N, _ = noisy_future.shape
        C = batch.context_len
        device = batch.device
        if timestep.ndim == 1:                                 # [B] -> [B, F]
            timestep = timestep[:, None].expand(-1, F)
        assert timestep.ndim == 2, "unet: per-object noise unsupported"

        # -- Tier-2 v2 (pose_native): rebuild the rigid rows of noisy_future
        # from the NOISY pose BEFORE positions/kNN/RoPE, so the uniform encoding
        # sees a pose-consistent rigid scene at every noise level.
        pose_pack = None
        if noisy_pose is not None:
            assert self.unet_cfg.pose_native, "noisy_pose given but model.pose_native is off"
            if batch.joint_type is not None and not self.unet_cfg.articulated_bodies:
                raise ValueError("the batch carries articulated joints but model.articulated_bodies is off; "
                                 "build its batches with articulated=False")
            from utils.pose_native import pose_native_inputs
            if noisy_ctx_pose is not None:
                # train.pose_noise_context: the context poses are noised at the context rows'
                # levels and their particle rows are rebuilt from them, exactly like the future.
                assert noisy_context is not None, \
                    "noisy_ctx_pose requires train.noise_context (a noised context stream)"
                noisy_future, noisy_context, pose_pack = pose_native_inputs(
                    batch, noisy_future, noisy_pose,
                    noisy_ctx_pose=noisy_ctx_pose, noisy_context=noisy_context)
            else:
                noisy_future, pose_pack = pose_native_inputs(batch, noisy_future, noisy_pose)
        elif (self.unet_cfg.pose_native and batch.body_type is not None
              and bool((batch.body_type[0] == 1).any())):
            raise RuntimeError(
                "pose_native checkpoint received a rigid-body batch without "
                "noisy_pose — use the pose-native inference path "
                "(run_diffusion_inference_df with emit_sidecar batches).")

        # -- delta channel: [context ; noisy future], normalized.
        # Token sequence = (C-1) context transitions + F future transitions.
        ctx_norm = noisy_context if noisy_context is not None \
            else batch.delta_stats.normalize(batch.context_deltas)  # [B,C-1,N,3]
        deltas_norm = torch.cat([ctx_norm, noisy_future], dim=1)    # [B,T',N,3]

        # -- INTEGRATED POSITIONS under the SHIFT convention. Each token's
        # coordinates are the START of its transition (exclusive cumsum), so
        # no token's geometry contains its OWN noise draw. Positions are
        # DETACHED (house rule).
        frame0 = batch.context[:, 0]                                # [B,N,3] clean
        ctx_raw = batch.delta_stats.denormalize(ctx_norm)
        fut_raw = batch.delta_stats.denormalize(noisy_future)
        ctx_cum = torch.cumsum(ctx_raw, dim=1)
        fut_cum = torch.cumsum(fut_raw, dim=1)
        last_ctx = frame0 + ctx_cum[:, -1]
        pos = torch.cat([frame0.unsqueeze(1) + (ctx_cum - ctx_raw),   # exclusive cumsum
                         last_ctx.unsqueeze(1) + (fut_cum - fut_raw)], dim=1)
        pos_norm = batch.position_stats.normalize(pos).detach()
        pos0_norm = batch.position_stats.normalize(frame0)

        # -- GEOMETRY currency: pos_meters / r. RoPE frequencies are fixed
        # functions of their input, so the band calibration (0.5..64 rad per
        # r-unit) is set against r — the fixed unet_geo_r_m, or the isotropic
        # delta half-range when loading a compatible historical checkpoint.
        _r = self._geo_radius(batch, device, pos.dtype)
        pos_geo = (pos / _r).detach()
        pos0_geo = frame0 / _r

        # -- the 10 input channels per token:
        #   [0:3]  normalized delta (noisy for future tokens) — the EVIDENCE
        #   [3:6]  displacement from own frame-0 position — relative identity
        #   [6:7]  NOISY z height (absolute z is deliberate: floor/gravity
        #          are absolute-z phenomena; x/y absolutes never enter)
        #   [7:10] physics params (E, nu, rho), normalized
        _parts = [deltas_norm]
        if not self._drop_off:
            _parts.append(pos_norm - pos0_norm.unsqueeze(1))
        _parts += [pos_norm[..., 2:3],
                   self._physics(batch).unsqueeze(1).expand(B, C - 1 + F, N, 3)]
        if batch.action_active is not None and not self._actions:
            raise ValueError('Scheduled MPM drag requires unet_actions=true')
        if self._actions:
            #   [10:13] commanded per-particle displacement (same normalizer as the
            #          evidence channel; context transitions carry ZERO — their motion
            #          is already in the delta channel)
            #   [13:14] actuated flag (time-invariant)
            _fd = deltas_norm.dtype
            # "no action" is normalize(0), not 0: the per-domain delta stats are asymmetric
            # (cloth: -4.6/+3.6 mm), so context rows use the same constant as passive rows.
            _zero = batch.delta_stats.normalize(torch.zeros(B, C - 1, N, 3, device=device)).to(_fd)
            # The explicit commanded collection must preserve command amplitude:
            # its p99 scale describes achieved motion, not a command limit. Keep
            # historical domains' input convention unchanged.
            clip_actions = batch.domain != "mpm_drag"
            act = _zero.new_zeros(B, C - 1 + F, N, 3) + _zero[:, :1] \
                if batch.actions is None else torch.cat([
                    _zero, batch.delta_stats.normalize(batch.actions, clip=clip_actions).to(_fd)], dim=1)
            amask = (batch.action_mask.to(_fd) if batch.action_mask is not None
                     else torch.zeros(B, N, device=device, dtype=_fd))
            flag = amask[:, None, :, None].expand(B, C - 1 + F, N, 1) if batch.action_active is None else torch.cat(
                [batch.context_action_active,batch.action_active],dim=1).to(_fd)[...,None]
            _parts += [act, flag]
        feats = torch.cat(_parts, dim=-1)

        mat_id = batch.mat_type.long() if batch.mat_type is not None \
            else torch.zeros(B, N, dtype=torch.long, device=device)
        ctx_t = context_timesteps.to(feats.dtype) if context_timesteps is not None \
            else torch.zeros(B, C - 1, device=device)
        t_tok = torch.cat([ctx_t, timestep.to(feats.dtype)], dim=1)

        obj_ids = self._object_ids(batch)
        pose = None
        if pose_pack is not None:
            pose = self._pose_pack(batch, pose_pack, obj_ids, _r, C, feats.dtype)
        _extra = {"phys": self._physics(batch)} if self._token_adaln else {}   # unet_token_adaln
        out = self.model(feats=feats, pos=pos_geo, pos0=pos0_geo,
                         obj_id=obj_ids, mat_id=mat_id,
                         t=t_tok, n_context=C - 1, pose=pose,
                         particle_rigid=particle_rigidity(batch) if self.unet_cfg.l1_rigid_max else None, **_extra)
        if pose is not None:
            part, pose_v = out
            self._last_ctx_pred = part[:, :C - 1] if (
                part is not None and context_timesteps is not None
                and getattr(self.cfg, "unet_supervise_context", False)) else None
            # Pose-side context supervision (train.pose_noise_context): stash the context rows
            # for the objective and for the sampler, which denoises them alongside the future.
            self._last_ctx_pose_pred = pose_v[:, :C - 1] if noisy_ctx_pose is not None else None
            # model.unet_deformable_context: the deformable rows' context particles, for the objective
            # and the sampler (rigid rows of it are unused: they are rebuilt from the context pose).
            self._last_ctx_part_pred = part[:, :C - 1] if (
                part is not None and noisy_ctx_pose is not None
                and getattr(self.cfg, "unet_deformable_context", False)) else None
            return (part[:, C - 1:] if part is not None else None), pose_v[:, C - 1:]
        # DF-on-context supervision: stash the context-token predictions for
        # train_step (side-channel convention; return shape stays future-only).
        self._last_ctx_pred = out[:, :C - 1] if (
            context_timesteps is not None
            and getattr(self.cfg, "unet_supervise_context", False)) else None
        return out[:, C - 1:]                                       # future tokens only

    def _pose_pack(self, batch, pack, obj_ids, r, C, dtype):
        """Assemble the UNet's `pose` dict from the sidecar + the input-rewrite
        pack. Token rows follow the particle convention: (C-1) context
        transitions (clean GT Δ6) then F future transitions (noisy Δ6);
        RoPE centres are the START of each transition (SHIFT convention)."""
        from utils.pose_native import body_object_index, context_pose_deltas, joint_features
        ridx = pack["rigid_idx"]
        if pack.get("noisy_ctx_pose") is not None:
            c6 = pack["noisy_ctx_pose"].to(dtype)                        # [B,C-1,Kr,6] noised
        else:
            c6 = batch.pose_delta_stats.normalize(
                context_pose_deltas(batch, ridx), clip=False).to(dtype)   # [B,C-1,Kr,6] clean GT
        feats = torch.cat([c6, pack["noisy_pose"].to(dtype)], dim=1)     # [B,T',Kr,6]
        ctx_c = (pack["ctx_centers"].float() if pack.get("ctx_centers") is not None
                 else batch.body_poses[:, :C - 1, ridx, :3, 3].float())   # frames 0..C-2
        fut_c = torch.cat([pack["anchor_centers"][:, None],
                           pack["centers"][:, :-1]], dim=1)               # start of each future transition
        pos = (torch.cat([ctx_c, fut_c], dim=1) / r).detach().to(dtype)   # [B,T',Kr,3]
        a6 = batch.pose_delta_stats.normalize(
            batch.pose_actions[:, :, ridx], clip=False).to(dtype)        # [B,F,Kr,6]
        out = dict(
            feats=feats, pos=pos,
            geom=batch.body_templates[:, ridx].float(),
            geom_mask=batch.body_template_mask[:, ridx],
            actions=a6, action_mask=batch.pose_action_mask[:, ridx],
            body_obj=body_object_index(batch, obj_ids, ridx),
            has_deform=bool((batch.body_type[0] == 0).any()),
        )
        if self.unet_cfg.articulated_bodies:
            out["joint_type"], out["joint_geom"] = joint_features(batch, ridx)
        return out
