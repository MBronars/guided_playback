"""TubeJoint: the joint-attention U (model.unet_core=tube_joint).

The SAME U as models/unet_std.UNet -- the same per-object FPS ladder and GraphProvider, the same
pointnet Pool / interpolation Unpool seams, the same widths / depths / registers / checkpointing
knobs. One thing changes: a level's blocks are JointBlocks (models/joint_block.py), so each
attention call is ONE kernel over that call's whole (rows x tokens) sequence instead of a temporal
pass followed by a spatial pass.

What a single call contains is set per level by model.unet_tube_sizes:

    tube levels   (P > 0): the level's tokens are cut into groups of P, and each group x ALL rows is
                  one call of P*T tokens. model.unet_tube_grouping picks the cut:
                    curves  consecutive chunks of a space-filling curve built on the CLEAN frame-0
                            positions (Morton codes, one axis permutation per curve order). Block i
                            of the level uses order i mod K, so the boundaries move between blocks.
                    random  uniformly random equal-size groups, re-drawn every forward
                            (models/tube_groups.py), `unet_tube_group_draws` draws cycled over blocks.
                  Either way a group is a fixed token set through time (Lagrangian).
    whole levels  (P = 0): one call over the level (all tokens x all rows). Its length grows with the
                  object count, which is in-distribution when training spans the range served.

model.unet_tube_within_object keeps a group inside one object (each object cut on its own, its last
group padded on its own). Registers ride whole-level calls only; see `register_mode`.

RoPE (RopeAxes): the rotated head dims are split over the token's noisy current x, y, z, its row
index, and an IDENTITY group (model.unet_rope_identity). frame0_pos spends 2 bands on each of the
token's CLEAN frame-0 x, y, z, so every row of a particle sits at identity offset exactly 0 -- an
exact "same particle" match whatever the noisy positions do. model.unet_rope_budget states the split
in dims explicitly; without it the budget is derived from unet_rope_fraction.

Head in fp32. Returns [B, T, N, 3] (all rows; the wrapper slices context rows). Particle
representation only -- no pose tokens, so rigid domains need data.rigid_poses=false.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from models.components.embeddings import SinusoidalTimeEmbed
from models.joint_block import JointBlock
from models.unet_std import (UNetConfig, GraphProvider, Pool, Unpool, gather_tokens, _morton3,
                             _ROPE_F_MAX, _ROPE_FRACTION)

_PERMS = ((0, 1, 2), (1, 2, 0), (2, 0, 1), (0, 2, 1))       # axis permutations = distinct Morton curve orders


class RopeAxes(nn.Module):
    """Multi-axis rotary tables. groups: list of (n_dims, frequencies[n_dims/2]) applied to coords[..., a] in
    order; the remaining head dims are identity pass-through. cos/sin are [B, 1, n, head_dim]."""

    def __init__(self, head_dim: int, groups):
        super().__init__()
        used = sum(g for g, _ in groups)
        assert used <= head_dim and all(g % 2 == 0 and f.numel() == g // 2 for g, f in groups), \
            (head_dim, [(g, f.numel()) for g, f in groups])
        self.dims = [g for g, _ in groups]
        self.pad = head_dim - used
        for i, (_, f) in enumerate(groups):
            self.register_buffer(f"f{i}", f.float(), persistent=False)

    @torch.no_grad()
    def tables(self, coords):                                  # coords [B, n, A] with A == len(groups)
        cs, ss = [], []
        for i, g in enumerate(self.dims):
            ang = coords[..., i:i + 1].float() * getattr(self, f"f{i}")     # [B, n, g/2]
            c, s = torch.cos(ang), torch.sin(ang)
            cs.append(torch.stack([c, c], -1).flatten(-2)); ss.append(torch.stack([s, s], -1).flatten(-2))
        cos, sin = torch.cat(cs, -1), torch.cat(ss, -1)
        if self.pad:
            cos = torch.cat([cos, torch.ones_like(cos[..., :self.pad])], -1)
            sin = torch.cat([sin, torch.zeros_like(sin[..., :self.pad])], -1)
        return cos[:, None], sin[:, None]


def _band(lo_period, hi_period, n):
    return torch.exp(torch.linspace(math.log(2 * math.pi / hi_period), math.log(2 * math.pi / lo_period), n))


def rope_groups(head_dim, f_min, f_max=_ROPE_F_MAX, fraction=_ROPE_FRACTION, t_dims=8, t_max_period=64.0,
                identity="curve_rank", identity_dims=4, identity_min_period=8.0, identity_max_period=2048.0,
                identity_freqs=None):
    """DERIVED RoPE budget: round(head_dim * fraction) dims are rotated, the rest pass through.
    Returns (groups, axis names); groups apply to coords[..., a] in order.
    identity:
      "none"        x, y, z (noisy current position) + t (row index) share the rotated dims.
      "curve_rank"  + 1 identity axis (the token's frame-0 curve rank, 2 bands), carved out of the rotated budget.
      "frame0_pos"  + 3 identity axes = the token's CLEAN frame-0 x, y, z, len(identity_freqs) bands each (rad per
                    geometry unit), taken from the PASS-THROUGH dims: the noisy xyz and row bands are unchanged."""
    g_t = (int(t_dims) // 2) * 2
    g_i = (int(identity_dims) // 2) * 2 if identity == "curve_rank" else 0
    g_id = 6 * len(identity_freqs) if identity == "frame0_pos" else 0     # frame0_pos lives in the un-rotated remainder
    rot = min(int(round(head_dim * fraction)), head_dim - g_id)            # fraction 1.0 = no pass-through beyond identity
    g_s = ((rot - g_t - g_i) // 6) * 2
    assert g_s >= 4 and g_t >= 2, \
        f"head_dim {head_dim}: RoPE budget too small (spatial {g_s}, time {g_t}, identity {g_i})"
    f_sp = torch.exp(torch.linspace(math.log(f_min), math.log(f_max), g_s // 2))
    groups = [(g_s, f_sp), (g_s, f_sp), (g_s, f_sp), (g_t, _band(2.0, t_max_period, g_t // 2))]
    names = ["x", "y", "z", "t"]
    if identity == "curve_rank":
        groups.append((g_i, _band(identity_min_period, identity_max_period, g_i // 2))); names.append("identity")
    elif identity == "frame0_pos":
        f_id = torch.as_tensor(identity_freqs, dtype=torch.float32)
        assert f_id.numel() >= 1
        groups += [(2 * f_id.numel(), f_id)] * 3; names += ["id_x0", "id_y0", "id_z0"]
        assert sum(g for g, _ in groups) <= head_dim, \
            f"head_dim {head_dim}: no pass-through room for the frame0_pos identity"
    else:
        assert identity == "none", identity
    return groups, names


def rope_budget(head_dim, f_min, f_max, budget, t_max_period, identity_freqs, t_min_period=2.0, t_base=0.0):
    """EXPLICIT RoPE budget (model.unet_rope_budget = "spatial,row,identity,pass"):
      spatial   dims on the noisy x, y, z (spatial/6 log-spaced bands per axis, f_min..f_max); must be % 6 == 0
      row       dims on the row index, log-spaced periods t_min_period..t_max_period rows (2 = the row Nyquist;
                2 pi = the standard RoPE's finest band, 1 rad per row). t_base > 0 instead uses the standard
                inverse-base bands theta^(-2i/d) (10000 = CogVideoX / Wan, 256 = HunyuanVideo).
      identity  dims on the clean frame-0 x, y, z together = 6 * len(identity_freqs)
      pass      dims left unrotated
    The four must add up to head_dim exactly."""
    s_dims, t_dims, i_dims, p_dims = (int(v) for v in budget)
    g_s, g_t, g_id = s_dims // 3, t_dims, 2 * len(identity_freqs)
    assert s_dims % 6 == 0 and g_t % 2 == 0 and g_t >= 2, (s_dims, t_dims)
    assert 3 * g_id == i_dims, f"identity budget {i_dims} != 6 x {len(identity_freqs)} bands"
    assert s_dims + t_dims + i_dims + p_dims == int(head_dim), \
        f"RoPE budget: spatial {s_dims} + row {t_dims} + identity {i_dims} + pass {p_dims} != head dim {head_dim}"
    f_sp = torch.exp(torch.linspace(math.log(f_min), math.log(f_max), g_s // 2))
    if t_base and float(t_base) > 0:
        f_t = torch.flip(float(t_base) ** (-torch.arange(0, g_t, 2, dtype=torch.float32) / g_t), dims=[0])
    else:
        f_t = _band(float(t_min_period), t_max_period, g_t // 2)
    f_id = torch.as_tensor(identity_freqs, dtype=torch.float32)
    groups = [(g_s, f_sp)] * 3 + [(g_t, f_t)] + [(g_id, f_id)] * 3
    return groups, ["x", "y", "z", "t", "id_x0", "id_y0", "id_z0"]


class TubeJointCore(nn.Module):
    """The joint-attention U (see the module docstring). Reads the same UNetConfig as UNet."""

    def __init__(self, cfg: UNetConfig, tube_sizes, curve_orders: int = 2, rope_t_dims: int = 8,
                 rope_t_max_period: float = 64.0, rope_t_min_period: float = 2.0, rope_t_base: float = 0.0,
                 rope_identity: str = "curve_rank", rope_identity_max_period: float = 2048.0,
                 rope_identity_periods_m=(0.012, 0.4), rope_fraction: float = 0.75, rope_budget_dims=None,
                 fp32_head: bool = True, geo_r: float = 0.133155, within_object_levels=(), head_dims=None,
                 token_adaln: bool = False, long_skip: bool = False, dec_mirror: bool = False,
                 grouping: str = "curves", group_draws: int = 2, register_mode: str = "carried",
                 register_objs: float = 0.0, ckpt_levels=None, ckpt_seams=None):
        super().__init__()
        W, L = cfg.widths, len(cfg.strides)
        self.cfg, self.geo_r, self.K = cfg, float(geo_r), int(curve_orders)
        self.fp32_head, self.token_adaln = bool(fp32_head), bool(token_adaln)
        # unet_tube_long_skip: the UNet Block's HunyuanDiT-style block long skips -- decoder block i of a level
        # merges the output of that level's encoder block (D_enc - 1 - i). unet_tube_dec_mirror: that decoder
        # block also cuts its groups exactly as the encoder block it mirrors, so the pair shares one partition.
        self.long_skip, self.dec_mirror = bool(long_skip), bool(dec_mirror)
        # unet_tube_grouping: how a tubed level's tokens are cut into attention calls.
        #   curves  (default) a Morton curve cut every tube-size tokens; block i uses curve order i mod K
        #   random  models/tube_groups.py: uniformly random equal-size groups, no locality. Re-drawn every
        #           forward: group_draws independent draws per level, block i of a level uses draw i mod draws.
        assert grouping in ("curves", "random"), grouping
        self.grouping, self.group_draws = grouping, max(1, int(group_draws))
        assert rope_identity in ("none", "curve_rank", "frame0_pos"), rope_identity
        self.identity = rope_identity
        assert cfg.pool_mode == "pointnet" and cfg.unpool_mode == "pointnet", "tube_joint: pointnet seams"
        tube_sizes = [int(v) for v in tube_sizes]
        assert len(tube_sizes) == L + 1, f"unet_tube_sizes needs one entry per level ({L + 1}), got {tube_sizes}"
        assert len(cfg.enc_depths) == L + 1 and len(cfg.dec_depths) == L
        self.tube = tube_sizes
        # unet_tube_within_object: at these levels a group never straddles objects -- the curve runs object by
        # object (each on its OWN frame-0 bbox), or the random draw groups each object on its own; an object
        # larger than P fills several groups and its last one is padded.
        self.within_object = frozenset(int(v) for v in within_object_levels)
        assert all(0 <= l <= L and tube_sizes[l] > 0 for l in self.within_object), \
            f"unet_tube_within_object levels {sorted(self.within_object)} need a tube size > 0 (tubes {tube_sizes})"
        assert 1 <= self.K <= len(_PERMS)

        # -- registers ------------------------------------------------------------------------------------
        # Learned register rows, appended per ROW to a level's token stream (every frame gets its own copy, so
        # they carry that frame's adaLN modulation) and given pass-through RoPE. Off unless
        # unet_register_from_level >= 0. Only whole-level calls (tube size 0) can host them.
        #   per_level  (unet_tube_register_mode) every level >= the birth level owns a set: it is re-born from
        #              that level's learned init at the start of each run of blocks, carried ACROSS THE BLOCKS
        #              of that run, and dropped at the seam. Nothing crosses a seam.
        #   carried    ONE set, born at the birth level, width-projected by a linear at each deeper seam
        #              (reg_up going down, reg_down coming up) and alive until it returns.
        assert register_mode in ("carried", "per_level"), register_mode
        self.reg_mode, self.reg_l0, self.n_reg = register_mode, int(cfg.register_from_level), {}
        if self.reg_l0 >= 0:
            assert 1 <= self.reg_l0 <= L and all(t == 0 for t in tube_sizes[self.reg_l0:]), \
                f"tube_joint: registers need whole-level calls (tube size 0) from level {self.reg_l0} down (tubes {tube_sizes})"
            # rows per set: unet_register_count is an explicit count at every level; otherwise
            # `objs` OBJECTS' WORTH of that level's per-object tokens, which scales down the ladder.
            # unet_tube_register_objs takes a fraction (0.5 = half an object) where unet_register_objs,
            # being an integer multiplier shared with the base UNet, cannot.
            objs = float(register_objs) if float(register_objs) > 0 else float(cfg.register_objs)
            count = lambda l: int(cfg.register_count) or int(round(objs * cfg.level_sizes[l]))
            if self.reg_mode == "per_level":
                self.n_reg = {l: count(l) for l in range(self.reg_l0, L + 1)}
                assert all(v >= 1 for v in self.n_reg.values()), self.n_reg
                self.reg_init = nn.ParameterDict(
                    {str(l): nn.Parameter(torch.randn(self.n_reg[l], W[l]) * 0.02) for l in range(self.reg_l0, L + 1)})
            else:
                n_carry = count(self.reg_l0)
                assert n_carry >= 1
                self.n_reg = {l: n_carry for l in range(self.reg_l0, L + 1)}
                self.reg_init = nn.Parameter(torch.randn(n_carry, W[self.reg_l0]) * 0.02)
                self.reg_up = nn.ModuleDict({str(l): nn.Linear(W[l], W[l + 1]) for l in range(self.reg_l0, L)})
                self.reg_down = nn.ModuleDict({str(l): nn.Linear(W[l + 1], W[l]) for l in range(self.reg_l0, L)})

        # -- blocks and seams -----------------------------------------------------------------------------
        self.stem = nn.Sequential(nn.Linear(cfg.in_dim, W[0]), nn.SiLU(), nn.Linear(W[0], W[0]))
        self.mat_emb = nn.Embedding(cfg.num_materials, W[0])
        self.t_embed = SinusoidalTimeEmbed(cfg.t_dim)
        self.t_mlp = nn.Sequential(nn.Linear(cfg.t_dim, cfg.t_dim), nn.SiLU(), nn.Linear(cfg.t_dim, cfg.t_dim))
        # per-level head dims (unet_head_dims "44,88,88,88"): narrower heads at the narrow fine levels keep >= 2 heads.
        self.head_dims = [min(int(h), W[l]) for l, h in enumerate(head_dims)] if head_dims \
            else [min(cfg.head_dim, W[l]) for l in range(L + 1)]
        assert len(self.head_dims) == L + 1 and all(W[l] % self.head_dims[l] == 0 for l in range(L + 1)), \
            (self.head_dims, W)
        mk = lambda l, n, skip=False: nn.ModuleList(
            JointBlock(W[l], self.head_dims[l], cfg.t_dim, cfg.expansion, cfg.dropout, skip=skip,
                       token_adaln=self.token_adaln) for _ in range(n))
        if self.token_adaln:
            # unet_token_adaln: per-token conditioning vector = material embedding + physics MLP -> SiLU
            self.obj_mat_emb = nn.Embedding(cfg.num_materials, cfg.t_dim)
            self.obj_phys = nn.Sequential(nn.Linear(3, cfg.t_dim), nn.SiLU(), nn.Linear(cfg.t_dim, cfg.t_dim))
        self.enc = nn.ModuleList(mk(l, cfg.enc_depths[l]) for l in range(L + 1))     # enc[L] = bottleneck
        self.dec = nn.ModuleList(mk(l, cfg.dec_depths[l], skip=self.long_skip) for l in range(L))
        assert not self.long_skip or all(cfg.dec_depths[l] <= cfg.enc_depths[l] for l in range(L)), \
            "unet_tube_long_skip pairs every decoder block with an encoder block of its level: dec depth <= enc depth"
        self.pools = nn.ModuleList(Pool(cfg, W[l], W[l + 1]) for l in range(L))
        self.unpools = nn.ModuleList(Unpool(cfg, W[l + 1], W[l]) for l in range(L))
        self.head = nn.Sequential(nn.LayerNorm(W[0]), nn.Linear(W[0], W[0]), nn.SiLU(), nn.Linear(W[0], 3))

        # -- RoPE tables ----------------------------------------------------------------------------------
        if cfg.stratified:
            # reference count ratio per level for the identity band period: the largest table budget with
            # L0 = 4 x L1 (the quarter convention). The RATIO, not an actual object, is what matters.
            n1_ref = max(cfg._l1_candidates())
            per_obj = [4 * n1_ref] + cfg.ladder(n1_ref)
        else:
            per_obj = [cfg.points]
            for st in cfg.strides:
                per_obj.append(per_obj[-1] // st)
        self.ropes, self.identity_levels, self.identity_periods_m, self.level_axes = nn.ModuleList(), [], [], []
        for l in range(L + 1):
            freqs = None
            # DERIVED budget only: frame0_pos spends 6 dims per band out of the pass-through remainder, which a
            # head narrower than 64 does not have -- those levels fall back to no identity axis. An EXPLICIT
            # unet_rope_budget states the dims itself, so it keeps the identity axis at every head dim.
            level_identity = self.identity
            if self.identity == "frame0_pos" and not rope_budget_dims and self.head_dims[l] < 64:
                level_identity = "none"
            if level_identity == "frame0_pos":
                self.identity_levels.append(l)
                # fine period scales with the level's token spacing (~ sqrt of the per-object count ratio on a
                # surface); coarse period fixed. Periods in metres -> rad per geometry unit (coords are m / geo_r).
                fine = float(rope_identity_periods_m[0]) * math.sqrt(per_obj[0] / per_obj[l])
                coarse = float(rope_identity_periods_m[1])
                self.identity_periods_m.append((fine, coarse))
                freqs = sorted([2 * math.pi * self.geo_r / coarse, 2 * math.pi * self.geo_r / fine])
            if rope_budget_dims:
                assert level_identity == "frame0_pos", "unet_rope_budget needs unet_rope_identity=frame0_pos at every level"
                groups, axes = rope_budget(self.head_dims[l], cfg.rope_f_min, cfg.rope_f_max, rope_budget_dims,
                                           rope_t_max_period, freqs, t_min_period=rope_t_min_period, t_base=rope_t_base)
            else:
                groups, axes = rope_groups(self.head_dims[l], cfg.rope_f_min, f_max=cfg.rope_f_max,
                                           fraction=rope_fraction, t_dims=rope_t_dims,
                                           t_max_period=rope_t_max_period, identity=level_identity,
                                           identity_max_period=rope_identity_max_period, identity_freqs=freqs)
            self.level_axes.append(axes)
            self.ropes.append(RopeAxes(self.head_dims[l], groups))

        # -- activation checkpointing ----------------------------------------------------------------------
        # ckpt_levels / ckpt_seams (level / seam indices; seam l is pool l -> l+1 and its unpool, seam -1 =
        # stem + head). None = the UNet knobs: levels >= cfg.ckpt_from_level, and with cfg.ckpt_seams every
        # seam from that level plus stem/head when it is 0.
        if ckpt_levels is None:
            ckpt_levels = list(range(cfg.ckpt_from_level, L + 1)) if cfg.ckpt_from_level >= 0 else []
        if ckpt_seams is None:
            ckpt_seams = ([l for l in range(L) if l >= cfg.ckpt_from_level] + ([-1] if cfg.ckpt_from_level == 0 else [])) \
                if (cfg.ckpt_seams and cfg.ckpt_from_level >= 0) else []
        self.ckpt_levels = frozenset(int(v) for v in ckpt_levels)
        self.ckpt_seam_set = frozenset(int(v) for v in ckpt_seams)
        assert all(0 <= l <= L for l in self.ckpt_levels) and all(-1 <= l < L for l in self.ckpt_seam_set), \
            (sorted(self.ckpt_levels), sorted(self.ckpt_seam_set))
        if cfg.ckpt_ingraph:        # the selected levels' blocks checkpoint inside their own (compilable) forward
            for l in self.ckpt_levels:
                for blk in list(self.enc[l]) + (list(self.dec[l]) if l < len(self.dec) else []):
                    blk.ckpt_ingraph = True

    # -- geometry ------------------------------------------------------------------------------------------
    @staticmethod
    @torch.compiler.disable
    def _build_provider(cfg, pos, pos0, obj_id, mat_id, particle_rigid):
        return GraphProvider(cfg, pos, pos0, obj_id, mat_id, particle_rigid)

    @staticmethod
    @torch.compiler.disable
    def curve_orders(pos0, K, obj=None):
        """Morton orders on the clean frame-0 positions of one level: K x (order, inverse) [B, n] long.
        Order k uses axis permutation _PERMS[k]. The rank along order 0 is the curve_rank identity coordinate.
        obj [B, n] (within-object levels): the key is (object id, Morton code on that object's OWN bbox), so the
        order visits objects one after another and a group never straddles objects (the last group of each
        object is padded by _run_level)."""
        if obj is None:
            mn, mx = pos0.amin(dim=1, keepdim=True), pos0.amax(dim=1, keepdim=True)
        else:
            O = int(obj.max()) + 1
            idx = obj[..., None].expand(-1, -1, 3)
            mn = pos0.new_full((pos0.shape[0], O, 3), float("inf")).scatter_reduce(1, idx, pos0, "amin").gather(1, idx)
            mx = pos0.new_full((pos0.shape[0], O, 3), float("-inf")).scatter_reduce(1, idx, pos0, "amax").gather(1, idx)
        u = (pos0 - mn) / (mx - mn + 1e-6)
        out = []
        for k in range(K):
            code = _morton3(u[..., _PERMS[k % len(_PERMS)]])
            if obj is not None:
                code = code | (obj.long() << 30)                    # 30-bit Morton codes: object id is the major key
            order = code.argsort(dim=1)
            out.append((order, order.argsort(dim=1)))
        return out

    @staticmethod
    @torch.compiler.disable
    def _level_obj(provider, l):
        """[B, n_l] object id of every token at level l (the provider lays tokens out as contiguous per-object blocks)."""
        ids = torch.cat([torch.full((n,), o, dtype=torch.long, device=provider.pos0[l].device)
                         for o, (_, n) in enumerate(provider.segs[l])])
        return ids[None].expand(provider.pos0[l].shape[0], -1)

    def _coords(self, pos_l, idc_l, l=None):
        """[B, T, n, 3] noisy positions (geometry units) + row index + identity coordinate -> [B, T, n, A].
        idc_l: [B, n] frame-0 curve rank (curve_rank) or [B, n, 3] clean frame-0 position (frame0_pos)."""
        B, T, n, _ = pos_l.shape
        parts = [pos_l, torch.arange(T, device=pos_l.device, dtype=pos_l.dtype)[None, :, None, None].expand(B, T, n, 1)]
        if self.identity == "curve_rank":
            parts.append(idc_l.to(pos_l.dtype)[:, None, :, None].expand(B, T, n, 1))
        elif self.identity == "frame0_pos" and (l is None or l in self.identity_levels):
            parts.append(idc_l.to(pos_l.dtype)[:, None].expand(B, T, n, 3))
        return torch.cat(parts, dim=-1)

    @torch.no_grad()
    @torch.compiler.disable
    def _draw_groups(self, provider, l):
        """unet_tube_grouping=random: `group_draws` independent random groupings of level l's tokens. Within-object
        levels group each object on its own (objects of equal size are batched into one call); other levels group
        the whole scene. Returns (list of global labels [B, n], number of groups K, slots per group S)."""
        from models.tube_groups import random_cells
        P, D = self.tube[l], self.group_draws
        pos0 = provider.pos0[l]
        B, n, _ = pos0.shape
        segs = [(int(o), int(c)) for o, c in provider.segs[l]] if l in self.within_object else [(0, n)]
        ks = [-(-c // P) for _, c in segs]; offs = [sum(ks[:i]) for i in range(len(ks))]
        glab = torch.zeros(D, B, n, dtype=torch.long, device=pos0.device); S = 0
        for size in sorted({c for _, c in segs}):
            idx = [i for i, (_, c) in enumerate(segs) if c == size]
            X = torch.cat([pos0[:, segs[i][0]:segs[i][0] + size] for i in idx], dim=0).repeat(D, 1, 1)
            lab, k, cap = random_cells(X, P)
            lab = lab.reshape(D, len(idx), B, size); S = max(S, cap)
            for j, i in enumerate(idx):
                glab[:, :, segs[i][0]:segs[i][0] + size] = lab[:, j] + offs[i]
        return [glab[d] for d in range(D)], sum(ks), S

    # -- one level -----------------------------------------------------------------------------------------
    _SLOT_CACHE = {}

    @staticmethod
    def tube_slots(n, P, counts=None, device=None):
        """Slot table of a tubed level under `curves`: the level's n tokens in curve order are laid into
        consecutive tubes of P slots. counts (within-object levels): the per-object token counts in curve order,
        each object padded to a multiple of P on its own; None: the scene padded once at its end.
        Returns (slot2pos [S] long: curve position of each slot, -1 = pad; pos2slot [n] long; n_tubes).
        Depends only on the layout, so it is built once per (n, P, counts, device) and cached."""
        key = (int(n), int(P), None if counts is None else tuple(int(c) for c in counts),
               None if device is None else str(device))
        hit = TubeJointCore._SLOT_CACHE.get(key)
        if hit is not None:
            return hit
        parts = [n] if counts is None else [int(c) for c in counts]
        slot2pos, off = [], 0
        for c in parts:
            k = -(-c // P)
            row = torch.full((k * P,), -1, dtype=torch.long)
            row[:c] = torch.arange(off, off + c)
            slot2pos.append(row); off += c
        slot2pos = torch.cat(slot2pos)
        pos2slot = torch.full((n,), -1, dtype=torch.long)
        real = slot2pos >= 0
        pos2slot[slot2pos[real]] = torch.nonzero(real).flatten()
        if device is not None:
            slot2pos, pos2slot = slot2pos.to(device), pos2slot.to(device)
        out = (slot2pos, pos2slot, slot2pos.numel() // P)
        TubeJointCore._SLOT_CACHE[key] = out
        TubeJointCore._SLOT_CACHE[("padded",) + key] = bool(sum(-(-c // P) * P for c in parts) != n)
        return out

    def _run_level(self, l, blocks, x, pos_l, rank_l, orders_l, t_emb, n_reg: int = 0, counts_l=None,
                   cond_l=None, skips_in=None, collect: bool = False, i0: int = 0, groups_l=None, istep: int = 1):
        """x [B, T, n + n_reg, w] through the level's blocks: one whole-level call (tube 0) or one call per group.
        Trailing rows per frame: n_reg register rows (no position, pass-through RoPE). counts_l: per-object token
        counts (within-object tube levels), so every object's last group pads on its own; None = one scene padding.
        Pad slots exist only inside the attention calls (zero tokens masked as keys, dropped on the way out).
        cond_l [B, n, t_dim]: per-token material conditioning (unet_token_adaln), None = off; registers and pad
        slots get zeros (the block's adaln_obj has no bias, so zero = no contribution).
        skips_in: one [B, T, nt, w] tensor per block (long skips, laid out exactly like x), None = off.
        collect: also return every block's output in token order (the encoder side of the long skips).
        i0 / istep: which curve order or grouping draw block i uses -- (i0 + istep * i)."""
        if len(blocks) == 0:
            return (x, []) if collect else x
        outs = []
        B, T, nt, w = x.shape
        ck = (l in self.ckpt_levels and self.training and torch.is_grad_enabled())

        def run(blk, *a, **kw):
            for name in ("obj_cond", "skip"):        # off: keep the block call's plain signature
                if kw.get(name, 0) is None:
                    kw.pop(name)
            if ck and not getattr(blk, "ckpt_ingraph", False):      # in-graph blocks checkpoint themselves
                from torch.utils.checkpoint import checkpoint as _ckpt
                return _ckpt(blk, *a, use_reentrant=False, **kw)
            return blk(*a, **kw)

        n = nt - n_reg
        coords = self._coords(pos_l, rank_l, l)                                     # [B, T, n, A]
        P, rope = self.tube[l], self.ropes[l]
        if P == 0:
            cos, sin = rope.tables(coords.reshape(B, T * n, -1))                    # [B, 1, T*n, hd]
            if n_reg:                                                               # registers: unrotated
                hd = cos.shape[-1]
                c4, s4 = cos.reshape(B, 1, T, n, hd), sin.reshape(B, 1, T, n, hd)
                cos = torch.cat([c4, cos.new_ones(B, 1, T, n_reg, hd)], 3).reshape(B, 1, T * nt, hd)
                sin = torch.cat([s4, sin.new_zeros(B, 1, T, n_reg, hd)], 3).reshape(B, 1, T * nt, hd)
            z = x.reshape(B, T * nt, w)
            oc = cond_l
            if oc is not None and n_reg:                                            # registers: zero conditioning
                oc = torch.cat([oc, oc.new_zeros(B, n_reg, oc.shape[-1])], dim=1)
            for i, blk in enumerate(blocks):
                sk = None if skips_in is None else skips_in[i].reshape(B, T * nt, w)
                z = run(blk, z, cos, sin, t_emb, nt, obj_cond=oc, skip=sk)
                if collect:
                    outs.append(z.reshape(B, T, nt, w))
            return (z.reshape(B, T, nt, w), outs) if collect else z.reshape(B, T, nt, w)
        assert n_reg == 0, f"level {l}: registers cannot ride a tubed level"
        if groups_l is not None:                                                    # unet_tube_grouping=random
            return self._run_level_groups(l, blocks, x, coords, t_emb, cond_l, groups_l, run, skips_in,
                                          collect, i0, istep)
        slot2pos, pos2slot, k = self.tube_slots(n, P, counts_l, device=x.device)
        _key = (int(n), int(P), None if counts_l is None else tuple(int(c) for c in counts_l), str(x.device))
        padded = self._SLOT_CACHE[("padded",) + _key]

        def chunk(seq, size):                  # [B, T, k*size, C] (curve order) -> [B*k, T*size, C], k
            kk = seq.shape[2] // size
            return seq.reshape(B, T, kk, size, -1).permute(0, 2, 1, 3, 4).reshape(B * kk, T * size, -1), kk

        def unchunk(z, kk, size):              # inverse of chunk
            return z.reshape(B, kk, T, size, -1).permute(0, 2, 1, 3, 4).reshape(B, T, kk * size, -1)

        def temb(kk):                          # per-call copy of the per-row noise embeddings
            return t_emb[:, None].expand(B, kk, T, -1).reshape(B * kk, T, -1)

        def cchunk(c, size):                   # [B, k*size, C] (tube layout) -> [B*k, size, C]; None when off
            return None if c is None else c.reshape(B * (c.shape[1] // size), size, -1)

        # tables / masks / RoPE depend on (level, curve order) and this forward's positions: the level's encoder
        # and decoder calls share them (self._fwd_cache lives for one forward)
        shared = getattr(self, "_fwd_cache", None)
        cache = shared.setdefault(l, {}) if shared is not None else {}
        for i, blk in enumerate(blocks):
            k_ord = (i0 + istep * i) % self.K
            order, inv = orders_l[k_ord]
            sk_i = None if skips_in is None else skips_in[i]
            if not padded:
                if k_ord not in cache:
                    cache[k_ord] = rope.tables(chunk(gather_tokens(coords, order), P)[0])
                tabs = cache[k_ord]
                co = None if cond_l is None else cond_l.gather(1, order[..., None].expand(-1, -1, cond_l.shape[-1]))
                z, kk = chunk(gather_tokens(x, order), P)
                sks = None if sk_i is None else chunk(gather_tokens(sk_i, order), P)[0]
                x = gather_tokens(unchunk(run(blk, z, *tabs, temb(kk), P, obj_cond=cchunk(co, P), skip=sks), kk, P), inv)
                if collect:
                    outs.append(x)
                continue
            # padded path: the curve order and the slot table are COMPOSED into one index each way -- src [B, k*P]
            # reads the token-order stream [tokens | one zero row] straight into tube layout (pads read the zero
            # row), back [B, n] reads the real tokens out.
            if k_ord not in cache:
                flat = slot2pos[None].expand(B, -1)
                src = torch.where(flat < 0, torch.full_like(flat, n),
                                  torch.gather(order, 1, flat.clamp(min=0)))
                back = pos2slot[inv]                                                     # [B, n]
                mask = chunk((flat >= 0).reshape(B, 1, k * P, 1).expand(B, T, k * P, 1), P)[0][..., 0][:, None, None]
                c_ext = torch.cat([coords, coords.new_zeros(B, T, 1, coords.shape[-1])], dim=2)
                cache[k_ord] = (src, back, mask, rope.tables(chunk(gather_tokens(c_ext, src), P)[0]))
                if cond_l is not None:                                                   # pad slots read zeros
                    c_src = torch.cat([cond_l, cond_l.new_zeros(B, 1, cond_l.shape[-1])], dim=1)
                    cache[("cond", k_ord)] = cchunk(c_src.gather(1, src[..., None].expand(-1, -1, c_src.shape[-1])), P)
            src, back, mask, tabs = cache[k_ord]
            z, kk = chunk(gather_tokens(torch.cat([x, x.new_zeros(B, T, 1, w)], dim=2), src), P)
            sk_z = None if sk_i is None else chunk(gather_tokens(torch.cat([sk_i, sk_i.new_zeros(B, T, 1, w)], dim=2), src), P)[0]
            xs = unchunk(run(blk, z, *tabs, temb(kk), P, attn_mask=mask, obj_cond=cache.get(("cond", k_ord)),
                             skip=sk_z), kk, P)                                          # [B, T, k*P, w]
            x = gather_tokens(xs, back)                                                  # the n real tokens, token order
            if collect:
                outs.append(x)
        return (x, outs) if collect else x

    def _run_level_groups(self, l, blocks, x, coords, t_emb, cond_l, groups_l, run, skips_in, collect, i0, istep=1):
        """A tubed level under unet_tube_grouping=random: block i attends inside the groups of draw
        (i0 + istep * i) mod group_draws. Same slot / mask / long-skip / material-cond machinery as the padded
        curve tubes: tokens reach the group layout by one gather and come back by one."""
        from models.tube_groups import label_slots
        labels, K, S = groups_l
        B, T, n, w = x.shape
        shared = getattr(self, "_fwd_cache", None)
        cache = shared.setdefault(("groups", l), {}) if shared is not None else {}
        rope, outs = self.ropes[l], []

        def chunk(seq):                        # [B, T, K*S, C] (group layout) -> [B*K, T*S, C]
            return seq.reshape(B, T, K, S, -1).permute(0, 2, 1, 3, 4).reshape(B * K, T * S, -1)

        te = t_emb[:, None].expand(B, K, T, -1).reshape(B * K, T, -1)
        for i, blk in enumerate(blocks):
            d = (i0 + istep * i) % len(labels)
            if d not in cache:
                src, back = label_slots(labels[d], K, S)
                real = src != n                                                          # a layout without pads keeps flash
                mask = None if bool(real.all()) else real.reshape(B, K, 1, S).expand(B, K, T, S).reshape(B * K, 1, 1, T * S)
                c_ext = torch.cat([coords, coords.new_zeros(B, T, 1, coords.shape[-1])], dim=2)
                co_t = None
                if cond_l is not None:                                                   # pads read zeros
                    ce = torch.cat([cond_l, cond_l.new_zeros(B, 1, cond_l.shape[-1])], dim=1)
                    co_t = ce.gather(1, src[..., None].expand(-1, -1, ce.shape[-1])).reshape(B * K, S, -1)
                cache[d] = (src, back, mask, rope.tables(chunk(gather_tokens(c_ext, src))), co_t)
            src, back, mask, tabs, co_t = cache[d]
            sk_z = None
            if skips_in is not None:
                sk = skips_in[i]
                sk_z = chunk(gather_tokens(torch.cat([sk, sk.new_zeros(B, T, 1, w)], dim=2), src))
            xs = run(blk, chunk(gather_tokens(torch.cat([x, x.new_zeros(B, T, 1, w)], dim=2), src)), *tabs, te, S,
                     attn_mask=mask, obj_cond=co_t, skip=sk_z)
            xs = xs.reshape(B, K, T, S, -1).permute(0, 2, 1, 3, 4).reshape(B, T, K * S, -1)
            x = gather_tokens(xs, back)
            if collect:
                outs.append(x)
        return (x, outs) if collect else x

    # -- the U ---------------------------------------------------------------------------------------------
    def forward(self, feats, pos, pos0, obj_id, mat_id, t, n_context: int, pose=None, particle_rigid=None, phys=None):
        """Returns [B, T, N, 3] (all rows; the wrapper slices the context rows)."""
        assert pose is None, "tube_joint: pose tokens are not supported (use data.rigid_poses=false)"
        B, T, N, _ = feats.shape
        L = len(self.cfg.strides)
        provider = self._build_provider(self.cfg, pos, pos0, obj_id, mat_id, particle_rigid)
        obj_of = lambda l: self._level_obj(provider, l) if l in self.within_object else None
        groups = [self._draw_groups(provider, l) if (self.grouping == "random" and self.tube[l] > 0) else None
                  for l in range(L + 1)]
        # curves are built only where they are read: a tubed level under `curves`, or any level when the identity
        # coordinate IS the curve rank
        need_order = lambda l: self.identity == "curve_rank" or (self.grouping == "curves" and self.tube[l] > 0)
        orders = [self.curve_orders(provider.pos0[l], self.K, obj=obj_of(l)) if need_order(l) else None
                  for l in range(L + 1)]
        if self.identity == "frame0_pos":
            ranks = [provider.pos0[l] for l in range(L + 1)]            # clean frame-0 xyz per level token
        else:                                                          # inverse of order 0 = the curve rank
            ranks = [orders[l][0][1] if orders[l] is not None else None for l in range(L + 1)]
        t_emb = self.t_mlp(self.t_embed(t.reshape(-1)).reshape(B, T, -1))

        # cfg.ckpt_seams: checkpoint the stem, the head and the pool / unpool seams too -- the activations
        # outside the blocks, which dominate at L0 on a large scene (the UNet's knob, same semantics)
        from torch.utils.checkpoint import checkpoint as _ckpt
        _grad = self.training and torch.is_grad_enabled()
        seam_ck = lambda l: _grad and l in self.ckpt_seam_set
        _ck_ends = seam_ck(-1)
        x = (_ckpt(self.stem, feats, use_reentrant=False) if _ck_ends else self.stem(feats)) + self.mat_emb(mat_id)[:, None]

        # unet_token_adaln: per-token material + physics vector, carried to every level by the anchors' particle ids
        cond = [None] * (L + 1)
        if self.token_adaln:
            assert phys is not None, "unet_token_adaln: the wrapper must pass phys [B, N, 3]"
            c0 = nn.functional.silu(self.obj_mat_emb(mat_id) + self.obj_phys(phys.to(self.obj_phys[0].weight.dtype)))
            cond = [c0] + [c0.gather(1, provider.ids[l][..., None].expand(-1, -1, c0.shape[-1])) for l in range(1, L + 1)]
        # within-object tube levels: the per-object token counts, so each object's last group pads on its own
        counts_of = lambda l: [n_ for _, n_ in provider.segs[l]] if (l in self.within_object and self.tube[l]) else None
        nreg = lambda l: self.n_reg.get(l, 0)
        # registers present in the token stream BETWEEN levels: only the carried set travels the seams; a
        # per_level set is appended and stripped inside run(), so the seams never see it.
        n_tail = lambda l: nreg(l) if self.reg_mode == "carried" else 0

        def run(l, blocks, z, **kw):
            """One run of blocks at level l, with this level's per_level registers appended and stripped again."""
            nr = nreg(l)
            fresh = self.reg_mode == "per_level" and nr > 0
            if fresh:
                z = torch.cat([z, self.reg_init[str(l)][None, None].to(z.dtype).expand(B, T, -1, -1)], dim=2)
            out = self._run_level(l, blocks, z, provider.pos[l], ranks[l], orders[l], t_emb, nr,
                                  counts_of(l), cond[l], groups_l=groups[l], **kw)
            if not fresh:
                return out
            # the COLLECTED outputs keep their register rows: a decoder long skip is laid out like the decoder
            # stream, which carries that level's own fresh set in the same trailing slots.
            if kw.get("collect"):
                z_, outs_ = out
                return z_[:, :, :-nr], outs_
            return out[:, :, :-nr]

        self._fwd_cache = {}
        skips, enc_outs = [], {}
        for l in range(L):
            if self.reg_mode == "carried" and l == self.reg_l0:                     # birth: one copy per row
                x = torch.cat([x, self.reg_init[None, None].to(x.dtype).expand(B, T, -1, -1)], dim=2)
            out = run(l, self.enc[l], x, collect=self.long_skip)
            x, enc_outs[l] = out if self.long_skip else (out, None)
            skips.append(x)
            xp, xr = (x[:, :, :-n_tail(l)], x[:, :, -n_tail(l):]) if n_tail(l) else (x, None)
            if seam_ck(l):
                xp = _ckpt(lambda a, t_, _l=l: self.pools[_l](a, provider.pos[_l], provider, _l, t_), xp, t_emb,
                           use_reentrant=False)
            else:
                xp = self.pools[l](xp, provider.pos[l], provider, l, t_emb)
            x = torch.cat([xp, self.reg_up[str(l)](xr)], dim=2) if xr is not None else xp
        if self.reg_mode == "carried" and L == self.reg_l0:
            x = torch.cat([x, self.reg_init[None, None].to(x.dtype).expand(B, T, -1, -1)], dim=2)
        x = run(L, self.enc[L], x)
        for l in reversed(range(L)):
            nr = n_tail(l + 1)
            xp, xr = (x[:, :, :-nr], x[:, :, -nr:]) if nr else (x, None)
            skp = skips[l][:, :, :-n_tail(l)] if n_tail(l) else skips[l]
            if seam_ck(l):
                x = _ckpt(lambda s_, c_, t_, _l=l: self.unpools[_l](s_, c_, provider, _l, t_), skp, xp, t_emb,
                          use_reentrant=False)
            else:
                x = self.unpools[l](skp, xp, provider, l, t_emb)
            if xr is not None and n_tail(l):            # width back down; dropped below the birth level
                x = torch.cat([x, self.reg_down[str(l)](xr)], dim=2)
            pair = [enc_outs[l][len(self.enc[l]) - 1 - i] for i in range(len(self.dec[l]))] if self.long_skip else None
            x = run(l, self.dec[l], x, skips_in=pair,
                    i0=(len(self.enc[l]) - 1) if self.dec_mirror else 0, istep=-1 if self.dec_mirror else 1)
        self._fwd_cache = None
        head = (lambda z: _ckpt(self.head, z, use_reentrant=False)) if _ck_ends else self.head
        if self.fp32_head:
            with torch.autocast(x.device.type, enabled=False):
                return head(x.float())
        return head(x)
