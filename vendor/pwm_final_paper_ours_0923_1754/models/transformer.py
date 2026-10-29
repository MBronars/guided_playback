"""
TrajectoryTransformer: Clean trajectory prediction model.

Architecture:
1. Embed inputs (points, time, actions, physics)
2. FPS + KNN for efficient attention (N → L)
3. Encode to latent space via KNN attention
4. Process through transformer blocks with RoPE
5. Decode back to full resolution (L → N)
6. Output prediction
"""

from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from pytorch3d.ops import sample_farthest_points, knn_points
from diffusers.models.attention import Attention

from models.components.embeddings import (
    SinusoidalTimeEmbed,
    SimplePointEmbed,
    RotaryPositionEncoding,
    RotaryPositionEncoding3D,
    FourierSpatialEmbed,
    TemporalEmbed,
)
from models.components.attention import (
    RotaryAttnProcessor,
    KNNAttention,
    TransformerBlock,
    create_sink_window_mask,
)
from data.batch import TrajectoryBatch


from models.config import ModelConfig  # re-export for existing callers


class TrajectoryTransformer(nn.Module):
    """
    Clean trajectory prediction model.

    Takes a TrajectoryBatch and diffusion timestep, returns predicted noise/x0.

    Architecture:
    - FPS + KNN for N → L compression
    - Encoder: KNN attention to aggregate neighbors
    - Transformer: temporal cross/self + spatial self attention with RoPE
    - Decoder: cross-attention L → N + temporal attention
    - Output: LayerNorm → Linear → 3D positions
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        if not cfg.all_relative:
            raise ValueError("Unified Transformer requires all_relative=True")
        self.cfg = cfg
        d = cfg.dim
        head_dim = d // cfg.heads

        print(f"[Model] use_rotary: {cfg.use_rotary}")
        print(f"[Model] use_knn_attention: {cfg.use_knn_attention}")
        print(f"[Model] use_stratified_encoding: {cfg.use_stratified_encoding}")
        print(f"[Model] use_stratified_decoding: {cfg.use_stratified_decoding}")
        print(f"[Model] use_windowed_temporal: {cfg.use_windowed_temporal}")
        if cfg.use_windowed_temporal:
            print(f"[Model]   window_size: {cfg.temporal_window_size}, use_sink: {cfg.temporal_use_sink}")
        print(f"[Model] causal_temporal: {cfg.causal_temporal}")

        # =====================================================================
        # Embeddings: xyz + [E, nu, rho] = 3 + 3 = 6 dims
        # =====================================================================
        delta_input_dim = 7 if cfg.concat_anchor_z else 6
        self.delta_physics_embed = nn.Sequential(
            nn.Linear(delta_input_dim, d),  # [dx, dy, dz, (anchor_z,) log10(E), nu, rho_scaled]
            nn.SiLU(),
            nn.Linear(d, d),
        )
        if not cfg.no_action_cond:
            self.action_embed = SimplePointEmbed(out_dim=d)  # Actions: just 3D
            self.no_action_token = nn.Parameter(torch.randn(d) * 0.02)

        # Material type embedding (categorical, additive)
        if cfg.num_mat_types > 0:
            self.mat_type_embed = nn.Embedding(cfg.num_mat_types, d)

        # Actuatable embedding (binary, additive)
        if cfg.embed_actuatable:
            self.actuatable_embed = nn.Embedding(2, d)  # 0=passive, 1=actuatable

        # Time embedding
        self.time_embed = SinusoidalTimeEmbed(dim=d)
        self.time_mlp = nn.Sequential(
            nn.Linear(d, d),
            nn.SiLU(),
            nn.Linear(d, d),
        )

        # =====================================================================
        # Position Encodings
        # =====================================================================
        self.rope_patch_pool = False
        if cfg.use_rotary:
            # Banded RoPE: master flag applies recommended defaults; individual knobs
            # override. All off => legacy RoPE (rotary_dim=None, f_min/f_max=None).
            _b = cfg.banded_rope
            eff_f_min = cfg.rope_f_min if cfg.rope_f_min is not None else (0.5 if _b else None)
            eff_f_max = cfg.rope_f_max if cfg.rope_f_max is not None else (64.0 if _b else None)
            eff_frac = cfg.rope_rotary_fraction if cfg.rope_rotary_fraction is not None else (0.75 if _b else None)
            self.rope_patch_pool = cfg.rope_patch_pool if cfg.rope_patch_pool is not None else _b
            rotary_dim = int(round(head_dim * eff_frac)) if eff_frac is not None else None
            # Rotary position encodings (multiplicative, applied in attention)
            self.rotary_spatial = RotaryPositionEncoding3D(
                dim=head_dim, max_period=cfg.rope_spatial_base,
                rotary_dim=rotary_dim, f_min=eff_f_min, f_max=eff_f_max,
            )
            self.rotary_temporal = RotaryPositionEncoding(dim=head_dim)
            if _b or eff_f_min is not None or rotary_dim is not None:
                print(f"[Model] banded-RoPE: f_min={eff_f_min} f_max={eff_f_max} "
                      f"rotary_dim={self.rotary_spatial.feature_dim}/{head_dim} "
                      f"patch_pool={self.rope_patch_pool}")
        else:
            # Additive position encodings (added to token embeddings)
            self.spatial_embed = FourierSpatialEmbed(out_dim=d, num_bands=6)
            self.temporal_embed = TemporalEmbed(out_dim=d, max_len=100)

        # Decoder query identity: clean frame-0 anchor position -> query value (opt-in).
        if cfg.decoder_anchor_identity:
            self.anchor_identity_embed = nn.Sequential(
                nn.Linear(3, d), nn.SiLU(), nn.Linear(d, d))

        # Decoder query self-attn: spatial+temporal self-attn on queries before cross-attn (opt-in).
        if cfg.decoder_query_self_attn:
            self.decoder_pre_spatial_norm = nn.LayerNorm(d)
            self.decoder_pre_spatial_attn = Attention(
                query_dim=d, heads=cfg.heads, dim_head=head_dim, dropout=cfg.dropout, bias=True)
            self.decoder_pre_temporal_norm = nn.LayerNorm(d)
            self.decoder_pre_temporal_attn = Attention(
                query_dim=d, heads=cfg.heads, dim_head=head_dim, dropout=cfg.dropout, bias=True)

        # =====================================================================
        # Encoder (N → L)
        # =====================================================================
        if not cfg.bypass_compression:
            self.norm_queries = nn.LayerNorm(d)
            self.norm_kv = nn.LayerNorm(d)

        if cfg.use_knn_attention:
            # KNN attention: each query attends to K nearest neighbors
            self.knn_attn = KNNAttention(
                dim=d,
                heads=cfg.heads,
                dim_head=head_dim,
                dropout=cfg.dropout,
            )
        elif not cfg.bypass_compression:
            # Standard cross-attention: each query attends to ALL N points
            self.encoder_cross_attn = Attention(
                query_dim=d,
                cross_attention_dim=d,
                heads=cfg.heads,
                dim_head=head_dim,
                dropout=cfg.dropout,
                bias=True,
            )
            
        # =====================================================================
        # Transformer Blocks
        # =====================================================================
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=d,
                heads=cfg.heads,
                dropout=cfg.dropout,
            )
            for _ in range(cfg.depth)
        ])

        # =====================================================================
        # Decoder (L → N)
        # =====================================================================
        if not cfg.bypass_compression:
            self.decoder_cross_attn = Attention(
                query_dim=d,
                cross_attention_dim=d,
                heads=cfg.heads,
                dim_head=head_dim,
                dropout=cfg.dropout,
                bias=True,
            )
            self.decoder_norm = nn.LayerNorm(d)
            self.attn_processor = RotaryAttnProcessor()

            # Temporal attention for coherence
            self.decoder_temporal_attn = Attention(
                query_dim=d,
                heads=cfg.heads,
                dim_head=head_dim,
                dropout=cfg.dropout,
                bias=True,
            )
            self.decoder_temporal_norm = nn.LayerNorm(d)

        # =====================================================================
        # Output Head
        # =====================================================================
        self.final_norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, 3)

        if cfg.pose_native:
            # Tier-2 v2: per-(rigid body, frame) pose tokens carrying the NOISY pose
            # delta (part of the diffusion variable). Token = embed(normalized Δ6)
            # + template-geometry summary + type vector; the pose-action skip rides
            # the block's action channel; zero-init v-head (identity start, the v9
            # convention).
            self.pose_token_embed = nn.Sequential(nn.Linear(6, d), nn.SiLU(), nn.Linear(d, d))
            self.pose_geom_embed = nn.Sequential(nn.Linear(3, 64), nn.SiLU(), nn.Linear(64, d))
            self.pose_action_skip = nn.Linear(6, d)
            self.pose_no_action_token = nn.Parameter(torch.randn(d) * 0.02)
            self.pose_token_type = nn.Parameter(torch.randn(d) * 0.02)
            if cfg.articulated_bodies:
                # each body's joint description (free / fixed / prismatic / revolute, axis and
                # pivot in the parent frame), so the model reads a gripper's structure
                self.pose_joint_type = nn.Parameter(torch.randn(4, d) * 0.02)
                self.pose_joint_geom = nn.Sequential(nn.Linear(6, 64), nn.SiLU(), nn.Linear(64, d))
            self.pose_v_head = nn.Linear(d, 6)
            nn.init.zeros_(self.pose_v_head.weight)
            nn.init.zeros_(self.pose_v_head.bias)

        # Cache for temporal attention masks (created lazily)
        self._temporal_mask_cache: Dict[Tuple[int, torch.device, torch.dtype], torch.Tensor] = {}
        self._causal_mask_cache: Dict[Tuple[int, torch.device, torch.dtype], torch.Tensor] = {}

    def _get_causal_temporal_mask(
        self,
        num_frames: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        """
        Get or create causal mask for temporal self-attention in transformer blocks.

        Returns:
            [F, F] attention mask (0.0 = attend, -inf = blocked), or None if disabled
        """
        if not self.cfg.causal_temporal:
            return None

        cache_key = (num_frames, device, dtype)
        if cache_key not in self._causal_mask_cache:
            # Lower-triangular: frame t attends to frames 0..t only
            mask = torch.zeros(num_frames, num_frames, device=device, dtype=dtype)
            mask.masked_fill_(
                ~torch.tril(torch.ones(num_frames, num_frames, device=device, dtype=torch.bool)),
                float('-inf'),
            )
            self._causal_mask_cache[cache_key] = mask
        return self._causal_mask_cache[cache_key]

    def _get_temporal_mask(
        self,
        num_frames: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        """
        Get or create temporal attention mask for windowed attention.

        The mask is cached by (num_frames, device, dtype) since it's static.

        Args:
            num_frames: Number of future frames F
            device: Target device
            dtype: Target dtype

        Returns:
            [F, F] attention mask (0.0 = attend, -inf = blocked), or None if disabled
        """
        if not self.cfg.use_windowed_temporal:
            return None

        cache_key = (num_frames, device, dtype)
        if cache_key not in self._temporal_mask_cache:
            self._temporal_mask_cache[cache_key] = create_sink_window_mask(
                num_frames=num_frames,
                window_size=self.cfg.temporal_window_size,
                use_sink=self.cfg.temporal_use_sink,
                device=device,
                dtype=dtype,
            )
        return self._temporal_mask_cache[cache_key]

    def _get_decoder_temporal_mask(
        self,
        num_frames: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        """
        Get combined temporal mask for decoder: windowed + causal.

        Both masks use 0.0=attend, -inf=blocked, so taking element-wise min
        gives the intersection (attend only where BOTH masks allow).
        """
        windowed = self._get_temporal_mask(num_frames, device, dtype)
        causal = self._get_causal_temporal_mask(num_frames, device, dtype)

        if windowed is not None and causal is not None:
            return torch.minimum(windowed, causal)
        return windowed if windowed is not None else causal
    
    def forward(
        self,
        batch: TrajectoryBatch,
        noisy_future: torch.Tensor,                          # [B, F, N, 3]
        timestep: torch.Tensor,                              # [B] or [B, F]
        noisy_context: Optional[torch.Tensor] = None,        # [B, C_ctx, N, 3] normalized, all_relative only
        context_timesteps: Optional[torch.Tensor] = None,    # [B, C_ctx]
        noisy_pose: Optional[torch.Tensor] = None,           # [B, F, K_rigid, 6] normalized (pose_native)
    ) -> torch.Tensor:
        """Predict the per-particle trajectory target. Returns [B, F, N, 3];
        under pose_native with noisy_pose given, returns the tuple
        (particle_v [B,F,N,3] or None when the batch has no deformable rows,
         pose_v [B,F,K_rigid,6]). Rigid rows of the incoming noisy_future are
        IGNORED under pose_native — they are rebuilt from the noisy pose."""
        if batch.prediction_target != "delta":
            raise ValueError("Unified Transformer supports delta targets only")
        B = batch.batch_size
        F = batch.num_future_frames
        N = batch.num_particles
        C = batch.context_len
        L = self.cfg.num_latents
        k = self.cfg.k_neighbors
        d = self.cfg.dim
        device = batch.device

        # Context is represented by C-1 transitions; frame zero stays clean.
        # Set once here — everything downstream uses this, no more scattered C_ctx = C - 1 assignments
        C_ctx = C - 1

        # =====================================================================
        # 1. Time embedding — one noise level per future frame
        # =====================================================================
        if timestep.ndim == 1:
            timestep = timestep.unsqueeze(1).expand(-1, F)
        if timestep.shape != (B, F):
            raise ValueError(f"Expected timestep [B] or [B,F], got {tuple(timestep.shape)}")
        t_flat = timestep.reshape(B * F)
        t_emb = self.time_mlp(self.time_embed(t_flat)).reshape(B, F, d)

        # =====================================================================
        # 1b. Tier-2 v2: rebuild rigid rows of noisy_future from the NOISY pose
        #     (templates @ integrate(noisy Δ)) BEFORE positions/KNN/RoPE — the
        #     uniform encoding then sees a pose-consistent rigid scene at every
        #     noise level (v9's self-consistency, generalized).
        # =====================================================================
        pose_pack = None
        if self.cfg.pose_native and noisy_pose is not None:
            if batch.joint_type is not None and not self.cfg.articulated_bodies:
                raise ValueError("the batch carries articulated joints but model.articulated_bodies is off; "
                                 "build its batches with articulated=False")
            noisy_future, pose_pack = self._pose_native_inputs(batch, noisy_future, noisy_pose)
        elif (self.cfg.pose_native and batch.body_type is not None
              and bool((batch.body_type[0] == 1).any())):
            # Rigid bodies present but no noisy_pose: under pose_native the rigid
            # particle rows were NEVER supervised — the legacy path would emit
            # garbage for them silently. Fail loudly instead.
            raise RuntimeError(
                "pose_native checkpoint received a rigid-body batch without "
                "noisy_pose — use the pose-native inference path "
                "(run_diffusion_inference_df with emit_sidecar batches).")

        # =====================================================================
        # 2. Compute positions for KNN and RoPE
        #    all_relative: integrate from frame 0 through context then future deltas
        #                  anchor is always frame 0, never noised
        #    outputs: ctx_positions [B, C_ctx, N, 3], future_positions [B, F, N, 3]
        # =====================================================================
        frame0 = batch.context[:, 0]  # [B, N, 3] — always clean, never noised

        # Integrate context deltas from frame0
        # If context is noised, denormalize the noisy version; else use clean batch deltas
        if noisy_context is not None:
            ctx_deltas_raw = batch.delta_stats.denormalize(noisy_context)  # [B, C_ctx, N, 3]
        else:
            ctx_deltas_raw = batch.context_deltas  # [B, C_ctx, N, 3]
        ctx_positions = frame0.unsqueeze(1) + torch.cumsum(ctx_deltas_raw, dim=1)  # [B, C_ctx, N, 3]

        # Integrate future deltas from last context position (continuous chain from frame0)
        last_ctx_pos = ctx_positions[:, -1]  # [B, N, 3]
        future_deltas_raw = batch.prediction_stats.denormalize(noisy_future)  # [B, F, N, 3]
        future_positions = last_ctx_pos.unsqueeze(1) + torch.cumsum(future_deltas_raw, dim=1)  # [B, F, N, 3]

        # Normalize positions for RoPE: map to [-1, 1]. The spatial base frequency
        # (rope_spatial_base) controls discrimination — no additional scaling needed.
        if self.cfg.use_rotary and batch.position_stats is not None:
            rope_scale = self.cfg.rope_position_scale  # legacy, default 1.0
            rope_future_positions = (batch.position_stats.normalize(future_positions) * rope_scale).detach()
            rope_ctx_positions = (batch.position_stats.normalize(ctx_positions) * rope_scale).detach()
        else:
            rope_future_positions = future_positions
            rope_ctx_positions = ctx_positions

        # =====================================================================
        # 3. FPS + KNN (global encoding path only)
        #    Uses ctx_positions and future_positions already computed in step 2
        #    ctx_positions is [B, C_ctx, N, 3]
        # =====================================================================
        fps_idx = None
        knn_idx_future = None
        knn_idx_context = None
        decode_info = None

        if not self.cfg.bypass_compression and not self.cfg.use_stratified_encoding:
            anchor = batch.anchor  # [B, N, 3]
            _, fps_idx = sample_farthest_points(anchor, lengths=batch.valid_lengths, K=L)

            if self.cfg.use_knn_attention:
                anchor_fps = self._gather(anchor, fps_idx)  # [B, L, 3]
                valid_exp = batch.valid_lengths.unsqueeze(1)  # [B, 1]

                if self.cfg.per_frame_knn:
                    # Future: query from FPS particles at their current frame positions [B, F, L, k]
                    fps_positions_f = self._gather_temporal(future_positions, fps_idx).reshape(B * F, L, 3)
                    _, knn_idx_future, _ = knn_points(
                        fps_positions_f,
                        future_positions.reshape(B * F, N, 3),
                        lengths2=valid_exp.expand(-1, F).reshape(B * F),
                        K=k,
                    )
                    knn_idx_future = knn_idx_future.reshape(B, F, L, k)

                    # Context: query from FPS particles at their current context frame positions
                    fps_positions_c = self._gather_temporal(ctx_positions, fps_idx).reshape(B * C_ctx, L, 3)
                    _, knn_idx_context, _ = knn_points(
                        fps_positions_c,
                        ctx_positions.reshape(B * C_ctx, N, 3),
                        lengths2=valid_exp.expand(-1, C_ctx).reshape(B * C_ctx),
                        K=k,
                    )
                    knn_idx_context = knn_idx_context.reshape(B, C_ctx, L, k)

                else:
                    # Anchor-frame KNN broadcast across all frames [B, L, k]
                    _, knn_idx, _ = knn_points(
                        anchor_fps, anchor,
                        lengths2=batch.valid_lengths,
                        K=k,
                    )
                    knn_idx_future = knn_idx
                    knn_idx_context = knn_idx

        # =====================================================================
        # 4. Embed all inputs
        #    physics_feat and future_emb always computed here (not inside conditionals)
        # =====================================================================
        physics_feat = self._prepare_physics_features(batch)  # [B, N, 3]

        # Anchor z feature: frame0's z-position, broadcast across frames and normalized
        # Used to give the model absolute height info for gravity-sensitive tasks
        if self.cfg.concat_anchor_z:
            anchor_z = frame0[:, :, 2:3]  # [B, N, 1]
            if batch.position_stats is not None:
                # Normalize z using the z-component of position stats
                z_min = batch.position_stats.min_val[2:3].to(device)
                z_max = batch.position_stats.max_val[2:3].to(device)
                anchor_z = 2 * (anchor_z - z_min) / (z_max - z_min + 1e-8) - 1
            anchor_z_future = anchor_z.unsqueeze(1).expand(-1, F, -1, -1)    # [B, F, N, 1]
            anchor_z_context = anchor_z.unsqueeze(1).expand(-1, C_ctx, -1, -1)  # [B, C_ctx, N, 1]

        # Future: concatenate [noisy_delta, (anchor_z,) physics] -> embed
        future_parts = [noisy_future, physics_feat.unsqueeze(1).expand(-1, F, -1, -1)]
        if self.cfg.concat_anchor_z:
            future_parts.insert(1, anchor_z_future)
        future_with_physics = torch.cat(future_parts, dim=-1)  # [B, F, N, 6 or 7]
        future_emb = self._embed_temporal(future_with_physics, self.delta_physics_embed)  # [B, F, N, d]

        # Context: embed normalized step deltas and physics.
        # noisy_context is pre-normalized; fallback to clean batch deltas if not provided
        if noisy_context is not None:
            ctx_deltas_norm = noisy_context  # [B, C_ctx, N, 3] already normalized
        else:
            ctx_deltas_norm = batch.delta_stats.normalize(batch.context_deltas) \
                            if batch.delta_stats is not None else batch.context_deltas
        ctx_parts = [ctx_deltas_norm, physics_feat.unsqueeze(1).expand(-1, C_ctx, -1, -1)]
        if self.cfg.concat_anchor_z:
            ctx_parts.insert(1, anchor_z_context)
        context_with_physics = torch.cat(ctx_parts, dim=-1)
        context_emb = self._embed_temporal(context_with_physics, self.delta_physics_embed)  # [B, C_ctx, N, d]

        # Context noise level embedding — passed to blocks for AdaLN modulation
        if context_timesteps is not None:
            ct_flat = context_timesteps.reshape(B * C_ctx)
            ctx_t_emb = self.time_mlp(self.time_embed(ct_flat)).reshape(B, C_ctx, d)
        else:
            ctx_t_emb = None

        # Material type embedding (additive, categorical)
        if batch.mat_type is not None and hasattr(self, 'mat_type_embed'):
            mat_emb = self.mat_type_embed(batch.mat_type).unsqueeze(1)  # [B, 1, N, d]
            future_emb = future_emb + mat_emb
            context_emb = context_emb + mat_emb

        # Actuatable embedding (additive, binary)
        if hasattr(self, 'actuatable_embed'):
            if batch.action_active is None:
                act_emb = self.actuatable_embed(batch.action_mask.long()).unsqueeze(1)  # [B, 1, N, d]
                future_emb = future_emb + act_emb
                context_emb = context_emb + act_emb
            else:
                future_emb = future_emb + self.actuatable_embed(batch.action_active.long())
                context_emb = context_emb + self.actuatable_embed(batch.context_action_active.long())

        # Particle action deltas share the prediction normalization. Rigid-body
        # actions additionally enter through the unified pose-token action stream.
        if batch.action_active is not None and self.cfg.no_action_cond:
            raise ValueError('Scheduled MPM drag requires particle action conditioning')
        if not self.cfg.no_action_cond:
            actions_normalized = batch.delta_stats.normalize(batch.actions)
            actions_emb = self._embed_temporal(actions_normalized, self.action_embed)
            actions_emb = self._mask_actions(actions_emb, batch.action_mask if batch.action_active is None else batch.action_active)
        else:
            actions_emb = None

        # =====================================================================
        # 5. Position encodings
        #    Rotary: computed from ctx_positions / future_positions, passed to blocks
        #    Additive: ctx_positions used (not batch.context) so all_relative is consistent
        # =====================================================================
        rotary_emb = None
        if self.cfg.use_rotary and not self.cfg.use_stratified_encoding:
            rotary_emb = self._compute_rotary_emb(
                rope_future_positions, fps_idx, B, F, C_ctx, L, device
            )
        elif not self.cfg.use_rotary:
            future_spatial_pe = self.spatial_embed(future_positions)    # [B, F, N, d]
            context_spatial_pe = self.spatial_embed(ctx_positions)      # [B, C_ctx, N, d]

            context_frame_idx = torch.arange(C_ctx, device=device)
            future_frame_idx = torch.arange(C_ctx, C_ctx + F, device=device)

            context_temporal_pe = self.temporal_embed(context_frame_idx)[None, :, None, :]  # [1, C_ctx, 1, d]
            future_temporal_pe = self.temporal_embed(future_frame_idx)[None, :, None, :]    # [1, F, 1, d]

            future_emb = future_emb + future_spatial_pe + future_temporal_pe
            context_emb = context_emb + context_spatial_pe + context_temporal_pe
            if actions_emb is not None:
                actions_emb = actions_emb + future_temporal_pe

        # =====================================================================
        # 6. Encode to latent space (N → L)
        # =====================================================================
        if self.cfg.use_stratified_encoding:
            anchor = batch.anchor
            future_latent, _, decode_info, fps_idx_global = self._stratified_encode(
                future_emb, rope_future_positions, anchor, batch.object_ids, batch.valid_lengths
            )
            # Pass ctx_positions instead of batch.context — correct for all_relative
            context_latent, _, _, _ = self._stratified_encode(
                context_emb, rope_ctx_positions, anchor, batch.object_ids, batch.valid_lengths
            )
            actions_latent = self._gather_temporal(actions_emb, fps_idx_global) if actions_emb is not None else None
            if self.cfg.use_rotary:
                rotary_emb = self._compute_rotary_emb(
                    rope_future_positions, fps_idx_global, B, F, C_ctx, L, device
                )
        else:
            if self.cfg.bypass_compression:
                future_latent = future_emb
                context_latent = context_emb
                actions_latent = actions_emb
            else:
                future_latent = self._encode(
                    future_emb, fps_idx, knn_idx_future, batch.valid_lengths, positions=rope_future_positions
                )
                context_latent = self._encode(
                    context_emb, fps_idx, knn_idx_context, batch.valid_lengths, positions=rope_ctx_positions
                )
                actions_latent = self._gather_temporal(actions_emb, fps_idx) if actions_emb is not None else None

        context_combined = context_latent  # [B, C_ctx, L, d]

        # =====================================================================
        # 6c. Tier-2 v2: pose tokens join the three latent streams (+Kr columns
        #     each) and rotary is recomputed over the extended token set. The
        #     block code is untouched — shapes simply grow by Kr.
        # =====================================================================
        L_particle = future_latent.shape[2]
        if pose_pack is not None:
            from utils.pose_native import context_pose_deltas, joint_features
            assert self.cfg.use_stratified_encoding and self.cfg.use_rotary, \
                "pose_native currently supports the stratified+rotary recipe"
            ridx = pose_pack["rigid_idx"]
            Kr = int(ridx.numel())
            tmpl_all = batch.body_templates[:, ridx]                     # [B, Kr, P, 3]
            tmask = batch.body_template_mask[:, ridx]                    # [B, Kr, P]
            g = self.pose_geom_embed(tmpl_all.to(future_latent.dtype))
            g = g.masked_fill(~tmask[..., None], float("-inf")).amax(dim=2)   # [B, Kr, d]
            if self.cfg.articulated_bodies:                              # each body's joint description
                jt, jgeom = joint_features(batch, ridx)
                g = g + self.pose_joint_type[jt] + self.pose_joint_geom(jgeom.to(g.dtype))
            ptok = self.pose_token_embed(pose_pack["noisy_pose"].to(g.dtype))
            ptok = ptok + g[:, None] + self.pose_token_type              # [B, F, Kr, d]
            # context pose tokens: GT context pose deltas (clean conditioning, the
            # pose-side analog of the particle context stream)
            c6 = batch.pose_delta_stats.normalize(
                context_pose_deltas(batch, ridx), clip=False).to(g.dtype)
            ctok = self.pose_token_embed(c6) + g[:, None] + self.pose_token_type
            # pose-space action rows ride the block's action channel (v8's idea,
            # per-body): commanded Δ6 of actuated bodies, no_action elsewhere
            a6 = batch.pose_delta_stats.normalize(
                batch.pose_actions[:, :, ridx], clip=False).to(g.dtype)
            arow = self.pose_action_skip(a6)                             # [B, F, Kr, d]
            amask = batch.pose_action_mask[:, ridx]                      # [B, Kr]
            arow = torch.where(amask[:, None, :, None], arow,
                               self.pose_no_action_token.to(arow.dtype))
            future_latent = torch.cat([future_latent, ptok], dim=2)
            context_combined = torch.cat([context_combined, ctok], dim=2)
            if actions_latent is not None:
                actions_latent = torch.cat([actions_latent, arow], dim=2)
            lat_pos = self._gather_temporal(rope_future_positions, fps_idx_global)
            # New collections contain valid centres outside the historical
            # +/-0.8 m range (Yikai reaches 1.37 m). Their particle RoPE inputs
            # are already linear/unclipped; keep pose tokens in the same frame.
            # Preserve the historical domains' checkpoint arithmetic.
            clip_centers = batch.domain not in ("folding", "yikai", "dex")
            cpos = (batch.position_stats.normalize(pose_pack["centers"].float(), clip=clip_centers)
                    * self.cfg.rope_position_scale).detach().to(lat_pos.dtype)
            rotary_emb = self._compute_rotary_emb(
                torch.cat([lat_pos, cpos], dim=2), None, B, F, C_ctx,
                L_particle + Kr, device)

        # =====================================================================
        # 7. Transformer blocks
        # =====================================================================
        x = future_latent  # [B, F, L, d]
        causal_mask = self._get_causal_temporal_mask(F, device, x.dtype)

        for block in self.blocks:
            if self.training and self.cfg.gradient_checkpointing:
                x = checkpoint(
                    block,
                    x, context_combined, actions_latent, t_emb, ctx_t_emb, rotary_emb, causal_mask,
                    use_reentrant=False,
                )
            else:
                x = block(
                    x=x,
                    context=context_combined,
                    actions=actions_latent,
                    t_emb=t_emb,
                    ctx_t_emb=ctx_t_emb,
                    rotary_emb=rotary_emb,
                    temporal_attn_mask=causal_mask,
                )

        # =====================================================================
        # 7b. Tier-2 v2: split pose tokens off; their v-prediction is a direct
        #     linear readout. If the batch has no deformable rows (pickup,
        #     pushing) the particle decode is skipped entirely.
        # =====================================================================
        pose_v = None
        if pose_pack is not None:
            pose_hidden = x[:, :, L_particle:]
            x = x[:, :, :L_particle]
            pose_v = self.pose_v_head(self.final_norm(pose_hidden))      # [B, F, Kr, 6]
            deform_rows = ((batch.body_type.gather(1, batch.particle_body) == 0)
                           & batch.get_valid_mask())
            if not bool(deform_rows.any()):
                return None, pose_v

        # =====================================================================
        # 8. Decode (L → N)
        # =====================================================================
        temporal_mask = self._get_decoder_temporal_mask(F, device, x.dtype)

        # Decoder query identity: clean frame-0 anchor [B, F, N, d] (opt-in), fed to whichever
        # decode path runs (stratified slices it per object).
        anchor_emb = None
        if self.cfg.decoder_anchor_identity and not self.cfg.bypass_compression:
            f0n = batch.position_stats.normalize(frame0) if batch.position_stats is not None else frame0
            anchor_emb = self.anchor_identity_embed(f0n).unsqueeze(1).expand(-1, F, -1, -1)
        # gradient_checkpointing == "full" is implemented at per-attention-op
        # granularity inside the encode/decode seams (see _ckpt_full): wrapping the
        # WHOLE decode as one region does not fit — its backward recompute
        # re-materializes the full [B, F, N, d] working set at once (measured OOM
        # on cloth N=10701 at dim 768, B=4, 40 GB).
        if self.cfg.use_stratified_decoding and decode_info is not None:
            x = self._stratified_decode(
                x, future_emb, rope_future_positions, batch.object_ids, decode_info, temporal_mask,
                anchor_emb=anchor_emb,
            )
        elif not self.cfg.bypass_compression:
            x = self._decode(x, future_emb, temporal_mask, positions=rope_future_positions,
                             fps_idx=fps_idx, anchor_emb=anchor_emb)

        # =====================================================================
        # 9. Output head
        # =====================================================================
        particle_hidden = self.final_norm(x)                    # [B, F, N, d]
        per_particle_delta = self.head(particle_hidden)         # [B, F, N, 3]

        if pose_v is not None:
            return per_particle_delta, pose_v
        return per_particle_delta

    # =========================================================================
    # Helper methods
    # =========================================================================


    def _pose_native_inputs(self, batch, noisy_future, noisy_pose):
        """Tier-2 v2 input side — delegates to the shared helper
        (utils/pose_native.py; also used by unet_std). Returns
        (patched_noisy_future, pack) exactly as before."""
        from utils.pose_native import pose_native_inputs
        return pose_native_inputs(batch, noisy_future, noisy_pose)

    def _encode(
        self,
        x: torch.Tensor,         # [B, T, N, d]
        fps_idx: torch.Tensor,   # [B, L]
        knn_idx: torch.Tensor | None = None,  # [B, L, k], only needed for KNN attention
        valid_lengths: torch.Tensor | None = None,  # [B], for masking in standard attention
        positions: torch.Tensor | None = None,  # [B, T, N, 3] for spatial RoPE
    ) -> torch.Tensor:
        """Encode to latent space via attention (KNN or standard cross-attention)."""
        B, T, N, d = x.shape
        L = fps_idx.shape[1]

        # Get queries from FPS points
        queries = self._gather_temporal(x, fps_idx)  # [B, T, L, d]
        queries_flat = queries.reshape(B * T, L, d)
        queries_norm = self.norm_queries(queries_flat)

        if self.cfg.use_knn_attention:
            # KNN attention: each query attends to K nearest neighbors
            kv = self._gather_knn(x, knn_idx)  # [B, T, L, k, d]
            kv_flat = kv.reshape(B * T, L, -1, d)
            kv_norm = self.norm_kv(kv_flat)

            # Compute spatial RoPE if using rotary and positions are provided
            rotary_emb_q = None
            rotary_emb_k = None
            if self.cfg.use_rotary and positions is not None:
                k = knn_idx.shape[-1]  # number of neighbors — works for both [B, L, k] and [B, T, L, k]


                # KNN neighbor positions [B, T, L, k, 3] (also the patch for pooled queries)
                knn_positions = self._gather_knn_positions(positions, knn_idx)  # [B, T, L, k, 3]
                knn_positions_flat = knn_positions.reshape(B * T, L, k, 3)

                # RoPE for queries: patch-pooled over the latent's KNN patch (banded), else single FPS point
                if self.rope_patch_pool:
                    rotary_emb_q = self.rotary_spatial.forward_pooled(knn_positions_flat)  # (cos,sin) [B*T, L, Dh]
                else:
                    fps_positions = self._gather_temporal(positions, fps_idx)  # [B, T, L, 3]
                    rotary_emb_q = self.rotary_spatial(fps_positions.reshape(B * T, L, 3))

                # RoPE for keys: each KNN neighbor at its own position [B, T, L, k, 3] -> [B*T, L, k, Dh]
                knn_pos_for_rope = knn_positions_flat.reshape(B * T * L, k, 3)
                cos_k, sin_k = self.rotary_spatial(knn_pos_for_rope)  # each [B*T*L, k, Dh]
                cos_k = cos_k.reshape(B * T, L, k, -1)
                sin_k = sin_k.reshape(B * T, L, k, -1)
                rotary_emb_k = (cos_k, sin_k)

            out = self._ckpt_full(self.knn_attn, queries_norm, kv_norm, rotary_emb_q=rotary_emb_q, rotary_emb_k=rotary_emb_k)
        else:
            # Standard cross-attention: each query attends to ALL N points
            x_flat = x.reshape(B * T, N, d)
            x_norm = self.norm_kv(x_flat)

            # Build attention mask for padded points (float mask for correct SDPA behavior)
            attn_mask = None
            if valid_lengths is not None:
                # [B, N] where True = INVALID
                indices = torch.arange(N, device=x.device)
                invalid_mask = indices >= valid_lengths[:, None]
                # Expand for T frames: [B, N] -> [B*T, 1, 1, N]
                invalid_mask = invalid_mask.unsqueeze(1).expand(B, T, N).reshape(B * T, 1, 1, N)
                # Convert to float mask: 0 for valid, -inf for invalid
                attn_mask = torch.zeros(B * T, 1, 1, N, device=x.device, dtype=x.dtype)
                attn_mask.masked_fill_(invalid_mask, float('-inf'))

            # Compute spatial RoPE if using rotary and positions are provided
            rotary_emb_q = None
            rotary_emb_k = None
            if self.cfg.use_rotary and positions is not None:
                # RoPE for queries: FPS positions at each frame
                fps_positions = self._gather_temporal(positions, fps_idx)  # [B, T, L, 3]
                fps_positions_flat = fps_positions.reshape(B * T, L, 3)
                rotary_emb_q = self.rotary_spatial(fps_positions_flat)  # (cos, sin) each [B*T, L, Dh]

                # RoPE for keys: ALL N positions at each frame
                positions_flat = positions.reshape(B * T, N, 3)
                rotary_emb_k = self.rotary_spatial(positions_flat)  # (cos, sin) each [B*T, N, Dh]

            out = self.attn_processor(
                self.encoder_cross_attn,
                queries_norm,
                encoder_hidden_states=x_norm,
                attention_mask=attn_mask,
                rotary_emb=rotary_emb_q,
                rotary_emb_k=rotary_emb_k,
            )

        # Residual connection
        out = queries_flat + out
        return out.reshape(B, T, L, d)

    def _compute_rotary_emb(
        self,
        future_positions: torch.Tensor,  # [B, F, N, 3] - integrated future positions (already denormalized)
        fps_idx: torch.Tensor,           # [B, L] or None (bypass_compression)
        B: int, F: int, C: int, L: int,
        device: torch.device,
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """Compute RoPE embeddings for temporal and spatial attention in transformer blocks."""
        # Actual latent dim: L from config when using FPS, N when bypass_compression
        L_act = fps_idx.shape[1] if fps_idx is not None else future_positions.shape[2]

        # =====================================================================
        # Temporal RoPE for QUERIES (future frames): indices [0, 1, ..., F-1]
        # =====================================================================
        # For transformer blocks, temporal attention is over [B*L_act, F, d]
        future_pos = torch.arange(F, device=device).float()
        future_pos = future_pos.unsqueeze(0).expand(B * L_act, -1)  # [B*L_act, F]
        future_cos, future_sin = self.rotary_temporal(future_pos)

        # =====================================================================
        # Temporal RoPE for KEYS (context + actions)
        # =====================================================================
        # context_combined = context_frames with shape [B, C, L, d]
        # (physics is now baked into context embeddings, no separate tokens)

        # Context frames at indices [-C, ..., -1]
        context_pos = torch.arange(-C, 0, device=device).float()
        context_pos = context_pos.unsqueeze(0).expand(B * L_act, -1)  # [B*L_act, C]
        context_cos, context_sin = self.rotary_temporal(context_pos)

        # Actions are at same time indices as future frames [0, 1, ..., F-1]
        if self.cfg.no_action_cond:
            # No actions: KV is context only → [B*L, C, head_dim]
            kv_cos, kv_sin = context_cos, context_sin
        else:
            action_cos, action_sin = future_cos, future_sin
            # Concatenate: [context, actions] for cross-attention keys
            # KV sequence is [context_combined, actions] = [context, actions]
            # Shape: [B*L, C+F, head_dim]
            kv_cos = torch.cat([context_cos, action_cos], dim=1)
            kv_sin = torch.cat([context_sin, action_sin], dim=1)

        # =====================================================================
        # Spatial RoPE: based on 3D positions at each future frame
        # =====================================================================
        # Gather FPS positions at each future frame (positions already in world coordinates)
        if fps_idx is not None:
            integrated_fps = self._gather_temporal(future_positions, fps_idx)  # [B, F, L, 3]
        else:
            integrated_fps = future_positions  # [B, F, N, 3] when bypass_compression
        integrated_fps_flat = integrated_fps.reshape(B * F, -1, 3)
        spatial_cos, spatial_sin = self.rotary_spatial(integrated_fps_flat)

        return {
            'temporal_q': (future_cos, future_sin),   # For future frame queries (cross & self attn)
            'temporal_kv': (kv_cos, kv_sin),          # For keys in temporal cross-attention
            'spatial': (spatial_cos, spatial_sin),    # For spatial self-attention
        }

    def _ckpt_full(self, fn, *args, **kwargs):
        """gradient_checkpointing == "full": checkpoint individual attention ops in
        the encode/decode seams (training only). Per-op granularity keeps one
        attention's activations live at a time during recompute; True/False take
        the plain-call branch, so existing configs are byte-identical.
        """
        if self.training and self.cfg.gradient_checkpointing == "full":
            return checkpoint(fn, *args, use_reentrant=False, **kwargs)
        return fn(*args, **kwargs)

    def _decode(
        self,
        latents: torch.Tensor,  # [B, F, L, d]
        queries: torch.Tensor,  # [B, F, N, d]
        temporal_mask: Optional[torch.Tensor] = None,  # [F, F] windowed attention mask
        positions: Optional[torch.Tensor] = None,  # [B, F, N, 3] for spatial RoPE in cross-attn
        fps_idx: Optional[torch.Tensor] = None,    # [B, L] to gather latent positions
        anchor_emb: Optional[torch.Tensor] = None, # [B, F, N, d] clean frame-0 identity (opt-in)
    ) -> torch.Tensor:
        """Decode from latent space: [B, F, L, d] -> [B, F, N, d]"""
        B, F, L, d = latents.shape
        N = queries.shape[2]

        # Decoder query identity: add the clean frame-0 anchor to the query value so it
        # keeps a stable identity even when its reconstructed position is noisy.
        if self.cfg.decoder_anchor_identity and anchor_emb is not None:
            queries = queries + anchor_emb

        # Flatten for attention: [B*F, L, d] and [B*F, N, d]
        latents_flat = latents.reshape(B * F, L, d)
        queries_flat = queries.reshape(B * F, N, d)

        # Spatial RoPE for decoder cross-attention
        rotary_emb_q = None
        rotary_emb_k = None
        if self.cfg.use_rotary and positions is not None:
            # Queries: all N particle positions per frame
            positions_flat = positions.reshape(B * F, N, 3)
            rotary_emb_q = self.rotary_spatial(positions_flat)  # (cos, sin) each [B*F, N, Dh]

            # Keys: L latent positions (gathered from FPS indices)
            if fps_idx is not None:
                latent_positions = self._gather_temporal(positions, fps_idx)  # [B, F, L, 3]
            else:
                latent_positions = positions
            latent_positions_flat = latent_positions.reshape(B * F, L, 3)
            rotary_emb_k = self.rotary_spatial(latent_positions_flat)  # (cos, sin) each [B*F, L, Dh]

        # Decoder query self-attn (opt-in): spatial then temporal self-attn on the queries
        # before cross-attn, so each query gathers neighbor context before reading the latents.
        if self.cfg.decoder_query_self_attn:
            qn = self.decoder_pre_spatial_norm(queries_flat)
            queries_flat = queries_flat + self.attn_processor(
                self.decoder_pre_spatial_attn, qn, rotary_emb=rotary_emb_q)
            qt = queries_flat.reshape(B, F, N, d).permute(0, 2, 1, 3).reshape(B * N, F, d)
            trope = None
            if self.cfg.use_rotary:
                tp = torch.arange(F, device=qt.device).float()[None].expand(B * N, -1)
                trope = self.rotary_temporal(tp)
            qtn = self.decoder_pre_temporal_norm(qt)
            qt = qt + self.attn_processor(
                self.decoder_pre_temporal_attn, qtn, attention_mask=temporal_mask, rotary_emb=trope)
            queries_flat = qt.reshape(B, N, F, d).permute(0, 2, 1, 3).reshape(B * F, N, d)

        # Cross-attention: queries attend to latents (with spatial RoPE)
        queries_norm = self.decoder_norm(queries_flat)
        attn_out = self._ckpt_full(
            self.attn_processor,
            self.decoder_cross_attn,
            queries_norm,
            encoder_hidden_states=latents_flat,
            rotary_emb=rotary_emb_q,
            rotary_emb_k=rotary_emb_k,
        )
        x = queries_flat + attn_out  # [B*F, N, d]

        # Reshape for temporal attention: [B, F, N, d] -> [B*N, F, d]
        x = x.reshape(B, F, N, d)
        x_temporal = x.permute(0, 2, 1, 3).reshape(B * N, F, d)

        # Temporal attention (with or without RoPE)
        rotary_emb = None
        if self.cfg.use_rotary:
            temporal_pos = torch.arange(F, device=x.device).float()
            temporal_pos = temporal_pos.unsqueeze(0).expand(B * N, -1)
            temporal_cos, temporal_sin = self.rotary_temporal(temporal_pos)
            rotary_emb = (temporal_cos, temporal_sin)
        else:
            # Additive mode: add temporal PE explicitly
            temporal_idx = torch.arange(F, device=x.device)
            temporal_pe = self.temporal_embed(temporal_idx)  # [F, d]
            x_temporal = x_temporal + temporal_pe[None, :, :]  # [B*N, F, d] + [1, F, d]

        # Temporal self-attention (with optional windowed mask)
        x_norm = self.decoder_temporal_norm(x_temporal)
        attn_out = self._ckpt_full(
            self.attn_processor,
            self.decoder_temporal_attn,
            x_norm,
            attention_mask=temporal_mask,
            rotary_emb=rotary_emb,
        )
        x_temporal = x_temporal + attn_out

        # Reshape back: [B*N, F, d] -> [B, F, N, d]
        x = x_temporal.reshape(B, N, F, d).permute(0, 2, 1, 3)

        return x

    # =========================================================================
    # Stratified Encoding/Decoding (per-object FPS + KNN)
    # =========================================================================

    def _compute_latents_per_object_single(
        self,
        object_ids: torch.Tensor,  # [N] single sample
        valid_length: int,
    ) -> Dict[int, int]:
        """
        Compute number of latents per object proportional to valid particle count.

        This is for a SINGLE sample - no batch dimension.

        Args:
            object_ids: [N] object type per particle (single sample)
            valid_length: number of valid particles for this sample

        Returns:
            Dict mapping object_id -> num_latents
        """
        N = object_ids.shape[0]
        device = object_ids.device
        unique_objs = object_ids.unique().tolist()

        # Create valid mask for this sample
        particle_indices = torch.arange(N, device=device)
        valid_mask = particle_indices < valid_length

        # Count VALID particles per object
        counts = {}
        total_valid = 0
        for obj_id in unique_objs:
            obj_valid_count = ((object_ids == obj_id) & valid_mask).sum().item()
            counts[obj_id] = obj_valid_count
            total_valid += obj_valid_count

        # Allocate latents proportionally based on valid particle counts
        latents_per_obj = {}
        allocated = 0
        for i, obj_id in enumerate(unique_objs):
            if i == len(unique_objs) - 1:
                # Last object gets remaining to ensure exact total
                # Use max(0, ...) to handle edge case where rounding caused over-allocation
                latents_per_obj[obj_id] = max(0, self.cfg.num_latents - allocated)
            else:
                proportion = counts[obj_id] / total_valid if total_valid > 0 else 0
                num_latents = int(round(self.cfg.num_latents * proportion))
                # Can't have more latents than valid particles
                num_latents = min(num_latents, counts[obj_id])
                # Don't exceed remaining budget
                num_latents = min(num_latents, self.cfg.num_latents - allocated)
                latents_per_obj[obj_id] = num_latents
                allocated += num_latents

        return latents_per_obj

    def _stratified_encode(
        self,
        x: torch.Tensor,              # [B, T, N, d] embedded features
        positions: torch.Tensor,      # [B, T, N, 3] for spatial RoPE
        anchor_positions: torch.Tensor,  # [B, N, 3] anchor frame for FPS selection
        object_ids: torch.Tensor,     # [B, N]
        valid_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict, torch.Tensor]:
        """
        Encode each object separately via per-object FPS + KNN, then concatenate.

        This prevents "Frankenstein tokens" where FPS picks anchors in gaps
        between objects, mixing gripper and object information.

        Processes each sample INDEPENDENTLY to handle heterogeneous batches
        where different samples may have different valid_lengths and thus
        different per-object latent allocations.

        Args:
            x: [B, T, N, d] embedded features to encode
            positions: [B, T, N, 3] positions for spatial RoPE in encoder
            anchor_positions: [B, N, 3] anchor frame positions for FPS selection
            object_ids: [B, N] object type per particle
            valid_lengths: [B] number of valid particles

        Returns:
            latents: [B, T, num_latents, d] concatenated pure latents
            fps_anchor_positions: [B, num_latents, 3] FPS positions at anchor frame
            decode_info: dict with per-sample object_ranges for stratified decoding
            fps_indices_global: [B, num_latents] global particle indices for gathering actions
        """
        B, T, N, d = x.shape

        # Process each sample independently
        all_sample_latents = []
        all_sample_anchor_pos = []
        all_sample_fps_idx = []
        all_sample_ranges = []  # Per-sample object ranges

        for b in range(B):
            # Extract single sample (keep batch dim for compatibility with _encode_object)
            x_b = x[b:b+1]                          # [1, T, N, d]
            pos_b = positions[b:b+1]                # [1, T, N, 3]
            anchor_pos_b = anchor_positions[b:b+1]  # [1, N, 3]
            obj_ids_b = object_ids[b]               # [N]
            valid_len_b = valid_lengths[b].item()

            # Compute per-sample latent allocation
            latents_per_obj_b = self._compute_latents_per_object_single(obj_ids_b, valid_len_b)

            # Encode this sample
            latents_b, anchor_pos_b_out, ranges_b, fps_idx_b = self._stratified_encode_single(
                x_b, pos_b, anchor_pos_b, obj_ids_b, valid_len_b, latents_per_obj_b
            )

            all_sample_latents.append(latents_b)        # [1, T, num_latents, d]
            all_sample_anchor_pos.append(anchor_pos_b_out)  # [1, num_latents, 3]
            all_sample_fps_idx.append(fps_idx_b)        # [1, num_latents]
            all_sample_ranges.append(ranges_b)          # dict

        # Stack across batch dimension
        latents = torch.cat(all_sample_latents, dim=0)           # [B, T, num_latents, d]
        fps_anchor_positions = torch.cat(all_sample_anchor_pos, dim=0)  # [B, num_latents, 3]
        fps_indices_global = torch.cat(all_sample_fps_idx, dim=0)       # [B, num_latents]

        # Store per-sample object ranges and FPS indices for decoding
        decode_info = {
            'object_ranges_per_sample': all_sample_ranges,  # List[Dict[int, Tuple[int, int]]]
            'fps_indices_global': fps_indices_global,        # [B, num_latents] for decoder spatial RoPE
        }

        return latents, fps_anchor_positions, decode_info, fps_indices_global

    def _stratified_encode_single(
        self,
        x: torch.Tensor,              # [1, T, N, d] single sample
        positions: torch.Tensor,      # [1, T, N, 3]
        anchor_positions: torch.Tensor,  # [1, N, 3]
        object_ids: torch.Tensor,     # [N] (no batch dim)
        valid_length: int,
        latents_per_obj: Dict[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict, torch.Tensor]:
        """
        Encode a single sample with per-object FPS + KNN.

        Returns:
            latents: [1, T, num_latents, d]
            fps_anchor_positions: [1, num_latents, 3]
            object_ranges: Dict[int, Tuple[int, int]] mapping obj_id -> (start, end)
            fps_indices_global: [1, num_latents]
        """
        _, T, N, d = x.shape
        device = x.device
        k = self.cfg.k_neighbors

        all_latents = []
        all_anchor_pos = []
        all_fps_global_idx = []
        object_ranges = {}

        latent_offset = 0
        unique_objs = object_ids.unique().tolist()

        # Create valid mask for this sample
        particle_indices = torch.arange(N, device=device)
        valid_mask = particle_indices < valid_length

        for obj_id in unique_objs:
            L_obj = latents_per_obj.get(obj_id, 0)
            if L_obj == 0:
                continue

            # 1. Get indices for this object (only valid particles)
            obj_mask = (object_ids == obj_id) & valid_mask
            obj_indices = obj_mask.nonzero(as_tuple=True)[0]  # [n_obj_valid]
            n_obj_valid = len(obj_indices)

            # Safety check: can't have more latents than valid particles
            L_obj = min(L_obj, n_obj_valid)
            if L_obj == 0:
                continue

            # 2. Extract this object's features: [1, T, n_obj_valid, d]
            obj_x = x[:, :, obj_indices, :]
            obj_pos = positions[:, :, obj_indices, :]
            anchor_pos_obj = anchor_positions[:, obj_indices, :]  # [1, n_obj_valid, 3]

            # 3. FPS on anchor frame (no lengths needed - all particles are valid)
            _, fps_idx_local = sample_farthest_points(anchor_pos_obj, K=L_obj)
            # fps_idx_local: [1, L_obj] - indices within this object's valid particles

            # 4. Convert to global particle indices
            fps_idx_global = obj_indices[fps_idx_local]  # [1, L_obj]

            # 5. KNN within this object
            K_actual = min(k, n_obj_valid)
            fps_positions = self._gather(anchor_pos_obj, fps_idx_local)  # [1, L_obj, 3] — always needed for anchor_pos output

            if self.cfg.per_frame_knn:
                # Per-frame KNN: query from FPS particles at their current positions each frame
                fps_positions_t = self._gather_temporal(obj_pos, fps_idx_local)  # [1, T, L_obj, 3]
                fps_flat = fps_positions_t.reshape(T, L_obj, 3)
                obj_pos_flat = obj_pos.squeeze(0)  # [T, n_obj_valid, 3]
                _, knn_idx_local, _ = knn_points(fps_flat, obj_pos_flat, K=K_actual)
                knn_idx_local = knn_idx_local.unsqueeze(0)  # [1, T, L_obj, k]
            else:
                # Anchor-frame KNN broadcast across all frames
                _, knn_idx_local, _ = knn_points(fps_positions, anchor_pos_obj, K=K_actual)
                # knn_idx_local: [1, L_obj, k]

            # 6. Encode via KNN attention
            obj_latents = self._encode_object(obj_x, fps_idx_local, knn_idx_local, obj_pos)
            # obj_latents: [1, T, L_obj, d]

            # 7. Store results
            all_latents.append(obj_latents)
            all_anchor_pos.append(fps_positions)
            all_fps_global_idx.append(fps_idx_global)

            object_ranges[obj_id] = (latent_offset, latent_offset + L_obj)
            latent_offset += L_obj

        # Concatenate across objects
        latents = torch.cat(all_latents, dim=2)  # [1, T, num_latents, d]
        anchor_pos_out = torch.cat(all_anchor_pos, dim=1)  # [1, num_latents, 3]
        fps_indices_global = torch.cat(all_fps_global_idx, dim=1)  # [1, num_latents]

        return latents, anchor_pos_out, object_ranges, fps_indices_global

    def _encode_object(
        self,
        x: torch.Tensor,         # [B, T, n_obj, d]
        fps_idx: torch.Tensor,   # [B, L_obj]
        knn_idx: torch.Tensor,   # [B, L_obj, k] OR [B, T, L_obj, k]
        positions: torch.Tensor, # [B, T, n_obj, 3]
    ) -> torch.Tensor:
        """Encode a single object's particles via KNN attention."""
        B, T, n_obj, d = x.shape
        L_obj = fps_idx.shape[1]
        k = knn_idx.shape[-1]  # number of neighbors — works for both [B, L, k] and [B, T, L, k]


        # Get queries from FPS points
        queries = self._gather_temporal(x, fps_idx)  # [B, T, L_obj, d]
        queries_flat = queries.reshape(B * T, L_obj, d)
        queries_norm = self.norm_queries(queries_flat)

        if self.cfg.use_knn_attention:
            # KNN attention: gather neighbors
            kv = self._gather_knn_local(x, knn_idx)  # [B, T, L_obj, k, d]
            kv_flat = kv.reshape(B * T, L_obj, k, d)
            kv_norm = self.norm_kv(kv_flat)

            # Compute spatial RoPE if using rotary
            rotary_emb_q = None
            rotary_emb_k = None
            if self.cfg.use_rotary:
                # KNN neighbor positions (also the patch for pooled queries)
                knn_positions = self._gather_knn_positions_local(positions, knn_idx)  # [B, T, L_obj, k, 3]
                knn_positions_flat = knn_positions.reshape(B * T, L_obj, k, 3)

                # RoPE for queries: patch-pooled over the KNN patch (banded), else single FPS point
                if self.rope_patch_pool:
                    rotary_emb_q = self.rotary_spatial.forward_pooled(knn_positions_flat)
                else:
                    fps_positions = self._gather_temporal(positions, fps_idx)  # [B, T, L_obj, 3]
                    rotary_emb_q = self.rotary_spatial(fps_positions.reshape(B * T, L_obj, 3))

                # RoPE for keys (KNN neighbors, each at its own position)
                knn_pos_for_rope = knn_positions_flat.reshape(B * T * L_obj, k, 3)
                cos_k, sin_k = self.rotary_spatial(knn_pos_for_rope)
                cos_k = cos_k.reshape(B * T, L_obj, k, -1)
                sin_k = sin_k.reshape(B * T, L_obj, k, -1)
                rotary_emb_k = (cos_k, sin_k)

            out = self._ckpt_full(self.knn_attn, queries_norm, kv_norm, rotary_emb_q=rotary_emb_q, rotary_emb_k=rotary_emb_k)
        else:
            # Standard cross-attention to all object particles
            x_flat = x.reshape(B * T, n_obj, d)
            x_norm = self.norm_kv(x_flat)

            rotary_emb_q = None
            rotary_emb_k = None
            if self.cfg.use_rotary:
                fps_positions = self._gather_temporal(positions, fps_idx)
                fps_positions_flat = fps_positions.reshape(B * T, L_obj, 3)
                rotary_emb_q = self.rotary_spatial(fps_positions_flat)

                positions_flat = positions.reshape(B * T, n_obj, 3)
                rotary_emb_k = self.rotary_spatial(positions_flat)

            out = self.attn_processor(
                self.encoder_cross_attn,
                queries_norm,
                encoder_hidden_states=x_norm,
                rotary_emb=rotary_emb_q,
                rotary_emb_k=rotary_emb_k,
            )

        # Residual connection
        out = queries_flat + out
        return out.reshape(B, T, L_obj, d)


    def _stratified_decode(
        self,
        latents: torch.Tensor,      # [B, F, L_total, d]
        queries: torch.Tensor,      # [B, F, N, d]
        positions: torch.Tensor,    # [B, F, N, 3] scaled positions for spatial RoPE
        object_ids: torch.Tensor,   # [B, N]
        decode_info: Dict,
        temporal_mask: Optional[torch.Tensor] = None,  # [F, F] windowed attention mask
        anchor_emb: Optional[torch.Tensor] = None,                 # [B, F, N, d] clean frame-0 identity (opt-in)
    ) -> torch.Tensor:
        """
        Decode each object from its own latents only.

        This prevents motion bleeding: gripper particles only attend to
        gripper latents, object particles only attend to object latents.

        Processes each sample INDEPENDENTLY since different samples may have
        different per-object latent ranges.
        """
        B, F, N, d = queries.shape
        device = queries.device
        output = torch.zeros(B, F, N, d, device=device, dtype=queries.dtype)

        object_ranges_per_sample = decode_info['object_ranges_per_sample']
        fps_indices_global = decode_info.get('fps_indices_global')  # [B, L_total]

        # Process each sample independently
        for b in range(B):
            object_ranges_b = object_ranges_per_sample[b]

            for obj_id, (start, end) in object_ranges_b.items():
                # Get this object's particles for this sample
                obj_mask = (object_ids[b] == obj_id)
                obj_indices = obj_mask.nonzero(as_tuple=True)[0]

                obj_queries = queries[b:b+1, :, obj_indices, :]     # [1, F, n_obj, d]
                obj_positions = positions[b:b+1, :, obj_indices, :] # [1, F, n_obj, 3]
                obj_anchor = anchor_emb[b:b+1, :, obj_indices, :] if anchor_emb is not None else None

                # Fix A: cloth (obj_id=0) attends to ALL latents when flag set, so its
                # decoder cross-attention can directly see gripper latents.
                use_all = (obj_id == 0 and self.cfg.cloth_attends_all_latents)
                if use_all:
                    obj_latents = latents[b:b+1]                                        # [1, F, L_total, d]
                    obj_latent_positions = (
                        positions[b:b+1, :, fps_indices_global[b], :]
                        if fps_indices_global is not None else None
                    )
                else:
                    obj_latents = latents[b:b+1, :, start:end, :]                       # [1, F, L_obj, d]
                    obj_latent_positions = (
                        positions[b:b+1, :, fps_indices_global[b, start:end], :]
                        if fps_indices_global is not None else None
                    )

                # Decode this object
                obj_output = self._decode_object(obj_latents, obj_queries, temporal_mask,
                                                  obj_positions, obj_latent_positions,
                                                  anchor_emb=obj_anchor)

                # Place back into full output
                output[b:b+1, :, obj_indices, :] = obj_output

        return output

    def _decode_object(
        self,
        latents: torch.Tensor,  # [B, F, L_obj, d]
        queries: torch.Tensor,  # [B, F, n_obj, d]
        temporal_mask: Optional[torch.Tensor] = None,  # [F, F] windowed attention mask
        positions: Optional[torch.Tensor] = None,       # [B, F, n_obj, 3] particle positions for RoPE
        latent_positions: Optional[torch.Tensor] = None, # [B, F, L_obj, 3] latent positions for RoPE
        anchor_emb: Optional[torch.Tensor] = None,       # [B, F, n_obj, d] clean frame-0 identity (opt-in)
    ) -> torch.Tensor:
        """Decode single object: cross-attention to latents + temporal self-attention."""
        B, F, L_obj, d = latents.shape
        n_obj = queries.shape[2]

        # Decoder query identity: add the clean frame-0 anchor to the query value.
        if self.cfg.decoder_anchor_identity and anchor_emb is not None:
            queries = queries + anchor_emb

        # 1. Cross-attention (merge B and F)
        latents_flat = latents.reshape(B * F, L_obj, d)
        queries_flat = queries.reshape(B * F, n_obj, d)

        # Spatial RoPE for decoder cross-attention
        rotary_emb_q = None
        rotary_emb_k = None
        if self.cfg.use_rotary and positions is not None and latent_positions is not None:
            positions_flat = positions.reshape(B * F, n_obj, 3)
            rotary_emb_q = self.rotary_spatial(positions_flat)

            latent_pos_flat = latent_positions.reshape(B * F, L_obj, 3)
            rotary_emb_k = self.rotary_spatial(latent_pos_flat)

        queries_norm = self.decoder_norm(queries_flat)
        attn_out = self._ckpt_full(
            self.attn_processor,
            self.decoder_cross_attn,
            queries_norm,
            encoder_hidden_states=latents_flat,
            rotary_emb=rotary_emb_q,
            rotary_emb_k=rotary_emb_k,
        )
        x = queries_flat + attn_out  # [B*F, n_obj, d]

        # 2. Temporal self-attention (merge B and n_obj)
        x = x.reshape(B, F, n_obj, d)
        x_temporal = x.permute(0, 2, 1, 3).reshape(B * n_obj, F, d)

        rotary_emb = None
        if self.cfg.use_rotary:
            temporal_pos = torch.arange(F, device=x.device).float()
            temporal_pos = temporal_pos.unsqueeze(0).expand(B * n_obj, -1)
            temporal_cos, temporal_sin = self.rotary_temporal(temporal_pos)
            rotary_emb = (temporal_cos, temporal_sin)
        else:
            temporal_idx = torch.arange(F, device=x.device)
            temporal_pe = self.temporal_embed(temporal_idx)
            x_temporal = x_temporal + temporal_pe[None, :, :]

        # Temporal self-attention (with optional windowed mask)
        x_norm = self.decoder_temporal_norm(x_temporal)
        attn_out = self._ckpt_full(
            self.attn_processor,
            self.decoder_temporal_attn,
            x_norm,
            attention_mask=temporal_mask,
            rotary_emb=rotary_emb,
        )
        x_temporal = x_temporal + attn_out

        # Reshape back: [B*n_obj, F, d] -> [B, F, n_obj, d]
        x = x_temporal.reshape(B, n_obj, F, d).permute(0, 2, 1, 3)
        return x

    def _embed_temporal(
        self,
        x: torch.Tensor,  # [B, T, N, D] where D is input dim (3 or 6)
        embed_fn: nn.Module,
    ) -> torch.Tensor:
        """Embed temporal sequence: [B, T, N, D] -> [B, T, N, d]"""
        B, T, N, D = x.shape
        x_flat = x.reshape(B * T, N, D)
        emb_flat = embed_fn(x_flat)
        return emb_flat.reshape(B, T, N, -1)

    def _prepare_physics_features(self, batch) -> torch.Tensor:
        """
        Prepare normalized physics features for concatenation with positions/deltas.

        Returns:
            [B, N, 3] tensor with [log10(E), nu, rho_scaled] per particle
        """
        E_norm = torch.log10(batch.E + 1e-8)  # log scale for Young's modulus
        nu_norm = batch.nu                      # Poisson's ratio already in [0, 0.5]
        rho_norm = batch.rho / 1000.0           # scale density to ~1
        feats = torch.stack([E_norm, nu_norm, rho_norm], dim=-1)  # [B, N, 3]
        if getattr(self.cfg, 'normalize_physics_features', False):
            # Optional fixed physics-feature standardization (see ModelConfig).
            feats = torch.stack([
                (feats[..., 0] - 5.5) / 1.5,
                (feats[..., 1] - 0.25) / 0.2,
                (feats[..., 2] - 1.75) / 1.25,
            ], dim=-1)
        return feats

    def _mask_actions(
        self,
        actions_emb: torch.Tensor,  # [B, F, N, d]
        action_mask: torch.Tensor,  # [B, N]
    ) -> torch.Tensor:
        """Replace non-gripper action embeddings with no_action_token."""
        B, F, N, d = actions_emb.shape

        # Expand mask: [B, N] -> [B, 1, N, 1] -> [B, F, N, d]
        mask = (action_mask[:, None, :, None] if action_mask.ndim==2 else action_mask[...,None]).expand(B, F, N, d)

        # Expand no_action_token: [d] -> [B, F, N, d]
        no_action = self.no_action_token.view(1, 1, 1, d).expand(B, F, N, d)

        return torch.where(mask, actions_emb, no_action)

    def _gather(
        self,
        x: torch.Tensor,    # [B, N, D]
        idx: torch.Tensor,  # [B, L]
    ) -> torch.Tensor:
        """Gather points by index: [B, N, D] x [B, L] -> [B, L, D]"""
        D = x.shape[-1]
        idx_exp = idx.unsqueeze(-1).expand(-1, -1, D)
        return x.gather(1, idx_exp)

    def _gather_temporal(
        self,
        x: torch.Tensor,    # [B, T, N, D]
        idx: torch.Tensor,  # [B, L]
    ) -> torch.Tensor:
        """Gather temporal: [B, T, N, D] x [B, L] -> [B, T, L, D]"""
        B, T, N, D = x.shape
        L = idx.shape[1]
        idx_exp = idx[:, None, :, None].expand(B, T, L, D)
        return x.gather(2, idx_exp)

    def _gather_knn(
        self,
        x: torch.Tensor,        # [B, T, N, d]
        knn_idx: torch.Tensor,  # [B, L, k] OR [B, T, L, k]
    ) -> torch.Tensor:          # [B, T, L, k, d]
        B, T, N, d = x.shape
        if knn_idx.ndim == 3:
            # Anchor-frame indices — broadcast across T
            x_t = x.transpose(1, 2)  # [B, N, T, d]
            batch_idx = torch.arange(B, device=x.device)[:, None, None]
            gathered = x_t[batch_idx, knn_idx]       # [B, L, k, T, d]
            return gathered.permute(0, 3, 1, 2, 4)   # [B, T, L, k, d]
        else:
            # Per-frame indices [B, T, L, k]
            B_idx = torch.arange(B, device=x.device).view(B, 1, 1, 1)
            T_idx = torch.arange(T, device=x.device).view(1, T, 1, 1)
            return x[B_idx, T_idx, knn_idx]          # [B, T, L, k, d]

    # Gathering is identical for feature/position channels and scene/object-local
    # indices. Keep the historical method names for callers and checkpoint code.
    _gather_knn_positions = _gather_knn
    _gather_knn_local = _gather_knn
    _gather_knn_positions_local = _gather_knn
