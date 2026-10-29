"""Shared configuration for the supported particle backbones.

Defaults preserve checkpoint behavior; choose an explicit recipe in configs/.
Only fields implemented by at least one retained backbone are listed here.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Unified Transformer and hierarchical UNet settings."""

    # Particle backbone
    dim: int = 256
    depth: int = 4
    heads: int = 8
    num_latents: int = 256
    k_neighbors: int = 32
    dropout: float = 0.1
    context_len: int = 2
    use_rotary: bool = True
    use_knn_attention: bool = True
    gradient_checkpointing: object = False
    use_stratified_encoding: bool = False
    use_stratified_decoding: bool = False
    use_windowed_temporal: bool = False
    temporal_window_size: int = 5
    temporal_use_sink: bool = True
    no_action_cond: bool = False
    num_mat_types: int = 8
    causal_temporal: bool = False
    bypass_compression: bool = True
    all_relative: bool = True
    per_frame_knn: bool = False
    concat_anchor_z: bool = False
    normalize_physics_features: bool = False
    embed_actuatable: bool = False
    rope_position_scale: float = 1.0
    rope_spatial_base: float = 0.1
    banded_rope: bool = False
    rope_f_min: Optional[float] = None
    rope_f_max: Optional[float] = None
    rope_rotary_fraction: Optional[float] = None
    rope_patch_pool: Optional[bool] = None
    decoder_anchor_identity: bool = False
    decoder_query_self_attn: bool = False

    # Unified rigid-body pose tokens
    pose_native: bool = False
    articulated_bodies: bool = False   # joint description on pose tokens (TrajectoryBatch articulation)
    cloth_attends_all_latents: bool = False

    # Hierarchical UNet
    unet_points: int = 0
    unet_strides: str = ''
    unet_widths: str = ''
    unet_enc_depths: str = ''
    unet_dec_depths: str = ''
    unet_head_dim: int = 64
    unet_expansion: float = 2.67
    unet_t_dim: int = 256
    unet_serial_patch: int = 256
    unet_causal_context: bool = False
    unet_bidirectional_temporal: bool = False
    unet_fps_random_start: bool = False
    unet_supervise_context: bool = False
    unet_ckpt_from_level: int = -1
    unet_drop_offset_channel: bool = False
    # Coarsest spatial RoPE band, rad per geometry unit: the unique range is 2*pi*unet_geo_r_m/f_min.
    # 0.15 -> 5.58 m per axis. 0.5 (1.67 m, the historical value) was calibrated on a 0.80 m clean
    # scene; measured 2026-09-20 on the v7 rulers, mpm_fresh exceeds it in 598 of 600 windows once its
    # own ruler is integrated into the coordinates (median z extent 2.11 m, worst 3.16 m), and cloth
    # reaches 1.64 m on geometry alone. Checkpoints saved without this key keep 0.5 (models/factory.py).
    unet_rope_f_min: float = 0.15
    unet_ckpt_ingraph: bool = False
    unet_serial_levels: int = 0
    unet_dual_ffn: bool = False
    unet_register_from_level: int = -1
    unet_register_objs: int = 1
    # Classic core: how register rows live. 'per_block' (historical) gives every block its own
    # scratch set, appended and dropped inside the block; 'per_level' gives each level one learned
    # set that is carried through that level's blocks and stripped at the seam (the joint core's
    # unet_tube_register_mode=per_level, for model.unet_core=unet). Rows per set: unet_register_count.
    unet_register_mode: str = 'per_block'
    unet_pool_mode: str = 'attn'
    unet_unpool_mode: str = 'attn'
    unet_pool_k: int = 16
    unet_interp_k: int = 3
    unet_l1_by_mat: str = ''
    unet_l1_default: int = 0
    unet_geo_r_m: float = 0.0
    unet_l1_rule: str = ''
    unet_l1_rule_min: int = 32
    unet_l1_rule_max: int = 1024
    unet_l1_scene_budget: int = 0
    unet_l1_rigid_max: int = 0
    unet_actions: bool = False
    unet_ckpt_seams: bool = False

    # -- Joint-attention core (models/tube_joint.py) --------------------------------------------------------
    # unet_core picks what a level's blocks are. 'unet' (default) is the factorized production UNet and every
    # key below is unread; 'tube_joint' swaps in JointBlocks -- one attention kernel per call over the call's
    # whole (rows x tokens) sequence. Same ladder, seams, widths, depths and checkpointing either way.
    unet_core: str = 'unet'                     # 'unet' | 'tube_joint'
    unet_tube_sizes: str = ''                   # tokens per attention call, one entry per level; 0 = whole level
    unet_tube_grouping: str = 'curves'          # how a tubed level is cut: 'curves' (Morton) | 'random'
    unet_tube_group_draws: int = 2              # random grouping: independent draws per level, cycled over blocks
    unet_tube_within_object: str = ''           # levels where a call never straddles objects
    unet_tube_long_skip: bool = False           # decoder block i merges encoder block (D_enc-1-i) of its level
    unet_tube_dec_mirror: bool = False          # ... and cuts its groups exactly as that encoder block did
    unet_head_dims: str = ''                    # per-level head dim override, e.g. '44,88,88,88'; '' = unet_head_dim
    unet_fp32_head: bool = True

    # Per-token adaLN: a zero-init projection of each token's material + physics vector, summed into the
    # block's six adaLN slots on top of the per-row noise modulation (no bias, so a register contributes
    # nothing). The base UNet conditions only per row, so this is the knob that makes material reach the blocks.
    unet_token_adaln: bool = False

    # How an ACTUATED body is conditioned on (pose-native runs).
    #   skip   (default) the learned pose-space action vector (pose_action_skip) on its future rows
    #   clean  no action vector at all: a GIVEN actuated body is shown unnoised -- its pose rows are
    #          pinned to the clean value, future and context, in training and at every sampler step --
    #          and its tokens are told t = 0 through the per-object adaLN. Which bodies are given is
    #          train.action_dropout's per-element draw (utils/conditioning.hide_actions).
    unet_pose_action_mode: str = 'skip'
    # Deformable context generation (default off). With train.pose_noise_context the context is a
    # generated channel, but only through the rigid POSES; a scene with deformable particles was
    # refused. On: the deformable rows' context particles are denoised and supervised like the
    # future rows (same target, same velocity_weight_cap), and the sampler steps them too.
    unet_deformable_context: bool = False

    # Register rows, off unless unet_register_from_level >= 0. They ride whole-level calls only.
    #   per_level  a fresh set per level, carried across that level's blocks and dropped at every seam
    #   carried    one set, width-projected across the seams, alive until it returns
    unet_tube_register_mode: str = 'carried'    # 'carried' | 'per_level'
    # objects' worth of register rows per level, as a FRACTION (0.5 = half an object: 64 rows at a level
    # holding 128 tokens per object, 32 at one holding 64). 0 = follow the integer unet_register_objs.
    unet_tube_register_objs: float = 0.0
    unet_register_count: int = 0                # rows per set; 0 = unet_register_objs x the level's per-object count

    # RoPE budget of the joint blocks. unet_rope_budget states the split in head dims explicitly as
    # 'spatial,row,identity,pass' (e.g. '42,6,12,4' at head dim 64); '' derives it from unet_rope_fraction.
    unet_rope_budget: str = ''
    unet_rope_fraction: float = 0.75            # derived budget: share of the head dim that is rotated
    unet_rope_f_max: float = 64.0               # finest spatial band, rad per geometry unit
    unet_rope_t_dims: int = 8                   # derived budget: dims on the row index
    unet_rope_t_max_period: float = 64.0        # rows; > 2 x (rows - 1) so signed row offsets never alias
    unet_rope_t_min_period: float = 2.0         # rows; 2 = the row Nyquist, 2*pi = 1 rad per row
    unet_rope_t_base: float = 0.0               # > 0: standard inverse-base row bands (10000 = Wan, 256 = Hunyuan)
    # Identity axis: 'frame0_pos' spends bands on the token's CLEAN frame-0 xyz, so every row of a particle
    # sits at identity offset exactly 0. 'curve_rank' uses its rank along curve order 0. 'none' = neither.
    unet_rope_identity: str = 'curve_rank'      # 'none' | 'curve_rank' | 'frame0_pos'
    unet_rope_identity_periods_m: str = '0.012,0.4'   # frame0_pos: fine (scaled by token spacing), then coarse
    unet_rope_identity_max_period: float = 2048.0     # curve_rank: coarsest band, in curve steps
    unet_curve_orders: int = 2                  # 'curves': distinct Morton axis permutations; block i uses i mod K
