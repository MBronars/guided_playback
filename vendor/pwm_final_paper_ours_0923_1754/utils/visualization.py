"""
Visualization utilities for particle trajectory diffusion.

Creates ground-truth/prediction GIFs with optional multi-view and ensemble panels.
"""

from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import to_rgba
import imageio

def height_colormap(z_values: np.ndarray) -> np.ndarray:
    """Map z-values to a blue→cyan→green→yellow colormap (matches sim1 export_rerun)."""
    z_min, z_max = z_values.min(), z_values.max()
    z_range = z_max - z_min if z_max > z_min else 1.0
    t = (z_values - z_min) / z_range
    r = np.clip(t * 2 - 0.5, 0, 1)
    g = np.clip(1.0 - np.abs(t - 0.5) * 2, 0.3, 1)
    b = np.clip(1.0 - t * 2, 0, 1)
    a = np.ones_like(t)
    return (np.stack([r, g, b, a], axis=-1) * 255).astype(np.uint8)

ROTATIONS = [
    (30,  -60, 'Perspective'),
    (90,  -90, 'Top-down'),
    (0,   -90, 'Side'),
]

def create_trajectory_comparison_gif(
    gt_positions: torch.Tensor,      # [B, F, N, 3]
    pred_positions: torch.Tensor,    # [B, F, N, 3]  OR  [S, B, F, N, 3]
    action_mask: torch.Tensor,       # [B, N]
    save_path: str,
    sample_idx: int = 0,
    fps: int = 5,
    figsize: tuple = (14, 6),
    valid_mask: Optional[torch.Tensor] = None,   # [B, N]
    object_ids: Optional[torch.Tensor] = None,   # [B, N]
    actions: Optional[torch.Tensor] = None,      # [B, F, N, 3]
    arrow_scale: float = 10.0,
    context_len: int = None,
    future_len: int = None,
    ground_plane_z: Optional[float] = 0.0,
    show_gt: bool = True,
    multi_view: bool = False,        # If True, render 3 camera views per row
    obj0_height_color: bool = False,                # color obj0 by z-height (cloth-style)
    axis_limits: Optional[Tuple[Tuple[float, float], ...]] = None,
) -> None:
    """
    Create side-by-side GIF comparing ground truth and predicted trajectories.

    Particles are colored by object_ids when provided (distinct color per object),
    otherwise by action_mask (red=gripper, blue=object).

    Args:
        gt_positions: Ground truth positions [B, F, N, 3]
        pred_positions: Predicted positions [B, F, N, 3] or [S, B, F, N, 3] for
                        S ensemble runs, separate in single-view mode and overlaid
                        on the prediction row in multi-view mode.
        action_mask: Boolean mask, True for gripper particles [B, N]
        save_path: Where to save the GIF
        sample_idx: Which sample in the batch to visualize
        fps: Frames per second for GIF
        figsize: Figure size. When multi_view=True consider a wider value like (18, 10).
        valid_mask: Optional validity mask [B, N], True for valid particles
        object_ids: Optional object IDs [B, N] for per-object coloring
        actions: Optional action vectors [B, F, N, 3] to show as arrows on gripper
        arrow_scale: Scale factor for action arrows
        context_len: Number of context frames C (for border indicator)
        future_len: Frames predicted per step F (for border indicator)
        ground_plane_z: Z value for transparent ground plane, or None to disable
        show_gt: If False, only the prediction panel is shown
        multi_view: If True, render 3 camera angles (Perspective / Top-down / Side)
                    in columns instead of a single view per panel. Ensemble predictions
                    (5-D pred_positions) are supported with or without this flag.
        axis_limits: Optional ((xmin, xmax), (ymin, ymax), (zmin, zmax)) shared
                     across all frames/panels, for comparisons between separate GIFs.
    """
    # ------------------------------------------------------------------ #
    # 1. Extract single sample
    # ------------------------------------------------------------------ #
    gt   = gt_positions[sample_idx].cpu().numpy()   # [F, N, 3]
    mask = action_mask[sample_idx].cpu().numpy()    # [N]
    acts = actions[sample_idx].cpu().numpy() if actions is not None else None
    valid = (valid_mask[sample_idx].cpu().numpy()
             if valid_mask is not None
             else np.ones(mask.shape[0], dtype=bool))
    obj_ids = (object_ids[sample_idx].cpu().numpy().astype(int)
               if object_ids is not None else None)
    invalid = ~valid

    # Build list of prediction arrays — always [F, N, 3] per run
    if pred_positions.ndim == 5:                    # [S, B, F, N, 3]
        pred_list = [pred_positions[s, sample_idx].cpu().numpy()
                     for s in range(pred_positions.shape[0])]
    else:                                           # [B, F, N, 3]
        pred_list = [pred_positions[sample_idx].cpu().numpy()]

    num_runs = len(pred_list)
    if num_runs == 0:
        raise ValueError("pred_positions must contain at least one ensemble run")

    # Ensemble colour / alpha (no-op when num_runs == 1)
    cmap_name       = 'tab10' if num_runs <= 10 else 'tab20'
    ensemble_cmap   = plt.colormaps[cmap_name].resampled(max(num_runs, 1))
    ensemble_colors = [ensemble_cmap(i) for i in range(num_runs)]
    ensemble_alpha  = max(0.2, 0.7 / (num_runs ** 0.5))

    # Pad GT / predictions to a common length
    max_pred_len = max(p.shape[0] for p in pred_list)
    if gt.shape[0] < max_pred_len:
        pad = max_pred_len - gt.shape[0]
        gt  = np.concatenate([gt, np.tile(gt[-1:], (pad, 1, 1))], axis=0)
    pred_list = [
        np.concatenate([p, np.tile(p[-1:], (gt.shape[0] - p.shape[0], 1, 1))], axis=0)
        if p.shape[0] < gt.shape[0] else p
        for p in pred_list
    ]
    F, N, _ = gt.shape

    # ------------------------------------------------------------------ #
    # 2. Shared axis limits (cube aspect)
    # ------------------------------------------------------------------ #
    valid_gt = gt[:, valid, :]
    points_to_stack = [valid_gt.reshape(-1, 3)]
    for p in pred_list:
        points_to_stack.append(p[:, valid, :].reshape(-1, 3))

    all_positions = np.concatenate(points_to_stack, axis=0)
    x_min, x_max = np.nanmin(all_positions[:, 0]), np.nanmax(all_positions[:, 0])
    y_min, y_max = np.nanmin(all_positions[:, 1]), np.nanmax(all_positions[:, 1])
    z_min, z_max = np.nanmin(all_positions[:, 2]), np.nanmax(all_positions[:, 2])

    x_mid     = (x_max + x_min) / 2
    y_mid     = (y_max + y_min) / 2
    z_mid     = (z_max + z_min) / 2
    half_span = max(x_max - x_min, y_max - y_min, z_max - z_min, 1e-6) / 2 * 1.1

    x_lim = (x_mid - half_span, x_mid + half_span)
    y_lim = (y_mid - half_span, y_mid + half_span)
    z_lim = (z_mid - half_span, z_mid + half_span)
    if axis_limits is not None:
        limits = np.asarray(axis_limits, dtype=float)
        if (limits.shape != (3, 2) or not np.isfinite(limits).all()
                or not (limits[:, 0] < limits[:, 1]).all()):
            raise ValueError("axis_limits must contain three finite increasing (min, max) pairs")
        x_lim, y_lim, z_lim = limits

    # ------------------------------------------------------------------ #
    # 3. Per-object colour setup
    # ------------------------------------------------------------------ #
    if obj_ids is not None:
        unique_objs = sorted(set(obj_ids[valid].tolist()))
        obj_cmap    = plt.cm.tab10
        obj_colors  = {oid: obj_cmap(i % 10) for i, oid in enumerate(unique_objs)}

    # ------------------------------------------------------------------ #
    # 4. Autoregressive border
    # ------------------------------------------------------------------ #
    show_border = (context_len is not None) and (future_len is not None)
    ar_boundary = future_len if show_border else None

    # ------------------------------------------------------------------ #
    # 5. Panel renderer
    # ------------------------------------------------------------------ #
    def _render_panel(
        ax, pos, frame_acts, title, frame_idx,
        scatter_color=None, scatter_alpha=None,
        run_label=None, draw_extras=True,
    ):
        # Default alpha: opaque for single run, dimmed for ensemble
        if scatter_alpha is None:
            scatter_alpha = ensemble_alpha

        if obj_ids is not None:
            # ONE scatter for every object, not one per object.
            #
            # mplot3d depth-sorts POINTS inside a single Path3DCollection, but sorts separate
            # collections against each other only by their AVERAGE depth. Drawing cloth and
            # grippers as separate scatters therefore rendered the whole cloth cloud either
            # entirely in front of or entirely behind the whole gripper, depending on which
            # centroid happened to be nearer the camera. Measured on a real grasp: at
            # elev=20/azim=-60 that drew ~4.1k cloth points which are genuinely BEHIND the
            # gripper on top of it, in 12/12 frames; at elev=30/azim=45, 0/12. That is why the
            # artifact appeared on "some" examples -- it tracks the view angle, not the episode.
            # Concatenating into one collection makes the occlusion correct automatically.
            _pts, _cols, _sizes = [], [], []
            for oid in unique_objs:
                obj_mask   = valid & (obj_ids == oid)
                if obj_mask.sum() == 0:
                    continue
                is_gripper = mask[obj_mask].any()
                n_pts      = int(obj_mask.sum())
                if obj0_height_color and oid == 0 and scatter_color is None:
                    cols = height_colormap(pos[obj_mask, 2]).astype(np.float32) / 255.0
                else:
                    color = scatter_color if scatter_color is not None else obj_colors[oid]
                    cols  = np.tile(np.asarray(to_rgba(color), dtype=np.float32), (n_pts, 1))
                _pts.append(pos[obj_mask])
                _cols.append(cols)
                _sizes.append(np.full(n_pts, 3.0 if is_gripper else 1.0, dtype=np.float32))
            if _pts:
                _P = np.concatenate(_pts); _C = np.concatenate(_cols); _S = np.concatenate(_sizes)
                ax.scatter(_P[:, 0], _P[:, 1], _P[:, 2],
                           c=_C, s=_S, alpha=scatter_alpha, label=run_label)
        else:
            valid_object  = valid & ~mask
            valid_gripper = valid &  mask
            obj_color     = scatter_color if scatter_color is not None else 'blue'
            grip_color    = scatter_color if scatter_color is not None else 'red'
            if valid_object.sum() > 0:
                ax.scatter(
                    pos[valid_object,  0], pos[valid_object,  1], pos[valid_object,  2],
                    c=[obj_color], s=1, alpha=scatter_alpha, label=run_label or 'Object',
                )
            if valid_gripper.sum() > 0:
                ax.scatter(
                    pos[valid_gripper, 0], pos[valid_gripper, 1], pos[valid_gripper, 2],
                    c=[grip_color], s=3, alpha=min(scatter_alpha + 0.2, 1.0), label='Gripper',
                )

        if not draw_extras:
            ax.set_xlim(x_lim); ax.set_ylim(y_lim); ax.set_zlim(z_lim)
            return

        # Action arrows on gripper particles
        if frame_acts is not None:
            valid_gripper = valid & mask
            if valid_gripper.sum() > 0:
                gp   = pos[valid_gripper]
                ga   = frame_acts[valid_gripper]
                step = max(1, gp.shape[0] // 50)
                ax.quiver(
                    gp[::step, 0], gp[::step, 1], gp[::step, 2],
                    ga[::step, 0] * arrow_scale,
                    ga[::step, 1] * arrow_scale,
                    ga[::step, 2] * arrow_scale,
                    color='green', arrow_length_ratio=0.3, linewidth=0.8,
                )

        # Invalid (padded) particles
        if invalid.sum() > 0:
            ax.scatter(
                pos[invalid, 0], pos[invalid, 1], pos[invalid, 2],
                c='gray', s=0.5, alpha=0.1, label='Invalid',
            )

        # Ground plane
        if ground_plane_z is not None:
            xx, yy = np.meshgrid(
                np.linspace(x_lim[0], x_lim[1], 2),
                np.linspace(y_lim[0], y_lim[1], 2),
            )
            ax.plot_surface(xx, yy, np.full_like(xx, ground_plane_z),
                            color='gray', alpha=0.15, zorder=0)

        ax.set_xlim(x_lim); ax.set_ylim(y_lim); ax.set_zlim(z_lim)
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.set_title(title, fontsize=8 if multi_view else 10)
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc='upper right', fontsize=6 if multi_view else 8)

    # ------------------------------------------------------------------ #
    # 6. Build GIF frames
    # ------------------------------------------------------------------ #
    # Layout dimensions
    # multi_view=False  →  num_cols = number of GT/Pred panels (classic behaviour)
    # multi_view=True   →  num_cols = 3 camera views, num_rows = GT + Pred rows
    if multi_view:
        views    = ROTATIONS
        num_cols = len(views)
        num_rows = (1 if show_gt else 0) + 1
    else:
        views    = None
        num_cols = num_runs + int(show_gt)
        num_rows = 1

    # figsize: when show_gt=False and single-view, halve width (original behaviour)
    if not multi_view and not show_gt and num_runs == 1:
        effective_figsize = (figsize[0] // 2, figsize[1])
    else:
        effective_figsize = figsize

    frames = []
    for f in range(F):
        fig, axes = plt.subplots(
            num_rows, num_cols,
            figsize=effective_figsize,
            subplot_kw={'projection': '3d'},
        )
        # Normalise to 2-D array for uniform indexing
        if num_rows == 1 and num_cols == 1:
            axes = np.array([[axes]])
        elif num_rows == 1:
            axes = axes[np.newaxis, :]
        elif num_cols == 1:
            axes = axes[:, np.newaxis]

        # Border colouring for autoregressive boundary
        mode_label = ''
        if show_border:
            border_color = 'green' if f < ar_boundary else 'red'
            mode_label   = 'GT Context' if f < ar_boundary else 'Autoregressive'
            fig.patch.set_linewidth(8)
            fig.patch.set_edgecolor(border_color)

        suffix     = f' [{mode_label}]' if show_border else ''
        frame_acts = acts[f] if acts is not None else None

        if multi_view:
            # ── Multi-view layout ──────────────────────────────────────
            # GT row
            if show_gt:
                for col, (elev, azim, view_label) in enumerate(views):
                    ax = axes[0, col]
                    ax.view_init(elev=elev, azim=azim)
                    _render_panel(
                        ax, gt[f], frame_acts,
                        f'GT – {view_label} (f {f}/{F-1}){suffix}', f,
                    )

            # Prediction row
            pred_row = 1 if show_gt else 0
            for col, (elev, azim, view_label) in enumerate(views):
                ax = axes[pred_row, col]
                ax.view_init(elev=elev, azim=azim)
                for run_idx, pred_run in enumerate(pred_list):
                    is_first  = (run_idx == 0)
                    run_label = f'Run {run_idx}' if num_runs > 1 else None
                    _render_panel(
                        ax, pred_run[f],
                        frame_acts if is_first else None,
                        f'Pred – {view_label} (f {f}/{F-1}){suffix}', f,
                        scatter_color=ensemble_colors[run_idx] if num_runs > 1 else None,
                        scatter_alpha=ensemble_alpha if num_runs > 1 else None,
                        run_label=run_label,
                        draw_extras=is_first,
                    )
                # Restore axis labels/title/legend after multiple runs
                if num_runs > 1:
                    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
                    ax.set_title(f'Pred – {view_label} (f {f}/{F-1}){suffix}', fontsize=8)
                    ax.legend(loc='upper right', fontsize=6)

        else:
            # ── Classic single-view layout ────────
            col = 0
            if show_gt:
                _render_panel(
                    axes[0, col], gt[f], frame_acts,
                    f'Ground Truth (Frame {f}/{F-1}){suffix}', f,
                )
                col += 1

            for run_idx, pred_run in enumerate(pred_list):
                run_label = f'Run {run_idx}' if num_runs > 1 else None
                _render_panel(
                    axes[0, col + run_idx], pred_run[f],
                    frame_acts,
                    (f'Prediction' if num_runs == 1 else f'Run {run_idx}')
                    + f' (Frame {f}/{F-1}){suffix}', f,
                    scatter_color=ensemble_colors[run_idx] if num_runs > 1 else None,
                    scatter_alpha=ensemble_alpha if num_runs > 1 else None,
                    run_label=run_label,
                    draw_extras=True,
                )

        plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.98] if show_border else None)
        fig.canvas.draw()
        img = np.array(fig.canvas.buffer_rgba())[:, :, :3]
        frames.append(img)
        plt.close(fig)

    # ------------------------------------------------------------------ #
    # 7. Save
    # ------------------------------------------------------------------ #
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(save_path, frames, duration=1000 / fps, loop=0)
    print(f"Saved: {save_path}")
