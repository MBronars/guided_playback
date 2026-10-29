"""
Unified-model inference on passive MPM h5 data.

Reads h5 files directly, constructs LongTrajectoryBatch, rolls each clip out with
inference.windowed_rollout (the evaluators' rollout) and scores it with
utils.evaluation.TrajectoryScores (millimetres, the evaluators' vocabulary).

Usage:
    python scripts/inference_mpm/inference_mpm_multi.py \
        --checkpoint runs/flow_mpm_multi_v2_6gpu/checkpoints/step_0195000.pt \
        --h5_dir /data/.../mpm_dataset/multi_object_v2/h5/3obj \
        --max_files 5 \
        --output_dir ./inference_results/mpm_multi_v2 \
        --visualize --verbose
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path when running as script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import h5py
import torch
from tqdm import tqdm

from data.normalization import domain_normalizers, resolve_normalization
from data.unified_dataset import _shift
from data.test_batch import LongTrajectoryBatch
from inference import load_model_and_diffusion, plot_error_vs_frames, windowed_rollout
from utils.evaluation import TrajectoryScores, validate_request
from utils.reproducibility import code_revision, file_sha256


def load_h5_as_long_batch(
    h5_path: str,
    delta_stats,
    position_stats,
    prediction_target: str = "delta",
) -> LongTrajectoryBatch:
    """
    Load a single h5 file into a LongTrajectoryBatch (B=1).

    h5 keys:
        x: [T, N, 3] positions
        object_ids: [N] per-particle object assignment
        E: [n_objects] Young's modulus
        nu: [n_objects] Poisson's ratio
        density: [n_objects] density
        mat_type: [n_objects] material type index
    """
    with h5py.File(h5_path, "r") as hf:
        for key in ('selected','command_velocity','transition_active','is_grabbed','grab_velocity','drag_active'):
            if key in hf and hf[key][()].any():
                raise ValueError('Passive H5 inference cannot discard active MPM commands: '+key)
        x = torch.from_numpy(hf["x"][:]).float()              # [T, N, 3]
        obj_ids = torch.from_numpy(hf["object_ids"][:]).long() # [N]
        E_obj = torch.from_numpy(hf["E"][:]).float()           # [n_objects]
        nu_obj = torch.from_numpy(hf["nu"][:]).float()         # [n_objects]
        rho_obj = torch.from_numpy(hf["density"][:]).float()   # [n_objects]
        mat_obj = torch.from_numpy(hf["mat_type"][:]).long()   # [n_objects]

    if x.ndim != 3 or x.shape[-1] != 3 or x.shape[0] < 2 or x.shape[1] < 1:
        raise ValueError(f"{h5_path}: expected nonempty [T>=2,N,3] positions")
    T, N, _ = x.shape
    if obj_ids.shape != (N,) or obj_ids.min() < 0:
        raise ValueError(f"{h5_path}: invalid per-particle object IDs")
    for values in (E_obj, nu_obj, rho_obj, mat_obj):
        if values.ndim != 1 or obj_ids.max() >= len(values) or not torch.isfinite(values).all():
            raise ValueError(f"{h5_path}: invalid object property table")
    if not torch.isfinite(x).all():
        raise ValueError(f"{h5_path}: nonfinite particle positions")
    x = x + _shift("mpm")

    # Map per-object properties to per-particle
    E = E_obj[obj_ids]        # [N]
    nu = nu_obj[obj_ids]      # [N]
    rho = rho_obj[obj_ids]    # [N]
    mat_type = mat_obj[obj_ids]  # [N]

    # Passive simulation: no actions, no actuated particles
    actions = torch.zeros(1, T - 1, N, 3)
    action_mask = torch.zeros(1, N, dtype=torch.bool)

    return LongTrajectoryBatch(
        positions=x.unsqueeze(0),           # [1, T, N, 3]
        actions=actions,                     # [1, T-1, N, 3]
        action_mask=action_mask,             # [1, N]
        object_ids=obj_ids.unsqueeze(0),     # [1, N]
        valid_lengths=torch.tensor([N]),     # [1]
        E=E.unsqueeze(0),                   # [1, N]
        nu=nu.unsqueeze(0),                 # [1, N]
        rho=rho.unsqueeze(0),               # [1, N]
        mat_type=mat_type.unsqueeze(0),     # [1, N]
        delta_stats=delta_stats,
        position_stats=position_stats,
        prediction_target=prediction_target,
    )


def main():
    parser = argparse.ArgumentParser(description="MPM multi-object inference on h5 data")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--h5_dir", type=str, required=True, help="Directory with .h5 files")
    parser.add_argument("--output_dir", type=str, default="./inference_results/mpm_multi",
                        help="Output directory for results")
    parser.add_argument("--max_files", type=int, default=None, help="Max h5 files to process")
    parser.add_argument("--visualize", action="store_true", help="Generate visualization GIFs")
    parser.add_argument("--verbose", action="store_true", help="Print detailed progress")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.max_files is not None and args.max_files < 1:
        raise ValueError("max_files must be positive")
    if "drag" in str(Path(args.h5_dir).resolve()).lower():
        raise ValueError("Active MPM drag is not supported by the passive unified adapter")
    torch.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model, diffusion, cfg, inference_steps = load_model_and_diffusion(args.checkpoint, device)

    normalization = resolve_normalization(cfg.normalization)
    delta_stats, position_stats, _ = domain_normalizers(normalization, "mpm")

    prediction_target = getattr(cfg, "prediction_target", "delta")
    if cfg.data.format != "unified" or prediction_target != "delta":
        raise ValueError("H5 MPM evaluation requires a unified delta checkpoint")
    context_len = int(cfg.data.context_len)
    future_len = int(cfg.data.future_len)
    validate_request(1, context_len, future_len)
    schedule = cfg.get("df_schedule", "pyramid")
    uncertainty_scale = float(cfg.get("df_uncertainty_scale", 1.0))
    print(f"Rollout: inference.windowed_rollout, C={context_len}, F={future_len}")

    # Find h5 files
    h5_dir = Path(args.h5_dir)
    h5_files = sorted(h5_dir.glob("*.h5"))
    if not h5_files:
        raise ValueError(f"No .h5 files found in {h5_dir}")
    if args.max_files:
        h5_files = h5_files[: args.max_files]
    print(f"Found {len(h5_files)} h5 files")

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    overall, per_file = TrajectoryScores(), []
    for file_idx, h5_path in enumerate(tqdm(h5_files, desc="Processing h5 files")):
        sequence_data = load_h5_as_long_batch(
            h5_path, delta_stats, position_stats, prediction_target,
        ).to(device)
        T = sequence_data.num_frames
        if T <= context_len:
            raise ValueError(f"{h5_path}: no future frames beyond checkpoint context")
        batch = sequence_data.get_window(start_frame=0, context_len=context_len, future_len=T - context_len)

        if args.verbose:
            print(f"\nFile {file_idx}: {h5_path.name}")
            print(f"  Shape: T={T}, N={sequence_data.num_particles}")

        predicted = windowed_rollout(model, diffusion, batch, horizon=T - context_len, future_len=future_len,
                                     num_inference_steps=inference_steps, schedule=schedule,
                                     uncertainty_scale=uncertainty_scale)["positions"]
        scores = TrajectoryScores(per_frame=True)
        scores.add(batch, predicted[:, context_len:], batch.positions[:, context_len:])
        overall.add(batch, predicted[:, context_len:], batch.positions[:, context_len:])
        per_file.append({"file": h5_path.name, **scores.result()})

        if args.verbose:
            print(f"  particle_mean_mm {per_file[-1]['particle_mean_mm']:.3f}  "
                  f"particle_final_mm {per_file[-1]['particle_final_mm']:.3f}")

        # Visualization
        if args.visualize:
            from utils.visualization import create_trajectory_comparison_gif

            viz_dir = output_dir / "visualizations"
            viz_dir.mkdir(exist_ok=True)
            gif_path = str(viz_dir / f"{h5_path.stem}_trajectory.gif")
            create_trajectory_comparison_gif(
                gt_positions=batch.positions,
                pred_positions=predicted,
                action_mask=sequence_data.action_mask,
                save_path=gif_path,
                sample_idx=0,
                valid_mask=sequence_data.get_valid_mask(),
                object_ids=sequence_data.object_ids,
                context_len=context_len,
                future_len=future_len,
            )
            if args.verbose:
                print(f"  Saved: {gif_path}")

    result = overall.result()
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Files processed: {len(per_file)}")
    print(f"particle_mean_mm: {result['particle_mean_mm']:.3f}")
    print(f"particle_final_mm: {result['particle_final_mm']:.3f}")

    # Plot error vs frames
    plot_path = output_dir / "error_vs_frames.png"
    _, mean_curve, std_curve = plot_error_vs_frames(
        all_per_frame_mse=[m["particle_per_frame_mm"] for m in per_file],
        context_len=context_len,
        future_len=future_len,
        save_path=str(plot_path),
        title=f"MPM particle error (n={len(per_file)} files)",
        ylabel="Particle error (mm)",
    )
    print(f"Saved error plot to: {plot_path}")

    # Save metrics
    metrics_path = output_dir / "metrics.json"
    metrics_dict = {
        "num_files": len(per_file),
        **result,
        "mean_per_frame_mm": mean_curve.tolist(),
        "std_per_frame_mm": std_curve.tolist(),
        "per_file": [{key: m[key] for key in ("file", "particle_mean_mm", "particle_final_mm")}
                     for m in per_file],
        "config": {
            "checkpoint": args.checkpoint,
            "checkpoint_sha256": file_sha256(args.checkpoint),
            "normalization": normalization,
            "canonical_shift_m": _shift("mpm").tolist(),
            "seed": args.seed,
            "rollout": "inference.windowed_rollout",
            "df_schedule": schedule,
            "df_uncertainty_scale": uncertainty_scale,
            "h5_dir": args.h5_dir,
            "context_len": context_len,
            "future_len": future_len,
            "num_inference_steps": inference_steps,
        },
        **code_revision(),
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics_dict, f, indent=2, allow_nan=False)
    print(f"Saved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
