"""Render saved physical-unit overfit predictions and measured learning curves.

No model execution or changes to receipts. Point thinning is for display only;
reported metrics come from the full saved training batch in overfit_prod.py.
"""
from __future__ import annotations

import argparse
import html
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from PIL import Image


def stable_limits(points):
    lo, hi = points.min(axis=(0, 1)), points.max(axis=(0, 1))
    center, radius = (lo + hi) / 2, max(float((hi - lo).max()) / 2, .015) * 1.12
    return [(float(c - radius), float(c + radius)) for c in center]


def scene(ax, points, colors, limits, title):
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=2, c=colors, alpha=.8, linewidths=0)
    ax.set(xlim=limits[0], ylim=limits[1], zlim=limits[2], title=title,
           xlabel="x (m)", ylabel="y (m)", zlabel="z (m)")
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=24, azim=-55)
    ax.tick_params(labelsize=7)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_major_locator(MaxNLocator(nbins=4))
        axis.label.set_fontsize(9)


def render_sample(path, out, title, deformable):
    with np.load(path, allow_pickle=False) as z:
        gt, pred, ctx = z["gt"], z["pred"], z["context"]
        actions, ids = z["action_mask"], z["object_ids"]
    selected = np.linspace(0, gt.shape[1] - 1, min(1500, gt.shape[1])).astype(int)
    gt, pred, ctx, actions, ids = gt[:, selected], pred[:, selected], ctx[:, selected], actions[selected], ids[selected]
    primary = np.ones(len(actions), bool) if deformable else ~actions
    colors = plt.get_cmap("tab10")(ids % 10)
    colors[actions] = [.55, .55, .55, .55]
    limits = stable_limits(np.concatenate([gt, pred, ctx], 0))
    errors = np.linalg.norm(pred - gt, axis=-1) * 1000
    frames = sorted(set([0, len(gt) // 2, len(gt) - 1]))
    fig = plt.figure(figsize=(12, 9.5))
    fig.suptitle(title + " — recorded / sampled prediction (metres)\n"
                 "Prediction captions report mean display-point error", fontsize=12)
    for col, frame in enumerate(frames):
        scene(fig.add_subplot(2, 3, col + 1, projection="3d"), gt[frame], colors, limits, f"Recorded, future {frame + 1}")
        scene(fig.add_subplot(2, 3, col + 4, projection="3d"), pred[frame], colors, limits,
              f"Prediction · {errors[frame, primary].mean():.2f} mm")
    fig.subplots_adjust(left=.015, right=.97, bottom=.06, top=.88, hspace=.22, wspace=.03)
    fig.savefig(out.with_suffix(".png"), dpi=140)
    plt.close(fig)
    images = []
    for frame in range(len(gt)):
        fig = plt.figure(figsize=(8, 4))
        scene(fig.add_subplot(121, projection="3d"), gt[frame], colors, limits, f"Recorded — future {frame + 1}")
        scene(fig.add_subplot(122, projection="3d"), pred[frame], colors, limits, f"Prediction — future {frame + 1}")
        fig.suptitle(title)
        fig.tight_layout()
        memory = io.BytesIO()
        fig.savefig(memory, format="png", dpi=85)
        memory.seek(0)
        images.append(Image.open(memory).convert("RGB"))
        plt.close(fig)
    images[0].save(out.with_suffix(".gif"), save_all=True, append_images=images[1:], duration=150, loop=0)


def render_curves(result, out, title):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    loss = np.asarray(result["history"])
    if len(loss):
        width = min(50, len(loss))
        axes[0].plot(np.arange(width, len(loss) + 1), np.convolve(loss, np.ones(width) / width, "valid"))
        axes[0].set_yscale("log")
    axes[0].set(title="Training objective, 50-update mean", xlabel="Updates", ylabel="Loss")
    evals = result["evals"]
    x = [r["step"] for r in evals]
    axes[1].plot(x, [r["sample_primary_mean_mm"] for r in evals], "o-", label="Fit fixture")
    if all("holdout_primary_mean_mm" in r for r in evals):
        axes[1].plot(x, [r["holdout_primary_mean_mm"] for r in evals], "o-", label="Development fixture")
    for name, values in result["primary_baselines"].items():
        axes[1].axhline(values[0], linestyle="--", label="Fit " + name.replace("_", " "))
    axes[1].set(title="Sampler error on predicted particles", xlabel="Updates", ylabel="Mean position error (mm)")
    axes[1].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def build(root, output):
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows, sections = [], []
    for path in sorted(root.glob("*/*.json")):
        result = json.loads(path.read_text())
        if "evals" not in result or "primary_baselines" not in result:
            continue
        run, domain = path.parent.name, result["domain"]
        title = f"{domain} · {result['arch']} · {result['evals'][-1]['step']:,} updates"
        note = ("Superseded diagnostic: this initial UNet drag run clipped command inputs."
                if run == "unet_mpm_drag_v1" else "")
        complete = "final_multi_seed_primary_mm" in result
        metric = result.get("final_multi_seed_primary_mm", {})
        baseline = min(v[0] for v in result["primary_baselines"].values())
        mean = metric.get("mean", result["evals"][-1]["sample_primary_mean_mm"])
        rows.append(dict(run=run, domain=domain, architecture=result["arch"],
            updates=result["evals"][-1]["step"], complete=complete, seeds=metric.get("n", 1),
            primary_mean_mm=mean, primary_std_mm=metric.get("std"), best_baseline_mm=baseline,
            baseline_ratio=mean / baseline, fixture_sha256=result["batch_receipt"]["sha256"],
            receipt=str(path), N=result["N"], note=note))
        curve = output / f"{run}_curves.png"
        render_curves(result, curve, title)
        sample = path.parent / f"{domain}_sample_step{result['evals'][-1]['step']:05d}.npz"
        visual = output / f"{run}_sample"
        if sample.exists():
            render_sample(sample, visual, title, domain in ("mpm_fresh", "mpm_drag"))
        sections.append(f'<section id="{html.escape(run)}"><h2>{html.escape(title)}</h2>'
            f'<p><code>{html.escape(run)}</code> {html.escape(note)}</p><p>{mean:.2f} mm '
            f'({metric.get("n", 1)} sampler seeds); best fit-fixture baseline {baseline:.2f} mm. '
            f'{"Complete" if complete else "In progress"}. Full batch N={result["N"]}.</p>'
            f'<img src="{curve.name}"><a href="{visual.with_suffix(".png").name}">'
            f'<img src="{visual.with_suffix(".gif").name}"></a></section>')
    (output / "comparison.json").write_text(json.dumps(rows, indent=2) + "\n")
    lines = ["| Run | Updates | Predicted-particle error (mm) | Best baseline (mm) | Ratio | Seeds |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f'| {row["run"]} | {row["updates"]} | {row["primary_mean_mm"]:.2f} | '
            f'{row["best_baseline_mm"]:.2f} | {row["baseline_ratio"]:.3f} | {row["seeds"]} |')
    (output / "COMPARISON.md").write_text("\n".join(lines) + "\n")
    table = ('<table><thead><tr><th>Run</th><th>Updates</th><th>Fit error, mm</th>'
             '<th>Best simple baseline, mm</th><th>Error / baseline</th></tr></thead><tbody>')
    for row in rows:
        uncertainty = (f' ± {row["primary_std_mm"]:.2f}'
                       if row["primary_std_mm"] is not None else " (unfinished)")
        table += (f'<tr><td><a href="#{html.escape(row["run"])}">{html.escape(row["run"])}</a></td>'
                  f'<td>{row["updates"]:,}</td><td>{row["primary_mean_mm"]:.2f}{uncertainty}</td>'
                  f'<td>{row["best_baseline_mm"]:.2f}</td><td>{row["baseline_ratio"]:.3f}</td></tr>')
    table += '</tbody></table>'
    (output / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>Collection overfit evidence</title>'
        '<style>body{font:16px system-ui;max-width:1150px;margin:35px auto;padding:0 15px;background:#fafafa;color:#18212b}'
        'section{margin:40px 0;border-top:1px solid #ccc}img{max-width:100%;display:block;margin:15px 0}'
        'table{border-collapse:collapse;width:100%;font-size:13px}td,th{text-align:left;padding:7px;border-bottom:1px solid #ddd}</style>'
        '<h1>Collection overfit evidence</h1><p>Production UNet and Transformer, C=4 / F=16, batch size two. '
        'No training noise, no target clipping. The development fixture is never trained on. '
        'These are fixed-fixture engineering checks, not estimates of generalization or long-rollout quality. '
        'Gray points are actuated. Display uses at most 1,500 points from the saved preview; metrics use the full batch. '
        'The sampling seed for animations is fixed at 4242. Compare architectures at equal update budgets and on '
        'the same fixture; longer runs start from scratch with their own schedule. Error bars summarize eight '
        'sampler seeds, not eight training runs. The original unet_mpm_drag_v1 is superseded by the corrected '
        'command-input runs.</p>' + table + ''.join(sections))
    print(f"Rendered {len(rows)} runs in {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    build(args.root, args.out)
