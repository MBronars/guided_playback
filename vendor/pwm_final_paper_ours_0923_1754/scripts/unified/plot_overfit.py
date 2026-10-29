"""Plot completed production-batch diagnostics from full-precision JSON.

Three separate views: noisy optimization history, controlled-t objectives, and
physical rollout errors. Enforces matched fixtures/update counts and never infers
generalization from memorization. See docs/loss_viz_rules.md.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf

from scripts.unified.lossprobe import guard, tail
from utils.trajectory_objective import trajectory_loss_options

COLORS = ("#1967a3", "#ce5317", "#27834d", "#8054a2")


def load_runs(runs, domains, *, physical_only=False):
    records = {}
    for label, directory in runs:
        records[label] = {}
        for domain in domains:
            path = directory / f"{domain}.json"
            data = json.loads(path.read_text())
            guard(f"{label}/{domain}", data["history"])
            if data["evals"][-1]["step"] != data["steps"]:
                raise ValueError(f"{path}: unfinished run")
            if data["evals"][-1]["domain_updates"] != len(data["history"]):
                raise ValueError(f"{path}: inconsistent update counts")
            values = np.asarray(data["final_multi_seed_mm"]["values"], dtype=float)
            if len(values) < 2 or not np.isfinite(values).all() or (values < 0).any():
                raise ValueError(f"{path}: need at least two finite nonnegative rollout scores")
            records[label][domain] = data
    for domain in domains:
        group = [r[domain] for r in records.values()]
        hashes = {r["batch_receipt"].get("sha256") for r in group}
        if len(hashes) != 1 or None in hashes:
            raise ValueError(f"{domain}: runs need identical hashed batch receipts")
        if len({len(r["history"]) for r in group}) != 1:
            raise ValueError(f"{domain}: runs have different update counts")
        rows = [r["batch_receipt"].get("selected_rows",
                    list(range(r["B"])) if "B" in r else None) for r in group]
        if any(row != rows[0] for row in rows):
            raise ValueError(f"{domain}: selected fixture rows differ")
        if any(any(r.get(key) != group[0].get(key) for key in ("B", "C", "F")) for r in group):
            raise ValueError(f"{domain}: batch/window contracts differ")
        if any(r["baselines"] != group[0]["baselines"] for r in group):
            raise ValueError(f"{domain}: physical baseline contracts differ")
        if not physical_only:
            if len({r["stats_set"] for r in group}) != 1:
                raise ValueError(f"{domain}: normalization differs; use --physical_only")
            configs = [r.get("provenance", {}).get("resolved_config", {}) for r in group]
            snapshots = [c.get("normalization", {}).get("sha256") for c in configs]
            targets = [c.get("flow_matching", {}).get("prediction_type") for c in configs]
            if any(value != snapshots[0] for value in snapshots):
                raise ValueError(f"{domain}: normalization snapshot differs; use --physical_only")
            if any(value != targets[0] for value in targets):
                raise ValueError(f"{domain}: prediction parameterization differs; use --physical_only")
            objectives = [trajectory_loss_options(OmegaConf.create({"train": {}, **c})) for c in configs]
            if any(value != objectives[0] for value in objectives):
                raise ValueError(f"{domain}: objective settings differ; use --physical_only")
    return records


def panels(domains, title, subtitle):
    cols = min(2, len(domains))
    rows = (len(domains) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12.8, max(5.6, 4.3 * rows)), squeeze=False)
    fig.suptitle(title, fontsize=16, y=0.995)
    fig.text(0.5, 0.955, subtitle, ha="center", va="top", fontsize=10)
    for ax, domain in zip(axes.flat, domains):
        ax.set_title(domain.upper(), loc="left", fontsize=12, fontweight="bold")
        ax.grid(alpha=0.16, which="both")
        ax.spines[["top", "right"]].set_visible(False)
    for ax in list(axes.flat)[len(domains):]:
        ax.set_visible(False)
    return fig, list(axes.flat)[:len(domains)]


def save(fig, axes, output, caption):
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.034),
               ncol=min(4, len(labels)), frameon=False)
    fig.text(0.5, 0.012, caption, ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.09, 1, 0.93), h_pad=2.0, w_pad=2.2)
    fig.savefig(output, dpi=160, facecolor="white")
    plt.close(fig)


def _plot_objectives(records, domains, output, title):
    arms = list(records)
    fig, axes = panels(domains, title + " — optimization history",
                       "Raw random-t losses (faint) + trailing 201-update medians (bold); descriptive, not a quality gate")
    for domain, ax in zip(domains, axes):
        notes = []
        for index, label in enumerate(arms):
            data = records[label][domain]
            values = guard(f"{label}/{domain}", data["history"])
            x = np.arange(1, len(values) + 1)
            medians = np.median(np.lib.stride_tricks.sliding_window_view(values, 201), axis=-1)
            ax.plot(x, values, color=COLORS[index], alpha=0.12, lw=0.6)
            ax.plot(x[200:], medians, color=COLORS[index], lw=2, label=label)
            stats = tail(values)
            notes.append(f"{label}: tail median {stats['median']:.3g}; scatter {stats['scatter']:.1f}×\n"
                         f"  tail log-slope {stats['slope_dex_per_1k']:+.2f} dex / 1K updates")
        ax.text(0.98, 0.97, "\n".join(notes), transform=ax.transAxes, ha="right", va="top",
                fontsize=8, bbox=dict(facecolor="white", edgecolor="none", alpha=0.82))
        ax.set_yscale("log")
        ax.set_xlabel("Optimizer updates on this domain")
        ax.set_ylabel("Production objective (random-t mixture)")
    save(fig, axes, output / "loss.png",
         "Tail = last third; scatter = max/min. Slopes are noisy descriptive statistics, not evidence of a plateau.")

    fig, axes = panels(domains, title + " — controlled noise levels",
                       "Mean of four fixed noise seeds per t; shaded ±1 SD. Dashed = initialization, solid = final weights")
    for domain, ax in zip(domains, axes):
        grids = []
        for index, label in enumerate(arms):
            data = records[label][domain]
            for position in (0, -1):
                sweep = data["evals"][position]["fixed_t_objective"]
                grid = sorted(float(t) for t in sweep)
                grids.append(grid)
                means, std = np.asarray([sweep[str(t)] for t in grid], dtype=float).T
                guard(label, means, min_points=1)
                if not np.isfinite(std).all() or (std < 0).any() or (means - std <= 0).any():
                    raise ValueError(f"{label}/{domain}: SD band cannot be drawn honestly on a log axis")
                final = position == -1
                ax.plot(grid, means, color=COLORS[index], ls="-" if final else "--",
                        lw=2 if final else 1, alpha=1 if final else 0.5,
                        label=label if final else None)
                if final:
                    ax.fill_between(grid, means - std, means + std, color=COLORS[index], alpha=0.14)
        if any(grid != grids[0] for grid in grids):
            raise ValueError(f"{domain}: fixed-t grids differ")
        ax.set_yscale("log")
        ax.set_xlabel("Normalized time t (all noised context/future frames)")
        ax.set_ylabel("Controlled-t production objective")
    save(fig, axes, output / "fixed_t.png",
         "Controlled slices are not the training expectation. Different prediction parameterizations are not comparable in loss height.")

def plot(records, domains, output, title, *, physical_only=False):
    output.mkdir(parents=True, exist_ok=True)
    if physical_only and any((output / name).exists() for name in ("loss.png", "fixed_t.png")):
        raise ValueError("Choose a new physical-only output directory; stale loss plots would be misleading")
    if not physical_only:
        _plot_objectives(records, domains, output, title)
    arms = list(records)
    fig, axes = panels(domains, title + " — physical rollout error",
                       "Lines: one fixed sampling seed through training. Dots at right: fixed final seeds; bars: mean ±1 SD")
    for domain, ax in zip(domains, axes):
        all_values = []
        for index, label in enumerate(arms):
            data = records[label][domain]
            x = [entry["domain_updates"] for entry in data["evals"]]
            values = np.asarray([entry["sample_mean_mm"] for entry in data["evals"]])
            seeds = np.asarray(data["final_multi_seed_mm"]["values"])
            all_values.extend(values)
            all_values.extend(seeds)
            ax.plot(x, values, "o-", color=COLORS[index], ms=4, lw=1.8, label=label)
            center = x[-1] * (1.055 + index * 0.04)
            jitter = np.linspace(-0.009, 0.009, len(seeds)) * x[-1]
            ax.scatter(center + jitter, seeds, color=COLORS[index], s=15, alpha=0.6)
            ax.errorbar(center, seeds.mean(), yerr=seeds.std(), color=COLORS[index], fmt="_", capsize=4)
        first = records[arms[0]][domain]
        for key, style, label in (("zero_motion", "-", "No motion"),
                                   ("hold_velocity", "--", "Last velocity")):
            baseline = float(first["baselines"][key][0])
            all_values.append(baseline)
            ax.axhline(baseline, color="#666666", ls=style, lw=1.1, label=label)
        if not np.isfinite(all_values).all() or np.min(all_values) < 0:
            raise ValueError(f"{domain}: invalid physical errors")
        if np.min(all_values) > 0:
            ax.set_yscale("log")
        ax.set_xlabel("Optimizer updates on this domain (final-seed dots offset)")
        ax.set_ylabel("Mean particle-position error (mm)")
    save(fig, axes, output / "rollout.png",
         "All-valid, frame-mean errors; raw baselines ignore known actions. Memorization only; sampling SD is not a generalization CI.")
    summary = {label: {domain: {"batch_receipt": data["batch_receipt"],
                "domain_updates": len(data["history"]), "tail": tail(data["history"]),
                "baselines": data["baselines"], "final_multi_seed_mm": data["final_multi_seed_mm"]}
                for domain, data in group.items()} for label, group in records.items()}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True, metavar="LABEL=DIRECTORY")
    parser.add_argument("--domains", default="mpm,pickup,cloth,pushing")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="Full-capacity fixed-batch diagnostic")
    parser.add_argument("--physical_only", action="store_true",
                        help="Compare physical mm across profiles/parameterizations without overlaying loss heights")
    args = parser.parse_args()
    runs = [(item.split("=", 1)[0], Path(item.split("=", 1)[1])) for item in args.runs]
    if not 1 <= len(runs) <= len(COLORS) or len({label for label, _ in runs}) != len(runs):
        parser.error("Use one to four distinctly named runs")
    domains = args.domains.split(",")
    plot(load_runs(runs, domains, physical_only=args.physical_only), domains,
         args.output, args.title, physical_only=args.physical_only)


if __name__ == "__main__":
    main()
