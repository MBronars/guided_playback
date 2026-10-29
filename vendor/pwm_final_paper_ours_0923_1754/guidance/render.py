"""Render a run_guidance dump as a GIF: ground-truth episode beside the planned
rollout, top-down (x, y) and side (x, z) views, the passive body's target marked.

    PYTHONPATH=. python guidance/render.py dump.npz --element 0 --out plan.gif
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter


def render(path: str, out: str, element: int = 0, fps: int = 6):
    d = np.load(path, allow_pickle=True)
    e = element
    gt, pl = d["gt"][e], d["planned"][e]                      # [T, N, 3]
    C = int(d["context_len"])
    valid = d["valid"][e]
    act, passive = d["action_mask"][e] & valid, d["passive_rows"][e] & valid
    other = valid & ~act & ~passive
    tgt = d["target_t"][e]
    T = gt.shape[0]
    allp = np.concatenate([gt, pl], 0)[:, valid]
    lo, hi = allp.min(axis=(0, 1)), allp.max(axis=(0, 1))
    lo, hi = np.minimum(lo, tgt - 0.01), np.maximum(hi, tgt + 0.01)
    pad = 0.06 * (hi - lo).max()
    fig, axes = plt.subplots(2, 2, figsize=(9, 7.4))
    panels = [(0, 0, "ground truth", gt, 0, 1), (0, 1, f"planned ({str(d['planner'])})", pl, 0, 1),
              (1, 0, "ground truth, side", gt, 0, 2), (1, 1, "planned, side", pl, 0, 2)]
    arts = []
    for r, c, title, pos, i, j in panels:
        ax = axes[r, c]
        ax.set_title(title, fontsize=10)
        ax.set_xlim(lo[i] - pad, hi[i] + pad); ax.set_ylim(lo[j] - pad, hi[j] + pad)
        ax.set_aspect("equal"); ax.tick_params(labelsize=7)
        ax.set_xlabel("xyz"[i] + " (m)", fontsize=8); ax.set_ylabel("xyz"[j] + " (m)", fontsize=8)
        s_other = ax.scatter(pos[0, other, i], pos[0, other, j], s=3, c="#9aa3ad")
        s_pass = ax.scatter(pos[0, passive, i], pos[0, passive, j], s=5, c="#1e6e8c", label="passive body")
        s_act = ax.scatter(pos[0, act, i], pos[0, act, j], s=5, c="#c0392b", label="actuator")
        ax.plot(gt[-1, passive, i].mean(), gt[-1, passive, j].mean(), "x", c="#1e6e8c", ms=8, label="GT final centre")
        ax.plot(tgt[i], tgt[j], "*", c="#2e8b57", ms=12, label="target")
        txt = ax.text(0.02, 0.96, "", transform=ax.transAxes, fontsize=8, va="top")
        arts.append((s_other, s_pass, s_act, txt, pos, i, j))
    axes[0, 1].legend(fontsize=7, loc="lower right")
    dist_final = float(np.linalg.norm(pl[-1, passive].mean(0) - tgt) * 1000)
    fig.suptitle(f"{str(d['objective'])}: planned body centre ends {dist_final:.1f} mm from target "
                 f"(receipt {float(d['target_distance_mm'][e]):.1f} mm)", fontsize=10)

    def update(f):
        for s_other, s_pass, s_act, txt, pos, i, j in arts:
            s_other.set_offsets(pos[f][other][:, [i, j]])
            s_pass.set_offsets(pos[f][passive][:, [i, j]])
            s_act.set_offsets(pos[f][act][:, [i, j]])
            txt.set_text(f"frame {f}" + (" (context)" if f < C else ""))
        return [a for tup in arts for a in tup[:4]]

    anim = FuncAnimation(fig, update, frames=T, interval=1000 // fps, blit=False)
    anim.save(out, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dump"); ap.add_argument("--element", type=int, default=0)
    ap.add_argument("--out", required=True); ap.add_argument("--fps", type=int, default=6)
    a = ap.parse_args()
    print(render(a.dump, a.out, a.element, a.fps))
