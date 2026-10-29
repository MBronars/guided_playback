"""Block-pushing figure from one render folder (default push_v3): Ours pushes the block onto the target, the baselines stall.
  --layout column : rows = method, columns = Episode 1 / Episode 2   (single paper column, 3.4 in wide)
  --layout wide   : rows = episode, columns = method                 (full width, 7 in)
Each episode is cropped to the union of its three panels' content (same crop for every method), padded to 4:3.
  .../foundationpose/bin/python bin/make_push_pdf.py [--layout column|wide] [--src figures/push_v3] [--out figures/paper_pdf/push_ours_vs_stall]"""
import argparse, glob, json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches, font_manager as fm
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
ap = argparse.ArgumentParser(); ap.add_argument("--src", default=os.path.join(ROOT, "figures", "push_v4"))
ap.add_argument("--layout", default="column", choices=("column", "wide")); ap.add_argument("--out", default=None)
ap.add_argument("--pad", type=float, default=0.05, help="crop padding, fraction of the image width")
A = ap.parse_args(); out = A.out or os.path.join(ROOT, "figures", "paper_pdf", "push_ours_vs_stall" + ("" if A.layout == "column" else "_wide"))
for p in glob.glob(os.path.expanduser("~/.local/share/fonts/Times*.TTF")): fm.fontManager.addfont(p)   # Microsoft core-fonts Times New Roman
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"], "mathtext.fontset": "custom", "mathtext.rm": "Times New Roman",
                     "mathtext.it": "Times New Roman:italic", "mathtext.bf": "Times New Roman:bold", "font.size": 7.5, "pdf.fonttype": 42})
cases = {(r["ep"], r["method"]): r for r in json.load(open(os.path.join(A.src, "push_cases.json"))) if "method" in r}
EPS = sorted({e for e, _ in cases}); METHODS = ["Ours", "PointWorld", "GNS"]
INK = "#1c2321"; MUTED = "#5d6864"; MAG = "#d6288a"; GREEN = "#14a06e"; BLUE = "#3e68b4"; ROD = "#9dbcf0"; BLOCK = "#f7c98c"


INIT = "Initial configuration"
PUSH_RUN = "/data/user_data/mbronars/packages/pwm_final_paper/results/block_pushing/ours_paper_video_source_pop3_gw2_lookaheads/runs/seed42_ep{ep:02d}/replay_set_pose/sim_trajectory.npz"
def start_cm(e):                                                # block-to-target distance at t = 0 (same scene for every method)
    z = np.load(PUSH_RUN.format(ep=e)); return float(np.linalg.norm(z["block_pos"][0, :2] - z["target_xy"])) * 100
def load(e, m):
    fn = f"push_ep{e:02d}_start.png" if m == INIT else f"push_ep{e:02d}_{m.lower()}_end.png"
    return np.asarray(Image.open(os.path.join(A.src, fn)).convert("RGB"))


crops = {}
for e in EPS:                                                   # content = anything clearly brighter / more saturated than the dark wood
    ims = [load(e, m) for m in [INIT] + METHODS]; H, W = ims[0].shape[:2]
    mask = np.zeros((H, W), bool)
    for im in ims:
        f = im.astype(int); mask |= (f.max(2) > 115) | ((f.max(2) - f.min(2)) > 60)   # wood: max channel < 90, spread < 40
    ys, xs = np.nonzero(mask); pad = A.pad * W
    x0, x1, y0, y1 = xs.min() - pad, xs.max() + pad, ys.min() - pad, ys.max() + pad
    cw, ch = x1 - x0, y1 - y0
    if cw / ch < 4 / 3: cw = ch * 4 / 3
    else: ch = cw * 3 / 4
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    cx = np.clip(cx, cw / 2, W - cw / 2); cy = np.clip(cy, ch / 2, H - ch / 2)
    crops[e] = tuple(int(round(v)) for v in (cx - cw / 2, cy - ch / 2, cx + cw / 2, cy + ch / 2)); print("episode", e, "crop", crops[e])


def caption(e, m, two=False):
    if m == INIT: return f"block starts {start_cm(e):.1f} cm" + ("\n" if two else " ") + "from target"
    r = cases[(e, m)]; return f"block moved {r['block_moved_mm']} mm" + ("\n" if two else ", ") + f"{r['sim_final_cm']:.1f} cm from target"


if A.layout == "column":
    ROWS, COLS = [INIT] + METHODS, EPS; W_IN = 3.4; lab_w = 0.84; gap = 0.04; head_h = 0.22
else:
    ROWS, COLS = EPS, [INIT] + METHODS; W_IN = 7.0; lab_w = 0.70; gap = 0.05; head_h = 0.22
nC = len(COLS); img_w = (W_IN - lab_w - gap * (nC + 1)) / nC; img_h = img_w * 0.75; cap_h = 0.27; leg_h = 0.42 if A.layout == "column" else 0.26
H_IN = head_h + len(ROWS) * (img_h + cap_h) + leg_h
fig = plt.figure(figsize=(W_IN, H_IN))
def box(x, y, w, h): return fig.add_axes([x / W_IN, 1 - (y + h) / H_IN, w / W_IN, h / H_IN])
def name(v): return f"Episode {EPS.index(v) + 1}" if isinstance(v, (int, np.integer)) else ("Initial\nconfiguration" if v == INIT and A.layout == "column" else v)
for ci, c in enumerate(COLS):
    x = lab_w + gap + ci * (img_w + gap)
    fig.text((x + img_w / 2) / W_IN, 1 - 0.11 / H_IN, name(c), ha="center", va="center", fontsize=9, fontweight="bold", color=INK)
for ri, rr in enumerate(ROWS):
    y0 = head_h + ri * (img_h + cap_h)
    fig.text((lab_w - 0.06) / W_IN, 1 - (y0 + img_h / 2) / H_IN, name(rr), ha="right", va="center", ma="right", linespacing=1.1, fontsize=8.5, fontweight="bold", color=INK)
    for ci, c in enumerate(COLS):
        m, e = (rr, c) if A.layout == "column" else (c, rr)
        x = lab_w + gap + ci * (img_w + gap); im = load(e, m); a, b, cc, d = crops[e]
        ax = box(x, y0, img_w, img_h); ax.imshow(im[b:d, a:cc], interpolation="none"); ax.set_axis_off()   # native resolution, no resampling
        fig.text((x + img_w / 2) / W_IN, 1 - (y0 + img_h + 0.035) / H_IN, caption(e, m, True), ha="center", va="top", linespacing=1.15, fontsize=7, color=MUTED)

# legend, wrapped onto a second line when it runs out of width
items = [("ring", GREEN, "target"), ("line", MAG, "block path"), ("line", BLUE, "pusher path"), ("rod", ROD, "pusher")]
r_ = fig.canvas.get_renderer(); rows_ = [[]]; used = 0.0; avail = W_IN - 0.3
for kind, col, label in items:                                   # pass 1: measure and wrap
    t_ = fig.text(0, 0, label, fontsize=7, color=MUTED, va="center"); w_ = 0.2 + t_.get_window_extent(renderer=r_).width / fig.dpi
    if rows_[-1] and used + w_ > avail: rows_.append([]); used = 0.0
    rows_[-1].append((kind, col, t_, w_)); used += w_ + 0.22
for li, row in enumerate(rows_):                                 # pass 2: place each row centred on the page
    tot = sum(w_ for *_, w_ in row) + 0.22 * (len(row) - 1); cx = (W_IN - tot) / 2
    yy = 1 - (H_IN - leg_h + 0.14 + 0.17 * li) / H_IN
    for kind, col, t_, w_ in row:
        t_.set_position(((cx + 0.2) / W_IN, yy))
        if kind == "ring": fig.add_artist(patches.Ellipse(((cx + 0.06) / W_IN, yy), 0.1 / W_IN, 0.1 / H_IN, fill=False, ec=col, lw=1.6, transform=fig.transFigure))
        elif kind == "line": fig.add_artist(plt.Line2D([cx / W_IN, (cx + 0.14) / W_IN], [yy, yy], color=col, lw=1.8, solid_capstyle="round"))
        elif kind == "dot": fig.add_artist(patches.Ellipse(((cx + 0.06) / W_IN, yy), 0.065 / W_IN, 0.065 / H_IN, fc=col, ec="white", lw=0.6, transform=fig.transFigure))
        else: fig.add_artist(patches.FancyBboxPatch(((cx + 0.02) / W_IN, yy - 0.04 / H_IN), 0.08 / W_IN, 0.08 / H_IN, boxstyle="round,pad=0,rounding_size=0.01", fc=col, ec="none", transform=fig.transFigure))
        cx += w_ + 0.22
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out + ".pdf", dpi=600); fig.savefig(out + ".png", dpi=300); print("wrote", out + ".pdf", f"{W_IN:.1f} x {H_IN:.2f} in")
