"""3 x 4 pile-manipulation figure: rows = method, columns = (episode 36, episode 47) x (model prediction, simulated outcome).
Reads the renders + pile_cases.json of one render folder (default push_pile_v8). Writes PDF + PNG preview.
  .../foundationpose/bin/python bin/make_pile_pdf.py [--src figures/push_pile_v8] [--out figures/paper_pdf/pile_prediction_vs_outcome]"""
import argparse, json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
ap = argparse.ArgumentParser(); ap.add_argument("--src", default=os.path.join(ROOT, "figures", "push_pile_v12"))
ap.add_argument("--out", default=os.path.join(ROOT, "figures", "paper_pdf", "pile_prediction_vs_outcome")); ap.add_argument("--crop_top", type=float, default=0.06)
A = ap.parse_args()
from matplotlib import font_manager as _fm
for _p in __import__("glob").glob(os.path.expanduser("~/.local/share/fonts/Times*.TTF")): _fm.fontManager.addfont(_p)   # Microsoft core-fonts Times New Roman (times32.exe from corefonts)
_serif = ["Times New Roman"]
plt.rcParams.update({"font.family": "serif", "font.serif": _serif, "mathtext.fontset": "custom", "mathtext.rm": "Times New Roman", "mathtext.it": "Times New Roman:italic", "mathtext.bf": "Times New Roman:bold", "font.size": 7.5, "pdf.fonttype": 42})
cases = {(r["ep"], r["method"]): r for r in json.load(open(os.path.join(A.src, "pile_cases.json"))) if "method" in r}
EPS = sorted({e for e, _ in cases}); METHODS = ["Ours", "PointWorld", "GNS"]
COLS = [(e, k) for e in EPS for k in ("pred", "sim")]
W_IN = 7.0; lab_w = 0.80; gap = 0.04; img_w = (W_IN - lab_w - gap * 5) / 4; img_h = img_w * 0.75 * (1 - A.crop_top); cap_h = 0.16; head_h = 0.42
H_IN = head_h + 3 * (img_h + cap_h) + 0.30
fig = plt.figure(figsize=(W_IN, H_IN)); INK = "#1c2321"; MUTED = "#5d6864"; MAG = "#d6288a"; GREEN = "#14a06e"
def box(x, y, w, h): return fig.add_axes([x / W_IN, 1 - (y + h) / H_IN, w / W_IN, h / H_IN])
for ci, (e, k) in enumerate(COLS):
    x = lab_w + gap + ci * (img_w + gap)
    if k == "pred":                                                                # episode header spanning its two columns
        fig.text((x + img_w + gap / 2) / W_IN, 1 - 0.13 / H_IN, f"Episode {EPS.index(e) + 1}", ha="center", va="center", fontsize=9, fontweight="bold", color=INK)
        fig.add_artist(plt.Line2D([(x + 0.05) / W_IN, (x + 2 * img_w + gap - 0.05) / W_IN], [1 - 0.23 / H_IN] * 2, color="#c9d0cd", lw=0.8))
    fig.text((x + img_w / 2) / W_IN, 1 - 0.34 / H_IN, "Model prediction" if k == "pred" else "Simulated outcome", ha="center", va="center", fontsize=8, color=INK, style="italic" if k == "pred" else "normal")
for ri, m in enumerate(METHODS):
    y0 = head_h + ri * (img_h + cap_h)
    fig.text((lab_w - 0.06) / W_IN, 1 - (y0 + img_h / 2) / H_IN, m, ha="right", va="center", fontsize=8.5, fontweight="bold", color=INK)
    for ci, (e, k) in enumerate(COLS):
        x = lab_w + gap + ci * (img_w + gap); r = cases[(e, m)]
        im = Image.open(os.path.join(A.src, f"pile_ep{e:03d}_{m.lower()}_{k}.png")).convert("RGB"); w, h = im.size
        im = np.asarray(im.crop((0, int(h * A.crop_top), w, h)))
        ax = box(x, y0, img_w, img_h); ax.imshow(im, interpolation="none"); ax.set_axis_off()                  # embed at native resolution, no resampling
        mv = r["pred_moved_mm"] if k == "pred" else r["sim_moved_mm"]
        txt = f"predicted movement {mv} mm" if k == "pred" else f"actual movement {mv} mm"
        fig.text((x + img_w / 2) / W_IN, 1 - (y0 + img_h + 0.075) / H_IN, txt, ha="center", va="center", fontsize=7, color=MUTED)
# legend
ly = H_IN - 0.16; lx = lab_w + gap
items = [("ring", GREEN, "target"), ("arrow", MAG, "pile center, start to end"), ("dash", "#8f9a96", "starting footprint"), ("box", "#7ba1dc", "paddle")]
cx = lx
for kind, col, label in items:
    yy = 1 - ly / H_IN
    if kind == "ring": fig.add_artist(patches.Ellipse(((cx + 0.06) / W_IN, yy), 0.11 / W_IN, 0.05 / H_IN, fill=False, ec=col, lw=1.6, transform=fig.transFigure))
    elif kind == "arrow": fig.add_artist(patches.FancyArrow((cx) / W_IN, yy, 0.13 / W_IN, 0, width=0.026 / H_IN, head_width=0.085 / H_IN, head_length=0.06 / W_IN, color=col, transform=fig.transFigure, length_includes_head=True))
    elif kind == "dash": fig.add_artist(plt.Line2D([cx / W_IN, (cx + 0.14) / W_IN], [yy, yy], color=col, lw=1.2, ls=(0, (3, 2))))
    else: fig.add_artist(patches.Rectangle((cx / W_IN, yy - 0.035 / H_IN), 0.1 / W_IN, 0.07 / H_IN, color=col, transform=fig.transFigure))
    t_ = fig.text((cx + 0.2) / W_IN, yy, label, va="center", fontsize=7, color=MUTED)
    cx += 0.2 + t_.get_window_extent(renderer=fig.canvas.get_renderer()).width / fig.dpi + 0.32   # measured text width: even gaps whatever the font
fig.savefig(A.out + ".pdf", dpi=600); fig.savefig(A.out + ".png", dpi=300); print("wrote", A.out + ".pdf", f"{W_IN:.1f} x {H_IN:.2f} in")
