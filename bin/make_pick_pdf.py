"""Pick figure: rows = object (milk carton t0, soda can t1), columns = initial configuration, unguided baseline, PointWorld, ours.
Isaac replays rendered by render_pick_views.py --shared_cam (one camera per object). Times New Roman, native-resolution images.
  .../foundationpose/bin/python bin/make_pick_pdf.py [--src figures/pick_panel_v2] [--out figures/paper_pdf/pick_baseline_pointworld_ours]"""
import argparse, glob, json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches, font_manager as fm
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
ap = argparse.ArgumentParser(); ap.add_argument("--src", default=os.path.join(ROOT, "figures", "pick_panel_v2"))
ap.add_argument("--out", default=os.path.join(ROOT, "figures", "paper_pdf", "pick_baseline_pointworld_ours"))
ap.add_argument("--crop_top", type=float, default=0.0); A = ap.parse_args()
for p in glob.glob(os.path.expanduser("~/.local/share/fonts/Times*.TTF")): fm.fontManager.addfont(p)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"], "font.size": 7.5, "pdf.fonttype": 42})
INK = "#1c2321"; MUTED = "#5d6864"; MAG = "#d6288a"; GREEN = "#1faa78"
ROWS = [("Milk carton", "milk_carton", 192.1, {"Ours Unguided": ("mppiargmin_full19_t0_milk_carton_K20", 0), "PointWorld": ("pw_benchmark", 0), "Ours": ("cyl_t0_milk_carton_it10", 0)}),
        ("Soda can", "soda_can", 153.0, {"Ours Unguided": ("mppiargmin_full19_t1_soda_can_K20", 0), "PointWorld": ("pw_benchmark", 8), "Ours": ("cyl_t1_soda_can_it10", 0)})]
COLS = ["Initial configuration", "Ours Unguided", "PointWorld", "Ours"]
cs = json.load(open(os.path.join(A.src, "cases.json")))
def rec(run, m, obj): return next(c for c in cs if c["run"] == run and c["member"] == m and c["object"] == obj)
def img(obj, run, m, tag):
    im = Image.open(os.path.join(A.src, f"{obj}__{run}__m{m}__az-60_el18_{tag}.png")).convert("RGB"); w, h = im.size
    return np.asarray(im.crop((0, int(h * A.crop_top), w, h)))
W_IN = 7.0; lab_w = 0.62; gap = 0.05; head_h = 0.22; cap_h = 0.30; leg_h = 0.26
img_w = (W_IN - lab_w - gap * 5) / 4; img_h = img_w * 0.75 * (1 - A.crop_top); H_IN = head_h + 2 * (img_h + cap_h) + leg_h
fig = plt.figure(figsize=(W_IN, H_IN))
def box(x, y, w, h): return fig.add_axes([x / W_IN, 1 - (y + h) / H_IN, w / W_IN, h / H_IN])
for ci, c in enumerate(COLS):
    x = lab_w + gap + ci * (img_w + gap)
    fig.text((x + img_w / 2) / W_IN, 1 - 0.11 / H_IN, c, ha="center", va="center", fontsize=9, fontweight="bold", color=INK)
for ri, (name, obj, start_mm, runs) in enumerate(ROWS):
    y0 = head_h + ri * (img_h + cap_h)
    fig.text((lab_w - 0.06) / W_IN, 1 - (y0 + img_h / 2) / H_IN, name.replace(" ", "\n"), ha="right", va="center", ma="right", linespacing=1.1, fontsize=8.5, fontweight="bold", color=INK)
    for ci, c in enumerate(COLS):
        x = lab_w + gap + ci * (img_w + gap)
        if c == COLS[0]:
            run, m = runs["Ours Unguided"]; im = img(obj, run, m, "start"); cap = f"starts {start_mm / 10:.1f} cm\nfrom target"
        else:
            run, m = runs[c]; r = rec(run, m, obj); im = img(obj, run, m, "end")
            cap = f"lifted {r['lift_peak_mm'] / 10:.1f} cm\nends {r['sim_to_target_mm'] / 10:.1f} cm from target"
        ax = box(x, y0, img_w, img_h); ax.imshow(im, interpolation="none"); ax.set_axis_off()
        fig.text((x + img_w / 2) / W_IN, 1 - (y0 + img_h + 0.035) / H_IN, cap, ha="center", va="top", linespacing=1.15, fontsize=7, color=MUTED)
# centred legend
items = [("dot", GREEN, "target"), ("line", MAG, "object path")]
r_ = fig.canvas.get_renderer(); parts = []
for kind, col, label in items:
    t_ = fig.text(0, 0, label, fontsize=7, color=MUTED, va="center"); parts.append((kind, col, t_, 0.2 + t_.get_window_extent(renderer=r_).width / fig.dpi))
tot = sum(p[3] for p in parts) + 0.3 * (len(parts) - 1); cx = (W_IN - tot) / 2; yy = 1 - (H_IN - leg_h + 0.13) / H_IN
for kind, col, t_, w_ in parts:
    t_.set_position(((cx + 0.2) / W_IN, yy))
    if kind == "dot": fig.add_artist(patches.Ellipse(((cx + 0.07) / W_IN, yy), 0.075 / W_IN, 0.075 / H_IN, fc=col, ec="white", lw=0.6, transform=fig.transFigure))
    else: fig.add_artist(plt.Line2D([cx / W_IN, (cx + 0.14) / W_IN], [yy, yy], color=col, lw=1.6, solid_capstyle="round"))
    cx += w_ + 0.3
os.makedirs(os.path.dirname(A.out), exist_ok=True)
fig.savefig(A.out + ".pdf", dpi=600); fig.savefig(A.out + ".png", dpi=300); print("wrote", A.out + ".pdf", f"{W_IN:.1f} x {H_IN:.2f} in")
