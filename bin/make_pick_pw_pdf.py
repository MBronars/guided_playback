"""PointWorld pick prediction over time: rows = object (milk carton t0, soda can t1), columns = frames; object-centred renders
from render_pick_pw_evolution.py. Times New Roman, native-resolution images.
  .../foundationpose/bin/python bin/make_pick_pw_pdf.py [--src figures/pick_pw_evolution_v3]"""
import argparse, glob, json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
ap = argparse.ArgumentParser(); ap.add_argument("--src", default=os.path.join(ROOT, "figures", "pick_pw_scene_v2")); ap.add_argument("--frames", default="0,15,30,45,60")
ap.add_argument("--out", default=os.path.join(ROOT, "figures", "paper_pdf", "pick_pointworld_prediction")); ap.add_argument("--dt", type=float, default=0.05); A = ap.parse_args()
for p in glob.glob(os.path.expanduser("~/.local/share/fonts/Times*.TTF")): fm.fontManager.addfont(p)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"], "font.size": 7.5, "pdf.fonttype": 42})
INK = "#1c2321"; MUTED = "#5d6864"; NAME = {"milk_carton": "Milk\ncarton", "soda_can": "Soda\ncan"}
FR = [int(v) for v in A.frames.split(",")]
cases = json.load(open(os.path.join(A.src, "cases.json")))
if "frames" not in cases[0]:                                                      # in-scene renders (render_pick_views.py --pw_cloud): sizes from the prediction itself
    z = np.load(os.path.join(ROOT, "runs", "pickup", "pw_benchmark", "rollout.npz"), allow_pickle=True)
    def rms(Y): return float(np.sqrt(((Y - Y.mean(0)) ** 2).sum(1).mean()))
    for r in cases:
        o = z["predicted"][r["member"]][:, z["roles"][r["member"]] == 2]; r["obj_scale"] = [rms(o[f]) / rms(o[0]) for f in range(len(o))]
        r["images"] = [f"{r['object']}__{r['run']}__pwcloud__m{r['member']}__az-60_el18_f{f:02d}.png" for f in FR]
    cases = sorted(cases, key=lambda r: ["milk_carton", "soda_can"].index(r["object"]))
W_IN = 7.0; lab_w = 0.62; gap = 0.05; head_h = 0.22; cap_h = 0.20; n = len(FR)
img_w = (W_IN - lab_w - gap * (n + 1)) / n; img_h = img_w * 0.75; H_IN = head_h + len(cases) * (img_h + cap_h) + 0.28
fig = plt.figure(figsize=(W_IN, H_IN))
def box(x, y, w, h): return fig.add_axes([x / W_IN, 1 - (y + h) / H_IN, w / W_IN, h / H_IN])
for ci, f in enumerate(FR):
    x = lab_w + gap + ci * (img_w + gap)
    fig.text((x + img_w / 2) / W_IN, 1 - 0.11 / H_IN, "Observed, t = 0 s" if f == 0 else f"Predicted, t = {f * A.dt:.2f} s", ha="center", va="center", fontsize=8.5, fontweight="bold", color=INK)
for ri, r in enumerate(cases):
    y0 = head_h + ri * (img_h + cap_h)
    fig.text((lab_w - 0.06) / W_IN, 1 - (y0 + img_h / 2) / H_IN, NAME[r["object"]], ha="right", va="center", ma="right", linespacing=1.1, fontsize=8.5, fontweight="bold", color=INK)
    for ci, (f, fn) in enumerate(zip(FR, r["images"])):
        x = lab_w + gap + ci * (img_w + gap)
        ax = box(x, y0, img_w, img_h); ax.imshow(np.asarray(Image.open(os.path.join(A.src, fn)).convert("RGB")), interpolation="none"); ax.set_axis_off()
        cap = "observed" if f == 0 else f"size {r['obj_scale'][f]:.2f}×"
        fig.text((x + img_w / 2) / W_IN, 1 - (y0 + img_h + 0.035) / H_IN, cap, ha="center", va="top", fontsize=7, color=MUTED)
os.makedirs(os.path.dirname(A.out), exist_ok=True)
from matplotlib import patches
LEG = [("#5c6e96", "predicted gripper"), ("#e2803c", "predicted object"), ("#1faa78", "target")]
r_ = fig.canvas.get_renderer(); parts = []
for col, label in LEG:
    t_ = fig.text(0, 0, label, fontsize=7, color=MUTED, va="center"); parts.append((col, t_, 0.17 + t_.get_window_extent(renderer=r_).width / fig.dpi))
tot = sum(p[2] for p in parts) + 0.28 * (len(parts) - 1); cx = (W_IN - tot) / 2; yy = 1 - (H_IN - 0.13) / H_IN
for col, t_, w_ in parts:
    t_.set_position(((cx + 0.17) / W_IN, yy)); fig.add_artist(patches.Ellipse(((cx + 0.06) / W_IN, yy), 0.075 / W_IN, 0.075 / H_IN, fc=col, ec="white", lw=0.6, transform=fig.transFigure)); cx += w_ + 0.28
fig.savefig(A.out + ".pdf", dpi=600); fig.savefig(A.out + ".png", dpi=300); print("wrote", A.out + ".pdf", f"{W_IN:.1f} x {H_IN:.2f} in")
