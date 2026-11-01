"""Planning-task overview: columns = Pushing / Pile Manipulation / Picking, rows = initial configuration with its target and the
executed LNO plan (simulator outcome). Reads the render folders of one figures/planning_nov1-style directory (push/, pile/, pick/
written by render_push_pile.py and render_pick_views.py) and their cases json. Times New Roman, native-resolution images.
  .../foundationpose/bin/python bin/make_task_pdf.py --dir figures/planning_nov1 --push_ep 14 --pile_ep 40 --pick lno_water_bottle_t3:2:water_bottle"""
import argparse, glob, json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
ap = argparse.ArgumentParser(); ap.add_argument("--dir", default=os.path.join(ROOT, "figures", "planning_nov1"))
ap.add_argument("--push_ep", type=int, default=14); ap.add_argument("--pile_ep", type=int, default=40)
ap.add_argument("--pick", default="lno_water_bottle_t3:2:water_bottle", help="ALIAS:MEMBER:OBJECT of the pick render to show")
ap.add_argument("--pick_start_mm", type=float, default=None, help="pick start distance (default: from pick_config.json)")
ap.add_argument("--out", default=None); A = ap.parse_args(); D = A.dir; out = A.out or os.path.join(D, "pdf", "planning_tasks")
for p in glob.glob(os.path.expanduser("~/.local/share/fonts/Times*.TTF")): fm.fontManager.addfont(p)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"], "font.size": 7.5, "pdf.fonttype": 42})
INK = "#1c2321"; MUTED = "#5d6864"


def push_crop(ims, pad=0.05):
    """make_push_pdf's crop: union of the content brighter / more saturated than the wood, padded to 4:3."""
    H, W = ims[0].shape[:2]; mask = np.zeros((H, W), bool)
    for im in ims:
        f = im.astype(int); mask |= (f.max(2) > 115) | ((f.max(2) - f.min(2)) > 60)
    ys, xs = np.nonzero(mask); p = pad * W; x0, x1, y0, y1 = xs.min() - p, xs.max() + p, ys.min() - p, ys.max() + p
    cw, ch = x1 - x0, y1 - y0
    if cw / ch < 4 / 3: cw = ch * 4 / 3
    else: ch = cw * 3 / 4
    cx = np.clip((x0 + x1) / 2, cw / 2, W - cw / 2); cy = np.clip((y0 + y1) / 2, ch / 2, H - ch / 2)
    a, b, c, d = (int(round(v)) for v in (cx - cw / 2, cy - ch / 2, cx + cw / 2, cy + ch / 2)); return [im[b:d, a:c] for im in ims]


def rgb(path): return np.asarray(Image.open(path).convert("RGB"))
def top_crop(im, f=0.06): return im[int(im.shape[0] * f):]


# pushing
pc = json.load(open(os.path.join(D, "push", "push_cases.json")))
ps = next(r for r in pc if r.get("ep") == A.push_ep and "start_cm" in r); po = next(r for r in pc if r.get("ep") == A.push_ep and r.get("method") == "Ours")
push = push_crop([rgb(os.path.join(D, "push", f"push_ep{A.push_ep:02d}_start.png")), rgb(os.path.join(D, "push", f"push_ep{A.push_ep:02d}_ours_end.png"))])
# pile
lc = json.load(open(os.path.join(D, "pile", "pile_cases.json"))); lo = next(r for r in lc if r.get("ep") == A.pile_ep and r.get("method") == "Ours")
SC = "/data/user_data/mbronars/packages/pwm_final_paper/results/moulding/scenarios/seed_42/ep_{:03d}.npz"
pile_start = float(np.linalg.norm(np.load(SC.format(A.pile_ep))["target_offset_xy"])) * 1e3
pile = [top_crop(rgb(os.path.join(D, "pile", f"pile_ep{A.pile_ep:03d}_start.png"))), top_crop(rgb(os.path.join(D, "pile", f"pile_ep{A.pile_ep:03d}_ours_sim.png")))]
# picking
alias, m, obj = A.pick.split(":"); kc = json.load(open(os.path.join(D, "pick", "cases.json")))
ko = next(r for r in kc if r["run"] == alias and r["member"] == int(m) and r["object"] == obj)
pick_start = A.pick_start_mm or next(r[2] for r in json.load(open(os.path.join(D, "pick_config.json")))["rows"] if r[1] == obj)
pick = [rgb(os.path.join(D, "pick", f"{obj}__{alias}__m{m}__az-60_el18_{t}.png")) for t in ("start", "end")]

COLS = [("Pushing", "60 steps, ManiSkill", push, f"block starts {ps['start_cm']:.1f} cm from target", f"block ends {po['sim_final_cm']:.1f} cm from target"),
        ("Pile Manipulation", "150 steps, Genesis", pile, f"pile center starts {pile_start / 10:.1f} cm from target", f"pile center ends {lo['sim_to_target_mm'] / 10:.1f} cm from target"),
        ("Picking", "60 steps, Isaac Lab", pick, f"object starts {pick_start / 10:.1f} cm from target", f"object ends {ko['sim_to_target_mm'] / 10:.1f} cm from target")]
ROWS = ["Initial\nconfiguration", "Executed\nLNO plan"]
W_IN = 7.0; lab_w = 0.78; gap = 0.06; head_h = 0.36; cap_h = 0.20
img_w = (W_IN - lab_w - gap * 4) / 3; img_h = img_w * 0.75; H_IN = head_h + 2 * (img_h + cap_h) + 0.05
fig = plt.figure(figsize=(W_IN, H_IN))
def box(x, y, w, h): return fig.add_axes([x / W_IN, 1 - (y + h) / H_IN, w / W_IN, h / H_IN])
for ci, (name, sub, ims, c0, c1) in enumerate(COLS):
    x = lab_w + gap + ci * (img_w + gap)
    fig.text((x + img_w / 2) / W_IN, 1 - 0.13 / H_IN, name, ha="center", va="center", fontsize=9, fontweight="bold", color=INK)
    fig.text((x + img_w / 2) / W_IN, 1 - 0.27 / H_IN, sub, ha="center", va="center", fontsize=7, color=MUTED)
    for ri, (im, cap) in enumerate(zip(ims, (c0, c1))):
        y0 = head_h + ri * (img_h + cap_h); h, w = im.shape[:2]
        if w / h > 4 / 3: d = int((w - h * 4 / 3) / 2); im = im[:, d:w - d]                       # centre-crop to 4:3
        elif w / h < 4 / 3: d = int((h - w * 3 / 4) / 2); im = im[d:h - d]
        ax = box(x, y0, img_w, img_h); ax.imshow(im, interpolation="none"); ax.set_axis_off()
        fig.text((x + img_w / 2) / W_IN, 1 - (y0 + img_h + 0.035) / H_IN, cap, ha="center", va="top", fontsize=7, color=MUTED)
for ri, r in enumerate(ROWS):
    y0 = head_h + ri * (img_h + cap_h)
    fig.text((lab_w - 0.06) / W_IN, 1 - (y0 + img_h / 2) / H_IN, r, ha="right", va="center", ma="right", linespacing=1.1, fontsize=8.5, fontweight="bold", color=INK)
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out + ".pdf", dpi=600); fig.savefig(out + ".png", dpi=300); print("wrote", out + ".pdf", f"{W_IN:.1f} x {H_IN:.2f} in")
