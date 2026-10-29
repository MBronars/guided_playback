"""Four-panel figure: Initial position | Unconditional sample | Guided | Guided with obstacle.

usage: python bin/four_panel.py --style hull|tan|slate|blue2 [--tight]
  hull  = the original yellow/pink renders (render / render_unguided / render_x / render_x_wall55)
  tan|slate|blue2 = asset-mesh renders with the carton tinted (render_asset_tint_<c>, ..._unguided_tint_<c>, ..._wall55_tint_<c>)
  --tight = crop every panel to the SAME bounding box of non-white content (union over the four) and use a 6 px gap
Writes site_armwall/panel/four_panel[_rendered_<c>][_tight].{png,pdf}. Needs the foundationpose env (imageio + PIL).
"""
import argparse, numpy as np, imageio.v2 as iio
from PIL import Image, ImageDraw, ImageFont

ap = argparse.ArgumentParser(); ap.add_argument("--style", default="hull"); ap.add_argument("--tight", action="store_true"); ap.add_argument("--grid", default="1x4", help="1x4 or 2x2")
ap.add_argument("--label", default="above", help="above = title strip over each panel; inside = title drawn in the panel's empty top-right corner")
ap.add_argument("--gap", type=int, default=None); A = ap.parse_args()
W, H, CROP = 960, 720, 75
N = "runs/pickup/ground30_nowall_milk_carton_asset_0037/replay_isaac_arm/"
Wd = "runs/pickup/wall12w20ground_pickonly_it40_lr0.1_shell_bw10_clr10/replay_isaac_arm/"
end_panel = lambda p: Image.open(p).convert("RGB").crop((W + 10, 0, 2 * W + 10, H - CROP))
first = lambda g: Image.fromarray(np.asarray(iio.mimread(g, memtest=False)[0])[:H - CROP, :, :3])
if A.style == "hull":
    src = [N + "render/env0_ep0.gif", N + "render_unguided/keyframes_trace_env2.png", N + "render_x/keyframes_trace_env0.png", Wd + "render_x_wall55/keyframes_trace_env0.png"]
else:
    c = A.style                                                          # e.g. tan, or tan_greytable
    ct = c.split("_")[0]; suf = c[len(ct):]
    src = [N + f"render_asset_unguided_tint_{ct}{suf}/env2_ep2.gif",   # frame 0 from the no-target render: no X in the initial panel
           N + f"render_asset_unguided_tint_{ct}{suf}/keyframes_trace_env2.png",
           N + f"render_asset_tint_{ct}{suf}/keyframes_trace_env0.png", Wd + f"render_asset_wall55_tint_{ct}{suf}/keyframes_trace_env0.png"]
ims = [first(src[0])] + [end_panel(p) for p in src[1:]]
titles = ["Initial position", "Unconditional sample", "Guided", "Guided with obstacle"]
gap, head, fs = 24, 84, 50
if A.tight:
    boxes = [np.argwhere(np.asarray(im).min(2) < 245) for im in ims]
    y0 = min(b[:, 0].min() for b in boxes); y1 = max(b[:, 0].max() for b in boxes)
    x0 = min(b[:, 1].min() for b in boxes); x1 = max(b[:, 1].max() for b in boxes)
    pad = 0 if A.grid == "2x2" else 6; y0, x0 = max(0, y0 - pad), max(0, x0 - pad); y1, x1 = min(ims[0].height, y1 + pad), min(W, x1 + pad)
    ims = [im.crop((x0, y0, x1, y1)) for im in ims]; gap, head = 6, 64
w, h = ims[0].size
if A.gap is not None: gap = A.gap
rows, cols = (2, 2) if A.grid == "2x2" else (1, 4)
if A.label == "inside": head = 0
if A.grid == "2x2" and A.label == "above": head, fs = 52, 40
font = ImageFont.truetype("/usr/share/fonts/urw-base35/NimbusRoman-Regular.otf", fs)
out = Image.new("RGB", (cols * w + (cols - 1) * gap, rows * (h + head) + (rows - 1) * gap), (255, 255, 255)); dr = ImageDraw.Draw(out)
for i, (t, im) in enumerate(zip(titles, ims)):
    r, c = divmod(i, cols); x = c * (w + gap); y = r * (h + head + gap)
    out.paste(im, (x, y + head)); tw = dr.textlength(t, font=font)
    if A.label == "inside": dr.text((x + 34, y + h - fs - 6), t, fill=(255, 255, 255), font=font)   # bottom-left corner is table in all four panels: white text on the table
    else: dr.text((x + (w - tw) / 2, y + (head - fs) / 2 - 4), t, fill=(20, 20, 20), font=font)
base = "site_armwall/panel/four_panel" + ("" if A.style == "hull" else f"_rendered_{A.style}") + ("_tight" if A.tight else "") + ("_2x2" if A.grid == "2x2" else "") + ("_inlabel" if A.label == "inside" else "")
if A.tight:                                                             # no white margin at all around the figure
    a = np.asarray(out); nz = np.argwhere(a.min(2) < 250); (ty0, tx0), (ty1, tx1) = nz.min(0), nz.max(0)
    out = out.crop((tx0, ty0, tx1 + 1, ty1 + 1))
out.save(base + ".png"); out.save(base + ".pdf", "PDF", resolution=300.0); print("wrote", base, out.size)
