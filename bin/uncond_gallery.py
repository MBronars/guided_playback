"""Gallery of unconditional (unguided) arm replays: one tile per draw = the render frame at the CUT frame with the smoothed carton
path, where cut = the last frame up to the carton's peak height at which every robot link origin projects inside the image
(top/side margin), so the robot is never cut off. Labels: seed/member, carton height at the cut, approach angle at grasp.
  python bin/uncond_gallery.py --out site_armwall/gallery --eye ... --look ..."""
import argparse, glob, json, os, numpy as np, imageio.v2 as iio
from PIL import Image, ImageDraw, ImageFont
from scipy.signal import savgol_filter
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--eye", required=True); ap.add_argument("--look", required=True)
ap.add_argument("--size", default="960x720"); ap.add_argument("--margin", type=int, default=14); ap.add_argument("--font", default="/usr/share/fonts/urw-base35/NimbusRoman-Regular.otf")
a = ap.parse_args(); W, H = [int(v) for v in a.size.split("x")]; os.makedirs(a.out, exist_ok=True)
eye = np.array([float(v) for v in a.eye.split(",")]); look = np.array([float(v) for v in a.look.split(",")])
f = look - eye; f /= np.linalg.norm(f); s = np.cross(f, [0, 0, 1]); s /= np.linalg.norm(s); u = np.cross(s, f)
Vi = np.eye(4); Vi[:3, 0], Vi[:3, 1], Vi[:3, 2], Vi[:3, 3] = s, u, -f, eye; V = np.linalg.inv(Vi); fy = 1 / np.tan(np.pi / 4.5 / 2); fx = fy / (W / H)
def px(p):
    c = V @ np.append(p, 1.0); return ((c[0] / -c[2]) * fx * 0.5 + 0.5) * W, (1 - ((c[1] / -c[2]) * fy * 0.5 + 0.5)) * H
RUNS = [("seed 0", "runs/pickup/ground30_nowall_milk_carton_asset_0037")] + [(f"seed {i}", f"runs/pickup/uncond_milk_carton_seed{i}") for i in range(1, 6)]
tiles, rows = [], []
for label, run in RUNS:
    d = os.path.join(run, "replay_isaac_arm"); R = np.load(os.path.join(d, "isaac_unguided", "replay.npz")); z = np.load(os.path.join(run, "rollout.npz"))
    for k in range(len(R["env_episode"])):
        m = int(R["env_episode"][k]); op = R["obj_pos"][k]; T = len(op); peak = int(op[:, 2].argmax())
        # robot fully in frame: every link origin (plus a 12 cm radius for link bulk at the top) inside the image
        ok = []
        for t in range(T):
            P = R["link_pos"][k, t]; xs, ys = zip(*[px(p) for p in P]); top = min(ys) - 0.12 * fy * 0.5 * H / max(1e-6, float((V @ np.append(P[np.argmin(ys)], 1))[2] * -1))
            ok.append(min(xs) > a.margin and max(xs) < W - a.margin and top > a.margin)
        cand = [t for t in range(peak + 1) if ok[t]]; cut = cand[-1] if cand else 0
        # approach angle at grasp from the generated rollout (same rule as the ranking)
        Pg = z["unguided"][m]; roles = z["roles"][m]; pb = z["particle_body"][m]; fing = roles == 1; hand = roles == 0
        fb = sorted(set(pb[fing].tolist())); f1 = Pg[:, pb == fb[0]].mean(1); f2 = Pg[:, pb == fb[1]].mean(1); h = Pg[:, hand].mean(1); sep = np.linalg.norm(f1 - f2, axis=1)
        cl = np.nonzero(sep < sep[0] - 0.02)[0]; fc = int(cl[0]) if len(cl) else 0; ap_ = (f1 + f2) / 2 - h; ap_ = ap_[fc] / np.linalg.norm(ap_[fc]); ang = float(np.degrees(np.arccos(np.clip(-ap_[2], -1, 1))))
        gif = os.path.join(d, "render_unguided", f"env{k}_ep{m}.gif"); fr = iio.mimread(gif, memtest=False)
        sm = savgol_filter(op[:cut + 1], min(11, (cut + 1) - ((cut + 1) + 1) % 2), 3, axis=0) if cut + 1 >= 11 else op[:cut + 1]; pts = [px(p) for p in sm]
        im = Image.fromarray(np.asarray(fr[cut])[:, :, :3]).convert("RGBA"); ov = Image.new("RGBA", im.size, (0, 0, 0, 0)); dr = ImageDraw.Draw(ov)
        for i in range(len(pts) - 1): dr.line([pts[i], pts[i + 1]], fill=(200, 40, 120, int(90 + 165 * i / max(1, len(pts) - 2))), width=6, joint="curve")
        for (x, y) in (pts[0], pts[-1]): dr.ellipse([x - 8, y - 8, x + 8, y + 8], fill=(200, 40, 120, 255), outline=(255, 255, 255, 255), width=2)
        im = Image.alpha_composite(im, ov).convert("RGB"); name = f"{label.replace(' ', '')}_m{m}"; im.save(os.path.join(a.out, name + ".png"))
        lift_cut = (op[cut, 2] - op[0, 2]) * 1e3; lift_peak = (op[peak, 2] - op[0, 2]) * 1e3
        rows.append(dict(id=name, seed=label, member=m, cut_frame=cut, peak_frame=peak, lift_at_cut_mm=round(lift_cut), peak_lift_mm=round(lift_peak), approach_deg=round(ang), sideways_mm=round(float(np.linalg.norm(op[cut, :2] - op[0, :2]) * 1e3)),
                         hand_err_max_mm=round(float(np.linalg.norm(R["hand_pos"][k] - R["target_pos"][k], axis=1).max() * 1e3), 1)))
        tiles.append(((f"{label} · m{m}", f"cut f{cut} · lift {lift_cut:.0f} mm (peak {lift_peak:.0f}) · approach {ang:.0f}°"), im))
        print(rows[-1], flush=True)
json.dump(rows, open(os.path.join(a.out, "gallery.json"), "w"), indent=1)
font = ImageFont.truetype(a.font, 30); font2 = ImageFont.truetype(a.font, 22); cols, gap, head = 4, 20, 68; tw, th = W // 2, H // 2; n = len(tiles); nrows = (n + cols - 1) // cols
sheet = Image.new("RGB", (cols * tw + (cols - 1) * gap, nrows * (th + head) + (nrows - 1) * gap), (255, 255, 255)); dr = ImageDraw.Draw(sheet)
for i, (t, im) in enumerate(tiles):
    r, c = divmod(i, cols); x, y = c * (tw + gap), r * (th + head + gap); dr.text((x + 8, y + 6), t[0], fill=(20, 20, 20), font=font); dr.text((x + 8, y + 40), t[1], fill=(70, 70, 70), font=font2); sheet.paste(im.resize((tw, th), Image.LANCZOS), (x, y + head))
sheet.save(os.path.join(a.out, "gallery.png")); print("gallery", sheet.size)
