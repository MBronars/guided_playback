"""Start | end image of a commanded-arm replay render, the end frame overlaid with the SMOOTHED path of the object centroid
(from isaac/replay.npz, projected with the render's camera). Smoothing: Savitzky-Golay (window 11 frames, order 3) on x,y,z.
  python bin/arm_keyframes.py <replay_isaac_arm dir> <env index> --eye ... --look ... [--size 960x720]"""
import argparse, json, os, numpy as np, imageio.v2 as iio
from PIL import Image, ImageDraw
from scipy.signal import savgol_filter
ap = argparse.ArgumentParser(); ap.add_argument("arm_dir"); ap.add_argument("env", type=int); ap.add_argument("--eye", required=True); ap.add_argument("--look", required=True)
ap.add_argument("--size", default="960x720"); ap.add_argument("--window", type=int, default=11); ap.add_argument("--out", default=None)
ap.add_argument("--render_dir", default="render"); ap.add_argument("--sim_dir", default="isaac"); ap.add_argument("--end", type=int, default=None, help="last frame to use (inclusive); default: the last frame")
a = ap.parse_args(); W, H = [int(v) for v in a.size.split("x")]
R = np.load(os.path.join(a.arm_dir, a.sim_dir, "replay.npz")); k = a.env; ep = int(R["env_episode"][k])
gif = os.path.join(a.arm_dir, a.render_dir, f"env{k}_ep{ep}.gif"); fr = iio.mimread(gif, memtest=False); fr = fr[:(a.end + 1) if a.end is not None else len(fr)]; T = len(fr)
eye = np.array([float(v) for v in a.eye.split(",")]); look = np.array([float(v) for v in a.look.split(",")])
# same camera as render_arm_replay.py: look_at + PerspectiveCamera(yfov=pi/4.5, aspect W/H)
f = look - eye; f /= np.linalg.norm(f); s = np.cross(f, [0, 0, 1]); s /= np.linalg.norm(s); u = np.cross(s, f)
Vi = np.eye(4); Vi[:3, 0], Vi[:3, 1], Vi[:3, 2], Vi[:3, 3] = s, u, -f, eye; V = np.linalg.inv(Vi)
fy = 1 / np.tan(np.pi / 4.5 / 2); fx = fy / (W / H)
def px(p):
    c = V @ np.append(p, 1.0); return ((c[0] / -c[2]) * fx * 0.5 + 0.5) * W, (1 - ((c[1] / -c[2]) * fy * 0.5 + 0.5)) * H
op = R["obj_pos"][k][:T]; sm = savgol_filter(op, min(a.window, T - (T + 1) % 2), 3, axis=0) if T >= a.window else op
pts = [px(p) for p in sm]
end = Image.fromarray(np.asarray(fr[-1])[:, :, :3]).convert("RGBA"); ov = Image.new("RGBA", end.size, (0, 0, 0, 0)); dr = ImageDraw.Draw(ov)
for i in range(len(pts) - 1):                                                   # fade from start (light) to end (solid)
    al = int(90 + 165 * i / (len(pts) - 2)); dr.line([pts[i], pts[i + 1]], fill=(200, 40, 120, al), width=6, joint="curve")
for (x, y), col in ((pts[0], (200, 40, 120, 255)), (pts[-1], (200, 40, 120, 255))): dr.ellipse([x - 8, y - 8, x + 8, y + 8], fill=col, outline=(255, 255, 255, 255), width=2)
end = Image.alpha_composite(end, ov).convert("RGB")
start = Image.fromarray(np.asarray(fr[0])[:, :, :3])
out = Image.new("RGB", (2 * W + 10, H), (255, 255, 255)); out.paste(start, (0, 0)); out.paste(end, (W + 10, 0))
dst = a.out or os.path.join(a.arm_dir, a.render_dir, f"keyframes_trace_env{k}.png"); out.save(dst); print("wrote", dst, "| path length %.0f mm, peak height %.0f mm" % (np.linalg.norm(np.diff(sm, axis=0), axis=1).sum() * 1e3, (sm[:, 2].max() - sm[0, 2]) * 1e3))
