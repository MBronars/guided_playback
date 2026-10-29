"""Planning box (wall_thickness x, wall_width y, wall_height z) of an obstacle asset under a layout, from its simplified.obj:
  python bin/obstacle_box.py <asset dir> [--rpy r,p,y deg] [--layout single|stack:N|row:N]"""
import argparse, os, numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("asset"); ap.add_argument("--rpy", default="0,0,0"); ap.add_argument("--layout", default="single"); a = ap.parse_args()
base = os.path.basename(a.asset.rstrip("/")); V = np.array([[float(t) for t in l.split()[1:4]] for l in open(os.path.join(a.asset, base + "_simplified.obj")) if l.startswith("v ")])
r, p, y = np.radians([float(v) for v in a.rpy.split(",")])
cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
R = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]]) @ np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]]) @ np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
ext = np.ptp(V @ R.T, axis=0); parts = a.layout.split(":"); n = int(parts[1]) if len(parts) > 1 else 1; n2 = int(parts[2]) if len(parts) > 2 else 1
box = ext.copy()
if a.layout.startswith("stack"): box[2] *= n
elif a.layout.startswith("rowstack"): box[1] *= n; box[2] *= n2
elif a.layout.startswith("row"): box[1] *= n
print(f"{base} rpy {a.rpy} layout {a.layout}: unit {ext.round(4).tolist()} m -> planning box thickness(x) {box[0]:.4f} width(y) {box[1]:.4f} height(z) {box[2]:.4f}")
