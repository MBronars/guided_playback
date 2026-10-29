"""The particles themselves, as small spheres in the same tank scene as render_water/render_rollout (same camera,
light, glass and bounds), for the page's water-surface / particles toggle. Raw box units, y up.
    window dump: python scripts/render_water3d/render_points.py --npz dump_w0.npz --out points_w0   -> gt_NNN.png, pred_NNN.png
    trajectory:  python scripts/render_water3d/render_points.py --traj trajectory.npz --frames 0:801:16 --out gt_points
Sphere radius = --radius_frac x the mean nearest-neighbour spacing of the first frame, fixed over the sequence."""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_water import BOUNDS, particle_spacing, scene_dict, write_ply


def icosphere():
    t = (1 + 5 ** 0.5) / 2
    v = np.array([[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
                  [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]], float)
    f = [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
         [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]]
    v = list(v / np.linalg.norm(v, axis=1, keepdims=True)); cache, out = {}, []
    def mid(i, j):
        k = (min(i, j), max(i, j))
        if k not in cache:
            m = (v[i] + v[j]) / 2; v.append(m / np.linalg.norm(m)); cache[k] = len(v) - 1
        return cache[k]
    for a, b, c in f:
        ab, bc, ca = mid(a, b), mid(b, c), mid(c, a); out += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
    return np.array(v), np.array(out)


UNIT, FACES = icosphere()


def cloud_ply(path, pts, r):
    n = len(pts)
    verts = (pts[:, None, :] + r * UNIT[None]).reshape(-1, 3)
    faces = (FACES[None] + len(UNIT) * np.arange(n)[:, None, None]).reshape(-1, 3)
    write_ply(path, verts.astype(np.float32), faces.astype(np.int32), np.tile(UNIT, (n, 1)).astype(np.float32))


def render(mi, pts, r, ply, png, res, spp):
    cloud_ply(ply, pts, r)
    sc = scene_dict(ply, BOUNDS, res, spp)
    sc["water"]["bsdf"] = {"type": "plastic", "diffuse_reflectance": {"type": "rgb", "value": [0.10, 0.36, 0.70]}, "int_ior": 1.5}
    mi.util.write_bitmap(png, mi.render(mi.load_dict(sc), spp=spp))
    os.remove(ply)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz"); ap.add_argument("--traj"); ap.add_argument("--frames", default="0:801:16")
    ap.add_argument("--out", required=True); ap.add_argument("--spp", type=int, default=96); ap.add_argument("--res", type=int, default=448)
    ap.add_argument("--radius_frac", type=float, default=0.35); ap.add_argument("--variant", default="cuda_ad_rgb")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    import mitsuba as mi; mi.set_variant(a.variant)
    if a.npz:
        d = np.load(a.npz)
        seq = {"gt": np.concatenate([d["context"], d["gt"]]), "pred": np.concatenate([d["context"], d["pred"]])}
        r = a.radius_frac * particle_spacing(seq["gt"][0].astype(np.float64))
        for t in range(seq["gt"].shape[0]):
            for k in ("gt", "pred"):
                render(mi, seq[k][t].astype(np.float64), r, os.path.join(a.out, f"{k}_{t:03d}.ply"), os.path.join(a.out, f"{k}_{t:03d}.png"), a.res, a.spp)
            print("frame", t, flush=True)
    else:
        pos = np.load(a.traj)["positions"]; s = slice(*[int(x) for x in a.frames.split(":")])
        idx = list(range(pos.shape[0]))[s]; r = a.radius_frac * particle_spacing(pos[idx[0]].astype(np.float64))
        for i, t in enumerate(idx):
            render(mi, pos[t].astype(np.float64), r, os.path.join(a.out, f"frame_{i:03d}.ply"), os.path.join(a.out, f"frame_{i:03d}.png"), a.res, a.spp)
            print("frame", t, flush=True)
    json.dump({"args": vars(a), "radius": float(r)}, open(os.path.join(a.out, "receipt.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
