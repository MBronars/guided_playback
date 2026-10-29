"""Place runs (held start, robert_pickup_data_place_success): the object's own CoACD convex decomposition posed at frame 0,
instead of one convex hull of its particles (a hull fills a cup or bowl, so a rim grasp cannot be held).

Index (bin/build_place_index.py): one row per select label with the episode's tar/key/element, asset dir, the rigid
transform canon -> model frame fitted by ICP to the frame-0 particles, and the recorded density / friction.
Only runs whose result.json has task == "place" go through this path; pick / lift runs are untouched."""
import glob, json, os
import numpy as np

INDEX = "/data/user_data/mbronars/packages/guided_playback/sweeps/place_val_index.json"


def _obj_vertices(path):
    """the 'v x y z' lines of a Wavefront OBJ (no trimesh in the pwm211 env)"""
    return np.array([[float(x) for x in ln.split()[1:4]] for ln in open(path) if ln.startswith("v ")], dtype=np.float64)


def load_pieces(asset_dir):
    name = os.path.basename(asset_dir)
    files = sorted(glob.glob(os.path.join(asset_dir, f"{name}_coacd", "coacd_convex_piece_*.obj")), key=lambda p: int(p.rsplit("_", 1)[1][:-4]))
    return [_obj_vertices(f) for f in files]


def place_hulls(z, index_path=INDEX, tol_m=0.02):
    """[B, Vmax, 3] hull vertices in the model frame and [B, Hmax + 1] start offsets (-1 padded), plus per-member rho / mu."""
    idx = {r["label"]: r for r in json.load(open(index_path))}
    B = z["predicted"].shape[0]; V, O, rho, mu, rows = [], [], [], [], []
    for m in range(B):
        lab = str(z["episodes"][m]); r = idx[lab]
        R, t = np.array(r["R"]), np.array(r["t"])
        pieces = [p @ R.T + t for p in load_pieces(r["asset_dir"])]
        w = np.concatenate(pieces); off = np.cumsum([0] + [len(p) for p in pieces])
        p0 = z["predicted"][m, 0, z["object_mask"][m]]
        err = max(np.abs(w.min(0) - p0.min(0)).max(), np.abs(w.max(0) - p0.max(0)).max())
        assert err < tol_m, f"member {m} ({lab}): CoACD hulls off the frame-0 particles by {err * 1e3:.1f} mm"
        V.append(w); O.append(off); rho.append(float(r["rho"])); mu.append(float(r["mu"]))
        rows.append(dict(member=m, label=lab, asset=r["asset_dir"].split("/assets/")[-1], pieces=len(pieces), bbox_err_mm=round(err * 1e3, 1), icp_rms_mm=r["icp_rms_mm"]))
    Vmax, Hmax = max(len(v) for v in V), max(len(o) for o in O)
    hv = np.zeros((B, Vmax, 3)); ho = np.full((B, Hmax), -1, np.int64)
    for m in range(B): hv[m, :len(V[m])] = V[m]; ho[m, :len(O[m])] = O[m]
    return hv, ho, np.array(rho), np.array(mu), rows
