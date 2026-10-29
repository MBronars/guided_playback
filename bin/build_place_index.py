"""Index of place_success VAL episodes for the Isaac replay of place runs: label -> tar/key/element, asset dir, the rigid
transform canon -> model frame (ICP of a dense sample of the asset's simplified mesh to the 512 frame-0 particles), density,
friction.  python bin/build_place_index.py [--select sweeps/place_val_candidates.json] [--out sweeps/place_val_index.json]
Run in the foundationpose env (trimesh + scipy)."""
import argparse, glob, json, sys
import numpy as np, trimesh
from scipy.spatial import cKDTree
sys.path.insert(0, "/data/user_data/mbronars/packages/pickup_replay_isaac/scripts")
from build_specs import iter_samples, SHIFT
D = "/data/group_data/katefgroup/VLA/3DPCWM/robert_pickup_data_place_success"
LIB = sorted(glob.glob("/data/group_data/katefgroup-ssd/sim_scene_gen/assets/*/*/asset_*"))
ap = argparse.ArgumentParser(); ap.add_argument("--select", default="/data/user_data/mbronars/packages/guided_playback/sweeps/place_val_candidates.json")
ap.add_argument("--out", default="/data/user_data/mbronars/packages/guided_playback/sweeps/place_val_index.json"); ap.add_argument("--n_shards", type=int, default=6); ap.add_argument("--samples", type=int, default=20000)
A = ap.parse_args()
cands = json.load(open(A.select)); want = {i: np.array(c["f0_obj_centroid_model"]) for i, c in enumerate(cands)}


def kabsch(src, dst):
    cs, cd = src.mean(0), dst.mean(0); H = (src - cs).T @ (dst - cd); U, _, Vt = np.linalg.svd(H); d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T; return R, cd - R @ cs


def fit(surf, pts):
    tree = cKDTree(surf); best = (np.inf, None, None)
    for flip in (np.eye(3), np.diag([1, -1, -1]), np.diag([-1, 1, -1])):
        for yaw in np.linspace(0, 2 * np.pi, 24, endpoint=False):
            c, s = np.cos(yaw), np.sin(yaw); R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ flip; t = pts.mean(0) - R @ surf.mean(0)
            for _ in range(40):
                _, j = tree.query((pts - t) @ R); R, t = kabsch(surf[j], pts)
            d, _ = tree.query((pts - t) @ R); res = float(np.sqrt((d ** 2).mean()))
            if res < best[0]: best = (res, R, t)
    return best


keys = {"particle_pos", "obj_ids", "asset_id", "material_properties"}; found = {}; cache = {}
for tar in sorted(glob.glob(D + "/gpu0/*.tar"))[-A.n_shards:]:
    for key, s in iter_samples(tar, keys):
        for k in range(s["particle_pos"].shape[0]):
            oid = s["obj_ids"][k]; pts0 = s["particle_pos"][k][0][oid == 0].astype(np.float64) + SHIFT; c = pts0.mean(0)
            for i, w in want.items():
                if i in found or np.abs(c - w).max() >= 1e-4: continue
                aid = int(s["asset_id"][k]); adir = LIB[aid]; name = adir.split("/")[-1]
                if adir not in cache: np.random.seed(0); cache[adir] = trimesh.load(f"{adir}/{name}_simplified.obj", force="mesh").sample(A.samples)
                res, R, t = fit(cache[adir], pts0); mp = s["material_properties"][k][oid == 0][0]
                found[i] = dict(label=cands[i]["label"], tar=tar.split("/3DPCWM/")[1], key=key, element=k, asset_id=aid, asset_dir=adir, category=adir.split("/")[-2],
                                icp_rms_mm=round(res * 1e3, 3), R=R.tolist(), t=t.tolist(), rho=float(mp[2]), mu=float(mp[3]), f0_obj_centroid_model=c.tolist())
                print(f"{found[i]['label']} {found[i]['category']}/{name} icp rms {res * 1e3:.2f} mm rho {mp[2]:.0f} mu {mp[3]:.2f}", flush=True)
    if len(found) == len(want): break
missing = [cands[i]["label"] for i in want if i not in found]
json.dump([found[i] for i in sorted(found)], open(A.out, "w"), indent=1); print("wrote", A.out, len(found), "rows; missing", missing)
