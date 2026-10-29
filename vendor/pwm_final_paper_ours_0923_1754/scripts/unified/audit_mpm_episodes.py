"""Stage 1 of the mpm corpus audit: per-OBJECT geometry/motion metrics for every episode
in the given webdataset roots, read straight from the tars (no adapter, no windowing),
so whole-episode failures are visible: an object contracting to a point, exploding,
sinking through the floor, never moving, particles turning invalid mid-episode, NaNs.

Raw layout (robert_mpm_v2): sample = a batch of B=4 episodes,
  particle_pos [B,T=49,N,3]  valid_particles [B,T,N]  object_ids [B,N]  n_objects [B]
  material_properties [B,N,5] = (E, nu, rho, col3, col4)   particle_vel [B,T,N,3]
  drag_active [B,T]  is_grabbed [B,N]  initial_vel [B,N,3]
N = 1024 x max objects of the group; validity marks padding (frame-0 mask is what the
unified adapter uses: valid_lengths = valid[:, 0].sum).

Per object we record (metres, frames): particle count, radius of gyration Rg(t) at
t = 0 / min / max / last with the argmin frame, bbox diagonal, unique-position fraction at
the Rg-argmin frame (a true "one point" has 1 unique position), min z over time, max
per-particle per-frame step, centroid path length and net displacement, per-frame validity
count range, NaN/Inf count, the 5 material columns, initial velocity, grab/drag flags, and
the min centroid gap to the other objects. Flags are NOT decided here (stage 2 sets
thresholds from the distributions); only raw numbers are written, one JSON line per object.

    PYTHONPATH=. python scripts/unified/audit_mpm_episodes.py \
        --root train=/data/group_data/katefgroup/VLA/3DPCWM/robert_mpm_v2/webdataset_full49 \
        --root test_seen=/data/group_data/katefgroup/VLA/3DPCWM/robert_mpm_v2_test_seen/webdataset_full49 \
        --out reports/unified/data_audit_0903/mpm_objects.jsonl --workers 16
"""
import argparse, glob, json, os, pickle, tarfile, time
from multiprocessing import Pool
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--root", action="append", required=True, help="label=path; path holds <k>obj/*.tar")
ap.add_argument("--out", required=True)
ap.add_argument("--workers", type=int, default=16)
ap.add_argument("--limit_tars", type=int, default=0, help="debug: only the first N tars per root")
a = ap.parse_args()


def _load(tf, name):
    return pickle.loads(tf.extractfile(name).read())


def scan_tar(job):
    label, tar = job
    rows, t0 = [], time.time()
    try:
        tf = tarfile.open(tar)
        names = tf.getnames()
        keys = sorted(set(n.split(".")[0] for n in names))
        for key in keys:
            P = np.asarray(_load(tf, f"{key}.particle_pos.pyd"), dtype=np.float32)        # [B,T,N,3]
            V = np.asarray(_load(tf, f"{key}.valid_particles.pyd"), dtype=bool)           # [B,T,N]
            O = np.asarray(_load(tf, f"{key}.object_ids.pyd"))                              # [B,N]
            NO = np.asarray(_load(tf, f"{key}.n_objects.pyd")).reshape(-1)                  # [B]
            M = np.asarray(_load(tf, f"{key}.material_properties.pyd"), dtype=np.float32)  # [B,N,5]
            VEL = np.asarray(_load(tf, f"{key}.particle_vel.pyd"), dtype=np.float32) if f"{key}.particle_vel.pyd" in names else None
            DRAG = np.asarray(_load(tf, f"{key}.drag_active.pyd")) if f"{key}.drag_active.pyd" in names else None
            GRAB = np.asarray(_load(tf, f"{key}.is_grabbed.pyd")) if f"{key}.is_grabbed.pyd" in names else None
            IV = np.asarray(_load(tf, f"{key}.initial_vel.pyd"), dtype=np.float32) if f"{key}.initial_vel.pyd" in names else None
            B, T, N, _ = P.shape
            for b in range(B):
                n_obj = int(NO[b])
                objs = {}
                for o in range(n_obj):
                    m0 = (O[b] == o) & V[b, 0]
                    n = int(m0.sum())
                    row = {"root": label, "tar": os.path.relpath(tar, os.path.dirname(os.path.dirname(tar))),
                           "key": key, "b": b, "obj": o, "n_objects": n_obj, "N": int(N), "T": int(T), "n": n}
                    if n == 0:
                        row.update({"empty": True}); rows.append(row); continue
                    X = P[b][:, m0]                                                         # [T,n,3]
                    finite = np.isfinite(X).all(axis=(1, 2))                                # [T]
                    row["n_nonfinite"] = int((~np.isfinite(X)).sum())
                    if not finite.any():
                        row.update({"all_nonfinite": True}); rows.append(row); continue
                    Xf = np.where(np.isfinite(X), X, np.nan)
                    c = np.nanmean(Xf, axis=1)                                              # [T,3]
                    rg = np.sqrt(np.nanmean(((Xf - c[:, None]) ** 2).sum(-1), axis=1))     # [T]
                    lo, hi = np.nanmin(Xf, axis=1), np.nanmax(Xf, axis=1)
                    diag = np.linalg.norm(hi - lo, axis=-1)                                 # [T]
                    rg_ok = np.where(finite, rg, np.nan)
                    t_min = int(np.nanargmin(rg_ok)); t_max = int(np.nanargmax(rg_ok))
                    # unique positions (0.01 mm grid) at the most contracted frame and the last frame
                    def uniq_frac(t):
                        q = np.round(X[t] * 1e5).astype(np.int64)
                        return float(len(np.unique(q, axis=0)) / n)
                    step = np.linalg.norm(np.diff(Xf, axis=0), axis=-1)                      # [T-1,n]
                    cpath = np.nansum(np.linalg.norm(np.diff(c, axis=0), axis=-1))
                    vcount = (V[b] & (O[b] == o)[None]).sum(1)                              # [T]
                    row.update({
                        "rg0": float(rg[0]), "rg_min": float(np.nanmin(rg_ok)), "rg_max": float(np.nanmax(rg_ok)),
                        "rg_last": float(rg[-1]), "t_rg_min": t_min, "t_rg_max": t_max,
                        "diag0": float(diag[0]), "diag_min": float(np.nanmin(np.where(finite, diag, np.nan))),
                        "uniq_frac_tmin": uniq_frac(t_min), "uniq_frac_last": uniq_frac(T - 1),
                        "minz": float(np.nanmin(Xf[..., 2])), "minz0": float(np.nanmin(X[0, :, 2])),
                        "maxz": float(np.nanmax(Xf[..., 2])),
                        "min_xy": [float(v) for v in np.nanmin(Xf[..., :2], axis=(0, 1))],
                        "max_xy": [float(v) for v in np.nanmax(Xf[..., :2], axis=(0, 1))],
                        "max_step": float(np.nanmax(step)), "p99_step": float(np.nanpercentile(step, 99)),
                        "t_max_step": int(np.nanargmax(np.nanmax(step, axis=1))),
                        "centroid_path": float(cpath), "centroid_net": float(np.linalg.norm(c[-1] - c[0])),
                        "centroid0": [float(v) for v in c[0]], "centroid_last": [float(v) for v in c[-1]],
                        "vcount_min": int(vcount.min()), "vcount_max": int(vcount.max()),
                        "valid_constant": bool((V[b, :1] == V[b]).all()),
                        "mat": [float(v) for v in M[b][m0][0]],
                        "mat_uniform": bool((M[b][m0] == M[b][m0][0]).all()),
                        "init_vel": [float(v) for v in (IV[b][m0].mean(0) if IV is not None else np.zeros(3))],
                        "sim_vmax": float(np.nanmax(np.linalg.norm(VEL[b][:, m0], axis=-1))) if VEL is not None else None,
                        "grabbed": bool(GRAB[b][m0].any()) if GRAB is not None else None,
                        "drag_any": bool(DRAG[b].any()) if DRAG is not None else None,
                    })
                    objs[o] = (c, rg)
                    rows.append(row)
                # inter-object centroid gaps (min over time of |ci-cj| - (rgi+rgj))
                if len(objs) > 1:
                    for o, (c, rg) in objs.items():
                        gaps = []
                        for o2, (c2, rg2) in objs.items():
                            if o2 == o: continue
                            g = np.linalg.norm(c - c2, axis=-1) - (rg + rg2)
                            gaps.append(float(np.nanmin(g)))
                        for r in rows:
                            if r["key"] == key and r["b"] == b and r["obj"] == o and "rg0" in r:
                                r["min_pair_gap"] = min(gaps)
                # padded (invalid at frame 0) particles: where do they sit?
                pad = ~V[b, 0]
                if pad.any():
                    q = np.round(P[b, 0, pad] * 1e5).astype(np.int64)
                    rows.append({"root": label, "tar": rows[-1]["tar"], "key": key, "b": b, "obj": -1,
                                 "pad_count": int(pad.sum()), "pad_unique_pos": int(len(np.unique(q, axis=0))),
                                 "pad_finite": bool(np.isfinite(P[b, :, pad]).all())})
        tf.close()
    except Exception as e:                                   # never lose the whole pass to one shard
        rows.append({"root": label, "tar": tar, "error": repr(e)})
    return rows, time.time() - t0


jobs = []
for spec in a.root:
    label, path = spec.split("=", 1)
    tars = sorted(glob.glob(f"{path}/*obj/*.tar"))
    if a.limit_tars: tars = tars[:a.limit_tars]
    jobs += [(label, t) for t in tars]
print(f"[audit-mpm] {len(jobs)} tars over {len(a.root)} roots -> {a.out}", flush=True)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
n_rows, t0 = 0, time.time()
with open(a.out, "w") as f, Pool(a.workers) as pool:
    for i, (rows, dt) in enumerate(pool.imap_unordered(scan_tar, jobs)):
        for r in rows:
            f.write(json.dumps(r) + "\n"); n_rows += 1
        f.flush()
        if i % 10 == 0 or i == len(jobs) - 1:
            print(f"[audit-mpm] {i + 1}/{len(jobs)} tars, {n_rows} rows, {time.time() - t0:.0f}s", flush=True)
print(f"[audit-mpm] done: {n_rows} rows in {time.time() - t0:.0f}s", flush=True)
