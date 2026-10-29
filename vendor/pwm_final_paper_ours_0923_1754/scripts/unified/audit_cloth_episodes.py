"""Stage 1 of the cloth corpus audit: per-EPISODE assembly/geometry/motion metrics for every
episode in the given unified_v1/cloth roots, read straight from the tars.

Raw layout (materializer output): sample = one episode,
  particle_pos [T,N,3] (canonical, metres)   obj_ids [N] (0=cloth, 1=left EE, 2=right EE)
  valid_particles [N] (prefix mask; padded cloth slots sit at one constant point)
  n_cloth (int)   dt (s)
Sample key = <garment>_<action>, e.g. g20_05_lift_random.

Assembly checks on the cloth particles at frame 0 (and the same at the last frame):
  nearest-neighbour spacing (median s, p99, max), isolated fraction (NN > 4 s), duplicate
  fraction (NN < 0.1 mm), connected components at radius 2.5 s (count of components with
  >= 1% of particles, largest-component fraction, top sizes), bbox extents, z range.
Temporal checks: a fixed k=8 neighbour graph from frame 0; per frame, each particle's
displacement vs the mean displacement of its neighbours -> "identity glitch" fraction
(particles moving unlike their neighbours by > 2 s); edge stretch p99/max of neighbour
distances at 5 frames relative to frame 0; total cloth motion; per-frame max step.
Gripper checks: per EE particle count, rigidity (pairwise distances of 64 sampled particles
frame 0 vs last, max relative change), max step. Also NaN/Inf, valid-prefix property,
n_cloth vs actual cloth count, min z (table penetration), out-of-box extents.
Flags are decided in stage 2 from the distributions; this writes raw numbers only.

    PYTHONPATH=. python scripts/unified/audit_cloth_episodes.py \
        --root train=/data/group_data/katefgroup/VLA/3DPCWM/unified_v1/cloth/train \
        --root val=/data/group_data/katefgroup/VLA/3DPCWM/unified_v1/cloth/val \
        --out reports/unified/data_audit_0903/cloth_episodes.jsonl --workers 8
"""
import argparse, glob, json, os, pickle, tarfile, time
from multiprocessing import Pool
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

ap = argparse.ArgumentParser()
ap.add_argument("--root", action="append", required=True, help="label=path; path holds *.tar")
ap.add_argument("--out", required=True)
ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--limit_tars", type=int, default=0)
a = ap.parse_args()
K = 8


def geom(X):
    """Frame-level assembly metrics on cloth points X [n,3]."""
    tree = cKDTree(X)
    d, _ = tree.query(X, k=2)
    nn = d[:, 1]
    s = float(np.median(nn))
    pairs = tree.query_pairs(2.5 * s, output_type="ndarray")
    n = len(X)
    g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    ncomp, lab = connected_components(g, directed=False)
    sizes = np.sort(np.bincount(lab))[::-1]
    lo, hi = X.min(0), X.max(0)
    # superposed panels: where two panels share one plane the NN distance drops well below
    # the single-layer spacing, so the NN distribution splits; count the close fraction
    p75 = float(np.percentile(nn, 75))
    overlap_frac = float((nn < 0.6 * p75).mean())
    # superposed panels double the in-plane density over PART of the silhouette (a proper
    # two-layer garment doubles it everywhere, a single panel nowhere): fraction of particles
    # whose xy-neighbour count within 2 spacings exceeds 1.5x the median count
    cnt = np.asarray(cKDTree(X[:, :2]).query_ball_point(X[:, :2], r=2 * s, return_length=True))
    dense_frac = float((cnt > 1.5 * np.median(cnt)).mean())
    return {"nn_med": s, "nn_p25": float(np.percentile(nn, 25)), "nn_p75": p75, "overlap_frac": overlap_frac,
            "dense_frac": dense_frac, "z_over_spacing": float((hi[2] - lo[2]) / max(s, 1e-6)),
            "nn_p99": float(np.percentile(nn, 99)), "nn_max": float(nn.max()),
            "isolated_frac": float((nn > 4 * s).mean()), "dup_frac": float((nn < 1e-4).mean()),
            "n_comp": int(ncomp), "n_comp_big": int((sizes >= 0.01 * n).sum()),
            "largest_frac": float(sizes[0] / n), "comp_sizes": [int(v) for v in sizes[:5]],
            "extent": [float(v) for v in (hi - lo)], "zmin": float(lo[2]), "zmax": float(hi[2])}, tree


def scan_tar(job):
    label, tar = job
    rows, t0 = [], time.time()
    try:
        tf = tarfile.open(tar)
        names = tf.getnames()
        keys = sorted(set(n.split(".")[0] for n in names))
        for key in keys:
            ld = lambda s: pickle.loads(tf.extractfile(f"{key}.{s}.pyd").read())
            P = np.asarray(ld("particle_pos"), dtype=np.float32)        # [T,N,3]
            oid = np.asarray(ld("obj_ids")); valid = np.asarray(ld("valid_particles"), dtype=bool)
            n_cloth = int(ld("n_cloth")); dt = float(ld("dt"))
            T, N, _ = P.shape
            parts = key.split("_")
            row = {"root": label, "tar": os.path.basename(tar), "key": key,
                   "garment": "_".join(parts[:2]), "action": "_".join(parts[2:]),
                   "T": int(T), "N": int(N), "dt": dt, "n_cloth_meta": n_cloth,
                   "n_valid": int(valid.sum()), "valid_prefix": bool(valid[:valid.sum()].all() and not valid[valid.sum():].any()),
                   "n_nonfinite": int((~np.isfinite(P[:, valid])).sum()),
                   "ids_valid": {int(i): int(((oid == i) & valid).sum()) for i in np.unique(oid[valid])}}
            ci = np.where((oid == 0) & valid)[0]
            row["n_cloth_actual"] = int(len(ci))
            if len(ci) < 50 or row["n_nonfinite"] > 0:
                rows.append(row); continue
            X0, XL = P[0, ci], P[-1, ci]
            g0, tree0 = geom(X0); gl, _ = geom(XL)
            row["f0"], row["flast"] = g0, gl
            s = g0["nn_med"]
            # fixed neighbour graph from frame 0
            _, nb = tree0.query(X0, k=K + 1); nb = nb[:, 1:]                        # [n,K]
            C = P[:, ci]                                                              # [T,n,3]
            D = np.diff(C, axis=0)                                                    # [T-1,n,3]
            Dn = D[:, nb].mean(axis=2)                                                # [T-1,n,3]
            e = np.linalg.norm(D - Dn, axis=-1)                                       # [T-1,n]
            glitch = (e > 2 * s).mean(axis=1)                                         # per frame
            step = np.linalg.norm(D, axis=-1)
            fr = sorted(set([T // 4, T // 2, 3 * T // 4, T - 1]))
            L0 = np.linalg.norm(C[0][:, None] - C[0][nb], axis=-1)                    # [n,K]
            stretch = []
            for t in fr:
                Lt = np.linalg.norm(C[t][:, None] - C[t][nb], axis=-1)
                stretch.append(Lt / np.maximum(L0, 1e-6))
            stretch = np.stack(stretch)
            # layer-robust stretch: garments lie as two layers ~1 spacing apart, so frame-0
            # neighbours across layers separate when the cloth is lifted (not a stretch).
            # In-layer proxy = pairs that stay within 2 s at every sampled frame; report the
            # max over ALL frames of their length ratio (catches transient stretch too).
            Lall = np.linalg.norm(C[:, :, None] - C[:, nb], axis=-1)                 # [T,n,K]
            inlayer = (Lall[fr] < 2 * s).all(axis=0) & (L0 < 2 * s)                  # [n,K]
            rat = Lall / np.maximum(L0, 1e-6)[None]
            sep_frac = float((stretch.max(axis=0) > 3).mean())
            inl = rat[:, inlayer] if inlayer.any() else rat[:, :0]
            # layer-aware identity check: compare each particle's step with its IN-LAYER
            # neighbours only (particles with >= 3 such neighbours), so a fold sliding one
            # layer over the other does not register as a glitch
            w = inlayer.astype(np.float32)                                            # [n,K]
            cnt = w.sum(1)                                                            # [n]
            Dn_in = (D[:, nb] * w[None, :, :, None]).sum(2) / np.maximum(cnt, 1)[None, :, None]
            e_in = np.linalg.norm(D - Dn_in, axis=-1)[:, cnt >= 3]                    # [T-1,n']
            glitch_in = (e_in > 2 * s).mean(axis=1) if e_in.shape[1] else np.zeros(T - 1)
            net = np.linalg.norm(C[-1] - C[0], axis=-1)
            row.update({
                "glitch_frac_max": float(glitch.max()), "t_glitch_max": int(glitch.argmax()),
                "glitch_frames": int((glitch > 0.005).sum()),
                "glitch_in_frac_max": float(glitch_in.max()), "t_glitch_in_max": int(glitch_in.argmax()),
                "glitch_in_frames": int((glitch_in > 0.005).sum()), "inlayer_particles_frac": float((cnt >= 3).mean()),
                "stretch_p99": float(np.percentile(stretch, 99)), "stretch_max": float(stretch.max()),
                "compress_p01": float(np.percentile(stretch, 1)),
                "sep_frac": sep_frac, "inlayer_frac": float(inlayer.mean()),
                "inlayer_stretch_p99": float(np.percentile(inl, 99)) if inl.size else None,
                "inlayer_stretch_max": float(inl.max()) if inl.size else None,
                "max_step": float(step.max()), "p99_step": float(np.percentile(step, 99)),
                "t_max_step": int(step.max(axis=1).argmax()),
                "mean_net": float(net.mean()), "max_net": float(net.max()),
                "mean_speed": float(step.mean()), "zmin_all": float(C[..., 2].min()), "zmax_all": float(C[..., 2].max()),
                "xy_absmax": float(np.abs(C[..., :2]).max()),
            })
            for eid, nm in ((1, "left"), (2, "right")):
                gi = np.where((oid == eid) & valid)[0]
                if len(gi) < 8:
                    row[f"ee_{nm}"] = {"n": int(len(gi))}; continue
                Gp = P[:, gi]
                rng = np.random.default_rng(0); sub = rng.choice(len(gi), size=min(64, len(gi)), replace=False)
                pd0 = np.linalg.norm(Gp[0][sub][:, None] - Gp[0][sub][None], axis=-1)
                pdl = np.linalg.norm(Gp[-1][sub][:, None] - Gp[-1][sub][None], axis=-1)
                iu = np.triu_indices(len(sub), 1)
                rel = np.abs(pdl[iu] - pd0[iu]) / np.maximum(pd0[iu], 1e-4)
                gs = np.linalg.norm(np.diff(Gp, axis=0), axis=-1)
                row[f"ee_{nm}"] = {"n": int(len(gi)), "rigid_maxrel": float(rel.max()),
                                   "rigid_medrel": float(np.median(rel)),
                                   "extent": [float(v) for v in (Gp[0].max(0) - Gp[0].min(0))],
                                   "extent_last": [float(v) for v in (Gp[-1].max(0) - Gp[-1].min(0))],
                                   "centroid0": [float(v) for v in Gp[0].mean(0)], "centroid_last": [float(v) for v in Gp[-1].mean(0)],
                                   "n_unique0": int(len(np.unique(np.round(Gp[0] * 1e5).astype(np.int64), axis=0))),
                                   "max_step": float(gs.max()), "net": float(np.linalg.norm(Gp[-1].mean(0) - Gp[0].mean(0)))}
            rows.append(row)
        tf.close()
    except Exception as ex:
        rows.append({"root": label, "tar": tar, "error": repr(ex)})
    return rows, time.time() - t0


jobs = []
for spec in a.root:
    label, path = spec.split("=", 1)
    tars = sorted(glob.glob(f"{path}/*.tar"))
    if a.limit_tars: tars = tars[:a.limit_tars]
    jobs += [(label, t) for t in tars]
print(f"[audit-cloth] {len(jobs)} tars over {len(a.root)} roots -> {a.out}", flush=True)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
n_rows, t0 = 0, time.time()
with open(a.out, "w") as f, Pool(a.workers) as pool:
    for i, (rows, dt) in enumerate(pool.imap_unordered(scan_tar, jobs)):
        for r in rows:
            f.write(json.dumps(r) + "\n"); n_rows += 1
        f.flush()
        if i % 5 == 0 or i == len(jobs) - 1:
            print(f"[audit-cloth] {i + 1}/{len(jobs)} tars, {n_rows} rows, {time.time() - t0:.0f}s", flush=True)
print(f"[audit-cloth] done: {n_rows} rows in {time.time() - t0:.0f}s", flush=True)
