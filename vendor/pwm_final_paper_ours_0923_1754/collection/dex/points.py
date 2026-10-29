"""Point clouds for one episode of a dex-hand shard, rebuilt from its recorded poses, with a contact sanity check.

Hand: URDF visual-mesh points per body, placed by the recorded link poses. Object: the env's own pieces (a screwdriver
is several convex pieces), placed by the recorded object pose. The piece vertices are already in the object frame, so
the stored piece offsets (objgeom_pos/quat) must not be applied again: doing so floats a screwdriver ~5 cm off the fingers.

    python -m collection.dex.points --shard <out_dir>/<teacher>.npz --env 0 [--png episode.png]
"""
import argparse
import json

import numpy as np

from .geometry import canonical_points, merge_fixed, outer_surface_points, parse_urdf, quat_to_R


def hand_bodies(link_names, urdf, n_points=1600):
    """Canonical points [P_k, 3] for each recorded link, in that link's frame."""
    links, joints = parse_urdf(urdf)
    canon = canonical_points(links, urdf, total=n_points)
    order, members, _, _ = merge_fixed(links, joints)
    pts = {root: np.concatenate([canon[n] @ R.T + t for n, R, t in members[i] if n in canon])
           for i, root in enumerate(order)}
    missing = [n for n in link_names if n not in pts]
    assert not missing, f"recorded links without URDF bodies: {missing}"
    return [pts[n] for n in link_names]


def object_points(z, env, n=512, seed=0):
    """(points [n, 3] in the object frame, piece ids) for one env's object."""
    geoms = sorted(int(k.rsplit("_", 1)[1]) for k in z.files if k.startswith("objgeom_envs_") and env in z[k])
    # outer surface only: faces between the convex pieces of a multi-piece object are inside it
    pts = outer_surface_points([(z[f"objgeom_verts_{i}"], z[f"objgeom_faces_{i}"]) for i in geoms], n,
                               np.random.default_rng(seed))
    return pts, geoms


def episode_points(z, env, hand, obj):
    """[T, N, 3] world points of one episode: the object first, then the hand bodies in recorded link order."""
    T = int(z["episode_length"][env])
    # an NpzFile re-reads the whole array on every z[key], so slice each array once, outside the loop
    lp, lq = z["link_pos"][:T, env].astype(np.float64), z["link_quat"][:T, env].astype(np.float64)
    op, oq = z["obj_pos"][:T, env].astype(np.float64), z["obj_quat"][:T, env].astype(np.float64)
    frames = []
    for t in range(T):
        P = [obj @ quat_to_R(oq[t]).T + op[t]]
        for k, b in enumerate(hand):
            P.append(b @ quat_to_R(lq[t, k]).T + lp[t, k])
        frames.append(np.concatenate(P))
    return np.stack(frames)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shard", required=True)
    ap.add_argument("--env", type=int, default=0)
    ap.add_argument("--urdf", default=None, help="robot URDF; default: the manifest's robot_file")
    ap.add_argument("--png", default=None)
    a = ap.parse_args()
    z = np.load(a.shard)
    urdf = a.urdf or json.load(open(a.shard[:-len(".npz")] + ".json"))["robot_file"]
    hand = hand_bodies([str(x) for x in z["link_names"]], urdf)
    obj, geoms = object_points(z, a.env)
    X = episode_points(z, a.env, hand, obj)
    n_obj = len(obj)
    d = np.linalg.norm(X[:, :n_obj, None, :] - X[:, None, n_obj:, :], axis=-1).min(-1)   # object point -> nearest hand point
    print(f"env {a.env}: {X.shape[0]} frames, {X.shape[1]} points, geoms {geoms}, "
          f"object extent {np.round((obj.max(0) - obj.min(0)) * 1000).tolist()} mm")
    print(f"object-to-hand gap: min {d.min() * 1000:.1f} mm, median over frames of per-frame min "
          f"{np.median(d.min(1)) * 1000:.1f} mm; finite {bool(np.isfinite(X).all())}")
    if a.png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(9, 4.4), dpi=110)
        for c, t in enumerate((0, X.shape[0] - 1)):
            ax = fig.add_subplot(1, 2, c + 1, projection="3d")
            ax.scatter(*X[t, n_obj:].T, s=1.5, c="#8a8f98", depthshade=False, linewidths=0)
            ax.scatter(*X[t, :n_obj].T, s=2.5, c="#d1495b", depthshade=False, linewidths=0)
            m, r = X[t].mean(0), 0.12
            ax.set_xlim(m[0] - r, m[0] + r); ax.set_ylim(m[1] - r, m[1] + r); ax.set_zlim(m[2] - r, m[2] + r)
            ax.set_box_aspect((1, 1, 1)); ax.set_title(f"frame {t}", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        fig.tight_layout(); fig.savefig(a.png, facecolor="white")
        print("wrote", a.png)


if __name__ == "__main__":
    main()
