"""Hand and object geometry for dex-hand shards: URDF meshes, surface sampling and fixed-link folding.

Genesis welds fixed URDF links into their parents, so each recorded link is one body: the root, or a revolute child
with its fixed descendants folded in. Canonical points live in each body's own frame; the recorded poses place them.
"""
import os
import struct
import xml.etree.ElementTree as ET

import numpy as np


def rpy_to_R(rpy):
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp,     cp * sr,                cp * cr]])


def quat_to_R(q):
    """Rotation matrices [..., 3, 3] from wxyz quaternions [..., 4] (Genesis order), not renormalized."""
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                     2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                     2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
                    -1).reshape(q.shape[:-1] + (3, 3))


def load_stl(path):
    """Triangles [n, 3, 3] from a binary or ASCII STL (xhand1 ships binary, sharpa ASCII)."""
    b = open(path, "rb").read()
    if len(b) >= 84:
        n = struct.unpack("<I", b[80:84])[0]
        if len(b) == 84 + 50 * n:
            rec = np.frombuffer(b[84:], dtype=np.uint8).reshape(n, 50)
            return np.ascontiguousarray(rec[:, 12:48]).view("<f4").reshape(n, 3, 3).astype(np.float64)
    v = np.array([line.split()[1:4] for line in b.decode("ascii", errors="ignore").splitlines()
                  if line.strip().startswith("vertex")], dtype=np.float64)
    assert len(v) and len(v) % 3 == 0, f"unreadable STL: {path}"
    return v.reshape(-1, 3, 3)


def sample_surface(tri, n, rng):
    """n area-weighted surface points on triangles [m, 3, 3]; returns (points [n, 3], total area)."""
    v0, v1, v2 = tri[:, 0], tri[:, 1], tri[:, 2]
    area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    idx = rng.choice(len(tri), size=n, p=area / area.sum())
    u, v = rng.random((n, 1)), rng.random((n, 1))
    flip = (u + v) > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    return v0[idx] + u * (v1[idx] - v0[idx]) + v * (v2[idx] - v0[idx]), area.sum()


def outer_surface_points(pieces, n, rng, step=5e-4):
    """n area-weighted points [n, 3] on the outer surface of an object made of convex pieces [(verts, faces), ...].

    One piece: exactly sample_surface. Several: a sampled point is dropped when a step of `step` metres outward along
    its face normal lands strictly inside another piece, i.e. it lies on a face shared with (or buried in) a
    neighbour; sampling repeats until n points remain.
    """
    tris = [np.asarray(v, np.float64)[f] for v, f in pieces]
    if len(tris) == 1:
        return sample_surface(tris[0], n, rng)[0]
    tri = np.concatenate(tris)
    owner = np.concatenate([np.full(len(t), i) for i, t in enumerate(tris)])
    v0, v1, v2 = tri[:, 0], tri[:, 1], tri[:, 2]
    normal = np.cross(v1 - v0, v2 - v0)
    area = 0.5 * np.linalg.norm(normal, axis=1)
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-18)
    centre = np.stack([np.asarray(v, np.float64).mean(0) for v, _ in pieces])
    normal *= np.where(((tri.mean(1) - centre[owner]) * normal).sum(1, keepdims=True) < 0, -1.0, 1.0)   # outward
    # a convex piece is the intersection of its faces' half-spaces (inside: normal . x < normal . v0); a flat piece
    # (vertices within 10 um of a plane) encloses nothing
    planes = []
    for i, (v, _) in enumerate(pieces):
        flat = np.linalg.svd(np.asarray(v, np.float64) - centre[i], compute_uv=False)[-1] < 1e-5 * np.sqrt(len(v))
        m = owner == i
        planes.append(None if flat else np.concatenate([normal[m], -(normal[m] * v0[m]).sum(1, keepdims=True)], 1))
    kept = []
    while sum(len(k) for k in kept) < n:
        idx = rng.choice(len(tri), size=2 * n, p=area / area.sum())
        u, v = rng.random((2 * n, 1)), rng.random((2 * n, 1))
        flip = (u + v) > 1
        u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
        p = v0[idx] + u * (v1[idx] - v0[idx]) + v * (v2[idx] - v0[idx])
        q = p + step * normal[idx]
        buried = np.zeros(len(p), bool)
        for j, eq in enumerate(planes):
            if eq is not None:
                other = owner[idx] != j
                buried[other] |= (q[other] @ eq[:, :3].T + eq[:, 3]).max(1) < 0
        kept.append(p[~buried])
    return np.concatenate(kept)[:n]


def parse_urdf(path):
    """(links, joints): every visual mesh of each link (a finger link can carry its tip too) and every joint."""
    root = ET.parse(path).getroot()
    links, joints = {}, []
    for l in root.findall("link"):
        visuals = []
        for v in l.findall("visual"):
            mesh = v.find("geometry/mesh")
            if mesh is None:
                continue
            o = v.find("origin")
            visuals.append({
                "mesh": mesh.get("filename"),
                "scale": np.array([float(x) for x in mesh.get("scale", "1 1 1").split()]),
                "origin_xyz": np.array([float(x) for x in o.get("xyz", "0 0 0").split()]) if o is not None else np.zeros(3),
                "origin_rpy": np.array([float(x) for x in o.get("rpy", "0 0 0").split()]) if o is not None else np.zeros(3),
            })
        links[l.get("name")] = {"visuals": visuals}
    for j in root.findall("joint"):
        o, a, lim = j.find("origin"), j.find("axis"), j.find("limit")
        joints.append({
            "name": j.get("name"), "type": j.get("type"),
            "parent": j.find("parent").get("link"), "child": j.find("child").get("link"),
            "xyz": np.array([float(x) for x in o.get("xyz", "0 0 0").split()]),
            "rpy": np.array([float(x) for x in o.get("rpy", "0 0 0").split()]),
            "axis": np.array([float(x) for x in a.get("xyz").split()]) if a is not None else np.array([0., 0., 1.]),
            "limit": (float(lim.get("lower")), float(lim.get("upper"))) if lim is not None and lim.get("lower") else None,
        })
    return links, joints


def canonical_points(links, urdf, total=2400, seed=0):
    """Surface points per link in its own frame, allocated by mesh area (at least 24 per link)."""
    rng = np.random.default_rng(seed)
    base = os.path.dirname(urdf)
    tri_sets, areas = {}, {}
    for n, l in links.items():
        if not l["visuals"]:
            continue
        tris = []
        for v in l["visuals"]:                          # each mesh into its link frame first
            t = load_stl(os.path.join(base, v["mesh"])) * v["scale"]
            R_v, t_v = rpy_to_R(v["origin_rpy"]), v["origin_xyz"]
            tris.append(t @ R_v.T + t_v)
        t = np.concatenate(tris)
        tri_sets[n] = t
        areas[n] = float(0.5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1).sum())
    tot = sum(areas.values())
    return {n: sample_surface(t, max(24, int(round(total * areas[n] / tot))), rng)[0] for n, t in tri_sets.items()}


def merge_fixed(links, joints):
    """Bodies as Genesis exposes them: the root and each revolute child, with fixed descendants folded in.

    Returns (order, members, parent_body, joint_of): order[i] is body i's link name; members[i] lists (link, R, t), the
    constant transform of each folded link into body i's frame; parent_body[i] is the parent body (-1 for the root);
    joint_of[i] is the revolute joint that moves body i (None for the root).
    """
    by_child = {j["child"]: j for j in joints}
    root = next(n for n in links if n not in by_child)

    def owner(link):                       # the nearest ancestor reached only through fixed joints
        while link != root and by_child[link]["type"] == "fixed":
            link = by_child[link]["parent"]
        return link

    order = [root] + [j["child"] for j in joints if j["type"] == "revolute"]
    body_of = {n: order.index(owner(n)) for n in links}
    members = {i: [] for i in range(len(order))}
    for n in links:
        R, t = np.eye(3), np.zeros(3)      # transform of link n into its body's frame
        cur = n
        while cur != order[body_of[n]]:
            j = by_child[cur]
            Rj, tj = rpy_to_R(j["rpy"]), j["xyz"]
            R, t = Rj @ R, Rj @ t + tj
            cur = j["parent"]
        members[body_of[n]].append((n, R, t))
    parent_body = [-1] + [body_of[by_child[c]["parent"]] for c in order[1:]]
    joint_of = [None] + [by_child[c] for c in order[1:]]
    return order, members, parent_body, joint_of
