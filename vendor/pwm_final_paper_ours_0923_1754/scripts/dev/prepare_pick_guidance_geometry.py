"""Match a pinned pickup fixture to source collision geometry, offline.

Requires trimesh and scipy for preparation only. Runtime guidance uses the saved
convex planes. Object components are verified convex and watertight; gripper
meshes are explicitly convex-hull proxies. No simulator collision equivalence is
claimed from URDF files alone.
"""
import argparse
import glob
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import sys

import numpy as np
import torch
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from utils.reproducibility import file_sha256


def align(source, target):
    x, y = np.asarray(source, dtype=np.float64), np.asarray(target, dtype=np.float64)
    u, _, vt = np.linalg.svd((x - x.mean(0)).T @ (y - y.mean(0)))
    R = vt.T @ u.T
    if np.linalg.det(R) < 0:
        vt[-1] *= -1
        R = vt.T @ u.T
    t = y.mean(0) - x.mean(0) @ R.T
    error = float(np.linalg.norm(x @ R.T + t - y, axis=1).max())
    if error > 2e-5:
        raise ValueError(f"Source/template alignment failed: {error} metres")
    return R, t, error


def origin(element):
    node = element.find('origin')
    if node is None:
        return np.eye(3), np.zeros(3)
    R = Rotation.from_euler('xyz', [float(v) for v in node.get('rpy', '0 0 0').split()]).as_matrix()
    t = np.array([float(v) for v in node.get('xyz', '0 0 0').split()])
    return R, t


def supporting_plane_violation(vertices, faces):
    """Largest outward-face violation in metres, with oriented triangle faces."""
    tri = vertices[faces]
    normals = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
    lengths = np.linalg.norm(normals, axis=-1)
    usable = lengths > 1e-16
    if not usable.any():
        raise ValueError('No nondegenerate collision faces')
    normals = normals[usable]/lengths[usable, None]
    offsets = -(normals*tri[usable, 0]).sum(-1)
    return float((vertices @ normals.T+offsets).max())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True)
    p.add_argument('--episode', required=True)
    p.add_argument('--source-geometry', required=True)
    p.add_argument('--asset-root', required=True)
    p.add_argument('--gripper-urdf', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--asset-path',help='Explicit recovered asset directory; exact surface/template alignment still required')
    p.add_argument('--object-hull-proxy',action='store_true',help='Explicitly use a convex hull of EACH object collision component, preserving the decomposition; report as proxy')
    p.add_argument('--trimesh-site', help='Optional installed mesh-reader package directory, after numpy/torch are imported')
    args = p.parse_args()
    if args.trimesh_site:
        sys.path.append(args.trimesh_site)
    import trimesh
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    batch = torch.load(args.input, map_location='cpu', weights_only=False)
    if batch.batch_size != 1 or batch.body_type.shape[1] != 4:
        raise ValueError('Expected one pinned pickup scene with four bodies')
    episode = json.loads(Path(args.episode).read_text())
    raw = np.load(args.source_geometry)
    paths = sorted(path for path in glob.glob(str(Path(args.asset_root) / '*/*/*/')) if Path(path).is_dir())
    asset = Path(args.asset_path or paths[episode['asset_id']]).resolve()
    visual = asset / f'{asset.name}_visual.ply'
    mesh = trimesh.load(visual, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    points, _ = trimesh.sample.sample_surface(mesh, 512, seed=int(np.random.default_rng(0).integers(0, 2**31 - 1)))
    points = points.astype(np.float32)
    mappings, records, bodies = {}, [], {}
    counts = batch.body_template_mask[0].sum(1).tolist()
    mappings[0] = align(points, batch.body_templates[0, 0, :counts[0]].numpy())
    urdf = Path(args.gripper_urdf).resolve()
    robot = ET.parse(urdf).getroot()
    body_links = {1: ('panda_hand', 8), 2: ('panda_leftfinger', 9), 3: ('panda_rightfinger', 10)}
    inputs = {str(Path(path).resolve()): file_sha256(path) for path in
              (args.input, args.episode, args.source_geometry, str(visual), str(urdf))}
    for body, (name, row) in body_links.items():
        n = int(raw['link_local_npts'][row])
        mappings[body] = align(raw['link_local_pts'][row, :n], batch.body_templates[0, body, :n].numpy())
    object_urdf = asset / f'{asset.name}_coacd/coacd.urdf'
    inputs[str(object_urdf)] = file_sha256(object_urdf)
    for body in range(4):
        if body == 0:
            link = ET.parse(object_urdf).getroot().find('link')
        else:
            link = robot.find(f"link[@name='{body_links[body][0]}']")
        planes, verts = [], []
        R, t, error = mappings[body]
        for element in link.findall('collision'):
            spec = element.find('geometry/mesh')
            name = spec.get('filename')
            if body == 0:
                path = object_urdf.parent / Path(name).name
            else:
                path = urdf.parent.parent / name.removeprefix('package://')
            path = path.resolve()
            m = trimesh.load_mesh(path, process=True)
            scale = np.array([float(v) for v in spec.get('scale', '1 1 1').split()])
            Rg, tg = origin(element)
            vertices = ((m.vertices * scale) @ Rg.T + tg) @ R.T + t
            hull = ConvexHull(vertices)
            violation = supporting_plane_violation(vertices, m.faces) if body == 0 else None
            expected_volume = abs(float(m.volume)*float(np.prod(scale)))
            volume_relative_error = abs(hull.volume-expected_volume)/max(hull.volume,1e-20)
            if body == 0 and not args.object_hull_proxy and not (m.is_watertight and m.is_winding_consistent and
                                  violation <= 2e-6 and volume_relative_error <= 1e-5):
                raise ValueError(f'Object collision component is not closed and convex within 2 micrometres: '
                                 f'{path}; plane violation={violation}, relative volume error={volume_relative_error}')
            _, idx = np.unique(np.round(hull.equations, 7), axis=0, return_index=True)
            planes.append(torch.tensor(hull.equations[np.sort(idx)], dtype=torch.float32))
            verts.append(torch.tensor(vertices, dtype=torch.float32))
            inputs[str(path)] = file_sha256(path)
            records.append(dict(body=body, path=str(path), watertight=bool(m.is_watertight),
                                convex=bool(m.is_convex), template_max_error_m=error,
                                hull_volume=float(hull.volume), source_volume=float(m.volume),
                                supporting_plane_violation_m=violation,volume_relative_error=volume_relative_error))
        bodies[body] = dict(planes=planes, vertices=torch.cat(verts),
                            kind=('per-component object convex hull proxies' if args.object_hull_proxy else 'convex collision components') if body == 0 else 'gripper collision convex hull proxy')
    # A support reference inferred from observed, initially resting object geometry.
    # The shard did not save its table pose; do not call this a measured table plane.
    v = bodies[0]['vertices']
    C = batch.context_len
    R0, t0 = batch.body_poses[0, :C, 0, :3, :3], batch.body_poses[0, :C, 0, :3, 3]
    bottom = (torch.einsum('fij,nj->fni', R0, v) + t0[:, None])[..., 2].amin(-1)
    support_z = float(bottom.median())
    report = dict(source_episode=episode, asset=str(asset), input_hashes=inputs,
                  object_hull_proxy=args.object_hull_proxy,
                  object_convexity_tolerance_m=2e-6,
                  components=records, support_z=support_z, observed_object_bottom_z=bottom.tolist(),
                  support_kind='inferred from minimum collision-mesh height in observed context; table pose not recorded',
                  penetration_kind='sampled points inside convex collision components; reverse direction uses gripper hull proxies',
                  limitations=['No continuous collision detection or mesh-intersection completeness.',
                               'Source URDF collision geometry is not independently verified against the original USD cooking.',
                               'Zero detected depth does not certify collision-free motion.'])
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(bodies=bodies, metadata=report), out)
    out.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(asset=str(asset), support_z=support_z,
                         alignments={k: v[2] for k, v in mappings.items()}), indent=2))


if __name__ == '__main__':
    main()
