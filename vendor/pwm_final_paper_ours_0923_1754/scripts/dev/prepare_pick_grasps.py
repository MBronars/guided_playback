"""Prepare observed-geometry grasp proposals and absolute finger calibration.

Uses collision components, URDF finger limits and observed joint positions. The
recorded future is used only to audit the coordinate calibration, never to pick
contacts, an approach or closure timing. Run with the geometry-preparation env.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import tarfile
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial import ConvexHull
import torch

from scripts.dev.prepare_pick_guidance_geometry import align
from utils.reproducibility import file_sha256


def margin(points, components):
    return np.min([np.max(points @ p[:, :3].T+p[:, 3], axis=-1) for p in components], axis=0)


def surface(geometry, n=1200):
    """Area-sampled hull surfaces, excluding points internal to another component."""
    rng = np.random.default_rng(781)
    planes = [p.numpy() for p in geometry['planes']]
    vertices = geometry['vertices'].numpy()
    points, normals = [], []
    for i, p in enumerate(planes):
        v = vertices[np.max(vertices @ p[:, :3].T+p[:, 3], axis=-1) < 2e-5]
        hull = ConvexHull(v)
        tri = v[hull.simplices]
        area = np.linalg.norm(np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]), axis=1)/2
        ids = rng.choice(len(tri), max(128, n//len(planes)), p=area/area.sum())
        uv = rng.random((len(ids), 2))
        uv[uv.sum(1)>1] = 1-uv[uv.sum(1)>1]
        x = tri[ids, 0] + uv[:, :1]*(tri[ids, 1]-tri[ids, 0]) + uv[:, 1:]*(tri[ids, 2]-tri[ids, 0])
        keep = np.ones(len(x), dtype=bool)
        for j, other in enumerate(planes):
            if j != i:
                keep &= np.max(x @ other[:, :3].T+other[:, 3], axis=-1) >= -1e-5
        points.append(x[keep])
        normals.append(hull.equations[ids, :3][keep])
    return np.concatenate(points), np.concatenate(normals)


def prepare(batch, geometry, source, raw_q, urdf, max_proposals=8, allow_empty=False):
    C = batch.context_len
    pose = batch.body_poses[0].numpy()
    R0, t0 = pose[C-1, :, :3, :3], pose[C-1, :, :3, 3]
    axis = batch.joint_axis[0, [2, 3]].numpy()
    translations = pose[..., :3, 3]
    relative = np.einsum('tji,tfj->tfi', pose[:, 1, :3, :3], translations[:, [2, 3]]-translations[:, 1, None])
    q0 = raw_q[C-1, -2:]
    recovered = q0 + ((relative-relative[C-1])*axis).sum(-1)
    error = float(np.max(np.abs(recovered-raw_q[:, -2:])))
    if error > 1e-5:
        raise ValueError(f'Absolute joint calibration disagrees with source: {error} m')
    tree = ET.parse(urdf)
    limits, pads, normals, mappings = [], [], [], {}
    for body, row in ((1, 8), (2, 9), (3, 10)):
        n = int(source['link_local_npts'][row])
        mappings[body] = align(source['link_local_pts'][row, :n], batch.body_templates[0, body, :n].numpy())
    for body, side, joint in ((2, 'left', 1), (3, 'right', 2)):
        lim = tree.find(f"joint[@name='panda_finger_joint{joint}']/limit")
        limits.append([float(lim.get(k)) for k in ('lower', 'upper', 'velocity')])
        tip = tree.find(f"joint[@name='panda_{side}finger_tip_joint']/origin")
        tip = np.array([float(v) for v in tip.get('xyz').split()])
        mr, mt, _ = mappings[body]
        tip = mr @ tip + mt
        inward = R0[body].T @ R0[1] @ (-axis[body-2])
        best = None
        vertices = geometry['bodies'][body]['vertices'].numpy()
        for plane in geometry['bodies'][body]['planes'][0].numpy():
            if plane[:3] @ inward < .8:
                continue
            face = vertices[np.abs(vertices @ plane[:3]+plane[3]) < 2e-5]
            if len(face) < 3:
                continue
            point = face.mean(0)
            score = np.linalg.norm(point-tip)
            if best is None or score < best[0]:
                best = (score, point, plane[:3])
        if best is None:
            raise ValueError(f'No inward collision pad face for body {body}')
        pads.append(best[1]); normals.append(best[2])
    pads, normals, limits = np.array(pads), np.array(normals), np.array(limits)
    pad_hand = relative[C-1] + np.einsum('fij,fj->fi', R0[1].T @ R0[[2, 3]], pads)
    pad_zero = pad_hand-q0[:, None]*axis
    pad_mid = pad_zero.mean(0)
    closing = axis[0]/np.linalg.norm(axis[0])
    zero_width = (pad_zero[0]-pad_zero[1]) @ closing
    approach = mappings[1][0] @ np.array([0., 0., 1.])
    approach -= closing*(approach @ closing)
    approach /= np.linalg.norm(approach)
    local_basis = np.stack([closing, np.cross(approach, closing), approach], axis=1)
    x, normals_object = surface(geometry['bodies'][0])
    d = x[:, None]-x[None]
    width = np.linalg.norm(d, axis=-1)
    direction = d/np.maximum(width[..., None], 1e-9)
    a = np.einsum('ijk,ik->ij', direction, normals_object)
    b = -np.einsum('ijk,jk->ij', direction, normals_object)
    q = (width-zero_width)/2
    mask = (width>.012) & (q>limits[:, 0].max()+.001) & (q<limits[:, 1].min()-.001) & (a>.9) & (b>.9)
    pairs = np.stack(np.where(mask), axis=1)
    if len(pairs) == 0 and not allow_empty:
        raise ValueError('No opposing contact pairs fit this calibrated gripper')
    centres = (x[pairs[:, 0]]+x[pairs[:, 1]])/2
    mid_world = centres @ R0[0].T+t0[0]
    order = np.argsort(np.linalg.norm(mid_world-t0[1], axis=1))[:500]
    components = [p.numpy() for p in geometry['bodies'][0]['planes']]
    templates = {k: batch.body_templates[0, k, batch.body_template_mask[0, k]].numpy() for k in (1, 2, 3)}
    finger_R = R0[1].T @ R0[[2, 3]]
    candidates = []
    for index in order:
        left, right = pairs[index]
        contacts = x[[left, right]]
        world_axis = R0[0] @ direction[left, right]
        reference = R0[1] @ approach
        forward = reference-world_axis*(reference @ world_axis)
        if np.linalg.norm(forward) < 1e-6:
            forward = np.cross(world_axis, [1., 0., 0.])
        if np.linalg.norm(forward) < 1e-6:
            forward = np.cross(world_axis, [0., 1., 0.])
        forward /= np.linalg.norm(forward)
        for angle in (0., -np.pi/3, np.pi/3, np.pi):
            fwd = forward*np.cos(angle)+np.cross(world_axis, forward)*np.sin(angle)
            rh = np.stack([world_axis, np.cross(fwd, world_axis), fwd], axis=1) @ local_basis.T
            th = (R0[0] @ contacts.mean(0)+t0[0])-rh @ pad_mid
            qc = np.full(2, q[left, right])
            qo = np.minimum(qc+.012, limits[:, 1])
            max_depth, min_bottom = 0., float('inf')
            for alpha in (0., .5, 1.):
                ht = th-(1-alpha)*.035*fwd
                qq = (1-alpha)*qo+alpha*qc
                clouds = [templates[1] @ rh.T+ht]
                for k in (2, 3):
                    ft = ht+rh @ (relative[C-1, k-2]+(qq[k-2]-q0[k-2])*axis[k-2])
                    clouds.append(templates[k] @ (rh @ finger_R[k-2]).T+ft)
                cloud = np.concatenate(clouds)
                max_depth = max(max_depth, float((-margin((cloud-t0[0]) @ R0[0], components)).clip(0).max()))
                min_bottom = min(min_bottom, float(cloud[:, 2].min()))
            if max_depth > .002 or min_bottom < geometry['metadata']['support_z']-.002:
                continue
            orientation = np.linalg.norm(rh-R0[1])/np.sqrt(2)
            score = np.linalg.norm(th-t0[1])+.04*orientation
            candidates.append(dict(contacts=torch.tensor(contacts, dtype=torch.float32),
                normals=torch.tensor(normals_object[[left, right]], dtype=torch.float32),
                hand_R_object=torch.tensor(R0[0].T @ rh, dtype=torch.float32),
                hand_t_object=torch.tensor(R0[0].T @ (th-t0[0]), dtype=torch.float32),
                q_close=torch.tensor(qc, dtype=torch.float32), q_open=torch.tensor(qo, dtype=torch.float32),
                approach_distance=.035, score=float(score), sampled_approach_depth_m=max_depth,
                sampled_approach_bottom_m=min_bottom))
    selected = []
    for p in sorted(candidates, key=lambda v: v['score']):
        if any(float((p['contacts']-old['contacts']).norm()) < .008 and
               float((p['hand_R_object']-old['hand_R_object']).norm()) < .3 for old in selected):
            continue
        selected.append(p)
        if len(selected) >= max_proposals:
            break
    if not selected and not allow_empty:
        raise ValueError('All geometric grasp approaches failed the sampled clearance checks')
    calibration = dict(q_anchor=q0, finger_anchor=relative[C-1], axis=axis, pad_local=pads,
        pad_normal_local=normals, limits=limits, approach_local=approach, dt=.05)
    calibration = {k: torch.tensor(v, dtype=torch.float32) if isinstance(v, np.ndarray) else v
                   for k, v in calibration.items()}
    return dict(calibration=calibration, proposals=selected,
        metadata=dict(joint_recovery_max_error_m=error, available_surface_points=len(x),
            opposing_pairs=len(pairs), sampled_clear_approaches=len(candidates),
            proposal_selection='observed geometry, observed hand pose and URDF limits only',
            pad_definition='inward collision-hull face centroid closest to URDF fingertip marker',
            limits='sampled geometric feasibility only; no force or full-arm reachability guarantee'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for k in ('input', 'geometry', 'source-geometry', 'episode', 'gripper-urdf', 'out'):
        parser.add_argument('--'+k, required=True)
    parser.add_argument('--allow-empty',action='store_true',help='Save calibration and an explicit empty proposal set on grasp-search failure')
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    batch = torch.load(args.input, weights_only=False, map_location='cpu')
    geometry = torch.load(args.geometry, weights_only=False, map_location='cpu')
    episode = json.loads(Path(args.episode).read_text())
    with tarfile.open(episode['shard']) as archive:
        raw = pickle.load(archive.extractfile(episode['key']+'.joint_pos_obs.pyd'))
    result = prepare(batch, geometry, np.load(args.source_geometry), np.asarray(raw)[episode['elem']], args.gripper_urdf,
                     allow_empty=args.allow_empty)
    result['metadata']['input_hashes'] = {str(Path(v).resolve()): file_sha256(v) for v in
        (args.input, args.geometry, args.source_geometry, args.episode, args.gripper_urdf)}
    result['metadata']['episode'] = episode
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, out)
    def serial(v):
        if torch.is_tensor(v): return v.tolist()
        if isinstance(v, dict): return {k: serial(x) for k,x in v.items()}
        if isinstance(v, list): return [serial(x) for x in v]
        return v
    out.with_suffix('.json').write_text(json.dumps(serial(result), indent=2)+'\n')
    print(json.dumps(dict(proposals=len(result['proposals']), **result['metadata']), indent=2))


if __name__ == '__main__':
    main()
