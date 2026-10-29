"""Observed joint calibration and task targets, without constructing grasp proposals."""
import argparse
import json
from pathlib import Path
import pickle
import tarfile
import xml.etree.ElementTree as ET

import numpy as np
import torch

from guidance.replay_control import native_tracks_to_model
from scripts.dev.prepare_pick_guidance_geometry import align
from utils.reproducibility import file_sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene',required=True);p.add_argument('--urdf',required=True)
    p.add_argument('--task',choices=['pick','place'],required=True)
    args=p.parse_args();root=Path(args.scene).resolve()
    b=torch.load(root/'input.pt',weights_only=False);g=torch.load(root/'collision_geometry.pt',weights_only=False)
    episode=json.loads((root/'source_episode.json').read_text()); C=b.context_len
    with tarfile.open(episode['shard']) as tar:
        q=np.asarray(pickle.load(tar.extractfile(episode['key']+'.joint_pos_obs.pyd')))[episode['elem']]
    poses=b.body_poses[0]; R=poses[:,:,:3,:3]
    t=poses[:,:,:3,3];local=(R[:,1].transpose(-1,-2)[:,None]@(t[:,[2,3]]-t[:,1,None])[...,None])[...,0]
    q0=torch.as_tensor(q[C-1,-2:]).float();axis=b.joint_axis[0,[2,3]]
    recovered=q0+((local-local[C-1])*axis).sum(-1)
    error=float((recovered-torch.as_tensor(q[:len(local),-2:])).abs().max())
    if error>1e-5: raise ValueError(f'Joint source audit failed: {error}')
    tree=ET.parse(args.urdf)
    limits=torch.tensor([[float(tree.find(f"joint[@name='panda_finger_joint{i}']/limit").get(k))
                         for k in ('lower','upper','velocity')] for i in (1,2)])
    calibration=dict(q_anchor=q0,finger_anchor=local[C-1],axis=axis,limits=limits,dt=.05)
    target=b.body_poses[0,C-1,0,:3,3]+torch.tensor([0,0,.1]); target_R=None
    if args.task=='place':
        # The geometry preparation already qualified correspondence to these
        # exact native surface samples. Use the same deterministic sample draw.
        import sys
        sys.path.append('/data/user_data/robertre/conda_envs/isaaclab_rlskills/lib/python3.10/site-packages')
        import trimesh
        asset=Path(g['metadata']['asset']);mesh=trimesh.load(asset/f'{asset.name}_visual.ply',process=False)
        if isinstance(mesh,trimesh.Scene): mesh=trimesh.util.concatenate(tuple(mesh.geometry.values()))
        points,_=trimesh.sample.sample_surface(mesh,512,seed=int(np.random.default_rng(0).integers(0,2**31-1)))
        mr,mt,_=align(points.astype(np.float32),b.body_templates[0,0,:512].numpy())
        target_R,target=native_tracks_to_model(episode['requested_goal_native'],mr,mt,[-.45,0,-.1])
        # Table pose was not saved. This reference is derived from the requested
        # resting pose and the collector's 2 mm clearance, never an airborne anchor.
        bottom=(g['bodies'][0]['vertices'].double()@target_R.T+target)[...,2].min()
        g['metadata'].update(support_z=float(bottom)-.002,
            support_kind='inferred from requested resting-pose collision bottom minus collector 2 mm clearance; table pose not saved',
            original_anchor_bottom_reference=g['metadata']['support_z'])
        torch.save(g,root/'collision_geometry.pt')
        (root/'collision_geometry.json').write_text(json.dumps(g['metadata'],indent=2)+'\n')
    hashes={str(root/name):file_sha256(root/name) for name in ('input.pt','collision_geometry.pt','source_episode.json')}
    hashes[str(Path(args.urdf).resolve())]=file_sha256(args.urdf)
    metadata=dict(input_hashes=hashes,joint_recovery_max_error_m=error,
        calibration_source='observed joint q, observed finger transforms and URDF; future q audits only',proposals='none')
    torch.save(dict(calibration=calibration,metadata=metadata),root/'calibration.pt')
    task=dict(task=args.task,target=target.tolist(),target_R=target_R.tolist() if target_R is not None else None,
        goal_provenance=episode.get('goal_provenance','observed anchor +10 cm vertical'),
        support_z=g['metadata']['support_z'],support_kind=g['metadata']['support_kind'])
    (root/'task.json').write_text(json.dumps(task,indent=2)+'\n')
    print(json.dumps(dict(scene=str(root),task=task,joint_audit_max_error_m=error)),flush=True)


if __name__=='__main__':main()
