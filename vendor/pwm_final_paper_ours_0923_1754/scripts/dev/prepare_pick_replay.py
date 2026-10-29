"""Export a pinned pickup trajectory to source-link coordinates for execution.

The output separates recorded commanded joints from generated achieved hand
poses. Object poses after initialization are reference measurements, not commands.
"""
import argparse
import json
from pathlib import Path
import pickle
import sys
import tarfile

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from scripts.dev.prepare_pick_guidance_geometry import align
from utils.reproducibility import file_sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('input','geometry','source-geometry','episode','grasp','out'):
        p.add_argument('--'+key,required=True)
    p.add_argument('--trajectory',help='Generated NPZ; absent means recorded hand poses')
    p.add_argument('--trimesh-site',default='/data/user_data/robertre/conda_envs/isaaclab_rlskills/lib/python3.10/site-packages')
    args=p.parse_args()
    out=Path(args.out)
    if out.exists(): raise FileExistsError(out)
    # Keep the model environment's numpy/torch; use only the available mesh reader.
    sys.path.append(args.trimesh_site)
    import trimesh
    b=torch.load(args.input,weights_only=False,map_location='cpu')
    g=torch.load(args.geometry,weights_only=False,map_location='cpu')
    grasp=torch.load(args.grasp,weights_only=False,map_location='cpu')
    episode=json.loads(Path(args.episode).read_text())
    source=np.load(args.source_geometry)
    keys=('joint_pos_obs','joint_vel_obs','joint_pos_target','material_properties')
    with tarfile.open(episode['shard']) as archive:
        raw={k:np.asarray(pickle.load(archive.extractfile(episode['key']+'.'+k+'.pyd')))[episode['elem']] for k in keys}
    asset=Path(g['metadata']['asset'])
    mesh=trimesh.load(asset/(asset.name+'_visual.ply'),process=False)
    if isinstance(mesh,trimesh.Scene): mesh=trimesh.util.concatenate(tuple(mesh.geometry.values()))
    pts,_=trimesh.sample.sample_surface(mesh,512,seed=int(np.random.default_rng(0).integers(0,2**31-1)))
    maps=[align(pts.astype(np.float32),b.body_templates[0,0,:512].numpy())]
    for k,row in ((1,8),(2,9),(3,10)):
        n=int(source['link_local_npts'][row])
        maps.append(align(source['link_local_pts'][row,:n],b.body_templates[0,k,:n].numpy()))
    mr,mt=np.stack([v[0] for v in maps]),np.stack([v[1] for v in maps])
    shift=np.array([-.45,0.,-.10])
    observed=b.body_poses[0].numpy()
    def native(R,t):
        rr=R @ mr
        tt=np.einsum('tkij,kj->tki',R,mt)+t-shift
        quat=Rotation.from_matrix(rr.reshape(-1,3,3)).as_quat().reshape(*rr.shape[:-2],4)[...,[3,0,1,2]]
        return np.concatenate([tt,quat],axis=-1)
    reference=native(observed[...,:3,:3],observed[...,:3,3])
    alignment=float(np.max(np.linalg.norm(reference[:,1:,:3]-source['link_pos'][:,[8,9,10]],axis=-1)))
    if alignment>2e-5: raise ValueError(f'Source hand-frame conversion mismatch: {alignment} m')
    if args.trajectory:
        track=np.load(args.trajectory); R,t=track['R'][0],track['T'][0]
        C=int(track['context_len'])
        if C!=b.context_len: raise ValueError('Context lengths disagree')
        np.testing.assert_allclose(R[:C],observed[:C,:,:3,:3],atol=2e-6,rtol=0)
        np.testing.assert_allclose(t[:C],observed[:C,:,:3,3],atol=2e-6,rtol=0)
    else:
        R,t=observed[...,:3,:3],observed[...,:3,3]; C=b.context_len
    requested_target=np.asarray(track['target'][0]) if args.trajectory and 'target' in track else np.empty(0)
    trajectory=native(R,t)
    c=grasp['calibration']
    rel=np.einsum('tji,tfj->tfi',R[:,1],t[:,[2,3]]-t[:,1,None])
    q=c['q_anchor'].numpy()+((rel-c['finger_anchor'].numpy())*c['axis'].numpy()).sum(-1)
    H=len(t)-C; start=C-1
    initial_velocity=np.zeros(6)
    initial_velocity[:3]=(reference[start,0,:3]-reference[start-1,0,:3])/.05
    r=Rotation.from_quat(reference[:,0,[4,5,6,3]])
    initial_velocity[3:]=(r[start]*r[start-1].inv()).as_rotvec()/.05
    out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out,initial_joint_pos=raw['joint_pos_obs'][start],
        initial_joint_vel=raw['joint_vel_obs'][start],initial_object_pose=reference[start,0],
        initial_object_velocity=initial_velocity,recorded_commands=raw['joint_pos_target'][start:start+H],
        reference_joint_pos=raw['joint_pos_obs'][start:start+H+1],
        reference_body_pose=reference[start:start+H+1],desired_body_pose=trajectory[start:],
        desired_finger_q=q[start:],native_to_template_R=mr,native_to_template_t=mt,
        canonical_shift=shift,context_len=C,dt=.05,requested_target=requested_target)
    files=(args.input,args.geometry,args.source_geometry,args.episode,args.grasp)+( (args.trajectory,) if args.trajectory else () )
    receipt=dict(episode=episode,source_hand_conversion_max_error_m=alignment,
        requested_target=requested_target.tolist(),requested_target_frame='canonical model template origin, not native object link origin',
        object_usd=str(asset/(asset.name+'_coacd/coacd_final.usd')),
        robot_usd='/data/user_data/robertre/RLSkills_IsaacLab/assets/franka_gripper/usd/franka_gripper.usd',
        collector_config='/data/user_data/robertre/RLSkills_IsaacLab/isaaclabenvs/tasks/franka_gripper/base_env_cfg.py',
        table_top_z=float(g['metadata']['support_z']-shift[2]),
        object_material_properties=raw['material_properties'][-1].tolist(),
        gripper_material_properties=raw['material_properties'][0].tolist(),
        horizon=H,source_kind='generated achieved actuator trajectory' if args.trajectory else 'recorded positive control',
        input_hashes={str(Path(v).resolve()):file_sha256(v) for v in files},
        limitations=['Table height inferred from observed support; historical table pose not saved.',
            'Object initial velocity estimated from observed poses; no complete historical simulator snapshot.',
            'Current source controller and USD cooking must be qualified using recorded commands.'])
    out.with_suffix('.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__': main()
