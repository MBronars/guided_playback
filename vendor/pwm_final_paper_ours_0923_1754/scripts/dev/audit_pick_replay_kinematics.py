"""Check recorded joint/hand consistency using URDF math, without a simulator."""
import argparse
import json
import pickle
from pathlib import Path
import tarfile
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from utils.reproducibility import file_sha256


def origin(node):
    pose=np.eye(4)
    if node is not None:
        pose[:3,3]=np.fromstring(node.get('xyz','0 0 0'),sep=' ')
        pose[:3,:3]=Rotation.from_euler('xyz',np.fromstring(node.get('rpy','0 0 0'),sep=' ')).as_matrix()
    return pose


def forward_kinematics(urdf,values):
    """URDF parent-to-child transforms; values maps joint names to [T] arrays."""
    T=len(next(iter(values.values())))
    children={j.find('child').get('link') for j in urdf.findall('joint')}
    poses={node.get('name'):np.broadcast_to(np.eye(4),(T,4,4)).copy()
        for node in urdf.findall('link') if node.get('name') not in children}
    pending=list(urdf.findall('joint'))
    while pending:
        ready=[j for j in pending if j.find('parent').get('link') in poses]
        if not ready: raise ValueError('Disconnected or cyclic URDF joints')
        for joint in ready:
            kind=joint.get('type'); motion=np.broadcast_to(np.eye(4),(T,4,4)).copy()
            if kind!='fixed':
                axis_node=joint.find('axis')
                axis=np.fromstring(axis_node.get('xyz') if axis_node is not None else '1 0 0',sep=' ')
                axis=axis/np.linalg.norm(axis); q=np.asarray(values[joint.get('name')])
                if kind in ('revolute','continuous'): motion[:,:3,:3]=Rotation.from_rotvec(q[:,None]*axis).as_matrix()
                elif kind=='prismatic': motion[:,:3,3]=q[:,None]*axis
                else: raise ValueError('Unsupported joint type: '+kind)
            poses[joint.find('child').get('link')]=poses[joint.find('parent').get('link')] @ origin(joint.find('origin')) @ motion
            pending.remove(joint)
    return poses


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('episode','source-geometry','urdf','out'): p.add_argument('--'+key,required=True)
    p.add_argument('--context-len',type=int,default=4); a=p.parse_args()
    episode=json.loads(Path(a.episode).read_text()); source=np.load(a.source_geometry)
    with tarfile.open(episode['shard']) as archive:
        raw={name:np.asarray(pickle.load(archive.extractfile(episode['key']+'.'+name+'.pyd')))[episode['elem']]
            for name in ('joint_pos_obs','joint_pos_target')}
    names=[f'panda_joint{i}' for i in range(1,8)]+['panda_finger_joint1','panda_finger_joint2']
    urdf=ET.parse(a.urdf).getroot(); fk=forward_kinematics(urdf,dict(zip(names,raw['joint_pos_obs'].T)))
    links=['panda_hand','panda_leftfinger','panda_rightfinger']; T=len(raw['joint_pos_obs']); C=a.context_len
    observed=np.broadcast_to(np.eye(4),(T,3,4,4)).copy()
    observed[:,:,:3,3]=source['link_pos'][:,[8,9,10]]
    quat=source['link_quat'][:,[8,9,10]][...,[1,2,3,0]]
    observed[:,:,:3,:3]=Rotation.from_quat(quat.reshape(-1,4)).as_matrix().reshape(T,3,3,3)
    conventions=[]
    for convention in ('link','inertial'):
        candidate=np.stack([fk[name] @ (origin(urdf.find(f"link[@name='{name}']/inertial/origin"))
            if convention=='inertial' else np.eye(4)) for name in links],axis=1)
        root_transform=observed[C-1,0] @ np.linalg.inv(candidate[C-1,0])
        prediction=root_transform @ candidate
        translation=np.linalg.norm(prediction[:,:,:3,3]-observed[:,:,:3,3],axis=-1)
        relative=prediction[:,:,:3,:3] @ observed[:,:,:3,:3].transpose(0,1,3,2)
        angle=np.linalg.norm(Rotation.from_matrix(relative.reshape(-1,3,3)).as_rotvec(),axis=-1).reshape(T,3)
        conventions.append(dict(native_frame_hypothesis=convention,root_transform_from_observed_anchor=root_transform.tolist(),
            observed_translation_max_mm=float(translation[:C].max()*1000),
            future_translation_max_mm=float(translation[C:].max()*1000),
            future_rotation_max_deg=float(np.rad2deg(angle[C:].max())),
            per_link_future_max_mm=dict(zip(links,(translation[C:].max(0)*1000).tolist()))))
    selected=min(conventions,key=lambda r:r['observed_translation_max_mm'])
    # Commanded closure and achieved opening differ under object contact. This
    # comparison qualifies a controller input assumption, not contact forces.
    gap=raw['joint_pos_obs'][-6:,-2:]-raw['joint_pos_target'][-6:,-2:]
    report=dict(episode=episode,conventions=conventions,selected_using_observed_context=selected['native_frame_hypothesis'],
        last_six_frames_observed_minus_commanded_finger_q_mm=(gap*1000).tolist(),
        input_hashes={str(Path(v).resolve()):file_sha256(v) for v in (a.episode,a.source_geometry,a.urdf)},
        interpretation='URDF/recording consistency only; simulator startup, controller tracking and object dynamics remain unqualified')
    Path(a.out).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(selected=selected,mean_finger_command_gap_mm=(gap.mean(0)*1000).tolist()),indent=2))


if __name__=='__main__': main()
