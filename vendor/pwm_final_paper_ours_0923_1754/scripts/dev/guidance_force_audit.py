"""Inspect forces and obstacle contacts at every solver step of an exact replay."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

from guidance.persistent_replay import PersistentPickupReplay
from scripts.dev.audit_replay_contact_forces import ContactRecorder
from utils.reproducibility import file_sha256


class DetailedContactRecorder(ContactRecorder):
    def __init__(self,engine):
        super().__init__(engine)
        self.obstacles={i for i in range(engine.model.ngeom)
            if (engine.mujoco.mj_id2name(engine.model,engine.mujoco.mjtObj.mjOBJ_GEOM,i) or '').startswith('obstacle_')}
        self.body_column={int(body):i for i,body in enumerate(engine.ids)}

    def mj_step(self,model,data):
        super().mj_step(model,data)
        counts=np.zeros(4,dtype=int);depth=np.zeros(4);normal=np.zeros(4)
        dynamic_pair_depth=np.zeros(3)
        for i in range(data.ncon):
            c=data.contact[i];a,z=int(c.geom1),int(c.geom2)
            body_a,body_z=int(model.geom_bodyid[a]),int(model.geom_bodyid[z])
            if self.object in (body_a,body_z):
                other=body_a if body_z==self.object else body_z
                if other in self.body_column and self.body_column[other]>0:
                    col=self.body_column[other]-1
                    dynamic_pair_depth[col]=max(dynamic_pair_depth[col],max(0.,-float(c.dist)))
            if a not in self.obstacles and z not in self.obstacles:continue
            dynamic=body_z if a in self.obstacles else body_a
            if dynamic not in self.body_column:continue
            col=self.body_column[dynamic];counts[col]+=1;depth[col]=max(depth[col],max(0.,-float(c.dist)))
            force=np.empty(6);self.module.mj_contactForce(model,data,i,force);normal[col]+=force[0]
        self.rows[-1].update(obstacle_contacts=counts,obstacle_depth=depth,obstacle_normal=normal,
                             object_gripper_depth=dynamic_pair_depth)


def audit_forces(replay, out, *, vendor_paths=()):
    replay,out=Path(replay).resolve(),Path(out).resolve()
    for path in vendor_paths:
        if str(path) not in sys.path:sys.path.append(str(path))
    reference=np.load(replay/'execution.npz');states=np.load(replay/'integration_states.npz')
    provenance=json.loads((replay/'protocol.json').read_text());options=provenance['arguments']
    if options['finger_mode']!='mesh_preload' or options.get('finger_velocity_feedforward'):
        raise ValueError('Force audit requires the qualified fixed mesh-preload command program')
    c=torch.load(Path(options['scene'])/'calibration.pt',weights_only=False,map_location='cpu')['calibration']
    engine=PersistentPickupReplay(replay,frame=0);recorder=DetailedContactRecorder(engine);engine.mujoco=recorder
    result=engine.advance(reference['desired_R'][:,1],reference['desired_T'][:,1],reference['commanded_q'],
                          dt=float(reference['dt']),joint_limits=np.asarray(c['limits'])[:,:2])
    for key,expected in [('R',reference['model_R']),('T',reference['model_T']),('q',reference['joint_q']),
                         ('integration_states',states['states'])]:
        np.testing.assert_array_equal(result[key],expected)
    arrays={key:np.asarray([row[key] for row in recorder.rows]) for key in recorder.rows[0]}
    out.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(out/'solver_steps.npz',**arrays,R=result['R'],T=result['T'],pose_time=result['time'])
    force=arrays['force'];finger_normal=arrays['normal'][:,2:]
    both=(finger_normal>.01).all(-1)
    report=dict(status='passed',bitwise_pose_joint_and_integration_replay=True,
        source_sha256=file_sha256(__file__),replay_state_sha256=file_sha256(replay/'integration_states.npz'),
        replay_protocol_sha256=file_sha256(replay/'protocol.json'),execution_sha256=file_sha256(replay/'execution.npz'),
        timestep=float(engine.model.opt.timestep),samples=len(recorder.rows),
        body_mass_kg=float(engine.model.body_mass[engine.ids[0]]),
        contact_partner_order=['static world (table and any obstacles)','hand','finger0','finger1'],
        mean_upward_force_by_partner_N=force[...,2].mean(0).tolist(),
        peak_normal_force_by_partner_N=arrays['normal'].max(0).tolist(),
        both_fingers_positive_normal_fraction=float(both.mean()),
        first_both_fingers_positive_normal_time=float(arrays['time'][np.flatnonzero(both)[0]]) if both.any() else None,
        force_conversion_error_max=float(arrays['force_conversion_error'].max()),
        peak_object_gripper_depth_mm=float(arrays['object_gripper_depth'].max()*1000),
        obstacle_count=len(recorder.obstacles),obstacle_contact_solver_steps=int((arrays['obstacle_contacts']>0).any(-1).sum()),
        obstacle_peak_depth_mm=float(arrays['obstacle_depth'].max()*1000),
        obstacle_peak_normal_force_N=float(arrays['obstacle_normal'].max()),
        limitations='Independent engine forces at each solver evaluation, before position integration; time labels are step ends. No threshold here establishes source-simulator or robot force accuracy. Dense obstacle contacts supplement geometric clearance probes; absence of contacts does not establish a 5 mm continuous clearance.')
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--replay',required=True);p.add_argument('--out',required=True)
    p.add_argument('--vendor',action='append',default=[]);a=p.parse_args()
    result=audit_forces(a.replay,a.out,vendor_paths=a.vendor)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
