"""Read contact forces during identical replay, without changing its dynamics."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

from guidance.persistent_replay import PersistentPickupReplay


class ContactRecorder:
    """Delegate all engine operations; inspect only the completed solver step."""
    def __init__(self, engine):
        self.module=engine.mujoco;self.rows=[]
        self.object=engine.ids[0];self.groups={0:0,**{body:i+1 for i,body in enumerate(engine.ids[1:])}}
        self.dof=int(engine.model.joint('object_free').dofadr[0]);self.jq=engine.jq

    def __getattr__(self,name):return getattr(self.module,name)

    def mj_step(self,model,data):
        self.module.mj_step(model,data)
        force=np.zeros((4,3));normal=np.zeros(4);counts=np.zeros(4,dtype=int)
        for i in range(data.ncon):
            contact=data.contact[i]
            a,z=model.geom_bodyid[contact.geom1],model.geom_bodyid[contact.geom2]
            if self.object not in (a,z):continue
            other=a if z==self.object else z
            if other not in self.groups:raise ValueError('Unrecognized object contact partner')
            j=self.groups[other];local=np.empty(6)
            self.module.mj_contactForce(model,data,i,local)
            world=contact.frame.reshape(3,3).T@local[:3]
            force[j]+=world*(1 if z==self.object else -1)
            normal[j]+=local[0];counts[j]+=1
        # Independent engine check of the sign/frame conversion. The passive
        # object's free translational DOFs have only contact constraints.
        expected=data.qfrc_constraint[self.dof:self.dof+3]
        np.testing.assert_allclose(force.sum(0),expected,rtol=1e-8,atol=1e-7)
        self.rows.append(dict(time=float(data.time),force=force,normal=normal,count=counts,
            q=data.qpos[self.jq].copy(),command=data.ctrl.copy(),actuator_force=data.actuator_force.copy(),
            force_conversion_error=float(np.max(np.abs(force.sum(0)-expected)))))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',required=True,type=Path)
    parser.add_argument('--feedback-run',required=True,type=Path);parser.add_argument('--out',required=True,type=Path)
    args=parser.parse_args();root=args.root.resolve();out=args.out.resolve();out.mkdir(exist_ok=False)
    for path in [root/'vendor_mujoco',root.parent/'overnight_2026-09-14/vendor']:sys.path.append(str(path))
    scenes=json.loads((root/'scenes_v2.json').read_text())['scenes'];work=[]
    for scene in scenes:
        for step in ['dt0005','dt00025']:
            path=root/'persistent_replay_controls_v1'/scene['name']/step
            work.append(dict(name=scene['name']+'_recorded_'+step,scene=scene['name'],kind='recorded',path=path,frame=0,
                             commands=None,expected=path/'execution.npz',directory=Path(scene['directory'])))
    feedback=args.feedback_run.resolve();report=json.loads((feedback/'report.json').read_text());a=report['arguments']
    options=json.loads((Path(a['base_run'])/'protocol.json').read_text())['arguments']
    for i,path in enumerate([Path(a['replay']),Path(a['fine_replay'])]):
        for variant in ['continue_original','feedback_sample','feedback_noise_opt']:
            work.append(dict(name=feedback.name+'_'+variant+f'_step{i}',scene=Path(options['input']).parent.name,kind=variant,
                path=path,frame=report['prefix_frames'],commands=feedback/(variant+'_commands.npz'),expected=feedback/f'{variant}_step{i}.npz',directory=Path(options['input']).parent))
    rows=[]
    for item in work:
        path=item['path'];reference=np.load(path/'execution.npz');frame=item['frame']
        c=torch.load(item['directory']/'calibration.pt',weights_only=False,map_location='cpu')['calibration']
        c={k:np.asarray(v,dtype=np.float64) for k,v in c.items()}
        command=np.load(item['commands']) if item['commands'] else dict(R=reference['desired_R'],T=reference['desired_T'],commanded=reference['commanded_q'])
        engine=PersistentPickupReplay(path,frame=frame);recorder=ContactRecorder(engine);engine.mujoco=recorder
        result=engine.advance(command['R'][:,1],command['T'][:,1],command['commanded'],dt=float(reference['dt']),joint_limits=c['limits'][:,:2])
        expected=np.load(item['expected'])
        if item['kind']=='recorded':
            np.testing.assert_array_equal(result['R'],expected['model_R']);np.testing.assert_array_equal(result['T'],expected['model_T'])
            np.testing.assert_array_equal(result['q'],expected['joint_q'])
            np.testing.assert_array_equal(result['integration_states'],np.load(path/'integration_states.npz')['states'])
        else:
            for k in ['R','T']:np.testing.assert_array_equal(result[k],expected[k][frame:])
            for k in ['q','time','integration_states']:np.testing.assert_array_equal(result[k],expected[k])
        arrays={k:np.array([v[k] for v in recorder.rows]) for k in recorder.rows[0]}
        np.savez_compressed(out/(item['name']+'.npz'),**arrays,pose_time=result['time'],R=result['R'],T=result['T'])
        row=dict(name=item['name'],scene=item['scene'],kind=item['kind'],frame=frame,
            step=float(engine.model.opt.timestep),mass_kg=float(engine.model.body_mass[engine.ids[0]]),
            samples=len(recorder.rows),bitwise_replay=True,force_conversion_error_max=float(arrays['force_conversion_error'].max()),
            mean_upward_force_by_partner_N=arrays['force'][...,2].mean(0).tolist(),
            peak_normal_force_by_partner_N=arrays['normal'].max(0).tolist(),
            fraction_both_fingers_positive_normal=float((arrays['normal'][:,2:]>.01).all(-1).mean()),
            object_lift_mm=float((result['T'][-1,0,2]-result['T'][0,0,2])*1000),
            replay_state_sha256=hashlib.sha256((path/'integration_states.npz').read_bytes()).hexdigest(),
            commands_sha256=hashlib.sha256((item['commands'] or path/'execution.npz').read_bytes()).hexdigest())
        rows.append(row);print(item['name'],row['object_lift_mm'],row['fraction_both_fingers_positive_normal'],flush=True)
    (out/'report.json').write_text(json.dumps(dict(status='passed',results=rows,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        partner_order=['table','hand','finger0','finger1'],
        definition='Read mj_contactForce immediately after each mj_step. Contact-frame forces are transformed to world coordinates and signed for the object. Sum agrees with object translational qfrc_constraint at every step. Pose and full integration state are bitwise unchanged.',
        limitations='Forces describe this independent engine and controller. Solver forces correspond to the force evaluation in the step, before position integration; time labels are step ends. Recorded cases use the qualified fixed mesh-preload approximation, not recorded motor commands.'),indent=2)+'\n')


if __name__=='__main__':main()
