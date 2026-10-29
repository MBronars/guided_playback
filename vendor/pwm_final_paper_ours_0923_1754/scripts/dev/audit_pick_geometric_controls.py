"""Check prescribed ideal paths on real meshes, separately from model outputs.

The object is deliberately translated with the constructed hand after closure.
These are geometry controls, never generated or physically executed successes.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

from guidance.grasp import StagedPickObjective
from guidance.pick_validation import PickValidator
from guidance.waypoints import grasp_waypoints
from utils.reproducibility import file_sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('input','geometry','grasp','out'): p.add_argument('--'+key,required=True)
    p.add_argument('--offset',default='0,0,0.1');p.add_argument('--horizon',type=int,default=36)
    p.add_argument('--approach-frames',type=int,default=12);p.add_argument('--close-frame',type=int,default=18)
    p.add_argument('--hold-frames',type=int,default=6);a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    b=torch.load(a.input,map_location='cpu',weights_only=False)
    geometry=torch.load(a.geometry,map_location='cpu',weights_only=False)
    grasp=torch.load(a.grasp,map_location='cpu',weights_only=False)
    for key in ('input','geometry'):
        path=str(Path(getattr(a,key)).resolve())
        if grasp['metadata']['input_hashes'].get(path)!=file_sha256(path): raise ValueError('Mismatched grasp input: '+path)
    C,H=b.context_len,a.horizon; offset=torch.tensor([float(x) for x in a.offset.split(',')]);target=b.body_poses[:,C-1,0,:3,3]+offset
    validator=PickValidator(b,target,geometry,grasp['calibration'],hold_frames=a.hold_frames)
    settings=dict(horizon=H,approach_frames=a.approach_frames,close_frame=a.close_frame,hold_frames=a.hold_frames)
    rows=[]
    for prop in range(len(grasp['proposals'])):
        control=grasp_waypoints(b,target,grasp,proposal=prop,**settings)
        pose=b.body_poses[:,C-1:C].expand(-1,H+1,-1,-1,-1).clone();pose[:,:,1:]=control['actuator_poses']
        tick=torch.arange(H+1,dtype=torch.float32)
        x=((tick-a.close_frame)/(H-a.hold_frames+1-a.close_frame)).clamp(0,1);lift=x*x*(3-2*x)
        pose[:,:,0,:3,3]+=lift[None,:,None]*offset
        R=pose[:,1:,:,:3,:3];T=pose[:,1:,:,:3,3];pos=torch.zeros(1,H,b.num_particles,3)
        for k in range(4):
            template=b.body_templates[:,k,b.body_template_mask[0,k]]
            pos[:,:,b.particle_body[0]==k]=template[:,None] @ R[:,:,k].transpose(-1,-2)+T[:,:,k,None]
        pred=dict(positions=pos,R=R,t=T,rigid_idx=torch.arange(4))
        objective=StagedPickObjective(b,target,geometry,grasp,proposal=prop,
            approach_frames=a.approach_frames,close_frame=a.close_frame,hold_frames=a.hold_frames)
        loss,terms=objective(pred,b)
        rows.append(dict(proposal=prop,validation=validator(pred),stage_loss=float(loss),stages=terms))
    report=dict(arguments=vars(a),input_sha256=file_sha256(a.input),grasp_sha256=file_sha256(a.grasp),
        description=__doc__,rows=rows)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,key,label,threshold in zip(axes,('contact','penetration'),('Worst held finger gap (mm)','Max sampled intersection (mm)'),(.005,.001)):
        ax.bar([r['proposal'] for r in rows],[1000*(r['validation']['residuals_m'][key]+threshold) for r in rows],
            color=['#198754' if r['validation']['accepted'] else '#d68131' for r in rows])
        ax.axhline(1000*threshold,c='.3',ls='--');ax.set(xlabel='Observed-geometry grasp proposal',ylabel=label);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Constructed geometry controls: object follows prescribed hand after closure\nGreen passes all geometric checks; orange fails at least one. These are NOT model or simulator results.',fontsize=10)
    fig.tight_layout();fig.savefig(out/'geometric_controls.png',dpi=150);plt.close(fig)
    print(json.dumps(dict(accepted=sum(r['validation']['accepted'] for r in rows),total=len(rows),out=str(out))))


if __name__=='__main__': main()
