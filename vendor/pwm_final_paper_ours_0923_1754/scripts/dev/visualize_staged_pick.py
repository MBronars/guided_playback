"""Real-scene grasp proposals and paired guidance movies with shared axes."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.ticker import MaxNLocator
import numpy as np
import torch

from guidance.grasp import StagedPickObjective


def proposal_pose(batch, grasp, proposal):
    pose = batch.body_poses[0, batch.context_len-1].clone()
    R, t = pose[:, :3, :3], pose[:, :3, 3]
    c, p = grasp['calibration'], grasp['proposals'][proposal]
    rh, th = R[0] @ p['hand_R_object'], R[0] @ p['hand_t_object']+t[0]
    rf = R[1].T @ R[[2, 3]]
    R[1], t[1] = rh, th
    for i,k in enumerate((2, 3)):
        R[k] = rh @ rf[i]
        t[k] = th+rh @ (c['finger_anchor'][i]+(p['q_close'][i]-c['q_anchor'][i])*c['axis'][i])
    return R,t


def points(batch,R,t):
    result = torch.empty_like(batch.positions[0,0])
    for k in range(4):
        result[batch.particle_body[0]==k] = batch.body_templates[0,k,batch.body_template_mask[0,k]] @ R[k].T+t[k]
    return result.numpy()


def setup_axis(ax,lo,hi):
    centre=(lo+hi)/2
    radius=max(hi-lo)*.55
    ax.set(xlim=(centre[0]-radius,centre[0]+radius),ylim=(centre[1]-radius,centre[1]+radius),
           zlim=(centre[2]-radius,centre[2]+radius))
    ax.set_box_aspect((1,1,1)); ax.set_proj_type('ortho'); ax.view_init(elev=24,azim=-65)
    ax.tick_params(labelsize=7)
    for axis in (ax.xaxis,ax.yaxis,ax.zaxis): axis.set_major_locator(MaxNLocator(4))
    ax.set_xlabel('x (m)',fontsize=8); ax.set_ylabel('y (m)',fontsize=8); ax.set_zlabel('z (m)',fontsize=8)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True)
    parser.add_argument('--geometry',required=True)
    parser.add_argument('--grasp',required=True)
    parser.add_argument('--out',required=True)
    parser.add_argument('--runs',nargs='*',default=[],help='label=directory containing seed_<seed>.npz')
    parser.add_argument('--seed',type=int,default=54321)
    parser.add_argument('--include-recording',action='store_true')
    args=parser.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    b=torch.load(args.input,map_location='cpu',weights_only=False)
    g=torch.load(args.geometry,map_location='cpu',weights_only=False)
    grasp=torch.load(args.grasp,map_location='cpu',weights_only=False)
    body=b.particle_body[0].numpy(); C=b.context_len
    proposals=[points(b,*proposal_pose(b,grasp,i)) for i in range(len(grasp['proposals']))]
    observed=b.positions[0,C-1].numpy()
    extent=np.concatenate(proposals+[observed]); lo,hi=extent.min(0),extent.max(0)
    fig=plt.figure(figsize=(16,10))
    for i,x in enumerate(proposals):
        ax=fig.add_subplot(2,4,i+1,projection='3d'); setup_axis(ax,lo,hi)
        ax.scatter(*observed[body>0].T,s=1,c='grey',alpha=.15)
        for k,color in ((0,'#287fb8'),(1,'#d75050'),(2,'#d75050'),(3,'#d75050')):
            ax.scatter(*x[body==k].T,s=2,c=color,depthshade=False)
        pose=b.body_poses[0,C-1,0]
        p=grasp['proposals'][i]
        contact=(p['contacts'] @ pose[:3,:3].T+pose[:3,3]).numpy()
        ax.scatter(*contact.T,c='#0aab62',s=35,marker='o',depthshade=False)
        ax.set_title(f'Proposal {i} · joint opening {p["q_close"][0]*1000:.1f} mm',fontsize=10)
    fig.suptitle('Observed object: geometric grasp candidates\nBlue: object · red: proposed gripper · green: target contacts · grey: observed gripper',fontsize=13)
    fig.text(.5,.015,'Proposals use observed geometry only. These are constructed grasp poses, not model outputs or simulator successes.',ha='center',fontsize=10)
    fig.subplots_adjust(left=.03,right=.96,bottom=.09,top=.85,wspace=.2,hspace=.35)
    fig.savefig(out/'grasp_proposals.png',dpi=130); plt.close(fig)
    if not args.runs: return
    panels=[]; phase_settings=set()
    for spec in args.runs:
        label,path=spec.split('=',1)
        data=np.load(Path(path)/f'seed_{args.seed}.npz')
        proto=json.loads((Path(path)/'protocol.json').read_text())
        chosen=json.loads((Path(path)/f'seed_{args.seed}.json').read_text()).get('selected_control',{})
        settings=dict(proto['arguments'],**chosen)
        phase_settings.add((settings.get('approach_frames',12),settings.get('close_frame',18)))
        target=torch.tensor(data['target'])
        diag=StagedPickObjective(b,target,g,grasp,proposal=settings.get('proposal',0),
            approach_frames=settings.get('approach_frames',12),
            close_frame=settings.get('close_frame',18),hold_frames=settings.get('hold_frames',6))
        pred=dict(positions=torch.tensor(data['positions'][:,C:]),R=torch.tensor(data['R'][:,C:]),
                  t=torch.tensor(data['T'][:,C:]),rigid_idx=torch.arange(4))
        metric=diag.measure(pred)
        panels.append((label,data['positions'][0],metric,settings.get('proposal',0)))
    if args.include_recording:
        pred=dict(positions=b.future,R=b.body_poses[:,C:,:,:3,:3],
            t=b.body_poses[:,C:,:,:3,3],rigid_idx=torch.arange(4))
        reference=diag.measure(pred); reference['reference_recording']=True
        panels.append(('Recorded future (reference)',b.positions[0].numpy(),reference,0))
    allp=np.concatenate([x for _,x,_,_ in panels]); lo,hi=allp.min((0,1)),allp.max((0,1))
    target=target[0].numpy(); lo=np.minimum(lo,target); hi=np.maximum(hi,target)
    ncols=min(3,len(panels)); nrows=(len(panels)+ncols-1)//ncols
    fig=plt.figure(figsize=(5*ncols,4.5*nrows)); artists=[]
    for i,(label,x,m,prop) in enumerate(panels):
        ax=fig.add_subplot(nrows,ncols,i+1,projection='3d'); setup_axis(ax,lo,hi)
        ax.set_title(label,fontsize=11)
        clouds=[ax.scatter(*x[0,body==k].T,s=2,c='#287fb8' if k==0 else '#d75050',depthshade=False) for k in range(4)]
        ax.scatter(*target,c='#0aab62',marker='*',s=85)
        caption=ax.text2D(.01,-.23,'',transform=ax.transAxes,fontsize=9)
        artists.append((clouds,caption,x,m))
    title=fig.suptitle('',fontsize=14)
    fig.text(.5,.015,'Blue: object · red: gripper · green: goal. Inpainted paths prescribe actuator motion. Execution is unverified. Playback 4× slower.',ha='center',fontsize=9)
    fig.subplots_adjust(top=.90,bottom=.18,hspace=.35,wspace=.10)
    def update(frame):
        phase='observed' if frame<C else 'future'
        title.set_text(f'Paired picking · seed {args.seed} · frame {frame:02d} · {phase}')
        for clouds,caption,x,m in artists:
            for k,a in enumerate(clouds): a._offsets3d=x[frame,body==k].T
            if frame<C: caption.set_text('Shared observed history')
            else:
                gap=max(m['traces']['gaps'][frame-C+1])*1000
                depth=m['traces']['depth'][frame-C]*1000
                goal_line='Recording follows a different goal' if m.get('reference_recording') else f'Final object error: {m["target_distance_mm"]:.1f} mm'
                caption.set_text(f'Farther-finger gap: {gap:.1f} mm · intersection: {depth:.1f} mm\n{goal_line}')
    frames=list(range(len(panels[0][1])))+[len(panels[0][1])-1]*5
    anim=FuncAnimation(fig,update,frames=frames,interval=200,blit=False)
    anim.save(out/f'comparison_seed_{args.seed}.gif',writer=PillowWriter(fps=5),dpi=95)
    update(frames[-1]); fig.savefig(out/f'final_seed_{args.seed}.png',dpi=120); plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(10,8),sharex=True)
    for label,x,m,prop in panels:
        f=(np.arange(len(m['traces']['pad_gap']))+1)*.05
        axes[0].plot(f,np.max(m['traces']['gaps'][1:],axis=-1)*1000,label=label)
        axes[1].plot(f,np.array(m['traces']['depth'])*1000)
        axes[2].plot(f,np.array(m['traces']['hold_goal'])*1000)
    for ax in axes:
        if len(phase_settings)==1:
            for tick in next(iter(phase_settings)): ax.axvline(tick*.05,c='.6',ls=':')
        ax.grid(alpha=.2)
    axes[0].axhline(5,c='.5',ls='--'); axes[1].axhline(1,c='.5',ls='--'); axes[2].axhline(20,c='.5',ls='--')
    axes[0].set_ylabel('Farther-finger gap (mm)'); axes[0].legend(fontsize=8)
    axes[1].set_ylabel('Maximum sampled\nintersection (mm)')
    axes[2].set_ylabel('Object target error (mm)'); axes[2].set_xlabel('Future time (s)')
    fig.suptitle('Paired traces: proximity alone is insufficient; also inspect penetration and held motion')
    fig.tight_layout(); fig.savefig(out/f'traces_seed_{args.seed}.png',dpi=140); plt.close(fig)
    (out/f'metrics_seed_{args.seed}.json').write_text(json.dumps({label:m for label,_,m,_ in panels},indent=2)+'\n')


if __name__=='__main__': main()
