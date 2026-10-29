"""Observed-geometry actuator waypoints for approximate flow inpainting.

The actuator path is prescribed here; passive object motion remains generated.
These are desired achieved poses, not validated robot commands.
"""
import torch

from utils.pose_native import articulated_deltas, joint_layout, pose_channel_mask, JOINT_PRISMATIC
from utils.pose_ops import so3_exp, so3_log


def grasp_waypoints(batch,target,grasp,*,proposal=0,horizon=36,approach_frames=12,
                    close_frame=18,hold_frames=6,scope='actuated'):
    if batch.batch_size!=1 or not 0<approach_frames<close_frame<horizon-hold_frames+1:
        raise ValueError('Require a reach, closure and complete hold within one-scene horizon')
    if scope not in ('actuated','fingers'):
        raise ValueError('Inpainting scope must be actuated or fingers')
    if not 0<=proposal<len(grasp['proposals']):
        raise ValueError('No requested geometric grasp proposal')
    c={k:v.to(batch.positions) if torch.is_tensor(v) else v for k,v in grasp['calibration'].items()}
    p={k:v.to(batch.positions) if torch.is_tensor(v) else v for k,v in grasp['proposals'][proposal].items()}
    anchor=batch.body_poses[:,batch.context_len-1].float()
    if anchor.shape[1]!=4 or not bool(batch.pose_action_mask[0,1:].all()) or bool(batch.pose_action_mask[0,0]):
        raise ValueError('Expected object, hand and two actuated fingers')
    R0,t0=anchor[:,:,:3,:3],anchor[:,:,:3,3]
    Rh=R0[:,0] @ p['hand_R_object']
    th=(R0[:,0] @ p['hand_t_object'][None,:,None])[...,0]+t0[:,0]
    forward=(Rh @ c['approach_local'][None,:,None])[...,0]
    pre=th-float(p['approach_distance'])*forward
    ticks=torch.arange(horizon+1,device=batch.device,dtype=anchor.dtype)
    def smooth(x):
        x=x.clamp(0,1)
        return x*x*(3-2*x)
    reach=smooth(ticks/approach_frames)
    close=smooth((ticks-approach_frames)/(close_frame-approach_frames))
    lift=smooth((ticks-close_frame)/(horizon-hold_frames+1-close_frame))
    hand_t=torch.lerp(t0[:,1,None],pre[:,None],reach[None,:,None])
    hand_t=torch.lerp(hand_t,th[:,None],close[None,:,None])
    hand_t=hand_t+lift[None,:,None]*(target-t0[:,0])[:,None]
    rotation=so3_log(Rh @ R0[:,1].transpose(-1,-2))
    hand_R=so3_exp(rotation[:,None]*reach[None,:,None]) @ R0[:,1,None]
    q=torch.lerp(c['q_anchor'][None,None],p['q_open'][None,None],reach[None,:,None])
    q=torch.lerp(q,p['q_close'][None,None],close[None,:,None])
    poses=anchor[:,None].expand(-1,horizon+1,-1,-1,-1).clone()
    poses[:,:,1,:3,:3]=hand_R; poses[:,:,1,:3,3]=hand_t
    for i,k in enumerate((2,3)):
        relative_R=R0[:,1].transpose(-1,-2) @ R0[:,k]
        poses[:,:,k,:3,:3]=hand_R @ relative_R[:,None]
        local=c['finger_anchor'][i]+(q[:,:,i]-c['q_anchor'][i])[...,None]*c['axis'][i]
        poses[:,:,k,:3,3]=hand_t+(hand_R @ local[...,None])[...,0]
    # Preserve the exact observed anchor before encoding deltas across the whole horizon.
    poses[:,0]=anchor
    delta_t=poses[:,1:,:,:3,3]-poses[:,:-1,:,:3,3]
    delta_R=poses[:,1:,:,:3,:3] @ poses[:,:-1,:,:3,:3].transpose(-1,-2)
    delta=torch.cat([delta_t,so3_log(delta_R)],dim=-1)
    ridx=torch.arange(4,device=batch.device)
    layout=joint_layout(batch,ridx)
    if layout is None or not bool((layout[1][2:]==JOINT_PRISMATIC).all()):
        raise ValueError('Grasp waypoints require two articulated prismatic fingers')
    delta=articulated_deltas(poses,delta,layout)
    normalized=batch.pose_delta_stats.normalize(delta,clip=False)
    mask=pose_channel_mask(batch,ridx).to(device=batch.device,dtype=torch.bool)[None,None] & batch.pose_action_mask[:,None,:,None]
    if scope=='fingers': mask[:,:,1]=False
    return dict(known_pose=torch.where(mask,normalized,torch.zeros_like(normalized)),known_mask=mask,
        actuator_poses=poses[:,:,1:],finger_q=q,
        description='prescribed geometric actuator path; passive object remains generated; execution unverified')
