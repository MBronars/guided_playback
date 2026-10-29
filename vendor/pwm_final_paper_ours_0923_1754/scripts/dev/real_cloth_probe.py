"""RGBench recorded-surface forecasts using an unmodified cloth-trained checkpoint.

The observed surface is sampled at the anchor; short prefix correspondences are
nearest-surface estimates, not ground-truth tracks. Recorded joint trajectories
drive the actual Piper URDF gripper geometry. No future cloth enters the input.
"""
from __future__ import annotations

import argparse
import html
import json
import struct
import time
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import torch

from data.batch import TrajectoryBatch
from data.normalization import domain_normalizers
from data.unified_dataset import _augment_pose_channels
from inference import windowed_rollout
from scripts.dev.observation_probe_common import (Predictor, geometry_residual_mm, require_given_actions,
                                                  write_json, write_frozen_run)
from utils.reproducibility import code_revision, config_for_json, file_sha256, seed_for_device


def read_pcd(path, world_from_camera):
    raw = Path(path).read_bytes()
    header, data = raw.split(b'DATA binary\n', 1)
    if b'FIELDS x y z rgb' not in header or b'TYPE F F F F' not in header:
        raise ValueError('Unsupported PCD layout')
    rows = np.frombuffer(data, dtype='<f4').reshape(-1, 4)
    bits = rows[:, 3].copy().view('<u4')
    rgb = np.stack([(bits >> 16) & 255, (bits >> 8) & 255, bits & 255], axis=-1) / 255
    xyz = rows[:, :3].astype(np.float64) @ world_from_camera[:3, :3].T + world_from_camera[:3, 3]
    valid = np.isfinite(xyz).all(1)
    return xyz[valid], rgb[valid]


def stl_points(path, count, seed):
    raw = Path(path).read_bytes()
    n = struct.unpack_from('<I', raw, 80)[0]
    dtype = np.dtype([('normal','<f4',(3,)),('vertices','<f4',(3,3)),('attribute','<u2')])
    assert len(raw) == 84 + n*50, 'Expected binary STL'
    triangles = np.frombuffer(raw, dtype=dtype, offset=84)['vertices'].astype(np.float64)
    area = np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=1)
    rng = np.random.default_rng(seed)
    tri = triangles[rng.choice(n, count, p=area/area.sum())]
    uv = rng.random((count,2)); uv[uv.sum(1)>1] = 1-uv[uv.sum(1)>1]
    return tri[:,0] + uv[:,:1]*(tri[:,1]-tri[:,0]) + uv[:,1:]*(tri[:,2]-tri[:,0])


def xyz_attr(element, key, default):
    return np.fromstring(element.get(key, default), sep=' ')


def register_prefix(source, target, iterations=30):
    """Rigid ICP initializer for short observed-surface tracks; no future data.

    The subsequent nearest-surface refinement retains a coherent tangential
    displacement instead of projecting the unchanged anchor into each frame.
    This is a state-estimation assumption, not measured particle correspondence.
    """
    tree=cKDTree(target)
    R=np.eye(3); translation=target.mean(0)-source.mean(0)
    moved=source+translation
    for _ in range(iterations):
        distance,index=tree.query(moved,workers=4)
        keep=distance<=np.quantile(distance,.95)
        p=moved[keep];q=target[index[keep]]
        pc=p.mean(0);qc=q.mean(0)
        U,_,Vt=np.linalg.svd((p-pc).T@(q-qc))
        rotation=Vt.T@U.T
        if np.linalg.det(rotation)<0:
            Vt[-1]*=-1;rotation=Vt.T@U.T
        shift=qc-rotation@pc
        moved=moved@rotation.T+shift
        R=rotation@R;translation=rotation@translation+shift
        if np.linalg.norm(shift)<1e-7 and np.linalg.norm(rotation-np.eye(3))<1e-6:break
    return moved,{'translation_m':translation.tolist(),
                  'rotation_degrees':float(np.rad2deg(Rotation.from_matrix(R).magnitude())),
                  'nearest_surface_mean_mm':float(tree.query(moved,workers=4)[0].mean()*1000)}


def piper_geometry(root, csv_path, times, side):
    """URDF FK, with the released benchmark's left +0.25/right -0.25 base offsets."""
    table = np.genfromtxt(csv_path, delimiter=',', names=True)
    order = np.argsort(table['time']); table=table[order]
    if times[0] < table['time'][0] or times[-1] > table['time'][-1]:
        raise ValueError('Requested gripper times outside recorded coverage')
    values = {f'joint{i}':np.interp(times,table['time'],np.unwrap(table[f'pos_{i}'])) for i in range(1,7)}
    opening = np.interp(times,table['time'],table['pos_gripper'])
    if opening.min() < -1e-5 or opening.max() > .03501:
        raise ValueError('Piper opening outside published URDF limits')
    values['joint7'], values['joint8'] = opening, -opening
    urdf = ET.parse(root/'piper/piper_with_gripper.urdf').getroot()
    B=len(times)
    initial=np.tile(np.eye(4),(B,1,1)); initial[:,1,3]=.25 if side=='left' else -.25
    pending=list(urdf.findall('joint'))
    roots={x.get('name') for x in urdf.findall('link')}-{j.find('child').get('link') for j in pending}
    if len(roots)!=1:raise ValueError('URDF needs one root link')
    poses={roots.pop():initial}
    while pending:
        progress=False
        for joint in list(pending):
            parent=joint.find('parent').get('link'); child=joint.find('child').get('link')
            if parent not in poses:
                continue
            origin=joint.find('origin'); base=np.eye(4)
            base[:3,:3]=Rotation.from_euler('xyz',xyz_attr(origin,'rpy','0 0 0')).as_matrix()
            base[:3,3]=xyz_attr(origin,'xyz','0 0 0')
            motion=np.tile(np.eye(4),(B,1,1)); kind=joint.get('type')
            if kind!='fixed':
                axis=xyz_attr(joint.find('axis'),'xyz','1 0 0'); q=values[joint.get('name')]
                if kind=='revolute': motion[:,:3,:3]=Rotation.from_rotvec(q[:,None]*axis).as_matrix()
                elif kind=='prismatic': motion[:,:3,3]=q[:,None]*axis
                else: raise ValueError(kind)
            poses[child]=poses[parent]@base@motion
            pending.remove(joint); progress=True
        if not progress:
            raise ValueError('URDF parent cycle')
    result=[]
    for index,(name,count) in enumerate((('gripper_base',200),('link7',400),('link8',400))):
        points=stl_points(root/f'piper/meshes/{name}.STL',count,110+index)
        center=points.mean(0); points-=center
        p=poses[name].copy(); p[:,:3,3]+=np.einsum('tij,j->ti',p[:,:3,:3],center)
        result.append((points,p))
    return result


def smooth_prefix_motion(prefix, radius):
    """Average observed displacements locally, preserving the measured anchor.

    This is an optional observation preprocessing experiment, not a learned
    tracker. A finite radius prevents borrowing motion from distant cloth.
    """
    if radius <= 0:
        raise ValueError('Motion smoothing radius must be positive')
    anchor = prefix[-1]
    distance, indices = cKDTree(anchor).query(anchor, k=min(16, len(anchor)), workers=4)
    if distance.ndim == 1: distance, indices = distance[:, None], indices[:, None]
    weights = np.exp(-.5*(distance/(radius/2))**2)*(distance <= radius)
    weights /= weights.sum(1, keepdims=True)
    motion = prefix-anchor[None]
    filtered = anchor[None]+(motion[:, indices]*weights[None, :, :, None]).sum(2)
    filtered[-1] = anchor
    return filtered


def build_input(root, capture, cfg, *, anchor_offset, horizon, n_particles=4096, action_delay=0, tracking='nearest', motion_smoothing_radius=0):
    folder=root/'green_tshirt'/capture
    files=sorted((folder/'segment_pcds').glob('*.pcd'))
    stamps=np.array([float(p.name.split('_')[1]) for p in files])
    W=np.linalg.inv(np.array(json.loads((folder/'calibration/world_to_camera_transform.json').read_text())))
    idx=int(np.searchsorted(stamps, stamps[0]+anchor_offset))
    if idx>=len(files) or idx<1: raise ValueError('No anchor frame')
    anchor_t=stamps[idx]; dt=1/30; C=int(cfg.data.context_len)
    requested_horizon=horizon
    last_time=stamps[-1]
    for side in ('left','right'):
        table=np.genfromtxt(folder/f'joints/{side}_arm_joint_states_and_end_pose.csv',delimiter=',',names=True)
        last_time=min(last_time,float(table['time'].max())-action_delay)
    horizon=min(horizon,int(np.floor((last_time-anchor_t)/dt)))
    if horizon<int(cfg.data.future_len):raise ValueError('Insufficient future coverage for one trained window')
    times=anchor_t+(np.arange(C+horizon)-(C-1))*dt
    if times[-1]>stamps[-1]: raise ValueError('Requested horizon outside camera coverage')
    # Prefix reconstruction cannot open any cloud with a timestamp after the anchor.
    used_prefix=[]
    anchor,rgb=read_pcd(files[idx],W); used_prefix.append(str(files[idx]))
    _, voxel_rows=np.unique(np.floor(anchor/.006).astype(np.int64),axis=0,return_index=True)
    n_particles=min(n_particles,len(voxel_rows))
    if n_particles<1024: raise ValueError('Insufficient measured surface coverage')
    rows=np.random.default_rng(1104).choice(voxel_rows,n_particles,replace=False)
    anchor=anchor[rows]; rgb=rgb[rows]
    prefix=[]; residuals=[]; registrations=[]
    for t in times[:C]:
        lower=max(0,int(np.searchsorted(stamps[:idx+1],t,side='right'))-1)
        upper=min(idx,lower+1)
        values=[]
        for j in (lower,upper):
            p,_=read_pcd(files[j],W); used_prefix.append(str(files[j]))
            query=anchor
            if tracking=='icp' and j!=idx:
                _,uniform=np.unique(np.floor(p/.006).astype(np.int64),axis=0,return_index=True)
                target=p[uniform]
                if len(target)>10000:target=target[np.linspace(0,len(target)-1,10000).astype(int)]
                query,registration=register_prefix(anchor,target)
                registrations.append({'cloud':str(files[j]),**registration})
            distance,nearest=cKDTree(p).query(query,k=8,workers=4)
            weight=1/np.maximum(distance,1e-4)**2; weight/=weight.sum(1,keepdims=True)
            estimated=(p[nearest]*weight[:,:,None]).sum(1)
            values.append(estimated); residuals.append(float(np.quantile(distance[:,0],.95)))
        alpha=0 if upper==lower else np.clip((t-stamps[lower])/(stamps[upper]-stamps[lower]),0,1)
        prefix.append((1-alpha)*values[0]+alpha*values[1])
    prefix=np.asarray(prefix); prefix[-1]=anchor
    smoothing_change_mm = 0.
    if motion_smoothing_radius > 0:
        filtered = smooth_prefix_motion(prefix, motion_smoothing_radius)
        smoothing_change_mm = float(np.sqrt(np.mean(np.sum((filtered-prefix)**2, axis=-1)))*1000)
        prefix = filtered
    # Fixed coordinate translation chosen from observed geometry; z/table calibration unchanged.
    origin=np.array([anchor[:,0].mean(),anchor[:,1].mean(),0.])
    prefix-=origin
    grippers=[]
    for side in ('left','right'):
        grippers+=piper_geometry(root,folder/f'joints/{side}_arm_joint_states_and_end_pose.csv',
                                times+action_delay,side)
    T=len(times); N=n_particles+2000; K=7
    positions=np.zeros((T,N,3),dtype=np.float32)
    positions[:C,:n_particles]=prefix
    positions[C:,:n_particles]=prefix[-1]  # future cloth is an inert placeholder
    poses=np.tile(np.eye(4,dtype=np.float32),(T,K,1,1))
    tmpl=np.zeros((K,400,3),np.float32); mask=np.zeros((K,400),bool)
    pb=np.zeros(N,np.int64); oid=np.zeros(N,np.int64); off=n_particles
    for j,(points,pose) in enumerate(grippers):
        pose=pose.copy();pose[:,:3,3]-=origin
        n=len(points); k=j+1
        positions[:,off:off+n]=np.einsum('tij,pj->tpi',pose[:,:3,:3],points)+pose[:,None,:3,3]
        poses[:,k]=pose;tmpl[k,:n]=points;mask[k,:n]=True
        pb[off:off+n]=k;oid[off:off+n]=1+j//3;off+=n
    positions=torch.from_numpy(positions)[None]
    actuator=torch.from_numpy(oid>0)[None]
    delta,normal_position,normal_pose=domain_normalizers(cfg.normalization,'cloth')
    sidecar={'body_poses':torch.from_numpy(poses)[None], 'body_templates':torch.from_numpy(tmpl)[None],
             'body_template_mask':torch.from_numpy(mask)[None], 'body_type':torch.tensor([[0,1,1,1,1,1,1]]),
             'particle_body':torch.from_numpy(pb)[None]}
    sidecar=_augment_pose_channels(sidecar,actuator,C,normal_pose)
    batch=TrajectoryBatch(positions=positions, actions=(positions[:,C:]-positions[:,C-1:-1])*actuator[:,None,:,None],
                          action_mask=actuator,object_ids=torch.from_numpy(oid)[None],valid_lengths=torch.tensor([N]),
                          E=torch.where(actuator,1e9,1e5),nu=torch.full((1,N),.3),rho=torch.full((1,N),300.),
                          mat_type=torch.where(actuator,4,5),context_len=C,delta_stats=delta,
                          position_stats=normal_position,domain='cloth',**sidecar)
    assert geometry_residual_mm(batch)<.002
    assert all(float(Path(f).name.split('_')[1])<=anchor_t for f in used_prefix)
    receipt={'capture':capture,'anchor_offset_seconds':anchor_offset,'anchor_timestamp':anchor_t,
             'requested_horizon_frames':requested_horizon,'actual_horizon_frames':horizon,
             'model_times':times.tolist(),'model_dt_seconds':dt,'actual_cloud_dt_median':float(np.median(np.diff(stamps))),
             'prefix_clouds':sorted(set(used_prefix)), 'prefix_nearest_surface_p95_max_mm':1000*max(residuals),
             'origin_world':origin.tolist(),'cloth_particles':n_particles,'actuator_particles':2000,
             'action_timestamp_offset_seconds':action_delay,'geometry_residual_mm':geometry_residual_mm(batch),
             'tracking_method':tracking,'prefix_registrations':registrations,
             'motion_smoothing_radius_m':motion_smoothing_radius,
             'history_motion_smoothing_change_rms_mm':smoothing_change_mm,
             'input_assumptions':['Approximate nearest-surface prefix tracking; no true particle correspondence',
                                 'Observed surface only; occluded cloth not reconstructed',
                                 'Known Piper gripper URDF; FK from recorded joint angles',
                                 'Cloth training material defaults E=1e5, nu=0.3, rho=300; not measured garment properties',
                                 'Recorded future actions given; world delta commands',
                                 'Camera and robot timestamps aligned directly; camera latency not calibrated',
                                 'Model cadence 30 Hz; prefix interpolated within observed interval only']}
    return batch,receipt,{'files':files,'stamps':stamps,'world_from_camera':W,'origin':origin,'rgb':rgb,'n_cloth':n_particles}


def measure(prediction, batch, metadata, receipt):
    times=np.array(receipt['model_times']); C=batch.context_len; n=metadata['n_cloth']
    values=[]; visual_clouds=[]; visual_colors=[]; eval_times=[]
    for path,stamp in zip(metadata['files'],metadata['stamps']):
        if stamp<=times[C-1] or stamp>times[-1]: continue
        frame=float((stamp-times[C-1])*30)
        low=int(np.floor(frame))+C-1; high=min(low+1,len(times)-1); alpha=frame-np.floor(frame)
        low=min(low,len(times)-1)
        pred=(1-alpha)*prediction[low,:n]+alpha*prediction[high,:n]
        real,rgb=read_pcd(path,metadata['world_from_camera']); real-=metadata['origin']
        distance=cKDTree(pred).query(real,workers=4)[0]
        reverse=cKDTree(real).query(pred,workers=4)[0]
        still=cKDTree(batch.anchor[0,:n].numpy()).query(real,workers=4)[0]
        values.append({'time_after_anchor_s':float(stamp-times[C-1]),
                       'measured_surface_to_prediction_mean_mm':float(distance.mean()*1000),
                       'prediction_to_measured_surface_mean_mm':float(reverse.mean()*1000),
                       'measured_surface_to_static_anchor_mean_mm':float(still.mean()*1000)})
        rows=np.linspace(0,len(real)-1,min(4000,len(real))).astype(int)
        visual_clouds.append(real[rows].astype(np.float32));visual_colors.append(rgb[rows].astype(np.float32));eval_times.append(stamp)
    return values,visual_clouds,visual_colors,eval_times


def render_real(out, key, prediction, batch, metadata, receipt, measurements, clouds, colors, eval_times):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter
    from scripts.dev.overnight_report import axes3d
    n=metadata['n_cloth']; C=batch.context_len; times=np.array(receipt['model_times'])
    lo=np.minimum(np.min([x.min(0) for x in clouds],axis=0),prediction[:,:n].min((0,1)))
    hi=np.maximum(np.max([x.max(0) for x in clouds],axis=0),prediction[:,:n].max((0,1)))
    fig=plt.figure(figsize=(13.5,5.5))
    writer=FFMpegWriter(fps=10,codec='libx264',extra_args=['-vf','pad=ceil(iw/2)*2:ceil(ih/2)*2','-pix_fmt','yuv420p','-crf','21'])
    with writer.saving(fig,str(out/(key+'.mp4')),dpi=110):
        display=list(range(len(clouds)))+[len(clouds)-1]*15
        for i in display:
            fig.clear(); f=min(len(prediction)-1,int(round((eval_times[i]-times[0])*30)))
            for panel in (1,2,3):
                ax=fig.add_subplot(1,3,panel,projection='3d');axes3d(ax,lo,hi)
                if panel in (1,3): ax.scatter(*clouds[i].T,c=colors[i],s=2,alpha=.65)
                if panel in (2,3): ax.scatter(*prediction[f,:n:2].T,c='#df8e2c',s=3,alpha=.8)
                if panel==2: ax.scatter(*prediction[f,n::4].T,c='#526681',s=3,alpha=.8)
                ax.set_title(('Real measured surface','Our predicted cloth + given grippers','Overlay')[panel-1],fontsize=11)
            fig.suptitle(f"RGBench real cloth | {receipt['capture']} | {key.split('__')[-1]}",fontsize=11)
            fig.text(.5,.018,f"{measurements[i]['time_after_anchor_s']:.2f} s after observed prefix | recorded actions given | depth points, not RGB video | approximate prefix tracks",ha='center',fontsize=8)
            writer.grab_frame()
            if i==len(clouds)//2: fig.savefig(out/(key+'.png'),dpi=140)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4),constrained_layout=True)
    ax.plot([v['time_after_anchor_s'] for v in measurements],[v['measured_surface_to_prediction_mean_mm'] for v in measurements],label='Our predicted surface')
    ax.plot([v['time_after_anchor_s'] for v in measurements],[v['measured_surface_to_static_anchor_mean_mm'] for v in measurements],label='No-motion anchor reference')
    ax.set(xlabel='Seconds after observed prefix',ylabel='Measured surface → candidate distance (mm)',title='Partial-surface distance; correspondence-free')
    ax.legend();ax.grid(alpha=.2);fig.savefig(out/(key+'_error.png'),dpi=150);plt.close(fig)


def report(out, cases):
    content=['<!doctype html><html><head><meta charset="utf-8"><title>Real cloth predictions</title>',
             '<style>body{font:17px/1.5 system-ui;max-width:1200px;margin:40px auto;padding:20px;color:#283746}video,img{max-width:100%}section{padding:24px;background:#f4f7fa;margin:24px 0}small{color:#617080}</style></head><body>',
             '<h1>Real cloth: observed surface to predicted motion</h1>',
             '<p>Green is measured RGBench depth geometry. Orange is our frozen simulation-trained model prediction. Robot actions are recorded and given. The model receives only a short cloth history; its rollout then feeds back its own predictions.</p>',
             '<p>This is a preliminary transfer diagnostic: partial surfaces, approximate nearest-surface tracking, training-default material values, and uncalibrated camera latency. It does not establish accurate full-state reconstruction or autonomous robot control.</p>']
    for case in cases:
        key=case['key'];content.append('<section><h2>'+html.escape(key)+'</h2>')
        if case['status']=='ok':
            content.append(f'<video controls loop muted src="{key}.mp4"></video><img src="{key}_error.png"><p><a href="{key}.json">Inputs, metrics and assumptions</a></p>')
        else: content.append('<pre>'+html.escape(case.get('error','failed'))+'</pre>')
        content.append('</section>')
    content.append('<small>Real observations and Piper geometry: <a href="https://rgbench.github.io/">RGBench, Hu et al., AAAI 2026</a>. Dataset CC BY 4.0. The source authors’ simulation videos are not our predictions.</small></body></html>')
    (out/'report.html').write_text('\n'.join(content))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--source',default='validation/public_real_cloth_2026-09-14')
    ap.add_argument('--prepare-only',action='store_true');ap.add_argument('--hours',type=float,default=4)
    ap.add_argument('--horizon',type=int,default=96)
    ap.add_argument('--phase',choices=['early','annotated'],default='early')
    ap.add_argument('--tracking',choices=['nearest','icp'],default='nearest')
    a=ap.parse_args();torch.set_num_threads(8)
    root=Path(a.source);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    from utils.checkpoint import checkpoint_config, read_checkpoint
    cfg=checkpoint_config(read_checkpoint(a.checkpoint))
    if cfg.model.get('articulated_bodies',False):raise ValueError('This adapter currently supplies free link poses')
    require_given_actions(cfg)
    specs=[('green_tshirt_grasp_2025-07-19-19-14-58',2.,a.horizon),
           ('green_tshirt_grasp_2025-07-19-19-14-03',2.,a.horizon),
           ('green_tshirt_fold_2025-07-28-18-21-20',1.,a.horizon)]
    if a.phase=='annotated':
        specs=[('green_tshirt_grasp_2025-07-19-19-14-58',4.,a.horizon),
               ('green_tshirt_grasp_2025-07-19-19-14-03',3.8,a.horizon),
               ('green_tshirt_fold_2025-07-28-18-21-20',5.4,a.horizon)]
    prepared=[];cases=[];start=time.time()
    write_frozen_run(out/'run.json',{'checkpoint':str(Path(a.checkpoint).resolve()),'checkpoint_sha256':file_sha256(a.checkpoint),
                              'config':config_for_json(cfg),'code':code_revision(),'specs':specs,
                              'sampler_seeds':[710001,710002,710003],'start_unix':start,
                              'phase':a.phase,'phase_selection':'Fixed early offsets or authors experiment_library.yaml start_calculate_time, relative to first released cloud',
                              'tracking':a.tracking,
                              'source_manifest_sha256':file_sha256(root/'selected_captures.json'),
                              'gripper_asset_sha256':{str(p.relative_to(root)):file_sha256(p) for p in (root/'piper').rglob('*') if p.is_file()}},
                     ('specs','sampler_seeds','phase','tracking','source_manifest_sha256','gripper_asset_sha256'))
    for capture,offset,horizon in specs:
        try:
            batch,receipt,metadata=build_input(root,capture,cfg,anchor_offset=offset,horizon=horizon,tracking=a.tracking)
            torch.save(batch,out/(capture+'_input.pt'));write_json(out/(capture+'_input.json'),receipt)
            prepared.append((capture,batch,receipt,metadata))
            print('PREPARED',capture,json.dumps({'geometry_residual_mm':receipt['geometry_residual_mm'],
                                                'prefix_p95_mm':receipt['prefix_nearest_surface_p95_max_mm']}),flush=True)
        except Exception:
            error=traceback.format_exc();cases.append({'key':capture,'status':'failed','error':error});print(error,flush=True)
    write_json(out/'preparation.json',{'prepared':[x[0] for x in prepared],'failures':cases})
    if a.prepare_only:return
    if not prepared:
        report(out,cases);write_json(out/'status.json',{'state':'failed','cases':cases});return
    predictor=Predictor(a.checkpoint)
    for capture,batch,receipt,metadata in prepared:
        for seed in (710001,710002,710003):
            if time.time()-start > a.hours*3600:break
            key=capture+'__seed'+str(seed)
            existing=out/(key+'.json')
            if existing.exists() and (out/(key+'.mp4')).exists() and (out/(key+'.mp4')).stat().st_size>1000:
                cases.append({'key':key,'status':'ok'});continue
            try:
                b=batch.to(predictor.device);seed_for_device(seed,predictor.device)
                predictor.calls=0;torch.cuda.synchronize();begun=time.monotonic()
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    rollout=windowed_rollout(predictor.model,predictor.diffusion,b,horizon=b.num_future_frames,
                                             future_len=int(cfg.data.future_len),num_inference_steps=predictor.steps,
                                             schedule=cfg.get('df_schedule','pyramid'),
                                             uncertainty_scale=float(cfg.get('df_uncertainty_scale',1.)),
                                             clamp_actuated=True)
                torch.cuda.synchronize();elapsed=time.monotonic()-begun
                pred=rollout['positions'][0].float().cpu().numpy()
                if not np.isfinite(pred).all():raise ValueError('Non-finite real-cloth rollout')
                measurements,clouds,colors,eval_times=measure(pred,batch,metadata,receipt)
                np.savez_compressed(out/(key+'.npz'),prediction=pred,context=batch.context[0].numpy(),
                                    object_ids=batch.object_ids[0].numpy(),model_times=receipt['model_times'])
                write_json(existing,{'input':receipt,'seed':seed,'seconds':elapsed,'model_calls':predictor.calls,
                                     'windows':rollout['windows'],'clamp_given_actuators':True,'metrics':measurements,
                                     'metric_note':'Observed surface to prediction avoids assuming hidden surfaces were observed; reverse distance is coverage-dependent',
                                     'evidence':'Preliminary sim-to-real diagnostic, approximate observed-surface tracks'})
                print('PREDICTED',key,elapsed,'seconds',flush=True)
                render_real(out,key,pred,batch,metadata,receipt,measurements,clouds,colors,eval_times)
                cases.append({'key':key,'status':'ok'})
            except Exception:
                error=traceback.format_exc();cases.append({'key':key,'status':'failed','error':error});print(error,flush=True)
                write_json(out/(key+'_failure.json'),cases[-1]);torch.cuda.empty_cache()
            report(out,cases);write_json(out/'status.json',{'state':'running','cases':cases,'elapsed_seconds':time.time()-start})
    report(out,cases);write_json(out/'status.json',{'state':'finished','cases':cases,'elapsed_seconds':time.time()-start})


if __name__=='__main__':main()
