"""Optional obstacle comparisons after a primary guidance GPU lane finishes.

No primary worker is interrupted. Fixtures depend only on observations, supplied
geometry and task requests. The existing full-objective plan is re-evaluated in
the same obstacle world as the new avoidance plan, without re-optimizing it.
"""
import argparse
import copy
from datetime import datetime
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts.dev.guidance_study import (REPO, PYTHON, read, write, sha, now, command,
    execute, audit_one, visualize_one)


MINIMUM_SECONDS = 2400  # 30-minute optimization cap, ten minutes for both audits/GIFs.


def find_fixture(geometry, anchor_R, anchor_T, target, target_R, vendors=()):
    """First clear-endpoint box that blocks a nominal straight object transport.

    This is a deterministic fixture rule, not a reachability or grasp guarantee.
    The pick endpoint uses the observed orientation because no rotation is asked.
    """
    import numpy as np
    import torch
    from guidance.obstacles import StaticObstacleDistance, interpolate_poses
    start=anchor_T[0];delta=np.asarray(target)-start
    horizontal=delta[:2];length=np.linalg.norm(horizontal)
    if length<.01:return None,dict(reason='Requested lateral travel is below 10 mm')
    normal=horizontal/length;tangent=np.array([-normal[1],normal[0]])
    rotation=np.eye(3);rotation[:2,0]=normal;rotation[:2,1]=tangent
    goal_R=anchor_R[0] if target_R is None else np.asarray(target_R)
    nominal_R,nominal_T=interpolate_poses(
        torch.tensor(np.stack([anchor_R[0],goal_R]))[None,:,None],
        torch.tensor(np.stack([start,target]))[None,:,None],20)
    nominal_R=nominal_R[0,:,0].numpy();nominal_T=nominal_T[0,:,0].numpy()
    counts=dict(tested=0,anchor_clear=0,endpoints_clear=0)
    keys=tuple(sorted(geometry['bodies']))
    for height in (.14,.10,.075,.05,.025):
        box=dict(name='tabletop_barrier',size=[.025,.09,height],centre=[0.,0.,0.],rotation=rotation.tolist())
        spec=dict(version=1,units='metres',boxes=[box])
        queries=StaticObstacleDistance(geometry,spec,vendor_paths=vendors).queries
        def distance(body,R,T,centre):
            return float(queries[keys.index(body)].query(np.stack([R,rotation]),np.stack([T,centre]))[0][0])
        for fraction in (.5,.35,.65,.2,.8):
            for offset in (0.,.03,-.03,.06,-.06):
                counts['tested']+=1
                centre=start+fraction*delta
                centre[:2]+=offset*tangent
                centre[2]=float(geometry['metadata']['support_z'])+height/2
                initial=[distance(k,anchor_R[k],anchor_T[k],centre) for k in keys]
                if min(initial)<.01:continue
                counts['anchor_clear']+=1
                endpoint=distance(0,goal_R,np.asarray(target),centre)
                if endpoint<.01:continue
                counts['endpoints_clear']+=1
                nominal=[distance(0,r,t,centre) for r,t in zip(nominal_R,nominal_T)]
                if min(nominal)>-.001:continue
                box['centre']=centre.tolist()
                return spec,dict(**counts,fraction=fraction,lateral_offset_m=offset,
                    initial_clearance_by_body_mm={str(k):d*1000 for k,d in zip(keys,initial)},
                    target_object_clearance_mm=endpoint*1000,nominal_min_clearance_mm=min(nominal)*1000,
                    target_orientation='requested' if target_R is not None else 'observed (unconstrained pick goal)',
                    reachability_verified=False)
    return None,dict(**counts,reason='No box in the fixed candidate grid met endpoint-clearance and blocking criteria')


def prepare(root):
    import torch
    parent=read(root/'plan.json');folder=root/'obstacles'
    if (folder/'plan.json').exists():raise FileExistsError(folder/'plan.json')
    scenes=[]
    for source in parent['scenes']:
        directory=Path(source['directory'])
        for name,expected in source['hashes'].items():
            if sha(directory/name)!=expected:raise ValueError('Changed source fixture')
        batch=torch.load(directory/'input.pt',map_location='cpu',weights_only=False)
        geometry=torch.load(directory/'collision_geometry.pt',map_location='cpu',weights_only=False)
        anchor=batch.body_poses[0,batch.context_len-1].double().numpy()
        spec,receipt=find_fixture(geometry,anchor[:,:3,:3],anchor[:,:3,3],source['target'],source['target_R'],parent['vendors'])
        scene=copy.deepcopy(source);scene['obstacle_fixture']=receipt
        scene['obstacle_state']='ready' if spec else 'not eligible: no blocking fixture'
        if spec:
            path=folder/'fixtures'/source['name']/ 'obstacles.json';write(path,spec)
            scene.update(obstacles=str(path),obstacles_sha256=sha(path))
        scenes.append(scene)
    plan=dict(parent,scenes=scenes,source=str(folder/'source'),baseline_root=str(root),reference_root=str(root),
        arms=dict(reference=dict(label='Full objective without avoidance, evaluated with barrier',terms=None,source_arm='full'),
            avoid=dict(label='Full objective + obstacle avoidance',terms=None,obstacles=True)),
        parent_plan_sha256=sha(root/'plan.json'),deadline_utc=read(root/'launch.json')['deadline_utc'],
        minimum_seconds=MINIMUM_SECONDS,priority='Only after a complete primary GPU lane; never extend the original deadline',
        fixture_rule='Fixed candidate grid; all observed bodies and requested object endpoint clear by 10 mm; '+
            'nominal interpolated object transport intersects by at least 1 mm. No generated or recorded future used.',
        source_receipt=str(folder/'source_receipt.json'))
    write(folder/'plan.json',plan)
    write(folder/'status.json',dict(state='waiting for primary experiments',created_utc=now(),entries=[]))


def second_barrier_fixture(geometry,anchor_R,anchor_T,target,target_R,first_spec,vendors=()):
    """Keep the first box and add a separated box using observations and the goal."""
    import numpy as np
    import torch
    from guidance.obstacles import StaticObstacleDistance,box_geometry,interpolate_poses,validate_obstacles
    first_spec=validate_obstacles(first_spec)
    if len(first_spec['boxes'])!=1:raise ValueError('Expected the original one-barrier fixture')
    first=first_spec['boxes'][0];start=anchor_T[0];delta=np.asarray(target)-start
    length=np.linalg.norm(delta[:2])
    if length<.01:raise ValueError('Requested lateral travel is below 10 mm')
    normal=delta[:2]/length;tangent=np.array([-normal[1],normal[0]])
    rotation=np.asarray(first['rotation']);goal_R=anchor_R[0] if target_R is None else np.asarray(target_R)
    R,T=interpolate_poses(torch.tensor(np.stack([anchor_R[0],goal_R]))[None,:,None],
        torch.tensor(np.stack([start,target]))[None,:,None],40)
    keys=tuple(sorted(geometry['bodies']));tested=0
    for height in (first['size'][2],.10,.075,.05):
        box=dict(name='second_barrier',size=[.025,.09,height],centre=[0.,0.,0.],rotation=rotation.tolist())
        spec=dict(version=1,units='metres',boxes=[box])
        queries=StaticObstacleDistance(geometry,spec,vendor_paths=vendors).queries
        pair=StaticObstacleDistance(dict(bodies={0:box_geometry(first['size'])}),spec,vendor_paths=vendors).queries[0]
        def distance(body,r,t,centre):
            return float(queries[keys.index(body)].query(np.stack([r,rotation]),np.stack([t,centre]))[0][0])
        for fraction in (.8,.75,.25,.2,.65,.35):
            for offset in (.045,-.045,0.,.075,-.075):
                tested+=1;centre=start+fraction*delta;centre[:2]+=offset*tangent
                centre[2]=float(geometry['metadata']['support_z'])+height/2
                separation=float(pair.query(np.stack([np.asarray(first['rotation']),rotation]),
                    np.stack([np.asarray(first['centre']),centre]))[0][0])
                if separation<.01:continue
                initial=[distance(k,anchor_R[k],anchor_T[k],centre) for k in keys]
                endpoint=distance(0,goal_R,np.asarray(target),centre)
                if min(initial)<.01 or endpoint<.01:continue
                nominal=[distance(0,r.numpy(),t.numpy(),centre) for r,t in zip(R[0,:,0],T[0,:,0])]
                if min(nominal)>-.001:continue
                box['centre']=centre.tolist()
                return dict(version=1,units='metres',boxes=[copy.deepcopy(first),box]),dict(
                    tested=tested,fraction=fraction,lateral_offset_m=offset,
                    initial_clearance_by_body_mm={str(k):d*1000 for k,d in zip(keys,initial)},
                    target_object_clearance_mm=endpoint*1000,between_obstacles_clearance_mm=separation*1000,
                    nominal_second_barrier_clearance_mm=min(nominal)*1000,first_barrier_unchanged=True,
                    reachability_verified=False,selection='Fixed candidate order using only observed poses, geometry, and requested goal.')
    raise ValueError('No separated second barrier meets the endpoint-clearance and nominal-blocking criteria')


def available_lane(root,plan,remaining,*,only_gpu=None):
    """Only release a lane when every originally planned run has a completion receipt."""
    if remaining<MINIMUM_SECONDS:return None
    for gpu in ((only_gpu,) if only_gpu is not None else (0,1)):
        path=root/f'gpu{gpu}.json'
        if not path.exists() or read(path)['state']!='completed':continue
        paths=[root/'runs'/s['name']/a for s in plan['scenes'] for a in plan['lanes'][str(gpu)]]
        if all((p/'results.json').exists() and p.with_suffix('.status.json').exists()
               and read(p.with_suffix('.status.json'))['state']=='completed' for p in paths):return gpu
    return None


def run_scene(root,name,gpu):
    """Runs inside the existing Slurm allocation, on a fully released lane."""
    folder=root/'obstacles';plan=read(folder/'plan.json')
    parent=read(root/'plan.json');deadline=datetime.fromisoformat(plan['deadline_utc']).timestamp()
    if available_lane(root,parent,deadline-time.time(),only_gpu=gpu)!=gpu:
        raise RuntimeError('Primary lane unavailable or insufficient remaining budget')
    if sha(root/'plan.json')!=plan['parent_plan_sha256']:raise ValueError('Primary plan changed')
    if read(root/'qualification.json')['status']!='passed':raise RuntimeError('Primary qualification did not pass')
    scene=next(s for s in plan['scenes'] if s['name']==name)
    if sha(scene['obstacles'])!=scene['obstacles_sha256']:raise ValueError('Obstacle fixture changed')
    for file,expected in scene['hashes'].items():
        if sha(Path(scene['directory'])/file)!=expected:raise ValueError('Scene fixture changed')
    if sha(plan['controls'])!=plan['controls_sha256']:raise ValueError('Controller qualification changed')
    os.environ['CUDA_VISIBLE_DEVICES']=plan['gpus'][gpu]
    for path in plan['replay_vendors']:sys.path.append(path)
    reference=root/'runs'/name/'goal'/f'seed_{scene["seed"]}.pt'
    original=root/'runs'/name/'full'
    if not (original/'results.json').exists():raise RuntimeError('Full-objective reference unavailable')
    out=folder/'runs'/name/'avoid'
    result=execute(plan,command(plan,scene,'avoid',out,reference=reference),out,deadline-600,timeout=1800)
    if result['state']!='completed':raise RuntimeError('Obstacle optimization failed; inspect its retained log')
    for arm in ('reference','avoid'):
        audit_one(folder,plan,scene,arm);visualize_one(folder,plan,scene,arm)
    # These are the same physical world and optimizer starting sample. Changing
    # the cost is the only experimental intervention; the reference is not rerun.
    protocols=[read(folder/'artifacts'/name/a/'plan/protocol.json') for a in ('reference','avoid')]
    for key in ('input_sha256','geometry_sha256','grasp_sha256','checkpoint_sha256','obstacles_sha256',
                'target','target_rotation','sampling'):
        if protocols[0].get(key)!=protocols[1].get(key):raise AssertionError(f'Paired comparison mismatch: {key}')
    write(folder/'artifacts'/name/'pair.json',dict(state='completed',utc=now(),
        original_full_run=str(original),original_full_trajectory_sha256=sha(original/f'seed_{scene["seed"]}.pt'),
        obstacle_sha256=scene['obstacles_sha256'],same_initial_noise_and_routing=True,
        original_reference_reoptimized=False,identical_obstacle_in_both_physics_replays=True))


def wait_and_run(root):
    import fcntl
    with (root/'obstacles/worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        _wait_and_run(root)


def _wait_and_run(root):
    folder=root/'obstacles';plan=read(folder/'plan.json');parent=read(root/'plan.json')
    state=read(folder/'status.json')
    if state['state']!='waiting for primary experiments':raise RuntimeError('Optional worker was already started')
    deadline=datetime.fromisoformat(plan['deadline_utc']).timestamp()
    state.update(watcher_pid=os.getpid(),watcher_started_utc=now());write(folder/'status.json',state)
    try:
        gpu=None
        while deadline-time.time()>=MINIMUM_SECONDS:
            gpu=available_lane(root,parent,deadline-time.time())
            if gpu is not None:break
            time.sleep(30)
        if gpu is None:
            state.update(state='not run: insufficient time',reason='No primary lane finished with the required 40-minute reserve')
            return
        state.update(state='running',gpu=gpu);write(folder/'status.json',state)
        for scene in plan['scenes']:
            entry=dict(scene=scene['name'],state=scene['obstacle_state']);state['entries'].append(entry)
            if scene['obstacle_state']!='ready':continue
            if deadline-time.time()<MINIMUM_SECONDS:
                entry['state']='not run: insufficient time';continue
            if not (root/'runs'/scene['name']/'full/results.json').exists():
                entry['state']='not run: full-objective reference unavailable';continue
            entry.update(state='running',started_utc=now());write(folder/'status.json',state)
            cmd=['srun',f'--jobid={parent["allocation"]}','--overlap','--ntasks=1','--gres=gpu:2',
                '--time='+str(max(1,math.floor((deadline-time.time()-30)/60))),PYTHON,
                str(Path(plan['source'])/'scripts/dev/guidance_obstacle_study.py'),'scene',
                '--root',str(root),'--scene',scene['name'],'--gpu',str(gpu)]
            log=folder/'logs'/(scene['name']+'.log');log.parent.mkdir(parents=True,exist_ok=True)
            with log.open('x') as stream:
                result=subprocess.run(cmd,cwd=plan['source'],stdout=stream,stderr=subprocess.STDOUT,
                    env=dict(os.environ,PYTHONPATH=plan['source'],OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',WANDB_MODE='disabled'))
            entry.update(state='completed' if result.returncode==0 else 'failed',returncode=result.returncode,finished_utc=now())
            write(folder/'status.json',state)
            if result.returncode:raise RuntimeError('Optional obstacle trial failed; further optional trials stopped')
        state['state']='completed'
    except Exception as error:
        state.update(state='failed',error=repr(error));raise
    finally:
        state['updated_utc']=now();write(folder/'status.json',state)


def report_lines(root,report_path):
    folder=root/'obstacles';plan=read(folder/'plan.json');state=read(folder/'status.json')
    prefix=os.path.relpath(folder,report_path.parent)
    entries={e['scene']:e for e in state.get('entries',[])}
    lines=['## Optional obstacle comparison','',f'Status: **{state["state"]}**. '+
        'Runs only after a GPU completes its original lane, with at least 40 minutes left. The original deadline is unchanged.','',
        'Same scenes, targets, seed, 68 frames and 20 updates: existing full guidance versus full guidance plus '+
        'a 5 mm obstacle-clearance cost (weight 5). The obstacle is supplied to the objective, not observed by the checkpoint. '+
        'Both plans are evaluated and replayed with the identical barrier. The original full-guidance trajectory is reused unchanged.','',
        'Fixtures use only observed poses, geometry and requested endpoints. They leave 10 mm initial/endpoint clearance '+
        'and obstruct a nominal straight object transport; this does not prove task reachability. '+
        'Scenes without a qualifying fixture remain listed. All bodies are checked, with three objective probes and ten audit probes per transition; '+
        'these are sampled geometry checks, not continuous collision guarantees. Simulator contact and finger-limit failures remain visible.','',
        '| Scene | State |','|---|---|']
    for scene in plan['scenes']:
        entry=entries.get(scene['name'],{})
        status=entry.get('state',scene['obstacle_state'] if scene['obstacle_state']!='ready' else
            ('not run: insufficient time' if state['state']=='not run: insufficient time' else 'waiting'))
        lines.append(f'| {scene["name"]} | {status} |')
    lines+=['',f'[Frozen obstacle fixtures and settings]({prefix}/plan.json) · [Queue status]({prefix}/status.json)','']
    for scene in plan['scenes']:
        for arm in plan['arms']:
            artifact=folder/'artifacts'/scene['name']/arm
            if not (artifact/'visualization.json').exists():continue
            audit=read(artifact/'audit.json');metrics=audit['model_metrics'];visual=read(artifact/'visualization.json')
            clearance=metrics['validation']['obstacle_min_clearance_mm']
            contact=sum(r['forces']['obstacle_contact_solver_steps'] for r in audit['replays'])
            lines+=['<details>',f'<summary>{scene["name"]} — {plan["arms"][arm]["label"]}</summary>','',
                f'Model held goal error: {metrics["held_worst_goal_mm"]:.1f} mm; minimum obstacle clearance: {clearance:.1f} mm. '+
                f'Replay steps with obstacle contact (both time steps): {contact}. '+
                f'Qualified execution pass: {audit["qualified_task_pass"]}. Peak finger overtravel: {visual["replay_finger_limit_peak_mm"]:.1f} mm.','',
                'Left: the same saved original unconditional sample. Middle: the indicated guided plan. '+
                'Right: replay of that plan with the same barrier. Purple points show the obstacle.','',
                f'![{scene["name"]} {arm}]({prefix}/artifacts/{scene["name"]}/{arm}/comparison_points.gif)','', '</details>','']
    return lines


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['prepare','wait','scene'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--scene');p.add_argument('--gpu',type=int,choices=[0,1])
    a=p.parse_args();root=a.root.resolve()
    if a.mode=='prepare':prepare(root)
    elif a.mode=='wait':wait_and_run(root)
    else:
        if a.scene is None or a.gpu is None:p.error('scene requires --scene and --gpu')
        run_scene(root,a.scene,a.gpu)


if __name__=='__main__':main()
