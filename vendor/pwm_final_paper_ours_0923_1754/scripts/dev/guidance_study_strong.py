"""Quality-first continuation: reproduce references, establish baselines, then ablate.

The old 20-update study stays intact. Within each scene, every comparison uses
the selected baseline's seed, horizon, update budget and goal weight. Only the
declared objective family or optimizer/sampler variant changes.
"""
import argparse
import copy
from datetime import datetime,timezone,timedelta
import fcntl
import json
import os
from pathlib import Path
import signal
import sys
import time

from scripts.dev.guidance_study import (REPO,PYTHON,read,write,sha,now,command,execute,
    audit_one,visualize_one)

TERMINAL=('failed','completed','needs investigation','budget exhausted')

def accepted_baseline(metrics):
    """A prerequisite for useful ablations, not a contact-force/execution certificate."""
    residual=metrics['validation']['residuals_m']
    return bool(metrics['held_worst_goal_mm']<=20 and metrics['peak_overlap_mm']<=2
        and all(residual.get(k,float('inf'))<=0 for k in ('support','gripper_support','opening','speed')))


def local_plan(plan,scene,*,iterations=None,goal_weight=None,baseline_arm=None):
    return dict(plan,horizon=scene['horizon'],iterations=iterations or scene.get('effective_iterations',64),
        goal_weight=goal_weight or scene.get('effective_goal_weight',1.),
        baseline_arm=baseline_arm or 'full',save_iterations=','.join(str(n) for n in (20,40,64,128)
            if n<=(iterations or scene.get('effective_iterations',64))))


def obstacle_plan(root,plan,task):
    scene=task['scene']
    if sha(scene['obstacles'])!=scene['obstacles_sha256']:raise ValueError('Obstacle fixture changed')
    cfg=local_plan(plan,scene,iterations=task.get('iterations'),goal_weight=task.get('goal_weight'))
    reference=task.get('reference_arm','obstacle_reference');count=len(read(scene['obstacles'])['boxes'])
    cfg['reference_root']=str(root)
    cfg['arms']=dict(cfg['arms'],**{
        task['arm']:dict(label=f'Full objective + avoidance ({count} obstacles)',terms=None,obstacles=True),
        reference:dict(label=f'Existing full plan with {count} obstacles; avoidance off',terms=None,source_arm='full')})
    return cfg


def prepare(parent,hours):
    root=parent/'upgrade64';old=read(parent/'plan.json')
    if (root/'plan.json').exists():raise FileExistsError(root/'plan.json')
    plan=copy.deepcopy(old)
    plan.update(version=2,created_utc=now(),iterations=64,seed=8311,goal_reduction='mean',goal_weight=1.,
        baseline_arm='full',source=str(root/'source'),source_receipt=str(root/'source_receipt.json'),
        parent_root=str(parent),parent_plan_sha256=sha(parent/'plan.json'),
        report=str(REPO/'docs/guidance/README.md'),hours=hours,
        gate='Held position <=20 mm; peak convex overlap <=2 mm; support and joint-limit/speed screens pass. '+
            'Contact/slip/orientation/full task and independent execution are still reported separately.',
        repair_policy='If baseline64 fails the gate, try 128 updates at weight 1, then 128 at position weight 3. '+
            'First passing recipe is shared by every arm for that scene. All attempts are retained; '+
            'if none passes, that scene remains visible and its ablations wait for investigation.',
        phases=['two historical references','64-update baselines on all ten scenes','bounded baseline repair','matched ablations and speed variants'],
        priority='Baselines before comparisons; optional obstacles only after the main matrix.',
        old_comparison='20-update results are a separate historical experiment; no pooled comparison.')
    plan.pop('baseline_root',None)
    for scene in plan['scenes']:
        scene.update(seed=8311,horizon=68 if scene['task']=='pick' else 36)
    for name in ('fast10','cosine','coarse4','coarse_residual'):
        plan['arms'][name]['terms']=None
        plan['arms'][name]['label']=plan['arms'][name]['label'].replace('Goal only:','Full objective:')
    plan['lanes']={'0':['fast10','coarse_residual','coarse4','cosine'],'1':['goal','geometry','contact']}
    campaign=REPO/'validation/guidance_push_2026-09-15'
    refs=[]
    for task,name,scene_name in (
        ('pick','bottle_8270_left_long_h68_i64_r050','pick_reserved_00_asset_8270'),
        ('place','teacup_7803_left_i64_r050','place_development_03_asset_7803')):
        source=campaign/name;protocol=read(source/'protocol.json')
        scene=copy.deepcopy(next(s for s in plan['scenes'] if s['name']==scene_name))
        scene.update(target=protocol['target'][0],target_R=protocol['target_rotation'][0]
            if protocol.get('target_rotation') is not None else None)
        if task=='place':scene['name']='reference_teacup_7803'
        for key,field in [('input.pt','input_sha256'),('collision_geometry.pt','geometry_sha256'),('calibration.pt','grasp_sha256')]:
            if scene['hashes'][key]!=protocol[field]:raise ValueError('Historical reference fixture mismatch')
        if protocol['checkpoint_sha256']!=plan['checkpoint_sha256']:raise ValueError('Historical checkpoint mismatch')
        refs.append(dict(scene=scene,source=str(source),initial=str(source/'seed_8311.pt'),
            protocol_sha256=sha(source/'protocol.json'),initial_sha256=sha(source/'seed_8311.pt'),
            historical_metrics=read(campaign/'exports'/name/'receipt.json')['metrics']))
    plan['references']=refs
    tasks=[]
    def add(scene,arm,phase,**extra):
        tasks.append(dict(id=f'{scene["name"]}__{arm}',scene=scene,arm=arm,phase=phase,state='pending',**extra))
    for ref in refs:add(ref['scene'],'full64',0,reference=ref)
    for scene in plan['scenes']:
        if scene['name']!=refs[0]['scene']['name']:add(scene,'full64',1)
    write(root/'plan.json',plan)
    write(root/'queue.json',dict(state='running',tasks=tasks,selected={},gate_failures=[],started_utc=now()))
    write(root/'budget.json',dict(deadline_utc=(datetime.now(timezone.utc)+timedelta(hours=hours)).isoformat(),
        authorized_extension='User: we can extend the deadline, make sure the results are actually what we want though',hours=hours))


def task_cost_seconds(task):
    # Conservative reservations based on the current deterministic 20-update runs.
    factor=task.get('iterations',64)/64
    obstacle_factor=1.5 if task['phase']==4 else 1.
    return (3600 if task['scene']['horizon']==68 else 2200)*factor*obstacle_factor+600


def prioritize_obstacle_pilot(root,names,gpu):
    """Record an approved queue amendment without changing the frozen experiment plan."""
    plan=read(root/'plan.json')
    fixtures={s['name']:s for s in read(Path(plan['parent_root'])/'obstacles/plan.json')['scenes']}
    with (root/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        queue=read(root/'queue.json')
        if queue['state']!='running':raise ValueError('Study is not running')
        if (root/'obstacle_pilot.json').exists():raise FileExistsError('Pilot already prepared')
        if len(names)!=len(set(names)) or not names:raise ValueError('Pilot scenes must be unique')
        tasks=[]
        for name in names:
            scene=next(s for s in plan['scenes'] if s['name']==name)
            selected=queue['selected'][name];fixture=fixtures[name]
            metrics=read(root/'artifacts'/name/selected['arm']/'audit.json')['model_metrics']
            if not metrics['geometry_pass']:raise ValueError('Pilot requires passing full model checks')
            if fixture['obstacle_state']!='ready' or sha(fixture['obstacles'])!=fixture['obstacles_sha256']:
                raise ValueError('Prepared obstacle is unavailable or changed')
            if any(t['id']==f'{name}__avoid' for t in queue['tasks']):raise ValueError('Avoidance task already exists')
            effective=dict(scene,effective_iterations=selected['iterations'],effective_goal_weight=selected['goal_weight'],
                obstacles=fixture['obstacles'],obstacles_sha256=fixture['obstacles_sha256'])
            tasks.append(dict(id=f'{name}__avoid',scene=effective,arm='avoid',phase=4,state='pending',
                iterations=selected['iterations'],goal_weight=selected['goal_weight'],priority=-1,assigned_gpu=gpu))
        write(root/'obstacle_pilot_before_queue.json',queue)
        write(root/'obstacle_pilot.json',dict(created_utc=now(),scenes=names,gpu=gpu,
            authorization='User approved the two-scene obstacle pilot before remaining optimizer variants.',
            parent_plan_sha256=sha(root/'plan.json'),tasks=tasks,
            comparison='Same initial noise, target, sampler and selected baseline budget; add only the obstacle cost. '+
                'Re-evaluate and replay the unchanged baseline with the identical obstacle.',
            settings=dict(weight=5.,clearance_mm=5.,optimization_substeps=3,audit_substeps=10),
            limitation='The checkpoint does not observe the barrier. Interpolated geometric checks are not continuous collision detection.'))
        queue['tasks'].extend(tasks);queue['updated_utc']=now();write(root/'queue.json',queue)


def claim(root,gpu):
    with (root/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        queue=read(root/'queue.json');plan=read(root/'plan.json')
        if queue['state'] in TERMINAL:return None
        unfinished=[x for x in queue['tasks'] if x['state'] in ('pending','running')]
        if not unfinished:
            queue['state']='needs investigation' if queue['gate_failures'] else 'budget exhausted'
            write(root/'queue.json',queue);return None
        lane_tasks=[x for x in unfinished if x.get('assigned_gpu',gpu)==gpu]
        if not lane_tasks:return None
        priority,phase=min((x.get('priority',0),x['phase']) for x in lane_tasks)
        remaining=datetime.fromisoformat(read(root/'budget.json')['deadline_utc']).timestamp()-time.time()
        for task in unfinished:
            if task['state']!='pending' or (task.get('priority',0),task['phase'])!=(priority,phase):continue
            if task.get('assigned_gpu',gpu)!=gpu:continue
            if phase==3 and task['arm'] not in plan['lanes'][str(gpu)]:continue
            if remaining<task_cost_seconds(task):
                task.update(state='not run: deadline reserve',updated_utc=now());continue
            task.update(state='running',gpu=gpu,started_utc=now())
            write(root/'queue.json',queue);return copy.deepcopy(task)
        write(root/'queue.json',queue)
        return None


def finish_handoff(root,gpu):
    parent=Path(read(root/'plan.json')['parent_root'])
    row=next(x for x in read(root/'handoff.json')['workers'] if x['gpu']==gpu)
    proc=Path('/proc')/str(row['child_pid']);driver=Path('/proc')/str(row['driver_pid'])
    while proc.exists():
        stat=proc.joinpath('stat').read_text().rsplit(')',1)[1].split()
        if stat[0]=='Z':
            if int(stat[49])!=0:raise RuntimeError('Last historical trial exited unsuccessfully')
            break
        time.sleep(10)
    out=Path(row['directory'])
    if not (out/'results.json').exists() or not (out/'seed_9515.pt').exists():
        raise RuntimeError('Historical trial did not save complete outputs')
    status_file=parent/f'gpu{gpu}.json';status=read(status_file)
    write(root/f'previous_gpu{gpu}_status.json',status)
    if driver.exists():
        args=driver.joinpath('cmdline').read_bytes().split(b'\0')
        if [v.decode() for v in args if v]!=row['driver_command']:raise RuntimeError('Driver PID identity changed')
        os.kill(row['driver_pid'],signal.SIGTERM);os.kill(row['driver_pid'],signal.SIGCONT)
    trial=read(out.with_suffix('.status.json'))
    trial.update(state='completed',returncode=0,finished_utc=now(),
        handoff_note='Completion confirmed from child exit and saved outputs while old queue was paused; finish timestamp is observation time')
    write(out.with_suffix('.status.json'),trial)
    for entry in status['entries']:
        if entry.get('directory')==str(out):entry.update(trial)
    status.update(state='superseded by stronger recipe',updated_utc=now())
    write(status_file,status)
    old_plan=read(parent/'plan.json')
    scene=next(s for s in old_plan['scenes'] if s['name']==out.parent.name)
    audit_one(parent,old_plan,scene,out.name);visualize_one(parent,old_plan,scene,out.name)
    from scripts.dev.guidance_study import report as old_report
    with (root/'handoff_report.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        old_report(parent,output=parent/'REPORT_20_UPDATES.md')
    write(root/f'handoff_gpu{gpu}.json',dict(state='completed',utc=now(),last_trial=str(out)))


def reference_check(root,task,result):
    import torch
    ref=task['reference'];path=Path(ref['initial'])
    if sha(path)!=ref['initial_sha256'] or sha(path.parent/'protocol.json')!=ref['protocol_sha256']:
        raise ValueError('Historical reference changed')
    old=torch.load(path,map_location='cpu',weights_only=False)['initial']
    current=torch.load(result/'seed_8311.pt',map_location='cpu',weights_only=False)['initial']
    errors={key:float((old[key]-current[key]).abs().max()) for key in ('positions','R','T')}
    if max(errors.values())>1e-5:raise AssertionError(f'Historical initial-trajectory mismatch: {errors}')
    a=read(root/'artifacts'/task['scene']['name']/task['arm']/'audit.json')
    metric=a['model_metrics'];passed=metric['held_worst_goal_mm']<=20 and metric['peak_overlap_mm']<=2
    write(root/'references'/(task['scene']['name']+'.json'),dict(state='passed' if passed else 'failed',
        historical_run=ref['source'],historical_metrics=ref['historical_metrics'],current_metrics=metric,
        initial_max_absolute_errors=errors,deterministic_current_run=True,
        caveat='Historical run did not request deterministic kernels; compare declared tolerance and goal/overlap, not bitwise optimization paths.'))
    if not passed:raise AssertionError('Historical good-result reproduction failed; pause before new comparisons')


def complete(root,task):
    plan=read(root/'plan.json');artifact=root/'artifacts'/task['scene']['name']/task['arm']
    metrics=read(artifact/'audit.json')['model_metrics']
    with (root/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        queue=read(root/'queue.json');entry=next(x for x in queue['tasks'] if x['id']==task['id'])
        entry.update(state='completed',finished_utc=now(),metrics=metrics)
        name=task['scene']['name'];in_matrix=any(s['name']==name for s in plan['scenes'])
        if task['phase']<=2 and in_matrix:
            if accepted_baseline(metrics):
                selected=dict(arm=task['arm'],iterations=task.get('iterations',64),goal_weight=task.get('goal_weight',1.))
                queue['selected'][name]=selected
                for directory in ('runs','artifacts'):
                    alias=root/directory/name/'full'
                    if alias.exists():raise FileExistsError(alias)
                    alias.symlink_to(task['arm'],target_is_directory=True)
            elif task['arm']!='full128_goal3':
                arm='full128' if task['arm']=='full64' else 'full128_goal3'
                queue['tasks'].append(dict(id=f'{name}__{arm}',scene=task['scene'],arm=arm,phase=2,state='pending',
                    iterations=128,goal_weight=1. if arm=='full128' else 3.))
            else:queue['gate_failures'].append(name)
        unfinished=[t for t in queue['tasks'] if t['state'] in ('pending','running') and t['phase']<=2]
        if not unfinished and not any(t['phase']==3 for t in queue['tasks']):
            for scene in plan['scenes']:
                if scene['name'] not in queue['selected']:continue
                settings=queue['selected'][scene['name']]
                effective=dict(scene,effective_iterations=settings['iterations'],effective_goal_weight=settings['goal_weight'])
                for arm in ('goal','geometry','contact','fast10','coarse_residual','coarse4','cosine'):
                    queue['tasks'].append(dict(id=f'{scene["name"]}__{arm}',scene=effective,arm=arm,phase=3,state='pending',
                        iterations=settings['iterations'],goal_weight=settings['goal_weight']))
        if (len(queue['selected'])==len(plan['scenes']) and any(t['phase']==3 for t in queue['tasks'])
                and all(t['state']=='completed' for t in queue['tasks']) and not any(t['phase']==4 for t in queue['tasks'])):
            obstacles=Path(plan['parent_root'])/'obstacles/plan.json'
            fixtures={s['name']:s for s in read(obstacles)['scenes']} if obstacles.exists() else {}
            for scene in plan['scenes']:
                fixture=fixtures.get(scene['name'],{})
                if fixture.get('obstacle_state')!='ready':continue
                settings=queue['selected'][scene['name']]
                effective=dict(scene,effective_iterations=settings['iterations'],effective_goal_weight=settings['goal_weight'],
                    obstacles=fixture['obstacles'],obstacles_sha256=fixture['obstacles_sha256'])
                queue['tasks'].append(dict(id=f'{scene["name"]}__avoid',scene=effective,arm='avoid',phase=4,state='pending',
                    iterations=settings['iterations'],goal_weight=settings['goal_weight']))
        if all(t['state'] not in ('pending','running') for t in queue['tasks']):
            queue['state']='needs investigation' if queue['gate_failures'] else 'completed'
        queue['updated_utc']=now();write(root/'queue.json',queue)


def finish_continuation_handoff(root,gpu,path):
    """Finish a paused controller's active child before taking over its queue lane."""
    row=read(path)
    if row['gpu']!=gpu:raise ValueError('Handoff belongs to another GPU')
    if row['state']=='completed':return
    child=Path('/proc')/str(row['child_pid']);driver=Path('/proc')/str(row['driver_pid'])
    while True:
        stat=child.joinpath('stat').read_text().rsplit(')',1)[1].split()
        if stat[19]!=row['child_start_ticks']:raise RuntimeError('Child PID identity changed')
        if stat[0]=='Z':
            if int(stat[49])!=0:raise RuntimeError('Prior optimization exited unsuccessfully')
            break
        time.sleep(10)
    task=row['task'];scene=task['scene'];arm=task['arm'];out=root/'runs'/scene['name']/arm
    if not (out/'results.json').exists() or not (out/f'seed_{scene["seed"]}.pt').exists():
        raise RuntimeError('Prior optimization did not save complete outputs')
    args=[v.decode() for v in driver.joinpath('cmdline').read_bytes().split(b'\0') if v]
    if args!=row['driver_command']:raise RuntimeError('Driver PID identity changed')
    os.kill(row['driver_pid'],signal.SIGTERM);os.kill(row['driver_pid'],signal.SIGCONT)
    # The original parent was paused during subprocess.wait. Preserve an explicit
    # approximate end time from the child's last output, not the later handoff.
    receipt=row['trial_status'];end=max(p.stat().st_mtime for p in out.iterdir())
    receipt.update(state='completed',returncode=0,finished_utc=datetime.fromtimestamp(end,timezone.utc).isoformat(),
        elapsed_seconds=end-datetime.fromisoformat(receipt['started_utc']).timestamp(),
        timing_note='End time estimated from last child output; exit code verified before controller handoff.')
    write(out.with_suffix('.status.json'),receipt)
    plan=read(root/'plan.json');cfg=local_plan(plan,scene,iterations=task.get('iterations'),goal_weight=task.get('goal_weight'))
    audit_one(root,cfg,scene,arm);visualize_one(root,cfg,scene,arm);complete(root,task)
    row.update(state='completed',finished_utc=now());write(path,row)


def run(root,gpu,*,handoff=None):
    plan=read(root/'plan.json');os.environ['CUDA_VISIBLE_DEVICES']=plan['gpus'][gpu]
    for p in plan['replay_vendors']:sys.path.append(p)
    status=dict(state='starting',gpu=gpu,started_utc=now());write(root/f'gpu{gpu}.json',status)
    try:
        if handoff is None:finish_handoff(root,gpu)
        else:finish_continuation_handoff(root,gpu,handoff)
        status.update(state='running');write(root/f'gpu{gpu}.json',status)
        while True:
            queue=read(root/'queue.json')
            if queue['state'] in TERMINAL:break
            if time.time()>=datetime.fromisoformat(read(root/'budget.json')['deadline_utc']).timestamp():break
            task=claim(root,gpu)
            if task is None:time.sleep(15);continue
            scene=task['scene'];arm=task['arm'];out=root/'runs'/scene['name']/arm
            status.update(current_task=task['id'],updated_utc=now());write(root/f'gpu{gpu}.json',status)
            cfg=local_plan(plan,scene,iterations=task.get('iterations'),goal_weight=task.get('goal_weight'),
                baseline_arm=arm if task['phase']<=2 else 'full')
            if task['phase']<=2:cfg['arms']=dict(cfg['arms'],**{arm:dict(label=f'Full objective: {cfg["iterations"]} updates, goal weight {cfg["goal_weight"]:g}',terms=None)})
            if task['phase']==4:
                cfg=obstacle_plan(root,plan,task)
            ref=task.get('reference',{}).get('initial')
            if task['phase']==2:ref=str(root/'runs'/scene['name']/'full64/seed_8311.pt')
            if task['phase']>=3:ref=str(root/'runs'/scene['name']/'full/seed_8311.pt')
            for file,expected in scene['hashes'].items():
                if sha(Path(scene['directory'])/file)!=expected:raise ValueError('Frozen fixture changed')
            deadline=datetime.fromisoformat(read(root/'budget.json')['deadline_utc']).timestamp()
            receipt=execute(cfg,command(cfg,scene,arm,out,reference=ref),out,deadline-600,timeout=task_cost_seconds(task)-300)
            if receipt['state']!='completed':raise RuntimeError(f'{task["id"]} failed; inspect retained log')
            # Reproduction probe has the source scene's independent-controller record.
            audit_scene=scene
            if scene['name']=='reference_teacup_7803':
                # audit_one looks up qualification by scene name. Supply an honest
                # alias in a derived controls manifest, preserving the control hashes.
                controls=read(plan['controls']);controls=copy.deepcopy(controls)
                original=next(x for x in controls['results'] if x['scene']=='place_development_03_asset_7803')
                alias=dict(original,scene=scene['name']);controls['results'].append(alias)
                path=root/'references/controls_with_teacup_alias.json';write(path,controls);cfg['controls']=str(path)
            audit_one(root,cfg,audit_scene,arm);visualize_one(root,cfg,audit_scene,arm)
            if task['phase']==4:
                reference=task.get('reference_arm','obstacle_reference')
                audit_one(root,cfg,audit_scene,reference);visualize_one(root,cfg,audit_scene,reference)
            if task.get('reference'):reference_check(root,task,out)
            complete(root,task)
        status.update(state='completed',updated_utc=now());write(root/f'gpu{gpu}.json',status)
    except Exception as error:
        status.update(state='failed',error=repr(error),updated_utc=now());write(root/f'gpu{gpu}.json',status)
        with (root/'queue.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);queue=read(root/'queue.json')
            for entry in queue['tasks']:
                if entry['id']==status.get('current_task'):entry.update(state='failed',error=repr(error))
            queue.update(state='failed',error=repr(error),updated_utc=now());write(root/'queue.json',queue)
        raise


def comparison_sections(root,plan,queue,link):
    """Report each arm against its selected baseline, retaining failed checks."""
    objectives=('goal','geometry','contact')
    speed=('fast10','coarse_residual','coarse4','cosine')
    tasks={(t['scene']['name'],t['arm']):t for t in queue['tasks'] if t['phase']==3}
    labels={'goal':'Goal only','geometry':'Goal + geometry/joints','contact':'Geometry/joints + contact',
        'fast10':'10 sampler steps','coarse_residual':'Four noise knots + residual',
        'coarse4':'Four noise knots','cosine':'Cosine rate + normalized gradient'}
    failures={'goal':'goal position','goal_rotation':'goal orientation','penetration':'overlap',
        'support':'object/table','gripper_support':'gripper/table','opening':'finger limits',
        'speed':'finger speed','contact':'finger proximity','relative_translation':'relative slip',
        'relative_rotation':'relative rotation','rest':'rest on table','release':'release','settle':'settling'}

    def state(name,arm):
        return tasks.get((name,arm),{}).get('state','not queued')

    def rows(scene,arms,with_time=False):
        name=scene['name'];selected=queue['selected'].get(name)
        if selected is None:return []
        baseline=selected['arm'];base_time=None
        status=root/'runs'/name/(baseline+'.status.json')
        if status.exists():base_time=read(status).get('elapsed_seconds')
        result=[f'### {name} — '+('speed comparison' if with_time else 'objective comparison'),'',
            f'{selected["iterations"]} updates; goal weight {selected["goal_weight"]:g}; '+
            f'{scene["horizon"]} frames; seed {scene["seed"]}. Full objective uses `{baseline}`.','']
        if with_time:
            result+=['| Method | State | Held error (mm) | Overlap (mm) | Minutes | Speedup | Model checks | Replay | GIF |',
                '|---|---|---:|---:|---:|---:|---|---|---|']
        else:
            result+=['| Objective | State | Held error (mm) | Overlap (mm) | Model checks | Replay | GIF |',
                '|---|---|---:|---:|---|---|---|']
        animations=[]
        for arm in (baseline,*arms):
            label='Full objective (baseline)' if arm==baseline else labels[arm]
            status='completed' if arm==baseline else state(name,arm)
            if status!='completed':
                result.append(f'| {label} | {status} | — | — | '+('— | — | ' if with_time else '')+'— | — | — |')
                continue
            artifact=root/'artifacts'/name/arm;audit=read(artifact/'audit.json');metric=audit['model_metrics']
            failed=[failures.get(k,k) for k,v in metric['validation']['residuals_m'].items() if v>0]
            checks='Pass' if metric['geometry_pass'] else 'Fail: '+(', '.join(failed) or 'see audit')
            duration=''
            if with_time:
                receipt=root/'runs'/name/(arm+'.status.json');receipt=read(receipt) if receipt.exists() else {}
                seconds=receipt.get('elapsed_seconds');approx='~' if receipt.get('timing_note') else ''
                duration=(f'{approx}{seconds/60:.1f} | ' if seconds is not None else '— | ')+(
                    f'{approx}{base_time/seconds:.2f}× | ' if base_time and seconds else '— | ')
            result.append(f'| {label} | completed | {metric["held_worst_goal_mm"]:.1f} | {metric["peak_overlap_mm"]:.2f} | '+
                duration+checks+' | '+('Pass' if audit['qualified_task_pass'] else 'Fail')+
                ' | [Open GIF]('+link(Path('artifacts')/name/arm/'comparison_points.gif')+') |')
            if not with_time:
                gif=link(Path('artifacts')/name/arm/'comparison_points.gif')
                animations+=[f'**{label}**','',
                    f'Held error: {metric["held_worst_goal_mm"]:.1f} mm. Overlap: {metric["peak_overlap_mm"]:.2f} mm. '+
                    f'Model checks: {checks}. Replay: '+('Pass.' if audit['qualified_task_pass'] else 'Fail.'),'',
                    f'[Open GIF]({gif})','',f'![{name} — {label}]({gif})','']
        result+=['']
        if animations:
            result+=['The labeled panels show the original unconditional sample, the guided prediction, and simulator replay. '+
                'Only completed runs are embedded; pending results appear here automatically when ready.','']+animations
        return result

    objective_done=sum(state(s['name'],a)=='completed' for s in plan['scenes'] for a in objectives)
    scenes_done=sum(all(state(s['name'],a)=='completed' for a in objectives) for s in plan['scenes'])
    lines=['## Objective ablations','',
        f'**{objective_done}/{len(plan["scenes"])*len(objectives)} ablation runs complete; '+
        f'{scenes_done}/{len(plan["scenes"])} scenes have the full four-way comparison.** '+
        'Completed means audited, not necessarily successful.','',
        'Each comparison uses that scene’s selected full-objective baseline: identical checkpoint, target, seed, initial trajectory, '+
        'routing, horizon, optimizer settings and update budget. Earlier baseline-refinement attempts and obstacle runs are excluded.','',
        '| Objective | Terms used during optimization |','|---|---|',
        '| Goal only | Requested position and, for placement, orientation |',
        '| Goal + geometry/joints | Goal plus object–gripper overlap, object/table and gripper/table intrusion, finger limits and speed |',
        '| Geometry/joints + contact | Previous row plus proximity to both fingers during lifted motion; picking also asks for final proximity |',
        '| Full objective | Previous row plus unsupported upward motion and relative slip; placement also includes rest, release and settling |','',
        'All arms keep the noise regularizer and RMS cap. These are incremental groups, not individual-term leave-one-out tests. '+
        'The full independent model and replay checks remain active even when their corresponding optimization terms are removed. '+
        'These checks are geometric proxies; they do not certify force closure. See [the objective pseudocode](CURRENT_GUIDANCE.tex).','',
        '**Coverage across the fixed five picks and five places**','',
        '| Scene | Selected full baseline | Goal only | Geometry/joints | + Contact |',
        '|---|---|---|---|---|']
    for scene in plan['scenes']:
        name=scene['name'];selected=queue['selected'].get(name)
        lines.append(f'| {name} | '+(f'{selected["iterations"]} updates, weight {selected["goal_weight"]:g}' if selected else 'pending')+
            ' | '+' | '.join(state(name,a) for a in objectives)+' |')
    lines+=['','Held error is the **worst position error over the final six frames**. Overlap is peak object–gripper penetration. '+
        'Read accuracy together with the failed checks; a small position error can coexist with a poor grasp.','']
    for scene in plan['scenes']:
        if any(state(scene['name'],a) not in ('pending','not queued') for a in objectives):lines+=rows(scene,objectives)
    speed_done=sum(state(s['name'],a)=='completed' for s in plan['scenes'] for a in speed)
    lines+=['## Speed and noise optimization','',
        f'**{speed_done}/{len(plan["scenes"])*len(speed)} variant runs complete.** All retain the full objective.','',
        '| Variant | Change from the full-objective baseline |','|---|---|',
        '| 10 sampler steps | Reduce nominal denoising steps from 25 to 10 |',
        '| Four noise knots | Optimize four temporally interpolated update knots per window, preserving the original Gaussian draw |',
        '| Four noise knots + residual | Add a per-frame residual capped at RMS 0.10; total update remains capped at 0.50 |',
        '| Cosine rate + normalized gradient | Two-update warmup, cosine learning rate and normalized noise gradient |','',
        'Budgets match optimization updates, not model calls. Speedup is relative to the selected full baseline on the same scene. '+
        'Times cover the planning subprocess, including startup, but exclude replay and GIF rendering; `~` marks an output-timestamp estimate. '+
        'The 10-step sampler shares initial noise and routing but produces a different initial trajectory. '+
        'Reducing the number of noise-update parameters does not reduce the number of model evaluations.','']
    for scene in plan['scenes']:
        if any(state(scene['name'],a) not in ('pending','not queued') for a in speed):lines+=rows(scene,speed,True)
    return lines


def report(root):
    plan=read(root/'plan.json');queue=read(root/'queue.json');path=Path(plan['report']);prefix=os.path.relpath(root,path.parent)
    link=lambda p:prefix+'/'+str(p)
    done=[t for t in queue['tasks'] if t['state']=='completed'];selected=queue['selected']
    lines=['# Guidance study: stronger baselines and matched ablations','',
        f'Updated {now()}. State: **{queue["state"]}**. {len(done)} planning runs finished; '+
        f'**{len(selected)}/10 baselines meet the comparison prerequisite**.','',
        '[Objective ablations](#objective-ablations) · [Speed and noise optimization](#speed-and-noise-optimization) · '+
        '[Obstacle pilot](#obstacle-pilot) · [Point-cloud GIFs](#completed-point-cloud-comparisons). Animations are expanded below; '+
        'each also has an **Open GIF** link.','',
        'This replaces the 20-update screening study. Earlier good bottle and teacup trajectories used 64 updates; '+
        'the new study first reproduces those references, then applies their recipe to the same five pick and five place scenes. '+
        'The earlier runs remain intact in the [20-update report]('+os.path.relpath(Path(plan['parent_root'])/'REPORT_20_UPDATES.md',path.parent)+').','',
        '**Shared recipe:** frozen EMA checkpoint `99ec6cc6102a0d65…`; seed 8311; 64 Adam updates; learning rate 0.03; '+
        'noise-change RMS cap 0.5; prior 0.001; 25-step pyramid sampling; mean position error over the final six frames. '+
        'Picking uses 68 frames, placing 36. The original ten scenes and their requested destinations stay fixed. '+
        'The teacup reproduction alone uses its historical leftward goal and is listed separately.','',
        'The baseline includes all current geometric/interaction terms. Objective ablations remove only their declared families. '+
        'Speed/optimizer variants retain the full objective; fast10 alone uses ten sampler steps. '+
        'Every arm for a scene uses the same saved initial noise and routing.','',
        '**Baseline check:** '+plan['gate']+' '+plan['repair_policy'],'',
        'A passing baseline check is not a successful simulator grasp. Full model checks, independent execution, '+
        'contact failures and finger overtravel remain visible. All attempts and all ten scenes are reported; no failed scene is substituted.','',
        'Each GIF shows **original unconditional sample → guided prediction → simulator replay of the generated gripper**, '+
        'using points only. The same original sample appears across a scene’s comparisons.','',
        'Deadline: '+read(root/'budget.json')['deadline_utc']+'. '+
        ('The approved obstacle pilots precede remaining optimizer variants on GPU 0.'
         if (root/'obstacle_pilot.json').exists() else 'Obstacles remain optional after the main comparison.'),'']
    if (root/'obstacle_pilot.json').exists():
        pilot_file='obstacle_pilot_pick12.json' if (root/'obstacle_pilot_pick12.json').exists() else 'obstacle_pilot.json'
        pilot=read(root/pilot_file)
        lines+=['## Obstacle pilot','',
            'Purple points show the supplied barrier. Compare the unchanged baseline and avoidance plan in the same obstacle world. '+
            'The checkpoint does not observe the barrier; only the planning cost changes. Clearance checks include every body and interpolated poses.','',
            '| Scene | Plan | State | Held goal error | Minimum clearance | Replay passes | GIF |',
            '|---|---|---|---:|---:|---|---|']
        variants=pilot.get('variants',[dict(scene=n,arm='avoid',reference_arm='obstacle_reference',obstacle_count=1) for n in pilot['scenes']])
        for variant in variants:
            name=variant['scene'];task=next(t for t in queue['tasks'] if t['id']==f'{name}__{variant["arm"]}')
            for arm,label in [(variant['reference_arm'],'Unchanged baseline'),(variant['arm'],'Avoidance')]:
                artifact=root/'artifacts'/name/arm;receipt=artifact/'audit.json'
                if receipt.exists():
                    audit=read(receipt);metric=audit['model_metrics']
                    values=f'{metric["held_worst_goal_mm"]:.1f} mm | {metric["validation"]["obstacle_min_clearance_mm"]:.1f} mm | {audit["qualified_task_pass"]}'
                else:values='pending | pending | pending'
                gif=artifact/'comparison_points.gif'
                lines.append(f'| {name} ({variant["obstacle_count"]} obstacles) | {label} | '+('audited' if receipt.exists() else task['state'])+
                    ' | '+values+' | '+('[Open GIF]('+link(gif.relative_to(root))+')' if gif.exists() else 'pending')+' |')
        lines+=['','[Pilot settings]('+link(pilot_file)+'). Negative clearance means penetration.','']
    lines+=['## Historical reproduction checks','',
        '| Reference | State | Current worst held error | Earlier worst held error |','|---|---|---:|---:|']
    for ref in plan['references']:
        name=ref['scene']['name'];p=root/'references'/(name+'.json');d=read(p) if p.exists() else None
        lines.append(f'| {name} | {d["state"] if d else "pending"} | '+
            (f'{d["current_metrics"]["held_worst_goal_mm"]:.1f} mm' if d else 'pending')+
            f' | {ref["historical_metrics"]["held_worst_goal_mm"]:.1f} mm |')
    lines+=['','## Five picks and five places','',
        '| Scene | Baseline recipe | Worst held error | Model overlap | Prerequisite |','|---|---|---:|---:|---|']
    for scene in plan['scenes']:
        name=scene['name'];attempts=[t for t in done if t['scene']['name']==name and t['phase']<=2]
        chosen=selected.get(name);latest=next((t for t in attempts if chosen and t['arm']==chosen['arm']),attempts[-1] if attempts else None)
        lines.append(f'| {name} | '+(f'{chosen["iterations"]} updates, goal weight {chosen["goal_weight"]:g}' if chosen else 'being established')+' | '+
            (f'{latest["metrics"]["held_worst_goal_mm"]:.1f} mm | {latest["metrics"]["peak_overlap_mm"]:.2f} mm' if latest else 'pending | pending')+
            ' | '+('passed' if chosen else 'needs investigation' if name in queue['gate_failures'] else 'pending')+' |')
    lines+=['']+comparison_sections(root,plan,queue,link)+['## Completed point-cloud comparisons','']
    for scene in [*(r['scene'] for r in plan['references'] if r['scene']['name'].startswith('reference_')),*plan['scenes']]:
        name=scene['name'];rows=[t for t in done if t['scene']['name']==name]
        rows=[item for t in rows for item in ([dict(t,arm=t.get('reference_arm','obstacle_reference')),t] if t['phase']==4 else [t])]
        if not rows:continue
        lines+=[f'### {name}','']
        for task in rows:
            arm=task['arm'];artifact=root/'artifacts'/name/arm;audit=read(artifact/'audit.json');v=read(artifact/'visualization.json')
            lines+=[f'**{arm} — {task.get("iterations",64)} updates; goal weight {task.get("goal_weight",1):g}**','',
                f'Model worst held error: {audit["model_metrics"]["held_worst_goal_mm"]:.1f} mm. '+
                f'Full model checks: {audit["model_metrics"]["geometry_pass"]}. Qualified execution: {audit["qualified_task_pass"]}. '+
                f'Replay finger overtravel: {v["replay_finger_limit_peak_mm"]:.1f} mm.','',
                '[Open GIF]('+link(Path('artifacts')/name/arm/'comparison_points.gif')+')','',
                f'![{name} {arm}]('+link(Path('artifacts')/name/arm/'comparison_points.gif')+')','']
    lines+=['## Queue and provenance','',f'[Frozen plan]({link("plan.json")}) · [Queue]({link("queue.json")}) · '+
        f'[Source snapshot]({link("source_receipt.json")}) · [Budget]({link("budget.json")})','']
    tmp=path.with_suffix('.pending.md');tmp.write_text('\n'.join(lines));tmp.replace(path)


def monitor(root):
    status=dict(state='running',started_utc=now(),errors=[]);write(root/'reporter.json',status)
    try:
        while True:
            report(root);status['updated_utc']=now();write(root/'reporter.json',status)
            if read(root/'queue.json')['state'] in TERMINAL:break
            if time.time()>=datetime.fromisoformat(read(root/'budget.json')['deadline_utc']).timestamp():break
            time.sleep(30)
        status.update(state='completed',updated_utc=now());write(root/'reporter.json',status)
    except Exception as error:
        status.update(state='failed',error=repr(error),updated_utc=now());write(root/'reporter.json',status);raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['prepare','run','report','monitor','pilot'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--gpu',type=int,choices=(0,1));p.add_argument('--hours',type=float,default=40)
    p.add_argument('--handoff',type=Path);p.add_argument('--scenes',nargs='+')
    a=p.parse_args();root=a.root.resolve()
    if a.mode=='prepare':prepare(root,a.hours)
    elif a.mode=='run':
        if a.gpu is None:p.error('run needs --gpu')
        run(root,a.gpu,handoff=a.handoff)
    elif a.mode=='pilot':
        if a.gpu is None or not a.scenes:p.error('pilot needs --gpu and --scenes')
        prioritize_obstacle_pilot(root,a.scenes,a.gpu)
    elif a.mode=='report':report(root)
    else:monitor(root)


if __name__=='__main__':main()
