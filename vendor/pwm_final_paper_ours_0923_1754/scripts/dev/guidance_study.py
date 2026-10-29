"""One manifest, two GPU workers, and one Markdown report for guidance studies.

The workers use an immutable source archive. Raw run receipts and replay states
stay below the study directory; only the report is a user-facing document.
"""
import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


REPO = Path(__file__).resolve().parents[2]
PYTHON = '/data/user_data/robertre/conda_envs/pwm_blackwell/bin/python'
CHECKPOINT = '/data/group_data/katefgroup/runs_pwm_robertre/pickup_jointed_unet_bidir_uncond_cleanctx20k_from40k_0914/checkpoints/final.pt'
GPUS = ['GPU-439f4fb8-a94f-1bf2-47e9-246e9a18a799', 'GPU-d120b7b7-ce6b-5280-619f-d830a5765efa']
GEOMETRY = ['penetration', 'support', 'gripper_support', 'joint', 'joint_speed']
ARMS = {
    'goal': dict(label='Goal only', terms='none'),
    'full': dict(label='Full current objective', terms=None),
    'geometry': dict(label='Goal + geometry/joints', terms=','.join(GEOMETRY)),
    'contact': dict(label='Geometry/joints + contact', terms=','.join(GEOMETRY + ['contact'])),
    'fast10': dict(label='Goal only: 10 sampler steps', terms='none', steps=10),
    'cosine': dict(label='Goal only: cosine + normalized gradients', terms='none',
                   extra=['--lr-schedule','cosine','--warmup-updates','2','--normalize-noise-gradient']),
    'coarse4': dict(label='Goal only: four temporal noise knots/window', terms='none', extra=['--noise-knots','4']),
    'coarse_residual': dict(label='Goal only: four knots + bounded residual', terms='none',
                          extra=['--noise-knots','4','--residual-max-rms','.1']),
}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.pending')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n'); tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def prepare(root, hours):
    import torch
    if (root/'plan.json').exists(): raise FileExistsError(root/'plan.json')
    registry = REPO/'validation/guidance_noise_2026-09-15/scenes_v2.json'
    all_scenes = read(registry)['scenes']; counts = {'pick':0, 'place':0}; scenes = []
    for source in all_scenes:
        task = source['task']; index = counts[task]
        if index == 5: continue
        counts[task] += 1; directory = Path(source['directory'])
        batch = torch.load(directory/'input.pt', map_location='cpu', weights_only=False)
        if task == 'pick':
            target = batch.body_poses[0, batch.context_len-1, 0, :3, 3].double()
            target = target + torch.tensor([-.15 if index % 2 == 0 else .15, 0., .15], dtype=torch.float64)
            target_R = None
        else:
            request = read(directory/'task.json')
            target = torch.tensor(request['target'], dtype=torch.float64)
            target[0] += -.15 if index % 2 == 0 else .15
            target_R = request['target_R']
        scenes.append(dict(**source, target=target.tolist(), target_R=target_R,
            seed=9515, hashes={name:sha(directory/name) for name in ('input.pt','collision_geometry.pt','calibration.pt')}))
    if counts != {'pick':5, 'place':5}: raise ValueError('Insufficient fixed fixtures')
    vendors = [str(REPO/'validation/guidance_noise_2026-09-15/vendor_fcl'),
        str(REPO/'validation/overnight_2026-09-14/vendor')]
    controls = REPO/'validation/guidance_noise_2026-09-15/mujoco_controls_matched_v2_summary.json'
    quals = {x['scene']:x for x in read(controls)['results']}
    for scene in scenes:
        scene['recorded_control_qualified'] = quals[scene['name']]['qualified_for_generated_diagnostic']
    plan = dict(version=1, created_utc=now(), hours=hours, allocation=10446392, gpus=GPUS,
        checkpoint=CHECKPOINT, checkpoint_sha256=sha(CHECKPOINT), registry_sha256=sha(registry),
        controls=str(controls), controls_sha256=sha(controls), vendors=vendors,
        replay_vendors=[str(REPO/'validation/guidance_noise_2026-09-15/vendor_mujoco'),*vendors],
        source=str(root/'source'), report=str(REPO/'docs/guidance/README.md'),
        horizon=68, iterations=20, seed=9515, hold_frames=6, goal_reduction='max',
        arms=ARMS, scenes=scenes, lanes={'0':['fast10','cosine','coarse4','coarse_residual'],
            '1':['goal','full','geometry','contact']},
        selection='First five scenes of each task in the existing frozen registry, without filtering on model or replay outcomes.',
        interpretation='Previously inspected fixtures; no claim of held-out training data or population success. One seed per scene.',
        qualification_required=True, deterministic=True)
    write(root/'plan.json', plan)
    report(root)


def command(plan, scene, arm, out, *, reference=None, no_checkpoint=False, iterations=None, steps=None):
    a = plan['arms'][arm]; directory = Path(scene['directory'])
    cmd = [PYTHON,'-m','guidance.run_unconditional','--checkpoint',plan['checkpoint'],
        '--input',str(directory/'input.pt'),'--geometry',str(directory/'collision_geometry.pt'),
        '--calibration',str(directory/'calibration.pt'),'--collision-backend','fcl',
        '--out',str(out),'--method','noise_opt','--objective','physical','--task',scene['task'],
        '--target='+','.join(map(str,scene['target'])),'--seeds',str(scene['seed']),
        '--horizon',str(plan['horizon']),'--context-only-future','--hold-frames',str(plan['hold_frames']),
        '--goal-reduction',plan['goal_reduction'],'--iterations',str(plan['iterations'] if iterations is None else iterations),
        '--lr','.03','--max-noise-rms','.5','--prior-weight','.001','--scope','all',
        '--sampling-steps',str(steps or a.get('steps',25))]
    if scene['task'] == 'pick': cmd += ['--pick-goal-hold']
    if 'goal_weight' in plan: cmd += ['--goal-weight',str(plan['goal_weight'])]
    if plan.get('deterministic', False): cmd += ['--deterministic']
    if scene['target_R'] is not None:
        cmd += ['--target-rotation='+','.join(str(v) for row in scene['target_R'] for v in row)]
    for vendor in plan['vendors']: cmd += ['--collision-vendor',vendor]
    if a['terms'] is not None: cmd += ['--physical-terms',a['terms']]
    cmd += a.get('extra',[])
    if a.get('obstacles'):
        cmd += ['--obstacles',scene['obstacles'],'--obstacle-weight','5',
            '--obstacle-clearance-mm','5','--obstacle-substeps','3','--obstacle-audit-substeps','10']
    if reference: cmd += ['--initial-rollout',str(reference)]
    if no_checkpoint: cmd += ['--no-activation-checkpointing']
    if iterations is None: cmd += ['--save-iterations',plan.get('save_iterations','10')]
    return cmd


def execute(plan, cmd, out, deadline, timeout=2400):
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    row = dict(state='running', started_utc=now(), command=cmd)
    if out.exists(): raise FileExistsError(out)
    started = time.monotonic()
    with out.with_suffix('.log').open('x') as log:
        process = subprocess.Popen(cmd, cwd=plan['source'], stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, env=dict(os.environ, PYTHONPATH=plan['source'],
                OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', WANDB_MODE='disabled'))
        row['pid'] = process.pid; write(out.with_suffix('.status.json'), row)
        try: code = process.wait(timeout=min(timeout,max(1,deadline-time.time())))
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try: process.wait(timeout=15)
            except subprocess.TimeoutExpired: os.killpg(process.pid, signal.SIGKILL); process.wait()
            code = 124
    row.update(state='completed' if code==0 else 'failed', returncode=code,
        elapsed_seconds=time.monotonic()-started, finished_utc=now())
    write(out.with_suffix('.status.json'), row)
    return row


def profile(root, plan, deadline):
    """Bounded real-model parity/latency checks; OOM is a measured limitation."""
    import numpy as np
    results=[]
    for task in ('pick','place'):
        scene=next(s for s in plan['scenes'] if s['task']==task)
        base=root/'qualification'/('deterministic' if plan.get('deterministic') else '.')/task/'baseline'
        if not (base/'results.json').exists():raise RuntimeError('Real-model qualification missing')
        reference=base/f'seed_{scene["seed"]}.pt'
        alternatives={}
        for name in ('no_checkpoint','legacy_zero_weights','steps10'):
            out=root/'profiles'/task/name
            cmd=command(plan,scene,'goal',out,iterations=1,reference=reference,
                no_checkpoint=name=='no_checkpoint',steps=10 if name=='steps10' else 25)
            if name=='legacy_zero_weights':
                i=cmd.index('--physical-terms');del cmd[i:i+2];cmd+=['--physical-weight','0']
            state=execute(plan,cmd,out,deadline,900)
            item=dict(**state)
            if state['state']=='completed':
                a=read(base/'results.json')[0]; b=read(out/'results.json')[0]
                za=np.load(base/f'seed_{scene["seed"]}.npz');zb=np.load(out/f'seed_{scene["seed"]}.npz')
                item.update(seconds=b['seconds'],model_calls=b['actual_model_calls_including_recomputation'],
                    allocated_gib=b['max_allocated_bytes']/2**30,
                    initial_position_max_error_m=float(np.abs(za['initial_T']-zb['initial_T']).max()),
                    selected_position_max_error_m=float(np.abs(za['T']-zb['T']).max()),
                    gradient_norm_abs_error=abs(a['trace'][0]['gradient_norm']-b['trace'][0]['gradient_norm']),
                    updated_objective_abs_error=abs(a['trace'][1]['loss']-b['trace'][1]['loss']))
                if name!='steps10':
                    item['parity_pass']=bool(item['initial_position_max_error_m']<=1e-7
                        and item['selected_position_max_error_m']<=1e-6
                        and item['updated_objective_abs_error']<=1e-6
                        and item['gradient_norm_abs_error']<=1e-5)
                    if not item['parity_pass']:raise AssertionError(f'{task}/{name} parity failed')
            elif name!='no_checkpoint':raise RuntimeError(f'Profile failed: {task}/{name}')
            else:
                log=out.with_suffix('.log').read_text()
                if 'out of memory' not in log.lower():raise RuntimeError('Unexpected uncheckpointed profile failure')
                item['limitation']='GPU memory exhaustion; retain checkpointing'
            alternatives[name]=item
        baseline=read(base/'results.json')[0]
        results.append(dict(task=task,baseline_seconds=baseline['seconds'],
            baseline_calls=baseline['actual_model_calls_including_recomputation'],alternatives=alternatives))
        write(root/'profiles.json',dict(state='running',results=results))
    # Keep experimental methods checkpointed. A faster equivalent implementation
    # is documented separately; avoid choosing it from only two geometry sizes.
    write(root/'profiles.json',dict(state='completed',results=results,
        main_policy='checkpointed on all ten scenes; no-checkpoint latency/parity is a separate two-scene probe'))


def qualify(root,gpu):
    import torch
    plan=read(root/'plan.json');os.environ['CUDA_VISIBLE_DEVICES']=plan['gpus'][gpu]
    task=('pick','place')[gpu];scene=next(s for s in plan['scenes'] if s['task']==task)
    base=root/'qualification'/('deterministic' if plan.get('deterministic') else '.')/task/'baseline'
    row=execute(plan,command(plan,scene,'goal',base,iterations=1),base,time.time()+1800,900)
    if row['state']!='completed':raise RuntimeError('Real model baseline failed')
    reference=base/f'seed_{scene["seed"]}.pt';paired=base.parent/'paired'
    row=execute(plan,command(plan,scene,'goal',paired,iterations=1,reference=reference),paired,time.time()+900,900)
    if row['state']!='completed':raise RuntimeError('Saved-noise CLI replay failed')
    a=torch.load(reference,map_location='cpu',weights_only=False)
    b=torch.load(paired/f'seed_{scene["seed"]}.pt',map_location='cpu',weights_only=False)
    for key in ('positions','R','T'):
        torch.testing.assert_close(a['initial'][key],b['initial'][key],atol=0,rtol=0)
        torch.testing.assert_close(a['result']['rollout'][key],b['result']['rollout'][key],atol=0,rtol=0)
    ra=read(base/'results.json')[0];rb=read(paired/'results.json')[0]
    for k in ('loss','task_loss','gradient_norm'):
        assert ra['trace'][0][k]==rb['trace'][0][k],k
    assert ra['trace'][1]['loss']==rb['trace'][1]['loss']
    assert ra['trace'][0]['gradient_norm']>0
    write(root/'qualification'/f'gpu{gpu}.json',dict(status='passed',task=task,utc=now(),
        checks=['68-frame CLI forward/backward with finite nonzero gradient',
            'saved initial noise/RNG replay exactly matches initial and optimized trajectories',
            'exact first-gradient norm and updated objective parity'],
        source_revision=read(plan.get('source_receipt',root/'source_receipt.json'))['revision']))


def run(root, gpu, *, resume=False):
    plan=read(root/'plan.json'); launch=read(root/'launch.json')
    deadline=datetime.fromisoformat(launch['deadline_utc']).timestamp()
    if read(root/'qualification.json')['status']!='passed':raise RuntimeError('Qualification has not passed')
    if sha(plan['controls']) != plan['controls_sha256']:raise ValueError('Controller qualification changed')
    for scene in plan['scenes']:
        for name,expected in scene['hashes'].items():
            if sha(Path(scene['directory'])/name)!=expected:raise ValueError('Frozen fixture changed')
    status_file=root/f'gpu{gpu}.json'
    if resume:
        status=read(status_file)
        if status['state']!='failed':raise ValueError('Only a stopped failed worker can resume')
        revision=(REPO/'SOURCE_REVISION').read_text().strip()
        backup=root/'recovery'/f'gpu{gpu}_{int(time.time())}'
        backup.mkdir(parents=True,exist_ok=False)
        write(backup/'previous_status.json',status)
        plan=dict(plan,source=str(REPO))
        status.update(state='running',resumed_utc=now(),recovery_source=str(REPO),
            recovery_revision=revision,recovery_backup=str(backup),previous_error=status.get('error'),error=None)
    else:
        if status_file.exists():raise FileExistsError(status_file)
        status=dict(state='running',gpu=gpu,plan_sha256=sha(root/'plan.json'),entries=[],started_utc=now())
    os.environ['CUDA_VISIBLE_DEVICES']=plan['gpus'][gpu]
    write(status_file,status)
    try:
        if gpu==0 and not resume:profile(root,plan,deadline)
        # GPU1 supplies one common initial-noise bank, then objective ablations.
        pairs=([(s,a) for a in plan['lanes']['1'] for s in plan['scenes']] if gpu==1 else
               [(s,a) for s in plan['scenes'] for a in plan['lanes']['0']])
        consecutive_failures=0
        for scene,arm in pairs:
            out=root/'runs'/scene['name']/arm
            if resume and (out/'results.json').exists():continue
            if resume and out.exists():
                old=read(out.with_suffix('.status.json'))
                if old['state']!='failed':raise RuntimeError('Cannot replace an unfinished run')
                destination=backup/scene['name'];destination.mkdir(parents=True,exist_ok=True)
                for path in (out,out.with_suffix('.log'),out.with_suffix('.status.json')):
                    path.rename(destination/path.name)
            row=dict(scene=scene['name'],arm=arm,state='waiting')
            status['entries'].append(row);write(status_file,status)
            reference=None
            if arm!='goal':
                base=root/'runs'/scene['name']/'goal'
                while not (base/'results.json').exists() and time.time()<deadline-1800:
                    upstream=base.with_suffix('.status.json')
                    if upstream.exists() and read(upstream)['state']=='failed':break
                    time.sleep(15)
                if not (base/'results.json').exists():
                    row['state']='not run: reference unavailable';write(status_file,status);continue
                reference=base/f'seed_{scene["seed"]}.pt'
            if time.time()>deadline-1800:
                row['state']='not run: deadline reserve';write(status_file,status);continue
            row.update(state='running',started_utc=now(),directory=str(out));write(status_file,status)
            receipt=execute(plan,command(plan,scene,arm,out,reference=reference),out,deadline-900)
            row.update(receipt);status['updated_utc']=now();write(status_file,status)
            consecutive_failures=consecutive_failures+1 if row['state']=='failed' else 0
            if consecutive_failures>=2:raise RuntimeError('Two consecutive infrastructure failures')
        status['state']='completed'
    except Exception as error:
        status.update(state='failed',error=repr(error));raise
    finally:
        status['updated_utc']=now();write(status_file,status)


def run_directory(root,plan,scene,arm):
    source_arm=plan['arms'][arm].get('source_arm')
    return (Path(plan['reference_root'])/'runs'/scene['name']/source_arm if source_arm else
            root/'runs'/scene['name']/arm)


def audit_one(root,plan,scene,arm):
    import numpy as np
    import torch
    from scripts.dev.render_guidance_push import load_variant
    from scripts.dev.guidance_report_utils import replay_pair
    from scripts.dev.guidance_force_audit import audit_forces
    key=scene['name']+'__'+arm;run_dir=run_directory(root,plan,scene,arm)
    artifact=root/'artifacts'/scene['name']/arm;artifact.mkdir(parents=True,exist_ok=True)
    if (artifact/'audit.json').exists():return
    entry=dict(key='plan',run=str(run_dir),seed=scene['seed'],label=plan['arms'][arm]['label'],
        checkpoint_sha256=plan['checkpoint_sha256'])
    if plan['arms'][arm].get('source_arm'):entry['evaluation_obstacles']=scene['obstacles']
    loaded=load_variant(entry,artifact)
    if loaded is None:return
    export=Path(loaded['export']);protocol=loaded['protocol']
    directories=[artifact/tag for tag in ('dt0005','dt00025')]
    for directory,step in zip(directories,('.0005','.00025')):
        if (directory/'result.json').exists():continue
        if directory.exists():raise RuntimeError(f'Incomplete replay: {directory}')
        cmd=[PYTHON,'-m','scripts.dev.replay_pickup_mujoco','--scene',scene['directory'],
            '--trajectory',str(export/'trajectory.npz'),'--frames',str(plan['horizon']),
            '--finger-mode','mesh_preload','--step',step,'--save-frame-states','--out',str(directory)]
        if protocol.get('obstacle_spec'):cmd+=['--obstacles',protocol['arguments']['obstacles']]
        for vendor in plan['replay_vendors']:cmd+=['--dependency-path',vendor]
        with directory.with_suffix('.log').open('x') as log:
            subprocess.run(cmd,cwd=plan['source'],stdout=log,stderr=subprocess.STDOUT,
                env=dict(os.environ,PYTHONPATH=plan['source'],CUDA_VISIBLE_DEVICES=''),check=True,timeout=300)
    qualification=next(x for x in read(plan['controls'])['results'] if x['scene']==scene['name'])
    audit=replay_pair(protocol,export/'trajectory.npz',directories,qualification)
    for replay,directory in zip(audit['replays'],directories):
        force_dir=directory/'forces'
        force=read(force_dir/'report.json') if (force_dir/'report.json').exists() else audit_forces(
            directory,force_dir,vendor_paths=plan['replay_vendors'])
        if not force['bitwise_pose_joint_and_integration_replay'] or force['execution_sha256']!=sha(directory/'execution.npz'):
            raise AssertionError('Instrumented replay does not match')
        replay['forces']=force
        replay['dense_contacts_pass']=force['peak_object_gripper_depth_mm']<=2 and force['obstacle_contact_solver_steps']==0
    audit['qualified_task_pass']=audit['qualified_task_pass'] and all(x['dense_contacts_pass'] for x in audit['replays'])
    saved=torch.load(run_dir/f'seed_{scene["seed"]}.pt',map_location='cpu',weights_only=False)
    baseline_root=Path(plan.get('baseline_root',root))
    baseline_arm=plan.get('baseline_arm','goal')
    protocol_initial=read(baseline_root/'runs'/scene['name']/baseline_arm/'protocol.json')
    reference=torch.load(baseline_root/'runs'/scene['name']/baseline_arm/f'seed_{scene["seed"]}.pt',map_location='cpu',weights_only=False)['initial']
    for before,after in zip(reference['noise'],saved['initial']['noise']):
        for a,b in zip(before,after):torch.testing.assert_close(a,b,atol=0,rtol=0)
    for before,after in zip(reference['rng_states'],saved['initial']['rng_states']):
        for a,b in zip(before,after):
            if a is not None:torch.testing.assert_close(a,b,atol=0,rtol=0)
    same_sampler=protocol['sampling']==protocol_initial['sampling']
    if same_sampler:
        for key in ('positions','R','T'):torch.testing.assert_close(reference[key],saved['initial'][key],atol=0,rtol=0)
    audit.update(model_metrics=loaded['metrics'],paired_noise_and_routing=True,
        identical_initial_trajectory=same_sampler,protocol_sha256=sha(run_dir/'protocol.json'),
        evaluation_protocol_sha256=sha(export/'protocol.json'),
        trajectory_sha256=sha(export/'trajectory.npz'),scene=scene['name'],arm=arm)
    write(artifact/'audit.json',audit)


def visualize_one(root,plan,scene,arm):
    """Render the actual template points; never project replay joints into range."""
    import numpy as np
    import torch
    from scripts.dev.render_guidance_surfaces import render
    artifact=root/'artifacts'/scene['name']/arm;run_dir=run_directory(root,plan,scene,arm)
    if (artifact/'visualization.json').exists():return
    protocol=read(artifact/'plan/protocol.json')
    batch=torch.load(Path(scene['directory'])/'input.pt',map_location='cpu',weights_only=False)
    geometry=torch.load(Path(scene['directory'])/'collision_geometry.pt',map_location='cpu',weights_only=False)
    calibration=torch.load(Path(scene['directory'])/'calibration.pt',map_location='cpu',weights_only=False)['calibration']
    c={k:v.double().numpy() if torch.is_tensor(v) else v for k,v in calibration.items()}
    saved=torch.load(run_dir/f'seed_{scene["seed"]}.pt',map_location='cpu',weights_only=False)
    original_file=Path(plan.get('baseline_root',root))/'runs'/scene['name']/plan.get('baseline_arm','goal')/f'seed_{scene["seed"]}.pt'
    original_protocol=read(original_file.parent/'protocol.json')
    if original_protocol['actions']!='hidden':raise ValueError('Original sample must be unconditional on future actions')
    original=(saved if original_file==run_dir/f'seed_{scene["seed"]}.pt' else torch.load(original_file,map_location='cpu',weights_only=False))['initial']
    C=batch.context_len;ordinary=original;generated=saved['result']['rollout']
    executed=np.load(artifact/'dt00025/execution.npz')
    tracks=[(ordinary['R'][0,C-1:].numpy(),ordinary['T'][0,C-1:].numpy()),
        (generated['R'][0,C-1:].numpy(),generated['T'][0,C-1:].numpy()),
        (executed['model_R'],executed['model_T'])]
    templates={k:batch.body_templates[0,k,batch.body_template_mask[0,k]].numpy() for k in geometry['bodies']}
    materialization_error=0.;point_tracks=[]
    for rollout,(R,T) in zip((ordinary,generated),tracks):
        cloud={}
        for k,points in templates.items():
            actual=rollout['positions'][0,C-1:].numpy()[:,batch.particle_body[0].numpy()==k]
            shown=np.einsum('fij,nj->fni',R[:,k],points)+T[:,k,None]
            materialization_error=max(materialization_error,float(np.abs(actual-shown).max()))
            cloud[k]=actual
        point_tracks.append(cloud)
    q=executed['joint_q']
    excess=np.maximum(c['limits'][:,0]-q,q-c['limits'][:,1]).clip(0).max(-1)*1000
    R,T=tracks[2]
    point_tracks.append({k:np.einsum('fij,nj->fni',R[:,k],points)+T[:,k,None] for k,points in templates.items()})
    local=np.einsum('fji,fkj->fki',R[:,1],T[:,2:]-T[:,1,None])
    expected=c['finger_anchor'][None]+(q-c['q_anchor'])[:,:,None]*c['axis'][None]
    fk_error=float(np.linalg.norm(local-expected,axis=-1).max()*1000)
    notes=([f'INVALID replay: peak finger overtravel {excess.max():.1f} mm\nCurrent overtravel: {e:.1f} mm' for e in excess]
        if excess.max()>.1 else ['Finger travel within 0.1 mm of limits']*len(excess))
    render(scene['name'],protocol,geometry,tracks,artifact/'comparison_points.gif',artifact/'final_points.png',
        point_templates=templates,point_tracks=point_tracks,frame_notes=[None,None,notes],
        panel_titles=['Original unconditional sample',
            'Existing guided plan (avoidance off)' if plan['arms'][arm].get('source_arm') else 'After noise optimization',
            'Simulator replay of guided gripper'],
        heading=scene['name']+' / '+plan['arms'][arm]['label'],
        subtitle='Points: orange object; grey hand; blue/green fingers. Red cross: goal.'+
            (' Purple points: supplied obstacle.' if protocol.get('obstacle_spec') else ''))
    write(artifact/'visualization.json',dict(kind='particle_points',all_points=True,
        original_sample_source=str(original_file),original_sample_field='initial',original_sample_sha256=sha(original_file),
        original_sample_sampling=original_protocol['sampling'],
        model_points_source='saved particle positions',replay_points_source='same local templates at saved simulator body poses',
        point_count=sum(len(p) for p in templates.values()),model_materialization_error_m=materialization_error,
        replay_finger_limit_peak_mm=float(excess.max()),replay_finger_fk_error_mm=fk_error,
        replay_finger_limit_excess_by_frame_mm=excess.tolist(),
        execution_sha256=sha(artifact/'dt00025/execution.npz'),source_sha256=sha(run_dir/f'seed_{scene["seed"]}.pt'),
        renderer_sha256=sha(REPO/'scripts/dev/render_guidance_surfaces.py')))


def report(root,*,output=None):
    plan=read(root/'plan.json');report_path=Path(output or plan['report']);prefix=os.path.relpath(root,report_path.parent)
    def link(path):return prefix+'/'+str(path)
    launch=read(root/'launch.json') if (root/'launch.json').exists() else {}
    states={str(i):read(root/f'gpu{i}.json') if (root/f'gpu{i}.json').exists() else {'state':'not started'} for i in (0,1)}
    worker_states={(r['scene'],r['arm']):r['state'] for s in states.values() for r in s.get('entries',[])}
    rows=[]
    for scene in plan['scenes']:
        for arm in plan['arms']:
            run=root/'runs'/scene['name']/arm;artifact=root/'artifacts'/scene['name']/arm
            result=read(run/'results.json')[0] if (run/'results.json').exists() else None
            audit=read(artifact/'audit.json') if (artifact/'audit.json').exists() else None
            state='completed' if result else (read(run.with_suffix('.status.json'))['state'] if run.with_suffix('.status.json').exists()
                else worker_states.get((scene['name'],arm),'pending'))
            visual=read(artifact/'visualization.json') if (artifact/'visualization.json').exists() else None
            rows.append(dict(scene=scene['name'],task=scene['task'],arm=arm,state=state,result=result,audit=audit,visual=visual))
    done=sum(r['result'] is not None for r in rows);audited=sum(r['audit'] is not None for r in rows)
    visualized=sum(r['visual'] is not None for r in rows)
    lines=['# Guidance study: speed, simplicity and execution','',
        f'Updated {now()}. **{done}/80 planned runs completed; {audited} audited; {visualized} point-cloud GIFs ready.**',
        f'GPU 0: **{states["0"]["state"]}**. GPU 1: **{states["1"]["state"]}**. '+
        ('Deadline: '+launch.get('deadline_utc','not launched')+'.'),'',
        'This is the single entry point for the current study. Per-run data, logs and reproducibility receipts are nested '+
        'under one [study directory]('+link('.')+'). Older campaigns remain intact; see [the preceding report](PUSH_2026-09-15.md).','']
    if (root/'recovery.json').exists():
        lines += ['GPU 0 stopped because CUDA linear-interpolation backward does not support deterministic mode. '+
            'Recovery replaces that operation with equivalent indexed interpolation; successful runs remain intact, '+
            'failed attempts are retained, and the original deadline is unchanged. '+
            '[Recovery and validation receipt]('+link('recovery.json')+').','']
    lines += ['## Start here: rollout GIFs','',
        'Each point-cloud GIF shows **original unconditional sample**, **after noise optimization**, and '+
        '**simulator replay of the guided gripper**. '+
        'The left panel reads the saved pre-optimization rollout from the goal-only baseline. '+
        'Every variant for a scene shows that same original 25-step sample, including the cheaper-sampler variant. '+
        'The cheaper sampler starts optimization from its own 10-step rollout with the same initial noise; '+
        'that distinct starting rollout remains in its run files. '+
        'Orange is the object; the red cross is the requested target. In replay only the generated gripper is commanded. '+
        'These are the first pick and place fixtures in the fixed list, not a selection of successful results. '+
        'Every completed variant is also embedded in its expandable scene section below.','']
    for task in ('pick','place'):
        scene=next(s for s in plan['scenes'] if s['task']==task)
        for arm in ('goal','full'):
            row=next(r for r in rows if r['scene']==scene['name'] and r['arm']==arm)
            if row['visual']:
                audit=row['audit']
                lines += ['**'+scene['name']+' — '+plan['arms'][arm]['label']+'**','',
                    f'Model held error: {audit["model_metrics"]["held_worst_goal_mm"]:.1f} mm. '+
                    f'Full model checks: {audit["model_metrics"]["geometry_pass"]}. '+
                    f'Qualified execution checks: {audit["qualified_task_pass"]}. '+
                    f'Replay finger travel beyond limits: {row["visual"]["replay_finger_limit_peak_mm"]:.1f} mm.','',
                    '!['+scene['name']+' '+arm+']('+link(Path('artifacts')/scene['name']/arm/'comparison_points.gif')+')','']
    if not visualized:lines += ['Point-cloud GIFs are regenerating from the saved trajectories.','']
    lines += ['Replay finger overtravel is a simulator/controller failure. A flagged replay cannot establish that '+
        'the planned gripper motion executes faithfully; the points show the saved state without clamping or hiding it.','']
    if (root/'recovery/replay_limits.json').exists():
        lines += ['The first goal-only pick also commands the gripper through the table. '+
            'A controlled probe changes only the simulator finger-limit time constant to 1 ms; '+
            'the hand poses and applied finger commands are verified identical. '+
            'The tighter setting reduces the failure but still violates the joint range. '+
            'These are controller diagnostics; they do not replace the primary replay results.','',
            '| Same-command replay, 0.25 ms step | Original overtravel | Firmer-limit overtravel |',
            '|---|---:|---:|']
        for item in read(root/'recovery/replay_limits.json')['results']:
            if item['step']==.00025:
                lines.append(f'| {item["scene"]} | {item["baseline_peak_overtravel_mm"]:.1f} mm | {item["firmer_peak_overtravel_mm"]:.1f} mm |')
        lines += ['', '[Both time steps and raw replay receipts]('+link('recovery/replay_limits.json')+').','']
    lines += ['## Questions and fixed protocol','',
        '- GPU 0: can cheaper sampling or task-independent noise parameterizations improve quality per second?',
        '- GPU 1: which families of current penalties improve generated motion, goal reaching and independent execution?',
        '- Both tracks use the same five pick and five place fixtures, the same requested goals and seed 9515. '+
        'Pose/particle noise and routing are explicitly paired; sampler changes may change the initial trajectory.',
        '- Horizon **68 frames / 3.4 seconds**, 20 Adam updates, learning rate 0.03, noise-change RMS cap 0.50, prior 0.001. '+
        'The goal is the worst position error over the last six frames; placement retains its orientation goal.',
        '- Picking requests ±15 cm lateral and +15 cm vertical from the observation. Placing offsets the recorded task request by ±15 cm lateral; '+
        'it does not use a recorded achieved endpoint. Directions alternate by the fixed registry order.',
        '- Same frozen EMA checkpoint (`99ec6cc6102a0d65…`), FP32, 16-frame internal windows and clean context. No training.',
        '- Deterministic PyTorch algorithms are enabled for this study. Qualification requires bitwise agreement of '+
        'initial and one-update trajectories when replaying saved noise and routing, within this pinned environment. '+
        'An earlier qualification without this setting showed submicron object-position differences after backward accumulation; '+
        'those receipts are retained under `qualification/pick` and `qualification/place`.',
        '- This is a one-seed exploratory comparison on previously inspected fixtures. No scenes are replaced for poor results. '+
        'The primary matrix matches update count and duration, not model-call budget across samplers. Saved traces show progress versus elapsed time.',
        '', '## What the current terms mean','',
        '| Family | Terms and default weights | What it assumes |','|---|---|---|',
        '| Task request | position 1; placement rotation 0.04 m/rad | Requested object origin and, for place, orientation |',
        '| Geometry/robot | overlap 5; object floor 5; gripper floor 5; joint travel 10; joint speed 10 | Supplied convex geometry, a horizontal support plane and calibrated two-finger limits |',
        '| Contact | finger proximity 2 | A two-finger grasp during lifted motion; picking also asks for final proximity |',
        '| Interaction heuristics | unsupported rise 5; slip 1 | Upward motion should have nearby fingers; a held object should move with the hand |',
        '| Placement ending | rest 5; release 2; settle 2 | End on the support plane, open fingers, reduce translation |',
        '| Search regularization | noise-change squared mean 0.001 plus RMS cap 0.50 | Stay near a sampled noise draw; not a physical-validity guarantee |','',
        'All terms, tolerances and pseudocode are in [the TeX note](CURRENT_GUIDANCE.tex). '+
        'The geometry-only arm still uses known geometry and robot limits. Contact and the interaction/ending costs encode task strategy; '+
        'grouping them does not make them universal physics. The current planner supports rigid scenes.','',
        '## GPU 1: controlled objective ablations','',
        '| Arm | Enabled objective beyond the noise regularizer |','|---|---|',
        '| Goal only | Position and placement orientation |',
        '| Geometry/joints | Goal plus overlap, support planes, joint travel/speed |',
        '| Geometry/joints + contact | Previous arm plus finger proximity |',
        '| Full | All current terms, including rise/slip and placement ending |','',
        'This tests groups, not the necessity of each individual scalar term. The full-minus-contact comparison is not represented by the incremental contact arm. '+
        'Disabled terms remain in the independent audit. No feasibility screening changes candidate selection.','',
        '## GPU 0: speed and general noise updates','',
        '| Arm | Change from goal-only baseline |','|---|---|',
        '| fast10 | 10 nominal sampler steps instead of 25 |',
        '| cosine | Cosine learning rate, two-update warmup, normalized full-trajectory gradient |',
        '| coarse4 | Four interpolated noise-update knots per window, original Gaussian draw preserved |',
        '| coarse_residual | Four knots plus full-frame residual capped at RMS 0.10; total remains capped at 0.50 |','',
        'These arms share the goal-only objective. They are hypotheses, not claimed improvements. '+
        'The profile separately compares activation checkpointing and goal-only evaluation without unused geometry queries; '+
        'one-update profiles include startup within the measured planning block and are not long-run throughput estimates.','']
    if (root/'profiles.json').exists():
        prof=read(root/'profiles.json');lines += [f'Profile status: **{prof["state"]}**. [All measurements and parity checks]('+link('profiles.json')+').','']
        for item in prof['results']:
            for name,value in item['alternatives'].items():
                lines.append(f'- {item["task"]} / {name}: '+(f'{value["seconds"]:.1f} s; {value["allocated_gib"]:.1f} GiB; '+
                    f'{value["model_calls"]} model calls; parity {value.get("parity_pass","different sampler")}' if value['state']=='completed' else value.get('limitation',value['state'])))
        lines.append('')
    if (root/'obstacles/plan.json').exists():
        from scripts.dev.guidance_obstacle_study import report_lines
        lines += report_lines(root,report_path)
    lines += ['## Results so far','',
        'Generated geometry and independent execution are separate. MuJoCo commands only the gripper; the object moves dynamically. '+
        'Both 0.5 ms and 0.25 ms steps are audited, including dense contacts. Controller qualification is inherited from shorter recorded clips; '+
        'the 3.4-second generated horizon exceeds those controls. This is a floating-hand diagnostic with approximate contact/inertia, not Isaac/PhysX or hardware.',
        '', '| Arm | Finished / 10 | Model position hits | Full model checks | Qualified full execution |',
        '|---|---:|---:|---:|---:|']
    for arm in plan['arms']:
        selected=[r for r in rows if r['arm']==arm];ready=[r['audit'] for r in selected if r['audit']]
        lines.append(f'| {arm} | {sum(r["result"] is not None for r in selected)} | '+
            f'{sum(a["model_metrics"]["goal_pass"] for a in ready)}/{len(ready)} audited | '+
            f'{sum(a["model_metrics"]["geometry_pass"] for a in ready)}/{len(ready)} audited | '+
            f'{sum(a["qualified_task_pass"] for a in ready)}/{sum(a["control_qualified"] for a in ready)} control-qualified |')
    for scene in plan['scenes']:
        lines += ['', '<details>', '<summary>'+scene['name']+'</summary>', '',
            f'Recorded-control qualification: **{scene["recorded_control_qualified"]}**. Target: '+str(scene['target'])+' m.', '',
            '| Arm | State | Model held error | Replay held error (worse step) | Model / replay full checks | Minutes / model calls | GIF |',
            '|---|---|---:|---:|---|---|---|']
        for row in [r for r in rows if r['scene']==scene['name']]:
            a=row['audit'];result=row['result'];arm=row['arm']
            model=f'{a["model_metrics"]["held_worst_goal_mm"]:.1f} mm' if a else 'pending audit'
            replay=f'{max(x["held_worst_goal_mm"] for x in a["replays"]):.1f} mm' if a else 'pending'
            checks=f'{a["model_metrics"]["geometry_pass"]} / {a["qualified_task_pass"]}' if a else 'pending'
            runtime=f'{result["seconds"]/60:.2f} / {result["actual_model_calls_including_recomputation"]}' if result else '—'
            movie='[View]('+link(Path('artifacts')/scene['name']/arm/'comparison_points.gif')+')' if row['visual'] else 'pending'
            lines.append(f'| {arm} | {row["state"]} | {model} | {replay} | {checks} | {runtime} | {movie} |')
        for arm in plan['arms']:
            if (root/'artifacts'/scene['name']/arm/'visualization.json').exists():
                lines += ['', '**'+plan['arms'][arm]['label']+'**', '',
                    '!['+scene['name']+' '+arm+']('+link(Path('artifacts')/scene['name']/arm/'comparison_points.gif')+')']
        lines += ['', '</details>']
    receipts=[('Frozen plan and fixture hashes','plan.json'),('Qualification','qualification.json'),
        ('GPU 0 status','gpu0.json'),('GPU 1 status','gpu1.json'),('Reporter status','reporter.json')]
    lines += ['', '## Reproducibility and scope','',
        ' · '.join('['+label+']('+link(file)+')' for label,file in receipts if (root/file).exists())+'.',
        'Unqualified replay fixtures remain in the model comparison and are flagged in execution reporting. '+
        'A failed physical task is a result, not a reason to substitute a different scene. '+
        'Low goal error alone does not establish contact forces, plausibility or a successful task.', '']
    report_path.parent.mkdir(parents=True,exist_ok=True)
    temporary=report_path.with_suffix('.pending.md');temporary.write_text('\n'.join(lines));temporary.replace(report_path)
    write(root/'summary.json',dict(updated_utc=now(),completed=done,audited=audited,visualized=visualized,
        rows=[{k:v for k,v in r.items() if k not in ('result','audit','visual')} for r in rows]))


def monitor(root):
    plan=read(root/'plan.json');deadline=datetime.fromisoformat(read(root/'launch.json')['deadline_utc']).timestamp()
    os.environ['CUDA_VISIBLE_DEVICES']=''
    for vendor in plan['replay_vendors']:sys.path.append(vendor)
    status=dict(state='running',errors=[],started_utc=now());write(root/'reporter.json',status)
    while time.time()<deadline:
        changed=False
        for scene in plan['scenes']:
            for arm in plan['arms']:
                if (root/'runs'/scene['name']/arm/'results.json').exists() and not (root/'artifacts'/scene['name']/arm/'visualization.json').exists():
                    try:audit_one(root,plan,scene,arm);visualize_one(root,plan,scene,arm);changed=True
                    except Exception as error:
                        status['errors'].append(dict(scene=scene['name'],arm=arm,error=repr(error),utc=now()))
                        status['state']='failed';write(root/'reporter.json',status);report(root);raise
                    report(root)
                if time.time()>=deadline:break
        report(root);status['updated_utc']=now();write(root/'reporter.json',status)
        finished=all((root/f'gpu{i}.json').exists() and read(root/f'gpu{i}.json')['state'] in ('completed','failed') for i in (0,1))
        optional=root/'obstacles/status.json'
        optional_finished=not optional.exists() or read(optional)['state'] in ('completed','failed','not run: insufficient time')
        if finished and not changed and optional_finished:break
        time.sleep(30)
    status.update(state='completed',updated_utc=now());write(root/'reporter.json',status);report(root)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['prepare','qualify','run','monitor','report'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--hours',type=float,default=12)
    p.add_argument('--gpu',type=int,choices=[0,1]);p.add_argument('--resume',action='store_true')
    a=p.parse_args();root=a.root.resolve()
    if a.resume and a.mode!='run':p.error('--resume is only valid for run')
    if a.mode=='prepare':prepare(root,a.hours)
    elif a.mode in ('run','qualify'):
        if a.gpu is None:p.error('run requires --gpu')
        if a.mode=='run':run(root,a.gpu,resume=a.resume)
        else:qualify(root,a.gpu)
    elif a.mode=='monitor':monitor(root)
    else:report(root)


if __name__=='__main__':main()
