"""Stronger recipes must retain matched settings and cannot skip baseline checks."""
from datetime import datetime,timezone,timedelta

from scripts.dev.guidance_study import read,write,command,sha,run_directory
from scripts.dev.guidance_study_strong import accepted_baseline,local_plan,claim,complete,prioritize_obstacle_pilot,obstacle_plan


def metric(error=5.,overlap=1.):
    return dict(held_worst_goal_mm=error,peak_overlap_mm=overlap,
        validation=dict(residuals_m={k:-.001 for k in ('support','gripper_support','opening','speed')}))


def test_baseline_gate_does_not_confuse_position_hit_with_geometry():
    assert accepted_baseline(metric())
    assert not accepted_baseline(metric(error=21))
    assert not accepted_baseline(metric(overlap=3))
    x=metric();x['validation']['residuals_m']['opening']=.001
    assert not accepted_baseline(x)


def test_every_variant_inherits_its_baseline_budget_and_weight():
    scene=dict(name='place',directory='fixture',task='place',target=[1,2,3],target_R=None,seed=8311,horizon=36,
        effective_iterations=128,effective_goal_weight=3.)
    plan=dict(checkpoint='checkpoint',arms=dict(full=dict(terms=None),goal=dict(terms='none'),
        fast10=dict(terms=None,steps=10)),vendors=[],hold_frames=6,goal_reduction='mean')
    for arm in plan['arms']:
        cmd=command(local_plan(plan,scene),scene,arm,'out',reference='original.pt')
        for flag,expected in (('--iterations','128'),('--goal-weight','3.0'),('--horizon','36'),
                              ('--seeds','8311'),('--save-iterations','20,40,64,128')):
            assert cmd[cmd.index(flag)+1]==expected
        assert cmd[cmd.index('--sampling-steps')+1]==('10' if arm=='fast10' else '25')


def setup_queue(root):
    scene=dict(name='pick',horizon=68)
    plan=dict(scenes=[scene],parent_root=str(root/'previous'),lanes={'0':['fast10','coarse_residual','coarse4','cosine'],'1':['goal','geometry','contact']})
    task=dict(id='pick__full64',scene=scene,arm='full64',phase=1,state='running')
    write(root/'plan.json',plan)
    write(root/'queue.json',dict(state='running',tasks=[task],selected={},gate_failures=[]))
    write(root/'budget.json',dict(deadline_utc=(datetime.now(timezone.utc)+timedelta(hours=12)).isoformat()))
    return task


def test_passing_baseline_freezes_settings_and_unblocks_matched_arms(tmp_path):
    task=setup_queue(tmp_path)
    write(tmp_path/'artifacts/pick/full64/audit.json',dict(model_metrics=metric()))
    (tmp_path/'runs/pick/full64').mkdir(parents=True)
    complete(tmp_path,task)
    q=read(tmp_path/'queue.json')
    assert q['selected']['pick']==dict(arm='full64',iterations=64,goal_weight=1.)
    assert len(q['tasks'])==8
    assert (tmp_path/'runs/pick/full').resolve()==tmp_path/'runs/pick/full64'
    assert claim(tmp_path,0)['arm']=='fast10'
    assert claim(tmp_path,1)['arm']=='goal'


def test_bad_baseline_gets_bounded_repairs_without_launching_ablations(tmp_path):
    task=setup_queue(tmp_path)
    for arm,expected in [('full64','full128'),('full128','full128_goal3')]:
        write(tmp_path/f'artifacts/pick/{arm}/audit.json',dict(model_metrics=metric(error=100)))
        complete(tmp_path,task)
        q=read(tmp_path/'queue.json')
        assert not q['selected'] and not any(t['phase']==3 for t in q['tasks'])
        task=claim(tmp_path,0)
        assert task['arm']==expected and task['iterations']==128
    write(tmp_path/'artifacts/pick/full128_goal3/audit.json',dict(model_metrics=metric(error=100)))
    complete(tmp_path,task)
    q=read(tmp_path/'queue.json')
    assert q['state']=='needs investigation' and q['gate_failures']==['pick']
    assert len(q['tasks'])==3


def test_reproduction_phase_finishes_before_new_baselines(tmp_path):
    task=setup_queue(tmp_path)
    q=read(tmp_path/'queue.json');q['tasks'][0]['state']='pending'
    q['tasks'].append(dict(id='ref',phase=0,state='running',arm='full64',scene=task['scene']))
    write(tmp_path/'queue.json',q)
    assert claim(tmp_path,0) is None


def test_obstacles_wait_for_entire_main_matrix_and_inherit_selected_recipe(tmp_path):
    task=setup_queue(tmp_path)
    write(tmp_path/'artifacts/pick/full64/audit.json',dict(model_metrics=metric()))
    (tmp_path/'runs/pick/full64').mkdir(parents=True)
    complete(tmp_path,task)
    q=read(tmp_path/'queue.json')
    assert not any(t['phase']==4 for t in q['tasks'])
    for t in q['tasks']:t['state']='completed'
    last=q['tasks'][-1];last['state']='running';write(tmp_path/'queue.json',q)
    write(tmp_path/'previous/obstacles/plan.json',dict(scenes=[dict(name='pick',obstacle_state='ready',
        obstacles='box.json',obstacles_sha256='hash')]))
    write(tmp_path/'artifacts/pick'/last['arm']/'audit.json',dict(model_metrics=metric()))
    complete(tmp_path,last)
    q=read(tmp_path/'queue.json');optional=[t for t in q['tasks'] if t['phase']==4]
    assert len(optional)==1 and optional[0]['iterations']==64 and optional[0]['goal_weight']==1.
    assert optional[0]['scene']['obstacles']=='box.json'


def test_approved_pilot_precedes_its_lane_without_blocking_other_gpu(tmp_path):
    task=setup_queue(tmp_path)
    write(tmp_path/'artifacts/pick/full64/audit.json',dict(model_metrics=dict(metric(),geometry_pass=True)))
    (tmp_path/'runs/pick/full64').mkdir(parents=True)
    complete(tmp_path,task)
    box=tmp_path/'box.json';write(box,dict(version=1))
    write(tmp_path/'previous/obstacles/plan.json',dict(scenes=[dict(name='pick',obstacle_state='ready',
        obstacles=str(box),obstacles_sha256=sha(box))]))
    original=read(tmp_path/'queue.json');plan_hash=sha(tmp_path/'plan.json')
    prioritize_obstacle_pilot(tmp_path,['pick'],0)
    q=read(tmp_path/'queue.json')
    assert q['tasks'][:-1]==original['tasks']
    assert sha(tmp_path/'plan.json')==plan_hash
    assert read(tmp_path/'obstacle_pilot_before_queue.json')==original
    pilot=claim(tmp_path,0)
    assert pilot['arm']=='avoid' and pilot['iterations']==64 and pilot['goal_weight']==1.
    assert claim(tmp_path,1)['arm']=='goal'
    write(tmp_path/'artifacts/pick/avoid/audit.json',dict(model_metrics=metric()))
    complete(tmp_path,pilot)
    assert claim(tmp_path,0)['arm']=='fast10'


def test_pilot_rejects_changed_barrier_without_modifying_queue(tmp_path):
    import pytest
    task=setup_queue(tmp_path)
    write(tmp_path/'artifacts/pick/full64/audit.json',dict(model_metrics=dict(metric(),geometry_pass=True)))
    (tmp_path/'runs/pick/full64').mkdir(parents=True)
    complete(tmp_path,task)
    box=tmp_path/'box.json';write(box,dict(version=1))
    write(tmp_path/'previous/obstacles/plan.json',dict(scenes=[dict(name='pick',obstacle_state='ready',
        obstacles=str(box),obstacles_sha256='changed')]))
    before=read(tmp_path/'queue.json')
    with pytest.raises(ValueError,match='changed'):prioritize_obstacle_pilot(tmp_path,['pick'],0)
    assert read(tmp_path/'queue.json')==before
    assert not (tmp_path/'obstacle_pilot.json').exists()


def test_one_and_two_obstacles_keep_the_same_baseline_and_optimizer(tmp_path):
    plan=dict(checkpoint='checkpoint',arms={},vendors=[],hold_frames=6,goal_reduction='mean')
    scene=dict(name='pick',directory='fixture',task='pick',target=[1,2,3],target_R=None,seed=8311,horizon=68)
    for count,arm,reference in [(1,'avoid','obstacle_reference'),(2,'avoid2','obstacle_reference2')]:
        path=tmp_path/f'boxes{count}.json'
        write(path,dict(boxes=[dict(name=str(i)) for i in range(count)]))
        s=dict(scene,obstacles=str(path),obstacles_sha256=sha(path))
        task=dict(scene=s,arm=arm,reference_arm=reference,phase=4,iterations=128,goal_weight=3.)
        cfg=obstacle_plan(tmp_path,plan,task);cmd=command(cfg,s,arm,tmp_path/arm,reference='same_initial.pt')
        for flag,expected in [('--iterations','128'),('--goal-weight','3.0'),('--sampling-steps','25'),
                              ('--initial-rollout','same_initial.pt'),('--obstacles',str(path))]:
            assert cmd[cmd.index(flag)+1]==expected
        assert run_directory(tmp_path,cfg,s,reference)==tmp_path/'runs/pick/full'
        assert run_directory(tmp_path,cfg,s,arm)==tmp_path/'runs/pick'/arm
    assert plan['arms']=={}
