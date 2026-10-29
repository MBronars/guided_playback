"""Unconditional articulated planning: production parity and actual gradient routes."""
from dataclasses import replace

import pytest
import torch

from diffusion.flow_matching import TrajectoryFlowMatching
from guidance.objectives import body_target
from guidance.unconditional import noise_opt, prediction, sample_rollout, best_of_k, select_then_refine, multistart
from guidance.unconditional import noise_learning_rate, expand_noise_update
from inference import windowed_rollout
from models.unet_std import UNetWrapper
from tests.test_articulated_bodies import gripper
from tests.test_unet_std_pose_native import _model_cfg, open_gates
from utils.pose_native import pose_channel_mask


def setup(random_fps=True):
    batch = gripper(second="prismatic")[0].slice(1)
    torch.manual_seed(998)
    model = UNetWrapper(_model_cfg(articulated_bodies=True, unet_actions=False,
                                   unet_bidirectional_temporal=True,
                                   unet_fps_random_start=random_fps)).eval()
    open_gates(model, std=0.035)
    for p in model.parameters():
        p.requires_grad_(False)
    return batch, model, TrajectoryFlowMatching()


@pytest.mark.parametrize("schedule", ["pyramid", "full_sequence"])
def test_complete_rollout_matches_production_and_replays_random_fps(schedule):
    batch, model, diffusion = setup()
    kwargs = dict(future_len=4, horizon=6, schedule=schedule)
    torch.manual_seed(197)
    expected = windowed_rollout(model, diffusion, batch, num_inference_steps=2,
                                hide_given_actions=True, **kwargs)
    expected_rng = torch.get_rng_state()
    torch.manual_seed(197)
    with torch.no_grad():
        actual = sample_rollout(model, diffusion, batch, steps=2, **kwargs)
    for k in ("positions", "R", "T"):
        torch.testing.assert_close(actual[k], expected[k], rtol=0, atol=0)
    assert actual["windows"] == [(3, 4, 0), (7, 2, 0)]
    assert torch.equal(torch.get_rng_state(), expected_rng)
    before = torch.get_rng_state()
    with torch.no_grad():
        replay = sample_rollout(model, diffusion, batch, steps=2,
            noise=actual["noise"], rng_states=actual["rng_states"], **kwargs)
    assert torch.equal(torch.get_rng_state(), before)
    torch.testing.assert_close(replay["positions"], actual["positions"], rtol=0, atol=0)
    mask = pose_channel_mask(batch, actual["rigid_idx"])
    assert torch.count_nonzero(actual["pose_deltas"] * (1 - mask)) == 0


def test_future_state_and_action_corruption_cannot_affect_unconditional_rollout():
    batch, model, diffusion = setup()
    C = batch.context_len
    positions, poses = batch.positions.clone(), batch.body_poses.clone()
    positions[:, C:] += 3.0
    poses[:, C:, :, :3, 3] -= 2.0
    altered = replace(batch, positions=positions, body_poses=poses,
                      actions=torch.full_like(batch.actions, 77.0),
                      pose_actions=torch.full_like(batch.pose_actions, -91.0),
                      pose_deltas_gt=torch.full_like(batch.pose_deltas_gt, 31.0))
    torch.manual_seed(54)
    with torch.no_grad():
        ref = sample_rollout(model, diffusion, batch, future_len=4, steps=2)
        out = sample_rollout(model, diffusion, altered, future_len=4, steps=2,
                            noise=ref["noise"], rng_states=ref["rng_states"])
    torch.testing.assert_close(out["positions"], ref["positions"], rtol=0, atol=0)
    # Recorded actuator future is not pasted into the generated rollout.
    rows = batch.action_mask[0]
    assert not torch.allclose(out["positions"][:, C:, rows], batch.future[:, :, rows])
    torch.testing.assert_close(batch.positions[:, C:] + 3.0, altered.positions[:, C:])


def test_checkpointing_policy_preserves_complete_rollout_and_noise_gradient():
    batch,model,diffusion=setup()
    with torch.no_grad():
        initial=sample_rollout(model,diffusion,batch,future_len=4,steps=2)
    outputs=[]
    for enabled in (True,False):
        noise=[(p.clone(),q.clone().requires_grad_()) for p,q in initial['noise']]
        out=sample_rollout(model,diffusion,batch,future_len=4,steps=2,noise=noise,
            rng_states=initial['rng_states'],use_checkpoint=enabled)
        grads=torch.autograd.grad(out['T'][:,-1,0].square().sum(),[q for _,q in noise])
        outputs.append((out['T'].detach(),grads))
    torch.testing.assert_close(outputs[0][0],outputs[1][0],atol=0,rtol=0)
    for a,b in zip(outputs[0][1],outputs[1][1]):torch.testing.assert_close(a,b,atol=0,rtol=0)


def test_terminal_gradient_reaches_early_window_and_matches_finite_difference():
    batch, model, diffusion = setup(random_fps=False)
    with torch.no_grad():
        ref = sample_rollout(model, diffusion, batch, future_len=4, steps=2)
    noise = [(a.clone(), b.clone().requires_grad_()) for a, b in ref["noise"]]
    out = sample_rollout(model, diffusion, batch, future_len=4, steps=2,
                         noise=noise, rng_states=ref["rng_states"])
    loss = out["T"][0, -1, 3, 0]
    grads = torch.autograd.grad(loss, [z[1] for z in noise])
    for g in grads:
        assert torch.isfinite(g).all() and float(g.norm()) > 1e-7
    # Perturb an early-window object translation channel: the objective is in the
    # final, separate window. Detaching carried context breaks this derivative.
    index = (0, 0, 3, 0)
    eps = 1e-3
    vals = []
    with torch.no_grad():
        for sign in (-1, 1):
            zs = [(a.clone(), b.detach().clone()) for a, b in noise]
            zs[0][1][index] += sign * eps
            perturbed = sample_rollout(model, diffusion, batch, future_len=4, steps=2,
                                      noise=zs, rng_states=ref["rng_states"])
            vals.append(perturbed["T"][0, -1, 3, 0])
    finite_difference = (vals[1] - vals[0]) / (2 * eps)
    torch.testing.assert_close(grads[0][index], finite_difference, rtol=0.08, atol=1e-4)
    assert all(p.grad is None for p in model.parameters())


def test_actuator_noise_optimization_improves_goal_without_changing_passive_noise():
    batch, model, diffusion = setup(random_fps=False)
    with torch.no_grad():
        ref = sample_rollout(model, diffusion, batch, future_len=4, steps=2)
    target = ref["T"][:, -1, 3].clone() + torch.tensor([0.005, 0, 0])
    objective = body_target(3, target)
    result = noise_opt(model, diffusion, batch, objective, initial=ref,
        iterations=2, lr=0.01, scope="actuated", max_noise_rms=0.008,
        future_len=4, steps=2)
    assert result["iteration"] > 0
    assert result["loss"] < result["trace"][0]["loss"]
    assert result["nfe"] == 3 * ref["nfe"]
    mask = pose_channel_mask(batch, ref["rigid_idx"]).bool()
    active = mask & batch.pose_action_mask[0, :, None]
    for original, changed in zip(ref["noise"], result["rollout"]["noise"]):
        torch.testing.assert_close(changed[0], original[0], rtol=0, atol=0)
        diff = changed[1] - original[1]
        assert torch.count_nonzero(diff[..., ~active]) == 0
        assert float(diff[..., active].square().mean().sqrt()) <= 0.008001
        assert torch.count_nonzero(diff[..., active]) > 0
    assert all(p.grad is None for p in model.parameters())
    assert all(row["gradient_norm"] > 0 for row in result["trace"][:-1])


def test_zero_updates_reproduces_initial_sample():
    batch, model, diffusion = setup()
    with torch.no_grad():
        ref = sample_rollout(model, diffusion, batch, future_len=4, steps=2)
    target = batch.body_poses[:, batch.context_len - 1, 3, :3, 3]
    result = noise_opt(model, diffusion, batch, body_target(3, target),
                       initial=ref, iterations=0, future_len=4, steps=2)
    torch.testing.assert_close(result["rollout"]["positions"], ref["positions"], rtol=0, atol=0)


def test_hybrid_and_multistart_preserve_candidate_identity_without_updates():
    batch,model,diffusion=setup()
    objective=body_target(3,batch.body_poses[:,batch.context_len-1,3,:3,3])
    sampling=dict(future_len=4,steps=2)
    captured=[]
    torch.manual_seed(451)
    selected=best_of_k(model,diffusion,batch,objective,k=2,
        candidate_callback=lambda i,out,row:captured.append((i,out['positions'].clone())),**sampling)
    assert len(captured)==2
    torch.manual_seed(451)
    hybrid=select_then_refine(model,diffusion,batch,objective,k=2,iterations=0,**sampling)
    torch.testing.assert_close(hybrid['rollout']['positions'],selected['rollout']['positions'],rtol=0,atol=0)
    assert hybrid['initial_draw']==selected['draw']
    torch.manual_seed(451)
    multi=multistart(model,diffusion,batch,objective,starts=2,iterations=0,**sampling)
    torch.testing.assert_close(multi['rollout']['positions'],selected['rollout']['positions'],rtol=0,atol=0)
    assert multi['start']==selected['draw']
    assert multi['nfe']==2*selected['nfe']


def test_local_guidance_zero_scale_is_exact_and_active_updates_keep_articulation():
    batch,model,diffusion=setup()
    seen=[]
    class PrefixCost:
        def prefix_loss(self,pred,*,complete):
            seen.append((pred['t'].shape[1],complete))
            return (pred['t'][:,-1,3,0]-.08).square().mean()
    objective=PrefixCost()
    sampling=dict(future_len=4,steps=2)
    with torch.no_grad():
        ref=sample_rollout(model,diffusion,batch,**sampling)
        zero=sample_rollout(model,diffusion,batch,noise=ref['noise'],rng_states=ref['rng_states'],
            guidance_objective=objective,guidance_scale=0,**sampling)
        assert not seen
        torch.testing.assert_close(zero['positions'],ref['positions'],rtol=0,atol=0)
        guided=sample_rollout(model,diffusion,batch,noise=ref['noise'],rng_states=ref['rng_states'],
            guidance_objective=objective,guidance_scale=.1,guidance_threshold=.8,**sampling)
        replay=sample_rollout(model,diffusion,batch,noise=ref['noise'],rng_states=ref['rng_states'],
            guidance_objective=objective,guidance_scale=.1,guidance_threshold=.8,**sampling)
    assert seen and any(complete for _,complete in seen)
    assert all((length==6)==complete for length,complete in seen)
    assert guided['guidance_trace'] and torch.isfinite(guided['positions']).all()
    assert not torch.equal(guided['positions'],ref['positions'])
    torch.testing.assert_close(guided['positions'],replay['positions'],rtol=0,atol=0)
    torch.testing.assert_close(guided['positions'][:,:batch.context_len],batch.context,rtol=0,atol=0)
    mask=pose_channel_mask(batch,guided['rigid_idx'])
    assert torch.count_nonzero(guided['pose_deltas']*(1-mask))==0
    assert guided['nfe']==ref['nfe']


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('knots,frames', [(1,1),(1,16),(4,1),(4,16),(4,13),(16,16),(7,3)])
def test_temporal_interpolation_matches_values_and_gradients_deterministically(device,knots,frames):
    if device=='cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA is required for the deterministic-backward regression')
    was_deterministic=torch.are_deterministic_algorithms_enabled()
    was_warn_only=torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.manual_seed(410)
        update=torch.randn(2,knots,3,6,device=device,requires_grad=True)
        weight=torch.randn(2,frames,3,6,device=device)
        torch.use_deterministic_algorithms(False)
        flat=update.permute(0,2,3,1).reshape(2,18,knots)
        expected=torch.nn.functional.interpolate(flat,size=frames,mode='linear',align_corners=True)
        expected=expected.reshape(2,3,6,frames).permute(0,3,1,2)
        reference_grad=torch.autograd.grad((expected*weight).sum(),update)[0]
        torch.use_deterministic_algorithms(True)
        gradients=[]
        for _ in range(3):
            actual=expand_noise_update(update,frames)
            torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-6)
            gradient=torch.autograd.grad((actual*weight).sum(),update)[0]
            torch.testing.assert_close(gradient,reference_grad,atol=2e-6,rtol=2e-6)
            gradients.append(gradient)
        for gradient in gradients[1:]:
            torch.testing.assert_close(gradient,gradients[0],atol=0,rtol=0)
    finally:
        torch.use_deterministic_algorithms(was_deterministic,warn_only=was_warn_only)


def test_study_resume_preserves_completed_runs_and_archives_failed_attempts(tmp_path,monkeypatch):
    from datetime import datetime,timezone,timedelta
    from scripts.dev import guidance_study as study
    root=tmp_path/'study';root.mkdir()
    source=tmp_path/'source';source.mkdir();(source/'SOURCE_REVISION').write_text('recovery-revision\n')
    monkeypatch.setattr(study,'REPO',source)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    control=root/'control.json';study.write(control,{})
    plan=dict(source='original-frozen-source',controls=str(control),controls_sha256=study.sha(control),
        scenes=[dict(name='scene',directory='unused',hashes={},seed=5)],gpus=['test-gpu'],
        lanes={'0':['fast10','coarse4']})
    study.write(root/'plan.json',plan);original_plan=(root/'plan.json').read_bytes()
    study.write(root/'launch.json',dict(deadline_utc=(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat()))
    study.write(root/'qualification.json',dict(status='passed'))
    study.write(root/'gpu0.json',dict(state='failed',entries=[dict(scene='scene',arm='fast10',state='completed'),
        dict(scene='scene',arm='coarse4',state='failed')],error='prior failure'))
    directory=root/'runs/scene'
    study.write(directory/'goal/results.json',[{}]);(directory/'goal/seed_5.pt').write_bytes(b'reference')
    study.write(directory/'fast10/results.json',[dict(keep=True)])
    failed=directory/'coarse4';failed.mkdir();(failed/'partial.pt').write_bytes(b'old attempt')
    study.write(failed.with_suffix('.status.json'),dict(state='failed'))
    failed.with_suffix('.log').write_text('original error')
    monkeypatch.setattr(study,'command',lambda *a,**kw:['test'])
    def execute(plan,cmd,out,deadline):
        assert out==failed and not out.exists()
        assert plan['source']==str(source)
        study.write(out/'results.json',[dict(recovered=True)])
        return dict(state='completed')
    monkeypatch.setattr(study,'execute',execute)
    monkeypatch.setattr(study,'profile',lambda *a:pytest.fail('Completed profiles must not repeat'))
    study.run(root,0,resume=True)
    state=study.read(root/'gpu0.json')
    assert state['state']=='completed' and len(state['entries'])==3
    assert state['entries'][1]['state']=='failed' and state['entries'][2]['state']=='completed'
    from pathlib import Path
    backup=Path(state['recovery_backup'])
    assert (backup/'scene/coarse4/partial.pt').read_bytes()==b'old attempt'
    assert (backup/'scene/coarse4.log').read_text()=='original error'
    assert study.read(backup/'previous_status.json')['state']=='failed'
    assert study.read(directory/'fast10/results.json')==[dict(keep=True)]
    assert (root/'plan.json').read_bytes()==original_plan
    with pytest.raises(ValueError,match='stopped failed'):
        study.run(root,0,resume=True)


def test_coarse_updates_keep_the_original_draw_and_bound_actual_expanded_noise():
    batch,model,diffusion=setup(random_fps=False)
    with torch.no_grad(): initial=sample_rollout(model,diffusion,batch,future_len=4,steps=2)
    objective=body_target(3,initial['T'][:,-1,3]+torch.tensor([.01,0,0]))
    zero=noise_opt(model,diffusion,batch,objective,initial=initial,iterations=0,temporal_knots=2,future_len=4,steps=2)
    torch.testing.assert_close(zero['rollout']['positions'],initial['positions'],atol=0,rtol=0)
    changed=noise_opt(model,diffusion,batch,objective,initial=initial,iterations=2,lr=.01,
        temporal_knots=2,max_noise_rms=.003,future_len=4,steps=2)
    assert changed['loss']<changed['trace'][0]['loss']
    mask=pose_channel_mask(batch,initial['rigid_idx']).bool()
    for old,new in zip(initial['noise'],changed['rollout']['noise']):
        delta=new[1]-old[1]
        assert float(delta[...,mask].square().mean().sqrt())<=.003001
        assert torch.count_nonzero(delta[...,~mask])==0
        if delta.shape[1]>2:
            torch.testing.assert_close(delta[:,2:]-2*delta[:,1:-1]+delta[:,:-2],torch.zeros_like(delta[:,2:]),atol=3e-7,rtol=0)


def test_supplied_action_cli_rejects_unconditional_checkpoint(monkeypatch):
    import sys
    from types import SimpleNamespace
    import inference
    from guidance.run_guidance import main

    monkeypatch.setattr(inference, 'load_model_and_diffusion', lambda *args:
                        (None, None, SimpleNamespace(train={'action_dropout': 1.0}), 25))
    monkeypatch.setattr(sys, 'argv', ['run_guidance', '--checkpoint', 'unused',
                        '--domain', 'pickup', '--out', 'unused.json'])
    with pytest.raises(ValueError, match='guidance.run_unconditional'):
        main()


def test_actuator_first_releases_passive_noise_only_after_the_requested_updates(monkeypatch):
    import guidance.unconditional as module
    batch,model,diffusion=setup(random_fps=False)
    sampling=dict(future_len=4,steps=2)
    with torch.no_grad(): initial=sample_rollout(model,diffusion,batch,**sampling)
    seen=[]
    def observe(*args,**kwargs):
        seen.append([(a.detach().clone(),b.detach().clone()) for a,b in kwargs['noise']])
        return sample_rollout(*args,**kwargs)
    monkeypatch.setattr(module,'sample_rollout',observe)
    target=initial['T'][:,-1,3]+torch.tensor([.01,0,0])
    result=noise_opt(model,diffusion,batch,body_target(3,target),initial=initial,
        iterations=3,actuator_updates=1,scope='actuator_first',lr=.01,**sampling)
    mask=pose_channel_mask(batch,initial['rigid_idx']).bool()
    passive=mask & ~batch.pose_action_mask[0,:,None]
    actuated=mask & batch.pose_action_mask[0,:,None]
    assert len(seen)==4
    for original,first,later in zip(initial['noise'],seen[1],seen[2]):
        assert torch.count_nonzero((first[1]-original[1])[...,passive])==0
        assert torch.count_nonzero((first[1]-original[1])[...,actuated])>0
        assert torch.count_nonzero((later[1]-original[1])[...,passive])>0
        assert torch.count_nonzero((later[1]-original[1])[...,~mask])==0
    assert result['loss']<result['trace'][0]['loss']


def test_coarse_plus_residual_preserves_draw_and_bounds_both_components():
    batch,model,diffusion=setup(random_fps=False)
    sampling=dict(future_len=4,steps=2)
    with torch.no_grad(): initial=sample_rollout(model,diffusion,batch,**sampling)
    objective=body_target(3,initial['T'][:,-1,3]+torch.tensor([.01,0,0]))
    zero=noise_opt(model,diffusion,batch,objective,initial=initial,iterations=0,
        temporal_knots=2,residual_max_rms=.001,**sampling)
    torch.testing.assert_close(zero['rollout']['positions'],initial['positions'],rtol=0,atol=0)
    changed=noise_opt(model,diffusion,batch,objective,initial=initial,iterations=2,
        temporal_knots=2,residual_max_rms=.001,max_noise_rms=.003,lr=.01,**sampling)
    assert changed['loss']<changed['trace'][0]['loss']
    assert all(row['residual_noise_rms']<=.001001 for row in changed['trace'])
    assert changed['trace'][-1]['residual_noise_rms']>0
    mask=pose_channel_mask(batch,initial['rigid_idx']).bool()
    for original,selected in zip(initial['noise'],changed['rollout']['noise']):
        delta=selected[1]-original[1]
        assert float(delta[...,mask].square().mean().sqrt())<=.003001
        assert torch.count_nonzero(delta[...,~mask])==0
        torch.testing.assert_close(original[0],selected[0],rtol=0,atol=0)


def test_zero_actuator_phase_matches_all_body_optimizer_exactly():
    batch,model,diffusion=setup(random_fps=False)
    sampling=dict(future_len=4,steps=2)
    with torch.no_grad(): initial=sample_rollout(model,diffusion,batch,**sampling)
    objective=body_target(3,initial['T'][:,-1,3]+torch.tensor([.01,0,0]))
    ref=noise_opt(model,diffusion,batch,objective,initial=initial,iterations=2,**sampling)
    result=noise_opt(model,diffusion,batch,objective,initial=initial,iterations=2,
        scope='actuator_first',actuator_updates=0,**sampling)
    torch.testing.assert_close(result['rollout']['positions'],ref['rollout']['positions'],rtol=0,atol=0)


def test_cosine_noise_optimizer_preserves_scope_and_improves_goal(monkeypatch):
    batch,model,diffusion=setup(random_fps=False)
    sampling=dict(future_len=4,steps=2)
    with torch.no_grad(): initial=sample_rollout(model,diffusion,batch,**sampling)
    objective=body_target(3,initial['T'][:,-1,3]+torch.tensor([.005,0,0]))
    norms=[]
    step=torch.optim.Adam.step
    def observe(self,*args,**kwargs):
        norms.append(float(torch.stack([p.grad.norm() for group in self.param_groups for p in group['params']]).norm()))
        return step(self,*args,**kwargs)
    monkeypatch.setattr(torch.optim.Adam,'step',observe)
    out=noise_opt(model,diffusion,batch,objective,initial=initial,iterations=4,lr=.01,
        scope='actuated',max_noise_rms=.008,lr_schedule='cosine',warmup_updates=2,
        normalize_gradient=True,**sampling)
    assert norms==pytest.approx([1.]*4,abs=1e-6)
    assert [r['learning_rate'] for r in out['trace'][:-1]]==pytest.approx([.005,.01,.01,.001])
    assert out['loss']<out['trace'][0]['loss']
    passive=~batch.pose_action_mask[0,initial['rigid_idx']]
    for before,after in zip(initial['noise'],out['rollout']['noise']):
        torch.testing.assert_close(before[1][:,:,passive],after[1][:,:,passive],rtol=0,atol=0)


def test_noise_learning_rate_rejects_unused_warmup_and_invalid_budget():
    with pytest.raises(ValueError,match='Warmup requires'):
        noise_learning_rate(.03,0,10,warmup_updates=2)
    with pytest.raises(ValueError,match='budget'):
        noise_learning_rate(.03,0,2,schedule='cosine',warmup_updates=3)
    assert noise_learning_rate(.03,0,1,schedule='cosine')==.03


def test_saved_optimizer_prefix_matches_an_independent_short_run():
    batch,model,diffusion=setup(random_fps=True)
    sampling=dict(future_len=4,steps=2)
    with torch.no_grad(): initial=sample_rollout(model,diffusion,batch,**sampling)
    objective=body_target(3,initial['T'][:,-1,3]+torch.tensor([.01,0,0]))
    saved={}
    def checkpoint(update,best):
        if update==1: saved.update(best)
    noise_opt(model,diffusion,batch,objective,initial=initial,iterations=3,
        checkpoint_callback=checkpoint,**sampling)
    short=noise_opt(model,diffusion,batch,objective,initial=initial,iterations=1,**sampling)
    assert saved['iteration']==short['iteration'] and saved['loss']==short['loss']
    assert not saved['rollout']['positions'].requires_grad
    torch.testing.assert_close(saved['rollout']['positions'],short['rollout']['positions'],rtol=0,atol=0)


def test_fresh_routing_changes_only_rng_and_replays_without_redrawing_noise():
    from guidance.unconditional import fresh_routing_states
    batch,model,diffusion=setup(random_fps=True)
    sampling=dict(future_len=4,steps=2)
    with torch.no_grad(): original=sample_rollout(model,diffusion,batch,**sampling)
    before=torch.get_rng_state().clone()
    states=fresh_routing_states(len(original['noise']),981,batch.device)
    assert torch.equal(before,torch.get_rng_state())
    assert not torch.equal(states[0][0],states[1][0])
    assert not torch.equal(states[0][0],original['rng_states'][0][0])
    with torch.no_grad():
        changed=sample_rollout(model,diffusion,batch,noise=original['noise'],rng_states=states,**sampling)
        repeated=sample_rollout(model,diffusion,batch,noise=original['noise'],rng_states=states,**sampling)
    assert torch.equal(before,torch.get_rng_state())
    for original_window,new_window in zip(original['noise'],changed['noise']):
        for original_array,new_array in zip(original_window,new_window):
            torch.testing.assert_close(original_array,new_array,rtol=0,atol=0)
    torch.testing.assert_close(changed['positions'],repeated['positions'],rtol=0,atol=0)
    torch.testing.assert_close(changed['positions'][:,:batch.context_len],batch.context,rtol=0,atol=0)


def test_detached_context_keeps_forward_exact_but_removes_terminal_credit_to_earlier_windows():
    batch,model,diffusion=setup(random_fps=False)
    sampling=dict(future_len=4,steps=2)
    with torch.no_grad(): initial=sample_rollout(model,diffusion,batch,**sampling)
    noise=[(p.clone(),r.clone().requires_grad_()) for p,r in initial['noise']]
    full=sample_rollout(model,diffusion,batch,noise=noise,rng_states=initial['rng_states'],**sampling)
    cut=sample_rollout(model,diffusion,batch,noise=noise,rng_states=initial['rng_states'],detach_context=True,**sampling)
    torch.testing.assert_close(full['positions'],cut['positions'],rtol=0,atol=0)
    gfull=torch.autograd.grad(full['T'][0,-1,3,0],[z[1] for z in noise])
    gcut=torch.autograd.grad(cut['T'][0,-1,3,0],[z[1] for z in noise],retain_graph=True)
    assert float(gfull[0].norm())>1e-7
    assert torch.count_nonzero(gcut[0])==0
    assert float(gcut[-1].norm())>1e-7
    # A cost on the earlier output still differentiates into that window.
    local=torch.autograd.grad(cut['T'][0,batch.context_len+1,3,0],noise[0][1])[0]
    assert float(local.norm())>1e-7


def test_goal_only_cli_requires_neither_mesh_nor_grasp_and_supports_single_window(tmp_path,monkeypatch):
    import json
    import sys
    import numpy as np
    from omegaconf import OmegaConf
    import guidance.run_unconditional as cli
    batch,model,diffusion=setup(random_fps=False)
    checkpoint=tmp_path/'checkpoint.pt';checkpoint.write_bytes(b'loader replaced by a real small model for CLI integration test')
    input_path=tmp_path/'input.pt';torch.save(batch,input_path)
    cfg=OmegaConf.create(dict(train=dict(action_dropout=1),data=dict(future_len=4),normalization={}))
    monkeypatch.setattr(cli,'load_model_and_diffusion',lambda *_:(model,diffusion,cfg,2))
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    out=tmp_path/'output'
    monkeypatch.setattr(sys,'argv',['run_unconditional','--checkpoint',str(checkpoint),'--input',str(input_path),
        '--out',str(out),'--objective','goal','--goal-body','3','--horizon','4','--iterations','0','--seeds','123'])
    cli.main()
    protocol=json.loads((out/'protocol.json').read_text())
    assert protocol['geometry_sha256'] is None and protocol['grasp_sha256'] is None
    result=json.loads((out/'results.json').read_text())[0]
    assert result['windows']==[[batch.context_len,4,0]]
    assert result['selected']['physical_diagnostics']=='not requested'
    arrays=np.load(out/'seed_123.npz')
    np.testing.assert_array_equal(arrays['positions'],arrays['initial'])
    torch.manual_seed(123)
    with torch.no_grad(): expected=sample_rollout(model,diffusion,batch,future_len=4,horizon=4,steps=2)
    np.testing.assert_array_equal(arrays['positions'],expected['positions'].numpy())


def test_plain_goal_cli_ignores_grasp_proposals_and_audits_without_changing_selection(tmp_path,monkeypatch):
    import json
    import sys
    import numpy as np
    from omegaconf import OmegaConf
    import guidance.run_unconditional as cli
    from tests.test_grasp_guidance import calibrated_scene
    from tests.test_guidance_waypoints import articulated_calibrated_scene
    from utils.reproducibility import file_sha256
    batch,grasp=articulated_calibrated_scene();_,geometry,_,_,_=calibrated_scene()
    _,model,diffusion=setup(random_fps=False)
    checkpoint=tmp_path/'checkpoint.pt';checkpoint.write_bytes(b'small model loader replaced')
    inp=tmp_path/'input.pt';torch.save(batch,inp)
    geometry['metadata']['input_hashes']={str(inp.resolve()):file_sha256(inp)}
    geom=tmp_path/'geometry.pt';torch.save(geometry,geom)
    metadata=dict(input_hashes={str(inp.resolve()):file_sha256(inp),str(geom.resolve()):file_sha256(geom)})
    cfg=OmegaConf.create(dict(train=dict(action_dropout=1),data=dict(future_len=4),normalization={}))
    monkeypatch.setattr(cli,'load_model_and_diffusion',lambda *_:(model,diffusion,cfg,2))
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    arrays=[]
    for i,proposals in enumerate((None,[{}])):
        artifact=dict(calibration=grasp['calibration'],metadata=metadata)
        if proposals is not None: artifact['proposals']=proposals
        calibration=tmp_path/f'calibration_{i}.pt';torch.save(artifact,calibration)
        out=tmp_path/f'out_{i}'
        monkeypatch.setattr(sys,'argv',['run_unconditional','--checkpoint',str(checkpoint),'--input',str(inp),
            '--geometry',str(geom),'--calibration',str(calibration),'--out',str(out),
            '--objective','goal','--horizon','4','--hold-frames','2','--iterations','0','--seeds','123'])
        cli.main()
        protocol=json.loads((out/'protocol.json').read_text())
        result=json.loads((out/'results.json').read_text())[0]
        assert protocol['stage_weights'] is None
        assert result['validation'] is not None
        assert result['selection_status'].startswith('selected_by_objective;')
        arrays.append(np.load(out/'seed_123.npz')['positions'])
    np.testing.assert_array_equal(*arrays)
