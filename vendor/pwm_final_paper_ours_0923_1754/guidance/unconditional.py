"""Unconditional rigid-scene planning through the existing differentiable sampler.

Optimize initial pose noise over a complete windowed rollout. Future actions are
hidden, generated actuator poses are retained, and gradients cross window
boundaries. Random FPS is replayed per window while evaluating optimizer iterates.
This is a separate interface from planners over supplied actuator actions.
"""
from __future__ import annotations

from contextlib import contextmanager
import math

import torch

from guidance.objectives import materialize
from guidance.planners import window_nfe
from guidance.sampler import draw_noise, sample_window
from guidance.pick_validation import selection_key
from inference import rollout_window, window_bounds
from utils.conditioning import hide_actions, model_metadata
from utils.pose_native import pose_channel_mask


def _check_model(model, batch):
    cfg = model_metadata(model).cfg
    if not cfg.pose_native or cfg.unet_actions or not hasattr(model_metadata(model), "unet_cfg"):
        raise ValueError("Unconditional guidance requires a pose-native UNet with unet_actions=false")
    if batch.body_type is None or not bool((batch.body_type == 1).all()):
        raise ValueError("This planner currently supports entirely rigid scenes")
    if batch.action_active is not None:
        raise ValueError("Pose-only hiding cannot hide temporal particle actions")


def _rng_state(device):
    return (torch.get_rng_state(), torch.cuda.get_rng_state(device) if device.type == "cuda" else None)


def fresh_routing_states(windows, seed, device):
    """Independent per-window FPS streams without drawing or changing model noise.

    Explicit generators leave the caller's CPU/CUDA RNG untouched. Use these
    with saved rollout noise to audit dependence on representative-point choices.
    """
    device=torch.device(device)
    states=[]
    for i in range(windows):
        value=seed+i*1_000_003
        cpu=torch.Generator(device='cpu').manual_seed(value).get_state()
        gpu=torch.Generator(device=device).manual_seed(value).get_state() if device.type=='cuda' else None
        states.append((cpu,gpu))
    return states


@contextmanager
def _replay_rng(state, device):
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.set_rng_state(state[0].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state(state[1].cpu(), device)
        yield


def sample_rollout(model, diffusion, batch, *, future_len: int, steps: int,
                   horizon: int | None = None, schedule: str = "pyramid",
                   uncertainty_scale: float = 1.0, noise=None, rng_states=None,
                   use_checkpoint: bool = True, guidance_objective=None,
                   guidance_scale=5., guidance_threshold=.35, guidance_max_rms=.03,
                   known_pose=None,known_mask=None,detach_context=False):
    """Production window semantics, with gradients and optional replayable noise.

    With no supplied noise, draws in production order, including FPS between
    windows. Returns full context+future tracks, the drawn noise and FPS RNG states.
    With supplied noise, rng_states must also be supplied; caller RNG is preserved.
    detach_context is a gradient-only ablation at window boundaries. It preserves
    the forward trajectory and direct objective gradients on earlier outputs.
    """
    _check_model(model, batch)
    C = batch.context_len
    H = batch.num_future_frames if horizon is None else int(horizon)
    if H < 1 or H > batch.num_future_frames or future_len < 1:
        raise ValueError("Invalid rollout horizon or window length")
    if (known_pose is None)!=(known_mask is None):
        raise ValueError('Supply both the whole-horizon actuator deltas and their mask')
    if known_pose is not None:
        if known_pose.shape!=(batch.batch_size,H,batch.body_type.shape[1],6):
            raise ValueError('Known actuator deltas must cover the whole requested horizon')
        known_mask=known_mask.to(device=batch.device,dtype=torch.bool).expand_as(known_pose)
        if bool((known_mask & ~batch.pose_action_mask[:,None,:,None]).any()):
            raise ValueError('This inpainting interface only prescribes actuated bodies')
    if (noise is None) != (rng_states is None):
        raise ValueError("Supply both noise and its per-window RNG states")
    expected = (H + future_len - 1) // future_len
    if noise is not None and (len(noise) != expected or len(rng_states) != expected):
        raise ValueError("Noise/RNG window counts do not match the horizon")
    ridx = (batch.body_type[0] == 1).nonzero(as_tuple=True)[0]
    pos = batch.positions[:, :C].clone()
    R = batch.body_poses[:, :C, ridx, :3, :3].float().clone()
    t = batch.body_poses[:, :C, ridx, :3, 3].float().clone()
    drawn, states, windows, pose_outputs, nfe, guidance_traces = [], [], [], [], 0, []
    while pos.shape[1] < C + H:
        start, length, skip = window_bounds(pos.shape[1], C + H, future_len, C)
        win = rollout_window(batch, pos.detach() if detach_context else pos, start, length, rigid_idx=ridx,
            R_track=R.detach() if detach_context else R,T_track=t.detach() if detach_context else t)
        i = len(windows)
        local_trace=[]
        def local_objective(particles,poses,ready_frames):
            candidate=materialize(win,particles,poses)
            stop=max(skip,ready_frames)
            prefix=dict(positions=torch.cat([pos[:,C:],candidate['positions'][:,skip:stop]],dim=1),
                R=torch.cat([R[:,C:],candidate['R'][:,skip:stop]],dim=1),
                t=torch.cat([t[:,C:],candidate['t'][:,skip:stop]],dim=1),rigid_idx=ridx)
            return guidance_objective.prefix_loss(prefix,complete=prefix['positions'].shape[1]==H)
        guidance=dict(guidance_fn=local_objective,guidance_scale=guidance_scale,
            guidance_threshold=guidance_threshold,guidance_max_rms=guidance_max_rms,guidance_trace=local_trace) if guidance_objective is not None else {}
        if known_pose is not None:
            guidance.update(known_pose=known_pose[:,start-C:start-C+length],
                known_mask=known_mask[:,start-C:start-C+length])
        if noise is None:
            z = draw_noise(model, win)
            state = _rng_state(batch.device)
            xp, xr = sample_window(model, diffusion, hide_actions(win, 1.0), z,
                steps=steps, schedule=schedule, uncertainty_scale=uncertainty_scale,
                use_checkpoint=use_checkpoint,**guidance)
        else:
            z, state = noise[i], rng_states[i]
            if z[0].shape != win.future.shape or z[1].shape != (batch.batch_size, length, len(ridx), 6):
                raise ValueError("Noise shape does not match its rollout window")
            with _replay_rng(state, batch.device):
                xp, xr = sample_window(model, diffusion, hide_actions(win, 1.0), z,
                    steps=steps, schedule=schedule, uncertainty_scale=uncertainty_scale,
                    use_checkpoint=use_checkpoint,**guidance)
        pred = materialize(win, xp, xr)
        pos = torch.cat([pos, pred["positions"][:, skip:]], dim=1)
        R = torch.cat([R, pred["R"][:, skip:]], dim=1)
        t = torch.cat([t, pred["t"][:, skip:]], dim=1)
        pose_outputs.append(xr[:, skip:])
        drawn.append(z)
        states.append(state)
        windows.append((start, length, skip))
        guidance_traces.extend(dict(window=i,**row) for row in local_trace)
        nfe += window_nfe(diffusion, win, steps, schedule, uncertainty_scale)
    return dict(positions=pos, R=R, T=t, rigid_idx=ridx, pose_deltas=torch.cat(pose_outputs, dim=1),
                noise=drawn, rng_states=states, windows=windows, horizon=H, nfe=nfe,guidance_trace=guidance_traces)


def prediction(rollout, context_len):
    """The future-only, physical-unit objective interface shared with planners.py."""
    return dict(positions=rollout["positions"][:, context_len:],
                R=rollout["R"][:, context_len:], t=rollout["T"][:, context_len:],
                rigid_idx=rollout["rigid_idx"])


def _detach(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, dict):
        return {k: _detach(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_detach(v) for v in value)
    return value


def expand_noise_update(update,frames):
    """Interpolate an update in time while keeping the original Gaussian draw."""
    if update.shape[1]==frames: return update
    # CUDA linear-upsampling backward rejects deterministic mode. Indexed
    # interpolation has the same align_corners=True coordinates and supports
    # deterministic accumulation into the temporal knots.
    knots=update.shape[1]
    coordinate=torch.arange(frames,device=update.device,dtype=update.dtype)
    coordinate=coordinate*((knots-1)/(frames-1) if frames>1 else 0.)
    left=coordinate.floor().long().clamp(max=knots-1)
    right=(left+1).clamp(max=knots-1)
    fraction=(coordinate-left).reshape(1,frames,1,1)
    return update.index_select(1,left)*(1-fraction)+update.index_select(1,right)*fraction


def noise_learning_rate(lr, update, iterations, *, schedule='constant', warmup_updates=0,
                        min_ratio=.1):
    """Actual-update schedule; cosine includes an explicit nonzero final rate."""
    if schedule not in ('constant','cosine') or not 0<=min_ratio<=1:
        raise ValueError('Invalid noise learning-rate schedule')
    if not isinstance(warmup_updates,int) or not 0<=warmup_updates<=iterations:
        raise ValueError('Warmup updates must lie within the optimization budget')
    if schedule=='constant':
        if warmup_updates: raise ValueError('Warmup requires the cosine schedule')
        return lr
    if update<warmup_updates:
        return lr*(update+1)/warmup_updates
    progress=(update-warmup_updates)/max(1,iterations-warmup_updates-1)
    return lr*(min_ratio+(1-min_ratio)*.5*(1+math.cos(math.pi*min(1.,progress))))


def noise_opt(model, diffusion, batch, objective, *, iterations: int = 6, lr: float = 0.03,
              scope: str = "all", max_noise_rms: float = 0.25, prior_weight: float = 0.001,
              initial=None, callback=None, validator=None, temporal_knots=0,
              checkpoint_callback=None,
              actuator_updates=0, residual_max_rms=0., lr_schedule='constant',
              warmup_updates=0, lr_min_ratio=.1, normalize_gradient=False, **sampling):
    """Adam over masked initial pose noise, with a trust region around each draw.

    scope='actuated' holds passive initial noise fixed; the network still predicts
    passive motion jointly. This is an optimization restriction, not a dynamics
    guarantee. Only B=1 is supported so selecting the best iterate is scene-wise.
    iterations counts actual updates; the final updated iterate is evaluated.
    nfe counts forward sampling calls; backward/recomputation is extra compute.
    actuator_first masks passive gradients for actuator_updates, then releases
    them. Its prior and total RMS use all valid channels throughout both phases.
    A bounded full-frame residual can supplement coarse temporal updates. Both
    components share the total trust bound around the original Gaussian draw.
    """
    _check_model(model, batch)
    if batch.batch_size != 1:
        raise ValueError("Optimize one scene/draw at a time (batch_size=1)")
    if iterations < 0 or lr < 0 or max_noise_rms <= 0 or prior_weight < 0:
        raise ValueError("Invalid optimizer settings")
    noise_learning_rate(lr,0,iterations,schedule=lr_schedule,warmup_updates=warmup_updates,min_ratio=lr_min_ratio)
    if scope not in ("all", "actuated", "actuator_first"):
        raise ValueError("Noise scope must be all, actuated or actuator_first")
    if not isinstance(actuator_updates,int) or not 0<=actuator_updates<=iterations:
        raise ValueError('Actuator update count must lie within the optimization budget')
    if scope!='actuator_first' and actuator_updates:
        raise ValueError('Actuator update count requires actuator_first scope')
    if residual_max_rms<0 or (residual_max_rms and not temporal_knots):
        raise ValueError('A nonnegative residual bound requires coarse temporal controls')
    if isinstance(temporal_knots,bool) or not isinstance(temporal_knots,int) or temporal_knots<0:
        raise ValueError('Temporal knot count must be a nonnegative integer')
    if any(p.requires_grad for p in model.parameters()):
        raise ValueError("Freeze model parameters before noise optimization")
    nfe = 0
    if initial is None:
        with torch.no_grad():
            initial = sample_rollout(model, diffusion, batch, **sampling)
        nfe += initial["nfe"]
    noise = _detach(initial["noise"])
    ridx = initial["rigid_idx"]
    mask = pose_channel_mask(batch, ridx)
    mask = torch.ones(len(ridx), 6, device=batch.device) if mask is None else mask.to(batch.device)
    if scope == "actuated":
        mask = mask * batch.pose_action_mask[:, ridx, None].to(mask)
    mask = mask.expand_as(noise[0][1][:, :1]).bool()
    if not bool(mask.any()):
        raise ValueError("No active pose channels to optimize")
    actuator_mask=None
    if scope=='actuator_first':
        if batch.pose_action_mask is None:
            raise ValueError('Actuator-first optimization requires an actuator mask')
        actuator_mask=mask & batch.pose_action_mask[:,None,ridx,None].bool()
        if not bool(actuator_mask.any()):
            raise ValueError('No actuated pose channels for actuator-first optimization')
    change = [torch.zeros_like(z[1][:,:min(temporal_knots,z[1].shape[1])],requires_grad=True)
              if temporal_knots else torch.zeros_like(z[1],requires_grad=True) for z in noise]
    residual=[torch.zeros_like(z[1],requires_grad=True) for z in noise] if residual_max_rms else []
    parameters=change+residual
    opt = torch.optim.Adam(parameters, lr=lr)
    best, trace, best_key = None, [], None
    for update in range(iterations + 1):
        opt.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(update < iterations):
            expanded=[expand_noise_update(dz,z[1].shape[1]) for z,dz in zip(noise,change)]
            if residual: expanded=[coarse+fine for coarse,fine in zip(expanded,residual)]
            current = [(z[0], z[1] + dz * mask) for z, dz in zip(noise, expanded)]
            out = sample_rollout(model, diffusion, batch, noise=current,
                                 rng_states=initial["rng_states"], **sampling)
            task_loss, diag = objective(prediction(out, batch.context_len), batch)
            squared = torch.cat([dz.masked_select(mask.expand_as(dz)).square() for dz in expanded]).mean()
            loss = task_loss + prior_weight * squared
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite guidance objective")
            nfe += out["nfe"]
            row = dict(iteration=update, loss=float(loss.detach()), task_loss=float(task_loss.detach()),
                       noise_rms=float(squared.detach().sqrt()), **diag)
            row['noise_rms_by_window']=[float(dz.masked_select(mask.expand_as(dz)).square().mean().detach().sqrt()) for dz in expanded]
            if scope=='actuator_first':
                row['update_scope']='actuated' if update<actuator_updates else 'all'
            if residual:
                row['residual_noise_rms']=float(torch.cat([r.masked_select(mask.expand_as(r)).square() for r in residual]).mean().detach().sqrt())
            validation = validator(prediction(out,batch.context_len)) if validator is not None else None
            comparable = float(diag['selection_loss'])+prior_weight*float(squared.detach()) if 'selection_loss' in diag else row['loss']
            row['selection_loss'] = comparable
            key = selection_key(comparable,validation)
            if validation is not None: row['validation'] = validation
            if best is None or key < best_key:
                best_key = key
                best = dict(rollout=_detach(out), loss=comparable, diagnostics=diag, iteration=update,
                            validation=validation)
            if update < iterations:
                rate=noise_learning_rate(lr,update,iterations,schedule=lr_schedule,
                    warmup_updates=warmup_updates,min_ratio=lr_min_ratio)
                for group in opt.param_groups: group['lr']=rate
                row['learning_rate']=rate
                grads = torch.autograd.grad(loss, parameters)
                if scope=='actuator_first' and update<actuator_updates:
                    grads=tuple(g*actuator_mask for g in grads)
                if not all(bool(torch.isfinite(g).all()) for g in grads):
                    raise FloatingPointError("Nonfinite noise gradient")
                row["gradient_norm"] = float(torch.stack([g.norm() for g in grads]).norm())
                row['gradient_norm_by_window']=[float(torch.stack([g.norm() for g in grads[i::len(noise)]]).norm()) for i in range(len(noise))]
                if normalize_gradient:
                    # One norm over the complete active trajectory (B=1), not
                    # independent window norms that change their relative credit.
                    norm=torch.stack([g.norm() for g in grads]).norm().clamp_min(1e-12)
                    grads=tuple(g/norm for g in grads)
                for dz, g in zip(parameters, grads):
                    dz.grad = g
                opt.step()
                with torch.no_grad():
                    for i,(z,dz) in enumerate(zip(noise,change)):
                        dz.mul_(mask)
                        if residual:
                            fine=residual[i]; fine.mul_(mask)
                            fine_rms=fine.masked_select(mask.expand_as(fine)).square().mean().sqrt()
                            fine.mul_((residual_max_rms/fine_rms.clamp_min(1e-12)).clamp(max=1.))
                        actual=expand_noise_update(dz,z[1].shape[1])
                        if residual: actual=actual+residual[i]
                        rms = actual.masked_select(mask.expand_as(actual)).square().mean().sqrt()
                        scale=(max_noise_rms / rms.clamp_min(1e-12)).clamp(max=1.0)
                        dz.mul_(scale)
                        if residual: residual[i].mul_(scale)
                if hasattr(objective,'after_step'):
                    objective.after_step(diag,update)
        trace.append(row)
        if callback is not None:
            callback(row)
        if checkpoint_callback is not None:
            # best contains detached tensors. A persistence callback can retain
            # selected prefixes of a long run without another sampling pass.
            checkpoint_callback(update,best)
    return dict(**best, trace=trace, nfe=nfe, updates=iterations, scope=scope,
                max_noise_rms=max_noise_rms, prior_weight=prior_weight,temporal_knots=temporal_knots,
                actuator_updates=actuator_updates,residual_max_rms=residual_max_rms,
                lr_schedule=lr_schedule,warmup_updates=warmup_updates,lr_min_ratio=lr_min_ratio,
                normalize_gradient=normalize_gradient)


def best_of_k(model, diffusion, batch, objective, *, k: int = 8, callback=None, validator=None,
              candidate_callback=None, **sampling):
    """Select one complete generated interaction, keeping its hand and object."""
    if k < 1 or batch.batch_size != 1:
        raise ValueError("best_of_k requires k>=1 and batch_size=1")
    best, trace, nfe, best_key = None, [], 0, None
    with torch.no_grad():
        for draw in range(k):
            out = sample_rollout(model, diffusion, batch, **sampling)
            loss, diag = objective(prediction(out, batch.context_len), batch)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite candidate objective")
            row = dict(draw=draw, loss=float(loss), **diag)
            validation = validator(prediction(out,batch.context_len)) if validator is not None else None
            key = selection_key(row['loss'],validation)
            if validation is not None: row['validation'] = validation
            trace.append(row)
            nfe += out["nfe"]
            if best is None or key < best_key:
                best_key = key
                best = dict(rollout=out, loss=row["loss"], diagnostics=diag, draw=draw, validation=validation)
            if callback is not None:
                callback(row)
            if candidate_callback is not None:
                candidate_callback(draw,out,row)
    return dict(**best, trace=trace, nfe=nfe)


def select_then_refine(model,diffusion,batch,objective,*,k=8,iterations=3,validator=None,callback=None,
                       candidate_callback=None,**kwargs):
    """Keep one generated interaction from a bank, then refine its own saved noise."""
    opt_keys={'lr','scope','max_noise_rms','prior_weight','temporal_knots','actuator_updates','residual_max_rms',
              'lr_schedule','warmup_updates','lr_min_ratio','normalize_gradient'}
    sampling={k:v for k,v in kwargs.items() if k not in opt_keys}
    opt={k:v for k,v in kwargs.items() if k in opt_keys}
    emit=lambda phase: (lambda row:callback(dict(phase=phase,**row))) if callback is not None else None
    selected=best_of_k(model,diffusion,batch,objective,k=k,validator=validator,callback=emit('selection'),
                       candidate_callback=candidate_callback,**sampling)
    refined=noise_opt(model,diffusion,batch,objective,initial=selected['rollout'],iterations=iterations,
                      validator=validator,callback=emit('refinement'),**opt,**sampling)
    refined['selection_trace']=selected['trace']
    refined['initial_draw']=selected['draw']
    refined['nfe']+=selected['nfe']
    return refined


def multistart(model,diffusion,batch,objective,*,starts=2,iterations=2,validator=None,callback=None,**kwargs):
    """Independent initial draws; each retains its own FPS replay and gradient path."""
    if starts<1: raise ValueError('At least one optimization start is required')
    best,best_key,total,trace=None,None,0,[]
    for start in range(starts):
        emit=(lambda row:callback(dict(start=start,**row))) if callback is not None else None
        result=noise_opt(model,diffusion,batch,objective,iterations=iterations,validator=validator,callback=emit,**kwargs)
        key=selection_key(result['loss'],result['validation'])
        total+=result['nfe']
        trace.append(dict(start=start,loss=result['loss'],validation=result['validation'],trace=result['trace']))
        if best is None or key<best_key:
            best,best_key=result,key
            best['start']=start
    best['nfe']=total; best['start_traces']=trace
    return best
