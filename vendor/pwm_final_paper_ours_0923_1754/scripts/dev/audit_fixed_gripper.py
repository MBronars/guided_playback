"""Resample passive noise while inpainting a previously generated actuator path.

This is a model-consistency diagnostic, not a physics replay or an exact
conditional sampler. The inpainted reference itself changes the denoising path.
"""
import argparse
import json
from pathlib import Path
import time
import traceback

import torch

from guidance.feedback import rigid_planning_batch
from guidance.unconditional import sample_rollout
from inference import load_model_and_diffusion
from scripts.dev.guidance_report_utils import task_validator, rollout_metrics
from utils.pose_native import pose_channel_mask
from utils.reproducibility import file_sha256


def passive_redraw(noise, passive_mask, seed):
    """Redraw valid passive pose channels, preserving actuator and routing noise."""
    generator = torch.Generator(device=noise[0][1].device).manual_seed(seed)
    return [(particle.clone(), torch.where(passive_mask,
        torch.randn(pose.shape, device=pose.device, dtype=pose.dtype, generator=generator), pose))
        for particle, pose in noise]


def cpu_tracks(rollout):
    return {key: rollout[key].detach().cpu() for key in
            ('positions', 'R', 'T', 'pose_deltas', 'rigid_idx')}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--seed', type=int, default=8311)
    p.add_argument('--redraw-seeds', default='9301,9302,9303,9304,9305,9306,9307,9308')
    args = p.parse_args(); out = args.out.resolve(); out.mkdir(parents=True, exist_ok=False)
    protocol_path = args.run/'protocol.json'; protocol = json.loads(protocol_path.read_text())
    source = args.run/f'seed_{args.seed}.pt'
    saved = torch.load(source, map_location='cpu', weights_only=False)
    selected = saved['result']['rollout']; sampling = protocol['sampling']
    report = dict(status='running', checks=[], variants=[], sampling=sampling,
        source_sha256=file_sha256(source), protocol_sha256=file_sha256(protocol_path),
        source_root=str(Path(__file__).resolve().parents[2]),
        interpretation='Fixed generated actuator path, passive-noise resampling with inpainting. '
            'Not an exact conditional distribution, causal intervention, or independent physics validation.')
    start = time.monotonic()
    def save():
        report['seconds'] = time.monotonic()-start
        temporary = out/'report.pending.json'
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        temporary.replace(out/'report.json')
    def check(name, passed, **details):
        row = dict(name=name, passed=bool(passed), **details); report['checks'].append(row)
        print(json.dumps(row), flush=True); save()
        if not passed: raise AssertionError(name)
    try:
        if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
        batch, geometry, collision, validator = task_validator(protocol)
        C = batch.context_len
        batch = rigid_planning_batch(batch, horizon=sampling['horizon']).to('cuda')
        checkpoint = protocol['arguments']['checkpoint']
        check('checkpoint matches planned trajectory', file_sha256(checkpoint)==protocol['checkpoint_sha256'])
        model, diffusion, cfg, steps = load_model_and_diffusion(checkpoint, torch.device('cuda'))
        for parameter in model.parameters(): parameter.requires_grad_(False)
        check('unconditional bidirectional checkpoint', float(cfg.train.action_dropout)==1 and bool(cfg.model.unet_bidirectional_temporal))
        hidden = [True]
        def hook(_, positional, kwargs):
            given = positional[0]
            hidden[0] &= not bool(given.pose_action_mask.any() or given.pose_actions.any())
        handle = model.register_forward_pre_hook(hook, with_kwargs=True)
        noise = [(a.cuda(), z.cuda()) for a, z in selected['noise']]
        common = dict(noise=noise, rng_states=selected['rng_states'], use_checkpoint=False, **sampling)
        ridx = selected['rigid_idx'].cuda()
        valid = pose_channel_mask(batch, ridx).to(device=batch.device, dtype=torch.bool)
        actuated = batch.pose_action_mask[:, None, ridx, None]
        mask = valid & actuated; passive_mask = valid & ~actuated
        check('only valid passive channels are redrawn', bool(passive_mask.any()) and not bool((mask & passive_mask).any()))
        with torch.no_grad():
            replay = sample_rollout(model, diffusion, batch, **common)
        check('original saved sample replays bitwise before inpainting',
            all(torch.equal(replay[k].cpu(), selected[k]) for k in ('positions', 'R', 'T')))
        del replay
        rows = [('selected plan', None, selected)]
        seeds = [int(v) for v in args.redraw_seeds.split(',')]
        check('redraw seeds are unique', len(seeds)==len(set(seeds)))
        for i, seed in enumerate([None]+seeds):
            trial_noise = noise if seed is None else passive_redraw(noise, passive_mask, seed)
            if seed is not None:
                check(f'redraw {seed} preserves all non-passive inputs', all(
                    torch.equal(a, aa) and torch.equal(z[~passive_mask.expand_as(z)], zz[~passive_mask.expand_as(z)])
                    for (a,z),(aa,zz) in zip(noise,trial_noise)))
            with torch.no_grad():
                roll = sample_rollout(model, diffusion, batch, **dict(common, noise=trial_noise),
                    known_pose=selected['pose_deltas'].cuda(), known_mask=mask)
            track = cpu_tracks(roll); del roll
            body_mask = batch.pose_action_mask[0, ridx].cpu()
            dt = (track['T'][:,:,body_mask]-selected['T'][:,:,body_mask]).abs().max().item()
            dr = (track['R'][:,:,body_mask]-selected['R'][:,:,body_mask]).abs().max().item()
            check(f'inpainted path {i} preserves actuator poses across windows', dt<=2e-6 and dr<=1e-5,
                max_position_component_error_m=dt, max_rotation_matrix_error=dr)
            check(f'inpainted path {i} preserves exact valid actuator deltas',
                torch.equal(track['pose_deltas'][mask.cpu().expand_as(track['pose_deltas'])],
                            selected['pose_deltas'][mask.cpu().expand_as(track['pose_deltas'])]))
            label = 'inpainted reference' if seed is None else f'passive redraw {seed}'
            rows.append((label, seed, track))
            torch.save(track, out/f'track_{i:02d}.pt')
            metrics = rollout_metrics(track, protocol, collision, validator)
            report['variants'].append(dict(label=label, seed=seed, metrics=metrics, track=f'track_{i:02d}.pt'))
            save()
        check('future actions remain hidden at the model input', hidden[0]); handle.remove()
        check('weights remain frozen', all(v.grad is None for v in model.parameters()))
        report['selected_metrics'] = rollout_metrics(selected, protocol, collision, validator)
        report['checkpoint_sha256'] = protocol['checkpoint_sha256']
        report['status'] = 'passed'
        render(out, protocol, rows, C, report)
    except Exception:
        report['status']='failed'; report['exception']=traceback.format_exc(); raise
    finally: save()


def render(out, protocol, rows, C, report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axs = plt.subplots(1,3,figsize=(15,4.7))
    goal=np.asarray(protocol['target'][0])
    for i,(label,seed,roll) in enumerate(rows):
        track=roll['T'][0,C-1:,0].numpy(); t=np.arange(len(track))*.05
        style=dict(color='#bd2775',lw=2.5) if i==0 else dict(color='#242424',lw=2,ls='--') if i==1 else dict(color='#427fc0',lw=1,alpha=.6)
        legend=label if i<3 else None
        axs[0].plot(track[:,0],track[:,1],label=legend,**style)
        axs[1].plot(t,track[:,2],label=legend,**style)
        axs[2].plot(t,np.linalg.norm(track-goal,axis=-1)*1000,label=legend,**style)
    axs[0].scatter(goal[0],goal[1],color='#bd2775',marker='X',s=70)
    axs[0].set(xlabel='x (m)',ylabel='y (m)',title='Object origin, top view');axs[0].axis('equal')
    axs[1].axhline(goal[2],color='#bd2775',ls=':');axs[1].set(xlabel='Time (s)',ylabel='Object origin z (m)',title='Height')
    axs[2].axhline(20,color='#bd2775',ls=':');axs[2].set(xlabel='Time (s)',ylabel='Goal error (mm)',title='Target reaching')
    axs[0].legend(fontsize=8)
    for ax in axs: ax.grid(alpha=.2)
    fig.suptitle('Identical generated gripper path; different passive diffusion noise\nInpainting diagnostic within the frozen model — execution is tested separately')
    fig.tight_layout();fig.savefig(out/'paths.png',dpi=145);plt.close(fig)
    lines=['# Fixed-gripper consistency diagnostic','',report['interpretation'],'',
        'The black dashed reference uses the selected noise with actuator inpainting. '+
        'Inpainting changes intermediate denoising states, so that reference can differ from the original optimized plan.',
        '', '![Fixed actuator, resampled object](paths.png)','',
        '| Variant | Final goal error | Worst held error | Full geometry |','|---|---:|---:|---|']
    metrics=[('selected plan',report['selected_metrics'])]+[(v['label'],v['metrics']) for v in report['variants']]
    for label,m in metrics: lines.append(f'| {label} | {m["final_goal_mm"]:.1f} mm | {m["held_worst_goal_mm"]:.1f} mm | {m["geometry_pass"]} |')
    lines+=['','[Checks, seeds and hashes](report.json). No recorded future actuator path is used.','']
    (out/'README.md').write_text('\n'.join(lines))


if __name__=='__main__': main()
