"""Separate changing history errors from a matched, constant observation offset."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from scripts.dev.observation_probe_common import (Predictor, clone_batch, corrupt_observation,
    geometry_residual_mm, observed_input, write_json, write_frozen_run)
from scripts.dev.overnight_robustness import fixtures
from utils.reproducibility import code_revision, evaluation_seed, file_sha256


def constant_history_offset(reference, jitter):
    """Retain jitter's endpoint error, but preserve the true context displacements.

The jitter protocol translates each rigid object coherently, so its anchor
translation can also be applied consistently to every historical link pose.
"""
    result = clone_batch(jitter)
    C = reference.context_len
    shift = jitter.anchor-reference.anchor
    result.positions[:, :C] = reference.context+shift[:, None]
    result.positions[:, C-1] = jitter.anchor
    result.positions[:, C:] = jitter.anchor[:, None]
    if reference.body_poses is not None:
        pose_shift = jitter.body_poses[:, C-1, :, :3, 3]-reference.body_poses[:, C-1, :, :3, 3]
        result.body_poses[:, :C] = reference.body_poses[:, :C]
        result.body_poses[:, :C, :, :3, 3] += pose_shift[:, None]
        result.body_poses[:, C:] = result.body_poses[:, C-1:C]
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', required=True); ap.add_argument('--root', required=True)
    ap.add_argument('--wait-for-stability', action='store_true')
    a = ap.parse_args(); root = Path(a.root); out = root/'motion_noise'; out.mkdir(exist_ok=True)
    start = time.monotonic()
    while a.wait_for_stability:
        p = root/'stability/status.json'
        state = json.loads(p.read_text()) if p.exists() else {}
        if state.get('state') == 'finished': break
        if time.monotonic()-start > 1800:
            write_json(out/'status.json', {'state': 'not_run', 'reason': 'Stability worker wait exceeded 30 minutes'}); return
        write_json(out/'status.json', {'state': 'waiting_for_stability'})
        time.sleep(30)
    torch.set_num_threads(8); predictor = Predictor(a.checkpoint)
    batches, manifest = fixtures(Path('validation/architecture_review_2026-09-12/deep_dive'),
                                predictor.cfg, root/'cloth_fixtures/manifest.json')
    selected = [f'{d}_{i}_row0' for d in ('mpm', 'pickup', 'pushing') for i in range(4)]
    selected += [f'cloth_middle_{i}_0' for i in range(4)]
    write_frozen_run(out/'run.json', {'checkpoint': str(Path(a.checkpoint).resolve()),
        'checkpoint_sha256': file_sha256(a.checkpoint), 'code': code_revision(),
        'clips': [x for x in manifest if x['key'] in selected], 'sampler_seeds': [710001, 710002],
        'corruption_seed': 9201, 'severity_mm': 1,
        'comparison': 'Same anchor error; independent frame jitter versus constant per-particle/object offset. Constant-offset history preserves true context displacements and is a controlled synthetic counterfactual, not a deployable state estimator.'},
        ('clips','sampler_seeds','corruption_seed','severity_mm','comparison'))
    results = []
    for key in selected:
        ref = batches[key]
        jitter = corrupt_observation(ref, 'history_jitter', 1, evaluation_seed(9201, 0, key+'history_jitter'))
        fixed = constant_history_offset(ref, jitter)
        motion_delta = (fixed.context_deltas-ref.context_deltas).abs().max().item()
        assert torch.equal(fixed.anchor, jitter.anchor)
        assert motion_delta < 2e-7, motion_delta
        assert torch.equal(fixed.actions, ref.actions)
        if ref.pose_actions is not None: assert torch.equal(fixed.pose_actions, ref.pose_actions)
        assert abs(geometry_residual_mm(fixed)-geometry_residual_mm(ref)) < .002
        passive = ref.get_valid_mask()[0] & ~ref.action_mask[0]
        rows = []
        for seed in (710001, 710002):
            clean, _ = predictor(observed_input(ref), seed)
            for label, batch in (('independent_frame_jitter', jitter), ('constant_offset', fixed)):
                pred, timing = predictor(batch, seed)
                response = (pred[0, :, passive]-clean[0, :, passive]).square().sum(-1).mean().sqrt()*1000
                error = (pred[0, :, passive]-ref.future[0, :, passive]).norm(dim=-1).mean()*1000
                clean_error = (clean[0, :, passive]-ref.future[0, :, passive]).norm(dim=-1).mean()*1000
                rows.append({'kind': label, 'sampler_seed': seed, 'response_rms_mm': float(response),
                             'mean_error_mm': float(error), 'clean_mean_error_mm': float(clean_error), **timing})
        result = {'clip': key, 'domain': ref.domain, 'cases': rows,
                  'max_constant_offset_context_delta_error_m': motion_delta,
                  'anchor_exactly_shared': True, 'geometry_residual_mm': geometry_residual_mm(fixed)}
        write_json(out/(key+'.json'), result); results.append(result)
        write_json(out/'status.json', {'state': 'running', 'completed': len(results), 'planned': len(selected)})
        print('MOTION_NOISE', key, [(r['kind'], r['response_rms_mm']) for r in rows], flush=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for i, domain in enumerate(('mpm', 'pickup', 'pushing', 'cloth')):
        clips = [x for x in results if x['domain'] == domain]
        for label, shift, color, title in (('independent_frame_jitter', -.18, '#d88e37', 'Changing frame errors'),
                                          ('constant_offset', .18, '#5f8fac', 'Same offset every frame')):
            response = [np.mean([r['response_rms_mm'] for r in x['cases'] if r['kind']==label]) for x in clips]
            delta = [np.mean([r['mean_error_mm']-r['clean_mean_error_mm'] for r in x['cases'] if r['kind']==label]) for x in clips]
            for ax, values in zip(axes, (response, delta)):
                ax.bar(i+shift, np.mean(values), width=.34, color=color, label=title if i==0 else None)
                ax.scatter(i+shift+np.linspace(-.07, .07, len(values)), values, c='#293747', s=14)
    for ax in axes:
        ax.set(xticks=range(4), xticklabels=['mpm', 'pickup', 'pushing', 'cloth']); ax.axhline(0, color='#888', lw=.6)
        ax.legend(fontsize=8)
    axes[0].set_ylabel('Change in predicted trajectory, RMS (mm)')
    axes[1].set_ylabel('Increase in mean error against truth (mm)')
    fig.suptitle('Same 1 mm endpoint error; different errors in motion history | four clips/domain, two sampler seeds', fontsize=11)
    fig.savefig(out/'constant_vs_changing_history.png', dpi=160); plt.close(fig)
    write_json(out/'summary.json', {'clips': results,
        'interpretation': 'Constant offset preserves true context displacements. This isolates temporal error structure; it does not isolate a particular neural layer or show a practical denoising method.',
        'aggregation': 'Average sampler draws within clip, then average four clips per domain; dots show clips'})
    write_json(out/'status.json', {'state': 'finished', 'completed': len(results), 'planned': len(selected)})


if __name__ == '__main__': main()
