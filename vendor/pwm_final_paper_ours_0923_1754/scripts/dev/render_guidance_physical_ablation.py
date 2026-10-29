"""Compare matched 24-update plans with and without physical penalties."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.dev.guidance_report_utils import task_validator
from scripts.dev.render_guidance_surfaces import render
from utils.reproducibility import file_sha256


def load(root, key):
    export = root/'exports'/key
    replay = root/'replays'/key
    if not (replay/'audit.json').exists():
        return None
    protocol = json.loads((export/'protocol.json').read_text())
    receipt = json.loads((export/'receipt.json').read_text())
    audit = json.loads((replay/'audit.json').read_text())
    batch, geometry, collision, _ = task_validator(protocol)
    planned = np.load(export/'trajectory.npz')
    executed = np.load(replay/'dt00025'/'execution.npz')
    C = int(planned['context_len'])
    R, T = planned['R'][0, C-1:], planned['T'][0, C-1:]
    np.testing.assert_array_equal(executed['desired_R'], R)
    np.testing.assert_array_equal(executed['desired_T'], T)
    assert file_sha256(export/'trajectory.npz') == receipt['trajectory_sha256']
    ridx = (batch.body_type[0] == 1).nonzero(as_tuple=True)[0]
    distances = collision(torch.tensor(R)[None], torch.tensor(T)[None], ridx)[0].numpy()
    close = distances[:, 1:].max(-1) <= .002
    dz = np.diff(T[:, 0, 2])
    unsupported = (dz > .0005) & ~close[:-1] & ~close[1:]
    metrics = dict(receipt['metrics'], upward_without_proximity_mm=float(dz[unsupported].sum()*1000),
        both_fingers_close_fraction=float(close[1:].mean()),
        replay_worst_held_mm=max(r['held_worst_goal_mm'] for r in audit['replays']),
        paired_execution_pass=audit['qualified_task_pass'])
    return dict(key=key, protocol=protocol, geometry=geometry, metrics=metrics,
        planned=(R, T), executed=(executed['model_R'], executed['model_T']),
        hashes=dict(trajectory=file_sha256(export/'trajectory.npz'),
            execution=file_sha256(replay/'dt00025'/'execution.npz'),
            audit=file_sha256(replay/'audit.json')))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True, type=Path)
    root = p.parse_args().root.resolve()
    folder = root/'physical_ablation_figures'
    folder.mkdir(exist_ok=True)
    lines = ['# What the physical penalties change', '',
        'Both arms start from the same seed and use 24 updates, a 0.50 noise-change cap, '
        'a 36-frame horizon and the same mean held-position goal. The left arm uses the '
        'standard physical costs. The right arm scales those costs to zero, preserving '
        'position/rotation goals and the noise prior. Neither arm uses a prescribed grasp path.', '',
        'Top panels are model predictions. Bottom panels execute the corresponding gripper '
        'commands in independent physics at 0.25 ms. Every plan is also audited at 0.50 ms. '
        '[Exact initial-draw pairing checks](initial_pairing_checks.json) and '
        '[all execution criteria](EXECUTION_GALLERY.md) are reported separately.', '']
    summaries = []
    for scene in ('bottle_8270', 'teacup_7803'):
        variants = [load(root, scene+'_left_'+suffix) for suffix in ('i24_r050', 'no_physical_i24_r050')]
        lines += ['## '+scene, '']
        if any(v is None for v in variants):
            lines += ['Awaiting completed predictions and paired replays.', '']
            continue
        base, ablated = variants
        a, b = base['protocol'], ablated['protocol']
        for field in ('checkpoint_sha256', 'input_sha256', 'geometry_sha256', 'target', 'target_rotation', 'sampling'):
            assert a.get(field) == b.get(field), field
        for field in ('lr', 'max_noise_rms', 'scope', 'hold_frames', 'prior_weight', 'objective'):
            assert a['arguments'].get(field) == b['arguments'].get(field), field
        assert a['arguments'].get('goal_weight', 1.) == b['arguments'].get('goal_weight', 1.)
        assert a['arguments'].get('physical_weight', 1.) == 1.
        assert b['arguments']['physical_weight'] == 0.
        assert a['arguments'].get('goal_reduction', 'mean') == b['arguments'].get('goal_reduction', 'mean') == 'mean'
        assert b['arguments']['iterations'] == 24
        signature = file_sha256(root/'exports'/ablated['key']/'trajectory.npz')[:12]+'_'+file_sha256(__file__)[:8]+'_'+file_sha256(Path(__file__).with_name('render_guidance_surfaces.py'))[:8]
        movie = folder/f'{scene}_{signature}.gif'
        still = movie.with_suffix('.png')
        if not movie.exists():
            render(scene, a, base['geometry'], [v['planned'] for v in variants]+[v['executed'] for v in variants], movie, still,
                panel_titles=['Model: standard physical penalties', 'Model: physical penalties disabled',
                    'Physics: standard-penalty plan', 'Physics: zero-penalty plan'],
                heading=scene.replace('_', ' ')+' · matched 24-update comparison',
                subtitle='Top: model predictions. Bottom: independent execution of each generated gripper plan')
        lines += [f'![Physical-cost comparison]({movie.relative_to(root)})', '',
            '| Measurement | Standard penalties | Penalties disabled |', '|---|---:|---:|']
        fields = [('Model worst held-position error', 'held_worst_goal_mm', 'mm'),
            ('Model peak mesh overlap', 'peak_overlap_mm', 'mm'),
            ('Upward motion without both-finger proximity', 'upward_without_proximity_mm', 'mm'),
            ('Model frames with both fingers within 2 mm', 'both_fingers_close_fraction', '%'),
            ('Worst held-position error across both physics steps', 'replay_worst_held_mm', 'mm')]
        for label, field, unit in fields:
            values = [v['metrics'][field]*(100 if unit == '%' else 1) for v in variants]
            lines.append(f'| {label} | {values[0]:.2f} {unit} | {values[1]:.2f} {unit} |')
        lines += ['| Full model geometry | '+' | '.join(str(v['metrics']['geometry_pass']) for v in variants)+' |',
            '| Full paired execution | '+' | '.join(str(v['metrics']['paired_execution_pass']) for v in variants)+' |', '',
            f'[Final frame]({still.relative_to(root)})', '']
        summaries.append(dict(scene=scene, variants=[{k:v[k] for k in ('key','metrics','hashes')} for v in variants],
            movie=str(movie.relative_to(root))))
    lines += ['The upward-motion proxy sums positive vertical steps greater than 0.5 mm when '
        'at least one finger is more than 2 mm away at both ends of the transition, measured '
        'with full convex-component distances. It flags suspicious motion for these grasp '
        'tasks; it is not a contact-force test or a general law ruling out ballistic motion. '
        'The proximity fraction includes the approach and release, so it is descriptive '
        'rather than an acceptance threshold.', '',
        'Independent replay uses a floating hand and approximate inertia/contact parameters. '
        'These comparisons do not establish source-simulator or robot performance.', '']
    (root/'PHYSICAL_ABLATION.md').write_text('\n'.join(lines))
    (root/'physical_ablation_summary.json').write_text(json.dumps(summaries, indent=2)+'\n')


if __name__ == '__main__':
    main()
