"""Visualize saved consistency probes without any additional model sampling."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

from scripts.dev.render_guidance_surfaces import render
from utils.reproducibility import file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    summary = json.loads((root/'fixed_gripper_summary.json').read_text())
    manifest = json.loads((root/'gallery_manifest.json').read_text())
    entries = {entry['key']: entry for group in manifest['groups'] for entry in group['variants']}
    folder = root/'fixed_gripper_surfaces'
    folder.mkdir(exist_ok=True)
    lines = ['# Same gripper motion, different predicted object motion', '',
        'Every panel is a prediction from the same frozen model. None is a physics replay. '
        'The lower panels show the best and worst held-position errors among the eight fresh '
        'passive-object noise draws. They are selected to illustrate the observed range; '
        '[all eight draws and geometry checks remain in the full diagnostic](FIXED_GRIPPER.md).', '',
        'The top-right reference inpaints the optimized gripper path with the selected noise. '
        'It can differ from the original top-left plan because inpainting changes intermediate '
        'denoising states. All inpainted variants preserve the generated actuator path. '
        'Orange is the passive object; grey, blue and green are the gripper.', '']
    receipts = []
    for row in summary['rows']:
        diagnostic = Path(row['report']).parent
        report = json.loads((diagnostic/'report.json').read_text())
        if report['status'] != 'passed' or not all(c['passed'] for c in report['checks']):
            raise ValueError(f'Unqualified consistency probe: {diagnostic}')
        suffix = 'max_actuated_i64_r050' if 'actuator noise;' in row['scope'] else 'i64_r050'
        key = row['scene']+'_left_'+suffix
        run = Path(entries[key]['run'])
        protocol_path = run/'protocol.json'
        protocol = json.loads(protocol_path.read_text())
        for vendor in protocol['arguments']['collision_vendor']:
            if vendor not in sys.path:
                sys.path.append(vendor)
        seed_path = run/f"seed_{entries[key].get('seed', 8311)}.pt"
        assert file_sha256(protocol_path) == report['protocol_sha256']
        assert file_sha256(seed_path) == report['source_sha256']
        assert file_sha256(protocol['arguments']['geometry']) == protocol['geometry_sha256']
        original = torch.load(seed_path, map_location='cpu', weights_only=False)['result']['rollout']
        geometry = torch.load(protocol['arguments']['geometry'], map_location='cpu', weights_only=False)
        variants = report['variants']
        reference = next(v for v in variants if v['seed'] is None)
        redraws = [v for v in variants if v['seed'] is not None]
        assert len(redraws) == 8
        best = min(redraws, key=lambda v: v['metrics']['held_worst_goal_mm'])
        worst = max(redraws, key=lambda v: v['metrics']['held_worst_goal_mm'])
        selected = [reference, best, worst]
        rolls = [original]+[torch.load(diagnostic/v['track'], map_location='cpu', weights_only=False) for v in selected]
        C = original['T'].shape[1]-protocol['sampling']['horizon']
        assert C > 0 and all(v['T'].shape == original['T'].shape for v in rolls)
        tracks = [(v['R'][0, C-1:].numpy(), v['T'][0, C-1:].numpy()) for v in rolls]
        for R, T in tracks[1:]:
            np.testing.assert_allclose(T[:, 1:], tracks[0][1][:, 1:], atol=2e-6, rtol=0)
            np.testing.assert_allclose(R[:, 1:], tracks[0][0][:, 1:], atol=1e-5, rtol=0)
        titles = ['Original optimized prediction', 'Inpainted reference',
            f'Best of 8 fresh draws: seed {best["seed"]}', f'Worst of 8 fresh draws: seed {worst["seed"]}']
        signature = file_sha256(diagnostic/'report.json')[:12]+'_'+file_sha256(__file__)[:8]+'_'+file_sha256(Path(__file__).with_name('render_guidance_surfaces.py'))[:8]
        movie = folder/f'{key}_{signature}.gif'
        still = movie.with_suffix('.png')
        if not movie.exists():
            render(key, protocol, geometry, tracks, movie, still, panel_titles=titles,
                heading=row['scene'].replace('_', ' ')+' · '+row['scope'],
                subtitle='All panels are model predictions; actuator path is fixed in every inpainted draw')
        errors = [report['selected_metrics']['held_worst_goal_mm']]+[v['metrics']['held_worst_goal_mm'] for v in selected]
        lines += ['## '+row['scene']+' / '+row['scope'], '',
            f'![Fixed gripper comparison]({movie.relative_to(root)})', '',
            'Worst held-position errors, in panel order: '+', '.join(f'{e:.1f} mm' for e in errors)+'.', '',
            f'[Final frame]({still.relative_to(root)}) · [All draws and checks]({diagnostic.relative_to(root)}/README.md)', '']
        receipts.append(dict(key=key, report_sha256=file_sha256(diagnostic/'report.json'),
            selected_redraw_seeds=[best['seed'], worst['seed']], movie=str(movie.relative_to(root)),
            held_errors_mm=errors))
    lines += ['Equal generated gripper poses do not necessarily produce equal commands under the '
        'geometry-dependent preload adapter. These panels test model consistency only. '
        '[Independent executions are reported separately](EXECUTION_GALLERY.md).', '']
    (root/'FIXED_GRIPPER_SURFACES.md').write_text('\n'.join(lines))
    (root/'fixed_gripper_surface_summary.json').write_text(json.dumps(receipts, indent=2)+'\n')


if __name__ == '__main__':
    main()
