"""Report every saved candidate in the obstacle selection/refinement experiment."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from utils.reproducibility import file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    root = parser.parse_args().root.resolve()
    lines = ['# Obstacle candidate selection', '',
        'Each experiment draws 16 complete object-and-gripper trajectories, selects the '
        'lowest weighted objective, and performs 24 noise-optimization updates from that '
        'candidate. No prescribed grasp path is supplied. Every candidate is listed below; '
        'individual candidates have not all received independent physics replay.', '',
        'Candidate clearances use the three interpolation probes per transition used during '
        'optimization. They are not the denser ten-probe final audit and do not certify '
        'continuous clearance. The final selected/refined trajectory is audited separately.', '']
    all_rows = []
    for scene in ('bottle_8270', 'teacup_7803'):
        run = root/(scene+'_left_obstacle_select16_refine24_r050')
        lines += ['## '+scene, '']
        if not (run/'results.json').exists():
            lines += ['Experiment pending or not run; consult the packet status.', '']
            continue
        protocol = json.loads((run/'protocol.json').read_text())
        args = protocol['arguments']
        assert args['method'] == 'select_refine' and args['k'] == 16 and not args['screen']
        result = torch.load(run/'seed_8311.pt', map_location='cpu', weights_only=False)['result']
        target = np.asarray(protocol['target'][0])
        rows, tracks = [], []
        for draw in range(args['k']):
            file = run/'seed_8311_candidates'/f'draw_{draw:03d}.pt'
            record = json.loads(file.with_suffix('.json').read_text())
            roll = torch.load(file, map_location='cpu', weights_only=False)
            C = roll['T'].shape[1]-protocol['sampling']['horizon']
            track = roll['T'][0, C-1:, 0].numpy()
            distance = np.linalg.norm(track-target, axis=-1)*1000
            rows.append(dict(draw=draw, weighted_selection_loss=record['loss'],
                held_error_mm=float(distance[-args['hold_frames']:].max()),
                objective_min_clearance_mm=record['obstacle_min_clearance_mm'],
                trajectory_sha256=file_sha256(file)))
            tracks.append(track)
        selected = min(range(len(rows)), key=lambda i: rows[i]['weighted_selection_loss'])
        assert selected == result['initial_draw']
        refined = result['rollout']['T'][0, C-1:, 0].numpy()
        figure = plot(root, scene, tracks, selected, refined, target)
        lines += [f'![All candidate paths]({figure.relative_to(root)})', '',
            f'Selected draw: **{selected}**. Grey curves show every initial draw; orange '
            'shows the selected candidate and green its refined prediction.', '',
            '| Draw | Weighted selection loss × 1000 | Worst held-position error | Three-probe minimum clearance |',
            '|---|---:|---:|---:|']
        for row in rows:
            label = str(row['draw'])+(' (selected)' if row['draw'] == selected else '')
            lines.append(f'| {label} | {row["weighted_selection_loss"]*1000:.2f} | '
                f'{row["held_error_mm"]:.2f} mm | {row["objective_min_clearance_mm"]:.2f} mm |')
        lines += ['', '[Final dense audits and independent execution](EXECUTION_GALLERY.md) · '
            '[All model comparisons](GALLERY.md) · [Measured compute](COMPUTE.md)', '']
        all_rows.append(dict(scene=scene, selected_draw=selected, candidates=rows,
            protocol_sha256=file_sha256(run/'protocol.json'), result_sha256=file_sha256(run/'seed_8311.pt')))
    (root/'OBSTACLE_CANDIDATES.md').write_text('\n'.join(lines))
    (root/'obstacle_candidate_summary.json').write_text(json.dumps(all_rows, indent=2)+'\n')


def plot(root, scene, tracks, selected, refined, target):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.3))
    for track in tracks:
        for ax, x, y in [(axs[0], track[:, 0], track[:, 1]),
            (axs[1], np.arange(len(track))*.05, track[:, 2]),
            (axs[2], np.arange(len(track))*.05, np.linalg.norm(track-target, axis=-1)*1000)]:
            ax.plot(x, y, color='#929292', alpha=.35, lw=.8)
    for label, track, color in [('Selected initial draw', tracks[selected], '#bc7714'),
        ('After refinement', refined, '#187d62')]:
        axs[0].plot(track[:, 0], track[:, 1], color=color, label=label, lw=2)
        axs[1].plot(np.arange(len(track))*.05, track[:, 2], color=color, lw=2)
        axs[2].plot(np.arange(len(track))*.05, np.linalg.norm(track-target, axis=-1)*1000, color=color, lw=2)
    axs[0].scatter(*target[:2], marker='X', color='#bc2775', s=65)
    axs[0].set(xlabel='Object origin x (m)', ylabel='Object origin y (m)', title='Top view');axs[0].axis('equal')
    axs[0].legend(fontsize=8)
    axs[1].set(xlabel='Time (s)', ylabel='Object origin z (m)', title='Height')
    axs[1].axhline(target[2], color='#bc2775', ls=':', lw=1)
    axs[2].set(xlabel='Time (s)', ylabel='Position error (mm)', title='Target reaching')
    axs[2].axhline(20, color='#bc2775', ls=':', lw=1)
    for ax in axs:ax.grid(alpha=.2)
    fig.suptitle(scene+' · all 16 initial candidates and the selected refinement\nModel predictions; obstacle clearance and execution are checked separately')
    fig.tight_layout();folder=root/'figures';folder.mkdir(exist_ok=True)
    file=folder/(scene+'_obstacle_candidates.png');fig.savefig(file,dpi=150);plt.close(fig)
    return file


if __name__ == '__main__':
    main()
