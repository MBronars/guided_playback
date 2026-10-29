"""Fixed-camera, real-time GIF of EXPORTED ACTUAL cloth and gripper states."""
import argparse
import json
from pathlib import Path

import numpy as np

from collection.common import check_hashes, digest, write_once


def render(roots, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    data = []
    for root in map(Path, roots):
        receipt = json.loads((root / 'RESULT.json').read_text())
        if receipt['status'] != 'complete_diagnostic_export': raise ValueError('Completed actual export required')
        check_hashes(root, receipt['artifacts'])
        with np.load(root / 'episode.npz', allow_pickle=False) as z:
            data.append({k: z[k] for k in ('cloth_points', 'left_ee_points', 'right_ee_points', 'state_time_s', 'openness')})
    for d in data[1:]: np.testing.assert_array_equal(d['state_time_s'], data[0]['state_time_s'])
    # Sample fixed particle identities and fixed camera limits; no centering/rotation per frame.
    qmin = np.min([d['cloth_points'].min(axis=(0, 1)) for d in data], axis=0)
    qmax = np.max([d['cloth_points'].max(axis=(0, 1)) for d in data], axis=0)
    bounds = [(min(.15, qmin[0] - .05), max(1.15, qmax[0] + .05)),
              (min(-.55, qmin[1] - .05), max(.55, qmax[1] + .05)),
              (min(.75, qmin[2] - .05), max(1.4, qmax[2] + .05))]
    fig, axes = plt.subplots(len(data), 2, figsize=(10, 4 * len(data)), squeeze=False)
    handles = []
    for row, d in enumerate(data):
        ids = np.linspace(0, d['cloth_points'].shape[1] - 1, min(2500, d['cloth_points'].shape[1]), dtype=int)
        colors = d['cloth_points'][0, ids, 1]
        for col, dims in enumerate(((0, 1), (0, 2))):
            ax = axes[row, col]
            ax.set(xlim=bounds[dims[0]], ylim=bounds[dims[1]], xlabel='x (m)',
                   ylabel=('y' if col == 0 else 'z') + ' (m)', aspect='equal')
            if col == 1: ax.axhline(.8, color='.4', lw=1)
            ax.grid(alpha=.2)
            cloth = ax.scatter([], [], s=2, cmap='coolwarm', vmin=float(colors.min()), vmax=float(colors.max()))
            left = ax.scatter([], [], s=2, c='#159447')
            right = ax.scatter([], [], s=2, c='#ee941a')
            handles.append((row, dims, ids, colors, ax, cloth, left, right))
    indices = list(range(0, len(data[0]['state_time_s']), 6))
    if indices[-1] != len(data[0]['state_time_s']) - 1: indices.append(len(data[0]['state_time_s']) - 1)
    def update(index):
        for row, dims, ids, colors, ax, cloth, left, right in handles:
            d = data[row]
            cloth.set_offsets(d['cloth_points'][index, ids][:, dims]); cloth.set_array(colors)
            left.set_offsets(d['left_ee_points'][index, ::4][:, dims])
            right.set_offsets(d['right_ee_points'][index, ::4][:, dims])
            op = d['openness'][index]
            ax.set_title(f'{Path(roots[row]).name} | t={d["state_time_s"][index]:.2f}s | jaws L {op[0]:.2f} R {op[1]:.2f}', fontsize=9)
        return []
    fig.suptitle('Actual simulated cloth + gripper surfaces (green left / orange right)\nCloth colors follow fixed initial material identities; full robot arms omitted', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, .94))
    animation = FuncAnimation(fig, update, frames=indices, interval=200, blit=False)
    animation.save(out / 'actual-manipulation.gif', writer=PillowWriter(fps=5))
    for label, index in (('initial', 0), ('middle', len(data[0]['state_time_s']) // 2), ('final', len(data[0]['state_time_s']) - 1)):
        update(index); fig.savefig(out / (label + '.png'), dpi=120)
    plt.close(fig)
    write_once(out / 'VISUALIZATION.json', dict(roots=list(map(str, roots)), source_state_indices=indices,
        playback_fps=5, native_export_fps=30, spatial_units='m', fixed_camera=True,
        provenance='actual simulated states, not model predictions or commanded FK',
        terminal_frame_duration_note='final partial sampling interval displayed as 0.2s',
        source_export_receipt_sha256={str(root): digest(Path(root) / 'RESULT.json') for root in roots},
        artifacts={p.name: digest(p) for p in out.iterdir() if p.is_file()}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exports', nargs='+', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    render(args.exports, args.out)
