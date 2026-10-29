"""Show frozen replication destinations using observed geometry only."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    path = parser.parse_args().manifest.resolve(); root = path.parent
    manifest = json.loads(path.read_text())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(manifest['rows']), 2, figsize=(12, 5*len(manifest['rows'])), squeeze=False)
    lines = ['# Frozen replication goals', '', manifest['target_rule'], '',
        'These are requested destinations drawn with the observed object geometry. '
        'They are not predictions or recorded future states. Grey points show the observed gripper. '
        'Orange is the observed object; blue and purple show the two requested goal poses.', '',
        'For picking, the pictured goal orientation simply copies the observed orientation; '
        'the pick objective constrains position only. Placement also requests the task rotation.', '',
        '![Observed scenes and distinct requested destinations](figures/replication_targets.png)', '',
        '| Scene | Direction | Target x, y, z (m) | Initial origin-to-target distance |',
        '|---|---|---|---:|']
    for row_index, row in enumerate(manifest['rows']):
        directory = Path(row['scene']['directory'])
        source = directory/'input.pt'
        if hashlib.sha256(source.read_bytes()).hexdigest() != row['fixture_hashes']['input.pt']:
            raise ValueError('Observed fixture changed since target freeze')
        batch = torch.load(source, map_location='cpu', weights_only=False)
        positions = batch.positions[0, batch.context_len-1].numpy()
        bodies = batch.particle_body[0].numpy()
        pose = batch.body_poses[0, batch.context_len-1, 0].numpy()
        anchor = pose[:3, 3]
        if not np.allclose(anchor, row['observed_anchor'], atol=1e-7, rtol=0):
            raise ValueError('Goal manifest anchor differs from observed pose')
        object_points = positions[bodies==0]
        local = (object_points-anchor) @ pose[:3, :3]
        target_rotation = pose[:3, :3] if row['target_rotation'] is None else np.array(row['target_rotation'])
        name = 'Spice jar: pick' if row['scene']['task']=='pick' else 'Bowl: place'
        for column, dims in enumerate([(0, 1), (0, 2)]):
            axis = axes[row_index, column]
            observed = positions[bodies!=0][::5]
            axis.scatter(observed[:, dims[0]], observed[:, dims[1]], color='#777777', alpha=.4, s=3,
                         label='Observed gripper')
            axis.scatter(object_points[::4, dims[0]], object_points[::4, dims[1]], color='#da9300', s=4,
                         alpha=.65, label='Observed object')
            for goal, color in zip(row['goals'], ['#267ea7', '#8b4d9e']):
                target = np.array(goal['target'])
                ghost = local @ target_rotation.T+target
                axis.scatter(ghost[::5, dims[0]], ghost[::5, dims[1]], color=color, alpha=.3, s=4)
                axis.scatter(target[dims[0]], target[dims[1]], marker='X', color=color, s=70,
                             label=goal['label'].title()+' destination')
            axis.set(xlabel='x (m)', ylabel='y (m)' if column==0 else 'z (m)',
                     title=name+(' — top view' if column==0 else ' — side view'))
            axis.axis('equal'); axis.grid(alpha=.2)
            if column==0: axis.legend(fontsize=8)
        for goal in row['goals']:
            xyz=', '.join(f'{v:.4f}' for v in goal['target'])
            distance=np.linalg.norm(np.array(goal['target'])-anchor)*1000
            lines.append(f'| {name} | {goal["label"]} | {xyz} | {distance:.1f} mm |')
    fig.suptitle('New goals fixed before choosing the guidance method\nObserved geometry and requested endpoints only; no prescribed path')
    fig.tight_layout(); (root/'figures').mkdir(exist_ok=True)
    fig.savefig(root/'figures/replication_targets.png', dpi=160); plt.close(fig)
    lines.extend(['', 'One common method is selected on the bottle and teacup development comparisons. '
        'The two directions share seed 9351. All four outcomes will be reported. '
        'These previously inspected fixtures are not held out from model training.', '',
        '[Frozen targets, observations and hashes](replication_targets.json) · '
        '[Method-selection rule](replication_selection_rule.json) · [Results gallery](GALLERY.md)', ''])
    (root/'REPLICATION_TARGETS.md').write_text('\n'.join(lines))


if __name__=='__main__': main()
