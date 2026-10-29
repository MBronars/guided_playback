"""Readable lift-height diagnostics from saved real-cloth forecasts; no inference."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

from scripts.dev.observation_probe_common import write_json
from scripts.dev.real_cloth_probe import read_pcd


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', required=True)
    ap.add_argument('--source', default='validation/public_real_cloth_2026-09-14')
    a = ap.parse_args(); root, source = Path(a.root), Path(a.source)
    out = root/'real_motion'; out.mkdir(exist_ok=True)
    cached = {}; rows = []
    for phase in ('real_cloth_full', 'real_cloth_icp', 'real_cloth_short', 'real_cloth_short_smooth'):
        for path in sorted((root/phase).glob('*__seed710001.json')):
            data = json.loads(path.read_text()); inp = data['input']; capture = inp['capture']
            folder = source/'green_tshirt'/capture
            if capture not in cached:
                W = np.linalg.inv(np.array(json.loads((folder/'calibration/world_to_camera_transform.json').read_text())))
                cached[capture] = [(float(p.name.split('_')[1]), read_pcd(p, W)[0])
                                   for p in sorted((folder/'segment_pcds').glob('*.pcd'))]
            times = np.array(inp['model_times']); C = 4; n = inp['cloth_particles']
            pred = np.load(path.with_suffix('.npz'))['prediction'][:, :n]
            values = []
            for stamp, cloud in cached[capture]:
                if stamp <= times[C-1] or stamp > times[-1]: continue
                f = (stamp-times[0])*30
                lo = min(int(np.floor(f)), len(pred)-1); hi = min(lo+1, len(pred)-1); alpha = f-np.floor(f)
                p = (1-alpha)*pred[lo]+alpha*pred[hi]+np.array(inp['origin_world'])
                distances = cKDTree(p).query(cloud, workers=4)[0]*1000
                values.append({'time_after_anchor_s': float(stamp-times[C-1]),
                    'measured_visible_height_p95_mm': float(np.quantile(cloud[:, 2], .95)*1000),
                    'predicted_particle_height_p95_mm': float(np.quantile(p[:, 2], .95)*1000),
                    'observed_surface_distance_p95_mm': float(np.quantile(distances, .95))})
            row = {'phase': phase, 'capture': capture, 'key': path.stem,
                   'anchor_height_p95_mm': float(np.quantile(pred[C-1, :, 2]+inp['origin_world'][2], .95)*1000),
                   'values': values, 'source_receipt': str(path.relative_to(root))}
            write_json(out/(phase+'__'+path.stem+'.json'), row); rows.append(row)
    captures = sorted(cached)
    fig, axes = plt.subplots(3, len(captures), figsize=(15, 11), constrained_layout=True, squeeze=False)
    for j, capture in enumerate(captures):
        for i, phase in enumerate(('real_cloth_full', 'real_cloth_icp', 'real_cloth_short')):
            candidates = [x for x in rows if x['capture']==capture and x['phase']==phase]
            if not candidates: continue
            # Last of the four predeclared time anchors, not selected by forecast error.
            chosen = candidates[-1]; v = chosen['values']; t = [x['time_after_anchor_s'] for x in v]
            ax = axes[i, j]
            ax.plot(t, [x['measured_visible_height_p95_mm'] for x in v], color='#239980', label='Measured visible cloth')
            ax.plot(t, [x['predicted_particle_height_p95_mm'] for x in v], color='#d78d30', label='Predicted cloth')
            ax.axhline(chosen['anchor_height_p95_mm'], color='#88929e', ls='--', label='Observed anchor')
            if phase=='real_cloth_short':
                smoothed = next((x for x in rows if x['phase']=='real_cloth_short_smooth' and x['key']==chosen['key']), None)
                if smoothed:
                    ax.plot([x['time_after_anchor_s'] for x in smoothed['values']],
                            [x['predicted_particle_height_p95_mm'] for x in smoothed['values']],
                            color='#557bb5', ls=':', label='With motion smoothing')
            label = 'Fold' if 'fold' in capture else ('Grasp A' if capture.endswith('58') else 'Grasp B')
            ax.set(title=label+(' — early long rollout', ' — annotated long rollout', ' — later short forecast')[i],
                   xlabel='Seconds after observed prefix', ylabel='Cloth height: 95th percentile z (mm)')
            ax.grid(alpha=.15); ax.legend(fontsize=8)
    fig.suptitle('Does the predicted cloth rise with the real cloth? | primary seed, three real recordings', fontsize=15)
    fig.savefig(out/'lift_height.png', dpi=160); plt.close(fig)
    write_json(out/'summary.json', {'cases': rows,
        'figure_selection': 'All three early and annotated ICP long rollouts and the last of four predeclared short-forecast anchors, primary seed 710001',
        'interpretation': 'Height is a visible-geometry diagnostic: source depth and predicted particles can have different surface coverage. It is not full-shape tracking error. The p95 surface distance adds a tail diagnostic to the existing means.'})
    baseline = {p.stem: json.loads(p.read_text()) for p in (root/'real_cloth_short').glob('*__seed*.json')}
    comparisons = []
    for path in sorted((root/'real_cloth_short_smooth').glob('*__seed*.json')):
        candidate = json.loads(path.read_text()); old = baseline[path.stem]
        assert old['input']['anchor_timestamp']==candidate['input']['anchor_timestamp']
        assert old['input']['origin_world']==candidate['input']['origin_world']
        assert [x['time_after_anchor_s'] for x in old['metrics']]==[x['time_after_anchor_s'] for x in candidate['metrics']]
        comparisons.append({'case': path.stem, 'baseline_mean_surface_mm': old['mean_surface_mm'],
                            'smoothed_mean_surface_mm': candidate['mean_surface_mm'],
                            'paired_change_mm': candidate['mean_surface_mm']-old['mean_surface_mm'],
                            'input_smoothing_change_rms_mm': candidate['input']['history_motion_smoothing_change_rms_mm']})
    write_json(out/'motion_smoothing_comparison.json', {'cases': comparisons,
        'interpretation': 'Same observed anchors, evaluation timestamps, robot motion and sampler seeds. Negative paired change favors smoothing. These 36 cases come from three captures.'})
    print('REAL_MOTION_REPORT', len(rows), 'primary-seed forecasts', flush=True)


if __name__ == '__main__': main()
