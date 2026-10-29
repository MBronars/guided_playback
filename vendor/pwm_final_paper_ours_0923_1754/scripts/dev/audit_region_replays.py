"""Evaluate declared position-region tasks using existing independent replay audits.

Only the goal residual changes. Controller qualification, timestep stability,
tracking, and every physical residual retain their original definitions.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    rows = []
    for path in sorted((root / 'generated_replay_v1').glob('*/audit_v2.json')):
        base = json.loads(path.read_text())
        protocol_path = root / base['run'] / 'protocol.json'
        protocol = json.loads(protocol_path.read_text())
        if protocol.get('goal_kind') != 'position_region':
            continue
        bounds = protocol['goal_region_bounds']
        lower, upper = [np.array([float(x) for x in bounds[k].split(',')])
                        for k in ('lower', 'upper')]
        hold = bounds['hold_frames']
        assert lower.shape == upper.shape == (3,) and np.all(lower < upper)
        values = []
        for original in base['replays']:
            with np.load(path.parent / original['step'] / 'execution.npz') as track:
                positions = track['model_T'][1:, 0]
            assert len(positions) >= hold and np.isfinite(positions).all()
            coordinate_violation = np.maximum(lower - positions, positions - upper)
            residual = float(coordinate_violation[-hold:].max())
            distance = np.linalg.norm(np.maximum(coordinate_violation, 0), axis=-1)
            physical = original['validation']
            residuals = dict(physical['residuals_m'], goal=residual)
            validation = dict(physical, residuals_m=residuals,
                              accepted=all(v <= 0 for v in residuals.values()),
                              max_violation_m=max(0., max(residuals.values())),
                              label='Declared position-region task; ' + physical['label'])
            values.append(dict(original, validation=validation,
                               region_final_distance_mm=float(distance[-1] * 1000),
                               region_held_max_distance_mm=float(distance[-hold:].max() * 1000),
                               region_held_signed_residual_mm=residual * 1000))
        row = dict(base, replays=values, goal_region_bounds=bounds,
                   primary_point_task_pass=base['qualified_task_pass'],
                   qualified_task_pass=bool(base['control_qualified'] and base['timestep_stable']
                       and all(v['tracking_ok'] and v['validation']['accepted'] for v in values)),
                   task_definition='Exploratory minimum-height lift in an XY workspace; distinct from precise point target.',
                   source_audit_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                   source_protocol_sha256=hashlib.sha256(protocol_path.read_bytes()).hexdigest())
        (path.parent / 'region_audit_v1.json').write_text(json.dumps(row, indent=2) + '\n')
        rows.append(row)
    summary = dict(results=rows, completed=len(rows),
                   qualified_task_passes=sum(row['qualified_task_pass'] for row in rows),
                   note='Includes selected and optimization-prefix diagnostics; these are not independent trials.')
    (root / 'region_replay_summary_v1.json').write_text(json.dumps(summary, indent=2) + '\n')
    print({k: v for k, v in summary.items() if k != 'results'})


if __name__ == '__main__':
    main()
