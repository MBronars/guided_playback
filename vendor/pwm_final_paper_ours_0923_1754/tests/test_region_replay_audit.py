"""Region reinterpretation must retain independent physical/qualification failures."""
import json
import sys

import numpy as np

from scripts.dev.audit_region_replays import main


def test_region_audit_preserves_non_goal_failures_and_source(tmp_path, monkeypatch):
    cases = [(.5, True, True, -.001, True),
             (1.1, True, True, -.001, False),
             (.5, True, True, .001, False),
             (.5, False, True, -.001, False),
             (.5, True, False, -.001, False)]
    originals = {}
    for index, (x, control, stable, penetration, expected) in enumerate(cases):
        run = f'run_{index}'
        (tmp_path / run).mkdir()
        (tmp_path / run / 'protocol.json').write_text(json.dumps(dict(
            goal_kind='position_region', goal_region_bounds=dict(
                lower='0,0,0', upper='1,1,inf', hold_frames=2))))
        directory = tmp_path / 'generated_replay_v1' / run
        directory.mkdir(parents=True)
        replays = []
        for step in ('dt0005', 'dt00025'):
            (directory / step).mkdir()
            positions = np.full((4, 1, 3), .5)
            positions[-1, 0, 0] = x
            np.savez(directory / step / 'execution.npz', model_T=positions)
            replays.append(dict(step=step, tracking_ok=True, validation=dict(
                accepted=False, interaction_accepted=penetration <= 0,
                max_interaction_violation_m=max(0, penetration),
                residuals_m=dict(goal=.2, penetration=penetration), label='physical screen')))
        path = directory / 'audit_v2.json'
        path.write_text(json.dumps(dict(run=run, control_qualified=control,
            timestep_stable=stable, qualified_task_pass=False, replays=replays)))
        originals[path] = path.read_bytes()
    monkeypatch.setattr(sys, 'argv', ['audit', '--root', str(tmp_path)])
    main()
    rows = json.loads((tmp_path / 'region_replay_summary_v1.json').read_text())['results']
    assert [row['qualified_task_pass'] for row in rows] == [case[-1] for case in cases]
    assert all(path.read_bytes() == data for path, data in originals.items())
    assert rows[0]['replays'][0]['region_held_signed_residual_mm'] == -500.
    assert abs(rows[1]['replays'][0]['region_held_signed_residual_mm'] - 100.) < 1e-9
    for row, case in zip(rows, cases):
        assert row['primary_point_task_pass'] is False
        assert row['replays'][0]['validation']['residuals_m']['penetration'] == case[3]
