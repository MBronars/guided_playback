"""Keep controller-policy qualification separate from model optimization results."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.dev.guidance_report_utils import replay_pair
from utils.pose_ops import rotation_geodesic_error
from utils.reproducibility import file_sha256


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True,type=Path)
    p.add_argument('--reference',required=True,type=Path);args=p.parse_args();root=args.root.resolve()
    reference=json.loads(args.reference.read_text());qualified=[]
    for old in reference['results']:
        paths=[root/'preload_latched/control'/old['scene']/tag for tag in ('dt0005','dt00025')]
        results=[json.loads((path/'result.json').read_text()) for path in paths]
        arrays=[np.load(path/'execution.npz') for path in paths]
        for key in ('desired_R','desired_T','commanded_q','preload_active'):
            np.testing.assert_array_equal(arrays[0][key],arrays[1][key])
        old_arrays=np.load(Path(old['paths']['mesh_preload'])/'execution.npz')
        for key in ('desired_R','desired_T','requested_q'):
            np.testing.assert_array_equal(arrays[0][key],old_arrays[key])
        dp=np.linalg.norm(arrays[0]['model_T'][:,0]-arrays[1]['model_T'][:,0],axis=-1)*1000
        dr=rotation_geodesic_error(torch.tensor(arrays[0]['model_R'][:,0]),torch.tensor(arrays[1]['model_R'][:,0])).numpy()*180/np.pi
        stable=bool(dp.max()<=5 and dr.max()<=5)
        def agreed(r):
            return bool(r['finite'] and not any(r['warning_counts']) and all(r[k]<=v for k,v in reference['control_thresholds'].items()))
        q=dict(scene=old['scene'],task=old['task'],group=old['group'],
            paths=dict(mesh_preload=str(paths[0]),mesh_preload_fine=str(paths[1])),
            results=dict(mesh_preload=results[0],mesh_preload_fine=results[1]),
            control_agreement=dict(mesh_preload=agreed(results[0]),mesh_preload_fine=agreed(results[1])),
            timestep_stable=stable,timestep_position_max_mm=float(dp.max()),timestep_rotation_max_deg=float(dr.max()),
            qualified_for_generated_diagnostic=stable and all(agreed(r) for r in results),
            original_qualified=old['qualified_for_generated_diagnostic'])
        qualified.append(q)
    summary=dict(protocol='Latched 6 mm preload, released/rearmed by a 2 mm mean opening/closing stroke; all recorded controls at 0.5/0.25 ms.',
        reference_sha256=file_sha256(args.reference),control_thresholds=reference['control_thresholds'],
        timestep_thresholds=reference['timestep_thresholds'],results=qualified)
    (root/'preload_latched_qualification.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# Preload-policy diagnostic','','This changes the independent replay adapter, not the generated model plan. '+
        'The original adapter adds/removes 6 mm of closure whenever a two-finger proximity flag changes. '+
        'The optional latched policy retains closure through proximity flicker and releases after a deliberate 2 mm mean opening. '+
        'Instant remains the default. Stateful feedback has not been integrated and is explicitly rejected for this policy.','',
        '| Recorded control | Original qualified | Latched qualified | Latched endpoint errors, two steps | Timestep agreement |',
        '|---|---|---|---:|---|']
    for q in qualified:
        errors=' / '.join(f'{v["object_prediction_final_error_mm"]:.1f}' for v in q['results'].values())
        lines.append(f'| {q["scene"]} | {q["original_qualified"]} | {q["qualified_for_generated_diagnostic"]} | {errors} mm | {q["timestep_stable"]} |')
    lines+=['','## Fixed generated plans','','| Plan | Original replay final goal error | Latched replay final goal error | Latched timestep agreement | Latched complete task |','|---|---:|---:|---|---|']
    generated=[]
    for directory in sorted((root/'preload_latched/generated').iterdir()):
        key=directory.name;export=root/'exports'/key;protocol=json.loads((export/'protocol.json').read_text())
        scene=protocol['arguments']['task']+'_'+Path(protocol['arguments']['input']).parent.name
        q=next(v for v in qualified if v['scene']==scene)
        audit=replay_pair(protocol,export/'trajectory.npz',[directory/'dt0005',directory/'dt00025'],q)
        audit.update(key=key,controller_policy='latched',qualification_sha256=file_sha256(root/'preload_latched_qualification.json'))
        (directory/'audit.json').write_text(json.dumps(audit,indent=2)+'\n');generated.append(audit)
        old=json.loads((root/'replays'/key/'audit.json').read_text())
        before=' / '.join(f'{v["final_goal_mm"]:.1f}' for v in old['replays']);after=' / '.join(f'{v["final_goal_mm"]:.1f}' for v in audit['replays'])
        lines.append(f'| {key} | {before} mm | {after} mm | {audit["timestep_stable"]} | {audit["qualified_task_pass"]} |')
    (root/'preload_latched_generated_summary.json').write_text(json.dumps(dict(results=generated),indent=2)+'\n')
    lines+=['','[Complete recorded-control qualification](preload_latched_qualification.json) · [Generated-plan audit](preload_latched_generated_summary.json).',
        'Controller qualification is specific to these fixtures and this independent engine. Fixing a command reversal does not establish physical task success.','']
    (root/'PRELOAD_DIAGNOSTIC.md').write_text('\n'.join(lines))
    print('Original qualified:',sum(v['original_qualified'] for v in qualified),'Latched qualified:',sum(v['qualified_for_generated_diagnostic'] for v in qualified),
          'Generated complete passes:',sum(v['qualified_task_pass'] for v in generated))


if __name__=='__main__':main()
