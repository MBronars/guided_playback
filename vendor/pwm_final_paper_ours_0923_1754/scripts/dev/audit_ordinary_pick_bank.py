"""Audit every retained ordinary draw in a frozen, within-checkpoint pick bank."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.dev.guidance_report_utils import task_validator, rollout_metrics, replay_pair
from scripts.dev.guidance_force_audit import audit_forces
from utils.reproducibility import file_sha256


def prepare(root,banks,vendors):
    entries=[]
    for bank in banks:
        original=json.loads((bank/'protocol.json').read_text());a=original['arguments']
        if original['checkpoint_sha256']!='99ec6cc6102a0d654c375c9f7b9cc9c86edd0964acf9de1acb42d4fb578c9fba':raise ValueError('Wrong checkpoint')
        if original['sampling']!=dict(future_len=16,horizon=36,steps=25,schedule='pyramid',uncertainty_scale=1.):raise ValueError('Wrong sampler')
        # This is a new audit receipt, preserving the original generation receipt.
        protocol=json.loads(json.dumps(original))
        protocol['arguments'].update(task='pick',collision_backend='fcl',collision_vendor=vendors,
            mesh_contact_tolerance_mm=2.,mesh_penetration_tolerance_mm=2.)
        protocol['audit_adaptation']=dict(original_protocol=str(bank/'protocol.json'),sha256=file_sha256(bank/'protocol.json'),
            changed_generation=False,description='Apply current full convex-surface audit to all retained ordinary draws; no resampling or optimization.')
        batch,geometry,collision,validator=task_validator(protocol);C=batch.context_len
        candidates=sorted((bank/'seed_8311_candidates').glob('draw_*.pt'))
        if len(candidates)!=16:raise ValueError('Require all sixteen retained draws')
        for candidate in candidates:
            roll=torch.load(candidate,map_location='cpu',weights_only=False)
            if roll['windows']!=[(4,16,0),(20,16,0),(36,4,0)]:raise ValueError('Old discontinuous window contract')
            if not torch.equal(roll['positions'][:,:C],batch.context):raise ValueError('Changed observed context')
            key=Path(a['input']).parent.name+'_'+candidate.stem;out=root/'exports'/key;out.mkdir(parents=True,exist_ok=False)
            metrics=rollout_metrics(roll,protocol,collision,validator)
            np.savez_compressed(out/'trajectory.npz',positions=roll['positions'].numpy(),R=roll['R'].numpy(),T=roll['T'].numpy(),
                particle_body=batch.particle_body.numpy(),context_len=C,target=np.asarray(protocol['target']))
            (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
            entry=dict(key=key,scene='pick_'+Path(a['input']).parent.name,export=str(out),
                source=str(candidate),source_sha256=file_sha256(candidate),model_metrics=metrics,
                predicted_lift_mm=float((roll['T'][0,-1,0,2]-roll['T'][0,C-1,0,2])*1000))
            entries.append(entry)
    (root/'manifest.json').write_text(json.dumps(dict(entries=entries,selection='All 16 ordinary candidates from each of two previously inspected, controller-qualified pick fixtures. No outcome-based draw selection.'),indent=2)+'\n')


def audit(root,controls,vendors):
    manifest=json.loads((root/'manifest.json').read_text());qualifications=json.loads(controls.read_text())['results'];results=[]
    lines=['# Ordinary pick bank: independent execution','',manifest['selection'],'',
        'The frozen model generated these trajectories without future actuator inputs or noise optimization. '+
        'The bank had an observed-anchor +10 cm target for selection; every retained draw is audited here, not only the selected draw.',
        '', 'The separate replay-fidelity criterion requires ≥50 mm executed lift, ≤20 mm / 20° final object prediction error, '+
        'passing non-goal geometry, ≤2 mm solver-step overlap, hand tracking and agreement between 0.5/0.25 ms replays. '+
        'It does not require the unguided sample to hit an externally requested destination.',
        '', '| Scene / draw | Predicted lift | Executed lift, two steps | Final prediction error, two steps | Timestep stable | Replay fidelity |',
        '|---|---:|---:|---:|---|---|']
    for entry in manifest['entries']:
        export=Path(entry['export']);protocol=json.loads((export/'protocol.json').read_text())
        q=next(v for v in qualifications if v['scene']==entry['scene'])
        directories=[root/'replays'/entry['key']/step for step in ('dt0005','dt00025')]
        if not all((d/'result.json').exists() for d in directories):raise FileNotFoundError('Bank replay incomplete: '+entry['key'])
        result=replay_pair(protocol,export/'trajectory.npz',directories,q)
        passed=[]
        for row,directory in zip(result['replays'],directories):
            force_dir=directory/'forces'
            force=json.loads((force_dir/'report.json').read_text()) if (force_dir/'report.json').exists() else audit_forces(directory,force_dir,vendor_paths=vendors)
            if force['execution_sha256']!=file_sha256(directory/'execution.npz') or not force['bitwise_pose_joint_and_integration_replay']:raise ValueError('Force audit mismatch')
            r=row['result'];okay=(row['tracking_ok'] and row['validation']['interaction_accepted'] and
                r['object_lift_mm']>=50 and r['object_prediction_final_error_mm']<=20 and r['object_rotation_final_error_deg']<=20 and force['peak_object_gripper_depth_mm']<=2)
            row['forces']=force;row['fidelity_pass']=okay;passed.append(okay)
        result.update(entry=entry,replay_fidelity_pass=q['qualified_for_generated_diagnostic'] and result['timestep_stable'] and all(passed))
        (root/'replays'/entry['key']/'audit.json').write_text(json.dumps(result,indent=2)+'\n');results.append(result)
        lift=' / '.join(f'{v["result"]["object_lift_mm"]:.1f}' for v in result['replays'])
        error=' / '.join(f'{v["result"]["object_prediction_final_error_mm"]:.1f}' for v in result['replays'])
        lines.append(f'| {entry["key"]} | {entry["predicted_lift_mm"]:.1f} mm | {lift} mm | {error} mm | {result["timestep_stable"]} | {result["replay_fidelity_pass"]} |')
        print(entry['key'],result['replay_fidelity_pass'],flush=True)
    summary=dict(count=len(results),fidelity_passes=sum(v['replay_fidelity_pass'] for v in results),results=results)
    (root/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    lines+=['',f'**{summary["fidelity_passes"]}/{summary["count"]}** pass this diagnostic. '+
        'This is a small, previously inspected fixture bank, not a population estimate or a source-simulator / robot success rate.',
        '', '[Complete audit](summary.json).','']
    (root/'README.md').write_text('\n'.join(lines))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True,type=Path)
    p.add_argument('--bank',action='append',type=Path);p.add_argument('--controls',type=Path)
    p.add_argument('--vendor',action='append',default=[]);a=p.parse_args();root=a.root.resolve();root.mkdir(parents=True,exist_ok=True)
    if a.bank:prepare(root,a.bank,a.vendor)
    elif a.controls:audit(root,a.controls,a.vendor)
    else:p.error('Supply banks to prepare, or controls to audit completed replays')


if __name__=='__main__':main()
