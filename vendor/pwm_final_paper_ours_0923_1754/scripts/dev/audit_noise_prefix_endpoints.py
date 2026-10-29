"""Measure saved-prefix endpoints directly, independently of objective reductions.

Legacy placing optimizer diagnostics called the mean held-goal cost
target_distance_mm. This repairs report measurements without changing saved
trajectories, objective values, candidate selection or the shared mesh screen.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    root=parser.parse_args().root.resolve()
    source=root/'prefix_mesh_summary_v2.json'
    rows=[]
    for previous in json.loads(source.read_text())['results']:
        directory=root/previous['run']
        stem=f'through_{previous["updates"]:04d}'
        prefix=directory/f'seed_{previous["seed"]}_iterates'/stem
        output=prefix.with_name(stem+'_mesh_audit_v3.json')
        fingerprint=hashlib.sha256(json.dumps(previous,sort_keys=True).encode()).hexdigest()
        if output.exists():
            row=json.loads(output.read_text())
            if row.get('previous_audit_sha256')==fingerprint:
                rows.append(row)
                continue
        saved=torch.load(prefix.with_suffix('.pt'),weights_only=False,map_location='cpu')
        protocol=json.loads((directory/'protocol.json').read_text())
        rollout=saved['rollout']
        column=(rollout['rigid_idx']==protocol['arguments'].get('goal_body',0)).nonzero(as_tuple=True)[0].item()
        target=torch.tensor(protocol['target'],dtype=rollout['T'].dtype)
        distance=(rollout['T'][:,:,column]-target[:,None]).norm(dim=-1)*1000
        hold=protocol['arguments'].get('hold_frames',6)
        endpoint=float(distance[:,-1].mean())
        held=float(distance[:,-hold:].max())
        if abs(held-(previous['validation']['residuals_m']['goal']+.020)*1000)>.002:
            raise ValueError(f'Pose-derived held distance disagrees with fixed task screen: {prefix}')
        row=dict(previous,goal_error_mm=endpoint,hold_goal_error_mm=held,
                 legacy_optimizer_target_distance_mm=saved['diagnostics']['target_distance_mm'],
                 objective_goal_cost_mm=saved['diagnostics'].get('goal_cost_mm'),
                 previous_audit_sha256=fingerprint,
                 endpoint_measurement='Euclidean final template-origin distance, measured directly from saved rollout T and protocol target')
        output.write_text(json.dumps(row,indent=2)+'\n')
        rows.append(row)
    summary=dict(results=rows,protocol=__doc__)
    temporary=root/'prefix_mesh_summary_v3.pending'
    temporary.write_text(json.dumps(summary,indent=2)+'\n')
    temporary.replace(root/'prefix_mesh_summary_v3.json')
    print(f'Measured {len(rows)} saved-prefix endpoints; shared mesh screens unchanged')


if __name__=='__main__':main()
