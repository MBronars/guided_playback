"""Verify paired noise and shared control trajectories across feedback ablations."""
import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import torch


def equal_tree(left,right):
    if torch.is_tensor(left):
        torch.testing.assert_close(left,right,rtol=0,atol=0)
    elif isinstance(left,(list,tuple)):
        assert len(left)==len(right)
        for a,z in zip(left,right):equal_tree(a,z)
    elif isinstance(left,dict):
        assert left.keys()==right.keys()
        for key in left:equal_tree(left[key],right[key])
    else:assert left==right


def shared_execution(left,right,variant):
    count=0
    for step in (0,1):
        a=np.load(left/f'{variant}_step{step}.npz');z=np.load(right/f'{variant}_step{step}.npz')
        for key in ('R','T','q','time','integration_states'):
            np.testing.assert_array_equal(a[key],z[key])
        count+=1
    return count


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',required=True,type=Path)
    root=parser.parse_args().root.resolve();runs=[];loaded={};checks=[]
    for path in sorted(root.glob('*/report.json')):
        if not re.match(r'g[01]v\d+_',path.parent.name):continue
        r=json.loads(path.read_text())
        if r.get('status')=='completed' and 'prefix_frames' in r:runs.append((path.parent,r))
    def tensors(path,repeat):
        key=(path,repeat)
        if key not in loaded:loaded[key]=torch.load(path/('replans.pt' if repeat else 'plans.pt'),weights_only=False,map_location='cpu')
        return loaded[key]
    identity=('checkpoint_sha256','base_plan_sha256','goal','goal_rotation','goal_region_bounds','sampling','seed','prefix_frames','total_frames')
    for index,(a,ar) in enumerate(runs):
        for z,zr in runs[index+1:]:
            if any(ar[k]!=zr[k] for k in identity):continue
            aa,za=ar['arguments'],zr['arguments'];am,zm=aa.get('method','noise_opt'),za.get('method','noise_opt')
            ac,zc=aa.get('context_source','observed'),za.get('context_source','observed')
            repeated=aa.get('replan_every') is not None
            relation=None;fields=[];replays=0
            if aa.get('replan_every')!=za.get('replan_every'):
                if ac!=zc or {am,zm}!={'noise_opt','best_of_k'}:continue
                m,bank=(a,z) if repeated else (z,a)
                mr=ar if repeated else zr
                first=tensors(m,True)['feedback_sample_0']['initial']
                equal_tree(first,tensors(bank,False)['initial'])
                # The long sampled plan and MPC execute the same first segment;
                # later MPC commands intentionally change after observing again.
                commit=mr['replan_schedule'][0]['commit'];prefix=mr['prefix_frames']
                for step in (0,1):
                    left=np.load(m/f'feedback_sample_step{step}.npz');right=np.load(bank/f'feedback_sample_step{step}.npz')
                    for key in ('R','T'):np.testing.assert_array_equal(left[key][:prefix+commit+1],right[key][:prefix+commit+1])
                    for key in ('q','time','integration_states'):np.testing.assert_array_equal(left[key][:commit+1],right[key][:commit+1])
                checks.append(dict(left=a.name,right=z.name,relation='early sampling versus first MPC plan',
                    bitwise_fields=['complete first fresh plan'],physics_replays_bitwise=shared_execution(a,z,'continue_original'),
                    shared_fresh_execution_through_frame=prefix+commit,shared_fresh_execution_steps=2,
                    left_model_calls=ar['actual_model_calls'],right_model_calls=zr['actual_model_calls'],
                    equal_model_calls=ar['actual_model_calls']==zr['actual_model_calls'],
                    report_sha256={p.name:hashlib.sha256((p/'report.json').read_bytes()).hexdigest() for p in (a,z)}))
                continue
            if repeated:
                if ac!=zc or aa['iterations']==za['iterations']:continue
                assert ar['replan_schedule']==zr['replan_schedule']
                relation='updates per replan';left,right=tensors(a,True),tensors(z,True)
                for stage in range(len(ar['replan_schedule'])):
                    key=f'feedback_sample_{stage}'
                    equal_tree(left[key]['initial'],right[key]['initial'])
                    for field in ('noise','rng_states'):
                        equal_tree(left[f'feedback_noise_opt_{stage}']['initial'][field],right[f'feedback_noise_opt_{stage}']['initial'][field])
                fields=['complete fresh rollouts at every replan','optimizer initial noise and routing at every replan']
                for variant in ('continue_original','feedback_sample'):replays+=shared_execution(a,z,variant)
            else:
                left,right=tensors(a,False)['initial'],tensors(z,False)['initial']
                if ac!=zc and am==zm and aa['iterations']==za['iterations'] and aa.get('k')==za.get('k'):
                    relation='observed versus predicted context'
                    fields=['noise','rng_states']
                    for field in fields:equal_tree(left[field],right[field])
                    replays=shared_execution(a,z,'continue_original')
                elif ac==zc and {am,zm}=={'noise_opt','best_of_k'}:
                    relation='noise optimization versus sample selection'
                    fields=['R','T','positions','noise','rng_states']
                    for field in fields:equal_tree(left[field],right[field])
                    for variant in ('continue_original','feedback_sample'):replays+=shared_execution(a,z,variant)
            if relation:
                checks.append(dict(left=a.name,right=z.name,relation=relation,bitwise_fields=fields,
                    physics_replays_bitwise=replays,left_model_calls=ar['actual_model_calls'],right_model_calls=zr['actual_model_calls'],
                    equal_model_calls=ar['actual_model_calls']==zr['actual_model_calls'],
                    report_sha256={p.name:hashlib.sha256((p/'report.json').read_bytes()).hexdigest() for p in (a,z)}))
    report=dict(status='passed',completed_experiments=len(runs),completed_pairs=len(checks),checks=checks,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        note='Absent pairs are pending, not failed or passed. Equal forward-invocation counts do not mean equal FLOPs or wall time. Context ablations share noise and the original continuation; their refreshed trajectories intentionally differ. Budget ablations share the fresh-sampling control; optimized histories may diverge.')
    (root/'feedback_pairing_v1.json').write_text(json.dumps(report,indent=2)+'\n');print('Feedback pairing:',len(checks),'completed pairs passed')


if __name__=='__main__':main()
