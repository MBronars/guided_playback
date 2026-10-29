"""Re-run saved optimized noise with different representative-point RNG streams."""
import argparse
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from guidance.pick_validation import PickValidator
from guidance.unconditional import fresh_routing_states, prediction, sample_rollout
from inference import load_model_and_diffusion
from utils.reproducibility import code_revision, file_sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',nargs='+',required=True,help='label=completed run directory')
    p.add_argument('--seeds',default='54321,54322')
    p.add_argument('--routing-seeds',default='771,772,773')
    p.add_argument('--out',required=True)
    args=p.parse_args(); out=Path(args.out); out.mkdir(parents=True,exist_ok=False)
    runs=[(v.split('=',1)[0],Path(v.split('=',1)[1])) for v in args.runs]
    protocols=[json.loads((path/'protocol.json').read_text()) for _,path in runs]
    first=protocols[0]
    for protocol in protocols:
        for key in ('checkpoint_sha256','input_sha256','geometry_sha256','grasp_sha256','sampling','normalization','target','precision'):
            if protocol[key]!=first[key]: raise ValueError('Mismatched routing-audit inputs: '+key)
        a=protocol['arguments']
        if a['method'] not in ('sample','best_of_k','noise_opt','select_refine','multistart'):
            raise ValueError('This replay supports saved unguided/noise-optimized paths without additional inpainting or denoising corrections')
        if a.get('observation_translation_mm',0) or a.get('observation_rotation_deg',0) or a.get('density_scale',1)!=1:
            raise ValueError('Reconstruct perturbed observations before auditing a biased-input experiment')
    a=first['arguments']
    for key in ('checkpoint','input','geometry','grasp'):
        if file_sha256(a[key])!=first[key+'_sha256']: raise ValueError('Changed input file: '+key)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch=torch.load(a['input'],map_location='cpu',weights_only=False).to(device)
    geometry=torch.load(a['geometry'],map_location='cpu',weights_only=False)
    grasp=torch.load(a['grasp'],map_location='cpu',weights_only=False)
    model,diffusion,cfg,_=load_model_and_diffusion(a['checkpoint'],device)
    if float(cfg.train.get('action_dropout',0))!=1: raise ValueError('Requires an unconditional checkpoint')
    model.eval(); model.requires_grad_(False)
    target=torch.tensor(first['target'],device=device,dtype=torch.float32)
    validator=PickValidator(batch,target,geometry,grasp['calibration'],hold_frames=a['hold_frames'])
    calls=[0]
    def check_forward(_,positional,kwargs):
        b=positional[0]
        if bool(b.pose_action_mask.any()) or bool(b.pose_actions.any()): raise AssertionError('Future actions reached the model')
        calls[0]+=1
    hook=model.register_forward_pre_hook(check_forward,with_kwargs=True)
    rows=[]; receipts=[]; t0=time.perf_counter()
    with torch.no_grad():
        for method,path in runs:
            for seed in [int(v) for v in args.seeds.split(',')]:
                saved_path=path/f'seed_{seed}.pt'; saved=torch.load(saved_path,map_location=device,weights_only=False)['result']['rollout']
                repeated=sample_rollout(model,diffusion,batch,noise=saved['noise'],rng_states=saved['rng_states'],**first['sampling'])
                error=float((repeated['positions']-saved['positions']).abs().max())
                if error>2e-6: raise ValueError(f'Saved FPS/noise failed to reproduce {saved_path}: {error} m')
                receipts.append(dict(method=method,seed=seed,saved_path=str(saved_path),sha256=file_sha256(saved_path),replay_max_coordinate_error_m=error))
                for route_seed in [None]+[int(v) for v in args.routing_seeds.split(',')]:
                    result=repeated if route_seed is None else sample_rollout(model,diffusion,batch,
                        noise=saved['noise'],rng_states=fresh_routing_states(len(saved['noise']),route_seed,device),**first['sampling'])
                    for original_window,new_window in zip(saved['noise'],result['noise']):
                        for original_array,new_array in zip(original_window,new_window):
                            if not torch.equal(original_array,new_array): raise AssertionError('Gaussian arrays changed')
                    validation=validator(prediction(result,batch.context_len))
                    shift=(result['T'][:,batch.context_len:,0]-saved['T'][:,batch.context_len:,0]).norm(dim=-1)
                    row=dict(method=method,seed=seed,routing_seed=route_seed,
                        object_path_max_change_mm=float(shift.max())*1000,validation=validation)
                    rows.append(row)
                    if route_seed is not None:
                        np.savez_compressed(out/f'{path.name}_seed_{seed}_routing_{route_seed}.npz',
                            positions=result['positions'].cpu().numpy(),R=result['R'].cpu().numpy(),T=result['T'].cpu().numpy(),
                            original_positions=saved['positions'].cpu().numpy(),particle_body=batch.particle_body.cpu().numpy())
                    print(json.dumps(row),flush=True)
    hook.remove()
    if device.type=='cuda': torch.cuda.synchronize(device)
    report=dict(arguments=vars(args),checkpoint_sha256=first['checkpoint_sha256'],sampling=first['sampling'],
        input_protocols=[str(path/'protocol.json') for _,path in runs],source_root=str(Path(__file__).resolve().parents[2]),
        receipts=receipts,results=rows,model_calls=calls[0],seconds=time.perf_counter()-t0,
        interpretation='Same saved optimized Gaussian arrays; only per-window representative-point RNG changes. No re-optimization, inpainting, or simulator execution.',**code_revision())
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    fig,axes=plt.subplots(1,3,figsize=(14,5))
    for i,(method,_) in enumerate(runs):
        data=[r for r in rows if r['method']==method]
        for seed in sorted({r['seed'] for r in data}):
            selected=[r for r in data if r['seed']==seed]
            values=[[r['object_path_max_change_mm'] for r in selected],
                [1000*(r['validation']['residuals_m']['contact']+.005) for r in selected],
                [max(0.,1000*(r['validation']['residuals_m']['penetration']+.001)) for r in selected]]
            for ax,y in zip(axes,values):
                ax.scatter([i]*len(y),y,alpha=.75,s=25,marker='o' if seed==min(r['seed'] for r in data) else '^')
                ax.scatter([i],[y[0]],facecolors='none',edgecolors='black',s=100)
    for ax,label in zip(axes,['Max object-path change (mm)','Worst held finger gap (mm)','Maximum sampled intersection (mm)']):
        ax.set(xticks=range(len(runs)),xticklabels=[m for m,_ in runs],ylabel=label);ax.tick_params(axis='x',rotation=22);ax.grid(alpha=.2)
    axes[1].axhline(5,c='.4',ls='--');axes[2].axhline(1,c='.4',ls='--')
    fig.suptitle('Sensitivity to representative-point choices\nBlack rings: original routing. Filled points: saved and fresh routing, two original noise draws; execution unverified.')
    fig.tight_layout();fig.savefig(out/'routing_sensitivity.png',dpi=150);plt.close(fig)


if __name__=='__main__': main()
