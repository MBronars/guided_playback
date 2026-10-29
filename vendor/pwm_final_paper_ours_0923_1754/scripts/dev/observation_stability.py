"""Separate observation sensitivity from ordinary diffusion-sample variation.

Six sampler draws per fixed clip, with the same draw paired to 1 mm history
jitter. This is a conditional sample-spread diagnostic, not calibration of
uncertainty about hidden physical state or materials.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from scripts.dev.observation_probe_common import Predictor, corrupt_observation, observed_input, write_json, write_frozen_run
from scripts.dev.overnight_robustness import fixtures
from utils.reproducibility import code_revision, file_sha256, evaluation_seed


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint',required=True);ap.add_argument('--root',required=True)
    ap.add_argument('--wait-for-core',action='store_true');ap.add_argument('--max-wait-hours',type=float,default=4.5)
    a=ap.parse_args();root=Path(a.root);out=root/'stability';out.mkdir(parents=True,exist_ok=True)
    start=time.monotonic()
    while a.wait_for_core:
        path=root/'robustness/status.json'
        state=json.loads(path.read_text()) if path.exists() else {}
        if state.get('state')=='sampling_finished':break
        if time.monotonic()-start>a.max_wait_hours*3600:
            write_json(out/'status.json',{'state':'not_run','reason':'Core sweep did not finish within wait budget'});return
        write_json(out/'status.json',{'state':'waiting_for_core','core_completed':state.get('completed',0)})
        time.sleep(30)
    torch.set_num_threads(8);predictor=Predictor(a.checkpoint)
    batches,manifest=fixtures(Path('validation/architecture_review_2026-09-12/deep_dive'),predictor.cfg,root/'cloth_fixtures/manifest.json')
    selected=[f'{d}_{i}_row0' for d in ('mpm','pickup','pushing') for i in range(4)]
    selected += [f'cloth_middle_{i}_0' for i in range(4)]
    seeds=list(range(710001,710007));results=[]
    write_frozen_run(out/'run.json',{'checkpoint':str(Path(a.checkpoint).resolve()),'checkpoint_sha256':file_sha256(a.checkpoint),
                              'code':code_revision(),'clips':[x for x in manifest if x['key'] in selected],
                              'sampler_seeds':seeds,'history_jitter_mm':1,'corruption_seed':9201,
                              'note':'Sampler seeds vary diffusion noise and randomized FPS. Conditional sample spread does not measure state-estimation uncertainty or establish calibration'},
                     ('clips','sampler_seeds','history_jitter_mm','corruption_seed'))
    for key in selected:
        target=out/(key+'.json')
        if target.exists():results.append(json.loads(target.read_text()));continue
        batch=batches[key];noise_seed=evaluation_seed(9201,0,key+'history_jitter')
        corrupted=corrupt_observation(batch,'history_jitter',1,noise_seed)
        valid=batch.get_valid_mask()[0] & ~batch.action_mask[0]
        clean_draws=[];noisy_draws=[];seconds=0;calls=0
        for seed in seeds:
            clean,timing=predictor(observed_input(batch),seed);seconds+=timing['seconds'];calls+=timing['model_calls']
            noisy,timing=predictor(corrupted,seed);seconds+=timing['seconds'];calls+=timing['model_calls']
            clean_draws.append(clean[0,:,valid]);noisy_draws.append(noisy[0,:,valid])
        clean=torch.stack(clean_draws);noisy=torch.stack(noisy_draws);truth=batch.future[0,:,valid]
        # Every estimator uses Euclidean 3D displacement, then RMS across particle/frame/draw.
        spread=(clean-clean.mean(0)).square().sum(-1).mean().sqrt()*1000
        response=(noisy-clean).square().sum(-1).mean().sqrt()*1000
        clean_error=(clean-truth).norm(dim=-1).mean()*1000
        noisy_error=(noisy-truth).norm(dim=-1).mean()*1000
        result={'clip':key,'domain':batch.domain,'conditional_sample_rms_spread_mm':float(spread),
                'matched_seed_1mm_input_response_rms_mm':float(response),
                'clean_mean_error_mm':float(clean_error),'noisy_mean_error_mm':float(noisy_error),
                'per_frame_spread_mm':((clean-clean.mean(0)).square().sum(-1).mean((0,2)).sqrt()*1000).tolist(),
                'per_frame_input_response_mm':((noisy-clean).square().sum(-1).mean((0,2)).sqrt()*1000).tolist(),
                'seconds':seconds,'model_calls':calls,'n_draws':len(seeds)}
        write_json(target,result);results.append(result)
        write_json(out/'status.json',{'state':'running','completed':len(results),'planned':len(selected)})
        print('STABILITY',json.dumps(result),flush=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    for i,domain in enumerate(('mpm','pickup','pushing','cloth')):
        values=[x for x in results if x['domain']==domain]
        for shift,key,color,label in ((-.18,'conditional_sample_rms_spread_mm','#6c8dbd','Conditional sample spread'),
                                      (.18,'matched_seed_1mm_input_response_rms_mm','#d98c38','Adding 1 mm history jitter')):
            y=np.array([x[key] for x in values]);axes[0].bar(i+shift,y.mean(),width=.34,color=color,label=label if i==0 else None)
            axes[0].scatter(i+shift+np.linspace(-.07,.07,len(y)),y,c='#263344',s=12)
        axes[1].plot(np.arange(1,17),np.mean([x['per_frame_input_response_mm'] for x in values],axis=0),label=domain)
    axes[0].set(xticks=range(4),xticklabels=['mpm','pickup','pushing','cloth'],ylabel='Prediction displacement RMS (mm)',
                title='Sampling variation versus input sensitivity')
    axes[0].legend(fontsize=8);axes[1].set(xlabel='Future frame',ylabel='Paired prediction response RMS (mm)',title='Response to 1 mm history jitter')
    axes[1].legend();fig.suptitle('Six draws per clip; four fixed clips per domain | sample spread, not calibrated uncertainty',fontsize=11)
    fig.savefig(out/'input_response_vs_sample_spread.png',dpi=160);plt.close(fig)
    write_json(out/'summary.json',{'clips':results,'estimator':'RMS Euclidean displacement over draws, frames and passive particles',
                                  'interpretation':'A paired input response measures sensitivity; conditional diffusion spread alone is not state uncertainty'})
    write_json(out/'status.json',{'state':'finished','completed':len(results),'planned':len(selected)})


if __name__=='__main__':main()
