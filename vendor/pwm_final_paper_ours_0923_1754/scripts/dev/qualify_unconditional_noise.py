"""Real-checkpoint gate for standard initial-noise optimization, with receipts."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time
import traceback

import torch

from guidance.objectives import body_target
from guidance.unconditional import noise_opt, prediction, sample_rollout
from inference import load_model_and_diffusion, windowed_rollout
from utils.pose_native import pose_channel_mask
from utils.reproducibility import file_sha256, code_revision


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True); p.add_argument('--input',required=True)
    p.add_argument('--out',required=True); args=p.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=False)
    report=dict(status='running',checkpoint_sha256=file_sha256(args.checkpoint),
        input_sha256=file_sha256(args.input),checks=[],**code_revision())
    start=time.monotonic()
    def save():
        report['seconds']=time.monotonic()-start
        (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    def check(name,ok,**details):
        row=dict(name=name,passed=bool(ok),**details); report['checks'].append(row)
        print(json.dumps(row),flush=True); save()
        if not ok: raise AssertionError(name)
    try:
        if not torch.cuda.is_available(): raise RuntimeError('GPU qualification requires CUDA')
        device=torch.device('cuda'); batch=torch.load(args.input,weights_only=False,map_location='cpu').to(device)
        model,diffusion,cfg,steps=load_model_and_diffusion(args.checkpoint,device)
        for parameter in model.parameters(): parameter.requires_grad_(False)
        check('unconditional bidirectional checkpoint',float(cfg.train.action_dropout)==1 and
              bool(cfg.model.unet_bidirectional_temporal))
        C=batch.context_len
        sampling=dict(future_len=int(cfg.data.future_len),steps=int(steps),
            schedule=cfg.get('df_schedule','pyramid'),uncertainty_scale=float(cfg.get('df_uncertainty_scale',1.)))
        report['sampling']=sampling
        for H in (16,32,36):
            torch.manual_seed(6172)
            expected=windowed_rollout(model,diffusion,batch,future_len=sampling['future_len'],horizon=H,
                num_inference_steps=steps,schedule=sampling['schedule'],uncertainty_scale=sampling['uncertainty_scale'],
                hide_given_actions=True)
            torch.manual_seed(6172)
            with torch.no_grad(): initial=sample_rollout(model,diffusion,batch,horizon=H,**sampling)
            errors={k:float((initial[k]-expected[k]).abs().max()) for k in ('positions','R','T')}
            check(f'H{H} production parity',all(v==0 for v in errors.values()),max_errors=errors)
            with torch.no_grad():
                replay=sample_rollout(model,diffusion,batch,horizon=H,noise=initial['noise'],rng_states=initial['rng_states'],**sampling)
            check(f'H{H} exact replay',torch.equal(replay['positions'],initial['positions']))
            mask=pose_channel_mask(batch,initial['rigid_idx']).to(initial['pose_deltas'])
            check(f'H{H} masked joint channels',torch.count_nonzero(initial['pose_deltas']*(1-mask))==0)
            if H==32: reference=initial
            del expected,replay
        H=32
        positions,poses=batch.positions.clone(),batch.body_poses.clone()
        positions[:,C:]+=3; poses[:,C:,:,:3,3]-=2
        corrupted=replace(batch,positions=positions,body_poses=poses,actions=torch.full_like(batch.actions,77),
            pose_actions=torch.full_like(batch.pose_actions,-91),pose_deltas_gt=torch.full_like(batch.pose_deltas_gt,31))
        with torch.no_grad():
            other=sample_rollout(model,diffusion,corrupted,horizon=H,noise=reference['noise'],rng_states=reference['rng_states'],**sampling)
        check('future state/action leakage',torch.equal(other['positions'],reference['positions']))
        target=reference['T'][:,-1,0]+batch.positions.new_tensor([.01,0,0])
        objective=body_target(0,target)
        # This nearby generated-endpoint target is a functional derivative probe,
        # not a picking task or a task-success measurement.
        report['functional_probe_target']='initial generated endpoint +10 mm X; not a task evaluation'
        zero=noise_opt(model,diffusion,batch,objective,initial=reference,iterations=0,horizon=H,**sampling)
        check('zero-update parity',torch.equal(zero['rollout']['positions'],reference['positions']))
        grads_by_policy=[]
        for detached in (False,True):
            noise=[(a,b.detach().clone().requires_grad_()) for a,b in reference['noise']]
            roll=sample_rollout(model,diffusion,batch,horizon=H,noise=noise,rng_states=reference['rng_states'],
                detach_context=detached,**sampling)
            loss=roll['T'][0,-1,0,0]
            gs=torch.autograd.grad(loss,[z[1] for z in noise],allow_unused=True)
            gs=[torch.zeros_like(z[1]) if g is None else g.detach() for g,z in zip(gs,noise)]
            check(f'detach={detached} forward parity',torch.equal(roll['positions'],reference['positions']))
            check(f'detach={detached} finite gradients',all(bool(torch.isfinite(g).all()) for g in gs),norms=[float(g.norm()) for g in gs])
            grads_by_policy.append(gs); del roll,loss,noise
        full,detached=grads_by_policy
        check('early-window gradient route',float(full[0].norm())>1e-7 and float(detached[0].norm())==0 and float(detached[-1].norm())>1e-7)
        index=int(full[0].abs().flatten().argmax()); analytic=float(full[0].flatten()[index]); estimates=[]
        with torch.no_grad():
            for eps in (.001,.0003,.0001):
                vals=[]
                for sign in (-1,1):
                    noise=[(a,b.clone()) for a,b in reference['noise']]
                    noise[0][1].flatten()[index]+=sign*eps
                    roll=sample_rollout(model,diffusion,batch,horizon=H,noise=noise,rng_states=reference['rng_states'],**sampling)
                    vals.append(float(roll['T'][0,-1,0,0]))
                fd=(vals[1]-vals[0])/(2*eps)
                estimates.append(dict(epsilon=eps,finite_difference=fd,absolute_error=abs(fd-analytic)))
        check('early-window finite difference',any(v['absolute_error']<=max(1e-4,.2*abs(analytic)) for v in estimates),
            analytic=analytic,index=index,estimates=estimates)
        result=noise_opt(model,diffusion,batch,objective,initial=reference,iterations=1,lr=.001,
            max_noise_rms=.01,prior_weight=0,horizon=H,**sampling)
        check('one update improves declared goal',result['loss']<result['trace'][0]['loss'],
            initial_loss=result['trace'][0]['loss'],selected_loss=result['loss'],iteration=result['iteration'])
        check('model weights frozen',all(p.grad is None for p in model.parameters()))
        torch.save(dict(initial=reference,result=result),out/'functional_probe.pt')
        report['status']='passed'
    except Exception:
        report['status']='failed'; report['exception']=traceback.format_exc(); raise
    finally: save()


if __name__=='__main__': main()
