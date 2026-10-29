"""Real-checkpoint checks for observation-only horizons and obstacle guidance."""
import argparse
import json
from pathlib import Path
import time
import traceback

import torch

from guidance.feedback import rigid_planning_batch
from guidance.mesh_physical import MeshPickPhysicalObjective
from guidance.obstacles import load_obstacles, StaticObstacleDistance, ObstacleObjective
from guidance.unconditional import sample_rollout, prediction
from inference import load_model_and_diffusion
from utils.reproducibility import file_sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','input','geometry','calibration','obstacles','out'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--vendor',action='append',default=[])
    args=p.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    report=dict(status='running',checks=[],source_root=str(Path(__file__).resolve().parents[2]),
        hashes={name:file_sha256(getattr(args,name)) for name in ('checkpoint','input','geometry','calibration','obstacles')})
    started=time.monotonic()
    def save():
        report['seconds']=time.monotonic()-started
        (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    def check(name,condition,**details):
        row=dict(name=name,passed=bool(condition),**details);report['checks'].append(row)
        print(json.dumps(row),flush=True);save()
        if not condition:raise AssertionError(name)
    try:
        if not torch.cuda.is_available():raise RuntimeError('CUDA required')
        b=torch.load(args.input,map_location='cpu',weights_only=False).to('cuda')
        g=torch.load(args.geometry,map_location='cpu',weights_only=False)
        c=torch.load(args.calibration,map_location='cpu',weights_only=False)['calibration']
        model,diffusion,cfg,steps=load_model_and_diffusion(args.checkpoint,torch.device('cuda'))
        for parameter in model.parameters():parameter.requires_grad_(False)
        check('unconditional bidirectional checkpoint',float(cfg.train.action_dropout)==1 and bool(cfg.model.unet_bidirectional_temporal))
        hidden=[True]
        def hook(_,positional,kwargs):
            given=positional[0]
            hidden[0] &= not bool(given.pose_action_mask.any() or given.pose_actions.any())
        handle=model.register_forward_pre_hook(hook,with_kwargs=True)
        sampling=dict(future_len=int(cfg.data.future_len),steps=int(steps),
            schedule=cfg.get('df_schedule','pyramid'),uncertainty_scale=float(cfg.get('df_uncertainty_scale',1.)))
        short=rigid_planning_batch(b,horizon=36);long=rigid_planning_batch(b,horizon=68)
        with torch.no_grad():
            torch.manual_seed(8311);original=sample_rollout(model,diffusion,b,horizon=36,**sampling)
            rebuilt=sample_rollout(model,diffusion,short,horizon=36,noise=original['noise'],rng_states=original['rng_states'],**sampling)
            check('H36 context-only future exact parity',all(torch.equal(original[k],rebuilt[k]) for k in ('positions','R','T')))
            torch.manual_seed(8311);extended=sample_rollout(model,diffusion,long,horizon=68,**sampling)
            check('H68 first two complete windows match H36',torch.equal(extended['positions'][:,:b.context_len+32],original['positions'][:,:b.context_len+32]))
        check('H68 covers all five windows',len(extended['windows'])==5 and extended['positions'].shape[1]==b.context_len+68,
              windows=extended['windows'])
        target=extended['T'][:,-1,0]+b.positions.new_tensor([.01,0,0])
        report['target_use']='Generated endpoint +10 mm: functional gradient probe only, not a task-success experiment'
        base=MeshPickPhysicalObjective(long,target,g,c,vendor_paths=args.vendor,goal_frames=6)
        stronger=MeshPickPhysicalObjective(long,target,g,c,vendor_paths=args.vendor,goal_frames=6,goal_weight=3.)
        pred=prediction(extended,b.context_len)
        with torch.no_grad():
            a,_=base(pred,long);d,_=stronger(pred,long);goal=base.terms(base.features(pred))['goal']
            check('real-sample position weight isolates goal',torch.allclose(d-a,2*goal,atol=2e-7,rtol=1e-5))
        obstacles=StaticObstacleDistance(g,load_obstacles(args.obstacles),vendor_paths=args.vendor)
        objective=ObstacleObjective(base,obstacles)
        noise=[(a,z.clone().requires_grad_()) for a,z in extended['noise']]
        roll=sample_rollout(model,diffusion,long,horizon=68,noise=noise,rng_states=extended['rng_states'],**sampling)
        loss,diagnostics=objective(prediction(roll,b.context_len),long)
        grads=torch.autograd.grad(loss,[z for _,z in noise])
        norms=[float(v.norm()) for v in grads]
        check('H68 obstacle physical objective has finite gradients to all windows',
            bool(torch.isfinite(loss)) and all(bool(torch.isfinite(v).all()) and n>0 for v,n in zip(grads,norms)),
            gradient_norm_by_window=norms,diagnostics=diagnostics)
        check('future actions stay hidden in every forward pass',hidden[0])
        check('model weights remain frozen',all(v.grad is None for v in model.parameters()))
        handle.remove()
        torch.save(dict(initial=extended,diagnostics=diagnostics),out/'extended_sample.pt')
        report['status']='passed'
    except Exception:
        report['status']='failed';report['exception']=traceback.format_exc();raise
    finally:save()


if __name__=='__main__':main()
