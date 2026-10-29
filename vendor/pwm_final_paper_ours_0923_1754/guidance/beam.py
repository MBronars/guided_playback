"""Stage-aware beam search over complete diffusion-forcing windows.

Branches draw fresh next-window noise; this is search, not a weighted SMC sampler.
Only complete trajectories are eligible for the independent pick validator.
"""
import torch

from guidance.objectives import materialize
from guidance.pick_validation import selection_key
from guidance.planners import window_nfe
from guidance.sampler import draw_noise, sample_window
from guidance.unconditional import _check_model, _rng_state, prediction
from inference import rollout_window, window_bounds
from utils.conditioning import hide_actions


@torch.no_grad()
def beam_search(model,diffusion,batch,objective,*,width=2,branches=4,future_len=16,
                horizon=None,steps=25,schedule='pyramid',uncertainty_scale=1.,
                validator=None,callback=None,candidate_callback=None,
                diversity_body=1,diversity_translation=.005,diversity_rotation=.087):
    _check_model(model,batch)
    H=batch.num_future_frames if horizon is None else int(horizon)
    if batch.batch_size!=1 or min(width,branches,H,future_len)<1 or H>batch.num_future_frames:
        raise ValueError('Beam search requires one scene and positive valid sizes')
    C=batch.context_len
    ridx=(batch.body_type[0]==1).nonzero(as_tuple=True)[0]
    if diversity_body not in ridx.tolist() or min(diversity_translation,diversity_rotation)<0:
        raise ValueError('Invalid beam diversity body or threshold')
    hand=ridx.tolist().index(diversity_body)
    initial=dict(positions=batch.context.clone(),R=batch.body_poses[:,:C,ridx,:3,:3].float().clone(),
        T=batch.body_poses[:,:C,ridx,:3,3].float().clone(),rigid_idx=ridx,
        pose_deltas=torch.empty(1,0,len(ridx),6,device=batch.device),noise=[],rng_states=[],
        windows=[],horizon=H,nfe=0,guidance_trace=[])
    beam=[dict(rollout=initial,lineage=[])]; total=0; trace=[]
    while beam[0]['rollout']['positions'].shape[1]<C+H:
        candidates=[]
        for parent in beam:
            old=parent['rollout']; pos,R,t=old['positions'],old['R'],old['T']
            start,length,skip=window_bounds(pos.shape[1],C+H,future_len,C)
            win=rollout_window(batch,pos,start,length,rigid_idx=ridx,R_track=R,T_track=t)
            for branch in range(branches):
                noise=draw_noise(model,win); state=_rng_state(batch.device)
                xp,xr=sample_window(model,diffusion,hide_actions(win,1.),noise,steps=steps,
                    schedule=schedule,uncertainty_scale=uncertainty_scale,use_checkpoint=False)
                pred=materialize(win,xp,xr)
                calls=window_nfe(diffusion,win,steps,schedule,uncertainty_scale); total+=calls
                out=dict(positions=torch.cat([pos,pred['positions'][:,skip:]],1),
                    R=torch.cat([R,pred['R'][:,skip:]],1),T=torch.cat([t,pred['t'][:,skip:]],1),
                    rigid_idx=ridx,pose_deltas=torch.cat([old['pose_deltas'],xr[:,skip:]],1),
                    noise=old['noise']+[noise],rng_states=old['rng_states']+[state],
                    windows=old['windows']+[(start,length,skip)],horizon=H,nfe=old['nfe']+calls,guidance_trace=[])
                complete=out['positions'].shape[1]==C+H
                cost=objective.prefix_loss(prediction(out,C),complete=complete)
                if not bool(torch.isfinite(cost)): raise FloatingPointError('Nonfinite beam score')
                validation=validator(prediction(out,C)) if complete and validator is not None else None
                lineage=parent['lineage']+[branch]
                row=dict(lineage=lineage,frames=out['positions'].shape[1]-C,loss=float(cost),
                    complete=complete,validation=validation)
                trace.append(row)
                if callback is not None: callback(row)
                if complete and candidate_callback is not None:
                    candidate_callback(len(candidates),out,row)
                candidates.append(dict(rollout=out,lineage=lineage,loss=float(cost),validation=validation,
                    key=selection_key(float(cost),validation)))
        candidates.sort(key=lambda item:item['key'])
        if complete:
            best=candidates[0]
            return dict(rollout=best['rollout'],loss=best['loss'],validation=best['validation'],
                lineage=best['lineage'],trace=trace,nfe=total,width=width,branches=branches)
        # Prefer distinct terminal hand poses when available, while always keeping
        # the best scoring prefix. Diversity changes selection, not sample weights.
        beam=[]; deferred=[]
        for item in candidates:
            out=item['rollout']
            distinct=all(float((out['T'][:,-1,hand]-b['rollout']['T'][:,-1,hand]).norm())>diversity_translation
                or float((out['R'][:,-1,hand]-b['rollout']['R'][:,-1,hand]).norm()/2**.5)>diversity_rotation
                for b in beam)
            if distinct and len(beam)<width: beam.append(item)
            else: deferred.append(item)
        beam.extend(deferred[:max(0,width-len(beam))])
