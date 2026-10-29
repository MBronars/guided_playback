"""Search observed-geometry actuator paths with paired stochastic object futures."""
import torch

from guidance.pick_validation import selection_key
from guidance.unconditional import sample_rollout, prediction
from guidance.waypoints import grasp_waypoints


@torch.no_grad()
def waypoint_search(model,diffusion,batch,target,grasp,*,initial,validator,
                    proposals=8,phases=((8,14),(12,18),(16,22)),hold_frames=6,scope='actuated',
                    callback=None,candidate_callback=None,**sampling):
    """Prescribe actuator deltas only; rank independently measured interactions.

    Every candidate reuses the initial rollout's Gaussian draws and FPS states.
    Physical phases and contact proposals come from observed scene geometry.
    """
    if validator is None or proposals<1 or not phases or not grasp['proposals']:
        raise ValueError('Waypoint search requires proposals, phases and independent validation')
    H=sampling.get('horizon',batch.num_future_frames)
    if len(set(phases))!=len(phases): raise ValueError('Duplicate physical stage schedules')
    if any(not 0<a<c<H-hold_frames+1 for a,c in phases):
        raise ValueError('Every stage schedule must fit the reach, closure, lift and hold')
    best=None; best_key=None; trace=[]; nfe=0
    for proposal in range(min(proposals,len(grasp['proposals']))):
        for approach,close in phases:
            control=grasp_waypoints(batch,target,grasp,proposal=proposal,horizon=H,
                approach_frames=approach,close_frame=close,hold_frames=hold_frames,scope=scope)
            out=sample_rollout(model,diffusion,batch,noise=initial['noise'],rng_states=initial['rng_states'],
                known_pose=control['known_pose'],known_mask=control['known_mask'],**sampling)
            pred=prediction(out,batch.context_len)
            loss=float((pred['t'][:,-1,0]-target).norm(dim=-1).mean())
            validation=validator(pred); key=selection_key(loss,validation)
            nfe+=out['nfe']
            row=dict(draw=len(trace),proposal=proposal,approach_frames=approach,close_frame=close,
                loss=loss,validation=validation)
            trace.append(row)
            if best is None or key<best_key:
                best_key=key; best=dict(rollout=out,control=control,loss=loss,validation=validation,
                    draw=row['draw'],proposal=proposal,approach_frames=approach,close_frame=close)
            if callback is not None: callback(row)
            if candidate_callback is not None: candidate_callback(row['draw'],out,row)
    return dict(**best,trace=trace,nfe=nfe,
        pairing='same Gaussian draws and FPS RNG states across all actuator paths',scope=scope)
