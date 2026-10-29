"""Re-evaluate optimized noise with fresh FPS randomness, without more optimization.

Run with PYTHONPATH pointing at the experiment's frozen source and its vendor deps.
Every method in a repeat uses the same new per-window RNG states. Initial noise is
unchanged. This separates sensitivity to sampled attention points from goal tuning.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from guidance.pick_objective import PickObjective
from guidance.unconditional import prediction, sample_rollout
from inference import load_model_and_diffusion
from utils.reproducibility import file_sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--routing-seeds', default='81001,81002')
    p.add_argument('--arms', default='sample,noise_goal,noise_interaction,noise_actuated')
    a = p.parse_args()
    root, out = a.root.resolve(), a.out.resolve()
    if out.exists():
        raise FileExistsError(out)
    protocol = json.loads((root/'sample/protocol.json').read_text())
    checkpoint = protocol['arguments']['checkpoint']
    assert file_sha256(checkpoint) == protocol['checkpoint_sha256']
    device = torch.device('cuda')
    batch = torch.load(protocol['arguments']['input'], map_location='cpu', weights_only=False).to(device)
    geometry = torch.load(protocol['arguments']['geometry'], map_location='cpu', weights_only=False)
    model, diffusion, cfg, _ = load_model_and_diffusion(checkpoint, device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    diagnostic = PickObjective(batch, torch.tensor(protocol['target'], device=device), geometry)
    seeds = [row['seed'] for row in json.loads((root/'sample/results.json').read_text())]
    arms = tuple(a.arms.split(','))
    if 'sample' not in arms:
        raise ValueError('Include sample to verify replay and the paired baseline')
    out.mkdir(parents=True)
    rows, replays = [], []
    with torch.no_grad():
        for seed in seeds:
            saved = {arm: torch.load(root/arm/f'seed_{seed}.pt', map_location=device, weights_only=False)['result']['rollout'] for arm in arms}
            # Verify ordinary replay on the actual GPU/model before changing randomness.
            original = saved['sample']
            replay = sample_rollout(model, diffusion, batch, noise=original['noise'],
                                    rng_states=original['rng_states'], **protocol['sampling'])
            difference = float((replay['positions']-original['positions']).abs().max())
            if difference > 5e-6:
                raise AssertionError(f'Saved ordinary rollout failed replay: {difference} m')
            replays.append(dict(seed=seed, max_position_error_m=difference,
                                bitwise_equal=torch.equal(replay['positions'], original['positions'])))
            for routing_seed in map(int, a.routing_seeds.split(',')):
                states = []
                for i in range(len(original['noise'])):
                    torch.manual_seed(routing_seed+i*1009)
                    states.append((torch.get_rng_state(), torch.cuda.get_rng_state(device)))
                torch.save(states, out/f'rng_{seed}_{routing_seed}.pt')
                for arm in arms:
                    torch.cuda.synchronize()
                    begin = time.perf_counter()
                    rollout = sample_rollout(model, diffusion, batch, noise=saved[arm]['noise'],
                                             rng_states=states, **protocol['sampling'])
                    torch.cuda.synchronize()
                    metric = diagnostic.measure(prediction(rollout, batch.context_len))
                    change = (rollout['positions']-saved[arm]['positions']).norm(dim=-1)
                    row = dict(seed=seed, routing_seed=routing_seed, arm=arm, metrics=metric,
                               seconds=time.perf_counter()-begin, forward_nfe=rollout['nfe'],
                               original_position_difference_max_mm=float(change.max())*1000,
                               original_position_difference_mean_mm=float(change.mean())*1000)
                    rows.append(row)
                    torch.save(dict(positions=rollout['positions'].cpu(), R=rollout['R'].cpu(), T=rollout['T'].cpu()),
                               out/f'{arm}_{seed}_{routing_seed}.pt')
                    (out/'results.json').write_text(json.dumps(dict(rows=rows, replays=replays), indent=2)+'\n')
                    print(json.dumps({k:v for k,v in row.items() if k!='metrics'} |
                                     {k:v for k,v in metric.items() if k!='traces'}), flush=True)
    (out/'protocol.json').write_text(json.dumps(dict(source_protocol=protocol,
        routing_seeds=a.routing_seeds, arms=arms, ordinary_replay=replays,
        sampler_source=str(Path(sample_rollout.__code__.co_filename).resolve()),
        scope='Fresh FPS RNG with fixed saved initial noise; no retraining or reoptimization',
        support='Same geometric proxies as the main experiment'), indent=2)+'\n')


if __name__ == '__main__':
    main()
