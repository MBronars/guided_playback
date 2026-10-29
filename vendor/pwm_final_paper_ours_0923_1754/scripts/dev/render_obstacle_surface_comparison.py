"""Before/after obstacle costs, with identical collision worlds in both replays."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

from scripts.dev.render_guidance_surfaces import render
from utils.reproducibility import file_sha256


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True,type=Path)
    root=p.parse_args().root.resolve();folder=root/'obstacle_surface_figures';folder.mkdir(exist_ok=True)
    lines=['# Obstacle avoidance: before and after','',
        '**Top row:** model predictions. **Bottom row:** independent physics executing each generated gripper plan. '+
        '**Left:** the original plan ignores the barrier. **Right:** the planner penalizes proximity to the barrier.',
        '', 'The same red barrier, observation, target, geometry and controller appear in all panels. '+
        'The original plan is frozen and replayed in the obstacle world. It is not reoptimized for this comparison. '+
        'The replay timestep is 0.25 ms; paired 0.5/0.25 ms checks remain in the linked audits.',
        '', '[Complete model comparisons](GALLERY.md) · [Execution checks](EXECUTION_GALLERY.md)','']
    summary=[]
    for scene,label in [('bottle_8270','Bottle'),('teacup_7803','Teacup')]:
        keys=[scene+'_left_ignored_obstacle',scene+'_left_obstacle_h36_i64_r050'];loaded=[]
        for key in keys:
            export=root/'exports'/key;replay=root/'replays'/key/'dt00025';audit_path=replay.parent/'audit.json'
            if not audit_path.exists():break
            protocol=json.loads((export/'protocol.json').read_text());audit=json.loads(audit_path.read_text())
            planned=np.load(export/'trajectory.npz');executed=np.load(replay/'execution.npz');C=int(planned['context_len'])
            np.testing.assert_array_equal(executed['desired_T'],planned['T'][0,C-1:])
            np.testing.assert_array_equal(executed['desired_R'],planned['R'][0,C-1:])
            loaded.append(dict(protocol=protocol,audit=audit,plan=(planned['R'][0,C-1:],planned['T'][0,C-1:]),
                replay=(executed['model_R'],executed['model_T']),hashes=[file_sha256(export/'trajectory.npz'),file_sha256(replay/'execution.npz')]))
        lines+=['## '+label,'']
        if len(loaded)!=2:lines+=['Pending.',''];continue
        a,b=[v['protocol'] for v in loaded]
        for field in ('input_sha256','geometry_sha256','obstacles_sha256','target','sampling'):
            if a[field]!=b[field]:raise ValueError('Inconsistent comparison: '+field)
        geometry=torch.load(a['arguments']['geometry'],weights_only=False,map_location='cpu')
        if file_sha256(a['arguments']['geometry'])!=a['geometry_sha256']:raise ValueError('Changed geometry')
        sys.path.extend(a['arguments']['collision_vendor'])
        signature=hashlib.sha256(json.dumps([v['hashes'] for v in loaded]).encode()+
            Path(__file__).read_bytes()+Path(sys.modules[render.__module__].__file__).read_bytes()).hexdigest()[:12]
        movie=folder/f'{scene}_{signature}.gif';still=movie.with_suffix('.png')
        if not movie.exists():
            tracks=[loaded[0]['plan'],loaded[1]['plan'],loaded[0]['replay'],loaded[1]['replay']]
            render(scene,b,geometry,tracks,movie,still,panel_titles=['Model: no avoidance cost','Model: avoidance cost',
                'Physics: original plan','Physics: avoidance plan'],heading=label+': barrier avoidance comparison')
        lines+=[f'![{label} obstacle comparison]({movie.relative_to(root)})','',
            '| Plan | Planned minimum clearance | Planned final goal error | Replay final goal error | Replay obstacle-contact solver steps |',
            '|---|---:|---:|---:|---:|']
        for key,v in zip(keys,loaded):
            m=v['audit']['model_metrics'];r=v['audit']['replays'][1]
            lines.append(f'| {"Original" if "ignored" in key else "Avoidance"} | {m["validation"]["obstacle_min_clearance_mm"]:.1f} mm | {m["final_goal_mm"]:.1f} mm | {r["final_goal_mm"]:.1f} mm | {r["forces"]["obstacle_contact_solver_steps"]} |')
        lines+=['',f'[Final frame]({still.relative_to(root)}) · [Original-plan audit](replays/{keys[0]}/audit.json) · [Avoidance-plan audit](replays/{keys[1]}/audit.json)','']
        summary.append(dict(scene=scene,keys=keys,movie=str(movie.relative_to(root)),still=str(still.relative_to(root))))
    lines+=['Planned clearance uses ten per-body interpolation probes per transition, not exact continuous collision detection. '+
        'The replay audits contacts at every solver step. Avoiding a barrier in the model does not ensure reaching the goal or executing the plan.','']
    (root/'OBSTACLE_GALLERY.md').write_text('\n'.join(lines));(root/'obstacle_surface_summary.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':main()
