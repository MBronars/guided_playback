"""Show how recorded task outcomes differ from the requested guidance goals."""
import argparse
import html
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True,type=Path)
    root=parser.parse_args().root.resolve()
    out=root/'task_reference_report_v1';out.mkdir(exist_ok=True)
    audits=json.loads((root/'recorded_task_screen_v1/summary.json').read_text())['results']
    motion={r['scene']:r for r in json.loads((root/'recorded_task_screen_v1/source_reward_and_motion.json').read_text())['results']}
    fig,axes=plt.subplots(1,3,figsize=(15,5)); pick=[r for r in audits if r['task']=='pick'];place=[r for r in audits if r['task']=='place']
    colors=['#0072b2' if r['group']=='development' else '#d55e00' for r in pick]
    names=[r['scene'].replace('pick_','').replace('_asset_',' · ') for r in pick]
    y=np.arange(len(pick))
    axes[0].barh(y,[motion[r['scene']]['final_lift_mm'] for r in pick],color=colors)
    axes[0].axvspan(80,120,color='#cc0077',alpha=.14,label='Vertical component of point tolerance')
    axes[0].axvline(100,color='#cc0077',ls='--');axes[0].set(yticks=y,yticklabels=names,xlabel='Final height change (mm)',title='Recorded picks lift beyond the point goal')
    axes[0].invert_yaxis();axes[0].legend(fontsize=7,loc='lower right')
    axes[1].barh(y,[motion[r['scene']]['final_xy_travel_mm'] for r in pick],color=colors)
    axes[1].axvline(20,color='#cc0077',ls='--',label='Point goal allows ≤20 mm total 3D error')
    axes[1].set(yticks=y,yticklabels=[],xlabel='Final XY displacement (mm)',title='They also move horizontally')
    axes[1].invert_yaxis();axes[1].legend(fontsize=7)
    # release = requested 5 mm clearance minus the minimum signed finger gap.
    gaps=[5.-1000*r['validation']['residuals_m']['release'] for r in place]
    names_place=[r['scene'].replace('place_','').replace('_asset_',' · ') for r in place]
    axes[2].barh(y,gaps,color=colors)
    axes[2].axvline(5,color='#cc0077',ls='--',label='Requested release clearance')
    axes[2].axvline(0,color='#777777',lw=.8)
    axes[2].set(yticks=y,yticklabels=names_place,xlabel='Final held minimum finger gap (mm)',title='Recorded placing clips retain contact')
    axes[2].invert_yaxis();axes[2].legend(fontsize=7)
    for ax in axes:ax.grid(axis='x',alpha=.2)
    fig.suptitle('Recorded references and the stricter requested guidance task\nBlue: development; orange: reserved for guidance evaluation. All 16 scenes retained.')
    fig.tight_layout(rect=(0,0,1,.9));fig.savefig(out/'reference_task_semantics.png',dpi=140);plt.close(fig)
    rows=[]
    for row in audits:
        values=row['validation']['residuals_m']
        failures=', '.join(k for k,v in values.items() if v>0)
        rows.append(f'<tr><td>{html.escape(row["scene"])}</td><td>{row["goal_held_max_mm"]:.1f}</td><td>{"PASS" if row["validation"]["interaction_accepted"] else "FAIL"}</td><td>{html.escape(failures)}</td></tr>')
    page='''<!doctype html><meta charset="utf-8"><title>What task are we asking for?</title><style>body{font:16px system-ui;line-height:1.5;margin:30px;max-width:1500px}img{max-width:100%}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}table{border-collapse:collapse}</style>
<h1>Our precise guidance task differs from the recorded motions</h1>
<p>For picking, the primary experiment asks the object to finish exactly 100 mm above its observed position, within 20 mm throughout the final six frames. The source clips lift higher and move sideways. For placing, the primary task also asks the gripper to release and the object to settle; the recorded clips retain finger contact at the end.</p>
<p>Consequently, none of these 16 recorded references passes the complete primary task screen. This does not establish that the recorded motions are physically wrong. It means that zero complete-task passes must be interpreted alongside separate goal, contact and collision measurements. It is not the source environment's success rate.</p>
<img src="reference_task_semantics.png">
<p>The pink band on the first plot shows only the vertical part of the 20 mm point tolerance; the real criterion uses 3D distance. The placing plot shows signed mesh separation: negative values mean overlap, and positive values mean a gap. All original targets and results are retained.</p>
<p>The separate workspace-lift experiment asks for at least 100 mm lift within a 200 × 200 mm XY workspace. That is a different exploratory task, motivated after examining these references. Its geometric checks still do not establish physical execution.</p>
<table><tr><th>Recorded scene</th><th>Worst held point distance (mm)</th><th>Interaction-only screen</th><th>Failed complete-task criteria</th></tr>'''+''.join(rows)+'''</table>
<p>Recorded references are read only for evaluation and controller qualification. Their future states and controls are not supplied to unconditional generation or latent-noise optimization. The exact historical collection launch has not been recovered; current collector code alone cannot establish its original success definition.</p>
<p><a href="../recorded_task_screen_v1/summary.json">Complete reference measurements</a> · <a href="../region_replay_report_v1/index.html">Workspace lifts: model versus physics</a> · <a href="../index.html">Main report</a></p>'''
    (out/'index.html').write_text(page)
    print('Rendered all16 reference task comparisons')


if __name__=='__main__':main()
