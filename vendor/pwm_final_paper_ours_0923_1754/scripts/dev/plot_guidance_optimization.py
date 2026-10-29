"""Plot paired optimization traces without confusing mean loss with held success."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',required=True,type=Path)
    root=parser.parse_args().root.resolve();fig,axes=plt.subplots(2,2,figsize=(12,8))
    for row,(scene,title) in enumerate([('bottle_8270','Bottle left'),('teacup_7803','Teacup left')]):
        for cap,color in [(25,'#2876ac'),(50,'#d76c2c')]:
            run=root/f'{scene}_left_i64_r{cap:03d}';file=run/'seed_8311_trace.jsonl'
            if not file.exists():continue
            lines=file.read_text().splitlines();values=[]
            for line in lines:
                try:values.append(json.loads(line))
                except json.JSONDecodeError:break
            if not values:continue
            updates=[v['iteration'] for v in values]
            axes[row,0].plot(updates,[v['goal_cost_mm'] for v in values],color=color,label=f'Cap {cap/100:.2f}: mean held error')
            axes[row,0].plot(updates,[v['target_distance_mm'] for v in values],color=color,ls=':',alpha=.65,label=f'Cap {cap/100:.2f}: endpoint')
            for index,style in enumerate(['-', '--', ':']):
                axes[row,1].plot(updates,[v['noise_rms_by_window'][index] for v in values],color=color,ls=style,
                    label=f'Cap {cap/100:.2f}, window {index+1}')
            axes[row,1].axhline(cap/100,color=color,alpha=.2)
        for ax in axes[row]:
            ax.axvline(24,color='#555555',ls='--',alpha=.4);ax.grid(alpha=.2);ax.set_xlabel('Optimization updates')
            ax.legend(fontsize=8)
        axes[row,0].set(title=title+' / goal progress',ylabel='Error (mm)')
        axes[row,1].set(title=title+' / noise-change RMS',ylabel='RMS over valid pose channels')
    fig.suptitle('Same checkpoint, seed, target and physical costs\nSolid: mean held goal cost; dotted: endpoint error. Neither alone certifies every held frame.')
    fig.tight_layout(rect=(0,0,1,.93));out=root/'figures';out.mkdir(exist_ok=True)
    fig.savefig(out/'optimization_progress.png',dpi=140);plt.close(fig)
    print(out/'optimization_progress.png')


if __name__=='__main__':main()
