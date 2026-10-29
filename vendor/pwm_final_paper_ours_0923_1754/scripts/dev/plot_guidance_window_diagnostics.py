"""Plot saved optimization traces by rollout window; no additional model calls."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


ARMS = [
    ('left_i64_r025', '36 frames, all noise, cap 0.25'),
    ('left_i64_r050', '36 frames, all noise, cap 0.50'),
    ('left_long_h68_i64_r050', '68 frames, all noise, cap 0.50'),
    ('left_max_actuated_i64_r050', '36 frames, actuator noise, max goal'),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    root = parser.parse_args().root.resolve()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    lines = ['# How the optimizer uses multiple windows', '',
        'Each color denotes a successive rollout window. Top: used fraction of the per-window noise-change RMS cap. '
        'Bottom: the saved total-objective gradient L2 norm divided by the square root of active noise coordinates in that window.', '',
        'A value near 1 in the top row means that window is at its allowed noise-change bound. '
        'A nonzero gradient means the total objective can influence that window; it does not by itself isolate the gradient from a later window. '
        'The separate sampler qualification checks cross-window gradient propagation.', '',
        'The bottom row accounts for the shorter four-frame tail. Gradients include physical costs and the noise prior, '
        'so their sizes are not direct measures of physical causality or task difficulty. Adam also rescales updates using its gradient history.', '',
        'The longer horizon changes physical time and compute. The actuator-only arm also uses a maximum held-position cost; '
        'its comparison here is descriptive, not a one-factor ablation.', '']
    records = []
    folder = root/'figures'; folder.mkdir(exist_ok=True)
    for scene, title in [('bottle_8270', 'Bottle'), ('teacup_7803', 'Teacup')]:
        fig, axes = plt.subplots(2, len(ARMS), figsize=(17, 7), squeeze=False, sharey='row')
        for column, (arm, label) in enumerate(ARMS):
            directory = root/(scene+'_'+arm)
            if not (directory/'results.json').exists():
                for axis in axes[:, column]:
                    axis.text(.5, .5, 'Pending', transform=axis.transAxes, ha='center'); axis.set_axis_off()
                axes[0, column].set_title(label)
                continue
            protocol = json.loads((directory/'protocol.json').read_text())
            args = protocol['arguments']; sampling = protocol['sampling']
            if protocol['checkpoint_sha256'] != '99ec6cc6102a0d654c375c9f7b9cc9c86edd0964acf9de1acb42d4fb578c9fba':
                raise ValueError('Active-channel counts are qualified only for this campaign checkpoint')
            if args.get('temporal_knots', 0) or args.get('residual_max_rms', 0):
                raise ValueError('Per-frame coordinate normalization does not cover temporal-knot parameters')
            if args['scope'] not in ('all', 'actuated'):
                raise ValueError('Only fixed noise scopes are supported')
            # This campaign's articulated pickup representation: object 6,
            # hand 6, two scalar finger joints. No invalid channels are counted.
            channels = 14 if args['scope']=='all' else 8
            horizon, window = sampling['horizon'], sampling['future_len']
            lengths = [min(window, horizon-start) for start in range(0, horizon, window)]
            trace_path = directory/'seed_8311_trace.jsonl'
            trace = [json.loads(line) for line in trace_path.read_text().splitlines()]
            cap = args['max_noise_rms']
            rms = np.array([row['noise_rms_by_window'] for row in trace])
            grad_rows = [row for row in trace if 'gradient_norm_by_window' in row]
            gradients = np.array([row['gradient_norm_by_window'] for row in grad_rows])
            if rms.shape[1] != len(lengths) or gradients.shape[1] != len(lengths):
                raise ValueError('Trace window count differs from the declared sampling protocol')
            if not np.isfinite(rms).all() or not np.isfinite(gradients).all() or (rms > cap+1e-6).any():
                raise ValueError('Invalid saved optimization trace')
            normalized = gradients/np.sqrt(np.array(lengths)*channels)[None]
            for index, length in enumerate(lengths):
                name = f'Window {index+1} ({length} frames)'
                color = plt.get_cmap('tab10')(index)
                axes[0, column].plot([row['iteration'] for row in trace], rms[:, index]/cap, color=color, label=name)
                axes[1, column].plot([row['iteration'] for row in grad_rows], np.maximum(normalized[:, index], 1e-12), color=color)
            axes[0, column].axhline(1, color='#777777', linestyle=':', linewidth=.8)
            axes[0, column].set(ylim=(0, 1.06), title=label)
            axes[0, column].legend(fontsize=7, loc='lower right')
            axes[1, column].set(yscale='log', xlabel='Adam update')
            for axis in axes[:, column]: axis.grid(alpha=.18)
            records.append(dict(key=directory.name, window_lengths=lengths, active_channels=channels,
                trace_sha256=hashlib.sha256(trace_path.read_bytes()).hexdigest(),
                final_rms_fraction=(rms[-1]/cap).tolist(),
                median_coordinate_gradient_rms=np.median(normalized, axis=0).tolist(),
                last_gradient_update=grad_rows[-1]['iteration']))
        axes[0, 0].set_ylabel('Used noise-change RMS / allowed RMS')
        axes[1, 0].set_ylabel('Total-objective gradient RMS per coordinate')
        fig.suptitle(title+': noise budget and gradients across rollout windows')
        fig.tight_layout()
        filename = scene+'_window_diagnostics.png'
        fig.savefig(folder/filename, dpi=145); plt.close(fig)
        lines.extend(['## '+title, '', f'![{title}: window diagnostics](figures/{filename})', ''])
    lines.extend(['These are traces of individual optimizations, without smoothing or seed averaging. '
        'They include all evaluated iterates; the returned trajectory is the best weighted objective seen so far.', '',
        '[Trace hashes and numeric summaries](window_diagnostics.json) · [Goal and execution scorecard](SCORECARD.md)', ''])
    (root/'window_diagnostics.json').write_text(json.dumps(dict(updated_utc=datetime.now(timezone.utc).isoformat(), records=records), indent=2)+'\n')
    (root/'WINDOW_DIAGNOSTICS.md').write_text('\n'.join(lines))


if __name__=='__main__': main()
