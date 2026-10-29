"""Measure saved guidance noise against Gaussian reference draws, without inference."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from utils.pose_native import pose_channel_mask


def statistics(windows):
    flat = np.concatenate([x.reshape(-1) for x in windows])
    products = np.concatenate([(x[:-1]*x[1:]).reshape(-1) for x in windows if len(x)>1])
    return dict(mean=float(flat.mean()), mean_square=float(np.square(flat).mean()),
                adjacent_product=float(products.mean()))


def reference(lengths, channels, count=10000):
    rng = np.random.default_rng(17401)
    windows = [rng.normal(size=(count, length, channels)) for length in lengths]
    flat = np.concatenate(windows, axis=1)
    products = np.concatenate([x[:, :-1]*x[:, 1:] for x in windows if x.shape[1]>1], axis=1)
    values = dict(mean=flat.mean((1, 2)), mean_square=np.square(flat).mean((1, 2)),
                  adjacent_product=products.mean((1, 2)))
    return {key: np.quantile(value, [.025, .975]).tolist() for key, value in values.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True, type=Path)
    root = p.parse_args().root.resolve()
    manifest = json.loads((root/'gallery_manifest.json').read_text())
    entries = {entry['key']: entry for group in manifest['groups'] for entry in group['variants']}
    rows = []; references = {}; cache = {}
    for key, entry in entries.items():
        run = Path(entry['run']); seed = entry.get('seed', 8311)
        source = run/f'seed_{seed}.pt'
        if entry.get('ordinary') or not (run/'results.json').exists(): continue
        protocol = json.loads((run/'protocol.json').read_text())
        if protocol['arguments']['method'] != 'noise_opt' or protocol.get('audit_adaptation'): continue
        if str(source) not in cache:
            cache[str(source)] = torch.load(source, map_location='cpu', weights_only=False)
        saved = cache[str(source)]
        prefix = entry.get('updates')
        selected_source = run/f'seed_{seed}_iterates'/f'through_{prefix:04d}.pt' if prefix is not None else source
        if not selected_source.exists(): continue
        chosen = (torch.load(selected_source, map_location='cpu', weights_only=False)['rollout']
                  if prefix is not None else saved['result']['rollout'])
        batch = torch.load(protocol['arguments']['input'], map_location='cpu', weights_only=False)
        valid = pose_channel_mask(batch, chosen['rigid_idx']).bool().numpy()
        if valid.ndim != 2: raise ValueError('Expected a shared rigid-body channel mask')
        initial = [z[0].numpy()[:, valid] for _, z in saved['initial']['noise']]
        selected = [z[0].numpy()[:, valid] for _, z in chosen['noise']]
        if any(a.shape != b.shape for a, b in zip(initial, selected)) or len(initial)!=len(selected):
            raise ValueError('Noise draw and selected rollout have different windows')
        lengths = [len(x) for x in selected]; channels = int(valid.sum())
        spec = ','.join(map(str, lengths))+' / '+str(channels)+' channels'
        if spec not in references: references[spec] = reference(lengths, channels)
        change = [b-a for a, b in zip(initial, selected)]
        rows.append(dict(key=key, reference=spec, initial=statistics(initial), selected=statistics(selected),
            delta_rms_by_window=[float(np.sqrt(np.square(x).mean())) for x in change],
            scope=protocol['arguments']['scope'],
            source_sha256=hashlib.sha256(selected_source.read_bytes()).hexdigest(),
            protocol_sha256=hashlib.sha256((run/'protocol.json').read_bytes()).hexdigest()))
    result = dict(updated_utc=datetime.now(timezone.utc).isoformat(), rows=rows, references=references,
        definition='All valid pose channels, including frozen passive channels in actuator-only optimization. Adjacent products use within-window pairs only; no centering or variance normalization.',
        reference='Central 95% of 10000 fresh iid standard Gaussian draws per shape, seed 17401. Descriptive baseline, not a physical-validity classifier, calibrated likelihood, or multiple-comparison significance test.')
    (root/'latent_noise_audit.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    render(root, result)


def render(root, result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = result['rows']; labels = [row['key'].replace('_left_', '\n') for row in rows]
    fig, axes = plt.subplots(1, 3, figsize=(15, max(6, .38*len(rows))))
    for axis, (metric, title) in zip(axes, [('mean', 'Mean noise'), ('mean_square', 'Mean squared noise'),
                                          ('adjacent_product', 'Adjacent-frame noise product')]):
        for index, row in enumerate(rows):
            low, high = result['references'][row['reference']][metric]
            axis.plot([low, high], [index, index], color='#bbbbbb', linewidth=5, alpha=.4)
            before, after = row['initial'][metric], row['selected'][metric]
            axis.plot([before, after], [index, index], color='#777777', linewidth=.8)
            axis.scatter(before, index, facecolors='white', edgecolors='#555555', s=30, zorder=3)
            axis.scatter(after, index, color='#a72269', s=24, zorder=4)
        axis.set(title=title, yticks=range(len(rows)), yticklabels=labels if axis is axes[0] else [])
        axis.invert_yaxis(); axis.grid(axis='x', alpha=.2)
    fig.suptitle('Initial draw (open) → selected optimized noise (magenta)\nGrey bars: central 95% of independent Gaussian reference draws with the same shape')
    fig.tight_layout(); (root/'figures').mkdir(exist_ok=True)
    fig.savefig(root/'figures/latent_noise_audit.png', dpi=145); plt.close(fig)
    lines = ['# What happens to the optimized noise?', '', result['definition'], '', result['reference'], '',
        '![Initial and optimized noise statistics](figures/latent_noise_audit.png)', '',
        'The trust cap bounds changes relative to a particular initial draw. It does not enforce every statistic of independent Gaussian noise. '
        'Conversely, unusual marginal statistics do not prove that the decoded motion is impossible. Physics replay remains a separate test.', '',
        '[DNO](https://arxiv.org/html/2312.11994) studies temporal latent decorrelation for human-motion refinement. '
        'That motivates measuring these statistics here; no decorrelation penalty has been added to these runs, and their physical effects have not been established.', '',
        '[Values, definitions and hashes](latent_noise_audit.json) · [Execution scorecard](SCORECARD.md)', '']
    (root/'LATENT_NOISE_AUDIT.md').write_text('\n'.join(lines))


if __name__=='__main__': main()
