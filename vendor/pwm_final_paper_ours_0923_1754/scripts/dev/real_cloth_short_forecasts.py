"""Fixed short-horizon real-cloth probes, restarting from measured observations."""
import argparse
import html
import json
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from inference import windowed_rollout
from scripts.dev.observation_probe_common import Predictor, require_given_actions, write_json, write_frozen_run
from scripts.dev.real_cloth_probe import build_input, measure, render_real
from utils.checkpoint import checkpoint_config, read_checkpoint
from utils.reproducibility import code_revision, file_sha256, seed_for_device


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--source', default='validation/public_real_cloth_2026-09-14')
    ap.add_argument('--motion-smoothing-radius-mm', type=float, default=0)
    a = ap.parse_args()
    if a.motion_smoothing_radius_mm < 0:
        raise ValueError('Motion smoothing radius cannot be negative')
    torch.set_num_threads(8)
    source, out = Path(a.source), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = checkpoint_config(read_checkpoint(a.checkpoint))
    if int(cfg.data.context_len) != 4 or int(cfg.data.future_len) != 16:
        raise ValueError('This diagnostic uses the C4/F16 checkpoint contract')
    require_given_actions(cfg)
    captures = sorted(p.name for p in (source/'green_tshirt').iterdir() if p.is_dir())
    specs = []
    for capture in captures:
        files = sorted((source/'green_tshirt'/capture/'segment_pcds').glob('*.pcd'))
        stamps = np.array([float(p.name.split('_')[1]) for p in files])
        # Fixed fractions of the usable interval, selected before any short forecasts.
        end = float(stamps[-1]-stamps[0]) - 16/30 - .15
        for i, fraction in enumerate((.15, .35, .55, .75)):
            specs.append({'capture': capture, 'window': i,
                          'anchor_offset': float(.3 + fraction*(end-.3))})
    write_frozen_run(out/'run.json', {
        'checkpoint': str(Path(a.checkpoint).resolve()), 'checkpoint_sha256': file_sha256(a.checkpoint),
        'code': code_revision(), 'specs': specs, 'seeds': [710001, 710002, 710003],
        'source_manifest_sha256': file_sha256(source/'selected_captures.json'),
        'protocol': 'Four fixed anchors per capture; each C4/F16 forecast starts from measured history. These are independent half-second forecasts, not one long rollout.',
        'tracking': 'icp', 'motion_smoothing_radius_mm': a.motion_smoothing_radius_mm,
        'selection': '15%, 35%, 55%, 75% of usable capture time, before inspecting these predictions',
        'time_unix': time.time()}, ('specs','seeds','source_manifest_sha256','tracking','motion_smoothing_radius_mm','protocol'))
    predictor = Predictor(a.checkpoint)
    cases = []
    for spec in specs:
        capture = spec['capture']; stem = f"{capture}__window{spec['window']}"
        try:
            batch, receipt, metadata = build_input(source, capture, cfg,
                anchor_offset=spec['anchor_offset'], horizon=16, tracking='icp',
                motion_smoothing_radius=a.motion_smoothing_radius_mm/1000)
            if batch.num_future_frames != 16:
                raise ValueError('Incomplete short forecast')
            torch.save(batch, out/(stem+'_input.pt'))
            write_json(out/(stem+'_input.json'), receipt)
        except Exception:
            cases.append({'key': stem, 'status': 'failed', 'error': traceback.format_exc()})
            write_json(out/'status.json', {'state': 'running', 'cases': cases})
            continue
        for seed in (710001, 710002, 710003):
            key = stem+'__seed'+str(seed)
            try:
                b = batch.to(predictor.device)
                seed_for_device(seed, predictor.device)
                predictor.calls = 0
                torch.cuda.synchronize(); start = time.monotonic()
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    result = windowed_rollout(predictor.model, predictor.diffusion, b,
                        horizon=16, future_len=16, num_inference_steps=predictor.steps,
                        schedule=cfg.get('df_schedule', 'pyramid'),
                        uncertainty_scale=float(cfg.get('df_uncertainty_scale', 1.)), clamp_actuated=True)
                torch.cuda.synchronize(); seconds = time.monotonic()-start
                pred = result['positions'][0].float().cpu().numpy()
                if not np.isfinite(pred).all():
                    raise ValueError('Non-finite prediction')
                values, clouds, colors, times = measure(pred, batch, metadata, receipt)
                if not values:
                    raise ValueError('No future measured surfaces inside this forecast')
                np.savez_compressed(out/(key+'.npz'), prediction=pred,
                                    context=batch.context[0].numpy(), model_times=receipt['model_times'])
                case = {'key': key, 'status': 'ok', 'capture': capture, 'window': spec['window'],
                        'seed': seed, 'seconds': seconds, 'model_calls': predictor.calls,
                        'input': receipt, 'metrics': values, 'clamp_given_actuators': True,
                        'mean_surface_mm': float(np.mean([m['measured_surface_to_prediction_mean_mm'] for m in values])),
                        'static_surface_mm': float(np.mean([m['measured_surface_to_static_anchor_mean_mm'] for m in values]))}
                write_json(out/(key+'.json'), case)
                if seed == 710001:
                    render_real(out, key, pred, batch, metadata, receipt, values, clouds, colors, times)
                cases.append(case)
                print('SHORT', key, case['mean_surface_mm'], case['static_surface_mm'], flush=True)
            except Exception:
                cases.append({'key': key, 'status': 'failed', 'error': traceback.format_exc()})
            write_json(out/'status.json', {'state': 'running', 'completed': len(cases), 'planned': len(specs)*3})
    chunks = ['<!doctype html><meta charset="utf-8"><title>Short real-cloth forecasts</title>',
              '<style>body{font:17px/1.5 system-ui;max-width:1200px;margin:auto;padding:32px}video,img{max-width:100%}section{margin:32px 0}</style>',
              '<h1>Half-second forecasts from fresh real observations</h1>',
              '<p>Each clip starts again from a measured cloth history. These clips are independent short predictions, not a continuous long forecast. Four anchors per recording were fixed by capture time; all three sampler seeds are retained. Videos show the predeclared primary seed.</p>',
              '<p>Inputs use approximate coherent surface tracking, known recorded robot motion and training-default cloth properties. The reference is partial depth geometry. Distances below are measured surface to predicted surface, not full-shape errors.</p>',
              f'<p>Optional observed-motion smoothing radius: {a.motion_smoothing_radius_mm:g} mm. This averages local estimated displacements while keeping the measured anchor and robot inputs unchanged. It can also blur true local motion.</p>',
              '<p><a href="run.json">Frozen protocol</a> | <a href="summary.json">All measurements</a></p>']
    for case in cases:
        if case['status'] != 'ok':
            chunks.append('<pre>'+html.escape(str(case))+'</pre>')
        elif case['seed'] == 710001:
            key = case['key']
            chunks += [f'<section><h2>{html.escape(key)}</h2>',
                       f"<p>Anchor: {case['input']['anchor_offset_seconds']:.2f} s after capture start. Mean surface distance: {case['mean_surface_mm']:.1f} mm; static anchor: {case['static_surface_mm']:.1f} mm.</p>",
                       f'<video controls muted src="{key}.mp4"></video><img src="{key}_error.png"></section>']
    chunks.append('<p>Real data: RGBench, Hu et al., AAAI 2026; CC BY 4.0. Frozen simulation-trained four-domain 40K UNet.</p>')
    (out/'report.html').write_text('\n'.join(chunks))
    write_json(out/'summary.json', {'cases': cases, 'aggregation': 'Per forecast mean over available depth timestamps; four anchors from one capture are not independent captures'})
    write_json(out/'status.json', {'state': 'finished', 'completed': sum(x['status']=='ok' for x in cases),
                                   'failed': sum(x['status']!='ok' for x in cases), 'planned': len(specs)*3})


if __name__ == '__main__':
    main()
