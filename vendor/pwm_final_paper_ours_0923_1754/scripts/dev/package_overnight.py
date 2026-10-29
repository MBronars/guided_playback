"""Build the morning montage and status summary after bounded workers finish."""
import argparse
import json
import subprocess
import time
import zipfile
from pathlib import Path

import numpy as np

from scripts.dev.observation_probe_common import reported_checkpoint, write_json


def status(path):
    return json.loads(path.read_text()) if path.exists() else {}


def real_summary(root):
    rows=[]
    for phase in ('real_cloth','real_cloth_full','real_cloth_annotated','real_cloth_icp','real_cloth_short','real_cloth_short_smooth'):
        for path in sorted((root/phase).glob('*__seed*.json')):
            if path.name.endswith('_failure.json'):continue
            data=json.loads(path.read_text())
            if not data.get('metrics'):continue
            metrics=data['metrics'];inp=data['input']
            rows.append({'phase':phase,'capture':inp['capture'],'seed':data['seed'],
                         'horizon_frames':len(inp['model_times'])-4,'anchor_offset_s':inp['anchor_offset_seconds'],
                         'observed_to_prediction_mean_mm':float(np.mean([m['measured_surface_to_prediction_mean_mm'] for m in metrics])),
                         'observed_to_static_mean_mm':float(np.mean([m['measured_surface_to_static_anchor_mean_mm'] for m in metrics])),
                         'observed_to_prediction_final_mm':metrics[-1]['measured_surface_to_prediction_mean_mm'],
                         'receipt':str(path.relative_to(root)),
                         'video':str((path.parent/(path.stem+'.mp4')).relative_to(root))})
    selection=[x for x in rows if x['seed']==710001 and x['phase'] in ('real_cloth_full','real_cloth_icp')
               and (root/x['video']).exists() and (root/x['video']).stat().st_size>1000]
    write_json(root/'real_results.json',{'cases':rows,'montage_cases':selection,
                'selection':'Primary seed 710001 for all three captures in the early long-rollout and coherent-tracking annotated-phase cohorts; no best-of-seed selection',
                'scope':'Three captures of one green T-shirt, two grasp and one fold. Partial-surface distances, not ground-truth particle or full-shape errors.',
                'phase_comparison':'Early and annotated phases have different anchors/horizons; their mean errors are not matched protocol comparisons.',
                'tracking_comparison':'Annotated nearest-surface and ICP cohorts use the same anchors, horizons, observations and sampler seeds.'})
    return rows,selection


def montage(root, selection):
    if not selection:return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    assets=root/'montage_assets';assets.mkdir(exist_ok=True)
    parts=[]
    def card(name,title,subtitle,footer,seconds):
        image=assets/(name+'.png');video=assets/(name+'.mp4')
        fig=plt.figure(figsize=(14.86,6.06),facecolor='#172537')
        fig.text(.06,.69,title,color='white',fontsize=28,weight='bold')
        fig.text(.06,.45,subtitle,color='#d7e4ee',fontsize=18,linespacing=1.7)
        fig.text(.06,.12,footer,color='#8cabbf',fontsize=12)
        fig.savefig(image,dpi=100,facecolor=fig.get_facecolor());plt.close(fig)
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-loop','1','-i',str(image),'-t',str(seconds),
                        '-r','10','-vf','pad=ceil(iw/2)*2:ceil(ih/2)*2','-c:v','libx264','-threads','4','-pix_fmt','yuv420p',str(video)],check=True)
        parts.append(video.resolve())
    card('intro','Forecasting real cloth from measured depth',
         'Green: recorded cloth     Orange: model prediction\nA short observed history and the recorded robot motion are given.',
         'Frozen simulation-trained 40K checkpoint | Three captures | Includes failure cases | RGBench, Hu et al., AAAI 2026',3)
    names={'green_tshirt_grasp_2025-07-19-19-14-58':'Grasp A','green_tshirt_grasp_2025-07-19-19-14-03':'Grasp B',
           'green_tshirt_fold_2025-07-28-18-21-20':'Fold'}
    for i,case in enumerate(selection):
        label='Early observed state' if case['phase']=='real_cloth_full' else 'Annotated phase; coherent prefix tracking'
        card(f'case{i}',names.get(case['capture'],case['capture']),
             label+f"\n{case['horizon_frames']/30:.2f} seconds of prediction; primary sampler seed",
             'Measured surface | Predicted cloth and given grippers | Overlay. These are depth-point visualizations.',1)
        parts.append((root/case['video']).resolve())
    playlist=assets/'concat.txt'
    playlist.write_text('\n'.join("file '"+str(p).replace("'","'\\''")+"'" for p in parts)+'\n')
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','concat','-safe','0','-i',str(playlist),
                    '-r','10','-c:v','libx264','-threads','4','-preset','fast','-crf','21','-pix_fmt','yuv420p',str(root/'real_world_montage.mp4')],check=True)


def findings(root, states):
    core=states.get('robustness',{}); summary=status(root/'robustness/summary.json')
    lines=['# What the overnight experiments found','',
           'Frozen four-domain 40K UNet, EMA, C4/F16, checkpoint-owned normalization. '
           'This checkpoint was trained before the current cleanup and noisy-context fix. '
           'These are diagnostics, not results from a newly trained checkpoint or a final held-out benchmark.','',
           f"Core sweep: {core.get('completed',0)} successful / {core.get('planned','?')} planned; {core.get('failed',0)} execution failures.", '',
           '## Small history errors can substantially change predictions','',
           'Mean passive-particle error over the 16-frame window, in mm. Draws are averaged '
           'within clips, then clips receive equal weight. The full report includes confidence intervals.','',
           '| Domain | Clips | Clean input | 1 mm history jitter | 10 mm history jitter |',
           '| --- | ---: | ---: | ---: | ---: |']
    conditions=summary.get('conditions',[])
    for domain in ('mpm','pickup','pushing','cloth'):
        selected=[next((x for x in conditions if x['domain']==domain and x['family']==f and x['severity']==s),None)
                  for f,s in [('clean',0),('history_jitter',1),('history_jitter',10)]]
        nums=[f"{x['absolute_mm_mean_ci95'][0]:.2f}" if x else 'pending' for x in selected]
        lines.append(f"| {domain} | {selected[0]['clips'] if selected[0] else '?'} | "+' | '.join(nums)+' |')
    lines += ['', '## The temporal pattern of estimation error matters','',
              'This control holds the noisy endpoint exactly fixed. A constant offset throughout '
              'history preserves the true context displacements; independently changing errors corrupt them. '
              'The table is prediction response RMS in mm, averaged over two sampler seeds within '
              'each of four fixed clips per domain. It is a different estimator and subset from the table above.','',
              '| Domain | Changing frame errors | Constant offset |', '| --- | ---: | ---: |']
    motion=status(root/'motion_noise/summary.json')
    for domain in ('mpm','pickup','pushing','cloth'):
        clips=[x for x in motion.get('clips',[]) if x['domain']==domain]
        numbers=[]
        for kind in ('independent_frame_jitter','constant_offset'):
            per_clip=[np.mean([r['response_rms_mm'] for r in x['cases'] if r['kind']==kind]) for x in clips]
            numbers.append(f'{np.mean(per_clip):.2f}' if per_clip else 'pending')
        lines.append('| '+domain+' | '+' | '.join(numbers)+' |')
    lines += ['', 'This shows sensitivity to temporal error structure. It does not isolate RoPE, KNN '
              'or another neural mechanism. Constructing the constant-offset control uses clean '
              'reference motion, so it is not a deployable state estimator. Six-seed sampling variation '
              'is analyzed separately; it is not calibrated uncertainty about the physical state.','',
              '## The real-data pipeline works; large lifting transfer remains poor','',
              'Three public RGBench captures provide real colored depth clouds and recorded Piper '
              'actions. Raw RGB footage synchronized to these captures is not released. Only observed '
              'cloth history and known robot actions initialize prediction; future cloth clouds are used for scoring.','',
              'The longer predictions miss much of the lift. Low mean distances early in the recordings '
              'partly reflect nearly stationary cloth. The half-second forecasts restart from measured '
              'state and must not be presented as a continuous long rollout. Inspect `real_motion/lift_height.png` '
              'and the montage, which includes all three captures and their failures.','']
    paired=status(root/'real_motion/motion_smoothing_comparison.json').get('cases',[])
    if paired:
        change=np.mean([x['paired_change_mm'] for x in paired])
        lines.append(f'Local observed-motion smoothing changed mean short-forecast surface distance by {change:+.3f} mm '
                     f'across {len(paired)} paired cases from three recordings. That is not a meaningful improvement here.')
    lines += ['', 'Robot FK conventions were verified against released pose tracks. Centimeter-scale '
              'finger-to-visible-cloth gaps remain in some lifting frames; occluded contact surfaces '
              'and uncalibrated camera latency limit interpretation. Persistent particle tracking, '
              'hidden geometry and real material values remain estimated or assumed.','',
              '## What to do next','',
              'Rerun the frozen input and corruption manifests on the improved checkpoint. Keep the '
              'current examples and seeds fixed. The next targeted training comparison should consider '
              'physically consistent observation-error augmentation and temporal tracking quality; '
              'nothing here establishes a need to change RoPE or neighborhood selection. For real transfer, '
              'verify contact alignment and tracking on a small controlled capture before expanding the video set.','',
              'Dynamic occlusion, segmentation mistakes, lost identities, new hardware execution and '
              'closed-loop control were not tested. Small material-parameter effects on these short clips '
              'do not by themselves prove useful material conditioning or robustness.']
    (root/'FINDINGS.md').write_text('\n'.join(lines)+'\n')


def portable_bundle(root, lanes):
    """Keep viewable reports and receipts together without large tensors or vendors."""
    names=['index.html','README.md','FINDINGS.md','morning_status.json','real_results.json','real_world_montage.mp4']
    files=[root/name for name in names if (root/name).exists()]
    for lane in [*lanes,'real_motion']:
        files += [p for p in (root/lane).rglob('*') if p.is_file() and p.suffix in ('.html','.png','.mp4','.json','.csv')]
    target=root/'viewable_results.zip'; temporary=target.with_suffix('.zip.partial')
    with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as archive:
        for path in sorted(set(files)): archive.write(path,path.relative_to(root))
    temporary.replace(target)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--wait',action='store_true')
    ap.add_argument('--max-wait-hours',type=float,default=5);ap.add_argument('--real-only',action='store_true')
    a=ap.parse_args();root=Path(a.root);start=time.monotonic()
    lanes=['real_cloth','real_cloth_full','real_cloth_annotated','real_cloth_icp','real_cloth_short','real_cloth_short_smooth']
    if not a.real_only:lanes+=['robustness','stability','motion_noise']
    while a.wait:
        states={name:status(root/name/'status.json') for name in lanes}
        finished=all(x.get('state') in ('finished','sampling_finished','failed','not_run') for x in states.values())
        # A fresh index alone can still contain partial plots. Wait for the report
        # whose actual case count matches the core sampler's final status.
        core=states.get('robustness',{}); report=status(root/'robustness/report_status.json')
        report_ready=(a.real_only or core.get('state') in ('failed','not_run') or
                      (report.get('state')=='finished' and
                       report.get('case_count')==core.get('completed',0)+core.get('failed',0)))
        if finished and report_ready:break
        if time.monotonic()-start>a.max_wait_hours*3600:break
        write_json(root/'packaging_status.json',{'state':'waiting','lanes':{n:v.get('state','not_started') for n,v in states.items()}})
        time.sleep(30)
    checkpoint=reported_checkpoint(root)        # the labels and text below describe only this checkpoint
    rows,selected=real_summary(root);montage(root,selected)
    subprocess.run(['/data/user_data/robertre/conda_envs/pwm_blackwell/bin/python',
                    'scripts/dev/real_cloth_motion_report.py', '--root', str(root)], check=True)
    states={name:status(root/name/'status.json') for name in lanes}
    write_json(root/'morning_status.json',{'lanes':states,'real_cases_with_metrics':len(rows),
                                         'generated_unix':time.time(),'checkpoint':checkpoint,
                                         'notes':['Historical checkpoint; diagnostic validation, not a final held-out paper benchmark',
                                                  'Real-state tracking and visibility limitations are retained in per-case receipts',
                                                  'No retraining or hardware execution was performed']})
    lines=['# Overnight experiment results','',
           'Open `index.html` for the plots, videos, and per-case receipts.',
           'The checkpoint is the completed four-domain 40K UNet; these are diagnostics, not a final paper benchmark.','',
           f"Real-cloth results: {len(rows)} case records across three recordings and the saved protocol variants.",
           f"Robustness: {states.get('robustness',{}).get('completed','pending')} successful cases / {states.get('robustness',{}).get('planned','pending')} planned.",
           f"Sampling-variation follow-up: {states.get('stability',{}).get('state','pending')}.",'',
           f"Temporal-error control: {states.get('motion_noise',{}).get('state','pending')}.",'',
           'Real observations are colored depth clouds. Surface tracking is approximate; future robot actions are given.',
           'Early and later-phase rollouts have different horizons. Compare trackers only within the annotated-phase protocol.',
           'The short-forecast cohort restarts from measured history at four fixed times per recording; it is not a continuous rollout.',
           'Every failed case and the historical checkpoint limitations remain recorded.','',
           'Important files: `real_world_montage.mp4`, `real_results.json`, `robustness/summary.json`,',
           '`robustness/results.csv`, `stability/summary.json`, `motion_noise/summary.json`, and `morning_status.json`.']
    lines += ['', 'Read `FINDINGS.md` first. To view on another computer, copy `viewable_results.zip`,',
              'extract the entire archive, and open its `index.html`. It includes videos, plots and receipts;',
              'raw model inputs, prediction tensors, dependencies and checkpoint weights stay in the original workspace.']
    (root/'README.md').write_text('\n'.join(lines)+'\n')
    findings(root,states)
    subprocess.run(['/data/user_data/robertre/conda_envs/pwm_blackwell/bin/python','scripts/dev/overnight_report.py',
                    '--root',str(root),'--index-only'],check=True)
    portable_bundle(root,lanes)
    write_json(root/'packaging_status.json',{'state':'finished','time_unix':time.time(),'real_cases':len(rows)})
    print('PACKAGED',root,flush=True)


if __name__=='__main__':main()
