"""Show a recorded-motion control beside a failed generated plan in the same scene."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--root', required=True, type=Path)
    root = parser.parse_args().root.resolve(); out = root/'qualified_control_report_v1'; out.mkdir(exist_ok=True)
    scene = 'pick_reserved_00_asset_8270'
    item = next(s for s in json.loads((root/'scenes_v2.json').read_text())['scenes'] if s['name'] == scene)
    audit = next(r for r in json.loads((root/'region_replay_summary_v1.json').read_text())['results']
                 if r['scene'] == scene and r['variant'] == 'selected')
    assert audit['control_qualified'] and audit['timestep_stable']
    directory = Path(item['directory'])
    batch = torch.load(directory/'input.pt', weights_only=False, map_location='cpu')
    geometry = torch.load(directory/'collision_geometry.pt', weights_only=False, map_location='cpu')
    recorded_path = root/'persistent_replay_controls_v1'/scene/'dt0005/execution.npz'
    generated_path = root/'generated_replay_v1'/f'{audit["run"]}_seed8311_selected/dt0005/execution.npz'
    recorded, generated = np.load(recorded_path), np.load(generated_path)
    poses = [(z[key+'_R'], z[key+'_T']) for z in (recorded, generated) for key in ('desired', 'model')]
    for R, T in poses:
        np.testing.assert_allclose(R[0], poses[0][0][0], rtol=0, atol=1e-6)
        np.testing.assert_allclose(T[0], poses[0][1][0], rtol=0, atol=1e-6)
        assert len(T) == 37
    body = batch.particle_body[0].numpy()
    subset = np.concatenate([np.where(body == k)[0][::max(1, int((body == k).sum())//150)] for k in range(4)])
    colors = np.array(['#e69f00', '#777777', '#0072b2', '#009e73'])[body[subset]]
    tracks = []
    for R, T in poses:
        points = np.empty((len(T), len(body), 3))
        for k in range(4):
            template = batch.body_templates[0, k, batch.body_template_mask[0, k]].numpy()
            points[:, body == k] = np.einsum('tij,nj->tni', R[:, k], template)+T[:, k, None]
        tracks.append(points[:, subset])
    cloud = np.concatenate(tracks).reshape(-1, 3)
    center = (cloud.min(0)+cloud.max(0))/2; span = max(np.ptp(cloud, axis=0).max()*1.12, .25)
    labels = ['Recorded dataset motion', 'Physics follows the recorded gripper motion',
              'Noise-optimized model prediction', 'Physics follows the generated gripper motion']
    fig = plt.figure(figsize=(11, 10)); artists, texts = [], []
    for i, (track, label) in enumerate(zip(tracks, labels)):
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        artists.append(ax.scatter(*track[0].T, c=colors, s=8))
        ax.set(xlim=(center[0]-span/2, center[0]+span/2), ylim=(center[1]-span/2, center[1]+span/2),
               zlim=(center[2]-span/2, center[2]+span/2), title=label, xticks=[], yticks=[], zlabel='height (m)')
        ax.set_box_aspect((1, 1, 1)); ax.view_init(22, -65)
        xx, yy = np.meshgrid(center[0]+np.array([-.5, .5])*span, center[1]+np.array([-.5, .5])*span)
        ax.plot_surface(xx, yy, np.full_like(xx, geometry['metadata']['support_z']), color='#999999', alpha=.12)
        texts.append(ax.text2D(.08, -.03, '', transform=ax.transAxes, fontsize=10))
    title = fig.suptitle(''); fig.subplots_adjust(left=.01, right=.97, top=.85, bottom=.07, hspace=.2, wspace=.08)
    origin = poses[0][1][0, 0, 2]
    def update(frame):
        for artist, track, text, (_, T) in zip(artists, tracks, texts, poses):
            artist._offsets3d = tuple(track[frame].T)
            text.set_text(f'Object height change: {(T[frame, 0, 2]-origin)*1000:.1f} mm')
        title.set_text(f'Reserved pick scene8270 · {frame*.05:.2f} s\n'
                       'Top: recorded-motion control. Bottom: generated plan, seed8311, 24 noise updates.\n'
                       'Both right panels use the same floating-hand controller and 0.5 ms physics step.\n'
                       'Orange is the object; physics commands only the gripper.')
        return artists+texts+[title]
    FuncAnimation(fig, update, frames=37, interval=100).save(out/'recorded_vs_guided_8270.gif', writer=PillowWriter(fps=10))
    update(36); fig.savefig(out/'recorded_vs_guided_8270_final.png', dpi=140); plt.close(fig)
    qualification = next(s for s in json.loads((root/'mujoco_controls_matched_v2_summary.json').read_text())['results'] if s['scene'] == scene)
    payload = dict(scene=scene, run=audit['run'], seed=8311,
                   final_lift_mm={label: float((T[-1, 0, 2]-origin)*1000) for label, (_, T) in zip(labels, poses)},
                   generated_timestep_stable=audit['timestep_stable'], controller_qualified=audit['control_qualified'],
                   recorded_control_error_mm=qualification['results']['mesh_preload']['object_prediction_final_error_mm'],
                   generated_qualified_task_pass=audit['qualified_task_pass'],
                   sources={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                            [recorded_path, generated_path, directory/'input.pt', directory/'collision_geometry.pt']},
                   source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   note='The recorded reference is a controller diagnostic, not a generated success or the held-workspace task target. Future recorded motion is used only for this separate reference replay. Approximate actuation and independent-engine differences remain possible contributors to generated failures.')
    (out/'summary.json').write_text(json.dumps(payload, indent=2)+'\n')
    page = '''<!doctype html><meta charset="utf-8"><title>Recorded control and generated-plan execution</title>
<style>body{font:17px system-ui;line-height:1.5;margin:30px;max-width:1200px}img{max-width:100%}</style>
<h1>The recorded motion replays; the generated lift does not</h1>
<p>Top left is a recorded dataset trajectory, and top right executes its gripper motion using the existing fixed preload controller. Bottom left is a24-update noise-optimized plan from the same observed scene; bottom right executes that plan's gripper motion. The object moves dynamically in both right panels.</p>
<img src="recorded_vs_guided_8270.gif">
<p>The recorded motion lifts136.75mm; its0.5ms replay lifts131.82mm. The generated model predicts211.96mm, while its replay leaves the object within0.002mm of its initial height. The generated replay is also stable at0.25ms, and the scene passes the recorded-controller qualification. This helps rule out a blanket inability of this replay setup to lift this object. It does not establish that the approximate motor-command conversion is equally faithful for every generated grasp.</p>
<p>The recorded trajectory is a separate reference, not a generated success and not the requested held-workspace target. The source recording never conditions the generated future. This is an independent floating-hand diagnostic, not robot or source-PhysX execution.</p>
<p><a href="summary.json">Measurements and source hashes</a> · <a href="../region_contact_timing_v1/index.html">Why the fingers approach different object trajectories</a> · <a href="../review_11am/index.html">Guidance review</a></p>'''
    (out/'index.html').write_text(page); print('Rendered recorded/generated comparison:', scene)


if __name__ == '__main__':
    main()
