"""Audit and visualize saved unconditional picking experiments (no model calls).

Example: python scripts/dev/analyze_pick_guidance.py --root validation/guidance_pick_2026-09-14
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import pickle
import tarfile
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.ticker import MaxNLocator
import numpy as np
import torch
from scipy.spatial import cKDTree

from utils.reproducibility import file_sha256


ARMS = dict(sample='Ordinary sample', best8_goal='Best of 8: goal',
            best8_interaction='Best of 8: interaction', noise_goal='Noise opt: goal only',
            noise_interaction='Noise opt: interaction', noise_actuated='Noise opt: actuator only')
GRIPPER = Path('/data/user_data/robertre/RLSkills_IsaacLab/assets/franka_gripper/robots/franka_gripper.urdf')


def joint_reference(root, batch):
    """Recover absolute joint openings; the model represents relative slides only."""
    episode = json.loads((root/'source_episode.json').read_text())
    member = episode['key'] + '.joint_pos_obs.pyd'
    with tarfile.open(episode['shard']) as archive:
        q = np.asarray(pickle.load(archive.extractfile(member)))[episode['elem'], :, -2:]
    limits = []
    tree = ET.parse(GRIPPER)
    for name in ('panda_finger_joint1', 'panda_finger_joint2'):
        limit = tree.find(f"joint[@name='{name}']/limit")
        limits.append([float(limit.get(k)) for k in ('lower', 'upper', 'velocity')])
    axis = batch.joint_axis[0, [2, 3]].numpy()
    pose = batch.body_poses[0].numpy()
    anchor = relative_fingers(pose[..., :3, :3], pose[..., :3, 3])[batch.context_len-1]
    recovered = q[batch.context_len-1] + ((relative_fingers(pose[..., :3, :3], pose[..., :3, 3]) - anchor)*axis).sum(-1)
    error = float(np.max(np.abs(q-recovered)))
    if error > 1e-5:
        raise ValueError(f'Joint opening recovery disagrees with recording: {error} m')
    return dict(q=q, axis=axis, anchor=anchor, limits=np.array(limits),
                recovery_max_error_m=error, urdf_sha256=file_sha256(GRIPPER))


def relative_fingers(R, t):
    return np.einsum('tji,tfj->tfi', R[:, 1], t[:, [2, 3]]-t[:, 1, None])


def joint_metrics(R, t, reference, C):
    q = reference['q'][C-1] + ((relative_fingers(R, t)-reference['anchor'])*reference['axis']).sum(-1)
    limits = reference['limits']
    violation = np.maximum(limits[:, 0]-q, q-limits[:, 1]).clip(0)
    # Pickup's adapter uses 0.05 seconds per observed frame.
    speed = np.abs(np.diff(q[C-1:], axis=0)) / .05
    return dict(finger_limit_violation_max_mm=float(violation[C:].max()*1000),
                finger_limit_over_0p1mm_frames=int((violation[C:].max(-1)>.0001).sum()),
                finger_speed_max_m_s=float(speed.max()),
                finger_speed_over_urdf_frames=int((speed>limits[:, 2]+1e-5).any(-1).sum()),
                finger_asymmetry_max_mm=float(np.abs(q[C:, 0]-q[C:, 1]).max()*1000),
                finger_opening_min_mm=float(q[C:].min()*1000),
                finger_opening_max_mm=float(q[C:].max()*1000))


def distant_gripper_rise(positions, body, t, C):
    """Stricter separation check: every sampled gripper point is far from the object.

    Rising in free flight after an impulse is possible; this remains a diagnostic,
    not a force-balance test. It does not equate missing one finger with no contact.
    """
    gaps = np.array([cKDTree(frame[body==0]).query(frame[body>0])[0].min()
                     for frame in positions[C-1:]])
    dz = np.diff(t[C-1:, 0, 2])
    flagged = (dz>.0005) & (gaps[:-1]>.005) & (gaps[1:]>.005)
    return dict(rise_with_all_gripper_points_over_5mm_mm=float(dz[flagged].sum()*1000),
                rise_with_all_gripper_points_over_5mm_transitions=int(flagged.sum()),
                final_nearest_gripper_gap_mm=float(gaps[-1]*1000))


def render(root, seed, datasets, records, protocol, out, panel_arms=None):
    base = datasets['sample']
    C, target = int(base['context_len']), base['target'][0]
    body = base['particle_body'][0]
    panels = [('Recording (reference)', base['gt'][0], records['recording'])]
    panel_arms = panel_arms or ('sample', 'best8_interaction', 'noise_goal', 'noise_interaction', 'noise_actuated')
    labels = dict(ARMS, noise_strong='Stronger interaction: 12 updates')
    for arm in panel_arms:
        panels.append((labels[arm], datasets[arm]['positions'][0], records[arm]['selected']))
    allp = np.concatenate([p[1] for p in panels], axis=0)
    lo, hi = allp.min((0, 1)), allp.max((0, 1))
    lo, hi = np.minimum(lo, target), np.maximum(hi, target)
    extent = max(hi-lo)*1.10
    centre = (lo+hi)/2
    limits = np.stack([centre-extent/2, centre+extent/2], axis=-1)
    floor = protocol['geometry']['support_z']
    fig = plt.figure(figsize=(15, 10), facecolor='white')
    fig.subplots_adjust(left=.01, right=.99, bottom=.09, top=.86, wspace=.02, hspace=.20)
    heading = fig.text(.5, .965, '', ha='center', va='top', fontsize=17, weight='bold')
    fig.text(.5, .92, 'Blue: object   |   Red: hand + fingers   |   Green star: requested 10 cm lift\n'
             'Same camera and scale in every panel. Recording shows a different, higher lift.', ha='center', va='top', fontsize=10)
    fig.text(.5, .025, 'Gap = farther finger\'s nearest sampled object point (5 mm proximity threshold). '
             'Depth = sampled gripper points inside object collision components.\n'
             'These are geometric checks, not simulator execution. Grey plane is inferred from observed object support. '
             'Playback: 4× slower than the 20 Hz data.', ha='center', fontsize=9)
    artists = []
    for i, (title, positions, metric) in enumerate(panels):
        ax = fig.add_subplot(2, 3, i+1, projection='3d')
        ax.view_init(elev=22, azim=-60)
        ax.set_proj_type('ortho')
        ax.set_box_aspect((1, 1, 1))
        ax.set(xlim=limits[0], ylim=limits[1], zlim=limits[2])
        ax.set_xlabel('x (m)', fontsize=8, labelpad=-2)
        ax.set_ylabel('y (m)', fontsize=8, labelpad=-2)
        ax.set_zlabel('z (m)', fontsize=8, labelpad=-2)
        ax.tick_params(labelsize=7, pad=0)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_major_locator(MaxNLocator(4))
        ax.set_title(title, fontsize=11, pad=4)
        xx, yy = np.meshgrid(limits[0], limits[1])
        ax.plot_surface(xx, yy, np.full_like(xx, floor), color='#dce1e7', alpha=.20, shade=False)
        ax.scatter(*target, marker='*', c='#209b5c', s=105, depthshade=False)
        clouds = [ax.scatter(*positions[0, body==b].T, c='#217bb8' if b==0 else '#d95050',
                            s=3 if b==0 else 2, alpha=.85, depthshade=False) for b in range(4)]
        centerline, = ax.plot([], [], [], color='#217bb8', linewidth=1, alpha=.6)
        text = ax.text2D(.03, -.08, '', transform=ax.transAxes, fontsize=9,
                         bbox=dict(facecolor='white', edgecolor='none', alpha=.85))
        artists.append((clouds, centerline, text, positions, metric))

    def update(f):
        phase = 'OBSERVED' if f<C else 'GENERATED'
        heading.set_text(f'Unconditional picking guidance · seed {seed} · frame {f:02d}/39 · {phase}')
        for clouds, path, label, positions, metric in artists:
            for b, cloud in enumerate(clouds):
                cloud._offsets3d = positions[f, body==b].T
            centres = positions[:f+1, body==0].mean(1)
            path.set_data_3d(*centres.T)
            if f<C:
                label.set_text('Shared observed history')
            else:
                j = f-C+1
                gap = max(metric['traces']['gaps'][j])*1000
                depth = max(metric['traces']['inside_object'][j])*1000
                label.set_text(f'Finger gap: {gap:.1f} mm   |   Inside depth: {depth:.1f} mm\n'
                               f'Final target error: {metric["target_distance_mm"]:.1f} mm')
        return [heading]

    frames = list(range(C)) + [C-1]*3 + list(range(C, len(base['gt'][0]))) + [len(base['gt'][0])-1]*8
    animation = FuncAnimation(fig, update, frames=frames, interval=200, blit=False)
    animation.save(out/f'comparison_seed_{seed}.gif', writer=PillowWriter(fps=5), dpi=90)
    update(len(base['gt'][0])-1)
    fig.savefig(out/f'final_seed_{seed}.png', dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for arm in panel_arms:
        m = records[arm]['selected']['traces']
        frames = np.arange(C-1, C-1+len(m['gaps']))
        axes[0].plot(frames, np.max(m['gaps'], axis=-1)*1000, label=labels[arm])
        axes[1].plot(frames, np.max(m['inside_object'], axis=-1)*1000)
        axes[2].plot(frames, (np.array(m['bottom'])-floor)*1000)
    gt = records['recording']['traces']
    for ax, values in zip(axes, (np.max(gt['gaps'], axis=-1)*1000,
                               np.max(gt['inside_object'], axis=-1)*1000,
                               (np.array(gt['bottom'])-floor)*1000)):
        ax.plot(frames, values, c='black', linestyle='--', label='Recording')
        ax.axvline(C-.5, color='.7', linewidth=1)
        for boundary in (20, 36):
            ax.axvline(boundary-.5, color='.8', linestyle=':')
        ax.grid(alpha=.2)
    axes[0].axhline(5, color='.4', linestyle=':', label='5 mm proximity threshold')
    axes[1].axhline(1, color='.4', linestyle=':')
    axes[2].axhline(0, color='.4', linestyle=':')
    axes[0].set_ylabel('Farther-finger gap (mm)')
    axes[1].set_ylabel('Gripper inside object (mm)')
    axes[2].set_ylabel('Object mesh bottom above\ninferred support (mm)')
    axes[2].set_xlabel('Frame (20 Hz); dotted vertical lines mark new rollout windows')
    axes[0].legend(ncol=2, fontsize=8)
    fig.suptitle(f'Guidance diagnostic · one scene · seed {seed}\nPoint contact is a proximity proxy; zero detected depth does not prove collision clearance')
    fig.tight_layout()
    fig.savefig(out/f'traces_seed_{seed}.png', dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--skip-render', action='store_true')
    p.add_argument('--include-followup', action='store_true')
    args = p.parse_args()
    root = args.root.resolve()
    protocol = json.loads((root/'sample/protocol.json').read_text())
    batch = torch.load(protocol['arguments']['input'], map_location='cpu', weights_only=False)
    reference = joint_reference(root, batch)
    out = root/('analysis_followup' if args.include_followup else 'analysis')
    out.mkdir(exist_ok=True)
    seeds = [row['seed'] for row in json.loads((root/'sample/results.json').read_text())]
    rows, audits = [], []
    arms = list(ARMS) + (['noise_strong'] if args.include_followup else [])
    for seed in seeds:
        datasets = {arm: dict(np.load(root/arm/f'seed_{seed}.npz')) for arm in arms}
        records = {arm: json.loads((root/arm/f'seed_{seed}.json').read_text()) for arm in arms}
        records['recording'] = json.loads((root/'sample/recording.json').read_text())
        baseline = datasets['sample']
        C = int(baseline['context_len'])
        for arm, d in datasets.items():
            for key in ('checkpoint_sha256', 'input_sha256', 'geometry_sha256', 'sampling', 'normalization'):
                assert json.loads((root/arm/'protocol.json').read_text())[key] == protocol[key], (arm, key)
            assert np.array_equal(d['initial'], baseline['positions']), (arm, seed, 'unpaired initial')
            assert np.array_equal(d['positions'][:, :C], d['gt'][:, :C]), (arm, seed, 'altered context')
            assert records[arm]['windows'] == [[4, 16, 0], [20, 16, 0], [36, 4, 0]]
            r = records[arm]
            metric = {k:v for k,v in r['selected'].items() if k!='traces'}
            metric.update(joint_metrics(d['R'][0], d['T'][0], reference, C))
            metric.update(distant_gripper_rise(d['positions'][0], d['particle_body'][0], d['T'][0], C))
            rows.append(dict(arm=arm, seed=seed, **metric,
                             selected_iteration=r['selected_iteration'], selected_draw=r['selected_draw'],
                             seconds=r['seconds'], forward_nfe=r['forward_nfe'],
                             actual_model_calls=r['actual_model_calls_including_recomputation'],
                             max_allocated_GB=r['max_allocated_bytes']/1e9))
            audits.append(dict(arm=arm, seed=seed, paired_initial=True, context_unchanged=True,
                               protocol_match=True, windows_match=True))
        if not args.skip_render:
            panels = ('sample', 'noise_goal', 'noise_interaction', 'noise_actuated', 'noise_strong') if args.include_followup else None
            render(root, seed, datasets, records, protocol, out, panel_arms=panels)
    reference_metrics = joint_metrics(batch.body_poses[0, :, :, :3, :3].numpy(),
                                     batch.body_poses[0, :, :, :3, 3].numpy(), reference, batch.context_len)
    reference_separation = distant_gripper_rise(batch.positions[0].numpy(), batch.particle_body[0].numpy(),
                                               batch.body_poses[0, :, :, :3, 3].numpy(), batch.context_len)
    audit = dict(comparisons=audits, joint_reference_recovery_max_error_m=reference['recovery_max_error_m'],
                 joint_urdf=str(GRIPPER), joint_urdf_sha256=reference['urdf_sha256'],
                 finger_limits=reference['limits'].tolist(), dt_seconds=.05,
                 recording_joint_metrics=reference_metrics, recording_separation_metrics=reference_separation)
    (out/'audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    (out/'metrics.json').write_text(json.dumps(rows, indent=2, allow_nan=False)+'\n')
    with (out/'metrics.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(dict(rows=len(rows), seeds=seeds, output=str(out), audit='passed')))


if __name__ == '__main__':
    main()
