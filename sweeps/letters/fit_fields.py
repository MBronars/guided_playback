"""Fit the letter footprints once, exactly as the old alphabet evaluation did (pwm_efficient/sweep/run_alphabet_eval_worker.py):
rotation = default_rng(SeedSequence([MASTER_SEED, ord(L)])).uniform(-40, 40), density scale 0.75, coverage 0.75 x n_dough,
patch frac 0.2, fitted to the frame-0 blob of mould episode 0. Runs in the pctrl env (scipy).

  /data/user_data/mbronars/.miniconda3/envs/pctrl/bin/python sweeps/letters/fit_fields.py --letters I,C,L,R [--seed 42]
  [--anchor <rollout.npz with gt frame 0 and roles>]  -> sweeps/letters/fields/letter_<L>_seed<S>.npz"""
import argparse, os, sys
import numpy as np, torch
PWM_EFF = "/data/user_data/mbronars/packages/pwm_efficient"; sys.path.insert(0, PWM_EFF)
import letter_template as lt
HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(); ap.add_argument("--letters", default="I,C,L,R"); ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--anchor", default="/data/user_data/mbronars/packages/guided_playback/runs/mould/paper42_ep000/rollout.npz")
ap.add_argument("--bank", default=os.path.join(PWM_EFF, "stats", "letter_bank_highres.npz")); ap.add_argument("--density_scale", type=float, default=0.75)
ap.add_argument("--coverage_frac", type=float, default=0.75); ap.add_argument("--patch_frac", type=float, default=0.2); ap.add_argument("--out", default=os.path.join(HERE, "fields"))
A = ap.parse_args(); os.makedirs(A.out, exist_ok=True)
z = np.load(A.anchor); anchor = z["gt"][0, 0].astype(np.float32); roles = z["roles"][0]
dough = roles == 2; n = int(dough.sum())
anchor_xy = torch.from_numpy(anchor[None, :, :2]); block_mask = torch.from_numpy(dough[None]); anchor_z = torch.tensor([float(anchor[dough, 2].mean())])
images, letters = lt.load_letter_bank(A.bank)
for L in A.letters.split(","):
    rot = float(np.random.default_rng(np.random.SeedSequence([A.seed] + [ord(c) for c in L])).uniform(-40.0, 40.0))
    variants = lt.pre_rotate_variants(lt.get_letter_variants(images, letters, L), rot)
    params = lt.FitParams(rot_range_deg=0.0, n_rots=1, coverage_points=max(1, int(round(A.coverage_frac * n))), patch_frac=A.patch_frac, letter_density_scale=A.density_scale)
    f = lt.fit_letter_field(anchor_xy, block_mask, anchor_z, variants, params); fit = f.info[0] or {}
    out = os.path.join(A.out, f"letter_{L}_seed{A.seed}.npz")
    np.savez(out, letter=L, coverage=f.coverage[0].numpy(), contour=f.contour[0].numpy(), extent=f.extent[0].numpy(), plane_z=float(f.plane_z[0]),
             anchor_particles=anchor, chosen_rot_deg=rot, **{f"fit_{k}": v for k, v in fit.items() if isinstance(v, (int, float, str)) or v is None})
    print(f"{L}: rot {rot:+.2f} deg, iou {fit.get('iou'):.3f}, scale {fit.get('scale'):.4f} m, {f.coverage.shape[1]} coverage points -> {out}", flush=True)
