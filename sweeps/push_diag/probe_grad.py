"""E1 gradient probe for block-pushing guidance (docs/PUSH_GUIDANCE_DIAG.md).

For each held-out scenario (hard seed 0) and each of B members (same scenario, member seeds from one generator):
  x = the sampler's draws (P0 initial pose noise, E churn bank); J(x) = final-frame cube-centroid distance to target (m).
  g = dJ/dx. Subsets of x that receive the step:  P0 | Ehi (bank rows at noise level >= 0.8) | Eall (every used bank row) | all.
  Step = the optimizer's own unit-gradient step: per (member, row) the direction is normalized to norm sqrt(D), times alpha,
  then shell re-projection (exactly gp/guide.py unitgrad + shell). Random control: a Gaussian direction, same normalization.
  For alpha in ALPHAS record J after the gradient step and after the random step; also J re-evaluated at x twice (noise floor).
Writes one JSON line per (scenario, subset, alpha) and per-scenario info to --out.
"""
import argparse, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, "/data/user_data/mbronars/packages/guided_playback")
from gp.adapters import get_adapter
from gp.adapters import clean_core as lib
from gp import push_scenes
from gp.objectives import Objective, centroid

ap = argparse.ArgumentParser()
ap.add_argument("--scenarios", default="0-19"); ap.add_argument("--B", type=int, default=4); ap.add_argument("--precision", default="bf16", choices=("bf16", "fp32"))
ap.add_argument("--alphas", default="0.003,0.01,0.03,0.1,0.3"); ap.add_argument("--steps", type=int, default=6); ap.add_argument("--transitions", type=int, default=60)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", required=True); ap.add_argument("--compile", action="store_true")
ap.add_argument("--ckpt", default="/data/user_data/mbronars/packages/pwm_final_clean/runs/push0418_rigid_x0_rope_regpass_mix_cont200k_stream/checkpoints/final.pt")
A = ap.parse_args()
a, b = (A.scenarios.split("-") + [None])[:2]; SC = list(range(int(a), int(b) + 1)) if b else [int(v) for v in A.scenarios.split(",")]
ALPHAS = [float(v) for v in A.alphas.split(",")]
dev = torch.device("cuda"); amp = torch.bfloat16 if A.precision == "bf16" else None
ad = get_adapter("clean_push").load(A.ckpt, dev, compile=A.compile)
done = set()
if os.path.exists(A.out):
    for l in open(A.out):
        r = json.loads(l); done.add(r["scenario"]) if r.get("kind") == "scenario_done" else None
fo = open(A.out, "a")
wr = lambda r: (fo.write(json.dumps(r) + "\n"), fo.flush())

for s in SC:
    if s in done: continue
    t0 = time.time()
    batch = ad.episodes([0] * A.B, A.transitions, domain="pushing")
    scen = push_scenes.load(f"/data/user_data/mbronars/packages/PointWorld/scenarios/hard/seed_0/ep_{s:03d}.npz")
    target = push_scenes.apply(batch, ad, [scen] * A.B)
    mask = ad.object_mask(batch); obj = Objective(target, mask, batch.get_valid_mask())
    sc = lib.Staircase(ad.model, ad.diffusion, ad.cfg, batch, A.steps, horizon=A.transitions, seed=A.seed, amp_dtype=amp); sc.reseed = True
    X0, P0, E = sc.draws(torch.Generator(device=dev).manual_seed(A.seed))
    used = sc.opt_mask(0.0).expand_as(E).float(); hi = sc.opt_mask(0.8).expand_as(E).float()
    shellP = P0.flatten(2).norm(dim=-1)[..., None, None]; shellE = E.flatten(3).norm(dim=-1)[..., None, None]
    DP, DE = P0.shape[-1] * P0.shape[-2], E.shape[-1] * E.shape[-2]

    def J(P, Ee, grad=False):
        with torch.set_grad_enabled(grad):
            pos = sc.rollout(X0, P, Ee, "churn"); d, _ = obj(pos)
        return d, pos

    P = P0.clone().requires_grad_(True); Ee = E.clone().requires_grad_(True)
    d0, pos0 = J(P, Ee, True)
    gP, gE = torch.autograd.grad(d0.sum(), [P, Ee])
    d0 = d0.detach()
    moved = (centroid(pos0[:, -1].detach(), mask) - centroid(pos0[:, 0].detach(), mask))[:, :2].norm(dim=-1)
    rep = [J(P0, E)[0] for _ in range(2)]
    unit = lambda g, D, dims: g * (D ** 0.5) / g.flatten(dims).norm(dim=-1).clamp_min(1e-12)[(...,) + (None,) * 2]
    gnorm = dict(P0=gP.flatten(1).norm(dim=-1).tolist(), Eall=(gE * used).permute(2, 0, 1, 3, 4).flatten(1).norm(dim=-1).tolist(),
                 Ehi=(gE * hi).permute(2, 0, 1, 3, 4).flatten(1).norm(dim=-1).tolist())
    info = dict(kind="scenario", scenario=s, precision=A.precision, J0_mm=(d0 * 1e3).tolist(), repeat_mm=[(r * 1e3).tolist() for r in rep],
                cube_moved_mm=(moved * 1e3).tolist(), grad_norm=gnorm, free_rows=dict(Ehi=int(sc.opt_mask(0.8)[..., 0, 0].sum()), Eall=int(sc.opt_mask(0.0)[..., 0, 0].sum()), P0=int(P0.shape[1])))
    wr(info)
    g = torch.Generator(device=dev).manual_seed(1000 + s)
    rP = torch.randn(P0.shape, device=dev, generator=g); rE = torch.randn(E.shape, device=dev, generator=g)
    dirs = dict(grad=(unit(gP, DP, 2), unit(gE, DE, 3)), rand=(unit(rP, DP, 2), unit(rE, DE, 3)))
    for subset, (mP, mE) in dict(P0=(1.0, 0.0 * used), Ehi=(0.0, hi), Eall=(0.0, used), all=(1.0, used)).items():
        for al in ALPHAS:
            row = dict(kind="step", scenario=s, precision=A.precision, subset=subset, alpha=al)
            for name, (dP, dE) in dirs.items():
                with torch.no_grad():
                    Pn = P0 - al * dP * mP; En = E - al * dE * mE
                    Pn = Pn * shellP / Pn.flatten(2).norm(dim=-1).clamp_min(1e-8)[..., None, None]
                    En = En * shellE / En.flatten(3).norm(dim=-1).clamp_min(1e-8)[..., None, None]
                    row[name + "_mm"] = (J(Pn, En)[0] * 1e3).tolist()
                    if name == "grad":                       # first-order forecast of the change for the un-projected step, mm
                        lin = -(al * ((gP * dP * mP).flatten(1).sum(1) + (gE * dE * mE).permute(2, 0, 1, 3, 4).flatten(1).sum(1)))
                        row["linear_mm"] = (lin * 1e3).tolist()
            wr(row)
    wr(dict(kind="scenario_done", scenario=s, seconds=round(time.time() - t0, 1)))
    print(f"scenario {s}: J0 {np.round(d0.cpu().numpy() * 1e3, 1)} moved {np.round(moved.cpu().numpy() * 1e3, 1)} ({time.time() - t0:.0f} s)", flush=True)
print("PROBE_DONE", flush=True)
