"""PointWorld pick planner for the open/closed-loop comparison: reports/pw_mpc/benchmark_mppi.plan() UNCHANGED (MPPI argmin,
internal commit 5, every internal plan rolled out to the task deadline and scored ONLY by the object-centroid distance to the
target at that last frame), started from a given state. Run in pointworld-env, cwd pwm_final_paper:
  python pw_plan_pick.py --inp in.npz --out out.npz --ckpt <frozen ckpt> [--population 88 --max_batch 88]
in.npz: assets [B] (str), points [B, N, 3] current cloud (actuator rows = Gripper points of the given state), R0 [B,3,3], t0 [B,3],
        sep0 [B] (finger-centroid separation), target [B, 3], horizon (frames left to frame 60), seed
out.npz: imagined [B, H+1, N, 3], hand_R [B, H+1, 3, 3], hand_t [B, H+1, 3], sep [B, H+1], plan_seconds [B] (+ out.json replan rows)"""
import argparse, copy, json, os, sys
import numpy as np
PWM = "/data/user_data/mbronars/reports/pw_mpc"; sys.path.insert(0, PWM)
ap = argparse.ArgumentParser(); ap.add_argument("--inp", required=True); ap.add_argument("--out", required=True); ap.add_argument("--ckpt", required=True)
ap.add_argument("--stats", default=None, help="norm stats folder (default: the benchmark's three_domain_h10)"); ap.add_argument("--population", type=int, default=88); ap.add_argument("--max_batch", type=int, default=88); A = ap.parse_args()
sys.argv = [sys.argv[0]]
import benchmark_mppi as M                                                  # noqa: E402
args = M.parse(); args.population, args.max_batch, args.ckpt = A.population, A.max_batch, A.ckpt
if A.stats: args.stats = A.stats
from eval_protocol_pointworld import bootstrap                              # noqa: E402
bootstrap()
import torch                                                                 # noqa: E402
from pw_batched import BatchedPointWorld                                     # noqa: E402
from gripper import Gripper                                                  # noqa: E402
Z = np.load(M.SCENES); I = np.load(A.inp); B = I["points"].shape[0]; args.horizon = int(I["horizon"]); args.seed = int(I["seed"]); H = args.horizon
model = BatchedPointWorld(A.ckpt, args.stats, device="cuda", domain="pickup_release", bf16=not args.fp32)
lim = dict(t=args.max_step_mm / 1000.0, r=np.radians(args.max_rot_deg), close=args.ds_close_mm / 1000.0, open=args.ds_open_mm / 1000.0)


class AtState:
    """the scene's Gripper, started at a given (R, t, separation); points() keeps the frame-0 separation reference."""
    def __init__(self, g, R, t, s):
        self.g = g; self.R0, self.t0, self.sep0 = R, t, float(s)
        self.act_rows, self.obj_rows = g.act_rows, g.obj_rows
    def integrate(self, U, state=None): return self.g.integrate(U, state)
    def points(self, R, t, s): return self.g.points(R, t, s)
    def track(self, U, state=None): return self.g.points(*self.g.integrate(U, state))


out = dict(imagined=np.zeros((B, H + 1) + I["points"].shape[1:]), hand_R=np.zeros((B, H + 1, 3, 3)), hand_t=np.zeros((B, H + 1, 3)), sep=np.zeros((B, H + 1)), plan_seconds=np.zeros(B))
rows = []
for b in range(B):
    a = str(I["assets"][b])
    g = Gripper.from_scene(Z[f"{a}__points_model"].astype(np.float64), Z[f"{a}__roles"], Z[f"{a}__particle_body"], Z[f"{a}__body_poses0_model"])
    gs = AtState(g, I["R0"][b].astype(np.float64), I["t0"][b].astype(np.float64), I["sep0"][b])
    im, hR, ht, sep, U, rep, secs = M.plan(model, gs, I["points"][b].astype(np.float64), I["target"][b].astype(np.float64), args, lim, lambda m: print(m, flush=True))
    out["imagined"][b], out["hand_R"][b], out["hand_t"][b], out["sep"][b], out["plan_seconds"][b] = im, hR, ht, sep, secs
    oc = im[:, g.obj_rows].mean(1)
    rows.append(dict(asset=a, pred_mm=round(float(np.linalg.norm(oc[-1] - I["target"][b]) * 1e3), 1), pred_lift_mm=round(float((oc[:, 2].max() - oc[0, 2]) * 1e3), 1),
                     seconds=round(secs, 1), replans=rep))
    print(f"{a}: planned {rows[-1]['pred_mm']} mm, lift {rows[-1]['pred_lift_mm']} mm, {secs:.1f} s", flush=True)
np.savez(A.out, **{k: np.asarray(v, np.float32) for k, v in out.items()})
json.dump(dict(checkpoint=A.ckpt, stats=args.stats, model_horizon=model.horizon, domain="pickup_release", horizon=H, seed=args.seed, population=args.population, commit=args.commit, scored="object-centroid distance to target at the last frame of every internal rollout (= task frame 60)", rows=rows),
          open(A.out[:-4] + ".json", "w"), indent=1)
