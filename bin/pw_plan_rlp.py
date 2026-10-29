"""PointWorld MPPI-argmin planner for the rl_push open/closed-loop comparison: reports/pw_mpc/rlpush_mppi.plan() unchanged
(the 50-push benchmark's planner: K=88, internal commit 5, horizon shrinking to the deadline, sampling widths from the recorded
rl_push steps, fingers held at the start opening), called for several problems in one process.
Run in pointworld-env, cwd /data/user_data/mbronars/packages/pwm_final_paper:
  python pw_plan_rlp.py --inp in.npz --out out.npz [--ckpt ...]
in.npz: obj [B, 512, 3] current object points (model frame), R0 [B, 3, 3], t0 [B, 3] current hand pose, fq0 [B, 2],
        target [B, 3], horizon (int, frames left), seed (int)
out.npz: hand_pos [B, H+1, 3], hand_quat [B, H+1, 4] wxyz, finger_q [B, H+1, 2], object_pred [B, H+1, 512, 3], plan_seconds [B]
         + out.json (per-problem replan rows)"""
import argparse, json, os, sys, time
import numpy as np
sys.argv_saved = list(sys.argv); PWM = "/data/user_data/mbronars/reports/pw_mpc"; sys.path.insert(0, PWM)
ap = argparse.ArgumentParser(); ap.add_argument("--inp", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--ckpt", default=os.path.join(PWM, "ckpt_frozen.pt")); A = ap.parse_args()
sys.argv = [sys.argv[0]]
import rlpush_mppi as M                                                    # noqa: E402
args = M.parse()                                                            # the benchmark's own defaults (K 88, commit 5, sigmas, clamps)
from eval_protocol_pointworld import bootstrap                             # noqa: E402
bootstrap()
import torch                                                                # noqa: E402
from pw_batched import BatchedPointWorld                                    # noqa: E402
from make_rollout import gripper_points                                     # noqa: E402
I = np.load(A.inp); B = I["obj"].shape[0]; args.horizon = int(I["horizon"]); args.seed = int(I["seed"])
meta = torch.load(A.ckpt, map_location="cpu", weights_only=False); step = int(meta.get("sample_count", 0)) // 32; del meta
model = BatchedPointWorld(A.ckpt, args.stats, device="cuda", domain="pickup_rl_push", bf16=not args.fp32)
lim = dict(t=args.max_step_mm / 1000.0, r=np.radians(args.max_rot_deg))
H = args.horizon; out = dict(hand_pos=np.zeros((B, H + 1, 3)), hand_quat=np.zeros((B, H + 1, 4)), finger_q=np.zeros((B, H + 1, 2)),
                             object_pred=np.zeros((B, H + 1, I["obj"].shape[1], 3)), plan_seconds=np.zeros(B)); rows = []
for b in range(B):
    hR, ht, op, U, replans, secs = M.plan(model, gripper_points, I["obj"][b].astype(np.float64), I["R0"][b].astype(np.float64), I["t0"][b].astype(np.float64),
                                          I["fq0"][b].astype(np.float64), I["target"][b].astype(np.float64), args, lim, lambda m: print(m, flush=True))
    out["hand_pos"][b] = ht; out["hand_quat"][b] = M.mat_to_quat(hR); out["finger_q"][b] = np.repeat(I["fq0"][b][None], H + 1, 0)
    out["object_pred"][b] = op; out["plan_seconds"][b] = secs
    rows.append(dict(pred_mm=round(float(np.linalg.norm(op[-1].mean(0) - I["target"][b]) * 1e3), 1), seconds=round(secs, 1), replans=replans))
np.savez(A.out, **{k: np.asarray(v, np.float32) for k, v in out.items()})
json.dump(dict(checkpoint=A.ckpt, matched_step=step, horizon=H, seed=args.seed, population=args.population, commit=args.commit, sigma_t_m=args.sigma_t,
               sigma_r_rad=args.sigma_r, clamps=[args.max_step_mm, args.max_rot_deg], rows=rows), open(A.out[:-4] + ".json", "w"), indent=1)
print("wrote", A.out, [r["pred_mm"] for r in rows], flush=True)
