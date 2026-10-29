"""GNS planner for the rl_push open/closed-loop comparison: the same receding-horizon best-of-K loop as
gns-pytorch/run_rlpush_benchmark_gns.py (32 candidates per internal replan, internal commit 5, context noise 6.7e-4,
gripper projected onto its rigid + prismatic manifold at internal replan boundaries), run from a given 6-frame context.
Run in gns_torch:  python gns_plan_rlp.py --inp in.npz --out out.npz [--checkpoint ...]
in.npz: hist [B, 6, N, 3] context frames (oldest first; the last one is the current frame), body [B, N] (0 object, 1 hand,
        2 left, 3 right finger), target [B, 3], horizon (int, frames left), seed (int)
out.npz: traj [B, H, N, 3] committed future frames (projected), plan_seconds [B], pred_mm [B]"""
import argparse, json, os, sys, time
import numpy as np, torch
GNS = "/data/user_data/mbronars/packages/gns-pytorch"; sys.path.insert(0, GNS)
ap = argparse.ArgumentParser(); ap.add_argument("--inp", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--checkpoint", default="/data/group_data/katefgroup/runs_gns_mbronars/gns_robotics3_direct_v1/model-200000.pt")
ap.add_argument("--data_dir", default="/data/group_data/katefgroup/datasets/gns_robotics3_lazy")
ap.add_argument("--replan_every", type=int, default=5); ap.add_argument("--cands_per_replan", type=int, default=32); ap.add_argument("--noise_std", type=float, default=6.7e-4)
A = ap.parse_args()
from gripper_projection import load_canonical                                # noqa: E402
from run_pick_benchmark_gns import CONTEXT, BODY_TO_TYPE, project_points      # noqa: E402
from gns import train as gns_train                                            # noqa: E402
meta = json.load(open(f"{A.data_dir}/metadata.json")); dev = torch.device("cuda")
sim = gns_train._get_simulator(meta, 6.7e-4, 6.7e-4, dev); sim.load(A.checkpoint); sim.to(dev).eval(); canon = load_canonical()
I = np.load(A.inp); B, _, N, _ = I["hist"].shape; H = int(I["horizon"]); seed0 = int(I["seed"]); assert I["hist"].shape[1] == CONTEXT
traj = np.zeros((B, H, N, 3), np.float32); secs = np.zeros(B); pred = np.zeros(B); rows = []
for b in range(B):
    body = I["body"][b].astype(np.int64); obj = body == 0; tgt = I["target"][b].astype(np.float64)
    ptt = torch.from_numpy(np.array([BODY_TO_TYPE[int(v)] for v in body], np.int64)).to(dev); npp = torch.tensor([N], dtype=torch.long, device=dev)

    def free_run(ctx, n):
        out = []
        with torch.no_grad():
            for _ in range(n):
                nx = sim.predict_positions(ctx, npp, ptt); out.append(nx.cpu().numpy()); ctx = torch.cat([ctx[:, 1:], nx.unsqueeze(1)], 1).contiguous()
        return np.stack(out)

    def make_ctx(frames, seed):
        c = torch.from_numpy(np.ascontiguousarray(np.transpose(frames, (1, 0, 2)))).float()
        if A.noise_std > 0: c = c + torch.randn(c.shape, generator=torch.Generator().manual_seed(seed)) * A.noise_std
        return c.to(dev).contiguous()

    t0 = time.time(); hist0 = I["hist"][b].astype(np.float32); hist = hist0; committed, k, reps = [], 0, []
    while len(committed) < H:
        rem = H - len(committed); best, best_cost = None, None
        for c in range(A.cands_per_replan):
            fut = free_run(make_ctx(hist, seed0 + 991 * k + c), rem); cost = float(np.linalg.norm(fut[-1][obj].mean(0) - tgt))
            if best_cost is None or cost < best_cost: best_cost, best = cost, fut
        take = np.stack([project_points(f, body, canon) for f in best[:min(A.replan_every, rem)]])
        committed.extend(list(take)); hist = np.concatenate([hist0, np.stack(committed)], 0)[-CONTEXT:].astype(np.float32)   # as the benchmark: last CONTEXT of (start context + committed)
        reps.append(dict(replan=k, frame=H - rem, best_mm=round(best_cost * 1e3, 1))); k += 1
    traj[b] = np.stack(committed); secs[b] = time.time() - t0; pred[b] = np.linalg.norm(traj[b, -1][obj].mean(0) - tgt) * 1e3
    rows.append(dict(pred_mm=round(float(pred[b]), 1), seconds=round(float(secs[b]), 1), replans=reps)); print(b, rows[-1]["pred_mm"], rows[-1]["seconds"], flush=True)
np.savez(A.out, traj=traj, plan_seconds=secs, pred_mm=pred)
json.dump(dict(checkpoint=A.checkpoint, horizon=H, seed=seed0, replan_every=A.replan_every, cands_per_replan=A.cands_per_replan, noise_std=A.noise_std, rows=rows),
          open(A.out[:-4] + ".json", "w"), indent=1)
print("wrote", A.out, flush=True)
