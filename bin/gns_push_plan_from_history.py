"""GNS best-of-N block-pushing plan from an EXECUTED history (closed-loop baseline). Same model, templates, candidate
count, scoring and action extraction as gns-pytorch/pushing_eval_gns.py (the paper's GNS pushing baseline); the only
change is the context: open loop gives GNS 6 copies of the scenario start (zero velocity); here the 6 context frames are the
most recent executed frames (SAPIEN block pose + rod xy), front-padded with executed frame 0 when fewer than 6 exist -
with no executed frames this is exactly the open-loop context. Each candidate's own template is posed at the executed
poses. Rolls out the REMAINING frames (total - c), scores the final block centroid vs the target, writes the winner's
rod actions (xy deltas, from executed frame c on) and imagined tracks.
  python bin/gns_push_plan_from_history.py --state_npz <all.npz> --scenario_idx 11 --history <npz or none> --total_frames 60 --out <dir>"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch
G = Path("/data/user_data/mbronars/packages/gns-pytorch"); sys.path.insert(0, str(G))
import pushing_eval_gns as E                                                   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", default="/data/user_data/mbronars/runs/gns_pushing_direct_v1/model-7185000.pt")
ap.add_argument("--metadata_json", default="/data/group_data/katefgroup/datasets/gns_pushing_direct_npz/metadata.json")
ap.add_argument("--state_npz", required=True); ap.add_argument("--scenario_idx", type=int, required=True)
ap.add_argument("--history", default="none", help="npz with block_pos [c+1,3], block_quat [c+1,4] wxyz, gripper_xy [c+1,2] (executed, world)")
ap.add_argument("--total_frames", type=int, default=60); ap.add_argument("--population_size", type=int, default=256); ap.add_argument("--chunk_size", type=int, default=32)
ap.add_argument("--template_seed", type=int, default=42); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--n_block", type=int, default=128); ap.add_argument("--n_hand", type=int, default=128)
ap.add_argument("--out", required=True)
A = ap.parse_args()
dev = torch.device("cuda"); np.random.seed(A.seed); torch.manual_seed(A.seed)
meta = json.load(open(A.metadata_json)); sim = E.make_simulator(meta, dev); sim.load(A.checkpoint); sim.to(dev).eval()
sc = E.load_scenario(Path(A.state_npz), A.scenario_idx)
if A.history == "none":
    H = dict(block_pos=sc["block_pos"][None], block_quat=sc["block_quat"][None], gripper_xy=sc["gripper_xy"][None])
else:
    z = np.load(A.history); H = {k: z[k].astype(np.float32) for k in ("block_pos", "block_quat", "gripper_xy")}
c = H["block_pos"].shape[0] - 1; C = E.INPUT_SEQUENCE_LENGTH
idx = [max(0, j) for j in range(c - C + 1, c + 1)]                             # most recent C executed frames, front-padded with frame 0
F = A.total_frames - c; K, N = A.population_size, A.n_hand + A.n_block
ctx = np.zeros((K, C, N, 3), np.float32); tpl_hand = np.zeros((K, A.n_hand, 3), np.float32)


def qmat(q):
    w, x, y, z = [float(v) for v in q]
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]], np.float64)


# sample_templates also draws from numpy's GLOBAL rng, so candidate k's points depend on the call sequence: replicate the
# open-loop sequence exactly (np.random.seed(seed) above, candidates 0..K-1 built once, in order, at the scenario start),
# recover each candidate's local cube / rod templates, then pose those FIXED templates at every executed frame.
Rs0 = qmat(sc["block_quat"]); pts0_all = []
for k in range(K):
    rng = np.random.default_rng((A.template_seed, A.scenario_idx, k))
    pts, th, gp0 = E.build_initial_particles(sc, A.n_block, A.n_hand, rng); pts0_all.append(pts)
    cube_local = ((pts[A.n_hand:].astype(np.float64) - sc["block_pos"].astype(np.float64)) @ Rs0).astype(np.float32)
    tpl_hand[k] = th
    for i, j in enumerate(idx):
        gp = np.array([H["gripper_xy"][j][0], H["gripper_xy"][j][1], E.GRIPPER_Z_FIXED], np.float32)
        blk = (cube_local.astype(np.float64) @ qmat(H["block_quat"][j]).T + H["block_pos"][j].astype(np.float64)).astype(np.float32)
        ctx[k, i] = np.concatenate([E.apply_pose(th, gp, E._IDENTITY_QUAT_WXYZ), blk], 0)
recon = max(float(np.abs(ctx[k, -1] - pts0_all[k]).max()) for k in range(K)) if c == 0 else None
# offset exactly as open loop: candidate 0's block points at the SCENARIO start (fixed across replans)
off = E.compute_traj_offset_from_initial(pts0_all[0][A.n_hand:])
ctx_c = torch.from_numpy(ctx - off[None, None, None]).to(dev).float().permute(0, 2, 1, 3).contiguous()     # (K, N, C, 3)
ptype = torch.from_numpy(np.concatenate([np.full(A.n_hand, 1), np.full(A.n_block, 0)]).astype(np.int64)).to(dev)
tgt = torch.from_numpy((sc["target_xy"] - off[:2]).astype(np.float32)).to(dev)
t0 = time.time(); fut = []
for cs in range(0, K, A.chunk_size):
    ce = min(cs + A.chunk_size, K); fut.append(E.free_rollout_batch(sim, ctx_c[cs:ce], ptype, N, K=ce - cs, F=F, device=dev))
fut = torch.cat(fut, 0); secs = time.time() - t0
costs = ((fut[:, -1, A.n_hand:, :2].mean(1) - tgt[None]) ** 2).sum(1).cpu().numpy(); kb = int(np.argmin(costs))
best = np.concatenate([ctx[kb] - off[None, None], fut[kb].cpu().numpy()], 0)                               # (C+F, N, 3) centred
rod_t, _ = E.fit_rigid_traj(E.apply_pose(tpl_hand[kb], np.zeros(3, np.float32), E._IDENTITY_QUAT_WXYZ), best[:, :A.n_hand])
gxy = rod_t[:, :2] + off[:2] - E.apply_pose(tpl_hand[kb], np.zeros(3, np.float32), E._IDENTITY_QUAT_WXYZ).mean(0)[:2] * 0   # rod origin track (world)
actions = np.diff(rod_t[C - 1:, :2], axis=0).astype(np.float32)                                            # from executed frame c on
bxy = best[:, A.n_hand:, :2].mean(1) + off[:2]
out = Path(A.out); out.mkdir(parents=True, exist_ok=True)
imag = float(np.linalg.norm(bxy[-1] - sc["target_xy"]))
np.savez(out / "plan.npz", actions=actions, imagined_block_xy=bxy[C - 1:].astype(np.float32), imagined_gripper_xy=(rod_t[C - 1:, :2] + off[:2]).astype(np.float32),
         predicted_centered=best.astype(np.float32), offset_world=off.astype(np.float32), context_frames=np.array(idx), final_dist=np.float32(imag), plan_seconds=np.float32(secs))
json.dump(dict(method="gns_bestofN_freeroll_direct", checkpoint=A.checkpoint, scenario_idx=A.scenario_idx, executed=c, context_frames=idx, frames_rolled=F,
               population=K, best_k=kb, imagined_final_dist_m=imag, plan_seconds=secs), open(out / "plan.json", "w"), indent=1)
print(f"GNS plan: executed {c}, context frames {idx}, rolled {F}, best k {kb}, imagined {imag * 1e3:.1f} mm, {secs:.1f} s" + (f", frame-0 context vs open-loop particles max {recon * 1e3:.4f} mm" if recon is not None else ""))
