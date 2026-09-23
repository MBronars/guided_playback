"""Genesis 0.4.6 MPM replay of a mould rollout (packages/Genesis/examples/replay_planner_output.py, genesis env, cwd
= packages/Genesis so the LOCAL Genesis is imported). Per (member, variant): the generated paddle rows are Kabsch-fit
to a paddle pose per frame, the frame-0 generated dough particles are injected, the paddle is PD-tracked with
frame_skip sim steps per model frame; the simulated dough particles are read back at every model frame.

Output contract: sim dough = simulated particles; sim paddle = frame-0 generated paddle rows under the simulated
paddle pose relative to its frame-0 pose."""
import json, os, pickle, subprocess
import numpy as np
from . import common as C

GENESIS = "/data/user_data/mbronars/packages/Genesis"
PY = "/data/user_data/mbronars/.miniconda3/envs/genesis/bin/python"


def run(run_dir, out, variants=("guided", "unguided"), script="examples/replay_planner_output.py", frame_skip=25, gpu="0"):
    z, rj = C.load_run(run_dir); os.makedirs(out, exist_ok=True)
    B, T, N = z["predicted"].shape[:3]; V = len(variants); roles = z["roles"]; obj_mask = z["object_mask"]
    pred = np.stack([z[C.VARIANT_KEY[v]] for v in variants], 1).astype(np.float32)                 # [B, V, T, N, 3]
    oid = np.where(obj_mask, 0, 1).astype(np.int64)                                                   # 0 dough, 1 paddle
    inp = os.path.abspath(os.path.join(out, "generated.npz"))
    np.savez(inp, predicted=pred, object_ids=oid, context_len=np.array(1))
    n_dough = int(obj_mask[0].sum())
    sim = np.zeros((V, B, T, N, 3), np.float32); sim_roles = roles.copy(); rows = []
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    log = open(os.path.join(out, "replay.log"), "w")
    for m in range(B):
        for v, var in enumerate(variants):
            rd = os.path.abspath(os.path.join(out, f"genesis_m{m}_{var}"))
            print(f"[genesis] member {m} {var} -> {rd}", flush=True)
            subprocess.run([PY, script, "--input_npz", inp, "--batch_idx", str(m), "--sample_idx", str(v), "--frame_skip", str(frame_skip),
                            "--surface_points", str(n_dough), "--record_dir", rd, "--no_video"], cwd=GENESIS, check=True, stdout=log, stderr=subprocess.STDOUT, env=env)
            d = pickle.load(open(os.path.join(rd, "replay.pkl"), "rb"))
            sb = d["sim_object_particles"]; sp, sq = d["sim_pusher_pos"], d["sim_pusher_quat"]; fs = int(d["frame_skip"])
            idx = np.clip(np.arange(1, T) * fs - 1, 0, len(sb) - 1)
            gen = pred[m, v]; dough = obj_mask[m]; pad = ~dough
            dst = sim[v, m]; dst[0] = gen[0]
            dst[1:, dough] = sb[idx] if sb[idx].shape[1] == n_dough else sb[idx][:, :n_dough]
            # paddle rows: frame-0 generated rows moved by the simulated paddle pose relative to the predicted frame-0 pose
            R0 = C.qmat(d["pred_pusher_quat"][0]); p0 = d["pred_pusher_pos"][0]
            loc = (gen[0, pad] - p0) @ R0                                                              # paddle-local
            Rt = C.qmat(sq[idx]); pt = sp[idx]
            dst[1:, pad] = np.einsum("tij,pj->tpi", Rt, loc) + pt[:, None]
            row = C.analysis_row(gen, sim[v, m], roles[m], dough, z["target"][m], m, var)
            row.update(genesis_dir=os.path.relpath(rd, out))
            rows.append(row)
    C.write(out, sim, sim_roles, list(variants), rows, extra=dict(sim="genesis 0.4.6 MPM, PD paddle", frame_skip=frame_skip))
