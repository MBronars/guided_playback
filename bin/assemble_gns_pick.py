"""The 25 GNS pick-benchmark plans as one guided_playback run, so the benchmark's own Isaac scorer (bin/replay.py ->
gp/replay/isaac_float.py, the unchanged pick path: task 'lift') replays them exactly like pw_benchmark.
  python bin/assemble_gns_pick.py <gns plan dir> <run name>
Plans come as <prefix>_t<id>_<asset>_rollout.npz with predicted [1, 61, 1183, 3] in the benchmark point order and model
frame (checked against sweeps/pick_benchmark_5x5_scenes.npz at frame 0). Roles / particle_body / target come from the
benchmark scene file, so nothing is taken from the GNS side except the predicted cloud."""
import glob, json, os, re, sys
import numpy as np
GP = "/data/user_data/mbronars/packages/guided_playback"
SRC, NAME = sys.argv[1], sys.argv[2]
OUT = f"{GP}/runs/pickup/{NAME}"
assert not os.path.exists(OUT), f"{OUT} exists; pick a new run name"
Z = np.load(f"{GP}/sweeps/pick_benchmark_5x5_scenes.npz")
summ = json.load(open(glob.glob(f"{SRC}/*summary.json")[0]))
pred, roles, omask, pb, tgt, eps, secs, pmm = [], [], [], [], [], [], [], []
for t in range(5):
    for asset in ("milk_carton_asset_0037", "pasta_box_asset_0058", "peanut_butter_jar_asset_0003", "soda_can_asset_0012", "water_bottle_asset_0087"):
        f = [g for g in glob.glob(f"{SRC}/*_rollout.npz") if re.search(rf"_t{t}_{asset}_rollout\.npz$", g)]
        assert len(f) == 1, (t, asset, f)
        P = np.load(f[0])["predicted"][0]; pm = Z[f"{asset}__points_model"]
        err = float(np.abs(P[0] - pm).max()); assert err < 1e-4, f"{f[0]}: frame 0 differs from the benchmark scene by {err * 1e3:.2f} mm"
        r = Z[f"{asset}__roles"]; row = [x for x in summ if x["asset"] == asset and int(x["target"]) == t][0]
        pred.append(P); roles.append(r); omask.append(r == 2); pb.append(Z[f"{asset}__particle_body"]); tgt.append(Z[f"{asset}__targets_model"][t])
        eps.append(f"{asset}_t{t}"); secs.append(float(row["plan_seconds"])); pmm.append(float(row["predicted_mm"]))
os.makedirs(OUT)
P = np.stack(pred).astype(np.float32)
np.savez(os.path.join(OUT, "rollout.npz"), predicted=P, unguided=P, roles=np.stack(roles).astype(np.int8), object_mask=np.stack(omask),
         particle_body=np.stack(pb), episodes=np.array(eps), target=np.stack(tgt).astype(np.float32), offsets=np.zeros((len(eps), 3), np.float32),
         boxes=np.zeros((0, 6), np.float32), trace_mm=np.zeros((1, len(eps)), np.float32), step=np.array(0), plan_seconds=np.array(secs))
json.dump(dict(adapter="gns", domain="pickup", sim="isaac", task="lift", frame_dt=1 / 20, mode=os.path.basename(SRC), source=os.path.abspath(SRC),
               episodes=eps, joint_limits_mm=[13.5, 53.5], best_mm=pmm, unguided_mm=pmm, plan_seconds=secs),
          open(os.path.join(OUT, "result.json"), "w"), indent=1)
print(f"assembled {len(eps)} GNS plans -> {OUT}  predicted {P.shape}")
