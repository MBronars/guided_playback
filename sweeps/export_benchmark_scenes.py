"""Frame-0 scene of every benchmark episode for baselines -> sweeps/pick_benchmark_5x5_scenes.npz (+ .json summary).
Model frame = raw data frame + SHIFT (-0.45, 0, -0.10) m (pickup canonical translation)."""
import json, numpy as np
SHIFT = np.array([-0.45, 0.0, -0.10])
spec = json.load(open("sweeps/pick_benchmark_5x5.json"))
pool = json.load(open("/data/user_data/mbronars/packages/pickup_replay_isaac/out/pool_pick_val/pool.json"))
P = np.load("/data/user_data/mbronars/packages/pickup_replay_isaac/out/pool_pick_val/specs.npz")
sel = json.load(open("sweeps/assetsweep_select.json"))
IDX = {"milk_carton_asset_0037": 11, "peanut_butter_jar_asset_0003": 1, "soda_can_asset_0012": 13, "pasta_box_asset_0058": 9, "water_bottle_asset_0087": 16}
out, summ = {}, []
for ai, a in enumerate(spec["assets"]):
    name = a["asset"]; s = sel[IDX[name]]; pi = s["pool_index"]; row = pool[pi]
    z = np.load(f"runs/pickup/cyl_t0_{name.rsplit('_asset', 1)[0]}_it10/rollout.npz")
    pts = z["gt"][0, 0]; roles = z["roles"][0]
    c = pts[roles == 2].mean(0)
    assert np.abs(c - np.array(a["f0_obj_centroid"])).max() < 1e-4, (name, c)
    assert np.abs(P["obj_centroid"][pi, 0, 0] + SHIFT - c).max() < 1e-4, (name, "pool mismatch")
    out[f"{name}__points_model"] = pts.astype(np.float32)                     # [1183, 3] frame-0 particles, model frame
    out[f"{name}__roles"] = roles                                             # 0 hand, 1 finger, 2 object, -1 pad
    out[f"{name}__particle_body"] = z["particle_body"][0]
    out[f"{name}__body_poses0_model"] = z["body_poses0"][0]                   # [K, 4, 4] rigid body poses at frame 0, model frame
    out[f"{name}__hand_pos0_raw"] = P["hand_pos"][pi, 0, 0]; out[f"{name}__hand_quat0_wxyz"] = P["hand_quat"][pi, 0, 0]
    out[f"{name}__finger_q0"] = P["finger_q"][pi, 0, 0]
    out[f"{name}__obj_pts0_raw"] = P["obj_pts0"][pi]
    tg = np.array([t["target_m"] for t in a["targets"]])
    out[f"{name}__targets_model"] = tg; out[f"{name}__targets_raw"] = tg - SHIFT
    summ.append(dict(asset=name, asset_mesh=s["asset"], asset_id=s["asset_id"], dataset_shard=f"gpu0/{row['tar']}.tar", key=row["key"], element=row["element"],
                     pool_index=pi, rho=s["rho"], mu=s["mu"], dims_mm=s["dims_mm"], f0_obj_centroid_model=np.round(c, 6).tolist(),
                     f0_obj_centroid_raw=np.round(c - SHIFT, 6).tolist(), hand_pos0_raw=np.round(P["hand_pos"][pi, 0, 0], 6).tolist(),
                     hand_pos0_model=np.round(P["hand_pos"][pi, 0, 0] + SHIFT, 6).tolist(), hand_quat0_wxyz=np.round(P["hand_quat"][pi, 0, 0], 6).tolist(),
                     finger_q0_m=np.round(P["finger_q"][pi, 0, 0], 5).tolist(), n_points=int((roles >= 0).sum()),
                     n_points_by_role={r: int((roles == k).sum()) for k, r in ((0, "hand"), (1, "fingers"), (2, "object"))},
                     targets_model=np.round(tg, 6).tolist(), targets_raw=np.round(tg - SHIFT, 6).tolist()))
np.savez("sweeps/pick_benchmark_5x5_scenes.npz", **out)
json.dump(summ, open("sweeps/pick_benchmark_5x5_scenes.json", "w"), indent=1)
for r in summ: print(json.dumps({k: r[k] for k in ("asset", "dataset_shard", "key", "element", "f0_obj_centroid_model", "hand_pos0_model", "finger_q0_m", "n_points_by_role")}))
