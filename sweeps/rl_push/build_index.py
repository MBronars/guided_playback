"""Index of the rl_push TEST episodes in loader order (the indices bin/guide.py --episodes takes for --domain rl_push).
Replicates data/rl_push.py UnifiedRLPushDataset._eval with batch_size 1: objects in _eval_order, episodes of an object
sorted (longest first, then episode id). Records per loader index: file, episode, asset, category, frames, recorded goal,
push direction, cell, object bbox (m) from object_cloud_outer_surface, hand start and object start (release frame, m).
Writes sweeps/rl_push/test_index.json."""
import json, os, sys, glob, numpy as np
sys.path.insert(0, "/data/user_data/mbronars/packages/pwm_final_paper"); os.chdir("/data/user_data/mbronars/packages/pwm_final_paper")
from data.rl_push import UnifiedRLPushDataset, load_push_object, asset_key_of
ROOT = "/data/group_data/katefgroup-ssd/world_model/particle_world_model/releases/pwm-data-v1/rl_push/test"
EXC = "/data/group_data/katefgroup-ssd/world_model/particle_world_model/releases/pwm-data-v1/rl_push/manifests/rl_push_exclusions_20260918.json"
ds = UnifiedRLPushDataset(ROOT, split="all", training=False, shuffle=False, manifests=[EXC], raw=True); ds.future_len = 10 ** 6
ds.first_window = True; ds.batch_size = 1
order = ds._eval_order()
by_object = {}
for i, e in ds.episodes: by_object.setdefault(i, []).append(e)
rows = []
for i in order:
    obj = ds.objects[i]
    z = np.load(os.path.join(ROOT, obj.file), allow_pickle=True)
    pts = z["object_cloud_outer_surface"] * float(z["scale"]) if False else z["object_cloud_outer_surface"]
    bb = pts.max(0) - pts.min(0)
    for L, e, start in ds._eval_windows(i, by_object.get(i, ())):
        s = int(z["episode_start"][e]); n = int(z["episode_len"][e])
        rows.append(dict(loader=len(rows), file=obj.file, asset=obj.asset_key, category=obj.category, e=int(e), frames=int(L), start=int(start), n=n,
                         goal_pos=z["goal_pos"][e].tolist(), goal_dist=float(z["goal_dist_m"][e]), push_dir=float(z["push_dir_deg"][e]), repo=float(z["reposition_deg"][e]),
                         cell=str(z["cell_names"][int(z["cell"][e])]), success_step=int(z["success_step"][e]),
                         obj_pos0=z["object_pos"][s + start].tolist(), obj_pos_end=z["object_pos"][s + n - 1].tolist(), hand_pos0=z["hand_pos"][s + start].tolist(),
                         obj_quat0=z["object_quat_wxyz"][s + start].tolist(), obj_quat_end=z["object_quat_wxyz"][s + n - 1].tolist(),
                         bbox_mm=[round(float(v) * 1e3, 1) for v in bb], scale=float(z["scale"])))
json.dump(rows, open("/data/user_data/mbronars/packages/guided_playback/sweeps/rl_push/test_index.json", "w"), indent=1)
print(len(rows), "episodes;", len(set(r["asset"] for r in rows)), "objects")
for r in rows[:8]: print(r["loader"], r["asset"], r["e"], r["frames"], r["bbox_mm"], r["cell"], round(r["goal_dist"], 3))
