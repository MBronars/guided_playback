"""Generate one dex-hand shard: roll out a tactile-genesis teacher policy and record rigid state.

One shard = one teacher, one rollout of --steps control steps from reset across --num_envs envs. Each env contributes
one episode: its frames up to and including its first termination (or the whole rollout). A seeded --perturb_frac of
envs gets Gaussian noise on the raw action so that failures exist (not calibrated per task, see README.md).

Recorded per control step (40 Hz): every robot link pose, the object pose, dof positions, the executed raw action,
reward, done and time-out flags, which termination fired (done_reason, names in the manifest's done_reasons, including
Eden's reset of envs the Genesis rigid solver flags), and each env's object geometry. Point clouds are rebuilt offline
(collection.dex.points). Run from the repository root with the checkout's Python, on a GPU:

    <checkout>/dexterous-hands/.venv/bin/python -m collection.dex.generate \\
        --tactile_genesis <checkout> --teacher in_palm_rotate-xhand1 --out_dir <new run dir>
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

from collection.common import digest

TEACHERS = ("in_palm_rotate-xhand1", "in_hand_repose-xhand1", "in_hand_repose-sharpa", "screwdriver-xhand1",
            "screwdriver-sharpa")
PINNED_COMMIT = "3023bcd2d43fe947327713ddce3c2123d475735d"   # the checkout tactile_genesis_teachers_v1 used


def git_head(root):
    r = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True)
    return r.stdout.strip() or None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tactile_genesis", required=True, help="checkout made by setup_tactile_genesis.sh")
    ap.add_argument("--teacher", required=True, choices=TEACHERS)
    ap.add_argument("--out_dir", required=True, help="new run directory; an existing shard is never overwritten")
    ap.add_argument("--num_envs", type=int, default=2048)
    ap.add_argument("--steps", type=int, default=0, help="0 = the task's episode length")
    ap.add_argument("--perturb_frac", type=float, default=0.2)
    ap.add_argument("--noise_std", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grasps", default=None,
                    help="screwdriver teachers: grasp file (.pt) used instead of the task's own for this hand")
    ap.add_argument("--objects", default=None,
                    help="in-hand teachers: JSON list of mesh files (.obj, metres, origin at the bounding-box centre) "
                         "replacing the task's object list; relative paths are relative to the JSON file")
    ap.add_argument("--object_variants", default=None,
                    help="comma-separated indices of the task's own object variants to keep (e.g. 0 = screwdriver model 0)")
    ap.add_argument("--part", default=None,
                    help="names the shard <teacher>@<part> so one teacher can have several shards in one folder")
    ap.add_argument("--max_collision_pairs", type=int, default=None,
                    help="replaces the task's rigid-solver collision-pair budget (in-hand tasks: 30 repose, 48 in-palm, sized "
                         "for single-hull objects); an env that overflows its contact buffers ends as a solver error")
    a = ap.parse_args()
    task, hand = a.teacher.rsplit("-", 1)
    if a.grasps and task != "screwdriver":
        ap.error("--grasps applies to the screwdriver teachers only")
    if a.objects and task == "screwdriver":
        ap.error("--objects applies to the in-hand teachers only")
    if a.objects and a.object_variants:
        ap.error("--object_variants selects from the task's own objects; --objects replaces them")
    variants_kept = [int(v) for v in a.object_variants.split(",")] if a.object_variants else None
    if variants_kept and task == "screwdriver" and variants_kept != list(range(len(variants_kept))):
        # the grasp file maps the scene's models to its own by position: a non-prefix subset needs a file with
        # exactly those models
        import torch
        if not a.grasps or len(torch.load(a.grasps, map_location="cpu")["geom_idx"].unique()) != len(variants_kept):
            ap.error("screwdriver: keep a prefix of the models, or pass --grasps holding exactly the kept models")

    root = os.path.abspath(a.tactile_genesis)
    hands = os.path.join(root, "dexterous-hands")
    config = os.path.join(hands, "checkpoints", "teachers", f"{a.teacher}.yaml")
    checkpoint = os.path.join(hands, "checkpoints", "teachers", f"{a.teacher}.pt")
    out_dir = os.path.abspath(a.out_dir)
    stem = a.teacher + (f"@{a.part}" if a.part else "")
    shard = os.path.join(out_dir, f"{stem}.npz")
    if os.path.exists(shard):
        raise FileExistsError(f"refusing to overwrite {shard}")
    source = {"tactile_genesis_commit": git_head(root), "generator_sha256": digest(os.path.abspath(__file__)),
              "config_sha256": digest(config), "checkpoint_sha256": digest(checkpoint)}
    grasps = os.path.abspath(a.grasps) if a.grasps else None
    if grasps:
        source.update(grasps_file=grasps, grasps_sha256=digest(grasps))
    object_files = None
    if a.objects:
        objects = os.path.abspath(a.objects)
        object_files = [os.path.join(os.path.dirname(objects), f) for f in json.load(open(objects))]
        missing = [f for f in object_files if not os.path.isfile(f)]
        if missing:
            raise FileNotFoundError(f"object meshes not found: {missing[:3]}")
        source.update(objects_file=objects, objects_sha256=digest(objects), object_files=object_files)
    if variants_kept:
        source.update(object_variants_kept=variants_kept)
    if source["tactile_genesis_commit"] != PINNED_COMMIT:
        print(f"[generate] WARNING: checkout at {source['tactile_genesis_commit']}, v1 used {PINNED_COMMIT}", flush=True)

    # run the simulator stack the way tactile-genesis's own scripts do: from dexterous-hands/ with src/ importable,
    # and without this repository on the path (both have a top-level `utils`)
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.curdir) != here]
    src = os.path.join(hands, "src")
    sys.path.insert(0, src)
    os.environ["PYTHONPATH"] = src
    os.makedirs(out_dir, exist_ok=True)
    os.chdir(hands)

    import torch
    import eden as en
    import genesis as gs
    from eden.envs.wrappers.rsl_rl_env import RslRlVecEnvWrapper
    from eden.utils.configs import EdenRLConfig
    from registry import _load_tasks, make_runner

    en.init(backend=gs.gpu, log_root_path=os.path.join(out_dir, "_eden_logs"), performance_mode=True, deterministic=False)
    _load_tasks()
    cfg = EdenRLConfig.load_from_file(config)
    cfg.env_options.num_envs = a.num_envs
    if hasattr(cfg.env_options, "num_eval_envs"):
        cfg.env_options.num_eval_envs = a.num_envs
    if getattr(cfg, "runner_options", None) is not None:
        cfg.runner_options.logger = "tensorboard"
    if a.max_collision_pairs:
        cfg.env_options.max_collision_pairs = a.max_collision_pairs
    if grasps:
        cfg.event_options.load_grasp.grasps_paths[hand] = grasps
    if object_files:
        # the way tactile-genesis's ManipulationObjectMod builds the object and its goal ghost, with mesh variants in
        # place of the primitives; material stays unset like the primitives' (rho 200), Genesis convexifies the meshes
        import copy
        from eden.options.entities import EntityOptions
        from task_mods import _make_options_ghost
        scene = cfg.scene_options
        scene.obj.grouped_entities = [EntityOptions(file=f) for f in object_files]
        ghosts = copy.deepcopy(scene.obj.grouped_entities)
        for ghost in ghosts:
            _make_options_ghost(ghost)
        scene.vis_obj.grouped_entities = ghosts
    if variants_kept:
        scene = cfg.scene_options
        for name in ("obj", "vis_obj"):
            entity = getattr(scene, name, None)
            if entity is not None:
                entity.grouped_entities = [entity.grouped_entities[v] for v in variants_kept]
    t_build = time.time()
    env = RslRlVecEnvWrapper.from_config(cfg, show_viewer=False, eval_mode=True)
    policy = make_runner(env, checkpoint).get_inference_policy(device=gs.device)
    base = env.unwrapped
    robot, obj = base.entities["robot"], base.entities["obj"]
    steps = a.steps or int(base.max_episode_length)
    ctrl_dt = float(cfg.env_options.sim_dt * cfg.env_options.decimation)
    build_s = time.time() - t_build

    g = torch.Generator(device="cpu").manual_seed(a.seed)
    # Genesis sets torch's default device to CUDA, so CPU tensors must say so explicitly
    perturbed = torch.zeros(a.num_envs, dtype=torch.bool, device="cpu")
    perturbed[torch.randperm(a.num_envs, generator=g, device="cpu")[: round(a.perturb_frac * a.num_envs)]] = True
    perturbed_dev = perturbed.to(gs.device)

    obs, _ = env.reset()
    # object geometry: a grouped object has one or more geoms per variant (a screwdriver is several convex pieces),
    # each active in some envs, so an env's object is the union of its geoms
    geo = {}
    geoms_of_env = [[] for _ in range(a.num_envs)]
    for i, geom in enumerate(obj.links[0].geoms):
        geo[f"objgeom_verts_{i}"] = np.asarray(geom.init_verts, dtype=np.float32)
        geo[f"objgeom_faces_{i}"] = np.asarray(geom.init_faces, dtype=np.int64)
        geo[f"objgeom_pos_{i}"] = np.asarray(geom.init_pos, dtype=np.float32)
        geo[f"objgeom_quat_{i}"] = np.asarray(geom.init_quat, dtype=np.float32)
        envs = getattr(geom, "active_envs_idx", None)
        envs = np.arange(a.num_envs) if envs is None else np.asarray(envs, dtype=np.int64)
        geo[f"objgeom_envs_{i}"] = envs
        for e in envs:
            geoms_of_env[int(e)].append(i)
    assert all(geoms_of_env), "some envs have no object geometry"
    variants = {v: k for k, v in enumerate(sorted({tuple(gs_) for gs_ in geoms_of_env}))}
    shape_of_env = np.array([variants[tuple(gs_)] for gs_ in geoms_of_env], dtype=np.int64)

    T, E = steps, a.num_envs
    L, D = len(robot.links), int(robot.n_dofs)
    buf = {"link_pos": np.zeros((T, E, L, 3), np.float32), "link_quat": np.zeros((T, E, L, 4), np.float32),
           "obj_pos": np.zeros((T, E, 3), np.float32), "obj_quat": np.zeros((T, E, 4), np.float32),
           "dofs": np.zeros((T, E, D), np.float32), "reward": np.zeros((T, E), np.float32),
           "done": np.zeros((T, E), bool), "time_out": np.zeros((T, E), bool), "done_reason": np.zeros((T, E), np.uint8)}
    # which termination ended an env: the task's terms in the manager's order (their flags stay set after the step),
    # else Eden's reset of envs the Genesis rigid solver flags (contact-buffer overflow, NaN), which has no term
    terms = base.termination_manager
    term_names = list(terms._term_names)
    timeout_names = [n for n in term_names if terms._terms[n].time_out]
    done_reasons = ["none", *term_names, "solver_error"]
    action_dim = None
    t0 = time.time()
    for t in range(T):
        with torch.no_grad():
            act = policy(obs)
            act = act + a.noise_std * torch.randn_like(act) * perturbed_dev[:, None]
        if action_dim is None:
            action_dim = act.shape[1]
            buf["action"] = np.zeros((T, E, action_dim), np.float32)
        buf["link_pos"][t] = robot.get_links_pos().cpu().numpy()
        buf["link_quat"][t] = robot.get_links_quat().cpu().numpy()
        buf["obj_pos"][t] = obj.get_pos().cpu().numpy()
        buf["obj_quat"][t] = obj.get_quat().cpu().numpy()
        buf["dofs"][t] = robot.get_dofs_pos().cpu().numpy()
        buf["action"][t] = act.cpu().numpy()
        obs, rew, done, extras = env.step(act)
        buf["reward"][t] = rew.cpu().numpy()
        done = done.bool()
        buf["done"][t] = done.cpu().numpy()
        reason = torch.zeros(E, dtype=torch.uint8, device=done.device)
        for code, name in enumerate(term_names, 1):
            reason[(reason == 0) & terms.get_term(name)] = code
        reason[(reason == 0) & done] = len(done_reasons) - 1
        buf["done_reason"][t] = reason.cpu().numpy()
        buf["time_out"][t] = torch.stack([terms.get_term(n) for n in timeout_names]).any(0).cpu().numpy()
    sim_s = time.time() - t0

    # episode table: each env's first termination ends its episode
    first_done = np.where(buf["done"].any(0), buf["done"].argmax(0), T - 1)
    length = first_done + 1
    ended_early = buf["done"].any(0) & ~buf["time_out"][first_done, np.arange(E)] & (first_done < T - 1)
    episodes = {"env": np.arange(E), "length": length, "terminated_early": ended_early,
                "perturbed": perturbed.numpy(), "shape_id": shape_of_env,
                "end_reason": buf["done_reason"][first_done, np.arange(E)],
                "return": np.array([buf["reward"][: length[e], e].sum() for e in range(E)], np.float32)}

    t_w = time.time()
    np.savez(shard, **buf, **geo, **{f"episode_{k}": v for k, v in episodes.items()},
             link_names=np.array([l.name for l in robot.links]),
             dof_names=np.array([getattr(j, "name", str(i)) for i, j in enumerate(getattr(robot, "joints", []))]),
             control_dt=np.array(ctrl_dt))
    write_s = time.time() - t_w
    size_gb = os.path.getsize(shard) / 1e9

    def rate(mask):
        return float(ended_early[mask].mean()) if mask.any() else None

    # the manifest is written last: a shard without one is incomplete
    manifest = {
        "name": a.teacher, "part": a.part, "shard": shard, "config": config, "checkpoint": checkpoint,
        "robot_file": str(getattr(robot.morph, "file", "")), "num_envs": E, "steps": T, "control_dt": ctrl_dt,
        "perturb_frac": a.perturb_frac, "noise_std": a.noise_std, "seed": a.seed,
        "links": L, "dofs": D, "action_dim": int(action_dim), "object_variants": len(variants), "object_geoms": len(obj.links[0].geoms),
        "max_collision_pairs": int(cfg.env_options.max_collision_pairs),
        "multiplier_collision_broad_phase": int(cfg.env_options.multiplier_collision_broad_phase),
        "episodes": E, "mean_length_steps": float(length.mean()),
        "done_reasons": done_reasons,
        "end_reasons": {r: int((episodes["end_reason"] == i).sum()) for i, r in enumerate(done_reasons)},
        "terminated_early_rate": {"clean": rate(~perturbed.numpy()), "perturbed": rate(perturbed.numpy())},
        "timing_s": {"build": round(build_s, 1), "simulate": round(sim_s, 1), "write": round(write_s, 1)},
        "env_steps_per_s": round(T * E / sim_s), "size_gb": round(size_gb, 3),
        **source,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    json.dump(manifest, open(os.path.join(out_dir, f"{stem}.json"), "w"), indent=2)
    print("[generate]", json.dumps({k: manifest[k] for k in ("name", "episodes", "mean_length_steps", "terminated_early_rate", "end_reasons",
                                                             "timing_s", "env_steps_per_s", "size_gb")}), flush=True)


if __name__ == "__main__":
    main()
