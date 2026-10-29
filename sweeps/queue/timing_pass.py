"""Serial, plan-only timing pass: every curve point re-planned on 3 problems, one process at a time on an otherwise idle node, no sim replays.
Commands are the done queue entries themselves, renamed T_<name>, with --no_replay and a 3-problem limit; push MPPI runs with batch = K.
  python sweeps/queue/timing_pass.py <GPU> <HOST> [only-substring]"""
import glob, json, os, re, subprocess, sys
GPU, HOST = sys.argv[1], sys.argv[2]; ONLY = sys.argv[3] if len(sys.argv) > 3 else ""
Q = "/data/group_data/katefgroup/runs_pwm_mbronars/guided_playback_budget/queue"; C = Q + "/../_cfgs"; HERE = "/data/user_data/mbronars/packages/guided_playback"
def ent(name):
    f = sorted(glob.glob(f"{Q}/done/*_{name}.json")); return json.load(open(f[-1]))["cmd"] if f else None
push = ["FM_push_oneshot"] + [f"G_push_pool{n}_I0" for n in (4, 8, 16, 32)] + [f"G_push_pool32_I{i}_P1" for i in (1, 2, 3, 5)] + ["G_push_pool32_I5_P2", "F_push_P3", "F_push_P4"] + \
       ["G_push_pool32_I5_P6", "F_push_P8", "F_push_P12"] + [f"G_push_pool32_I5_P{p}" for p in (16, 22, 24, 28, 32, 40)] + \
       [f"FM_push_mppi_K{k}" for k in (1, 2, 4, 12, 16, 28, 48)] + [f"XT_mppi_K{k}_s0" for k in (8, 20, 40)]
pick = ["FM_pick_oneshot"] + [f"G_pick_pool{n}_I0" for n in (4, 8, 16, 32)] + [f"G_pick_pool32_I{i}_P1" for i in (1, 2, 3, 5)] + ["G_pick_pool32_I5_P2", "F_pick_P3", "F_pick_P4", "G_pick_pool32_I5_P5",
       "G_pick_pool32_I5_P6", "F_pick_P8", "G_pick_pool32_I5_P10", "F_pick_P12"] + [f"FM_pick_mppi_K{k}" for k in (1, 2, 4, 12, 16, 24, 28)] + ["F_pick_mppi_K8", "F_pick_mppi_K20"]
pile = ["FM_pile_oneshot_c0"] + [f"G_pile_pool{n}_I0_c0" for n in (4, 8, 16, 32)] + [f"G_pile_pool32_I{i}_P1_c0" for i in (1, 2, 3, 5)] + ["G_pile_pool32_I5_P2_c0", "S10_pile_P3_I5_c0", "S10_pile_all_g0p01_c0",
       "G_pile_pool32_I5_P5_c0", "G_pile_pool32_I5_P6_c0", "S10_pile_P8_I5_c0"] + [f"FM_pile_mppi_K{k}_c0" for k in (1, 2, 3, 4, 6)] + ["PG_mppi_K8_a"]
todo = []
for task, names in (("push", push), ("pick", pick), ("pile", pile)):
    for n in names:
        c = ent(n)
        if c is None: print("MISSING", n, flush=True); continue
        assert f"--name {n}" in c, n
        c = c.replace(f"--name {n}", f"--name T_{n} --no_replay")
        if task == "push":
            c, k = re.subn(r"--n \d+", "--n 3", c); assert k == 1, n
            m = re.search(r"--K (\d+) --members \d+", c)
            if m: c = c.replace(m.group(0), f"--K {m.group(1)} --members {m.group(1)}")       # MPPI: the K candidates of a replan as one batch
        elif task == "pick": c = c.replace(f"--name T_{n}", f"--name T_{n} --n 3")
        else:
            c, k = re.subn(r"--start \d+ --n \d+", "--start 0 --n 3", c); assert k == 1, n
        todo.append((f"T_{n}", task, c))
if ONLY == "list":
    for t in todo: print(t[0], "|", t[2][t[2].index("--name"):][:150])
    sys.exit()
for name, task, c in todo:
    if ONLY and ONLY not in name: continue
    rec = f"{C}/{name}__{dict(push='test', pick='pick', pile='mould')[task]}.json"
    if os.path.exists(rec) and len(json.load(open(rec)).get("rows", [])) >= 3: print("have", name, flush=True); continue
    print("RUN", name, flush=True)
    rc = subprocess.call(["bash", "-c", c.format(GPU=GPU, HOST=HOST, URL="", CORES="", JOB="")], cwd=HERE, env=dict(os.environ, CUDA_VISIBLE_DEVICES=GPU))
    print("DONE", name, rc, flush=True)
print("TIMING_PASS_DONE", flush=True)
