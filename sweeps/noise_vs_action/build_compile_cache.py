"""Warm every planning shape of the pick experiment in ONE process (the same guided_run code path the pool workers use),
then save torch's compile-cache artifacts: workers load them before compiling (python -m gp.pool work ... --cache <file>).
GPU-specific: build on the GPU type the workers run on (L40S).
  python sweeps/noise_vs_action/build_compile_cache.py <out .mega> [--quick]"""
import os, sys, time, json, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from gp.session import DEFAULTS, Args, load_adapter, guided_run
out = sys.argv[1]; quick = "--quick" in sys.argv
HERE = "/data/user_data/mbronars/packages/guided_playback"
CKPT = "/data/group_data/katefgroup/runs_pwm_mbronars/pwm_v1_pickup_drop50_100m_unet_bidir/checkpoints/final.pt"
gpu = torch.cuda.get_device_name(0); print("GPU", gpu, flush=True)
ad = load_adapter(Args(DEFAULTS, adapter="paper", ckpt=CKPT, eager=False, joint_limits=None))
base = dict(domain="pickup", task="lift", select=f"{HERE}/sweeps/assetsweep_select.json", transitions=60, steps=10, offset="-0.015,0.119,0.15",
            seed=0, churn=True, t_freeze=0.0, fix_fps=True, runs=os.path.join(os.path.dirname(out), "warm_runs"))
DNO = dict(optimizer="adamw", lr=0.1, shell=False, prob_reg=0.01, prob_mode="dno", prob_k=4, prob_shuffles=10, grad_clip=1.0)
jobs = [("B4 sampling", 4, dict(planner="cem", cem_K=4, cem_rounds=1, iterations=0)),
        ("B4 actions", 4, dict(planner="cem_actions", cem_K=4, cem_rounds=1, iterations=0))]
for n in ([] if quick else [8, 16, 32]):
    jobs.append((f"B{n} no-grad", n, dict(iterations=0)))
for P in ([1] if quick else [1, 2, 3, 4, 6, 8, 12]):
    jobs.append((f"B{P} grad", P, dict(DNO, iterations=1, **({"screen_pool": 32} if P <= 2 else {"prefilter": max(0, round(32 / P) - 1)}))))
log = []
for name, B, k in jobs:
    t = time.time(); A = Args(DEFAULTS, **base, **k, episodes=",".join(["11"] * B), name=f"warm_{name.replace(' ', '_')}")
    guided_run(ad, A); log.append((name, round(time.time() - t, 1))); print("warmed", name, log[-1][1], "s", flush=True)
b, info = torch.compiler.save_cache_artifacts(); open(out, "wb").write(b)
json.dump(dict(gpu=gpu, shapes=log, bytes=len(b), overrides=os.environ.get("PWM_MODEL_OVERRIDES"), torch=torch.__version__), open(out + ".json", "w"), indent=1)
print("saved", len(b) // 2**20, "MB ->", out, flush=True)
