"""Timing calibration for the budget study: per population P, compile (1-iteration warmup), then one timed run with I
iterations on held-out hard seed-0 scenario 0. Records grad-iteration seconds, final no-grad pass, peak GiB.
Base recipe: S 6, unitgrad lr 0.1 constant, t_freeze 0.8, shell, full churn, fixed FPS, 60 transitions."""
import json, sys, time, urllib.request
URL = sys.argv[1]; POPS = [int(v) for v in sys.argv[2].split(",")]; ITS = int(sys.argv[3]) if len(sys.argv) > 3 else 4
SC = "/data/user_data/mbronars/packages/PointWorld/scenarios/hard/seed_0/ep_000.npz"
OUT = "/data/user_data/mbronars/packages/guided_playback/sweeps/budget_push/calib.jsonl"
def post(b):
    r = urllib.request.Request(URL, data=json.dumps(b).encode(), headers={"Content-Type": "application/json"}, method="POST")
    return json.loads(urllib.request.urlopen(r, timeout=24 * 3600).read())
def body(P, it, name):
    return dict(domain="pushing", task="push", episodes=",".join(["0"] * P), transitions=60, steps=6, churn=True,
                optimizer="unitgrad", shell=True, t_freeze=0.8, scenario=SC, iterations=it, lr=0.1, lr_final=None,
                name=name, desc="budget-study timing calibration", free_fps=False)
for P in POPS:
    t0 = time.time(); post(body(P, 1, f"_budget_calib_warm_P{P}")); tc = time.time() - t0
    t0 = time.time(); r = post(body(P, ITS, f"_budget_calib_P{P}_I{ITS}")); tw = time.time() - t0
    s = r["iteration_seconds"]
    rec = dict(P=P, I=ITS, compile_s=round(tc, 1), wall_s=round(tw, 1), iter_s=s, grad_iter_mean=round(sum(s[1:-1]) / max(1, len(s) - 2), 2),
               first_iter=s[0], final_pass=s[-1], total=round(sum(s), 1), peak_gib=r.get("peak_gib"), best_mm=r["best_mm"])
    print(json.dumps(rec), flush=True); open(OUT, "a").write(json.dumps(rec) + "\n")
print("CALIB_DONE", flush=True)
