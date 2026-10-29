"""Letter-moulding sweep through ONE warm server on ONE GPU: guide the arms back to back, replay each finished arm's members in
Genesis in the background (one replay at a time, on the same GPU), and keep a summary table current.

  CUDA_VISIBLE_DEVICES=<gpu> python bin/letter_sweep.py --arms sweeps/letters/stage1.json --url http://localhost:8795/guide
Idempotent: arms with a result.json are not re-guided, arms with replay_genesis/analysis.json are not re-replayed.
Summary: sweeps/letters/<arms>_summary.{md,json}. Numbers: imagined = best all-frame objective per member (result.json best_mm,
RMS mm); imagined last = last-frame chamfer of the picked member; sim = Genesis last-frame chamfer (the paper metric)."""
import argparse, json, os, subprocess, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, ROOT)
from gp.letter import chamfer_rms_np
ap = argparse.ArgumentParser(); ap.add_argument("--arms", required=True); ap.add_argument("--url", required=True); ap.add_argument("--members", default=None, help="replay only these members (default all)")
A = ap.parse_args()
spec = json.load(open(A.arms)); base = spec.get("base", {}); arms = [dict(base, **a) for a in spec["arms"]]
PY = sys.executable; RUNS = os.path.join(base.get("runs", os.path.join(ROOT, "runs")), "mould"); TAG = os.path.splitext(os.path.basename(A.arms))[0]
SUMMARY = os.path.join(os.path.dirname(os.path.abspath(A.arms)), TAG + "_summary")
LOG = open(SUMMARY + ".log", "a")


def say(m):
    line = time.strftime("%H:%M:%S ") + m; print(line, flush=True); LOG.write(line + "\n"); LOG.flush()


def post(body):
    import urllib.request
    r = urllib.request.Request(A.url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST")
    return json.loads(urllib.request.urlopen(r, timeout=24 * 3600).read())


def knobs(a):
    lr = f"{a['lr']}" + (f"->{a['lr_final']}" if a.get("lr_final") else "")
    return f"{a['letter']} it{a['iterations']} lr {lr} {a['optimizer']} S{a['steps']} tf{a['t_freeze']} N{a['transitions']} frames {a['letter_frames']} seed {a['seed']}"


def row(a):
    run = os.path.join(RUNS, a["name"]); rp = os.path.join(run, "result.json")
    if not os.path.exists(rp): return None
    rj = json.load(open(rp)); z = np.load(os.path.join(run, "rollout.npz")); lf = np.load(os.path.join(run, "letter_field.npz"), allow_pickle=True)
    cov = lf["coverage"]; dough = z["roles"][0] == 2; pick = int(np.argmin(rj["best_mm"]))
    last = [chamfer_rms_np(z["predicted"][m, -1, dough, :2], cov) * 1e3 for m in range(z["predicted"].shape[0])]
    ung = [chamfer_rms_np(z["unguided"][m, -1, dough, :2], cov) * 1e3 for m in range(z["unguided"].shape[0])]
    r = dict(name=a["name"], letter=a["letter"], knobs=knobs(a), picked=pick, imagined_obj_mm=rj["best_mm"], imagined_last_mm=[round(v, 2) for v in last],
             unguided_last_mm=[round(v, 2) for v in ung], guide_s=round(sum(rj["iteration_seconds"]), 1), peak_gib=max(rj["peak_gib"]))
    ap_ = os.path.join(run, "replay_genesis", "analysis.json")
    if os.path.exists(ap_):
        rows = [x for x in json.load(open(ap_))["rows"] if x["variant"] == "guided" and "sim_chamfer_mm" in x]
        sim = {x["member"]: x["sim_chamfer_mm"] for x in rows}
        if sim:
            r.update(sim_mm={k: round(v, 2) for k, v in sim.items()}, sim_picked_mm=round(sim.get(pick, float("nan")), 2), sim_best_mm=round(min(sim.values()), 2),
                     sim_best_member=int(min(sim, key=sim.get)), gap_mm=round(max(abs(sim[k] - last[k]) for k in sim), 2))
    return r


def write_summary():
    rows = [r for r in (row(a) for a in arms) if r]
    json.dump(rows, open(SUMMARY + ".json", "w"), indent=1)
    out = [f"# {TAG}: letter moulding arms (mould ctxgen step 60k, 3 members, episode 0)", "",
           "imagined = best all-frame objective per member (RMS mm); last = last-frame chamfer of the picked member (imagined); "
           "sim = Genesis last-frame chamfer (paper metric), picked member and best member; unguided = last-frame chamfer of the unguided draws.", "",
           "| arm | knobs | picked | imagined obj | imagined last | sim picked | sim best (m) | unguided last | gap | guide s |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r['name']} | {r['knobs']} | m{r['picked']} | {min(r['imagined_obj_mm']):.2f} | {r['imagined_last_mm'][r['picked']]:.2f} | "
                   f"{r.get('sim_picked_mm', '')} | {r.get('sim_best_mm', '')} ({'m' + str(r['sim_best_member']) if 'sim_best_member' in r else ''}) | "
                   f"{min(r['unguided_last_mm']):.2f} | {r.get('gap_mm', '')} | {r['guide_s']} |")
    open(SUMMARY + ".md", "w").write("\n".join(out) + "\n")


pending = None
def replay(a):
    global pending
    if pending is not None:
        pending[0].wait(); say(f"replay done {pending[1]} exit {pending[0].returncode}"); write_summary(); pending = None
    run = os.path.join(RUNS, a["name"])
    if os.path.exists(os.path.join(run, "replay_genesis", "analysis.json")): return
    cmd = [PY, "-u", os.path.join(HERE, "replay.py"), "--run", run, "--variants", "guided", "--gpu", "0"] + (["--members", A.members] if A.members else [])
    pending = (subprocess.Popen(cmd, cwd=ROOT, stdout=open(os.path.join(ROOT, "logs", a["name"] + "_replay.log"), "w"), stderr=subprocess.STDOUT), a["name"])
    say(f"replay start {a['name']}")


for a in arms:
    run = os.path.join(RUNS, a["name"])
    if not os.path.exists(os.path.join(run, "result.json")):
        body = {k: v for k, v in a.items()}; body["desc"] = knobs(a)
        say(f"guide start {a['name']}: {knobs(a)}"); t = time.time()
        try:
            res = post(body)
        except Exception as e:
            say(f"guide FAILED {a['name']}: {e}"); continue
        say(f"guide done {a['name']} in {time.time() - t:.0f} s: unguided {res.get('unguided_mm')} best {res.get('best_mm')}")
    write_summary(); replay(a)
if pending is not None:
    pending[0].wait(); say(f"replay done {pending[1]} exit {pending[0].returncode}")
write_summary(); say("sweep finished: " + SUMMARY + ".md")
