"""Run a list of guided requests over one or more servers in parallel, replay each finished run in its simulator, and
write a summary table. Arms = JSON list of request dicts (each needs "name"); "base" keys are merged into every arm.

  python bin/sweep.py --arms sweeps/<file>.json --servers http://localhost:8766,http://localhost:8767 --gpus 0,1
"""
import argparse, json, os, queue, subprocess, sys, threading, time, urllib.request
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
ap = argparse.ArgumentParser(); ap.add_argument("--arms", required=True); ap.add_argument("--servers", required=True); ap.add_argument("--gpus", required=True)
ap.add_argument("--skip_done", action="store_true")
A = ap.parse_args()
spec = json.load(open(A.arms)); base = spec.get("base", {}); arms = [dict(base, **a) for a in spec["arms"]]
PY = sys.executable; Q = queue.Queue(); LOCK = threading.Lock()
for a in arms:
    Q.put(a)
log = open(os.path.splitext(A.arms)[0] + ".log", "a")


def say(msg):
    with LOCK:
        line = time.strftime("%H:%M:%S ") + msg; print(line, flush=True); log.write(line + "\n"); log.flush()


def worker(url, gpu):
    while True:
        try:
            a = Q.get_nowait()
        except queue.Empty:
            return
        run = os.path.join(ROOT, "runs", a.get("domain", "pickup"), a["name"])
        if not (A.skip_done and os.path.exists(os.path.join(run, "result.json"))):
            say(f"[{url}] start {a['name']}")
            t = time.time()
            try:
                r = urllib.request.Request(url + "/guide", data=json.dumps(a).encode(), headers={"Content-Type": "application/json"}, method="POST")
                res = json.loads(urllib.request.urlopen(r, timeout=24 * 3600).read())
            except Exception as e:
                say(f"[{url}] FAILED {a['name']}: {e}"); continue
            say(f"[{url}] done {a['name']} in {time.time() - t:.0f} s: unguided {res.get('unguided_mm')} best {res.get('best_mm')}")
        if not os.path.exists(os.path.join(run, "replay_isaac", "analysis.json")):
            rl = open(os.path.join(ROOT, "logs", a["name"] + "_replay.log"), "w")
            p = subprocess.run([PY, "-u", os.path.join(HERE, "replay.py"), "--run", run, "--gpu", str(gpu)], stdout=rl, stderr=subprocess.STDOUT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)))
            say(f"[{url}] replay {a['name']} exit {p.returncode}")


ts = [threading.Thread(target=worker, args=(u, g)) for u, g in zip(A.servers.split(","), A.gpus.split(","))]
[t.start() for t in ts]; [t.join() for t in ts]
subprocess.run([PY, os.path.join(HERE, "summarize.py"), A.arms])
