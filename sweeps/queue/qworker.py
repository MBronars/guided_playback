"""Queue worker for one GPU. Jobs are JSON files in <Q>/pending/*.json: {"cmd": "...", "gpu_only": bool}. The worker claims one by an
atomic rename into <Q>/running/, runs cmd (bash) with {URL} {GPU} {CORES} {HOST} {JOB} substituted and CUDA_VISIBLE_DEVICES set,
then moves it to done/ or failed/ with rc and seconds. Jobs are taken in name order. Polls forever (until the Slurm job ends)."""
import json, os, subprocess, sys, time, socket
Q, GPU, URL, CORES = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
JOB = os.environ.get("SLURM_JOB_ID", "nojob"); HOST = socket.gethostname()
for d in ("pending", "running", "done", "failed", "logs"): os.makedirs(os.path.join(Q, d), exist_ok=True)
while True:
    if os.path.exists(os.path.join(Q, "STOP")): break
    got = None
    for f in sorted(os.listdir(os.path.join(Q, "pending"))):
        if not f.endswith(".json"): continue
        dst = os.path.join(Q, "running", f"{f[:-5]}__{HOST}_g{GPU}.json")
        try: os.rename(os.path.join(Q, "pending", f), dst); got = (f, dst); break
        except OSError: continue
    if got is None: time.sleep(20); continue
    f, dst = got; spec = json.load(open(dst)); t0 = time.time()
    cmd = spec["cmd"].format(URL=URL, GPU=GPU, CORES=CORES, HOST=HOST, JOB=JOB)
    log = os.path.join(Q, "logs", f[:-5] + ".log")
    print(time.strftime("%H:%M:%S"), "RUN", f, flush=True)
    rc = subprocess.call(["bash", "-c", cmd], stdout=open(log, "a"), stderr=subprocess.STDOUT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=GPU),
                         cwd="/data/user_data/mbronars/packages/guided_playback")
    spec.update(rc=rc, seconds=round(time.time() - t0, 1), host=HOST, gpu=GPU, slurm=JOB)
    json.dump(spec, open(os.path.join(Q, "done" if rc == 0 else "failed", f), "w"), indent=1); os.remove(dst)
    print(time.strftime("%H:%M:%S"), "END", f, "rc", rc, flush=True)
