"""Claims budget-study cells (cells.json, longest budget first) with an atomic mkdir under <root>/_claims and runs them
one after another against this worker's server. Usage: worker.py --url http://host:port/guide [--replay_gpu g]"""
import argparse, json, os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__)); PY = "/data/user_data/mbronars/.miniconda3/envs/pwm211/bin/python"
ap = argparse.ArgumentParser(); ap.add_argument("--url", required=True); ap.add_argument("--replay_gpu", default="0")
ap.add_argument("--root", default="/data/group_data/katefgroup/runs_pwm_mbronars/guided_playback_budget"); ap.add_argument("--replays", default="3"); ap.add_argument("--cells", default=os.path.join(HERE, "cells.json"))
A = ap.parse_args(); os.makedirs(os.path.join(A.root, "_claims"), exist_ok=True)
JOB = os.environ.get("SLURM_JOB_ID", "nojob")


def claim(cd, cell):
    """atomic mkdir; a claim whose Slurm job is gone (cancelled / timed out) and whose cell has no summary is taken over"""
    try:
        os.mkdir(cd); return True
    except FileExistsError:
        pass
    rec = os.path.join(A.root, "_cells", f"{cell}.json")
    if os.path.exists(rec) and "summary" in json.load(open(rec)):
        return False
    try:
        owner = open(os.path.join(cd, "owner")).read().split()[0]
    except Exception:
        return False                                              # claim being written right now
    if owner == JOB or not owner.isdigit():
        return False
    if subprocess.run(["squeue", "-h", "-j", owner], capture_output=True, text=True).stdout.strip():
        return False                                              # owner still running
    try:
        os.rename(cd, f"{cd}_stale_{owner}")                      # only one worker wins the rename
        os.mkdir(cd); print("TAKEOVER", cell, "from dead job", owner, flush=True); return True
    except OSError:
        return False
for c in json.load(open(A.cells)):
    cell = f"{c['curve']}{c['budget']:03d}"
    cd = os.path.join(A.root, "_claims", cell)
    if not claim(cd, cell):
        continue
    open(os.path.join(cd, "owner"), "w").write(f"{JOB} {os.uname().nodename} {A.url} pid {os.getpid()}\n")
    print("CLAIM", cell, flush=True)
    if c["curve"] == "M":
        cmd = [PY, "-u", os.path.join(HERE, "run_cell_mppi.py"), "--url", A.url, "--budget", str(c["budget"]), "--root", A.root, "--replay_gpu", A.replay_gpu]
    else:
        cmd = [PY, "-u", os.path.join(HERE, "run_cell.py"), "--url", A.url, "--curve", c["curve"], "--budget", str(c["budget"]),
               "--P", str(c["P"]), "--I", str(c["I"]), "--root", A.root, "--replay_gpu", A.replay_gpu, "--replays", A.replays]
    rc = subprocess.call(cmd)
    print("END", cell, "rc", rc, flush=True)
print("WORKER_DONE", flush=True)
