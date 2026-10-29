"""Work pool for many short (preemptible) planning jobs + one central Isaac replayer.

Layout under a pool root:
  units/<id>.json        one plan to make: {"id", "shape", "req": guided_run knobs (one problem, one planner setting)}
  claims/<id>/           a worker's claim (atomic mkdir); claims/<id>/hb = "<worker> <unix time>" refreshed every HB s
  plans/<id>/            the run folder written by guided_run (rollout.npz, result.json)
  done/<id>.json         plan finished: imagined distance, request_seconds, picked member, plan folder   (atomic)
  replayed/<id>.json     Isaac result for that plan                                                          (atomic)
  failed/<id>.json       a unit that raised (traceback); not retried automatically
A claim whose heartbeat is older than LEASE seconds and has no done/ file is stale: the next worker renames it away
(atomic: only one stealer wins) and claims the unit again, so preempted work is redone and finished work never is.

  worker:   python -m gp.pool work <root> --shapes <s1,s2,..> [--max_minutes M]     (one GPU, model loaded once)
  replayer: python -m gp.pool replay <root> [--gpu_srun "<srun prefix>"] [--every 120]
  status:   python -m gp.pool status <root>
"""
import math, argparse, glob, json, os, shutil, socket, sys, threading, time, traceback

LEASE, HB = int(os.environ.get("GP_POOL_LEASE", 180)), int(os.environ.get("GP_POOL_HB", 30))
NAN_RETRIES = int(os.environ.get("GP_POOL_NAN_RETRIES", 5))                 # re-plans of a unit after a non-finite rollout before the worker gives up


def _atomic_json(path, obj):
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def dirs(root):
    for d in ("units", "claims", "plans", "done", "replayed", "failed", "stale"):
        os.makedirs(os.path.join(root, d), exist_ok=True)


def add_units(root, units):
    """units: list of {"id", "shape", "req"}; existing ids are left untouched."""
    dirs(root); n = 0
    for u in units:
        p = os.path.join(root, "units", u["id"] + ".json")
        if not os.path.exists(p):
            _atomic_json(p, u); n += 1
    return n


def _hb_age(cdir):
    try:
        return time.time() - float(open(os.path.join(cdir, "hb")).read().split()[1])
    except Exception:
        try:
            return time.time() - os.path.getmtime(cdir)
        except FileNotFoundError:
            return 0.0


def try_claim(root, uid, me):
    if os.path.exists(os.path.join(root, "done", uid + ".json")) or os.path.exists(os.path.join(root, "failed", uid + ".json")):
        return False
    cdir = os.path.join(root, "claims", uid)
    try:
        os.mkdir(cdir)
    except FileExistsError:
        if _hb_age(cdir) < LEASE:
            return False
        try:                                                                 # stale: move it away (only one stealer succeeds)
            os.rename(cdir, os.path.join(root, "stale", f"{uid}.{int(time.time())}.{me}"))
        except (FileNotFoundError, OSError):
            return False
        try:
            os.mkdir(cdir)
        except FileExistsError:
            return False
    open(os.path.join(cdir, "hb"), "w").write(f"{me} {time.time()}")
    return True


class Heartbeat(threading.Thread):
    def __init__(self, cdir, me):
        super().__init__(daemon=True); self.cdir, self.me, self.stop = cdir, me, threading.Event()

    def run(self):
        while not self.stop.wait(HB):
            try:
                open(os.path.join(self.cdir, "hb"), "w").write(f"{self.me} {time.time()}")
            except FileNotFoundError:
                return


def work(root, shapes, max_minutes=None, cache=None, linger=0):
    """Load the model once, warm each shape (untimed), then plan units of those shapes until none are left."""
    import torch
    if cache and os.path.exists(cache):
        torch.compiler.load_cache_artifacts(open(cache, "rb").read()); print(f"[pool] compile cache {cache}", flush=True)
    from gp.session import DEFAULTS, Args, load_adapter, guided_run
    me = f"{socket.gethostname()}:{os.environ.get('SLURM_JOB_ID', 'nojob')}:{os.getpid()}"
    t_end = None if not max_minutes else time.time() + 60 * max_minutes
    units = lambda: sorted(glob.glob(os.path.join(root, "units", "*.json")))
    ov = os.path.join(root, "model_overrides.txt")                         # pool-level model settings beat the job's environment
    if os.path.exists(ov):
        os.environ["PWM_MODEL_OVERRIDES"] = open(ov).read().strip(); print("[pool] model overrides from pool:", os.environ["PWM_MODEL_OVERRIDES"], flush=True)
    ckpt = json.load(open(units()[0]))["req"]["ckpt"]
    ad = load_adapter(Args(DEFAULTS, adapter="paper", ckpt=ckpt, eager=False, joint_limits=None))
    pw = os.path.join(root, "prewarm.json")                                   # one untimed plan of this request before any unit (see 26 Oct note)
    if os.path.exists(pw):
        W = json.load(open(pw)); W.update(name="_prewarm", runs=os.path.join(root, "warm"))
        guided_run(ad, Args(DEFAULTS, **W)); print("[pool] prewarm plan done:", W.get("desc"), flush=True)
    warmed = set(); last_work = time.time(); nan_streak = 0; skip_until = {}
    while True:
        if t_end and time.time() > t_end - 60:
            print("[pool] out of time", flush=True); return
        todo = [(J["shape"], u) for u in units() if not os.path.exists(os.path.join(root, "done", os.path.basename(u)))
                and not os.path.exists(os.path.join(root, "failed", os.path.basename(u)))
                for J in [json.load(open(u))] if J["req"].get("ckpt", ckpt) == ckpt]                    # failed units are not retried; only units for the checkpoint this worker loaded
        todo = [(sh, u) for sh, u in todo if skip_until.get(os.path.basename(u)[:-5], 0) < time.time()]
        todo = [u for sh, u in sorted(todo, key=lambda x: (x[0] not in warmed, x[1])) if "all" in shapes or sh in shapes]   # stay on warmed shapes first
        if not todo and any(t > time.time() for t in skip_until.values()):   # only released NaN units left: wait them out
            time.sleep(30); continue
        if not todo:
            if linger and time.time() - last_work < 60 * linger:                 # closed loop: the next round's units arrive later
                time.sleep(15); continue
            print("[pool] nothing left for", shapes, flush=True); return
        claimed = None
        for u in todo:
            uid = os.path.basename(u)[:-5]
            if try_claim(root, uid, me):
                claimed = (uid, json.load(open(u))); break
        if claimed is None:
            time.sleep(20); continue
        uid, U = claimed; cdir = os.path.join(root, "claims", uid); hb = Heartbeat(cdir, me); hb.start()
        try:
            if U["shape"] not in warmed:                                     # one untimed plan of this shape: compiles / loads graphs
                W = dict(U["req"]); W.update(name=f"_warm_{U['shape']}", runs=os.path.join(root, "warm"))
                guided_run(ad, Args(DEFAULTS, **W)); warmed.add(U["shape"])
            import gp.cem as _cem; rej0 = _cem.REJECTED[0]
            R = dict(U["req"]); R.update(name=uid, runs=os.path.join(root, "plans_tmp", me.replace(":", "_")))
            for attempt in range(NAN_RETRIES + 1):                           # an occasional non-finite candidate: plan the unit again in this process
                try:
                    out, res = guided_run(ad, Args(DEFAULTS, **R))
                    if not any(math.isfinite(float(v)) for v in res["best_mm"]):  # guide path (DNO) returns inf instead of raising (27 Oct: whole pool-32 batches NaN on babel-s5-28)
                        shutil.rmtree(out, ignore_errors=True); raise _cem.NaNRollout(f"every member non-finite (best_mm {res['best_mm'][:4]})")
                    break
                except Exception as e:
                    if type(e).__name__ != "NaNRollout" or attempt == NAN_RETRIES: raise
                    os.makedirs(os.path.join(root, "nan_retries"), exist_ok=True)
                    _atomic_json(os.path.join(root, "nan_retries", f"{uid}.{me.replace(':', '_')}.{attempt}.json"), dict(worker=me, unit=uid, attempt=attempt, error=str(e), at=time.time()))
                    print(f"[pool] {uid}: non-finite rollouts ({e}); planning it again (attempt {attempt + 2})", flush=True)
            final = os.path.join(root, "plans", uid)
            if os.path.exists(os.path.join(root, "done", uid + ".json")):    # another worker finished this unit first (lease race, 27 Oct):
                shutil.rmtree(out, ignore_errors=True)                         # keep its plan, which may already be replayed
                print(f"[pool] {uid}: already done by another worker; discarded this plan", flush=True); continue
            if os.path.exists(final): shutil.rmtree(final)
            os.replace(out, final)
            rj = json.load(open(os.path.join(final, "result.json")))
            pick = min(range(len(res["best_mm"])), key=lambda i: res["best_mm"][i])
            _atomic_json(os.path.join(root, "done", uid + ".json"), dict(id=uid, worker=me, member=pick, imagined_mm=res["best_mm"][pick],
                         request_seconds=rj.get("request_seconds"), plan=final, finished=time.time(), gpu=torch.cuda.get_device_name(0),
                         rejected_nonfinite=_cem.REJECTED[0] - rej0))
            print(f"[pool] {uid}: {res['best_mm'][pick]:.1f} mm imagined, {rj.get('request_seconds')} s", flush=True); last_work = time.time(); nan_streak = 0
        except Exception as e:
            if type(e).__name__ == "NaNRollout":                              # still non-finite after the retries: GPU / process fault, give the unit back, stop this worker
                os.makedirs(os.path.join(root, "bad_workers"), exist_ok=True)
                _atomic_json(os.path.join(root, "bad_workers", me.replace(":", "_") + ".json"), dict(worker=me, unit=uid, error=str(e), gpu=torch.cuda.get_device_name(0)))
                hb.stop.set(); shutil.rmtree(cdir, ignore_errors=True)
                nan_streak += 1
                if nan_streak >= 3:
                    print(f"[pool] {uid}: non-finite rollouts ({e}); {nan_streak} units in a row, exiting this worker", flush=True)
                    sys.exit(3)
                skip_until[uid] = time.time() + 600                              # let another worker (or this one later) take it
                print(f"[pool] {uid}: non-finite rollouts ({e}) after {NAN_RETRIES} re-plans; released, moving on", flush=True)
                continue
            tb = traceback.format_exc()
            if "out of memory" in tb or "OutOfMemoryError" in tb or "CUDA error" in tb or "CUBLAS_STATUS" in tb:   # the CUDA context / allocator is suspect now
                gd = os.path.join(root, "gpu_errors"); os.makedirs(gd, exist_ok=True)
                _atomic_json(os.path.join(gd, f"{uid}.{me.replace(':', '_')}.json"), dict(id=uid, worker=me, error=tb[-3000:], at=time.time()))
                seen = {f.split(".", 1)[1] for f in os.listdir(gd) if f.startswith(uid + ".")}
                if len(seen) >= 2:                                           # the same unit broke two workers: it is the unit, not the worker
                    _atomic_json(os.path.join(root, "failed", uid + ".json"), dict(id=uid, worker=me, error=tb))
                hb.stop.set(); shutil.rmtree(cdir, ignore_errors=True)
                print(f"[pool] {uid}: GPU error ({tb.strip().splitlines()[-1][:120]}); releasing it and exiting this worker", flush=True)
                sys.exit(4)
            _atomic_json(os.path.join(root, "failed", uid + ".json"), dict(id=uid, worker=me, error=tb))
            print(f"[pool] {uid} FAILED", flush=True)
        finally:
            hb.stop.set(); shutil.rmtree(cdir, ignore_errors=True)


def _stack(root, ids, out):
    """Stack the picked member of every plan into one run folder (one Isaac launch)."""
    import numpy as np
    rows, arrs, other = [], {}, {}
    for uid in ids:
        d = json.load(open(os.path.join(root, "done", uid + ".json"))); m = d["member"]
        z = np.load(os.path.join(d["plan"], "rollout.npz"))
        for k in z.files:
            v = z[k]
            if v.ndim >= 1 and v.shape[0] == z["predicted"].shape[0] and k not in ("trace_mm",):
                arrs.setdefault(k, []).append(v[m])
            elif k not in ("trace_mm", "step"):
                other[k] = v                                                     # per-run arrays (e.g. boxes): last plan's
        rows.append(dict(id=uid, member=m, imagined_mm=d["imagined_mm"]))
        rj = json.load(open(os.path.join(d["plan"], "result.json")))
    os.makedirs(out, exist_ok=True)
    np.savez(os.path.join(out, "rollout.npz"), **other, **{k: np.stack(v) for k, v in arrs.items()}, trace_mm=np.zeros((1, len(ids)), np.float32), step=np.array(rj.get("step", 0)))
    rj.update(episodes=[r["id"] for r in rows], best_mm=[r["imagined_mm"] for r in rows], unguided_mm=[r["imagined_mm"] for r in rows], pool_rows=rows)
    json.dump(rj, open(os.path.join(out, "result.json"), "w"), indent=1)


def replay(root, gpu_srun="", every=120, batch_max=64, once=False):
    """Central replayer: every `every` s, stack all planned-but-unreplayed units (same point count) and replay them in one
    Isaac launch (GP_ISAAC_EXTRA passes the physics settings), then write replayed/<id>.json per unit."""
    import numpy as np, subprocess
    py = sys.executable; here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    while True:
        done = sorted(os.path.basename(p)[:-5] for p in glob.glob(os.path.join(root, "done", "*.json")))
        todo = [u for u in done if not os.path.exists(os.path.join(root, "replayed", u + ".json"))]
        groups = {}
        for u in todo:
            d = json.load(open(os.path.join(root, "done", u + ".json")))
            n = int(np.load(os.path.join(d["plan"], "rollout.npz"))["predicted"].shape[2]); groups.setdefault(n, []).append(u)
        for n, ids in groups.items():
            for s in range(0, len(ids), batch_max):
                chunk = ids[s:s + batch_max]; out = os.path.join(root, "replay_batches", f"b{int(time.time())}_{n}_{s}")
                _stack(root, chunk, out)
                cmd = f"{gpu_srun} {py} -u {here}/bin/replay.py --run {out} --sim isaac --variants guided --gpu 0"
                print(f"[replay] {len(chunk)} plans -> {out}", flush=True)
                rc = subprocess.run(["bash", "-c", cmd], cwd=here, stdout=open(out + ".log", "w"), stderr=subprocess.STDOUT).returncode
                try:
                    rows = json.load(open(os.path.join(out, "replay_isaac", "analysis.json")))["rows"]
                    for r in rows:
                        uid = chunk[r["member"]]
                        _atomic_json(os.path.join(root, "replayed", uid + ".json"), dict(id=uid, batch=out, **{k: r[k] for k in r if k != "member"}))
                    print(f"[replay] wrote {len(rows)} results", flush=True)
                except Exception as e:
                    print(f"[replay] batch failed rc={rc}: {e!r}", flush=True)
        if once:
            return
        time.sleep(every)


def status(root):
    c = lambda d: len(glob.glob(os.path.join(root, d, "*.json")))
    live = [p for p in glob.glob(os.path.join(root, "claims", "*")) if _hb_age(p) < LEASE]
    print(f"units {c('units')} | claimed now {len(live)} | planned {c('done')} | replayed {c('replayed')} | failed {c('failed')} | stale claims moved {len(os.listdir(os.path.join(root, 'stale')))}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=("work", "replay", "status")); ap.add_argument("root")
    ap.add_argument("--shapes", default=""); ap.add_argument("--max_minutes", type=float, default=None); ap.add_argument("--cache", default=None); ap.add_argument("--linger", type=float, default=0, help="minutes to keep waiting for new units after the last one (closed loop)")
    ap.add_argument("--gpu_srun", default=""); ap.add_argument("--every", type=int, default=120); ap.add_argument("--once", action="store_true")
    a = ap.parse_args(); dirs(a.root)
    if a.cmd == "work": work(a.root, a.shapes.split(","), a.max_minutes, a.cache, a.linger)
    elif a.cmd == "replay": replay(a.root, a.gpu_srun, a.every, once=a.once)
    else: status(a.root)
