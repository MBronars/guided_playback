"""Timing breakdown for the paper adapter: batch scaling, churn cost, and where a guidance iteration goes.

  python bin/bench.py --ckpt <ckpt> --domain mould [--transitions 150] [--steps 10] [--eager]

Prints, for each configuration, seconds per GENERATE (no grad) and per guidance ITERATION (forward + checkpointed
backward), plus a section profile of one configuration (cuda-synchronised timers around each part of every model
call, so the sum is an upper bound). Population = repeated episodes, the batch IS the population."""
import argparse, os, sys, time, collections, torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from gp.adapters.paper import PaperAdapter
from gp.adapters import paper_core as sc

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True); ap.add_argument("--domain", default="mould")
ap.add_argument("--transitions", type=int, default=150); ap.add_argument("--steps", type=int, default=10)
ap.add_argument("--keep", type=int, default=3); ap.add_argument("--t_freeze", type=float, default=0.8)
ap.add_argument("--eager", action="store_true"); ap.add_argument("--reps", type=int, default=2)
ap.add_argument("--batches", default="1,2,3,4")
ap.add_argument("--compile_mode", default=None, help="torch.compile mode: reduce-overhead | max-autotune-no-cudagraphs")
ap.add_argument("--only", default=None, help="scaling | churn | profile (default: all)")
A = ap.parse_args()
torch.backends.cuda.matmul.allow_tf32 = True
dev = torch.device("cuda")
ad = PaperAdapter().load(A.ckpt, dev, compile=not A.eager and A.compile_mode is None)
if A.compile_mode and not A.eager:
    from utils.compilation import prepare_compilation
    t0 = time.time()
    prepare_compilation(ad.model, {"compile": True, "compile_regional": True, "compile_dynamic": False,
                                   "compile_l0": True, "compile_mode": A.compile_mode})
    ad.mode = f"compiled regional + L0, bf16, mode={A.compile_mode}"
    print(f"compile setup {time.time() - t0:.0f} s", flush=True)
ad.joint_limits = None
print(f"mode: {ad.mode}; domain {A.domain}; N={A.transitions} S={A.steps} keep={A.keep} t_freeze={A.t_freeze}", flush=True)


def time_cfg(B, churn, part_churn, reps):
    batch = ad.episodes([0] * B, A.transitions, domain=A.domain)          # population: one episode, B members
    st = ad.prepare(batch, A.transitions, A.steps, A.keep, 0, churn, A.t_freeze, True)
    if churn and not part_churn and st.info.get("particle_churn"):        # drop the particle table -> pose-only churn
        st.vars = st.vars[:1]; st.free = st.free[:1]; st.shell = st.shell[:1]; st.info.pop("particle_churn")
    out = {}
    def once(grad):
        if grad:
            pos = ad.rollout(batch, st, grad=True); pos[:, -1].sum().backward()
            for v in st.vars:
                v.grad = None
        else:
            with torch.no_grad():                                         # the draws require grad: no_grad or the
                ad.rollout(batch, st, grad=False)                         # whole 104-call graph is kept (OOM)
    for tag, grad in (("generate", False), ("iteration", True)):
        once(grad)                                                        # warm (compile)
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t0 = time.time()
        for _ in range(reps):
            once(grad)
        torch.cuda.synchronize()
        out[tag] = (time.time() - t0) / reps
        out[tag + "_gib"] = torch.cuda.max_memory_allocated() / 2**30
    del batch, st; torch.cuda.empty_cache()
    return out


if A.only in (None, "scaling"):
    print("\n== batch scaling (full churn, particle churn on)")
base = {}
for B in ([int(v) for v in A.batches.split(",")] if A.only in (None, "scaling") else []):
    r = time_cfg(B, True, True, A.reps); base[B] = r
    print(f"  B={B}: generate {r['generate']:6.1f} s ({r['generate']/B:5.1f} s/member)   "
          f"iteration {r['iteration']:6.1f} s ({r['iteration']/B:5.1f} s/member)   peak {r['iteration_gib']:4.1f} GiB", flush=True)
if 1 in base and 3 in base:
    print(f"  -> population of 3: one batch {base[3]['generate']:.1f} s vs three runs of 1 {3*base[1]['generate']:.1f} s (generate); "
          f"{base[3]['iteration']:.1f} s vs {3*base[1]['iteration']:.1f} s (iteration)")

CHURN_CFGS = (("full churn + particle churn", True, True), ("churn, pose only", True, False), ("no churn (Euler carry)", False, False))
if A.only in (None, "churn"):
    print("\n== churn cost at B=3")
    for tag, churn, part in CHURN_CFGS:
        r = time_cfg(3, churn, part, A.reps)
        print(f"  {tag:28s} generate {r['generate']:6.1f} s   iteration {r['iteration']:6.1f} s   peak {r['iteration_gib']:4.1f} GiB", flush=True)

if A.only in (None, 'profile'):
    print("\n== section profile at B=3 (full churn + particle churn), generate only")
    batch = ad.episodes([0] * 3, A.transitions, domain=A.domain)
    st = ad.prepare(batch, A.transitions, A.steps, A.keep, 0, True, A.t_freeze, True)
    T = collections.defaultdict(float)
    S = ad.S
    orig_call, orig_anchor, orig_mat = S._call, S._step_anchor, S._materialize
    def timed(name, fn):
        def wrap(*a, **k):
            torch.cuda.synchronize(); t0 = time.time(); out = fn(*a, **k); torch.cuda.synchronize(); T[name] += time.time() - t0
            return out
        return wrap
    S._call, S._step_anchor, S._materialize = timed("model call", orig_call), timed("anchor step", orig_anchor), timed("materialize", orig_mat)
    with torch.no_grad():
        ad.rollout(batch, st, grad=False)                                     # warm
        torch.cuda.synchronize(); t0 = time.time(); T.clear()
        ad.rollout(batch, st, grad=False)
        torch.cuda.synchronize(); total = time.time() - t0
    S._call, S._step_anchor, S._materialize = orig_call, orig_anchor, orig_mat
    for k, v in sorted(T.items(), key=lambda kv: -kv[1]):
        print(f"  {k:14s} {v:6.2f} s  {v/total*100:5.1f}%")
    print(f"  {'everything else':14s} {total - sum(T.values()):6.2f} s  {(total - sum(T.values()))/total*100:5.1f}%   (total {total:.2f} s)")
