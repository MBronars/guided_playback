"""One row per guidance arm, restricted to the scenarios ALL listed arms share. Headline = mean SIM distance to target.
  python bin/sweep_table.py runs/_paper_eval/<a>_summary.json runs/_paper_eval/<b>_summary.json ...  [--scenarios 6,27,...]"""
import json, sys, numpy as np
args = sys.argv[1:]; want = None
if "--scenarios" in args:
    i = args.index("--scenarios"); want = set(int(v) for v in args[i + 1].split(",")); del args[i:i + 2]
runs = []
for p in args:
    d = json.load(open(p)); rows = {r["scenario"]: r for r in d["rows"]}
    runs.append((d, rows))
shared = set.intersection(*[set(r) for _, r in runs]) if runs else set()
if want: shared &= want
S = sorted(shared)
print(f"{len(S)} shared scenarios: {S}\n")
print(f"{'arm':26s} {'S':>2s} {'it':>3s} {'lr':>22s} {'tf':>4s} | {'sim mean':>8s} {'median':>7s} {'<=30':>4s} {'<=50':>4s} {'neg':>3s} | {'imag':>6s} {'gap':>6s} | {'s/scn':>5s}")
for d, rows in runs:
    s = d["settings"]; r = [rows[k] for k in S]
    sim = np.array([x["sim_mm"] for x in r]); tgt = np.array([x["target_mm"] for x in r])
    img = np.array([x["imagined_mm"] for x in r]); gap = np.array([x["gap_mm"] for x in r]); sec = np.array([x["guide_seconds"] for x in r])
    lr = f"{s['lr']}" + ("" if s["lr_final"] is None else f"->{s['lr_final']}")
    if not s.get("shell", True): lr += " noshell"
    if s.get("optimizer", "unitgrad") != "unitgrad": lr += " " + s["optimizer"]
    print(f"{d['tag']:26s} {s['steps']:2d} {s['iterations']:3d} {lr:>22s} {s['t_freeze']:4.1f} | {sim.mean():8.1f} {np.median(sim):7.1f} {(sim<=30).sum():4d} {(sim<=50).sum():4d} {(tgt-sim<0).sum():3d} | {img.mean():6.1f} {gap.mean():6.1f} | {sec.mean():5.1f}")
