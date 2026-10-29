"""Aggregate one or two paper_eval_push summaries. Headline = SIM CUBE-CENTROID DISTANCE TO TARGET.

  python bin/push_report.py runs/_paper_eval/<tag>_summary.json [<other>_summary.json]

With two summaries it also prints the paired per-scenario comparison (same scenario, both arms)."""
import json, os, sys
import numpy as np


def load(p):
    d = json.load(open(p)); return d, {r["scenario"]: r for r in d["rows"]}


def stats(d, rows):
    r = list(rows.values())
    sim = np.array([x["sim_mm"] for x in r]); tgt = np.array([x["target_mm"] for x in r])
    img = np.array([x["imagined_mm"] for x in r]); ung = np.array([x["unguided_imagined_mm"] for x in r])
    gap = np.array([x["gap_mm"] for x in r]); sec = np.array([x["guide_seconds"] for x in r])
    mov = np.array([x["sim_moved_mm"] for x in r])
    s = d["settings"]
    print(f"== {d['tag']}  {d.get('difficulty','?')} seed {d['seed']}  {len(r)} scenarios")
    print(f"   S={s['steps']} it={s['iterations']} lr={s['lr']}{'' if s['lr_final'] is None else ' -> ' + str(s['lr_final'])}"
          f" t_freeze={s['t_freeze']} members={s['members']}")
    print(f"  SIM DIST TO TARGET  mean {sim.mean():6.1f} mm  median {np.median(sim):6.1f}  min {sim.min():6.1f}  max {sim.max():6.1f}")
    print(f"    within 15 mm {int((sim<=15).sum()):2d}/{len(r)}   within 30 mm {int((sim<=30).sum()):2d}/{len(r)}   within 50 mm {int((sim<=50).sum()):2d}/{len(r)}")
    print(f"  start distance      mean {tgt.mean():6.1f} mm      distance closed mean {(tgt-sim).mean():6.1f} mm ({int((tgt-sim<0).sum())} negative)")
    print(f"  imagined dist       mean {img.mean():6.1f} mm      unguided imagined {ung.mean():6.1f} mm")
    print(f"  sim cube motion     mean {mov.mean():6.1f} mm      sim vs generated gap mean {gap.mean():5.1f} (max {gap.max():5.1f})")
    print(f"  guide seconds       mean {sec.mean():6.1f} s   median {np.median(sec):6.1f}   max {sec.max():6.1f}")
    return sim, tgt


A = [load(p) for p in sys.argv[1:]]
for d, rows in A:
    stats(d, rows); print()
if len(A) == 2:
    (da, ra), (db, rb) = A
    common = sorted(set(ra) & set(rb))
    a = np.array([ra[s]["sim_mm"] for s in common]); b = np.array([rb[s]["sim_mm"] for s in common])
    print(f"== paired on {len(common)} shared scenarios: {da['tag']} vs {db['tag']}")
    print(f"  mean sim dist   {a.mean():6.1f} vs {b.mean():6.1f} mm   (difference {a.mean()-b.mean():+.1f})")
    print(f"  median          {np.median(a):6.1f} vs {np.median(b):6.1f} mm")
    print(f"  per-scenario wins: {da['tag']} {int((a<b).sum())}   {db['tag']} {int((b<a).sum())}   ties {int((a==b).sum())}")
    dif = a - b
    print(f"  paired difference mean {dif.mean():+.1f} mm  std {dif.std(ddof=1):5.1f}  median {np.median(dif):+.1f}")
