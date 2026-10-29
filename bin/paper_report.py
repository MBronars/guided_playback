"""Aggregate a paper_eval summary. Headline = DISTANCE CLOSED: how far the dough centroid was
actually moved toward the target, target_mm - sim_mm (frame-0 distance to target IS target_mm,
since the target is the frame-0 centroid plus the scenario offset).

  python bin/paper_report.py [--summary runs/_paper_eval/paper42_summary.json]
"""
import argparse, json, os
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--summary", default=os.path.join(HERE, "runs", "_paper_eval", "paper42_summary.json"))
A = ap.parse_args()
d = json.load(open(A.summary)); rows = d["rows"]
tgt = np.array([r["target_mm"] for r in rows]); sim = np.array([r["sim_mm"] for r in rows])
img = np.array([r["imagined_mm"] for r in rows]); ung = np.array([r["unguided_imagined_mm"] for r in rows])
gap = np.array([r["gap_mm"] for r in rows]); sec = np.array([r["guide_seconds"] for r in rows])
rep = np.array([r["replay_seconds"] for r in rows]); mov = np.array([r["sim_moved_mm"] for r in rows])
closed, frac = tgt - sim, (tgt - sim) / tgt
icl = tgt - img
print(f"== {d['tag']} seed {d['seed']}: {len(rows)} scenarios  {d['settings']}")
print(f"  DISTANCE CLOSED (sim)   mean {closed.mean():6.1f} mm   median {np.median(closed):6.1f}   "
      f"min {closed.min():6.1f}   max {closed.max():6.1f}   negative {int((closed<0).sum())}")
print(f"  fraction of target closed  mean {frac.mean()*100:5.1f} %   median {np.median(frac)*100:5.1f} %")
print(f"  distance closed (imagined) mean {icl.mean():6.1f} mm      unguided imagined closed {(tgt-ung).mean():6.1f} mm")
print(f"  target distance         mean {tgt.mean():6.1f} mm   sim centroid to target mean {sim.mean():6.1f} mm")
print(f"  sim dough motion        mean {mov.mean():6.1f} mm      sim vs generated gap mean {gap.mean():5.1f} (max {gap.max():5.1f})")
print(f"  guide seconds (warm)    mean {sec.mean():6.1f} s    median {np.median(sec):6.1f}   min {sec.min():6.1f}   max {sec.max():6.1f}")
print(f"  replay seconds          mean {rep.mean():6.1f} s")
