"""PointWorld + CEM pile units: every ladder setting x the 50 paper scenarios (seed 42) -> <pool>/units/.
  python make_units.py <pool> [--eps 0-49] [--only K4_R1,...]"""
import argparse, sys
sys.path.insert(0, "/data/user_data/mbronars/packages/guided_playback")
from gp.pool import add_units
LADDER = [(4, 1), (8, 1), (16, 1), (32, 1), (32, 2), (32, 3), (64, 3), (128, 3), (192, 3), (256, 3), (384, 3), (512, 3), (768, 3)]
LADDER += [(64, 6), (64, 12), (64, 18), (64, 24), (64, 36)]                     # depth series: population 64, rounds scaled (24 Oct)
ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("--eps", default="0-49"); ap.add_argument("--only", default=None)
ap.add_argument("--nozlim", action="store_true")
ap.add_argument("--variant", default=None, choices=[None, "free4", "se3"])     # free4: no clamps, yaw only; se3: no clamps, roll/pitch/yaw (26 Oct)                               # no paddle height limit (26 Oct rerun)
A = ap.parse_args(); lo, hi = map(int, A.eps.split("-"))
units = []
for K, R in LADDER:
    tag = f"pwp_cem_K{K}_R{R}" + ("_nz" if A.nozlim else "") + (f"_{A.variant}" if A.variant else "")
    var = {"free4": dict(zlim=False, clamp=False), "se3": dict(zlim=False, clamp=False, dof=6)}.get(A.variant, {})
    if A.only and f"K{K}_R{R}" not in A.only.split(","): continue
    for ep in range(lo, hi + 1):
        units.append(dict(id=f"{tag}__ep{ep:03d}", shape="pw_pile", tag=tag, problem=f"ep{ep:03d}", req=dict(ep=ep, K=K, rounds=R, seed=0, **({"zlim": False} if A.nozlim else {}), **var)))
print(add_units(A.root, units), "new units of", len(units))
