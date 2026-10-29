"""Check the torch_geometric-free copy against the original implementation.

Runs the SAME checkpoint on the SAME scene through both and compares. Needs a
torch_geometric install, so it is a developer check, not something a user of
generate.py has to run. Also reports whether radius_graph's max_num_neighbors
cap is ever reached, the one place the two could legitimately differ.
"""
import argparse, json, os, sys
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--metadata", required=True)
ap.add_argument("--mpm_fresh_test", required=True)
ap.add_argument("--cases", type=int, nargs="+", default=[0, 1, 2])
ap.add_argument("--steps", type=int, default=12)
ap.add_argument("--gns_repo", default="/data/user_data/mbronars/packages/gns-pytorch")
a = ap.parse_args()

sys.path.insert(0, HERE)
import generate                                                    # noqa: E402
metadata = json.load(open(a.metadata))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- reference: the original torch_geometric implementation ---
sys.path.insert(0, a.gns_repo)
from gns import train as gns_train                                 # noqa: E402
ref = gns_train._get_simulator(metadata, 6.7e-4, 6.7e-4, device)
state = torch.load(a.checkpoint, map_location="cpu")
ref.load_state_dict(state["state_dict"] if "state_dict" in state else state)
ref.to(device).eval()

mine = generate.build_simulator(metadata, a.checkpoint, device)

# A multi-step rollout is chaotic and CUDA reductions are not bit-reproducible, so
# the reference does not even reproduce ITSELF over many steps. The meaningful test
# is therefore the SINGLE step, with the reference's own rerun as the yardstick for
# what any multi-step number can mean.
print(f"{'case':>5} {'particles':>10} {'max nbrs':>9} {'1 step':>12} {'ref vs self':>12} "
      f"{str(a.steps)+' steps':>12} {'ref vs self':>12}")
worst = 0.0
for c in a.cases:
    pos, pt, mat = generate.load_mpm_fresh_case(a.mpm_fresh_test, c, metadata)
    # does the 128-neighbour cap ever bind on this scene?
    x = torch.from_numpy(pos[generate.CONTEXT_LEN - 1]).to(device).float()
    d = torch.cdist(x, x)
    deg = int(((d < metadata["default_connectivity_radius"]).sum(1) - 1).max())
    def roll(sim, n):
        return generate.rollout(sim, pos, pt, mat, n, device)

    one_mine, one_ref, one_ref2 = roll(mine, 1), roll(ref, 1), roll(ref, 1)
    many_mine, many_ref, many_ref2 = roll(mine, a.steps), roll(ref, a.steps), roll(ref, a.steps)
    d1 = float(np.abs(one_mine - one_ref).max()); s1 = float(np.abs(one_ref - one_ref2).max())
    dn = float(np.abs(many_mine - many_ref).max()); sn = float(np.abs(many_ref - many_ref2).max())
    worst = max(worst, d1)
    print(f"{c:>5} {pos.shape[1]:>10,} {deg:>9} {d1:>12.2e} {s1:>12.2e} {dn:>12.2e} {sn:>12.2e}")

print(f"\nworst SINGLE-step difference: {worst:.3e} m = {worst*1e6:.3f} micrometres")
print("float32 round-off on metre-scale positions is ~6e-8 m. The multi-step columns are")
print("only meaningful next to 'ref vs self': if they match, the divergence is the model's")
print("own chaos under non-deterministic CUDA reductions, not this implementation.")
