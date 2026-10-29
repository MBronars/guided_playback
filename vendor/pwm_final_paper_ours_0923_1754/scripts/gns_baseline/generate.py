"""Roll out a trained GNS baseline checkpoint. Inference only.

Needs torch and numpy -- nothing else, and no training code. The model files in
this folder are copies of the GNS implementation with its two torch_geometric
imports pointed at pyg_shim.py.

Two ways to give it a scene:

  # an episode straight out of the pwm-data-v1 mpm_fresh test release
  python generate.py --checkpoint mpm_fresh.pt --metadata metadata.json \
      --mpm_fresh_test .../releases/pwm-data-v1/mpm_fresh/test --case 0 --out out.npz

  # any scene of your own: an .npz with positions [T, N, 3] in metres and
  # particle_type [N] (+ material_property [N, K] if the checkpoint uses it)
  python generate.py --checkpoint mpm_fresh.pt --metadata metadata.json \
      --input scene.npz --out out.npz

The first 6 frames are context; everything after is predicted autoregressively.
With --mpm_fresh_test the ground truth is carried through to `out.npz` and the
per-frame error is printed, so a run is self-checking.

`metadata.json` ships next to each checkpoint and holds the normalization the
model was trained with. Nothing here refits it.
"""
import argparse, glob, io, json, os, pickle, sys, tarfile
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import learned_simulator                                          # noqa: E402

CONTEXT_LEN = 6                   # INPUT_SEQUENCE_LENGTH the models were trained with
NUM_PARTICLE_TYPES = 9
MPM_MAT_TYPE_COL = 3              # material_properties column holding mat_type


def build_simulator(metadata, checkpoint, device, noise_std=6.7e-4):
    stats = {
        "acceleration": {
            "mean": torch.FloatTensor(metadata["acc_mean"]).to(device),
            "std": torch.sqrt(torch.FloatTensor(metadata["acc_std"]) ** 2 + noise_std ** 2).to(device)},
        "velocity": {
            "mean": torch.FloatTensor(metadata["vel_mean"]).to(device),
            "std": torch.sqrt(torch.FloatTensor(metadata["vel_std"]) ** 2 + noise_std ** 2).to(device)},
    }
    nnode_in = metadata.get("nnode_in", 37 if metadata["dim"] == 3 else 30)
    nedge_in = metadata.get("nedge_in", metadata["dim"] + 1)
    sim = learned_simulator.LearnedSimulator(
        particle_dimensions=metadata["dim"], nnode_in=nnode_in, nedge_in=nedge_in,
        latent_dim=128, nmessage_passing_steps=10, nmlp_layers=2, mlp_hidden_dim=128,
        connectivity_radius=metadata["default_connectivity_radius"],
        boundaries=np.array(metadata["bounds"]), normalization_stats=stats,
        nparticle_types=NUM_PARTICLE_TYPES, particle_type_embedding_size=16,
        boundary_clamp_limit=metadata.get("boundary_augment", 1.0), device=device)
    state = torch.load(checkpoint, map_location="cpu")
    sim.load_state_dict(state["state_dict"] if "state_dict" in state else state)
    sim.to(device).eval()
    return sim


def mpm_material_vector(mat_raw, metadata):
    """The per-particle material channels the mpm_fresh checkpoints take.

    metadata.json records exactly how training built them under `material_norm`:
    which raw columns, the pre-transform (log10 on Young's modulus, which spans
    two decades), and the train-set mean/std used to z-score. Applying anything
    else here would feed the network numbers on a different scale than it saw.
    Returns None for checkpoints trained without material channels.
    """
    norm = metadata.get("material_norm")
    if norm is None:
        return None
    cols = np.asarray(mat_raw[:, norm["source_columns"]], dtype=np.float64)
    for i, how in enumerate(norm["pre_transform"]):
        if how == "log10":
            cols[:, i] = np.log10(np.maximum(cols[:, i], 1e-12))
        elif how != "identity":
            raise SystemExit(f"unknown pre_transform {how!r} in metadata material_norm")
    return ((cols - np.asarray(norm["mean"])) / np.asarray(norm["std"])).astype(np.float32)


def load_mpm_fresh_case(test_dir, case, metadata=None):
    """One episode from the release test shards: tarfile + pickle, no deps."""
    shards = sorted(glob.glob(os.path.join(test_dir, "**", "*.tar"), recursive=True))
    seen = 0
    for sp in shards:
        with tarfile.open(sp, "r") as tar:
            for s in sorted(set(x.split(".", 1)[0] for x in tar.getnames())):
                try:
                    pos = pickle.load(io.BytesIO(tar.extractfile(f"{s}.particle_pos.pyd").read()))
                    mat = pickle.load(io.BytesIO(tar.extractfile(f"{s}.material_properties.pyd").read()))
                    val = pickle.load(io.BytesIO(tar.extractfile(f"{s}.valid_particles.pyd").read()))
                except KeyError:
                    continue
                for b in range(pos.shape[0]):
                    if seen == case:
                        keep = val[b].all(axis=0) if not val[b].all() else np.ones(pos.shape[2], bool)
                        raw = mat[b][keep]
                        material = mpm_material_vector(raw, metadata) if metadata else None
                        return (pos[b][:, keep].astype(np.float32),
                                raw[:, MPM_MAT_TYPE_COL].astype(np.int64), material)
                    seen += 1
    raise SystemExit(f"case {case} not found ({seen} episodes in {test_dir})")


def rollout(sim, positions, ptype, material, n_steps, device):
    ctx = torch.from_numpy(np.ascontiguousarray(
        np.transpose(positions[:CONTEXT_LEN], (1, 0, 2)))).to(device).float().contiguous()
    pt = torch.from_numpy(np.asarray(ptype)).to(device).long()
    mp = torch.from_numpy(np.asarray(material)).to(device).float() if material is not None else None
    npp = torch.tensor([ctx.shape[0]], dtype=torch.long, device=device)
    out = []
    with torch.no_grad():
        for _ in range(n_steps):
            nxt = sim.predict_positions(ctx, npp, pt, material_property=mp)
            out.append(nxt.cpu().numpy())
            ctx = torch.cat([ctx[:, 1:], nxt.unsqueeze(1)], dim=1).contiguous()
    return np.stack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--metadata", required=True, help="metadata.json shipped with the checkpoint")
    ap.add_argument("--mpm_fresh_test", help="pwm-data-v1/mpm_fresh/test")
    ap.add_argument("--case", type=int, default=0)
    ap.add_argument("--input", help=".npz with positions [T,N,3] (+ particle_type, material_property)")
    ap.add_argument("--steps", type=int, default=None, help="default: to the end of the episode")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    if bool(a.mpm_fresh_test) == bool(a.input):
        ap.error("give exactly one of --mpm_fresh_test or --input")
    metadata = json.load(open(a.metadata))
    if a.mpm_fresh_test:
        positions, ptype, material = load_mpm_fresh_case(a.mpm_fresh_test, a.case, metadata)
    else:
        z = np.load(a.input, allow_pickle=True)
        positions = z["positions"].astype(np.float32)
        ptype = z["particle_type"] if "particle_type" in z else np.zeros(positions.shape[1], np.int64)
        material = z["material_property"] if "material_property" in z else None
    if positions.shape[0] <= CONTEXT_LEN:
        raise SystemExit(f"need more than {CONTEXT_LEN} frames, got {positions.shape[0]}")

    if "nnode_in" in metadata and metadata["nnode_in"] > 37 and material is None:
        raise SystemExit(f"this checkpoint expects a material vector (nnode_in={metadata['nnode_in']}); "
                         "supply material_property in the input npz")
    device = torch.device(a.device)
    sim = build_simulator(metadata, a.checkpoint, device)

    n_steps = a.steps if a.steps else positions.shape[0] - CONTEXT_LEN
    print(f"{positions.shape[1]} particles, {CONTEXT_LEN} context frames, {n_steps} predicted")
    pred = rollout(sim, positions, ptype, material, n_steps, device)

    save = dict(predicted=pred, particle_type=np.asarray(ptype),
                context=positions[:CONTEXT_LEN], context_len=CONTEXT_LEN)
    gt = positions[CONTEXT_LEN:CONTEXT_LEN + n_steps]
    if gt.shape[0] == pred.shape[0]:
        err = np.linalg.norm(pred - gt, axis=-1) * 1000.0
        save["ground_truth"] = gt
        save["per_frame_mm"] = err.mean(axis=1)
        print(f"error vs ground truth: mean {err.mean():.2f} mm, final frame {err[-1].mean():.2f} mm")
    np.savez(a.out, **save)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    raise SystemExit(main())
