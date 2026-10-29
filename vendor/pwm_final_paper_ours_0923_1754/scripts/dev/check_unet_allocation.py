"""Engineering checks on trusted, previously pinned trajectory batches.

Counts/graphs on all supplied files; reduced-width production-wrapper backward
and diffusion sampling on one file per domain. No fitting or accuracy comparison.
Inputs are trusted local TrajectoryBatch pickles, never untrusted downloads.
"""
import argparse
import json
from pathlib import Path

import torch

from diffusion import TrajectoryFlowMatching
from inference import run_diffusion_inference_df
from models.factory import build_particle_model, parse_model_config
from models.unet_std import GraphProvider, build_unet_config, particle_rigidity
from utils.config import load_config
from utils.pose_native import materialize_pose_native
from utils.reproducibility import code_revision, config_for_json, file_sha256
from utils.trajectory_objective import trajectory_objective


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(631)
    config = load_config("configs/unet_adaptive.yaml", ["train.compile=false",
        "model.unet_widths=16,32,64,64", "model.unet_enc_depths=1,1,1,1",
        "model.unet_dec_depths=1,1,1", "model.unet_head_dim=8", "model.unet_t_dim=32",
        "model.unet_fps_random_start=false"])
    model = build_particle_model(config).to(args.device)
    with torch.no_grad():
        for p in model.parameters():
            if torch.count_nonzero(p) == 0:
                p.normal_(0, .025)
    original = build_unet_config(parse_model_config(load_config("configs/unet.yaml").model)[1])
    report = dict(provenance=code_revision(), seed=631, checkpoint=None,
        protocol="Graph invariants for all 16 pinned files; one forward/backward and one "
                 "4-step pyramid diffusion window per domain on reduced-width random weights. "
                 "No optimizer updates or prediction-quality metrics; no corpus frequency estimate.",
        config=config_for_json(config), device=str(args.device), torch_version=str(torch.__version__),
        files=[], model_checks=[], source_sha256={str(p): file_sha256(p) for p in (
            Path(__file__), Path("models/unet_std.py"), Path("utils/pose_native.py"))})
    receipt = args.samples / "geometry_results.json"
    if receipt.exists():
        report["selection_receipt"] = dict(path=str(receipt.resolve()), sha256=file_sha256(receipt))
    seen = set()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    for domain in ("mpm", "pickup", "cloth", "pushing"):
        for index in range(4):
            path = args.samples / f"{domain}_{index}.pt"
            batch = torch.load(path, weights_only=False, map_location="cpu").to(args.device)
            # Pickup stores gripper id 1 before object id 0. Use the same run-order
            # canonicalization as the production forward; do not reorder particles.
            oid, mid = model._object_ids(batch), batch.mat_type
            counts = torch.unique_consecutive(oid[0], return_counts=True)[1].tolist()
            starts = [0] + list(torch.tensor(counts).cumsum(0).tolist())[:-1]
            mats = [int(mid[0, s]) for s in starts]
            with torch.no_grad():
                # Full clean trajectory frames exercise every per-frame graph.
                provider = GraphProvider(model.model.cfg, batch.positions.float(),
                                         batch.positions[:, 0].float(), oid, mid, particle_rigidity(batch))
            for l in range(1, len(provider.sizes)):
                assert torch.equal(provider.ids[l - 1].gather(1, provider.anchor_slots[l - 1]), provider.ids[l])
                for (raw, n), (off, count) in zip(provider.segs[0], provider.segs[l]):
                    ids = provider.ids[l][:, off:off + count]
                    assert ((ids >= raw) & (ids < raw + n)).all()
            row = dict(file=str(path.resolve()), sha256=file_sha256(path), domain=domain,
                       batch_shape=list(batch.positions.shape), material_ids=mats,
                       counts=counts, fixed_l1=original.l1_for_scene(mats, counts),
                       adaptive_levels=provider.sizes, rigid_objects=provider.rigid_objects, graph_valid=True)
            report["files"].append(row)
            print(json.dumps(row), flush=True)
            del provider
            if domain not in seen:
                seen.add(domain)
                model.train()
                model.zero_grad(set_to_none=True)
                torch.manual_seed(631)
                with torch.autocast(torch.device(args.device).type, dtype=torch.bfloat16,
                                    enabled=torch.device(args.device).type == "cuda"):
                    loss = trajectory_objective(model, batch, TrajectoryFlowMatching(), noise_context=False)[0]
                assert torch.isfinite(loss)
                loss.backward()
                grads = [p.grad for p in model.parameters() if p.grad is not None]
                assert grads and all(torch.isfinite(g).all() for g in grads)
                assert any(g.abs().sum() > 0 for g in grads)
                model.zero_grad(set_to_none=True)
                model.eval()
                with torch.no_grad(), torch.autocast(torch.device(args.device).type, dtype=torch.bfloat16,
                                                    enabled=torch.device(args.device).type == "cuda"):
                    sampled = run_diffusion_inference_df(model, batch, TrajectoryFlowMatching(),
                                                        num_inference_steps=4, schedule="pyramid")
                    if isinstance(sampled, tuple):
                        world = materialize_pose_native(batch, *sampled)[0]
                    else:
                        world = batch.predictions_to_positions(batch.prediction_stats.denormalize(sampled))
                assert world.shape == batch.future.shape and torch.isfinite(world).all()
                check = dict(domain=domain, file=str(path.resolve()), backward_finite=True,
                             sample_finite=True, sample_shape=list(world.shape), pyramid_steps=4)
                report["model_checks"].append(check)
                print(json.dumps(check), flush=True)
                del loss, grads, sampled, world
            save()
    report["complete"] = True
    save()


if __name__ == "__main__":
    main()
