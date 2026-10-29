"""Paired, unguided sampler check on the four pinned phantom-motion fixtures.

Use the frozen source and input receipt from phantom_regeneration_2026-09-14.
The baseline records each window's RNG state; other schedules replay those
states. This preserves the saved baseline and pairs later-window initial noise
despite the random FPS draws consumed by different numbers of model forwards.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time


ARMS = (("pyramid25", "pyramid", 25), ("pyramid50", "pyramid", 50),
        ("joint36", "full_sequence", 36), ("joint61", "full_sequence", 61))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--collections", default="pick,place,convex,upright")
    args = parser.parse_args()
    reference, outdir = args.reference.resolve(), args.out.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    source = reference / "source"
    sys.path.insert(0, str(source))
    import numpy as np
    import torch
    import inference
    from omegaconf import OmegaConf
    from diffusion.scheduling import generate_scheduling_matrix
    from utils.reproducibility import file_sha256, seed_for_device

    def save_json(path, value):
        path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")

    def tensor_hash(value):
        array = value.detach().contiguous().cpu().numpy()
        return hashlib.sha256(array.tobytes()).hexdigest()

    manifest = json.loads((reference / "source_manifest.json").read_text())

    def verify_source():
        for name, digest in manifest.items():
            assert file_sha256(source / name) == digest, name

    verify_source()
    inputs = json.loads((reference / "inputs.json").read_text())
    checkpoint = inputs["original"]["checkpoint"]
    assert file_sha256(checkpoint) == inputs["checkpoint_sha256"]
    device = torch.device("cuda")
    model, diffusion, cfg, _ = inference.load_model_and_diffusion(checkpoint, device)
    assert cfg.model.unet_bidirectional_temporal and cfg.model.pose_native
    assert cfg.model.articulated_bodies and not cfg.model.unet_actions
    assert cfg.train.action_dropout == 1 and cfg.train.noise_context
    assert cfg.model.unet_fps_random_start and cfg.flow_matching.prediction_type == "velocity"
    assert not model.training and next(model.parameters()).dtype == torch.float32
    assert (cfg.data.context_len, cfg.data.future_len) == (4, 16)
    print("Loaded pinned bidirectional EMA checkpoint, FP32,", torch.cuda.get_device_name(), flush=True)
    sampler = inference.run_diffusion_inference_df

    for clip in inputs["clips"]:
        name = clip["collection"]
        if name not in args.collections.split(","):
            continue
        assert file_sha256(reference / f"{name}_input.pt") == clip["input_sha256"]
        one = torch.load(reference / f"{name}_input.pt", weights_only=False).to(device)
        batch = one._map_tensors(lambda v: v.repeat(8, *([1] * (v.dim() - 1))))
        old = np.load(reference / f"{name}.npz")
        assert one.positions.shape == (1, 40, 1183, 3)
        assert np.array_equal(one.positions[0].cpu().numpy(), old["gt"])
        states, baseline_noise = [], []
        receipts = []

        for arm, schedule, steps in ARMS:
            assert not (outdir / f"{name}_{arm}.npz").exists(), "Use a fresh output directory"
            seed_for_device(clip["seed"], device)
            windows = []

            def paired_sampler(model, win, diffusion, **kwargs):
                index = len(windows)
                if arm == "pyramid25":
                    states.append(dict(cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(device)))
                else:
                    torch.set_rng_state(states[index]["cpu"])
                    torch.cuda.set_rng_state(states[index]["cuda"], device)
                assert not win.pose_action_mask.any() and not win.pose_actions.any()
                row = dict(length=win.num_future_frames, calls=0)

                def observe(module, model_args, model_kwargs):
                    if row["calls"] == 0:
                        row["particle_noise_sha256"] = tensor_hash(model_args[1])
                        row["pose_noise_sha256"] = tensor_hash(model_kwargs["noisy_pose"])
                        assert not model_kwargs["context_timesteps"].any()
                        expected = win.delta_stats.normalize(win.context_deltas)
                        assert torch.equal(expected, model_kwargs["noisy_context"])
                    row["calls"] += 1

                hook = model.register_forward_pre_hook(observe, with_kwargs=True)
                try:
                    result = sampler(model, win, diffusion, **kwargs)
                finally:
                    hook.remove()
                matrix = generate_scheduling_matrix(win.num_future_frames, steps, schedule, 1.)
                mapped = np.clip((matrix.astype(np.float64) / steps * diffusion.num_train_timesteps)
                                 .astype(np.int64), 0, diffusion.num_train_timesteps - 1)
                expected_calls = int(np.any(np.diff(mapped, axis=0), axis=1).sum())
                assert row["calls"] == expected_calls
                noise = {k: row[k] for k in ("particle_noise_sha256", "pose_noise_sha256")}
                if arm == "pyramid25":
                    baseline_noise.append(noise)
                else:
                    assert noise == baseline_noise[index], "Initial window noise was not paired"
                windows.append(row)
                return result

            inference.run_diffusion_inference_df = paired_sampler
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize()
            start = time.monotonic()
            try:
                result = inference.windowed_rollout(model, diffusion, batch, horizon=36, future_len=16,
                    num_inference_steps=steps, schedule=schedule, uncertainty_scale=1.,
                    hide_given_actions=True, clamp_actuated=False)
            finally:
                inference.run_diffusion_inference_df = sampler
            torch.cuda.synchronize()
            seconds = time.monotonic() - start
            full = result["positions"].cpu().numpy()
            assert np.isfinite(full).all() and full.shape == (8, 40, 1183, 3)
            assert result["windows"] == [(4, 16, 0), (20, 16, 0), (36, 4, 0)]
            assert np.array_equal(full[:, :4], np.broadcast_to(old["gt"][None, :4], full[:, :4].shape))
            if arm == "pyramid25":
                assert np.array_equal(full, old["samples"]), "Pinned baseline failed bitwise reproduction"
                for key in ("R", "T", "rigid_idx"):
                    assert np.array_equal(result[key].cpu().numpy(), old[key]), key
                torch.save(states, outdir / f"{name}_window_rng.pt")
            record = dict(collection=name, arm=arm, schedule=schedule, steps=steps,
                seed=clip["seed"], seconds=seconds, calls=sum(w["calls"] for w in windows),
                max_allocated_bytes=torch.cuda.max_memory_allocated(device), windows=windows,
                initial_noise_paired=True, clean_context_verified=True, actions_hidden_verified=True,
                reference_bitwise_equal=bool(np.array_equal(full, old["samples"])))
            path = outdir / f"{name}_{arm}.npz"
            np.savez_compressed(path, gt=old["gt"], samples=full, action_mask=old["action_mask"],
                object_ids=old["object_ids"], context_len=4, particle_body=old["particle_body"],
                R=result["R"].cpu().numpy(), T=result["T"].cpu().numpy(),
                rigid_idx=result["rigid_idx"].cpu().numpy())
            record["samples_sha256"] = file_sha256(path)
            save_json(outdir / f"{name}_{arm}_receipt.json", record)
            receipts.append(record)
            print(json.dumps(record), flush=True)

        verify_source()
        save_json(outdir / f"{name}_experiment.json", dict(
            checkpoint=checkpoint, checkpoint_sha256=inputs["checkpoint_sha256"],
            source=str(source), source_manifest_sha256=file_sha256(reference / "source_manifest.json"),
            harness_sha256=file_sha256(__file__), input=clip,
            reference_samples_sha256=file_sha256(reference / f"{name}.npz"),
            normalization=OmegaConf.to_container(cfg.normalization, resolve=True),
            weights="EMA", precision="float32; autocast disabled", guidance=False,
            model=dict(bidirectional=True, random_fps=True, training_noise_context=True,
                       inference_context="clean, timestep 0", action_dropout=1.),
            runtime=dict(torch=torch.__version__, cuda=torch.version.cuda, python=platform.python_version(),
                         node=platform.node(), gpu=torch.cuda.get_device_name(),
                         allow_tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
                         float32_matmul_precision=torch.get_float32_matmul_precision()),
            pairing="Replay baseline CPU/CUDA RNG state immediately before each window sampler; "
                    "verify particle and masked-pose noise hashes; retain random FPS. "
                    "Extra forwards consume additional FPS draws within the window only.",
            timing="Wall time of an 8-draw batch rollout including first-call noise hashes; "
                   "model load and disk output excluded; baseline may include cold-start overhead. "
                   "Matched forward counts do not imply matched FLOPs with a short last window.",
            arms=receipts))


if __name__ == "__main__":
    main()
