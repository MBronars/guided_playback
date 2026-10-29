"""Build and validate an explicit production recipe for frozen collections.

Writes a new YAML without changing the three canonical backbone recipes. Paths,
domain selection and normalization are mandatory inputs; no live staging glob is
used by training. The same recipe works with train.py and the overfit harness.
"""
import argparse
from pathlib import Path

from omegaconf import OmegaConf

from data.indexed_episodes import evaluation_stratum, load_index
from data.unified_dataset import INDEXED_DOMAINS, RL_PUSH_EXCLUSIONS, RL_PUSH_VAL_ASSETS, selected_domains
from utils.config import DEFAULTS, resolve_config


# Training window policy decided 2026-09-17 on the release's train folders
# (scratchpad/pwm_data_v1_build/receipts/window_motion.json): p(start) ∝ (mean future step of the
# predicted particles + floor)^alpha. These values take the share of draws on quiet windows from
# 47 / 44 / 77% (fresh / drag / folding) to 22 / 22 / 42% and draw no window beyond about 4x its
# uniform rate. Yikai, dex and rl_push stay uniform; MPM batches mix scenes of equal point count.
WINDOW_MOTION = {"mpm_fresh": dict(alpha=1.0, floor_mm=2.0), "mpm_drag": dict(alpha=1.0, floor_mm=0.25),
                 "folding": dict(alpha=1.0, floor_mm=0.02)}
BATCH_ACROSS_EPISODES = ("mpm_fresh", "mpm_drag")
# The UNet's regional compile is static-shape (train.compile_dynamic=false): every distinct point count compiles
# again. Datasets with a handful of counts keep it (pushing 256, rl_push and pickup 1,183, MPM 1,024 per object). Yikai
# scenes, folding garments and dex hands bring hundreds of counts: the 2026-09-18 smoke trainings ran the UNet on
# yikai at 0.08 it/s compiled against 0.60 it/s eager, and the 20K pilots of 2026-09-17 ran eager.
FEW_POINT_COUNTS = ("pushing", "rl_push", "mpm_fresh", "mpm_drag", "pickup")
# UNet sizes of the pwm-data-v1 launch plan (decided 2026-09-18): specialists a little above 100M parameters, the unified
# seven-dataset model about 500M. A preset changes widths and the bottleneck depth only; strides, head dimension, pooling
# and registers stay those of configs/unet.yaml. A width must be a multiple of the head dimension (64), so `500m` widens
# levels 1 to 3 by 1.25 and keeps level 0 at 128. tests/test_collection_contract.py pins the exact parameter counts.
UNET_SIZES = {"331m": {},                                                              # configs/unet.yaml as written
              "100m": dict(unet_widths="64,128,256,512", unet_enc_depths="1,2,4,9"),   # half width, one more bottleneck block
              "500m": dict(unet_widths="128,320,640,1280")}


def validation_folder(indices, domain):
    """`<indices>/<domain>/val` when the layout has one, else the release layout's `test` folder."""
    for name in ("val", "test"):
        folder = Path(indices).resolve() / domain / name
        if (folder / "INDEX.json").is_file():
            return folder
    raise FileNotFoundError(f"{domain}: no val or test INDEX.json under {Path(indices).resolve() / domain}")


def shard_exclusions(indices, folder, domain):
    """Episode exclusion lists a release ships for a shard dataset: `<folder>/manifests/<domain>_exclusions_train*.json`
    for the train folder and `..._exclusions_test*.json` for the validation folder (data/blocklists.py format; pushing's
    failed simulations, 2026-09-19). Empty when the release has none, so older layouts give the same recipe text."""
    manifests = Path(indices).resolve() / folder / "manifests"
    found = {split: sorted(str(p) for p in manifests.glob(f"{domain}_exclusions_{name}*.json"))
             for split, name in (("train", "train"), ("val", "test"))}
    return {split: files for split, files in found.items() if files}


def declared_shift(indices, folder):
    """`canonical_translation_m` of `<indices>/<folder>/RELEASE.json`, or None: a shard dataset folder may state the
    translation its loader must apply (pwm-data-v1/pickup: the registered -0.10 m leaves the table top at +10 mm)."""
    path = Path(indices).resolve() / folder / "RELEASE.json"
    if not path.is_file():
        return None
    import json
    value = json.loads(path.read_text()).get("canonical_translation_m")
    return None if value is None else [float(v) for v in value]


def rl_push_roots(indices, rl_push_release=None):
    """A release layout (`<indices>/rl_push/{train,test}`, pwm-data-v1) is read whole per folder with every
    episode exclusion list it ships in `rl_push/manifests/`: the folders already are the object split. A
    single-folder release (rl-push-lowdim-20260914) is split at run time by the registered object list."""
    layout = Path(indices).resolve() / "rl_push"
    if (layout / "train").is_dir():
        if rl_push_release is not None:
            raise ValueError(f"{layout} is the rl_push release; drop --rl-push-release")
        lists = sorted(str(p) for p in (layout / "manifests").glob("rl_push_exclusions*.json"))
        if not lists:
            raise FileNotFoundError(f"{layout / 'manifests'}: no rl_push_exclusions*.json")
        return dict(train=str(layout / "train"), val=str(validation_folder(indices, "rl_push")),
                    exclusions={s: list(lists) for s in ("train", "val")})
    if rl_push_release is None:
        raise ValueError("An explicit frozen rl_push release is required")
    return dict(train=str(Path(rl_push_release).resolve()),
                exclusions={s: [str(Path(p).resolve()) for p in (RL_PUSH_EXCLUSIONS, RL_PUSH_VAL_ASSETS)] for s in ("train", "val")})


def collection_recipe(architecture, domains, indices, normalization, *, batch_size=2,
                      rl_push_release=None, run_name="collection_verification", folders=None, unet_size="331m",
                      light_context_noise_std=0., prediction_type=None, ema_decay=None, timestep_schedule=None,
                      loss_spike_threshold=None, val_batches=None, sampler_batches=None,
                      light_pose_noise_std=None, light_pose_noise_anchor=False, light_pose_noise_target="restore",
                      light_pose_noise_type="iid"):
    """Protocol options (decided per experiment, 2026-09-19); each default leaves the recipe text unchanged.
    light_context_noise_std: Gaussian jitter on the normalized deformable context deltas, training only (0.05 in every
        earlier MPM production run). prediction_type: `sample` trains the particle head in x0 space with a loss flat in t
        and scores rigid poses the same way (train.pose_loss_space=sample: without it the pose error is divided by the
        noise level, which rejected 82% of the steps of an x0 pushing run); None keeps the backbone recipe's velocity target. ema_decay: written together with train.ema_warmup=true, because a
        long average without warm-up keeps the initial weights for tens of thousands of steps. timestep_schedule: a
        {name: weight} mixture of the training noise-level patterns of the future frames (diffusion/timestep_schedules.py);
        each sample draws one. The launch plan uses independent 0.6 + ramp 0.4: the ramp is the staircase our pyramid sampler
        visits, and the share follows RNR-DP (Chen et al. 2025, p_linear = 0.4; either pattern alone loses 20 to 27 points of
        task success) and Streaming Diffusion Policy (Hoeg et al. 2024, Push-T: independent 0.85, ramp only 0.36, 67% + 33%
        0.87). ramp_slope and ramp_jitter are not written: their defaults follow the run's own sampler.
    loss_spike_threshold: train.loss_spike_threshold, the multiple of a rank's running loss above which a step is rejected
        (trainer default 10). The rule was built for the velocity loss, which is about equally large on every batch. The
        plain x0 loss grows with the motion in a batch, so the rule rejects motion-rich valid batches, more of them as the
        running loss falls (yikai, 2026-09-19: 2% of the steps after step 1,000, among them ordinary carried-object
        motion). `inf` switches the rule off; the non-finite guard, the gradient-norm gate and gradient clipping are
        separate and stay.
    val_batches, sampler_batches: log.val_batches and log.sampler_batches, the validation batches read for the loss and
        for the sampler at every validation (default: the backbone recipe's 40 and 8, raised to cover every stratum). A
        batch is not a fixed number of windows: an rl_push validation batch holds the windows of ONE object, so in
        pwm-data-v1's 200-episode rl_push test folder (one or two episodes per object, 2026-09-19) it holds one or two
        windows where the complete folder gave 16; its recipe therefore reads all 117 batches for the loss and 60 for
        the sampler (about 100 windows over 60 objects, the window count the sampler scored before). Each must still be a
        multiple of the number of domains and cover the strata, as the automatic values do.
    light_pose_noise_std (with light_pose_noise_anchor, _target, _type): train.light_pose_noise_*, light noise on the
        RIGID bodies' history as whole-body pose error, training only (utils/pose_native.jitter_rigid_history). It is
        not light_context_noise_std above: that one jitters the deformable particles' context deltas and leaves every
        rigid row clean (measured: exactly 0 mm on an all-rigid dataset), so without this option a rigid dataset
        trains on exact history. std is in normalized pose-delta units. anchor=true also moves the passive bodies'
        last context pose; target=restore then patches the first future transition so the model has to return to
        the recorded trajectory, target=carry keeps the recorded increments from the moved pose; type=iid draws each
        history transition independently, random_walk accumulates the draws as GNS does. The pickup comparison of
        2026-09-16/17 used std 0.05, anchor, restore, iid under the velocity objective (36-frame rollout at 200K
        steps: last frame 6.89 against 7.24 mm, mean 4.48 against 3.86 mm, one seed). None writes the usual
        light_pose_noise_std: 0.0 and nothing else."""
    if unet_size not in UNET_SIZES:
        raise ValueError(f"unet_size must be one of {sorted(UNET_SIZES)}, got {unet_size!r}")
    if architecture != "unet" and unet_size != "331m":
        raise ValueError("unet_size selects a UNet preset; the transformer recipe has one size")
    domains = selected_domains(domains)[0]
    base = OmegaConf.load(Path(__file__).resolve().parents[2] / "configs" / f"{architecture}.yaml")
    # A dataset lives in the layout folder of its domain name unless `folders` names a variant folder
    # (pwm-data-v1: folding_8mm for folding). Indexed domains always come from the layout; a legacy shard
    # domain (pushing) does when the layout has its folder, so a run never falls back to the registered corpus.
    folder = {d: (folders or {}).get(d, d) for d in domains}
    unknown = set(folders or {}) - set(domains)
    if unknown:
        raise ValueError(f"--folder names domains that are not selected: {sorted(unknown)}")
    roots = {d: dict(train=str(Path(indices).resolve() / folder[d] / "train"), val=str(validation_folder(indices, folder[d])))
             for d in domains if d in INDEXED_DOMAINS or (d != "rl_push" and (Path(indices) / folder[d] / "train").is_dir())}
    for d in roots:
        if d not in INDEXED_DOMAINS and shard_exclusions(indices, folder[d], d):
            roots[d]["exclusions"] = shard_exclusions(indices, folder[d], d)
        if d not in INDEXED_DOMAINS and declared_shift(indices, folder[d]) is not None:
            roots[d]["shift_m"] = declared_shift(indices, folder[d])
    if "rl_push" in domains:
        roots["rl_push"] = rl_push_roots(indices, rl_push_release)
    data = dict(domains=domains, roots=roots, context_len=4, future_len=16, batch_size=batch_size, num_workers=4)
    if any(d in WINDOW_MOTION for d in domains):
        data["window_motion"] = {d: dict(WINDOW_MOTION[d]) for d in domains if d in WINDOW_MOTION}
    if any(d in BATCH_ACROSS_EPISODES for d in domains):
        data["batch_across_episodes"] = [d for d in domains if d in BATCH_ACROSS_EPISODES]
    overrides = dict(run_name=run_name, data=data,
        normalization=dict(profile="xy_p99_per_domain", artifact=str(Path(normalization).resolve())),
        model=dict(articulated_bodies=True),
        train=dict(noise_context=False, light_context_noise_std=float(light_context_noise_std), light_pose_noise_std=0.,
                   clip_deltas=False, clip_pose_deltas=False, action_dropout=0., stop_after_step=0))
    if prediction_type is not None:
        if prediction_type not in ("velocity", "sample"):
            raise ValueError(f"prediction_type must be velocity or sample, got {prediction_type!r}")
        overrides["flow_matching"] = dict(prediction_type=prediction_type)
        if prediction_type == "sample":
            overrides["train"]["pose_loss_space"] = "sample"
    if ema_decay is not None:
        overrides["train"].update(ema_decay=float(ema_decay), ema_warmup=True)
    if loss_spike_threshold is not None:
        if not float(loss_spike_threshold) > 0:
            raise ValueError(f"loss_spike_threshold must be positive (inf switches the rule off), got {loss_spike_threshold!r}")
        overrides["train"]["loss_spike_threshold"] = float(loss_spike_threshold)
    if light_pose_noise_std is not None:
        if isinstance(light_pose_noise_std, bool) or not float(light_pose_noise_std) > 0:
            raise ValueError(f"light_pose_noise_std must be positive, got {light_pose_noise_std!r}")
        # All four keys are written, so the recipe states the whole mechanism; resolve_config checks the combination.
        overrides["train"].update(light_pose_noise_std=float(light_pose_noise_std),
                                  light_pose_noise_anchor=bool(light_pose_noise_anchor),
                                  light_pose_noise_target=str(light_pose_noise_target),
                                  light_pose_noise_type=str(light_pose_noise_type))
    if timestep_schedule is not None:
        from diffusion.timestep_schedules import resolve_mixture
        resolve_mixture(dict(timestep_schedule))                     # unknown names and bad weights fail here
        overrides["train"]["timestep_schedule"] = {str(k): float(v) for k, v in dict(timestep_schedule).items()}
    # Strict round-robin is balanced only after a whole number of domain cycles.
    # The ordered prefix must also cover every Dex teacher / folding action.
    # Small specialist or historical four-domain defaults stay unchanged when
    # they already cover their strata. This balances checkpoint selection domains.
    count = len(domains)
    strata = max([1] + [len({evaluation_stratum(row, domain)
                            for row in load_index(roots[domain]["val"], domain, ("val", "test"))["records"]})
                        for domain in domains if domain in INDEXED_DOMAINS])
    overrides["log"] = {name: ((max(count * strata,
                                   int(base.get("log", {}).get(name, DEFAULTS["log"][name])))
                               + count - 1) // count) * count
                        for name in ("val_batches", "sampler_batches")}
    for name, value in (("val_batches", val_batches), ("sampler_batches", sampler_batches)):
        if value is None:
            continue
        if isinstance(value, bool) or int(value) != value or int(value) < count * strata or int(value) % count:
            raise ValueError(f"{name} must be a whole multiple of the {count} domain(s) that covers the {strata} validation "
                             f"strata (at least {count * strata}), got {value!r}")
        overrides["log"][name] = int(value)
    if architecture == "unet" and "mpm_drag" in domains:
        overrides["model"]["unet_actions"] = True
    overrides["model"].update(UNET_SIZES[unet_size])
    if architecture == "unet" and any(d not in FEW_POINT_COUNTS for d in domains):
        overrides["train"]["compile"] = False
    recipe = OmegaConf.merge(base, OmegaConf.create(overrides))
    resolve_config(recipe)  # Fail before saving or allocating a GPU.
    return recipe


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--architecture", choices=["unet", "transformer"], required=True)
    p.add_argument("--domains", required=True)
    p.add_argument("--indices", required=True)
    p.add_argument("--normalization", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--run-name", required=True)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--rl-push-release", help="a single-folder rl_push release split by the registered object list; "
                   "not needed when --indices holds rl_push/train")
    p.add_argument("--unet-size", choices=sorted(UNET_SIZES), default="331m",
                   help="UNet size preset of the pwm-data-v1 launch plan: 100m for a specialist, 500m for the unified model")
    p.add_argument("--light-context-noise", type=float, default=0., metavar="STD",
                   help="train.light_context_noise_std: jitter on the normalized deformable context deltas (0.05 in the MPM runs)")
    p.add_argument("--prediction-type", choices=["velocity", "sample"], default=None,
                   help="flow_matching.prediction_type; sample = x0 prediction with a loss flat in t (default: the backbone recipe's)")
    p.add_argument("--ema-decay", type=float, default=None,
                   help="train.ema_decay, written with train.ema_warmup=true (default: the backbone recipe's 0.99, no warm-up)")
    p.add_argument("--timestep-schedule", default=None, metavar="NAME=WEIGHT[,NAME=WEIGHT]",
                   help="train.timestep_schedule mixture, e.g. independent=0.6,ramp=0.4 (default: independent noise levels per frame)")
    p.add_argument("--loss-spike-threshold", type=float, default=None, metavar="X",
                   help="train.loss_spike_threshold: reject a step whose loss exceeds X times the running loss (trainer default 10); "
                        "inf switches the rule off, which an x0 recipe on mostly still data needs (see collection_recipe)")
    p.add_argument("--val-batches", type=int, default=None, metavar="N",
                   help="log.val_batches: validation batches read for the loss (default: 40, raised to cover every stratum)")
    p.add_argument("--sampler-batches", type=int, default=None, metavar="N",
                   help="log.sampler_batches: validation batches sampled (default: 8, raised to cover every stratum); an rl_push "
                        "batch holds one object's windows, so a small test folder needs more batches for the same window count")
    p.add_argument("--light-pose-noise", type=float, default=None, metavar="STD",
                   help="train.light_pose_noise_std: light noise on the rigid bodies' history as whole-body pose error "
                        "(--light-context-noise leaves rigid rows clean); see collection_recipe for the three options below")
    p.add_argument("--light-pose-noise-anchor", action="store_true",
                   help="train.light_pose_noise_anchor: also move the passive bodies' last context pose")
    p.add_argument("--light-pose-noise-target", choices=["restore", "carry"], default="restore",
                   help="train.light_pose_noise_target: restore returns to the recorded trajectory, carry keeps the recorded increments")
    p.add_argument("--light-pose-noise-type", choices=["iid", "random_walk"], default="iid",
                   help="train.light_pose_noise_type: independent draws per history transition, or a GNS-style random walk")
    p.add_argument("--folder", action="append", default=[], metavar="DOMAIN=NAME",
                   help="read a domain from a variant folder of the layout, e.g. folding=folding_8mm")
    a = p.parse_args()
    folders = dict(item.split("=", 1) for item in a.folder)
    schedule = None if a.timestep_schedule is None else {
        name: float(weight) for name, weight in (item.split("=", 1) for item in a.timestep_schedule.split(","))}
    recipe = collection_recipe(a.architecture, a.domains.split(","), a.indices, a.normalization,
        batch_size=a.batch_size, rl_push_release=a.rl_push_release, run_name=a.run_name, folders=folders or None,
        unet_size=a.unet_size, light_context_noise_std=a.light_context_noise, prediction_type=a.prediction_type,
        ema_decay=a.ema_decay, timestep_schedule=schedule, loss_spike_threshold=a.loss_spike_threshold,
        val_batches=a.val_batches, sampler_batches=a.sampler_batches,
        light_pose_noise_std=a.light_pose_noise, light_pose_noise_anchor=a.light_pose_noise_anchor,
        light_pose_noise_target=a.light_pose_noise_target, light_pose_noise_type=a.light_pose_noise_type)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as f:
        f.write(OmegaConf.to_yaml(recipe))
    print(out)


if __name__ == "__main__":
    main()
