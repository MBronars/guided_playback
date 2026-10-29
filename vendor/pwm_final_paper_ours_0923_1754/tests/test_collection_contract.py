"""Scientific configuration and continuation contracts for opt-in collections."""
import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from data.indexed_episodes import INDEX_SCHEMA
from scripts.unified.collection_recipe import collection_recipe
from utils.checkpoint import corpus_identity, validate_resume_configuration
from utils.config import load_config, resolve_config


def recipe(tmp_path, domain="mpm_fresh", architecture="transformer", validation="val", **builder_options):
    artifact = json.loads(Path("stats/normalization_rl_push.json").read_text())
    scales = artifact["profiles"]["xy_p99_per_domain"]
    scales["particle"][domain] = [.01, .02]
    if domain in ("folding", "yikai", "dex"):
        scales["pose"][domain] = [.01, .02, .03, .04]
    artifact_path = tmp_path / "normalization.json"
    artifact_path.write_text(json.dumps(artifact))
    for split in ("train", validation):
        root = tmp_path / domain / split
        root.mkdir(parents=True)
        source = root / "source.npz"
        source.write_bytes(b"index identity fixture, never loaded as a trajectory")
        stat = source.stat()
        row = dict(file=str(source), size=stat.st_size, mtime_ns=stat.st_mtime_ns,
                   episode_id=split, group=split, frames=49)
        (root / "INDEX.json").write_text(json.dumps(dict(
            schema=INDEX_SCHEMA, domain=domain, split=split, records=[row])))
    return collection_recipe(architecture, [domain], tmp_path, artifact_path, **builder_options)


@pytest.mark.parametrize("domain", ["mpm_fresh", "mpm_drag", "folding", "yikai", "dex"])
@pytest.mark.parametrize("architecture", ["unet", "transformer"])
def test_collection_recipe_is_explicit_and_unclipped(tmp_path, domain, architecture):
    cfg = resolve_config(recipe(tmp_path, domain, architecture))
    assert list(cfg.data.domains) == [domain]
    assert not cfg.train.clip_deltas and not cfg.train.clip_pose_deltas
    assert not cfg.train.noise_context and cfg.train.light_context_noise_std == 0
    assert cfg.train.light_pose_noise_std == 0
    if domain == "mpm_drag" and architecture == "unet":
        assert cfg.model.unet_actions and cfg.train.action_dropout == 0


SIZE_LINES = {"331m": {}, "100m": dict(unet_widths="64,128,256,512", unet_enc_depths="1,2,4,9"), "500m": dict(unet_widths="128,320,640,1280")}


@pytest.mark.parametrize("unet_size,domain,parameters", [
    ("331m", "mpm_fresh", 331_628_269), ("100m", "mpm_fresh", 103_096_585), ("100m", "mpm_drag", 103_096_841),
    ("500m", "mpm_drag", 502_218_277)])
def test_unet_size_presets_change_only_the_size_lines(tmp_path, unet_size, domain, parameters):
    """The launch plan of pwm-data-v1: specialists a little above 100M parameters, the unified model about 500M.
    A preset differs from the recipe of configs/unet.yaml in its size lines and nowhere else; the counts are exact
    (mpm_drag adds the four drag action channels at the stem)."""
    import torch
    from models.factory import build_particle_model
    for name in ("base", "sized"):
        (tmp_path / name).mkdir()
    base, sized = recipe(tmp_path / "base", domain, "unet"), recipe(tmp_path / "sized", domain, "unet", unet_size=unet_size)
    a, b = OmegaConf.to_container(base.model), OmegaConf.to_container(sized.model)
    assert {key: b[key] for key in b if a.get(key) != b[key]} == SIZE_LINES[unet_size] and set(a) == set(b)
    for block in ("train", "log", "normalization"):
        if block == "normalization":   # the two recipes read their own copy of the same fixture artifact
            assert base.normalization.profile == sized.normalization.profile
        else:
            assert OmegaConf.to_container(base[block]) == OmegaConf.to_container(sized[block])
    with torch.device("meta"):
        model = build_particle_model(resolve_config(sized))
    assert sum(p.numel() for p in model.parameters()) == parameters


def test_unet_size_is_a_unet_option_with_known_names(tmp_path):
    for name in ("transformer", "unknown"):
        (tmp_path / name).mkdir()
    with pytest.raises(ValueError, match="unet_size selects a UNet preset"):
        recipe(tmp_path / "transformer", architecture="transformer", unet_size="100m")
    with pytest.raises(ValueError, match="unet_size must be one of"):
        recipe(tmp_path / "unknown", architecture="unet", unet_size="1b")


def test_protocol_options_change_only_their_lines(tmp_path):
    # Light context noise, x0 prediction and a long warmed-up EMA (decided per experiment, 2026-09-19). The defaults
    # write nothing new, so every recipe built before the options existed is reproduced unchanged.
    for name in ("default", "chosen", "velocity"):
        (tmp_path / name).mkdir()
    default = recipe(tmp_path / "default", architecture="unet")
    chosen = recipe(tmp_path / "chosen", architecture="unet", batch_size=8, light_context_noise_std=0.05,
                    prediction_type="sample", ema_decay=0.9999, timestep_schedule=dict(independent=0.6, ramp=0.4))
    assert "flow_matching" not in default and "ema_warmup" not in default.train
    assert default.train.light_context_noise_std == 0. and default.data.batch_size == 2
    assert chosen.flow_matching.prediction_type == "sample" and chosen.train.pose_loss_space == "sample"
    assert "pose_loss_space" not in default.train and "timestep_schedule" not in default.train
    assert dict(chosen.train.timestep_schedule) == dict(independent=0.6, ramp=0.4)
    assert "ramp_slope" not in chosen.train and "ramp_jitter" not in chosen.train     # they follow the run's sampler
    explicit_velocity = recipe(tmp_path / "velocity", architecture="unet", prediction_type="velocity")
    assert "pose_loss_space" not in explicit_velocity.train
    assert (chosen.train.light_context_noise_std, chosen.train.ema_decay, chosen.train.ema_warmup) == (0.05, 0.9999, True)
    assert chosen.data.batch_size == 8 and not chosen.train.clip_deltas and not chosen.train.noise_context
    resolved = resolve_config(chosen)
    assert resolved.flow_matching.prediction_type == "sample" and resolved.train.ema_warmup is True
    plain = lambda cfg: OmegaConf.to_container(cfg, resolve=True)
    a, b = plain(default), plain(chosen)
    b.pop("flow_matching")
    for key in ("light_context_noise_std", "ema_decay", "ema_warmup", "pose_loss_space", "timestep_schedule"):
        a["train"].pop(key, None), b["train"].pop(key, None)
    a["data"].pop("batch_size"), b["data"].pop("batch_size")
    for cfg in (a, b):   # the two recipes read their own temporary folders
        cfg["data"].pop("roots"), cfg["normalization"].pop("artifact")
    assert a == b
    # The loss spike rule: absent by default, `inf` switches it off (an x0 recipe on mostly still data rejects motion-rich batches).
    assert "loss_spike_threshold" not in default.train and "loss_spike_threshold" not in chosen.train
    (tmp_path / "gate").mkdir()
    gate_off = recipe(tmp_path / "gate", architecture="unet", prediction_type="sample", loss_spike_threshold=float("inf"))
    assert gate_off.train.loss_spike_threshold == float("inf")
    reloaded = OmegaConf.create(OmegaConf.to_yaml(gate_off))                     # the recipe file round-trips the infinity
    assert resolve_config(reloaded).train.loss_spike_threshold == float("inf")
    with pytest.raises(ValueError, match="loss_spike_threshold must be positive"):
        (tmp_path / "badgate").mkdir()
        recipe(tmp_path / "badgate", architecture="unet", loss_spike_threshold=0)
    with pytest.raises(ValueError, match="Unknown timestep_schedule"):
        (tmp_path / "badschedule").mkdir()
        recipe(tmp_path / "badschedule", architecture="unet", timestep_schedule=dict(staircase=1.0))
    with pytest.raises(ValueError, match="prediction_type must be velocity or sample"):
        (tmp_path / "bad").mkdir()
        recipe(tmp_path / "bad", architecture="unet", prediction_type="epsilon")
    # Validation batch counts: automatic by default, explicit when a batch holds few windows (rl_push: one object per batch).
    (tmp_path / "batches").mkdir()
    more = recipe(tmp_path / "batches", architecture="unet", val_batches=117, sampler_batches=60)
    assert (more.log.val_batches, more.log.sampler_batches) == (117, 60)
    assert (default.log.val_batches, default.log.sampler_batches) != (117, 60)
    c = plain(more)
    for cfg in (c,):
        cfg["data"].pop("roots"), cfg["normalization"].pop("artifact")
    d = plain(default)
    d["data"].pop("roots"), d["normalization"].pop("artifact")
    c["log"].pop("val_batches"), c["log"].pop("sampler_batches"), d["log"].pop("val_batches"), d["log"].pop("sampler_batches")
    assert c == d                                                                  # nothing else moves
    for bad in (0, 2.5, True):
        with pytest.raises(ValueError, match="sampler_batches must be a whole multiple"):
            (tmp_path / f"badbatches{bad}").mkdir()
            recipe(tmp_path / f"badbatches{bad}", architecture="unet", sampler_batches=bad)
    # Rigid history noise (train.light_pose_noise_*): absent by default; the option writes its four keys and nothing else.
    (tmp_path / "posenoise").mkdir()
    noisy = recipe(tmp_path / "posenoise", architecture="unet", light_pose_noise_std=0.05, light_pose_noise_anchor=True)
    assert default.train.light_pose_noise_std == 0. and "light_pose_noise_anchor" not in default.train
    assert (noisy.train.light_pose_noise_std, noisy.train.light_pose_noise_anchor,
            noisy.train.light_pose_noise_target, noisy.train.light_pose_noise_type) == (0.05, True, "restore", "iid")
    e, f = plain(noisy), plain(default)
    for cfg in (e, f):
        cfg["data"].pop("roots"), cfg["normalization"].pop("artifact")
        for key in ("light_pose_noise_std", "light_pose_noise_anchor", "light_pose_noise_target", "light_pose_noise_type"):
            cfg["train"].pop(key, None)
    assert e == f
    with pytest.raises(ValueError, match="light_pose_noise_target must be restore, or carry with anchor=true"):
        (tmp_path / "badcarry").mkdir()
        recipe(tmp_path / "badcarry", architecture="unet", light_pose_noise_std=0.05, light_pose_noise_target="carry")
    with pytest.raises(ValueError, match="light_pose_noise_std must be positive"):
        (tmp_path / "badstd").mkdir()
        recipe(tmp_path / "badstd", architecture="unet", light_pose_noise_std=0)


@pytest.mark.parametrize("split,validation", [("train", "val"), ("val", "val"), ("train", "test"), ("test", "test")])
def test_resume_rejects_in_place_index_change(tmp_path, split, validation):
    # `validation` is the folder the recipe reads as its validation root: a release's test folder since 2026-09-17.
    cfg = resolve_config(recipe(tmp_path, validation=validation))
    corpus = corpus_identity(cfg)
    assert str(tmp_path / "mpm_fresh" / validation / "INDEX.json") in corpus["val"]["mpm_fresh"]
    state = dict(cfg=OmegaConf.to_container(cfg, resolve=True), corpus=corpus)
    validate_resume_configuration(state, cfg)
    path = tmp_path / "mpm_fresh" / split / "INDEX.json"
    value = json.loads(path.read_text())
    value["records"][0]["episode_id"] += "-different-episode"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="corpus files differ"):
        validate_resume_configuration(state, cfg)


def test_rigid_collection_does_not_block_deformable_floor_auxiliary(tmp_path):
    cfg = recipe(tmp_path, "yikai")
    cfg.train.aux_losses = dict(floor=1., floor_z=0.)
    # The objective masks rigid particles out of this auxiliary. A Yikai scene
    # therefore cannot make the MPM/cloth floor plane incorrect in a mixed run.
    # This option is not a contact constraint for Yikai's variable tabletops.
    resolved = resolve_config(cfg)
    assert resolved.train.aux_losses.floor_z == 0.


def test_new_domains_require_explicit_indices():
    with pytest.raises(ValueError, match="explicit"):
        load_config("configs/transformer.yaml", ["data.domains=[dex]"])


def test_drag_recipe_cannot_drop_or_omit_commands(tmp_path):
    cfg = recipe(tmp_path, "mpm_drag", "unet")
    cfg.model.unet_actions = False
    with pytest.raises(ValueError, match="unet_actions"):
        resolve_config(cfg)


def test_evaluation_receipt_pins_episode_membership(tmp_path):
    from data.indexed_episodes import load_index
    from scripts.unified.eval_unified import evaluation_dataset, _input_provenance
    cfg = resolve_config(recipe(tmp_path))
    ds = evaluation_dataset(cfg, "mpm_fresh", 4, 16)
    receipt = _input_provenance(ds, cfg.normalization)
    expected = load_index(tmp_path / "mpm_fresh" / "val")["index_sha256"]
    assert receipt["episode_index_sha256"] == expected
    assert receipt["episodes_in_index"] == 1
    assert receipt["roots_source"] == "checkpoint data.roots"


def _report_folder(tmp_path, name, episode, split="test", domain="mpm_fresh"):
    root = tmp_path / domain / name
    root.mkdir(parents=True)
    source = root / "source.npz"
    source.write_bytes(b"index identity fixture, never loaded as a trajectory")
    stat = source.stat()
    row = dict(file=str(source), size=stat.st_size, mtime_ns=stat.st_mtime_ns, episode_id=episode, group=episode, frames=49)
    (root / "INDEX.json").write_text(json.dumps(dict(schema=INDEX_SCHEMA, domain=domain, split=split, records=[row])))
    return root


def test_report_folder_replaces_the_validation_folder_for_one_evaluation(tmp_path):
    """pwm-data-v1/mpm_fresh/test_seen: a test folder no recipe reads, scored with --test_folder."""
    from data.indexed_episodes import load_index
    from scripts.unified.eval_unified import evaluation_dataset, _input_provenance
    cfg = resolve_config(recipe(tmp_path, validation="test"))
    report = _report_folder(tmp_path, "test_seen", "seen-episode")
    ds = evaluation_dataset(cfg, "mpm_fresh", 4, 16, test_folders={"mpm_fresh": str(report)})
    receipt = _input_provenance(ds, cfg.normalization, command_line=True)
    assert receipt["episode_index_sha256"] == load_index(report)["index_sha256"]
    assert receipt["episode_index_sha256"] != load_index(tmp_path / "mpm_fresh" / "test")["index_sha256"]
    assert receipt["roots_source"] == "command line" and receipt["roots"] == [str(report.resolve())]
    assert [r["episode_id"] for r in ds.records] == ["seen-episode"]
    # the checkpoint's own validation folder is what the next plain call reads
    assert [r["episode_id"] for r in evaluation_dataset(cfg, "mpm_fresh", 4, 16).records] == ["test"]


def test_report_folder_must_be_held_out_and_of_a_trained_indexed_dataset(tmp_path):
    from scripts.unified.eval_unified import evaluation_dataset
    cfg = resolve_config(recipe(tmp_path, validation="test"))
    leaking = _report_folder(tmp_path, "test_leak", "train")                 # the train folder's episode
    with pytest.raises(ValueError, match="overlap"):
        evaluation_dataset(cfg, "mpm_fresh", 4, 16, test_folders={"mpm_fresh": str(leaking)})
    training_split = _report_folder(tmp_path, "test_as_train", "another", split="train")
    with pytest.raises(ValueError, match="split"):
        evaluation_dataset(cfg, "mpm_fresh", 4, 16, test_folders={"mpm_fresh": str(training_split)})
    with pytest.raises(ValueError, match="own data.roots"):
        evaluation_dataset(cfg, "mpm_drag", 4, 16, test_folders={"mpm_drag": str(leaking)})


def test_explicit_indexed_evaluation_checks_normalization(tmp_path):
    from data.indexed_episodes import IndexedEpisodeDataset
    from scripts.unified.eval_unified import evaluation_dataset, _check_explicit_dataset
    cfg = resolve_config(recipe(tmp_path))
    ds = evaluation_dataset(cfg, "mpm_fresh", 4, 16)
    assert ds.normalization["sha256"] == cfg.normalization.sha256
    _check_explicit_dataset(ds, cfg, 4)
    different = IndexedEpisodeDataset(tmp_path / "mpm_fresh" / "val", domain="mpm_fresh",
                                     training=False, normalization="xy_p99_global")
    with pytest.raises(ValueError, match="normalization differs"):
        _check_explicit_dataset(different, cfg, 4)


def test_calibration_raw_dataset_cannot_be_scored_as_checkpoint_normalized(tmp_path):
    from data.indexed_episodes import IndexedEpisodeDataset
    from scripts.unified.eval_unified import _check_explicit_dataset
    cfg = resolve_config(recipe(tmp_path))
    ds = IndexedEpisodeDataset(tmp_path / "mpm_fresh" / "val", domain="mpm_fresh",
                               training=False, raw=True)
    with pytest.raises(ValueError, match="raw calibration"):
        _check_explicit_dataset(ds, cfg, 4)


@pytest.mark.parametrize("domains", [["mpm_fresh", "mpm_drag", "dex"],
                                    ["mpm_fresh", "mpm_drag", "folding", "yikai", "dex"]])
def test_mixed_recipe_scores_whole_validation_cycles(tmp_path, domains):
    merged = None
    for domain in domains:
        recipe(tmp_path, domain)
        current = json.loads((tmp_path / "normalization.json").read_text())
        if merged is None:
            merged = current
        else:
            for channel in ("particle", "pose"):
                merged["profiles"]["xy_p99_per_domain"][channel].update(
                    current["profiles"]["xy_p99_per_domain"][channel])
    artifact = tmp_path / "normalization.json"
    artifact.write_text(json.dumps(merged))
    cfg = resolve_config(collection_recipe("unet", domains, tmp_path, artifact))
    for name, minimum in (("val_batches", 40), ("sampler_batches", 8)):
        budget = int(cfg.log[name])
        assert budget >= minimum and budget % len(domains) == 0


@pytest.mark.parametrize("architecture", ["unet", "transformer"])
@pytest.mark.parametrize("stratified,field", [("dex", "teacher"), ("folding", "action")])
def test_sampler_budget_covers_every_indexed_validation_stratum(tmp_path, architecture, stratified, field):
    domains = ["mpm_fresh", "mpm_drag", stratified]
    merged = None
    for domain in domains:
        recipe(tmp_path, domain)
        current = json.loads((tmp_path / "normalization.json").read_text())
        if merged is None:
            merged = current
        else:
            for channel in ("particle", "pose"):
                merged["profiles"]["xy_p99_per_domain"][channel].update(
                    current["profiles"]["xy_p99_per_domain"][channel])
    val_path = tmp_path / stratified / "val" / "INDEX.json"
    value = json.loads(val_path.read_text())
    original = value["records"][0]
    value["records"] = [dict(original, episode_id=f"val-{i}", group=f"val-{i}",
                             env=i, **{field: str(i)}) for i in range(5)]
    val_path.write_text(json.dumps(value))
    artifact = tmp_path / "normalization.json"
    artifact.write_text(json.dumps(merged))
    cfg = resolve_config(collection_recipe(architecture, domains, tmp_path, artifact))
    assert cfg.log.sampler_batches == 15
    assert cfg.log.val_batches == 42


def test_recipe_takes_a_release_test_folder_as_validation(tmp_path):
    cfg = resolve_config(recipe(tmp_path, validation="test"))
    assert cfg.data.roots.mpm_fresh.val == str(tmp_path / "mpm_fresh" / "test")
    from scripts.unified.collection_recipe import validation_folder
    with pytest.raises(FileNotFoundError, match="no val or test INDEX.json"):
        validation_folder(tmp_path, "mpm_drag")


def test_recipe_points_a_release_shard_domain_at_its_folders(tmp_path):
    recipe(tmp_path, validation="test")                      # writes the artifact and an mpm_fresh layout
    for split in ("train", "test"):
        folder = tmp_path / "pushing" / split
        folder.mkdir(parents=True)
        (folder / "00000.tar").write_bytes(b"")
        (folder / "INDEX.json").write_text(json.dumps(dict(schema="pwm-shard-index-v1", domain="pushing", split=split, shards=[])))
    cfg = resolve_config(collection_recipe("unet", ["mpm_fresh", "pushing"], tmp_path, tmp_path / "normalization.json"))
    assert cfg.data.roots.pushing.train == str(tmp_path / "pushing" / "train")
    assert cfg.data.roots.pushing.val == str(tmp_path / "pushing" / "test")
    assert "exclusions" not in collection_recipe("unet", ["pushing"], tmp_path, tmp_path / "normalization.json").data.roots.pushing
    without = resolve_config(collection_recipe("unet", ["mpm_fresh"], tmp_path, tmp_path / "normalization.json"))
    assert "pushing" not in without.data.roots
    # A release that ships episode exclusion lists for the shard dataset: the recipe names them per split.
    manifests = tmp_path / "pushing" / "manifests"
    manifests.mkdir()
    for name in ("pushing_exclusions_train_20260919.json", "pushing_exclusions_test_20260919.json", "notes.json"):
        (manifests / name).write_text(json.dumps(dict(blocklist=[dict(shard="00000.tar", key="batch_00000000", elem=0)])))
    listed = collection_recipe("unet", ["pushing"], tmp_path, tmp_path / "normalization.json").data.roots.pushing.exclusions
    assert list(listed.train) == [str(manifests / "pushing_exclusions_train_20260919.json")]
    assert list(listed.val) == [str(manifests / "pushing_exclusions_test_20260919.json")]


def test_recipe_reads_a_release_rl_push_layout_whole(tmp_path):
    recipe(tmp_path, validation="test")                      # writes the artifact and an mpm_fresh layout
    artifact = tmp_path / "normalization.json"
    with pytest.raises(ValueError, match="explicit frozen rl_push release"):
        collection_recipe("unet", ["mpm_fresh", "rl_push"], tmp_path, artifact)
    layout = tmp_path / "rl_push"
    for split in ("train", "test"):
        (layout / split / "objects").mkdir(parents=True)
        (layout / split / "INDEX.json").write_text(json.dumps(dict(schema="pwm-shard-index-v1", domain="rl_push", split=split)))
    (layout / "manifests").mkdir()
    (layout / "manifests" / "rl_push_val_assets.json").write_text(json.dumps(dict(kind="rl_push_val_assets", assets=[])))
    with pytest.raises(FileNotFoundError, match="rl_push_exclusions"):
        collection_recipe("unet", ["mpm_fresh", "rl_push"], tmp_path, artifact)
    excl = layout / "manifests" / "rl_push_exclusions_20260918.json"
    excl.write_text(json.dumps(dict(kind="rl_push_episode_exclusions", episodes=[])))
    cfg = resolve_config(collection_recipe("unet", ["mpm_fresh", "rl_push"], tmp_path, artifact))
    # Both folders are read whole with the release's own exclusion list; no object list splits them again.
    assert OmegaConf.to_container(cfg.data.roots.rl_push) == dict(
        train=str(layout / "train"), val=str(layout / "test"), exclusions={"train": [str(excl)], "val": [str(excl)]})
    with pytest.raises(ValueError, match="drop --rl-push-release"):
        collection_recipe("unet", ["rl_push"], tmp_path, artifact, rl_push_release=str(layout / "train"))


@pytest.mark.parametrize("domain,motion,across", [("mpm_fresh", dict(alpha=1.0, floor_mm=2.0), ["mpm_fresh"]),
                                                  ("mpm_drag", dict(alpha=1.0, floor_mm=0.25), ["mpm_drag"]),
                                                  ("folding", dict(alpha=1.0, floor_mm=0.02), None),
                                                  ("dex", None, None)])
def test_recipe_carries_the_window_policy(tmp_path, domain, motion, across):
    cfg = resolve_config(recipe(tmp_path, domain))
    if motion is None:
        assert "window_motion" not in cfg.data
    else:
        assert OmegaConf.to_container(cfg.data.window_motion) == {domain: motion}
    if across is None:
        assert "batch_across_episodes" not in cfg.data
    else:
        assert list(cfg.data.batch_across_episodes) == across


@pytest.mark.parametrize("domain,compiled", [("mpm_fresh", True), ("mpm_drag", True), ("yikai", False),
                                             ("folding", False), ("dex", False)])
def test_unet_recipe_compiles_only_datasets_with_few_point_counts(tmp_path, domain, compiled):
    for architecture in ("unet", "transformer"):
        (tmp_path / architecture).mkdir()
    assert bool(resolve_config(recipe(tmp_path / "unet", domain, "unet")).train.compile) is compiled
    assert not resolve_config(recipe(tmp_path / "transformer", domain, "transformer")).train.compile


def test_mixed_unet_recipe_runs_eager(tmp_path):
    recipe(tmp_path, "mpm_fresh")
    artifact = json.loads((tmp_path / "normalization.json").read_text())
    recipe(tmp_path, "yikai")
    merged = json.loads((tmp_path / "normalization.json").read_text())
    merged["profiles"]["xy_p99_per_domain"]["particle"]["mpm_fresh"] = artifact["profiles"]["xy_p99_per_domain"]["particle"]["mpm_fresh"]
    (tmp_path / "normalization.json").write_text(json.dumps(merged))
    cfg = resolve_config(collection_recipe("unet", ["mpm_fresh", "yikai"], tmp_path, tmp_path / "normalization.json"))
    assert not cfg.train.compile
    alone = resolve_config(collection_recipe("unet", ["mpm_fresh"], tmp_path, tmp_path / "normalization.json"))
    assert alone.train.compile


def test_recipe_reads_a_domain_from_a_variant_folder(tmp_path):
    recipe(tmp_path, "folding", validation="test")                  # artifact + folding/{train,test}
    import shutil
    shutil.copytree(tmp_path / "folding", tmp_path / "folding_8mm")
    artifact = tmp_path / "normalization.json"
    cfg = resolve_config(collection_recipe("unet", ["folding"], tmp_path, artifact, folders={"folding": "folding_8mm"}))
    assert cfg.data.roots.folding.train == str(tmp_path / "folding_8mm" / "train")
    assert cfg.data.roots.folding.val == str(tmp_path / "folding_8mm" / "test")
    plain = resolve_config(collection_recipe("unet", ["folding"], tmp_path, artifact))
    assert plain.data.roots.folding.train == str(tmp_path / "folding" / "train")
    with pytest.raises(ValueError, match="not selected"):
        collection_recipe("unet", ["folding"], tmp_path, artifact, folders={"dex": "dex_x"})
