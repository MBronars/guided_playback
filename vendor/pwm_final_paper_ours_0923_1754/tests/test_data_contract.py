"""Batch metadata and requested data settings must survive every adapter."""
from dataclasses import fields, replace

import pytest
import torch

from data.batch import TrajectoryBatch
from data.dataset import BaseGraspDataset, NormalizationStats, SkipSample, create_dataloader
from tests.test_train import create_dummy_batch
from tests.test_unet_std_pose_native import make_batch


@pytest.mark.parametrize("operation", ["slice", "float", "half", "to"])
def test_batch_transforms_preserve_every_field(operation):
    batch = make_batch(deform_n=8, rigid=((1, 8),))
    batch.domain = "cloth"
    transformed = {"slice": lambda: batch.slice(1), "float": batch.float,
                   "half": batch.half, "to": lambda: batch.to("cpu")}[operation]()
    for field in fields(batch):
        before, after = getattr(batch, field.name), getattr(transformed, field.name)
        if torch.is_tensor(before):
            expected = before[:1] if operation == "slice" else before
            if operation == "half" and before.is_floating_point():
                expected = expected.half()
            elif operation == "float" and before.is_floating_point():
                expected = expected.float()
            torch.testing.assert_close(after, expected, rtol=0, atol=0)
        else:
            assert after is before


@pytest.mark.parametrize("mismatch", ["domain", "delta_stats", "mat_type", "context_len"])
def test_collate_rejects_mixed_contracts(mismatch):
    a = create_dummy_batch(B=1, T=6, N=8, context_len=2)
    b = replace(a)
    if mismatch == "domain":
        a.domain, b.domain = "mpm", "cloth"
    elif mismatch == "delta_stats":
        b.delta_stats = NormalizationStats(-torch.ones(3) * 2, torch.ones(3) * 2)
    elif mismatch == "mat_type":
        a.mat_type = None
        b.mat_type = torch.zeros(1, 8, dtype=torch.long)
    else:
        b.context_len = 3  # malformed input must not inherit element zero's metadata
    with pytest.raises(ValueError, match="collate|Collat"):
        TrajectoryBatch.collate([a, b])


def test_collate_accepts_equal_independently_loaded_stats():
    a = create_dummy_batch(B=1, T=6, N=8, context_len=2)
    b = replace(a, delta_stats=NormalizationStats(a.delta_stats.min_val.clone(),
                                                 a.delta_stats.max_val.clone()))
    assert TrajectoryBatch.collate([a, b]).batch_size == 2
    with pytest.raises(ValueError, match="empty"):
        TrajectoryBatch.collate([])


def test_unknown_target_is_not_silently_treated_as_delta():
    with pytest.raises(ValueError, match="prediction_target"):
        replace(create_dummy_batch(B=1), prediction_target="dleta")


@pytest.mark.parametrize("target", ["position", "anchor_offset", "frame0_offset"])
def test_retired_targets_fail_at_the_batch_boundary(target):
    with pytest.raises(ValueError, match="prediction_target.*delta"):
        replace(create_dummy_batch(B=1), prediction_target=target)


@pytest.mark.parametrize("field", ["offset_stats", "pose_action_deltas", "gripper_anchor_pose"])
def test_retired_specialist_fields_are_not_part_of_new_batch_contract(field):
    assert field not in {item.name for item in fields(TrajectoryBatch)}
    with pytest.raises(TypeError, match=field):
        replace(create_dummy_batch(B=1), **{field: None})


def test_old_unified_pickle_state_loads_and_drops_unused_historical_fields():
    import pickle
    batch = make_batch(deform_n=8, rigid=((1, 8),))
    batch.domain = "cloth"
    # Pickle restores instance dictionaries, including fields recorded by the
    # old dataclass. No historical numerical scale is substituted on load.
    batch.offset_stats = batch.delta_stats
    batch.pose_action_deltas = None
    batch.gripper_anchor_pose = None
    restored = pickle.loads(pickle.dumps(batch))
    cleaned = restored.to("cpu")
    torch.testing.assert_close(cleaned.positions, batch.positions, atol=0, rtol=0)
    torch.testing.assert_close(cleaned.pose_deltas_gt, batch.pose_deltas_gt, atol=0, rtol=0)
    assert cleaned.domain == "cloth"
    assert not hasattr(cleaned, "offset_stats")
    assert not hasattr(cleaned, "pose_action_deltas")


@pytest.mark.parametrize("options", [
    {"prediction_target": "frame0_offset"}, {"window_movement_min_mm": 1.0},
    {"window_movement_power": 1.0}, {"max_objects": 1}, {"num_collate": 2},
    {"temporal_stride": 2}, {"typo_option": False},
])
def test_unified_rejects_unsupported_settings_before_io(options):
    with pytest.raises(ValueError, match="unified|Unified"):
        create_dataloader("unified://v1", normalization="unified://xy_p99", format="unified",
                          num_workers=0, **options)


def test_first_window_is_forwarded_and_used():
    from data.unified_dataset import _UnifiedWDSBase
    stream = create_dataloader("unified://v1", normalization="unified://xy_p99", format="unified",
                               first_window=True, context_len=2, future_len=3)
    assert stream.kw["first_window"] is True
    dataset = object.__new__(_UnifiedWDSBase)
    dataset.context_len, dataset.future_len = 2, 3
    dataset.training, dataset.first_window = True, True
    positions = torch.arange(30).reshape(1, 10, 1, 3)
    assert torch.equal(dataset._window(positions), positions[:, :5])


def test_first_window_reaches_validation_datasets(monkeypatch):
    from data import unified_dataset
    class Empty(torch.utils.data.IterableDataset):
        def __iter__(self):
            return iter(())
    monkeypatch.setattr(unified_dataset, "build_domain_dataset", lambda *a, **k: Empty())
    loader = create_dataloader("unified://v1", normalization="unified://xy_p99", format="unified",
                                training=False, first_window=True, num_workers=0)
    assert all(item.dataset.first_window for item in loader.loaders.values())


def test_processing_errors_include_source_and_are_not_swallowed():
    class Broken(BaseGraspDataset):
        def __init__(self):
            self.num_collate = 1

        def _create_pipeline(self):
            return iter([{"__url__": "example.tar", "__key__": "episode17"}])

        def _process_sample(self, sample):
            raise KeyError("missing_physics")

    with pytest.raises(RuntimeError, match="example.tar.*episode17") as error:
        next(iter(Broken()))
    assert isinstance(error.value.__cause__, KeyError)


def test_intentional_adapter_exclusions_still_skip_and_report_identity(capsys):
    class Filtered(BaseGraspDataset):
        num_collate = 1

        def __init__(self):
            pass

        def _create_pipeline(self):
            return iter([{"__url__": "example.tar", "__key__": key} for key in ("bad", "good")])

        def _process_sample(self, sample):
            if sample["__key__"] == "bad":
                raise SkipSample("unusable link metadata")
            return create_dummy_batch(B=1)

    assert len(list(Filtered())) == 1
    assert "example.tar:bad: unusable link metadata" in capsys.readouterr().out


def test_cloth_unusable_link_metadata_is_an_explicit_exclusion():
    from data.unified_dataset import UnifiedClothDataset
    dataset = object.__new__(UnifiedClothDataset)
    dataset._ee_ids = {"bad": torch.zeros(553)}
    with pytest.raises(SkipSample, match="unusable ee_link_ids"):
        dataset._cloth_sidecar({"__key__": "bad", "n_cloth.pyd": torch.tensor(10)}, None, None, None)


def test_bucket_flush_repeats_samples_not_prebatched_records():
    class Prebatched(BaseGraspDataset):
        def __init__(self):
            self.num_collate, self.batch_size, self.bucket_max_items = 1, 4, 1
            self.batches = [create_dummy_batch(B=3, N=8), create_dummy_batch(B=2, N=12)]
            for index in range(3):
                self.batches[0].positions[index].fill_(index + 1)

        def _create_pipeline(self):
            return iter([{"index": 0}, {"index": 1}])

        def _process_sample(self, sample):
            return self.batches[sample["index"]]

    batches = list(Prebatched())
    assert [batch.batch_size for batch in batches] == [4, 2]
    torch.testing.assert_close(batches[0].positions[:, 0, 0, 0], torch.tensor([1., 2., 3., 1.]))
