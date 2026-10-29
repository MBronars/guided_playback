"""Shape-bucket and exact sample-count contracts extracted from legacy model tests."""
import torch

from data.batch import TrajectoryBatch
from data.dataset import BaseGraspDataset, NormalizationStats


def test_dataset_bucket_collate():
    """__iter__ with num_collate>1 must bucket by particle count N, never mixing shapes."""
    def tiny_batch(N):
        ident = NormalizationStats(
            min_val=torch.tensor([-1.0, -1.0, -1.0]),
            max_val=torch.tensor([1.0, 1.0, 1.0]),
        )
        return TrajectoryBatch(
            positions=torch.randn(1, 4, N, 3),
            actions=torch.randn(1, 2, N, 3),
            action_mask=torch.zeros(1, N, dtype=torch.bool),
            object_ids=torch.zeros(1, N, dtype=torch.long),
            valid_lengths=torch.full((1,), N, dtype=torch.long),
            E=torch.rand(1, N),
            nu=torch.rand(1, N),
            rho=torch.rand(1, N),
            context_len=2,
            delta_stats=ident,
            position_stats=ident,
        )

    # Interleaved shapes: old single-buffer logic would try to collate N=8 with N=16
    # (crash in torch.cat). Bucketing yields one full batch per bucket (B=2) plus the
    # end-of-stream flush of each part-filled bucket (B=1).
    shapes = [8, 16, 8, 16, 8, 16]
    ds = object.__new__(BaseGraspDataset)
    ds.num_collate = 2
    ds._create_pipeline = lambda: iter(shapes)
    ds._process_sample = tiny_batch

    yielded = list(BaseGraspDataset.__iter__(ds))
    total = sum(b.positions.shape[0] for b in yielded)
    assert total == len(shapes), f"lost samples: {total} != {len(shapes)}"
    got = sorted((b.positions.shape[0], b.positions.shape[2]) for b in yielded)
    assert got == [(1, 8), (1, 16), (2, 8), (2, 16)], f"unexpected batches: {got}"
    print("[dataset] shape-bucketed collation OK")



def _stub_ds(shapes, num_collate=1, batch_size=0, inner_B=1):
    """BaseGraspDataset stub yielding tiny batches with given particle counts."""
    def tiny(N, B):
        ident = NormalizationStats(
            min_val=torch.tensor([-1.0, -1.0, -1.0]),
            max_val=torch.tensor([1.0, 1.0, 1.0]))
        return TrajectoryBatch(
            positions=torch.randn(B, 4, N, 3), actions=torch.randn(B, 2, N, 3),
            action_mask=torch.zeros(B, N, dtype=torch.bool),
            object_ids=torch.zeros(B, N, dtype=torch.long),
            valid_lengths=torch.full((B,), N, dtype=torch.long),
            E=torch.rand(B, N), nu=torch.rand(B, N), rho=torch.rand(B, N),
            context_len=2, delta_stats=ident, position_stats=ident)
    ds = object.__new__(BaseGraspDataset)
    ds.num_collate = num_collate
    if batch_size:
        ds.batch_size = batch_size
    ds._create_pipeline = lambda: iter(shapes)
    ds._process_sample = lambda N: tiny(N, inner_B)
    return ds



def test_batch_size_samples_semantics():
    """batch_size counts SAMPLES: uniform stream of B=1 items, batch_size=3."""
    ds = _stub_ds([8] * 7, batch_size=3)
    got = [(b.positions.shape[0], b.positions.shape[2]) for b in BaseGraspDataset.__iter__(ds)]
    assert got == [(3, 8), (3, 8), (1, 8)], got   # two full batches + flush
    print("[batch_size] B=1 stream: 3-sample batches + flush OK")



def test_batch_size_prebatched():
    """Pre-batched items (inner B=2) with a MISALIGNED batch_size=3: items must be
    SPLIT at the boundary (remainder carried over), yielding exact 3-sample batches.
    num_collate could never express this; batch_size is packaging-independent."""
    ds = _stub_ds([8] * 5, batch_size=3, inner_B=2)
    out = list(BaseGraspDataset.__iter__(ds))
    got = [b.positions.shape[0] for b in out]
    assert got == [3, 3, 3, 1], got                       # 10 samples -> 3+3+3+1
    # content preservation: concatenated output == concatenated input stream
    torch.manual_seed(0)  # _stub_ds batches are random; instead check via count+shape
    total = sum(b.positions.shape[0] for b in out)
    assert total == 10 and all(b.positions.shape[2] == 8 for b in out)
    print("[batch_size] misaligned pre-batched items split exactly OK")



def test_batch_size_shape_buckets():
    """batch_size respects shape buckets: mixed N never collates together."""
    ds = _stub_ds([8, 16, 8, 16, 8, 16], batch_size=2)
    got = sorted((b.positions.shape[0], b.positions.shape[2])
                 for b in BaseGraspDataset.__iter__(ds))
    assert got == [(1, 8), (1, 16), (2, 8), (2, 16)], got
    print("[batch_size] shape-bucketed accumulation OK")



def test_num_collate_legacy_unchanged():
    """batch_size unset: legacy num_collate item-count behavior still holds."""
    ds = _stub_ds([8] * 5, num_collate=2)
    got = [b.positions.shape[0] for b in BaseGraspDataset.__iter__(ds)]
    assert got == [2, 2, 1], got
    print("[batch_size] legacy num_collate path unchanged OK")
