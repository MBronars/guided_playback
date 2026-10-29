"""The two torch_geometric pieces the GNS baseline uses, in plain torch.

The GNS model imports exactly two things from torch_geometric: `radius_graph`
and `MessagePassing`. Installing torch_geometric + torch_scatter + torch_cluster
is a compiled, CUDA-version-matched install; this file removes that so inference
needs only torch and numpy. The model files themselves are copied unchanged, so
nothing about the network is reimplemented here.

`scripts/gns_baseline/check_against_reference.py` checks a rollout from this shim
against the original torch_geometric implementation on the same checkpoint.
"""
import torch
import torch.nn as nn


def radius_graph(x, r, batch=None, loop=False, max_num_neighbors=128):
    """Edges between points closer than `r`, matching torch_cluster's output.

    Returns (2, E) with row 0 = source j, row 1 = target i, the
    "source_to_target" convention MessagePassing assumes.

    Points are compared only within their own `batch` entry. Distances are
    computed in blocks of target points so memory stays O(block x N) rather than
    O(N^2) -- a 4,096-point scene is 16.8 M pairs, which is fine, but a batch of
    scenes concatenated is not.

    `max_num_neighbors` caps neighbours per target, keeping the CLOSEST ones.
    torch_cluster's cap keeps an implementation-defined subset instead, so the
    two can differ only for a point with more than `max_num_neighbors` others
    inside `r`. check_against_reference.py reports whether that cap is ever
    reached on real data (it is not, at the radius these checkpoints use).
    """
    device = x.device
    n = x.shape[0]
    if batch is None:
        batch = torch.zeros(n, dtype=torch.long, device=device)
    block = max(1, int(2_000_000 // max(n, 1)))
    src_all, dst_all = [], []
    for start in range(0, n, block):
        end = min(start + block, n)
        # `donot_use_mm_for_euclid_dist` is required, not an optimisation choice:
        # cdist's default matrix-multiply form computes ||a||^2+||b||^2-2ab, whose
        # round-off flips pairs sitting near the radius. On a real 1,024-point scene
        # that produced 22 wrong edges out of 28,898 and a 0.6 mm single-step
        # difference. The exact form reproduces torch_cluster edge for edge.
        d = torch.cdist(x[start:end], x, compute_mode="donot_use_mm_for_euclid_dist")
        same = batch[start:end, None] == batch[None, :]
        keep = (d < float(r)) & same
        if not loop:
            idx = torch.arange(start, end, device=device)
            keep[torch.arange(end - start, device=device), idx] = False
        if max_num_neighbors is not None and max_num_neighbors < n:
            counts = keep.sum(dim=1)
            if int(counts.max()) > max_num_neighbors:
                masked = torch.where(keep, d, torch.full_like(d, float("inf")))
                nearest = masked.topk(max_num_neighbors, dim=1, largest=False).indices
                capped = torch.zeros_like(keep)
                capped.scatter_(1, nearest, True)
                keep &= capped
        tgt, src = keep.nonzero(as_tuple=True)
        src_all.append(src)
        dst_all.append(tgt + start)
    src = torch.cat(src_all) if src_all else torch.empty(0, dtype=torch.long, device=device)
    dst = torch.cat(dst_all) if dst_all else torch.empty(0, dtype=torch.long, device=device)
    return torch.stack([src, dst], dim=0)


class MessagePassing(nn.Module):
    """`aggr='add'`, flow `source_to_target`, for the propagate() call GNS makes.

    GNS calls `self.propagate(edge_index=..., x=..., edge_features=...)` and
    defines `message(x_i, x_j, edge_features)` and
    `update(x_updated, x, edge_features)`. That is the whole surface used, so
    this implements those three steps directly: gather endpoints, run message(),
    sum messages into target nodes with index_add_, then run update().
    """

    def __init__(self, aggr="add", **kwargs):
        super().__init__()
        # Processor is constructed with aggr='max' but never calls propagate --
        # only InteractionNetwork does, with 'add'. So the setting is recorded
        # and enforced where it would actually change a result, not here.
        self._aggr = aggr

    def propagate(self, edge_index, x, edge_features):
        if self._aggr != "add":
            raise NotImplementedError(f"propagate() with aggr={self._aggr!r} is not implemented")
        src, dst = edge_index[0], edge_index[1]      # source j, target i
        x_i, x_j = x[dst], x[src]
        msg = self.message(x_i=x_i, x_j=x_j, edge_features=edge_features)
        agg = torch.zeros(x.shape[0], msg.shape[-1], dtype=msg.dtype, device=msg.device)
        agg.index_add_(0, dst, msg)                  # aggr='add'
        return self.update(agg, x, edge_features)

    def message(self, x_i, x_j, edge_features):
        raise NotImplementedError

    def update(self, x_updated, x, edge_features):
        raise NotImplementedError
