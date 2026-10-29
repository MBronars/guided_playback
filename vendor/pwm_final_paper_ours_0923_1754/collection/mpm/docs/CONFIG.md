# Configurable MPM collection

Edit the `collection` section of [the operator config](../launch/mpm_collect.json),
or copy the complete config to a new JSON file and pass `--config /path/run.json`
**before** the command. Preserve the common registry/campaign paths and pinned
template. Settings are validated, frozen in the run bundle and copied into the
reservation/ready receipts. Unknown fields fail rather than being ignored.

Example training selection (replace only the `collection` object):

```json
{
  "split": "train",
  "evaluation": "train",
  "object_counts": [1, 2, 3],
  "object_weights": [1, 2, 1],
  "action_weights": {"drop": 1, "translate": 1, "collision": 2, "drop_on": 1},
  "max_attempts": 12000,
  "target_retained": 10000,
  "hardware": {"gpus": 4, "gpu_type": "RTX_PRO_6000", "walltime_hours": 24}
}
```

```bash
python3 collection/mpm/launch/mpm_collect.py --config /path/run.json plan --name alice-custom-one
python3 collection/mpm/launch/mpm_collect.py --config /path/run.json preview --out /path/new-preview --count 12
python3 collection/mpm/launch/mpm_collect.py --config /path/run.json preflight
# Real GPU submission, only after inspection:
python3 collection/mpm/launch/mpm_collect.py --config /path/run.json launch --name alice-custom-one
```

`preview` prepares actual initial particle states/manifests on CPU. It does not
simulate trajectories, produce rollout GIFs, reserve production sequences, export
training shards or certify a configuration. Preview directories cannot be used
as prepared production runs. It requires a settings-enabled pinned template.

## Choices and limits

| Field | Meaning |
|---|---|
| `split` | `train`, `validation`, or `test` |
| `evaluation` | `train`, `seen`, `unseen`, `count_seen`, `count_unseen` |
| `object_counts` | Distinct subset of 1–4, or 5–7 for `count_*` evaluation |
| `object_weights` | Positive relative weights, aligned with `object_counts` |
| `action_weights` | Positive relative weights; omit disabled actions |
| `max_attempts` | Exact simulation-attempt ceiling, 4–120,000; excludes matched controls |
| `target_retained` | Optional stopping target, at most `max_attempts`; not promised yield |
| `hardware.walltime_hours` | Integer 12–48; always retains the five-hour shutdown reserve |
| `hardware.gpus`, `gpu_type` | 2 or 4 × `RTX_PRO_6000` (v4 onward; default 4) |

One-object actions: `drop`, `translate`. Multi-object actions: `collision`,
`drop_on`, `stack`, `scatter`, `random`. Every selected count needs a compatible
action. There is no free-form solver/material/timestep/filter override here.

Weights govern **proposed** samples. Filtering and finite sample counts change
retained proportions; neither exact retained balance nor complete 10K shape
coverage is guaranteed for customized mixtures. The original full balanced
default still uses the historical sampler exactly. Custom recipes (and partial
chunks) use deterministic local RNGs, weighted counts/actions, and distinct
within-scene shape indices. Material and stiffness offsets span all nine pairings
over each nine-attempt block, rather than locking each material to one stiffness.
Custom shape coverage must be measured, not inferred
from the original cyclic sampler's coverage guarantee.

Maximum attempts are distributed exactly across bounded two/four-lane waves, with
1–60 coverage attempts per chunk. Rejections count as attempts. A retained target
can overshoot because already-started work is reviewed before stopping; the
attempt ceiling is strict. Very small runs may stop for zero retained examples
or insufficient first-wave quality—those checks are not waived.

## Split safety and current execution boundary

The bank role is derived, not a user-editable conflicting flag:

| Split/category | Counts | Asset bank |
|---|---|---|
| `train` / `train` | 1–4 subset | train |
| validation or test / `seen` | 1–4 subset | train |
| validation or test / `unseen` | 1–4 subset | corresponding held-out bank |
| validation or test / `count_seen` | 5–7 subset | train |
| validation or test / `count_unseen` | 5–7 subset | corresponding held-out bank |

For example, a minimal collection object for held-out validation preparation is
`{"split":"validation","evaluation":"unseen","object_counts":[2,4]}`.
Other fields receive defaults; weights become uniform over the selected counts.

**Only training configurations may launch production collection.** Evaluation
configurations support `plan` and CPU `preview`; `preflight`, `prepare`, and
`launch` reject them before a production reservation or job. Fast evaluation,
especially 5–7 objects, still requires a separately qualified execution path.
Do not relabel a held-out preview as training to bypass this boundary.

Each run continues to reserve a disjoint 2,000-cycle block from the common
registry; a smaller attempt budget consumes only part of its reserved block.
Unused cycles are not reclaimed automatically. The current eight-block capacity
is unchanged. A custom sampling signature is included in sequence keys, and
no RNG depends on process scheduling or the global NumPy RNG. Use a new run name
when changing settings; reusing a name with a changed recipe is rejected.

## Deployment and provenance

The settings-enabled shared launcher is `tools/mpm-v4/mpm_collect.py` under the
same lab SSD project root. It uses the existing shared assets, common sequence
registry and campaign directory; publishing it does not reserve sequences or
submit jobs. The adjacent shared `CONFIG.md` describes the same settings.

Custom settings require `configuration_schema = mpm-collection-settings-v1` in
the pinned template. Older `mpm-v1` remains runnable with the original defaults;
it refuses custom settings rather than silently ignoring them. Updating source
does not mutate any published or running bundle.

The preparation function now accepts a validated specification factory; default
generation remains unchanged. This intentional source change is recorded alongside
the original integration hashes in `SOURCE_PARITY.json`. Solver, guarded backend,
adaptive clock, normalization, exclusion thresholds and GPU startup comparisons
have not changed. Default-specification parity and customized split/count/seed
contracts are covered by the domain tests.
