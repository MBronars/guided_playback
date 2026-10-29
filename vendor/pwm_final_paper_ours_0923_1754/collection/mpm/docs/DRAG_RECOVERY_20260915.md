# Dragging recovery — September 15, 2026

## Outcome and current observation

The incorrect normalization assertion is fixed. All **334 retained clips** from
the failed campaign are published in a separate verified recovery dataset.
A replacement collection was submitted and has a live Slurm step
**10447018.60**, on **dev-blackwellf / babel-l9-16**, at **18:30 UTC**.
It uses two PRO6000 GPUs, 16 CPUs, 48 GiB, and a **36-hour** limit. The
19,000-attempt cap is a safety bound, not a throughput prediction. GPU startup
probes passed on both devices. At **18:33 UTC**, both GPUs measured **97%
utilization**, step 10447018.60 was live, and the first simulation groups were
running. At **18:39 UTC**, both lanes had completed their first groups:
**8 attempts, 7 retained raw clips, 471 driven saved states**, and one settling
rejection. Both continued collecting; GPU utilization measured **96% / 97%**.
There were no worker or export failure receipts. No new 108-attempt TAR block
had completed yet; the seven fresh clips are raws, separate from the 334
verified recovered clips. These are time-stamped observations, not a guarantee
of future liveness. Evidence: `BOTH_LANES_FIRST_GROUPS.json`.

## Failure and diagnosis

The original collection started at 07:38:59 UTC and stopped at 13:17:24 UTC.
Its export reviewer assumed that the historical float32 normalization and
denormalization were exact inverses to a fixed 1e-8 m absolute tolerance.
They are not: normalization divides by `scale + 1e-8`, while denormalization
multiplies by `scale`; float32 cancellation adds rounding error. One check
reported 103 failing elements and a maximum error of about **1.79e-8 m**.
The fail-stop supervisor then stopped both collector lanes.

The old assertion was reproduced on two profile checks in the pending
76-clip block. Both profiles now pass an independent float64 formula oracle
with magnitude-dependent float32 rounding bounds. Tests intentionally corrupting
either map still fail. Exact raw positions, action commands and activity masks
must still match the training loader bit for bit. **Neither training
normalization nor its statistics were changed.**

The operational failure was also that an isolated export-review problem halted
useful simulation. Previously, TARs were written under the training root before
consumer verification. They are now written outside that root and atomically
published only after both consumer profiles pass. An export failure retains raw
data and an `EXPORT_FAILURE.json`, quarantines unapproved exports, and continues
with the next independent block. Three consecutive export failures stop that
lane and the supervisor. Simulation failures still stop collection. This is
bounded failure isolation, not an automatic repair system for arbitrary bugs.

## Recovered data

| Blocks | Retained clips | Verification |
| --- | ---: | --- |
| 0–2 | 248 | Existing exact consumer approvals; artifact hashes rechecked |
| 3 | 76 | All physical decisions revalidated; both consumer profiles rechecked |
| 4 | 10 | Raw recovery, fresh export, physical and both-profile consumer checks |
| Total | **334** | Copied artifacts hash-verified in the new dataset |

These are from **444 completed attempts**, with **25,966 driven saved states**
in the retained raws. Rejected attempts were not silently replaced. Original
raw hashes and physical retention decisions are unchanged. The recovery copy
does not create additional independent samples.

Use this verified recovery root:

```text
/data/group_data/katefgroup-ssd/world_model/particle_world_model/datasets/mpm-drag-gentle-v4-recovered-20260915/train
```

Do **not** use the entire original campaign training root: its block 3 contains
TARs written before the old review failed. The old campaign remains unchanged as
evidence, not the recommended recovered training source.

New collection bundle (independent sequence namespace):

```text
/data/group_data/katefgroup-ssd/world_model/particle_world_model/staging/mpm-drag-gentle-v4-restart-20260915
```

Its approved blocks appear under `data/train/train/block-*/diagnostic-export`.
Pending or failed exports live under `export-review/`, outside the training root.
The first full block requires 108 attempts; raw completions precede TAR publication.

## Verification and provenance

- **432 collector tests + 26 training-adapter tests passed**, zero skips/failures
  in the final configured run. Earlier environment-setup failures remain in the
  local evidence; they are not the final results.
- Ten actual retained clips also passed the new operator's staging, exact
  consumer review and atomic-publication path.
- After recovery publication, the training loader consumed all **334 clips in
  52 shards from the new shared root** under both normalization profiles, with
  exact raw-window/command agreement. See `PUBLISHED_CONSUMER.json`.
- Critical source changes are limited to `collection/mpm/drag/review.py` and
  `collection/mpm/drag/launch.py`. Simulator, sampler, adapter, statistics,
  material clock, physical quality filters and the rest of the critical closure
  are byte-identical to the original qualified frozen bundle.
- The physical comparison, wall policy and visualization functions inside the
  changed reviewer are AST-identical. Existing GPU/visual qualification is
  explicitly inherited, not misrepresented as rerun GPU evidence.
- Original frozen source, failed gate history, failure receipts and raws remain
  intact. The new gate includes a hashed recovery amendment and test evidence.

Local receipts and reproducible recovery scripts:

```text
runs/mpm-drag-recovery-20260915/
  audit.py
  release.py
  start.py
  CPU_QUALIFIED.xml
  ADAPTER_TESTS.xml
  OPERATOR_REAL.json
  PUBLISHED_CONSUMER.json
  block-000003/RESULT.json
  block-000004/RESULT.json
  RELEASED.json
  PRODUCTION_SUBMITTED.json
  BOTH_LANES_FIRST_GROUPS.json
```

Shared evidence:

```text
/data/group_data/katefgroup-ssd/world_model/particle_world_model/evidence/mpm-drag-recovery-20260915
```

Scope remains g256, train shapes, 1–4 homogeneous elastic/plasticine objects,
10 kPa–1 MPa, no sand, gentle-v4 commands and mandatory fragmentation screening.
Shared-grid contact and point-connectivity limitations remain. Held-out dragging
and 5–7-object evaluation are not qualified by this train-only restart.

## Monitoring

Inspect the new bundle's `lane-0.log`, `lane-1.log`, `srun.err`, top-level
`FAILURE.json`, per-block `EXPORT_FAILURE.json`, startup receipts, and completed
raw `RESULT.json` files. Filesystem receipts alone do not prove live execution;
check `squeue --steps -j 10447018` as well. Launch receipts are not completed
collection. The code can isolate export failures without an interactive agent,
but does not promise agent monitoring or arbitrary fixes after handoff.

For a read-only combined count/failure/scheduler report:

```bash
/data/group_data/katefgroup-ssd/world_model/particle_world_model/environments/mpm-genesis-v1/bin/python runs/mpm-drag-recovery-20260915/status.py
```
