# MPM collection: operator runbook

## Start here

Configurable count/action mixtures, budgets and CPU evaluation previews are
documented in [CONFIG.md](CONFIG.md). The settings-enabled template preserves
the original training defaults; tables below describe that default recipe.

This is the supported entry point for **passive MPM training data, 1–4 objects,
two or four RTX PRO 6000 GPUs (default four), one 48-hour Slurm allocation**. It uses the already
qualified `fused4` implementation and rechecks the newly allocated GPUs before
collecting. No chat history, GPU UUID editing, manual seed selection, or second
job submission is required.

For lab members, use the shared entry point from any cluster login directory:

```bash
MPM_OPERATOR=/data/group_data/katefgroup-ssd/world_model/particle_world_model/tools/mpm-v4/mpm_collect.py
python3 "$MPM_OPERATOR" assets
python3 "$MPM_OPERATOR" preflight
python3 "$MPM_OPERATOR" launch --name alice-night-next
python3 "$MPM_OPERATOR" status --name alice-night-next
```

See [shared assets and multi-user access](ASSETS.md). The main project
checkout also provides the same front end and shared configuration:

```bash
cd /data/user_data/robertre/particle_world_model
python3 collection/mpm/launch/mpm_collect.py preflight
python3 collection/mpm/launch/mpm_collect.py launch --name mpm-night-next
python3 collection/mpm/launch/mpm_collect.py status --name mpm-night-next
```

**The `launch` command submits a real GPU job (four GPUs by default).** Choose a descriptive run
name. Repeating the same successful launch returns its recorded job ID instead
of submitting a duplicate. A new name means a new collection and GPU request.

Historical job **10429006** failed startup qualification and did not collect.
The collection step in allocation **10421336** was cancelled; an allocation can
remain running without a collector. This handoff does not replace, restart or
submit another copy of either. Check their current state before launching:

```bash
squeue -j 10429006,10421336
python3 collection/mpm/launch/mpm_collect.py status
```

The command without a run name reports registered operator runs plus the known
original, fast-two-PRO, Ada-pilot, and historical four-PRO collection roots. It
counts `COMPLETE.json` receipts rather than trusting lagging `STATUS.json`
counters. This is a status report, not a full data hash audit.

The `mpm-v4` release reuses each GPU's successfully qualified compiler artifacts,
removing repeated per-chunk compilation. It does not change simulation physics
or relax any qualification gates. See [PERFORMANCE.md](PERFORMANCE.md) for the
measured 24–26% full-chunk worker speedup, numerical diagnostics and limitations.
It also enables the four-thread bounded writer in both candidate qualification
and collection. The full-size paired test measured an additional ~12% worker
throughput gain. Those component measurements are not a sustained campaign-rate
claim. Serial references, numerical tolerances, controls and filters are unchanged.

## Inspect without submitting

```bash
python3 collection/mpm/launch/mpm_collect.py plan --name mpm-night-next
```

`plan` is read-only. Its proposed seed range is not reserved until preparation,
so a simultaneous operator can change which range is assigned next.

For a staged launch:

```bash
python3 collection/mpm/launch/mpm_collect.py prepare --name mpm-night-next
python3 collection/mpm/launch/mpm_collect.py submit --name mpm-night-next --test-only
python3 collection/mpm/launch/mpm_collect.py submit --name mpm-night-next
```

`prepare` reserves seeds and freezes a private run bundle but requests no GPUs.
`--test-only` asks Slurm whether the job can be scheduled; it creates no actual
job or submission claim. Its start-time prediction is not a guarantee. The
last command is the real submission. `launch` combines preparation and submission.

## Prerequisites and access

The front end uses Python 3's standard library. It launches the pinned
collection interpreter:

```
/data/group_data/katefgroup-ssd/world_model/particle_world_model/environments/mpm-genesis-v1/bin/python
```

Defaults are in [collection/mpm/launch/mpm_collect.json](../launch/mpm_collect.json): lab
partition `katef`, account `kfragki2`, QoS `katef_qos`, shared-SSD output roots,
and an immutable template SHA. Slurm commands must be available in the shell.

**Another person must belong to `kfragki2_group` and have their own permitted
Slurm account/QoS.** Shared assets/runtime are group-readable without granting
access to Robert's private home. This is not a portable installation on an
unrelated machine. Run
`preflight` as the actual operator: it checks interpreter execution, output
permissions/free space, frozen sources, shape-bank artifacts, and preserved
reference hashes. Use `submit --test-only` to check scheduler access. The shared
deployment has copied the required controls, qualification references and source
closure to the lab SSD. Original provenance can still name old paths, but those
are not executable dependencies. Do not change numerical hashes just to make
a moved dataset pass.

Agent filesystem sandboxes may report the shared SSD as unwritable even when
the cluster account has access. In that case run preflight through the approved
normal-cluster execution mechanism. Do not weaken its permission check.

The immutable template is:

```
/data/group_data/katefgroup-ssd/world_model/particle_world_model/tools/mpm-v2/template-v1
```

Its exact `BUNDLE.json` SHA-256 is pinned in the adjacent `mpm_collect.json`
and publication `READY.json`; preflight rejects any mismatch.

The template and its pinned input artifacts are required; **a Git clone alone
does not include the shared datasets/runtime**. All executed Python source comes
from frozen bundles, not mutable research modules. Preserve the shared
baseline bundle, banks and qualification data. Each new run receives its own
verified bundle; subsequent source edits cannot silently change it.

## Fixed collection contract

| Setting | Supported value |
|---|---|
| Hardware | 4 × RTX PRO 6000, 16 CPUs, 128 GiB host RAM |
| Allocation | 48 hours, no automatic requeue |
| Scope | Passive training, 1–4 objects; no validation/test generation |
| Implementation | One worker/GPU; four worlds/worker; block 64; fused walls |
| Per-object sampling | 1,024 particles, existing frozen 10K training-shape bank |
| Stiffness phases | Existing 10 kPa, 100 kPa, 1 MPa protocol |
| Recorded episode | 49 saved states, 48 transitions; unchanged native timestep |
| Wall policy | Existing certified sand-only wall-contact policy |
| Consumers | Both per-domain and global normalization profiles checked |
| Per-run seed capacity | 2,000 cycles × 60 coverage attempts = 120,000 attempts maximum |
| Admission shutdown | Stop new waves five hours before allocation end; drain pending work |
| Disk reserve | Stop admitting below 500 GiB free |

The attempt ceiling is not promised yield, and 48 hours does not mean 48 hours
of GPU admissions. Qualification, filtering, I/O, cold compilation, scheduling,
and shutdown reserve affect throughput. At the currently demonstrated two-PRO
rate near 1,000 clips/hour, four-GPU linear scaling is only an estimate until
the new allocation has measured its own sustained rate.

No timestep, constitutive model, collapse guard, filtering threshold, data
normalization, or source split is changed by this operator wrapper.

## What happens automatically

1. Reserve a disjoint cycle block under the project registry lock.
2. Freeze and verify the run's numerical code, loaders, bank and references.
3. Record a submit-once claim, then submit one Slurm job and record its ID.
4. On allocation, validate every configured PRO GPU UUID and the real Slurm
   deadline; check all 20 predeclared 1–4-object training cases on each GPU.
5. Require serial/batched and preserved-reference numerical parity, two serial
   repeats of sensitive case 3, exact fused grid behavior, failure isolation,
   matching export decisions and both loader profiles on all allocated GPUs.
6. Collect one fresh chunk per GPU. Check controls, journals, integrity,
   contraction/collapse, export rules, and both loader profiles; generate GIFs.
7. Continue automatically after all first-wave reviews pass. This is
   explicitly an **automated** gate, not a claim of a new manual visual review.
8. Keep simulation, bounded CPU prefetch, review, checked publication, and GIF
   generation running until the seed/time/disk budget ends or a check fails.

The original failed 1–7-object qualification is not rewritten. Startup checks
here approve training only. The same RMS limit (0.1 mm), exact saved-state wall
contact presence, and two-native-step timestamp tolerance remain enforced.
Qualification failure means no production launch; a collection failure preserves
completed chunks and partial evidence instead of quietly retrying.

## Where results live

For run `mpm-night-next`:

```
/data/group_data/katefgroup-ssd/world_model/particle_world_model/staging/
  mpm-operator-campaigns-v1/mpm-night-next/
    RESERVED.json
    READY.json
    SUBMISSION_CLAIM.json
    SUBMITTED.json
    bundle/
    slurm-<job-id>.log
    data/
      ALLOCATION.json
      qualification/
        lane-0.log ... lane-3.log
        TRAINING_PASS.json
      collection/
        STATUS.json
        PILOT_APPROVED.json
        chunks/train-<cycle>/
          COMPLETE.json
          samples.gif
          review.log
        datasets/train/train-<cycle>/
          MANIFEST.json
          <numeric-shard>.tar
        STOP.json
      FAILURE.json               # only on failure
```

The shared reservation registry is
`.../staging/mpm-operator-registry-v1`. All operators must use **this same
registry**, even if account/QoS settings differ. Do not create a second production
registry, delete reservations, or bypass the launcher with copied sbatch commands.

Reserved blocks start at 84000, in increments of 2000. They avoid all known
existing campaign ranges. The pinned sampler permits cycles only through
100000; the operator stops at its exclusive bound of 100000 rather than wrapping
or guessing a new range. There are eight initial blocks. Exhaustion requires an
explicit sampler/range extension and validation, not manual seed reuse.

## Monitoring, checking, and visual review

```bash
python3 collection/mpm/launch/mpm_collect.py status --name mpm-night-next
python3 collection/mpm/launch/mpm_collect.py audit --name mpm-night-next
```

`audit` rehashes every completed chunk and its bound artifacts, checks training
scope and cycle membership, and reports retained counts. It may be expensive
on a large dataset. It does not mutate data, claim unfinished chunks, or promote
the staged dataset to a training release. Qualification/collection workers
already enforce per-chunk loader checks before publishing.

Inspect first-wave and periodically sampled `samples.gif` files. In particular,
check material behavior, object separation/interaction, floor penetration,
collapse, unexpected jumps and non-sand wall contacts. The XZ GIF hides Y, so
it supplements—not replaces—the full numerical guards.

## Stop and continuation

Find the exact job ID in `SUBMITTED.json` or `status`. For a running **collection
phase**, request a graceful admission stop:

```bash
scancel --signal=USR1 --batch <job-id>
```

The collector finishes bounded in-flight work and drains reviews. Do not expect
instant termination. To cancel a queued job, or abort startup qualification,
use `scancel <job-id>` instead. Never cancel a job solely by a broad name match.

For another allocation with fresh trajectories:

```bash
python3 collection/mpm/launch/mpm_collect.py launch --name mpm-night-two --continue-from mpm-night-next
```

This is **dataset continuation, not physical episode resume**. It reserves a
new cycle block, preserves the old root, records the relationship, and qualifies
the newly allocated GPUs. Previous reviewed shards remain usable candidates;
incomplete old episodes stay excluded. It does not automatically reconstruct
unfinished CPU reviews from a killed run. Those need an explicit recovery audit.

If preparation fails, its reservation and partial bundle remain. Use a new run
name after diagnosing the cause; do not overwrite the failed root. If submission
times out or returns an ambiguous receipt, the launcher writes
`SUBMISSION_UNCERTAIN.json` and **refuses automatic retry**. The scheduler might
already have accepted the job. Search its recorded `mpm-operator:<name>:<hash>`
comment and reconcile the submission before any administrative repair. Do not
delete the claim merely to make the command run again.

## What this does not approve

- **5–7-object evaluation:** repeatability remains unresolved for fast collection.
- **Ada fused walls:** failed bit-exact check; the successful Ada pilot used
  unfused walls. This entry point does not launch Ada.
- **Active dragging:** this remains outside the passive collection contract.
- **Dataset release:** outputs retain `accepted_for_training: false`. A final
  combined manifest/split review and training smoke test remain separate from
  collection. Do not equate process exit zero with a released dataset.
- **Arbitrary deployments:** paths, environments, assets, scheduler access and
  shared-registry permissions must be present and pass preflight.

## Validation of this handoff

The operator has 12 CPU-only regression tests for read-only planning, concurrent
disjoint reservations, atomic no-overwrite receipts, idempotent submission,
ambiguous-submission protection, scheduler-only testing, continuation and status
counts. The MPM suite has 137 passing tests including parameterized cycle bounds
and the four-lane first-wave/stop orchestration checks. The pinned physical
source closure is unchanged from the production bundle.

Real operator validation also exercises preflight, creation/verification of an
isolated temporary run bundle, Slurm `--test-only`, and CPU preparation of its
first real 60-case training cycle. This validates the launch path without
claiming another GPU qualification or submitting duplicate collection jobs.

Developer/operator test command (no Slurm submission):

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s collection/mpm/tests -p 'test_mpm_collect.py' -v
```
