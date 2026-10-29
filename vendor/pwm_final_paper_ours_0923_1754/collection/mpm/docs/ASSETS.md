# Shared MPM assets and independent multi-user collection

The simulation-ready bank is published on the lab SSD:

```
/data/group_data/katefgroup-ssd/world_model/particle_world_model/assets/mpm/fresh-10k-v1
```

| Directory | Prepared objects | Role |
|---|---:|---|
| `train/` | 10,000 | Training geometry |
| `validation/` | 258 | Held-out validation geometry |
| `test/` | 262 | Held-out test geometry |

Each role has `MANIFEST.json`, indexing its particle NPZ files by stable asset
UID. These are 1,024-particle simulation assets, not a newly converted set of
meshes. `SPLITS.json` and `pool/` preserve the existing frozen object-level
assignment. Five protected source assets were previously excluded during bank
preparation; publishing does not reinstate them or change membership.

The published 711 MiB bank is byte-for-byte identical to the original prepared
bank. Payload and manifest checksums were verified before and after copying;
prepared train, validation and test UID sets are pairwise disjoint.

`COMPLETE.json` SHA-256:

```
6aedbb02b658ff09699a2aa0ec8b5b11103975d74c50f822dcaf559929e09922
```

`SHARED_ASSETS.json` records the publication and original source. Original
lineage strings can mention old storage paths; loading the prepared bank uses
the relative `file` entries and local pinned pool/split files, not those old
conversion inputs.

## Access policy

**Lab group: `kfragki2_group` (GID 60074).** Shared assets and launcher/reference
artifacts are group-readable and directories group-traversable, with no world
access. Assets are read-only to group members; the publishing owner retains
maintenance rights. No access to unrelated home/worktree files is granted.

Group membership and scheduler association are different. A collaborator not
in `kfragki2_group` must request membership/access through the lab administrators.
Every user also needs their own permitted Slurm account/QoS. Do not use somebody
else's credentials or broaden home-directory permissions to work around this.

## Common launcher

The shared operator entry point is published under:

```
/data/group_data/katefgroup-ssd/world_model/particle_world_model/tools/mpm-v4/mpm_collect.py
```

Use it from any cluster directory:

```bash
python3 /data/group_data/katefgroup-ssd/world_model/particle_world_model/tools/mpm-v4/mpm_collect.py assets
python3 /data/group_data/katefgroup-ssd/world_model/particle_world_model/tools/mpm-v4/mpm_collect.py preflight
python3 /data/group_data/katefgroup-ssd/world_model/particle_world_model/tools/mpm-v4/mpm_collect.py launch --name alice-night-one
```

`launch` defaults to a real **four-PRO6000, 48-hour, 1–4-object training** job
(v4 also accepts an explicit two-PRO configuration). A
second operator uses another run name, e.g. `bob-night-one`. Full instructions
are in the shared `RUNBOOK.md` and [the operator runbook](RUNBOOK.md).
`READY.json` is the publication completion receipt; absence means the shared
runtime publication has not finished and must not be treated as launch-ready.

## How sequences differ between people

All users use the **same registry**:

```
/data/group_data/katefgroup-ssd/world_model/particle_world_model/staging/mpm-operator-registry-v1
```

The launcher takes an exclusive filesystem lock and atomically reserves a new
2,000-cycle range. The initial blocks are `[84000,86000)`, `[86000,88000)`, etc.
They avoid the pre-existing campaign ranges. The current pinned sampler ends
at 100000; exhausted reservation space fails closed rather than wrapping.

Within a cycle, the existing sampler derives sequence keys from the cycle,
object count, stiffness phase and action family. Thus operators share the same
shape bank but receive different deterministic initialization/sequence keys.
This is not a guarantee that two runs never reuse a shape—shape reuse is
intentional. The exported training sequence keys do not overlap. The matched
positive/negative functionality controls are intentionally repeated and are
never exported as training examples.

Reservations are committed group-readable (`0640`); the registry lock is
group-readable/writable for exclusive locking on NFS. The shared parent directories use setgid
and sticky permissions so users can create their own runs without deleting
another UID's top-level records. Only the run's owner can submit it through
the front end. Repeating a successful launch name returns its job ID instead
of making a duplicate submission. Ambiguous submission errors retain the claim
and require reconciliation; they are never automatically retried.

Do not create a private copy of the registry, delete its reservations, manually
edit cycle ranges, or launch a copied sbatch with a reused bundle. Those actions
bypass the cross-user uniqueness guarantee.

**Readable test assets do not mean test-generation approval.** Training remains
restricted to the training bank; validation/test membership does not change.
Fast 5–7-object evaluation and active dragging remain separate capability gaps.

## Publication verification (2026-09-13)

The shared `READY.json` has been written. Verified:

- All bank payload/manifest hashes survived publication, with 10,000 / 258 /
  262 prepared assets and no cross-split UID overlap.
- The isolated shared Python runtime imports the required dependencies with
  versions matching the original environment. Its executable and all parent
  directories have lab-group access; no private home needs to be opened.
- Shared-launcher `assets` and `preflight` pass using actual host permissions.
  Slurm `--test-only` accepts the shared four-PRO launch command. This is a
  scheduler validation, **not a submitted collection job**.
- Thirteen CPU-only operator regression tests pass, including restrictive
  umasks, disjoint reservations, duplicate-submit prevention and bounded seeds.
- Four separate processes acquired the lock on the actual shared NFS volume
  and reserved four distinct ranges in an isolated test registry. Production
  reservations were untouched. Receipt:
  `/data/group_data/katefgroup-ssd/world_model/particle_world_model/staging/mpm-shared-lock-check-ns8wiv1c/CHECK.json`.
- Preparing 60 coverage cases at cycle 84000 and another 60 at 86000 using the
  shared banks produced 120 unique sequence keys with zero overlap; all 60
  paired particle-position initial states differed. Repeated matched controls
  were excluded from this comparison. This validates initialization diversity,
  not a new simulation-accuracy benchmark.

No other user's login was used: cross-user access is established by checking
the real lab-group ownership/modes and ancestor traversal, not by an executed
second-UID test. Collaborators should run `preflight` under their own account.
The current reservation space supports eight 2,000-cycle runs before requiring
a reviewed sampler-range extension. Do not bypass that limit.

The shared environment has not received a separate GPU qualification in this
publication task. Each new job still must pass the normal GPU startup gate
before production collection. Existing running/queued jobs were not modified.
