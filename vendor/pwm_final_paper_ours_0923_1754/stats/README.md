# Normalization

## Files in this folder

| File | What it is | Who reads it |
|---|---|---|
| `normalization.json` | The frozen four-domain specification (`frozen-xy-p99-2026-08-28`): mpm, pickup, cloth, pushing, both profiles. Never edited. | The default of every config; tests pin its digest. |
| `normalization_collections_20260919_v7.json` | The **one current scales artifact for the pwm-data-v1 release**: the frozen entries unchanged except pushing, which sits on ONE scale for all axes (2026-09-19: a planar task, its vertical and tilt scales were the p99 of motion that is almost always zero), plus rl_push, dex, folding, mpm_drag, mpm_fresh and yikai, each fit on the release's train folder. Byte-identical to the release's copy, which the release's `RELEASE.json` pins. | Every pwm-data-v1 recipe (through the release's copy). |
| `normalization_rl_push.json` | The rl_push entry of the 2026-09-14 rl_push release. | `launch/legacy/train_rl_push_unet.sbatch` (the 2026-09-15 specialist), the collection tests, `verify_collections.py` as the base of an assembled artifact. |
| `normalization_water3d.json` | The frozen specification plus a `particle.water` entry. | Opt-in water runs. |

**One current collections artifact.** A refit replaces the collections file: the new version is added and the one before it is
removed in the same commit, so the tree never holds a chain of versions. A checkpoint carries its complete resolved snapshot and
never needs the file it was trained with. Superseded versions `collections-xy-p99-2026-09-16-v1` to `-2026-09-18-v5` and the fit
record `normalization_rl_push_20260918.json` (its two rl_push entries are the ones in v6) were removed from the tree on 2026-09-19;
they are in git history (last present at `58ade7c`). v6 was removed on 2026-09-19 when v7 replaced it (last present at `c056d6a`).
The release's `stats/` folder holds only v7 since 2026-09-19: v2 to v6 moved, unchanged, to the release's
`_backup/stats_superseded_20260919/`, because no recipe of the launch plan reads them. `docs/DATASETS.md` records each version by artifact version and sha256.

The active unified system uses **one versioned specification**,
[`normalization.json`](normalization.json), with two independently selectable
experimental profiles. Neither profile is declared the scientific winner.

| Profile | Motion scales | Meaning |
|---|---:|---|
| `xy_p99_per_domain` | 20 | Two particle scales for each of four domains; four pose scales for each of three rigid-body domains. |
| `xy_p99_global` | 4 | Two translation scales shared by particles and body poses, plus two rotation scales; fixed across all domains. |

Particle arrays store `[translation_xy, translation_z]`. Pose arrays store
`[translation_xy, translation_z, rotation_xy, rotation_z]`. Bounds are symmetric;
X and Y share a scale. Translation is **metres per stored frame**, rotation is
**radians per stored frame**, not per second. All profiles retain the common
position bounds of +/-0.8 metres. Domain-specific canonical shifts remain a
separate dataset contract.

The four global values are componentwise **maxima over the fixed four-domain
reference**, including both particle and pose translation. They are not a
pooled-data percentile and must not be recomputed when selecting a specialist.

In the frozen bundle, MPM already supplies the largest translation bounds, so its
particle and position normalizers are bitwise identical under both profiles.
An MPM-only model therefore has the same normalization math in either profile;
joint training still changes because the other domains' scales differ. A newly
fitted candidate need not preserve this MPM equivalence.

## Selection and checkpoint continuation

```yaml
normalization:
  profile: xy_p99_per_domain  # or xy_p99_global
```

An audited candidate can be selected explicitly with an additional `artifact`
JSON path. Candidates use a distinct `artifact_version`; nothing automatically
fits statistics during training or evaluation.

[`data/normalization.py`](../data/normalization.py) supplies one resolver:

```python
snapshot = resolve_normalization({"profile": "xy_p99_global"})
particle, position, pose = domain_normalizers(snapshot, "pickup")
```

Unified dataset constructors and loaders accept only `normalization=` (a profile,
artifact selector, or snapshot). The obsolete `stats_location`, `stats_override`
and `stats_set` adapter options fail explicitly: particle and pose scales cannot
be selected through separate paths. Historical URI aliases remain accepted as
the value of `normalization`, and checkpoint loading migrates their old config
field into a complete snapshot.

The resolved snapshot contains numerical values, units, profile, schema/artifact
versions, provenance and a SHA-256 checksum. It is JSON-serializable and can be
loaded without any normalization files. Training checkpoints retain this
snapshot; resume and evaluation must use it rather than silently substituting
the latest bundle. A checksum detects accidental mutation, not malicious
rewriting by somebody able to recompute the checksum.

The aliases `unified://xy_p99` and `unified://xy_p99_global` map exactly to the
two retained profiles. The old literal **`per-domain` is a different baseline**;
it is rejected with reproduction instructions, never remapped to xy-p99.

## Frozen migration, not a refit

The bundled `frozen-xy-p99-2026-08-28` values reconstruct all motion bounds of
the seven historical per-domain and seven global NPZ assets bit-for-bit as
float32. The normalizer retains the original min/max arithmetic, epsilon and
caller-controlled clipping. Simplifying the packaging does not change those
numerics or demonstrate that the historical fit was statistically sound.

Historical assets and their manifest are recoverable at commit `55c386c`; the
pre-unified-only branch at `c5fac33` also preserves the specialist workflows.
The original fitter had nondeterministic sampling and incomplete cache identity.
A fresh audited candidate must therefore have separate provenance and a new
version, not claim that importing old numbers repaired their derivation.

## Calibration

The single entrypoint is `scripts/compute_stats.py`, with explicit `collect`,
`fit` and `audit` stages. Collection consumes raw physical quantities, independent
of normalization assets. It records training sources and exclusions, actual
episode/window identities, context/horizon, window policy and seed. Cache identity
includes those inputs and the implementation contract.

```bash
python scripts/compute_stats.py collect --output-dir validation/calibration \
  --context-len 4 --future-len 16 --max-records 32 --reservoir-rows 100000
# Use the collection-<hash>.json path printed by collect:
python scripts/compute_stats.py fit --collection PATH_TO_COLLECTION_JSON \
  --out validation/candidate-normalization.json --version candidate-c4f16-seed0-v1
python scripts/compute_stats.py audit --collection PATH_TO_COLLECTION_JSON \
  --artifact validation/candidate-normalization.json --out validation/calibration-audit.json
```

Collection defaults to all four domains and both `training` and `prefix` window
policies, each from **training episodes**. A separate `collect --split val`
produces validation-only audit caches; `fit` rejects them. Each policy has a
fixed-capacity uniform reservoir of valid particle-frame and complete rigid-body
transitions. It is not an episode-uniform sample. All adjacent particle transitions
in the C+F window are collected; body-pose targets cover its future F transitions.
The fitter takes maxima of per-policy p99 bounds, not percentiles of pooled policy
rows. Zero scales fail explicitly rather than inventing a numerical floor.

Check actual source coverage before interpreting a candidate. Shards are visited
in a seeded permutation, but records within a shard are traversed sequentially;
a small record cap can stop after only one shard. For example, a 32-record
engineering collection covered one pickup shard and one pushing shard. Many
particle rows from those records do not establish broad episode/shard coverage.
Collection output and fit/audit receipts report records, episode windows,
contributing/recorded-visited versus available shards and recorded skips. The
ledger does not log every blocklist-filtered record or trailing skip, so these
are recorded coverage counts, not an exhaustive I/O/exclusion audit. Different
window policies can revisit the same episodes; do not add their counts as unique
corpus coverage. A bounded example command is not an endorsed production-calibration budget.

Existing verified caches can be reused; candidates, audit files and corrupt/stale
caches are never silently overwritten. Tar identities use canonical path, size
and nanosecond mtime, while exclusions/sidecars and adapter code are byte-hashed.
This is not a claim of full-content hashing of every source tar.

Per-domain20 and global4 are fitted from the same bounded calibration samples.
Physical-unit distributions, normalized magnitudes, clipping and clipped-target
reconstruction diagnostics are reported. Each audit embeds both resolved
normalization snapshots, including artifact version and checksums, alongside
cache identities; its selected bounds do not depend on the report's filename.
Independent calibration/validation
audits never fit the bounds; final-test data must remain out of fitting and selection.

Compare candidates using matched trained rollouts in physical units, not the
normalized training loss alone: scales also affect physical noise amplitude,
relative objective weighting and clipping.
