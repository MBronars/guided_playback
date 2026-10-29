# Additional dragging collector: dev-blackwellb

User-authorized independent collection, September 16, 2026.

## Launch

- Allocation **10431325**, `dev-blackwellb`, node **babel-z9-16**.
- Collection step **10431325.424**, started **18:28 UTC / 14:28 EDT**.
- Two PRO6000 Blackwell Server Edition GPUs; 16 CPUs and 48 GiB for this step.
- Both allocated GPUs were idle with no compute processes before launch:
  `GPU-4d27c47e-22f1-6713-b0e2-d90fd67d6bc0` and
  `GPU-e693ac5b-80dc-1ad2-5590-588044e03e88`.
- Allocation expires **September 16, 20:19:35 EDT**. Step time is 349 minutes,
  bounded inside that allocation. Collector uses the earlier allocation deadline
  with its existing five-minute margin and one-hour admission/finalization reserve.
  This is **not** a fresh 48-hour allocation.
- Bundle cap: 1,000 attempts, six hours; these are limits, not a yield estimate.
- Existing dev-blackwellf collection **10447018.60** was not changed and was
  verified live after this launch.

At **18:31–18:32 UTC**, both fresh-device startup fault/isolation probes had
passed, both first simulation groups were running, and measured GPU utilization
was **97% / 78%** during startup. No worker failure receipts or scheduler errors
were present. First complete clips/TAR blocks had not yet been observed; this is
verified startup, not a claim of completed collection or continuous monitoring.

## Source, identity and outputs

Created through the verified frozen implementation under
`staging/mpm-drag-gentle-v4-restart-20260915/source`, using its unchanged gate,
shared interpreter, immutable shape bank and shared namespace registry. No
working-tree numerical edits were imported. Same g256 gentle-v4 protocol,
1–4 train objects, homogeneous elastic/plasticine, no sand, and all existing
quality/consumer checks. Each fresh GPU runs the normal startup fault probe.

Namespace: `drag-blackwellb-sep16-f2b2937842ad42799498fce7b189e165`.
Bundle SHA-256: `5a376992ea993863ea881b7993c855cca4e7067a15dbe2294a125d9a78def8e4`.
Distinct namespaces change seeds and shape ordering; shape reuse is expected.

```text
/data/group_data/katefgroup-ssd/world_model/particle_world_model/staging/mpm-drag-blackwellb-20260916/
  BUNDLE.json
  GATE.json
  SRUN_SUBMISSION_CLAIM.json
  source/
  collection/block-*/
  data/train/train/block-*/diagnostic-export/*.tar
  export-review/             # NEVER include this in training
  lane-0.log
  lane-1.log
  srun.err
```

Do not relaunch or copy this consumed bundle. The local attach script and
submission receipt are in `runs/mpm-drag-blackwellb-20260916/`.

For read-only counts, run from this bundle's `source/` with the shared interpreter:

```bash
python -m collection.mpm.drag.launch status --bundle /data/group_data/katefgroup-ssd/world_model/particle_world_model/staging/mpm-drag-blackwellb-20260916
squeue --steps -j 10431325
```

Status receipts do not prove GPU liveness. Inspect per-block `STARTUP.json`,
raw completions, lane logs and failure receipts too. As with the other run,
isolated export failures quarantine outputs and continue; repeated export
failures and simulation failures still stop safely.

Training field semantics and split limitations: [dataset guide](DRAG_DATASET_FORMAT.md).
