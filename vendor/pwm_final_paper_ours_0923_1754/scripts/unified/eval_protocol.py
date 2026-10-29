"""The evaluation protocol of pwm-data-v1 in one command (docs/EVAL_PROTOCOLS.md).

    PYTHONPATH=. python scripts/unified/eval_protocol.py --checkpoint <run>/checkpoints/final.pt

scores a checkpoint, a specialist or a unified model alike, on every dataset it was trained on: the dataset's
CURRENT frozen case list (CURRENT below), every case set of it, one `eval_unified.py --cases` command per set.
What changes a number is fixed here and is not an option: seed 0, the checkpoint's own window, sampler steps and
EMA weights, given actions, no clamping, the whole case set. So two people who run this on the same checkpoint
get the same receipts. The one protocol setting is the sampling schedule:

    --schedule pyramid | full_sequence     default: the checkpoint's own (`df_schedule`)

Each schedule has its own output folder and is named in every receipt and summary, so the two never mix.

Every rollout is scored to the end of its clip: `rollouts_full` for the long recordings (folding, yikai, dex),
`rollouts_first` for the others, whose clips that set already covers whole. That set is the dataset's headline.
A receipt holds, on the predicted rows: the Euclidean error in mm (mean, final, per frame), its square (MSE, mm^2),
the Chamfer distance per frame, the passive bodies' pose error, and the two trivial predictions (no motion, held
velocity) scored by the same scorer on the same clips. Its sidecar `<set>.cases.json` holds the curves of every
single case, and `<set>.predictions/` the predicted positions of every case (--predictions headline | all | none;
default headline: the whole-clip rollouts), for later visualization and for scoring other methods on the same cases.

Output, default <run>/eval_protocol/<checkpoint name>_<schedule>/:
    <dataset>/<case set>.json, .cases.json, .log, .predictions/    one evaluator receipt per case set
    SUMMARY.json, SUMMARY.md                                       every set next to its trivial references
--report_folders also scores the MPM report folders (test_seen, test_count_seen, test_count_unseen).
--datasets and --sets restrict a run (the summary is then marked partial); --resume keeps receipts that match this
checkpoint, case list, set and schedule; --dry_run prints the commands.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MANIFESTS = REPO / "data" / "manifests" / "eval_cases"
# dataset folder of pwm-data-v1 -> its current case list. A new list version changes this table in the same commit.
CURRENT = {"mpm_fresh": "mpm_fresh_v1.json", "mpm_drag": "mpm_drag_v1.json", "folding_8mm": "folding_8mm_v3.json",
           "pushing": "pushing_v2.json", "rl_push": "rl_push_v2.json", "yikai": "yikai_v3.json",
           "dex_v2_filtered2": "dex_v2_filtered2_v2.json", "pickup": "pickup_v1.json"}
REPORT_FOLDERS = {"mpm_fresh": {"test_seen": "mpm_fresh_test_seen_v1.json", "test_count_seen": "mpm_fresh_test_count_seen_v1.json",
                                "test_count_unseen": "mpm_fresh_test_count_unseen_v1.json"},
                  "mpm_drag": {"test_seen": "mpm_drag_test_seen_v1.json"}}
SCHEDULES = ("pyramid", "full_sequence")
CURVE_FRAMES = (16, 32, 64, 128, 256, 512)               # frames of the headline curve shown in SUMMARY.md
REFERENCE_TOLERANCE_MM = 0.01                            # the evaluator's zero-motion number against the manifest's


def headline_set(sets):
    """The rollout set that reaches the end of every clip."""
    return "rollouts_full" if "rollouts_full" in sets else "rollouts_first"


def checkpoint_datasets(cfg):
    """[(domain, dataset folder, validation folder)] of a checkpoint's configuration: the folder it validated on
    names the dataset, so dex on dex_v2_filtered2 and folding on folding_8mm find their own case lists."""
    from data.unified_dataset import selected_domains
    out = []
    for domain in selected_domains(cfg.data.get("domains"), cfg.data.get("domain_weights"))[0]:
        entry = (cfg.data.get("roots") or {}).get(domain)
        folder = entry.get("val") if entry is not None else None
        if not folder:
            raise ValueError(f"The checkpoint names no validation folder for {domain}: the protocol covers runs on a "
                             "pwm-data-v1 release (data.roots.<domain>.val = <release>/<dataset>/test)")
        folder = Path(folder if isinstance(folder, str) else list(folder)[0])
        out.append((domain, folder.parent.name, folder))
    return out


def plan(checkpoint, datasets, out_dir, schedule, *, report_folders=False, only_datasets=None, only_sets=None,
         predictions="headline", python=sys.executable):
    """One entry per evaluator command: dataset, folder, case set, manifest, receipt, words."""
    from utils.eval_cases import load_cases
    if schedule not in SCHEDULES or predictions not in ("headline", "all", "none"):
        raise ValueError(f"schedule is one of {SCHEDULES}, predictions one of headline, all, none")
    entries = []
    for domain, dataset, val_folder in datasets:
        if only_datasets and dataset not in only_datasets:
            continue
        if dataset not in CURRENT:
            raise ValueError(f"No current case list for {dataset}; the protocol covers {sorted(CURRENT)}")
        lists = [("test", CURRENT[dataset], None)]
        if report_folders:
            lists += [(folder, name, val_folder.parent / folder) for folder, name in REPORT_FOLDERS.get(dataset, {}).items()]
        for folder, name, test_folder in lists:
            manifest = MANIFESTS / name
            sets = load_cases(manifest)["sets"]
            for case_set, entry in sets.items():
                if only_sets and case_set not in only_sets:
                    continue
                headline = case_set == headline_set(sets)
                receipt = Path(out_dir) / (dataset if folder == "test" else f"{dataset}.{folder}") / f"{case_set}.json"
                words = [python, "scripts/unified/eval_unified.py", "--checkpoint", str(checkpoint), "--domains", domain,
                         "--cases", str(manifest), "--case_set", case_set, "--clips", "all", "--seed", "0",
                         "--schedule", schedule, "--out", str(receipt)]
                if case_set.startswith("rollouts"):
                    words += ["--rollout", "--rollout_frames", str(int(entry["summary"]["horizon_max"]))]
                if test_folder is not None:
                    words += ["--test_folder", f"{domain}={test_folder}"]
                if predictions == "all" or (predictions == "headline" and headline):
                    words += ["--save_predictions", str(receipt.with_suffix(".predictions"))]
                entries.append(dict(dataset=dataset, domain=domain, folder=folder, case_set=case_set, manifest=str(manifest),
                                    headline=headline, receipt=str(receipt), words=words, reference=entry["summary"]))
    if not entries:
        raise ValueError("Nothing to score: check --datasets and --sets")
    return entries


def _matches(receipt, entry, checkpoint_sha, manifest_sha, schedule):
    """A kept receipt (--resume) must be this checkpoint on this case list, set and schedule, with its sidecar."""
    path = Path(receipt)
    if not path.is_file() or not path.with_suffix(".cases.json").is_file():
        return False
    value = json.loads(path.read_text())
    cases = value.get("cases", {})
    return (value.get("checkpoint_sha256") == checkpoint_sha and cases.get("manifest_sha256") == manifest_sha
            and cases.get("case_set") == entry["case_set"] and value.get("seed") == 0 and value.get("df_schedule") == schedule)


def summarize(entries):
    """Rows of the summary: every case set next to the trivial references the same scorer gave on the same clips."""
    rows = []
    for entry in entries:
        receipt = json.loads(Path(entry["receipt"]).read_text())
        metrics = receipt[entry["domain"]]
        model, reference = metrics["rows"]["predicted"], metrics["reference"]
        zero, hold = reference["zero"]["rows"]["predicted"], reference["hold"]["rows"]["predicted"]
        frozen = entry["reference"]["zero_predicted_mm"]
        row = dict(dataset=entry["dataset"], folder=entry["folder"], case_set=entry["case_set"], headline=entry["headline"],
                   cases=receipt["cases"]["cases"], manifest=Path(entry["manifest"]).name,
                   manifest_sha256=receipt["cases"]["manifest_sha256"], schedule=receipt["df_schedule"],
                   horizon_frames=[metrics["horizon_frames_min"], metrics["horizon_frames_max"]],
                   model_mean_mm=model.get("particle_mean_mm"), model_final_mm=model.get("particle_final_mm"),
                   model_mse_mm2=model.get("particle_mse_mm2"), model_chamfer_mm=model.get("chamfer_mean_mm"),
                   zero_mean_mm=zero.get("particle_mean_mm"), zero_final_mm=zero.get("particle_final_mm"),
                   zero_mse_mm2=zero.get("particle_mse_mm2"), zero_chamfer_mm=zero.get("chamfer_mean_mm"),
                   hold_mean_mm=hold.get("particle_mean_mm"), hold_final_mm=hold.get("particle_final_mm"),
                   moving_mean_mm=metrics["rows"]["moving"].get("particle_mean_mm"),
                   still_mean_mm=metrics["rows"]["still"].get("particle_mean_mm"),
                   pose_passive_mm=metrics.get("pose_passive_mm"), pose_passive_deg=metrics.get("pose_passive_deg"),
                   zero_pose_passive_mm=reference["zero"].get("pose_passive_mm"),
                   reference_matches_manifest=(frozen is None or zero.get("particle_mean_mm") is None
                                               or abs(zero["particle_mean_mm"] - frozen) <= REFERENCE_TOLERANCE_MM),
                   predictions=receipt["cases"].get("predictions"), receipt=str(Path(entry["receipt"])))
        if entry["headline"]:                             # the per-frame curve at a few frames, model against no motion
            curve, base, counts = (model.get("particle_per_frame_mm") or [], zero.get("particle_per_frame_mm") or [],
                                   model.get("particle_per_frame_n") or [])
            row["curve"] = [dict(frame=f, clips=counts[f - 1], model_mm=curve[f - 1], zero_mm=base[f - 1])
                            for f in sorted({*CURVE_FRAMES, len(curve)}) if 0 < f <= len(curve)]
        rows.append(row)
    return rows


def render(summary):
    """SUMMARY.md: the headline rollouts first, then every set. Errors are on the predicted rows."""
    number = lambda v: "" if v is None else f"{v:.2f}"
    header = ["| dataset | folder | set | cases | frames | model mean mm | model final mm | no motion mean mm | no motion final mm | "
              "held velocity mean mm | model MSE mm2 | model Chamfer mm | no motion Chamfer mm | pose mm | pose deg |",
              "|" + "---|" * 15]
    cells = lambda r: (f"| {r['dataset']} | {r['folder']} | {r['case_set']} | {r['cases']} | {r['horizon_frames'][0]} to "
                       f"{r['horizon_frames'][1]} | {number(r['model_mean_mm'])} | {number(r['model_final_mm'])} | "
                       f"{number(r['zero_mean_mm'])} | {number(r['zero_final_mm'])} | {number(r['hold_mean_mm'])} | "
                       f"{number(r['model_mse_mm2'])} | {number(r['model_chamfer_mm'])} | {number(r['zero_chamfer_mm'])} | "
                       f"{number(r['pose_passive_mm'])} | {number(r['pose_passive_deg'])} |")
    lines = [f"# Evaluation protocol: {summary['checkpoint']}", "",
             f"Checkpoint sha256 `{summary['checkpoint_sha256'][:12]}`, code `{summary['code']}`, seed 0, schedule `{summary['schedule']}`"
             + ("" if summary["schedule"] == summary["checkpoint_schedule"] else f" (the checkpoint's own is `{summary['checkpoint_schedule']}`)")
             + ". " + ("This is `final.pt`, the reported checkpoint." if summary["reported_checkpoint"] else
                       "NOT `final.pt`: a monitoring number, not a reported one.")
             + (" PARTIAL run: not every dataset or case set was scored." if summary["partial"] else ""), "",
             "## Whole-clip rollouts (the headline of every dataset)", "", *header]
    lines += [cells(r) for r in summary["sets"] if r["headline"]]
    lines += ["", "Error along the rollout, model / no motion, mm (clips that reach the frame):", ""]
    for r in summary["sets"]:
        if r["headline"] and r.get("curve"):
            points = ", ".join(f"frame {c['frame']}: {c['model_mm']:.2f} / {c['zero_mm']:.2f} ({c['clips']})" for c in r["curve"])
            lines.append(f"- {r['dataset']} {r['folder']}: {points}")
    lines += ["", "## Every case set", "", *header] + [cells(r) for r in summary["sets"]]
    bad = [r for r in summary["sets"] if not r["reference_matches_manifest"]]
    lines += ["", "No-motion check against the frozen lists: " + ("every set matches." if not bad else
              "MISMATCH in " + ", ".join(f"{r['dataset']} {r['case_set']}" for r in bad) + ": the clips read differ from the frozen ones.")]
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="<run>/checkpoints/final.pt is the reported checkpoint")
    p.add_argument("--schedule", default=None, choices=SCHEDULES, help="sampling schedule; default: the checkpoint's own")
    p.add_argument("--out", default=None, help="default: <run>/eval_protocol/<checkpoint name>_<schedule>/")
    p.add_argument("--predictions", default="headline", choices=("headline", "all", "none"),
                   help="which case sets also save every case's predicted positions (default: the whole-clip rollouts)")
    p.add_argument("--report_folders", action="store_true", help="also the MPM report folders")
    p.add_argument("--datasets", default=None, help="comma-separated dataset folders; the summary is then marked partial")
    p.add_argument("--sets", default=None, help="comma-separated case sets; the summary is then marked partial")
    p.add_argument("--resume", action="store_true", help="keep receipts that match this checkpoint, case list, set and schedule")
    p.add_argument("--dry_run", action="store_true", help="print the evaluator commands and stop")
    args = p.parse_args()

    from utils.checkpoint import checkpoint_config, read_checkpoint
    from utils.reproducibility import code_revision, file_sha256
    checkpoint = Path(args.checkpoint).resolve()
    cfg = checkpoint_config(read_checkpoint(str(checkpoint)))
    own = str(cfg.get("df_schedule", "pyramid"))
    schedule = args.schedule or own
    out_dir = Path(args.out) if args.out else checkpoint.parent.parent / "eval_protocol" / f"{checkpoint.stem}_{schedule}"
    split = lambda value: [v.strip() for v in value.split(",") if v.strip()] if value else None
    entries = plan(checkpoint, checkpoint_datasets(cfg), out_dir, schedule, report_folders=args.report_folders,
                   only_datasets=split(args.datasets), only_sets=split(args.sets), predictions=args.predictions)
    if args.dry_run:
        for entry in entries:
            print("PYTHONPATH=. " + " ".join(entry["words"]))
        return
    checkpoint_sha = file_sha256(str(checkpoint))
    for i, entry in enumerate(entries, 1):
        receipt = Path(entry["receipt"])
        label = f"[{i}/{len(entries)}] {entry['dataset']} {entry['folder']} {entry['case_set']}"
        if args.resume and _matches(receipt, entry, checkpoint_sha, file_sha256(entry["manifest"]), schedule):
            print(f"{label}: kept {receipt}", flush=True)
            continue
        receipt.parent.mkdir(parents=True, exist_ok=True)
        started = time.time()
        print(f"{label}: scoring ...", flush=True)
        with open(receipt.with_suffix(".log"), "w") as log:
            done = subprocess.run(entry["words"], cwd=REPO, env={**os.environ, "PYTHONPATH": str(REPO)},
                                  stdout=log, stderr=subprocess.STDOUT)
        if done.returncode:
            raise SystemExit(f"{label}: the evaluator failed, see {receipt.with_suffix('.log')}")
        print(f"{label}: done in {(time.time() - started) / 60:.1f} min", flush=True)
    summary = dict(checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_sha, reported_checkpoint=checkpoint.name == "final.pt",
                   code=code_revision().get("git_commit", "")[:7], seed=0, schedule=schedule, checkpoint_schedule=own,
                   partial=bool(args.datasets or args.sets), report_folders=bool(args.report_folders), sets=summarize(entries))
    (out_dir / "SUMMARY.json").write_text(json.dumps(summary, indent=1, allow_nan=False) + "\n")
    (out_dir / "SUMMARY.md").write_text(render(summary))
    print(f"wrote {out_dir / 'SUMMARY.md'}")
    if any(not row["reference_matches_manifest"] for row in summary["sets"]):
        raise SystemExit("The evaluator's no-motion number differs from a frozen list: the clips read are not the frozen ones")


if __name__ == "__main__":
    main()
