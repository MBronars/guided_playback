"""Freeze published collection membership without modifying or copying a corpus.

Creates opt-in INDEX.json files with episode-level identities, source clocks,
file identities, explicit exclusions and train/validation grouping. Existing
outputs are never replaced. Source receipts and any supplied content hashes are
preserved; --verify-hashes independently checks those bytes during preparation.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tarfile

import numpy as np

from data.indexed_episodes import INDEX_SCHEMA, source_identity, validate_index_pair
from data.npz_arrays import npz_array


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_identity(path):
    # Preserve the absolute source spelling while recording physical identity.
    # Resolving every component of 12K deeply nested NAS paths is expensive;
    # stat's device/inode also detects symlink aliases across split indices.
    path = Path(path).absolute()
    stat = path.stat()
    return dict(file=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns,
                device=stat.st_dev, inode=stat.st_ino)


def split_group(group, seed):
    digest = hashlib.sha256(f"{seed}:{group}".encode()).digest()
    return "val" if int.from_bytes(digest[:8], "big") / 2**64 < .1 else "train"


def write_indices(out, domain, records, provenance, *, seed=916, verify_hashes=False):
    out = Path(out).resolve()
    partial = out.with_name(out.name + ".partial")
    if out.exists() or partial.exists():
        raise FileExistsError(f"Refusing to replace an indexed release: {out}")
    seen, unique, duplicates = {}, [], []
    for record in records:
        key = record["episode_id"]
        if key in seen:
            previous = seen[key]
            if not record.get("raw_sha256") or record.get("raw_sha256") != previous.get("raw_sha256"):
                raise ValueError(f"Conflicting/repeated episode identity: {key}")
            duplicates.append(dict(episode_id=key, kept=previous["file"], ignored=record["file"]))
            continue
        seen[key] = record; unique.append(record)
    checks = {r["file"]: r["sha256"] for r in unique if r.get("sha256")}
    if verify_hashes:
        with ThreadPoolExecutor(8) as pool:
            for path, found in zip(checks, pool.map(sha256, checks)):
                if found != checks[path]:
                    raise ValueError(f"Source content checksum mismatch: {path}")
    splits = {"train": [], "val": [], "test": []}
    for record in unique:
        record = dict(record)
        split = "test" if record.pop("official_test", False) else split_group(record["group"], seed)
        splits[split].append(record)
    if not splits["train"] or not splits["val"]:
        raise ValueError("Need nonempty train and validation groups; no automatic split fallback")
    if {r["group"] for r in splits["test"]} & {r["group"] for r in splits["train"] + splits["val"]}:
        raise ValueError("Official test groups overlap development/train groups")
    if ({source_identity(r) for r in splits["test"]}
            & {source_identity(r) for r in splits["train"] + splits["val"]}):
        raise ValueError("Official test physical source episodes overlap development/train episodes")
    common = dict(schema=INDEX_SCHEMA, domain=domain, created_utc=datetime.now(timezone.utc).isoformat(),
        split_policy=dict(seed=seed, validation_fraction=.1,
                          selection="sha256 of whole group; retained official test episodes are never reassigned"),
        source_content_hashes_verified=bool(verify_hashes and len(checks) == len({r['file'] for r in unique})),
        source_hashes_available=len(checks), source_files_verified=len(checks) if verify_hashes else 0,
        duplicate_episodes=duplicates, provenance=provenance)
    for split, rows in splits.items():
        if rows:
            directory = partial / split
            directory.mkdir(parents=True)
            (directory / "INDEX.json").write_text(json.dumps(dict(common, split=split,
                records=sorted(rows, key=lambda r: r["episode_id"])), indent=2) + "\n")
    validate_index_pair(partial / "train", partial / "val", domain)
    summary = {s: dict(episodes=len(rows), groups=len({r['group'] for r in rows}),
                       saved_states=sum(r['frames'] for r in rows)) for s, rows in splits.items()}
    (partial / "SUMMARY.json").write_text(json.dumps(dict(summary=summary, **common), indent=2) + "\n")
    partial.rename(out)
    print(json.dumps(dict(domain=domain, output=str(out), summary=summary), indent=2), flush=True)


def mpm_records(roots, drag=False):
    records, receipts, shards = [], {}, {}
    for root in map(Path, roots):
        if drag:
            for block in sorted(root.glob("block-*")):
                complete = block / "COMPLETE.json"
                export = block / "diagnostic-export" / "COMPLETE.json"
                if not complete.is_file() or not export.is_file():
                    continue
                receipt = json.loads(export.read_text())
                receipts[str(complete)] = sha256(complete)
                receipts[str(export)] = sha256(export)
                review = block / "LOADER_REVIEW.json"
                if review.exists():
                    receipts[str(review)] = sha256(review)
                for artifact in receipt["artifacts"]:
                    if artifact["path"].endswith(".tar"):
                        shards[str(export.parent / artifact["path"])] = artifact["sha256"]
        else:
            from scripts.unified.build_mpm_release import published_chunks
            for chunk, receipt in published_chunks(root):
                receipts[str(chunk / "MANIFEST.json")] = sha256(chunk / "MANIFEST.json")
                for name, digest in receipt["shards"].items():
                    shards[str(chunk / name)] = digest
    if not shards:
        raise ValueError("No completed published MPM shards")
    for path, digest in sorted(shards.items()):
        identity = file_identity(path)
        with tarfile.open(path) as archive:
            for member in archive:
                if not member.name.endswith(".provenance.json"):
                    continue
                rows = json.load(archive.extractfile(member))
                key = member.name[:-len(".provenance.json")]
                for row, provenance in enumerate(rows):
                    if provenance.get("task") != ("drag" if drag else "passive"):
                        raise ValueError("MPM task does not match selected domain")
                    if drag and provenance.get("action_schema") != "mpm-drag-command-sequence-v4":
                        raise ValueError("This preparation recipe selects gentle-v4 dragging only")
                    frames = len(provenance["state_time_s"])
                    if frames < 20:
                        raise ValueError("Published MPM episode too short")
                    records.append(dict(identity, episode_id=provenance["episode_id"],
                        group=provenance["episode_id"], key=key, row=row, frames=frames,
                        sha256=digest, raw_sha256=provenance.get("raw_sha256"),
                        asset_ids=provenance.get("asset_ids", []), dt=.0417,
                        action_schema=provenance.get("action_schema")))
    return records, dict(source_receipts=receipts, roots=roots,
        validation_scope="new episodes from the collected training asset population; not novel-shape validation",
        selection="completed published shards only; deduplicate stable episode_id/raw_sha256")


def folding_records(indices):
    records, receipts = [], {}
    for index in map(Path, indices):
        j = json.loads(index.read_text()); receipts[str(index.resolve())] = sha256(index)
        if j["native_fps"] != 30 or j["split"] not in ("train", "test"):
            raise ValueError("Expected final retained 30 Hz train/test folding indices")
        for row in j["records"]:
            template = Path(row["file"]).parent / "gripper-template-1000.npz"
            ti = file_identity(template)
            records.append(dict(file_identity(row["file"]), episode_id=f"{row['garment_id']}/{row['world']}",
                group=row["garment_id"], frames=row["frames"], sha256=row["sha256"],
                garment_id=row["garment_id"], action=row["world"], particles=row["particles"],
                template_file=ti["file"], template_size=ti["size"], template_mtime_ns=ti["mtime_ns"],
                template_sha256=sha256(template), dt=1 / 30, official_test=j["split"] == "test"))
    return records, dict(source_receipts=receipts, validation_scope="whole garments reserved from official train; official test untouched",
                         selection="final retained indices only; native vertices, no second settling removal or stride")


def yikai_records(roots, audit):
    audits = {}
    if audit:
        with open(audit) as stream:
            audits = {row["npz"]: row for row in map(json.loads, stream)}
    records, receipts, skipped, seen, pending = [], {}, [], set(), []
    for root in map(Path, roots):
        for population in ("gen", "fail"):
            manifest = root / population / "manifest.jsonl"
            receipts[str(manifest)] = sha256(manifest)
            for row in map(json.loads, manifest.open()):
                path = row.get("npz")
                if not path or path in seen:
                    continue
                seen.add(path)
                a = audits.get(path, {})
                if a.get("flags"):
                    # Conservative candidate quality filter, separate from task
                    # success: the audit flags launches, teleports and sinking.
                    skipped.append(dict(file=path, reason="motion quality audit flags", flags=a["flags"])); continue
                pending.append((root, population, row, a))
    def inspect(item):
        root, population, row, a = item
        path = row["npz"]
        try:
            identity = file_identity(path)
        except FileNotFoundError:
            return None, dict(file=path, reason="manifest points to absent export")
        # The supplied audit already read every geometry array. Its immutable
        # receipt is pinned below; runtime verifies this length against the NPY
        # header whenever a window is loaded. Without that audit, inspect now.
        frames = int(a["T"]) if "T" in a else int(npz_array(path, "state/particles").shape[0])
        if frames < 20:
            return None, dict(file=path, reason="shorter than C4+F16")
        return dict(identity, episode_id=hashlib.sha256(path.encode()).hexdigest(),
                    group=row["scene"], scene=row["scene"], frames=frames, source_tag=row["tag"],
                    source_object=row.get("src"), population=population, success=row.get("episode_success"),
                    dt=.05, official_test=root.name.endswith("_eval"), audit=a), None
    with ThreadPoolExecutor(16) as pool:
        for i, (record, exclusion) in enumerate(pool.map(inspect, pending)):
            if record is not None:
                records.append(record)
            else:
                skipped.append(exclusion)
            if (i + 1) % 1000 == 0:
                print(f"[index] yikai {i + 1}/{len(pending)} episodes", flush=True)
    return records, dict(source_receipts=receipts, audit=None if not audit else dict(file=str(Path(audit).resolve()), sha256=sha256(audit)),
        exclusions=skipped, validation_scope="whole training scenes reserved for development; official wm_v2_eval remains test",
        selection="present valid-length exports including task failures, excluding all motion-quality audit flags (launch/impulse/teleport/sink); thresholds belong to the pinned audit")


def dex_geometry(path, output):
    from scripts.unified.dex_geometry import canonical_points, merge_fixed, outer_surface_points, parse_urdf, rpy_to_R
    path, output = Path(path), Path(output)
    manifest = path.with_suffix(".json")
    info = json.loads(manifest.read_text())
    urdf = info["robot_file"]
    names = [str(n) for n in npz_array(path, "link_names")]
    links, joints = parse_urdf(urdf)
    canon = canonical_points(links, urdf, total=1600, seed=0)
    order, members, parents, joint_of = merge_fixed(links, joints)
    arrays = dict(link_names=np.asarray(names), parent=np.full(len(names), -1, dtype=np.int64),
                  axis=np.zeros((len(names), 3), dtype=np.float32), pivot=np.zeros((len(names), 3), dtype=np.float32))
    for k, name in enumerate(names):
        i = order.index(name)
        arrays[f"hand_{k}"] = np.concatenate([canon[n] @ R.T + t for n, R, t in members[i] if n in canon]).astype(np.float32)
        if parents[i] >= 0:
            p = parents[i]; j = joint_of[i]
            # The joint's URDF parent may be a fixed descendant of the recorded
            # parent body: include that fixed transform in both origin and axis.
            _, Rfixed, tfixed = next(m for m in members[p] if m[0] == j["parent"])
            arrays["parent"][k] = names.index(order[p])
            axis = Rfixed @ rpy_to_R(j["rpy"]) @ j["axis"]
            arrays["axis"][k] = axis / np.linalg.norm(axis)
            arrays["pivot"][k] = tfixed + Rfixed @ j["xyz"]
    with np.load(path, allow_pickle=False) as z:
        shapes = z["episode_shape_id"]
        for shape in np.unique(shapes):
            env = int(np.flatnonzero(shapes == shape)[0])
            pieces = sorted(int(key.rsplit("_", 1)[1]) for key in z.files if key.startswith("objgeom_envs_") and env in z[key])
            # a multi-piece object's faces between pieces are inside it: sample its outer surface only
            # (a single piece samples exactly as before)
            arrays[f"object_{shape}"] = outer_surface_points(
                [(z[f"objgeom_verts_{i}"], z[f"objgeom_faces_{i}"]) for i in pieces], 512, np.random.default_rng(0)).astype(np.float32)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    np.savez(output, **arrays)
    output.with_suffix(".json").write_text(json.dumps(dict(source=file_identity(path),
        manifest_sha256=sha256(manifest), urdf=str(urdf), urdf_sha256=sha256(urdf), geometry_sha256=sha256(output),
        sampling="seed 0; hand area-weighted total 1600 with minimum 24/link; object outer surface 512 (faces between convex pieces excluded)",
        object_piece_offsets="already baked into vertices; not applied again"), indent=2) + "\n")
    return output


def dex_records(root, geometry_dir):
    records, receipts, exclusions, custom_objects = [], {}, [], False
    for path in sorted(Path(root).glob("*.npz")):
        if path.stem.endswith("_outcomes"):
            continue
        manifest = path.with_suffix(".json")
        if not manifest.is_file():
            raise ValueError(f"Dex shard missing manifest: {path}")
        receipts[str(manifest)] = sha256(manifest)
        info = json.loads(manifest.read_text())
        # the shipped screwdriver grasp file starts model 0 14.8 cm high; a shard that names its own grasp file does not
        shipped_grasps = not info.get("grasps_file")
        custom_objects |= bool(info.get("objects_file") or info.get("grasps_file"))
        # shards that record which termination ended each episode also mark Eden's solver-error resets
        reasons = info.get("done_reasons")
        solver_error = reasons.index("solver_error") if reasons else None
        end_reason = npz_array(path, "episode_end_reason") if reasons else None
        # a teacher can have several shards, named <teacher>@<part>; the object of each episode, for held-out analysis
        teacher, _, part = path.stem.partition("@")
        objects = ([Path(f).stem for f in info["object_files"]] if info.get("object_files") else
                   [f"variant {v}" for v in info["object_variants_kept"]] if info.get("object_variants_kept") else None)
        geometry = dex_geometry(path, Path(geometry_dir) / path.name)
        gi = file_identity(geometry)
        outcome_path = path.with_name(path.stem + "_outcomes.npz")
        outcomes = npz_array(outcome_path, "outcome")
        receipts[str(outcome_path)] = sha256(outcome_path)
        lengths, shapes = npz_array(path, "episode_length"), npz_array(path, "episode_shape_id")
        perturbed = npz_array(path, "episode_perturbed")
        source = file_identity(path)
        for env, native_length in enumerate(lengths):
            identity = f"{path.stem}/{env:04d}"
            reason = ("screwdriver model 0 reset geometry" if teacher.startswith("screwdriver") and shapes[env] == 0 and shipped_grasps else
                      "perturbed screwdriver-sharpa" if teacher == "screwdriver-sharpa" and perturbed[env] else
                      "ended by a simulator solver error" if end_reason is not None and end_reason[env] == solver_error else
                      "shorter than 39 native states" if native_length < 39 else None)
            if reason:
                exclusions.append(dict(episode_id=identity, reason=reason)); continue
            records.append(dict(source, episode_id=identity, group=identity, teacher=teacher, env=env,
                shape_id=int(shapes[env]), outcome=int(outcomes[env]), perturbed=bool(perturbed[env]),
                **(dict(part=part) if part else {}),
                **(dict(object=objects[int(shapes[env])]) if objects else {}),
                frames=(int(native_length) + 1) // 2, native_frames=int(native_length), stride=2, dt=.05,
                geometry_file=gi["file"], geometry_size=gi["size"], geometry_mtime_ns=gi["mtime_ns"],
                geometry_sha256=sha256(geometry)))
    return records, dict(source_receipts=receipts, exclusions=exclusions,
        exclusion_counts=dict(Counter(r["reason"] for r in exclusions)),
        validation_scope=("whole episodes on object models chosen per shard (objects_file / grasps_file in the source manifests)"
                          if custom_objects else
                          "whole episodes, same teacher and object-model population; not unseen-teacher/shape generalization"),
        selection="first episode only, native 40 Hz stride 2; achieved hand motion, not unrecorded torque commands")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("domain", choices=["mpm_fresh", "mpm_drag", "folding", "yikai", "dex"])
    p.add_argument("--source", action="append", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=916)
    p.add_argument("--audit")
    p.add_argument("--geometry-dir")
    p.add_argument("--verify-hashes", action="store_true")
    a = p.parse_args()
    if a.domain.startswith("mpm_"):
        rows, provenance = mpm_records(a.source, drag=a.domain == "mpm_drag")
    elif a.domain == "folding":
        rows, provenance = folding_records(a.source)
    elif a.domain == "yikai":
        rows, provenance = yikai_records(a.source, a.audit)
    else:
        if len(a.source) != 1 or not a.geometry_dir:
            p.error("Dex requires one source root and --geometry-dir")
        rows, provenance = dex_records(a.source[0], a.geometry_dir)
    write_indices(a.out, a.domain, rows, provenance, seed=a.seed, verify_hashes=a.verify_hashes)


if __name__ == "__main__":
    main()
