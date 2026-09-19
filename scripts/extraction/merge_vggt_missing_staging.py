#!/usr/bin/env python3
"""Guarded, non-overwriting merge of validated VGGT staging sidecars."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from validate_vggt_sidecars import load_and_validate, locate_video, read_ids


REPO = Path("/gpfs/home4/geusdd/shuang/SpatialFocus")
HANDOFF = REPO / "vggt_extraction_handoff"
RAW_TRAIN = Path("/scratch-shared/geusdd/VLM3R/data/vlm3r")
RAW_EVAL = Path("/scratch-shared/geusdd/VLM3R/hf_cache/vsibench")
STAGING = Path("/scratch-shared/geusdd/shaoruei/VLM3R/spatial_features/vggt_missing_staging")
FINAL = Path("/scratch-shared/geusdd/shaoruei/VLM3R/spatial_features/vggt")
STABILITY_RECORD = STAGING / "_provenance/rsync_stability.json"
MINIMUM_STABILITY_SECONDS = 600
CORRUPT_ID = "dd685be466"


def media_ids(root: Path) -> set[str]:
    extensions = {".mp4", ".avi", ".mov", ".mkv"}
    return {path.stem for path in root.iterdir() if path.is_file() and path.suffix.lower() in extensions}


def targets() -> list[tuple[str, str, Path]]:
    specs = (
        ("scannetpp", HANDOFF / "training_repair_ids.txt", RAW_TRAIN / "scannetpp/videos"),
        ("scannet", HANDOFF / "eval_scannet_ids.txt", RAW_EVAL / "scannet"),
        ("scannetpp", HANDOFF / "eval_scannetpp_ids.txt", RAW_EVAL / "scannetpp"),
        ("arkitscenes", HANDOFF / "eval_arkitscenes_ids.txt", RAW_EVAL / "arkitscenes"),
    )
    result = []
    for dataset, ids_path, raw_root in specs:
        result.extend((dataset, identifier, locate_video(raw_root, identifier)) for identifier in read_ids(ids_path))
    if len(result) != 412 or len({(dataset, identifier) for dataset, identifier, _ in result}) != 412:
        raise ValueError("target union is not exactly 412 unique dataset/ID pairs")
    return result


def expected_existing_paths() -> set[str]:
    repair = set(read_ids(HANDOFF / "training_repair_ids.txt"))
    result = set()
    for dataset in ("scannet", "scannetpp", "arkitscenes"):
        identifiers = media_ids(RAW_TRAIN / dataset / "videos")
        if dataset == "scannetpp":
            identifiers -= repair
        result.update(f"{dataset}/{identifier}.pt" for identifier in identifiers)
    if len(result) != 2281:
        raise ValueError(f"expected-existing derivation produced {len(result)}, not 2281")
    return result


def final_tree_state() -> dict:
    expected = expected_existing_paths()
    target_map = {
        f"{dataset}/{identifier}.pt": source_video
        for dataset, identifier, source_video in targets()
    }
    files = sorted(path for path in FINAL.rglob("*") if path.is_file())
    pt_paths = {str(path.relative_to(FINAL)) for path in files if path.suffix == ".pt"}
    unexpected_pt = pt_paths - expected
    if unexpected_pt - set(target_map):
        raise ValueError(f"unexpected pre-merge sidecars: {sorted(unexpected_pt - set(target_map))[:20]}")
    missing = expected - pt_paths
    if missing:
        raise ValueError(f"rsync incomplete: {len(missing)} expected sidecars missing; first={sorted(missing)[:10]}")

    dd_state = "absent"
    dd_path = FINAL / "scannetpp" / f"{CORRUPT_ID}.pt"
    if dd_path.is_file():
        try:
            load_and_validate(
                dd_path,
                target_map[f"scannetpp/{CORRUPT_ID}.pt"],
                allow_legacy_layer4_metadata=True,
                require_recorded_source_match=False,
            )
            dd_state = "valid"
        except Exception as exc:
            dd_state = f"invalid: {exc}"
    for relative in sorted(unexpected_pt - {f"scannetpp/{CORRUPT_ID}.pt"}):
        load_and_validate(
            FINAL / relative,
            target_map[relative],
            allow_legacy_layer4_metadata=True,
            require_recorded_source_match=False,
        )

    digest = hashlib.sha256()
    entries = []
    for path in files:
        stat = path.stat()
        relative = str(path.relative_to(FINAL))
        entry = f"{relative}\0{stat.st_size}\0{stat.st_mtime_ns}"
        digest.update(entry.encode())
        entries.append({"path": relative, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    return {
        "timestamp_epoch": time.time(),
        "pt_count": len(pt_paths),
        "all_file_count": len(files),
        "fingerprint_sha256": digest.hexdigest(),
        "dd685be466_state": dd_state,
        "entries": entries,
    }


def record_stability() -> None:
    state = final_tree_state()
    STABILITY_RECORD.parent.mkdir(parents=True, exist_ok=True)
    temporary = STABILITY_RECORD.with_name(f".{STABILITY_RECORD.name}.tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(STABILITY_RECORD)
    print(json.dumps({key: value for key, value in state.items() if key != "entries"}, indent=2))
    print(f"Recorded rsync-tree fingerprint in {STABILITY_RECORD}; wait at least {MINIMUM_STABILITY_SECONDS}s before merge.")


def require_stable_tree() -> dict:
    previous = json.loads(STABILITY_RECORD.read_text())
    current = final_tree_state()
    elapsed = current["timestamp_epoch"] - previous["timestamp_epoch"]
    if elapsed < MINIMUM_STABILITY_SECONDS:
        raise ValueError(f"only {elapsed:.1f}s since stability record; require {MINIMUM_STABILITY_SECONDS}s")
    if current["fingerprint_sha256"] != previous["fingerprint_sha256"]:
        raise ValueError("final rsync tree changed since the recorded fingerprint; record stability again")
    return current


def merge() -> None:
    state = require_stable_tree()
    all_targets = targets()
    print(f"Final rsync tree stable and complete: {state['pt_count']} .pt files")

    validated = []
    for index, (dataset, identifier, source_video) in enumerate(all_targets, start=1):
        staged = STAGING / dataset / f"{identifier}.pt"
        load_and_validate(staged, source_video)
        validated.append((dataset, identifier, staged, source_video))
        if index % 25 == 0 or index == len(all_targets):
            print(f"Validated staged payloads: {index}/{len(all_targets)}", flush=True)

    if STAGING.stat().st_dev != FINAL.stat().st_dev:
        raise ValueError("staging and final roots are on different filesystems; atomic hard-link merge unavailable")

    linked = 0
    reused = 0
    quarantined = None
    for dataset, identifier, staged, source_video in validated:
        destination = FINAL / dataset / f"{identifier}.pt"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            try:
                load_and_validate(
                    destination,
                    source_video,
                    allow_legacy_layer4_metadata=True,
                    require_recorded_source_match=False,
                )
                print(f"REUSE_VALID {destination}")
                reused += 1
                continue
            except Exception as exc:
                if dataset != "scannetpp" or identifier != CORRUPT_ID:
                    raise ValueError(f"refusing to replace invalid unexpected destination {destination}: {exc}") from exc
                quarantine = destination.with_name(
                    f"{destination.name}.corrupt-pre-merge-{int(time.time())}"
                )
                os.replace(destination, quarantine)
                quarantined = str(quarantine)
                print(f"QUARANTINED {destination} -> {quarantine}: {exc}")
        os.link(staged, destination)
        print(f"LINKED {staged} -> {destination}")
        linked += 1

    report = {
        "timestamp_epoch": time.time(),
        "staged_validated": len(validated),
        "linked": linked,
        "reused_valid": reused,
        "quarantined_corrupt": quarantined,
    }
    report_path = STAGING / "_reports/merge.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--record-stability", action="store_true")
    actions.add_argument("--merge", action="store_true")
    args = parser.parse_args()
    if args.record_stability:
        record_stability()
    else:
        merge()


if __name__ == "__main__":
    main()
