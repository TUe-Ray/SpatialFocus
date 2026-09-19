#!/usr/bin/env python3
"""Load-audit the complete merged VGGT training/evaluation union."""

import json
import time
from pathlib import Path

from validate_vggt_sidecars import load_and_validate, locate_video, read_ids


REPO = Path("/gpfs/home4/geusdd/shuang/SpatialFocus")
HANDOFF = REPO / "vggt_extraction_handoff"
RAW_TRAIN = Path("/scratch-shared/geusdd/VLM3R/data/vlm3r")
RAW_EVAL = Path("/scratch-shared/geusdd/VLM3R/hf_cache/vsibench")
FINAL = Path("/scratch-shared/geusdd/shaoruei/VLM3R/spatial_features/vggt")
REPORT = Path("/scratch-shared/geusdd/shaoruei/VLM3R/spatial_features/vggt_missing_staging/_reports/final_audit.json")
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def training_videos(dataset: str) -> dict[str, Path]:
    root = RAW_TRAIN / dataset / "videos"
    result = {
        path.stem: path.resolve()
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    }
    return result


def main() -> None:
    start = time.time()
    training = {dataset: training_videos(dataset) for dataset in ("scannet", "scannetpp", "arkitscenes")}
    expected_training_counts = {"scannet": 1201, "scannetpp": 856, "arkitscenes": 348}
    actual_training_counts = {dataset: len(items) for dataset, items in training.items()}
    if actual_training_counts != expected_training_counts:
        raise ValueError(f"training ID counts changed: {actual_training_counts}")

    eval_specs = {
        "scannet": HANDOFF / "eval_scannet_ids.txt",
        "scannetpp": HANDOFF / "eval_scannetpp_ids.txt",
        "arkitscenes": HANDOFF / "eval_arkitscenes_ids.txt",
    }
    evaluation = {
        dataset: {
            identifier: locate_video(RAW_EVAL / dataset, identifier)
            for identifier in read_ids(ids_path)
        }
        for dataset, ids_path in eval_specs.items()
    }
    for dataset in training:
        overlap = set(training[dataset]) & set(evaluation[dataset])
        if overlap:
            raise ValueError(f"training/eval ID overlap for {dataset}: {sorted(overlap)[:10]}")

    expected = {
        (dataset, identifier): source
        for dataset, videos in training.items()
        for identifier, source in videos.items()
    }
    expected.update(
        {
            (dataset, identifier): source
            for dataset, videos in evaluation.items()
            for identifier, source in videos.items()
        }
    )
    if len(expected) != 2693:
        raise ValueError(f"overall expected union is {len(expected)}, not 2693")

    observed = {
        (dataset_dir.name, path.stem)
        for dataset_dir in FINAL.iterdir()
        if dataset_dir.is_dir()
        for path in dataset_dir.glob("*.pt")
    }
    missing = set(expected) - observed
    extra = observed - set(expected)
    if missing or extra:
        raise ValueError(f"coverage mismatch: missing={sorted(missing)[:10]} extra={sorted(extra)[:10]}")

    metadata_layer_counts = {}
    for index, ((dataset, identifier), source_video) in enumerate(sorted(expected.items()), start=1):
        path = FINAL / dataset / f"{identifier}.pt"
        schema = load_and_validate(
            path,
            source_video,
            allow_legacy_layer4_metadata=True,
            require_recorded_source_match=False,
        )
        metadata_layers = tuple(schema["meta_intermediate_layer_idx"])
        metadata_layer_counts[str(metadata_layers)] = metadata_layer_counts.get(str(metadata_layers), 0) + 1
        if index % 50 == 0 or index == len(expected):
            print(f"Loaded and validated: {index}/{len(expected)}", flush=True)

    repair_ids = set(read_ids(HANDOFF / "training_repair_ids.txt"))
    report = {
        "training_union": {"expected": 2405, "valid": sum(map(len, training.values()))},
        "training_repair": {"expected": 124, "valid": sum(("scannetpp", identifier) in observed for identifier in repair_ids)},
        "eval_scannet": {"expected": 88, "valid": len(evaluation["scannet"])},
        "eval_scannetpp": {"expected": 50, "valid": len(evaluation["scannetpp"])},
        "eval_arkitscenes": {"expected": 150, "valid": len(evaluation["arkitscenes"])},
        "eval_union": {"expected": 288, "valid": sum(map(len, evaluation.values()))},
        "overall_union": {"expected": 2693, "valid": len(expected)},
        "metadata_intermediate_layer_idx_counts": metadata_layer_counts,
        "all_payloads_loaded": True,
        "all_frame_indices_match_raw_media": True,
        "layer4_tensor_absent_everywhere": True,
        "runtime_seconds": time.time() - start,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    temporary = REPORT.with_name(f".{REPORT.name}.tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(REPORT)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
