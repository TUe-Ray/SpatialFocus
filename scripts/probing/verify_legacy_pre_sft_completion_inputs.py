#!/usr/bin/env python
"""Verify and record full Snellius inputs for legacy pre-SFT probes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-indices", type=Path, required=True)
    parser.add_argument("--forward-root", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--geometry-root", type=Path, required=True)
    parser.add_argument("--eomt-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reuse-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = read_json(args.sample_indices)
    videos = manifest.get("videos")
    if not isinstance(videos, list) or len(videos) != 1_199:
        raise ValueError("Expected the fixed 1,199-video sample manifest")
    scene_ids = [str(video.get("scene_id", "")) for video in videos]
    if any(not scene for scene in scene_ids) or len(set(scene_ids)) != 1_199:
        raise ValueError("The fixed manifest must contain 1,199 unique non-empty scene IDs")

    directories = {
        "forward_frames": (args.forward_root / "frames/scannet", 1_199),
        "probe_targets": (args.target_root / "targets/scannet/spatial_features_points", 1_199),
        # This authoritative container has 88 extra available sidecars. Only
        # the fixed 1,199-scene manifest is selected by extraction.
        "cut3r_final": (args.feature_root / "scannet/spatial_features", 1_287),
        "cut3r_dec_6": (args.feature_root / "scannet/spatial_features_dec_6", 1_199),
        "cut3r_dec_9": (args.feature_root / "scannet/spatial_features_dec_9", 1_199),
        "geometry_point_maps": (args.geometry_root / "scannet/spatial_features_points", 1_199),
        "eomt_class_logits": (args.eomt_root / "class_logits/scannet", 1_199),
        "eomt_object_masks": (args.eomt_root / "object_masks/scannet", 1_199),
        "eomt_selective_masks": (args.eomt_root / "selective_masks/scannet", 1_199),
    }
    records: dict[str, Any] = {}
    for name, (directory, expected_count) in directories.items():
        if not directory.is_dir():
            raise FileNotFoundError(f"Missing input directory: {directory}")
        files = list(directory.glob("*.pt"))
        missing = [scene for scene in scene_ids if not (directory / f"{scene}.pt").is_file()]
        if len(files) != expected_count or missing:
            raise ValueError(
                f"{name}: expected container count {expected_count} and complete fixed-manifest "
                f"coverage; found count={len(files)}, missing={missing[:5]}"
            )
        records[name] = {
            "path": str(directory.resolve()),
            "container_pt_count": len(files),
            "fixed_manifest_scene_count": len(scene_ids),
            "missing_fixed_manifest_scenes": 0,
        }

    validation = args.eomt_root / "validation.json"
    checksums = args.eomt_root / "checksums.json"
    if not validation.is_file() or not checksums.is_file():
        raise FileNotFoundError("Missing EoMT validation/checksum manifest")
    payload = {
        "schema_version": "legacy_pre_sft_completion_input_coverage_v1",
        "status": "PASS",
        "sample_indices": str(args.sample_indices.resolve()),
        "sample_indices_sha256": sha256(args.sample_indices),
        "fixed_manifest_scene_count": len(scene_ids),
        "inputs": records,
        "eomt_validation_sha256": sha256(validation),
        "eomt_checksums_sha256": sha256(checksums),
    }
    if args.output.exists():
        if not args.reuse_existing:
            raise FileExistsError(f"Refusing to overwrite input coverage record: {args.output}")
        if read_json(args.output) != payload:
            raise ValueError(f"Existing input coverage record differs from current inputs: {args.output}")
        print(json.dumps({"status": "PASS_REUSED", "output": str(args.output)}))
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": str(args.output), "inputs": len(records)}))


if __name__ == "__main__":
    main()
