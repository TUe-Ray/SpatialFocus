#!/usr/bin/env python
"""Audit exact frame coverage for a depth-feature cache assembled by resume.

The extractor records detailed runtime rows for videos processed by its current
invocation.  When a killed extraction is resumed, already complete videos are
skipped, so the final extraction provenance can legitimately contain fewer
runtime rows than the fixed manifest.  This audit does not rewrite that source
record.  It proves the union cache has exactly the fixed frame population and
records the limitation explicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.probing.legacy_pre_sft_completion_specs import FULL_FEATURE_LEVELS


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def frame_files(path: Path) -> set[str]:
    require(path.is_dir(), f"Missing cache directory: {path}")
    files = {item.stem.removeprefix("frame_") for item in path.glob("frame_*.pt")}
    unexpected = sorted(item.name for item in path.iterdir() if item.is_file() and not item.name.startswith("frame_"))
    require(not unexpected, f"Unexpected files under {path}: {unexpected[:5]}")
    return files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--sample-indices", type=Path, required=True)
    parser.add_argument("--artifact-manifest", type=Path, required=True)
    parser.add_argument("--extraction-provenance", type=Path, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--source-job-ids", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require(not args.output.exists(), f"Refusing to overwrite audit: {args.output}")
    sample = read_json(args.sample_indices)
    manifest = read_json(args.artifact_manifest)
    provenance = read_json(args.extraction_provenance)

    require(manifest.get("schema_version") == "legacy_pre_sft_completion_manifest_v1", "Wrong artifact-manifest schema")
    require(manifest.get("post_sft_state_loaded") is False, "Artifact manifest does not exclude post-SFT state")
    require(args.candidate in manifest.get("candidates", {}), f"Candidate missing from artifact manifest: {args.candidate}")
    candidate = manifest["candidates"][args.candidate]
    sample_hash = sha256(args.sample_indices)
    require(manifest.get("sample_indices", {}).get("sha256") == sample_hash, "Artifact manifest uses a different split")
    require(tuple(manifest.get("feature_levels", [])) == FULL_FEATURE_LEVELS, "Artifact manifest does not use the 15-level policy")
    require(provenance.get("git_commit") == manifest.get("git_commit"), "Extraction commit differs from artifact manifest")
    require(provenance.get("git_worktree_dirty") is False, "Extraction used a dirty worktree")
    require(provenance.get("sample_indices_sha256") == sample_hash, "Extraction uses a different split")
    require(provenance.get("no_vlm3r_sft_adapter_loaded") is True, "Extraction lacks no-post-SFT proof")
    require(set(provenance.get("requested_feature_levels", [])) == set(FULL_FEATURE_LEVELS), "Extraction requested incomplete feature coverage")
    require(provenance.get("c1_calibration_sha256") == candidate["c1"]["sha256"], "C1 hash differs from artifact manifest")
    if candidate.get("geometry_c1"):
        require(
            provenance.get("geometry_c1_calibration_sha256") == candidate["geometry_c1"]["sha256"],
            "Geometry C1 hash differs from artifact manifest",
        )

    videos = sample.get("videos", [])
    require(isinstance(videos, list) and len(videos) == 1199, "Fixed manifest does not contain 1199 videos")
    require(int(sample.get("train_videos", -1)) == 1006 and int(sample.get("val_videos", -1)) == 193, "Unexpected train/validation split")
    require(all(int(video.get("num_frames", -1)) == 32 for video in videos), "Not every source video has 32 model-forward frames")
    expected_frames = {
        frame["frame_sample_id"]
        for video in videos
        for frame in video.get("frames", [])
    }
    require(len(expected_frames) == 2398, f"Expected 2398 unique target frames, found {len(expected_frames)}")

    coverage: dict[str, Any] = {}
    for level in FULL_FEATURE_LEVELS:
        path = args.cache_root / "features" / args.model_label / level
        found = frame_files(path)
        require(found == expected_frames, f"{level}: frame population differs from the fixed manifest")
        coverage[level] = {"path": str(path), "frames": len(found), "exact_manifest_match": True}
    for cache_name in ("gt_depth", "metadata"):
        path = args.cache_root / cache_name
        found = frame_files(path)
        require(found == expected_frames, f"{cache_name}: frame population differs from the fixed manifest")
        coverage[cache_name] = {"path": str(path), "frames": len(found), "exact_manifest_match": True}

    runtime_rows = provenance.get("extraction_samples", [])
    require(isinstance(runtime_rows, list) and runtime_rows, "Extraction provenance has no runtime rows")
    require(all(int(row.get("source_video_num_frames", -1)) == 32 for row in runtime_rows), "A recorded extraction row did not use 32 frames")
    assertions = [row.get("first_video_runtime_assertions") for row in runtime_rows if row.get("first_video_runtime_assertions")]
    require(assertions and all(row.get("assessment") == "PASS" for row in assertions), "Runtime assertion did not pass")
    require(len(runtime_rows) <= len(videos), "Extraction provenance has more rows than fixed videos")

    payload = {
        "schema_version": "resumed_depth_feature_cache_audit_v1",
        "status": "PASS_WITH_EXPLICIT_RESUME_PROVENANCE",
        "candidate": args.candidate,
        "model_label": args.model_label,
        "source_job_ids": [value for value in args.source_job_ids.split(",") if value],
        "artifact_manifest": str(args.artifact_manifest.resolve()),
        "artifact_manifest_sha256": sha256(args.artifact_manifest),
        "extraction_provenance": str(args.extraction_provenance.resolve()),
        "extraction_provenance_sha256": sha256(args.extraction_provenance),
        "sample_indices": str(args.sample_indices.resolve()),
        "sample_indices_sha256": sample_hash,
        "fixed_videos": len(videos),
        "fixed_target_frames": len(expected_frames),
        "model_forward_frames_per_video": 32,
        "runtime_rows_preserved_by_final_resume_invocation": len(runtime_rows),
        "videos_skipped_from_final_runtime_rows_due_to_prior_complete_resume_cache": len(videos) - len(runtime_rows),
        "source_provenance_was_modified": False,
        "coverage": coverage,
        "limitations": [
            "The source extraction provenance contains detailed runtime rows only for videos processed by the final resume invocation.",
            "Exact union-cache frame coverage is proven against the immutable split; earlier runtime rows were not reconstructed or inserted into source provenance.",
        ],
        "post_sft_state_loaded": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "output": str(args.output), "levels": len(FULL_FEATURE_LEVELS)}))


if __name__ == "__main__":
    main()
