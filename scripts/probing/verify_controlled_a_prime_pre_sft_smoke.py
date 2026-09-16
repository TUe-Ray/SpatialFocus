#!/usr/bin/env python
"""Verify the baseline-first two-video A-prime pre-SFT smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.probing.prepare_controlled_a_prime_pre_sft import SCHEMA_VERSION as MANIFEST_SCHEMA
from scripts.probing.probe_layer_policy import PRE_SFT_FULL_FEATURE_LEVELS


SCHEMA_VERSION = "controlled_a_prime_pre_sft_smoke_verification_v1"


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_runtime_attestation(provenance: dict[str, Any]) -> None:
    samples = provenance.get("extraction_samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("A-prime smoke has no extraction sample attestation")
    attestation = next(
        (sample.get("first_video_runtime_assertions") for sample in samples if sample.get("first_video_runtime_assertions")),
        None,
    )
    if not isinstance(attestation, dict) or attestation.get("assessment") != "PASS":
        raise ValueError("A-prime smoke has no passing first-video runtime attestation")
    if attestation.get("architecture") != "A_prime":
        raise ValueError("A-prime smoke used the generic rather than architecture-specific runtime assertion")
    telemetry = attestation.get("runtime_telemetry", {})
    required = {
        "fusion_stage": "pre_mm_projector",
        "cut3r_source_layer": 12,
        "geometry_tokens": "patch_only",
        "camera_token_count": 0,
        "finite": True,
        "cut3r_detached": True,
        "visual_query_shape": [32, 729, 1152],
        "geometry_kv_shape": [32, 729, 768],
        "fused_shape": [32, 729, 1152],
        "mm_projector_input_shape": [32, 729, 1152],
        "mm_projector_output_shape": [32, 729, 3584],
    }
    mismatches = {key: telemetry.get(key) for key, expected in required.items() if telemetry.get(key) != expected}
    if mismatches:
        raise ValueError(f"A-prime runtime telemetry mismatch: {mismatches}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sample-indices", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite A-prime smoke marker: {args.output}")
    manifest = read_json(args.manifest)
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise ValueError("A-prime smoke received an incompatible experiment manifest")
    if manifest.get("post_sft_state_loaded") is not False or manifest.get("c1_enabled") is not False:
        raise ValueError("A-prime manifest does not prove plain non-C1 initialization")
    samples = read_json(args.sample_indices)
    if len(samples.get("videos", [])) != 2:
        raise ValueError("A-prime smoke must use exactly one train and one validation video")
    smoke_splits = sorted(str(video.get("split")) for video in samples["videos"])
    if smoke_splits != ["train", "val"]:
        raise ValueError(f"A-prime smoke has the wrong split composition: {smoke_splits}")
    smoke_sample_hash = sha256_file(args.sample_indices)
    expected_frames = 4
    candidates = {
        "BASE": ("pre_sft_base_vlm", "pre_sft_base_vlm"),
        "A_prime": ("controlled_a_prime", "pre_sft_fusion"),
    }
    report: dict[str, Any] = {}
    for identifier, (label, loading_mode) in candidates.items():
        root = args.cache_root / identifier
        feature_root = root / "features" / label
        provenance_path = feature_root / "extraction_provenance.json"
        provenance = read_json(provenance_path)
        if provenance.get("model_loading_mode") != loading_mode:
            raise ValueError(f"{identifier} loading-mode mismatch")
        if provenance.get("git_worktree_dirty") is not False:
            raise ValueError(f"{identifier} smoke was run from a dirty worktree")
        if provenance.get("git_commit") != manifest.get("git_commit"):
            raise ValueError(f"{identifier} smoke commit differs from the frozen manifest")
        if provenance.get("sample_indices_sha256") != smoke_sample_hash:
            raise ValueError(f"{identifier} smoke sample manifest hash mismatch")
        if provenance.get("no_vlm3r_sft_adapter_loaded") is not True:
            raise ValueError(f"{identifier} lacks proof that no post-SFT adapter was loaded")
        if provenance.get("requested_feature_levels") != list(PRE_SFT_FULL_FEATURE_LEVELS):
            raise ValueError(f"{identifier} does not use the complete 15-level pre-SFT policy")
        if identifier == "A_prime":
            if provenance.get("experiment_variant") != "controlled_a_prime":
                raise ValueError("A-prime extraction variant mismatch")
            if provenance.get("fusion_init_seed") != 42:
                raise ValueError("A-prime did not use official constructor seed 42")
            if provenance.get("c1_calibration_json") is not None:
                raise ValueError("A-prime unexpectedly loaded C1 state")
            if provenance.get("fresh_fusion_state_sha256") != manifest.get("fresh_fusion_state_sha256"):
                raise ValueError("A-prime fresh fusion state differs from the frozen constructor manifest")
            validate_runtime_attestation(provenance)
        counts: dict[str, int] = {}
        for level in PRE_SFT_FULL_FEATURE_LEVELS:
            count = len(list((feature_root / level).glob("frame_*.pt")))
            if count != expected_frames:
                raise ValueError(f"{identifier}/{level} has {count} tensors; expected {expected_frames}")
            metric = read_json(root / "probes" / label / level / "metrics.json")
            if not all(math.isfinite(float(metric.get(name, float("nan")))) for name in ("mae", "absrel", "delta125")):
                raise ValueError(f"{identifier}/{level} has non-finite probe metrics")
            if int(metric.get("num_tokens", 0)) <= 0:
                raise ValueError(f"{identifier}/{level} has no validation tokens")
            counts[level] = count
        report[identifier] = {
            "status": "PASS",
            "feature_counts": counts,
            "extraction_provenance": str(provenance_path),
            "extraction_provenance_sha256": sha256_file(provenance_path),
        }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "baseline_executed_first": True,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "sample_indices": str(args.sample_indices.resolve()),
        "sample_indices_sha256": smoke_sample_hash,
        "feature_levels": list(PRE_SFT_FULL_FEATURE_LEVELS),
        "selected_frame_tensors_per_level": expected_frames,
        "candidates": report,
        "post_sft_state_loaded": False,
        "candidate_model_optimizer_constructed": False,
        "candidate_model_optimizer_step": False,
        "depth_probe_optimizer_used": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": str(args.output)}))


if __name__ == "__main__":
    main()
