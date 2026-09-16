#!/usr/bin/env python
"""Fail closed on the baseline-first smoke for legacy probe completion."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--artifact-manifest", type=Path, required=True)
    parser.add_argument("--sample-indices", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def verify_one(
    root: Path,
    identifier: str,
    label: str,
    expected_levels: set[str],
    smoke_sample_sha256: str,
    expected_git_commit: str,
    c1_sha256: str | None,
    geometry_c1_sha256: str | None,
    eomt_validation_sha256: str | None,
    eomt_mode: str | None,
    geometry_architecture: str | None,
) -> dict[str, Any]:
    feature_root = root / "features" / label
    provenance_path = feature_root / "extraction_provenance.json"
    provenance = read_json(provenance_path)
    if provenance.get("git_worktree_dirty") is not False:
        raise ValueError(f"{identifier}: smoke extraction used a dirty worktree")
    if provenance.get("sample_indices_sha256") != smoke_sample_sha256:
        raise ValueError(f"{identifier}: smoke split differs from the common smoke manifest")
    if provenance.get("git_commit") != expected_git_commit:
        raise ValueError(f"{identifier}: smoke extraction commit differs from locked manifest")
    if provenance.get("no_vlm3r_sft_adapter_loaded") is not True:
        raise ValueError(f"{identifier}: smoke lacks no-post-SFT-adapter proof")
    if set(provenance.get("requested_feature_levels", [])) != expected_levels:
        raise ValueError(f"{identifier}: smoke does not cover all required representations")
    if c1_sha256 is None:
        if provenance.get("c1_calibration_json") is not None:
            raise ValueError("BASE unexpectedly loaded a C1 artifact")
    elif provenance.get("c1_calibration_sha256") != c1_sha256:
        raise ValueError(f"{identifier}: C1 hash differs from locked artifact")
    if geometry_c1_sha256 is not None and provenance.get("geometry_c1_calibration_sha256") != geometry_c1_sha256:
        raise ValueError(f"{identifier}: geometry C1 hash differs from locked artifact")
    if eomt_validation_sha256 is not None:
        if provenance.get("eomt_cache_validation_sha256") != eomt_validation_sha256:
            raise ValueError(f"{identifier}: EoMT validation identity differs from locked cache")
        if eomt_mode == "selective" and provenance.get("eomt_selective_kv_gate") is not True:
            raise ValueError(f"{identifier}: selective EoMT gate was not active")
        if eomt_mode == "object":
            assertion = provenance.get("extraction_samples", [{}])[0].get("first_video_runtime_assertions", {})
            if (
                provenance.get("experiment_variant") != "c1_eomt_object"
                or assertion.get("assessment") != "PASS"
                or assertion.get("architecture") != "eomt_object"
                or int(assertion.get("eomt_object_auxiliary_token_count", 0)) <= 0
                or assertion.get("eomt_object_sequence_order") != "after_ordinary_visual_tokens"
                or assertion.get("primary_probe_excludes_auxiliary_tokens") is not True
            ):
                raise ValueError(f"{identifier}: object-token runtime assertion did not pass")
    if geometry_architecture == "visual_geo_rope":
        assertion = provenance.get("extraction_samples", [{}])[0].get("first_video_runtime_assertions", {})
        if (
            provenance.get("active_geometry_architecture") != "visual_3d_rope"
            or provenance.get("geometry_point_map_key") != "point_maps_ref"
            or assertion.get("assessment") != "PASS"
            or assertion.get("architecture") != "visual_3d_rope"
            or assertion.get("model_forward_inputs", {}).get("point_maps") is not True
        ):
            raise ValueError(f"{identifier}: Visual GeoRoPE runtime assertion did not pass")
    counts: dict[str, int] = {}
    metrics: dict[str, dict[str, float]] = {}
    for level in sorted(expected_levels):
        count = len(list((feature_root / level).glob("frame_*.pt")))
        if count != 4:
            raise ValueError(f"{identifier}/{level}: expected four selected smoke-frame tensors, found {count}")
        metric = read_json(root / "probes" / label / level / "metrics.json")
        values = {name: float(metric.get(name, float("nan"))) for name in ("mae", "absrel", "delta125")}
        if not all(math.isfinite(value) for value in values.values()) or int(metric.get("num_tokens", 0)) <= 0:
            raise ValueError(f"{identifier}/{level}: non-finite or empty smoke metrics")
        counts[level] = count
        metrics[level] = values
    return {
        "feature_root": str(feature_root),
        "provenance": str(provenance_path),
        "provenance_sha256": sha256(provenance_path),
        "feature_counts": counts,
        "probe_metrics": metrics,
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite smoke marker: {args.output}")
    manifest = read_json(args.artifact_manifest)
    if manifest.get("schema_version") != "legacy_pre_sft_completion_manifest_v1":
        raise ValueError("Incompatible completion artifact manifest")
    if manifest.get("post_sft_state_loaded") is not False:
        raise ValueError("Manifest does not prove post-SFT state is excluded")
    smoke_sample_sha256 = sha256(args.sample_indices)
    expected_levels = set(manifest.get("feature_levels", []))
    if len(expected_levels) != 15:
        raise ValueError("Artifact manifest does not have the full 15-feature policy")
    records = {
        "BASE": verify_one(
            args.cache_root / "BASE", "BASE", "pre_sft_base_vlm", expected_levels, smoke_sample_sha256,
            str(manifest["git_commit"]), None, None, None, None, None
        )
    }
    for identifier, candidate in manifest["candidates"].items():
        records[identifier] = verify_one(
            args.cache_root / identifier,
            identifier,
            str(candidate["label"]),
            expected_levels,
            smoke_sample_sha256,
            str(manifest["git_commit"]),
            str(candidate["c1"]["sha256"]),
            str(candidate["geometry_c1"]["sha256"]) if candidate.get("geometry_c1") else None,
            str(manifest["eomt"]["validation_sha256"])
            if candidate.get("uses_eomt_selective_gate") or candidate.get("uses_eomt_object_tokens")
            else None,
            "selective" if candidate.get("uses_eomt_selective_gate") else "object"
            if candidate.get("uses_eomt_object_tokens") else None,
            str(candidate["geometry_c1"]["architecture"]) if candidate.get("geometry_c1") else None,
        )
    payload = {
        "schema_version": "legacy_pre_sft_completion_smoke_v1",
        "status": "PASS",
        "artifact_manifest": str(args.artifact_manifest.resolve()),
        "artifact_manifest_sha256": sha256(args.artifact_manifest),
        "sample_indices": str(args.sample_indices.resolve()),
        "smoke_sample_indices_sha256": smoke_sample_sha256,
        "candidates": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": str(args.output), "candidates": len(records)}))


if __name__ == "__main__":
    main()
