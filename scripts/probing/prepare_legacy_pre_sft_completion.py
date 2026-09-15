#!/usr/bin/env python
"""Lock and validate inputs for the seven legacy pre-SFT probe completions."""

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

from scripts.probing.legacy_pre_sft_completion_specs import (
    FULL_FEATURE_LEVELS,
    LEGACY_PARTIAL_CANDIDATES,
    REPO_OUTPUTS,
)


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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument(
        "--base-model",
        type=Path,
        default=Path("/mnt/DATA_SSD/shaoruei/models/base/LLaVA-NeXT-Video-7B-Qwen2"),
    )
    parser.add_argument(
        "--siglip-model",
        type=Path,
        default=Path("/mnt/DATA_SSD/shaoruei/models/base/siglip-so400m-patch14-384"),
    )
    parser.add_argument(
        "--sample-indices",
        type=Path,
        default=Path(
            "/home/shaoruei/probe_provenance/scannet_baseline_L6/"
            "scannet_baseline_L6_depth_provenance/splits/"
            "semantic_probe_scannet_final_usable_sample_indices.json"
        ),
    )
    parser.add_argument(
        "--geometry-root",
        type=Path,
        default=Path("/mnt/DATA_SSD/shaoruei/probing_data/cut3r_point_maps_32_v1"),
    )
    parser.add_argument(
        "--eomt-root",
        type=Path,
        default=Path("/home/shaoruei/probe_cache/eomt_consumer_grid_v2"),
    )
    parser.add_argument(
        "--c1-root",
        type=Path,
        default=REPO_OUTPUTS,
        help="Relocated root containing the candidate C1 trees recorded by the immutable specifications.",
    )
    parser.add_argument(
        "--candidates",
        default=",".join(candidate.identifier for candidate in LEGACY_PARTIAL_CANDIDATES),
        help="Comma-separated candidate subset; defaults to the complete seven-candidate campaign.",
    )
    parser.add_argument("--reuse-existing", action="store_true")
    return parser.parse_args()


def validate_base(path: Path) -> dict[str, Any]:
    config = path / "config.json"
    if not config.is_file():
        raise FileNotFoundError(f"Missing base config: {config}")
    forbidden = [
        str(candidate)
        for pattern in ("adapter_model.bin", "non_lora_trainables.bin", "adapter_config.json")
        for candidate in path.rglob(pattern)
    ]
    if forbidden:
        raise RuntimeError(f"Base model contains forbidden post-SFT artifacts: {forbidden[:3]}")
    return {"path": str(path.resolve()), "config_sha256": sha256(config), "forbidden_post_sft_artifacts_loaded": False}


def validate_manifest(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    videos = payload.get("videos")
    if not isinstance(videos, list) or len(videos) != 1_199:
        raise ValueError(f"Expected authoritative 1,199-video manifest: {path}")
    return {"path": str(path.resolve()), "sha256": sha256(path), "videos": len(videos)}


def validate_c1(path: Path, expected_architecture: str) -> dict[str, Any]:
    payload = read_json(path)
    if payload.get("schema_version") != "c1_calibration_v1":
        raise ValueError(f"Unexpected C1 schema: {path}")
    if payload.get("architecture") != expected_architecture:
        raise ValueError(f"C1 architecture mismatch at {path}: expected {expected_architecture}")
    if payload.get("no_training") is not True:
        raise ValueError(f"C1 artifact does not prove no_training=true: {path}")
    return {"path": str(path.resolve()), "sha256": sha256(path), "architecture": expected_architecture}


def validate_geometry_activation(path: Path, expected_architecture: str) -> dict[str, Any]:
    payload = read_json(path)
    if payload.get("schema_version") != "c1_geometry_activation_v1":
        raise ValueError(f"Unexpected geometry C1 schema: {path}")
    if payload.get("architecture") != expected_architecture or payload.get("no_training") is not True:
        raise ValueError(f"Invalid frozen geometry C1 activation: {path}")
    if payload.get("geometry_source") != "predicted CUT3R point_maps_ref":
        raise ValueError(f"Unexpected geometry coordinate source: {path}")
    return {"path": str(path.resolve()), "sha256": sha256(path), "architecture": expected_architecture}


def validate_geometry_root(root: Path) -> dict[str, Any]:
    sidecars = root / "scannet/spatial_features_points"
    count = sum(1 for _ in sidecars.glob("*.pt"))
    if count != 1_199:
        raise ValueError(f"Expected 1,199 full-frame ScanNet point maps at {sidecars}, found {count}")
    return {"root": str(root.resolve()), "sidecar_dir": str(sidecars.resolve()), "scannet_files": count, "point_map_key": "point_maps_ref"}


def validate_eomt(root: Path) -> dict[str, Any]:
    validation_path = root / "validation.json"
    payload = read_json(validation_path)
    if payload.get("schema_version") != "eomt_consumer_grid_v2" or payload.get("status") != "PASS":
        raise ValueError(f"EoMT cache validation is not PASS: {validation_path}")
    if int(payload.get("scene_count", -1)) != 1_199 or int(payload.get("file_count", -1)) != 3_597:
        raise ValueError(f"EoMT cache is not the expected 1,199-scene grid: {validation_path}")
    return {
        "root": str(root.resolve()),
        "validation": str(validation_path.resolve()),
        "validation_sha256": sha256(validation_path),
        "scene_count": int(payload["scene_count"]),
        "file_count": int(payload["file_count"]),
        "actual_vlm_forward_parity": payload.get("actual_vlm_forward_parity"),
    }


def main() -> None:
    args = parse_args()
    selected_ids = [value.strip() for value in args.candidates.split(",") if value.strip()]
    known_ids = {candidate.identifier for candidate in LEGACY_PARTIAL_CANDIDATES}
    unknown_ids = sorted(set(selected_ids).difference(known_ids))
    if unknown_ids or not selected_ids or len(selected_ids) != len(set(selected_ids)):
        raise ValueError(
            f"Candidates must be a non-empty unique subset of {sorted(known_ids)}; "
            f"unknown={unknown_ids}, requested={selected_ids}"
        )
    if args.output.exists() and not args.reuse_existing:
        raise FileExistsError(f"Refusing to overwrite locked completion manifest: {args.output}")

    base = validate_base(args.base_model)
    siglip_config = args.siglip_model / "config.json"
    if not siglip_config.is_file():
        raise FileNotFoundError(f"Missing SigLIP config: {siglip_config}")
    sample_indices = validate_manifest(args.sample_indices)
    geometry = validate_geometry_root(args.geometry_root)
    eomt = validate_eomt(args.eomt_root)

    candidates: dict[str, dict[str, Any]] = {}
    for candidate in LEGACY_PARTIAL_CANDIDATES:
        if candidate.identifier not in selected_ids:
            continue
        c1_path = args.c1_root / candidate.c1_artifact.relative_to(REPO_OUTPUTS)
        record: dict[str, Any] = {
            "label": candidate.label,
            "display_name": candidate.display_name,
            "fusion_variant": candidate.fusion_variant,
            "spatial_features_subdir": candidate.spatial_features_subdir,
            "spatialstack_cut3r_layers": candidate.spatialstack_cut3r_layers,
            "spatialstack_llm_layers": candidate.spatialstack_llm_layers,
            "c1": validate_c1(c1_path, candidate.c1_architecture),
            "uses_eomt_selective_gate": candidate.uses_eomt_selective_gate,
        }
        if candidate.geometry_c1_activation is not None:
            geometry_c1_path = args.c1_root / candidate.geometry_c1_activation.relative_to(REPO_OUTPUTS)
            record["geometry_c1"] = validate_geometry_activation(
                geometry_c1_path, str(candidate.geometry_architecture)
            )
        candidates[candidate.identifier] = record

    payload = {
        "schema_version": "legacy_pre_sft_completion_manifest_v1",
        "experiment_label": "legacy partial pre-SFT depth-probe completion",
        "git_commit": args.git_commit,
        "base_model": base,
        "siglip_model": {"path": str(args.siglip_model.resolve()), "config_sha256": sha256(siglip_config)},
        "sample_indices": sample_indices,
        "feature_levels": list(FULL_FEATURE_LEVELS),
        "geometry": geometry,
        "eomt": eomt,
        "candidates": candidates,
        "excluded_loss_only_variants": [
            "Hierarchical Add @ L0/1/2 + Depth Supervision",
            "Pre-Projector Cross-Attn + Depth Supervision",
        ],
        "post_sft_state_loaded": False,
        "no_candidate_optimizer_constructed": True,
        "no_candidate_optimizer_step": True,
    }
    if args.output.exists():
        existing = read_json(args.output)
        if existing != payload:
            raise ValueError(f"Existing completion manifest differs from current inputs: {args.output}")
        print(json.dumps({"status": "PASS_REUSED", "output": str(args.output)}))
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": str(args.output), "candidates": len(candidates)}))


if __name__ == "__main__":
    main()
