#!/usr/bin/env python
"""Run Common-7 LogME for cache-backed non-formal pre-SFT diagnostics.

The official five-model v2 study is immutable.  This runner only evaluates
additional caches whose extraction provenance explicitly proves pre-SFT state
and records unavailable requested models without substituting post-SFT data.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import resource
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.probing.depth_probe_common import load_frame_records
from scripts.probing.run_pre_sft_logme_proxy import (
    DEFAULT_SAMPLE_INDICES,
    SPLIT_SHA256,
    logme_from_statistics,
    select_videos,
    sha256_file,
    write_csv,
    write_json,
)
from scripts.probing.run_pre_sft_logme_proxy_v2_common7 import accumulate_statistics_on_device


PROTOCOL_NAME = "pre_sft_logme_proxy_remaining_diagnostics_v1"
DEFAULT_OUTPUT = REPO_ROOT / "logs" / PROTOCOL_NAME
LAYERS = (1, 3, 6, 9, 15, 21, 27)
TARGET_SIGNATURE = "b48e025fefb19e5d7414d2d540b9904a4e21d6de883552699d5e4dc194956d37"


@dataclass(frozen=True)
class Candidate:
    key: str
    label: str
    display_name: str
    root: Path
    required_special_provenance: str


CANDIDATES = (
    Candidate(
        "ss_depth",
        "ss_depth",
        "SS + depth (C1 pre-SFT, fresh point-map head)",
        Path("/home/shaoruei/probe_cache/pre_sft_logme_remaining6_recache/ss_depth"),
        "loss_only_pointmap_head",
    ),
    Candidate(
        "geo_rope_fusion",
        "c1_geo_rope_fusion",
        "GeoRoPE Fusion (C1 pre-SFT)",
        Path("/home/shaoruei/probe_cache/c1_geometry_pre_sft_v1/full/geo_rope_fusion"),
        "geometry_c1_calibration",
    ),
    Candidate(
        "extra_object_token",
        "c1_eomt_object",
        "Extra Object Token (C1 pre-SFT)",
        Path("/home/shaoruei/probe_cache/pre_sft_logme_remaining6_recache/eomt_object"),
        "eomt_object_consumer_cache",
    ),
    Candidate(
        "visual_geo_rope",
        "c1_visual_geo_rope",
        "Visual geo-RoPE (C1 pre-SFT)",
        Path("/home/shaoruei/probe_cache/pre_sft_logme_remaining6_recache/visual_geo_rope"),
        "geometry_c1_calibration",
    ),
)


def provenance_path(candidate: Candidate) -> Path:
    return candidate.root / "features" / candidate.label / "extraction_provenance.json"


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def validate_candidate(candidate: Candidate, sample_indices: Path) -> tuple[dict[str, Any] | None, str | None]:
    path = provenance_path(candidate)
    feature_root = candidate.root / "features" / candidate.label
    missing = [layer for layer in LAYERS if not (feature_root / f"layer_{layer}").is_dir()]
    if not path.is_file() or missing:
        return None, f"missing complete pre-SFT cache/provenance; missing Common-7 layers={missing}"
    payload = json.loads(path.read_text(encoding="utf-8"))
    failures = []
    if payload.get("model_label") != candidate.label:
        failures.append("model_label")
    if payload.get("model_loading_mode") != "pre_sft_fusion":
        failures.append("model_loading_mode")
    if payload.get("no_vlm3r_sft_adapter_loaded") is not True:
        failures.append("no_vlm3r_sft_adapter_loaded")
    if payload.get("post_sft_architecture") is not None:
        failures.append("post_sft_architecture")
    if payload.get("sample_indices") != str(sample_indices) or payload.get("sample_indices_sha256") != SPLIT_SHA256:
        failures.append("sample_indices")
    if sha256_file(sample_indices) != SPLIT_SHA256:
        failures.append("local_sample_indices_sha256")
    if payload.get("target_semantics") != "point_maps_cam -> camera_z":
        failures.append("target_semantics")
    if candidate.required_special_provenance == "geometry_c1_calibration":
        activation = payload.get("geometry_c1_calibration_json")
        expected_sha = payload.get("geometry_c1_calibration_sha256")
        if not activation or not Path(activation).is_file() or sha256_file(Path(activation)) != expected_sha:
            failures.append("geometry_c1_calibration")
    elif candidate.required_special_provenance == "eomt_object_consumer_cache":
        consumer = payload.get("eomt_consumer_cache_root")
        if not consumer or not Path(consumer).is_dir() or not payload.get("eomt_cache_validation_sha256"):
            failures.append("eomt_object_consumer_cache")
        assertions = [
            sample.get("first_video_runtime_assertions")
            for sample in payload.get("extraction_samples", [])
            if isinstance(sample, dict) and isinstance(sample.get("first_video_runtime_assertions"), dict)
        ]
        if not any(
            assertion.get("assessment") == "PASS"
            and assertion.get("architecture") == "eomt_object"
            and int(assertion.get("eomt_object_auxiliary_token_count", 0)) > 0
            and assertion.get("primary_probe_excludes_auxiliary_tokens") is True
            for assertion in assertions
        ):
            failures.append("eomt_object_first_video_runtime_assertions")
    elif candidate.required_special_provenance == "loss_only_pointmap_head":
        attestation = payload.get("loss_only_forward_equivalence_attestation")
        if not isinstance(attestation, dict):
            failures.append("loss_only_forward_equivalence_attestation")
        elif not (
            attestation.get("loss") == "pointmap"
            and attestation.get("auxiliary_head_class") == "PointMapHead"
            and attestation.get("auxiliary_head_freshly_initialized") is True
            and int(attestation.get("auxiliary_head_parameters", 0)) > 0
            and attestation.get("no_optimizer") is True
            and attestation.get("no_post_sft_state") is True
        ):
            failures.append("fresh_pointmap_head_provenance")
    if failures:
        return None, "invalid pre-SFT provenance: " + ",".join(failures)
    return payload, None


def compatible(row: dict[str, Any], candidate: Candidate, layer: int, protocol_sha: str) -> bool:
    try:
        return (
            row.get("status") == "complete"
            and row.get("architecture") == candidate.label
            and int(row.get("layer", -1)) == layer
            and row.get("protocol_sha256") == protocol_sha
            and row.get("target_signature") == TARGET_SIGNATURE
            and math.isfinite(float(row["logme"]))
        )
    except (KeyError, TypeError, ValueError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-indices", type=Path, default=DEFAULT_SAMPLE_INDICES)
    parser.add_argument("--candidate", choices=[candidate.key for candidate in CANDIDATES], action="append")
    parser.add_argument("--video-limit", type=int, default=None)
    parser.add_argument("--device", default="cpu", help="Statistics device only; no VLM is loaded.")
    parser.add_argument("--block-frames", type=int, default=8)
    parser.add_argument("--cpu-threads", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.block_frames < 1:
        raise ValueError("--block-frames must be positive")
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA is unavailable: {device}")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    sample_indices = args.sample_indices.resolve()
    all_records = load_frame_records(sample_indices, split="train")
    if len(all_records) != 2012 or len({str(row["video_sample_id"]) for row in all_records}) != 1006:
        raise RuntimeError("Expected the formal 1,006-video / 2,012-frame training split")
    records = select_videos(all_records, args.video_limit)
    selected_keys = set(args.candidate or [candidate.key for candidate in CANDIDATES])
    selected = [candidate for candidate in CANDIDATES if candidate.key in selected_keys]

    protocol = {
        "schema_version": PROTOCOL_NAME,
        "scope": "additional diagnostics only; never added to the frozen formal-five correlation",
        "definition": "Bayesian linear-regression maximum evidence, normalized log p(y|F)/N",
        "dtype": "float64",
        "alpha_init": 1.0,
        "beta_init": 1.0,
        "primary_layers": list(LAYERS),
        "sample_indices": str(sample_indices),
        "sample_indices_sha256": sha256_file(sample_indices),
        "split": "train",
        "training_videos": len({str(row["video_sample_id"]) for row in records}),
        "training_frames": len(records),
        "video_limit": args.video_limit,
        "formal_target_signature": TARGET_SIGNATURE,
        "no_vlm_forward": True,
        "no_optimizer": True,
        "no_post_sft_checkpoint": True,
        "candidate_registry": [
            {"key": candidate.key, "label": candidate.label, "cache_root": str(candidate.root)}
            for candidate in CANDIDATES
        ],
    }
    protocol["protocol_sha256"] = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    write_json(output / "protocol.json", protocol)

    availability = []
    valid: dict[str, dict[str, Any]] = {}
    for candidate in CANDIDATES:
        provenance, reason = validate_candidate(candidate, sample_indices)
        status = "available" if provenance is not None else "unavailable"
        availability.append({
            "requested_model": candidate.key,
            "architecture": candidate.label,
            "display_name": candidate.display_name,
            "status": status,
            "reason": reason or "complete provenance-verified pre-SFT Common-7 cache",
            "cache_root": str(candidate.root),
            "selected_this_invocation": candidate.key in selected_keys,
        })
        if provenance is not None:
            valid[candidate.key] = provenance
            write_json(output / f"source_provenance_{candidate.key}.json", provenance)
    write_csv(output / "availability.csv", availability)

    per_layer_path = output / "logme_per_layer.csv"
    existing = read_rows(per_layer_path)
    rows = [row for row in existing if any(row.get("architecture") == candidate.label for candidate in CANDIDATES)]
    started_all = time.perf_counter()
    for candidate in selected:
        if candidate.key not in valid:
            print(f"[UNAVAILABLE] {candidate.key}: {next(row['reason'] for row in availability if row['requested_model'] == candidate.key)}", flush=True)
            continue
        for layer in LAYERS:
            previous = next((row for row in rows if compatible(row, candidate, layer, protocol["protocol_sha256"])), None)
            if previous is not None and not args.force:
                print(f"[REUSE] {candidate.label}/L{layer}", flush=True)
                continue
            rows = [row for row in rows if not (row.get("architecture") == candidate.label and int(row.get("layer", -1)) == layer)]
            started = time.perf_counter()
            gram, cross, yy, count, frames, signature = accumulate_statistics_on_device(
                candidate, layer, records, device=device, block_frames=args.block_frames
            )
            if signature != TARGET_SIGNATURE:
                raise RuntimeError(f"{candidate.label}/L{layer}: target signature mismatch: {signature}")
            result = logme_from_statistics(gram, cross, yy, count)
            row = {
                "architecture": candidate.label,
                "display_name": candidate.display_name,
                "layer": layer,
                "feature_level": f"layer_{layer}",
                "cache_root": str(candidate.root),
                "cache_provenance": str(provenance_path(candidate)),
                "status": "complete",
                "protocol_sha256": protocol["protocol_sha256"],
                "logme": result["logme"],
                "alpha": result["alpha"],
                "beta": result["beta"],
                "gamma": result["gamma"],
                "iterations": result["iterations"],
                "converged": result["converged"],
                "residual_sq": result["residual_sq"],
                "minimum_eigenvalue": result["minimum_eigenvalue"],
                "valid_tokens": count,
                "feature_dimension": int(gram.shape[0]),
                "training_videos": protocol["training_videos"],
                "training_frames": frames,
                "target_signature": signature,
                "runtime_seconds": time.perf_counter() - started,
                "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                "statistics_device": str(device),
            }
            rows.append(row)
            write_csv(per_layer_path, rows)
            print(f"[DONE] {candidate.label}/L{layer} N={count} LogME={result['logme']:.12f} time={row['runtime_seconds']:.1f}s", flush=True)
            del gram, cross, yy
            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

    rows.sort(key=lambda row: (row["architecture"], int(row["layer"])))
    write_csv(per_layer_path, rows)
    scores = []
    for candidate in CANDIDATES:
        candidate_rows = [row for row in rows if row["architecture"] == candidate.label and row.get("status") == "complete"]
        if {int(row["layer"]) for row in candidate_rows} != set(LAYERS):
            continue
        values = {int(row["layer"]): float(row["logme"]) for row in candidate_rows}
        scores.append({
            "architecture": candidate.label,
            "display_name": candidate.display_name,
            "scope": "diagnostic_not_formal_v2_candidate",
            "common7_mean_logme": sum(values[layer] for layer in LAYERS) / len(LAYERS),
            "valid_tokens_per_layer": int(candidate_rows[0]["valid_tokens"]),
            "training_videos": int(candidate_rows[0]["training_videos"]),
            "target_signature": candidate_rows[0]["target_signature"],
        })
    scores.sort(key=lambda row: -float(row["common7_mean_logme"]))
    write_csv(output / "logme_architecture_scores.csv", scores)
    completed_labels = {row["architecture"] for row in scores}
    for row in availability:
        if row["architecture"] in completed_labels and row["status"] == "unavailable":
            row["status"] = "complete_result_cache_recycled"
            row["reason"] = "completed provenance-validated LogME rows retained; regeneratable feature tensors were recycled"
    write_csv(output / "availability.csv", availability)
    write_json(output / "logme_summary.json", {
        "protocol": protocol,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "availability": availability,
        "scores": scores,
        "per_layer_rows": len(rows),
        "source_provenance": valid,
        "runtime_memory": {
            "this_invocation_wall_seconds": time.perf_counter() - started_all,
            "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "statistics_device": str(device),
        },
    })
    print(json.dumps({"output": str(output), "completed_scores": scores, "availability": availability}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
