#!/usr/bin/env python
"""Validate and summarize the complete formal A-prime pre-SFT depth probe."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.probing.prepare_controlled_a_prime_pre_sft import (
    OFFICIAL_SPLIT_SHA256,
    SCHEMA_VERSION as MANIFEST_SCHEMA,
)
from scripts.probing.probe_layer_policy import PRE_SFT_FULL_FEATURE_LEVELS
from scripts.probing.verify_controlled_a_prime_pre_sft_smoke import validate_runtime_attestation


SCHEMA_VERSION = "controlled_a_prime_pre_sft_depth_probe_summary_v1"
EXPECTED_VALIDATION_TOKENS = 75656


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sample-indices", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = [args.output_dir / name for name in ("results.json", "metrics.csv", "summary.md")]
    if any(path.exists() for path in outputs):
        raise FileExistsError(f"Refusing to overwrite A-prime summary: {[str(path) for path in outputs if path.exists()]}")
    manifest = read_json(args.manifest)
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise ValueError("A-prime summary received an incompatible manifest")
    split_hash = sha256_file(args.sample_indices)
    if split_hash != OFFICIAL_SPLIT_SHA256 or split_hash != manifest.get("sample_indices_sha256"):
        raise ValueError("A-prime summary split differs from the official 1,199-video manifest")
    samples = read_json(args.sample_indices)
    if len(samples.get("videos", [])) != 1199 or sum(len(v.get("frames", [])) for v in samples["videos"]) != 2398:
        raise ValueError("A-prime summary requires exactly 1,199 videos and 2,398 selected frames")
    split_counts = {
        split: sum(str(video.get("split")) == split for video in samples["videos"])
        for split in ("train", "val")
    }
    if split_counts != {"train": 1006, "val": 193}:
        raise ValueError(f"A-prime summary received the wrong train/validation split: {split_counts}")
    entries = {
        "BASE": ("Plain pre-SFT base VLM", "pre_sft_base_vlm"),
        "A_prime": ("A-prime controlled pre-projector cross-attention", "controlled_a_prime"),
    }
    rows: list[dict[str, Any]] = []
    provenance_records: dict[str, Any] = {}
    for identifier, (display, label) in entries.items():
        root = args.results_root / identifier
        provenance_path = root / "extraction_provenance.json"
        provenance = read_json(provenance_path)
        if provenance.get("git_worktree_dirty") is not False or provenance.get("git_commit") != manifest.get("git_commit"):
            raise ValueError(f"{identifier} extraction is not a clean same-commit formal run")
        if provenance.get("sample_indices_sha256") != OFFICIAL_SPLIT_SHA256:
            raise ValueError(f"{identifier} did not use the official ScanNet split")
        if provenance.get("no_vlm3r_sft_adapter_loaded") is not True:
            raise ValueError(f"{identifier} lacks no-post-SFT attestation")
        if provenance.get("requested_feature_levels") != list(PRE_SFT_FULL_FEATURE_LEVELS):
            raise ValueError(f"{identifier} does not cover the complete 15-level policy")
        if identifier == "A_prime":
            if provenance.get("experiment_variant") != "controlled_a_prime":
                raise ValueError("A-prime full extraction variant mismatch")
            if provenance.get("fusion_init_seed") != 42 or provenance.get("c1_calibration_json") is not None:
                raise ValueError("A-prime initialization is not the official seed-42 non-C1 state")
            if provenance.get("fresh_fusion_state_sha256") != manifest.get("fresh_fusion_state_sha256"):
                raise ValueError("A-prime full extraction fusion state differs from the manifest")
            validate_runtime_attestation(provenance)
        provenance_records[identifier] = {"path": str(provenance_path), "sha256": sha256_file(provenance_path)}
        for level in PRE_SFT_FULL_FEATURE_LEVELS:
            metric_path = root / "probes" / level / "metrics.json"
            metric = read_json(metric_path)
            values = {name: float(metric.get(name, float("nan"))) for name in ("mae", "absrel", "delta125")}
            if not all(math.isfinite(value) for value in values.values()):
                raise ValueError(f"{identifier}/{level} has non-finite metrics")
            if int(metric.get("num_tokens", -1)) != EXPECTED_VALIDATION_TOKENS:
                raise ValueError(f"{identifier}/{level} does not contain {EXPECTED_VALIDATION_TOKENS} validation tokens")
            rows.append({
                "candidate": identifier,
                "display_name": display,
                "model_label": label,
                "feature_level": level,
                **values,
                "num_tokens": int(metric["num_tokens"]),
                "metrics_path": str(metric_path),
            })
    rankings = {
        level: [row["candidate"] for row in sorted((r for r in rows if r["feature_level"] == level), key=lambda r: r["mae"])]
        for level in PRE_SFT_FULL_FEATURE_LEVELS
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "experiment_label": "A-prime non-C1 pre-SFT architecture probe",
        "formal_existing_five_candidate_roster_modified": False,
        "controlled_b_c_d_e_h_artifacts_modified": False,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "sample_indices_sha256": split_hash,
        "video_count": 1199,
        "train_video_count": 1006,
        "validation_video_count": 193,
        "selected_frame_count": 2398,
        "feature_levels": list(PRE_SFT_FULL_FEATURE_LEVELS),
        "expected_validation_tokens": EXPECTED_VALIDATION_TOKENS,
        "provenance": provenance_records,
        "rows": rows,
        "mae_ranking_ascending_by_feature": rankings,
        "post_sft_state_loaded": False,
        "candidate_model_optimizer_constructed": False,
        "candidate_model_optimizer_step": False,
        "depth_probe_optimizer_used": True,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs[0].write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with outputs[1].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# A-prime non-C1 pre-SFT depth probe",
        "",
        "A-prime is a separate seed-42 extension and does not alter the frozen formal C1 roster.",
        "",
        "| Candidate | Feature | MAE | AbsRel | delta<1.25 | Validation tokens |",
        "|---|---|---:|---:|---:|---:|",
    ]
    lines.extend(
        f"| {row['candidate']} | {row['feature_level']} | {row['mae']:.6g} | {row['absrel']:.6g} | {row['delta125']:.6g} | {row['num_tokens']} |"
        for row in rows
    )
    outputs[2].write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "rows": len(rows), "output": str(outputs[0])}))


if __name__ == "__main__":
    main()
