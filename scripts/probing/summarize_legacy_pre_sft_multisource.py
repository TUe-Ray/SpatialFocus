#!/usr/bin/env python
"""Validate and combine seven legacy pre-SFT probes from audited source runs."""

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

from scripts.probing.legacy_pre_sft_completion_specs import (
    EXPECTED_VALIDATION_TOKENS,
    FULL_FEATURE_LEVELS,
    LEGACY_PARTIAL_CANDIDATES,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-indices", type=Path, required=True)
    parser.add_argument(
        "--source",
        action="append",
        nargs=4,
        metavar=("CANDIDATE", "ARTIFACT_MANIFEST", "RESULT_ROOT", "COVERAGE_AUDIT_OR_DASH"),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def validate_runtime_coverage(
    identifier: str,
    provenance: dict[str, Any],
    provenance_path: Path,
    audit_path: Path | None,
    sample_hash: str,
) -> dict[str, Any]:
    rows = provenance.get("extraction_samples", [])
    if len(rows) == 1199 and all(int(row.get("source_video_num_frames", -1)) == 32 for row in rows):
        return {"mode": "complete_extraction_provenance", "runtime_rows": 1199}
    require(audit_path is not None, f"{identifier}: incomplete runtime rows and no resume coverage audit")
    audit = read_json(audit_path)
    require(audit.get("status") == "PASS_WITH_EXPLICIT_RESUME_PROVENANCE", f"{identifier}: resume audit did not pass")
    require(audit.get("candidate") == identifier, f"{identifier}: resume audit candidate differs")
    require(audit.get("sample_indices_sha256") == sample_hash, f"{identifier}: resume audit split differs")
    require(audit.get("extraction_provenance_sha256") == sha256(provenance_path), f"{identifier}: resume audit provenance hash differs")
    require(int(audit.get("fixed_videos", -1)) == 1199, f"{identifier}: resume audit has wrong video count")
    require(int(audit.get("fixed_target_frames", -1)) == 2398, f"{identifier}: resume audit has wrong frame count")
    require(int(audit.get("model_forward_frames_per_video", -1)) == 32, f"{identifier}: resume audit has wrong forward-frame count")
    require(audit.get("source_provenance_was_modified") is False, f"{identifier}: source provenance was modified")
    require(audit.get("post_sft_state_loaded") is False, f"{identifier}: resume audit does not exclude post-SFT state")
    coverage = audit.get("coverage", {})
    require(
        all(coverage.get(level, {}).get("exact_manifest_match") is True for level in FULL_FEATURE_LEVELS),
        f"{identifier}: resume audit lacks exact 15-level frame coverage",
    )
    return {
        "mode": "explicit_resumed_cache_coverage_audit",
        "runtime_rows": len(rows),
        "audit": str(audit_path.resolve()),
        "audit_sha256": sha256(audit_path),
    }


def main() -> None:
    args = parse_args()
    output_paths = [args.output_dir / name for name in ("results.json", "metrics.csv", "summary.md")]
    require(not any(path.exists() for path in output_paths), "Refusing to overwrite existing multisource summary")
    expected_ids = [candidate.identifier for candidate in LEGACY_PARTIAL_CANDIDATES]
    sources = {entry[0]: entry[1:] for entry in args.source}
    require(len(sources) == len(args.source), "Duplicate --source candidate")
    require(set(sources) == set(expected_ids), f"Sources must be exactly {expected_ids}")
    sample_hash = sha256(args.sample_indices)

    rows: list[dict[str, Any]] = []
    source_records: dict[str, Any] = {}
    for spec in LEGACY_PARTIAL_CANDIDATES:
        identifier = spec.identifier
        manifest_path = Path(sources[identifier][0])
        result_root = Path(sources[identifier][1])
        audit_path = None if sources[identifier][2] == "-" else Path(sources[identifier][2])
        manifest = read_json(manifest_path)
        require(manifest.get("schema_version") == "legacy_pre_sft_completion_manifest_v1", f"{identifier}: wrong manifest schema")
        require(manifest.get("post_sft_state_loaded") is False, f"{identifier}: manifest does not exclude post-SFT state")
        require(manifest.get("sample_indices", {}).get("sha256") == sample_hash, f"{identifier}: manifest split differs")
        require(tuple(manifest.get("feature_levels", [])) == FULL_FEATURE_LEVELS, f"{identifier}: manifest lacks 15-level policy")
        require(identifier in manifest.get("candidates", {}), f"{identifier}: absent from source manifest")
        candidate = manifest["candidates"][identifier]

        provenance_path = result_root / "extraction_provenance.json"
        provenance = read_json(provenance_path)
        require(provenance.get("git_worktree_dirty") is False, f"{identifier}: dirty extraction worktree")
        require(provenance.get("git_commit") == manifest.get("git_commit"), f"{identifier}: extraction commit differs")
        require(provenance.get("sample_indices_sha256") == sample_hash, f"{identifier}: extraction split differs")
        require(provenance.get("no_vlm3r_sft_adapter_loaded") is True, f"{identifier}: no-post-SFT proof missing")
        require(set(provenance.get("requested_feature_levels", [])) == set(FULL_FEATURE_LEVELS), f"{identifier}: incomplete requested levels")
        require(provenance.get("c1_calibration_sha256") == candidate["c1"]["sha256"], f"{identifier}: C1 hash differs")
        if candidate.get("geometry_c1"):
            require(
                provenance.get("geometry_c1_calibration_sha256") == candidate["geometry_c1"]["sha256"],
                f"{identifier}: geometry C1 hash differs",
            )
        if candidate.get("uses_eomt_selective_gate"):
            require(provenance.get("eomt_selective_kv_gate") is True, f"{identifier}: selective EoMT gate inactive")
            require(
                provenance.get("eomt_cache_validation_sha256") == manifest["eomt"]["validation_sha256"],
                f"{identifier}: EoMT validation hash differs",
            )
        runtime_coverage = validate_runtime_coverage(identifier, provenance, provenance_path, audit_path, sample_hash)

        for level in FULL_FEATURE_LEVELS:
            metric_path = result_root / "probes" / level / "metrics.json"
            metric = read_json(metric_path)
            values = {name: float(metric.get(name, float("nan"))) for name in ("mae", "absrel", "delta125")}
            require(all(math.isfinite(value) for value in values.values()), f"{identifier}/{level}: non-finite metric")
            require(int(metric.get("num_tokens", -1)) == EXPECTED_VALIDATION_TOKENS, f"{identifier}/{level}: wrong token count")
            rows.append(
                {
                    "candidate": identifier,
                    "display_name": candidate["display_name"],
                    "model_label": candidate["label"],
                    "feature_level": level,
                    **values,
                    "best_epoch": int(metric["best_epoch"]),
                    "num_tokens": int(metric["num_tokens"]),
                    "metrics_path": str(metric_path.resolve()),
                }
            )
        source_records[identifier] = {
            "artifact_manifest": str(manifest_path.resolve()),
            "artifact_manifest_sha256": sha256(manifest_path),
            "git_commit": manifest["git_commit"],
            "result_root": str(result_root.resolve()),
            "extraction_provenance": str(provenance_path.resolve()),
            "extraction_provenance_sha256": sha256(provenance_path),
            "runtime_coverage": runtime_coverage,
        }

    payload = {
        "schema_version": "legacy_pre_sft_multisource_summary_v1",
        "experiment_label": "seven legacy pre-SFT depth probes",
        "sample_indices": str(args.sample_indices.resolve()),
        "sample_indices_sha256": sample_hash,
        "feature_levels": list(FULL_FEATURE_LEVELS),
        "expected_validation_tokens": EXPECTED_VALIDATION_TOKENS,
        "sources": source_records,
        "rows": rows,
        "post_sft_state_loaded": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_paths[0].write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with output_paths[1].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Seven legacy pre-SFT depth probes",
        "",
        "All rows use the fixed 1,199-video split, 75,656 validation tokens, and complete 15-representation policy.",
        "",
        "| Candidate | Feature | MAE | AbsRel | delta<1.25 | Best epoch |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['candidate']} | {row['feature_level']} | {row['mae']:.6g} | {row['absrel']:.6g} | "
            f"{row['delta125']:.6g} | {row['best_epoch']} |"
        )
    output_paths[2].write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": str(output_paths[0]), "candidates": len(expected_ids), "rows": len(rows)}))


if __name__ == "__main__":
    main()
