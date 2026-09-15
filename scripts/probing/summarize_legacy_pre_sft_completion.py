#!/usr/bin/env python
"""Validate and summarize complete 15-feature legacy pre-SFT probe results."""

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

from scripts.probing.legacy_pre_sft_completion_specs import EXPECTED_VALIDATION_TOKENS


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
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--artifact-manifest", type=Path, required=True)
    parser.add_argument("--sample-indices", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = [args.output_dir / name for name in ("results.json", "metrics.csv", "summary.md")]
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite completion summary files: {existing}")
    manifest = read_json(args.artifact_manifest)
    if manifest.get("schema_version") != "legacy_pre_sft_completion_manifest_v1":
        raise ValueError("Incompatible completion artifact manifest")
    if manifest.get("post_sft_state_loaded") is not False:
        raise ValueError("Completion manifest does not prove post-SFT state is excluded")
    sample_sha256 = sha256(args.sample_indices)
    if manifest.get("sample_indices", {}).get("sha256") != sample_sha256:
        raise ValueError("Summary split differs from locked artifact manifest")
    levels = list(manifest.get("feature_levels", []))
    if len(levels) != 15 or len(set(levels)) != 15:
        raise ValueError("Completion manifest lacks the complete 15-feature policy")

    rows: list[dict[str, Any]] = []
    provenance_records: dict[str, Any] = {}
    for identifier, candidate in manifest["candidates"].items():
        root = args.results_root / identifier
        provenance_path = root / "extraction_provenance.json"
        provenance = read_json(provenance_path)
        if provenance.get("git_worktree_dirty") is not False:
            raise ValueError(f"{identifier}: full extraction used a dirty worktree")
        if provenance.get("git_commit") != manifest.get("git_commit"):
            raise ValueError(f"{identifier}: full extraction commit differs from locked manifest")
        if provenance.get("sample_indices_sha256") != sample_sha256:
            raise ValueError(f"{identifier}: full extraction split differs from locked manifest")
        if provenance.get("no_vlm3r_sft_adapter_loaded") is not True:
            raise ValueError(f"{identifier}: full extraction lacks no-post-SFT-adapter proof")
        if set(provenance.get("requested_feature_levels", [])) != set(levels):
            raise ValueError(f"{identifier}: full extraction does not cover all 15 representations")
        if provenance.get("c1_calibration_sha256") != candidate["c1"]["sha256"]:
            raise ValueError(f"{identifier}: C1 artifact differs from locked manifest")
        if candidate.get("geometry_c1") and provenance.get("geometry_c1_calibration_sha256") != candidate["geometry_c1"]["sha256"]:
            raise ValueError(f"{identifier}: geometry C1 artifact differs from locked manifest")
        if candidate.get("uses_eomt_selective_gate"):
            if provenance.get("eomt_selective_kv_gate") is not True:
                raise ValueError(f"{identifier}: selective EoMT gate was not active")
            if provenance.get("eomt_cache_validation_sha256") != manifest["eomt"]["validation_sha256"]:
                raise ValueError(f"{identifier}: EoMT validation differs from locked manifest")
        provenance_records[identifier] = {"path": str(provenance_path), "sha256": sha256(provenance_path)}
        for level in levels:
            metric_path = root / "probes" / level / "metrics.json"
            metric = read_json(metric_path)
            values = {name: float(metric.get(name, float("nan"))) for name in ("mae", "absrel", "delta125")}
            if not all(math.isfinite(value) for value in values.values()):
                raise ValueError(f"{identifier}/{level}: non-finite metric")
            if int(metric.get("num_tokens", -1)) != EXPECTED_VALIDATION_TOKENS:
                raise ValueError(f"{identifier}/{level}: incorrect validation-token count")
            rows.append(
                {
                    "candidate": identifier,
                    "display_name": candidate["display_name"],
                    "model_label": candidate["label"],
                    "feature_level": level,
                    **values,
                    "num_tokens": int(metric["num_tokens"]),
                    "metrics_path": str(metric_path),
                }
            )

    payload = {
        "schema_version": "legacy_pre_sft_completion_summary_v1",
        "experiment_label": "legacy partial pre-SFT depth-probe completion",
        "artifact_manifest": str(args.artifact_manifest.resolve()),
        "artifact_manifest_sha256": sha256(args.artifact_manifest),
        "sample_indices": str(args.sample_indices.resolve()),
        "sample_indices_sha256": sample_sha256,
        "feature_levels": levels,
        "expected_validation_tokens": EXPECTED_VALIDATION_TOKENS,
        "provenance": provenance_records,
        "rows": rows,
        "post_sft_state_loaded": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs[0].write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with outputs[1].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Legacy partial pre-SFT depth-probe completion",
        "",
        "All rows use the complete current 15-representation policy. This completion is separate from the existing formal five-candidate zero-cost-proxy conclusions.",
        "",
        "| Candidate | Feature | MAE | AbsRel | delta<1.25 | Validation tokens |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['candidate']} | {row['feature_level']} | {row['mae']:.6g} | "
            f"{row['absrel']:.6g} | {row['delta125']:.6g} | {row['num_tokens']} |"
        )
    outputs[2].write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": str(outputs[0]), "rows": len(rows)}))


if __name__ == "__main__":
    main()
