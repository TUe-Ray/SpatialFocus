#!/usr/bin/env python
"""Verify loss-only pre-SFT feature identity and materialize LogME aliases."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.probing.run_pre_sft_logme_proxy import write_csv, write_json


LAYERS = (1, 3, 6, 9, 15, 21, 27)
# Compare exactly the retained hidden layers whose formal LogME values are
# reused.  The historical C1 source caches did not retain shared front-end
# features or v1-only L12/L18/L24, so requiring those would test files that
# are outside the amended Common-7 score rather than representation identity.
ALL_FEATURES = tuple(f"layer_{layer}" for layer in LAYERS)
SPLIT_SHA = "d478cb684958dfc25066821ec83d5216469577c9e282e33bdf87d3c88b200d8e"
FORMAL_ROWS = REPO_ROOT / "logs" / "pre_sft_logme_proxy_v2_common7" / "logme_per_layer.csv"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_leaves(value: Any, prefix: str = "root") -> dict[str, torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return {prefix: value}
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            result.update(tensor_leaves(item, f"{prefix}.{key}"))
        return result
    if isinstance(value, (list, tuple)):
        result = {}
        for index, item in enumerate(value):
            result.update(tensor_leaves(item, f"{prefix}[{index}]"))
        return result
    return {}


def load_provenance(root: Path, label: str) -> tuple[Path, dict[str, Any]]:
    path = root / "features" / label / "extraction_provenance.json"
    return path, json.loads(path.read_text(encoding="utf-8"))


def compare_pair(
    name: str,
    reference_root: Path,
    reference_label: str,
    auxiliary_root: Path,
    auxiliary_label: str,
    expected_loss: str,
) -> dict[str, Any]:
    reference_path, reference_provenance = load_provenance(reference_root, reference_label)
    auxiliary_path, auxiliary_provenance = load_provenance(auxiliary_root, auxiliary_label)
    for payload_name, payload in (("reference", reference_provenance), ("auxiliary", auxiliary_provenance)):
        if payload.get("model_loading_mode") != "pre_sft_fusion":
            raise RuntimeError(f"{name}/{payload_name}: not pre_sft_fusion")
        if payload.get("no_vlm3r_sft_adapter_loaded") is not True:
            raise RuntimeError(f"{name}/{payload_name}: no-adapter proof missing")
        if payload.get("sample_indices_sha256") != SPLIT_SHA:
            raise RuntimeError(f"{name}/{payload_name}: split mismatch")
    attestation = auxiliary_provenance.get("loss_only_forward_equivalence_attestation")
    if not isinstance(attestation, dict) or attestation.get("loss") != expected_loss:
        raise RuntimeError(f"{name}: auxiliary toggle attestation missing")
    if attestation.get("auxiliary_head_freshly_initialized") is not True:
        raise RuntimeError(f"{name}: fresh auxiliary-head proof missing")
    if int(attestation.get("auxiliary_head_parameters", 0)) <= 0:
        raise RuntimeError(f"{name}: auxiliary head has no parameters")
    for field in ("base_model_config_sha256", "siglip_config_sha256", "c1_calibration_sha256"):
        if reference_provenance.get(field) != auxiliary_provenance.get(field):
            raise RuntimeError(f"{name}: {field} differs")

    comparisons = []
    overall_max = 0.0
    compared_tensors = compared_elements = 0
    for level in ALL_FEATURES:
        reference_dir = reference_root / "features" / reference_label / level
        auxiliary_dir = auxiliary_root / "features" / auxiliary_label / level
        reference_files = {path.name: path for path in reference_dir.glob("frame_*.pt")}
        auxiliary_files = {path.name: path for path in auxiliary_dir.glob("frame_*.pt")}
        missing = sorted(set(auxiliary_files) - set(reference_files))
        if len(auxiliary_files) != 2 or missing:
            raise RuntimeError(
                f"{name}/{level}: expected the two auxiliary frames in the retained formal source cache; "
                f"auxiliary={len(auxiliary_files)}, reference={len(reference_files)}, missing={missing}"
            )
        level_max = 0.0
        level_equal = True
        for filename, auxiliary_file in sorted(auxiliary_files.items()):
            reference_file = reference_files[filename]
            reference = tensor_leaves(torch.load(reference_file, map_location="cpu"))
            auxiliary = tensor_leaves(torch.load(auxiliary_file, map_location="cpu"))
            if reference.keys() != auxiliary.keys():
                raise RuntimeError(f"{name}/{level}/{filename}: tensor schema differs")
            for key in reference:
                left, right = reference[key], auxiliary[key]
                if left.shape != right.shape or left.dtype != right.dtype:
                    raise RuntimeError(f"{name}/{level}/{filename}/{key}: shape or dtype differs")
                equal = torch.equal(left, right)
                level_equal = level_equal and equal
                difference = 0.0 if equal or left.numel() == 0 else float((left.float() - right.float()).abs().max().item())
                level_max = max(level_max, difference)
                compared_tensors += 1
                compared_elements += int(left.numel())
        overall_max = max(overall_max, level_max)
        comparisons.append({"feature_level": level, "bitwise_equal": level_equal, "maximum_absolute_difference": level_max})
    if not all(row["bitwise_equal"] for row in comparisons):
        raise RuntimeError(f"{name}: feature identity failed; max abs diff={overall_max}")
    return {
        "name": name,
        "assessment": "PASS_BITWISE_IDENTICAL",
        "loss_only_toggle": expected_loss,
        "reference_label": reference_label,
        "auxiliary_label": auxiliary_label,
        "reference_provenance": str(reference_path),
        "reference_provenance_sha256": sha256_file(reference_path),
        "auxiliary_provenance": str(auxiliary_path),
        "auxiliary_provenance_sha256": sha256_file(auxiliary_path),
        "compared_tensors": compared_tensors,
        "compared_elements": compared_elements,
        "maximum_absolute_difference": overall_max,
        "auxiliary_head_class": attestation["auxiliary_head_class"],
        "auxiliary_head_parameters": int(attestation["auxiliary_head_parameters"]),
        "features": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=Path("/home/shaoruei/probe_cache/pre_sft_logme_remaining6_equivalence"))
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "logs" / "pre_sft_logme_proxy_remaining_diagnostics_v1")
    args = parser.parse_args()
    cache = args.cache_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    results = [
        compare_pair("Baseline + depth", Path("/home/shaoruei/probe_cache/c1_vlm3r_v1/full"), "c1_vlm3r",
                     cache / "baseline_depth", "c1_vlm3r_depth_loss_attestation", "depth"),
        compare_pair("SS + depth", cache / "ss012_reference", "c1_ss012_equivalence_reference",
                     cache / "ss_depth", "c1_ss012_pointmap_loss_attestation", "pointmap"),
    ]
    attestation_path = output / "loss_only_forward_equivalence.json"
    write_json(attestation_path, {
        "schema_version": "pre_sft_loss_only_forward_equivalence_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "assessment": "PASS" if all(row["assessment"] == "PASS_BITWISE_IDENTICAL" for row in results) else "FAIL",
        "definition": (
            "Loss-only auxiliary heads consume hidden states after the representation forward. Baseline+depth "
            "is compared directly with its retained formal source cache and permits exact LogME reuse. "
            "SS+depth is compared with a current-code reference; its older formal source cache exhibits "
            "code-version numerical drift, so SS+depth must receive an independent full-cache LogME fit."
        ),
        "results": results,
    })

    with FORMAL_ROWS.open(newline="", encoding="utf-8") as handle:
        formal = list(csv.DictReader(handle))
    aliases = (("baseline_depth", "Baseline + depth", 59.6, "c1_vlm3r"),)
    rows = []
    scores = []
    for label, display, vsi, source in aliases:
        source_rows = [row for row in formal if row["architecture"] == source and int(row["layer"]) in LAYERS]
        if {int(row["layer"]) for row in source_rows} != set(LAYERS):
            raise RuntimeError(f"Incomplete formal source rows for {source}")
        layer_scores = []
        for source_row in sorted(source_rows, key=lambda row: int(row["layer"])):
            score = float(source_row["logme"])
            if not math.isfinite(score):
                raise RuntimeError(f"Non-finite source LogME for {source}/L{source_row['layer']}")
            layer_scores.append(score)
            alias_row = dict(source_row)
            alias_row.update({
                "architecture": label,
                "display_name": display,
                "layer": int(source_row["layer"]),
                "logme": score,
                "result_type": "exact_forward_equivalent_loss_only_alias",
                "source_architecture": source,
                "source_protocol_sha256": source_row["protocol_sha256"],
                "forward_equivalence_attestation": str(attestation_path),
                "valid_tokens": int(source_row["valid_tokens"]),
                "feature_dimension": int(source_row["feature_dimension"]),
                "training_videos": int(source_row["training_videos"]),
                "training_frames": int(source_row["training_frames"]),
                "target_signature": source_row["target_signature"],
            })
            rows.append(alias_row)
        scores.append({
            "architecture": label,
            "display_name": display,
            "common7_mean_logme": sum(layer_scores) / len(layer_scores),
            "post_sft_vsi_score": vsi,
            "result_type": "exact_forward_equivalent_loss_only_alias",
            "source_architecture": source,
            "forward_equivalence_attestation": str(attestation_path),
        })
    write_csv(output / "loss_only_logme_per_layer.csv", rows)
    write_csv(output / "loss_only_logme_architecture_scores.csv", scores)
    print(json.dumps({"attestation": str(attestation_path), "results": results, "scores": scores}, sort_keys=True))


if __name__ == "__main__":
    main()
