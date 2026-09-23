#!/usr/bin/env python3
"""Combine three frozen base-VLM depth probes and common-seven LogME in one CSV.

This exports existing diagnostic measurements; it does not run or modify a
model. Missing historical feature levels remain missing rather than imputed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


SPLIT_SHA256 = "d478cb684958dfc25066821ec83d5216469577c9e282e33bdf87d3c88b200d8e"
COMMON7 = (1, 3, 6, 9, 15, 21, 27)
VAL_TOKENS = 75656
TRAIN_TOKENS = 394352
BASE_ROOT = Path("/home/shaoruei/probe_cache/scannet_depth_layers_v1/full")
QWEN_ROOT = Path("/mnt/DATA_SSD/shaoruei/probing_data/qwen35_presft_probe_v1")
INTERN_ROOT = Path("/mnt/DATA_SSD/shaoruei/probing_data/internvl3_presft_probe_v2")
BASE_LOGME = Path("logs/pre_sft_logme_proxy_combined7")

FIELDS = (
    "record_type", "model", "model_label", "feature_level", "llm_layer",
    "mae", "absrel", "delta125", "logme", "best_epoch", "feature_dim",
    "val_valid_tokens", "train_valid_tokens", "train_videos", "val_videos",
    "forward_frames", "target_frames", "sample_indices_sha256", "logme_target_sha256",
    "checkpoint_path", "checkpoint_config_sha256", "weight_index_sha256",
    "source_path", "notes",
)


def read_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def finite(value: object, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Non-finite {name}: {value}")
    return number


def base_sources() -> dict:
    provenance_path = BASE_ROOT / "features/pre_sft_base_vlm/extraction_provenance.json"
    provenance = read_json(provenance_path)
    if provenance["sample_indices_sha256"] != SPLIT_SHA256:
        raise RuntimeError("Historical base VLM split mismatch")
    if not provenance["no_vlm3r_sft_adapter_loaded"] or not provenance["no_cut3r_or_spatial_sidecar_usage"]:
        raise RuntimeError("Historical base VLM is not a frozen no-spatial comparator")
    depth_paths = sorted((BASE_ROOT / "probes/pre_sft_base_vlm").glob("*/metrics.json"))
    depth = [(path, read_json(path)) for path in depth_paths]
    if len(depth) != 14 or {row["feature_level"] for _, row in depth} != {
        "siglip_output", "projected_features",
        *(f"layer_{layer}" for layer in (0, 1, 2, 3, 6, 9, 12, 15, 18, 21, 24, 27)),
    }:
        raise RuntimeError("Historical base depth layer set is not the audited 14-level cache")
    layer_path = BASE_LOGME / "combined_logme_per_layer.csv"
    with layer_path.open(newline="", encoding="utf-8") as handle:
        logme = [row for row in csv.DictReader(handle) if row["architecture"] == "pre_sft_base_vlm"]
    with (BASE_LOGME / "combined_logme_architecture_scores.csv").open(newline="", encoding="utf-8") as handle:
        summary = next(row for row in csv.DictReader(handle) if row["architecture"] == "pre_sft_base_vlm")
    return {
        "model": "LLaVA-NeXT-Video-7B-Qwen2",
        "label": "pre_sft_base_vlm",
        "depth": depth,
        "logme": [(layer_path, int(row["layer"]), row["logme"], int(row["valid_tokens"]),
                    row["target_signature"], int(row["feature_dimension"])) for row in logme],
        "mean_logme": summary["common7_mean_logme"],
        "checkpoint_path": provenance["base_model_path"],
        "checkpoint_config_sha256": provenance["base_model_config_sha256"],
        "weight_index_sha256": "",
        "notes": "Historical 14-level cache; fusion_output was not extracted; siglip_output is the visual level.",
    }


def new_sources(root: Path, label: str, model: str, expected_depth: int) -> dict:
    manifest_path = root / f"{label}_run_manifest.json"
    manifest = read_json(manifest_path)
    if (manifest.get("status"), manifest.get("complete_videos"), manifest.get("optimizer_steps")) != ("complete", 1199, 0):
        raise RuntimeError(f"Incomplete or non-frozen feature manifest: {manifest_path}")
    if manifest["sample_indices_sha256"] != SPLIT_SHA256:
        raise RuntimeError(f"Split mismatch: {manifest_path}")
    if manifest["frames_per_video"] != 32 or manifest["selected_target_frames_per_video"] != 2:
        raise RuntimeError(f"Frame protocol mismatch: {manifest_path}")
    depth_paths = sorted((root / "probes" / label).glob("*/metrics.json"))
    depth = [(path, read_json(path)) for path in depth_paths]
    if len(depth) != expected_depth or {row["feature_level"] for _, row in depth} != set(manifest["feature_levels"]):
        raise RuntimeError(f"Incomplete depth measurements for {label}")
    summary_path = root / "logme_common7" / label / "summary.json"
    summary = read_json(summary_path)
    if summary["sample_indices_sha256"] != SPLIT_SHA256 or tuple(summary["layers"]) != COMMON7:
        raise RuntimeError(f"LogME protocol mismatch for {label}")
    logme = []
    for layer in COMMON7:
        path = summary_path.parent / f"layer_{layer}.json"
        row = read_json(path)
        logme.append((path, layer, row["logme"], int(row["valid_tokens"]),
                      row["target_sha256"], int(row["feature_dim"])))
    return {
        "model": model, "label": label, "depth": depth,
        "logme": logme, "mean_logme": summary["mean_logme"],
        "checkpoint_path": manifest["model_root"],
        "checkpoint_config_sha256": manifest.get("model_config_sha256", ""),
        "weight_index_sha256": manifest["model_weight_index_sha256"],
        "notes": "No spatial tokens; fusion_output is an identity copy of visual_output.",
    }


def build_rows() -> list[dict]:
    sources = [
        base_sources(),
        new_sources(QWEN_ROOT, "qwen35_base_presft", "Qwen3.5-4B", 17),
        new_sources(INTERN_ROOT, "internvl3_8b_base_presft", "InternVL3-8B", 15),
    ]
    signatures = {signature for source in sources for _, _, _, _, signature, _ in source["logme"]}
    if len(signatures) != 1:
        raise RuntimeError(f"The three LogME target sequences differ: {signatures}")
    target_sha256 = signatures.pop()
    rows = []
    for source in sources:
        if len(source["logme"]) != len(COMMON7) or {layer for _, layer, *_ in source["logme"]} != set(COMMON7):
            raise RuntimeError(f"Common-seven LogME is incomplete for {source['label']}")
        scores = [finite(score, "LogME") for _, _, score, tokens, _, _ in source["logme"]]
        if any(tokens != TRAIN_TOKENS for _, _, _, tokens, _, _ in source["logme"]):
            raise RuntimeError(f"LogME training-token count mismatch for {source['label']}")
        if not math.isclose(sum(scores) / len(scores), finite(source["mean_logme"], "mean LogME"), abs_tol=1e-10):
            raise RuntimeError(f"LogME mean does not match per-layer values for {source['label']}")
        common = {
            "model": source["model"], "model_label": source["label"],
            "train_videos": 1006, "val_videos": 193, "forward_frames": 32,
            "target_frames": 2, "sample_indices_sha256": SPLIT_SHA256,
            "logme_target_sha256": target_sha256,
            "checkpoint_path": source["checkpoint_path"],
            "checkpoint_config_sha256": source["checkpoint_config_sha256"],
            "weight_index_sha256": source["weight_index_sha256"],
            "notes": source["notes"],
        }
        depth_rows = []
        for path, metric in source["depth"]:
            if metric["model_label"] != source["label"] or int(metric["num_tokens"]) != VAL_TOKENS:
                raise RuntimeError(f"Depth result identity/count mismatch: {path}")
            row = {
                **common, "record_type": "depth", "feature_level": metric["feature_level"],
                "llm_layer": metric["feature_level"].removeprefix("layer_") if metric["feature_level"].startswith("layer_") else "",
                "mae": finite(metric["mae"], "MAE"),
                "absrel": finite(metric["absrel"], "AbsRel"),
                "delta125": finite(metric["delta125"], "delta125"),
                "best_epoch": int(metric["best_epoch"]),
                "feature_dim": int(metric["d_in"]), "val_valid_tokens": VAL_TOKENS,
                "source_path": str(path),
            }
            depth_rows.append(row)
        best = min(depth_rows, key=lambda row: row["mae"])
        rows.append({
            **common, "record_type": "model_summary", "feature_level": best["feature_level"],
            "llm_layer": best["llm_layer"], "mae": best["mae"],
            "absrel": best["absrel"], "delta125": best["delta125"],
            "logme": finite(source["mean_logme"], "mean LogME"),
            "best_epoch": best["best_epoch"], "feature_dim": best["feature_dim"],
            "val_valid_tokens": VAL_TOKENS, "train_valid_tokens": TRAIN_TOKENS,
            "source_path": best["source_path"],
            "notes": "Best-MAE depth level and seven-layer mean LogME. " + source["notes"],
        })
        rows.extend(depth_rows)
        for path, layer, score, tokens, _, dim in source["logme"]:
            rows.append({
                **common, "record_type": "logme", "feature_level": f"layer_{layer}",
                "llm_layer": layer, "logme": finite(score, "LogME"),
                "feature_dim": dim, "train_valid_tokens": tokens,
                "source_path": str(path),
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("reports/three_base_vlm_presft_probe_logme_20260924.csv"))
    args = parser.parse_args()
    rows = build_rows()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # UTF-8 BOM lets Excel detect the file correctly without changing values.
    with args.output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{args.output}: {len(rows)} rows; 3 summaries, 46 depth, 21 LogME")


if __name__ == "__main__":
    main()
