#!/usr/bin/env python
"""Integrate the 17 requested Common-7 pre-SFT LogME model variants.

The frozen five-model v2 correlation is copied unchanged.  All other rows are
diagnostic, including exact representation aliases for auxiliary-loss models
and the plain-base construction used by the 0-spatial control.
"""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
FORMAL = ROOT / "logs/pre_sft_logme_proxy_v2_common7"
EXTRA = ROOT / "logs/pre_sft_logme_proxy_v2_common7_extra_diagnostics"
CONTROLLED = ROOT / "logs/pre_sft_logme_proxy_controlled_fusion_v1"
REMAINING = ROOT / "logs/pre_sft_logme_proxy_remaining_diagnostics_v1"
OUTPUT = ROOT / "logs/pre_sft_logme_proxy_complete17_common7"
LAYERS = (1, 3, 6, 9, 15, 21, 27)
TARGET_SIGNATURE = "b48e025fefb19e5d7414d2d540b9904a4e21d6de883552699d5e4dc194956d37"

MODEL_INFO = {
    "c1_vlm3r": ("C1 VLM3R Baseline", 59.3, "formal"),
    "c1_spatialstack_add": ("SpatialStack additive 0/1/2", 61.2, "formal"),
    "c1_spatialstack_add_123": ("SpatialStack additive 1/2/3", 62.2, "formal"),
    "c1_spatialstack_add_036": ("SpatialStack additive 0/3/6", 61.2, "formal"),
    "c1_spatialstack_cross_attn_v1": ("SpatialStack cross-attention 0/1/2", 60.6, "formal"),
    "ss_depth": ("SS + depth", 61.3, "diagnostic_exact_alias"),
    "baseline_depth": ("Baseline + depth", 59.6, "diagnostic_exact_alias"),
    "c1_geo_rope_fusion": ("GeoRoPE Fusion", 60.2, "diagnostic"),
    "c1_eomt_object": ("Extra Object Token", 58.6, "diagnostic"),
    "pre_sft_base_vlm": ("0 spatial / Base VLM", 56.4, "diagnostic_plain_base_identity"),
    "c1_visual_geo_rope": ("Visual geo-RoPE", 57.9, "diagnostic"),
    "c1_vlm3r_eomt_selective": ("Selective fusion", 57.2, "diagnostic"),
    "c1_controlled_b": ("Controlled B: pre-projector add dec12", 57.84309940577166, "diagnostic"),
    "c1_controlled_c": ("Controlled C: cross-attn dec12 -> L0", 60.60944047928047, "diagnostic"),
    "c1_controlled_d": ("Controlled D: additive dec12 -> L0", 61.17079824298335, "diagnostic"),
    "c1_controlled_e": ("Controlled E: additive dec12 -> L0/L1/L2", 61.008630184756484, "diagnostic"),
    "c1_controlled_h": ("Controlled H: cross-attn dec12 -> L0/L1/L2", 59.87772232326709, "diagnostic"),
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def finite(row: dict[str, Any], key: str) -> float:
    result = float(row[key])
    if not math.isfinite(result):
        raise RuntimeError(f"Non-finite {key}: {row.get('architecture')}/L{row.get('layer')}")
    return result


def fmt(value: float) -> str:
    return f"{value:.6f}"


def main() -> None:
    sources = {
        "formal_v2_common7": FORMAL / "logme_per_layer.csv",
        "existing_diagnostics": EXTRA / "logme_per_layer.csv",
        "controlled_fusion": CONTROLLED / "logme_per_layer.csv",
        "remaining_diagnostics": REMAINING / "logme_per_layer.csv",
        "loss_only_aliases": REMAINING / "loss_only_logme_per_layer.csv",
    }
    rows: list[dict[str, Any]] = []
    for source, path in sources.items():
        for original in read_csv(path):
            row: dict[str, Any] = dict(original)
            row["source_study"] = source
            rows.append(row)

    expected_pairs = {(model, layer) for model in MODEL_INFO for layer in LAYERS}
    actual_pairs = [(str(row["architecture"]), int(row["layer"])) for row in rows]
    if len(rows) != 119 or len(set(actual_pairs)) != 119 or set(actual_pairs) != expected_pairs:
        missing = sorted(expected_pairs - set(actual_pairs))
        duplicates = sorted(pair for pair in set(actual_pairs) if actual_pairs.count(pair) > 1)
        raise RuntimeError(f"Expected 17 x 7 complete rows; rows={len(rows)}, missing={missing}, duplicates={duplicates}")

    checks = {
        "common_seven_layers": {int(row["layer"]) for row in rows} == set(LAYERS),
        "same_target_signature": {row["target_signature"] for row in rows} == {TARGET_SIGNATURE},
        "same_valid_tokens": {int(row["valid_tokens"]) for row in rows} == {394352},
        "same_feature_dimension": {int(row["feature_dimension"]) for row in rows} == {3584},
        "same_training_videos": {int(row["training_videos"]) for row in rows} == {1006},
        "same_selected_frames": {int(row["training_frames"]) for row in rows} == {2012},
        "finite_logme": all(math.isfinite(finite(row, "logme")) for row in rows),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Protocol consistency failure: {[name for name, passed in checks.items() if not passed]}")

    formal_protocol = read_json(FORMAL / "protocol.json")
    remaining_protocol = read_json(REMAINING / "protocol.json")
    formal_summary = read_json(FORMAL / "logme_summary.json")
    loss_attestation = read_json(REMAINING / "loss_only_forward_equivalence.json")
    if loss_attestation.get("assessment") != "PASS":
        raise RuntimeError("Loss-only forward-equivalence attestation is not PASS")
    protocol_checks = {
        "float64_logme": formal_protocol.get("dtype") == remaining_protocol.get("dtype") == "float64",
        "no_optimizer": formal_protocol.get("no_optimizer") is True and remaining_protocol.get("no_optimizer") is True,
        "no_vlm_forward_during_logme": formal_protocol.get("no_vlm_forward") is True and remaining_protocol.get("no_vlm_forward") is True,
        "no_post_sft_substitution": formal_protocol.get("no_post_sft_cache") is True and remaining_protocol.get("no_post_sft_checkpoint") is True,
        "loss_only_aliases_bitwise_attested": loss_attestation.get("assessment") == "PASS",
        "zero_spatial_pre_sft_construction_is_plain_base": True,
    }
    if not all(protocol_checks.values()):
        raise RuntimeError(f"Protocol metadata failure: {protocol_checks}")

    by_key = {(str(row["architecture"]), int(row["layer"])): row for row in rows}
    means = {
        model: sum(finite(by_key[model, layer], "logme") for layer in LAYERS) / len(LAYERS)
        for model in MODEL_INFO
    }
    ranked = sorted(means, key=lambda model: (-means[model], model))
    architecture_rows = []
    for rank, model in enumerate(ranked, start=1):
        name, vsi, scope = MODEL_INFO[model]
        architecture_rows.append({
            "rank_all_17": rank,
            "architecture": model,
            "display_name": name,
            "common7_mean_logme": means[model],
            "post_sft_vsi_score": vsi,
            "formal_or_diagnostic": scope,
        })

    for row in rows:
        name, vsi, scope = MODEL_INFO[str(row["architecture"])]
        row["display_name"] = name
        row["post_sft_vsi_score"] = vsi
        row["formal_or_diagnostic"] = scope
    rows.sort(key=lambda row: (ranked.index(str(row["architecture"])), LAYERS.index(int(row["layer"]))))

    layer_rankings = []
    for layer in LAYERS:
        order = sorted(MODEL_INFO, key=lambda model: (-finite(by_key[model, layer], "logme"), model))
        for rank, model in enumerate(order, start=1):
            layer_rankings.append({
                "layer": layer, "rank": rank, "architecture": model,
                "display_name": MODEL_INFO[model][0], "logme": finite(by_key[model, layer], "logme"),
            })

    regions = {"early": (1, 3, 6), "middle": (9, 15), "late": (21, 27)}
    region_rows = []
    for model in ranked:
        region_rows.append({
            "architecture": model,
            "display_name": MODEL_INFO[model][0],
            **{
                region: sum(finite(by_key[model, layer], "logme") for layer in layers) / len(layers)
                for region, layers in regions.items()
            },
            "common7_mean_logme": means[model],
        })

    pairwise = [
        {
            "row_architecture": row_model,
            "column_architecture": column_model,
            "delta_row_minus_column": means[row_model] - means[column_model],
        }
        for row_model in ranked for column_model in ranked
    ]

    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT / "complete_logme_per_layer.csv", rows)
    write_csv(OUTPUT / "complete_logme_architecture_scores.csv", architecture_rows)
    write_csv(OUTPUT / "complete_pairwise_logme_deltas.csv", pairwise)
    write_csv(OUTPUT / "complete_layerwise_rankings.csv", layer_rankings)
    write_csv(OUTPUT / "complete_region_summary.csv", region_rows)

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_count": 17,
        "distinct_pre_sft_representations": 14,
        "exact_representation_aliases": {
            "ss_depth": "c1_spatialstack_add",
            "baseline_depth": "c1_vlm3r",
            "0_spatial": "pre_sft_base_vlm",
        },
        "sources": {name: str(path) for name, path in sources.items()},
        "data_checks": checks,
        "protocol_checks": protocol_checks,
        "architecture_scores": architecture_rows,
        "formal_five_correlation_unchanged": {
            "primary": formal_summary["primary_correlation"],
            "per_layer": formal_summary["per_layer_diagnostics"],
            "fixed_regions": formal_summary["fixed_region_diagnostics"],
            "leave_one_layer_out": formal_summary["leave_one_layer_out"],
            "leave_one_layer_out_spearman": formal_summary["leave_one_layer_out_spearman"],
        },
        "scope": "Only the original five architectures belong to the frozen formal correlation; all added models and all-model rankings are descriptive diagnostics.",
    }
    write_json(OUTPUT / "complete_summary.json", summary)

    per_layer_table = [
        "| Rank | Model | L1 | L3 | L6 | L9 | L15 | L21 | L27 | Mean | VSI | Scope |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for rank, model in enumerate(ranked, start=1):
        name, vsi, scope = MODEL_INFO[model]
        values = " | ".join(fmt(finite(by_key[model, layer], "logme")) for layer in LAYERS)
        per_layer_table.append(f"| {rank} | {name} | {values} | {fmt(means[model])} | {vsi:.3f} | {scope} |")
    primary = formal_summary["primary_correlation"]
    md = [
        "# Complete 17-model pre-SFT Common-7 LogME inventory", "",
        "This integrates the frozen formal five, selective fusion, the 0-spatial/plain-base control, five controlled variants, three geometry/object diagnostics, and two bitwise-attested auxiliary-loss aliases. No post-SFT representation was used. The official formal-five correlation is copied unchanged and was not recomputed with diagnostics.", "",
        *per_layer_table, "",
        "## Protocol", "",
        "All rows use L1/L3/L6/L9/L15/L21/L27, 1,006 calibration/train videos, 2,012 selected frames, 394,352 valid target tokens per layer, D=3,584 and the same target signature. LogME evidence is float64 and cache-only; there is no optimizer, update, SFT checkpoint substitution, or VLM forward during evidence fitting.", "",
        "`SS + depth` and `Baseline + depth` are exact aliases only after a one-video, all-retained-level bitwise hidden-feature equality test with fresh auxiliary heads. `0 spatial / Base VLM` uses the current extension protocol's explicit `plain_base` pre-SFT construction.", "",
        "## Frozen official result", "",
        f"Formal five Common-7 Spearman rho: **{float(primary['spearman_rho']):.12f}**; Kendall tau-b: **{float(primary['kendall_tau_b']):.12f}**.", "",
        "## Sources", "",
        *[f"- `{name}`: `{path}`" for name, path in sources.items()],
    ]
    (OUTPUT / "complete_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "models": 17, "rows": 119, "top": architecture_rows[:3]}, sort_keys=True))


if __name__ == "__main__":
    main()
