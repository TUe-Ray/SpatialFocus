#!/usr/bin/env python
"""Create a read-only Common-7 LogME inventory across all completed models.

This does not rerun evidence fitting or change the formal five-model study.
It only joins completed CSV files and validates the shared depth-probe sample
identity before computing descriptive all-model rankings.
"""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
FORMAL = ROOT / "logs" / "pre_sft_logme_proxy_v2_common7"
EXTRA = ROOT / "logs" / "pre_sft_logme_proxy_v2_common7_extra_diagnostics"
CONTROLLED = ROOT / "logs" / "pre_sft_logme_proxy_controlled_fusion_v1"
OUTPUT = ROOT / "logs" / "pre_sft_logme_proxy_all_available_common7"
LAYERS = (1, 3, 6, 9, 15, 21, 27)
TARGET_SIGNATURE = "b48e025fefb19e5d7414d2d540b9904a4e21d6de883552699d5e4dc194956d37"

FORMAL_VSI = {
    "c1_vlm3r": 59.3,
    "c1_spatialstack_add": 61.2,
    "c1_spatialstack_add_123": 62.2,
    "c1_spatialstack_add_036": 61.2,
    "c1_spatialstack_cross_attn_v1": 60.6,
}
CONTROLLED_DESCRIPTIONS = {
    "c1_controlled_b": "B: pre-projector additive, CUT3R dec12",
    "c1_controlled_c": "C: cross-attention, CUT3R dec12 → L0",
    "c1_controlled_d": "D: additive, CUT3R dec12 → L0",
    "c1_controlled_e": "E: additive, CUT3R dec12 → L0/L1/L2, site projectors",
    "c1_controlled_h": "H: cross-attention, CUT3R dec12 → L0/L1/L2",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(field for row in rows for field in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def value(row: dict[str, Any], field: str) -> float:
    result = float(row[field])
    if not math.isfinite(result):
        raise RuntimeError(f"Non-finite {field} in {row['architecture']}/L{row['layer']}")
    return result


def order(items: list[tuple[str, float]]) -> list[tuple[str, float]]:
    return sorted(items, key=lambda item: (-item[1], item[0]))


def fmt(number: float, precision: int = 6) -> str:
    return f"{number:.{precision}f}"


def common_score_fields(source: str) -> tuple[str, str]:
    if source == "formal_v2_common7":
        return "logme_primary", "formal"
    if source == "extra_diagnostics":
        return "logme_primary", "diagnostic"
    if source == "controlled_fusion":
        return "common7_mean_logme", "diagnostic"
    raise ValueError(source)


def main() -> None:
    source_rows = {
        "formal_v2_common7": read_csv(FORMAL / "logme_per_layer.csv"),
        "extra_diagnostics": read_csv(EXTRA / "logme_per_layer.csv"),
        "controlled_fusion": read_csv(CONTROLLED / "logme_per_layer.csv"),
    }
    source_scores = {
        "formal_v2_common7": read_csv(FORMAL / "logme_architecture_scores.csv"),
        "extra_diagnostics": read_csv(EXTRA / "logme_architecture_scores.csv"),
        "controlled_fusion": read_csv(CONTROLLED / "logme_architecture_scores.csv"),
    }
    formal_summary = read_json(FORMAL / "logme_summary.json")
    protocols = {
        "formal_v2_common7": read_json(FORMAL / "protocol.json"),
        "extra_diagnostics": read_json(EXTRA / "protocol.json"),
        "controlled_fusion": read_json(CONTROLLED / "protocol.json"),
    }

    rows: list[dict[str, Any]] = []
    for source, source_values in source_rows.items():
        _, scope = common_score_fields(source)
        for source_row in source_values:
            row = dict(source_row)
            row["source_study"] = source
            row["formal_or_diagnostic"] = scope
            row["vsi_score_if_validly_matched"] = FORMAL_VSI.get(row["architecture"])
            rows.append(row)

    labels = {row["architecture"] for row in rows}
    expected_pairs = {(label, layer) for label in labels for layer in LAYERS}
    pairs = {(row["architecture"], int(row["layer"])) for row in rows}
    if len(labels) != 12 or len(rows) != 84 or pairs != expected_pairs:
        raise RuntimeError(f"Expected 12 complete models x seven layers; got {len(labels)} models and {len(rows)} rows")

    checks = {
        "common_seven_layers": {int(row["layer"]) for row in rows} == set(LAYERS),
        "target_signature": {row["target_signature"] for row in rows} == {TARGET_SIGNATURE},
        "valid_tokens": {int(row["valid_tokens"]) for row in rows} == {394352},
        "feature_dimension": {int(row["feature_dimension"]) for row in rows} == {3584},
        "training_videos": {int(row["training_videos"]) for row in rows} == {1006},
        "training_frames": {int(row["training_frames"]) for row in rows} == {2012},
        "finite_converged": all(
            row["status"] == "complete"
            and row["converged"] == "True"
            and all(math.isfinite(value(row, field)) for field in ("logme", "alpha", "beta", "gamma"))
            for row in rows
        ),
        "float64_evidence": all(protocol["dtype"] == "float64" for protocol in protocols.values()),
        "no_optimizer": all(protocol["no_optimizer"] is True for protocol in protocols.values()),
        "no_post_sft": all(
            protocol.get("no_post_sft_checkpoint", protocol.get("no_post_sft_cache")) is True
            for protocol in protocols.values()
        ),
        "no_vlm_forward_during_logme": all(protocol["no_vlm_forward"] is True for protocol in protocols.values()),
    }
    if not all(checks.values()):
        bad = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"Protocol consistency failure: {bad}")

    by_key = {(row["architecture"], int(row["layer"])): row for row in rows}
    names = {
        label: CONTROLLED_DESCRIPTIONS.get(label, by_key[label, LAYERS[0]]["display_name"])
        for label in labels
    }
    means = {
        label: sum(value(by_key[label, layer], "logme") for layer in LAYERS) / len(LAYERS)
        for label in labels
    }
    source_means = {}
    for source, score_rows in source_scores.items():
        field, _ = common_score_fields(source)
        source_means.update({row["architecture"]: float(row[field]) for row in score_rows})
    for label, mean in means.items():
        if not math.isclose(mean, source_means[label], rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(f"Mean mismatch for {label}: {mean} vs saved {source_means[label]}")

    ranked = order(list(means.items()))
    ranks = {label: index for index, (label, _) in enumerate(ranked, start=1)}
    score_rows = []
    for label, mean in ranked:
        source = by_key[label, LAYERS[0]]["source_study"]
        score_rows.append({
            "all_available_logme_rank": ranks[label],
            "architecture": label,
            "display_name": names[label],
            "common7_mean_logme": mean,
            "source_study": source,
            "formal_or_diagnostic": by_key[label, LAYERS[0]]["formal_or_diagnostic"],
            "vsi_score_if_validly_matched": FORMAL_VSI.get(label),
            "vsi_note": "matched formal v2 VSI entry" if label in FORMAL_VSI else "not assigned; no provenance-verified post-SFT VSI pairing",
        })

    layer_rank_rows = []
    layer_order: dict[int, list[tuple[str, float]]] = {}
    rank_history = {label: [] for label in labels}
    for layer in LAYERS:
        layer_order[layer] = order([(label, value(by_key[label, layer], "logme")) for label in labels])
        for rank, (label, score) in enumerate(layer_order[layer], start=1):
            rank_history[label].append(rank)
            layer_rank_rows.append({"layer": layer, "rank": rank, "architecture": label, "display_name": names[label], "logme": score})

    region_layers = {"early": (1, 3, 6), "middle": (9, 15), "late": (21, 27)}
    region_rows = []
    for label, mean in ranked:
        region = {
            name: sum(value(by_key[label, layer], "logme") for layer in selected) / len(selected)
            for name, selected in region_layers.items()
        }
        region_rows.append({"architecture": label, "display_name": names[label], **region, "common7": mean})

    pairwise = []
    for row_label, row_mean in ranked:
        for column_label, column_mean in ranked:
            pairwise.append({
                "row_architecture": row_label,
                "column_architecture": column_label,
                "delta_row_minus_column": row_mean - column_mean,
            })

    fit_summary = []
    for label, _ in ranked:
        model_rows = [by_key[label, layer] for layer in LAYERS]
        fit_summary.append({
            "architecture": label,
            "display_name": names[label],
            "alpha_min": min(value(row, "alpha") for row in model_rows),
            "alpha_max": max(value(row, "alpha") for row in model_rows),
            "beta_min": min(value(row, "beta") for row in model_rows),
            "beta_max": max(value(row, "beta") for row in model_rows),
            "gamma_min": min(value(row, "gamma") for row in model_rows),
            "gamma_max": max(value(row, "gamma") for row in model_rows),
            "iterations_by_layer": ",".join(str(row["iterations"]) for row in model_rows),
            "runtime_seconds_total": sum(value(row, "runtime_seconds") for row in model_rows),
            "peak_process_rss_bytes_max": max(int(row["peak_process_rss_bytes"]) for row in model_rows),
            "all_converged": True,
        })

    row_order = {label: index for index, (label, _) in enumerate(ranked)}
    per_layer_rows = sorted(rows, key=lambda row: (row_order[row["architecture"]], LAYERS.index(int(row["layer"]))))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT / "all_logme_per_layer.csv", per_layer_rows)
    write_csv(OUTPUT / "all_logme_architecture_scores.csv", score_rows)
    write_csv(OUTPUT / "all_layerwise_rankings.csv", layer_rank_rows)
    write_csv(OUTPUT / "all_region_summary.csv", region_rows)
    write_csv(OUTPUT / "all_pairwise_logme_deltas.csv", pairwise)
    write_csv(OUTPUT / "all_fitting_summary.csv", fit_summary)

    primary = formal_summary["primary_correlation"]
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "All completed Common-7 LogME results. Only the original five formal architectures are included in the official VSI correlation.",
        "inputs": {
            "formal": str(FORMAL), "extra_diagnostics": str(EXTRA), "controlled_fusion": str(CONTROLLED),
        },
        "protocol_checks": checks,
        "all_available_architecture_scores": score_rows,
        "formal_five_correlation_unchanged": {
            "primary": primary,
            "per_layer": formal_summary["per_layer_diagnostics"],
            "fixed_regions": formal_summary["fixed_region_diagnostics"],
            "leave_one_layer_out": formal_summary["leave_one_layer_out"],
            "leave_one_layer_out_spearman": formal_summary["leave_one_layer_out_spearman"],
        },
        "rank_stability": {
            label: {"ranks": rank_history[label], "rank_span": max(rank_history[label]) - min(rank_history[label])}
            for label, _ in ranked
        },
        "numerical_warnings": [],
        "controlled_cache_note": "B/C/D/E/H features were regenerated from pre-SFT C1 initializations because their prior full feature tensors had been recycled. Their LogME fits were then cache-only; controlled protocol records no post-SFT checkpoint and no optimizer.",
    }
    write_json(OUTPUT / "all_logme_summary.json", summary)

    per_layer_table = [
        "| Model | L1 | L3 | L6 | L9 | L15 | L21 | L27 | Common-7 mean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, mean in ranked:
        scores = " | ".join(fmt(value(by_key[label, layer], "logme")) for layer in LAYERS)
        per_layer_table.append(f"| {names[label]} | {scores} | {fmt(mean)} |")
    ranking_table = [
        "| Rank | Model | Mean LogME | VSI if validly matched | Scope |",
        "|---:|---|---:|---:|---|",
    ]
    for row in score_rows:
        vsi = "—" if row["vsi_score_if_validly_matched"] is None else f"{row['vsi_score_if_validly_matched']:.1f}"
        ranking_table.append(f"| {row['all_available_logme_rank']} | {row['display_name']} | {fmt(row['common7_mean_logme'])} | {vsi} | {row['formal_or_diagnostic']} |")
    layer_table = [
        "| Layer | " + " | ".join(f"{index}{'st' if index == 1 else 'nd' if index == 2 else 'rd' if index == 3 else 'th'}" for index in range(1, 13)) + " |",
        "|---|" + "|".join("---" for _ in range(12)) + "|",
    ]
    for layer in LAYERS:
        layer_table.append("| L" + str(layer) + " | " + " | ".join(names[label] for label, _ in layer_order[layer]) + " |")
    regions_table = ["| Model | Early | Middle | Late | Common-7 |", "|---|---:|---:|---:|---:|"]
    for row in region_rows:
        regions_table.append(f"| {row['display_name']} | {fmt(row['early'])} | {fmt(row['middle'])} | {fmt(row['late'])} | {fmt(row['common7'])} |")
    fit_table = ["| Model | alpha range | beta range | gamma range | iterations (L1→L27) | fitted runtime |", "|---|---:|---:|---:|---|---:|"]
    for row in fit_summary:
        fit_table.append(
            f"| {row['display_name']} | {fmt(row['alpha_min'], 3)}–{fmt(row['alpha_max'], 3)} | "
            f"{fmt(row['beta_min'], 3)}–{fmt(row['beta_max'], 3)} | {fmt(row['gamma_min'], 3)}–{fmt(row['gamma_max'], 3)} | "
            f"{row['iterations_by_layer']} | {fmt(row['runtime_seconds_total'], 1)} s |"
        )
    md = [
        "# All currently completed pre-SFT Common-7 LogME results", "",
        "This is an integration of 12 completed pre-SFT representations: five formal v2 architectures, selective fusion and plain base-VLM diagnostics, and five latest controlled-fusion diagnostics. It does not recompute or broaden the official five-model VSI correlation.", "",
        "## Full per-layer scores", "", *per_layer_table, "",
        "## All-model descriptive ranking", "", *ranking_table, "",
        "Only the five formal architectures have explicitly matched post-SFT VSI values. All seven diagnostic rows intentionally retain VSI as unavailable; they are not included in any correlation below.", "",
        "## Layer-wise ranking", "", *layer_table, "",
        "## Region summaries", "", *regions_table, "",
        "## Numerical fitting status", "", *fit_table, "",
        "All 84 fits are finite and converged. Every row uses float64 evidence, D=3,584, 394,352 valid depth tokens, 1,006 training videos, 2,012 selected frames, the same target signature, no optimizer and no update. The detailed row-level alpha, beta, gamma, iteration count, runtime, peak RSS, cache source and protocol fields are in `all_logme_per_layer.csv`.", "",
        "## Data/provenance note", "",
        "The five formal and two pre-existing diagnostic results were calculated from retained pre-SFT feature caches. B/C/D/E/H lacked retained full hidden-feature tensors, so their features were regenerated from their recorded pre-SFT C1 initialization at controlled source commit `f4e2259451ecf12d4da9a87dce121639642a0524`; their evidence fits were cache-only. Their protocol records no post-SFT checkpoint and no optimizer. Their regenerated features were intentionally recycled after fitting to preserve disk capacity.", "",
        "## Unchanged official formal-five correlation", "",
        f"- Common-seven mean: Spearman rho **{fmt(float(primary['spearman_rho']))}**; Kendall tau-b **{fmt(float(primary['kendall_tau_b']))}**.",
        f"- Per-layer rho: L1={fmt(float(formal_summary['per_layer_diagnostics'][0]['spearman_rho']))}; L3/L6/L9/L15/L21/L27={fmt(float(primary['spearman_rho']))}.",
        f"- Leave-one-layer-out rho range/median: {fmt(float(formal_summary['leave_one_layer_out_spearman']['minimum']))}–{fmt(float(formal_summary['leave_one_layer_out_spearman']['maximum']))}; median {fmt(float(formal_summary['leave_one_layer_out_spearman']['median']))}.", "",
        "## Source files", "",
        f"- Formal: `{FORMAL / 'logme_per_layer.csv'}`, `{FORMAL / 'logme_architecture_scores.csv'}`, `{FORMAL / 'logme_summary.json'}`.",
        f"- Existing diagnostics: `{EXTRA / 'logme_per_layer.csv'}`, `{EXTRA / 'logme_architecture_scores.csv'}`, `{EXTRA / 'logme_extra_summary.json'}`.",
        f"- Controlled variants: `{CONTROLLED / 'logme_per_layer.csv'}`, `{CONTROLLED / 'logme_architecture_scores.csv'}`, `{CONTROLLED / 'protocol.json'}`.",
    ]
    (OUTPUT / "all_logme_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "models": len(labels), "fits": len(rows), "formal_primary": primary}, sort_keys=True))


if __name__ == "__main__":
    main()
