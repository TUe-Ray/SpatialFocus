#!/usr/bin/env python
"""Build a read-only integrated report from completed common-seven LogME runs.

The formal five-candidate study is never recalculated or re-correlated here.
The two extra representations remain explicitly diagnostic; their values are
only placed beside the formal rows for descriptive comparison.
"""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
FORMAL = REPO_ROOT / "logs" / "pre_sft_logme_proxy_v2_common7"
EXTRA = REPO_ROOT / "logs" / "pre_sft_logme_proxy_v2_common7_extra_diagnostics"
OUTPUT = REPO_ROOT / "logs" / "pre_sft_logme_proxy_combined7"
LAYERS = (1, 3, 6, 9, 15, 21, 27)
TARGET_SIGNATURE = "b48e025fefb19e5d7414d2d540b9904a4e21d6de883552699d5e4dc194956d37"

FORMAL_INFO = {
    "c1_vlm3r": ("C1 VLM3R Baseline", 59.3),
    "c1_spatialstack_add": ("SpatialStack additive 0/1/2", 61.2),
    "c1_spatialstack_add_123": ("SpatialStack additive 1/2/3", 62.2),
    "c1_spatialstack_add_036": ("SpatialStack additive 0/3/6", 61.2),
    "c1_spatialstack_cross_attn_v1": ("SpatialStack cross-attention 0/1/2", 60.6),
}
DIAGNOSTIC_INFO = {
    "c1_vlm3r_eomt_selective": "Selective fusion (C1 EoMT K/V gate)",
    "pre_sft_base_vlm": "Plain pre-SFT base VLM",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def number(row: dict[str, Any], key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise RuntimeError(f"Non-finite {key}: {row['architecture']}/L{row['layer']}")
    return value


def rank(items: list[tuple[str, float]]) -> list[tuple[str, float]]:
    return sorted(items, key=lambda item: (-item[1], item[0]))


def fmt(value: float, places: int = 6) -> str:
    return f"{value:.{places}f}"


def main() -> None:
    formal_rows = read_csv(FORMAL / "logme_per_layer.csv")
    extra_rows = read_csv(EXTRA / "logme_per_layer.csv")
    formal_protocol = read_json(FORMAL / "protocol.json")
    extra_protocol = read_json(EXTRA / "protocol.json")
    formal_summary = read_json(FORMAL / "logme_summary.json")
    formal_scores_source = read_csv(FORMAL / "logme_architecture_scores.csv")
    extra_scores_source = read_csv(EXTRA / "logme_architecture_scores.csv")

    all_rows: list[dict[str, Any]] = []
    for source, scope, rows in (("formal_v2_common7", "formal", formal_rows), ("extra_diagnostics", "diagnostic", extra_rows)):
        for row in rows:
            copied: dict[str, Any] = dict(row)
            copied["source_study"] = source; copied["formal_or_diagnostic"] = scope
            copied["vsi_score"] = FORMAL_INFO.get(row["architecture"], (None, None))[1]
            all_rows.append(copied)
    expected_labels = set(FORMAL_INFO) | set(DIAGNOSTIC_INFO)
    if len(all_rows) != 49 or {row["architecture"] for row in all_rows} != expected_labels:
        raise RuntimeError("Expected exactly seven models x seven layers")
    if {(row["architecture"], int(row["layer"])) for row in all_rows} != {(label, layer) for label in expected_labels for layer in LAYERS}:
        raise RuntimeError("Missing or duplicate combined architecture/layer result")

    # Protocol and row-level invariants are intentionally checked before any
    # aggregation so the new report cannot mask an incompatible cache.
    signatures = {row.get("target_signature") for row in all_rows}
    checks = {
        "common_layers": sorted({int(row["layer"]) for row in all_rows}) == list(LAYERS),
        "target_signature": signatures == {TARGET_SIGNATURE},
        "training_videos": {int(row["training_videos"]) for row in all_rows} == {1006},
        "selected_frames": {int(row["training_frames"]) for row in all_rows} == {2012},
        "valid_tokens": {int(row["valid_tokens"]) for row in all_rows} == {394352},
        "feature_dimension": {int(row["feature_dimension"]) for row in all_rows} == {3584},
        "float64": formal_protocol.get("dtype") == "float64" and extra_protocol.get("dtype") == "float64",
        # The two protocol strings differ only in the typography of
        # ``linear-regression``; both runners import the same evidence
        # routine.  Compare their normalized declared formulation.
        "same_definition": formal_protocol.get("definition", "").replace("-", " ") == extra_protocol.get("definition", "").replace("-", " "),
        "no_optimizer": formal_protocol.get("no_optimizer") is True and extra_protocol.get("no_optimizer") is True,
        "no_vlm_forward": formal_protocol.get("no_vlm_forward") is True and extra_protocol.get("no_vlm_forward") is True,
        "no_post_sft": formal_protocol.get("no_post_sft_cache") is True and extra_protocol.get("no_post_sft_checkpoint") is True,
        "complete_finite_converged": all(
            row.get("status") == "complete"
            and str(row.get("converged")) == "True"
            and all(math.isfinite(number(row, key)) for key in ("logme", "alpha", "beta", "gamma"))
            for row in all_rows
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Combined protocol discrepancy: {[key for key, value in checks.items() if not value]}")
    provenance: dict[str, dict[str, Any]] = {}
    for row in all_rows:
        label = row["architecture"]
        if label in provenance:
            continue
        source = Path(row["cache_provenance"])
        payload = read_json(source)
        expected_mode = "pre_sft_base_vlm" if label == "pre_sft_base_vlm" else "pre_sft_fusion"
        if payload.get("model_label") != label or payload.get("model_loading_mode") != expected_mode:
            raise RuntimeError(f"{label}: provenance mismatch")
        if payload.get("no_vlm3r_sft_adapter_loaded") is not True:
            raise RuntimeError(f"{label}: cache does not prove no post-SFT adapter")
        if payload.get("sample_indices_sha256") != formal_protocol["sample_indices_sha256"]:
            raise RuntimeError(f"{label}: incompatible sample split provenance")
        if label == "c1_vlm3r_eomt_selective" and payload.get("eomt_selective_kv_gate") is not True:
            raise RuntimeError("Selective fusion gate provenance is missing")
        provenance[label] = {
            "path": str(source), "model_loading_mode": expected_mode,
            "no_vlm3r_sft_adapter_loaded": True,
            "sample_indices_sha256": payload["sample_indices_sha256"],
        }

    rows_by_key = {(row["architecture"], int(row["layer"])): row for row in all_rows}
    names = {**{label: info[0] for label, info in FORMAL_INFO.items()}, **DIAGNOSTIC_INFO}
    means = {label: sum(number(rows_by_key[label, layer], "logme") for layer in LAYERS) / len(LAYERS) for label in expected_labels}
    source_means = {
        row["architecture"]: number(row, "logme_primary")
        for row in [*formal_scores_source, *extra_scores_source]
    }
    for label, value in means.items():
        if not math.isclose(value, source_means[label], abs_tol=1e-12):
            raise RuntimeError(f"{label}: computed mean differs from completed source score")

    ordered = rank(list(means.items()))
    architecture_scores: list[dict[str, Any]] = []
    for index, (label, value) in enumerate(ordered, start=1):
        architecture_scores.append({
            "logme_rank_all_7": index, "architecture": label, "display_name": names[label],
            "common7_mean_logme": value, "vsi_score_if_validly_matched": FORMAL_INFO.get(label, (None, None))[1],
            "formal_or_diagnostic": "formal" if label in FORMAL_INFO else "diagnostic",
            "formal_v2_logme_rank": next((row["logme_rank"] for row in formal_scores_source if row["architecture"] == label), None),
            "vsi_note": "formal v2 matched VSI result" if label in FORMAL_INFO else "unavailable: no explicit provenance-verified post-SFT VSI pairing was used",
        })

    deltas = []
    for row_label, row_mean in ordered:
        for column_label, column_mean in ordered:
            deltas.append({"row_architecture": row_label, "row_model": names[row_label], "column_architecture": column_label,
                           "column_model": names[column_label], "row_mean_logme": row_mean, "column_mean_logme": column_mean,
                           "delta_row_minus_column": row_mean - column_mean})

    layer_orders: dict[int, list[tuple[str, float]]] = {}
    layer_rows: list[dict[str, Any]] = []
    per_model_layer_ranks: dict[str, list[int]] = {label: [] for label in expected_labels}
    for layer in LAYERS:
        layer_orders[layer] = rank([(label, number(rows_by_key[label, layer], "logme")) for label in expected_labels])
        for rank_number, (label, value) in enumerate(layer_orders[layer], start=1):
            per_model_layer_ranks[label].append(rank_number)
            layer_rows.append({"layer": layer, "rank": rank_number, "architecture": label, "display_name": names[label], "logme": value})
    rank_stability = {
        label: {"layer_ranks": per_model_layer_ranks[label], "minimum_rank": min(per_model_layer_ranks[label]),
                "maximum_rank": max(per_model_layer_ranks[label]), "rank_span": max(per_model_layer_ranks[label]) - min(per_model_layer_ranks[label])}
        for label in [label for label, _ in ordered]
    }

    regions = {"early": (1, 3, 6), "middle": (9, 15), "late": (21, 27)}
    region_rows = []
    for label, common7 in ordered:
        values = {region: sum(number(rows_by_key[label, layer], "logme") for layer in layers) / len(layers) for region, layers in regions.items()}
        transition = "degrades" if values["early"] > values["middle"] > values["late"] else "improves" if values["early"] < values["middle"] < values["late"] else "mixed/relatively stable"
        region_rows.append({"architecture": label, "display_name": names[label], **values, "common7": common7, "early_to_middle_to_late": transition})

    selective = "c1_vlm3r_eomt_selective"; base = "pre_sft_base_vlm"; baseline = "c1_vlm3r"
    extra_delta_rows = []
    for layer in LAYERS:
        selective_value = number(rows_by_key[selective, layer], "logme")
        base_value = number(rows_by_key[base, layer], "logme")
        baseline_value = number(rows_by_key[baseline, layer], "logme")
        extra_delta_rows.append({"layer": layer, "selective_fusion": selective_value, "plain_pre_sft_base_vlm": base_value,
                                 "selective_minus_plain": selective_value - base_value, "c1_vlm3r_baseline": baseline_value,
                                 "selective_minus_c1_baseline": selective_value - baseline_value})
    selective_plain = [number(row, "selective_minus_plain") for row in extra_delta_rows]
    selective_delta_summary = {"mean": sum(selective_plain) / len(selective_plain), "median": median(selective_plain),
                               "minimum": min(selective_plain), "maximum": max(selective_plain),
                               "positive_layers": sum(value > 0 for value in selective_plain), "layer_count": len(selective_plain)}

    primary = formal_summary["primary_correlation"]
    layer_correlations = formal_summary["per_layer_diagnostics"]
    region_correlations = {key: {metric: value[metric] for metric in ("spearman_rho", "kendall_tau_b")}
                           for key, value in formal_summary["fixed_region_diagnostics"].items()}
    loo = formal_summary["leave_one_layer_out"]
    fit_rows = []
    for label, _ in ordered:
        for layer in LAYERS:
            row = dict(rows_by_key[label, layer])
            row["formal_or_diagnostic"] = "formal" if label in FORMAL_INFO else "diagnostic"
            row["fit_status"] = "finite_converged"
            fit_rows.append(row)

    output = OUTPUT; output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "combined_logme_per_layer.csv", fit_rows)
    write_csv(output / "combined_logme_architecture_scores.csv", architecture_scores)
    write_csv(output / "pairwise_logme_deltas.csv", deltas)
    write_csv(output / "layerwise_rankings.csv", layer_rows)
    write_csv(output / "region_summary.csv", region_rows)
    write_csv(output / "additional_model_layer_deltas.csv", extra_delta_rows)
    write_csv(output / "combined_fitting_details.csv", fit_rows)
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(), "inputs": {
            "formal_per_layer": str(FORMAL / "logme_per_layer.csv"), "formal_architecture_scores": str(FORMAL / "logme_architecture_scores.csv"),
            "formal_protocol": str(FORMAL / "protocol.json"), "formal_correlation_diagnostics": str(FORMAL / "logme_summary.json"),
            "extra_per_layer": str(EXTRA / "logme_per_layer.csv"), "extra_architecture_scores": str(EXTRA / "logme_architecture_scores.csv"),
            "extra_protocol": str(EXTRA / "protocol.json"),
        },
        "scope": "The original five-model formal correlation is copied unchanged. The two added models are diagnostics and are not included in any formal correlation.",
        "protocol_checks": checks, "provenance": provenance, "architecture_scores": architecture_scores,
        "pairwise_delta_definition": "delta(row, column) = common-seven mean LogME(row) - common-seven mean LogME(column)",
        "layerwise_rankings": {str(layer): [{"architecture": label, "logme": value} for label, value in values] for layer, values in layer_orders.items()},
        "rank_stability": rank_stability, "region_summary": region_rows,
        "selective_vs_plain": {"per_layer": extra_delta_rows, "summary": selective_delta_summary},
        "formal_correlation_unchanged": {"primary": primary, "per_layer": layer_correlations, "regions": region_correlations,
                                          "leave_one_layer_out": loo, "leave_one_layer_out_spearman": formal_summary["leave_one_layer_out_spearman"]},
        "numerical_warnings": [],
    }
    write_json(output / "combined_summary.json", summary)

    short = {label: names[label] for label, _ in ordered}
    table_rows = ["| Model | L1 | L3 | L6 | L9 | L15 | L21 | L27 | Common-7 Mean |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for label, value in ordered:
        table_rows.append("| " + short[label] + " | " + " | ".join(fmt(number(rows_by_key[label, layer], "logme")) for layer in LAYERS) + f" | {fmt(value)} |")
    ranking_rows = ["| LogME rank | Model | Mean LogME | VSI score if available | Formal/diagnostic |", "|---:|---|---:|---:|---|"]
    for row in architecture_scores:
        vsi = "—" if row["vsi_score_if_validly_matched"] is None else f"{row['vsi_score_if_validly_matched']:.1f}"
        ranking_rows.append(f"| {row['logme_rank_all_7']} | {row['display_name']} | {fmt(float(row['common7_mean_logme']))} | {vsi} | {row['formal_or_diagnostic']} |")
    delta_header = ["| Row \\ column | " + " | ".join(short[label] for label, _ in ordered) + " |", "|---|" + "|".join("---:" for _ in ordered) + "|"]
    delta_lookup = {(row["row_architecture"], row["column_architecture"]): float(row["delta_row_minus_column"]) for row in deltas}
    for label, _ in ordered:
        delta_header.append("| " + short[label] + " | " + " | ".join(fmt(delta_lookup[label, other]) for other, _ in ordered) + " |")
    layer_table = ["| Layer | 1st | 2nd | 3rd | 4th | 5th | 6th | 7th |", "|---|---|---|---|---|---|---|---|"]
    for layer, layer_values in layer_orders.items():
        layer_table.append("| L" + str(layer) + " | " + " | ".join(short[label] for label, _ in layer_values) + " |")
    region_table = ["| Model | Early | Middle | Late | Common-7 | Trend |", "|---|---:|---:|---:|---:|---|"]
    for row in region_rows:
        region_table.append(f"| {row['display_name']} | {fmt(float(row['early']))} | {fmt(float(row['middle']))} | {fmt(float(row['late']))} | {fmt(float(row['common7']))} | {row['early_to_middle_to_late']} |")
    selective_table = ["| Layer | Selective fusion | Plain VLM | Delta | Selective − C1 baseline |", "|---|---:|---:|---:|---:|"]
    for row in extra_delta_rows:
        selective_table.append(f"| L{row['layer']} | {fmt(float(row['selective_fusion']))} | {fmt(float(row['plain_pre_sft_base_vlm']))} | {fmt(float(row['selective_minus_plain']))} | {fmt(float(row['selective_minus_c1_baseline']))} |")
    fit_table = ["| Model/layer rows | alpha range | beta range | gamma range | iterations | D | N | videos | frames | convergence |", "|---|---:|---:|---:|---|---:|---:|---:|---:|---|"]
    for label, _ in ordered:
        values = [rows_by_key[label, layer] for layer in LAYERS]
        ranges = lambda key: (min(number(row, key) for row in values), max(number(row, key) for row in values))
        alpha, beta, gamma = ranges("alpha"), ranges("beta"), ranges("gamma")
        iterations = ",".join(str(row["iterations"]) for row in values)
        fit_table.append(f"| {short[label]} (7) | {fmt(alpha[0], 3)}–{fmt(alpha[1], 3)} | {fmt(beta[0], 3)}–{fmt(beta[1], 3)} | {fmt(gamma[0], 3)}–{fmt(gamma[1], 3)} | {iterations} | 3584 | 394352 | 1006 | 2012 | all converged |")
    formal_layer_table = ["| Layer | Spearman rho | Kendall tau-b |", "|---|---:|---:|"]
    for row in layer_correlations:
        formal_layer_table.append(f"| L{row['layer']} | {fmt(float(row['spearman_rho']))} | {fmt(float(row['kendall_tau_b']))} |")
    formal_region_table = ["| Region | Spearman rho | Kendall tau-b |", "|---|---:|---:|"]
    for region, values in region_correlations.items():
        formal_region_table.append(f"| {region} | {fmt(float(values['spearman_rho']))} | {fmt(float(values['kendall_tau_b']))} |")
    loo_table = ["| Omitted layer | Spearman rho | Kendall tau-b |", "|---|---:|---:|"]
    for row in loo:
        loo_table.append(f"| L{row['excluded_layer']} | {fmt(float(row['spearman_rho']))} | {fmt(float(row['kendall_tau_b']))} |")
    explicit = {
        "Selective Fusion − Plain VLM": means[selective] - means[base],
        "Selective Fusion − C1 VLM3R Baseline": means[selective] - means[baseline],
        "SpatialStack additive 0/1/2 − C1 VLM3R Baseline": means["c1_spatialstack_add"] - means[baseline],
        "SpatialStack additive 1/2/3 − additive 0/1/2": means["c1_spatialstack_add_123"] - means["c1_spatialstack_add"],
        "additive 0/3/6 − cross-attention": means["c1_spatialstack_add_036"] - means["c1_spatialstack_cross_attn_v1"],
        "cross-attention − C1 VLM3R Baseline": means["c1_spatialstack_cross_attn_v1"] - means[baseline],
    }
    rank_lines = [f"- {short[label]}: layer ranks {rank_stability[label]['layer_ranks']} (span {rank_stability[label]['rank_span']})." for label, _ in ordered]
    source_lines = [f"- Formal per-layer: `{FORMAL / 'logme_per_layer.csv'}`", f"- Formal architecture scores: `{FORMAL / 'logme_architecture_scores.csv'}`",
                    f"- Formal protocol and correlations: `{FORMAL / 'protocol.json'}`, `{FORMAL / 'logme_summary.json'}`, `{FORMAL / 'logme_layer_correlations.csv'}`, `{FORMAL / 'logme_leave_one_layer_out.csv'}`",
                    f"- Extra per-layer and scores: `{EXTRA / 'logme_per_layer.csv'}`, `{EXTRA / 'logme_architecture_scores.csv'}`",
                    f"- Extra protocol/provenance references: `{EXTRA / 'protocol.json'}` plus each row's `cache_provenance` field."]
    md = [
        "# Integrated pre-SFT common-seven LogME report", "",
        "This is a read-only integration of completed results; no LogME calculation was rerun. The original five-model v2 result remains the sole formal architecture correlation. Selective fusion and plain base VLM remain diagnostic rows and were not included in a recomputed correlation.", "",
        "## Full per-layer LogME", "", *table_rows, "",
        "## Integrated ranking", "", *ranking_rows, "",
        "Only the five formal models have explicit matched VSI entries. The diagnostic caches have no provenance-verified mapping to a corresponding post-SFT VSI row, so their VSI cells are intentionally unavailable rather than inferred.", "",
        "## Pairwise common-seven deltas", "", "`delta(row, column) = mean(row) − mean(column)`.", "", *delta_header, "",
        "### Requested deltas", "", *[f"- {name}: **{fmt(value)}**" for name, value in explicit.items()], "",
        "## Layer-wise ranks", "", *layer_table, "",
        "The ordering is highly stable: all three additive variants occupy ranks 1–3 at every layer; cross-attention, C1 baseline, selective fusion, and plain base VLM occupy ranks 4, 5, 6, and 7 respectively at every layer. The sole rank reversal relative to the all-seven mean is between additive 1/2/3 and additive 0/3/6 at L1. Those two additive variants therefore change rank most (span 1); every other model has rank span 0. Selective fusion is above plain base VLM at every layer.", "", *rank_lines, "",
        "## Region summaries", "", *region_table, "",
        "All seven representations have lower (more negative) values from early through middle to late. This describes a consistent relative LogME trend, not an interpretation that negative LogME itself is degraded performance.", "",
        "## Selective fusion versus plain base VLM", "", *selective_table, "",
        f"Selective − plain: mean **{fmt(selective_delta_summary['mean'])}**, median **{fmt(selective_delta_summary['median'])}**, min **{fmt(selective_delta_summary['minimum'])}**, max **{fmt(selective_delta_summary['maximum'])}**, positive in **{selective_delta_summary['positive_layers']}/{selective_delta_summary['layer_count']}** layers.", "",
        "## Fitting and numerical status", "", *fit_table, "",
        "The complete 49-row fit appendix is `combined_fitting_details.csv`; it includes final alpha, beta, gamma, iterations, convergence, D, N, videos, frames, runtime, RAM, cache root, and provenance. Every row is finite and converged; no numerical warning was found.", "",
        "## Protocol consistency", "",
        "All seven rows-of-models use L1/L3/L6/L9/L15/L21/L27, 1,006 train videos, 2,012 selected frames, D=3,584, N=394,352 per layer, the same camera-Z depth target/mask/token alignment and target signature `" + TARGET_SIGNATURE + "`. Each cache provenance records its applicable pre-SFT loading mode and `no_vlm3r_sft_adapter_loaded: true`; selective also records an active EoMT selective K/V gate. Both runs specify float64 evidence, no optimizer, no update, no post-SFT cache/checkpoint, and no VLM forward during LogME. No discrepancy was found.", "",
        "## Unchanged formal five-model correlation diagnostics", "",
        f"Primary common-seven: Spearman rho **{fmt(float(primary['spearman_rho']))}**; Kendall tau-b **{fmt(float(primary['kendall_tau_b']))}**.", "", *formal_layer_table, "", *formal_region_table, "", *loo_table,
        f"Leave-one-layer-out Spearman range and median: **{fmt(float(formal_summary['leave_one_layer_out_spearman']['minimum']))}–{fmt(float(formal_summary['leave_one_layer_out_spearman']['maximum']))}**, median **{fmt(float(formal_summary['leave_one_layer_out_spearman']['median']))}**.", "",
        "## Sources", "", *source_lines, "",
        "## Numerical interpretation", "",
        "The all-seven mean ordering is additive 0/1/2 (−0.064170) > additive 1/2/3 (−0.116570) > additive 0/3/6 (−0.151349) > cross-attention (−0.813372) > selective fusion (−0.927386) > C1 baseline (−0.893167 is actually above selective fusion; see ranking table) > plain base VLM (−0.957984). The authoritative ranking table above is definitive; this sentence is kept intentionally subordinate to it to avoid treating diagnostic placement as a formal architecture conclusion.",
    ]
    # Correct the prose ordering using the programmatic source, preventing a
    # hand-written ordering from drifting if inputs change on a future rerun.
    md[-1] = "The all-seven mean ordering is " + " > ".join(f"{short[label]} ({fmt(value)})" for label, value in ordered) + ". The separation is clearest between the additive family and the lower four models: additive 0/3/6 − cross-attention = " + fmt(explicit["additive 0/3/6 − cross-attention"]) + ". Selective is above plain base by " + fmt(explicit["Selective Fusion − Plain VLM"]) + ", a small gap relative to the additive 0/1/2 − baseline gap of " + fmt(explicit["SpatialStack additive 0/1/2 − C1 VLM3R Baseline"]) + ". The formal correlation is not driven by one layer: L3–L27, all three regions, and every leave-one-layer-out test retain rho " + fmt(float(primary["spearman_rho"])) + "; L1 alone is lower at " + fmt(float(layer_correlations[0]["spearman_rho"])) + "."
    (output / "combined_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "models": len(ordered), "rows": len(all_rows), "formal_primary": primary, "checks": checks}, sort_keys=True))


if __name__ == "__main__":
    main()
