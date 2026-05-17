#!/usr/bin/env python
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


ROOM_SIZE_TYPE = "room_size_estimation"
MRA_KEY = "MRA:.5:.95:.05"
DEPTH_VARIABLES = (
    "p95_depth",
    "max_depth",
    "fraction_depth_gt_8m",
    "fraction_depth_gt_10m",
)
SUMMARY_COLUMNS = (
    "grouping",
    "group",
    "bin_rule",
    "num_samples",
    "mean_mra",
    "median_mra",
    "std_mra",
    "p25_mra",
    "p75_mra",
    "fraction_mra_eq_0",
    "fraction_mra_ge_0_5",
    "fraction_mra_ge_0_75",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze continuous Room Size MRA versus CUT3R scene-depth statistics."
    )
    parser.add_argument(
        "--probe0-csv",
        default="outputs/probe0_room_size_depth_analysis/room_size_depth_analysis.csv",
        help="Existing Probe 0 per-sample Room Size depth CSV.",
    )
    parser.add_argument(
        "--vsibench-json",
        default=(
            "/leonardo_scratch/fast/EUHPC_D32_006/eval/logs/VLM3R/"
            "0507_2335_eval_rope_spherical_100p_40790070_vlm_3r_model_args_dd1948/"
            "vsibench.json"
        ),
        help="Original sample log, used only if the Probe 0 CSV lacks MRA scores.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/probe0_room_size_depth_analysis",
        help="Directory for the continuous-MRA summary outputs.",
    )
    return parser.parse_args()


def to_float(value, default=float("nan")):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def fmt(value, digits=4):
    if value is None:
        return "NA"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(value):
        return "NA"
    return f"{value:.{digits}f}"


def read_csv_rows(path):
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_mra_from_vsibench(path):
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    by_doc_id = {}
    by_sample_id = {}
    for sample in payload.get("logs", []):
        doc = sample.get("vsibench_score") or sample.get("doc") or {}
        if doc.get("question_type") != ROOM_SIZE_TYPE:
            continue
        mra_score = to_float(doc.get(MRA_KEY))
        by_doc_id[str(sample.get("doc_id", ""))] = mra_score
        by_sample_id[str(doc.get("id", ""))] = mra_score
    return by_doc_id, by_sample_id


def attach_mra(rows, vsibench_json):
    needs_log = not rows or ("mra_score" not in rows[0] and "score" not in rows[0])
    by_doc_id, by_sample_id = ({}, {})
    if needs_log:
        by_doc_id, by_sample_id = read_mra_from_vsibench(vsibench_json)

    output = []
    for row in rows:
        mra_score = to_float(row.get("mra_score", row.get("score")))
        if not math.isfinite(mra_score):
            mra_score = by_doc_id.get(str(row.get("doc_id", "")), float("nan"))
        if not math.isfinite(mra_score):
            mra_score = by_sample_id.get(str(row.get("sample_id", "")), float("nan"))
        new_row = dict(row)
        new_row["mra_score"] = mra_score
        output.append(new_row)
    return output


def percentile(sorted_values, q):
    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * q
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return float(sorted_values[low])
    weight = position - low
    return float(sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight)


def mean(values):
    values = [float(value) for value in values if math.isfinite(float(value))]
    return sum(values) / len(values) if values else float("nan")


def sample_std(values):
    values = [float(value) for value in values if math.isfinite(float(value))]
    if len(values) < 2:
        return float("nan")
    avg = sum(values) / len(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / (len(values) - 1))


def p95_depth_group(value):
    if value < 5.0:
        return "near"
    if value < 10.0:
        return "mid"
    return "far"


def max_depth_group(value):
    if value < 5.0:
        return "near"
    if value < 10.0:
        return "mid"
    return "far"


def fraction_group(value):
    if value <= 0.0:
        return "0"
    if value < 0.001:
        return "(0,0.001)"
    if value < 0.01:
        return "[0.001,0.01)"
    if value < 0.05:
        return "[0.01,0.05)"
    return ">=0.05"


def summarize_values(grouping, group, bin_rule, values):
    values = [float(value) for value in values if math.isfinite(float(value))]
    sorted_values = sorted(values)
    num_samples = len(sorted_values)
    return {
        "grouping": grouping,
        "group": group,
        "bin_rule": bin_rule,
        "num_samples": num_samples,
        "mean_mra": mean(sorted_values),
        "median_mra": percentile(sorted_values, 0.50),
        "std_mra": sample_std(sorted_values),
        "p25_mra": percentile(sorted_values, 0.25),
        "p75_mra": percentile(sorted_values, 0.75),
        "fraction_mra_eq_0": mean([1.0 if value == 0.0 else 0.0 for value in sorted_values]),
        "fraction_mra_ge_0_5": mean([1.0 if value >= 0.5 else 0.0 for value in sorted_values]),
        "fraction_mra_ge_0_75": mean([1.0 if value >= 0.75 else 0.0 for value in sorted_values]),
    }


def summarize_grouping(rows, grouping, value_key, group_fn, group_order, bin_rule):
    grouped = defaultdict(list)
    for row in rows:
        value = to_float(row.get(value_key))
        mra_score = to_float(row.get("mra_score"))
        if math.isfinite(value) and math.isfinite(mra_score):
            grouped[group_fn(value)].append(mra_score)

    return [
        summarize_values(grouping, group, bin_rule, grouped.get(group, []))
        for group in group_order
    ]


def average_ranks(values):
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    idx = 0
    while idx < len(indexed):
        end = idx + 1
        while end < len(indexed) and indexed[end][1] == indexed[idx][1]:
            end += 1
        rank = (idx + 1 + end) / 2.0
        for original_index, _ in indexed[idx:end]:
            ranks[original_index] = rank
        idx = end
    return ranks


def pearson(xs, ys):
    if len(xs) < 2:
        return float("nan")
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x == 0.0 or var_y == 0.0:
        return float("nan")
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    return cov / math.sqrt(var_x * var_y)


def spearman(xs, ys):
    pairs = [
        (float(x), float(y))
        for x, y in zip(xs, ys)
        if math.isfinite(float(x)) and math.isfinite(float(y))
    ]
    if len(pairs) < 2:
        return float("nan"), None, len(pairs), "rank_pearson"

    xs_clean, ys_clean = zip(*pairs)
    try:
        from scipy.stats import spearmanr

        result = spearmanr(xs_clean, ys_clean)
        return float(result.statistic), float(result.pvalue), len(pairs), "scipy"
    except Exception:
        rho = pearson(average_ranks(list(xs_clean)), average_ranks(list(ys_clean)))
        return rho, None, len(pairs), "rank_pearson"


def write_csv(path, rows, fieldnames):
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def group_lookup(summary_rows, grouping):
    return [row for row in summary_rows if row["grouping"] == grouping]


def row_value(rows, group, key):
    for row in rows:
        if row["group"] == group:
            return row.get(key, float("nan"))
    return float("nan")


def trend_clearly_decreases(rows):
    available = [row for row in rows if int(row["num_samples"]) > 0]
    if len(available) < 3:
        return False

    mean_values = [float(row["mean_mra"]) for row in available]
    median_values = [float(row["median_mra"]) for row in available]

    def monotonic_drop(values, min_drop):
        if any(not math.isfinite(value) for value in values):
            return False
        return all(left >= right for left, right in zip(values, values[1:])) and (values[0] - values[-1]) >= min_drop

    mean_clearly_drops = monotonic_drop(mean_values, 0.03)
    median_drops_with_mean_support = monotonic_drop(median_values, 0.05) and (mean_values[0] - mean_values[-1]) >= 0.03
    return mean_clearly_drops or median_drops_with_mean_support


def write_report(path, summary_rows, correlation_rows):
    p95_rows = group_lookup(summary_rows, "p95_depth")
    max_rows = group_lookup(summary_rows, "max_depth")
    frac8_rows = group_lookup(summary_rows, "fraction_depth_gt_8m")
    frac10_rows = group_lookup(summary_rows, "fraction_depth_gt_10m")

    corr_by_var = {row["variable"]: row for row in correlation_rows}
    p95_rho = to_float(corr_by_var.get("p95_depth", {}).get("rho"))
    frac10_rho = to_float(corr_by_var.get("fraction_depth_gt_10m", {}).get("rho"))
    depth_plausible = (
        trend_clearly_decreases(p95_rows)
        or trend_clearly_decreases(max_rows)
        or (math.isfinite(p95_rho) and p95_rho <= -0.20)
        or (math.isfinite(frac10_rho) and frac10_rho <= -0.20)
    )

    lines = [
        "# Room Size Continuous MRA vs Depth",
        "",
        "Primary signal: continuous `MRA:.5:.95:.05`, not binary `MRA > 0` correctness.",
        "",
        "## p95_depth Groups",
        "",
        "| Group | N | Mean MRA | Median MRA | P25 | P75 | MRA=0 | MRA>=0.5 | MRA>=0.75 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in p95_rows:
        lines.append(summary_markdown_row(row))

    lines.extend(["", "## max_depth Groups", "", "| Group | N | Mean MRA | Median MRA | P25 | P75 | MRA=0 | MRA>=0.5 | MRA>=0.75 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for row in max_rows:
        lines.append(summary_markdown_row(row))

    lines.extend(["", "## fraction_depth_gt_8m Bins", "", "| Group | N | Mean MRA | Median MRA | P25 | P75 | MRA=0 | MRA>=0.5 | MRA>=0.75 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for row in frac8_rows:
        lines.append(summary_markdown_row(row))

    lines.extend(["", "## fraction_depth_gt_10m Bins", "", "| Group | N | Mean MRA | Median MRA | P25 | P75 | MRA=0 | MRA>=0.5 | MRA>=0.75 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for row in frac10_rows:
        lines.append(summary_markdown_row(row))

    lines.extend(["", "## Spearman Correlations", "", "| Variable | rho | p_value | N | Method |", "|---|---:|---:|---:|---|"])
    for row in correlation_rows:
        lines.append(
            f"| {row['variable']} | {fmt(row['rho'])} | {fmt(row['p_value'])} | "
            f"{row['num_samples_used']} | {row['method']} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "Depth-range/clamp hypothesis remains plausible; continuous MRA decreases with depth "
                "or shows a meaningfully negative Spearman correlation."
                if depth_plausible
                else (
                    "Room Size continuous MRA does not clearly decrease with depth, and the key "
                    "Spearman correlations are not meaningfully negative. Room Size degradation is "
                    "probably not depth-range/clamp dominated; do not run the full expensive NTK matrix "
                    "based on this evidence alone."
                )
            ),
            "",
            (
                f"p95 near/mid/far mean MRA: {fmt(row_value(p95_rows, 'near', 'mean_mra'))}, "
                f"{fmt(row_value(p95_rows, 'mid', 'mean_mra'))}, "
                f"{fmt(row_value(p95_rows, 'far', 'mean_mra'))}."
            ),
            (
                f"max-depth near/mid/far mean MRA: {fmt(row_value(max_rows, 'near', 'mean_mra'))}, "
                f"{fmt(row_value(max_rows, 'mid', 'mean_mra'))}, "
                f"{fmt(row_value(max_rows, 'far', 'mean_mra'))}."
            ),
        ]
    )

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
        handle.write("\n")


def summary_markdown_row(row):
    return (
        f"| {row['group']} | {row['num_samples']} | {fmt(row['mean_mra'])} | "
        f"{fmt(row['median_mra'])} | {fmt(row['p25_mra'])} | {fmt(row['p75_mra'])} | "
        f"{fmt(row['fraction_mra_eq_0'])} | {fmt(row['fraction_mra_ge_0_5'])} | "
        f"{fmt(row['fraction_mra_ge_0_75'])} |"
    )


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = attach_mra(read_csv_rows(args.probe0_csv), args.vsibench_json)
    valid_rows = [row for row in rows if math.isfinite(to_float(row.get("mra_score")))]
    if not valid_rows:
        raise SystemExit("No finite MRA scores found.")

    summary_rows = []
    summary_rows.extend(
        summarize_grouping(
            valid_rows,
            "p95_depth",
            "p95_depth",
            p95_depth_group,
            ("near", "mid", "far"),
            "near:<5; mid:[5,10); far:>=10",
        )
    )
    summary_rows.extend(
        summarize_grouping(
            valid_rows,
            "max_depth",
            "max_depth",
            max_depth_group,
            ("near", "mid", "far"),
            "near:<5; mid:[5,10); far:>=10",
        )
    )
    fraction_order = ("0", "(0,0.001)", "[0.001,0.01)", "[0.01,0.05)", ">=0.05")
    for key in ("fraction_depth_gt_8m", "fraction_depth_gt_10m"):
        summary_rows.extend(
            summarize_grouping(
                valid_rows,
                key,
                key,
                fraction_group,
                fraction_order,
                "0; (0,0.001); [0.001,0.01); [0.01,0.05); >=0.05",
            )
        )

    correlation_rows = []
    mra_scores = [to_float(row.get("mra_score")) for row in valid_rows]
    for variable in DEPTH_VARIABLES:
        values = [to_float(row.get(variable)) for row in valid_rows]
        rho, p_value, num_samples, method = spearman(values, mra_scores)
        correlation_rows.append(
            {
                "variable": variable,
                "rho": rho,
                "p_value": "NA" if p_value is None else p_value,
                "num_samples_used": num_samples,
                "method": method,
            }
        )

    write_csv(output_dir / "room_size_continuous_mra_depth_summary.csv", summary_rows, SUMMARY_COLUMNS)
    write_csv(
        output_dir / "room_size_continuous_mra_depth_correlations.csv",
        correlation_rows,
        ("variable", "rho", "p_value", "num_samples_used", "method"),
    )
    write_report(output_dir / "room_size_continuous_mra_depth_analysis.md", summary_rows, correlation_rows)
    print(f"Wrote continuous Room Size MRA depth analysis to {output_dir}")


if __name__ == "__main__":
    main()
