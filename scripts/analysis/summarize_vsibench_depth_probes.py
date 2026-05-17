#!/usr/bin/env python
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


MCA_QUESTION_TYPES = {
    "object_rel_direction_easy",
    "object_rel_direction_medium",
    "object_rel_direction_hard",
    "object_rel_distance",
    "route_planning",
    "obj_appearance_order",
}
NA_QUESTION_TYPES = {
    "object_abs_distance",
    "object_counting",
    "object_size_estimation",
    "room_size_estimation",
}
MRA_KEY = "MRA:.5:.95:.05"

TABLE_METRICS = (
    ("Room Size", "room_size_estimation_MRA:.5:.95:.05"),
    ("Abs. Dist.", "object_abs_distance_MRA:.5:.95:.05"),
    ("Obj. Size", "object_size_estimation_MRA:.5:.95:.05"),
    ("Rel. Dist.", "object_rel_distance_accuracy"),
    ("Rel. Dir.", "object_rel_direction_accuracy"),
    ("Route Plan", "route_planning_accuracy"),
    ("Appearance Order", "obj_appearance_order_accuracy"),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Summarize per-category VSiBench metrics from saved vsibench.json sample logs "
            "for the GeoRoPE depth-range probes."
        )
    )
    parser.add_argument(
        "--variant",
        action="append",
        nargs=4,
        metavar=("NAME", "VSIBENCH_JSON_OR_DIR", "EVAL_MAX_DEPTH", "NTK_ON"),
        help=(
            "Variant to summarize. Repeat for comparisons. The second value can be a "
            "vsibench.json file or a directory containing one."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/depth_range_probe_eval_summary",
        help="Output directory for JSON/CSV/Markdown summaries.",
    )
    parser.add_argument("--results-json", default="eval_depth_ntk_results.json")
    parser.add_argument("--summary-csv", default="eval_depth_ntk_summary.csv")
    parser.add_argument("--summary-md", default="eval_depth_ntk_summary.md")
    return parser.parse_args()


def parse_bool(value):
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on", "ntk"}


def parse_depth(value):
    if str(value).strip().lower() in {"", "none", "null", "na", "n/a"}:
        return ""
    return float(value)


def to_float(value, default=float("nan")):
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def mean(values):
    values = [float(value) for value in values if math.isfinite(float(value))]
    return sum(values) / len(values) if values else float("nan")


def find_vsibench_json(path):
    path = Path(path)
    if path.is_file():
        return path
    direct = path / "vsibench.json"
    if direct.is_file():
        return direct
    matches = sorted(path.rglob("vsibench.json"), key=lambda item: item.stat().st_mtime, reverse=True)
    if not matches:
        raise FileNotFoundError(f"No vsibench.json found under {path}")
    if len(matches) > 1:
        print(f"[WARN] Found {len(matches)} vsibench.json files under {path}; using newest: {matches[0]}")
    return matches[0]


def load_vsibench_docs(path):
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    docs = []
    for sample in payload.get("logs", []):
        doc = sample.get("vsibench_score") or sample.get("doc") or {}
        if doc:
            docs.append(doc)
    if not docs:
        raise ValueError(f"No VSiBench sample docs found in {path}")
    return docs


def aggregate_docs(docs):
    grouped = defaultdict(list)
    for doc in docs:
        question_type = doc.get("question_type")
        grouped[question_type].append(doc)

    metrics = {}
    counts = {}
    for question_type, rows in grouped.items():
        if question_type in MCA_QUESTION_TYPES:
            key = "accuracy"
        elif question_type in NA_QUESTION_TYPES:
            key = MRA_KEY
        else:
            continue

        values = [to_float(row.get(key)) for row in rows]
        metric_name = f"{question_type}_{key}"
        metrics[metric_name] = mean(values)
        counts[question_type] = len(rows)

    direction_keys = (
        "object_rel_direction_easy_accuracy",
        "object_rel_direction_medium_accuracy",
        "object_rel_direction_hard_accuracy",
    )
    direction_values = [metrics.pop(key) for key in direction_keys if key in metrics]
    if direction_values:
        metrics["object_rel_direction_accuracy"] = mean(direction_values)

    metrics["overall"] = mean(metrics.values())
    return metrics, counts


def load_results_overall(vsibench_json):
    results_path = vsibench_json.with_name("results.json")
    if not results_path.is_file():
        return float("nan")
    try:
        with open(results_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return to_float(payload["results"]["vsibench"]["vsibench_score,none"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return float("nan")


def summarize_variant(name, source, eval_max_depth, ntk_on):
    vsibench_json = find_vsibench_json(source)
    docs = load_vsibench_docs(vsibench_json)
    metrics, counts = aggregate_docs(docs)
    row = {
        "variant_name": name,
        "eval_max_depth": parse_depth(eval_max_depth),
        "ntk_on": parse_bool(ntk_on),
        "overall_score": metrics.get("overall", float("nan")) * 100.0,
        "samples": len(docs),
        "source": str(vsibench_json),
        "results_json_overall_score": load_results_overall(vsibench_json),
    }
    for label, metric_key in TABLE_METRICS:
        row[label] = metrics.get(metric_key, float("nan"))
    return row, metrics, counts


def csv_value(value):
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def write_csv(path, rows):
    fieldnames = [
        "variant_name",
        "eval_max_depth",
        "ntk_on",
        "overall_score",
        *[label for label, _ in TABLE_METRICS],
        "samples",
        "results_json_overall_score",
        "source",
    ]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key, "")) for key in fieldnames})


def fmt(value, digits=4):
    if value in ("", None):
        return "n/a"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def best_row(rows, key):
    available = [row for row in rows if math.isfinite(float(row.get(key, float("nan"))))]
    if not available:
        return None
    return max(available, key=lambda row: float(row[key]))


def relative_task_average(row):
    keys = ("Rel. Dist.", "Rel. Dir.", "Route Plan", "Appearance Order")
    return mean(row.get(key, float("nan")) for key in keys)


def write_markdown(path, rows):
    best_room = best_row(rows, "Room Size")
    best_overall = best_row(rows, "overall_score")
    original = next((row for row in rows if row["variant_name"].lower() == "original"), rows[0] if rows else None)

    lines = [
        "# GeoRoPE Depth-Range Probe Summary",
        "",
        "Overall is reported on the lmms-eval 0-100 scale. Category metrics are raw VSiBench means in [0, 1].",
        "",
        "| Variant | Eval Max Depth | NTK | Overall | Room Size | Abs. Dist. | Obj. Size | Rel. Dist. | Rel. Dir. | Route Plan | Appearance Order |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['variant_name']} | {fmt(row['eval_max_depth'])} | {row['ntk_on']} | "
            f"{fmt(row['overall_score'])} | {fmt(row['Room Size'])} | {fmt(row['Abs. Dist.'])} | "
            f"{fmt(row['Obj. Size'])} | {fmt(row['Rel. Dist.'])} | {fmt(row['Rel. Dir.'])} | "
            f"{fmt(row['Route Plan'])} | {fmt(row['Appearance Order'])} |"
        )

    lines.extend(["", "## Interpretation", ""])
    if best_room is not None:
        lines.append(
            f"- Best Room Size variant: `{best_room['variant_name']}` "
            f"with Room Size={fmt(best_room['Room Size'])} and overall={fmt(best_room['overall_score'])}."
        )
    if best_overall is not None:
        lines.append(
            f"- Best overall variant: `{best_overall['variant_name']}` "
            f"with overall={fmt(best_overall['overall_score'])} and Room Size={fmt(best_overall['Room Size'])}."
        )

    by_depth = defaultdict(dict)
    for row in rows:
        depth = row["eval_max_depth"]
        if depth == "":
            continue
        by_depth[float(depth)][bool(row["ntk_on"])] = row
    ntk_lines = []
    for depth in sorted(by_depth):
        pair = by_depth[depth]
        if False in pair and True in pair:
            room_delta = pair[True]["Room Size"] - pair[False]["Room Size"]
            overall_delta = pair[True]["overall_score"] - pair[False]["overall_score"]
            ntk_lines.append(
                f"depth {fmt(depth)}: Room Size delta={fmt(room_delta)}, overall delta={fmt(overall_delta)}"
            )
    if ntk_lines:
        lines.append("- NTK vs no-NTK at matched eval depths: " + "; ".join(ntk_lines) + ".")
    else:
        lines.append("- NTK vs no-NTK at matched eval depths: not enough paired variants yet.")

    if original is not None:
        base_rel = relative_task_average(original)
        rel_deltas = []
        for row in rows:
            if row is original:
                continue
            rel_delta = relative_task_average(row) - base_rel
            if math.isfinite(rel_delta):
                rel_deltas.append((row["variant_name"], rel_delta))
        if rel_deltas:
            worst_name, worst_delta = min(rel_deltas, key=lambda item: item[1])
            best_name, best_delta = max(rel_deltas, key=lambda item: item[1])
            lines.append(
                "- Relative spatial task average versus original: "
                f"best `{best_name}` delta={fmt(best_delta)}, worst `{worst_name}` delta={fmt(worst_delta)}."
            )

    if best_overall is not None:
        lines.append(f"- Recommended setting from completed runs: `{best_overall['variant_name']}` for best overall trade-off.")

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
        handle.write("\n")


def main():
    args = parse_args()
    if not args.variant:
        raise SystemExit("Provide at least one --variant NAME VSIBENCH_JSON_OR_DIR EVAL_MAX_DEPTH NTK_ON")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    details = {}
    for name, source, eval_max_depth, ntk_on in args.variant:
        row, metrics, counts = summarize_variant(name, source, eval_max_depth, ntk_on)
        rows.append(row)
        details[name] = {
            "metrics": metrics,
            "counts": counts,
            "source": row["source"],
        }

    write_csv(output_dir / args.summary_csv, rows)
    write_markdown(output_dir / args.summary_md, rows)
    with open(output_dir / args.results_json, "w", encoding="utf-8") as handle:
        json.dump({"variants": rows, "details": details}, handle, indent=2)
        handle.write("\n")

    print(f"Wrote VSiBench depth-probe summary to {output_dir}")


if __name__ == "__main__":
    main()
