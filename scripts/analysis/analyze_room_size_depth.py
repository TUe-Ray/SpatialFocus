#!/usr/bin/env python
import argparse
import csv
import ctypes
import gc
import json
import math
from collections import defaultdict
from pathlib import Path

import torch

torch.set_num_threads(1)


ROOM_SIZE_TYPE = "room_size_estimation"
SCORE_KEY = "MRA:.5:.95:.05"
DEPTH_THRESHOLDS = (5.0, 8.0, 10.0, 15.0)
DEPTH_COLUMNS = (
    "max_depth",
    "p95_depth",
    "p90_depth",
    "mean_depth",
    "fraction_depth_gt_5m",
    "fraction_depth_gt_8m",
    "fraction_depth_gt_10m",
    "fraction_depth_gt_15m",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze whether VSiBench Room Size failures correlate with CUT3R scene depth."
    )
    parser.add_argument(
        "--vsibench-json",
        default=(
            "/leonardo_scratch/fast/EUHPC_D32_006/eval/logs/VLM3R/"
            "0507_2335_eval_rope_spherical_100p_40790070_vlm_3r_model_args_dd1948/"
            "vsibench.json"
        ),
        help="Saved lmms-eval sample log containing a top-level logs list.",
    )
    parser.add_argument(
        "--sidecar-root",
        default="/leonardo_scratch/fast/EUHPC_D32_006/data/vlm3r",
        help="Root containing <dataset>/<sidecar-subdir>/<scene>.pt CUT3R point-map sidecars.",
    )
    parser.add_argument("--sidecar-subdir", default="spatial_features_points")
    parser.add_argument(
        "--output-dir",
        default="outputs/probe0_room_size_depth_analysis",
        help="Directory for room_size_depth_analysis.csv, summary CSV, report, and optional plots.",
    )
    parser.add_argument(
        "--point-key-priority",
        default="point_maps,point_maps_cam,point_maps_ref",
        help="Comma-separated point-map keys to try, in eval-compatible order.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable optional matplotlib plot generation.",
    )
    return parser.parse_args()


def to_float(value, default=0.0):
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def read_room_size_rows(vsibench_json):
    with open(vsibench_json, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    rows = []
    for sample in payload.get("logs", []):
        doc = sample.get("vsibench_score") or sample.get("doc") or {}
        if doc.get("question_type") != ROOM_SIZE_TYPE:
            continue

        score = to_float(doc.get(SCORE_KEY), default=0.0)
        rows.append(
            {
                "doc_id": sample.get("doc_id", ""),
                "sample_id": doc.get("id", sample.get("doc_id", "")),
                "dataset": doc.get("dataset", ""),
                "scene_name": doc.get("scene_name", ""),
                "question_type": doc.get("question_type", ""),
                "category": doc.get("question_type", ""),
                "prediction": doc.get("prediction", ""),
                "ground_truth": doc.get("ground_truth", ""),
                "score": score,
                "is_correct": 1 if score > 0.0 else 0,
            }
        )
    return rows


def sidecar_path(sidecar_root, sidecar_subdir, dataset, scene_name):
    return Path(sidecar_root) / str(dataset) / sidecar_subdir / f"{scene_name}.pt"


def select_point_maps(payload, key_priority):
    for key in key_priority:
        value = payload.get(key)
        if isinstance(value, torch.Tensor):
            return key, value
    keys = ", ".join(key_priority)
    raise KeyError(f"Sidecar has none of the requested point-map keys: {keys}")


def load_sidecar(path):
    try:
        return torch.load(str(path), map_location="cpu", mmap=True)
    except (TypeError, RuntimeError):
        return torch.load(str(path), map_location="cpu")


def release_memory():
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (AttributeError, OSError):
        pass


def percentile(values, q):
    if values.numel() == 1:
        return float(values.item())
    k = max(1, min(values.numel(), int(math.ceil(q * values.numel()))))
    return float(values.kthvalue(k).values.item())


def compute_depth_stats(path, key_priority):
    payload = load_sidecar(path)
    point_key, point_maps = select_point_maps(payload, key_priority)
    points = point_maps.float()
    if points.shape[-1] != 3:
        if points.dim() >= 3 and points.shape[-3] == 3:
            points = points.transpose(-3, -2).transpose(-2, -1)
        else:
            raise ValueError(f"{path} point map key {point_key} has unsupported shape {tuple(points.shape)}")

    finite_points = torch.isfinite(points).all(dim=-1)
    radius = torch.linalg.vector_norm(points, dim=-1)
    valid = finite_points & torch.isfinite(radius) & (radius > 0)
    valid_depth = radius[valid]
    if valid_depth.numel() == 0:
        raise ValueError(f"{path} has no finite positive radial-depth points")

    valid_depth = valid_depth.reshape(-1)
    stats = {
        "point_key": point_key,
        "num_depth_points": int(valid_depth.numel()),
        "max_depth": float(valid_depth.max().item()),
        "p95_depth": percentile(valid_depth, 0.95),
        "p90_depth": percentile(valid_depth, 0.90),
        "mean_depth": float(valid_depth.mean().item()),
    }
    for threshold in DEPTH_THRESHOLDS:
        stats[f"fraction_depth_gt_{int(threshold)}m"] = float((valid_depth > threshold).float().mean().item())
    return stats


def p95_group(value):
    if value < 5.0:
        return "near"
    if value < 10.0:
        return "mid"
    return "far"


def max_group(value):
    if value < 5.0:
        return "near"
    if value < 10.0:
        return "mid"
    return "far"


def summarize_group(rows, grouping_name, group_fn, value_key):
    grouped = defaultdict(list)
    for row in rows:
        grouped[group_fn(float(row[value_key]))].append(row)

    output = []
    for group in ("near", "mid", "far"):
        group_rows = grouped.get(group, [])
        num_samples = len(group_rows)
        num_correct = sum(int(row["is_correct"]) for row in group_rows)
        num_wrong = num_samples - num_correct
        avg_score = sum(float(row["score"]) for row in group_rows) / num_samples if num_samples else float("nan")
        accuracy = num_correct / num_samples if num_samples else float("nan")
        output.append(
            {
                "grouping": grouping_name,
                "group": group,
                "num_samples": num_samples,
                "accuracy": accuracy,
                "average_score": avg_score,
                "num_correct": num_correct,
                "num_wrong": num_wrong,
            }
        )
    return output


def pearson(xs, ys):
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if math.isfinite(float(x)) and math.isfinite(float(y))]
    if len(pairs) < 2:
        return float("nan")
    xs, ys = zip(*pairs)
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x == 0 or var_y == 0:
        return float("nan")
    cov = sum((x - mean_x) * (y - mean_y) for x, y in pairs)
    return cov / math.sqrt(var_x * var_y)


def fmt(value, digits=4):
    if value is None or not math.isfinite(float(value)):
        return "n/a"
    return f"{float(value):.{digits}f}"


def write_csv(path, rows, fieldnames):
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def make_plots(rows, output_dir):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []

    made = []
    specs = (
        ("p95_depth", "p95_depth_vs_correctness.png"),
        ("max_depth", "max_depth_vs_correctness.png"),
        ("fraction_depth_gt_10m", "fraction_depth_gt_10m_vs_correctness.png"),
    )
    for key, filename in specs:
        xs = [float(row[key]) for row in rows]
        ys = [float(row["is_correct"]) for row in rows]
        scores = [float(row["score"]) for row in rows]
        plt.figure(figsize=(7, 4))
        plt.scatter(xs, ys, c=scores, cmap="viridis", alpha=0.75, edgecolors="none")
        plt.yticks([0, 1], ["wrong", "correct"])
        plt.xlabel(key)
        plt.ylabel("is_correct")
        plt.colorbar(label="Room Size MRA")
        plt.tight_layout()
        path = output_dir / filename
        plt.savefig(path, dpi=160)
        plt.close()
        made.append(path.name)
    return made


def write_report(path, rows, group_rows, plot_names):
    p95_summary = [row for row in group_rows if row["grouping"] == "p95_depth"]
    max_summary = [row for row in group_rows if row["grouping"] == "max_depth"]

    def group_acc(summary, group):
        for row in summary:
            if row["group"] == group:
                return row["accuracy"]
        return float("nan")

    p95_drop = group_acc(p95_summary, "near") - group_acc(p95_summary, "far")
    max_drop = group_acc(max_summary, "near") - group_acc(max_summary, "far")
    correct = [float(row["is_correct"]) for row in rows]
    score = [float(row["score"]) for row in rows]
    corr = {
        "p95_depth_correct": pearson([row["p95_depth"] for row in rows], correct),
        "max_depth_correct": pearson([row["max_depth"] for row in rows], correct),
        "p95_depth_score": pearson([row["p95_depth"] for row in rows], score),
        "max_depth_score": pearson([row["max_depth"] for row in rows], score),
        "frac_gt_8_correct": pearson([row["fraction_depth_gt_8m"] for row in rows], correct),
        "frac_gt_10_correct": pearson([row["fraction_depth_gt_10m"] for row in rows], correct),
    }
    failures = [row for row in rows if int(row["is_correct"]) == 0]
    successes = [row for row in rows if int(row["is_correct"]) == 1]

    def avg(key, subset):
        return sum(float(row[key]) for row in subset) / len(subset) if subset else float("nan")

    p95_more_predictive = abs(corr["p95_depth_correct"]) >= abs(corr["max_depth_correct"])
    supports_hypothesis = (
        math.isfinite(p95_drop)
        and p95_drop > 0.05
        and group_acc(p95_summary, "far") < group_acc(p95_summary, "near")
    )

    lines = [
        "# Room Size Depth Analysis",
        "",
        f"Samples: {len(rows)}",
        f"Correct definition: `{SCORE_KEY} > 0`",
        "",
        "## Depth Group Summary",
        "",
        "| Grouping | Group | N | Accuracy | Avg Score | Correct | Wrong |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in group_rows:
        lines.append(
            f"| {row['grouping']} | {row['group']} | {row['num_samples']} | "
            f"{fmt(row['accuracy'])} | {fmt(row['average_score'])} | "
            f"{row['num_correct']} | {row['num_wrong']} |"
        )

    lines.extend(
        [
            "",
            "## Questions",
            "",
            f"1. Accuracy drop with p95 depth: near minus far = {fmt(p95_drop)}.",
            f"2. Accuracy drop with max depth: near minus far = {fmt(max_drop)}.",
            (
                "3. More predictive depth statistic: "
                f"{'p95_depth' if p95_more_predictive else 'max_depth'} "
                f"(corr with correctness: p95={fmt(corr['p95_depth_correct'])}, "
                f"max={fmt(corr['max_depth_correct'])})."
            ),
            (
                "4. Fractions beyond 8m/10m: "
                f"wrong avg gt8={fmt(avg('fraction_depth_gt_8m', failures))}, "
                f"correct avg gt8={fmt(avg('fraction_depth_gt_8m', successes))}; "
                f"wrong avg gt10={fmt(avg('fraction_depth_gt_10m', failures))}, "
                f"correct avg gt10={fmt(avg('fraction_depth_gt_10m', successes))}."
            ),
            (
                "5. Depth-range/clamp hypothesis: "
                + (
                    "supported by this probe; far-depth Room Size samples are worse."
                    if supports_hypothesis
                    else "not clearly supported by this probe; far-depth failures are not systematically worse."
                )
            ),
            "",
            "## Correlations",
            "",
            "| Variable | Correctness corr | Score corr |",
            "|---|---:|---:|",
            f"| p95_depth | {fmt(corr['p95_depth_correct'])} | {fmt(corr['p95_depth_score'])} |",
            f"| max_depth | {fmt(corr['max_depth_correct'])} | {fmt(corr['max_depth_score'])} |",
            f"| fraction_depth_gt_8m | {fmt(corr['frac_gt_8_correct'])} | n/a |",
            f"| fraction_depth_gt_10m | {fmt(corr['frac_gt_10_correct'])} | n/a |",
        ]
    )
    if plot_names:
        lines.extend(["", "## Plots", ""])
        lines.extend(f"- `{name}`" for name in plot_names)

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
        handle.write("\n")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    key_priority = [key.strip() for key in args.point_key_priority.split(",") if key.strip()]

    rows = read_room_size_rows(args.vsibench_json)
    if not rows:
        raise SystemExit(f"No {ROOM_SIZE_TYPE} rows found in {args.vsibench_json}")

    depth_cache = {}
    enriched = []
    for row in rows:
        key = (row["dataset"], row["scene_name"])
        if key not in depth_cache:
            path = sidecar_path(args.sidecar_root, args.sidecar_subdir, row["dataset"], row["scene_name"])
            if not path.is_file():
                raise FileNotFoundError(f"Missing CUT3R point-map sidecar: {path}")
            depth_cache[key] = compute_depth_stats(path, key_priority)
            if len(depth_cache) == 1 or len(depth_cache) % 25 == 0:
                print(f"Loaded depth stats for {len(depth_cache)} scenes", flush=True)
            release_memory()
        enriched.append({**row, **depth_cache[key]})

    group_rows = []
    group_rows.extend(summarize_group(enriched, "p95_depth", p95_group, "p95_depth"))
    group_rows.extend(summarize_group(enriched, "max_depth", max_group, "max_depth"))

    analysis_fields = [
        "doc_id",
        "sample_id",
        "dataset",
        "scene_name",
        "question_type",
        "category",
        "prediction",
        "ground_truth",
        "score",
        "is_correct",
        "point_key",
        "num_depth_points",
        *DEPTH_COLUMNS,
    ]
    summary_fields = ["grouping", "group", "num_samples", "accuracy", "average_score", "num_correct", "num_wrong"]
    write_csv(output_dir / "room_size_depth_analysis.csv", enriched, analysis_fields)
    write_csv(output_dir / "room_size_depth_group_summary.csv", group_rows, summary_fields)
    plot_names = [] if args.no_plots else make_plots(enriched, output_dir)
    write_report(output_dir / "room_size_depth_analysis.md", enriched, group_rows, plot_names)
    print(f"Wrote Room Size depth analysis to {output_dir}")


if __name__ == "__main__":
    main()
