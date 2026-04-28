#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path


def load_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path):
    rows = []
    path = Path(path)
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_csv(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_json(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def safe_float(value):
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_div(num, den):
    return num / den if den else None


def fmt_delta(value, percent=False):
    if value is None:
        return "n/a"
    if percent:
        return f"{100.0 * value:+.1f}%"
    return f"{value:+.3f}"


def fmt_value(value, percent=False):
    if value is None:
        return "n/a"
    if percent:
        return f"{100.0 * value:.1f}%"
    return f"{value:.3f}"


def slug(text):
    text = str(text)
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "run"


def overall_accuracy(stats):
    if stats.get("prompt_variant") == "option_shuffle":
        return stats.get("accuracy_original")
    if stats.get("mca_samples"):
        return stats.get("mca_accuracy")
    if stats.get("numeric_samples"):
        return stats.get("numeric_mra_mean")
    return stats.get("open_ended_normalized_match_rate")


def semantic_consistency(stats):
    return (stats.get("semantic_consistency") or {}).get("semantic_consistency_rate")


def load_run(run_dir):
    run_dir = Path(run_dir)
    return {
        "run_dir": run_dir,
        "metadata": load_json(run_dir / "run_metadata.json", {}),
        "stats": load_json(run_dir / "stats.json", {}),
        "selected_samples": load_json(run_dir / "selected_samples.json", {}),
        "by_type": read_csv(run_dir / "stats_by_question_type.csv"),
        "letter_bias": read_csv(run_dir / "letter_bias.csv"),
        "predictions": read_jsonl(run_dir / "predictions.jsonl"),
    }


def run_name(run):
    return run["metadata"].get("run_name") or run["run_dir"].name


def by_type_accuracy(row):
    for key in ("accuracy_original", "accuracy", "mca_accuracy", "numeric_mra_mean", "open_ended_normalized_match_rate"):
        value = safe_float(row.get(key))
        if value is not None:
            return value
    return None


def compare_question_types(baseline, new):
    base_map = {row.get("question_type"): row for row in baseline["by_type"]}
    new_map = {row.get("question_type"): row for row in new["by_type"]}
    improved = []
    regressed = []
    for question_type in sorted(set(base_map) & set(new_map)):
        base_acc = by_type_accuracy(base_map[question_type])
        new_acc = by_type_accuracy(new_map[question_type])
        if base_acc is None or new_acc is None:
            continue
        delta = new_acc - base_acc
        row = {"question_type": question_type, "baseline_accuracy": base_acc, "new_accuracy": new_acc, "delta": delta}
        if delta > 0:
            improved.append(row)
        elif delta < 0:
            regressed.append(row)
    improved.sort(key=lambda row: row["delta"], reverse=True)
    regressed.sort(key=lambda row: row["delta"])
    return improved, regressed


def summarize_predictions(predictions):
    by_sample = {}
    grouped = {}
    for row in predictions:
        grouped.setdefault(row.get("sample_id"), []).append(row)

    for sample_id, rows in grouped.items():
        variant = rows[0].get("prompt_variant")
        if variant == "option_shuffle":
            parsed = [row for row in rows if row.get("parse_ok")]
            correct_values = [bool(row.get("correct_original_space")) for row in parsed if isinstance(row.get("correct_original_space"), bool)]
            semantic_answers = sorted({row.get("model_original_letter") for row in parsed if row.get("model_original_letter")})
            semantic_consistent = len(semantic_answers) == 1 if semantic_answers else False
        else:
            parsed = [row for row in rows if row.get("answer_parse_ok")]
            correct_values = [bool(row.get("correct")) for row in parsed if isinstance(row.get("correct"), bool)]
            semantic_answers = sorted({row.get("model_answer") for row in parsed if row.get("model_answer")})
            semantic_consistent = None

        by_sample[sample_id] = {
            "sample_id": sample_id,
            "probe_uids": [row.get("probe_uid") for row in rows if row.get("probe_uid")],
            "question_type": rows[0].get("question_type"),
            "question": rows[0].get("question"),
            "correct_rate": safe_div(sum(correct_values), len(correct_values)),
            "parsed_count": len(parsed),
            "total_count": len(rows),
            "semantic_answers": semantic_answers,
            "semantic_consistent": semantic_consistent,
        }
    return by_sample


def compare_samples(baseline, new):
    base_samples = summarize_predictions(baseline["predictions"])
    new_samples = summarize_predictions(new["predictions"])
    wrong_to_correct = []
    correct_to_wrong = []
    consistency_improved = []
    consistency_regressed = []

    for sample_id in sorted(set(base_samples) & set(new_samples)):
        base = base_samples[sample_id]
        new_item = new_samples[sample_id]
        base_correct = base["correct_rate"]
        new_correct = new_item["correct_rate"]
        payload = {"sample_id": sample_id, "question": base["question"], "question_type": base["question_type"], "baseline": base, "new": new_item}

        if base_correct == 0 and new_correct is not None and new_correct > 0:
            wrong_to_correct.append(payload)
        if base_correct is not None and new_correct is not None and base_correct > 0 and new_correct < base_correct:
            correct_to_wrong.append(payload)

        if base["semantic_consistent"] is False and new_item["semantic_consistent"] is True:
            consistency_improved.append(payload)
        if base["semantic_consistent"] is True and new_item["semantic_consistent"] is False:
            consistency_regressed.append(payload)

    return wrong_to_correct, correct_to_wrong, consistency_improved, consistency_regressed


def compare_letter_bias(baseline, new):
    base_map = {row.get("letter"): row for row in baseline["letter_bias"]}
    new_map = {row.get("letter"): row for row in new["letter_bias"]}
    rows = []
    for letter in sorted(set(base_map) | set(new_map)):
        base_bias = safe_float((base_map.get(letter) or {}).get("bias"))
        new_bias = safe_float((new_map.get(letter) or {}).get("bias"))
        rows.append(
            {
                "letter": letter,
                "baseline_bias": base_bias,
                "new_bias": new_bias,
                "delta": None if base_bias is None or new_bias is None else new_bias - base_bias,
            }
        )
    return rows


def selected_signature(run):
    selected = run.get("selected_samples") or {}
    metadata = run.get("metadata") or {}
    stats = run.get("stats") or {}
    return {
        "sample_seed": selected.get("sample_seed", metadata.get("sample_seed", stats.get("sample_seed"))),
        "num_samples": selected.get("num_samples", metadata.get("num_samples", stats.get("num_requested_samples"))),
        "actual_num_samples": selected.get("actual_num_samples"),
        "prompt_variant": selected.get("prompt_variant", metadata.get("prompt_variant", stats.get("prompt_variant"))),
        "option_shuffle_seeds": selected.get("option_shuffle_seeds", metadata.get("option_shuffle_seeds", stats.get("option_shuffle_seeds"))),
        "sample_order": selected.get("sample_order") or [],
    }


def compare_selected_samples_or_raise(baseline, new, allow_mismatch=False):
    base_sig = selected_signature(baseline)
    new_sig = selected_signature(new)
    mismatches = []
    for key in ["sample_seed", "num_samples", "actual_num_samples", "prompt_variant"]:
        if base_sig.get(key) != new_sig.get(key):
            mismatches.append(f"{key} differs: baseline={base_sig.get(key)!r}, new={new_sig.get(key)!r}")

    if base_sig.get("prompt_variant") == "option_shuffle" or new_sig.get("prompt_variant") == "option_shuffle":
        if base_sig.get("option_shuffle_seeds") != new_sig.get("option_shuffle_seeds"):
            mismatches.append(f"option_shuffle_seeds differ: baseline={base_sig.get('option_shuffle_seeds')!r}, new={new_sig.get('option_shuffle_seeds')!r}")

    if base_sig.get("sample_order") != new_sig.get("sample_order"):
        mismatches.append("sample_order differs")

    if mismatches and not allow_mismatch:
        raise ValueError("Refusing to compare mismatched VSiBench probe runs. " + " | ".join(mismatches) + " Pass --allow-mismatch to compare overlapping sample IDs only.")
    return mismatches


def build_report(stats):
    lines = ["# VSiBench Probe Comparison", ""]
    lines.append(f"Baseline: {stats.get('baseline_run')}")
    lines.append("")
    for comparison in stats.get("comparisons", []):
        lines.extend(
            [
                f"## {comparison['new_run']}",
                "",
                "| Metric | Baseline | New | Delta |",
                "|---|---:|---:|---:|",
                f"| Accuracy | {fmt_value(comparison.get('baseline_accuracy'), percent=True)} | {fmt_value(comparison.get('new_accuracy'), percent=True)} | {fmt_delta(comparison.get('accuracy_delta'), percent=True)} |",
                f"| Semantic consistency | {fmt_value(comparison.get('baseline_semantic_consistency'), percent=True)} | {fmt_value(comparison.get('new_semantic_consistency'), percent=True)} | {fmt_delta(comparison.get('semantic_consistency_delta'), percent=True)} |",
                "",
                "### Question types improved",
                "",
            ]
        )
        if comparison.get("question_types_improved"):
            for row in comparison["question_types_improved"][:10]:
                lines.append(f"- {row['question_type']}: {fmt_delta(row['delta'], percent=True)}")
        else:
            lines.append("No improved question types found.")
        lines.extend(["", "### Question types regressed", ""])
        if comparison.get("question_types_regressed"):
            for row in comparison["question_types_regressed"][:10]:
                lines.append(f"- {row['question_type']}: {fmt_delta(row['delta'], percent=True)}")
        else:
            lines.append("No regressed question types found.")
        lines.extend(
            [
                "",
                "### Sample changes",
                "",
                f"- Baseline wrong, new correct: {comparison.get('baseline_wrong_new_correct_count', 0)}",
                f"- Baseline correct, new wrong: {comparison.get('baseline_correct_new_wrong_count', 0)}",
                f"- Option-shuffle consistency improved: {comparison.get('consistency_improved_count', 0)}",
                f"- Option-shuffle consistency regressed: {comparison.get('consistency_regressed_count', 0)}",
                "",
            ]
        )
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(description="Compare two or more VSiBench probe run directories.")
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-mismatch", action="store_true", help="Allow comparison when selected_samples.json metadata/order differ; compares overlapping sample IDs only.")
    return parser.parse_args()


def main():
    args = parse_args()
    if len(args.runs) < 2:
        raise ValueError("Provide at least two run directories: baseline followed by comparison run(s).")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline = load_run(args.runs[0])
    comparisons = []

    for new_run_dir in args.runs[1:]:
        new = load_run(new_run_dir)
        sample_mismatches = compare_selected_samples_or_raise(baseline, new, allow_mismatch=args.allow_mismatch)
        new_slug = slug(run_name(new))
        improved, regressed = compare_question_types(baseline, new)
        wrong_to_correct, correct_to_wrong, consistency_improved, consistency_regressed = compare_samples(baseline, new)
        letter_bias = compare_letter_bias(baseline, new)

        write_jsonl(output_dir / f"{new_slug}_baseline_wrong_new_correct.jsonl", wrong_to_correct)
        write_jsonl(output_dir / f"{new_slug}_baseline_correct_new_wrong.jsonl", correct_to_wrong)
        write_jsonl(output_dir / f"{new_slug}_consistency_improved.jsonl", consistency_improved)
        write_jsonl(output_dir / f"{new_slug}_consistency_regressed.jsonl", consistency_regressed)

        base_acc = overall_accuracy(baseline["stats"])
        new_acc = overall_accuracy(new["stats"])
        base_consistency = semantic_consistency(baseline["stats"])
        new_consistency = semantic_consistency(new["stats"])
        comparisons.append(
            {
                "new_run": run_name(new),
                "new_run_dir": str(new["run_dir"]),
                "selected_samples_mismatches": sample_mismatches,
                "baseline_accuracy": base_acc,
                "new_accuracy": new_acc,
                "accuracy_delta": None if base_acc is None or new_acc is None else new_acc - base_acc,
                "baseline_semantic_consistency": base_consistency,
                "new_semantic_consistency": new_consistency,
                "semantic_consistency_delta": None if base_consistency is None or new_consistency is None else new_consistency - base_consistency,
                "letter_bias_difference": letter_bias,
                "question_types_improved": improved,
                "question_types_regressed": regressed,
                "baseline_wrong_new_correct_count": len(wrong_to_correct),
                "baseline_correct_new_wrong_count": len(correct_to_wrong),
                "consistency_improved_count": len(consistency_improved),
                "consistency_regressed_count": len(consistency_regressed),
                "files": {
                    "baseline_wrong_new_correct": f"{new_slug}_baseline_wrong_new_correct.jsonl",
                    "baseline_correct_new_wrong": f"{new_slug}_baseline_correct_new_wrong.jsonl",
                    "consistency_improved": f"{new_slug}_consistency_improved.jsonl",
                    "consistency_regressed": f"{new_slug}_consistency_regressed.jsonl",
                },
            }
        )

    stats = {"baseline_run": run_name(baseline), "baseline_run_dir": str(baseline["run_dir"]), "comparisons": comparisons}
    write_json(output_dir / "stats.json", stats)
    (output_dir / "report.md").write_text(build_report(stats), encoding="utf-8")
    print(f"Wrote VSiBench probe comparison to {output_dir}")


if __name__ == "__main__":
    main()
