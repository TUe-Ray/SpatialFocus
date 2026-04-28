#!/usr/bin/env python3
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


LETTERS = ["A", "B", "C", "D"]


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


def fmt_value(value, percent=False):
    if value is None or value == "":
        return "n/a"
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            return value
    if percent:
        return f"{100.0 * float(value):.1f}%"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def ordered_option_items(options):
    if not options:
        return []
    if isinstance(options, dict):
        ordered = []
        seen = set()
        for letter in LETTERS:
            if letter in options:
                ordered.append((letter, options[letter]))
                seen.add(letter)
        for letter, text in options.items():
            if letter not in seen:
                ordered.append((letter, text))
        return ordered
    if isinstance(options, str):
        return [(None, line.strip()) for line in options.splitlines() if line.strip()]
    ordered = []
    for item in options:
        text = str(item)
        parts = text.split(". ", 1)
        if len(parts) == 2 and parts[0].strip():
            ordered.append((parts[0].strip(), parts[1].strip()))
        else:
            ordered.append((None, text))
    return ordered


def append_options_block(lines, label, options, indent="  "):
    if not options:
        lines.append(f"{indent}{label}: n/a")
        return
    lines.append(f"{indent}{label}:")
    for letter, text in ordered_option_items(options):
        if letter:
            lines.append(f"{indent}  {letter}. {text}")
        else:
            lines.append(f"{indent}  {text}")


def format_presented_letter(letter):
    return letter or "n/a"


def format_original_letter(letter, original_options):
    if not letter:
        return "n/a"
    answer_text = None
    if isinstance(original_options, dict):
        answer_text = original_options.get(letter)
    return f"{letter} ({answer_text})" if answer_text else letter


def options_text(options):
    if not options:
        return ""
    if isinstance(options, dict):
        return ", ".join(f"{letter}. {text}" for letter, text in options.items())
    return ", ".join(str(item) for item in options)


def short_text(value, limit=500):
    text = "" if value is None else str(value).replace("\n", "\\n")
    return text if len(text) <= limit else text[: limit - 3] + "..."


def sample_answer(record):
    if record.get("prompt_variant") == "option_shuffle":
        return record.get("model_original_letter") or record.get("model_presented_letter") or "n/a"
    return record.get("model_answer") or "n/a"


def sample_ground_truth(record):
    if record.get("prompt_variant") == "option_shuffle":
        return record.get("gt_original_letter") or "n/a"
    return record.get("gt_letter") or record.get("gt_answer_text") or record.get("numeric_target") or "n/a"


def add_example(lines, record, include_raw=False, include_uncertainty=False):
    lines.append(f"- Question: {record.get('question')}")
    opts = record.get("presented_options") or record.get("options") or record.get("original_options")
    if opts:
        lines.append(f"  Options: {options_text(opts)}")
    lines.append(f"  Ground truth: {sample_ground_truth(record)} ({record.get('gt_answer_text') or record.get('numeric_target') or 'n/a'})")
    lines.append(f"  Parsed model answer: {sample_answer(record)} ({record.get('model_answer_text') or 'n/a'})")
    if include_raw:
        lines.append(f"  Raw model output: `{short_text(record.get('model_raw_prediction'))}`")
    if record.get("evidence_objects"):
        lines.append(f"  Evidence objects: {', '.join(record.get('evidence_objects') or [])}")
    if record.get("spatial_evidence"):
        lines.append(f"  Spatial evidence: {record.get('spatial_evidence')}")
    if record.get("visible_reference"):
        lines.append(f"  Visible reference: {record.get('visible_reference')}")
    if record.get("estimate_reason"):
        lines.append(f"  Estimate reason: {record.get('estimate_reason')}")
    if include_uncertainty:
        lines.append(f"  Uncertainty: {record.get('uncertainty') or 'n/a'}")


def add_option_shuffle_example(lines, record, include_raw=False):
    original_options = record.get("original_options") or {}
    presented_options = record.get("presented_options") or record.get("options")

    lines.append(f"- Question: {record.get('question')}")
    append_options_block(lines, "Presented options", presented_options)
    append_options_block(lines, "Original options", original_options)
    lines.append(f"  Ground-truth presented answer: {format_presented_letter(record.get('gt_presented_letter'))}")
    lines.append(f"  Ground-truth original answer: {format_original_letter(record.get('gt_original_letter'), original_options)}")
    lines.append(f"  Raw presented answer: {format_presented_letter(record.get('model_presented_letter'))}")
    lines.append(f"  Mapped original answer: {format_original_letter(record.get('model_original_letter'), original_options)}")
    if include_raw:
        lines.append(f"  Raw model output: `{short_text(record.get('model_raw_prediction'))}`")


def add_parse_failure(lines, record):
    if record.get("prompt_variant") == "option_shuffle":
        add_option_shuffle_example(lines, record, include_raw=True)
        lines.append(f"  Parse error: {record.get('parse_error') or 'n/a'}")
        return

    add_example(lines, record, include_raw=True)
    errors = []
    if record.get("parse_error"):
        errors.append(f"parse_error={record.get('parse_error')}")
    if record.get("json_parse_error"):
        errors.append(f"json_parse_error={record.get('json_parse_error')}")
    if record.get("answer_parse_error"):
        errors.append(f"answer_parse_error={record.get('answer_parse_error')}")
    lines.append(f"  Parse error: {'; '.join(errors) if errors else 'n/a'}")


def build_report(run_dir):
    run_dir = Path(run_dir)
    metadata = load_json(run_dir / "run_metadata.json", {})
    stats = load_json(run_dir / "stats.json", {})
    by_type = read_csv(run_dir / "stats_by_question_type.csv")
    letter_bias = read_csv(run_dir / "letter_bias.csv")
    predictions = read_jsonl(run_dir / "predictions.jsonl")
    variant = metadata.get("prompt_variant") or stats.get("prompt_variant")

    lines = ["# VSiBench Probe Report", ""]
    lines.extend(
        [
            "## Run metadata",
            "",
            f"- Run name: {metadata.get('run_name', 'n/a')}",
            f"- Model: {metadata.get('model_name_or_path') or metadata.get('model') or 'n/a'}",
            f"- Checkpoint: {metadata.get('checkpoint') or 'n/a'}",
            f"- Prompt variant: {variant or 'n/a'}",
            f"- Prompt template(s): {metadata.get('prompt_template_name') or 'n/a'}",
            f"- Number of samples: {metadata.get('num_samples', 'n/a')}",
            f"- Sample seed: {metadata.get('sample_seed', 'n/a')}",
            f"- Option shuffle seed(s): {metadata.get('option_shuffle_seeds') or 'n/a'}",
            f"- Git commit: {metadata.get('git_commit') or 'n/a'}",
            f"- Timestamp: {metadata.get('timestamp') or 'n/a'}",
            "",
        ]
    )

    warnings = stats.get("selected_samples_validation_warnings") or []
    if warnings:
        lines.extend(["## Selected sample warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings[:20])
        lines.append("")

    lines.extend(["## Overall results", "", "| Metric | Value |", "|---|---:|"])
    if variant == "option_shuffle":
        lines.append(f"| Base samples | {stats.get('num_base_samples', 0)} |")
        lines.append(f"| Evaluated rows | {stats.get('num_samples', 0)} |")
        lines.append(f"| Parse success rate | {fmt_value(stats.get('parse_success_rate'), percent=True)} |")
        lines.append(f"| Accuracy original | {fmt_value(stats.get('accuracy_original'), percent=True)} |")
        lines.append(f"| Accuracy original parsed | {fmt_value(stats.get('accuracy_original_parsed'), percent=True)} |")
    else:
        lines.append(f"| Samples | {stats.get('num_samples', 0)} |")
        lines.append(f"| JSON parse success rate | {fmt_value(stats.get('json_parse_success_rate'), percent=True)} |")
        lines.append(f"| Answer parse success rate | {fmt_value(stats.get('answer_parse_success_rate'), percent=True)} |")
        lines.append(f"| MCA accuracy | {fmt_value(stats.get('mca_accuracy'), percent=True)} |")
        lines.append(f"| MCA accuracy answer-parsed | {fmt_value(stats.get('mca_accuracy_answer_parsed'), percent=True)} |")
        lines.append(f"| Numeric MRA mean | {fmt_value(stats.get('numeric_mra_mean'))} |")
        lines.append(f"| Numeric within 10% | {fmt_value(stats.get('numeric_within_10pct_rate'), percent=True)} |")
        lines.append(f"| Numeric within 25% | {fmt_value(stats.get('numeric_within_25pct_rate'), percent=True)} |")
        lines.append(f"| Open-ended normalized match | {fmt_value(stats.get('open_ended_normalized_match_rate'), percent=True)} |")
    lines.append("")

    if variant == "option_shuffle":
        lines.extend(
            [
                "## Option shuffle analysis",
                "",
                "| Metric | Value |",
                "|---|---:|",
                f"| Semantic consistency rate | {fmt_value(stats.get('semantic_consistency_rate'), percent=True)} |",
                f"| Correctness flip rate | {fmt_value(stats.get('correctness_flip_rate'), percent=True)} |",
                f"| Average unique semantic answers per sample | {fmt_value(stats.get('avg_unique_semantic_answers_per_sample'))} |",
                "",
            ]
        )
        if letter_bias:
            lines.extend(["## Letter bias", "", "| Letter | Model frequency | Ground-truth frequency | Bias |", "|---|---:|---:|---:|"])
            for row in letter_bias:
                lines.append(f"| {row.get('letter')} | {fmt_value(row.get('model_frequency'), percent=True)} | {fmt_value(row.get('ground_truth_frequency'), percent=True)} | {fmt_value(row.get('bias'), percent=True)} |")
            lines.append("")
    else:
        lines.extend(
            [
                "## Evidence JSON analysis",
                "",
                "| Metric | Value |",
                "|---|---:|",
                f"| JSON parse success rate | {fmt_value(stats.get('json_parse_success_rate'), percent=True)} |",
                f"| Answer parse success rate | {fmt_value(stats.get('answer_parse_success_rate'), percent=True)} |",
                f"| Most common uncertainty | {stats.get('most_common_uncertainty') or 'n/a'} |",
                "",
            ]
        )
        common_objects = stats.get("most_common_evidence_objects") or []
        if common_objects:
            lines.append(f"Most common evidence objects: {', '.join(f'{name} ({count})' for name, count in common_objects[:10])}")
            lines.append("")

    lines.extend(["## By question type", ""])
    if variant == "option_shuffle":
        lines.extend(["| Question type | Samples | Accuracy | Parse rate | Semantic consistency |", "|---|---:|---:|---:|---:|"])
        for row in by_type:
            lines.append(
                f"| {row.get('question_type')} | {row.get('samples')} | {fmt_value(row.get('accuracy_original'), percent=True)} | {fmt_value(row.get('parse_success_rate'), percent=True)} | {fmt_value(row.get('semantic_consistency_rate'), percent=True)} |"
            )
    else:
        lines.extend(["| Question type | Samples | Answer parse | MCA acc | Numeric MRA | Within 25% |", "|---|---:|---:|---:|---:|---:|"])
        for row in by_type:
            lines.append(
                f"| {row.get('question_type')} | {row.get('samples')} | {fmt_value(row.get('answer_parse_success_rate'), percent=True)} | {fmt_value(row.get('mca_accuracy'), percent=True)} | {fmt_value(row.get('numeric_mra_mean'))} | {fmt_value(row.get('numeric_within_25pct_rate'), percent=True)} |"
            )
    lines.append("")

    lines.extend(["## Qualitative examples", ""])
    if variant == "evidence_json":
        correct = [row for row in predictions if row.get("correct") is True and row.get("answer_parse_ok")]
        wrong = [row for row in predictions if row.get("correct") is False and row.get("answer_parse_ok")]
        parse_failures = [row for row in predictions if not row.get("json_parse_ok") or not row.get("answer_parse_ok")]
    else:
        correct = [row for row in predictions if row.get("correct_original_space") is True and row.get("parse_ok")]
        wrong = [row for row in predictions if row.get("correct_original_space") is False and row.get("parse_ok")]
        parse_failures = [row for row in predictions if not row.get("parse_ok")]

    lines.extend(["### Correct examples", ""])
    for record in correct[:3]:
        if variant == "option_shuffle":
            add_option_shuffle_example(lines, record)
        else:
            add_example(lines, record)
        lines.append("")
    if not correct:
        lines.extend(["No parsed correct examples found.", ""])

    lines.extend(["### Wrong examples with raw output", ""])
    for record in wrong[:5]:
        if variant == "option_shuffle":
            add_option_shuffle_example(lines, record, include_raw=True)
        else:
            add_example(lines, record, include_raw=True, include_uncertainty=True)
        lines.append("")
    if not wrong:
        lines.extend(["No parsed wrong examples found.", ""])

    lines.extend(["### Parse failure examples", ""])
    for record in parse_failures[:5]:
        add_parse_failure(lines, record)
        lines.append("")
    if not parse_failures:
        lines.extend(["No parse failure examples found.", ""])

    if variant == "evidence_json":
        invalid_json_answer_parsed = [row for row in predictions if row.get("json_parse_ok") is False and row.get("answer_parse_ok") is True]
        lines.extend(["### Invalid JSON but answer parsed examples", ""])
        for record in invalid_json_answer_parsed[:5]:
            add_example(lines, record, include_raw=True, include_uncertainty=True)
            lines.append("")
        if not invalid_json_answer_parsed:
            lines.extend(["No invalid-JSON answer-parsed examples found.", ""])

    if variant == "option_shuffle":
        grouped = defaultdict(list)
        for record in predictions:
            grouped[record.get("sample_id")].append(record)
        inconsistent = []
        for rows in grouped.values():
            answers = sorted({row.get("model_original_letter") for row in rows if row.get("parse_ok") and row.get("model_original_letter")})
            if len(answers) > 1:
                inconsistent.append((rows[0], rows, answers))
        lines.extend(["### Option-shuffle inconsistent examples", ""])
        for first, rows, answers in inconsistent[:5]:
            original_options = first.get("original_options") or {}
            lines.append(f"- Question: {first.get('question')}")
            append_options_block(lines, "Original options", original_options)
            lines.append(f"  Semantic answers: {', '.join(format_original_letter(answer, original_options) for answer in answers)}")
            for row in sorted(rows, key=lambda item: item.get("option_shuffle_seed")):
                lines.append(f"  Seed {row.get('option_shuffle_seed')}:")
                lines.append(f"    Raw presented answer: {format_presented_letter(row.get('model_presented_letter'))}")
                lines.append(f"    Mapped original answer: {format_original_letter(row.get('model_original_letter'), original_options)}")
                append_options_block(lines, "Presented options", row.get("presented_options") or row.get("options"), indent="    ")
                lines.append(f"    Ground-truth presented answer: {format_presented_letter(row.get('gt_presented_letter'))}")
                lines.append(f"    Ground-truth original answer: {format_original_letter(row.get('gt_original_letter'), original_options)}")
                lines.append(f"    Correct: {row.get('correct_original_space')}")
                lines.append(f"    Raw model output: `{short_text(row.get('model_raw_prediction'))}`")
            lines.append("")
        if not inconsistent:
            lines.extend(["No parsed inconsistent examples found.", ""])

    files = ["predictions.jsonl", "stats.json", "stats_by_question_type.csv", "selected_samples.json", "run_metadata.json"]
    if (run_dir / "letter_bias.csv").exists():
        files.append("letter_bias.csv")
    lines.extend(["## Files generated", ""])
    lines.extend(f"- {name}" for name in files)
    lines.append("")
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a Markdown report for a VSiBench probe run.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output")
    return parser.parse_args()


def main():
    args = parse_args()
    report = build_report(args.run_dir)
    output = Path(args.output) if args.output else Path(args.run_dir) / "report.md"
    output.write_text(report, encoding="utf-8")
    print(f"Wrote VSiBench probe report to {output}")


if __name__ == "__main__":
    main()
