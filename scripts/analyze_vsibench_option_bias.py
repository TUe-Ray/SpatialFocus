#!/usr/bin/env python3
import argparse
import csv
import importlib.util
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
UTILS_PATH = REPO_ROOT / "thinking-in-space/lmms_eval/tasks/vsibench_probe/utils.py"
LETTERS = ["A", "B", "C", "D"]
SPATIAL_ANSWERS = {
    "left": ["left"],
    "right": ["right"],
    "front": ["front", "in front", "ahead"],
    "back": ["back", "behind"],
    "front-left": ["front left", "front-left", "left front"],
    "front-right": ["front right", "front-right", "right front"],
    "back-left": ["back left", "back-left", "left back"],
    "back-right": ["back right", "back-right", "right back"],
}


def load_probe_utils():
    spec = importlib.util.spec_from_file_location("vsibench_probe_utils", UTILS_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


utils = load_probe_utils()


def safe_div(num, den):
    return num / den if den else None


def write_json(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def write_csv(path, rows, fieldnames):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def normalize_text(value):
    text = "" if value is None else str(value).strip().lower()
    text = re.sub(r"^\s*[a-d]\s*[\.\):\-]\s*", "", text, flags=re.I)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .")


def normalize_spatial_answer(text):
    normalized = normalize_text(text)
    compact = normalized.replace("-", " ")
    for label, aliases in SPATIAL_ANSWERS.items():
        for alias in aliases:
            if compact == alias or compact.startswith(alias + " ") or compact.endswith(" " + alias):
                return label
    return None


def load_local_dataset(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Local dataset path does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            for key in ("data", "rows", "examples"):
                if isinstance(payload.get(key), list):
                    return payload[key]
        if isinstance(payload, list):
            return payload
        raise ValueError(f"Unsupported JSON dataset shape in {path}")
    if suffix == ".csv":
        with path.open("r", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    if suffix == ".parquet":
        try:
            from datasets import load_dataset
        except Exception as exc:
            raise RuntimeError("Reading parquet requires the datasets package. Install it or provide JSON/JSONL/CSV.") from exc
        return list(load_dataset("parquet", data_files=str(path), split="train"))
    raise ValueError(f"Unsupported local dataset extension: {suffix}. Use JSON, JSONL, CSV, or Parquet.")


def load_hf_dataset(args):
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError("Loading Hugging Face datasets requires the datasets package. Use --local-dataset-path for a local file.") from exc

    kwargs = {}
    if args.dataset_config:
        kwargs["name"] = args.dataset_config
    try:
        dataset = load_dataset(args.dataset_name, **kwargs)
    except Exception as exc:
        raise RuntimeError(f"Could not load dataset {args.dataset_name!r}: {exc}") from exc
    if args.split not in dataset:
        available = ", ".join(dataset.keys())
        raise ValueError(f"Split {args.split!r} is not available for {args.dataset_name!r}. Available splits: {available}")
    return list(dataset[args.split])


def load_rows(args):
    if args.local_dataset_path:
        return load_local_dataset(args.local_dataset_path)
    if not args.dataset_name:
        raise ValueError("Provide either --local-dataset-path or --dataset-name.")
    return load_hf_dataset(args)


def analyze(rows):
    mca_rows = []
    skipped = 0
    for row in rows:
        options = utils.parse_options(row.get("options"))
        question_type = row.get("question_type")
        if question_type not in utils.MCA_QUESTION_TYPES or not options:
            skipped += 1
            continue
        gt_letter = utils._resolve_gt_letter(options, row.get("ground_truth"))
        if gt_letter not in LETTERS:
            skipped += 1
            continue
        mca_rows.append(
            {
                "question_type": question_type,
                "gt_letter": gt_letter,
                "answer_text": options.get(gt_letter) or row.get("ground_truth"),
                "options": options,
            }
        )

    gt_letter = Counter(row["gt_letter"] for row in mca_rows)
    qtype = Counter(row["question_type"] for row in mca_rows)
    answer_text = Counter(normalize_text(row["answer_text"]) for row in mca_rows)

    gt_letter_rows = [{"letter": letter, "count": gt_letter[letter], "frequency": safe_div(gt_letter[letter], len(mca_rows)) or 0.0} for letter in LETTERS]

    gt_by_type_rows = []
    for question_type in sorted(qtype):
        rows_for_type = [row for row in mca_rows if row["question_type"] == question_type]
        total = len(rows_for_type)
        counts = Counter(row["gt_letter"] for row in rows_for_type)
        gt_by_type_rows.append(
            {
                "question_type": question_type,
                **{f"{letter}_count": counts[letter] for letter in LETTERS},
                **{f"{letter}_frequency": safe_div(counts[letter], total) or 0.0 for letter in LETTERS},
                "total": total,
            }
        )

    option_text_rows = []
    for letter in LETTERS:
        values = Counter()
        for row in mca_rows:
            if letter in row["options"]:
                values[normalize_text(row["options"][letter])] += 1
        total = sum(values.values())
        for text, count in values.most_common():
            option_text_rows.append({"letter": letter, "option_text_normalized": text, "count": count, "frequency_within_letter": safe_div(count, total) or 0.0})

    answer_text_rows = [
        {"answer_text_normalized": text, "count": count, "frequency": safe_div(count, len(mca_rows)) or 0.0}
        for text, count in answer_text.most_common()
    ]

    answer_by_type_rows = []
    for question_type in sorted(qtype):
        rows_for_type = [row for row in mca_rows if row["question_type"] == question_type]
        counts = Counter(normalize_text(row["answer_text"]) for row in rows_for_type)
        for text, count in counts.most_common():
            answer_by_type_rows.append(
                {
                    "question_type": question_type,
                    "answer_text_normalized": text,
                    "count": count,
                    "frequency_within_question_type": safe_div(count, len(rows_for_type)) or 0.0,
                }
            )

    qtype_rows = [{"question_type": question_type, "count": count, "frequency": safe_div(count, len(mca_rows)) or 0.0} for question_type, count in qtype.most_common()]

    spatial_counts = defaultdict(Counter)
    for row in mca_rows:
        spatial = normalize_spatial_answer(row["answer_text"])
        if spatial:
            spatial_counts[row["question_type"]][spatial] += 1
    spatial_rows = []
    for question_type in sorted(spatial_counts):
        total = sum(spatial_counts[question_type].values())
        for spatial, count in spatial_counts[question_type].most_common():
            spatial_rows.append({"question_type": question_type, "spatial_answer": spatial, "count": count, "frequency": safe_div(count, total) or 0.0})

    answer_letter_counts = defaultdict(Counter)
    for row in mca_rows:
        answer_letter_counts[normalize_text(row["answer_text"])][row["gt_letter"]] += 1
    strongest_semantic_ties = []
    for text, counts in answer_letter_counts.items():
        total = sum(counts.values())
        if not total:
            continue
        letter, count = counts.most_common(1)[0]
        strongest_semantic_ties.append({"answer_text_normalized": text, "letter": letter, "count": count, "total": total, "frequency": safe_div(count, total) or 0.0})
    strongest_semantic_ties.sort(key=lambda row: (row["frequency"], row["count"]), reverse=True)

    stats = {
        "total_rows": len(rows),
        "mca_rows": len(mca_rows),
        "skipped_rows": skipped,
        "gt_letter_distribution": {row["letter"]: row for row in gt_letter_rows},
        "question_type_distribution": {row["question_type"]: row for row in qtype_rows},
        "strongest_semantic_answer_letter_ties": strongest_semantic_ties[:25],
    }
    return {
        "stats": stats,
        "gt_letter_rows": gt_letter_rows,
        "gt_by_type_rows": gt_by_type_rows,
        "option_text_rows": option_text_rows,
        "answer_text_rows": answer_text_rows,
        "answer_by_type_rows": answer_by_type_rows,
        "qtype_rows": qtype_rows,
        "spatial_rows": spatial_rows,
    }


def fmt_pct(value):
    return "n/a" if value is None else f"{100.0 * value:.1f}%"


def build_report(result, args):
    stats = result["stats"]
    lines = ["# VSiBench Multiple-Choice Option Bias", ""]
    lines.append(f"- Split: {args.split}")
    lines.append(f"- MCA rows analyzed: {stats['mca_rows']}")
    lines.append(f"- Rows skipped: {stats['skipped_rows']}")
    lines.append("")

    lines.extend(["## Overall GT letter distribution", "", "| Letter | Count | Frequency |", "|---|---:|---:|"])
    for row in result["gt_letter_rows"]:
        lines.append(f"| {row['letter']} | {row['count']} | {fmt_pct(row['frequency'])} |")
    lines.append("")

    max_letter = max(result["gt_letter_rows"], key=lambda row: row["frequency"], default=None)
    if max_letter and max_letter["frequency"] > 0.35:
        lines.append(f"Warning: letter {max_letter['letter']} is overrepresented at {fmt_pct(max_letter['frequency'])}.")
        lines.append("")

    lines.extend(["## By question type GT letter distribution", "", "| Question type | A | B | C | D | Total |", "|---|---:|---:|---:|---:|---:|"])
    for row in result["gt_by_type_rows"]:
        lines.append(f"| {row['question_type']} | {fmt_pct(row['A_frequency'])} | {fmt_pct(row['B_frequency'])} | {fmt_pct(row['C_frequency'])} | {fmt_pct(row['D_frequency'])} | {row['total']} |")
    lines.append("")

    lines.extend(["## Strongest option-position biases", "", "| Answer text | Letter | Count | Total | Frequency |", "|---|---|---:|---:|---:|"])
    ties = [row for row in stats["strongest_semantic_answer_letter_ties"] if row["total"] >= 2 and row["frequency"] >= 0.75]
    for row in ties[:20]:
        lines.append(f"| {row['answer_text_normalized']} | {row['letter']} | {row['count']} | {row['total']} | {fmt_pct(row['frequency'])} |")
    if not ties:
        lines.append("| n/a | n/a | 0 | 0 | n/a |")
    lines.append("")

    lines.extend(["## Most common answer texts", "", "| Answer text | Count | Frequency |", "|---|---:|---:|"])
    for row in result["answer_text_rows"][:20]:
        lines.append(f"| {row['answer_text_normalized']} | {row['count']} | {fmt_pct(row['frequency'])} |")
    lines.append("")

    if ties:
        lines.append("Warning: some semantic answers are strongly tied to a specific answer letter. Check `option_text_by_letter.csv` and `answer_text_distribution.csv` before interpreting model letter preferences as spatial reasoning.")
        lines.append("")
    lines.append("This analysis is intended to test whether training data could teach answer-letter or option-position shortcuts.")
    lines.append("")
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze VSiBench multiple-choice answer-position and option-text bias.")
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--local-dataset-path")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        rows = load_rows(args)
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2
    result = analyze(rows)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "gt_letter_distribution.csv", result["gt_letter_rows"], ["letter", "count", "frequency"])
    write_csv(output / "gt_letter_by_question_type.csv", result["gt_by_type_rows"], ["question_type", "A_count", "B_count", "C_count", "D_count", "A_frequency", "B_frequency", "C_frequency", "D_frequency", "total"])
    write_csv(output / "option_text_by_letter.csv", result["option_text_rows"], ["letter", "option_text_normalized", "count", "frequency_within_letter"])
    write_csv(output / "answer_text_distribution.csv", result["answer_text_rows"], ["answer_text_normalized", "count", "frequency"])
    write_csv(output / "answer_text_by_question_type.csv", result["answer_by_type_rows"], ["question_type", "answer_text_normalized", "count", "frequency_within_question_type"])
    write_csv(output / "question_type_distribution.csv", result["qtype_rows"], ["question_type", "count", "frequency"])
    write_csv(output / "spatial_answer_bias.csv", result["spatial_rows"], ["question_type", "spatial_answer", "count", "frequency"])
    write_json(output / "stats.json", result["stats"])
    (output / "report.md").write_text(build_report(result, args), encoding="utf-8")
    print(f"Wrote VSiBench option-bias analysis to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
