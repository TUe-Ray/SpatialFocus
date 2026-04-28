#!/usr/bin/env python3
import argparse
import csv
import json
import os
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


LETTERS = ["A", "B", "C", "D"]
MCA_QUESTION_TYPES = {
    "object_rel_direction_easy",
    "object_rel_direction_medium",
    "object_rel_direction_hard",
    "object_rel_distance",
    "route_planning",
    "obj_appearance_order",
}


def load_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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


def write_csv(path, rows, fieldnames):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def safe_div(num, den):
    return num / den if den else None


def mean(values):
    values = [value for value in values if isinstance(value, (int, float))]
    return safe_div(sum(values), len(values))


def bool_mean(rows, key):
    values = [row.get(key) for row in rows if isinstance(row.get(key), bool)]
    return safe_div(sum(bool(value) for value in values), len(values))


def true_rate_over_total(rows, key):
    return safe_div(sum(row.get(key) is True for row in rows), len(rows))


def run_git(args):
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def parse_model_name(model_args):
    if not model_args:
        return ""
    for prefix in ("peft=", "delta=", "pretrained=", "model=", "path=", "engine="):
        if prefix in model_args:
            return model_args.split(prefix, 1)[1].split(",", 1)[0]
    return ""


def env_int(name):
    value = os.getenv(name)
    if value in (None, ""):
        return None
    try:
        return int(value)
    except ValueError:
        return None


def parse_seed_list(value):
    if value in (None, ""):
        return []
    if isinstance(value, list):
        return [int(item) for item in value if item is not None]
    return [int(item) for item in str(value).replace(",", " ").split() if item]


def find_raw_sample_file(run_dir, raw_lmms_eval_dir=None):
    candidates = []
    search_roots = []
    if raw_lmms_eval_dir:
        search_roots.append(Path(raw_lmms_eval_dir))
    search_roots.append(Path(run_dir) / "raw_lmms_eval")
    search_roots.append(Path(run_dir))

    for root in search_roots:
        if root.exists():
            candidates.extend(root.glob("**/vsibench_probe.json"))

    candidates = sorted(set(candidates), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"Could not find raw lmms-eval vsibench_probe.json under {run_dir}")
    return candidates[0]


def extract_records(raw_samples):
    records = []
    for log in raw_samples.get("logs", []):
        record = log.get("vsibench_probe_score") or log.get("vsibench_probe")
        if isinstance(record, dict):
            records.append(record)
    records.sort(key=lambda row: (row.get("probe_index", 0), row.get("option_shuffle_seed") if row.get("option_shuffle_seed") is not None else -1))
    return records


def build_selected_samples(records, args, prompt_variant, option_shuffle_seeds):
    seen = {}
    for record in records:
        sample_id = record.get("sample_id")
        if sample_id is None or sample_id in seen:
            continue
        seen[sample_id] = {
            "probe_index": record.get("probe_index"),
            "sample_id": sample_id,
            "dataset_index": record.get("dataset_index"),
            "scene_name": record.get("scene_name"),
            "question_type": record.get("question_type"),
        }

    sample_order = sorted(seen.values(), key=lambda row: row.get("probe_index", 0))
    return {
        "dataset": "vsibench",
        "prompt_variant": prompt_variant,
        "num_samples": args.num_samples,
        "actual_num_samples": len(sample_order),
        "sample_seed": args.sample_seed,
        "option_shuffle_seeds": option_shuffle_seeds if prompt_variant == "option_shuffle" else None,
        "reconstructed_from_predictions": True,
        "sample_order": sample_order,
    }


def validate_selected_samples(selected_samples, records, prompt_variant, option_shuffle_seeds):
    warnings = []
    sample_order = selected_samples.get("sample_order") or []
    selected_ids = [row.get("sample_id") for row in sample_order]
    selected_set = set(selected_ids)
    observed_ids = []
    seen = set()
    for record in sorted(records, key=lambda row: (row.get("probe_index", 0), row.get("option_shuffle_seed") if row.get("option_shuffle_seed") is not None else -1)):
        sample_id = record.get("sample_id")
        if sample_id not in seen:
            observed_ids.append(sample_id)
            seen.add(sample_id)
    observed_set = set(observed_ids)

    unexpected = sorted(observed_set - selected_set)
    missing = sorted(selected_set - observed_set)
    if unexpected:
        warnings.append(f"Predictions contain sample IDs not present in selected_samples.json: {unexpected[:10]}")
    if missing:
        warnings.append(f"selected_samples.json contains sample IDs with no predictions: {missing[:10]}")
    if selected_set and not observed_set.issubset(selected_set):
        warnings.append("Prediction sample IDs are not a subset of selected_samples sample_order")
    if selected_ids and observed_ids and observed_ids != [sample_id for sample_id in selected_ids if sample_id in observed_set]:
        warnings.append("Prediction sample order does not match selected_samples sample_order")

    if prompt_variant == "option_shuffle":
        expected_seeds = option_shuffle_seeds or selected_samples.get("option_shuffle_seeds") or []
        expected_seeds = [int(seed) for seed in expected_seeds]
        by_sample = defaultdict(set)
        for record in records:
            if record.get("option_shuffle_seed") is not None:
                by_sample[record.get("sample_id")].add(int(record.get("option_shuffle_seed")))
        for sample_id in selected_ids:
            found = by_sample.get(sample_id, set())
            missing_seeds = sorted(set(expected_seeds) - found)
            extra_seeds = sorted(found - set(expected_seeds))
            if missing_seeds:
                warnings.append(f"Sample {sample_id} is missing option shuffle seed records: {missing_seeds}")
            if extra_seeds:
                warnings.append(f"Sample {sample_id} has unexpected option shuffle seed records: {extra_seeds}")
    elif selected_ids:
        counts = Counter(record.get("sample_id") for record in records)
        for sample_id in selected_ids:
            if counts.get(sample_id, 0) != 1:
                warnings.append(f"Evidence JSON sample {sample_id} has {counts.get(sample_id, 0)} prediction records; expected 1")

    return warnings


def option_shuffle_stats(records):
    total = len(records)
    parsed = [row for row in records if row.get("parse_ok")]
    parsed_count = len(parsed)
    model_presented = Counter(row.get("model_presented_letter") for row in parsed if row.get("model_presented_letter") in LETTERS)
    model_original = Counter(row.get("model_original_letter") for row in parsed if row.get("model_original_letter") in LETTERS)
    gt_presented = Counter(row.get("gt_presented_letter") for row in records if row.get("gt_presented_letter") in LETTERS)
    gt_original = Counter(row.get("gt_original_letter") for row in records if row.get("gt_original_letter") in LETTERS)

    by_sample = defaultdict(list)
    for row in records:
        by_sample[row.get("sample_id")].append(row)

    consistency_rows = []
    for sample_id, sample_rows in by_sample.items():
        parsed_sample_rows = [row for row in sample_rows if row.get("parse_ok")]
        semantic_answers = [row.get("model_original_letter") for row in parsed_sample_rows if row.get("model_original_letter")]
        unique_answers = sorted(set(semantic_answers))
        correctness_values = [row.get("correct_original_space") for row in parsed_sample_rows if isinstance(row.get("correct_original_space"), bool)]
        consistency_rows.append(
            {
                "sample_id": sample_id,
                "question_type": sample_rows[0].get("question_type"),
                "semantic_consistent": len(unique_answers) == 1 if semantic_answers else False,
                "unique_semantic_answers": len(unique_answers),
                "correctness_flip": len(set(correctness_values)) > 1 if correctness_values else False,
            }
        )

    by_type_rows = []
    for question_type in sorted({row.get("question_type") for row in records}):
        rows = [row for row in records if row.get("question_type") == question_type]
        type_parsed = [row for row in rows if row.get("parse_ok")]
        type_consistency = [row for row in consistency_rows if row.get("question_type") == question_type]
        by_type_rows.append(
            {
                "question_type": question_type,
                "samples": len(rows),
                "base_samples": len({row.get("sample_id") for row in rows}),
                "parsed_predictions": len(type_parsed),
                "parse_success_rate": safe_div(len(type_parsed), len(rows)),
                "accuracy_presented": true_rate_over_total(rows, "correct_presented_space"),
                "accuracy_original": true_rate_over_total(rows, "correct_original_space"),
                "accuracy_presented_parsed": bool_mean(type_parsed, "correct_presented_space"),
                "accuracy_original_parsed": bool_mean(type_parsed, "correct_original_space"),
                "semantic_consistency_rate": bool_mean(type_consistency, "semantic_consistent"),
                "correctness_flip_rate": bool_mean(type_consistency, "correctness_flip"),
                "avg_unique_semantic_answers_per_sample": mean([row["unique_semantic_answers"] for row in type_consistency]),
            }
        )

    def counter_dict(counter):
        return {letter: counter[letter] for letter in LETTERS}

    stats = {
        "prompt_variant": "option_shuffle",
        "num_samples": total,
        "num_base_samples": len(by_sample),
        "parsed_predictions": parsed_count,
        "parse_success_rate": safe_div(parsed_count, total),
        "accuracy_presented": true_rate_over_total(records, "correct_presented_space"),
        "accuracy_original": true_rate_over_total(records, "correct_original_space"),
        "accuracy_presented_parsed": bool_mean(parsed, "correct_presented_space"),
        "accuracy_original_parsed": bool_mean(parsed, "correct_original_space"),
        "model_predicted_presented_letter_distribution": counter_dict(model_presented),
        "model_predicted_original_letter_distribution": counter_dict(model_original),
        "ground_truth_presented_letter_distribution": counter_dict(gt_presented),
        "ground_truth_original_letter_distribution": counter_dict(gt_original),
        "semantic_consistency_rate": bool_mean(consistency_rows, "semantic_consistent"),
        "correctness_flip_rate": bool_mean(consistency_rows, "correctness_flip"),
        "avg_unique_semantic_answers_per_sample": mean([row["unique_semantic_answers"] for row in consistency_rows]),
        "semantic_consistency": {
            "semantic_consistency_rate": bool_mean(consistency_rows, "semantic_consistent"),
            "avg_unique_semantic_answers_per_sample": mean([row["unique_semantic_answers"] for row in consistency_rows]),
            "correctness_flip_rate": bool_mean(consistency_rows, "correctness_flip"),
            "by_question_type": {
                row["question_type"]: {
                    "semantic_consistency_rate": row["semantic_consistency_rate"],
                    "correctness_flip_rate": row["correctness_flip_rate"],
                }
                for row in by_type_rows
            },
        },
        "by_question_type": {row["question_type"]: row for row in by_type_rows},
    }

    letter_rows = []
    for letter in LETTERS:
        model_frequency = safe_div(model_presented[letter], parsed_count) or 0.0
        gt_frequency = safe_div(gt_presented[letter], total) or 0.0
        letter_rows.append(
            {
                "letter": letter,
                "model_count": model_presented[letter],
                "model_frequency": model_frequency,
                "ground_truth_count": gt_presented[letter],
                "ground_truth_frequency": gt_frequency,
                "bias": model_frequency - gt_frequency,
            }
        )
    stats["letter_bias"] = {row["letter"]: row["bias"] for row in letter_rows}
    return stats, by_type_rows, letter_rows


def _is_mca_record(row):
    return bool(row.get("is_mca")) or row.get("question_type") in MCA_QUESTION_TYPES and bool(row.get("options"))


def _is_numeric_record(row):
    return bool(row.get("is_numeric")) or row.get("numeric_target") is not None


def evidence_json_stats(records):
    total = len(records)
    json_parsed = [row for row in records if row.get("json_parse_ok")]
    answer_parsed = [row for row in records if row.get("answer_parse_ok")]
    mca_rows = [row for row in records if _is_mca_record(row)]
    mca_answer_parsed = [row for row in mca_rows if row.get("answer_parse_ok")]
    numeric_rows = [row for row in records if not _is_mca_record(row) and _is_numeric_record(row)]
    numeric_answer_parsed = [row for row in numeric_rows if row.get("answer_parse_ok")]
    open_rows = [row for row in records if not _is_mca_record(row) and not _is_numeric_record(row)]
    open_answer_parsed = [row for row in open_rows if row.get("answer_parse_ok")]

    uncertainty = Counter(row.get("uncertainty") or "unknown" for row in records)
    uncertainty_correct_mca = Counter(row.get("uncertainty") or "unknown" for row in mca_rows if row.get("correct") is True)
    uncertainty_wrong_mca = Counter(row.get("uncertainty") or "unknown" for row in mca_rows if row.get("correct") is False)
    evidence_objects = Counter()
    wrong_mca_evidence_objects = Counter()
    for row in records:
        evidence_objects.update(row.get("evidence_objects") or [])
        if _is_mca_record(row) and row.get("correct") is False:
            wrong_mca_evidence_objects.update(row.get("evidence_objects") or [])

    by_type_rows = []
    for question_type in sorted({row.get("question_type") for row in records}):
        rows = [row for row in records if row.get("question_type") == question_type]
        type_mca = [row for row in rows if _is_mca_record(row)]
        type_mca_parsed = [row for row in type_mca if row.get("answer_parse_ok")]
        type_numeric = [row for row in rows if not _is_mca_record(row) and _is_numeric_record(row)]
        type_open = [row for row in rows if not _is_mca_record(row) and not _is_numeric_record(row)]
        by_type_rows.append(
            {
                "question_type": question_type,
                "samples": len(rows),
                "json_parse_success_rate": safe_div(sum(row.get("json_parse_ok") is True for row in rows), len(rows)),
                "answer_parse_success_rate": safe_div(sum(row.get("answer_parse_ok") is True for row in rows), len(rows)),
                "mca_samples": len(type_mca),
                "mca_accuracy": true_rate_over_total(type_mca, "correct"),
                "mca_accuracy_answer_parsed": bool_mean(type_mca_parsed, "correct"),
                "numeric_samples": len(type_numeric),
                "numeric_mra_mean": mean([row.get("numeric_mra") for row in type_numeric if row.get("answer_parse_ok")]),
                "numeric_within_10pct_rate": true_rate_over_total(type_numeric, "numeric_within_10pct"),
                "numeric_within_25pct_rate": true_rate_over_total(type_numeric, "numeric_within_25pct"),
                "open_ended_samples": len(type_open),
                "open_ended_normalized_match_rate": true_rate_over_total(type_open, "open_ended_normalized_match"),
            }
        )

    stats = {
        "prompt_variant": "evidence_json",
        "num_samples": total,
        "json_parsed": len(json_parsed),
        "json_parse_success_rate": safe_div(len(json_parsed), total),
        "answer_parsed": len(answer_parsed),
        "answer_parse_success_rate": safe_div(len(answer_parsed), total),
        "mca_samples": len(mca_rows),
        "mca_accuracy": true_rate_over_total(mca_rows, "correct"),
        "mca_accuracy_answer_parsed": bool_mean(mca_answer_parsed, "correct"),
        "numeric_samples": len(numeric_rows),
        "numeric_answer_parsed": len(numeric_answer_parsed),
        "numeric_mra_mean": mean([row.get("numeric_mra") for row in numeric_answer_parsed]),
        "numeric_within_10pct_rate": true_rate_over_total(numeric_rows, "numeric_within_10pct"),
        "numeric_within_10pct_rate_parsed": bool_mean(numeric_answer_parsed, "numeric_within_10pct"),
        "numeric_within_25pct_rate": true_rate_over_total(numeric_rows, "numeric_within_25pct"),
        "numeric_within_25pct_rate_parsed": bool_mean(numeric_answer_parsed, "numeric_within_25pct"),
        "open_ended_samples": len(open_rows),
        "open_ended_answer_parsed": len(open_answer_parsed),
        "open_ended_normalized_match_rate": true_rate_over_total(open_rows, "open_ended_normalized_match"),
        "uncertainty_distribution": dict(uncertainty),
        "uncertainty_distribution_correct_mca": dict(uncertainty_correct_mca),
        "uncertainty_distribution_wrong_mca": dict(uncertainty_wrong_mca),
        "most_common_uncertainty": uncertainty.most_common(1)[0][0] if uncertainty else None,
        "most_common_evidence_objects": evidence_objects.most_common(20),
        "most_common_evidence_objects_wrong_mca": wrong_mca_evidence_objects.most_common(20),
        "by_question_type": {row["question_type"]: row for row in by_type_rows},
    }
    return stats, by_type_rows


def infer_args(args, run_dir, raw_samples, records, existing_metadata, selected_samples):
    raw_args = raw_samples.get("args") or {}
    prompt_variant = args.prompt_variant or existing_metadata.get("prompt_variant") or (records[0].get("prompt_variant") if records else None) or os.getenv("PROMPT_VARIANT") or "option_shuffle"
    option_shuffle_seeds = args.option_shuffle_seeds
    if not option_shuffle_seeds:
        option_shuffle_seeds = parse_seed_list(existing_metadata.get("option_shuffle_seeds"))
    if not option_shuffle_seeds and selected_samples:
        option_shuffle_seeds = parse_seed_list(selected_samples.get("option_shuffle_seeds"))
    if not option_shuffle_seeds and prompt_variant == "option_shuffle":
        option_shuffle_seeds = sorted({int(row.get("option_shuffle_seed")) for row in records if row.get("option_shuffle_seed") is not None})
    if not option_shuffle_seeds and prompt_variant == "option_shuffle":
        option_shuffle_seeds = parse_seed_list(os.getenv("OPTION_SHUFFLE_SEEDS") or os.getenv("OPTION_SHUFFLE_SEED") or "0")

    args.prompt_variant = prompt_variant
    args.option_shuffle_seeds = option_shuffle_seeds
    args.run_name = args.run_name or existing_metadata.get("run_name") or run_dir.name
    args.num_samples = args.num_samples or existing_metadata.get("num_samples") or (selected_samples or {}).get("num_samples") or env_int("NUM_SAMPLES") or len({row.get("sample_id") for row in records})
    args.sample_seed = args.sample_seed if args.sample_seed is not None else existing_metadata.get("sample_seed")
    args.sample_seed = args.sample_seed if args.sample_seed is not None else (selected_samples or {}).get("sample_seed")
    args.sample_seed = args.sample_seed if args.sample_seed is not None else env_int("SAMPLE_SEED")
    args.sample_seed = args.sample_seed if args.sample_seed is not None else 42
    args.model = args.model or existing_metadata.get("model") or raw_args.get("model")
    args.model_args = args.model_args or existing_metadata.get("model_args") or raw_args.get("model_args") or ""
    args.model_name_or_path = args.model_name_or_path or existing_metadata.get("model_name_or_path") or parse_model_name(args.model_args)
    args.checkpoint = args.checkpoint or existing_metadata.get("checkpoint") or args.model_name_or_path
    args.notes = args.notes if args.notes is not None else existing_metadata.get("notes", "")
    return args


def build_metadata(raw_samples, args, records):
    raw_args = raw_samples.get("args") or {}
    model_configs = raw_samples.get("model_configs") or {}
    generation_kwargs = model_configs.get("generation_kwargs") or {}
    prompt_templates = {}
    for record in records:
        name = record.get("prompt_template_name")
        sha = record.get("prompt_template_sha256")
        if name and sha:
            prompt_templates[name] = sha
    template_names = sorted(prompt_templates)

    return {
        "run_name": args.run_name,
        "timestamp": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "model": raw_args.get("model") or args.model,
        "model_name_or_path": args.model_name_or_path,
        "model_args": args.model_args,
        "checkpoint": args.checkpoint,
        "task": "vsibench_probe",
        "prompt_variant": args.prompt_variant,
        "prompt_template_name": template_names[0] if len(template_names) == 1 else template_names,
        "prompt_template_sha256": prompt_templates[template_names[0]] if len(template_names) == 1 else prompt_templates,
        "prompt_template_names": template_names,
        "prompt_template_sha256_by_name": prompt_templates,
        "save_rendered_prompts": any("rendered_prompt" in record for record in records),
        "num_samples": args.num_samples,
        "sample_seed": args.sample_seed,
        "option_shuffle_seed": args.option_shuffle_seeds[0] if args.option_shuffle_seeds else None,
        "option_shuffle_seeds": args.option_shuffle_seeds,
        "generation_kwargs": generation_kwargs,
        "git_commit": run_git(["rev-parse", "HEAD"]),
        "git_branch": run_git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "raw_lmms_eval_file": str(args.raw_sample_file) if args.raw_sample_file else None,
        "notes": args.notes or "",
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze a VSiBench probe lmms-eval run.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--raw-lmms-eval-dir")
    parser.add_argument("--raw-sample-file")
    parser.add_argument("--run-name")
    parser.add_argument("--prompt-variant", choices=["option_shuffle", "evidence_json"])
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--sample-seed", type=int)
    parser.add_argument("--option-shuffle-seeds", type=int, nargs="*", default=[])
    parser.add_argument("--model")
    parser.add_argument("--model-args")
    parser.add_argument("--model-name-or-path")
    parser.add_argument("--checkpoint")
    parser.add_argument("--notes")
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    existing_metadata = load_json(run_dir / "run_metadata.json", {}) or {}

    raw_sample_file_value = args.raw_sample_file or existing_metadata.get("raw_lmms_eval_file")
    raw_sample_file = Path(raw_sample_file_value) if raw_sample_file_value else None
    if raw_sample_file is None or not raw_sample_file.exists():
        raw_sample_file = find_raw_sample_file(run_dir, args.raw_lmms_eval_dir)
    args.raw_sample_file = raw_sample_file
    raw_samples = load_json(raw_sample_file)
    records = extract_records(raw_samples)

    selected_path = run_dir / "selected_samples.json"
    selected_samples = load_json(selected_path, None)
    args = infer_args(args, run_dir, raw_samples, records, existing_metadata, selected_samples)

    write_jsonl(run_dir / "predictions.jsonl", records)

    if selected_samples is None:
        selected_samples = build_selected_samples(records, args, args.prompt_variant, args.option_shuffle_seeds)
        write_json(selected_path, selected_samples)

    validation_warnings = validate_selected_samples(selected_samples, records, args.prompt_variant, args.option_shuffle_seeds)
    metadata = build_metadata(raw_samples, args, records)
    write_json(run_dir / "run_metadata.json", metadata)

    if args.prompt_variant == "option_shuffle":
        stats, by_type_rows, letter_rows = option_shuffle_stats(records)
        write_csv(
            run_dir / "stats_by_question_type.csv",
            by_type_rows,
            [
                "question_type",
                "samples",
                "base_samples",
                "parsed_predictions",
                "parse_success_rate",
                "accuracy_presented",
                "accuracy_original",
                "accuracy_presented_parsed",
                "accuracy_original_parsed",
                "semantic_consistency_rate",
                "correctness_flip_rate",
                "avg_unique_semantic_answers_per_sample",
            ],
        )
        write_csv(run_dir / "letter_bias.csv", letter_rows, ["letter", "model_count", "model_frequency", "ground_truth_count", "ground_truth_frequency", "bias"])
    else:
        stats, by_type_rows = evidence_json_stats(records)
        write_csv(
            run_dir / "stats_by_question_type.csv",
            by_type_rows,
            [
                "question_type",
                "samples",
                "json_parse_success_rate",
                "answer_parse_success_rate",
                "mca_samples",
                "mca_accuracy",
                "mca_accuracy_answer_parsed",
                "numeric_samples",
                "numeric_mra_mean",
                "numeric_within_10pct_rate",
                "numeric_within_25pct_rate",
                "open_ended_samples",
                "open_ended_normalized_match_rate",
            ],
        )

    stats.update(
        {
            "run_name": args.run_name,
            "num_requested_samples": args.num_samples,
            "sample_seed": args.sample_seed,
            "option_shuffle_seeds": args.option_shuffle_seeds,
            "raw_lmms_eval_file": str(raw_sample_file),
            "selected_samples_reconstructed_from_predictions": bool(selected_samples.get("reconstructed_from_predictions")),
            "selected_samples_validation_warnings": validation_warnings,
        }
    )
    write_json(run_dir / "stats.json", stats)
    print(f"Wrote VSiBench probe analysis to {run_dir}")


if __name__ == "__main__":
    main()
