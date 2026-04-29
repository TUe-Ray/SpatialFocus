#!/usr/bin/env python3
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
UTILS_PATH = REPO_ROOT / "thinking-in-space/lmms_eval/tasks/vsibench_probe/utils.py"


def load_probe_utils():
    spec = importlib.util.spec_from_file_location("vsibench_probe_utils", UTILS_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


utils = load_probe_utils()


def load_script_module(name, relative_path):
    path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_cmd(args, check=True):
    result = subprocess.run([sys.executable, *args], cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if check and result.returncode != 0:
        raise AssertionError(f"Command failed: {' '.join(args)}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
    return result


def write_raw_run(run_dir, records, selected_samples):
    raw_dir = run_dir / "raw_lmms_eval" / "nested"
    raw_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "args": {"model": "fake", "model_args": "pretrained=fake-model"},
        "model_configs": {"generation_kwargs": {"max_new_tokens": 16}},
        "logs": [{"doc_id": idx, "vsibench_probe_score": record} for idx, record in enumerate(records)],
        "time": "smoke",
    }
    (raw_dir / "vsibench_probe.json").write_text(json.dumps(payload), encoding="utf-8")
    (run_dir / "selected_samples.json").write_text(json.dumps(selected_samples, indent=2), encoding="utf-8")


def option_doc(sample_id, probe_index, seed, gt_letter="A", question_type="object_rel_direction_hard"):
    original = {"A": "table", "B": "sofa", "C": "bed", "D": "TV"}
    presented, presented_to_original, original_to_presented = utils.shuffle_options_for_sample(original, sample_id, seed)
    return {
        "prompt_variant": "option_shuffle",
        "sample_id": sample_id,
        "probe_index": probe_index,
        "dataset_index": probe_index,
        "scene_name": f"scene_{probe_index}",
        "question_type": question_type,
        "question": f"Question {probe_index}?",
        "options": utils.format_options(presented),
        "original_options": original,
        "presented_options": presented,
        "presented_to_original": presented_to_original,
        "original_to_presented": original_to_presented,
        "option_shuffle_seed": seed,
        "gt_original_letter": gt_letter,
        "gt_presented_letter": original_to_presented[gt_letter],
        "gt_answer_text": original[gt_letter],
    }


def presented_for_original(doc, original_letter):
    return doc["original_to_presented"][original_letter]


def wrong_presented(doc, preferred_wrong_original="C"):
    wrong_original = preferred_wrong_original
    if wrong_original == doc["gt_original_letter"]:
        wrong_original = next(letter for letter in ["A", "B", "C", "D"] if letter != doc["gt_original_letter"])
    return presented_for_original(doc, wrong_original)


def selected(sample_ids, seeds=None, prompt_variant="option_shuffle", sentinel=True):
    payload = {
        "dataset": "vsibench",
        "prompt_variant": prompt_variant,
        "num_samples": len(sample_ids),
        "actual_num_samples": len(sample_ids),
        "sample_seed": 42,
        "option_shuffle_seeds": seeds if prompt_variant == "option_shuffle" else None,
        "sample_order": [
            {"probe_index": idx, "sample_id": sample_id, "dataset_index": idx, "scene_name": f"scene_{idx}", "question_type": "object_rel_direction_hard"}
            for idx, sample_id in enumerate(sample_ids)
        ],
    }
    if sentinel:
        payload["sentinel"] = "do_not_overwrite"
    return payload


def test_deterministic_selection():
    dataset = [
        {"id": idx, "dataset": "d", "scene_name": f"s{idx}", "question_type": "object_rel_direction_hard", "question": f"q{idx}", "ground_truth": "A", "options": ["A. x", "B. y", "C. z", "D. w"]}
        for idx in range(10)
    ]
    first = utils.select_probe_rows(dataset, "option_shuffle", 5, 42)
    second = utils.select_probe_rows(dataset, "option_shuffle", 5, 42)
    assert first == second
    assert len(first) == 5


def test_option_shuffle_mapping_and_parser():
    original = {"A": "table", "B": "sofa", "C": "bed", "D": "TV"}
    shuffled = utils.shuffle_options_for_sample(original, "sample", 0)
    assert shuffled == utils.shuffle_options_for_sample(original, "sample", 0)
    presented_options, presented_to_original, _ = shuffled
    assert list(presented_to_original.keys()) == ["A", "B", "C", "D"]
    assert list(presented_to_original.values()) != ["A", "B", "C", "D"]
    assert utils.parse_option_letter("A/B/C/D")[0] is None
    assert utils.parse_option_letter("A, B, C, or D")[0] is None
    assert utils.parse_option_letter("Answer: A/B/C/D")[0] is None
    assert utils.parse_option_letter("Final answer: A/B/C/D")[0] is None
    assert utils.parse_option_letter("Answer: A, B, C, or D")[0] is None
    assert utils.parse_option_letter("Final answer: C")[0] == "C"


def test_evidence_answer_and_numeric_parsing():
    doc = {
        "prompt_variant": "evidence_json",
        "sample_id": "s0",
        "probe_index": 0,
        "dataset_index": 0,
        "scene_name": "scene",
        "question_type": "object_rel_direction_hard",
        "question": "Which object?",
        "options": ["A. table", "B. sofa", "C. bed", "D. TV"],
        "original_options": {"A": "table", "B": "sofa", "C": "bed", "D": "TV"},
        "gt_original_letter": "B",
        "gt_answer_text": "sofa",
    }
    placeholder = utils._process_evidence_json(doc, '{"answer": "A/B/C/D"}')
    assert placeholder["json_parse_ok"] is True
    assert placeholder["answer_parse_ok"] is False
    valid = utils._process_evidence_json(doc, '{"answer": "B"}')
    assert valid["answer_parse_ok"] is True and valid["model_answer"] == "B"
    raw = utils._process_evidence_json(doc, "Final answer: B")
    assert raw["json_parse_ok"] is False and raw["answer_parse_ok"] is True and raw["correct"] is True

    numeric_doc = {"prompt_variant": "evidence_json", "sample_id": "n0", "question_type": "object_counting", "question": "How many?", "options": [], "ground_truth": "10"}
    numeric = utils._process_evidence_json(numeric_doc, "about 11")
    assert numeric["numeric_pred"] == 11.0
    assert numeric["numeric_target"] == 10.0
    assert numeric["numeric_within_10pct"] is True
    assert numeric["numeric_mra"] is not None


def test_analyzer_report_and_compare():
    with tempfile.TemporaryDirectory(prefix="vsibench_probe_tests_") as tmp:
        root = Path(tmp)
        base = root / "base"
        new = root / "new"
        base.mkdir()
        new.mkdir()
        base_records = []
        new_records = []
        for sample_id, probe_index, gt_letter in [("s0", 0, "A"), ("s1", 1, "B")]:
            for seed in [0, 1]:
                base_records.append(utils._process_option_shuffle(option_doc(sample_id, probe_index, seed, gt_letter), option_doc(sample_id, probe_index, seed, gt_letter)["gt_presented_letter"]))
                new_answer = "A" if sample_id == "s1" and seed == 1 else option_doc(sample_id, probe_index, seed, gt_letter)["gt_presented_letter"]
                new_records.append(utils._process_option_shuffle(option_doc(sample_id, probe_index, seed, gt_letter), new_answer))

        write_raw_run(base, base_records, selected(["s0", "s1"], [0, 1]))
        write_raw_run(new, new_records, selected(["s1", "s0"], [0, 1]))

        run_cmd(["scripts/analyze_vsibench_probe.py", "--run-dir", str(base)])
        after = json.loads((base / "selected_samples.json").read_text())
        assert after["sentinel"] == "do_not_overwrite"
        stats = json.loads((base / "stats.json").read_text())
        assert stats["semantic_consistency_rate"] == 1.0
        assert stats["robustness_summary"]["base_samples"] == 2
        assert (base / "sample_robustness.jsonl").exists()
        assert (base / "sample_robustness.csv").exists()
        assert (base / "robustness_by_question_type.csv").exists()
        assert "probe_uid" in (base / "predictions.jsonl").read_text()
        run_cmd(["scripts/generate_vsibench_probe_report.py", "--run-dir", str(base)])
        base_report = (base / "report.md").read_text()
        assert "Base samples" in base_report
        assert "Evaluated rows" in base_report
        assert "Presented options:" in base_report
        assert "Original options:" in base_report
        assert "Ground-truth presented answer:" in base_report
        assert "Ground-truth original answer:" in base_report
        assert "Raw presented answer:" in base_report
        assert "Mapped original answer:" in base_report

        run_cmd(["scripts/analyze_vsibench_probe.py", "--run-dir", str(new)])
        failed = run_cmd(["scripts/compare_vsibench_probe_runs.py", "--runs", str(base), str(new), "--output", str(root / "compare_fail")], check=False)
        assert failed.returncode != 0
        run_cmd(["scripts/compare_vsibench_probe_runs.py", "--runs", str(base), str(new), "--output", str(root / "compare_ok"), "--allow-mismatch"])

        option_report_run = root / "option_report"
        option_report_run.mkdir()
        option_report_records = []
        correct_doc = option_doc("fmt_s1", 1, 0, "B")
        option_report_records.append(utils._process_option_shuffle(correct_doc, correct_doc["gt_presented_letter"]))
        for seed in [0, 1, 2]:
            inconsistent_doc = option_doc("fmt_s0", 0, seed, "A")
            if seed == 0:
                prediction = inconsistent_doc["gt_presented_letter"]
            elif seed == 1:
                prediction = next(letter for letter in ["A", "B", "C", "D"] if letter != inconsistent_doc["gt_presented_letter"])
            else:
                prediction = "A/B/C/D"
            option_report_records.append(utils._process_option_shuffle(inconsistent_doc, prediction))
        for seed in [1, 2]:
            extra_correct_doc = option_doc("fmt_s1", 1, seed, "B")
            option_report_records.append(utils._process_option_shuffle(extra_correct_doc, extra_correct_doc["gt_presented_letter"]))

        write_raw_run(option_report_run, option_report_records, selected(["fmt_s0", "fmt_s1"], [0, 1, 2]))
        run_cmd(["scripts/analyze_vsibench_probe.py", "--run-dir", str(option_report_run)])
        run_cmd(["scripts/generate_vsibench_probe_report.py", "--run-dir", str(option_report_run)])
        option_report = (option_report_run / "report.md").read_text()
        assert "| Base samples | 2 |" in option_report
        assert "| Evaluated rows | 6 |" in option_report
        assert "### Wrong examples with raw output" in option_report
        assert "### Parse failure examples" in option_report
        assert "### Option-shuffle inconsistent examples" in option_report
        correct_section = option_report.split("### Correct examples", 1)[1].split("### Wrong examples with raw output", 1)[0]
        wrong_section = option_report.split("### Wrong examples with raw output", 1)[1].split("### Parse failure examples", 1)[0]
        assert correct_section.count("Question 1?") <= 1
        assert wrong_section.count("Question 0?") <= 1
        assert "Raw presented answer:" in option_report
        assert "Mapped original answer:" in option_report
        assert "Ground-truth presented answer:" in option_report
        assert "Ground-truth original answer:" in option_report
        assert "Raw model output: `A/B/C/D`" in option_report
        assert "Seed 0:" in option_report
        assert "Seed 1:" in option_report
        assert "Seed 2:" in option_report

        evidence = root / "evidence"
        evidence.mkdir()
        evidence_doc = {
            "prompt_variant": "evidence_json",
            "sample_id": "e0",
            "probe_index": 0,
            "dataset_index": 0,
            "scene_name": "scene",
            "question_type": "object_rel_direction_hard",
            "question": "Which object?",
            "options": ["A. table", "B. sofa", "C. bed", "D. TV"],
            "original_options": {"A": "table", "B": "sofa", "C": "bed", "D": "TV"},
            "gt_original_letter": "B",
            "gt_answer_text": "sofa",
        }
        evidence_records = [
            utils._process_evidence_json(evidence_doc, "Final answer: B"),
            utils._process_evidence_json({**evidence_doc, "sample_id": "e1", "probe_index": 1, "dataset_index": 1}, '{"answer": "A/B/C/D"}'),
        ]
        write_raw_run(evidence, evidence_records, selected(["e0", "e1"], None, prompt_variant="evidence_json"))
        run_cmd(["scripts/analyze_vsibench_probe.py", "--run-dir", str(evidence)])
        run_cmd(["scripts/generate_vsibench_probe_report.py", "--run-dir", str(evidence)])
        report = (evidence / "report.md").read_text()
        assert "Parse failure examples" in report
        assert "Invalid JSON but answer parsed examples" in report

        compare_base = root / "compare_evidence_base"
        compare_new = root / "compare_evidence_new"
        compare_base.mkdir()
        compare_new.mkdir()
        compare_base_records = [
            {
                "probe_uid": "mca::evidence_json",
                "sample_id": "mca",
                "prompt_variant": "evidence_json",
                "question_type": "object_rel_direction_hard",
                "question": "Which object?",
                "answer_parse_ok": True,
                "correct": True,
                "is_mca": True,
            },
            {
                "probe_uid": "num::evidence_json",
                "sample_id": "num",
                "prompt_variant": "evidence_json",
                "question_type": "object_counting",
                "question": "How many?",
                "answer_parse_ok": True,
                "correct": None,
                "is_numeric": True,
                "numeric_mra": 0.2,
            },
        ]
        compare_new_records = [
            {
                **compare_base_records[0],
                "correct": False,
            },
            {
                **compare_base_records[1],
                "numeric_mra": 0.8,
            },
        ]
        write_raw_run(compare_base, compare_base_records, selected(["mca", "num"], None, prompt_variant="evidence_json"))
        write_raw_run(compare_new, compare_new_records, selected(["mca", "num"], None, prompt_variant="evidence_json"))
        run_cmd(["scripts/analyze_vsibench_probe.py", "--run-dir", str(compare_base)])
        run_cmd(["scripts/analyze_vsibench_probe.py", "--run-dir", str(compare_new)])
        run_cmd(["scripts/compare_vsibench_probe_runs.py", "--runs", str(compare_base), str(compare_new), "--output", str(root / "compare_evidence")])
        compare_stats = json.loads((root / "compare_evidence" / "stats.json").read_text())
        regressed = compare_stats["comparisons"][0]["question_types_regressed"]
        improved = compare_stats["comparisons"][0]["question_types_improved"]
        regressed_types = {row["question_type"] for row in regressed}
        improved_types = {row["question_type"] for row in improved}
        assert "object_rel_direction_hard" in regressed_types
        assert "object_counting" in improved_types


def test_per_sample_robustness_categories():
    with tempfile.TemporaryDirectory(prefix="vsibench_probe_robust_") as tmp:
        root = Path(tmp)
        run = root / "robust"
        run.mkdir()
        records = []
        samples = [
            ("always_correct", "A"),
            ("flip", "A"),
            ("same_wrong", "A"),
            ("wrong_inconsistent", "A"),
            ("parse_failure", "A"),
        ]
        for probe_index, (sample_id, gt_letter) in enumerate(samples):
            for seed in [0, 1]:
                doc = option_doc(sample_id, probe_index, seed, gt_letter)
                if sample_id == "always_correct":
                    prediction = doc["gt_presented_letter"]
                elif sample_id == "flip":
                    prediction = doc["gt_presented_letter"] if seed == 0 else wrong_presented(doc, "C")
                elif sample_id == "same_wrong":
                    prediction = wrong_presented(doc, "C")
                elif sample_id == "wrong_inconsistent":
                    prediction = wrong_presented(doc, "C" if seed == 0 else "D")
                else:
                    prediction = "A/B/C/D" if seed == 0 else doc["gt_presented_letter"]
                records.append(utils._process_option_shuffle(doc, prediction))

        write_raw_run(run, records, selected([item[0] for item in samples], [0, 1]))
        run_cmd(["scripts/analyze_vsibench_probe.py", "--run-dir", str(run)])
        robustness = [json.loads(line) for line in (run / "sample_robustness.jsonl").read_text().splitlines() if line.strip()]
        by_sample = {row["sample_id"]: row for row in robustness}
        assert by_sample["always_correct"]["primary_robustness_category"] == "always_correct"
        assert by_sample["flip"]["primary_robustness_category"] == "flip"
        assert by_sample["same_wrong"]["primary_robustness_category"] == "always_same_wrong_answer"
        assert by_sample["wrong_inconsistent"]["primary_robustness_category"] == "always_wrong_inconsistent"
        assert by_sample["parse_failure"]["primary_robustness_category"] == "parse_failure_or_incomplete"
        assert by_sample["same_wrong"]["always_wrong"] is True
        assert by_sample["same_wrong"]["always_same_wrong_answer"] is True
        stats = json.loads((run / "stats.json").read_text())
        summary = stats["robustness_summary"]
        total = sum(summary[f"{category}_count"] for category in ["always_correct", "flip", "always_same_wrong_answer", "always_wrong_inconsistent", "parse_failure_or_incomplete"])
        assert total == summary["base_samples"] == 5


def test_paired_win_loss_helpers():
    with tempfile.TemporaryDirectory(prefix="vsibench_probe_pair_") as tmp:
        root = Path(tmp)
        base = root / "zero_spatial"
        new = root / "reproduction"
        base.mkdir()
        new.mkdir()
        base_records = []
        new_records = []
        specs = [("p0", 0, "A"), ("p1", 1, "B"), ("p2", 2, "A")]
        for sample_id, probe_index, gt_letter in specs:
            for seed in [0, 1]:
                doc = option_doc(sample_id, probe_index, seed, gt_letter)
                if sample_id == "p0":
                    base_pred = wrong_presented(doc, "C")
                    new_pred = doc["gt_presented_letter"] if seed == 0 else wrong_presented(doc, "C")
                elif sample_id == "p1":
                    base_pred = doc["gt_presented_letter"]
                    new_pred = wrong_presented(doc, "C") if seed == 0 else doc["gt_presented_letter"]
                else:
                    base_pred = doc["gt_presented_letter"] if seed == 0 else wrong_presented(doc, "C")
                    new_pred = doc["gt_presented_letter"] if seed == 0 else wrong_presented(doc, "C")
                base_records.append(utils._process_option_shuffle(doc, base_pred))
                new_records.append(utils._process_option_shuffle(doc, new_pred))

        same_selected = selected([item[0] for item in specs], [0, 1])
        write_raw_run(base, base_records, same_selected)
        write_raw_run(new, new_records, same_selected)
        run_cmd(["scripts/analyze_vsibench_probe.py", "--run-dir", str(base), "--run-name", "zero_spatial"])
        run_cmd(["scripts/analyze_vsibench_probe.py", "--run-dir", str(new), "--run-name", "reproduction"])
        out = root / "compare"
        run_cmd(["scripts/compare_vsibench_probe_runs.py", "--runs", str(base), str(new), "--output", str(out)])
        stats = json.loads((out / "stats.json").read_text())
        overall = stats["paired_row_win_loss_overall"]
        assert overall["baseline_wrong_new_correct"] == 1
        assert overall["baseline_correct_new_wrong"] == 1
        assert overall["both_correct"] == 2
        assert overall["both_wrong"] == 2
        assert overall["net_gain"] == 0
        sample_overall = stats["paired_sample_win_loss_overall"]
        assert sample_overall["improved_samples"] == 1
        assert sample_overall["regressed_samples"] == 1
        assert sample_overall["unchanged_samples"] == 1
        assert (out / "paired_win_loss_by_question_type_rows.csv").exists()
        assert (out / "paired_win_loss_by_question_type_samples.csv").exists()
        assert "Zero wrong -> Repro correct" in (out / "report.md").read_text()


def test_bias_script_tiny_mocked_dataset():
    with tempfile.TemporaryDirectory(prefix="vsibench_bias_") as tmp:
        root = Path(tmp)
        data_path = root / "mock.jsonl"
        rows = [
            {"id": "0", "dataset": "mock", "question_type": "object_rel_direction_hard", "question": "Where?", "options": ["A. left", "B. right", "C. front", "D. back"], "ground_truth": "A"},
            {"id": "1", "dataset": "mock", "question_type": "object_rel_direction_hard", "question": "Where?", "options": ["A. right", "B. left", "C. front", "D. back"], "ground_truth": "left"},
            {"id": "2", "dataset": "mock", "question_type": "object_rel_distance", "question": "Which?", "options": ["A. chair", "B. table", "C. sofa", "D. bed"], "ground_truth": "C"},
            {"id": "3", "dataset": "mock", "question_type": "object_counting", "question": "How many?", "options": [], "ground_truth": "2"},
        ]
        data_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        out = root / "out"
        run_cmd(["scripts/analyze_vsibench_option_bias.py", "--local-dataset-path", str(data_path), "--split", "train", "--output", str(out)])
        for name in [
            "gt_letter_distribution.csv",
            "gt_letter_by_question_type.csv",
            "option_text_by_letter.csv",
            "answer_text_distribution.csv",
            "answer_text_by_question_type.csv",
            "question_type_distribution.csv",
            "spatial_answer_bias.csv",
            "stats.json",
            "report.md",
        ]:
            assert (out / name).exists()
        stats = json.loads((out / "stats.json").read_text())
        assert stats["mca_rows"] == 3


def test_compare_selected_samples_allow_mismatch_gate():
    compare = load_script_module("compare_vsibench_probe_runs_for_test", "scripts/compare_vsibench_probe_runs.py")
    baseline = {"selected_samples": selected(["a"], [0]), "metadata": {}, "stats": {}}
    new = {"selected_samples": selected(["b"], [0]), "metadata": {}, "stats": {}}
    try:
        compare.compare_selected_samples_or_raise(baseline, new)
    except ValueError as exc:
        assert "sample_order differs" in str(exc)
    else:
        raise AssertionError("Expected selected sample mismatch to fail")
    mismatches = compare.compare_selected_samples_or_raise(baseline, new, allow_mismatch=True)
    assert "sample_order differs" in mismatches


def main():
    test_deterministic_selection()
    test_option_shuffle_mapping_and_parser()
    test_evidence_answer_and_numeric_parsing()
    test_analyzer_report_and_compare()
    test_per_sample_robustness_categories()
    test_paired_win_loss_helpers()
    test_bias_script_tiny_mocked_dataset()
    test_compare_selected_samples_allow_mismatch_gate()
    print("VSiBench probe helper tests passed")


if __name__ == "__main__":
    main()
