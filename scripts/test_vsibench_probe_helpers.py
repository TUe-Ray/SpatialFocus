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
        assert "probe_uid" in (base / "predictions.jsonl").read_text()
        run_cmd(["scripts/generate_vsibench_probe_report.py", "--run-dir", str(base)])
        base_report = (base / "report.md").read_text()
        assert "Base samples" in base_report
        assert "Evaluated rows" in base_report

        run_cmd(["scripts/analyze_vsibench_probe.py", "--run-dir", str(new)])
        failed = run_cmd(["scripts/compare_vsibench_probe_runs.py", "--runs", str(base), str(new), "--output", str(root / "compare_fail")], check=False)
        assert failed.returncode != 0
        run_cmd(["scripts/compare_vsibench_probe_runs.py", "--runs", str(base), str(new), "--output", str(root / "compare_ok"), "--allow-mismatch"])

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


def main():
    test_deterministic_selection()
    test_option_shuffle_mapping_and_parser()
    test_evidence_answer_and_numeric_parsing()
    test_analyzer_report_and_compare()
    print("VSiBench probe helper tests passed")


if __name__ == "__main__":
    main()
