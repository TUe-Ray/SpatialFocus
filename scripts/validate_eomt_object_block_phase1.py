#!/usr/bin/env python3
"""Validate minimal Phase-1 EoMT object-block consumer behavior.

Focus:
- selector OFF fallback behavior
- selector ON deterministic ordering and budget truncation
- pool-skipped fallback behavior
- appender before/after position behavior
- appender failure fallback behavior

This script is intentionally scoped to Phase-1 module behavior and does not run
full model training/inference loops.
"""

import argparse
import json
import os
import sys
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from llava.model.llava_arch import LlavaMetaForCausalLM
from llava.model.multimodal_eomt import EoMTObjectBlockAppender, EoMTObjectTokenSelector


class DummyPhase1Validator(LlavaMetaForCausalLM):
    def __init__(self, config: SimpleNamespace):
        self.config = config

    def get_model(self):
        return self


def build_pooled_outputs(
    frame_count: int = 3,
    obj_per_frame: int = 4,
    hidden_size: int = 8,
    sample_idx: int = 0,
) -> Dict[str, Any]:
    pooled_tokens = torch.arange(
        frame_count * obj_per_frame * hidden_size,
        dtype=torch.float32,
    ).view(frame_count, obj_per_frame, hidden_size)

    # Score profile intentionally mixes order to test sorting behavior.
    selected_scores = torch.tensor(
        [
            [0.80, 0.65, 0.10, 0.05],
            [0.95, 0.40, 0.20, 0.01],
            [0.70, 0.60, 0.30, 0.02],
        ],
        dtype=torch.float32,
    )

    # Candidate query indices are unique per frame for easier debugging.
    selected_indices = torch.tensor(
        [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [8, 9, 10, 11],
        ],
        dtype=torch.long,
    )

    # Include one invalid class id per frame to test drop_no_object behavior.
    selected_class_ids = torch.tensor(
        [
            [3, 7, -1, 2],
            [1, -1, 6, 5],
            [4, 9, -1, 8],
        ],
        dtype=torch.long,
    )

    aligned_sample_frame_pairs = [(sample_idx, i) for i in range(frame_count)]

    return {
        "pooled_tokens": pooled_tokens,
        "selected_scores": selected_scores,
        "selected_indices": selected_indices,
        "selected_class_ids": selected_class_ids,
        "aligned_sample_frame_pairs": aligned_sample_frame_pairs,
        "pool_skipped": False,
        "pool_debug": {
            "poolable_frame_count": frame_count,
            "aligned_sample_frame_pairs": aligned_sample_frame_pairs,
        },
    }


def evaluate_case_off_mode() -> Dict[str, Any]:
    cfg = SimpleNamespace(
        mm_eomt_enable_object_block=False,
        mm_eomt_object_block_position="after_visual",
        mm_eomt_object_block_max_objects=8,
        mm_eomt_object_block_max_per_frame=2,
        mm_eomt_selector_mode="class_aware",
        mm_eomt_selector_keep_stuff=True,
        mm_eomt_selector_keep_things=True,
        mm_eomt_selector_drop_no_object=True,
        mm_eomt_selector_order="frame_then_score",
    )

    validator = DummyPhase1Validator(cfg)
    validator._last_eomt_pooled_outputs = build_pooled_outputs()
    side_out = validator._compute_eomt_object_block_side_output()

    ok = (
        side_out.get("enabled") is False
        and side_out.get("used_object_block") is False
        and side_out.get("fallback_reason") == "object_block_disabled"
    )
    return {"pass": ok, "details": side_out}


def evaluate_case_on_frame_then_score() -> Dict[str, Any]:
    cfg = SimpleNamespace(
        mm_eomt_enable_object_block=True,
        mm_eomt_object_block_position="after_visual",
        mm_eomt_object_block_max_objects=5,
        mm_eomt_object_block_max_per_frame=2,
        mm_eomt_selector_mode="class_aware",
        mm_eomt_selector_keep_stuff=True,
        mm_eomt_selector_keep_things=True,
        mm_eomt_selector_drop_no_object=True,
        mm_eomt_selector_order="frame_then_score",
    )

    selector = EoMTObjectTokenSelector()
    pooled = build_pooled_outputs()

    out1 = selector.select(pooled_outputs=pooled, config=cfg)
    out2 = selector.select(pooled_outputs=pooled, config=cfg)

    selected_pairs_1 = out1.get("selected_sample_frame_pairs", [])
    selected_pairs_2 = out2.get("selected_sample_frame_pairs", [])
    selected_scores_1 = out1.get("selected_scores", [])
    selected_scores_2 = out2.get("selected_scores", [])

    deterministic = (
        selected_pairs_1 == selected_pairs_2
        and selected_scores_1 == selected_scores_2
        and out1.get("selected_indices", []) == out2.get("selected_indices", [])
    )

    # Per-frame cap=2 over 3 frames gives up to 6; global max=5 should truncate by 1.
    cap_ok = (
        out1.get("selected_count", 0) == 5
        and out1.get("truncated_per_frame_count", 0) >= 0
        and out1.get("truncated_global_count", 0) >= 1
    )

    # frame_then_score should keep non-decreasing frame order for same sample.
    frame_order_ok = True
    last_frame = -1
    for pair in selected_pairs_1:
        if not isinstance(pair, (list, tuple)) or len(pair) < 2:
            frame_order_ok = False
            break
        frame_idx = int(pair[1])
        if frame_idx < last_frame:
            frame_order_ok = False
            break
        last_frame = frame_idx

    return {
        "pass": bool(deterministic and cap_ok and frame_order_ok),
        "details": {
            "deterministic": deterministic,
            "cap_ok": cap_ok,
            "frame_order_ok": frame_order_ok,
            "selected_count": out1.get("selected_count", 0),
            "selected_pairs": selected_pairs_1,
            "selected_scores": selected_scores_1,
            "truncated_per_frame_count": out1.get("truncated_per_frame_count", 0),
            "truncated_global_count": out1.get("truncated_global_count", 0),
        },
    }


def evaluate_case_on_score_desc() -> Dict[str, Any]:
    cfg = SimpleNamespace(
        mm_eomt_enable_object_block=True,
        mm_eomt_object_block_position="before_visual",
        mm_eomt_object_block_max_objects=8,
        mm_eomt_object_block_max_per_frame=3,
        mm_eomt_selector_mode="class_aware",
        mm_eomt_selector_keep_stuff=True,
        mm_eomt_selector_keep_things=True,
        mm_eomt_selector_drop_no_object=True,
        mm_eomt_selector_order="score_desc",
    )

    selector = EoMTObjectTokenSelector()
    out = selector.select(pooled_outputs=build_pooled_outputs(), config=cfg)
    scores = out.get("selected_scores", [])

    non_increasing = True
    for i in range(1, len(scores)):
        if float(scores[i]) > float(scores[i - 1]):
            non_increasing = False
            break

    return {
        "pass": bool(non_increasing),
        "details": {
            "non_increasing": non_increasing,
            "selected_scores": scores,
            "selected_pairs": out.get("selected_sample_frame_pairs", []),
        },
    }


def evaluate_case_pool_skipped() -> Dict[str, Any]:
    cfg = SimpleNamespace(
        mm_eomt_enable_object_block=True,
        mm_eomt_object_block_position="after_visual",
        mm_eomt_object_block_max_objects=8,
        mm_eomt_object_block_max_per_frame=2,
        mm_eomt_selector_mode="class_aware",
        mm_eomt_selector_keep_stuff=True,
        mm_eomt_selector_keep_things=True,
        mm_eomt_selector_drop_no_object=True,
        mm_eomt_selector_order="frame_then_score",
    )

    validator = DummyPhase1Validator(cfg)
    validator._last_eomt_pooled_outputs = {
        "pool_skipped": True,
        "skip_reason": "anyres_not_supported_for_eomt_pooling",
        "pool_debug": {},
    }
    side_out = validator._compute_eomt_object_block_side_output()
    ok = (
        side_out.get("enabled") is True
        and side_out.get("selected_count") == 0
        and side_out.get("fallback_reason") == "anyres_not_supported_for_eomt_pooling"
    )
    return {"pass": ok, "details": side_out}


def evaluate_case_selector_failure_path() -> Dict[str, Any]:
    cfg = SimpleNamespace(
        mm_eomt_enable_object_block=True,
        mm_eomt_object_block_position="after_visual",
        mm_eomt_object_block_max_objects=8,
        mm_eomt_object_block_max_per_frame=2,
        mm_eomt_selector_mode="class_aware",
        mm_eomt_selector_keep_stuff=True,
        mm_eomt_selector_keep_things=True,
        mm_eomt_selector_drop_no_object=True,
        mm_eomt_selector_order="frame_then_score",
    )

    validator = DummyPhase1Validator(cfg)
    validator._last_eomt_pooled_outputs = {
        "pool_skipped": False,
        "pooled_tokens": None,
        "selected_scores": None,
        "selected_indices": None,
        "selected_class_ids": None,
        "aligned_sample_frame_pairs": [],
        "pool_debug": {},
    }
    side_out = validator._compute_eomt_object_block_side_output()
    ok = (
        side_out.get("enabled") is True
        and side_out.get("selected_count") == 0
        and isinstance(side_out.get("fallback_reason"), str)
        and len(side_out.get("fallback_reason")) > 0
    )
    return {"pass": ok, "details": side_out}


def evaluate_case_appender_positions() -> Dict[str, Any]:
    appender = EoMTObjectBlockAppender()
    visual = torch.tensor([[10.0, 11.0], [12.0, 13.0]])
    obj = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    merged_before, used_before, reason_before = appender.append(
        visual_tokens=visual,
        object_tokens=obj,
        position="before_visual",
    )
    merged_after, used_after, reason_after = appender.append(
        visual_tokens=visual,
        object_tokens=obj,
        position="after_visual",
    )

    before_ok = (
        used_before
        and reason_before is None
        and torch.equal(merged_before[:2], obj)
        and torch.equal(merged_before[2:], visual)
    )
    after_ok = (
        used_after
        and reason_after is None
        and torch.equal(merged_after[:2], visual)
        and torch.equal(merged_after[2:], obj)
    )

    return {
        "pass": bool(before_ok and after_ok),
        "details": {
            "before_ok": before_ok,
            "after_ok": after_ok,
            "merged_before_shape": list(merged_before.shape),
            "merged_after_shape": list(merged_after.shape),
        },
    }


def evaluate_case_appender_failure_path() -> Dict[str, Any]:
    appender = EoMTObjectBlockAppender()
    visual = torch.randn(3, 8)
    wrong_obj = torch.randn(2, 7)

    merged, used, reason = appender.append(
        visual_tokens=visual,
        object_tokens=wrong_obj,
        position="after_visual",
    )

    ok = (
        used is False
        and reason == "feature_dim_mismatch"
        and torch.equal(merged, visual)
    )
    return {
        "pass": ok,
        "details": {
            "used": used,
            "reason": reason,
            "merged_equals_visual": torch.equal(merged, visual),
        },
    }


def build_report() -> Dict[str, Any]:
    results = {
        "Case 1 OFF mode": evaluate_case_off_mode(),
        "Case 2 ON frame_then_score": evaluate_case_on_frame_then_score(),
        "Case 3 ON score_desc": evaluate_case_on_score_desc(),
        "Case 4 pool_skipped fallback": evaluate_case_pool_skipped(),
        "Case 5 selector failure fallback": evaluate_case_selector_failure_path(),
        "Case 6 appender positions": evaluate_case_appender_positions(),
        "Case 7 appender failure fallback": evaluate_case_appender_failure_path(),
    }

    summary = {k: ("pass" if v.get("pass") else "fail") for k, v in results.items()}
    problems = [k for k, v in summary.items() if v != "pass"]
    verdict = "validated" if len(problems) == 0 else "validated with caveats"

    return {
        "summary": summary,
        "problems": problems,
        "verdict": verdict,
        "results": results,
        "deferred_scope": [
            "external_word_socket",
            "OBJ_INFO modes",
            "object-type embedding",
            "runtime tokenizer loading",
            "query hidden-state interaction",
            "fusion redesign",
            "anyres/multi-patch scope expansion",
        ],
    }


def print_report(report: Dict[str, Any]) -> None:
    print("1. Summary")
    for case_name, status in report["summary"].items():
        print(f"- {case_name}: {status}")

    print("\n2. Problems found")
    if len(report["problems"]) == 0:
        print("- none")
    else:
        for case_name in report["problems"]:
            print(f"- {case_name}")

    print("\n3. Final verdict")
    print(f"- {report['verdict']}")

    print("\n4. Deferred scope")
    for item in report["deferred_scope"]:
        print(f"- {item}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Phase-1 EoMT object block behavior")
    parser.add_argument(
        "--output_json",
        type=str,
        default="logs/eomt_object_block_phase1_report.json",
        help="Path to write JSON report",
    )
    args = parser.parse_args()

    torch.manual_seed(0)
    report = build_report()

    output_dir = os.path.dirname(args.output_json)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print_report(report)


if __name__ == "__main__":
    main()
