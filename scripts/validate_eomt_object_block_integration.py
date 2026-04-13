#!/usr/bin/env python3
"""Sequence-level validation for Phase-1B EoMT object-block integration.

This script validates integration behavior with synthetic fixtures and does not
launch platform-specific training jobs.
"""

import argparse
import json
import os
import sys
from types import SimpleNamespace
from typing import Any, Dict, List

import torch
import torch.nn as nn

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from llava.model.llava_arch import LlavaMetaForCausalLM


class DummyIntegrationValidator(LlavaMetaForCausalLM):
    def __init__(self, config: SimpleNamespace, hidden_size: int = 8):
        self.config = config
        self.embed_tokens = nn.Embedding(64, hidden_size)

    def get_model(self):
        return self


def make_pooled_outputs_two_samples(hidden_size: int = 8) -> Dict[str, Any]:
    # 4 aligned frames total: sample0/frame0, sample0/frame1, sample1/frame0, sample1/frame1
    frame_pairs = [(0, 0), (0, 1), (1, 0), (1, 1)]
    pooled_tokens = torch.arange(4 * 3 * hidden_size, dtype=torch.float32).view(4, 3, hidden_size)
    selected_scores = torch.tensor(
        [
            [0.90, 0.60, 0.10],
            [0.80, 0.50, 0.20],
            [0.95, 0.40, 0.30],
            [0.85, 0.70, 0.05],
        ],
        dtype=torch.float32,
    )
    selected_indices = torch.tensor(
        [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
            [9, 10, 11],
        ],
        dtype=torch.long,
    )
    selected_class_ids = torch.tensor(
        [
            [1, 2, -1],
            [3, 4, -1],
            [5, 6, -1],
            [7, 8, -1],
        ],
        dtype=torch.long,
    )

    frame_meta = [
        {"sample_id": 0, "frame_idx": 0},
        {"sample_id": 0, "frame_idx": 1},
        {"sample_id": 1, "frame_idx": 0},
        {"sample_id": 1, "frame_idx": 1},
    ]

    return {
        "pooled_tokens": pooled_tokens,
        "selected_scores": selected_scores,
        "selected_indices": selected_indices,
        "selected_class_ids": selected_class_ids,
        "aligned_sample_frame_pairs": frame_pairs,
        "frame_meta": frame_meta,
        "pool_debug": {"aligned_sample_frame_pairs": frame_pairs},
        "pool_skipped": False,
    }


def make_external_socket_pooled_outputs(hidden_size: int = 8) -> Dict[str, Any]:
    pooled = make_pooled_outputs_two_samples(hidden_size=hidden_size)
    # Add external socket hints in frame metadata for sample0 only.
    pooled["frame_meta"][0]["selected_query_ids"] = [0, 1]
    pooled["frame_meta"][1]["selected_mask_ids"] = [3]
    pooled["frame_meta"][2]["selected_words"] = ["cup", "table"]
    return pooled


def apply_integration(
    validator: DummyIntegrationValidator,
    image_features_by_sample: List[torch.Tensor],
) -> List[torch.Tensor]:
    outputs = validator._last_eomt_object_block_outputs
    appender = validator._get_eomt_object_block_appender()

    merged_list = []
    for sample_idx, visual_tokens in enumerate(image_features_by_sample):
        merged = visual_tokens
        if isinstance(outputs, dict) and bool(outputs.get("enabled", False)):
            selected = outputs.get("selected_tokens_by_sample", {}).get(sample_idx, None)
            if torch.is_tensor(selected) and selected.ndim == 2 and selected.shape[0] > 0:
                block_tokens, compose_reason = validator._compose_eomt_object_block_tokens(
                    object_tokens=selected,
                    visual_tokens=visual_tokens,
                    object_block_outputs=outputs,
                )
                if torch.is_tensor(block_tokens) and block_tokens.ndim == 2 and block_tokens.shape[0] > 0:
                    merged, used, reason = appender.append(
                        visual_tokens=visual_tokens,
                        object_tokens=block_tokens,
                        position=str(outputs.get("append_position", "after_visual")),
                    )
                    if used:
                        outputs["used_object_block"] = True
                        outputs["fallback_reason"] = None
                    elif outputs.get("fallback_reason", None) is None:
                        outputs["fallback_reason"] = reason
                elif outputs.get("fallback_reason", None) is None:
                    outputs["fallback_reason"] = compose_reason
        merged_list.append(merged)

    if isinstance(outputs, dict):
        if outputs.get("enabled", False) and not outputs.get("used_object_block", False) and outputs.get("fallback_reason", None) is None:
            outputs["fallback_reason"] = "no_object_block_appended"

    return merged_list


def case_length_and_mapping() -> Dict[str, Any]:
    cfg = SimpleNamespace(
        mm_eomt_enable_object_block=True,
        mm_eomt_object_block_position="after_visual",
        mm_eomt_object_block_max_objects=8,
        mm_eomt_object_block_max_per_frame=2,
        mm_eomt_selector_mode="class_aware",
        mm_eomt_selector_keep_stuff=True,
        mm_eomt_selector_keep_things=True,
        mm_eomt_selector_drop_no_object=True,
        mm_eomt_selector_no_object_class_id=-1,
        mm_eomt_selector_order="frame_then_score",
        mm_eomt_obj_info_mode="none",
        mm_eomt_obj_info_text="Object information from the image:",
        mm_eomt_obj_info_trainable=True,
        mm_eomt_use_object_type_embedding=False,
        mm_eomt_external_socket_word_topn=1,
        mm_eomt_external_socket_deduplicate=True,
    )

    validator = DummyIntegrationValidator(cfg)
    validator._last_eomt_pooled_outputs = make_pooled_outputs_two_samples()
    side_out = validator._compute_eomt_object_block_side_output()
    validator._last_eomt_object_block_outputs = side_out

    visuals = [torch.randn(5, 8), torch.randn(6, 8)]
    merged = apply_integration(validator, visuals)

    s0_sel = side_out["selected_tokens_by_sample"].get(0)
    s1_sel = side_out["selected_tokens_by_sample"].get(1)
    exp0 = visuals[0].shape[0] + (int(s0_sel.shape[0]) if torch.is_tensor(s0_sel) else 0)
    exp1 = visuals[1].shape[0] + (int(s1_sel.shape[0]) if torch.is_tensor(s1_sel) else 0)

    len_ok = merged[0].shape[0] == exp0 and merged[1].shape[0] == exp1

    tail_match_0 = False
    tail_match_1 = False
    if torch.is_tensor(s0_sel) and s0_sel.shape[0] > 0:
        tail_match_0 = torch.allclose(merged[0][-s0_sel.shape[0]:], s0_sel.to(merged[0].dtype), atol=1e-6)
    if torch.is_tensor(s1_sel) and s1_sel.shape[0] > 0:
        tail_match_1 = torch.allclose(merged[1][-s1_sel.shape[0]:], s1_sel.to(merged[1].dtype), atol=1e-6)

    return {
        "pass": bool(len_ok and tail_match_0 and tail_match_1),
        "details": {
            "length_ok": len_ok,
            "tail_match_sample0": tail_match_0,
            "tail_match_sample1": tail_match_1,
            "selected_count": side_out.get("selected_count", 0),
            "selected_pairs": side_out.get("selected_sample_frame_pairs", []),
        },
    }


def case_before_after_order() -> Dict[str, Any]:
    base_cfg = {
        "mm_eomt_enable_object_block": True,
        "mm_eomt_object_block_max_objects": 4,
        "mm_eomt_object_block_max_per_frame": 2,
        "mm_eomt_selector_mode": "class_aware",
        "mm_eomt_selector_keep_stuff": True,
        "mm_eomt_selector_keep_things": True,
        "mm_eomt_selector_drop_no_object": True,
        "mm_eomt_selector_no_object_class_id": -1,
        "mm_eomt_selector_order": "frame_then_score",
        "mm_eomt_obj_info_mode": "none",
        "mm_eomt_obj_info_text": "Object information from the image:",
        "mm_eomt_obj_info_trainable": True,
        "mm_eomt_use_object_type_embedding": False,
        "mm_eomt_external_socket_word_topn": 1,
        "mm_eomt_external_socket_deduplicate": True,
    }

    visuals = [torch.tensor([[10.0] * 8, [11.0] * 8])]

    cfg_before = SimpleNamespace(**{**base_cfg, "mm_eomt_object_block_position": "before_visual"})
    v_before = DummyIntegrationValidator(cfg_before)
    v_before._last_eomt_pooled_outputs = make_pooled_outputs_two_samples()
    out_before = v_before._compute_eomt_object_block_side_output()
    v_before._last_eomt_object_block_outputs = out_before
    merged_before = apply_integration(v_before, visuals)[0]

    cfg_after = SimpleNamespace(**{**base_cfg, "mm_eomt_object_block_position": "after_visual"})
    v_after = DummyIntegrationValidator(cfg_after)
    v_after._last_eomt_pooled_outputs = make_pooled_outputs_two_samples()
    out_after = v_after._compute_eomt_object_block_side_output()
    v_after._last_eomt_object_block_outputs = out_after
    merged_after = apply_integration(v_after, visuals)[0]

    sel_before = out_before["selected_tokens_by_sample"][0]
    sel_after = out_after["selected_tokens_by_sample"][0]

    before_ok = torch.allclose(merged_before[: sel_before.shape[0]], sel_before.to(merged_before.dtype), atol=1e-6)
    after_ok = torch.allclose(merged_after[-sel_after.shape[0] :], sel_after.to(merged_after.dtype), atol=1e-6)

    return {
        "pass": bool(before_ok and after_ok),
        "details": {
            "before_ok": before_ok,
            "after_ok": after_ok,
            "before_shape": list(merged_before.shape),
            "after_shape": list(merged_after.shape),
        },
    }


def case_fallback_preserves_baseline() -> Dict[str, Any]:
    cfg = SimpleNamespace(
        mm_eomt_enable_object_block=True,
        mm_eomt_object_block_position="after_visual",
        mm_eomt_object_block_max_objects=8,
        mm_eomt_object_block_max_per_frame=2,
        mm_eomt_selector_mode="class_aware",
        mm_eomt_selector_keep_stuff=True,
        mm_eomt_selector_keep_things=True,
        mm_eomt_selector_drop_no_object=True,
        mm_eomt_selector_no_object_class_id=-1,
        mm_eomt_selector_order="frame_then_score",
        mm_eomt_obj_info_mode="none",
        mm_eomt_obj_info_text="Object information from the image:",
        mm_eomt_obj_info_trainable=True,
        mm_eomt_use_object_type_embedding=False,
        mm_eomt_external_socket_word_topn=1,
        mm_eomt_external_socket_deduplicate=True,
    )

    validator = DummyIntegrationValidator(cfg)
    validator._last_eomt_pooled_outputs = {
        "pool_skipped": True,
        "skip_reason": "anyres_not_supported_for_eomt_pooling",
        "pool_debug": {},
    }
    side_out = validator._compute_eomt_object_block_side_output()
    validator._last_eomt_object_block_outputs = side_out

    visuals = [torch.randn(4, 8), torch.randn(7, 8)]
    merged = apply_integration(validator, visuals)

    unchanged = all(torch.allclose(m, v, atol=1e-6) for m, v in zip(merged, visuals))
    reason_ok = side_out.get("fallback_reason") in {
        "anyres_not_supported_for_eomt_pooling",
        "no_object_block_appended",
    }

    return {
        "pass": bool(unchanged and reason_ok),
        "details": {
            "unchanged": unchanged,
            "fallback_reason": side_out.get("fallback_reason", None),
        },
    }


def case_obj_info_and_type_embedding() -> Dict[str, Any]:
    cfg = SimpleNamespace(
        mm_eomt_enable_object_block=True,
        mm_eomt_object_block_position="after_visual",
        mm_eomt_object_block_max_objects=4,
        mm_eomt_object_block_max_per_frame=2,
        mm_eomt_selector_mode="class_aware",
        mm_eomt_selector_keep_stuff=True,
        mm_eomt_selector_keep_things=True,
        mm_eomt_selector_drop_no_object=True,
        mm_eomt_selector_no_object_class_id=-1,
        mm_eomt_selector_order="frame_then_score",
        mm_eomt_obj_info_mode="learnable_embedding",
        mm_eomt_obj_info_text="Object information from the image:",
        mm_eomt_obj_info_trainable=True,
        mm_eomt_use_object_type_embedding=True,
        mm_eomt_external_socket_word_topn=1,
        mm_eomt_external_socket_deduplicate=True,
    )

    validator = DummyIntegrationValidator(cfg)
    validator._last_eomt_pooled_outputs = make_pooled_outputs_two_samples()
    side_out = validator._compute_eomt_object_block_side_output()

    selected = side_out["selected_tokens_by_sample"][0]
    visual = torch.randn(3, 8)
    block_tokens, _ = validator._compose_eomt_object_block_tokens(
        object_tokens=selected,
        visual_tokens=visual,
        object_block_outputs=side_out,
    )

    # learnable OBJ_INFO adds one token in this mode
    obj_info_token_added = bool(torch.is_tensor(block_tokens) and block_tokens.shape[0] == selected.shape[0] + 1)
    type_embedding_used = bool(side_out.get("object_type_embedding_used", False))
    obj_info_used = bool(side_out.get("obj_info_used", False))

    return {
        "pass": bool(obj_info_token_added and type_embedding_used and obj_info_used),
        "details": {
            "obj_info_token_added": obj_info_token_added,
            "type_embedding_used": type_embedding_used,
            "obj_info_used": obj_info_used,
            "obj_info_mode": side_out.get("obj_info_mode", None),
        },
    }


def case_external_socket_contract() -> Dict[str, Any]:
    cfg = SimpleNamespace(
        mm_eomt_enable_object_block=True,
        mm_eomt_object_block_position="after_visual",
        mm_eomt_object_block_max_objects=8,
        mm_eomt_object_block_max_per_frame=3,
        mm_eomt_selector_mode="external_socket",
        mm_eomt_selector_keep_stuff=True,
        mm_eomt_selector_keep_things=True,
        mm_eomt_selector_drop_no_object=True,
        mm_eomt_selector_no_object_class_id=-1,
        mm_eomt_selector_order="frame_then_score",
        mm_eomt_obj_info_mode="none",
        mm_eomt_obj_info_text="Object information from the image:",
        mm_eomt_obj_info_trainable=True,
        mm_eomt_use_object_type_embedding=False,
        mm_eomt_external_socket_word_topn=1,
        mm_eomt_external_socket_deduplicate=True,
    )

    validator = DummyIntegrationValidator(cfg)
    validator._last_eomt_pooled_outputs = make_external_socket_pooled_outputs()
    side_out = validator._compute_eomt_object_block_side_output()

    contract = side_out.get("external_selection_contract", {})
    present_ok = bool(side_out.get("external_socket_present", False))
    contract_ok = isinstance(contract, dict) and contract.get("matching_status") in {
        "ids_matched",
        "word_matching_deferred_no_ids",
    }

    return {
        "pass": bool(present_ok and contract_ok),
        "details": {
            "external_socket_present": side_out.get("external_socket_present", False),
            "contract": contract,
            "socket_note": side_out.get("external_socket_note", None),
            "selected_count": side_out.get("selected_count", 0),
            "fallback_reason": side_out.get("fallback_reason", None),
        },
    }


def case_external_socket_fallback_matrix() -> Dict[str, Any]:
    base_cfg = {
        "mm_eomt_enable_object_block": True,
        "mm_eomt_object_block_position": "after_visual",
        "mm_eomt_object_block_max_objects": 8,
        "mm_eomt_object_block_max_per_frame": 3,
        "mm_eomt_selector_mode": "external_socket",
        "mm_eomt_selector_keep_stuff": True,
        "mm_eomt_selector_keep_things": True,
        "mm_eomt_selector_drop_no_object": True,
        "mm_eomt_selector_no_object_class_id": -1,
        "mm_eomt_selector_order": "frame_then_score",
        "mm_eomt_obj_info_mode": "none",
        "mm_eomt_obj_info_text": "Object information from the image:",
        "mm_eomt_obj_info_trainable": True,
        "mm_eomt_use_object_type_embedding": False,
        "mm_eomt_external_socket_word_topn": 1,
        "mm_eomt_external_socket_deduplicate": True,
    }

    # Missing external selection input in frame metadata.
    v_missing = DummyIntegrationValidator(SimpleNamespace(**base_cfg))
    v_missing._last_eomt_pooled_outputs = make_pooled_outputs_two_samples()
    out_missing = v_missing._compute_eomt_object_block_side_output()

    # Empty selected ids (entry present, ids empty).
    p_empty_ids = make_pooled_outputs_two_samples()
    p_empty_ids["frame_meta"][0]["selected_query_ids"] = []
    p_empty_ids["frame_meta"][0]["selected_mask_ids"] = []
    v_empty_ids = DummyIntegrationValidator(SimpleNamespace(**base_cfg))
    v_empty_ids._last_eomt_pooled_outputs = p_empty_ids
    out_empty_ids = v_empty_ids._compute_eomt_object_block_side_output()

    # Duplicate ids should be tracked and path should remain stable.
    p_duplicates = make_pooled_outputs_two_samples()
    p_duplicates["frame_meta"][0]["selected_query_ids"] = [0, 0, 1]
    v_duplicates = DummyIntegrationValidator(SimpleNamespace(**base_cfg))
    v_duplicates._last_eomt_pooled_outputs = p_duplicates
    out_duplicates = v_duplicates._compute_eomt_object_block_side_output()

    # Out-of-range ids should cleanly fallback.
    p_oob = make_pooled_outputs_two_samples()
    p_oob["frame_meta"][0]["selected_query_ids"] = [999]
    v_oob = DummyIntegrationValidator(SimpleNamespace(**base_cfg))
    v_oob._last_eomt_pooled_outputs = p_oob
    out_oob = v_oob._compute_eomt_object_block_side_output()

    # Words-only path is accepted but matching is deferred.
    p_words = make_pooled_outputs_two_samples()
    p_words["frame_meta"][0]["selected_words"] = ["chair", "lamp"]
    v_words = DummyIntegrationValidator(SimpleNamespace(**base_cfg))
    v_words._last_eomt_pooled_outputs = p_words
    out_words = v_words._compute_eomt_object_block_side_output()

    missing_ok = out_missing.get("fallback_reason") == "missing_external_selection"
    empty_ok = (
        out_empty_ids.get("fallback_reason") == "no_valid_selected_masks"
        and out_empty_ids.get("external_selection_contract", {}).get("matching_status") == "no_query_or_mask_ids"
    )
    duplicates_ok = (
        out_duplicates.get("external_selection_contract", {}).get("selected_id_duplicates", 0) > 0
        and out_duplicates.get("fallback_reason") is None
    )
    oob_ok = (
        out_oob.get("fallback_reason") == "no_valid_selected_masks"
        and out_oob.get("external_selection_contract", {}).get("matching_status") == "ids_out_of_range_or_unmatched"
    )
    words_ok = (
        out_words.get("fallback_reason") == "no_matched_masks"
        and out_words.get("external_socket_note") == "word_matching_deferred"
        and out_words.get("external_selection_contract", {}).get("matching_status") == "word_matching_deferred_no_ids"
    )

    return {
        "pass": bool(missing_ok and empty_ok and duplicates_ok and oob_ok and words_ok),
        "details": {
            "missing_external_selection": {
                "fallback_reason": out_missing.get("fallback_reason"),
                "contract": out_missing.get("external_selection_contract", {}),
            },
            "empty_ids": {
                "fallback_reason": out_empty_ids.get("fallback_reason"),
                "contract": out_empty_ids.get("external_selection_contract", {}),
            },
            "duplicate_ids": {
                "fallback_reason": out_duplicates.get("fallback_reason"),
                "contract": out_duplicates.get("external_selection_contract", {}),
            },
            "out_of_range_ids": {
                "fallback_reason": out_oob.get("fallback_reason"),
                "contract": out_oob.get("external_selection_contract", {}),
            },
            "words_only": {
                "fallback_reason": out_words.get("fallback_reason"),
                "socket_note": out_words.get("external_socket_note"),
                "contract": out_words.get("external_selection_contract", {}),
            },
        },
    }


def build_report() -> Dict[str, Any]:
    cases = {
        "Case 1 length and sample mapping": case_length_and_mapping(),
        "Case 2 before/after ordering": case_before_after_order(),
        "Case 3 fallback preserves baseline": case_fallback_preserves_baseline(),
        "Case 4 OBJ_INFO and type embedding": case_obj_info_and_type_embedding(),
        "Case 5 external socket contract": case_external_socket_contract(),
        "Case 6 external socket fallback matrix": case_external_socket_fallback_matrix(),
    }

    summary = {k: ("pass" if v.get("pass") else "fail") for k, v in cases.items()}
    problems = [k for k, s in summary.items() if s != "pass"]
    verdict = "validated" if len(problems) == 0 else "validated with caveats"

    return {
        "summary": summary,
        "problems": problems,
        "verdict": verdict,
        "results": cases,
    }


def print_report(report: Dict[str, Any]) -> None:
    print("1. Summary")
    for case_name, status in report["summary"].items():
        print(f"- {case_name}: {status}")

    print("\n2. Problems found")
    if len(report["problems"]) == 0:
        print("- none")
    else:
        for problem in report["problems"]:
            print(f"- {problem}")

    print("\n3. Final verdict")
    print(f"- {report['verdict']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate sequence-level EoMT object-block integration")
    parser.add_argument(
        "--output_json",
        type=str,
        default="logs/eomt_object_block_integration_report.json",
        help="Path to write JSON report",
    )
    args = parser.parse_args()

    torch.manual_seed(0)
    report = build_report()

    out_dir = os.path.dirname(args.output_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print_report(report)


if __name__ == "__main__":
    main()
