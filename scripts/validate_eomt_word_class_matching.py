#!/usr/bin/env python3
"""Focused validation for EoMT word-to-class matching and selective 3D gating."""

import json
import os
import sys
from typing import Any, Dict, List

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from llava.model.multimodal_eomt.selective_3d_gate import Selective3DConfig, apply_selective_3d_fusion
from llava.model.multimodal_eomt.word_class_matcher import (
    COCO_PANOPTIC_CLASS_NAMES,
    WordClassMatchConfig,
    coco_class_id_from_name,
    match_entry_words_to_class_names,
)


def _build_logits(class_names: List[str]) -> torch.Tensor:
    num_classes = len(COCO_PANOPTIC_CLASS_NAMES)
    logits = torch.full((1, len(class_names), num_classes + 1), fill_value=-6.0, dtype=torch.float32)
    for idx, class_name in enumerate(class_names):
        class_id = coco_class_id_from_name(class_name)
        if class_id is None:
            raise ValueError(f"Unknown COCO class name for test fixture: {class_name}")
        logits[0, idx, class_id] = 6.0 - float(idx)
        logits[0, idx, -1] = -8.0
    return logits


def _build_soft_masks(num_queries: int) -> torch.Tensor:
    base_masks = torch.tensor(
        [
            [[0.9, 0.1], [0.1, 0.1]],
            [[0.1, 0.9], [0.1, 0.1]],
            [[0.1, 0.1], [0.9, 0.1]],
        ],
        dtype=torch.float32,
    )
    return base_masks[:num_queries].unsqueeze(0)


def _build_eomt_outputs(class_names: List[str], frame_meta: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "soft_masks": _build_soft_masks(len(class_names)),
        "class_logits": _build_logits(class_names),
        "mask_resolution": (2, 2),
        "query_count": len(class_names),
        "frame_meta": [frame_meta],
        "stuff_class_ids": frozenset(),
    }


def case_matcher_aliases() -> Dict[str, Any]:
    entry = {
        "visible_grounded_words": ["sofa", "table", "monitor", "office chair", "blinds", "telephone"],
    }
    cfg = WordClassMatchConfig(
        enable=True,
        source="visible_grounded_words",
        mode="hybrid_safe",
        no_match_behavior="keep_masks",
        similarity_threshold=0.86,
    )
    result = match_entry_words_to_class_names(
        entry,
        candidate_class_names=["couch", "dining_table", "tv", "chair", "window_blind", "cell_phone"],
        match_config=cfg,
    )
    expected_classes = {"couch", "dining_table", "tv", "chair", "window_blind", "cell_phone"}
    methods = {item["word"]: item["method"] for item in result.word_matches}
    alias_words = {"sofa", "table", "monitor", "office chair", "blinds", "telephone"}
    alias_ok = all(methods.get(word) == "alias" for word in alias_words)
    return {
        "pass": bool(set(result.matched_class_names) == expected_classes and alias_ok),
        "details": result.to_dict(),
    }


def case_visible_source_ignores_reasoning() -> Dict[str, Any]:
    entry = {
        "visible_grounded_words": ["chair"],
        "selected_words": ["chair", "left", "right", "closest point", "in meters"],
    }
    cfg = WordClassMatchConfig(
        enable=True,
        source="visible_grounded_words",
        mode="hybrid_safe",
        no_match_behavior="keep_masks",
        similarity_threshold=0.86,
    )
    result = match_entry_words_to_class_names(
        entry,
        candidate_class_names=["chair", "light"],
        match_config=cfg,
    )
    return {
        "pass": bool(
            result.input_words == ["chair"]
            and result.matched_class_names == ["chair"]
            and result.unmatched_words == []
        ),
        "details": result.to_dict(),
    }


def case_selective_gate_word_filter() -> Dict[str, Any]:
    patch_tokens = torch.arange(8, dtype=torch.float32).view(1, 4, 2)
    config = Selective3DConfig(
        enable=True,
        selector_mode="confidence",
        score_threshold=0.50,
        topk=-1,
        word_match_enable=True,
        word_match_source="visible_grounded_words",
        word_match_mode="hybrid_safe",
        word_match_no_match="keep_masks",
        word_match_similarity_threshold=0.86,
    )
    outputs = _build_eomt_outputs(
        ["chair", "light", "couch"],
        {"visible_grounded_words": ["chair"]},
    )
    _, debug_infos = apply_selective_3d_fusion(patch_tokens, outputs, config)
    dbg = debug_infos[0]
    return {
        "pass": bool(
            dbg.word_filter_applied
            and dbg.word_match_matched_class_names == ["chair"]
            and dbg.selected_class_names == ["chair"]
        ),
        "details": {
            "selected_class_names": dbg.selected_class_names,
            "word_match": dbg.word_match_matched_class_names,
            "word_filter_reason": dbg.word_filter_reason,
        },
    }


def case_selective_gate_keep_masks_no_match() -> Dict[str, Any]:
    patch_tokens = torch.arange(8, dtype=torch.float32).view(1, 4, 2)
    config = Selective3DConfig(
        enable=True,
        selector_mode="confidence",
        score_threshold=0.50,
        topk=-1,
        word_match_enable=True,
        word_match_source="visible_grounded_words",
        word_match_mode="hybrid_safe",
        word_match_no_match="keep_masks",
        word_match_similarity_threshold=0.86,
    )
    outputs = _build_eomt_outputs(
        ["chair", "light", "couch"],
        {"visible_grounded_words": ["cabinet"]},
    )
    _, debug_infos = apply_selective_3d_fusion(patch_tokens, outputs, config)
    dbg = debug_infos[0]
    return {
        "pass": bool(
            not dbg.word_filter_applied
            and dbg.word_filter_reason == "no_word_class_match_keep_masks"
            and len(dbg.selected_class_names) == 3
        ),
        "details": {
            "selected_class_names": dbg.selected_class_names,
            "word_filter_reason": dbg.word_filter_reason,
        },
    }


def case_selective_gate_filter_out_no_match() -> Dict[str, Any]:
    patch_tokens = torch.arange(8, dtype=torch.float32).view(1, 4, 2)
    config = Selective3DConfig(
        enable=True,
        selector_mode="confidence",
        score_threshold=0.50,
        topk=-1,
        word_match_enable=True,
        word_match_source="visible_grounded_words",
        word_match_mode="hybrid_safe",
        word_match_no_match="filter_out",
        word_match_similarity_threshold=0.86,
    )
    outputs = _build_eomt_outputs(
        ["chair", "light", "couch"],
        {"visible_grounded_words": ["cabinet"]},
    )
    _, debug_infos = apply_selective_3d_fusion(patch_tokens, outputs, config)
    dbg = debug_infos[0]
    return {
        "pass": bool(
            dbg.word_filter_applied
            and dbg.fallback_reason == "no_queries_after_threshold_and_topk"
            and dbg.word_filter_reason == "no_word_class_match_filter_out"
        ),
        "details": {
            "fallback_reason": dbg.fallback_reason,
            "word_filter_reason": dbg.word_filter_reason,
            "selected_class_names": dbg.selected_class_names,
        },
    }


def case_selective_gate_keep_best_similar() -> Dict[str, Any]:
    patch_tokens = torch.arange(8, dtype=torch.float32).view(1, 4, 2)
    config = Selective3DConfig(
        enable=True,
        selector_mode="confidence",
        score_threshold=0.50,
        topk=-1,
        word_match_enable=True,
        word_match_source="visible_grounded_words",
        word_match_mode="hybrid_safe",
        word_match_no_match="keep_best_similar",
        word_match_similarity_threshold=0.86,
    )
    outputs = _build_eomt_outputs(
        ["window_blind", "chair", "couch"],
        {"visible_grounded_words": ["window blnd"]},
    )
    _, debug_infos = apply_selective_3d_fusion(patch_tokens, outputs, config)
    dbg = debug_infos[0]
    return {
        "pass": bool(
            dbg.word_filter_applied
            and dbg.word_match_kept_class_names == ["window_blind"]
            and dbg.selected_class_names == ["window_blind"]
        ),
        "details": {
            "word_filter_reason": dbg.word_filter_reason,
            "kept_class_names": dbg.word_match_kept_class_names,
            "selected_class_names": dbg.selected_class_names,
        },
    }


def build_report() -> Dict[str, Any]:
    cases = {
        "Matcher alias coverage": case_matcher_aliases(),
        "Visible source ignores reasoning": case_visible_source_ignores_reasoning(),
        "Selective gate word filter": case_selective_gate_word_filter(),
        "Selective gate keep_masks no-match": case_selective_gate_keep_masks_no_match(),
        "Selective gate filter_out no-match": case_selective_gate_filter_out_no_match(),
        "Selective gate keep_best_similar": case_selective_gate_keep_best_similar(),
    }
    summary = {name: ("pass" if case.get("pass") else "fail") for name, case in cases.items()}
    problems = [name for name, status in summary.items() if status != "pass"]
    verdict = "validated" if len(problems) == 0 else "validated with caveats"
    return {
        "summary": summary,
        "problems": problems,
        "verdict": verdict,
        "results": cases,
    }


def main() -> None:
    report = build_report()
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["problems"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
