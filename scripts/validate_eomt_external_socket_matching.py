#!/usr/bin/env python3
"""Lightweight validation for selector-side external socket word matching."""

import json
import os
import sys
from types import SimpleNamespace
from typing import Any, Dict

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from llava.model.multimodal_eomt.eomt_selector import EoMTObjectTokenSelector
from llava.model.multimodal_eomt.word_class_matcher import coco_class_id_from_name


def make_pooled_outputs(hidden_size: int = 8) -> Dict[str, Any]:
    pooled_tokens = torch.arange(3 * 3 * hidden_size, dtype=torch.float32).view(3, 3, hidden_size)
    selected_scores = torch.tensor(
        [
            [0.95, 0.85, 0.10],
            [0.92, 0.80, 0.05],
            [0.90, 0.70, 0.02],
        ],
        dtype=torch.float32,
    )
    selected_indices = torch.tensor(
        [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
        ],
        dtype=torch.long,
    )
    selected_class_ids = torch.tensor(
        [
            [int(coco_class_id_from_name("chair")), int(coco_class_id_from_name("light")), -1],
            [int(coco_class_id_from_name("cup")), int(coco_class_id_from_name("dining_table")), -1],
            [int(coco_class_id_from_name("couch")), int(coco_class_id_from_name("sink")), -1],
        ],
        dtype=torch.long,
    )
    return {
        "pooled_tokens": pooled_tokens,
        "selected_scores": selected_scores,
        "selected_indices": selected_indices,
        "selected_class_ids": selected_class_ids,
        "selected_valid_mask": torch.tensor(
            [
                [True, True, False],
                [True, True, False],
                [True, True, False],
            ],
            dtype=torch.bool,
        ),
        "aligned_sample_frame_pairs": [(0, 0), (0, 1), (0, 2)],
        "frame_meta": [{}, {}, {}],
        "stuff_class_ids": frozenset(),
        "pool_skipped": False,
    }


def make_config(**overrides: Any) -> SimpleNamespace:
    base = {
        "mm_eomt_selector_mode": "external_socket",
        "mm_eomt_selector_order": "frame_then_score",
        "mm_eomt_object_block_position": "after_visual",
        "mm_eomt_selector_drop_no_object": True,
        "mm_eomt_selector_no_object_class_id": -1,
        "mm_eomt_selector_keep_stuff": True,
        "mm_eomt_selector_keep_things": True,
        "mm_eomt_object_block_max_objects": 8,
        "mm_eomt_object_block_max_per_frame": 3,
        "mm_eomt_external_socket_deduplicate": True,
        "mm_eomt_external_socket_word_topn": 1,
        "mm_eomt_word_match_enable": True,
        "mm_eomt_word_match_source": "visible_grounded_words",
        "mm_eomt_word_match_mode": "hybrid_safe",
        "mm_eomt_word_match_no_match": "keep_masks",
        "mm_eomt_word_match_similarity_threshold": 0.86,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def build_external_selection(frame_meta):
    entries = []
    for frame_idx, meta in enumerate(frame_meta):
        socket = dict(meta)
        if any(key in socket for key in ("selected_query_ids", "selected_mask_ids", "selected_words", "visible_grounded_words")):
            entries.append(
                {
                    "sample_idx": 0,
                    "frame_idx": frame_idx,
                    "selected_query_ids": socket.get("selected_query_ids"),
                    "selected_mask_ids": socket.get("selected_mask_ids"),
                    "selected_words": socket.get("selected_words"),
                    "visible_grounded_words": socket.get("visible_grounded_words"),
                }
            )
    return {"entries": entries}


def case_words_only_matches() -> Dict[str, Any]:
    pooled = make_pooled_outputs()
    pooled["frame_meta"][0]["visible_grounded_words"] = ["chair", "lamp"]
    pooled["frame_meta"][0]["selected_words"] = ["chair", "lamp", "left", "right"]
    selector = EoMTObjectTokenSelector()
    result = selector.select(
        pooled_outputs=pooled,
        config=make_config(),
        external_selection=build_external_selection(pooled["frame_meta"]),
    )
    contract = result.get("external_selection_contract", {})
    return {
        "pass": bool(
            result.get("fallback_reason") is None
            and contract.get("matching_status") == "words_matched"
            and result.get("selected_count", 0) == 2
        ),
        "details": {
            "selected_count": result.get("selected_count", 0),
            "selected_indices": result.get("selected_indices", []),
            "contract": contract,
        },
    }


def case_mixed_ids_and_words() -> Dict[str, Any]:
    pooled = make_pooled_outputs()
    pooled["frame_meta"][0]["selected_query_ids"] = [0]
    pooled["frame_meta"][2]["visible_grounded_words"] = ["sofa", "sink"]
    selector = EoMTObjectTokenSelector()
    result = selector.select(
        pooled_outputs=pooled,
        config=make_config(),
        external_selection=build_external_selection(pooled["frame_meta"]),
    )
    contract = result.get("external_selection_contract", {})
    return {
        "pass": bool(
            result.get("fallback_reason") is None
            and contract.get("matching_status") == "mixed_ids_and_words"
            and contract.get("word_frames_with_matches", 0) == 1
            and result.get("selected_count", 0) == 3
        ),
        "details": {
            "selected_count": result.get("selected_count", 0),
            "selected_pairs": result.get("selected_sample_frame_pairs", []),
            "contract": contract,
        },
    }


def case_no_match_keep_masks() -> Dict[str, Any]:
    pooled = make_pooled_outputs()
    pooled["frame_meta"][0]["visible_grounded_words"] = ["cabinet"]
    selector = EoMTObjectTokenSelector()
    result = selector.select(
        pooled_outputs=pooled,
        config=make_config(mm_eomt_word_match_no_match="keep_masks"),
        external_selection=build_external_selection(pooled["frame_meta"]),
    )
    contract = result.get("external_selection_contract", {})
    return {
        "pass": bool(
            result.get("fallback_reason") is None
            and contract.get("matching_status") == "words_no_match_keep_masks"
            and result.get("selected_count", 0) == 2
        ),
        "details": {
            "selected_count": result.get("selected_count", 0),
            "contract": contract,
        },
    }


def main() -> None:
    report = {
        "Words-only matching": case_words_only_matches(),
        "Mixed ids and words": case_mixed_ids_and_words(),
        "No-match keep_masks": case_no_match_keep_masks(),
    }
    summary = {name: ("pass" if case["pass"] else "fail") for name, case in report.items()}
    problems = [name for name, status in summary.items() if status != "pass"]
    payload = {
        "summary": summary,
        "problems": problems,
        "results": report,
        "verdict": "validated" if not problems else "validated with caveats",
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
