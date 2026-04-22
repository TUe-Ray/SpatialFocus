#!/usr/bin/env python3
"""Stream VLM-3R selected-word artifacts and compare them against COCO classes."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_SELECTED_WORDS_ROOT = (
    WORKSPACE_ROOT / "Word-Selection_LLM" / "artifacts" / "vlm3r_word_selection" / "vsibench_split"
)

import sys

PROJECT_ROOT = str(SCRIPT_DIR.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from llava.model.multimodal_eomt.word_class_matcher import (
    COCO_PANOPTIC_CLASS_NAMES,
    WordClassMatchConfig,
    canonicalize_word_list,
    match_entry_words_to_class_names,
)


SOURCES = ("visible_grounded_words", "selected_words")
MODES = ("exact", "exact_alias", "hybrid_safe")
NO_MATCH_BEHAVIORS = ("keep_masks", "keep_best_similar", "filter_out")


def _make_stats() -> Dict[str, Any]:
    return {
        "rows": 0,
        "total_occurrences": 0,
        "unique_words": set(),
        "matched_occurrences": 0,
        "matched_unique_words": set(),
        "matched_rows": 0,
        "unmatched_counter": Counter(),
        "alias_counter": Counter(),
        "hybrid_counter": Counter(),
        "matched_class_counter": Counter(),
    }


def _make_ablation_stats() -> Dict[str, Any]:
    return {
        "rows": 0,
        "rows_with_words": 0,
        "rows_filter_applied": 0,
        "rows_filter_empty": 0,
        "rows_keep_all": 0,
        "rows_with_class_match": 0,
        "rows_with_best_similar_recovery": 0,
    }


def _percent(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _freeze_counter(counter: Counter, top_k: int) -> List[Dict[str, Any]]:
    return [
        {"item": item if not isinstance(item, tuple) else list(item), "count": count}
        for item, count in counter.most_common(top_k)
    ]


def _finalize_stats(stats: Dict[str, Any], top_k: int) -> Dict[str, Any]:
    unique_word_count = len(stats["unique_words"])
    matched_unique_count = len(stats["matched_unique_words"])
    total_occurrences = int(stats["total_occurrences"])
    matched_occurrences = int(stats["matched_occurrences"])
    rows = int(stats["rows"])
    matched_rows = int(stats["matched_rows"])
    return {
        "rows": rows,
        "total_occurrences": total_occurrences,
        "unique_words": unique_word_count,
        "matched_occurrences": matched_occurrences,
        "matched_occurrence_fraction": _percent(matched_occurrences, total_occurrences),
        "matched_unique_words": matched_unique_count,
        "matched_unique_fraction": _percent(matched_unique_count, unique_word_count),
        "matched_rows": matched_rows,
        "matched_row_fraction": _percent(matched_rows, rows),
        "top_unmatched_words": _freeze_counter(stats["unmatched_counter"], top_k),
        "top_alias_mappings": _freeze_counter(stats["alias_counter"], top_k),
        "top_hybrid_mappings": _freeze_counter(stats["hybrid_counter"], top_k),
        "top_matched_classes": _freeze_counter(stats["matched_class_counter"], top_k),
    }


def _finalize_ablation(stats: Dict[str, Any]) -> Dict[str, Any]:
    rows_with_words = int(stats["rows_with_words"])
    return {
        "rows": int(stats["rows"]),
        "rows_with_words": rows_with_words,
        "rows_filter_applied": int(stats["rows_filter_applied"]),
        "rows_filter_applied_fraction": _percent(stats["rows_filter_applied"], rows_with_words),
        "rows_filter_empty": int(stats["rows_filter_empty"]),
        "rows_filter_empty_fraction": _percent(stats["rows_filter_empty"], rows_with_words),
        "rows_keep_all": int(stats["rows_keep_all"]),
        "rows_keep_all_fraction": _percent(stats["rows_keep_all"], rows_with_words),
        "rows_with_class_match": int(stats["rows_with_class_match"]),
        "rows_with_best_similar_recovery": int(stats["rows_with_best_similar_recovery"]),
    }


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _entry_cache_key(entry: Dict[str, Any], source: str, mode: str, no_match_behavior: str) -> Tuple[Any, ...]:
    return (
        source,
        mode,
        no_match_behavior,
        tuple(canonicalize_word_list(entry.get("visible_grounded_words", []))),
        tuple(canonicalize_word_list(entry.get("selected_words", []))),
    )


def _get_match_result(
    entry: Dict[str, Any],
    source: str,
    mode: str,
    no_match_behavior: str,
    cache: Dict[Tuple[Any, ...], Any],
):
    key = _entry_cache_key(entry, source, mode, no_match_behavior)
    result = cache.get(key)
    if result is not None:
        return result

    cfg = WordClassMatchConfig(
        enable=True,
        source=source,
        mode=mode,
        no_match_behavior=no_match_behavior,
        similarity_threshold=0.86,
    )
    result = match_entry_words_to_class_names(
        entry,
        candidate_class_names=COCO_PANOPTIC_CLASS_NAMES,
        match_config=cfg,
    )
    cache[key] = result
    return result


def _update_mode_stats(
    stats: Dict[str, Any],
    entry: Dict[str, Any],
    source: str,
    mode: str,
    cache: Dict[Tuple[Any, ...], Any],
) -> None:
    result = _get_match_result(entry, source, mode, "keep_masks", cache)
    input_words = canonicalize_word_list(result.input_words)
    stats["rows"] += 1
    stats["total_occurrences"] += len(input_words)
    stats["unique_words"].update(input_words)
    if result.matched_class_names:
        stats["matched_rows"] += 1

    matched_words = set(result.matched_words)
    stats["matched_occurrences"] += sum(1 for word in input_words if word in matched_words)
    stats["matched_unique_words"].update(matched_words)

    for word in input_words:
        if word not in matched_words:
            stats["unmatched_counter"][word] += 1

    for match in result.word_matches:
        mapped_pair = (match["word"], match["class_name"])
        if match["method"] == "alias":
            stats["alias_counter"][mapped_pair] += 1
        elif match["method"] == "hybrid_safe":
            stats["hybrid_counter"][mapped_pair] += 1
        stats["matched_class_counter"][match["class_name"]] += 1


def _update_ablation_stats(
    stats: Dict[str, Any],
    entry: Dict[str, Any],
    source: str,
    mode: str,
    no_match_behavior: str,
    cache: Dict[Tuple[Any, ...], Any],
) -> None:
    result = _get_match_result(entry, source, mode, no_match_behavior, cache)
    stats["rows"] += 1
    if len(result.input_words) == 0:
        return
    stats["rows_with_words"] += 1
    if result.matched_class_names:
        stats["rows_with_class_match"] += 1
    if result.filter_reason == "no_word_class_match_keep_best_similar":
        stats["rows_with_best_similar_recovery"] += 1
    if result.filter_applied:
        stats["rows_filter_applied"] += 1
        if len(result.kept_class_names) == 0:
            stats["rows_filter_empty"] += 1
    else:
        stats["rows_keep_all"] += 1


def _collect_report(selected_words_root: Path, top_k: int) -> Dict[str, Any]:
    split_paths = sorted(selected_words_root.glob("*/selected_words.jsonl"))
    if len(split_paths) == 0:
        raise FileNotFoundError(f"No selected_words.jsonl files found under {selected_words_root}")

    split_reports: Dict[str, Any] = {}
    global_mode_stats = {
        source: {mode: _make_stats() for mode in MODES}
        for source in SOURCES
    }
    global_ablation_stats = {
        source: {
            mode: {behavior: _make_ablation_stats() for behavior in NO_MATCH_BEHAVIORS}
            for mode in MODES
        }
        for source in SOURCES
    }
    match_cache: Dict[Tuple[Any, ...], Any] = {}

    for split_path in split_paths:
        split_name = split_path.parent.name
        split_mode_stats = {
            source: {mode: _make_stats() for mode in MODES}
            for source in SOURCES
        }
        split_ablation_stats = {
            source: {
                mode: {behavior: _make_ablation_stats() for behavior in NO_MATCH_BEHAVIORS}
                for mode in MODES
            }
            for source in SOURCES
        }

        for entry in _iter_jsonl(split_path):
            for source in SOURCES:
                for mode in MODES:
                    _update_mode_stats(split_mode_stats[source][mode], entry, source, mode, match_cache)
                    _update_mode_stats(global_mode_stats[source][mode], entry, source, mode, match_cache)
                    for behavior in NO_MATCH_BEHAVIORS:
                        _update_ablation_stats(
                            split_ablation_stats[source][mode][behavior],
                            entry,
                            source,
                            mode,
                            behavior,
                            match_cache,
                        )
                        _update_ablation_stats(
                            global_ablation_stats[source][mode][behavior],
                            entry,
                            source,
                            mode,
                            behavior,
                            match_cache,
                        )

        split_reports[split_name] = {
            "file": str(split_path),
            "coverage": {
                source: {
                    mode: _finalize_stats(split_mode_stats[source][mode], top_k)
                    for mode in MODES
                }
                for source in SOURCES
            },
            "ablations": {
                source: {
                    mode: {
                        behavior: _finalize_ablation(split_ablation_stats[source][mode][behavior])
                        for behavior in NO_MATCH_BEHAVIORS
                    }
                    for mode in MODES
                }
                for source in SOURCES
            },
        }

    return {
        "selected_words_root": str(selected_words_root),
        "coco_class_count": len(COCO_PANOPTIC_CLASS_NAMES),
        "splits": split_reports,
        "global": {
            "coverage": {
                source: {
                    mode: _finalize_stats(global_mode_stats[source][mode], top_k)
                    for mode in MODES
                }
                for source in SOURCES
            },
            "ablations": {
                source: {
                    mode: {
                        behavior: _finalize_ablation(global_ablation_stats[source][mode][behavior])
                        for behavior in NO_MATCH_BEHAVIORS
                    }
                    for mode in MODES
                }
                for source in SOURCES
            },
        },
    }


def _build_text_summary(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("Coverage Summary")
    lines.append(f"- selected_words_root: {report['selected_words_root']}")
    lines.append(f"- coco_class_count: {report['coco_class_count']}")

    global_visible = report["global"]["coverage"]["visible_grounded_words"]
    lines.append("")
    lines.append("Global visible_grounded_words coverage")
    for mode in MODES:
        stats = global_visible[mode]
        lines.append(
            "- "
            f"{mode}: matched_unique={stats['matched_unique_words']}/{stats['unique_words']} "
            f"({stats['matched_unique_fraction']:.4f}), "
            f"matched_occurrences={stats['matched_occurrences']}/{stats['total_occurrences']} "
            f"({stats['matched_occurrence_fraction']:.4f})"
        )

    lines.append("")
    lines.append("Top unmatched visible_grounded_words (hybrid_safe)")
    for item in global_visible["hybrid_safe"]["top_unmatched_words"][:10]:
        lines.append(f"- {item['count']}: {item['item']}")

    lines.append("")
    lines.append("Top alias wins (visible_grounded_words, hybrid_safe)")
    for item in global_visible["hybrid_safe"]["top_alias_mappings"][:10]:
        word, class_name = item["item"]
        lines.append(f"- {item['count']}: {word} -> {class_name}")

    lines.append("")
    lines.append("Ablations (visible_grounded_words, hybrid_safe)")
    for behavior in NO_MATCH_BEHAVIORS:
        stats = report["global"]["ablations"]["visible_grounded_words"]["hybrid_safe"][behavior]
        lines.append(
            "- "
            f"{behavior}: filter_applied={stats['rows_filter_applied']}/{stats['rows_with_words']} "
            f"({stats['rows_filter_applied_fraction']:.4f}), "
            f"keep_all={stats['rows_keep_all']}/{stats['rows_with_words']} "
            f"({stats['rows_keep_all_fraction']:.4f})"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze VLM-3R selected words against COCO class names")
    parser.add_argument(
        "--selected-words-root",
        type=Path,
        default=DEFAULT_SELECTED_WORDS_ROOT,
        help="Directory containing vsibench_split/*/selected_words.jsonl",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to save the full JSON report",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=40,
        help="Number of top unmatched words / mappings to retain per table",
    )
    args = parser.parse_args()

    report = _collect_report(args.selected_words_root, top_k=max(args.top_k, 1))
    summary = _build_text_summary(report)
    print(summary)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print("")
        print(f"Saved JSON report to {args.output_json}")


if __name__ == "__main__":
    main()
