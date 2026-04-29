"""Minimal Phase-1B selector for EoMT pooled object tokens.

This selector consumes existing trusted pooled side outputs and returns a safe,
config-driven subset for optional object-block insertion.
"""

from typing import Any, Dict, List, Optional, Tuple

import torch

from .word_class_matcher import (
    WordClassMatchConfig,
    canonicalize_word_list,
    lookup_class_name,
    match_entry_words_to_class_names,
)


class EoMTObjectTokenSelector:
    """Select pooled object tokens with deterministic ordering and budgets."""

    SUPPORTED_MODES = {"class_aware", "external_socket"}
    SUPPORTED_ORDERS = {"score_desc", "frame_then_score"}

    def _empty_result(
        self,
        selector_mode: str,
        ordering_mode: str,
        fallback_reason: str,
        append_position: str,
        taxonomy_note: Optional[str] = None,
        external_socket_note: Optional[str] = None,
    ) -> Dict[str, Any]:
        return {
            "selector_mode": selector_mode,
            "ordering_mode": ordering_mode,
            "append_position": append_position,
            "selected_tokens_by_sample": {},
            "selected_count": 0,
            "selected_sample_frame_pairs": [],
            "selected_scores": [],
            "selected_indices": [],
            "truncated_per_frame_count": 0,
            "truncated_global_count": 0,
            "taxonomy_note": taxonomy_note,
            "external_socket_note": external_socket_note,
            "external_selection_contract": {},
            "fallback_reason": fallback_reason,
        }

    def _normalize_id_list(self, values: Any) -> List[int]:
        if isinstance(values, (list, tuple, set)):
            normalized = []
            for x in values:
                if isinstance(x, bool):
                    continue
                if isinstance(x, int):
                    normalized.append(int(x))
                    continue
                if torch.is_tensor(x) and x.ndim == 0 and x.dtype in {
                    torch.int8,
                    torch.int16,
                    torch.int32,
                    torch.int64,
                    torch.uint8,
                }:
                    normalized.append(int(x.item()))
            return normalized
        return []

    def _to_pairs(self, pairs: Any, frame_count: int) -> List[Tuple[int, int]]:
        if isinstance(pairs, list) and len(pairs) == frame_count:
            parsed = []
            for idx, pair in enumerate(pairs):
                if (
                    isinstance(pair, (list, tuple))
                    and len(pair) >= 2
                    and isinstance(pair[0], (int,))
                    and isinstance(pair[1], (int,))
                ):
                    parsed.append((int(pair[0]), int(pair[1])))
                else:
                    parsed.append((idx, idx))
            return parsed
        return [(idx, idx) for idx in range(frame_count)]

    def _get_stuff_class_ids(self, pooled_outputs: Dict[str, Any]) -> Optional[set]:
        for key in ("stuff_class_ids", "eomt_stuff_class_ids"):
            value = pooled_outputs.get(key, None)
            if isinstance(value, (list, tuple, set)):
                ids = {int(x) for x in value if isinstance(x, (int,))}
                if len(ids) > 0:
                    return ids
        return None

    def _is_no_object(self, class_id: int, no_object_class_id: int) -> bool:
        if no_object_class_id >= 0:
            return class_id == no_object_class_id or class_id < 0
        return class_id < 0

    def _merge_socket_word_fields(
        self,
        merged: Dict[str, Any],
        item: Dict[str, Any],
    ) -> None:
        for field_name in ("visible_grounded_words", "selected_words"):
            values = canonicalize_word_list(item.get(field_name, []))
            if not values:
                continue
            merged_values = canonicalize_word_list(merged.get(field_name, []))
            merged[field_name] = merged_values + [value for value in values if value not in set(merged_values)]

    def _apply_external_socket_words(
        self,
        candidates: List[Dict[str, Any]],
        word_entries_by_key: Dict[Tuple[int, int], Dict[str, Any]],
        config: Any,
        contract: Dict[str, Any],
    ) -> Tuple[List[Dict[str, Any]], Optional[str], Optional[str]]:
        match_config = WordClassMatchConfig.from_config(config)
        candidates_by_key: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
        for cand in candidates:
            key = (int(cand["sample_idx"]), int(cand["frame_idx"]))
            candidates_by_key.setdefault(key, []).append(cand)

        filtered: List[Dict[str, Any]] = []
        frame_results: List[Dict[str, Any]] = []
        matched_frames = 0
        kept_frames_without_match = 0
        recovered_frames = 0
        filter_out_frames = 0
        missing_candidate_frames = 0

        contract["word_match_enabled"] = bool(match_config.enable)
        contract["word_source_requested"] = match_config.source
        contract["word_match_mode"] = match_config.mode
        contract["word_no_match_behavior"] = match_config.no_match_behavior
        contract["word_similarity_threshold"] = float(match_config.similarity_threshold)

        for key, entry in word_entries_by_key.items():
            frame_candidates = candidates_by_key.get(key, [])
            if len(frame_candidates) == 0:
                missing_candidate_frames += 1
                frame_results.append(
                    {
                        "sample_idx": int(key[0]),
                        "frame_idx": int(key[1]),
                        "filter_applied": False,
                        "filter_reason": "missing_candidate_frame",
                    }
                )
                continue

            candidate_class_names: List[str] = []
            seen_class_names = set()
            for cand in frame_candidates:
                class_name = str(cand.get("class_name", ""))
                if not class_name or class_name in seen_class_names:
                    continue
                seen_class_names.add(class_name)
                candidate_class_names.append(class_name)

            match_result = match_entry_words_to_class_names(
                entry,
                candidate_class_names=candidate_class_names,
                match_config=match_config,
            )
            frame_result = match_result.to_dict()
            frame_result["sample_idx"] = int(key[0])
            frame_result["frame_idx"] = int(key[1])
            frame_result["candidate_count"] = len(frame_candidates)
            frame_results.append(frame_result)

            if match_result.matched_class_names:
                matched_frames += 1
            elif match_result.filter_reason == "no_word_class_match_keep_masks":
                kept_frames_without_match += 1
            elif match_result.filter_reason == "no_word_class_match_keep_best_similar":
                recovered_frames += 1
            elif match_result.filter_reason == "no_word_class_match_filter_out":
                filter_out_frames += 1

            if match_result.filter_applied:
                kept_class_names = set(match_result.kept_class_names)
                if kept_class_names:
                    filtered.extend(
                        cand for cand in frame_candidates if cand.get("class_name") in kept_class_names
                    )
                continue

            filtered.extend(frame_candidates)

        if bool(getattr(config, "mm_eomt_external_socket_deduplicate", True)):
            deduped: List[Dict[str, Any]] = []
            seen = set()
            for cand in filtered:
                dedupe_key = (cand["sample_idx"], cand["frame_idx"], cand["query_idx"])
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                deduped.append(cand)
            filtered = deduped

        contract["word_frame_result_count"] = len(frame_results)
        contract["word_frame_results"] = frame_results
        contract["word_frames_with_matches"] = matched_frames
        contract["word_frames_keep_masks"] = kept_frames_without_match
        contract["word_frames_keep_best_similar"] = recovered_frames
        contract["word_frames_filter_out"] = filter_out_frames
        contract["word_frames_missing_candidates"] = missing_candidate_frames
        contract["matched_candidate_count"] = int(len(filtered))

        if not match_config.enable:
            contract["matching_status"] = "word_match_disabled"
            return [], "no_matched_masks", "word_match_disabled"
        if len(filtered) == 0:
            if filter_out_frames > 0:
                contract["matching_status"] = "words_no_match_filter_out"
                return [], "no_matched_masks", "words_no_match_filter_out"
            contract["matching_status"] = "words_no_matched_candidates"
            return [], "no_matched_masks", "words_no_matched_candidates"
        if matched_frames > 0:
            contract["matching_status"] = "words_matched"
            return filtered, None, "words_matched"
        if recovered_frames > 0:
            contract["matching_status"] = "words_no_match_keep_best_similar"
            return filtered, None, "words_no_match_keep_best_similar"
        if kept_frames_without_match > 0:
            contract["matching_status"] = "words_no_match_keep_masks"
            return filtered, None, "words_no_match_keep_masks"

        contract["matching_status"] = "words_candidates_retained"
        return filtered, None, "words_candidates_retained"

    def _apply_external_socket(
        self,
        candidates: List[Dict[str, Any]],
        external_selection: Optional[Dict[str, Any]],
        config: Any,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Optional[str], Optional[str]]:
        # External selection schema:
        # {
        #   "entries": [
        #     {
        #       "sample_idx": int,
        #       "frame_idx": int,
        #       "selected_query_ids": Optional[List[int]],
        #       "selected_mask_ids": Optional[List[int]],
        #       "visible_grounded_words": Optional[List[str]],
        #       "selected_words": Optional[List[str]],
        #     },
        #     ...
        #   ]
        # }
        contract = {
            "selected_words_present": False,
            "visible_grounded_words_present": False,
            "selected_mask_ids_present": False,
            "selected_query_ids_present": False,
            "word_topn": int(getattr(config, "mm_eomt_external_socket_word_topn", 1)),
            "deduplicate": bool(getattr(config, "mm_eomt_external_socket_deduplicate", True)),
            "input_entry_count": 0,
            "parsed_entry_count": 0,
            "selected_id_total": 0,
            "selected_id_unique": 0,
            "selected_id_duplicates": 0,
            "selected_id_out_of_range": 0,
            "matched_candidate_count": 0,
            "matching_status": "not_requested",
        }

        if not isinstance(external_selection, dict):
            contract["matching_status"] = "missing_external_selection"
            return [], contract, "missing_external_selection", None

        entries = external_selection.get("entries", None)
        if not isinstance(entries, list) or len(entries) == 0:
            contract["matching_status"] = "empty_external_selection"
            return [], contract, "empty_external_selection", None
        contract["input_entry_count"] = int(len(entries))

        query_map: Dict[Tuple[int, int], set] = {}
        word_entries = 0
        word_entries_by_key: Dict[Tuple[int, int], Dict[str, Any]] = {}
        unique_ids_all = set()
        candidate_query_ids_by_key: Dict[Tuple[int, int], set] = {}
        for cand in candidates:
            key = (int(cand["sample_idx"]), int(cand["frame_idx"]))
            candidate_query_ids_by_key.setdefault(key, set()).add(int(cand["query_idx"]))

        for item in entries:
            if not isinstance(item, dict):
                continue
            sample_idx = int(item.get("sample_idx", -1))
            frame_idx = int(item.get("frame_idx", -1))
            if sample_idx < 0 or frame_idx < 0:
                continue
            contract["parsed_entry_count"] += 1

            selected_query_ids = self._normalize_id_list(item.get("selected_query_ids", None))
            selected_mask_ids = self._normalize_id_list(item.get("selected_mask_ids", None))
            selected_words = item.get("selected_words", None)
            visible_grounded_words = item.get("visible_grounded_words", None)
            has_word_payload = False

            if isinstance(selected_words, (list, tuple)) and len(selected_words) > 0:
                has_word_payload = True
                contract["selected_words_present"] = True
            if isinstance(visible_grounded_words, (list, tuple)) and len(visible_grounded_words) > 0:
                has_word_payload = True
                contract["visible_grounded_words_present"] = True
            if len(selected_mask_ids) > 0:
                contract["selected_mask_ids_present"] = True
            if len(selected_query_ids) > 0:
                contract["selected_query_ids_present"] = True

            selected_ids_raw: List[int] = []
            allowed_ids = set(selected_query_ids)
            if len(allowed_ids) == 0 and len(selected_mask_ids) > 0:
                # Phase-1B contract: treat mask IDs as query IDs when provided.
                allowed_ids = set(selected_mask_ids)
                selected_ids_raw = list(selected_mask_ids)
            elif len(allowed_ids) > 0:
                selected_ids_raw = list(selected_query_ids)

            if len(selected_ids_raw) > 0:
                contract["selected_id_total"] += int(len(selected_ids_raw))
                contract["selected_id_duplicates"] += int(len(selected_ids_raw) - len(set(selected_ids_raw)))
                unique_ids_all.update(set(selected_ids_raw))

            if len(allowed_ids) > 0:
                key = (sample_idx, frame_idx)
                if key not in query_map:
                    query_map[key] = set()
                query_map[key].update(allowed_ids)

            if has_word_payload:
                word_entries += 1
                key = (sample_idx, frame_idx)
                merged_entry = word_entries_by_key.setdefault(
                    key,
                    {
                        "sample_idx": sample_idx,
                        "frame_idx": frame_idx,
                        "selected_words": [],
                        "visible_grounded_words": [],
                    },
                )
                self._merge_socket_word_fields(merged_entry, item)

        contract["selected_id_unique"] = int(len(unique_ids_all))

        if len(query_map) == 0 and word_entries == 0:
            contract["matching_status"] = "no_query_or_mask_ids"
            return [], contract, "no_valid_selected_masks", None

        filtered_by_ids: List[Dict[str, Any]] = []
        id_reason: Optional[str] = None
        id_note: Optional[str] = None
        id_status: Optional[str] = None

        if len(query_map) > 0:
            out_of_range = 0
            for key, ids in query_map.items():
                valid_ids = candidate_query_ids_by_key.get(key, set())
                out_of_range += sum(1 for query_id in ids if query_id not in valid_ids)
            contract["selected_id_out_of_range"] = int(out_of_range)

            filtered_by_ids = [
                cand
                for cand in candidates
                if cand["query_idx"] in query_map.get((cand["sample_idx"], cand["frame_idx"]), set())
            ]

            if bool(getattr(config, "mm_eomt_external_socket_deduplicate", True)):
                deduped: List[Dict[str, Any]] = []
                seen = set()
                for cand in filtered_by_ids:
                    dedupe_key = (cand["sample_idx"], cand["frame_idx"], cand["query_idx"])
                    if dedupe_key in seen:
                        continue
                    seen.add(dedupe_key)
                    deduped.append(cand)
                filtered_by_ids = deduped

            if len(filtered_by_ids) == 0:
                if contract["selected_id_out_of_range"] > 0:
                    id_status = "ids_out_of_range_or_unmatched"
                else:
                    id_status = "ids_provided_but_no_matches"
                id_reason = "no_valid_selected_masks"
            elif contract["selected_id_duplicates"] > 0 and bool(getattr(config, "mm_eomt_external_socket_deduplicate", True)):
                id_status = "ids_matched_deduplicated"
            else:
                id_status = "ids_matched"

            contract["id_matching_status"] = id_status

        filtered_by_words: List[Dict[str, Any]] = []
        word_reason: Optional[str] = None
        word_note: Optional[str] = None
        word_status: Optional[str] = None
        word_entries_without_ids = {
            key: entry for key, entry in word_entries_by_key.items() if key not in query_map
        }
        if len(word_entries_without_ids) > 0:
            word_candidates = [
                cand
                for cand in candidates
                if (cand["sample_idx"], cand["frame_idx"]) in word_entries_without_ids
            ]
            filtered_by_words, word_reason, word_note = self._apply_external_socket_words(
                candidates=word_candidates,
                word_entries_by_key=word_entries_without_ids,
                config=config,
                contract=contract,
            )
            word_status = contract.get("matching_status")
            contract["word_matching_status"] = word_status

        filtered = filtered_by_ids + filtered_by_words
        if bool(getattr(config, "mm_eomt_external_socket_deduplicate", True)):
            deduped: List[Dict[str, Any]] = []
            seen = set()
            for cand in filtered:
                dedupe_key = (cand["sample_idx"], cand["frame_idx"], cand["query_idx"])
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                deduped.append(cand)
            filtered = deduped
        contract["matched_candidate_count"] = int(len(filtered))

        if len(filtered) == 0:
            if id_reason is not None:
                contract["matching_status"] = id_status or "ids_provided_but_no_matches"
                return [], contract, id_reason, id_note
            if word_reason is not None:
                contract["matching_status"] = word_status or "words_no_matched_candidates"
                return [], contract, word_reason, word_note
            contract["matching_status"] = "no_valid_selected_masks"
            return [], contract, "no_valid_selected_masks", None

        if id_status is not None and word_status is not None:
            contract["matching_status"] = "mixed_ids_and_words"
            return filtered, contract, None, "mixed_ids_and_words"
        if id_status is not None:
            contract["matching_status"] = id_status
            return filtered, contract, None, id_note
        contract["matching_status"] = word_status or "words_candidates_retained"
        return filtered, contract, None, word_note

    def _apply_class_aware_word_filter(
        self,
        candidates: List[Dict[str, Any]],
        external_selection: Optional[Dict[str, Any]],
        config: Any,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Optional[str], Optional[str]]:
        contract = {
            "word_filter_path": "class_aware",
            "word_match_enabled": bool(getattr(config, "mm_eomt_word_match_enable", False)),
            "matching_status": "not_requested",
            "input_entry_count": 0,
            "parsed_entry_count": 0,
            "word_entry_count": 0,
            "untouched_candidate_count": 0,
            "matched_candidate_count": int(len(candidates)),
        }

        if not contract["word_match_enabled"]:
            contract["matching_status"] = "word_match_disabled"
            return candidates, contract, None, "word_match_disabled"

        if not isinstance(external_selection, dict):
            contract["matching_status"] = "missing_external_selection"
            return candidates, contract, None, "missing_external_selection"

        entries = external_selection.get("entries", None)
        if not isinstance(entries, list) or len(entries) == 0:
            contract["matching_status"] = "empty_external_selection"
            return candidates, contract, None, "empty_external_selection"
        contract["input_entry_count"] = int(len(entries))

        word_entries_by_key: Dict[Tuple[int, int], Dict[str, Any]] = {}
        for item in entries:
            if not isinstance(item, dict):
                continue
            sample_idx = int(item.get("sample_idx", -1))
            frame_idx = int(item.get("frame_idx", -1))
            if sample_idx < 0 or frame_idx < 0:
                continue
            contract["parsed_entry_count"] += 1

            selected_words = item.get("selected_words", None)
            visible_grounded_words = item.get("visible_grounded_words", None)
            has_word_payload = (
                isinstance(selected_words, (list, tuple))
                and len(selected_words) > 0
            ) or (
                isinstance(visible_grounded_words, (list, tuple))
                and len(visible_grounded_words) > 0
            )
            if not has_word_payload:
                continue

            key = (sample_idx, frame_idx)
            merged_entry = word_entries_by_key.setdefault(
                key,
                {
                    "sample_idx": sample_idx,
                    "frame_idx": frame_idx,
                    "selected_words": [],
                    "visible_grounded_words": [],
                },
            )
            self._merge_socket_word_fields(merged_entry, item)

        contract["word_entry_count"] = int(len(word_entries_by_key))
        if len(word_entries_by_key) == 0:
            contract["matching_status"] = "no_word_entries"
            return candidates, contract, None, "no_word_entries"

        word_keys = set(word_entries_by_key.keys())
        untouched_candidates = [
            cand
            for cand in candidates
            if (int(cand["sample_idx"]), int(cand["frame_idx"])) not in word_keys
        ]
        word_candidates = [
            cand
            for cand in candidates
            if (int(cand["sample_idx"]), int(cand["frame_idx"])) in word_keys
        ]

        contract["untouched_candidate_count"] = int(len(untouched_candidates))
        filtered_word_candidates, word_reason, word_note = self._apply_external_socket_words(
            candidates=word_candidates,
            word_entries_by_key=word_entries_by_key,
            config=config,
            contract=contract,
        )

        filtered = untouched_candidates + filtered_word_candidates
        contract["matched_candidate_count"] = int(len(filtered))
        if len(filtered) == 0 and word_reason is not None:
            return [], contract, word_reason, word_note

        return filtered, contract, None, word_note

    def select(
        self,
        pooled_outputs: Dict[str, Any],
        config: Any,
        external_selection: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        selector_mode = str(getattr(config, "mm_eomt_selector_mode", "class_aware"))
        ordering_mode = str(getattr(config, "mm_eomt_selector_order", "frame_then_score"))
        append_position = str(getattr(config, "mm_eomt_object_block_position", "after_visual"))

        if selector_mode not in self.SUPPORTED_MODES:
            return self._empty_result(
                selector_mode=selector_mode,
                ordering_mode=ordering_mode,
                append_position=append_position,
                fallback_reason="unsupported_selector_mode",
            )

        if ordering_mode not in self.SUPPORTED_ORDERS:
            ordering_mode = "frame_then_score"

        if not isinstance(pooled_outputs, dict):
            return self._empty_result(
                selector_mode=selector_mode,
                ordering_mode=ordering_mode,
                append_position=append_position,
                fallback_reason="missing_pooled_outputs",
            )

        if bool(pooled_outputs.get("pool_skipped", False)):
            reason = str(pooled_outputs.get("skip_reason", "pool_skipped"))
            return self._empty_result(
                selector_mode=selector_mode,
                ordering_mode=ordering_mode,
                append_position=append_position,
                fallback_reason=reason,
            )

        pooled_tokens = pooled_outputs.get("pooled_tokens", None)
        selected_scores = pooled_outputs.get("selected_scores", None)
        selected_indices = pooled_outputs.get("selected_indices", None)
        selected_class_ids = pooled_outputs.get("selected_class_ids", None)
        selected_valid_mask = pooled_outputs.get("selected_valid_mask", None)
        sample_frame_pairs = pooled_outputs.get("aligned_sample_frame_pairs", [])

        try:
            if not torch.is_tensor(pooled_tokens) or pooled_tokens.ndim != 3:
                return self._empty_result(
                    selector_mode=selector_mode,
                    ordering_mode=ordering_mode,
                    append_position=append_position,
                    fallback_reason="invalid_pooled_tokens",
                )

            frame_count, obj_per_frame, _ = pooled_tokens.shape

            if not torch.is_tensor(selected_scores) or selected_scores.shape[:2] != (frame_count, obj_per_frame):
                return self._empty_result(
                    selector_mode=selector_mode,
                    ordering_mode=ordering_mode,
                    append_position=append_position,
                    fallback_reason="invalid_selected_scores",
                )

            if not torch.is_tensor(selected_indices) or selected_indices.shape[:2] != (frame_count, obj_per_frame):
                return self._empty_result(
                    selector_mode=selector_mode,
                    ordering_mode=ordering_mode,
                    append_position=append_position,
                    fallback_reason="invalid_selected_indices",
                )

            if not torch.is_tensor(selected_class_ids) or selected_class_ids.shape[:2] != (frame_count, obj_per_frame):
                selected_class_ids = torch.full_like(selected_indices, -1)

            if not torch.is_tensor(selected_valid_mask) or selected_valid_mask.shape[:2] != (frame_count, obj_per_frame):
                selected_valid_mask = torch.ones_like(selected_indices, dtype=torch.bool)

            drop_no_object = bool(getattr(config, "mm_eomt_selector_drop_no_object", True))
            no_object_class_id = int(getattr(config, "mm_eomt_selector_no_object_class_id", -1))
            keep_stuff = bool(getattr(config, "mm_eomt_selector_keep_stuff", True))
            keep_things = bool(getattr(config, "mm_eomt_selector_keep_things", True))
            max_objects = int(getattr(config, "mm_eomt_object_block_max_objects", 8))
            max_per_frame = int(getattr(config, "mm_eomt_object_block_max_per_frame", 2))

            unlimited_objects = max_objects <= 0
            unlimited_per_frame = max_per_frame <= 0

            parsed_pairs = self._to_pairs(sample_frame_pairs, frame_count)
            stuff_class_ids = self._get_stuff_class_ids(pooled_outputs)
            taxonomy_note = None
            if stuff_class_ids is None and ((not keep_stuff) or (not keep_things)):
                taxonomy_note = "taxonomy_unavailable_thing_stuff_filter_noop"

            external_socket_note = None
            external_selection_contract = {}

            candidates: List[Dict[str, Any]] = []
            for frame_idx in range(frame_count):
                sample_idx, sample_frame_idx = parsed_pairs[frame_idx]
                for obj_idx in range(obj_per_frame):
                    if not bool(selected_valid_mask[frame_idx, obj_idx].item()):
                        continue

                    class_id = int(selected_class_ids[frame_idx, obj_idx].item())
                    if drop_no_object and self._is_no_object(class_id, no_object_class_id):
                        continue

                    if stuff_class_ids is not None:
                        is_stuff = class_id in stuff_class_ids
                        is_thing = class_id >= 0 and (not is_stuff)
                        if is_stuff and (not keep_stuff):
                            continue
                        if is_thing and (not keep_things):
                            continue

                    score = float(selected_scores[frame_idx, obj_idx].item())
                    query_idx = int(selected_indices[frame_idx, obj_idx].item())
                    if query_idx < 0:
                        continue
                    class_name = lookup_class_name(class_id)
                    candidates.append(
                        {
                            "sample_idx": int(sample_idx),
                            "frame_idx": int(sample_frame_idx),
                            "query_idx": query_idx,
                            "class_id": class_id,
                            "class_name": class_name,
                            "score": score,
                            "token": pooled_tokens[frame_idx, obj_idx],
                            "pair": (int(sample_idx), int(sample_frame_idx)),
                        }
                    )

            if selector_mode == "external_socket":
                candidates, external_selection_contract, external_reason, external_socket_note = self._apply_external_socket(
                    candidates=candidates,
                    external_selection=external_selection,
                    config=config,
                )
                if external_reason is not None:
                    empty = self._empty_result(
                        selector_mode=selector_mode,
                        ordering_mode=ordering_mode,
                        append_position=append_position,
                        fallback_reason=external_reason,
                        taxonomy_note=taxonomy_note,
                        external_socket_note=external_socket_note,
                    )
                    empty["external_selection_contract"] = external_selection_contract
                    return empty
            elif bool(getattr(config, "mm_eomt_word_match_enable", False)):
                candidates, external_selection_contract, word_reason, external_socket_note = self._apply_class_aware_word_filter(
                    candidates=candidates,
                    external_selection=external_selection,
                    config=config,
                )
                if word_reason is not None and len(candidates) == 0:
                    empty = self._empty_result(
                        selector_mode=selector_mode,
                        ordering_mode=ordering_mode,
                        append_position=append_position,
                        fallback_reason=word_reason,
                        taxonomy_note=taxonomy_note,
                        external_socket_note=external_socket_note,
                    )
                    empty["external_selection_contract"] = external_selection_contract
                    return empty

            if len(candidates) == 0:
                empty = self._empty_result(
                    selector_mode=selector_mode,
                    ordering_mode=ordering_mode,
                    append_position=append_position,
                    fallback_reason="no_candidates_after_filter",
                    taxonomy_note=taxonomy_note,
                    external_socket_note=external_socket_note,
                )
                empty["external_selection_contract"] = external_selection_contract
                return empty

            if ordering_mode == "score_desc":
                candidates.sort(
                    key=lambda x: (
                        -x["score"],
                        x["sample_idx"],
                        x["frame_idx"],
                        x["query_idx"],
                    )
                )
            else:
                candidates.sort(
                    key=lambda x: (
                        x["sample_idx"],
                        x["frame_idx"],
                        -x["score"],
                        x["query_idx"],
                    )
                )

            per_frame_counts: Dict[Tuple[int, int], int] = {}
            after_per_frame: List[Dict[str, Any]] = []
            truncated_per_frame_count = 0
            for cand in candidates:
                frame_key = (cand["sample_idx"], cand["frame_idx"])
                current = per_frame_counts.get(frame_key, 0)
                if (not unlimited_per_frame) and current >= max_per_frame:
                    truncated_per_frame_count += 1
                    continue
                per_frame_counts[frame_key] = current + 1
                after_per_frame.append(cand)

            per_sample_counts: Dict[int, int] = {}
            final_selected: List[Dict[str, Any]] = []
            truncated_global_count = 0
            for cand in after_per_frame:
                current = per_sample_counts.get(cand["sample_idx"], 0)
                if (not unlimited_objects) and current >= max_objects:
                    truncated_global_count += 1
                    continue
                per_sample_counts[cand["sample_idx"]] = current + 1
                final_selected.append(cand)

            if len(final_selected) == 0:
                empty = self._empty_result(
                    selector_mode=selector_mode,
                    ordering_mode=ordering_mode,
                    append_position=append_position,
                    fallback_reason="all_candidates_truncated",
                    taxonomy_note=taxonomy_note,
                    external_socket_note=external_socket_note,
                )
                empty["external_selection_contract"] = external_selection_contract
                return empty

            selected_tokens_by_sample: Dict[int, torch.Tensor] = {}
            selected_pairs: List[Tuple[int, int]] = []
            selected_scores_list: List[float] = []
            selected_indices_list: List[int] = []

            by_sample: Dict[int, List[Dict[str, Any]]] = {}
            for cand in final_selected:
                by_sample.setdefault(cand["sample_idx"], []).append(cand)
                selected_pairs.append(cand["pair"])
                selected_scores_list.append(cand["score"])
                selected_indices_list.append(cand["query_idx"])

            for sample_idx, sample_candidates in by_sample.items():
                token_stack = torch.stack([c["token"] for c in sample_candidates], dim=0)
                selected_tokens_by_sample[int(sample_idx)] = token_stack

            return {
                "selector_mode": selector_mode,
                "ordering_mode": ordering_mode,
                "append_position": append_position,
                "selected_tokens_by_sample": selected_tokens_by_sample,
                "selected_count": len(final_selected),
                "selected_sample_frame_pairs": selected_pairs,
                "selected_scores": selected_scores_list,
                "selected_indices": selected_indices_list,
                "truncated_per_frame_count": truncated_per_frame_count,
                "truncated_global_count": truncated_global_count,
                "taxonomy_note": taxonomy_note,
                "external_socket_note": external_socket_note,
                "external_selection_contract": external_selection_contract,
                "no_object_class_id": no_object_class_id,
                "fallback_reason": None,
            }
        except Exception as exc:
            return self._empty_result(
                selector_mode=selector_mode,
                ordering_mode=ordering_mode,
                append_position=append_position,
                fallback_reason=f"selector_exception:{type(exc).__name__}",
            )
