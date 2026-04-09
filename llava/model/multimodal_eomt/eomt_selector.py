"""Minimal Phase-1B selector for EoMT pooled object tokens.

This selector consumes existing trusted pooled side outputs and returns a safe,
config-driven subset for optional object-block insertion.
"""

from typing import Any, Dict, List, Optional, Tuple

import torch


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
        #       "selected_words": Optional[List[str]],
        #     },
        #     ...
        #   ]
        # }
        contract = {
            "selected_words_present": False,
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

            if isinstance(selected_words, (list, tuple)) and len(selected_words) > 0:
                word_entries += 1
                contract["selected_words_present"] = True
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

        contract["selected_id_unique"] = int(len(unique_ids_all))

        if len(query_map) == 0:
            if word_entries > 0:
                contract["matching_status"] = "word_matching_deferred_no_ids"
                return [], contract, "no_matched_masks", "word_matching_deferred"
            contract["matching_status"] = "no_query_or_mask_ids"
            return [], contract, "no_valid_selected_masks", None

        out_of_range = 0
        for key, ids in query_map.items():
            valid_ids = candidate_query_ids_by_key.get(key, set())
            out_of_range += sum(1 for query_id in ids if query_id not in valid_ids)
        contract["selected_id_out_of_range"] = int(out_of_range)

        filtered = [
            cand
            for cand in candidates
            if cand["query_idx"] in query_map.get((cand["sample_idx"], cand["frame_idx"]), set())
        ]

        if bool(getattr(config, "mm_eomt_external_socket_deduplicate", True)):
            deduped = []
            seen = set()
            for cand in filtered:
                key = (cand["sample_idx"], cand["frame_idx"], cand["query_idx"])
                if key in seen:
                    continue
                seen.add(key)
                deduped.append(cand)
            filtered = deduped
        contract["matched_candidate_count"] = int(len(filtered))

        if len(filtered) == 0:
            if contract["selected_id_out_of_range"] > 0:
                contract["matching_status"] = "ids_out_of_range_or_unmatched"
            else:
                contract["matching_status"] = "ids_provided_but_no_matches"
            return [], contract, "no_valid_selected_masks", None

        if contract["selected_id_duplicates"] > 0 and bool(getattr(config, "mm_eomt_external_socket_deduplicate", True)):
            contract["matching_status"] = "ids_matched_deduplicated"
        else:
            contract["matching_status"] = "ids_matched"
        return filtered, contract, None, None

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

            drop_no_object = bool(getattr(config, "mm_eomt_selector_drop_no_object", True))
            no_object_class_id = int(getattr(config, "mm_eomt_selector_no_object_class_id", -1))
            keep_stuff = bool(getattr(config, "mm_eomt_selector_keep_stuff", True))
            keep_things = bool(getattr(config, "mm_eomt_selector_keep_things", True))
            max_objects = int(getattr(config, "mm_eomt_object_block_max_objects", 8))
            max_per_frame = int(getattr(config, "mm_eomt_object_block_max_per_frame", 2))

            if max_objects <= 0:
                return self._empty_result(
                    selector_mode=selector_mode,
                    ordering_mode=ordering_mode,
                    append_position=append_position,
                    fallback_reason="max_objects_le_zero",
                )

            if max_per_frame <= 0:
                return self._empty_result(
                    selector_mode=selector_mode,
                    ordering_mode=ordering_mode,
                    append_position=append_position,
                    fallback_reason="max_per_frame_le_zero",
                )

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
                    candidates.append(
                        {
                            "sample_idx": int(sample_idx),
                            "frame_idx": int(sample_frame_idx),
                            "query_idx": query_idx,
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
                    return self._empty_result(
                        selector_mode=selector_mode,
                        ordering_mode=ordering_mode,
                        append_position=append_position,
                        fallback_reason=external_reason,
                        taxonomy_note=taxonomy_note,
                        external_socket_note=external_socket_note,
                    )

            if len(candidates) == 0:
                return self._empty_result(
                    selector_mode=selector_mode,
                    ordering_mode=ordering_mode,
                    append_position=append_position,
                    fallback_reason="no_candidates_after_filter",
                    taxonomy_note=taxonomy_note,
                    external_socket_note=external_socket_note,
                )

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
                if current >= max_per_frame:
                    truncated_per_frame_count += 1
                    continue
                per_frame_counts[frame_key] = current + 1
                after_per_frame.append(cand)

            per_sample_counts: Dict[int, int] = {}
            final_selected: List[Dict[str, Any]] = []
            truncated_global_count = 0
            for cand in after_per_frame:
                current = per_sample_counts.get(cand["sample_idx"], 0)
                if current >= max_objects:
                    truncated_global_count += 1
                    continue
                per_sample_counts[cand["sample_idx"]] = current + 1
                final_selected.append(cand)

            if len(final_selected) == 0:
                return self._empty_result(
                    selector_mode=selector_mode,
                    ordering_mode=ordering_mode,
                    append_position=append_position,
                    fallback_reason="all_candidates_truncated",
                    taxonomy_note=taxonomy_note,
                    external_socket_note=external_socket_note,
                )

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
