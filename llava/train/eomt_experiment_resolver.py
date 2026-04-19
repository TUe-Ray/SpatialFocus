import json
import os
from typing import Any, Dict, Optional, Sequence, Set


class EoMTExperimentConfigError(ValueError):
    """Raised when an EoMT experiment-family config is invalid."""


ALLOWED_TOP_LEVEL_KEYS = {
    "experiment_family",
    "description",
    "metadata",
    "shared",
    "modes",
}

ALLOWED_EOMT_OVERRIDE_KEYS = {
    "eomt_config_path",
    "eomt_ckpt_path",
    "eomt_debug_mode",
    "eomt_debug_max_samples",
    "eomt_debug_top_k_masks",
    "eomt_pool_top_k",
    "eomt_pool_selection",
    "eomt_pool_mask_area_threshold",
    "eomt_pool_score_threshold",
    "mm_eomt_enable_object_block",
    "mm_eomt_object_block_position",
    "mm_eomt_object_block_max_objects",
    "mm_eomt_object_block_max_per_frame",
    "mm_eomt_selector_mode",
    "mm_eomt_selector_keep_stuff",
    "mm_eomt_selector_keep_things",
    "mm_eomt_selector_drop_no_object",
    "mm_eomt_selector_order",
    "mm_eomt_selector_no_object_class_id",
    "mm_eomt_obj_info_mode",
    "mm_eomt_obj_info_text",
    "mm_eomt_obj_info_trainable",
    "mm_eomt_use_object_type_embedding",
    "mm_eomt_external_socket_word_topn",
    "mm_eomt_external_socket_deduplicate",
    "mm_eomt_selective_3d_enable",
    "mm_eomt_selector_score_threshold",
    "mm_eomt_selector_topk",
    "mm_eomt_selective_3d_merge_mode",
    "mm_eomt_selective_3d_gate_type",
    "mm_eomt_selective_3d_floor",
    "mm_eomt_selective_3d_empty_fallback",
    "eomt_smoke_test_enable",
    "eomt_smoke_test_max_samples",
    "eomt_smoke_test_output_dir",
}

TRACKED_SUMMARY_KEYS = [
    "eomt_debug_mode",
    "eomt_debug_max_samples",
    "eomt_debug_top_k_masks",
    "eomt_pool_top_k",
    "eomt_pool_selection",
    "eomt_pool_mask_area_threshold",
    "eomt_pool_score_threshold",
    "mm_eomt_enable_object_block",
    "mm_eomt_obj_info_mode",
    "mm_eomt_object_block_position",
    "mm_eomt_selector_mode",
    "mm_eomt_selector_order",
    "mm_eomt_object_block_max_objects",
    "mm_eomt_object_block_max_per_frame",
    "mm_eomt_use_object_type_embedding",
    "mm_eomt_obj_info_text",
    "mm_eomt_selector_no_object_class_id",
    "mm_eomt_selector_keep_stuff",
    "mm_eomt_selector_keep_things",
    "mm_eomt_selector_drop_no_object",
    "eomt_config_path",
    "eomt_ckpt_path",
    "mm_eomt_selective_3d_enable",
    "mm_eomt_selector_score_threshold",
    "mm_eomt_selector_topk",
    "mm_eomt_selective_3d_merge_mode",
    "mm_eomt_selective_3d_gate_type",
    "mm_eomt_selective_3d_floor",
    "mm_eomt_selective_3d_empty_fallback",
    "eomt_smoke_test_enable",
    "eomt_smoke_test_max_samples",
    "eomt_smoke_test_output_dir",
]


def _normalize_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _normalize_value(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, list):
        return [_normalize_value(v) for v in value]
    return value


def _load_json_file(config_path: str) -> Dict[str, Any]:
    if not config_path:
        raise EoMTExperimentConfigError("EoMT experiment config path is empty.")

    if not os.path.isfile(config_path):
        raise EoMTExperimentConfigError(f"EoMT experiment config file not found: {config_path}")

    try:
        with open(config_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        raise EoMTExperimentConfigError(
            f"Failed to parse EoMT experiment JSON '{config_path}': {exc}"
        ) from exc

    if not isinstance(payload, dict):
        raise EoMTExperimentConfigError(
            f"EoMT experiment config '{config_path}' must contain a JSON object at the top level."
        )

    return payload


def _validate_override_block(block_name: str, overrides: Any) -> Dict[str, Any]:
    if overrides is None:
        return {}
    if not isinstance(overrides, dict):
        raise EoMTExperimentConfigError(
            f"EoMT experiment {block_name} must be a JSON object, got {type(overrides).__name__}."
        )

    unknown_keys = sorted(set(overrides.keys()) - ALLOWED_EOMT_OVERRIDE_KEYS)
    if unknown_keys:
        raise EoMTExperimentConfigError(
            f"Unknown EoMT experiment override keys in {block_name}: {unknown_keys}. "
            f"Allowed keys: {sorted(ALLOWED_EOMT_OVERRIDE_KEYS)}"
        )

    return {str(key): _normalize_value(value) for key, value in overrides.items()}


def _validate_payload(payload: Dict[str, Any], config_path: str) -> Dict[str, Any]:
    unknown_top_level = sorted(set(payload.keys()) - ALLOWED_TOP_LEVEL_KEYS)
    if unknown_top_level:
        raise EoMTExperimentConfigError(
            f"Unknown top-level keys in EoMT experiment config '{config_path}': {unknown_top_level}. "
            f"Allowed keys: {sorted(ALLOWED_TOP_LEVEL_KEYS)}"
        )

    family = payload.get("experiment_family", None)
    if not isinstance(family, str) or family.strip() == "":
        raise EoMTExperimentConfigError(
            f"EoMT experiment config '{config_path}' must define a non-empty string 'experiment_family'."
        )

    description = payload.get("description", "")
    if description is not None and not isinstance(description, str):
        raise EoMTExperimentConfigError(
            f"EoMT experiment config '{config_path}' field 'description' must be a string if provided."
        )

    metadata = payload.get("metadata", None)
    if metadata is not None and not isinstance(metadata, dict):
        raise EoMTExperimentConfigError(
            f"EoMT experiment config '{config_path}' field 'metadata' must be a JSON object if provided."
        )

    shared = _validate_override_block("shared", payload.get("shared", {}))
    modes_raw = payload.get("modes", None)
    if not isinstance(modes_raw, dict) or len(modes_raw) == 0:
        raise EoMTExperimentConfigError(
            f"EoMT experiment config '{config_path}' must define a non-empty 'modes' object."
        )

    modes: Dict[str, Dict[str, Any]] = {}
    for mode_name, overrides in modes_raw.items():
        if not isinstance(mode_name, str) or mode_name.strip() == "":
            raise EoMTExperimentConfigError(
                f"EoMT experiment config '{config_path}' contains an invalid mode name: {mode_name!r}."
            )
        modes[mode_name] = _validate_override_block(f"modes.{mode_name}", overrides)

    return {
        "experiment_family": family,
        "description": description or "",
        "metadata": _normalize_value(metadata or {}),
        "shared": shared,
        "modes": modes,
    }


def extract_explicit_eomt_cli_fields(raw_argv: Optional[Sequence[str]]) -> Set[str]:
    explicit_fields: Set[str] = set()
    if raw_argv is None:
        return explicit_fields

    for token in raw_argv:
        if not isinstance(token, str) or not token.startswith("--"):
            continue
        flag_name = token[2:].split("=", 1)[0]
        if flag_name in ALLOWED_EOMT_OVERRIDE_KEYS:
            explicit_fields.add(flag_name)

    return explicit_fields


def resolve_eomt_experiment_config(model_args: Any, raw_argv: Optional[Sequence[str]] = None) -> Optional[Dict[str, Any]]:
    config_path = getattr(model_args, "eomt_experiment_config_path", None)
    mode_name = getattr(model_args, "eomt_experiment_mode", None)

    if not config_path and not mode_name:
        return None

    if not config_path or not mode_name:
        raise EoMTExperimentConfigError(
            "EoMT experiment selection requires both --eomt_experiment_config_path and --eomt_experiment_mode."
        )

    normalized_config_path = os.path.abspath(config_path)
    payload = _validate_payload(_load_json_file(normalized_config_path), normalized_config_path)

    modes = payload["modes"]
    if mode_name not in modes:
        raise EoMTExperimentConfigError(
            f"Unknown EoMT experiment mode '{mode_name}' for family '{payload['experiment_family']}'. "
            f"Available modes: {sorted(modes.keys())}"
        )

    resolved = dict(payload["shared"])
    resolved.update(modes[mode_name])

    baseline_forced_disable = False
    if mode_name == "baseline":
        if resolved.get("mm_eomt_enable_object_block", False) is not False:
            baseline_forced_disable = True
        resolved["mm_eomt_enable_object_block"] = False

    explicit_cli_fields = extract_explicit_eomt_cli_fields(raw_argv)
    conflicting_fields = []
    duplicate_same_value_fields = []
    for key in sorted(set(resolved.keys()) & explicit_cli_fields):
        cli_value = getattr(model_args, key, None)
        resolved_value = resolved[key]
        if cli_value != resolved_value:
            conflicting_fields.append(
                {
                    "field": key,
                    "cli_value": _normalize_value(cli_value),
                    "resolved_value": _normalize_value(resolved_value),
                }
            )
        else:
            duplicate_same_value_fields.append(key)

    if conflicting_fields:
        raise EoMTExperimentConfigError(
            "EoMT experiment config conflicts with explicitly provided CLI flags: "
            + json.dumps(conflicting_fields, sort_keys=True)
        )

    for key, value in resolved.items():
        setattr(model_args, key, value)

    resolved_settings = {key: _normalize_value(resolved[key]) for key in sorted(resolved.keys())}
    tracked_summary = {
        key: _normalize_value(getattr(model_args, key, None))
        for key in TRACKED_SUMMARY_KEYS
        if hasattr(model_args, key)
    }

    return {
        "experiment_family": payload["experiment_family"],
        "description": payload["description"],
        "config_path": normalized_config_path,
        "mode_name": mode_name,
        "resolved_settings": resolved_settings,
        "tracked_summary": tracked_summary,
        "baseline_forced_disable": baseline_forced_disable,
        "explicit_cli_fields": sorted(explicit_cli_fields),
        "duplicate_same_value_fields": duplicate_same_value_fields,
        "metadata": payload["metadata"],
    }


def write_eomt_experiment_snapshot(summary: Optional[Dict[str, Any]], output_dir: Optional[str]) -> Optional[str]:
    if not summary or not output_dir:
        return None

    os.makedirs(output_dir, exist_ok=True)
    snapshot_path = os.path.join(output_dir, "eomt_experiment_resolved.json")
    with open(snapshot_path, "w", encoding="utf-8") as handle:
        json.dump(_normalize_value(summary), handle, indent=2, sort_keys=True)
        handle.write("\n")
    return snapshot_path