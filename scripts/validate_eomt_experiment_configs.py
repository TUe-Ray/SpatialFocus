#!/usr/bin/env python3
"""Validate EoMT experiment-family configs and selective-3D legacy aliases."""

import importlib.util
import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RESOLVER_PATH = PROJECT_ROOT / "llava" / "train" / "eomt_experiment_resolver.py"


def _load_resolver_module():
    spec = importlib.util.spec_from_file_location("eomt_experiment_resolver", RESOLVER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load resolver module from {RESOLVER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


resolver_module = _load_resolver_module()
resolve_eomt_experiment_config = resolver_module.resolve_eomt_experiment_config


def resolve_family(config_name: str, mode_name: str):
    model_args = SimpleNamespace(
        eomt_experiment_config_path=str(PROJECT_ROOT / "configs" / "eomt" / config_name),
        eomt_experiment_mode=mode_name,
    )
    return resolve_eomt_experiment_config(model_args=model_args, raw_argv=[])


def assert_true(condition: bool, message: str):
    if not condition:
        raise AssertionError(message)


def case_objinfo_unchanged():
    summary = resolve_family("eomt_objinfo_round1.json", "eomt_obj_text_phrase")
    resolved = summary["resolved_settings"]
    assert_true(summary["experiment_family"] == "eomt_objinfo_round1", "objinfo family mismatch")
    assert_true(resolved["mm_eomt_enable_object_block"] is True, "objinfo object block should be enabled")
    assert_true(resolved["mm_eomt_obj_info_mode"] == "text_phrase", "objinfo mode should remain text_phrase")
    assert_true(resolved["mm_eomt_selector_mode"] == "class_aware", "objinfo selector should stay class_aware")
    return {
        "family": summary["experiment_family"],
        "mode": summary["mode_name"],
        "selector_mode": resolved["mm_eomt_selector_mode"],
        "obj_info_mode": resolved["mm_eomt_obj_info_mode"],
    }


def case_selective_new_namespace():
    summary = resolve_family("eomt_selective_3d_round1.json", "selective_soft_with_floor")
    resolved = summary["resolved_settings"]
    assert_true(summary["experiment_family"] == "eomt_selective_3d_round1", "selective family mismatch")
    assert_true(resolved["mm_eomt_selective_3d_enable"] is True, "selective 3D should be enabled")
    assert_true(
        resolved["mm_eomt_selective_3d_selector_mode"] == "confidence",
        "selective selector mode should use the new namespace",
    )
    assert_true(
        "mm_eomt_selector_score_threshold" not in resolved,
        "selective config should not expose legacy threshold keys after resolution",
    )
    return {
        "family": summary["experiment_family"],
        "mode": summary["mode_name"],
        "selector_mode": resolved["mm_eomt_selective_3d_selector_mode"],
        "score_threshold": resolved["mm_eomt_selective_3d_score_threshold"],
    }


def case_combined_family():
    summary = resolve_family("eomt_combined_round1.json", "combined_soft_with_floor")
    resolved = summary["resolved_settings"]
    assert_true(summary["experiment_family"] == "eomt_combined_round1", "combined family mismatch")
    assert_true(resolved["mm_eomt_enable_object_block"] is True, "combined family should enable object block")
    assert_true(resolved["mm_eomt_selective_3d_enable"] is True, "combined family should enable selective 3D")
    assert_true(
        resolved["mm_eomt_selector_mode"] == "class_aware",
        "combined family should keep object-block selector settings",
    )
    assert_true(
        resolved["mm_eomt_selective_3d_selector_mode"] == "confidence",
        "combined family should use selective-3D selector namespace",
    )
    return {
        "family": summary["experiment_family"],
        "mode": summary["mode_name"],
        "object_block_enabled": resolved["mm_eomt_enable_object_block"],
        "selective_3d_enabled": resolved["mm_eomt_selective_3d_enable"],
    }


def case_legacy_selective_alias():
    legacy_payload = {
        "experiment_family": "eomt_selective_3d_round1",
        "description": "Legacy selective-3D key alias validation.",
        "shared": {
            "mm_eomt_selector_mode": "confidence",
            "mm_eomt_selector_score_threshold": 0.77,
            "mm_eomt_selector_topk": 3,
            "mm_eomt_selector_class_type": "things",
            "mm_eomt_word_match_enable": False,
            "mm_eomt_word_match_source": "selected_words",
            "mm_eomt_word_match_mode": "exact_alias",
            "mm_eomt_word_match_no_match": "filter_out",
            "mm_eomt_word_match_similarity_threshold": 0.9
        },
        "modes": {
            "legacy_selective": {
                "mm_eomt_selective_3d_enable": True,
                "mm_eomt_selective_3d_gate_type": "soft"
            }
        }
    }

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as handle:
        json.dump(legacy_payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temp_path = handle.name

    try:
        model_args = SimpleNamespace(
            eomt_experiment_config_path=temp_path,
            eomt_experiment_mode="legacy_selective",
        )
        summary = resolve_eomt_experiment_config(model_args=model_args, raw_argv=[])
    finally:
        os.unlink(temp_path)

    resolved = summary["resolved_settings"]
    assert_true(
        resolved["mm_eomt_selective_3d_selector_mode"] == "confidence",
        "legacy selective selector mode should alias into the new namespace",
    )
    assert_true(
        resolved["mm_eomt_selective_3d_score_threshold"] == 0.77,
        "legacy selective threshold should alias into the new namespace",
    )
    assert_true(
        resolved["mm_eomt_selective_3d_word_match_source"] == "selected_words",
        "legacy selective word-match source should alias into the new namespace",
    )
    assert_true(
        "mm_eomt_selector_score_threshold" not in resolved,
        "legacy selective threshold key should not survive normalization",
    )
    return {
        "family": summary["experiment_family"],
        "mode": summary["mode_name"],
        "aliased_threshold": resolved["mm_eomt_selective_3d_score_threshold"],
        "aliased_word_match_source": resolved["mm_eomt_selective_3d_word_match_source"],
    }


def main():
    report = {
        "objinfo_unchanged": case_objinfo_unchanged(),
        "selective_new_namespace": case_selective_new_namespace(),
        "combined_family": case_combined_family(),
        "legacy_selective_alias": case_legacy_selective_alias(),
    }
    print(
        json.dumps(
            {
                "verdict": "validated",
                "results": report,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
