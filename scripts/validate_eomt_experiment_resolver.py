#!/usr/bin/env python3

import argparse
import importlib.util
import json
import os
import sys
import tempfile
from types import SimpleNamespace


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def load_resolver_module():
    module_path = os.path.join(PROJECT_ROOT, "llava", "train", "eomt_experiment_resolver.py")
    spec = importlib.util.spec_from_file_location("eomt_experiment_resolver", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load resolver module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RESOLVER_MODULE = load_resolver_module()
EoMTExperimentConfigError = RESOLVER_MODULE.EoMTExperimentConfigError
resolve_eomt_experiment_config = RESOLVER_MODULE.resolve_eomt_experiment_config


def make_args(config_path, mode_name, **overrides):
    base = {
        "eomt_experiment_config_path": config_path,
        "eomt_experiment_mode": mode_name,
        "eomt_config_path": None,
        "eomt_ckpt_path": None,
        "mm_eomt_enable_object_block": False,
        "mm_eomt_object_block_position": "after_visual",
        "mm_eomt_object_block_max_objects": 8,
        "mm_eomt_object_block_max_per_frame": 2,
        "mm_eomt_selector_mode": "class_aware",
        "mm_eomt_selector_keep_stuff": True,
        "mm_eomt_selector_keep_things": True,
        "mm_eomt_selector_drop_no_object": True,
        "mm_eomt_selector_order": "frame_then_score",
        "mm_eomt_selector_no_object_class_id": -1,
        "mm_eomt_obj_info_mode": "none",
        "mm_eomt_obj_info_text": "Object information from the image:",
        "mm_eomt_obj_info_trainable": True,
        "mm_eomt_use_object_type_embedding": False,
        "mm_eomt_external_socket_word_topn": 1,
        "mm_eomt_external_socket_deduplicate": True,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def assert_true(condition, message):
    if not condition:
        raise AssertionError(message)


def expect_failure(fn, expected_substring):
    try:
        fn()
    except EoMTExperimentConfigError as exc:
        if expected_substring not in str(exc):
            raise AssertionError(
                f"Expected error containing '{expected_substring}', got '{exc}'"
            )
        return str(exc)
    raise AssertionError(f"Expected failure containing '{expected_substring}', but call succeeded")


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main():
    parser = argparse.ArgumentParser(description="Validate EoMT experiment-family resolver.")
    parser.add_argument(
        "--config_path",
        type=str,
        default=os.path.join(PROJECT_ROOT, "configs", "eomt", "eomt_objinfo_round1.json"),
    )
    args = parser.parse_args()

    config_path = os.path.abspath(args.config_path)
    results = []

    summary = resolve_eomt_experiment_config(make_args(config_path, "baseline"), raw_argv=[])
    assert_true(summary["experiment_family"] == "eomt_objinfo_round1", "Unexpected family name")
    assert_true(summary["tracked_summary"]["mm_eomt_enable_object_block"] is False, "Baseline must disable object block")
    results.append({"test": "baseline_mode", "status": "pass"})

    summary = resolve_eomt_experiment_config(make_args(config_path, "eomt_obj_only"), raw_argv=[])
    assert_true(summary["tracked_summary"]["mm_eomt_enable_object_block"] is True, "Object-only mode must enable object block")
    assert_true(summary["tracked_summary"]["mm_eomt_obj_info_mode"] == "none", "Object-only mode must keep obj_info_mode=none")
    results.append({"test": "obj_only_mode", "status": "pass"})

    summary = resolve_eomt_experiment_config(make_args(config_path, "eomt_obj_text_phrase"), raw_argv=[])
    assert_true(summary["tracked_summary"]["mm_eomt_enable_object_block"] is True, "Text-phrase mode must enable object block")
    assert_true(summary["tracked_summary"]["mm_eomt_obj_info_mode"] == "text_phrase", "Text-phrase mode must resolve correctly")
    assert_true(summary["tracked_summary"]["mm_eomt_selector_order"] == "frame_then_score", "Shared defaults must apply")
    results.append({"test": "text_phrase_mode", "status": "pass"})

    summary = resolve_eomt_experiment_config(make_args(config_path, "eomt_obj_learnable"), raw_argv=[])
    assert_true(summary["tracked_summary"]["mm_eomt_enable_object_block"] is True, "Learnable mode must enable object block")
    assert_true(summary["tracked_summary"]["mm_eomt_obj_info_mode"] == "learnable_embedding", "Learnable mode must resolve correctly")
    results.append({"test": "learnable_mode", "status": "pass"})

    expect_failure(
        lambda: resolve_eomt_experiment_config(make_args(config_path, "does_not_exist"), raw_argv=[]),
        "Unknown EoMT experiment mode",
    )
    results.append({"test": "invalid_mode", "status": "pass"})

    expect_failure(
        lambda: resolve_eomt_experiment_config(make_args("/tmp/this_file_should_not_exist.json", "baseline"), raw_argv=[]),
        "config file not found",
    )
    results.append({"test": "missing_file", "status": "pass"})

    with tempfile.TemporaryDirectory(prefix="eomt_experiment_validator_") as temp_dir:
        invalid_key_path = os.path.join(temp_dir, "invalid_key.json")
        write_json(
            invalid_key_path,
            {
                "experiment_family": "broken_family",
                "shared": {
                    "mm_eomt_enable_object_block": True,
                    "not_a_real_key": 1,
                },
                "modes": {
                    "baseline": {
                        "mm_eomt_enable_object_block": False,
                    }
                },
            },
        )
        expect_failure(
            lambda: resolve_eomt_experiment_config(make_args(invalid_key_path, "baseline"), raw_argv=[]),
            "Unknown EoMT experiment override keys",
        )
        results.append({"test": "invalid_key", "status": "pass"})

        conflict_path = os.path.join(temp_dir, "conflict.json")
        write_json(
            conflict_path,
            {
                "experiment_family": "conflict_family",
                "shared": {},
                "modes": {
                    "obj": {
                        "mm_eomt_enable_object_block": True,
                    }
                },
            },
        )
        expect_failure(
            lambda: resolve_eomt_experiment_config(
                make_args(conflict_path, "obj", mm_eomt_enable_object_block=False),
                raw_argv=["--mm_eomt_enable_object_block", "False"],
            ),
            "conflicts with explicitly provided CLI flags",
        )
        results.append({"test": "cli_conflict_detection", "status": "pass"})

    print(json.dumps({"status": "pass", "results": results}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()