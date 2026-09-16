#!/usr/bin/env python
"""Cache-only Common-7 LogME for default-initialized controlled A-prime."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import resource
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.probing.depth_probe_common import load_frame_records
from scripts.probing.run_pre_sft_logme_proxy import (
    DEFAULT_SAMPLE_INDICES,
    SPLIT_SHA256,
    logme_from_statistics,
    select_videos,
    sha256_file,
    write_csv,
    write_json,
)
from scripts.probing.run_pre_sft_logme_proxy_v2_common7 import accumulate_statistics_on_device


COMMON7_LAYERS = (1, 3, 6, 9, 15, 21, 27)
FULL_POLICY_LEVELS = (
    "siglip_output", "fusion_output", "projected_features", "layer_0", "layer_1", "layer_2", "layer_3",
    "layer_6", "layer_9", "layer_12", "layer_15", "layer_18", "layer_21", "layer_24", "layer_27",
)
LABEL = "pre_sft_controlled_a_prime"
DISPLAY_NAME = "Controlled A-prime: pre-projector cross-attention, CUT3R dec12 patch-only"
FORMAL_TARGET_SIGNATURE = "b48e025fefb19e5d7414d2d540b9904a4e21d6de883552699d5e4dc194956d37"
A_PRIME_ARCHITECTURE_COMMIT = "ec0679e"
DEFAULT_CACHE_ROOT = Path("/home/shaoruei/probe_cache/pre_sft_logme_a_prime_common7")
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "pre_sft_logme_proxy_a_prime_common7"


@dataclass(frozen=True)
class Candidate:
    label: str
    display_name: str
    root: Path


def read_existing(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def validate_cache(item: Candidate, sample_indices: Path) -> dict[str, Any]:
    feature_root = item.root / "features" / item.label
    missing = [level for level in FULL_POLICY_LEVELS if not (feature_root / level).is_dir()]
    if missing:
        raise RuntimeError(f"A-prime full-policy cache is incomplete; missing={missing}")
    path = feature_root / "extraction_provenance.json"
    if not path.is_file():
        raise RuntimeError("A-prime extraction provenance is missing")
    payload = json.loads(path.read_text(encoding="utf-8"))
    metadata = payload.get("pre_sft_fusion_metadata") or {}
    samples = payload.get("extraction_samples") or []
    assertion = samples[0].get("first_video_runtime_assertions", {}) if samples else {}
    checks = {
        "model_label": payload.get("model_label") == item.label,
        "mode": payload.get("model_loading_mode") == "pre_sft_fusion",
        "variant": payload.get("experiment_variant") == "controlled_a_prime",
        "seed": payload.get("fusion_init_seed") == 42,
        "no_adapter": payload.get("no_vlm3r_sft_adapter_loaded") is True,
        "no_c1": payload.get("c1_calibration_json") is None and payload.get("c1_calibration_sha256") is None,
        "clean_worktree": payload.get("git_worktree_dirty") is False,
        "split": payload.get("sample_indices") == str(sample_indices) and payload.get("sample_indices_sha256") == SPLIT_SHA256,
        "full_policy": payload.get("requested_feature_levels") == list(FULL_POLICY_LEVELS),
        "metadata": metadata.get("variant") == "controlled_a_prime" and metadata.get("controlled_fusion_id") == "A_prime",
        "default_init": metadata.get("fusion_init_seed") == 42 and metadata.get("c1_enabled") is False,
        "runtime": assertion.get("assessment") == "PASS" and assertion.get("architecture") == "controlled_a_prime",
        "patch_only": assertion.get("camera_tokens_excluded") is True and assertion.get("spatialstack_disabled") is True,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Invalid A-prime pre-SFT cache provenance: {[key for key, ok in checks.items() if not ok]}")
    return payload


def compatible(row: dict[str, Any], layer: int, protocol_sha: str) -> bool:
    try:
        return (
            row.get("status") == "complete"
            and row.get("architecture") == LABEL
            and int(row.get("layer", -1)) == layer
            and row.get("protocol_sha256") == protocol_sha
            and row.get("target_signature") == FORMAL_TARGET_SIGNATURE
            and math.isfinite(float(row["logme"]))
        )
    except (KeyError, TypeError, ValueError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-indices", type=Path, default=DEFAULT_SAMPLE_INDICES)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--block-frames", type=int, default=8)
    parser.add_argument("--cpu-threads", type=int, default=min(32, os.cpu_count() or 1))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.block_frames < 1:
        raise ValueError("--block-frames must be positive")
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA statistics device requested but CUDA is unavailable")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    sample_indices = args.sample_indices.resolve()
    if sha256_file(sample_indices) != SPLIT_SHA256:
        raise RuntimeError("Sample-index SHA differs from the formal train split")
    all_records = load_frame_records(sample_indices, split="train")
    records = select_videos(all_records, None)
    if len(records) != 2012 or len({str(row["video_sample_id"]) for row in records}) != 1006:
        raise RuntimeError("A-prime LogME requires 1,006 train videos / 2,012 selected frames")
    item = Candidate(LABEL, DISPLAY_NAME, args.cache_root.resolve())
    provenance = validate_cache(item, sample_indices)
    provenance_path = item.root / "features" / item.label / "extraction_provenance.json"
    durable_provenance = output / "source_provenance.json"
    shutil.copy2(provenance_path, durable_provenance)
    protocol = {
        "schema_version": "pre_sft_logme_proxy_a_prime_common7_v1",
        "scope": "additional diagnostic; excluded from the frozen formal-five correlation and prior 17-model audit",
        "architecture": "Controlled Pre-Projector Cross-Attention (A-prime)",
        "architecture_commit": A_PRIME_ARCHITECTURE_COMMIT,
        "initialization": "default PyTorch initialization reproduced with official training seed 42",
        "c1_calibration": False,
        "primary_layers": list(COMMON7_LAYERS),
        "primary_score": "mean Common-7 regression LogME",
        "definition": "Bayesian linear regression maximum evidence, normalized log p(y|F)/N",
        "dtype": "float64",
        "alpha_init": 1.0,
        "beta_init": 1.0,
        "sample_indices": str(sample_indices),
        "sample_indices_sha256": SPLIT_SHA256,
        "split": "train",
        "training_videos": 1006,
        "training_frames": 2012,
        "valid_tokens_per_layer": 394352,
        "formal_target_signature": FORMAL_TARGET_SIGNATURE,
        "full_feature_policy": list(FULL_POLICY_LEVELS),
        "source_provenance": str(durable_provenance),
        "source_provenance_sha256": sha256_file(durable_provenance),
        "no_vlm_forward_during_logme": True,
        "no_optimizer": True,
        "no_weight_update": True,
        "no_post_sft_checkpoint": True,
    }
    protocol["protocol_sha256"] = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode("utf-8")).hexdigest()
    write_json(output / "protocol.json", protocol)
    per_layer_path = output / "logme_per_layer.csv"
    rows_by_layer = {int(row["layer"]): row for row in read_existing(per_layer_path) if row.get("architecture") == LABEL}
    started_all = time.perf_counter()
    for layer in COMMON7_LAYERS:
        prior = rows_by_layer.get(layer)
        if prior is not None and compatible(prior, layer, protocol["protocol_sha256"]) and not args.force:
            print(f"[REUSE] A_prime/L{layer}", flush=True)
            continue
        started = time.perf_counter()
        gram, cross, yy, count, frames, signature = accumulate_statistics_on_device(
            item, layer, records, device=device, block_frames=args.block_frames
        )
        if signature != FORMAL_TARGET_SIGNATURE or count != 394352 or frames != 2012:
            raise RuntimeError(f"A_prime/L{layer}: target/mask alignment differs from formal Common-7")
        result = logme_from_statistics(gram, cross, yy, count)
        rows_by_layer[layer] = {
            "architecture": LABEL,
            "display_name": DISPLAY_NAME,
            "layer": layer,
            "feature_level": f"layer_{layer}",
            "cache_root": str(item.root),
            "cache_provenance": str(durable_provenance),
            "status": "complete",
            "protocol_sha256": protocol["protocol_sha256"],
            "logme": result["logme"],
            "alpha": result["alpha"],
            "beta": result["beta"],
            "gamma": result["gamma"],
            "iterations": result["iterations"],
            "converged": result["converged"],
            "residual_sq": result["residual_sq"],
            "minimum_eigenvalue": result["minimum_eigenvalue"],
            "valid_tokens": count,
            "feature_dimension": int(gram.shape[0]),
            "training_videos": 1006,
            "training_frames": frames,
            "target_signature": signature,
            "runtime_seconds": time.perf_counter() - started,
            "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "statistics_device": str(device),
            "fusion_init_seed": 42,
            "c1_calibration_applied": False,
            "extraction_provenance_git_commit": provenance["git_commit"],
        }
        write_csv(per_layer_path, [rows_by_layer[value] for value in sorted(rows_by_layer)])
        print(f"[DONE] A_prime/L{layer} N={count} LogME={result['logme']:.12f}", flush=True)
        del gram, cross, yy
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
    rows = [rows_by_layer[layer] for layer in COMMON7_LAYERS]
    mean = sum(float(row["logme"]) for row in rows) / len(rows)
    score = {
        "architecture": LABEL,
        "display_name": DISPLAY_NAME,
        "status": "complete",
        "common7_mean_logme": mean,
        "formal_or_diagnostic": "diagnostic",
        "post_sft_vsi_score": None,
        "post_sft_vsi_status": "final evaluation pending; milestone scores are not substituted",
        "valid_tokens_per_layer": 394352,
        "training_videos": 1006,
    }
    write_csv(output / "logme_architecture_scores.csv", [score])
    summary = {
        "protocol": protocol,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "scores": [score],
        "per_layer_rows": len(rows),
        "post_sft_milestone_vsi_diagnostics": {"1_percent": 37.4551, "5_percent": 46.5862, "25_percent": 53.3904, "50_percent": 56.1521},
        "runtime_memory": {
            "this_invocation_wall_seconds": time.perf_counter() - started_all,
            "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "statistics_device": str(device),
        },
    }
    write_json(output / "logme_summary.json", summary)
    print(json.dumps({"architecture": LABEL, "common7_mean_logme": mean, "output": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
