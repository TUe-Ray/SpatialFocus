#!/usr/bin/env python
"""Cache-only common-seven LogME for controlled-fusion B/C/D/E/H.

The controlled-fusion extension is outside the frozen five-candidate formal
study.  This runner reads only regenerated, provenance-checked pre-SFT feature
caches and imports the already-smoke-validated float64 sufficient-statistics
LogME implementation.  It neither loads a VLM nor constructs an optimizer.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import resource
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
    average_ranks,
    logme_from_statistics,
    select_videos,
    sha256_file,
    target_and_feature,
    write_csv,
    write_json,
)
from scripts.probing.run_pre_sft_logme_proxy_v2_common7 import accumulate_statistics_on_device


COMMON7_LAYERS = (1, 3, 6, 9, 15, 21, 27)
FULL_POLICY_LEVELS = (
    "siglip_output", "fusion_output", "projected_features", "layer_0", "layer_1", "layer_2", "layer_3",
    "layer_6", "layer_9", "layer_12", "layer_15", "layer_18", "layer_21", "layer_24", "layer_27",
)
CONTROLLED_COMMIT = "f4e2259451ecf12d4da9a87dce121639642a0524"
FORMAL_TARGET_SIGNATURE = "b48e025fefb19e5d7414d2d540b9904a4e21d6de883552699d5e4dc194956d37"
DEFAULT_CACHE_BASE = Path("/home/shaoruei/probe_cache/controlled_fusion_logme_recache_v1")
DEFAULT_C1_MANIFEST = Path("/home/shaoruei/probe_outputs/controlled_fusion_pre_sft_v3/c1/artifact_manifest.json")
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "pre_sft_logme_proxy_controlled_fusion_v1"


@dataclass(frozen=True)
class Candidate:
    identifier: str
    label: str
    display_name: str
    c1_sha256: str
    root: Path


SPECS = {
    "B": ("c1_controlled_b", "B: pre-projector add, CUT3R dec12", "a4f9e5411cd065e4acfff11376c142ccbcb0b46263a7a035b4e2dea0ef24f363"),
    "C": ("c1_controlled_c", "C: cross-attention, CUT3R dec12 -> L0", "2bc7aabb442502edc3ca624cc530d3bfed0c741b360bbeadd674f79d7bae81b1"),
    "D": ("c1_controlled_d", "D: additive, CUT3R dec12 -> L0", "cb367bed7d60b15d0a51d512cad4aebf4840e8686655579a640ec724b17d07e7"),
    "E": ("c1_controlled_e", "E: additive, CUT3R dec12 -> L0/L1/L2, site projectors", "7680486b2f43a5bd71fe4537f3a2a28a29615ab9309aa4162d0cec72c8de8fbf"),
    "H": ("c1_controlled_h", "H: cross-attention, CUT3R dec12 -> L0/L1/L2", "7b6661934b894abfe0fc54eaf1e24d7677d3b69d3bd7dbfca80a488c00073259"),
}


def candidate(identifier: str, cache_base: Path) -> Candidate:
    label, display_name, c1_sha256 = SPECS[identifier]
    return Candidate(identifier, label, display_name, c1_sha256, cache_base / identifier)


def provenance_path(item: Candidate) -> Path:
    return item.root / "features" / item.label / "extraction_provenance.json"


def validate_cache(item: Candidate, sample_indices: Path, c1_manifest: dict[str, Any]) -> dict[str, Any]:
    feature_root = item.root / "features" / item.label
    missing = [level for level in FULL_POLICY_LEVELS if not (feature_root / level).is_dir()]
    if missing:
        raise RuntimeError(f"{item.identifier}: incomplete regenerated full-policy cache; missing {missing}")
    path = provenance_path(item)
    if not path.is_file():
        raise RuntimeError(f"{item.identifier}: missing feature extraction provenance")
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_artifact = c1_manifest["artifacts"][item.identifier]
    if expected_artifact.get("sha256") != item.c1_sha256:
        raise RuntimeError(f"{item.identifier}: C1 manifest SHA differs from predeclared controlled C1 artifact")
    checks = {
        "model_label": payload.get("model_label") == item.label,
        "mode": payload.get("model_loading_mode") == "pre_sft_fusion",
        "variant": payload.get("experiment_variant") == item.label,
        "no_adapter": payload.get("no_vlm3r_sft_adapter_loaded") is True,
        "split": payload.get("sample_indices") == str(sample_indices) and payload.get("sample_indices_sha256") == SPLIT_SHA256,
        "c1_sha": payload.get("c1_calibration_sha256") == item.c1_sha256,
        "commit": payload.get("git_commit") == CONTROLLED_COMMIT,
    }
    if not all(checks.values()):
        raise RuntimeError(f"{item.identifier}: invalid pre-SFT cache provenance: {[key for key, value in checks.items() if not value]}")
    return payload


def compatible(row: dict[str, Any], item: Candidate, layer: int, protocol_sha: str) -> bool:
    try:
        return (
            row.get("status") == "complete" and row.get("architecture") == item.label and int(row.get("layer", -1)) == layer
            and row.get("protocol_sha256") == protocol_sha and row.get("target_signature") == FORMAL_TARGET_SIGNATURE
            and math.isfinite(float(row["logme"]))
        )
    except (KeyError, TypeError, ValueError):
        return False


def read_existing(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=tuple(SPECS), required=True, help="Run exactly one cache-ready controlled candidate.")
    parser.add_argument("--cache-base", type=Path, default=DEFAULT_CACHE_BASE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--c1-manifest", type=Path, default=DEFAULT_C1_MANIFEST)
    parser.add_argument("--sample-indices", type=Path, default=DEFAULT_SAMPLE_INDICES)
    parser.add_argument("--device", default="cpu", help="Statistics device only; never loads a VLM.")
    parser.add_argument("--block-frames", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--cpu-threads", type=int, default=min(32, os.cpu_count() or 1))
    args = parser.parse_args()
    if args.block_frames < 1:
        raise ValueError("--block-frames must be positive")
    torch.set_num_threads(args.cpu_threads)
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    cache_base = args.cache_base.resolve(); sample_indices = args.sample_indices.resolve()
    if sha256_file(sample_indices) != SPLIT_SHA256:
        raise RuntimeError("Sample-index SHA differs from the formal train split")
    all_records = load_frame_records(sample_indices, split="train")
    records = select_videos(all_records, None)
    if len(records) != 2012 or len({str(row["video_sample_id"]) for row in records}) != 1006:
        raise RuntimeError("Formal LogME input must be 1,006 train videos / 2,012 selected frames")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {device}")
    manifest = json.loads(args.c1_manifest.read_text(encoding="utf-8"))
    if manifest.get("post_sft_state_loaded") is not False or manifest.get("git_commit") != CONTROLLED_COMMIT:
        raise RuntimeError("Controlled C1 manifest does not prove the expected pre-SFT state/commit")
    item = candidate(args.candidate, cache_base)
    provenance = validate_cache(item, sample_indices, manifest)
    protocol = {
        "schema_version": "pre_sft_logme_proxy_controlled_fusion_v1",
        "scope": "controlled-fusion B/C/D/E/H diagnostics; excluded from frozen five-model formal correlation",
        "definition": "Bayesian linear regression maximum evidence, normalized log p(y|F)/N",
        "dtype": "float64", "alpha_init": 1.0, "beta_init": 1.0, "primary_layers": list(COMMON7_LAYERS),
        "sample_indices": str(sample_indices), "sample_indices_sha256": SPLIT_SHA256, "split": "train",
        "training_videos": 1006, "training_frames": 2012, "valid_tokens_per_layer": 394352,
        "formal_target_signature": FORMAL_TARGET_SIGNATURE, "controlled_source_commit": CONTROLLED_COMMIT,
        "c1_manifest": str(args.c1_manifest.resolve()), "c1_manifest_sha256": sha256_file(args.c1_manifest),
        "full_feature_policy": list(FULL_POLICY_LEVELS), "no_vlm_forward": True, "no_optimizer": True, "no_post_sft_checkpoint": True,
    }
    protocol["protocol_sha256"] = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode("utf-8")).hexdigest()
    write_json(output / "protocol.json", protocol)
    per_layer_path = output / "logme_per_layer.csv"; existing = read_existing(per_layer_path)
    rows_by_key = {(row.get("architecture"), int(row.get("layer", -1))): row for row in existing}
    started_all = time.perf_counter()
    for layer in COMMON7_LAYERS:
        prior = rows_by_key.get((item.label, layer))
        if prior is not None and compatible(prior, item, layer, protocol["protocol_sha256"]) and not args.force:
            print(f"[REUSE] {item.identifier}/L{layer}", flush=True); continue
        started = time.perf_counter()
        gram, cross, yy, count, frames, signature = accumulate_statistics_on_device(item, layer, records, device=device, block_frames=args.block_frames)
        if signature != FORMAL_TARGET_SIGNATURE or count != 394352 or frames != 2012:
            raise RuntimeError(f"{item.identifier}/L{layer}: target/mask alignment differs from formal common-seven pool")
        result = logme_from_statistics(gram, cross, yy, count)
        rows_by_key[(item.label, layer)] = {
            "architecture": item.label, "controlled_id": item.identifier, "display_name": item.display_name, "layer": layer,
            "feature_level": f"layer_{layer}", "cache_root": str(item.root), "cache_provenance": str(provenance_path(item)),
            "status": "complete", "protocol_sha256": protocol["protocol_sha256"], "logme": result["logme"],
            "alpha": result["alpha"], "beta": result["beta"], "gamma": result["gamma"], "iterations": result["iterations"],
            "converged": result["converged"], "residual_sq": result["residual_sq"], "minimum_eigenvalue": result["minimum_eigenvalue"],
            "valid_tokens": count, "feature_dimension": int(gram.shape[0]), "training_videos": 1006, "training_frames": frames,
            "target_signature": signature, "runtime_seconds": time.perf_counter() - started,
            "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024, "statistics_device": str(device),
            "c1_calibration_sha256": item.c1_sha256, "extraction_provenance_git_commit": provenance["git_commit"],
        }
        write_csv(per_layer_path, sorted(rows_by_key.values(), key=lambda row: (str(row["architecture"]), int(row["layer"]))))
        print(f"[DONE] {item.identifier}/L{layer} N={count} LogME={result['logme']:.12f} time={time.perf_counter() - started:.1f}s", flush=True)
        del gram, cross, yy
        if device.type == "cuda": torch.cuda.empty_cache()
        gc.collect()
    rows = sorted(rows_by_key.values(), key=lambda row: (str(row["architecture"]), int(row["layer"])))
    scores: list[dict[str, Any]] = []
    for identifier in SPECS:
        target = candidate(identifier, cache_base)
        values = [rows_by_key.get((target.label, layer)) for layer in COMMON7_LAYERS]
        if all(values):
            scores.append({"controlled_id": identifier, "architecture": target.label, "display_name": target.display_name,
                           "status": "complete", "common7_mean_logme": sum(float(row["logme"]) for row in values if row is not None) / len(COMMON7_LAYERS),
                           "formal_or_diagnostic": "diagnostic", "valid_tokens_per_layer": 394352, "training_videos": 1006})
        else:
            scores.append({"controlled_id": identifier, "architecture": target.label, "display_name": target.display_name,
                           "status": "pending_missing_full_cache_or_logme", "common7_mean_logme": None, "formal_or_diagnostic": "diagnostic"})
    complete = [row for row in scores if row["status"] == "complete"]
    for score, rank in zip(sorted(complete, key=lambda row: -float(row["common7_mean_logme"])), average_ranks([float(row["common7_mean_logme"]) for row in sorted(complete, key=lambda row: -float(row["common7_mean_logme"]))], higher_is_better=True)):
        score["rank_among_completed_controlled_diagnostics"] = rank
    write_csv(output / "logme_architecture_scores.csv", scores)
    summary = {
        "protocol": protocol, "completed_at": datetime.now(timezone.utc).isoformat(), "scores": scores,
        "per_layer_rows": len(rows), "current_candidate": item.identifier, "provenance": provenance,
        "runtime_memory": {"this_invocation_wall_seconds": time.perf_counter() - started_all,
                           "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024, "statistics_device": str(device)},
    }
    write_json(output / "logme_summary.json", summary)
    print(f"[COMPLETE] candidate={item.identifier} output={output}", flush=True)


if __name__ == "__main__":
    main()
