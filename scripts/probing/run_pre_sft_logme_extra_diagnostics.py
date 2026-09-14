#!/usr/bin/env python
"""Run cache-only common-seven LogME for explicitly non-formal diagnostics.

This runner deliberately keeps its outputs separate from
``pre_sft_logme_proxy_v2_common7``.  The frozen five-candidate C1 study and
its LogME--VSI correlation therefore remain unchanged.  It reuses the exact
float64 sufficient-statistics implementation and the same train-video split,
but evaluates only the predeclared extra representations listed below.
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
PROTOCOL_NAME = "pre_sft_logme_proxy_v2_common7_extra_diagnostics"
DEFAULT_OUTPUT = REPO_ROOT / "logs" / PROTOCOL_NAME
# This was recorded by the completed formal common-seven run.  It is a strict
# guard that these extra caches use precisely its target values and masks.
FORMAL_TARGET_SIGNATURE = "b48e025fefb19e5d7414d2d540b9904a4e21d6de883552699d5e4dc194956d37"


@dataclass(frozen=True)
class Candidate:
    label: str
    display_name: str
    root: Path
    allowed_loading_mode: str


CANDIDATES = (
    Candidate(
        "c1_vlm3r_eomt_selective",
        "C1 VLM3R selective fusion (EoMT K/V gate)",
        Path("/home/shaoruei/probe_cache/c1_eomt_selective_depth_probe_v1/full"),
        "pre_sft_fusion",
    ),
    Candidate(
        "pre_sft_base_vlm",
        "Plain pre-SFT base VLM",
        Path("/home/shaoruei/probe_cache/scannet_depth_layers_v1/full"),
        "pre_sft_base_vlm",
    ),
)


def provenance_path(candidate: Candidate) -> Path:
    return candidate.root / "features" / candidate.label / "extraction_provenance.json"


def validate_provenance(candidate: Candidate, sample_indices: Path) -> dict[str, Any]:
    """Accept only a cache that explicitly proves its declared pre-SFT state."""
    path = provenance_path(candidate)
    if not path.is_file():
        raise RuntimeError(f"{candidate.label}: missing extraction provenance: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("model_label") != candidate.label:
        raise RuntimeError(f"{candidate.label}: provenance model label mismatch")
    if payload.get("model_loading_mode") != candidate.allowed_loading_mode:
        raise RuntimeError(f"{candidate.label}: unexpected model-loading mode {payload.get('model_loading_mode')!r}")
    if payload.get("no_vlm3r_sft_adapter_loaded") is not True:
        raise RuntimeError(f"{candidate.label}: provenance does not prove that no post-SFT adapter was loaded")
    if payload.get("sample_indices_sha256") != SPLIT_SHA256 or sha256_file(sample_indices) != SPLIT_SHA256:
        raise RuntimeError(f"{candidate.label}: sample-index SHA-256 is not the formal train split")
    if payload.get("sample_indices") != str(sample_indices):
        raise RuntimeError(f"{candidate.label}: provenance references a different sample-index path")
    if candidate.label == "c1_vlm3r_eomt_selective" and payload.get("eomt_selective_kv_gate") is not True:
        raise RuntimeError("Selective-fusion cache does not prove its EoMT selective K/V gate")
    return payload


def missing_layers(candidate: Candidate) -> list[int]:
    feature_root = candidate.root / "features" / candidate.label
    return [layer for layer in COMMON7_LAYERS if not (feature_root / f"layer_{layer}").is_dir()]


def compatible(row: dict[str, Any], candidate: Candidate, layer: int, protocol_sha: str) -> bool:
    try:
        return (
            row.get("status") == "complete"
            and row.get("architecture") == candidate.label
            and int(row.get("layer", -1)) == layer
            and row.get("protocol_sha256") == protocol_sha
            and row.get("target_signature") == FORMAL_TARGET_SIGNATURE
            and math.isfinite(float(row["logme"]))
        )
    except (KeyError, TypeError, ValueError):
        return False


def load_existing(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def draw_figures(output: Path, rows: list[dict[str, Any]], means: list[dict[str, Any]]) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    figures: list[str] = []
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.bar([row["display_name"] for row in means], [float(row["logme_primary"]) for row in means])
    axis.set_ylabel("Pre-SFT common-seven mean LogME")
    axis.set_title("Extra diagnostic representations (not formal candidates)")
    axis.tick_params(axis="x", labelrotation=15, labelsize=8)
    figure.tight_layout(); figure.savefig(output / "logme_extra_architecture_scores.png", dpi=180); plt.close(figure)
    figures.append("logme_extra_architecture_scores.png")

    figure, axis = plt.subplots(figsize=(9, 3.2))
    matrix = [
        [float(next(row["logme"] for row in rows if row["architecture"] == candidate.label and int(row["layer"]) == layer)) for layer in COMMON7_LAYERS]
        for candidate in CANDIDATES
    ]
    image = axis.imshow(matrix, aspect="auto")
    figure.colorbar(image, ax=axis, label="Normalized LogME")
    axis.set_xticks(range(len(COMMON7_LAYERS)), [f"L{layer}" for layer in COMMON7_LAYERS])
    axis.set_yticks(range(len(CANDIDATES)), [candidate.display_name for candidate in CANDIDATES])
    axis.set_title("Extra diagnostic LogME layer profiles")
    figure.tight_layout(); figure.savefig(output / "logme_extra_layer_heatmap.png", dpi=180); plt.close(figure)
    figures.append("logme_extra_layer_heatmap.png")
    return figures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-indices", type=Path, default=DEFAULT_SAMPLE_INDICES)
    parser.add_argument("--video-limit", type=int, default=None, help="Future deterministic video-level subset; default uses all formal train videos.")
    parser.add_argument("--device", default="cpu", help="Statistics device only; never loads a VLM.")
    parser.add_argument("--block-frames", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--cpu-threads", type=int, default=min(32, os.cpu_count() or 1))
    args = parser.parse_args()
    if args.block_frames < 1:
        raise ValueError("--block-frames must be positive")
    torch.set_num_threads(args.cpu_threads)
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    sample_indices = args.sample_indices.resolve()
    all_records = load_frame_records(sample_indices, split="train")
    if len(all_records) != 2012 or len({str(row["video_sample_id"]) for row in all_records}) != 1006:
        raise RuntimeError("Formal split must contain exactly 1,006 train videos / 2,012 selected frames")
    records = select_videos(all_records, args.video_limit)
    video_count = len({str(row["video_sample_id"]) for row in records})
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {device}")
    provenance = {candidate.label: validate_provenance(candidate, sample_indices) for candidate in CANDIDATES}
    unavailable = {candidate.label: missing_layers(candidate) for candidate in CANDIDATES}
    if any(unavailable.values()):
        raise RuntimeError(f"Extra diagnostic cache unexpectedly lacks common-seven layers: {unavailable}")
    protocol = {
        "schema_version": PROTOCOL_NAME,
        "scope": "extra diagnostics only; excluded from the formal five-candidate v2 ranking and all its VSI correlations",
        "definition": "Bayesian linear-regression maximum evidence, normalized log p(y|F)/N",
        "dtype": "float64", "alpha_init": 1.0, "beta_init": 1.0, "primary_layers": list(COMMON7_LAYERS),
        "sample_indices": str(sample_indices), "sample_indices_sha256": sha256_file(sample_indices),
        "split": "train", "training_videos": video_count, "training_frames": len(records), "video_limit": args.video_limit,
        "target_processing": "exact existing cached depth target, validity mask, and token/pixel alignment",
        "formal_common7_target_signature": FORMAL_TARGET_SIGNATURE,
        "no_vlm_forward": True, "no_post_sft_checkpoint": True, "no_optimizer": True,
        "candidates": [
            {"label": candidate.label, "display_name": candidate.display_name, "cache_root": str(candidate.root),
             "cache_provenance": str(provenance_path(candidate)), "model_loading_mode": candidate.allowed_loading_mode}
            for candidate in CANDIDATES
        ],
    }
    protocol["protocol_sha256"] = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode("utf-8")).hexdigest()
    write_json(output / "protocol.json", protocol)
    per_layer_path = output / "logme_per_layer.csv"
    existing = load_existing(per_layer_path)
    rows: list[dict[str, Any]] = []
    started_all = time.perf_counter()
    for candidate in CANDIDATES:
        for layer in COMMON7_LAYERS:
            prior = next((row for row in existing if compatible(row, candidate, layer, protocol["protocol_sha256"])), None)
            if prior is not None and not args.force:
                rows.append(prior)
                print(f"[REUSE] {candidate.label}/L{layer}", flush=True)
                continue
            started = time.perf_counter()
            gram, cross, yy, count, frames, signature = accumulate_statistics_on_device(
                candidate, layer, records, device=device, block_frames=args.block_frames
            )
            if signature != FORMAL_TARGET_SIGNATURE:
                raise RuntimeError(
                    f"{candidate.label}/L{layer}: target/mask signature differs from formal common-seven run: {signature}"
                )
            # Keep the O(D^3) eigendecomposition on the requested statistics
            # device as well; this is still the unchanged float64 algorithm.
            result = logme_from_statistics(gram, cross, yy, count)
            row = {
                "architecture": candidate.label, "display_name": candidate.display_name, "layer": layer,
                "feature_level": f"layer_{layer}", "cache_root": str(candidate.root),
                "cache_provenance": str(provenance_path(candidate)), "cache_source_identifier": str(provenance_path(candidate)),
                "status": "complete", "protocol_sha256": protocol["protocol_sha256"],
                "logme": result["logme"], "alpha": result["alpha"], "beta": result["beta"],
                "iterations": result["iterations"], "converged": result["converged"], "gamma": result["gamma"],
                "residual_sq": result["residual_sq"], "minimum_eigenvalue": result["minimum_eigenvalue"],
                "valid_tokens": count, "feature_dimension": int(gram.shape[0]), "training_videos": video_count,
                "training_frames": frames, "target_signature": signature,
                "runtime_seconds": time.perf_counter() - started,
                "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                "statistics_device": str(device),
            }
            rows.append(row); write_csv(per_layer_path, rows)
            print(f"[DONE] {candidate.label}/L{layer} N={count} LogME={result['logme']:.12f} time={row['runtime_seconds']:.1f}s", flush=True)
            del gram, cross, yy
            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
    write_csv(per_layer_path, rows)
    scores: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        values = [float(next(row["logme"] for row in rows if row["architecture"] == candidate.label and int(row["layer"]) == layer)) for layer in COMMON7_LAYERS]
        scores.append({
            "architecture": candidate.label, "display_name": candidate.display_name,
            "scope": "extra_diagnostic_not_formal_v2_candidate", "logme_primary": sum(values) / len(values),
            "layer_count": len(COMMON7_LAYERS), "valid_tokens_per_layer": int(next(row["valid_tokens"] for row in rows if row["architecture"] == candidate.label)),
            "training_videos": video_count, "target_signature": FORMAL_TARGET_SIGNATURE,
        })
    for score, rank in zip(scores, average_ranks([float(row["logme_primary"]) for row in scores], higher_is_better=True)):
        score["diagnostic_rank_within_extra_pair"] = rank
    write_csv(output / "logme_architecture_scores.csv", scores)
    figures = draw_figures(output, rows, scores)
    summary = {
        "protocol": protocol, "completed_at": datetime.now(timezone.utc).isoformat(), "architectures": scores,
        "per_layer_rows": len(rows), "figures": figures,
        "provenance": provenance,
        "runtime_memory": {"overall_wall_seconds": time.perf_counter() - started_all,
                           "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                           "statistics_device": str(device)},
        "caveat": "These two representations are intentionally excluded from the pre_sft_logme_proxy_v2_common7 five-candidate primary ranking and LogME--VSI correlations.",
    }
    write_json(output / "logme_extra_summary.json", summary)
    lines = [
        "# Extra pre-SFT LogME diagnostics", "",
        "This cache-only run evaluates C1 EoMT selective fusion and the plain pre-SFT base VLM with the same validated float64 streaming regression LogME implementation, common-seven layer set (L1,L3,L6,L9,L15,L21,L27), formal train-video split, cached depth target, validity mask, and token alignment as the completed v2 study.", "",
        "It loads no VLM, no post-SFT checkpoint, and creates no optimizer. Both source caches explicitly record pre-SFT loading and `no_vlm3r_sft_adapter_loaded: true`.", "",
        "## Scope", "",
        "These are extra diagnostics, not additions to the formal five-candidate C1 architecture study. They are excluded from the frozen `pre_sft_logme_proxy_v2_common7` architecture ranking and all of its LogME--VSI correlations.", "",
        "## Scores", "", "| Representation | Seven-layer mean LogME | Diagnostic rank within extra pair | Valid tokens/layer |", "|---|---:|---:|---:|",
    ]
    for score in scores:
        lines.append(f"| {score['display_name']} | {float(score['logme_primary']):.12f} | {score['diagnostic_rank_within_extra_pair']} | {score['valid_tokens_per_layer']} |")
    lines.extend(["", "The exact target/mask signature for every one of the 14 computations matches the formal common-seven target signature: `" + FORMAL_TARGET_SIGNATURE + "`.", ""])
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("[COMPLETE] extra_diagnostic_scores=" + json.dumps({row["architecture"]: row["logme_primary"] for row in scores}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
