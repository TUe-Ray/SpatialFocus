#!/usr/bin/env python3
"""Score frozen InternVL3 no-spatial features with the audited common-seven LogME.

This is an independent base-VLM diagnostic; it does not alter the formal C1
five-candidate architecture registry or infer a VSI-Bench correlation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.probing.depth_probe_common import load_frame_records
from scripts.probing.run_pre_sft_logme_proxy import (
    Candidate, SPLIT_SHA256, logme_from_statistics, sha256_file, write_json,
)
from scripts.probing.run_pre_sft_logme_proxy_v2_common7 import accumulate_statistics_on_device


LABEL = "internvl3_8b_base_presft"
LAYERS = (1, 3, 6, 9, 15, 21, 27)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sample-indices", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--block-frames", type=int, default=16)
    args = parser.parse_args()
    if args.block_frames <= 0:
        raise ValueError("--block-frames must be positive")
    if sha256_file(args.sample_indices) != SPLIT_SHA256:
        raise RuntimeError("Unexpected fixed ScanNet sample split")
    manifest_path = args.output_root / f"{LABEL}_run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "complete" or manifest.get("complete_videos") != 1199:
        raise RuntimeError("LogME requires all 1,199 full 32-frame feature extractions")
    if manifest.get("candidate") != "base_no_spatial" or manifest.get("optimizer_steps") != 0:
        raise RuntimeError("Feature provenance is not a frozen no-spatial base comparator")
    if manifest.get("sample_indices_sha256") != SPLIT_SHA256:
        raise RuntimeError("Feature and LogME sample split differ")
    if any(f"layer_{layer}" not in manifest.get("feature_levels", []) for layer in LAYERS):
        raise RuntimeError("Incomplete common-seven layer coverage")
    train = load_frame_records(args.sample_indices, split="train")
    val = load_frame_records(args.sample_indices, split="val")
    if len(train) != 2012 or len(val) != 386:
        raise RuntimeError("Expected 1006/193 videos with two selected frames each")
    candidate = Candidate(LABEL, "InternVL3-8B base (no spatial)", args.output_root, "InternVL3-8B")
    result_dir = args.output_root / "logme_common7" / LABEL
    result_dir.mkdir(parents=True, exist_ok=True)
    identity = {
        "schema": "spatialfocus.internvl3_base_common7_logme.v1",
        "model_label": LABEL,
        "feature_manifest_sha256": sha256_file(manifest_path),
        "sample_indices_sha256": SPLIT_SHA256,
        "layers": list(LAYERS),
        "train_frames": len(train),
        "target": "camera-coordinate depth, same valid mask as weak MLP depth probe",
        "loss": "normalized Bayesian linear-regression log evidence",
        "optimizer_steps": 0,
    }
    identity_sha = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    rows = []
    for layer in LAYERS:
        row_path = result_dir / f"layer_{layer}.json"
        if row_path.exists():
            row = json.loads(row_path.read_text())
            if row.get("identity_sha256") != identity_sha:
                raise RuntimeError(f"Refusing to reuse incompatible L{layer} LogME result")
        else:
            started = time.perf_counter()
            gram, cross, yy, count, frames, target_sha = accumulate_statistics_on_device(
                candidate, layer, train, device=torch.device(args.device), block_frames=args.block_frames,
            )
            if frames != len(train):
                raise RuntimeError(f"L{layer} training frame count changed")
            score = logme_from_statistics(gram, cross, yy, count)
            row = {
                **score, "layer": layer, "feature_dim": gram.shape[0],
                "train_frames": frames, "valid_tokens": count,
                "target_sha256": target_sha, "identity_sha256": identity_sha,
                "runtime_seconds": time.perf_counter() - started,
            }
            write_json(row_path, row)
        rows.append(row)
        print(json.dumps({"layer": layer, "logme": row["logme"], "valid_tokens": row["valid_tokens"]}), flush=True)
    if len({row["target_sha256"] for row in rows}) != 1:
        raise RuntimeError("Common-seven target sequence or mask differs by layer")
    summary = {
        **identity, "identity_sha256": identity_sha,
        "mean_logme": sum(float(row["logme"]) for row in rows) / len(rows),
        "per_layer": rows,
        "target_sha256": rows[0]["target_sha256"],
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(result_dir / "summary.json", summary)
    print(json.dumps({"status": "complete", "mean_logme": summary["mean_logme"]}), flush=True)


if __name__ == "__main__":
    main()
