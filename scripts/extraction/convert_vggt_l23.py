#!/usr/bin/env python3
"""Convert the canonical VGGT cache to restartable, exact L23-only sidecars."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

import pandas as pd
import torch


REPO_ROOT = Path("/gpfs/home4/geusdd/shuang/SpatialFocus")
DATA_ROOT = Path("/scratch-shared/geusdd/VLM3R/data")
SOURCE_ROOT = Path("/scratch-shared/geusdd/shaoruei/VLM3R/spatial_features/vggt")
STAGING_ROOT = Path("/scratch-shared/geusdd/shaoruei/VLM3R/spatial_features/vggt_l23_staging")
FINAL_ROOT = Path("/scratch-shared/geusdd/shaoruei/VLM3R/spatial_features/vggt_l23")
ARTIFACTS_ROOT = Path("/scratch-shared/geusdd/shaoruei/VLM3R/spatial_features/vggt_l23_artifacts")

TRAIN_MANIFESTS = (
    DATA_ROOT / "vlm3r/VLM-3R-DATA/vsibench_train/merged_qa_scannet_train.json",
    DATA_ROOT / "vlm3r/VLM-3R-DATA/vsibench_train/merged_qa_scannetpp_train.json",
    DATA_ROOT / "vlm3r/VLM-3R-DATA/vsibench_train/merged_qa_route_plan_train.json",
)
EVAL_MANIFESTS = (
    DATA_ROOT / "vsibench/test_pruned.parquet",
    DATA_ROOT / "vsibench/test_debiased.parquet",
)
SOURCE_PROVENANCE = (
    REPO_ROOT / "vggt_extraction_handoff/checksums.sha256",
    Path(
        "/scratch-shared/geusdd/shaoruei/VLM3R/spatial_features/"
        "vggt_missing_staging/_reports/final_audit.json"
    ),
    Path(
        "/scratch-shared/geusdd/shaoruei/VLM3R/spatial_features/"
        "vggt_missing_staging/_reports/merge.json"
    ),
)

DATASETS = ("scannet", "scannetpp", "arkitscenes")
EXPECTED_DATASET_COUNTS = {"scannet": 1289, "scannetpp": 906, "arkitscenes": 498}
EXPECTED_TRAIN_COUNTS = {"scannet": 1201, "scannetpp": 856, "arkitscenes": 348}
EXPECTED_EVAL_COUNTS = {"scannet": 88, "scannetpp": 50, "arkitscenes": 150}
EXPECTED_TOTAL = 2693
SOURCE_LAYER_KEYS = {"11", "17", "23"}
OUTPUT_LAYER_KEYS = {"23"}
TOKEN_SHAPE = (32, 1374, 2048)
SMOKE_KEYS = {
    ("scannet", "scene0025_01"),
    ("scannetpp", "dd685be466"),
    ("arkitscenes", "41069025"),
}


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frame_idx_sha256(frame_idx: torch.Tensor) -> str:
    values = frame_idx.detach().cpu().contiguous().numpy().astype("<i8", copy=False)
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
    os.replace(temporary, path)


def _parse_train_video(value: Any, manifest: Path) -> tuple[str, str]:
    relative = PurePosixPath(str(value))
    if len(relative.parts) != 3 or relative.parts[1] != "videos":
        raise ValueError(f"unexpected training video path in {manifest}: {value!r}")
    dataset = relative.parts[0].lower()
    if dataset not in DATASETS:
        raise ValueError(f"unexpected dataset in {manifest}: {dataset!r}")
    return dataset, Path(relative.name).stem


def canonical_records() -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    membership: dict[tuple[str, str], set[str]] = {}
    canonical_inputs = []

    for manifest in TRAIN_MANIFESTS:
        rows = json.loads(manifest.read_text())
        if not isinstance(rows, list):
            raise ValueError(f"training manifest is not a list: {manifest}")
        for row in rows:
            dataset, sample_id = _parse_train_video(row["video"], manifest)
            membership.setdefault((dataset, sample_id), set()).add("train")
        canonical_inputs.append(
            {"path": str(manifest), "size": manifest.stat().st_size, "sha256": sha256_file(manifest)}
        )

    for manifest in EVAL_MANIFESTS:
        frame = pd.read_parquet(manifest, columns=["dataset", "scene_name"])
        for row in frame.itertuples(index=False):
            dataset = str(row.dataset).lower()
            sample_id = str(row.scene_name)
            if dataset not in DATASETS:
                raise ValueError(f"unexpected evaluation dataset in {manifest}: {dataset!r}")
            membership.setdefault((dataset, sample_id), set()).add("eval")
        canonical_inputs.append(
            {"path": str(manifest), "size": manifest.stat().st_size, "sha256": sha256_file(manifest)}
        )

    overlaps = sorted(key for key, scopes in membership.items() if len(scopes) != 1)
    if overlaps:
        raise ValueError(f"training/evaluation membership overlaps: {overlaps[:10]}")

    records = [
        {"dataset": dataset, "sample_id": sample_id, "scope": next(iter(scopes))}
        for (dataset, sample_id), scopes in sorted(membership.items())
    ]
    dataset_counts = Counter(record["dataset"] for record in records)
    train_counts = Counter(record["dataset"] for record in records if record["scope"] == "train")
    eval_counts = Counter(record["dataset"] for record in records if record["scope"] == "eval")
    if dict(dataset_counts) != EXPECTED_DATASET_COUNTS:
        raise ValueError(f"canonical dataset counts changed: {dict(dataset_counts)}")
    if dict(train_counts) != EXPECTED_TRAIN_COUNTS:
        raise ValueError(f"canonical training counts changed: {dict(train_counts)}")
    if dict(eval_counts) != EXPECTED_EVAL_COUNTS:
        raise ValueError(f"canonical evaluation counts changed: {dict(eval_counts)}")
    if len(records) != EXPECTED_TOTAL:
        raise ValueError(f"canonical union is {len(records)}, expected {EXPECTED_TOTAL}")
    return records, canonical_inputs


def expected_relative_paths(records: list[dict[str, str]]) -> set[str]:
    return {f"{record['dataset']}/{record['sample_id']}.pt" for record in records}


def tree_fingerprint(root: Path, expected: set[str]) -> dict[str, Any]:
    observed_paths = sorted(path for path in root.glob("*/*.pt") if path.is_file())
    observed = {str(path.relative_to(root)) for path in observed_paths}
    all_files = {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()}
    missing = expected - observed
    extra = all_files - expected
    if missing or extra:
        raise ValueError(
            f"tree coverage mismatch under {root}: "
            f"missing={sorted(missing)[:10]} extra={sorted(extra)[:10]}"
        )

    digest = hashlib.sha256()
    total_bytes = 0
    for path in observed_paths:
        stat = path.stat()
        relative = str(path.relative_to(root))
        total_bytes += stat.st_size
        digest.update(f"{relative}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode())
    return {
        "root": str(root.resolve()),
        "file_count": len(observed_paths),
        "total_bytes": total_bytes,
        "filename_size_mtime_sha256": digest.hexdigest(),
    }


def remove_stale_temporary_files(root: Path) -> None:
    if not root.exists():
        return
    for path in sorted(root.glob("**/.*.tmp.*")):
        if not path.is_file():
            raise ValueError(f"unexpected non-file temporary path: {path}")
        print(f"REMOVE_STALE_TEMP {path}", flush=True)
        path.unlink()


def _validate_common(payload: Any, path: Path, expected_shape: tuple[int, int, int]) -> tuple[dict, dict, torch.Tensor]:
    if not isinstance(payload, dict) or set(payload) != {"frames", "meta"}:
        raise ValueError(f"{path}: invalid top-level schema")
    frames = payload["frames"]
    meta = payload["meta"]
    if not isinstance(frames, dict) or set(frames) != {"aggregated_tokens", "frame_idx"}:
        raise ValueError(f"{path}: invalid frames schema")
    if not isinstance(meta, dict):
        raise ValueError(f"{path}: meta is not a dict")
    frame_idx = frames["frame_idx"]
    if not isinstance(frame_idx, torch.Tensor):
        raise ValueError(f"{path}: frame_idx is not a tensor")
    if frame_idx.device.type != "cpu" or frame_idx.dtype != torch.int64:
        raise ValueError(f"{path}: frame_idx must be CPU int64")
    if tuple(frame_idx.shape) != (expected_shape[0],):
        raise ValueError(f"{path}: frame_idx shape is {tuple(frame_idx.shape)}")
    expected_meta = {
        "num_frames": expected_shape[0],
        "input_size": 518,
        "model_image_hw": (518, 518),
        "patch_size": 14,
        "patch_start_idx": 5,
        "feature_dim": expected_shape[2],
        "token_dtype": "bfloat16",
        "schema": "vggt_aggregated_tokens_v1",
    }
    for key, expected_value in expected_meta.items():
        if meta.get(key) != expected_value:
            raise ValueError(
                f"{path}: meta.{key}={meta.get(key)!r}, expected {expected_value!r}"
            )
    return frames, meta, frame_idx


def validate_source_payload(
    payload: Any,
    path: Path,
    expected_shape: tuple[int, int, int] = TOKEN_SHAPE,
) -> tuple[dict, torch.Tensor]:
    frames, meta, frame_idx = _validate_common(payload, path, expected_shape)
    tokens = frames["aggregated_tokens"]
    if not isinstance(tokens, dict) or set(tokens) != SOURCE_LAYER_KEYS:
        raise ValueError(f"{path}: source layer keys are not {sorted(SOURCE_LAYER_KEYS)}")
    for layer in sorted(SOURCE_LAYER_KEYS, key=int):
        tensor = tokens[layer]
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"{path}: layer {layer} is not a tensor")
        if tensor.device.type != "cpu" or tensor.dtype != torch.bfloat16:
            raise ValueError(f"{path}: layer {layer} must be CPU bfloat16")
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(f"{path}: layer {layer} shape is {tuple(tensor.shape)}")
    if meta.get("intermediate_layer_idx") not in ([4, 11, 17, 23], [11, 17, 23]):
        raise ValueError(f"{path}: invalid source intermediate_layer_idx")
    return meta, frame_idx


def build_output_payload(source_payload: dict) -> dict:
    output_meta = dict(source_payload["meta"])
    output_meta["intermediate_layer_idx"] = [23]
    return {
        "frames": {
            "aggregated_tokens": {"23": source_payload["frames"]["aggregated_tokens"]["23"]},
            "frame_idx": source_payload["frames"]["frame_idx"],
        },
        "meta": output_meta,
    }


def validate_output_payload(
    output_payload: Any,
    path: Path,
    source_payload: dict,
    expected_shape: tuple[int, int, int] = TOKEN_SHAPE,
) -> torch.Tensor:
    frames, meta, frame_idx = _validate_common(output_payload, path, expected_shape)
    tokens = frames["aggregated_tokens"]
    if not isinstance(tokens, dict) or set(tokens) != OUTPUT_LAYER_KEYS:
        raise ValueError(f"{path}: output layer keys are not ['23']")
    tensor = tokens["23"]
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"{path}: output layer 23 is not a tensor")
    if tensor.device.type != "cpu" or tensor.dtype != torch.bfloat16:
        raise ValueError(f"{path}: output layer 23 must be CPU bfloat16")
    if tuple(tensor.shape) != expected_shape:
        raise ValueError(f"{path}: output layer 23 shape is {tuple(tensor.shape)}")
    if meta.get("intermediate_layer_idx") != [23]:
        raise ValueError(f"{path}: output intermediate_layer_idx is not [23]")

    expected_meta = dict(source_payload["meta"])
    expected_meta["intermediate_layer_idx"] = [23]
    if meta != expected_meta:
        raise ValueError(f"{path}: output metadata differs beyond intermediate_layer_idx")
    if not torch.equal(frame_idx, source_payload["frames"]["frame_idx"]):
        raise ValueError(f"{path}: output frame_idx differs from source")
    if not torch.equal(tensor, source_payload["frames"]["aggregated_tokens"]["23"]):
        raise ValueError(f"{path}: output L23 tensor differs from source")
    return frame_idx


def convert_one(source_path: Path, output_path: Path) -> tuple[str, dict[str, Any]]:
    source_payload = torch.load(source_path, map_location="cpu")
    temporary: Path | None = None
    try:
        validate_source_payload(source_payload, source_path)
        status = "converted"
        output_payload = None
        if output_path.is_file():
            try:
                output_payload = torch.load(output_path, map_location="cpu")
                validate_output_payload(output_payload, output_path, source_payload)
                status = "valid_existing"
            except Exception as exc:
                print(f"INVALID_EXISTING {output_path}: {exc}", flush=True)
                output_payload = None

        if output_payload is None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = output_path.with_name(
                f".{output_path.name}.tmp.{os.getpid()}.{time.time_ns()}"
            )
            torch.save(build_output_payload(source_payload), temporary)
            output_payload = torch.load(temporary, map_location="cpu")
            validate_output_payload(output_payload, temporary, source_payload)
            os.replace(temporary, output_path)
            temporary = None

        frame_idx = validate_output_payload(output_payload, output_path, source_payload)
        result = {
            "output_sha256": sha256_file(output_path),
            "output_size": output_path.stat().st_size,
            "frame_idx_sha256": frame_idx_sha256(frame_idx),
            "frame_idx_dtype": "int64",
            "frame_idx_shape": list(frame_idx.shape),
            "tensor_dtype": "bfloat16",
            "tensor_shape": list(source_payload["frames"]["aggregated_tokens"]["23"].shape),
        }
        return status, result
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
        try:
            del output_payload
        except UnboundLocalError:
            pass
        del source_payload
        gc.collect()


def file_provenance(paths: tuple[Path, ...]) -> list[dict[str, Any]]:
    result = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"required provenance file is missing: {path}")
        result.append({"path": str(path), "size": path.stat().st_size, "sha256": sha256_file(path)})
    return result


def build_counts(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "overall": len(records),
        "by_dataset": dict(sorted(Counter(record["dataset"] for record in records).items())),
        "by_scope": dict(sorted(Counter(record["scope"] for record in records).items())),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=("smoke", "full"), required=True)
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--staging-root", type=Path, default=STAGING_ROOT)
    parser.add_argument("--final-root", type=Path, default=FINAL_ROOT)
    parser.add_argument("--artifacts-root", type=Path, default=ARTIFACTS_ROOT)
    parser.add_argument("--publish", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.time()
    args.artifacts_root.mkdir(parents=True, exist_ok=True)

    if args.publish and args.scope != "full":
        raise ValueError("--publish is valid only with --scope full")
    if args.final_root.exists():
        raise FileExistsError(f"refusing to overwrite existing final root: {args.final_root}")

    remove_stale_temporary_files(args.staging_root)

    records, canonical_inputs = canonical_records()
    expected = expected_relative_paths(records)
    source_before = tree_fingerprint(args.source_root, expected)
    selected = (
        [record for record in records if (record["dataset"], record["sample_id"]) in SMOKE_KEYS]
        if args.scope == "smoke"
        else records
    )
    if args.scope == "smoke" and len(selected) != 3:
        raise ValueError(f"smoke selection produced {len(selected)} records, expected 3")

    inventory_records = []
    failures = []
    status_counts: Counter[str] = Counter()
    for index, record in enumerate(selected, start=1):
        relative = Path(record["dataset"]) / f"{record['sample_id']}.pt"
        source_path = args.source_root / relative
        staged_path = args.staging_root / relative
        final_path = args.final_root / relative
        try:
            status, details = convert_one(source_path, staged_path)
            status_counts[status] += 1
            inventory_records.append(
                {
                    "dataset": record["dataset"],
                    "sample_id": record["sample_id"],
                    "scope": record["scope"],
                    "relative_path": str(relative),
                    "source_sidecar": str(source_path.absolute()),
                    "output_sidecar": str(final_path.absolute()),
                    **details,
                }
            )
            print(
                f"{status.upper()} {index}/{len(selected)} {record['dataset']}/{record['sample_id']}",
                flush=True,
            )
        except Exception as exc:
            failures.append(
                {
                    "dataset": record["dataset"],
                    "sample_id": record["sample_id"],
                    "source_sidecar": str(source_path),
                    "staged_sidecar": str(staged_path),
                    "error": str(exc),
                }
            )
            print(
                f"FAILURE {index}/{len(selected)} {record['dataset']}/{record['sample_id']}: {exc}",
                file=sys.stderr,
                flush=True,
            )

    source_after = tree_fingerprint(args.source_root, expected)
    source_unchanged = source_before == source_after
    inventory_records.sort(key=lambda row: (row["dataset"], row["sample_id"]))
    provenance = file_provenance(SOURCE_PROVENANCE)
    complete = not failures and source_unchanged and len(inventory_records) == len(selected)
    output_tree = None
    if args.scope == "full" and complete:
        output_tree = tree_fingerprint(args.staging_root, expected)

    inventory = {
        "schema": "vggt_l23_inventory_v1",
        "created_at_epoch": time.time(),
        "scope": args.scope,
        "source_root": str(args.source_root.resolve()),
        "output_root": str(args.final_root.absolute()),
        "canonical_inputs": canonical_inputs,
        "source_provenance": provenance,
        "source_tree_fingerprint_before": source_before,
        "source_tree_fingerprint_after": source_after,
        "source_tree_unchanged": source_unchanged,
        "output_tree_fingerprint_staging": output_tree,
        "counts": build_counts(inventory_records),
        "records": inventory_records,
    }

    official_inventory_path = args.artifacts_root / "cached_vggt_l23_inventory.json"
    if args.scope == "smoke":
        inventory_write_path = args.artifacts_root / "smoke_vggt_l23_inventory.json"
        inventory_path = inventory_write_path
    elif not complete:
        inventory_write_path = args.artifacts_root / "cached_vggt_l23_inventory.partial.json"
        inventory_path = inventory_write_path
    elif args.publish:
        inventory_write_path = args.artifacts_root / ".cached_vggt_l23_inventory.pending.json"
        inventory_path = official_inventory_path
    else:
        inventory_write_path = args.artifacts_root / "cached_vggt_l23_inventory.unpublished.json"
        inventory_path = inventory_write_path
    report_path = args.artifacts_root / f"{args.scope}_conversion_report.json"
    report = {
        "schema": "vggt_l23_conversion_report_v1",
        "scope": args.scope,
        "complete": complete,
        "published": False,
        "loader_manifest_status": "pending_source_manifest",
        "source_full_sidecar_sha_sweep_performed": False,
        "selected_count": len(selected),
        "status_counts": dict(sorted(status_counts.items())),
        "failures": failures,
        "source_tree_unchanged": source_unchanged,
        "inventory_path": str(inventory_path),
        "inventory_sha256": None,
        "runtime_seconds": time.time() - started,
    }
    atomic_write_json(inventory_write_path, inventory)
    report["inventory_sha256"] = sha256_file(inventory_write_path)
    atomic_write_json(report_path, report)

    if not complete:
        print(json.dumps(report, indent=2), file=sys.stderr, flush=True)
        return 1

    if args.publish:
        if len(inventory_records) != EXPECTED_TOTAL:
            raise ValueError("refusing to publish a non-canonical inventory")
        os.replace(args.staging_root, args.final_root)
        os.replace(inventory_write_path, official_inventory_path)
        report["published"] = True
        report["published_root"] = str(args.final_root)
        report["runtime_seconds"] = time.time() - started
        atomic_write_json(report_path, report)
        completion = {
            "schema": "vggt_l23_completion_v1",
            "completed_at_epoch": time.time(),
            "output_root": str(args.final_root),
            "inventory_path": str(inventory_path),
            "inventory_sha256": report["inventory_sha256"],
            "record_count": len(inventory_records),
            "loader_manifest_status": "pending_source_manifest",
            "report_path": str(report_path),
            "report_sha256": sha256_file(report_path),
        }
        atomic_write_json(args.artifacts_root / "COMPLETE.json", completion)

    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
