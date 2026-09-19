#!/usr/bin/env python3
"""Strictly validate VGGT sidecars and their forced Decord frame indices."""

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch
from decord import VideoReader, cpu


VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")
EXPECTED_LAYERS = (11, 17, 23)
EXPECTED_LAYER_KEYS = {str(layer) for layer in EXPECTED_LAYERS}
EXPECTED_TOP_LEVEL_KEYS = {"frames", "meta"}
EXPECTED_FRAME_KEYS = {"aggregated_tokens", "frame_idx"}


def read_ids(path: Path) -> list[str]:
    identifiers = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError(f"duplicate identifiers in {path}")
    return identifiers


def locate_video(raw_root: Path, identifier: str) -> Path:
    candidates = [raw_root / f"{identifier}{extension}" for extension in VIDEO_EXTENSIONS]
    candidates += [
        raw_root / f"{identifier}{extension.upper()}" for extension in VIDEO_EXTENSIONS
    ]
    existing = [path for path in candidates if path.is_file()]
    if len(existing) != 1:
        raise ValueError(
            f"expected exactly one source video for {identifier} under {raw_root}; "
            f"found {existing}"
        )
    return existing[0].resolve()


def expected_frame_idx(video_path: Path, frames_upbound: int = 32, video_fps: int = 1) -> torch.Tensor:
    reader = VideoReader(str(video_path), ctx=cpu(0), num_threads=1)
    total_frames = len(reader)
    if total_frames <= 0:
        raise ValueError(f"empty video: {video_path}")
    fps_stride = max(1, round(reader.get_avg_fps() / video_fps))
    indices = list(range(0, total_frames, fps_stride))
    if len(indices) > frames_upbound or True:  # DataArguments.force_sample is hard-coded True.
        indices = np.linspace(0, total_frames - 1, frames_upbound, dtype=int).tolist()
    reader.seek(0)
    return torch.tensor(indices, dtype=torch.int64)


def validate_payload(
    payload,
    path: Path,
    expected_indices: torch.Tensor | None,
    source_video: Path | None,
    allow_legacy_layer4_metadata: bool = False,
    require_recorded_source_match: bool = True,
) -> dict:
    if not isinstance(payload, dict) or set(payload) != EXPECTED_TOP_LEVEL_KEYS:
        raise ValueError(f"{path}: invalid top-level schema")
    frames = payload["frames"]
    meta = payload["meta"]
    if not isinstance(frames, dict) or set(frames) != EXPECTED_FRAME_KEYS:
        raise ValueError(f"{path}: invalid frames schema")
    if not isinstance(meta, dict):
        raise ValueError(f"{path}: meta is not a dict")

    frame_idx = frames["frame_idx"]
    if not isinstance(frame_idx, torch.Tensor):
        raise ValueError(f"{path}: frame_idx is not a tensor")
    if frame_idx.device.type != "cpu" or frame_idx.dtype != torch.int64:
        raise ValueError(f"{path}: frame_idx must be CPU int64, got {frame_idx.device}/{frame_idx.dtype}")
    if tuple(frame_idx.shape) != (32,):
        raise ValueError(f"{path}: frame_idx shape is {tuple(frame_idx.shape)}, expected (32,)")
    if expected_indices is not None and not torch.equal(frame_idx, expected_indices):
        raise ValueError(
            f"{path}: frame_idx mismatch; stored={frame_idx.tolist()} expected={expected_indices.tolist()}"
        )

    aggregated_tokens = frames["aggregated_tokens"]
    if not isinstance(aggregated_tokens, dict) or set(aggregated_tokens) != EXPECTED_LAYER_KEYS:
        raise ValueError(
            f"{path}: layer keys are "
            f"{sorted(aggregated_tokens) if isinstance(aggregated_tokens, dict) else type(aggregated_tokens)}, "
            f"expected {sorted(EXPECTED_LAYER_KEYS)}"
        )
    for layer in EXPECTED_LAYERS:
        tensor = aggregated_tokens[str(layer)]
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"{path}: layer {layer} is not a tensor")
        if tuple(tensor.shape) != (32, 1374, 2048):
            raise ValueError(f"{path}: layer {layer} shape is {tuple(tensor.shape)}")
        if tensor.dtype != torch.bfloat16 or tensor.device.type != "cpu":
            raise ValueError(f"{path}: layer {layer} is {tensor.dtype}/{tensor.device}")

    expected_meta = {
        "num_frames": 32,
        "input_size": 518,
        "model_image_hw": (518, 518),
        "patch_size": 14,
        "patch_start_idx": 5,
        "feature_dim": 2048,
        "token_dtype": "bfloat16",
        "schema": "vggt_aggregated_tokens_v1",
    }
    for key, expected in expected_meta.items():
        if meta.get(key) != expected:
            raise ValueError(f"{path}: meta.{key}={meta.get(key)!r}, expected {expected!r}")
    allowed_layer_metadata = ([11, 17, 23], [4, 11, 17, 23]) if allow_legacy_layer4_metadata else ([11, 17, 23],)
    if meta.get("intermediate_layer_idx") not in allowed_layer_metadata:
        raise ValueError(
            f"{path}: meta.intermediate_layer_idx={meta.get('intermediate_layer_idx')!r}, "
            f"expected one of {allowed_layer_metadata!r}"
        )
    if not isinstance(meta.get("source_video"), str):
        raise ValueError(f"{path}: missing meta.source_video")
    if not isinstance(meta.get("vggt_weights_path"), str):
        raise ValueError(f"{path}: missing meta.vggt_weights_path")
    if source_video is not None and require_recorded_source_match:
        recorded_source = Path(meta["source_video"])
        if not recorded_source.exists() or recorded_source.resolve() != source_video:
            raise ValueError(
                f"{path}: recorded source {recorded_source} does not resolve to {source_video}"
            )

    return {
        "top_level_keys": sorted(payload),
        "frame_keys": sorted(frames),
        "layer_keys": sorted(aggregated_tokens, key=int),
        "frame_idx": frame_idx.tolist(),
        "tensor_shape": list(aggregated_tokens["11"].shape),
        "tensor_dtype": str(aggregated_tokens["11"].dtype),
        "meta_keys": sorted(meta),
        "meta_intermediate_layer_idx": list(meta["intermediate_layer_idx"]),
    }


def load_and_validate(
    path: Path,
    source_video: Path | None = None,
    allow_legacy_layer4_metadata: bool = False,
    require_recorded_source_match: bool = True,
) -> dict:
    indices = expected_frame_idx(source_video) if source_video is not None else None
    payload = torch.load(path, map_location="cpu")
    try:
        return validate_payload(
            payload,
            path,
            indices,
            source_video,
            allow_legacy_layer4_metadata=allow_legacy_layer4_metadata,
            require_recorded_source_match=require_recorded_source_match,
        )
    finally:
        del payload
        gc.collect()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spec",
        nargs=4,
        action="append",
        metavar=("LABEL", "IDS_FILE", "RAW_ROOT", "OUTPUT_DIR"),
        required=True,
        help="Repeat for each independently counted dataset/scope.",
    )
    parser.add_argument("--reference-sidecar", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    failures = []
    groups = {}
    first_new_schema = None
    total = 0
    for label, ids_file_text, raw_root_text, output_dir_text in args.spec:
        ids_file = Path(ids_file_text)
        raw_root = Path(raw_root_text)
        output_dir = Path(output_dir_text)
        identifiers = read_ids(ids_file)
        valid = 0
        for identifier in identifiers:
            output_path = output_dir / f"{identifier}.pt"
            try:
                source_video = locate_video(raw_root, identifier)
                schema = load_and_validate(output_path, source_video)
                if first_new_schema is None:
                    first_new_schema = schema
                valid += 1
                print(f"VALID {label} {identifier} {output_path}", flush=True)
            except Exception as exc:
                failures.append({"label": label, "id": identifier, "path": str(output_path), "error": str(exc)})
                print(f"INVALID {label} {identifier} {output_path}: {exc}", file=sys.stderr, flush=True)
        groups[label] = {"expected": len(identifiers), "valid": valid}
        total += len(identifiers)

    reference = None
    if args.reference_sidecar:
        try:
            reference = load_and_validate(
                args.reference_sidecar,
                allow_legacy_layer4_metadata=True,
            )
            if first_new_schema is None:
                raise ValueError("no new sidecar was successfully validated")
            fields = (
                "top_level_keys",
                "frame_keys",
                "layer_keys",
                "tensor_shape",
                "tensor_dtype",
                "meta_keys",
            )
            mismatches = {
                field: {"new": first_new_schema[field], "reference": reference[field]}
                for field in fields
                if first_new_schema[field] != reference[field]
            }
            if mismatches:
                raise ValueError(f"reference schema mismatch: {mismatches}")
            print(f"REFERENCE_COMPATIBLE {args.reference_sidecar}", flush=True)
        except Exception as exc:
            failures.append({"label": "reference", "path": str(args.reference_sidecar), "error": str(exc)})
            print(f"INVALID reference {args.reference_sidecar}: {exc}", file=sys.stderr, flush=True)

    report = {
        "groups": groups,
        "total_expected": total,
        "total_valid": sum(group["valid"] for group in groups.values()),
        "reference_sidecar": str(args.reference_sidecar) if args.reference_sidecar else None,
        "reference_compatible": bool(reference is not None and not any(f["label"] == "reference" for f in failures)),
        "failures": failures,
    }
    print(json.dumps(report, indent=2), flush=True)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        temporary_report = args.report.with_name(f".{args.report.name}.tmp")
        temporary_report.write_text(json.dumps(report, indent=2) + "\n")
        temporary_report.replace(args.report)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
