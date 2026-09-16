#!/usr/bin/env python
"""Freeze the non-C1 A-prime constructor and formal ScanNet probe inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from llava.model.controlled_fusion_pre_sft import CONTROLLED_A_PRIME_PRE_SFT_SPEC  # noqa: E402
from llava.model.multimodal_fusion_block.builder import build_multimodal_fusion_block  # noqa: E402
from scripts.diagnose_layerwise_spatial_hidden_scan import (  # noqa: E402
    module_state_sha256,
    seeded_fusion_initialization,
)
from scripts.probing.probe_layer_policy import PRE_SFT_FULL_FEATURE_LEVELS  # noqa: E402


SCHEMA_VERSION = "controlled_a_prime_pre_sft_manifest_v1"
OFFICIAL_SEED = 42
OFFICIAL_SPLIT_SHA256 = "d478cb684958dfc25066821ec83d5216469577c9e282e33bdf87d3c88b200d8e"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fresh_a_prime_state(seed: int = OFFICIAL_SEED) -> tuple[str, int]:
    config = SimpleNamespace(
        fusion_block="pre_projector_cross_attention_patch_only",
        mm_hidden_size=1152,
        hidden_size=3584,
        spatial_feature_dim=768,
        pre_projector_cross_attention_source_layer=12,
    )
    with seeded_fusion_initialization(seed):
        module = build_multimodal_fusion_block(config)
    return module_state_sha256(module), sum(parameter.numel() for parameter in module.parameters())


def git_value(*args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=REPO_ROOT, text=True).strip()


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    sample_hash = sha256_file(args.sample_indices)
    if sample_hash != OFFICIAL_SPLIT_SHA256:
        raise ValueError(f"A-prime requires official split {OFFICIAL_SPLIT_SHA256}, got {sample_hash}")
    samples = json.loads(args.sample_indices.read_text(encoding="utf-8"))
    videos = samples.get("videos")
    if not isinstance(videos, list) or len(videos) != 1199:
        raise ValueError(f"A-prime formal manifest requires 1,199 videos, got {len(videos or [])}")
    if any(not isinstance(video.get("frames"), list) or len(video["frames"]) != 2 for video in videos):
        raise ValueError("A-prime formal manifest requires exactly two selected target frames per video")
    split_counts = {
        split: sum(str(video.get("split")) == split for video in videos)
        for split in ("train", "val")
    }
    if split_counts != {"train": 1006, "val": 193}:
        raise ValueError(f"A-prime formal manifest has the wrong video split: {split_counts}")
    dec12_root = args.feature_root / "scannet" / "spatial_features"
    expected_scenes = {str(video.get("scene_id")) for video in videos}
    dec12_files = sorted(dec12_root / f"{scene}.pt" for scene in expected_scenes)
    missing_dec12 = [path for path in dec12_files if not path.is_file()]
    empty_dec12 = [path for path in dec12_files if path.is_file() and path.stat().st_size <= 0]
    if len(dec12_files) != 1199 or missing_dec12 or empty_dec12:
        raise ValueError(
            "A-prime requires exactly 1,199 non-empty formal ScanNet decoder-12 sidecars: "
            f"count={len(dec12_files)}, missing={[str(path) for path in missing_dec12[:10]]}, "
            f"empty={[str(path) for path in empty_dec12[:10]]}"
        )
    available_dec12_count = len(list(dec12_root.glob("*.pt")))
    reference_sidecar = dec12_root / "scene0384_00.pt"
    try:
        reference_payload = torch.load(reference_sidecar, map_location="cpu", weights_only=True)
    except TypeError:
        reference_payload = torch.load(reference_sidecar, map_location="cpu")
    if not isinstance(reference_payload, dict):
        raise TypeError(f"Invalid decoder-12 sidecar payload: {reference_sidecar}")
    patch_tokens = reference_payload.get("patch_tokens")
    if not isinstance(patch_tokens, torch.Tensor) or list(patch_tokens.shape) != [32, 729, 768]:
        raise ValueError(
            f"A-prime decoder-12 reference patch shape is invalid: {getattr(patch_tokens, 'shape', None)}"
        )
    status = git_value("status", "--short")
    if status:
        raise RuntimeError("Official A-prime manifest requires a clean Git worktree")
    forbidden = [
        str(path)
        for name in ("adapter_model.bin", "non_lora_trainables.bin", "adapter_config.json")
        for path in args.base_model.rglob(name)
    ]
    if forbidden:
        raise RuntimeError(f"Plain base contains forbidden post-SFT state: {forbidden}")
    state_hash, parameter_count = fresh_a_prime_state()
    spec = CONTROLLED_A_PRIME_PRE_SFT_SPEC
    return {
        "schema_version": SCHEMA_VERSION,
        "architecture_id": spec.identifier,
        "experiment_variant": spec.pre_sft_variant,
        "architecture": spec.architecture,
        "fusion_stage": "pre_mm_projector",
        "fusion_once": True,
        "query": "SigLIP visual patch features [32,729,1152]",
        "key_value": "CUT3R decoder-12 patch tokens [32,729,768]",
        "camera_tokens_included": False,
        "camera_token_count": 0,
        "cut3r_detached": True,
        "cut3r_source_layer": 12,
        "spatialstack_enabled": False,
        "constructor": "PyTorch default initialization",
        "constructor_seed": OFFICIAL_SEED,
        "c1_enabled": False,
        "c1_artifact": None,
        "fresh_fusion_state_sha256": state_hash,
        "fresh_fusion_state_hash_format": "sorted_state_dict_name_dtype_shape_raw_bytes_v1",
        "fresh_fusion_parameter_count": parameter_count,
        "feature_levels": list(PRE_SFT_FULL_FEATURE_LEVELS),
        "sample_indices": str(args.sample_indices.resolve()),
        "sample_indices_sha256": sample_hash,
        "video_count": 1199,
        "train_video_count": 1006,
        "validation_video_count": 193,
        "selected_frame_count": 2398,
        "expected_validation_tokens": 75656,
        "base_model": str(args.base_model.resolve()),
        "base_model_config_sha256": sha256_file(args.base_model / "config.json"),
        "siglip_model": str(args.siglip_model.resolve()),
        "siglip_config_sha256": sha256_file(args.siglip_model / "config.json"),
        "cut3r_feature_root": str(args.feature_root.resolve()),
        "cut3r_feature_subdir": "spatial_features",
        "cut3r_dec12_sidecar_root": str(dec12_root.resolve()),
        "cut3r_dec12_sidecar_count": len(dec12_files),
        "cut3r_dec12_available_sidecar_count": available_dec12_count,
        "cut3r_dec12_scene_identity_match": True,
        "cut3r_dec12_reference_sidecar": str(reference_sidecar.resolve()),
        "cut3r_dec12_reference_sidecar_sha256": sha256_file(reference_sidecar),
        "cut3r_dec12_reference_patch_shape": list(patch_tokens.shape),
        "torch_version": torch.__version__,
        "post_sft_state_loaded": False,
        "candidate_model_optimizer_constructed": False,
        "candidate_model_optimizer_step": False,
        "depth_probe_optimizer_planned": True,
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_worktree_dirty": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--siglip-model", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--sample-indices", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--check-existing",
        action="store_true",
        help="Recompute all identities and require --output to match exactly.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (
        args.base_model / "config.json",
        args.siglip_model / "config.json",
        args.feature_root,
        args.sample_indices,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.check_existing:
        if not args.output.is_file():
            raise FileNotFoundError(f"Missing A-prime manifest to check: {args.output}")
        observed = json.loads(args.output.read_text(encoding="utf-8"))
        expected = build_manifest(args)
        if observed != expected:
            differing = sorted(
                key for key in set(observed) | set(expected) if observed.get(key) != expected.get(key)
            )
            raise RuntimeError(f"Existing A-prime manifest is stale or mismatched: {differing}")
        print(json.dumps({"status": "PASS", "output": str(args.output), "checked": True}))
        return
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite A-prime manifest: {args.output}")
    payload = build_manifest(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": str(args.output), "fusion_sha256": payload["fresh_fusion_state_sha256"]}))


if __name__ == "__main__":
    main()
