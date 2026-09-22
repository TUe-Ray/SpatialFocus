#!/usr/bin/env python3
"""One-minibatch, pre-SFT QA gradient proxy for the held-out Qwen3.5 A/B pair.

The model/data/LoRA construction comes from the pinned external held-out
repository.  This runner only supplies the fixed SpatialFocus calibration,
read-only transferred inputs, grouped gradient reduction, and provenance.
It never constructs an optimizer or loads a candidate SFT checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch


SOURCE_COMMIT = "d37ed60aa45d444baf13e4d3ac57e14e6ac013ad"
BASE_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
TRANSFER_SHA256 = "b9615df512b5570686bac199348f2cd276e558f7338f0b1425b6eed8170ac629"
QA_ID = "7e8340c2-eb6e-4736-abfe-13ca82a9e62e"
VIDEO = "scannet/videos/scene0384_00.mp4"
FRAME_IDS = (0, 41, 82, 123, 165, 206, 247, 289, 330, 371, 412, 454, 495, 536, 578, 619,
             660, 701, 743, 784, 825, 867, 908, 949, 990, 1032, 1073, 1114, 1156, 1197, 1238, 1280)
CANDIDATES = {
    "candidate_a": {
        "source_id": "a_premerger_cross_attn",
        "layers": (23,),
        "source_manifest": "candidate-a-l23-full.json",
        "sidecar_sha256": "dc845feb1948bfa86612b48c05fa5264aaf598a49d12e11d940c275c79e8a794",
    },
    "candidate_b": {
        "source_id": "b_llm_add",
        "layers": (11, 17, 23),
        "source_manifest": "candidate-b-three-layer-full.json",
        "sidecar_sha256": "6553b606fdbf382f2f8756085cd806741f1791b67c1a14a7dd585629186ff6d1",
    },
}
FORBIDDEN_NAMES = {
    "adapter_model.bin", "adapter_model.safetensors", "non_lora_trainables.bin",
    "controlled_vggt_fusion.bin", "optimizer.pt", "optimizer.bin", "trainer_state.json",
}
EXPECTED_VERSIONS = {
    "torch": "2.10.0+cu129", "torchvision": "0.25.0+cu129", "transformers": "5.3.0",
    "peft": "0.21.0", "accelerate": "1.13.0", "flash_attn": "2.8.3",
    "causal-conv1d": "1.7.0", "flash-linear-attention": "0.6.0",
    "einops": "0.8.2", "qwen-vl-utils": "0.0.14", "decord": "0.6.0",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def git_head(root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def package_versions() -> dict[str, str]:
    versions = {name: importlib.metadata.version(name) for name in EXPECTED_VERSIONS}
    versions.update({"python": platform.python_version(), "cuda_runtime": str(torch.version.cuda)})
    for name, expected in EXPECTED_VERSIONS.items():
        if versions[name] != expected:
            raise RuntimeError(f"{name} must match source runtime {expected}, got {versions[name]}")
    if versions["python"] != "3.12.14" or versions["cuda_runtime"] != "12.9":
        raise RuntimeError(f"Python/CUDA source runtime mismatch: {versions}")
    return versions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "run", "aggregate"), required=True)
    parser.add_argument("--candidate", choices=tuple(CANDIDATES))
    parser.add_argument("--source-root", type=Path, default=Path("/scratch-shared/shuang/SpatialStack-Qwen35-VGGT-d37ed60"))
    parser.add_argument("--delivery-root", type=Path, default=Path("/scratch-shared/geusdd/to_shuang"))
    parser.add_argument("--qa-annotation", type=Path, default=Path("/home/shuang/probing_data/probe_targets_2f_v1/manifests/merged_qa_scannet_train.json"))
    parser.add_argument("--old-rgb-cache", type=Path, default=Path("/home/shuang/probing_data/forward_frames_32_v1/frames/scannet/scene0384_00.pt"))
    parser.add_argument("--output-root", type=Path, default=Path("/home/shuang/proxy_outputs/pre_sft_zero_cost_proxies_a100_v1/qwen35_cached_vggt_ab_v1"))
    args = parser.parse_args()
    if args.mode == "run" and args.candidate is None:
        parser.error("--candidate is required for --mode run")
    return args


def source_config(args: argparse.Namespace) -> None:
    if git_head(args.source_root) != SOURCE_COMMIT:
        raise RuntimeError("Held-out source checkout is not the pinned d37ed60 commit")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=args.source_root):
        raise RuntimeError("Held-out source checkout is dirty")
    source_python = args.source_root / "src"
    if str(source_python) not in sys.path:
        sys.path.insert(0, str(source_python))


def validate_transfer(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = args.delivery_root / "transfer-manifest.json"
    if sha256(manifest_path) != TRANSFER_SHA256:
        raise RuntimeError("Transferred manifest SHA-256 differs from verified handoff")
    transfer = json.loads(manifest_path.read_text(encoding="utf-8"))
    if transfer["base_model"]["huggingface_revision"] != BASE_REVISION:
        raise RuntimeError("Qwen base revision differs from the pinned pretrained snapshot")
    if transfer["base_model"]["forbidden_post_sft_artifacts_transferred"]:
        raise RuntimeError("Handoff contains forbidden post-SFT artifacts")
    for record in transfer["records"]:
        path = Path(record["destination_path"])
        if not path.is_relative_to(args.delivery_root) or not path.is_file():
            raise RuntimeError(f"Transferred file missing or outside handoff root: {path}")
        if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
            raise RuntimeError(f"Transferred file failed byte/hash verification: {path}")
    wheel_manifest_path = args.delivery_root / "wheels/wheel-manifest.json"
    wheel_manifest = json.loads(wheel_manifest_path.read_text(encoding="utf-8"))
    if (wheel_manifest.get("python"), wheel_manifest.get("torch"), wheel_manifest.get("cuda_runtime"),
            wheel_manifest.get("torch_cuda_arch_list")) != ("3.12.14", "2.10.0+cu129", "12.9", "8.0"):
        raise RuntimeError("Transferred fast-path wheels have incompatible runtime provenance")
    if (wheel_manifest["causal_conv1d"]["git_commit"],
            wheel_manifest["flash_linear_attention"]["git_commit"]) != (
                "cd81f0413cad2fc1e6f17e785ac39f59aae690cd",
                "954438d1fcb5e1bb05c22f9908de9c5c2df74ae5"):
        raise RuntimeError("Transferred fast-path wheels differ from source Git commits")
    transfer_wheels = {Path(item["destination_path"]).name: item["sha256"]
                       for item in transfer["records"] if "/wheels/" in item["destination_path"]}
    if {item["file"]: item["sha256"] for item in wheel_manifest["files"]} != transfer_wheels:
        raise RuntimeError("Wheel provenance manifest differs from verified transfer records")
    base = args.delivery_root / "models/base/Qwen3.5-4B"
    forbidden = sorted(str(path) for path in base.rglob("*") if path.name in FORBIDDEN_NAMES)
    if forbidden:
        raise RuntimeError(f"Forbidden trained state is present in base snapshot: {forbidden}")
    config = json.loads((base / "config.json").read_text(encoding="utf-8"))
    if config["model_type"] != "qwen3_5" or config.get("use_cached_vggt", False):
        raise RuntimeError("Base snapshot is not the plain pretrained Qwen3.5 model")
    index = json.loads((base / "model.safetensors.index.json").read_text(encoding="utf-8"))
    shards = set(index["weight_map"].values())
    if shards != set(transfer["base_model"]["referenced_shards"]):
        raise RuntimeError("Base snapshot shard index differs from transfer manifest")
    if any("controlled_vggt_fusion" in key or "lora_" in key for key in index["weight_map"]):
        raise RuntimeError("Base checkpoint index contains trained fusion or LoRA tensors")
    return {"manifest_path": str(manifest_path), "manifest_sha256": TRANSFER_SHA256,
            "wheel_manifest_path": str(wheel_manifest_path),
            "wheel_manifest_sha256": sha256(wheel_manifest_path),
            "verified_file_count": len(transfer["records"]), "base_model": str(base),
            "base_revision": BASE_REVISION, "base_shards": sorted(shards)}


def calibration_record(args: argparse.Namespace) -> dict[str, Any]:
    rows = json.loads(args.qa_annotation.read_text(encoding="utf-8"))
    found = [row for row in rows if row.get("id") == QA_ID]
    if len(found) != 1 or found[0].get("video") != VIDEO:
        raise RuntimeError("Fixed ScanNet QA ID/video is not uniquely present")
    row = found[0]
    if row["conversations"][-1]["value"].strip() != "1.3":
        raise RuntimeError("Fixed QA answer has changed")
    rgb = torch.load(args.old_rgb_cache, map_location="cpu", weights_only=False)
    if tuple(rgb["source_frame_indices"].tolist()) != FRAME_IDS or rgb["source_video_relative_path"] != VIDEO:
        raise RuntimeError("Existing 32-frame RGB cache identity differs from the fixed calibration")
    return row


def prepare_candidate(args: argparse.Namespace, candidate: str, calibration: dict[str, Any]) -> dict[str, Any]:
    spec = CANDIDATES[candidate]
    original = args.delivery_root / "manifests" / spec["source_manifest"]
    source_manifest = json.loads(original.read_text(encoding="utf-8"))
    matches = [row for row in source_manifest["records"] if row.get("dataset") == "vlm3r_scannet" and row.get("video") == VIDEO]
    if source_manifest.get("schema") != "spatialfocus.cached_vggt.v1" or len(matches) != 1:
        raise RuntimeError(f"Canonical {candidate} manifest lacks one exact calibration record")
    record = dict(matches[0])
    sidecar = args.delivery_root / candidate / "scene0384_00.pt"
    if record["sha256"] != spec["sidecar_sha256"] or sha256(sidecar) != spec["sidecar_sha256"]:
        raise RuntimeError(f"{candidate} sidecar hash differs from canonical manifest")
    if tuple(record["frame_idx"]) != FRAME_IDS:
        raise RuntimeError(f"{candidate} VGGT frame IDs differ from old RGB cache")
    if record.get("frame_positions") is not None:
        raise RuntimeError("Unexpected manifest frame subset; fixed 32-frame sidecar is required")
    record["sidecar"] = str(sidecar)
    candidate_root = args.output_root / candidate
    candidate_root.mkdir(parents=True, exist_ok=True)
    runtime_manifest = candidate_root / "runtime_manifest.json"
    write_json(runtime_manifest, {"schema": "spatialfocus.cached_vggt.v1", "records": [record]})
    one_qa = candidate_root / "calibration_qa.json"
    write_json(one_qa, [calibration])
    return {"source_manifest": str(original), "source_manifest_sha256": sha256(original),
            "runtime_manifest": str(runtime_manifest), "runtime_manifest_sha256": sha256(runtime_manifest),
            "sidecar": str(sidecar), "sidecar_sha256": spec["sidecar_sha256"],
            "calibration_qa": str(one_qa), "calibration_qa_sha256": sha256(one_qa),
            "layers": list(spec["layers"]), "frame_ids": list(FRAME_IDS)}


def to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=torch.bfloat16 if value.is_floating_point() else None)
    if isinstance(value, dict):
        return {key: to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [to_device(item, device) for item in value]
    return value


def group_scores(groups: dict[str, list[torch.nn.Parameter]]) -> dict[str, dict[str, float | int]]:
    scored: dict[str, dict[str, float | int]] = {}
    for name, parameters in groups.items():
        row: dict[str, float | int] = {"parameter_elements": sum(p.numel() for p in parameters),
                                       "parameters_with_gradient": 0, "gradnorm": 0.0, "snip": 0.0, "fisher": 0.0}
        for parameter in parameters:
            if parameter.grad is None:
                continue
            gradient = parameter.grad.detach().float()
            value = parameter.detach().float()
            if not bool(torch.isfinite(gradient).all()):
                raise RuntimeError(f"Nonfinite gradient in {name}")
            row["parameters_with_gradient"] += parameter.numel()
            row["gradnorm"] += float(torch.linalg.vector_norm(gradient).item())
            row["snip"] += float((value * gradient).abs().sum().item())
            row["fisher"] += float(gradient.square().sum().item())
        scored[name] = row
    group_rows = tuple(scored.values())
    scored["total"] = {key: sum(float(row[key]) for row in group_rows) for key in ("gradnorm", "snip", "fisher")}
    for key in ("parameter_elements", "parameters_with_gradient"):
        scored["total"][key] = sum(int(row[key]) for row in group_rows)
    return scored


def lora_initialization(model: torch.nn.Module) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for side in ("lora_A", "lora_B"):
        parameters = [parameter for name, parameter in model.named_parameters() if f".{side}." in name]
        if not parameters:
            raise RuntimeError(f"Missing fresh PEFT {side} tensors")
        summary[side] = {
            "tensor_count": len(parameters),
            "parameter_elements": sum(parameter.numel() for parameter in parameters),
            "nonzero_elements": sum(int(torch.count_nonzero(parameter.detach()).item()) for parameter in parameters),
            "max_abs": max(float(parameter.detach().float().abs().max().item()) for parameter in parameters),
        }
    summary["peft_version"] = importlib.metadata.version("peft")
    return summary


def run_candidate(args: argparse.Namespace, candidate: str, provenance: dict[str, Any]) -> dict[str, Any]:
    candidate_started = time.perf_counter()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("The proxy requires one allocated, visible CUDA GPU")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda:0")
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    source_config(args)
    model_root = provenance["transfer"]["base_model"]
    scene = provenance["candidates"][candidate]
    os.environ["VLM3R_SCANNET_ANNOTATION"] = scene["calibration_qa"]
    os.environ["VLM3R_SCANNET_MEDIA_ROOT"] = str(args.delivery_root / "media")
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    from transformers import AutoConfig, AutoProcessor, AutoTokenizer, set_seed
    from qwen_vl.model.modeling_qwen3_5 import Qwen3_5ForConditionalGenerationWithGeometry
    from qwen_vl.train.argument import DataArguments, ModelArguments
    from qwen_vl.train.train_qwen import add_language_lora, audit_controlled_trainable_parameters, set_model
    from qwen_vl.data.data_qwen import make_supervised_data_module

    spec = CANDIDATES[candidate]
    set_seed(42)
    model_args = ModelArguments(model_name_or_path=model_root, use_cached_vggt=True,
                                controlled_fusion_candidate=spec["source_id"], lora_enable=True,
                                lora_r=128, lora_alpha=256, lora_dropout=0.05)
    config = AutoConfig.from_pretrained(model_root, local_files_only=True)
    for key in ("use_geometry_encoder", "geometry_encoder_type", "geometry_encoder_path", "reference_frame",
                "feature_fusion_method", "fusion_num_layers", "geometry_merger_type", "geometry_fusion_layers",
                "geometry_encoder_layers", "include_camera_token", "pos_encoding_type", "vision_language_fusion_layers",
                "use_cached_vggt", "controlled_fusion_candidate", "controlled_cross_attention_heads",
                "controlled_fusion_dropout", "controlled_projector_hidden_dim"):
        setattr(config, key, getattr(model_args, key))
    dimensions = (config.vision_config.hidden_size, config.vision_config.out_hidden_size,
                  config.text_config.hidden_size, config.text_config.num_hidden_layers)
    if dimensions != (1024, 2560, 2560, 32):
        raise RuntimeError(f"Unexpected Qwen3.5-4B model dimensions: {dimensions}")
    load_started = time.perf_counter()
    model = Qwen3_5ForConditionalGenerationWithGeometry.from_pretrained(
        model_root, config=config, local_files_only=True, attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16)
    model.config.use_cache = False
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    set_model(model_args, model)
    # Match the source training order: PEFT initializes fresh LoRA on CPU,
    # then the Trainer moves the complete trainable state to its CUDA device.
    model = add_language_lora(model_args, model)
    model.to(device)
    audit = audit_controlled_trainable_parameters(model)
    model_load_seconds = time.perf_counter() - load_started
    groups = {
        "lora": [parameter for name, parameter in model.named_parameters() if ".lora_A." in name or ".lora_B." in name],
        "fusion": [parameter for name, parameter in model.named_parameters() if "controlled_vggt_fusion" in name and "lora_" not in name],
    }
    if not all(groups.values()) or len({id(p) for ps in groups.values() for p in ps}) != sum(map(len, groups.values())):
        raise RuntimeError("LoRA/fusion groups are empty or overlap")
    trainable = {id(p) for p in model.parameters() if p.requires_grad}
    selected = {id(p) for ps in groups.values() for p in ps}
    if trainable != selected or audit["total"] != sum(p.numel() for ps in groups.values() for p in ps):
        raise RuntimeError("Primary selected groups differ from the source SFT trainable scope")
    if any(p.is_meta or p.device.type != "cuda" for ps in groups.values() for p in ps):
        raise RuntimeError("An intended trainable parameter is on CPU or meta")
    if any(p.requires_grad for p in model.get_base_model().model.visual.parameters()):
        raise RuntimeError("The native Qwen vision tower/merger must be frozen")
    lora_init = lora_initialization(model)
    versions_before = {id(p): p._version for ps in groups.values() for p in ps}
    processor = AutoProcessor.from_pretrained(model_root, local_files_only=True, padding_side="right")
    tokenizer = AutoTokenizer.from_pretrained(model_root, local_files_only=True,
                                              model_max_length=12800, padding_side="right", use_fast=False)
    data_args = DataArguments(dataset_use="vlm3r_scannet", data_flatten=False,
                              cached_vggt_manifest=scene["runtime_manifest"], cached_vggt_num_frames=32,
                              cached_vggt_layers=list(spec["layers"]),
                              cached_vggt_require_exact_layers=candidate == "candidate_a")
    data_args.model_type = "qwen3.5"
    data_args.image_processor = processor.image_processor
    data_args.processor = processor
    data_args.use_cached_vggt = True
    data_module = make_supervised_data_module(tokenizer, data_args)
    if len(data_module["train_dataset"]) != 1:
        raise RuntimeError("Expected exactly one fixed calibration QA sample")
    sample = data_module["train_dataset"][0]
    batch = data_module["data_collator"]([sample])
    supervised_tokens = int((batch["labels"] != -100).sum().item())
    if supervised_tokens <= 0 or len(batch["cached_vggt_features"][0][str(spec["layers"][0])]) != 32:
        raise RuntimeError("The fixed QA calibration is not fully supervised with 32 frames")
    batch = to_device(batch, device)
    model.train(True)
    model.zero_grad(set_to_none=True)
    set_seed(42)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    output = model(**batch, use_cache=False, return_dict=True)
    loss = output.loss
    if loss is None or not bool(torch.isfinite(loss).all()):
        raise RuntimeError(f"Nonfinite or absent QA CE loss: {loss}")
    loss.backward()
    torch.cuda.synchronize(device)
    backward_seconds = time.perf_counter() - started
    scores = group_scores(groups)
    if any(row["parameters_with_gradient"] != row["parameter_elements"] for row in scores.values()):
        raise RuntimeError(f"Incomplete selected gradient coverage: {scores}")
    if any(not math.isfinite(float(row[key])) for row in scores.values() for key in ("gradnorm", "snip", "fisher")):
        raise RuntimeError("Nonfinite grouped proxy score")
    if any(p._version != versions_before[id(p)] for ps in groups.values() for p in ps):
        raise RuntimeError("A selected parameter was modified during the proxy")
    actual_dtypes = {name: sorted({str(parameter.dtype) for parameter in parameters})
                     for name, parameters in groups.items()}
    attention_implementation = str(model.config._attn_implementation)
    if attention_implementation != "flash_attention_2":
        raise RuntimeError(f"Unexpected Qwen attention implementation: {attention_implementation}")
    result = {
        "candidate": candidate, "source_candidate": spec["source_id"], "status": "PASS",
        "loss_definition": "L_proxy = L_QA", "ce_loss": float(loss.detach().float().item()),
        "proxy_groups": scores, "source_trainable_audit": audit,
        "lora_initialization": lora_init,
        "calibration": {"qa_id": QA_ID, "video": VIDEO, "answer": "1.3", "frame_ids": list(FRAME_IDS),
                        "input_ids_shape": list(batch["input_ids"].shape), "supervised_label_tokens": supervised_tokens},
        "runtime": {"hostname": socket.gethostname(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                    "gpu": torch.cuda.get_device_name(device),
                    "gpu_total_bytes": torch.cuda.get_device_properties(device).total_memory,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                    "model_load_seconds": model_load_seconds, "forward_backward_seconds": backward_seconds,
                    "candidate_total_seconds": time.perf_counter() - candidate_started,
                    "dtype": str(next(model.parameters()).dtype),
                    "selected_group_dtypes": actual_dtypes, "attention": attention_implementation,
                    "gradient_checkpointing": bool(model.is_gradient_checkpointing),
                    "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
                    "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
                    "cpu_meta_trainable_offload": False},
        "provenance": provenance, "optimizer_constructed": False, "optimizer_steps": 0,
        "post_sft_state_loaded": False, "parameter_versions_unchanged": True,
    }
    return result


def aggregate(args: argparse.Namespace) -> None:
    rows = []
    for candidate in CANDIDATES:
        path = args.output_root / candidate / "results.json"
        result = json.loads(path.read_text(encoding="utf-8"))
        if result["status"] != "PASS":
            raise RuntimeError(f"Cannot aggregate incomplete candidate {candidate}")
        total = result["proxy_groups"]["total"]
        row = {"candidate": candidate, "selected_params": total["parameter_elements"],
               "gradient_covered_params": total["parameters_with_gradient"],
               "ce_loss": result["ce_loss"], "gradnorm": total["gradnorm"],
               "snip": total["snip"], "fisher": total["fisher"],
               "peak_vram_bytes": result["runtime"]["peak_allocated_bytes"],
               "forward_backward_seconds": result["runtime"]["forward_backward_seconds"],
               "candidate_total_seconds": result["runtime"]["candidate_total_seconds"]}
        for group in ("lora", "fusion"):
            for key, value in result["proxy_groups"][group].items():
                row[f"{group}_{key}"] = value
        rows.append(row)
    with (args.output_root / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(args.output_root / "results.json", {"status": "PASS", "cohort": "Qwen3.5-4B + cached VGGT A/B", "results": rows})
    lines = ["# Qwen3.5 cached-VGGT A/B pre-SFT QA proxy", "",
             "Separate cohort from the LLaVA/CUT3R 17-model pilot. One QA minibatch and one backward per candidate; no optimizer/update.", "",
             "| Candidate | Selected / gradient-covered | CE | GradNorm | SNIP | Fisher | Peak allocated GiB | Forward+backward s | Total s |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['candidate']} | {row['selected_params']} / {row['gradient_covered_params']} | {row['ce_loss']:.6g} | {row['gradnorm']:.6g} | {row['snip']:.6g} | {row['fisher']:.6g} | {row['peak_vram_bytes']/2**30:.3f} | {row['forward_backward_seconds']:.2f} | {row['candidate_total_seconds']:.2f} |")
    lines.extend(["", "Group-level LoRA and fusion contributions are in `results.csv` and per-candidate JSON."])
    (args.output_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.mode == "aggregate":
        aggregate(args)
        return
    source_config(args)
    transfer = validate_transfer(args)
    calibration = calibration_record(args)
    versions = package_versions()
    args.output_root.mkdir(parents=True, exist_ok=True)
    candidates = {name: prepare_candidate(args, name, calibration) for name in CANDIDATES}
    provenance = {"spatialfocus_git_commit": git_head(Path(__file__).resolve().parents[2]),
                  "proxy_runner_sha256": sha256(Path(__file__).resolve()),
                  "heldout_source_git_commit": SOURCE_COMMIT, "transfer": transfer,
                  "calibration_annotation": str(args.qa_annotation),
                  "calibration_annotation_sha256": sha256(args.qa_annotation),
                  "old_rgb_cache_sha256": sha256(args.old_rgb_cache),
                  "environment": versions, "candidates": candidates,
                  "vggt_checkpoint": {"identifier": "facebook/VGGT-1B",
                                      "revision": "860abec7937da0a4c03c41d3c269c366e82abdf9",
                                      "model_safetensors_sha256": "f164acf60724910d8fe1578bb499d800850c7bb0948db7555c413f9fbe60467e"}}
    write_json(args.output_root / "preflight.json", provenance)
    if args.mode == "preflight":
        print(json.dumps({"status": "PASS", "mode": "preflight", "output_root": str(args.output_root)}))
        return
    result_path = args.output_root / args.candidate / "results.json"
    if result_path.exists():
        raise FileExistsError(f"Refusing to overwrite previous scientific result: {result_path}")
    try:
        result = run_candidate(args, args.candidate, provenance)
        write_json(result_path, result)
        print(json.dumps({"status": "PASS", "candidate": args.candidate, "result": str(result_path)}))
    except BaseException as exc:
        write_json(args.output_root / args.candidate / "error.json",
                   {"candidate": args.candidate, "error": str(exc), "traceback": traceback.format_exc(),
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved() if torch.cuda.is_available() else None})
        raise


if __name__ == "__main__":
    main()
