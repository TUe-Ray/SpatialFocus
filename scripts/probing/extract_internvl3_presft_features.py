#!/usr/bin/env python3
"""Frozen InternVL3-8B no-spatial ScanNet representation extraction.

Uses the fixed 32-frame/two-target-frame SpatialFocus split.  This is a base
VLM diagnostic, not a member of the formal five-candidate C1 fusion study.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import dispatch_model
from PIL import Image
from torchvision import transforms
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import Qwen2SdpaAttention

from probe_layer_policy import COMMON_PROBE_LAYERS


SPLIT_SHA256 = "d478cb684958dfc25066821ec83d5216469577c9e282e33bdf87d3c88b200d8e"
MODEL_REVISION = "853e3a797a661694b1b8ece0cb72dc2b23e3dac9"
MODEL_LABEL = "internvl3_8b_base_presft"
FEATURE_LEVELS = ("visual_output", "fusion_output", "projected_features") + tuple(
    f"layer_{layer}" for layer in COMMON_PROBE_LAYERS
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def grid14(tokens: torch.Tensor, side: int) -> torch.Tensor:
    if tokens.ndim != 2 or tokens.shape[0] != side * side:
        raise ValueError(f"Expected {side}x{side} tokens, got {tuple(tokens.shape)}")
    value = tokens.detach().float().cpu().T.reshape(1, -1, side, side)
    value = F.interpolate(value, size=(14, 14), mode="bilinear", align_corners=False)
    if not torch.isfinite(value).all():
        raise RuntimeError("Non-finite feature")
    return value.squeeze(0).permute(1, 2, 0).contiguous().half()


def load_prompts(paths: list[Path], videos: list[dict]) -> dict[str, str]:
    required = {str(v["video_sample_id"]): str(v["video_path"]) for v in videos}
    prompts: dict[str, str] = {}
    for path in paths:
        for item in json.loads(path.read_text()):
            sample_id = str(item.get("id", ""))
            if sample_id not in required:
                continue
            if str(item.get("video")) != required[sample_id] or sample_id in prompts:
                raise RuntimeError(f"Annotation identity mismatch or duplicate: {sample_id}")
            human = next((str(t.get("value", "")) for t in item.get("conversations", [])
                          if str(t.get("from", "")).lower() in {"human", "user"}), "")
            if not human.strip():
                raise RuntimeError(f"Missing human turn: {sample_id}")
            prompts[sample_id] = human
    if set(prompts) != set(required):
        raise RuntimeError(f"Missing {len(set(required) - set(prompts))} fixed prompts")
    return prompts


def model_device_map(num_layers: int) -> dict[str, int]:
    # The pinned model has ~0.57 GiB ViT and ~0.434 GiB per Qwen2 layer.
    # Keep ViT intact; leave roughly 4 GiB free on each TITAN V for activations.
    device_map = {
        "vision_model": 0,
        "mlp1": 0,
        "language_model.model.embed_tokens": 0,
        "language_model.model.norm": 1,
        "language_model.lm_head": 1,
    }
    for layer in range(num_layers):
        device_map[f"language_model.model.layers.{layer}"] = 0 if layer < 13 else 1
    return device_map


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--sample-indices", type=Path, required=True)
    parser.add_argument("--forward-frames-root", type=Path, required=True)
    parser.add_argument("--annotation-scannet", type=Path, required=True)
    parser.add_argument("--annotation-route-plan", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit-videos", type=int)
    args = parser.parse_args()

    if torch.cuda.device_count() != 2:
        raise RuntimeError("Formal 32-frame extraction requires two visible TITAN V GPUs")
    if sha256_file(args.sample_indices) != SPLIT_SHA256:
        raise RuntimeError("Fixed ScanNet sample split SHA256 differs")
    if any((args.model / name).exists() for name in (
        "adapter_model.bin", "adapter_model.safetensors", "non_lora_trainables.bin"
    )):
        raise RuntimeError("Refusing a candidate-trained adapter/non-LoRA checkpoint")
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    if config.llm_config.num_hidden_layers != 28 or max(COMMON_PROBE_LAYERS) >= 28:
        raise RuntimeError("Unexpected InternVL3 decoder depth")
    if config.force_image_size != 448 or config.downsample_ratio != 0.5:
        raise RuntimeError("Unexpected InternVL3 image token layout")
    samples = json.loads(args.sample_indices.read_text())
    videos = sorted((v for v in samples["videos"] if v["source_dataset"] == "scannet"),
                    key=lambda v: (int(v["selected_order"]), str(v["video_path"])))
    if len(videos) != 1199 or sum(v["split"] == "train" for v in videos) != 1006:
        raise RuntimeError("Expected the fixed ScanNet 1006/193 video split")
    prompts = load_prompts([args.annotation_scannet, args.annotation_route_plan], videos)
    if args.start_index < 0 or args.start_index >= len(videos):
        raise ValueError("Invalid start index")
    selected = videos[args.start_index:]
    if args.limit_videos is not None:
        selected = selected[:args.limit_videos]
    if not selected:
        raise ValueError("No selected videos")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    image_token = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
    if image_token == tokenizer.unk_token_id:
        raise RuntimeError("Missing InternVL image context token")
    device_map = model_device_map(28)
    # The official ViT constructor calls .item() on a linspace during init;
    # meta-device initialization used by from_pretrained(device_map=...) fails.
    # CPU initialization/loading avoids changing the official remote code.
    model = AutoModel.from_pretrained(
        args.model, trust_remote_code=True, local_files_only=True, torch_dtype=torch.float16,
        use_flash_attn=False, low_cpu_mem_usage=False,
    ).eval()
    for layer_idx, layer in enumerate(model.language_model.model.layers):
        replacement = Qwen2SdpaAttention(model.config.llm_config, layer_idx=layer_idx).to(
            dtype=layer.self_attn.q_proj.weight.dtype
        )
        replacement.load_state_dict(layer.self_attn.state_dict(), strict=True)
        layer.self_attn = replacement
    model.language_model.model._attn_implementation = "sdpa"
    model = dispatch_model(model, device_map=device_map)
    model.requires_grad_(False)
    # InternVL's no-FlashAttention constructor selects eager Qwen attention.
    # SDPA is essential for the fixed 8192-image-token, 32-frame protocol.
    model.language_model.config._attn_implementation = "sdpa"
    model.config.llm_config._attn_implementation = "sdpa"
    if any(p.requires_grad for p in model.parameters()):
        raise RuntimeError("Base model unexpectedly trainable")
    if any(p.device.type == "meta" for p in model.parameters()):
        raise RuntimeError("Unexpected meta parameter in two-GPU placement")
    if model.num_image_token != 256:
        raise RuntimeError(f"Unexpected tokens/image: {model.num_image_token}")

    image_transform = transforms.Compose([
        transforms.Resize((448, 448), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    feature_root = args.output_root / "features" / MODEL_LABEL
    manifest_path = args.output_root / f"{MODEL_LABEL}_run_manifest.json"
    manifest = {
        "schema": "spatialfocus.internvl3_base_presft_features.v1",
        "status": "in_progress", "candidate": "base_no_spatial", "model_label": MODEL_LABEL,
        "model_root": str(args.model.resolve()),
        "model_repo": "OpenGVLab/InternVL3-8B",
        "model_revision": MODEL_REVISION,
        "model_weight_index_sha256": sha256_file(args.model / "model.safetensors.index.json"),
        "model_config_sha256": sha256_file(args.model / "config.json"),
        "extractor_sha256": sha256_file(Path(__file__)),
        "sample_indices_sha256": SPLIT_SHA256,
        "annotation_sha256": {p.name: sha256_file(p) for p in [args.annotation_scannet, args.annotation_route_plan]},
        "feature_levels": list(FEATURE_LEVELS),
        "layer_indexing": "L -> hidden_states[L+1]; final L27 after model norm",
        "frames_per_video": 32, "selected_target_frames_per_video": 2,
        "image_size": 448, "image_tokens_per_frame": 256,
        "target_grid": [14, 14], "dtype": "float16", "attention_implementation": "sdpa",
        "spatial_features": "none; fusion_output is identity copy of visual_output",
        "checkpoint_state": "official base only; no candidate post-SFT adapter/non-LoRA/fusion state",
        "device_map": device_map, "optimizer_steps": 0,
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "gpu_names": [torch.cuda.get_device_name(i) for i in range(2)],
        "hostname": socket.gethostname(), "pid": __import__("os").getpid(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        for field in ("model_revision", "model_weight_index_sha256", "model_config_sha256",
                      "extractor_sha256", "sample_indices_sha256", "annotation_sha256",
                      "feature_levels", "frames_per_video", "image_size", "attention_implementation"):
            if old.get(field) != manifest[field]:
                raise RuntimeError(f"Refusing to mix incompatible extraction: {field}")
    else:
        write_json(manifest_path, manifest)

    for video_number, video in enumerate(selected, start=args.start_index):
        started = time.perf_counter()
        scene = Path(video["video_path"]).stem
        chosen = {int(f["frame_index"]): str(f["frame_sample_id"]) for f in video["frames"]}
        if len(chosen) != 2 or any(i < 0 or i >= 32 for i in chosen):
            raise RuntimeError(f"Invalid target frame indices: {scene}")
        expected = [feature_root / level / f"frame_{fsid}.pt"
                    for level in FEATURE_LEVELS for fsid in chosen.values()]
        if all(path.is_file() for path in expected):
            print(json.dumps({"status": "SKIP_COMPLETE", "video_index": video_number, "scene": scene}), flush=True)
            continue
        frame_path = args.forward_frames_root / "frames" / "scannet" / f"{scene}.pt"
        rgb = torch.load(frame_path, map_location="cpu", weights_only=False)
        frames = rgb["frames_rgb_uint8"]
        if frames.shape[0] != 32 or frames.dtype != torch.uint8 or frames.shape[-1] != 3:
            raise RuntimeError(f"Invalid 32-frame RGB cache: {frame_path}")
        visual_selected = {}
        projected_selected = {}
        all_projected = []
        for frame_index in range(32):
            pixels = image_transform(Image.fromarray(frames[frame_index].numpy())).unsqueeze(0)
            pixels = pixels.to(device="cuda:0", dtype=torch.float16)
            with torch.inference_mode():
                raw = model.vision_model(pixel_values=pixels, output_hidden_states=False,
                                         return_dict=True).last_hidden_state[:, 1:, :]
                raw = raw.reshape(1, 32, 32, -1)
                unshuffled = model.pixel_shuffle(raw, scale_factor=model.downsample_ratio)
                projected = model.mlp1(unshuffled.reshape(1, 256, -1))
            if frame_index in chosen:
                visual_selected[frame_index] = grid14(raw.reshape(1024, -1), 32)
                projected_selected[frame_index] = grid14(projected.reshape(256, -1), 16)
            all_projected.append(projected.squeeze(0))
        video_prompt = "\n".join(f"Frame {i + 1}: <image>" for i in range(32))
        human = prompts[str(video["video_sample_id"])]
        if human.count("<image>") > 1:
            raise RuntimeError(f"Unexpected multiple annotation image placeholders: {scene}")
        human = human.replace("<image>", video_prompt, 1) if "<image>" in human else video_prompt + "\n" + human
        template = copy.deepcopy(model.conv_template)
        template.system_message = model.system_message
        template.append_message(template.roles[0], human)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()
        image_tokens = "<img>" + "<IMG_CONTEXT>" * model.num_image_token + "</img>"
        for _ in range(32):
            query = query.replace("<image>", image_tokens, 1)
        if "<image>" in query:
            raise RuntimeError("Unexpanded image placeholder")
        encoded = tokenizer(query, return_tensors="pt")
        ids = encoded["input_ids"].to("cuda:0")
        positions = torch.nonzero(ids[0] == image_token, as_tuple=False).flatten()
        if positions.numel() != 32 * model.num_image_token:
            raise RuntimeError(f"Image-token count mismatch: {positions.numel()}")
        selected_positions = {
            frame: positions[frame * 256:(frame + 1) * 256] for frame in chosen
        }
        embeds = model.language_model.get_input_embeddings()(ids).clone()
        embeds[0, positions] = torch.cat(all_projected, dim=0).to(embeds.device)
        captured = {}
        handles = []

        def capture_layer(layer: int):
            def hook(_module, _args, output):
                hidden = output[0] if isinstance(output, tuple) else output
                captured[layer] = {
                    frame: hidden[0].index_select(0, index.to(hidden.device)).detach().cpu()
                    for frame, index in selected_positions.items()
                }
            return hook

        for layer in COMMON_PROBE_LAYERS:
            module = model.language_model.model.norm if layer == 27 else model.language_model.model.layers[layer]
            handles.append(module.register_forward_hook(capture_layer(layer)))
        try:
            with torch.inference_mode():
                model.language_model.model(
                    inputs_embeds=embeds,
                    attention_mask=encoded["attention_mask"].to(embeds.device),
                    use_cache=False, return_dict=True,
                )
        finally:
            for handle in handles:
                handle.remove()
        if set(captured) != set(COMMON_PROBE_LAYERS):
            raise RuntimeError(f"Missing LLM layer captures for {scene}")
        for frame, fsid in chosen.items():
            values = {
                "visual_output": visual_selected[frame],
                "fusion_output": visual_selected[frame],  # identity: no spatial fusion
                "projected_features": projected_selected[frame],
            }
            values.update({f"layer_{layer}": grid14(captured[layer][frame], 16)
                           for layer in COMMON_PROBE_LAYERS})
            for level, value in values.items():
                path = feature_root / level / f"frame_{fsid}.pt"
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(value, path)
        print(json.dumps({"status": "PASS", "video_index": video_number, "scene": scene,
                          "sequence_tokens": ids.shape[1], "selected_frames": sorted(chosen),
                          "runtime_seconds": round(time.perf_counter() - started, 3),
                          "peak_gpu_memory_bytes": [torch.cuda.max_memory_allocated(i) for i in range(2)]}),
              flush=True)

    complete = sum(all((feature_root / level / f"frame_{f['frame_sample_id']}.pt").is_file()
                       for level in FEATURE_LEVELS for f in video["frames"]) for video in videos)
    manifest.update(status="complete" if complete == len(videos) else "in_progress",
                    complete_videos=complete, updated_at=datetime.now(timezone.utc).isoformat())
    write_json(manifest_path, manifest)
    print(json.dumps({"status": manifest["status"], "complete_videos": complete,
                      "expected_videos": len(videos)}), flush=True)


if __name__ == "__main__":
    main()
