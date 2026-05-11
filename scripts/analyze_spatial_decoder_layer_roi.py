#!/usr/bin/env python
"""ROI similarity maps for CUT3R and PI3 decoder-layer outputs.

This is an inference-only diagnostic. It runs the spatial encoders on selected
VSI/VLM-3R samples, captures the last decoder layers, pools each native grid to
the target 14x14 visual-token grid, and visualizes anchor-token similarity maps.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import random
import sys
import types
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from transformers import AutoImageProcessor, AutoTokenizer

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_roi_similarity_maps import (  # noqa: E402
    build_anchors,
    corr_values,
    extract_category,
    extract_question,
    get_video_tensor,
    parse_anchor_coords,
    parse_int_list,
    parse_str_list,
    pool_grid,
    safe_id,
    sample_identifier,
    save_rgb_frame,
    select_frames,
    similarity_map,
    str2bool,
    tensor_to_uint8_image,
)


def parse_grid(value: str) -> tuple[int, int]:
    parts = value.lower().replace("x", ",").split(",")
    if len(parts) != 2:
        raise ValueError(f"Expected grid as HxW or H,W, got {value!r}")
    return int(parts[0]), int(parts[1])


def infer_square_hw(token_count: int, name: str) -> tuple[int, int]:
    side = int(math.isqrt(int(token_count)))
    if side * side != int(token_count):
        raise RuntimeError(f"{name} has non-square token count {token_count}; provide explicit handling.")
    return side, side


def load_raw_items(data_path: str) -> list[dict[str, Any]]:
    paths: list[str]
    if data_path.endswith((".yaml", ".yml")):
        with open(data_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        paths = [dataset["json_path"] for dataset in cfg.get("datasets", []) if dataset.get("json_path")]
    else:
        paths = [data_path]

    items: list[dict[str, Any]] = []
    for path in paths:
        if not os.path.exists(path):
            print(f"[WARN] data file does not exist, skipping: {path}")
            continue
        with open(path, "r", encoding="utf-8") as f:
            if path.endswith(".jsonl"):
                loaded = [json.loads(line) for line in f if line.strip()]
            else:
                loaded = json.load(f)
        for item in loaded:
            item["_annotation_path"] = path
        items.extend(loaded)
        print(f"Loaded {len(loaded)} samples from {path}")
    items.sort(key=lambda item: (
        str(item.get("_annotation_path", "")),
        str(item.get("id", "")),
        str(item.get("question_id", "")),
        str(item.get("video", "")),
        str(item.get("image", "")),
        str(item.get("data_source", "")),
        str((item.get("conversations") or [{}])[0].get("value", ""))[:128],
    ))
    print(f"Loaded {len(items)} total samples from {data_path}")
    return items


def select_samples(raw_items: list[dict[str, Any]], args: argparse.Namespace):
    categories = parse_str_list(args.categories)
    requested_ids = parse_str_list(args.sample_ids)
    selected = []
    skipped = []
    for idx, raw_item in enumerate(raw_items):
        sid = sample_identifier(raw_item, idx)
        category = extract_category(raw_item)
        if requested_ids is not None and sid not in requested_ids and str(raw_item.get("question_id", "")) not in requested_ids:
            continue
        if categories is not None and category not in categories:
            continue
        try:
            if "video" not in raw_item:
                skipped.append(f"{sid}: no video field")
                continue
            selected.append((idx, raw_item))
            if len(selected) >= args.num_samples:
                break
        except Exception as exc:
            skipped.append(f"{sid}: {exc}")
    return selected, skipped


def resolve_video_path(raw_item: dict[str, Any], video_folder: str) -> str:
    video = str(raw_item["video"])
    if os.path.isabs(video):
        return video
    return os.path.join(video_folder, video)


def sample_frame_indices(video_path: str, args: argparse.Namespace) -> list[int]:
    from decord import VideoReader, cpu

    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total = len(vr)
    avg_fps = round(vr.get_avg_fps() / args.video_fps)
    if avg_fps <= 0:
        avg_fps = 1
    indices = list(range(0, total, avg_fps))
    if args.frames_upbound > 0 and (len(indices) > args.frames_upbound or args.force_sample):
        indices = np.linspace(0, total - 1, args.frames_upbound, dtype=int).tolist()
    return indices


def load_video_batch(raw_item: dict[str, Any], args: argparse.Namespace, image_processor: Any) -> dict[str, Any]:
    from decord import VideoReader, cpu

    video_path = resolve_video_path(raw_item, args.video_folder)
    indices = sample_frame_indices(video_path, args)
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    frames = vr.get_batch(indices).asnumpy()
    processed = image_processor.preprocess(images=list(frames), return_tensors="pt")
    return {
        "images": [processed["pixel_values"]],
        "frame_order": indices,
        "video_path": video_path,
    }


def load_cut3r(args: argparse.Namespace, device: torch.device, dtype: torch.dtype):
    from llava.model.multimodal_spatial_encoder.cut3r_spatial_encoder import Cut3rSpatialConfig, Cut3rEncoder

    cfg = Cut3rSpatialConfig(weights_path=args.cut3r_weights)
    encoder = Cut3rEncoder(config=cfg).to(device=device, dtype=dtype).eval()
    return encoder.cut3r


def run_cut3r_decoder_layers(
    cut3r: torch.nn.Module,
    pixel_values: torch.Tensor,
    layer_indices: list[int],
) -> dict[str, torch.Tensor]:
    from llava.model.multimodal_spatial_encoder.cut3r_spatial_encoder import prepare_input

    views = prepare_input(pixel_values=pixel_values)
    shape, feat_ls, pos = cut3r._encode_views(views)
    feat = feat_ls[-1]
    state_feat, state_pos = cut3r._init_state(feat[0], pos[0])
    mem = cut3r.pose_retriever.mem.expand(feat[0].shape[0], -1, -1)
    init_state_feat = state_feat.clone()
    init_mem = mem.clone()
    outputs = {layer: [] for layer in layer_indices}

    for i in range(len(views)):
        feat_i = feat[i].to(pixel_values.dtype)
        pos_i = pos[i]
        global_img_feat_i = None
        if cut3r.pose_head_flag:
            global_img_feat_i = cut3r._get_img_level_feat(feat_i)
            if i == 0:
                pose_feat_i = cut3r.pose_token.expand(feat_i.shape[0], -1, -1)
            else:
                pose_feat_i = cut3r.pose_retriever.inquire(global_img_feat_i, mem)
            pose_pos_i = -torch.ones(feat_i.shape[0], 1, 2, device=feat_i.device, dtype=pos_i.dtype)
        else:
            pose_feat_i = None
            pose_pos_i = None

        new_state_feat, dec = cut3r._recurrent_rollout(
            state_feat,
            state_pos,
            feat_i,
            pos_i,
            pose_feat_i,
            pose_pos_i,
            init_state_feat,
            img_mask=views[i]["img_mask"],
            reset_mask=views[i]["reset"],
            update=views[i].get("update", None),
        )
        for layer in layer_indices:
            outputs[layer].append(dec[layer][:, 1:].detach().float().cpu())

        if global_img_feat_i is not None:
            out_pose_feat_i = dec[-1][:, 0:1]
            new_mem = cut3r.pose_retriever.update_mem(mem, global_img_feat_i, out_pose_feat_i)
        else:
            new_mem = mem

        img_mask = views[i]["img_mask"]
        update = views[i].get("update", None)
        update_mask = (img_mask & update if update is not None else img_mask)[:, None, None].to(pixel_values.dtype)
        state_feat = new_state_feat * update_mask + state_feat * (1 - update_mask)
        mem = new_mem * update_mask + mem * (1 - update_mask)
        reset_mask = views[i]["reset"]
        if reset_mask is not None:
            reset_mask = reset_mask[:, None, None].to(pixel_values.dtype)
            state_feat = init_state_feat * reset_mask + state_feat * (1 - reset_mask)
            mem = init_mem * reset_mask + mem * (1 - reset_mask)

    named = {}
    for layer in layer_indices:
        # [F, B=1, T, D] -> [F, T, D]
        named[f"CUT3R dec {layer}"] = torch.cat(outputs[layer], dim=0)
    return named


def load_pi3(args: argparse.Namespace, device: torch.device, dtype: torch.dtype):
    from llava.model.multimodal_spatial_encoder.pi3x_spatial_encoder import Pi3xSpatialConfig, Pi3xEncoder
    from src.croco.models.curope.curope2d import cuRoPE2D_func

    cfg = Pi3xSpatialConfig(weights_path=args.pi3x_weights, input_size=args.pi3x_input_size)
    encoder = Pi3xEncoder(config=cfg).to(device=device, dtype=dtype).eval()
    if getattr(encoder.pi3, "rope", None) is not None:
        def contiguous_rope_forward(self, tokens, positions):
            rope_tokens = tokens.transpose(1, 2).contiguous()
            cuRoPE2D_func.apply(rope_tokens, positions, self.base, self.F0)
            return rope_tokens.transpose(1, 2).contiguous()

        encoder.pi3.rope.forward = types.MethodType(contiguous_rope_forward, encoder.pi3.rope)
    return encoder.pi3


def run_pi3_decoder_layers(
    pi3: torch.nn.Module,
    pixel_values: torch.Tensor,
    layer_indices: list[int],
    input_size: int,
) -> dict[str, torch.Tensor]:
    from llava.model.multimodal_spatial_encoder.pi3x_spatial_encoder import prepare_input

    imgs = prepare_input(pixel_values, input_size=input_size)
    imgs_norm = (imgs - pi3.image_mean) / pi3.image_std
    bsz, num_frames, channels, height, width = imgs_norm.shape
    imgs_flat = imgs_norm.reshape(bsz * num_frames, channels, height, width)
    hidden = pi3.encoder(imgs_flat, is_training=True)
    if isinstance(hidden, dict):
        hidden = hidden["x_norm_patchtokens"]

    bn, hw, _ = hidden.shape
    batch = bn // num_frames
    hidden = hidden.reshape(batch * num_frames, hw, -1)
    register_token = pi3.register_token.repeat(batch, num_frames, 1, 1).reshape(batch * num_frames, *pi3.register_token.shape[-2:])
    hidden = torch.cat([register_token, hidden], dim=1)
    hw = hidden.shape[1]
    pos = pi3.position_getter(batch * num_frames, height // pi3.patch_size, width // pi3.patch_size, hidden.device)
    pos = pos + 1
    pos_special = torch.zeros(batch * num_frames, pi3.patch_start_idx, 2, device=hidden.device, dtype=pos.dtype)
    pos = torch.cat([pos_special, pos], dim=1)

    resolved = {idx if idx >= 0 else len(pi3.decoder) + idx for idx in layer_indices}
    captures: dict[int, torch.Tensor] = {}
    for i, blk in enumerate(pi3.decoder):
        if i % 2 == 0:
            pos = pos.reshape(batch * num_frames, hw, -1).contiguous()
            hidden = hidden.reshape(batch * num_frames, hw, -1).contiguous()
        else:
            pos = pos.reshape(batch, num_frames * hw, -1).contiguous()
            hidden = hidden.reshape(batch, num_frames * hw, -1).contiguous()
        hidden = blk(hidden, xpos=pos)
        if i in resolved:
            captures[i] = hidden.reshape(batch * num_frames, hw, -1).detach().float().cpu()

    named = {}
    for layer in layer_indices:
        resolved_idx = layer if layer >= 0 else len(pi3.decoder) + layer
        tokens = captures[resolved_idx][:, pi3.patch_start_idx :, :]
        named[f"PI3 dec {layer}"] = tokens
    return named


def frame_features_from_layers(
    layer_features: dict[str, torch.Tensor],
    local_frame_idx: int,
    target_hw: tuple[int, int],
    pool_mode: str,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    out = {}
    meta = {}
    for name, tensor in layer_features.items():
        if tensor.dim() != 3:
            raise RuntimeError(f"{name}: expected [F,T,D], got {tuple(tensor.shape)}")
        tokens = tensor[local_frame_idx].contiguous()
        raw_hw = infer_square_hw(tokens.shape[0], name)
        out[name] = pool_grid(tokens, raw_hw, target_hw, pool_mode)
        meta[name] = {
            "raw_token_count": int(tokens.shape[0]),
            "raw_grid_shape": list(raw_hw),
            "pooled_token_count": int(out[name].shape[0]),
        }
    return out, meta


def save_figure(
    path: Path,
    rgb: np.ndarray,
    anchor: dict[str, Any],
    raw_maps: dict[str, torch.Tensor],
    normalize_mode: str,
) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is required for figure generation.")
    ncols = 1 + len(raw_maps)
    fig, axes = plt.subplots(1, ncols, figsize=(3.0 * ncols, 3.35), squeeze=False)
    ax = axes[0, 0]
    ax.imshow(rgb)
    x = anchor["image_xy"][0] * (rgb.shape[1] - 1)
    y = anchor["image_xy"][1] * (rgb.shape[0] - 1)
    ax.scatter([x], [y], s=52, c="red", edgecolors="white", linewidths=1.2)
    ax.set_title("RGB", fontsize=9)
    ax.axis("off")
    if normalize_mode == "global_per_figure":
        values = torch.cat([value.float().flatten() for value in raw_maps.values()])
        vmin, vmax = float(values.min()), float(values.max())
    else:
        vmin = vmax = None
    for idx, (name, value) in enumerate(raw_maps.items(), start=1):
        ax = axes[0, idx]
        heat = F.interpolate(value[None, None].float(), size=rgb.shape[:2], mode="bilinear", align_corners=False)[0, 0].numpy()
        ax.imshow(heat, cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(name, fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def analyze_sample(
    args: argparse.Namespace,
    image_processor: Any,
    batch: dict[str, Any],
    raw_item: dict[str, Any],
    sample_idx: int,
    cut3r: torch.nn.Module | None,
    pi3: torch.nn.Module | None,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[list[dict[str, Any]], int]:
    sample_id = sample_identifier(raw_item, sample_idx)
    video_tensor = get_video_tensor(batch)
    frame_order = [int(x) for x in batch.get("frame_order", list(range(video_tensor.shape[0])))]
    selected_frames = select_frames(args, frame_order, raw_item)
    target_hw = parse_grid(args.target_grid)

    pixel_values = video_tensor.to(device=device, dtype=dtype)
    layer_features: dict[str, torch.Tensor] = {}
    with torch.inference_mode():
        if cut3r is not None:
            layer_features.update(run_cut3r_decoder_layers(cut3r, pixel_values, args.decoder_layers))
        if pi3 is not None:
            layer_features.update(run_pi3_decoder_layers(pi3, pixel_values, args.decoder_layers, args.pi3x_input_size))

    rows = []
    figures_dir = Path(args.output_dir) / "figures"
    raw_dir = Path(args.output_dir) / "raw"
    frames_dir = Path(args.output_dir) / "frames"
    figures_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)
    figure_count = 0

    for local_frame_idx in selected_frames:
        frame_id = frame_order[local_frame_idx]
        frame_features, feature_meta = frame_features_from_layers(layer_features, local_frame_idx, target_hw, args.pool_mode)
        anchors = build_anchors(args.anchor_mode, parse_anchor_coords(args.anchor_coords), target_hw, raw_item)
        if not anchors:
            continue
        rgb = tensor_to_uint8_image(video_tensor[local_frame_idx], image_processor)
        save_rgb_frame(frames_dir / f"{safe_id(sample_id)}_frame{frame_id}.png", rgb)
        for anchor in anchors:
            raw_maps = {name: similarity_map(value, anchor["token_index"], target_hw) for name, value in frame_features.items()}
            stem = f"{safe_id(sample_id)}_frame{frame_id}_anchor{anchor['anchor_id']}"
            save_figure(figures_dir / f"{stem}.png", rgb, anchor, raw_maps, args.normalize_mode)
            save_rgb_frame(frames_dir / f"{stem}_rgb_anchor.png", rgb, anchor)
            figure_count += 1

            if args.save_raw:
                torch.save(
                    {
                        "raw_similarity_maps": raw_maps,
                        "pooled_features": {name: value.cpu() for name, value in frame_features.items()} if args.save_features else {},
                    },
                    raw_dir / f"{stem}.pt",
                )

            cut_ref = raw_maps.get("CUT3R dec -1")
            pi3_ref = raw_maps.get("PI3 dec -1")
            for name, sim in raw_maps.items():
                cut_p = cut_s = pi3_p = pi3_s = float("nan")
                if cut_ref is not None:
                    cut_p, cut_s = corr_values(sim, cut_ref)
                if pi3_ref is not None:
                    pi3_p, pi3_s = corr_values(sim, pi3_ref)
                rows.append({
                    "sample_id": sample_id,
                    "category": extract_category(raw_item),
                    "frame_idx": frame_id,
                    "anchor_id": anchor["anchor_id"],
                    "representation": name,
                    "pearson_with_cut3r_dec_m1": cut_p,
                    "spearman_with_cut3r_dec_m1": cut_s,
                    "pearson_with_pi3_dec_m1": pi3_p,
                    "spearman_with_pi3_dec_m1": pi3_s,
                    "mean_similarity": float(sim.float().mean().item()),
                    "std_similarity": float(sim.float().std(unbiased=False).item()),
                })

            metadata = {
                "sample_id": sample_id,
                "sample_index": sample_idx,
                "question": extract_question(raw_item),
                "category": extract_category(raw_item),
                "frame_idx": frame_id,
                "local_frame_idx": local_frame_idx,
                "anchor_coordinate_image_space": anchor["image_xy"],
                "anchor_token_xy": anchor["token_xy"],
                "anchor_token_index": anchor["token_index"],
                "target_grid_shape": list(target_hw),
                "representations": list(raw_maps.keys()),
                "feature_metadata": feature_meta,
                "pool_mode": args.pool_mode,
                "cut3r_weights": args.cut3r_weights if cut3r is not None else None,
                "pi3x_weights": args.pi3x_weights if pi3 is not None else None,
                "number_of_frames": int(video_tensor.shape[0]),
            }
            with open(raw_dir / f"{stem}.json", "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
                f.write("\n")
            print(json.dumps({
                "sample_id": sample_id,
                "frame_idx": frame_id,
                "anchor": anchor["anchor_id"],
                "representations": list(raw_maps.keys()),
                "feature_metadata": feature_meta,
            }, indent=2))

    return rows, figure_count


def write_summary(output_dir: Path, rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    fieldnames = [
        "sample_id",
        "category",
        "frame_idx",
        "anchor_id",
        "representation",
        "pearson_with_cut3r_dec_m1",
        "spearman_with_cut3r_dec_m1",
        "pearson_with_pi3_dec_m1",
        "spearman_with_pi3_dec_m1",
        "mean_similarity",
        "std_similarity",
    ]
    with open(output_dir / "spatial_decoder_layer_roi_summary.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        rep = row["representation"]
        for key in ("spearman_with_cut3r_dec_m1", "spearman_with_pi3_dec_m1"):
            value = float(row[key])
            if not math.isnan(value):
                grouped[rep][key].append(value)
    return {
        rep: {key: float(np.mean(values)) for key, values in metrics.items() if values}
        for rep, metrics in grouped.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--sample_ids", default=None)
    parser.add_argument("--categories", default=None)
    parser.add_argument("--frames", default=None)
    parser.add_argument("--anchor_mode", choices=["manual", "center", "object", "grid"], default="center")
    parser.add_argument("--anchor_coords", default=None)
    parser.add_argument("--normalize_mode", choices=["global_per_figure", "per_map"], default="global_per_figure")
    parser.add_argument("--save_raw", type=str2bool, default=True)
    parser.add_argument("--save_features", action="store_true")
    parser.add_argument("--decoder_layers", type=parse_int_list, default="-3,-2,-1")
    parser.add_argument("--target_grid", default="14x14")
    parser.add_argument("--pool_mode", choices=["bilinear", "average", "max"], default="bilinear")

    parser.add_argument("--model_base", required=True)
    parser.add_argument("--siglip_path", required=True)
    parser.add_argument("--image_folder", default=".")
    parser.add_argument("--video_folder", default=".")
    parser.add_argument("--spatial_feature_dir", default="spatial_features")
    parser.add_argument("--frames_upbound", type=int, default=32)
    parser.add_argument("--video_fps", type=int, default=1)
    parser.add_argument("--image_aspect_ratio", default="anyres_max_9")
    parser.add_argument("--image_grid_pinpoints", default="(1x1),...,(6x6)")
    parser.add_argument("--add_time_instruction", type=str2bool, default=True)
    parser.add_argument("--force_sample", type=str2bool, default=True)

    parser.add_argument("--cut3r_weights", default=str(REPO_ROOT / "third_party/CUT3R/src/cut3r_512_dpt_4_64.pth"))
    parser.add_argument("--pi3x_weights", default="/leonardo_scratch/fast/EUHPC_D32_006/hf_models/Pi3X")
    parser.add_argument("--pi3x_input_size", type=int, default=518)
    parser.add_argument("--encoder", choices=["both", "cut3r", "pi3"], default="both")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if plt is None:
        raise RuntimeError("matplotlib is required for this script.")
    if args.decoder_layers is None:
        args.decoder_layers = [-3, -2, -1]

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)
        f.write("\n")

    tokenizer = AutoTokenizer.from_pretrained(args.model_base, use_fast=False, local_files_only=True)
    tokenizer.model_max_length = 32768
    tokenizer.padding_side = "right"
    image_processor = AutoImageProcessor.from_pretrained(args.siglip_path, local_files_only=True)
    raw_items = load_raw_items(args.data_json)
    samples, skipped = select_samples(raw_items, args)
    if not samples:
        raise RuntimeError("No samples selected. Skips:\n" + "\n".join(skipped[:20]))

    device = torch.device(args.device)
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    cut3r = load_cut3r(args, device, dtype) if args.encoder in {"both", "cut3r"} else None
    pi3 = load_pi3(args, device, dtype) if args.encoder in {"both", "pi3"} else None

    rows: list[dict[str, Any]] = []
    figure_count = 0
    for sample_idx, raw_item in samples:
        try:
            batch = load_video_batch(raw_item, args, image_processor)
            sample_rows, count = analyze_sample(args, image_processor, batch, raw_item, sample_idx, cut3r, pi3, device, dtype)
            rows.extend(sample_rows)
            figure_count += count
        except Exception as exc:
            import traceback
            skipped.append(f"{sample_identifier(raw_item, sample_idx)}: {exc}")
            print(f"[SKIP] {sample_identifier(raw_item, sample_idx)}: {exc}")
            print(traceback.format_exc())
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    averages = write_summary(output_dir, rows)
    with open(output_dir / "skipped_samples.json", "w", encoding="utf-8") as f:
        json.dump(skipped, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print("\n[DONE] Spatial decoder layer ROI")
    print(f"figures_generated={figure_count}")
    print(f"output_dir={output_dir}")
    print(f"skipped_samples={len(skipped)}")
    print("average_spearman:")
    for rep, metrics in sorted(averages.items()):
        rendered = ", ".join(f"{key}={value:.4f}" for key, value in sorted(metrics.items()))
        print(f"  {rep}: {rendered}")


if __name__ == "__main__":
    main()
