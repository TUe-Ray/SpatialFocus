"""Debug script to verify EoMT frame alignment with VLM-3R training pipeline.

Loads training samples from LazySupervisedDataset and uses the eomt_images /
eomt_meta fields that the dataset already populated.  EoMT is run on exactly
those same raw PIL frames — no independent reloading or resampling.

Usage:
    python scripts/debug_eomt_alignment.py \
        --data_path scripts/VLM_3R/vsibench_data.yaml \
        --video_folder /path/to/data \
        --eomt_config_path third_party/EoMT/configs/dinov2/coco/panoptic/eomt_large_640.yaml \
        --eomt_ckpt_path /path/to/eomt_weights.bin \
        --num_samples 5 \
        --output_dir debug_eomt
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so llava imports work
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Minimal DataArguments stub for LazySupervisedDataset
# ---------------------------------------------------------------------------
@dataclass
class MinimalDataArgs:
    data_path: str = ""
    lazy_preprocess: bool = True
    is_multimodal: bool = True
    early_mix_text: bool = False
    image_folder: Optional[str] = None
    image_aspect_ratio: str = "square"
    image_grid_pinpoints: Optional[str] = None
    image_crop_resolution: Optional[int] = None
    image_split_resolution: Optional[int] = None
    video_folder: Optional[str] = None
    video_fps: int = 1
    frames_upbound: int = 32
    add_time_instruction: bool = False
    force_sample: bool = True
    train_data_percentage: float = 100.0
    train_data_percentage_seed: int = 42
    train_data_shuffle: bool = False
    zero_spatial_features: bool = False
    spatial_tower_type: Optional[str] = None
    spatial_features_subdir: str = "spatial_features"
    image_processor: object = field(default=None, repr=False)
    mm_use_im_start_end: bool = False


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------
def overlay_mask_on_image(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.5,
    cmap_name: str = "jet",
) -> np.ndarray:
    cmap = plt.get_cmap(cmap_name)
    heatmap = (cmap(mask)[..., :3] * 255).astype(np.uint8)
    blended = alpha * image.astype(np.float32) + (1 - alpha) * heatmap.astype(np.float32)
    return np.clip(blended, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Debug EoMT frame alignment with VLM-3R training pipeline."
    )
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--video_folder", type=str, default=None)
    parser.add_argument("--image_folder", type=str, default=None)
    parser.add_argument("--eomt_config_path", type=str, required=True)
    parser.add_argument("--eomt_ckpt_path", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="debug_eomt")
    parser.add_argument("--top_k_masks", type=int, default=5)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Build EoMT extractor (reuse the project wrapper)
    # ------------------------------------------------------------------
    from llava.model.multimodal_eomt import EoMTExtractor

    eomt_cfg = {
        "config_path": args.eomt_config_path,
        "ckpt_path": args.eomt_ckpt_path,
        "device": args.device,
    }
    print(f"[EoMT] Loading from {args.eomt_ckpt_path} ...")
    eomt = EoMTExtractor(eomt_cfg)
    eomt.to(device)
    eomt.eval()

    # ------------------------------------------------------------------
    # 2. Build dataset
    # ------------------------------------------------------------------
    image_folder = args.image_folder or args.video_folder

    try:
        from transformers import SiglipImageProcessor
        processor = SiglipImageProcessor.from_pretrained("google/siglip-so400m-patch14-384")
    except Exception:
        try:
            from transformers import CLIPImageProcessor
            processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
        except Exception:
            print("WARNING: Could not load image processor. Dataset loading may fail.")
            processor = None

    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
    except Exception:
        print("WARNING: Could not load tokenizer.")
        tokenizer = None

    data_args = MinimalDataArgs(
        data_path=args.data_path,
        video_folder=args.video_folder,
        image_folder=image_folder,
        image_processor=processor,
    )

    from llava.train.train import LazySupervisedDataset

    print(f"[Dataset] Loading from {args.data_path} ...")
    dataset = LazySupervisedDataset(
        data_path=args.data_path,
        tokenizer=tokenizer,
        data_args=data_args,
    )
    print(f"[Dataset] Total samples: {len(dataset)}")

    num_to_process = min(args.num_samples, len(dataset))

    # ------------------------------------------------------------------
    # 3. Iterate over samples
    # ------------------------------------------------------------------
    for sample_idx in range(num_to_process):
        print(f"\n{'='*60}")
        print(f"Processing sample {sample_idx} / {num_to_process}")

        # Get the sample — this populates eomt_images and eomt_meta
        try:
            sample = dataset[sample_idx]
        except Exception as e:
            print(f"  WARNING: dataset[{sample_idx}] failed: {e}")
            continue

        if "eomt_images" not in sample or not sample["eomt_images"]:
            print(f"  Sample {sample_idx} has no eomt_images, skipping.")
            continue

        pil_frames = sample["eomt_images"]   # list of PIL.Image
        frame_metas = sample["eomt_meta"]    # list of per-frame dicts

        sample_dir = os.path.join(args.output_dir, f"sample_{sample_idx}")
        os.makedirs(sample_dir, exist_ok=True)

        print(f"  Frames: {len(pil_frames)}")
        print(f"  Modality: {frame_metas[0].get('modality', '?')}")
        print(f"  Source: {frame_metas[0].get('source_path', '?')}")

        # ---- Run EoMT on the exact frames the dataset produced ----
        try:
            with torch.no_grad():
                eomt_outputs = eomt(pil_frames, frame_metas)
        except Exception as e:
            print(f"  WARNING: EoMT inference failed: {e}")
            continue

        soft_masks = eomt_outputs["soft_masks"]    # (B, num_q, H, W)
        class_logits = eomt_outputs["class_logits"]  # (B, num_q, num_classes+1)

        # ---- Visualise per frame ----
        for fidx, (pil_img, fmeta) in enumerate(zip(pil_frames, frame_metas)):
            frame_index = fmeta.get("frame_index", fidx)

            # Save original frame
            orig_path = os.path.join(sample_dir, f"original_frame_{fidx}_vidx{frame_index}.png")
            pil_img.save(orig_path)
            print(f"  Saved {orig_path}")

            # Top-k mask overlays
            mask_batch = soft_masks[fidx]        # (num_q, H, W)
            cls_batch = class_logits[fidx]       # (num_q, num_classes+1)

            class_probs = torch.softmax(cls_batch.float(), dim=-1)
            max_scores, max_class_ids = class_probs[:, :-1].max(dim=-1)
            topk_indices = torch.argsort(max_scores, descending=True)[: args.top_k_masks]

            orig_np = np.array(pil_img)
            orig_h, orig_w = orig_np.shape[:2]

            for rank, q_idx in enumerate(topk_indices):
                q_idx_int = q_idx.item()
                score = max_scores[q_idx_int].item()
                cls_id = max_class_ids[q_idx_int].item()
                mask = mask_batch[q_idx_int]  # (H, W) already at EoMT img_size

                import torch.nn.functional as F
                mask_resized = F.interpolate(
                    mask.unsqueeze(0).unsqueeze(0),
                    size=(orig_h, orig_w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze().cpu().float().numpy()

                overlay = overlay_mask_on_image(orig_np, mask_resized, alpha=0.5)

                fig, ax = plt.subplots(1, 1, figsize=(8, 6))
                ax.imshow(overlay)
                ax.set_title(
                    f"Frame {fidx} (video_idx={frame_index}) | Query {q_idx_int} (rank {rank}) | "
                    f"class {cls_id} | score {score:.3f}",
                    fontsize=9,
                )
                ax.axis("off")

                save_path = os.path.join(
                    sample_dir,
                    f"soft_mask_frame{fidx}_vidx{frame_index}_query{q_idx_int}_rank{rank}.png",
                )
                fig.savefig(save_path, bbox_inches="tight", dpi=120)
                plt.close(fig)
                print(f"  Saved {save_path}")

        # ---- Save metadata ----
        meta_out = {
            "sample_idx": sample_idx,
            "num_frames": len(pil_frames),
            "eomt_query_count": eomt_outputs["query_count"],
            "mask_resolution": list(eomt_outputs["mask_resolution"]),
            "frame_metas": [
                {k: list(v) if isinstance(v, tuple) else v for k, v in fm.items()}
                for fm in frame_metas
            ],
        }
        meta_path = os.path.join(sample_dir, "meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta_out, f, indent=2)
        print(f"  Saved {meta_path}")

    print(f"\nDone. Debug outputs saved to {os.path.abspath(args.output_dir)}/")


if __name__ == "__main__":
    main()
