"""EoMT extractor module for VLM-3R.

Wraps the EoMT panoptic segmentation model to extract soft masks
from PIL images. This module is frozen (no gradients) and serves
as a side-branch for mask extraction only.
"""

import os
import sys
import warnings
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

EOMT_AVAILABLE = False


def _ensure_eomt_importable():
    """Add third_party/EoMT to sys.path so its modules can be imported."""
    global EOMT_AVAILABLE
    eomt_root = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "third_party", "EoMT")
    )
    if eomt_root not in sys.path:
        sys.path.insert(0, eomt_root)
    try:
        from models.eomt import EoMT as _EoMT  # noqa: F401
        from models.vit import ViT as _ViT  # noqa: F401

        EOMT_AVAILABLE = True
    except ImportError:
        EOMT_AVAILABLE = False


class EoMTExtractor(nn.Module):
    """Wrapper for EoMT model that extracts soft masks from PIL images.

    This module is frozen (no gradients) and serves as a side-branch
    for mask extraction only. It does NOT participate in training.
    """

    def __init__(self, eomt_config: dict):
        super().__init__()

        self.ckpt_path = eomt_config.get("ckpt_path")
        self.config_path = eomt_config.get("config_path")
        self.device_str = eomt_config.get("device", "cuda")
        self.dtype = eomt_config.get("dtype", torch.float16)

        _ensure_eomt_importable()
        self.is_available = EOMT_AVAILABLE

        if not self.is_available:
            warnings.warn(
                "EoMT is not available. EoMTExtractor will be a no-op. "
                "Make sure third_party/EoMT is present and its dependencies "
                "(timm, transformers) are installed."
            )
            self.img_size = eomt_config.get("img_size", (640, 640))
            self.num_q = 0
            self.num_classes = 0
            return

        import yaml
        from models.eomt import EoMT as _EoMT
        from models.vit import ViT as _ViT

        # Parse config
        with open(self.config_path, "r") as f:
            config = yaml.safe_load(f)

        network_cfg = config["model"]["init_args"]["network"]
        network_init = network_cfg.get("init_args", {})
        encoder_cfg = network_init.get("encoder", {})
        encoder_init = encoder_cfg.get("init_args", {})

        self.num_q = network_init.get("num_q", 200)
        num_blocks = network_init.get("num_blocks", 4)

        # Determine num_classes from data config (COCO panoptic = 133, etc.)
        data_init = config.get("data", {}).get("init_args", {})
        stuff_classes = data_init.get("stuff_classes", [])
        # num_classes for COCO panoptic: 80 thing + 53 stuff = 133
        # The stuff_classes list contains class IDs; num_classes = max(stuff_classes)+1
        # if stuff_classes is available, otherwise fall back to eomt_config
        if stuff_classes:
            self.num_classes = max(stuff_classes) + 1
        else:
            self.num_classes = eomt_config.get("num_classes", 133)

        # Determine img_size
        if "img_size" in eomt_config and eomt_config["img_size"] is not None:
            self.img_size = tuple(eomt_config["img_size"])
        else:
            # Default from common EoMT configs
            self.img_size = (640, 640)

        # Build encoder
        encoder = _ViT(
            img_size=self.img_size,
            **encoder_init,
        )

        # Build EoMT network (masked_attn disabled for inference)
        self.network = _EoMT(
            encoder=encoder,
            num_classes=self.num_classes,
            num_q=self.num_q,
            num_blocks=num_blocks,
            masked_attn_enabled=False,
        )

        # Load checkpoint weights
        if self.ckpt_path is not None and os.path.isfile(self.ckpt_path):
            state_dict = torch.load(
                self.ckpt_path, map_location="cpu", weights_only=True
            )
            # Handle Lightning-wrapped state dicts (keys prefixed with "network.")
            cleaned = {}
            for k, v in state_dict.items():
                if k.startswith("network."):
                    cleaned[k[len("network."):]] = v
                else:
                    cleaned[k] = v
            self.network.load_state_dict(cleaned, strict=False)

        # Freeze everything
        self.network.eval()
        for param in self.network.parameters():
            param.requires_grad = False

    def train(self, mode: bool = True):
        """Override train to keep the network always in eval mode."""
        super().train(mode)
        if self.is_available:
            self.network.eval()
        return self

    def preprocess(self, pil_images: list) -> torch.Tensor:
        """Preprocess a list of PIL RGB images into a batched tensor.

        Args:
            pil_images: List of PIL.Image.Image in RGB mode.

        Returns:
            Tensor of shape (B, 3, H, W) in [0, 255] range, float32.
        """
        H, W = self.img_size
        tensors = []
        for img in pil_images:
            if img.mode != "RGB":
                img = img.convert("RGB")
            img_resized = img.resize((W, H), Image.BILINEAR)
            # Convert to tensor: [0, 255] float32
            t = torch.from_numpy(
                np.array(img_resized, dtype=np.float32)
            )
            # (H, W, 3) -> (3, H, W)
            t = t.permute(2, 0, 1)
            tensors.append(t)
        return torch.stack(tensors, dim=0)

    @torch.no_grad()
    def forward(
        self,
        pil_images: list,
        frame_meta: Optional[list] = None,
    ) -> dict:
        """Run EoMT inference on a list of PIL images.

        Args:
            pil_images: List of PIL.Image.Image in RGB mode.
            frame_meta: Optional list of dicts with per-frame metadata
                (passed through to the output).

        Returns:
            Dict with keys:
                - mask_logits: (B, num_q, H_grid, W_grid) raw logits
                - soft_masks: (B, num_q, H_img, W_img) sigmoid of
                  interpolated logits
                - class_logits: (B, num_q, num_classes+1)
                - mask_resolution: (H_img, W_img)
                - query_count: int
                - frame_meta: pass-through
        """
        if not self.is_available:
            warnings.warn("EoMT is not available. Returning empty outputs.")
            B = len(pil_images)
            H, W = self.img_size
            return {
                "mask_logits": torch.zeros(B, 0, 1, 1),
                "soft_masks": torch.zeros(B, 0, H, W),
                "class_logits": torch.zeros(B, 0, 1),
                "mask_resolution": (H, W),
                "query_count": 0,
                "frame_meta": frame_meta,
            }

        # Preprocess
        imgs = self.preprocess(pil_images)  # (B, 3, H, W) in [0, 255]

        # Move to model device and run inference
        device = next(self.network.parameters()).device
        imgs = imgs.to(device)

        with torch.amp.autocast(device_type=device.type, dtype=self.dtype):
            # EoMT.forward expects input in [0, 1]; it normalizes internally
            x = imgs / 255.0
            mask_logits_per_layer, class_logits_per_layer = self.network(x)

        # Take the last layer predictions
        mask_logits = mask_logits_per_layer[-1]  # (B, num_q, H_grid, W_grid)
        class_logits = class_logits_per_layer[-1]  # (B, num_q, num_classes+1)

        # Compute soft masks: interpolate logits to img_size, then sigmoid
        H_img, W_img = self.img_size
        mask_logits_upsampled = F.interpolate(
            mask_logits.float(), size=(H_img, W_img), mode="bilinear", align_corners=False
        )
        soft_masks = torch.sigmoid(mask_logits_upsampled)

        return {
            "mask_logits": mask_logits,
            "soft_masks": soft_masks,
            "class_logits": class_logits,
            "mask_resolution": (H_img, W_img),
            "query_count": self.num_q,
            "frame_meta": frame_meta,
        }

    @torch.no_grad()
    def extract_topk_masks(self, outputs: dict, k: int = 5) -> dict:
        """Select top-k queries per image based on max class confidence.

        Args:
            outputs: Dict returned by forward().
            k: Number of top queries to select.

        Returns:
            Dict with keys:
                - topk_soft_masks: (B, k, H, W)
                - topk_class_ids: (B, k)
                - topk_scores: (B, k)
        """
        class_logits = outputs["class_logits"]  # (B, num_q, num_classes+1)
        soft_masks = outputs["soft_masks"]  # (B, num_q, H, W)

        # Exclude the last class ("no object") for scoring
        class_probs = torch.softmax(class_logits.float(), dim=-1)
        # Max class probability excluding "no object" (last index)
        valid_probs = class_probs[:, :, :-1]  # (B, num_q, num_classes)
        max_scores, max_class_ids = valid_probs.max(dim=-1)  # (B, num_q) each

        # Clamp k to available queries
        actual_k = min(k, max_scores.shape[1])

        # Top-k per image
        topk_scores, topk_indices = max_scores.topk(actual_k, dim=1)  # (B, k)
        topk_class_ids = torch.gather(max_class_ids, 1, topk_indices)  # (B, k)

        # Gather corresponding masks
        B = soft_masks.shape[0]
        H, W = soft_masks.shape[2], soft_masks.shape[3]
        # Expand indices for gathering: (B, k) -> (B, k, H, W)
        idx_expanded = topk_indices.unsqueeze(-1).unsqueeze(-1).expand(B, actual_k, H, W)
        topk_soft_masks = torch.gather(soft_masks, 1, idx_expanded)  # (B, k, H, W)

        return {
            "topk_soft_masks": topk_soft_masks,
            "topk_class_ids": topk_class_ids,
            "topk_scores": topk_scores,
        }
