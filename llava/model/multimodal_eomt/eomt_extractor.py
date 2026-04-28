"""EoMT extractor module for VLM-3R.

Wraps the EoMT panoptic segmentation model to extract soft masks
from PIL images. This module is frozen (no gradients) and serves
as a side-branch for mask extraction only.
"""

from contextlib import nullcontext
import os
import sys
import warnings
from importlib import import_module
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

EOMT_AVAILABLE = False
EOMT_IMPORT_ERROR = None


def _env_flag(name: str) -> bool:
    value = os.environ.get(name, "")
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def _resolve_cache_dir(explicit_cache_dir: Optional[str]) -> Optional[str]:
    candidates = [
        explicit_cache_dir,
        os.environ.get("EOMT_HF_CACHE_DIR"),
        os.environ.get("HF_HUB_CACHE"),
        os.environ.get("HUGGINGFACE_HUB_CACHE"),
        os.environ.get("HF_HOME"),
    ]
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip() != "":
            return os.path.expanduser(candidate)
    return None


def _resolve_local_backbone_path(explicit_local_backbone_path: Optional[str]) -> Optional[str]:
    candidates = [
        explicit_local_backbone_path,
        os.environ.get("EOMT_LOCAL_BACKBONE_PATH"),
    ]
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip() != "":
            return os.path.expanduser(candidate)
    return None


def _ensure_eomt_importable():
    """Make the vendored EoMT modules importable.

    Prefer the package-qualified import path to avoid collisions with other
    top-level packages named ``models``.
    """
    global EOMT_AVAILABLE, EOMT_IMPORT_ERROR
    eomt_root = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "third_party", "EoMT")
    )
    project_root = os.path.dirname(os.path.dirname(eomt_root))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    try:
        import_module("third_party.EoMT.models.eomt")
        import_module("third_party.EoMT.models.vit")

        EOMT_AVAILABLE = True
        EOMT_IMPORT_ERROR = None
    except ImportError as exc:
        EOMT_AVAILABLE = False
        EOMT_IMPORT_ERROR = exc


def _import_eomt_classes():
    eomt_mod = import_module("third_party.EoMT.models.eomt")
    vit_mod = import_module("third_party.EoMT.models.vit")
    return eomt_mod.EoMT, vit_mod.ViT


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
        self.cache_dir = _resolve_cache_dir(eomt_config.get("cache_dir"))
        self.local_backbone_path = _resolve_local_backbone_path(eomt_config.get("local_backbone_path"))
        local_files_only = eomt_config.get("local_files_only")
        if local_files_only is None:
            local_files_only = _env_flag("HF_HUB_OFFLINE") or _env_flag("TRANSFORMERS_OFFLINE")
        self.local_files_only = bool(local_files_only)

        _ensure_eomt_importable()
        self.is_available = EOMT_AVAILABLE

        if not self.is_available:
            warnings.warn(
                "EoMT is not available. EoMTExtractor will be a no-op. "
                "Make sure third_party/EoMT is present and its dependencies "
                "(timm, transformers) are installed. "
                f"Import error: {EOMT_IMPORT_ERROR!r}"
            )
            self.img_size = eomt_config.get("img_size", (640, 640))
            self.num_q = 0
            self.num_classes = 0
            self._warned_unavailable_forward = False
            return

        import yaml
        _EoMT, _ViT = _import_eomt_classes()

        # Parse config
        with open(self.config_path, "r") as f:
            config = yaml.safe_load(f)

        network_cfg = config["model"]["init_args"]["network"]
        network_init = network_cfg.get("init_args", {})
        encoder_cfg = network_init.get("encoder", {})
        encoder_init = dict(encoder_cfg.get("init_args", {}))

        if self.cache_dir is not None:
            encoder_init.setdefault("cache_dir", self.cache_dir)
        if self.local_backbone_path is not None:
            encoder_init.setdefault("local_backbone_path", self.local_backbone_path)
        encoder_init.setdefault("local_files_only", self.local_files_only)

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
        # Store stuff class IDs so downstream components (e.g. selective_3d_gate)
        # can filter queries by panoptic class type (things vs stuff).
        self.stuff_class_ids: frozenset = frozenset(int(c) for c in stuff_classes)

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
        if self.ckpt_path is None:
            warnings.warn("EoMT checkpoint path is not provided. Extractor will use randomly initialized weights.")
        elif not os.path.isfile(self.ckpt_path):
            raise FileNotFoundError(f"EoMT checkpoint not found: {self.ckpt_path}")
        else:
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
            if not getattr(self, "_warned_unavailable_forward", False):
                warnings.warn("EoMT is not available. Returning empty outputs.")
                self._warned_unavailable_forward = True
            B = len(pil_images)
            H, W = self.img_size
            return {
                "mask_logits": torch.zeros(B, 0, 1, 1),
                "soft_masks": torch.zeros(B, 0, H, W),
                "class_logits": torch.zeros(B, 0, 1),
                "mask_resolution": (H, W),
                "query_count": 0,
                "frame_meta": frame_meta,
                "is_available": False,
                "skip_reason": "eomt_unavailable",
            }

        # Preprocess
        imgs = self.preprocess(pil_images)  # (B, 3, H, W) in [0, 255]

        # Move to model device and run inference
        device = next(self.network.parameters()).device
        imgs = imgs.to(device)

        autocast_context = nullcontext()
        if device.type != "cpu" or self.dtype == torch.bfloat16:
            autocast_context = torch.amp.autocast(device_type=device.type, dtype=self.dtype)

        with autocast_context:
            # EoMT.forward expects input in [0, 1]; it normalizes internally
            x = imgs / 255.0
            mask_logits_per_layer, class_logits_per_layer = self.network(x)

        # Take the last layer predictions
        mask_logits = mask_logits_per_layer[-1]  # (B, num_q, H_grid, W_grid)
        class_logits = class_logits_per_layer[-1]  # (B, num_q, num_classes+1)

        # Compute soft masks at backbone grid resolution.
        # We intentionally do NOT upsample to img_size (e.g. 640×640) here:
        # storing (B, 200, 640, 640) float32 tensors costs ~9.77 GiB which
        # causes OOM during training.  All downstream consumers (MaskGuidedPooler,
        # Selective3DGate) resize to their own target resolution anyway.
        H_grid, W_grid = mask_logits.shape[-2], mask_logits.shape[-1]
        soft_masks = torch.sigmoid(mask_logits.float())  # (B, Q, H_grid, W_grid)

        return {
            "mask_logits": mask_logits,
            "soft_masks": soft_masks,
            "class_logits": class_logits,
            "mask_resolution": (H_grid, W_grid),
            "query_count": self.num_q,
            "frame_meta": frame_meta,
            "is_available": True,
            # Taxonomy: frozenset of model-space class IDs classified as "stuff".
            # Classes NOT in this set are "things" (countable instance categories).
            "stuff_class_ids": self.stuff_class_ids,
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
