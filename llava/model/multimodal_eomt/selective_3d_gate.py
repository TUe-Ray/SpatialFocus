"""Selective 3D gating via EoMT query foreground confidence selection.

This module gates only the 3D spatial features (patch_tokens) using top-k
selection by max foreground class probability from EoMT outputs, leaving 2D
features untouched.

Design:
    1. **select_masks_by_confidence** — threshold + top-k selection
    2. **build_selective_gate** — merge selected masks into a single gating map
    3. **apply_selective_3d_gate** — gate 3D features only

Round-1 intentionally supports only confidence-based selection and patch-only
3D fusion. Future question-aware or externally driven selection can plug into
the same merge -> gate stages later.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Config dataclass                                                            #
# --------------------------------------------------------------------------- #

@dataclass
class Selective3DConfig:
    """Configuration for selective 3D gating, read from model config attrs."""

    enable: bool = False
    selector_mode: str = "confidence_topk"
    score_threshold: float = 0.35
    topk: int = -1  # -1 = disabled: keep all masks whose score >= threshold
    merge_mode: str = "soft_max_union"
    gate_type: str = "soft_with_floor"
    floor: float = 0.1
    empty_fallback: str = "all_3d"

    VALID_SELECTOR_MODES = {"confidence_topk"}
    VALID_MERGE_MODES = {"soft_max_union"}
    VALID_GATE_TYPES = {"soft", "soft_with_floor"}
    VALID_FALLBACKS = {"all_3d", "zero_3d"}

    def __post_init__(self):
        if self.selector_mode not in self.VALID_SELECTOR_MODES:
            raise ValueError(
                f"Selective 3D round-1 only supports selector_mode='confidence_topk', got '{self.selector_mode}'. "
                f"Valid: {sorted(self.VALID_SELECTOR_MODES)}"
            )
        if self.topk != -1 and self.topk < 1:
            raise ValueError(
                f"mm_eomt_selector_topk must be -1 (disabled, keep all above threshold) or >= 1, got {self.topk}."
            )
        if self.merge_mode not in self.VALID_MERGE_MODES:
            raise ValueError(
                f"Invalid merge_mode '{self.merge_mode}'. "
                f"Valid: {sorted(self.VALID_MERGE_MODES)}"
            )
        if self.gate_type not in self.VALID_GATE_TYPES:
            raise ValueError(
                f"Invalid gate_type '{self.gate_type}'. "
                f"Valid: {sorted(self.VALID_GATE_TYPES)}"
            )
        if self.empty_fallback not in self.VALID_FALLBACKS:
            raise ValueError(
                f"Invalid empty_fallback '{self.empty_fallback}'. "
                f"Valid: {sorted(self.VALID_FALLBACKS)}"
            )

    @classmethod
    def from_model_config(cls, config) -> "Selective3DConfig":
        """Build from a model config object (getattr-based)."""
        return cls(
            enable=bool(getattr(config, "mm_eomt_selective_3d_enable", False)),
            selector_mode=str(getattr(config, "mm_eomt_selector_mode", "confidence_topk")),
            score_threshold=float(getattr(config, "mm_eomt_selector_score_threshold", 0.35)),
            topk=int(getattr(config, "mm_eomt_selector_topk", -1)),
            merge_mode=str(getattr(config, "mm_eomt_selective_3d_merge_mode", "soft_max_union")),
            gate_type=str(getattr(config, "mm_eomt_selective_3d_gate_type", "soft_with_floor")),
            floor=float(getattr(config, "mm_eomt_selective_3d_floor", 0.1)),
            empty_fallback=str(getattr(config, "mm_eomt_selective_3d_empty_fallback", "all_3d")),
        )


# --------------------------------------------------------------------------- #
#  Debug info container                                                        #
# --------------------------------------------------------------------------- #

@dataclass
class SelectiveGateDebugInfo:
    """Structured debug/logging payload for a single frame."""

    num_masks_total: int = 0
    num_masks_after_threshold: int = 0
    topk_disabled: bool = False
    num_masks_after_topk: int = 0
    selected_scores: List[float] = field(default_factory=list)
    selected_query_indices: List[int] = field(default_factory=list)
    used_fallback: bool = False
    fallback_mode: str = ""
    fallback_reason: str = ""
    gate_min: float = 0.0
    gate_max: float = 0.0
    gate_mean: float = 0.0

    def log(self, prefix: str = "") -> None:
        tag = f"[Selective3DGate] {prefix}" if prefix else "[Selective3DGate]"
        topk_msg = "disabled(all_above_thresh)" if self.topk_disabled else f"applied({self.num_masks_after_topk})"
        logger.info(
            "%s queries: total=%d, after_thresh=%d, topk=%s, "
            "foreground_scores=%s, fallback=%s(%s), reason=%s, gate_stats=(min=%.4f, max=%.4f, mean=%.4f)",
            tag,
            self.num_masks_total,
            self.num_masks_after_threshold,
            topk_msg,
            [f"{s:.4f}" for s in self.selected_scores],
            self.used_fallback,
            self.fallback_mode,
            self.fallback_reason,
            self.gate_min,
            self.gate_max,
            self.gate_mean,
        )


# --------------------------------------------------------------------------- #
#  A. Selector: threshold → top-k                                             #
# --------------------------------------------------------------------------- #

def select_masks_by_confidence(
    scores: torch.Tensor,
    threshold: float,
    topk: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Select query indices by foreground class confidence threshold, then optionally top-k.

    Args:
        scores: (Q,) per-query foreground class confidence scores for a single frame.
        threshold: Minimum score to keep.
        topk: Maximum number of masks to keep after thresholding.
            Pass -1 to disable top-k and keep all masks whose score >= threshold.

    Returns:
        selected_indices: (K,) long tensor of selected query indices.
        selected_scores: (K,) corresponding scores, sorted descending.
    """
    # 1) Threshold
    above_mask = scores >= threshold
    above_indices = torch.where(above_mask)[0]  # (M,)

    if above_indices.numel() == 0:
        return (
            torch.zeros(0, dtype=torch.long, device=scores.device),
            torch.zeros(0, dtype=scores.dtype, device=scores.device),
        )

    above_scores = scores[above_indices]  # (M,)

    # 2) Top-k (among survivors) — only when topk != -1
    if topk == -1:
        # Disabled: return all above threshold, sorted by score descending.
        sorted_order = torch.argsort(above_scores, descending=True)
        return above_indices[sorted_order], above_scores[sorted_order]

    k = min(topk, above_indices.numel())
    topk_scores, topk_local_idx = torch.topk(above_scores, k, sorted=True)  # (K,)
    selected_indices = above_indices[topk_local_idx]

    return selected_indices, topk_scores


def _build_fallback_debug_info(
    *,
    config: Selective3DConfig,
    num_queries: int,
    reason: str,
) -> SelectiveGateDebugInfo:
    gate_value = 1.0 if config.empty_fallback == "all_3d" else 0.0
    return SelectiveGateDebugInfo(
        num_masks_total=num_queries,
        used_fallback=True,
        fallback_mode=config.empty_fallback,
        fallback_reason=reason,
        gate_min=gate_value,
        gate_max=gate_value,
        gate_mean=gate_value,
    )


def _apply_fallback_to_frame(
    frame_3d: torch.Tensor,
    *,
    config: Selective3DConfig,
    reason: str,
    num_queries: int,
) -> Tuple[torch.Tensor, SelectiveGateDebugInfo]:
    """Apply the configured empty fallback to one frame.

    `zero_3d` is handled here by returning true zeroed 3D features directly,
    bypassing any floor logic from `soft_with_floor`.
    """
    debug = _build_fallback_debug_info(
        config=config,
        num_queries=num_queries,
        reason=reason,
    )

    if config.empty_fallback == "all_3d":
        return frame_3d, debug

    return torch.zeros_like(frame_3d), debug


def _apply_fallback_to_all_frames(
    patch_tokens: torch.Tensor,
    *,
    config: Selective3DConfig,
    reason: str,
    num_queries: int = 0,
) -> Tuple[torch.Tensor, List[SelectiveGateDebugInfo]]:
    gated_frames = []
    debug_infos = []
    for frame_idx in range(patch_tokens.shape[0]):
        gated_frame, debug = _apply_fallback_to_frame(
            patch_tokens[frame_idx:frame_idx + 1],
            config=config,
            reason=reason,
            num_queries=num_queries,
        )
        gated_frames.append(gated_frame)
        debug_infos.append(debug)
    return torch.cat(gated_frames, dim=0), debug_infos


# --------------------------------------------------------------------------- #
#  B. Merge: selected masks → single gating map                               #
# --------------------------------------------------------------------------- #

def build_selective_gate(
    selected_masks: torch.Tensor,
    target_hw: Tuple[int, int],
    merge_mode: str = "soft_max_union",
    empty_fallback: str = "all_3d",
    masks_are_logits: bool = False,
) -> Tuple[torch.Tensor, bool]:
    """Merge selected masks into a single spatial gating map.

    Args:
        selected_masks: (K, H_m, W_m) selected mask tensors.
            If masks_are_logits=True, sigmoid is applied first.
        target_hw: (H_target, W_target) spatial resolution of 3D feature map.
        merge_mode: Merge strategy. Currently only "soft_max_union".
        empty_fallback: "all_3d" → ones, "zero_3d" → zeros when K=0.
        masks_are_logits: If True, apply sigmoid before merging.

    Returns:
        gate: (1, 1, H_target, W_target) gating map in [0, 1].
        used_fallback: True if no masks were selected and fallback gate was built.
    """
    H_t, W_t = target_hw
    device = selected_masks.device if selected_masks.numel() > 0 else torch.device("cpu")
    dtype = selected_masks.dtype if selected_masks.numel() > 0 else torch.float32

    # Empty selection → fallback
    if selected_masks.numel() == 0 or selected_masks.shape[0] == 0:
        if empty_fallback == "all_3d":
            return torch.ones(1, 1, H_t, W_t, device=device, dtype=dtype), True
        else:  # zero_3d
            return torch.zeros(1, 1, H_t, W_t, device=device, dtype=dtype), True

    # Sigmoid if logits
    if masks_are_logits:
        selected_masks = torch.sigmoid(selected_masks.float())

    # Ensure float for interpolation
    selected_masks = selected_masks.float()

    # Resize to target spatial resolution if needed
    K = selected_masks.shape[0]
    H_m, W_m = selected_masks.shape[-2], selected_masks.shape[-1]
    if (H_m, W_m) != (H_t, W_t):
        # F.interpolate expects (N, C, H, W)
        selected_masks = F.interpolate(
            selected_masks.unsqueeze(1),  # (K, 1, H_m, W_m)
            size=(H_t, W_t),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)  # (K, H_t, W_t)

    # Merge: soft_max_union → pixelwise max
    if merge_mode == "soft_max_union":
        gate = selected_masks.max(dim=0).values  # (H_t, W_t)
    else:
        raise ValueError(f"Unsupported merge_mode: {merge_mode}")

    # Clamp to [0, 1] for safety
    gate = gate.clamp(0.0, 1.0)

    return gate.unsqueeze(0).unsqueeze(0), False  # (1, 1, H_t, W_t)


# --------------------------------------------------------------------------- #
#  C. 3D gate application                                                      #
# --------------------------------------------------------------------------- #

def apply_selective_3d_gate(
    feats_3d: torch.Tensor,
    gate: torch.Tensor,
    gate_type: str = "soft",
    floor: float = 0.0,
) -> torch.Tensor:
    """Apply selective gate to 3D spatial features only.

    Args:
        feats_3d: (B, T, D) 3D patch tokens where T = H*W spatial tokens.
        gate: (1, 1, H, W) gating map from build_selective_gate.
        gate_type: "soft" or "soft_with_floor".
        floor: Minimum non-fallback gate value for "soft_with_floor".

    Returns:
        gated_feats_3d: (B, T, D) gated 3D features.
    """
    B, T, D = feats_3d.shape
    H, W = gate.shape[-2], gate.shape[-1]

    # Flatten gate to (1, H*W, 1) to match (B, T, D)
    gate_flat = gate.reshape(1, H * W, 1)  # (1, H*W, 1)

    # If T != H*W, the gate was built at a different resolution.
    # This shouldn't happen after build_selective_gate with correct target_hw,
    # but handle gracefully.
    if gate_flat.shape[1] != T:
        gate_flat = F.interpolate(
            gate.float(),  # (1, 1, H, W)
            size=(int(math.isqrt(T)), int(math.isqrt(T))),
            mode="bilinear",
            align_corners=False,
        ).reshape(1, T, 1)

    gate_flat = gate_flat.to(dtype=feats_3d.dtype, device=feats_3d.device)

    if gate_type == "soft":
        return gate_flat * feats_3d
    elif gate_type == "soft_with_floor":
        effective_gate = floor + (1.0 - floor) * gate_flat
        return effective_gate * feats_3d
    else:
        raise ValueError(f"Unsupported gate_type: {gate_type}")


# --------------------------------------------------------------------------- #
#  Orchestrator: full selective 3D gating pipeline                             #
# --------------------------------------------------------------------------- #

def apply_selective_3d_fusion(
    patch_tokens: torch.Tensor,
    eomt_outputs: Optional[Dict],
    config: Selective3DConfig,
    frame_indices: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, List[SelectiveGateDebugInfo]]:
    """Full selective 3D gating pipeline.

    Orchestrates: select → merge → gate for each frame in the batch.

    Args:
        patch_tokens: (F, T, D) 3D spatial features where F = num_frames.
        eomt_outputs: Dict from EoMT extractor with keys:
            - "soft_masks": (F_eomt, Q, H_m, W_m) — soft masks (already sigmoided)
            - "class_logits": (F_eomt, Q, C+1) — per-query class logits
            - "mask_logits": (F_eomt, Q, H_grid, W_grid) — raw logits (optional)
        config: Selective3DConfig instance.
        frame_indices: Optional mapping from patch_tokens frames to eomt_outputs
            frames. If None, assumes 1:1 correspondence.

    Returns:
        gated_patch_tokens: (F, T, D) gated 3D features.
        debug_infos: List of SelectiveGateDebugInfo per frame.
    """
    if not config.enable:
        return patch_tokens, []

    if eomt_outputs is None:
        logger.warning(
            "[Selective3DGate] EoMT outputs unavailable; applying configured fallback '%s'.",
            config.empty_fallback,
        )
        return _apply_fallback_to_all_frames(
            patch_tokens,
            config=config,
            reason="missing_eomt_outputs",
        )

    F_batch, T, D = patch_tokens.shape
    patch_side = int(math.isqrt(T))
    target_hw = (patch_side, patch_side)

    # Extract EoMT data
    soft_masks = eomt_outputs.get("soft_masks")  # (F_eomt, Q, H_m, W_m)
    class_logits = eomt_outputs.get("class_logits")  # (F_eomt, Q, C+1)

    if soft_masks is None or class_logits is None:
        logger.warning(
            "[Selective3DGate] Missing soft_masks or class_logits; applying configured fallback '%s'.",
            config.empty_fallback,
        )
        return _apply_fallback_to_all_frames(
            patch_tokens,
            config=config,
            reason="missing_required_eomt_keys",
        )

    if soft_masks.ndim != 4 or class_logits.ndim != 3:
        logger.warning(
            "[Selective3DGate] Malformed EoMT output shapes soft_masks=%s class_logits=%s; applying configured fallback '%s'.",
            tuple(soft_masks.shape),
            tuple(class_logits.shape),
            config.empty_fallback,
        )
        return _apply_fallback_to_all_frames(
            patch_tokens,
            config=config,
            reason="malformed_eomt_shapes",
        )

    if soft_masks.shape[0] != class_logits.shape[0] or soft_masks.shape[1] != class_logits.shape[1]:
        logger.warning(
            "[Selective3DGate] Incompatible EoMT output shapes soft_masks=%s class_logits=%s; applying configured fallback '%s'.",
            tuple(soft_masks.shape),
            tuple(class_logits.shape),
            config.empty_fallback,
        )
        return _apply_fallback_to_all_frames(
            patch_tokens,
            config=config,
            reason="incompatible_eomt_shapes",
        )

    F_eomt = soft_masks.shape[0]
    Q = soft_masks.shape[1]

    if class_logits.shape[-1] <= 1:
        logger.warning(
            "[Selective3DGate] class_logits has no foreground classes; applying configured fallback '%s'.",
            config.empty_fallback,
        )
        return _apply_fallback_to_all_frames(
            patch_tokens,
            config=config,
            reason="no_foreground_classes",
            num_queries=Q,
        )

    # Compute per-query foreground class confidence: max foreground class probability.
    # class_logits shape: (F_eomt, Q, C+1), last class is "no object"
    fg_probs = torch.softmax(class_logits.float(), dim=-1)[:, :, :-1]  # (F_eomt, Q, C)
    per_query_scores = fg_probs.max(dim=-1).values  # (F_eomt, Q)

    # Gate each frame
    gated_frames = []
    debug_infos = []

    for f_idx in range(F_batch):
        # Map to EoMT frame index
        eomt_f_idx = f_idx if frame_indices is None else int(frame_indices[f_idx].item())

        if eomt_f_idx < 0 or eomt_f_idx >= F_eomt:
            logger.warning(
                "[Selective3DGate] Frame %d maps to EoMT frame %d but only %d EoMT frames are available; applying configured fallback '%s'.",
                f_idx, eomt_f_idx, F_eomt,
                config.empty_fallback,
            )
            gated_frame, debug = _apply_fallback_to_frame(
                patch_tokens[f_idx:f_idx + 1],
                config=config,
                reason="eomt_frame_mismatch",
                num_queries=Q,
            )
            gated_frames.append(gated_frame)
            debug_infos.append(debug)
            continue

        frame_scores = per_query_scores[eomt_f_idx]  # (Q,)
        frame_soft_masks = soft_masks[eomt_f_idx]  # (Q, H_m, W_m)

        debug = SelectiveGateDebugInfo(num_masks_total=Q)

        if config.selector_mode != "confidence_topk":
            raise ValueError(
                "Selective 3D round-1 only supports selector_mode='confidence_topk'. "
                f"Got '{config.selector_mode}'."
            )

        # A. Select using max foreground class probability.
        sel_indices, sel_scores = select_masks_by_confidence(
            frame_scores, config.score_threshold, config.topk,
        )

        debug.num_masks_after_threshold = int(
            (frame_scores >= config.score_threshold).sum().item()
        )
        debug.topk_disabled = (config.topk == -1)
        debug.num_masks_after_topk = sel_indices.numel()
        debug.selected_scores = sel_scores.tolist()
        debug.selected_query_indices = sel_indices.tolist()

        if sel_indices.numel() == 0:
            gated_frame, fallback_debug = _apply_fallback_to_frame(
                patch_tokens[f_idx:f_idx + 1],
                config=config,
                reason="no_queries_after_threshold_and_topk",
                num_queries=Q,
            )
            fallback_debug.num_masks_after_threshold = debug.num_masks_after_threshold
            fallback_debug.topk_disabled = debug.topk_disabled
            fallback_debug.num_masks_after_topk = 0
            fallback_debug.selected_scores = debug.selected_scores
            fallback_debug.selected_query_indices = []
            gated_frames.append(gated_frame)
            debug_infos.append(fallback_debug)
            continue

        selected = frame_soft_masks[sel_indices]  # (K, H_m, W_m)

        # B. Merge
        gate, used_fb = build_selective_gate(
            selected,
            target_hw=target_hw,
            merge_mode=config.merge_mode,
            empty_fallback=config.empty_fallback,
            masks_are_logits=False,  # soft_masks are already sigmoided
        )
        debug.used_fallback = used_fb
        if used_fb:
            debug.fallback_mode = config.empty_fallback
            debug.fallback_reason = "empty_selection_gate"

        # Gate stats
        gate_float = gate.float()
        debug.gate_min = float(gate_float.min().item())
        debug.gate_max = float(gate_float.max().item())
        debug.gate_mean = float(gate_float.mean().item())

        # C. Apply gate to 3D features for this frame
        frame_3d = patch_tokens[f_idx:f_idx + 1]  # (1, T, D)

        # Explicit zero_3d fallback guard: when the merge stage used the fallback
        # (used_fb=True) and the configured fallback is "zero_3d", return true
        # zeroed 3D features directly.  This bypasses apply_selective_3d_gate
        # entirely so that soft_with_floor cannot add a non-zero floor to what
        # must be all-zero 3D output.
        if used_fb and config.empty_fallback == "zero_3d":
            gated_frames.append(torch.zeros_like(frame_3d))
            debug_infos.append(debug)
            continue

        gated_frame = apply_selective_3d_gate(
            frame_3d, gate,
            gate_type=config.gate_type,
            floor=config.floor,
        )
        gated_frames.append(gated_frame)
        debug_infos.append(debug)

    # Log debug info
    for i, dbg in enumerate(debug_infos):
        dbg.log(prefix=f"frame={i}")

    gated_patch_tokens = torch.cat(gated_frames, dim=0)  # (F, T, D)
    return gated_patch_tokens, debug_infos
