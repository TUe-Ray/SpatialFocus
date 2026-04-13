"""Minimal Phase-1 appender for EoMT object token blocks."""

from typing import Optional, Tuple

import torch


class EoMTObjectBlockAppender:
    """Append or prepend selected object tokens around visual tokens."""

    SUPPORTED_POSITIONS = {"before_visual", "after_visual"}

    def append(
        self,
        visual_tokens: torch.Tensor,
        object_tokens: Optional[torch.Tensor],
        position: str,
    ) -> Tuple[torch.Tensor, bool, Optional[str]]:
        if not torch.is_tensor(visual_tokens) or visual_tokens.ndim != 2:
            return visual_tokens, False, "invalid_visual_tokens"

        if not torch.is_tensor(object_tokens) or object_tokens.ndim != 2 or object_tokens.shape[0] == 0:
            return visual_tokens, False, "empty_object_tokens"

        if position not in self.SUPPORTED_POSITIONS:
            position = "after_visual"

        try:
            object_tokens = object_tokens.to(device=visual_tokens.device, dtype=visual_tokens.dtype)
            if object_tokens.shape[1] != visual_tokens.shape[1]:
                return visual_tokens, False, "feature_dim_mismatch"

            if position == "before_visual":
                merged = torch.cat((object_tokens, visual_tokens), dim=0)
            else:
                merged = torch.cat((visual_tokens, object_tokens), dim=0)
            return merged, True, None
        except Exception as exc:
            return visual_tokens, False, f"append_exception:{type(exc).__name__}"
