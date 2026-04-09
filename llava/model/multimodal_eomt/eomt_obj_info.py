"""Config-driven OBJ_INFO prefix builder for EoMT object blocks."""

from typing import Optional, Tuple

import torch


class EoMTObjInfoBuilder:
    """Build optional prefix tokens for the object block.

    Supported modes:
    - none
    - text_phrase
    - learnable_embedding
    """

    SUPPORTED_MODES = {"none", "text_phrase", "learnable_embedding"}

    def _empty(self, mode: str, reason: Optional[str] = None) -> Tuple[torch.Tensor, dict]:
        return torch.zeros(0, 0), {
            "obj_info_mode": mode,
            "obj_info_used": False,
            "obj_info_reason": reason,
        }

    def _phrase_proxy_embedding(
        self,
        text: str,
        hidden_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        # Deterministic tokenizer-free proxy embedding for text_phrase mode.
        text = text or ""
        score = 0
        for idx, ch in enumerate(text):
            score += (idx + 1) * ord(ch)
        if score == 0:
            score = 1
        base = float(score)

        idxs = torch.arange(hidden_size, device=device, dtype=torch.float32)
        vec = torch.sin(idxs / 17.0 + base / 997.0) + torch.cos(idxs / 29.0 + base / 577.0)
        vec = vec / vec.norm(p=2).clamp_min(1e-6)
        return vec.to(dtype=dtype).unsqueeze(0)

    def build(
        self,
        mode: str,
        hidden_size: int,
        device: torch.device,
        dtype: torch.dtype,
        text_phrase: Optional[str] = None,
        learnable_embedding: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        if mode not in self.SUPPORTED_MODES:
            mode = "none"

        if mode == "none":
            tokens = torch.zeros((0, hidden_size), device=device, dtype=dtype)
            return tokens, {
                "obj_info_mode": mode,
                "obj_info_used": False,
                "obj_info_reason": "obj_info_disabled",
            }

        if mode == "text_phrase":
            tokens = self._phrase_proxy_embedding(
                text=str(text_phrase or ""),
                hidden_size=hidden_size,
                device=device,
                dtype=dtype,
            )
            return tokens, {
                "obj_info_mode": mode,
                "obj_info_used": True,
                "obj_info_reason": None,
                "obj_info_text": str(text_phrase or ""),
                "obj_info_token_count": int(tokens.shape[0]),
            }

        if mode == "learnable_embedding":
            if not torch.is_tensor(learnable_embedding) or learnable_embedding.ndim != 1:
                tokens = torch.zeros((0, hidden_size), device=device, dtype=dtype)
                return tokens, {
                    "obj_info_mode": mode,
                    "obj_info_used": False,
                    "obj_info_reason": "missing_learnable_obj_info_embedding",
                }
            tokens = learnable_embedding.to(device=device, dtype=dtype).unsqueeze(0)
            return tokens, {
                "obj_info_mode": mode,
                "obj_info_used": True,
                "obj_info_reason": None,
                "obj_info_token_count": int(tokens.shape[0]),
            }

        tokens = torch.zeros((0, hidden_size), device=device, dtype=dtype)
        return tokens, {
            "obj_info_mode": "none",
            "obj_info_used": False,
            "obj_info_reason": "unsupported_obj_info_mode",
        }
