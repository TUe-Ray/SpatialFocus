"""
Pi3XDecodedFeatures — accessor for pre-extracted Pi3X decoded feature .pt files
produced by scripts/extract_spatial_features_pi3x_decoded_schema.py.

Stored .pt schema
-----------------
{
  "frames": {
      "decoded_features": Tensor[F, T, C],   # main decoder output (C = 2048)
      "frame_idx":        Tensor[F],
  },
  "meta": {
      "decoded_pos_template": Tensor[T, 2],  # positional encoding, shared across frames
      "num_frames":       int,
      "input_size":       int,
      "patch_size":       int,
      "patch_start_idx":  int,               # = num_register_tokens = 5
  },
}

Token layout within T
---------------------
  [0 : patch_start_idx]   register / special tokens  (dim C = 2048)
  [patch_start_idx : ]    spatial patch tokens        (dim C = 2048)

Camera tokens (semantically correct)
-------------------------------------
Pi3 does NOT use a single register token as its camera representation.
The camera branch is a dedicated TransformerDecoder head:

    camera_hidden = pi3.camera_decoder(decoded_features, xpos=pos)  # (F, T, D)

where D = 512 (camera_decoder out_dim).  The register-token slice of this output:

    camera_tokens = camera_hidden[:, patch_start_idx:, :]             # (F, num_patches, D)

is the correct "camera token" representation — matching Pi3's camera_head which also
receives camera_hidden[:, patch_start_idx:] as input to produce camera poses.

Because the camera_decoder is a lightweight head (5 blocks, no image encoding),
it is run at training time from the stored decoded_features via:

    sf.compute_camera_tokens(pi3.camera_decoder, device=..., dtype=...)

This must be called before accessing sf.camera_tokens.

Legacy schema compatibility
---------------------------
The class also accepts the old flat schema:
  { "camera_tokens": Tensor[F,1,C], "patch_tokens": Tensor[F,P,C] }
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional


class Pi3XDecodedFeatures:
    """
    Accessor around a loaded Pi3X decoded-feature dict.

    Construct via:
      - ``Pi3XDecodedFeatures(data_dict)``       (dict already in memory)
      - ``Pi3XDecodedFeatures.from_file(path)``  (loads from .pt file)
      - ``Pi3XDecodedFeatures.from_loaded(data)`` (dict or already-wrapped instance)
    """

    _NEW_SCHEMA_KEY = "frames"
    _LEGACY_CAM_KEY = "camera_tokens"

    # ------------------------------------------------------------------ #
    #  Construction                                                        #
    # ------------------------------------------------------------------ #

    def __init__(self, data: dict) -> None:
        if self._NEW_SCHEMA_KEY in data:
            self._mode = "new"
            self._frames = data["frames"]
            self._meta   = data["meta"]
            # Cache for camera_decoder output, populated by compute_camera_tokens()
            self._camera_decoder_cache: Optional[torch.Tensor] = None
        elif self._LEGACY_CAM_KEY in data:
            self._mode = "legacy"
            self._legacy = data
        else:
            raise ValueError(
                "Unrecognised Pi3X feature dict. Expected 'frames' key (new schema) "
                f"or 'camera_tokens' key (legacy). Got: {list(data.keys())}"
            )

    @classmethod
    def from_file(cls, path: str, map_location: str = "cpu") -> "Pi3XDecodedFeatures":
        """Load a .pt file and return a Pi3XDecodedFeatures instance."""
        data = torch.load(path, map_location=map_location, weights_only=False)
        return cls(data)

    @classmethod
    def from_loaded(cls, data) -> "Pi3XDecodedFeatures":
        """
        Accept a raw dict (from torch.load) or an already-wrapped instance (no-op).
        Use this in llava_arch.py where the type depends on which script produced the file.
        """
        if isinstance(data, cls):
            return data
        return cls(data)

    # ------------------------------------------------------------------ #
    #  Core tensors                                                        #
    # ------------------------------------------------------------------ #

    @property
    def decoded_features(self) -> torch.Tensor:
        """Full main decoder output.  Shape: (F, T, C=2048).  New schema only."""
        if self._mode == "legacy":
            raise AttributeError("decoded_features is not available in legacy schema.")
        return self._frames["decoded_features"]

    @property
    def frame_idx(self) -> Optional[torch.Tensor]:
        """Original video frame indices.  Shape: (F,).  None in legacy schema."""
        if self._mode == "new":
            return self._frames["frame_idx"]
        return None

    @property
    def decoded_pos_template(self) -> Optional[torch.Tensor]:
        """
        Token positional-encoding template shared across all frames.
        Shape: (T, 2).  None in legacy schema.
        """
        if self._mode == "new":
            return self._meta["decoded_pos_template"]
        return None

    # ------------------------------------------------------------------ #
    #  Meta                                                                #
    # ------------------------------------------------------------------ #

    @property
    def patch_start_idx(self) -> int:
        """Number of register tokens = index where patch tokens start (= 5)."""
        if self._mode == "new":
            return int(self._meta["patch_start_idx"])
        return 1  # legacy: only 1 camera token stored

    @property
    def num_frames(self) -> int:
        if self._mode == "new":
            return int(self._meta["num_frames"])
        return int(self._legacy["camera_tokens"].shape[0])

    @property
    def input_size(self) -> Optional[int]:
        return int(self._meta["input_size"]) if self._mode == "new" else None

    @property
    def patch_size(self) -> Optional[int]:
        return int(self._meta["patch_size"]) if self._mode == "new" else None

    # ------------------------------------------------------------------ #
    #  Camera tokens — requires compute_camera_tokens() in new schema     #
    # ------------------------------------------------------------------ #

    def compute_camera_tokens(
        self,
        camera_decoder: nn.Module,
        device=None,
        dtype=None,
    ) -> torch.Tensor:
        """
        Run pi3.camera_decoder on the stored decoded_features and cache the result.

        Must be called before accessing .camera_tokens in new schema mode.

        Args:
            camera_decoder: pi3.camera_decoder (TransformerDecoder, out_dim=512).
                            Typically obtained via spatial_tower.camera_decoder.
            device: target device for computation (e.g. 'cuda:0').
            dtype:  target dtype (e.g. torch.bfloat16).

        Returns:
            camera_decoder_features  Tensor[F, T, 512]
        """
        if self._mode == "legacy":
            raise AttributeError("compute_camera_tokens is not needed in legacy schema.")

        feats = self.decoded_features          # (F, T, 2048)
        pos   = self.decoded_pos_template      # (T, 2)
        F     = feats.shape[0]
        pos_expanded = pos.unsqueeze(0).expand(F, -1, -1)  # (F, T, 2)

        if device is not None:
            feats        = feats.to(device=device)
            pos_expanded = pos_expanded.to(device=device)
        if dtype is not None:
            feats = feats.to(dtype=dtype)

        with torch.no_grad():
            cam_feats = camera_decoder(feats, xpos=pos_expanded)  # (F, T, 512)

        self._camera_decoder_cache = cam_feats
        return cam_feats

    @property
    def camera_tokens(self) -> torch.Tensor:
        """
        Register-token slice of the camera_decoder branch output.
        Shape: (F, num_patches, D=512)  — new schema (patch slice of camera_decoder output,
                                         matching what Pi3's camera_head receives).
        Shape: (F, 1, C)               — legacy schema.

        In new schema, call compute_camera_tokens(pi3.camera_decoder) first.
        """
        if self._mode == "legacy":
            return self._legacy["camera_tokens"]
        if self._camera_decoder_cache is None:
            raise RuntimeError(
                "camera_tokens requires compute_camera_tokens(pi3.camera_decoder) "
                "to be called first. In llava_arch.py call "
                "_sf.compute_camera_tokens(spatial_tower.camera_decoder, ...)."
            )
        return self._camera_decoder_cache[:, self.patch_start_idx :, :]

    @property
    def camera_decoder_features(self) -> torch.Tensor:
        """
        Full camera_decoder branch output.  Shape: (F, T, D=512).
        Available only after compute_camera_tokens() has been called.
        """
        if self._mode == "legacy":
            raise AttributeError("camera_decoder_features not available in legacy schema.")
        if self._camera_decoder_cache is None:
            raise RuntimeError(
                "camera_decoder_features requires compute_camera_tokens() to be called first."
            )
        return self._camera_decoder_cache

    # ------------------------------------------------------------------ #
    #  Patch tokens                                                        #
    # ------------------------------------------------------------------ #

    @property
    def register_tokens(self) -> torch.Tensor:
        """
        All register/special tokens from the main decoder (indices 0..patch_start_idx-1).
        Shape: (F, patch_start_idx, C=2048).  New schema only.
        """
        if self._mode == "legacy":
            raise AttributeError("register_tokens not available in legacy schema.")
        return self.decoded_features[:, : self.patch_start_idx, :]

    @property
    def patch_tokens(self) -> torch.Tensor:
        """
        Spatial patch tokens from the main decoder.
        Shape: (F, num_patches, C=2048).
        """
        if self._mode == "legacy":
            return self._legacy["patch_tokens"]
        return self.decoded_features[:, self.patch_start_idx :, :]

    @property
    def patch_pos(self) -> Optional[torch.Tensor]:
        """Positional encoding for patch tokens only.  Shape: (num_patches, 2)."""
        if self._mode == "new":
            return self.decoded_pos_template[self.patch_start_idx :]
        return None

    @property
    def all_tokens(self) -> torch.Tensor:
        """
        All main-decoder tokens (register + patch), dim C=2048.
        Note: camera_decoder_features (dim D=512) are separate and accessed via
        camera_tokens / camera_decoder_features after compute_camera_tokens().
        """
        if self._mode == "new":
            return self.decoded_features
        return torch.cat([self.camera_tokens, self.patch_tokens], dim=1)

    # ------------------------------------------------------------------ #
    #  Utilities                                                           #
    # ------------------------------------------------------------------ #

    def to(self, device=None, dtype=None) -> "Pi3XDecodedFeatures":
        """Move/cast all stored tensors in-place and return self."""
        def _cast(t):
            if t is None:
                return t
            if device is not None and dtype is not None:
                return t.to(device=device, dtype=dtype)
            if device is not None:
                return t.to(device=device)
            if dtype is not None:
                return t.to(dtype=dtype)
            return t

        if self._mode == "new":
            self._frames["decoded_features"] = _cast(self._frames["decoded_features"])
            if self._frames.get("frame_idx") is not None:
                self._frames["frame_idx"] = _cast(self._frames["frame_idx"])
            self._meta["decoded_pos_template"] = _cast(self._meta["decoded_pos_template"])
            if self._camera_decoder_cache is not None:
                self._camera_decoder_cache = _cast(self._camera_decoder_cache)
        else:
            self._legacy["camera_tokens"] = _cast(self._legacy["camera_tokens"])
            self._legacy["patch_tokens"]  = _cast(self._legacy["patch_tokens"])
        return self

    def is_new_schema(self) -> bool:
        return self._mode == "new"

    def __repr__(self) -> str:
        if self._mode == "new":
            f, t, c = self.decoded_features.shape
            cam_ready = self._camera_decoder_cache is not None
            return (
                f"Pi3XDecodedFeatures(schema=new, frames={f}, tokens={t}, "
                f"patch_start_idx={self.patch_start_idx}, C={c}, "
                f"camera_ready={cam_ready})"
            )
        return (
            f"Pi3XDecodedFeatures(schema=legacy, "
            f"camera_tokens={self._legacy['camera_tokens'].shape}, "
            f"patch_tokens={self._legacy['patch_tokens'].shape})"
        )
