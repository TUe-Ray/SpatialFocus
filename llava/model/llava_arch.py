#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod

import math
import re
import time
import torch
import torch.nn as nn
from .multimodal_encoder.builder import build_vision_tower
from .multimodal_spatial_encoder.builder import build_spatial_tower
from .multimodal_fusion_block.builder import build_multimodal_fusion_block
from .multimodal_resampler.builder import build_vision_resampler
from .multimodal_projector.builder import build_vision_projector
from .pi3x_decoded_features import Pi3XDecodedFeatures
from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from llava.mm_utils import get_anyres_image_grid_shape
from llava.utils import rank0_print, rank_print
import random
from einops import rearrange

import torch.nn.functional as F
import numpy as np
import cv2  # OpenCV for resizing and writing images
import matplotlib.cm as cm # For colormaps
import os
class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            delay_load = getattr(config, "delay_load", False)
            self.vision_tower = build_vision_tower(config, delay_load=delay_load)
            
            # create spatial tower and fusion block
            if hasattr(config, "spatial_tower"):
                self.spatial_tower = build_spatial_tower(config, delay_load=True)
            if hasattr(config, "fusion_block"):
                self.fusion_block = build_multimodal_fusion_block(config, delay_load=delay_load)

            self.vision_resampler = build_vision_resampler(config, vision_tower=self.vision_tower)
            self.mm_projector = build_vision_projector(config, vision_cfg=self.vision_tower.config)

            if "unpad" in getattr(config, "mm_patch_merge_type", ""):
                self.image_newline = nn.Parameter(torch.empty(config.hidden_size, dtype=self.dtype))

    def get_vision_tower(self):
        vision_tower = getattr(self, "vision_tower", None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def get_spatial_tower(self):
        spatial_tower = getattr(self, "spatial_tower", None)
        if type(spatial_tower) is list:
            spatial_tower = spatial_tower[0]
        return spatial_tower

    def get_fusion_block(self):
        fusion_block = getattr(self, "fusion_block", None)
        if type(fusion_block) is list:
            fusion_block = fusion_block[0]
        return fusion_block

    def initialize_spatial_tower(self, model_args, fsdp=None):
        cli_spatial_tower = model_args.spatial_tower
        self.config.mm_spatial_tower = cli_spatial_tower

        if self.get_spatial_tower() is None:
            # When creating the spatial tower for the first time, force eager load.
            # Otherwise some towers keep only config (delay_load=True) and fail at first forward.
            spatial_tower = build_spatial_tower(model_args, delay_load=False)

            if hasattr(spatial_tower.config, "to_dict"):
                cfg_dict = spatial_tower.config.to_dict()
            else:
                cfg_dict = dict(spatial_tower.config)

            for k, v in cfg_dict.items():
                setattr(self.config, k, v)

            if fsdp is not None and len(fsdp) > 0:
                self.spatial_tower = [spatial_tower]
            else:
                self.spatial_tower = spatial_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                spatial_tower = self.spatial_tower[0]
            else:
                spatial_tower = self.spatial_tower

            if hasattr(spatial_tower, "spatial_tower_name"):
                spatial_tower.spatial_tower_name = cli_spatial_tower
            elif hasattr(spatial_tower, "tower_name"):
                spatial_tower.tower_name = cli_spatial_tower
            elif hasattr(spatial_tower, "model_name"):
                spatial_tower.model_name = cli_spatial_tower
            print("[DEBUG] cli spatial_tower =", model_args.spatial_tower)
            print("[DEBUG] existing spatial tower =", spatial_tower)
            print("[DEBUG] spatial_tower_name =", getattr(spatial_tower, "spatial_tower_name", None))
            print("[DEBUG] tower_name =", getattr(spatial_tower, "tower_name", None))
            print("[DEBUG] model_name =", getattr(spatial_tower, "model_name", None))
            spatial_tower.load_model()

    def initialize_fusion_block(self, model_args, fsdp=None):
        requested_fusion_block = getattr(model_args, "fusion_block", None)
        if requested_fusion_block is not None:
            self.config.fusion_block = requested_fusion_block

        def _expected_class_name(fusion_block_type):
            if fusion_block_type in ["cross_attention", "svf_baseline"]:
                return "CrossAttentionFusion"
            if fusion_block_type == "svf_patch_only":
                return "CrossAttentionFusion"
            if fusion_block_type == "svf_patch_cam_concat":
                return "PatchCrossAttentionCameraConcatFusion"
            if fusion_block_type == "svf_geometry_bridge":
                return "GeometryBridgeFusion"
            if fusion_block_type == "svf_pose_geometry_bridge":
                return "GeometryBridgeFusion"
            if fusion_block_type == "svf_pose_geometry_bridge_reverse":
                return "ReverseGeometryBridgeFusion"
            if fusion_block_type == "svf_cat_feat":
                return "SvfCatFeatFusion"
            if fusion_block_type == "svf_pose_prepend":
                return "SvfPosePrependFusion"
            if fusion_block_type == "svf_pose_geometry_bridge_reverse":
                return "ReverseGeometryBridgeFusion"
            if fusion_block_type == "cross_attention_with_mlp":
                return "CrossAttentionFusionWithMLP"
            if fusion_block_type == "mlp_after_clip_proj":
                return "MLPFusion"
            if fusion_block_type == "transformer":
                return "TransformerFusion"
            if fusion_block_type == "concat_mlp":
                return "ConcatMLPFusion"
            if fusion_block_type == "concat_self_attention":
                return "ConcatSelfAttentionFusion"
            if fusion_block_type == "llava_3d_fusion_block":
                return "llava_3d_fusion_block"
            if fusion_block_type == "video_3d_llm_fusion_block":
                return "video_3d_llm_fusion_block"
            if isinstance(fusion_block_type, str) and fusion_block_type.endswith("_layer_cross_attention"):
                return "MultiLayerCrossAttentionFusion"
            return None

        existing_fusion_block = self.get_fusion_block()
        current_type = getattr(self.config, "fusion_block", None)
        expected_cls = _expected_class_name(current_type)
        needs_rebuild = (
            existing_fusion_block is None
            or (expected_cls is not None and existing_fusion_block.__class__.__name__ != expected_cls)
        )

        # Build/rebuild to keep the instantiated module aligned with config.fusion_block.
        if needs_rebuild:
            self.fusion_block = build_multimodal_fusion_block(self.config)
        else:
            # In case it is frozen by LoRA
            for p in self.fusion_block.parameters():
                p.requires_grad = True

        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location="cpu")

            def get_w(weights, keyword):
                return {k.split(keyword + ".")[1]: v for k, v in weights.items() if keyword in k}

            incompatible_keys = self.fusion_block.load_state_dict(get_w(mm_projector_weights, "fusion_block"), strict=False)
            rank0_print(f"Loaded fusion block weights from {pretrain_mm_mlp_adapter}. Incompatible keys: {incompatible_keys}")

    def get_eomt_extractor(self):
        return getattr(self, "eomt_extractor", None)

    def initialize_eomt_extractor(self, model_args, fsdp=None):
        """Initialize the EoMT extractor as a frozen side branch for mask extraction."""
        eomt_config_path = getattr(model_args, "eomt_config_path", None)
        eomt_ckpt_path = getattr(model_args, "eomt_ckpt_path", None)
        if eomt_config_path is None or eomt_ckpt_path is None:
            rank0_print("EoMT: No config/ckpt path provided, skipping EoMT extractor initialization.")
            return

        try:
            from llava.model.multimodal_eomt import EoMTExtractor
            eomt_cfg = {
                "config_path": eomt_config_path,
                "ckpt_path": eomt_ckpt_path,
                "device": "cpu",  # will be moved to correct device later
            }
            self.eomt_extractor = EoMTExtractor(eomt_cfg)
            rank0_print(f"EoMT extractor initialized from {eomt_ckpt_path}")
        except Exception as e:
            rank0_print(f"EoMT: Failed to initialize extractor: {e}")
            self.eomt_extractor = None

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.mm_vision_tower = vision_tower
        self.config.vision_tower_pretrained = getattr(model_args, "vision_tower_pretrained", "")

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)
            vision_resampler = build_vision_resampler(model_args, vision_tower=vision_tower)
            for k, v in vision_resampler.config.items():
                setattr(self.config, k, v)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
                self.vision_resampler = [vision_resampler]
            else:
                self.vision_tower = vision_tower
                self.vision_resampler = vision_resampler
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_resampler = self.vision_resampler[0]
                vt = self.vision_tower[0]
            else:
                vision_resampler = self.vision_resampler
                vt = self.vision_tower

            vt.vision_tower_name = model_args.vision_tower
            print("[DEBUG] cli vision_tower =", model_args.vision_tower)
            print("[DEBUG] existing vision tower name =", getattr(vt, "vision_tower_name", None))
            vt.load_model()

            for p in self.vision_resampler.parameters():
                p.requires_grad = True

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, "mm_projector_type", "linear")
        current_vt = self.get_vision_tower()
        self.config.mm_hidden_size = getattr(vision_resampler, "hidden_size", current_vt.hidden_size)
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type

        
        if not hasattr(self.config, 'add_faster_video'):
            if model_args.add_faster_video:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.faster_token = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )

        if getattr(self, "mm_projector", None) is None:
            self.mm_projector = build_vision_projector(self.config, vision_cfg=vision_tower.config)

            if "unpad" in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std)
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location="cpu")

            def get_w(weights, keyword):
                return {k.split(keyword + ".")[1]: v for k, v in weights.items() if keyword in k}

            incompatible_keys = self.mm_projector.load_state_dict(get_w(mm_projector_weights, "mm_projector"))
            rank0_print(f"Loaded mm projector weights from {pretrain_mm_mlp_adapter}. Incompatible keys: {incompatible_keys}")
            incompatible_keys = self.vision_resampler.load_state_dict(get_w(mm_projector_weights, "vision_resampler"), strict=False)
            rank0_print(f"Loaded vision resampler weights from {pretrain_mm_mlp_adapter}. Incompatible keys: {incompatible_keys}")


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of the image (height, width).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    # Compute aspect ratios
    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    # Determine padding size and direction
    if original_aspect_ratio > current_aspect_ratio:
        # Padding was added to the height
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding : current_height - padding, :]
    else:
        # Padding was added to the width
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding : current_width - padding]

    return unpadded_tensor


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def get_spatial_tower(self):
        return self.get_model().get_spatial_tower()

    def get_fusion_block(self):
        return self.get_model().get_fusion_block()

    def _get_eomt_mask_pooler(self):
        pooler = getattr(self, "_eomt_mask_pooler", None)
        if pooler is None:
            from llava.model.multimodal_eomt import MaskGuidedPooler

            pooler = MaskGuidedPooler()
            self._eomt_mask_pooler = pooler
        return pooler

    def _get_eomt_object_selector(self):
        selector = getattr(self, "_eomt_object_selector", None)
        if selector is None:
            from llava.model.multimodal_eomt import EoMTObjectTokenSelector

            selector = EoMTObjectTokenSelector()
            self._eomt_object_selector = selector
        return selector

    def _get_eomt_object_block_appender(self):
        appender = getattr(self, "_eomt_object_block_appender", None)
        if appender is None:
            from llava.model.multimodal_eomt import EoMTObjectBlockAppender

            appender = EoMTObjectBlockAppender()
            self._eomt_object_block_appender = appender
        return appender

    def _get_eomt_obj_info_builder(self):
        builder = getattr(self, "_eomt_obj_info_builder", None)
        if builder is None:
            from llava.model.multimodal_eomt import EoMTObjInfoBuilder

            builder = EoMTObjInfoBuilder()
            self._eomt_obj_info_builder = builder
        return builder

    def _get_or_create_eomt_vector_parameter(self, name, hidden_size, trainable=True):
        model = self.get_model()
        existing = getattr(model, name, None)
        if isinstance(existing, nn.Parameter):
            existing.requires_grad = bool(trainable)
            return existing

        ref = model.embed_tokens.weight
        param = nn.Parameter(
            torch.zeros(hidden_size, device=ref.device, dtype=ref.dtype),
            requires_grad=bool(trainable),
        )
        nn.init.normal_(param, std=0.02)
        setattr(model, name, param)
        return param

    def _build_external_selection_socket(self, pooled_outputs):
        frame_meta = pooled_outputs.get("frame_meta", []) if isinstance(pooled_outputs, dict) else []
        sample_frame_pairs = pooled_outputs.get("aligned_sample_frame_pairs", []) if isinstance(pooled_outputs, dict) else []
        entries = []

        if not isinstance(frame_meta, list):
            frame_meta = []

        for idx, meta in enumerate(frame_meta):
            if not isinstance(meta, dict):
                continue
            if idx < len(sample_frame_pairs) and isinstance(sample_frame_pairs[idx], (list, tuple)) and len(sample_frame_pairs[idx]) >= 2:
                sample_idx = int(sample_frame_pairs[idx][0])
                frame_idx = int(sample_frame_pairs[idx][1])
            else:
                sample_idx = idx
                frame_idx = idx

            socket_dict = meta.get("external_selection", meta)
            selected_words = socket_dict.get("selected_words", None)
            selected_mask_ids = socket_dict.get("selected_mask_ids", None)
            selected_query_ids = socket_dict.get("selected_query_ids", None)

            if any(x is not None for x in [selected_words, selected_mask_ids, selected_query_ids]):
                entries.append(
                    {
                        "sample_idx": sample_idx,
                        "frame_idx": frame_idx,
                        "selected_words": selected_words,
                        "selected_mask_ids": selected_mask_ids,
                        "selected_query_ids": selected_query_ids,
                    }
                )

        if len(entries) == 0:
            return None
        return {"entries": entries}

    def _build_eomt_obj_info_tokens(self, visual_tokens):
        hidden_size = int(visual_tokens.shape[1])
        obj_info_mode = str(getattr(self.config, "mm_eomt_obj_info_mode", "none"))
        obj_info_text = str(getattr(self.config, "mm_eomt_obj_info_text", "Object information from the image:"))
        obj_info_trainable = bool(getattr(self.config, "mm_eomt_obj_info_trainable", True))

        learnable_embedding = None
        if obj_info_mode == "learnable_embedding":
            learnable_embedding = self._get_or_create_eomt_vector_parameter(
                name="eomt_obj_info_learnable_embedding",
                hidden_size=hidden_size,
                trainable=obj_info_trainable,
            )

        builder = self._get_eomt_obj_info_builder()
        return builder.build(
            mode=obj_info_mode,
            hidden_size=hidden_size,
            device=visual_tokens.device,
            dtype=visual_tokens.dtype,
            text_phrase=obj_info_text,
            learnable_embedding=learnable_embedding,
        )

    def _compose_eomt_object_block_tokens(self, object_tokens, visual_tokens, object_block_outputs):
        if not torch.is_tensor(object_tokens) or object_tokens.ndim != 2 or object_tokens.shape[0] == 0:
            return None, "empty_selected_object_tokens"

        block_tokens = object_tokens.to(device=visual_tokens.device, dtype=visual_tokens.dtype)

        type_embedding_used = False
        if bool(getattr(self.config, "mm_eomt_use_object_type_embedding", False)):
            object_type_embedding = self._get_or_create_eomt_vector_parameter(
                name="eomt_object_type_embedding",
                hidden_size=int(block_tokens.shape[1]),
                trainable=True,
            )
            block_tokens = block_tokens + object_type_embedding.to(device=block_tokens.device, dtype=block_tokens.dtype).unsqueeze(0)
            type_embedding_used = True

        obj_info_tokens, obj_info_debug = self._build_eomt_obj_info_tokens(visual_tokens=visual_tokens)
        if torch.is_tensor(obj_info_tokens) and obj_info_tokens.ndim == 2 and obj_info_tokens.shape[0] > 0:
            block_tokens = torch.cat((obj_info_tokens.to(device=block_tokens.device, dtype=block_tokens.dtype), block_tokens), dim=0)

        if isinstance(object_block_outputs, dict):
            object_block_outputs["obj_info_mode"] = obj_info_debug.get("obj_info_mode", "none")
            object_block_outputs["obj_info_used"] = bool(obj_info_debug.get("obj_info_used", False))
            object_block_outputs["obj_info_reason"] = obj_info_debug.get("obj_info_reason", None)
            if "obj_info_text" in obj_info_debug:
                object_block_outputs["obj_info_text"] = obj_info_debug.get("obj_info_text", "")
            object_block_outputs["object_type_embedding_used"] = type_embedding_used
            object_block_outputs["object_block_token_count"] = int(block_tokens.shape[0])

        return block_tokens, None

    def _compute_eomt_object_block_side_output(self):
        enabled = bool(getattr(self.config, "mm_eomt_enable_object_block", False))
        selector_mode = str(getattr(self.config, "mm_eomt_selector_mode", "class_aware"))
        ordering_mode = str(getattr(self.config, "mm_eomt_selector_order", "frame_then_score"))
        append_position = str(getattr(self.config, "mm_eomt_object_block_position", "after_visual"))
        if append_position not in {"before_visual", "after_visual"}:
            append_position = "after_visual"

        debug = {
            "enabled": enabled,
            "used_object_block": False,
            "has_selected_objects": False,
            "selector_mode": selector_mode,
            "ordering_mode": ordering_mode,
            "append_position": append_position,
            "obj_info_mode": str(getattr(self.config, "mm_eomt_obj_info_mode", "none")),
            "obj_info_used": False,
            "obj_info_reason": None,
            "object_type_embedding_used": False,
            "selected_count": 0,
            "selected_frame_ids": [],
            "selected_sample_frame_pairs": [],
            "selected_scores": [],
            "selected_indices": [],
            "truncated_per_frame_count": 0,
            "truncated_global_count": 0,
            "fallback_reason": None,
            "taxonomy_note": None,
            "external_socket_present": False,
            "external_selection_contract": {},
            "external_socket_note": None,
            "no_object_class_id": int(getattr(self.config, "mm_eomt_selector_no_object_class_id", -1)),
            "selected_tokens_by_sample": {},
        }

        if not enabled:
            debug["fallback_reason"] = "object_block_disabled"
            return debug

        pooled_outputs = getattr(self, "_last_eomt_pooled_outputs", None)
        if not isinstance(pooled_outputs, dict):
            debug["fallback_reason"] = "missing_pooled_outputs"
            return debug

        external_selection = self._build_external_selection_socket(pooled_outputs)
        debug["external_socket_present"] = bool(
            isinstance(external_selection, dict)
            and isinstance(external_selection.get("entries", None), list)
            and len(external_selection.get("entries", [])) > 0
        )

        try:
            selector = self._get_eomt_object_selector()
            selected = selector.select(
                pooled_outputs=pooled_outputs,
                config=self.config,
                external_selection=external_selection,
            )
        except Exception as e:
            rank0_print(f"EoMT object selector error: {e}")
            debug["fallback_reason"] = "selector_execution_error"
            return debug

        if not isinstance(selected, dict):
            debug["fallback_reason"] = "invalid_selector_output"
            return debug

        debug["selector_mode"] = str(selected.get("selector_mode", selector_mode))
        debug["ordering_mode"] = str(selected.get("ordering_mode", ordering_mode))
        debug["append_position"] = str(selected.get("append_position", append_position))
        debug["selected_count"] = int(selected.get("selected_count", 0))
        debug["selected_sample_frame_pairs"] = list(selected.get("selected_sample_frame_pairs", []))
        debug["selected_frame_ids"] = [
            pair[1]
            for pair in debug["selected_sample_frame_pairs"]
            if isinstance(pair, (tuple, list)) and len(pair) >= 2
        ]
        debug["selected_scores"] = list(selected.get("selected_scores", []))
        debug["selected_indices"] = list(selected.get("selected_indices", []))
        debug["truncated_per_frame_count"] = int(selected.get("truncated_per_frame_count", 0))
        debug["truncated_global_count"] = int(selected.get("truncated_global_count", 0))
        debug["taxonomy_note"] = selected.get("taxonomy_note", None)
        debug["external_socket_note"] = selected.get("external_socket_note", None)
        debug["external_selection_contract"] = selected.get("external_selection_contract", {})
        debug["no_object_class_id"] = int(selected.get("no_object_class_id", debug["no_object_class_id"]))
        debug["selected_tokens_by_sample"] = selected.get("selected_tokens_by_sample", {})

        debug["has_selected_objects"] = debug["selected_count"] > 0
        fallback_reason = selected.get("fallback_reason", None)
        if fallback_reason is not None:
            debug["fallback_reason"] = str(fallback_reason)
        elif not debug["has_selected_objects"]:
            debug["fallback_reason"] = "no_selected_objects"

        return debug

    def _build_eomt_pool_side_cache(
        self,
        per_sample_visual_features,
        split_sizes,
        video_idx_in_batch,
        modalities,
        eomt_images,
        eomt_meta,
        image_aspect_ratio,
        mm_patch_merge_type,
        image_sizes,
    ):
        has_eomt_outputs = getattr(self, "_last_eomt_outputs", None) is not None

        if per_sample_visual_features is None:
            per_sample_visual_features = []

        total_samples = len(per_sample_visual_features)
        modalities = modalities if isinstance(modalities, list) else [modalities]

        if eomt_images is None:
            eomt_images_list = [None for _ in range(total_samples)]
        else:
            eomt_images_list = list(eomt_images)
            if len(eomt_images_list) < total_samples:
                eomt_images_list.extend([None for _ in range(total_samples - len(eomt_images_list))])

        if eomt_meta is None:
            eomt_meta_list = [None for _ in range(total_samples)]
        else:
            eomt_meta_list = list(eomt_meta)
            if len(eomt_meta_list) < total_samples:
                eomt_meta_list.extend([None for _ in range(total_samples - len(eomt_meta_list))])

        # Global frame index offsets follow the same flatten order used for EoMT input:
        # for sample_frames in eomt_images: all_frames.extend(sample_frames)
        sample_frame_offsets = []
        total_eomt_frames = 0
        for sample_idx in range(total_samples):
            sample_frame_offsets.append(total_eomt_frames)
            sample_frames = eomt_images_list[sample_idx]
            if sample_frames is None:
                sample_frames = []
            total_eomt_frames += len(sample_frames)

        aligned_visual_list = []
        aligned_frame_meta_list = []
        aligned_global_frame_indices = []
        aligned_sample_frame_pairs = []

        pooled_sample_indices = []
        skipped_sample_indices = []
        per_sample_debug = []
        enabled_mask = []
        skip_reasons = []

        anyres_in_aspect = isinstance(image_aspect_ratio, str) and ("anyres" in image_aspect_ratio)

        for sample_idx in range(total_samples):
            sample_visual = per_sample_visual_features[sample_idx]
            sample_visual_shape = tuple(sample_visual.shape) if torch.is_tensor(sample_visual) else None

            if sample_idx < len(modalities):
                sample_modality = modalities[sample_idx]
            elif sample_idx in video_idx_in_batch:
                sample_modality = "video"
            else:
                sample_modality = "image"

            sample_frames = eomt_images_list[sample_idx]
            sample_frame_meta = eomt_meta_list[sample_idx]
            if sample_frames is None:
                sample_frames = []
            if sample_frame_meta is None:
                sample_frame_meta = []

            skip_reason = None
            sample_poolable_frame_count = 0

            if eomt_images is None:
                skip_reason = "missing_eomt_images"
            elif not has_eomt_outputs:
                skip_reason = "missing_eomt_outputs"
            elif len(sample_frames) == 0:
                skip_reason = "empty_eomt_frames"
            elif not torch.is_tensor(sample_visual) or sample_visual.ndim != 3:
                skip_reason = "unsupported_visual_shape_for_eomt_pooling"
            elif sample_modality == "video":
                if len(sample_frames) != sample_visual.shape[0]:
                    skip_reason = "video_frame_count_mismatch"
                else:
                    for frame_idx in range(sample_visual.shape[0]):
                        aligned_visual_list.append(sample_visual[frame_idx : frame_idx + 1].detach())
                        if frame_idx < len(sample_frame_meta) and isinstance(sample_frame_meta[frame_idx], dict):
                            aligned_frame_meta_list.append(dict(sample_frame_meta[frame_idx]))
                        else:
                            aligned_frame_meta_list.append({})
                        aligned_global_frame_indices.append(sample_frame_offsets[sample_idx] + frame_idx)
                        aligned_sample_frame_pairs.append((sample_idx, frame_idx))
                        sample_poolable_frame_count += 1
            else:
                if sample_visual.shape[0] == 1:
                    if len(sample_frames) != 1:
                        skip_reason = "single_image_eomt_count_mismatch"
                    else:
                        aligned_visual_list.append(sample_visual[0:1].detach())
                        if len(sample_frame_meta) >= 1 and isinstance(sample_frame_meta[0], dict):
                            aligned_frame_meta_list.append(dict(sample_frame_meta[0]))
                        else:
                            aligned_frame_meta_list.append({})
                        aligned_global_frame_indices.append(sample_frame_offsets[sample_idx])
                        aligned_sample_frame_pairs.append((sample_idx, 0))
                        sample_poolable_frame_count = 1
                elif sample_visual.shape[0] > 1:
                    if anyres_in_aspect:
                        skip_reason = "anyres_not_supported_for_eomt_pooling"
                    else:
                        skip_reason = "multi_patch_not_supported_for_eomt_pooling"
                else:
                    skip_reason = "unsupported_visual_shape_for_eomt_pooling"

            is_poolable = sample_poolable_frame_count > 0
            if is_poolable:
                pooled_sample_indices.append(sample_idx)
                enabled_mask.append(True)
            else:
                skipped_sample_indices.append(sample_idx)
                enabled_mask.append(False)
                if skip_reason is not None:
                    skip_reasons.append(skip_reason)

            per_sample_debug.append(
                {
                    "sample_idx": sample_idx,
                    "modality": sample_modality,
                    "visual_shape_before_merge": sample_visual_shape,
                    "eomt_frame_count": len(sample_frames),
                    "is_poolable": is_poolable,
                    "poolable_frame_count": sample_poolable_frame_count,
                    "skip_reason": skip_reason,
                }
            )

        pooled_visual_features = None
        if len(aligned_visual_list) > 0:
            pooled_visual_features = torch.cat(aligned_visual_list, dim=0)

        pool_debug = {
            "total_samples": total_samples,
            "total_eomt_frames": total_eomt_frames,
            "poolable_frame_count": len(aligned_visual_list),
            "pooled_sample_indices": pooled_sample_indices,
            "skipped_sample_indices": skipped_sample_indices,
            "aligned_global_frame_indices": aligned_global_frame_indices,
            "aligned_sample_frame_pairs": aligned_sample_frame_pairs,
            "split_sizes": list(split_sizes) if split_sizes is not None else None,
            "image_aspect_ratio": image_aspect_ratio,
            "mm_patch_merge_type": mm_patch_merge_type,
            "per_sample": per_sample_debug,
        }

        self._last_eomt_pool_visual_features = pooled_visual_features
        self._last_eomt_pool_frame_meta = aligned_frame_meta_list
        self._last_eomt_pool_debug = pool_debug
        self._last_eomt_pool_enabled_mask = enabled_mask
        self._last_eomt_pool_skip_reasons = skip_reasons

        return pooled_visual_features, aligned_frame_meta_list, pool_debug

    def _compute_eomt_mask_pooled_side_output(self):
        eomt_outputs = getattr(self, "_last_eomt_outputs", None)
        if eomt_outputs is None:
            return None

        pool_visual_features = getattr(self, "_last_eomt_pool_visual_features", None)
        pool_frame_meta = getattr(self, "_last_eomt_pool_frame_meta", None)
        pool_debug = getattr(self, "_last_eomt_pool_debug", None)
        if pool_frame_meta is None:
            pool_frame_meta = []
        if pool_debug is None:
            pool_debug = {}

        soft_masks = eomt_outputs.get("soft_masks", None)
        class_logits = eomt_outputs.get("class_logits", None)

        pool_top_k = int(getattr(self.config, "eomt_pool_top_k", 5))
        pool_selection = str(getattr(self.config, "eomt_pool_selection", "mean_mask_confidence"))
        pool_area_threshold = float(getattr(self.config, "eomt_pool_mask_area_threshold", 0.5))

        def _skipped_result(reason):
            return {
                "pooled_tokens": None,
                "selected_indices": None,
                "selected_scores": None,
                "selected_class_ids": None,
                "selection_method": pool_selection,
                "frame_meta": [],
                "mask_resolution": eomt_outputs.get("mask_resolution", None),
                "query_count": eomt_outputs.get("query_count", None),
                "pool_debug": pool_debug,
                "pool_skipped": True,
                "skip_reason": reason,
            }

        if soft_masks is None:
            return _skipped_result("missing_soft_masks")

        if pool_visual_features is None or (torch.is_tensor(pool_visual_features) and pool_visual_features.shape[0] == 0):
            return _skipped_result("no_frame_aligned_visual_features")

        if not torch.is_tensor(pool_visual_features) or pool_visual_features.ndim != 3:
            return _skipped_result("invalid_aligned_visual_features")

        aligned_global_frame_indices = pool_debug.get("aligned_global_frame_indices", [])
        aligned_sample_frame_pairs = pool_debug.get("aligned_sample_frame_pairs", [])
        if len(aligned_global_frame_indices) == 0:
            return _skipped_result("no_frame_aligned_visual_features")

        if max(aligned_global_frame_indices) >= soft_masks.shape[0]:
            return _skipped_result("aligned_frame_index_out_of_range")

        aligned_index_tensor = torch.as_tensor(
            aligned_global_frame_indices,
            dtype=torch.long,
            device=soft_masks.device,
        )
        aligned_soft_masks = soft_masks.index_select(0, aligned_index_tensor)

        aligned_class_logits = None
        if class_logits is not None:
            if max(aligned_global_frame_indices) >= class_logits.shape[0]:
                return _skipped_result("aligned_frame_index_out_of_range")
            aligned_class_logits = class_logits.index_select(0, aligned_index_tensor)

        if len(pool_frame_meta) == 0:
            raw_frame_meta = eomt_outputs.get("frame_meta", None)
            if isinstance(raw_frame_meta, list):
                pool_frame_meta = [
                    raw_frame_meta[idx] if idx < len(raw_frame_meta) else {}
                    for idx in aligned_global_frame_indices
                ]

        try:
            pooler = self._get_eomt_mask_pooler()
            with torch.no_grad():
                pooled = pooler(
                    soft_masks=aligned_soft_masks.to(device=pool_visual_features.device),
                    visual_features=pool_visual_features,
                    class_logits=(
                        aligned_class_logits.to(device=pool_visual_features.device)
                        if aligned_class_logits is not None
                        else None
                    ),
                    top_k=pool_top_k,
                    selection=pool_selection,
                    mask_area_threshold=pool_area_threshold,
                )

            pooled["frame_meta"] = pool_frame_meta
            pooled["mask_resolution"] = eomt_outputs.get("mask_resolution", None)
            pooled["query_count"] = eomt_outputs.get("query_count", None)
            pooled["pool_debug"] = pool_debug
            pooled["aligned_global_frame_indices"] = aligned_global_frame_indices
            pooled["aligned_sample_frame_pairs"] = aligned_sample_frame_pairs
            pooled["pool_skipped"] = False
            return pooled
        except Exception as e:
            rank0_print(f"EoMT mask pooling side branch error: {e}")
            skipped = _skipped_result("pooler_execution_error")
            skipped["pool_error"] = str(e)
            return skipped

    def _split_prefix_tokens_for_square_grid(self, image_feature):
        num_tokens = image_feature.shape[1]
        side = int(math.isqrt(num_tokens))
        if side * side == num_tokens:
            return None, image_feature, side

        # Some fusion blocks prepend non-grid tokens (for example, camera tokens).
        for prefix_len in range(1, num_tokens):
            remaining = num_tokens - prefix_len
            side = int(math.isqrt(remaining))
            if side * side == remaining:
                prefix_tokens = image_feature[:, :prefix_len, :]
                grid_tokens = image_feature[:, prefix_len:, :]
                return prefix_tokens, grid_tokens, side

        raise ValueError(
            f"Cannot split tokens into prefix + square grid, got shape {tuple(image_feature.shape)}"
        )

    def get_2dPool(self, image_feature, stride=2):
        height = width = self.get_vision_tower().num_patches_per_side
        num_frames, num_tokens, num_dim = image_feature.shape
        expected_grid_tokens = height * width
        if num_tokens < expected_grid_tokens:
            raise ValueError(
                f"Insufficient tokens for {height}x{width} grid: got {num_tokens}"
            )

        prefix_tokens = None
        if num_tokens > expected_grid_tokens:
            prefix_len = num_tokens - expected_grid_tokens
            prefix_tokens = image_feature[:, :prefix_len, :]
            image_feature = image_feature[:, prefix_len:, :]

        image_feature = image_feature.view(num_frames, height, width, num_dim)
        image_feature = image_feature.permute(0, 3, 1, 2).contiguous()
        # image_feature = nn.functional.max_pool2d(image_feature, self.config.mm_spatial_pool_stride)
        if self.config.mm_spatial_pool_mode == "average":
            image_feature = nn.functional.avg_pool2d(image_feature, stride)
        elif self.config.mm_spatial_pool_mode == "max":
            image_feature = nn.functional.max_pool2d(image_feature, stride)
        elif self.config.mm_spatial_pool_mode == "bilinear":
            height, width = image_feature.shape[2:]
            scaled_shape = [math.ceil(height / stride), math.ceil(width / stride)]
            image_feature = nn.functional.interpolate(image_feature, size=scaled_shape, mode='bilinear')

        else:
            raise ValueError(f"Unexpected mm_spatial_pool_mode: {self.config.mm_spatial_pool_mode}")
        image_feature = image_feature.permute(0, 2, 3, 1)
        image_feature = image_feature.view(num_frames, -1, num_dim)
        if prefix_tokens is not None:
            image_feature = torch.cat((prefix_tokens, image_feature), dim=1)
        return image_feature

    # def encode_images(self, images):
    #     # vision features
    #     image_features = self.get_model().get_vision_tower()(images)
    #     # set brance
    #     if self.get_model().get_spatial_tower() is not None and self.get_model().get_fusion_block() is not None:
    #         # spatial features
    #         spatial_encoder_type = self.get_model().config.spatial_tower
    #         if spatial_encoder_type == "cut3r":
    #             # Scale up image by 16/14 before passing to spatial tower
    #             images_scaled = nn.functional.interpolate(images, size=(432, 432), mode='bilinear')
    #             images_for_spatial_tower = images_scaled.unsqueeze(1) ## FIXME: the first dimension is the number of frames in one batch
    #             image_spatial_features = self.get_model().get_spatial_tower()(images_for_spatial_tower)
    #         elif spatial_encoder_type == "vggt":
    #             images_scaled = nn.functional.interpolate(images, size=(378, 378), mode='bilinear')
    #             images_for_spatial_tower = images_scaled.unsqueeze(1) ## FIXME: the first dimension is the number of frames in one batch
    #             image_spatial_features = self.get_model().get_spatial_tower()(images_for_spatial_tower)
    #         elif spatial_encoder_type == "cut3r_points":
    #             images_scaled = nn.functional.interpolate(images, size=(432, 432), mode='bilinear')
    #             images_for_spatial_tower = images_scaled.unsqueeze(1) ## FIXME: the first dimension is the number of frames in one batch
    #             image_spatial_features = self.get_model().get_spatial_tower()(images_for_spatial_tower)
    #         else:
    #             raise ValueError(f"Unexpected spatial encoder type: {spatial_encoder_type}")
            
    #         fusion_block_type = self.get_model().config.fusion_block
            
    #         # Handle special case for mlp2x_gelu_cat first
    #         if fusion_block_type == "mlp2x_gelu_cat":
    #             image_features = torch.cat((image_features, image_spatial_features), dim=-1)
    #             image_features = self.get_model().get_fusion_block()(image_features)
    #         # Handle special case for mlp2x_gelu
    #         elif fusion_block_type == "mlp2x_gelu":
    #             image_features = self.get_model().get_fusion_block()(image_features)
    #             image_features = self.get_model().mm_projector(image_features)
    #         # Handle all other fusion types that follow the same pattern
    #         elif fusion_block_type in ["cross_attention", "mlp", "transformer"]:
    #             image_features = self.get_model().get_fusion_block()(image_features, image_spatial_features)
    #             image_features = self.get_model().mm_projector(image_features)
    #         elif fusion_block_type == "llava_3d_fusion_block":
    #             image_features = self.get_model().get_fusion_block()(image_features, image_spatial_features)
    #             image_features = self.get_model().mm_projector(image_features)
    #         else:
    #             raise ValueError(f"Unexpected fusion block type: {fusion_block_type}")
    #     else:
    #         # project features
    #         image_features = self.get_model().mm_projector(image_features)

    #     return image_features

    def encode_images(self, images, spatial_features=None, point_maps=None):
        # vision features
        image_features = self.get_model().get_vision_tower()(images)
        # fuse with spatial features
        if self.get_model().get_spatial_tower() is not None and self.get_model().get_fusion_block() is not None:
            spatial_encoder_type = self.get_model().config.spatial_tower
            fusion_block_type = self.get_model().config.fusion_block

            zero_spatial_features = getattr(self.get_model().config, "zero_spatial_features", False)
            if isinstance(zero_spatial_features, str):
                zero_spatial_features = zero_spatial_features.lower() in {"1", "true", "yes", "y", "on"}

            if spatial_encoder_type.endswith("points"):
                points = self.get_model().get_spatial_tower()(images)
                image_features = self.get_model().get_fusion_block()(image_features, points)
                image_features = self.get_model().mm_projector(image_features)

            else:
                spatial_tower = self.get_model().get_spatial_tower()
                cfg_spatial_tower_name = str(spatial_encoder_type or "").lower()
                runtime_spatial_tower_name = str(
                    getattr(spatial_tower, "spatial_tower_name", None)
                    or getattr(spatial_tower, "tower_name", None)
                    or getattr(spatial_tower, "model_name", None)
                    or ""
                ).lower()
                spatial_tower_module = ""
                spatial_tower_class_name = ""
                if spatial_tower is not None:
                    spatial_tower_module = getattr(spatial_tower.__class__, "__module__", "").lower()
                    spatial_tower_class_name = spatial_tower.__class__.__name__.lower()

                is_cut3r_spatial = any(
                    "cut3r" in value
                    for value in (
                        cfg_spatial_tower_name,
                        runtime_spatial_tower_name,
                        spatial_tower_module,
                        spatial_tower_class_name,
                    )
                )
                is_pi3x_spatial = any(
                    "pi3x" in value
                    for value in (
                        cfg_spatial_tower_name,
                        runtime_spatial_tower_name,
                        spatial_tower_module,
                        spatial_tower_class_name,
                    )
                )
                loaded_spatial_features = spatial_features[0] if spatial_features is not None else None
                has_token_pair_features = (
                    isinstance(loaded_spatial_features, dict)
                    and "camera_tokens" in loaded_spatial_features
                    and "patch_tokens" in loaded_spatial_features
                )

                _sf = None
                camera_pose = None

                if zero_spatial_features:
                    return self.get_model().mm_projector(image_features)
                elif spatial_features is not None and has_token_pair_features and (is_cut3r_spatial or not is_pi3x_spatial):
                    camera_tokens, patch_tokens = loaded_spatial_features["camera_tokens"], loaded_spatial_features["patch_tokens"]
                elif spatial_features is not None and is_pi3x_spatial:
                    _sf = Pi3XDecodedFeatures.from_loaded(loaded_spatial_features)
                    if _sf.is_new_schema():
                        # Camera tokens must be computed from the stored decoded_features
                        # by running pi3.camera_decoder (lightweight head, no re-encoding).
                        _spatial_tower = spatial_tower
                        _cam_dec = getattr(_spatial_tower, "camera_decoder", None)
                        if _cam_dec is None:
                            raise RuntimeError(
                                "Pi3X spatial tower must be loaded to compute camera_tokens "
                                "from decoded_features. Ensure spatial_tower is not None."
                            )
                        _sf.compute_camera_tokens(
                            _cam_dec,
                            device=images.device,
                            dtype=self.dtype,
                        )
                        # svf_pose_prepend needs camera_head to get the 12-value pose.
                        if fusion_block_type == 'svf_pose_prepend':
                            _cam_head = getattr(_spatial_tower, "camera_head", None)
                            if _cam_head is None:
                                raise RuntimeError(
                                    "svf_pose_prepend requires pi3.camera_head. "
                                    "Ensure Pi3X spatial tower is loaded."
                                )
                            _patch_h = _patch_w = _sf.input_size // _sf.patch_size
                            _sf.compute_camera_pose(_cam_head, _patch_h, _patch_w, device=images.device)
                            camera_pose = _sf.camera_pose
                    camera_tokens, patch_tokens = _sf.camera_tokens, _sf.patch_tokens
                else:
                    camera_tokens, patch_tokens = spatial_tower(images)
                    # Runtime path parity for svf_pose_prepend (pi3x only):
                    # compute camera_pose from camera_head using runtime camera_tokens.
                    if fusion_block_type == 'svf_pose_prepend' and is_pi3x_spatial:
                        _spatial_tower = spatial_tower
                        _cam_head = getattr(_spatial_tower, "camera_head", None)
                        if _cam_head is None:
                            raise RuntimeError(
                                "svf_pose_prepend requires pi3.camera_head in runtime path. "
                                "Ensure Pi3X spatial tower is loaded."
                            )

                        patch_token_num = int(patch_tokens.shape[1])
                        patch_side = int(math.isqrt(patch_token_num))
                        if patch_side * patch_side != patch_token_num:
                            raise RuntimeError(
                                f"svf_pose_prepend runtime path expects square patch grid, got {patch_token_num} tokens"
                            )

                        cam_tokens_for_pose = camera_tokens
                        cam_head_param = next(_cam_head.parameters(), None)
                        if cam_head_param is not None and cam_tokens_for_pose.dtype != cam_head_param.dtype:
                            cam_tokens_for_pose = cam_tokens_for_pose.to(dtype=cam_head_param.dtype)

                        with torch.no_grad():
                            pose_4x4 = _cam_head(cam_tokens_for_pose, patch_side, patch_side)
                        camera_pose = pose_4x4[:, :3, :].reshape(pose_4x4.shape[0], 12)
                
                if fusion_block_type in ['cross_attention', 'svf_baseline']:
                    # Build spatial KV tokens.
                    if fusion_block_type == 'svf_baseline':
                        # Ablation-1: Q=2D, KV=[camera, patch], output=2D+cross_attn.
                        if camera_tokens.shape[-1] != patch_tokens.shape[-1]:
                            # Camera branch features (512-dim from camera_decoder) cannot be directly
                            # concatenated with patch features (2048-dim from main decoder).
                            # svf_baseline requires a camera projection layer to reconcile dims.
                            # Fall back to patch-only KV until a camera projection is added to the model.
                            final_image_features = patch_tokens.to(self.dtype)
                        else:
                            final_image_features = torch.cat((camera_tokens, patch_tokens), dim=1).to(self.dtype)
                    else:
                        # Legacy cross_attention keeps runtime-selectable spatial token composition.
                        spatial_tower_select_feature = getattr(self.config, "spatial_tower_select_feature", "patch_tokens")
                        spatial_tower_select_feature_list = spatial_tower_select_feature.split(",")
                        selected_tokens = []
                        for spatial_tower_select_feature in spatial_tower_select_feature_list:
                            if spatial_tower_select_feature == "camera_tokens":
                                selected_tokens.append(camera_tokens)
                            elif spatial_tower_select_feature == "patch_tokens":
                                selected_tokens.append(patch_tokens)
                            elif spatial_tower_select_feature in ["all", "all_tokens"]:
                                selected_tokens = [camera_tokens, patch_tokens]
                            else:
                                raise ValueError(f"Unexpected spatial_tower_select_feature: {spatial_tower_select_feature}")
                        final_image_features = torch.cat(selected_tokens, dim=1).to(self.dtype)

                    image_features, attn_weights = self.get_model().get_fusion_block()(image_features, final_image_features)
                    image_features = self.get_model().mm_projector(image_features)

                elif fusion_block_type == 'svf_patch_cam_concat':
                    # Ablation-2: Q=2D, KV=patch only, then prepend projected camera tokens.
                    image_features, attn_weights = self.get_model().get_fusion_block()(
                        image_features,
                        patch_tokens.to(self.dtype),
                        camera_tokens.to(self.dtype),
                    )
                    image_features = self.get_model().mm_projector(image_features)

                elif fusion_block_type == 'svf_geometry_bridge':
                    # Ablation-3: camera->patch attention builds geometry-aware tokens, then 2D queries those tokens.
                    image_features, attn_weights = self.get_model().get_fusion_block()(
                        image_features,
                        camera_tokens.to(self.dtype),
                        patch_tokens.to(self.dtype),
                    )
                    image_features = self.get_model().mm_projector(image_features)
                
                elif fusion_block_type == 'svf_patch_only':
                    # Baseline: 2D cross-attends patch_tokens only.
                    image_features, attn_weights = self.get_model().get_fusion_block()(
                        image_features,
                        patch_tokens.to(self.dtype),
                    )
                    image_features = self.get_model().mm_projector(image_features)

                elif fusion_block_type == 'svf_cat_feat':
                    # Comparison 1: feature-dim concat [camera‖patch] as single KV stream.
                    image_features, attn_weights = self.get_model().get_fusion_block()(
                        image_features,
                        camera_tokens.to(self.dtype),
                        patch_tokens.to(self.dtype),
                    )
                    image_features = self.get_model().mm_projector(image_features)

                elif fusion_block_type == 'svf_pose_geometry_bridge':
                    # Comparison 2: camera tokens (from camera_decoder branch) query
                    # patch tokens to build geometry-aware tokens, then 2D queries them.
                    image_features, attn_weights = self.get_model().get_fusion_block()(
                        image_features,
                        camera_tokens.to(self.dtype),
                        patch_tokens.to(self.dtype),
                    )
                    image_features = self.get_model().mm_projector(image_features)

                elif fusion_block_type == 'svf_pose_geometry_bridge_reverse':
                    # Comparison 2b: patch tokens query camera tokens to build
                    # geometry-aware tokens, then 2D queries them.
                    image_features, attn_weights = self.get_model().get_fusion_block()(
                        image_features,
                        camera_tokens.to(self.dtype),
                        patch_tokens.to(self.dtype),
                    )
                    image_features = self.get_model().mm_projector(image_features)

                elif fusion_block_type == 'svf_pose_prepend':
                    # Comparison 3: 1 pose token (12-value camera matrix → Linear(12,d_clip))
                    # prepended to 2D-fused sequence. camera_pose = (F, 12).
                    if camera_pose is None:
                        raise RuntimeError(
                            "svf_pose_prepend: camera_pose not computed. "
                            "Check that the preextracted branch ran compute_camera_pose()."
                        )
                    image_features, attn_weights = self.get_model().get_fusion_block()(
                        image_features,
                        camera_pose.to(self.dtype),
                        patch_tokens.to(self.dtype),
                    )
                    image_features = self.get_model().mm_projector(image_features)

                elif fusion_block_type == 'svf_pose_geometry_bridge_reverse':
                    # Comparison 2b: patch tokens (Q=3D) attend to camera tokens (KV),
                    # then 2D tokens attend to those geometry-aware tokens.
                    image_features, attn_weights = self.get_model().get_fusion_block()(
                        image_features,
                        camera_tokens.to(self.dtype),
                        patch_tokens.to(self.dtype),
                    )
                    image_features = self.get_model().mm_projector(image_features)

                elif fusion_block_type == 'cross_attention_with_mlp':
                    image_features, attn_weights = self.get_model().get_fusion_block()(image_features, patch_tokens)
                    image_features = self.get_model().mm_projector(image_features)

                elif fusion_block_type == 'transformer':
                    spatial_tower_select_feature = getattr(self.config, "spatial_tower_select_feature", "patch_tokens")
                    if spatial_tower_select_feature in ["all", "all_tokens"]:
                        final_image_features = torch.cat((camera_tokens, patch_tokens), dim=1).to(self.dtype)
                        image_features = self.get_model().get_fusion_block()(image_features, final_image_features)
                        image_features = self.get_model().mm_projector(image_features)

                elif (fusion_block_type == 'mlp_after_clip_proj' 
                      or fusion_block_type == 'concat_mlp'
                      or fusion_block_type == 'concat_self_attention'):

                    image_features = self.get_model().mm_projector(image_features)
                    image_features = self.get_model().get_fusion_block()(image_features, patch_tokens)

                else:
                    raise ValueError(f"Unexpected fusion block type: {fusion_block_type}")

        elif self.get_model().get_spatial_tower() is None and self.get_model().get_fusion_block() is not None:
            assert point_maps is not None
            image_features = self.get_model().mm_projector(image_features)
            image_features = self.get_model().get_fusion_block()(image_features, point_maps[0]) # FIXME: point_maps is a list of tensors, each tensor is a point map for one image

        else:
            image_features = self.get_model().mm_projector(image_features)
        return image_features
    
    def encode_multimodals(self, videos_or_images, video_idx_in_batch, split_sizes=None):
        videos_or_images_features = self.get_model().get_vision_tower()(videos_or_images)
        per_videos_or_images_features = torch.split(videos_or_images_features, split_sizes, dim=0)  # tuple, (dim_1, 576, 4096)
        all_videos_or_images_features = []
        all_faster_video_features = []
        cur_mm_spatial_pool_stride = self.config.mm_spatial_pool_stride

        for idx, feat in enumerate(per_videos_or_images_features):
            
            feat = self.get_model().mm_projector(feat)
            faster_video_feature = 0
            slower_img_feat = 0
            if idx in video_idx_in_batch and cur_mm_spatial_pool_stride > 1:
                slower_img_feat = self.get_2dPool(feat,cur_mm_spatial_pool_stride)
                if self.config.add_faster_video:
                    cur_mm_spatial_pool_stride = cur_mm_spatial_pool_stride * 2
                    faster_video_feature = self.get_2dPool(feat,cur_mm_spatial_pool_stride)
            if slower_img_feat != 0:
                all_videos_or_images_features.append(slower_img_feat)
            else:
                all_videos_or_images_features.append(feat)
            all_faster_video_features.append(faster_video_feature)
        return all_videos_or_images_features,all_faster_video_features

    def add_token_per_grid(self, image_feature):
        prefix_tokens, image_feature, resize_h = self._split_prefix_tokens_for_square_grid(image_feature)
        num_frames = image_feature.shape[0]
        feature_dim = image_feature.shape[-1]

        image_feature = image_feature.view(num_frames, 1, resize_h, resize_h, -1)
        image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
        image_feature = image_feature.flatten(1, 2).flatten(2, 3)
        image_feature = torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
        if getattr(self.config, "add_faster_video", False):
            # import pdb; pdb.set_trace()
            # (3584, 832, 14) -> (3584, 64, 13, 14)
            image_feature = image_feature.view(feature_dim, num_frames,resize_h, -1)
            #  (3584, 64, 13, 14) -> (64, 13, 14, 3584)
            image_feature = image_feature.permute(1, 2, 3, 0).contiguous()
            # (64, 13, 14, 3584) -> (64, 13*14, 3584)
            image_feature = image_feature.flatten(1, 2)
            if prefix_tokens is not None:
                image_feature = torch.cat((prefix_tokens, image_feature), dim=1)
            # import pdb; pdb.set_trace()
            return image_feature
        # import pdb; pdb.set_trace()
        image_feature = image_feature.view(feature_dim, num_frames, resize_h, -1)
        image_feature = image_feature.permute(1, 2, 3, 0).contiguous()
        image_feature = image_feature.flatten(1, 2)
        if prefix_tokens is not None:
            image_feature = torch.cat((prefix_tokens, image_feature), dim=1)
        image_feature = image_feature.flatten(0, 1)
        return image_feature

    def add_token_per_frame(self, image_feature):
        image_feature = image_feature.permute(2, 0, 1).contiguous()
        image_feature =  torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
        image_feature = image_feature.permute(1, 2, 0).contiguous()
        return image_feature

    def prepare_inputs_labels_for_multimodal(self, input_ids, position_ids, attention_mask, past_key_values, labels, images, spatial_features=None, point_maps=None, modalities=["image"], image_sizes=None, eomt_images=None, eomt_meta=None):
        vision_tower = self.get_vision_tower()
        # rank_print(modalities)
        self._last_vision_grid_features = None
        self._last_eomt_pool_visual_features = None
        self._last_eomt_pool_frame_meta = None
        self._last_eomt_pool_debug = None
        self._last_eomt_pool_enabled_mask = None
        self._last_eomt_pool_skip_reasons = None
        self._last_eomt_outputs = None
        self._last_eomt_pooled_outputs = None
        self._last_eomt_object_block_outputs = None
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        # EoMT side branch: extract soft masks without injecting into main features
        eomt_extractor = self.get_model().get_eomt_extractor()
        if eomt_extractor is not None and eomt_images is not None:
            try:
                # eomt_images: list[B] of list[frames] of PIL images
                # eomt_meta:   list[B] of list[frames] of per-frame dicts
                # Flatten both into a single list aligned frame-by-frame
                all_frames = []
                all_frame_metas = []
                if eomt_meta is None:
                    eomt_meta = [[] for _ in eomt_images]
                for sample_frames, sample_frame_metas in zip(eomt_images, eomt_meta):
                    all_frames.extend(sample_frames)
                    all_frame_metas.extend(sample_frame_metas)
                eomt_outputs = eomt_extractor(all_frames, all_frame_metas)
                # Phase 1: store as debug-accessible attribute, do NOT inject into features
                self._last_eomt_outputs = eomt_outputs
            except Exception as e:
                rank0_print(f"EoMT side branch error: {e}")
                self._last_eomt_outputs = None

        if isinstance(modalities, str):
            modalities = [modalities]

        video_idx_in_batch = []

        # import pdb; pdb.set_trace()
        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]

            for _ in range(len(modalities)):
                if modalities[_] == "video":
                    video_idx_in_batch.append(_)

            images_list = []
            for image in images:
                if image.ndim == 4:
                    images_list.append(image)
                else:
                    images_list.append(image.unsqueeze(0))

            concat_images = torch.cat([image for image in images_list], dim=0)
            split_sizes = [image.shape[0] for image in images_list]
            encoded_image_features = self.encode_images(concat_images, spatial_features, point_maps)
            # if self.get_model().get_spatial_tower() is not None:
            #     if spatial_features is None:
            #         camera_tokens, patch_tokens = self.encode_spatial_features(concat_images)
            #     else:
            #         camera_tokens, patch_tokens = spatial_features[0]["camera_tokens"], spatial_features[0]["patch_tokens"]
            #     # fuse with spatial features
            #     spatial_tower_select_feature = getattr(self.config, "spatial_tower_select_feature", "patch_tokens")
            #     spatial_tower_select_feature_list = spatial_tower_select_feature.split(",")
            #     final_image_features = []
            #     for spatial_tower_select_feature in spatial_tower_select_feature_list:
            #         if spatial_tower_select_feature == "camera_tokens":
            #             final_image_features.append(camera_tokens)
            #         elif spatial_tower_select_feature == "patch_tokens":
            #             final_image_features.append(patch_tokens)
            #     final_image_features = torch.cat(final_image_features, dim=1)
            #     encoded_image_features = self.get_model().get_fusion_block()(encoded_image_features, final_image_features)
            # image_features,all_faster_video_features = self.encode_multimodals(concat_images, video_idx_in_batch, split_sizes)

            # This is a list, each element is [num_images, patch * patch, dim]
            # rank_print(f"Concat images : {concat_images.shape}")
            encoded_image_features = torch.split(encoded_image_features, split_sizes)
            image_features = []
            for idx, image_feat in enumerate(encoded_image_features):
                if idx in video_idx_in_batch:
                    image_features.append(self.get_2dPool(image_feat))
                else:
                    image_features.append(image_feat)

            # Build frame-aligned side cache for EoMT pooling before any merge/flatten
            # logic changes the per-frame structure.
            mm_patch_merge_type = getattr(self.config, "mm_patch_merge_type", "flat")
            image_aspect_ratio = getattr(self.config, "image_aspect_ratio", "square")
            self._build_eomt_pool_side_cache(
                per_sample_visual_features=image_features,
                split_sizes=split_sizes,
                video_idx_in_batch=video_idx_in_batch,
                modalities=modalities,
                eomt_images=eomt_images,
                eomt_meta=eomt_meta,
                image_aspect_ratio=image_aspect_ratio,
                mm_patch_merge_type=mm_patch_merge_type,
                image_sizes=image_sizes,
            )
            
            # if self.get_model().get_spatial_tower() is not None:
            #     if spatial_features is not None:
            #         encoded_camera_tokens = spatial_features[0]["camera_tokens"] ## FIXME: spatial_features is a list of dicts, each dict contains camera_tokens and patch_tokens
            #         encoded_patch_tokens = spatial_features[0]["patch_tokens"]
            #         # fusion block
            #         encoded_camera_tokens, encoded_patch_tokens = self.get_model().get_fusion_block()(encoded_camera_tokens, encoded_patch_tokens)
            #     else:
            #         encoded_camera_tokens, encoded_patch_tokens = self.encode_spatial_features(concat_images)
            #     camera_tokens = torch.split(encoded_camera_tokens, split_sizes)
            #     encoded_patch_tokens = torch.split(encoded_patch_tokens, split_sizes)
            #     # split and merge
            #     patch_tokens = []
            #     # pool patch tokens
            #     for idx, patch_token in enumerate(encoded_patch_tokens):
            #         if idx in video_idx_in_batch:
            #             patch_tokens.append(self.get_2dPool(patch_token))
            #         else:
            #             patch_tokens.append(patch_token)

            # image_features = self.encode_multimodals(concat_images, video_idx_in_batch, split_sizes)
            # rank_print(f"Encoded image feats : {[x.shape for x in image_features]}")
            # image_features = torch.split(image_features, split_sizes, dim=0)
            mm_patch_merge_type = getattr(self.config, "mm_patch_merge_type", "flat")
            image_aspect_ratio = getattr(self.config, "image_aspect_ratio", "square")
            mm_newline_position = getattr(self.config, "mm_newline_position", "one_token")

            if mm_patch_merge_type == "flat":
                image_features = [x.flatten(0, 1) for x in image_features]

            elif mm_patch_merge_type.startswith("spatial"):
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    # FIXME: now assume the image is square, and split to 2x2 patches
                    # num_patches = h * w, where h = w = sqrt(num_patches)
                    # currently image_feature is a tensor of shape (4, num_patches, hidden_size)
                    # we want to first unflatten it to (2, 2, h, w, hidden_size)
                    # rank0_print("At least we are reaching here")
                    # import pdb; pdb.set_trace()
                    if image_idx in video_idx_in_batch:  # video operations
                        # rank0_print("Video")
                        if mm_newline_position == "grid":
                            # Grid-wise
                            image_feature = self.add_token_per_grid(image_feature)
                            if getattr(self.config, "add_faster_video", False):
                                faster_video_feature = self.add_token_per_grid(all_faster_video_features[image_idx])
                                # Add a token for each frame
                                concat_slow_fater_token = []
                                # import pdb; pdb.set_trace()
                                for _ in range(image_feature.shape[0]):
                                    if _ % self.config.faster_token_stride == 0:
                                        concat_slow_fater_token.append(torch.cat((image_feature[_], self.model.faster_token[None].to(image_feature.device)), dim=0))
                                    else:
                                        concat_slow_fater_token.append(torch.cat((faster_video_feature[_], self.model.faster_token[None].to(image_feature.device)), dim=0))
                                # import pdb; pdb.set_trace()
                                image_feature = torch.cat(concat_slow_fater_token)

                                # print("!!!!!!!!!!!!")
                        
                            new_image_features.append(image_feature)
                        elif mm_newline_position == "frame":
                            # Frame-wise
                            image_feature = self.add_token_per_frame(image_feature)

                            new_image_features.append(image_feature.flatten(0, 1))
                            
                        elif mm_newline_position == "one_token":
                            # one-token
                            image_feature = image_feature.flatten(0, 1)
                            if 'unpad' in mm_patch_merge_type:
                                image_feature = torch.cat((
                                    image_feature,
                                    self.model.image_newline[None].to(image_feature.device)
                                ), dim=0)
                            new_image_features.append(image_feature)      
                        elif mm_newline_position == "no_token":
                            new_image_features.append(image_feature.flatten(0, 1))
                        else:
                            raise ValueError(f"Unexpected mm_newline_position: {mm_newline_position}")
                    elif image_feature.shape[0] > 1:  # multi patches and multi images operations
                        # rank0_print("Single-images")
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]

                        if "anyres_max" in image_aspect_ratio:
                            matched_anyres_max_num_patches = re.match(r"anyres_max_(\d+)", image_aspect_ratio)
                            if matched_anyres_max_num_patches:
                                max_num_patches = int(matched_anyres_max_num_patches.group(1))

                        if image_aspect_ratio == "anyres" or "anyres_max" in image_aspect_ratio:
                            if hasattr(self.get_vision_tower(), "image_size"):
                                vision_tower_image_size = self.get_vision_tower().image_size
                            else:
                                raise ValueError("vision_tower_image_size is not found in the vision tower.")
                            try:
                                num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, vision_tower_image_size)
                            except Exception as e:
                                rank0_print(f"Error: {e}")
                                num_patch_width, num_patch_height = 2, 2
                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                        else:
                            image_feature = image_feature.view(2, 2, height, width, -1)

                        if "maxpool2x2" in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = nn.functional.max_pool2d(image_feature, 2)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        elif "unpad" in mm_patch_merge_type and "anyres_max" in image_aspect_ratio and matched_anyres_max_num_patches:
                            unit = image_feature.shape[2]
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            c, h, w = image_feature.shape
                            times = math.sqrt(h * w / (max_num_patches * unit**2))
                            if times > 1.1:
                                image_feature = image_feature[None]
                                image_feature = nn.functional.interpolate(image_feature, [int(h // times), int(w // times)], mode="bilinear")[0]
                            image_feature = torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        elif "unpad" in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            image_feature = torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        else:
                            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                        if "nobase" in mm_patch_merge_type:
                            pass
                        else:
                            image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                        new_image_features.append(image_feature)
                    else:  # single image operations
                        # For single images, apply the same grid-wise newline logic
                        # as used for video frames to maintain consistency.
                        image_feature = self.add_token_per_grid(image_feature)
                        new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            image_features = self.encode_images(images)

            # Keep frame alignment explicit in non-list mode: split batched (B, T, D)
            # into true per-sample entries [(1, T, D), ...] before side-cache building.
            if torch.is_tensor(image_features) and image_features.ndim == 3:
                per_sample_visual_features = [
                    image_features[sample_idx : sample_idx + 1]
                    for sample_idx in range(image_features.shape[0])
                ]
                per_sample_split_sizes = [1 for _ in range(image_features.shape[0])]
            elif torch.is_tensor(image_features) and image_features.ndim == 2:
                per_sample_visual_features = [image_features.unsqueeze(0)]
                per_sample_split_sizes = [1]
            else:
                per_sample_visual_features = [image_features]
                if torch.is_tensor(image_features) and image_features.ndim > 0:
                    per_sample_split_sizes = [image_features.shape[0]]
                else:
                    per_sample_split_sizes = None

            self._build_eomt_pool_side_cache(
                per_sample_visual_features=per_sample_visual_features,
                split_sizes=per_sample_split_sizes,
                video_idx_in_batch=video_idx_in_batch,
                modalities=modalities,
                eomt_images=eomt_images,
                eomt_meta=eomt_meta,
                image_aspect_ratio=getattr(self.config, "image_aspect_ratio", "square"),
                mm_patch_merge_type=getattr(self.config, "mm_patch_merge_type", "flat"),
                image_sizes=image_sizes,
            )

        # Side-output path: consume EoMT masks to pool vision grid tokens.
        # This does not modify the main image_features path used by the model.
        self._last_eomt_pooled_outputs = self._compute_eomt_mask_pooled_side_output()
        self._last_eomt_object_block_outputs = self._compute_eomt_object_block_side_output()

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(self.config, "mm_use_im_start_end", False):
            raise NotImplementedError
        # rank_print(f"Total images : {len(image_features)}")

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        object_block_outputs = getattr(self, "_last_eomt_object_block_outputs", None)
        # rank_print("Inserting Images embedding")
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            # rank0_print(num_images)
            if num_images == 0:
                # Handle cases with no image tokens if necessary.
                # Original code appends empty features, adapt if needed.
                cur_image_features = image_features[cur_image_idx]
                # Also get corresponding spatial features
                # if self.get_model().get_spatial_tower() is not None:
                #     cur_camera_tokens = camera_tokens[cur_image_idx]
                #     cur_patch_tokens = patch_tokens[cur_image_idx]

                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)

                # Concatenate text embeds, visual embeds [0:0], and spatial embeds [0:0]?
                # This part of original code seems odd (using [0:0]), clarify its purpose.
                # Assuming you want to append actual features if available, otherwise skip.
                embeds_to_concat = [cur_input_embeds_1]
                # if cur_image_features is not None and cur_image_features.numel() > 0:
                #     embeds_to_concat.append(cur_image_features[0:0]) # Original behavior

                cur_input_embeds = torch.cat(embeds_to_concat, dim=0)

                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1 # Increment even if no image token? Check original logic intent.
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1 : image_token_indices[i + 1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                # Append text embeddings and labels
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])

                # If this segment was followed by an image token, insert features
                if i < num_images:
                    try:
                        # Get the visual/fused features for the current image
                        cur_image_features = image_features[cur_image_idx]
                        # Get the spatial features for the current image
                        # if self.get_model().get_spatial_tower() is not None:
                        #     cur_camera_tokens = camera_tokens[cur_image_idx]
                        #     cur_patch_tokens = patch_tokens[cur_image_idx]
                    except IndexError:
                         # Fallback logic from original code
                        cur_image_features = image_features[cur_image_idx - 1]
                        # if self.get_model().get_spatial_tower() is not None:
                        #     cur_camera_tokens = camera_tokens[cur_image_idx - 1]
                        #     cur_patch_tokens = patch_tokens[cur_image_idx - 1]

                    cur_object_sample_idx = batch_idx
                    cur_image_idx += 1

                    # Prepare combined features (visual + spatial)
                    features_to_insert = []
                    if cur_image_features is not None and cur_image_features.shape[0] > 0:
                        merged_visual_features = cur_image_features
                        if isinstance(object_block_outputs, dict) and bool(object_block_outputs.get("enabled", False)):
                            selected_tokens_by_sample = object_block_outputs.get("selected_tokens_by_sample", {})
                            cur_object_tokens = selected_tokens_by_sample.get(cur_object_sample_idx, None)
                            if (
                                torch.is_tensor(cur_object_tokens)
                                and cur_object_tokens.ndim == 2
                                and cur_object_tokens.shape[0] > 0
                            ):
                                object_block_tokens, compose_reason = self._compose_eomt_object_block_tokens(
                                    object_tokens=cur_object_tokens,
                                    visual_tokens=cur_image_features,
                                    object_block_outputs=object_block_outputs,
                                )
                                if (
                                    torch.is_tensor(object_block_tokens)
                                    and object_block_tokens.ndim == 2
                                    and object_block_tokens.shape[0] > 0
                                ):
                                    appender = self._get_eomt_object_block_appender()
                                    merged_visual_features, used_object_block, append_reason = appender.append(
                                        visual_tokens=cur_image_features,
                                        object_tokens=object_block_tokens,
                                        position=str(object_block_outputs.get("append_position", "after_visual")),
                                    )
                                    if used_object_block:
                                        object_block_outputs["used_object_block"] = True
                                        object_block_outputs["fallback_reason"] = None
                                    elif object_block_outputs.get("fallback_reason", None) is None:
                                        object_block_outputs["fallback_reason"] = append_reason
                                elif object_block_outputs.get("fallback_reason", None) is None:
                                    object_block_outputs["fallback_reason"] = compose_reason
                        features_to_insert.append(merged_visual_features)
                    # spatial_tower_select_feature = getattr(self.config, "spatial_tower_select_feature", None)
                    # if self.get_model().get_spatial_tower() is not None and spatial_tower_select_feature is not None:
                    #     spatial_feature_flags = spatial_tower_select_feature.split(",")
                        
                    #     if cur_camera_tokens is not None and cur_camera_tokens.shape[0] > 0 and "camera_tokens" in spatial_feature_flags:
                    #         features_to_insert.append(cur_camera_tokens.flatten(0, 1))
                    #     if cur_patch_tokens is not None and cur_patch_tokens.shape[0] > 0 and "patch_tokens" in spatial_feature_flags:
                    #         features_to_insert.append(cur_patch_tokens.flatten(0, 1))

                    if features_to_insert:
                        combined_features = torch.cat(features_to_insert, dim=0)
                        cur_new_input_embeds.append(combined_features)
                        # Add IGNORE_INDEX labels for the entire combined feature length
                        cur_new_labels.append(torch.full((combined_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, "tokenizer_model_max_length", None)
        # rank_print("Finishing Inserting")

        new_input_embeds = [x[:tokenizer_model_max_length] for x, modality in zip(new_input_embeds, modalities)]
        new_labels = [x[:tokenizer_model_max_length] for x, modality in zip(new_labels, modalities)]
        # TODO: Hard code for control loss spike
        # if tokenizer_model_max_length is not None:
        #     new_input_embeds = [x[:4096] if modality != "video" else x[:tokenizer_model_max_length] for x, modality in zip(new_input_embeds, modalities)]
        #     new_labels = [x[:4096] if modality != "video" else x[:tokenizer_model_max_length] for x, modality in zip(new_labels, modalities)]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)
        # rank0_print("Prepare pos id")

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, "tokenizer_padding_side", "right") == "left":
                new_input_embeds_padded.append(torch.cat((torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device), cur_new_embed), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((cur_new_embed, torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)
        # rank0_print("tokenizer padding")

        if isinstance(self._last_eomt_object_block_outputs, dict):
            if (
                self._last_eomt_object_block_outputs.get("enabled", False)
                and not self._last_eomt_object_block_outputs.get("used_object_block", False)
                and self._last_eomt_object_block_outputs.get("fallback_reason", None) is None
            ):
                self._last_eomt_object_block_outputs["fallback_reason"] = "no_object_block_appended"

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None
        if getattr(self.config, "use_pos_skipping", False) and self.training:
            position_ids = torch.arange(new_input_embeds.size(1), device=new_input_embeds.device).unsqueeze(0).to(new_input_embeds.device)
            split_position = random.randint(0, new_input_embeds.size(1))
            left_add = random.randint(0, self.config.pos_skipping_range)
            right_add = random.randint(left_add, self.config.pos_skipping_range)
            position_ids[:, :split_position] += left_add
            position_ids[:, split_position:] += right_add
        # import pdb; pdb.set_trace()
        # rank0_print("Finish preparing")
        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location="cpu")
                embed_tokens_weight = mm_projector_weights["model.embed_tokens.weight"]
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
