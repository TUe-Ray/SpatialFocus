from types import SimpleNamespace

import torch
import torch.nn as nn

from llava.model.llava_arch import LlavaMetaForCausalLM
from llava.model.controlled_fusion_pre_sft import (
    A_PRIME_PRE_SFT_SPEC,
    controlled_fusion_spec,
    controlled_fusion_spec_for_variant,
)
from llava.model.multimodal_fusion_block.builder import (
    CrossAttentionFusion,
    PreProjectorAddFusion,
    PreProjectorCrossAttentionPatchOnlyFusion,
    align_patch_tokens_to_visual_grid,
    build_multimodal_fusion_block,
)


class TinyVisionTower(nn.Module):
    def forward(self, images, return_raw_features=False):
        features = torch.ones(images.shape[0], 4, 4, dtype=images.dtype)
        return (features, features) if return_raw_features else features


class TinyCut3RSpatialTower(nn.Module):
    is_loaded = True


class RecordingProjector(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = None

    def forward(self, features):
        self.input = features.detach().clone()
        return torch.cat([features, features[..., :2]], dim=-1)


class TinyBase(nn.Module):
    def __init__(self, fusion_type="pre_projector_add"):
        super().__init__()
        self.config = SimpleNamespace(
            spatial_tower="cut3r",
            spatial_tower_preextracted_only=True,
            fusion_block=fusion_type,
            pre_projector_add_source_layer=12,
            pre_projector_cross_attention_source_layer=12,
            cut3r_spatialstack_feature_key="cut3r_dec_layers",
            zero_spatial_features=False,
        )
        self.vision_tower = TinyVisionTower()
        self.spatial_tower = TinyCut3RSpatialTower()
        if fusion_type == "pre_projector_add":
            self.fusion_block = PreProjectorAddFusion(4, 3, source_layer=12, zero_init=False)
        else:
            self.fusion_block = PreProjectorCrossAttentionPatchOnlyFusion(
                4, 3, 4, num_heads=2, source_layer=12
            )
        self.mm_projector = RecordingProjector()

    def get_vision_tower(self):
        return self.vision_tower

    def get_spatial_tower(self):
        return self.spatial_tower

    def get_fusion_block(self):
        return self.fusion_block


class EncodeHarness:
    encode_images = LlavaMetaForCausalLM.encode_images

    def __init__(self, fusion_type="pre_projector_add"):
        self._base = TinyBase(fusion_type=fusion_type)
        self.config = self._base.config

    def get_model(self):
        return self._base


def test_pre_projector_add_aligns_dec12_then_calls_mm_projector():
    torch.manual_seed(5)
    harness = EncodeHarness()
    dec12 = torch.randn(1, 9, 3)
    sidecar = {"cut3r_dec_layers": {"12": {"patch_tokens": dec12}}}
    output = harness.encode_images(torch.zeros(1, 3, 4, 4), spatial_features=[sidecar])

    assert output.shape == (1, 4, 6)
    assert torch.isfinite(output).all()
    assert harness._base.mm_projector.input.shape == (1, 4, 4)
    assert not torch.equal(harness._base.mm_projector.input, torch.ones(1, 4, 4))
    metrics = harness._base._last_pre_projector_add_metrics
    assert metrics["fusion_stage"] == "pre_mm_projector"
    assert metrics["cut3r_source_layer"] == 12
    assert metrics["raw_spatial_shape"] == [1, 9, 3]
    assert metrics["aligned_spatial_shape"] == [1, 4, 3]
    assert metrics["mm_projector_input_shape"] == [1, 4, 4]
    assert metrics["mm_projector_output_shape"] == [1, 4, 6]


def test_pre_projector_add_zero_init_is_identity_in_vision_space():
    fusion = PreProjectorAddFusion(4, 3, source_layer=12, zero_init=True)
    clip = torch.randn(2, 4, 4)
    spatial = torch.randn(2, 9, 3)
    assert torch.equal(fusion(clip, spatial), clip)


def test_pre_projector_add_c1_scalars_preserve_native_default_and_emit_diagnostics():
    fusion = PreProjectorAddFusion(4, 3, source_layer=12, zero_init=False)
    clip = torch.randn(2, 4, 4)
    spatial = torch.randn(2, 9, 3)
    native = fusion(clip, spatial)
    fusion.set_c1_state(
        enabled=True,
        pre_gelu_scale=0.75,
        residual_gain=0.0,
        collect_diagnostics=True,
    )
    assert torch.equal(fusion(clip, spatial), clip)
    assert not torch.equal(native, clip)
    assert set(fusion._c1_last_diagnostics) == {
        "clip",
        "z_pre_raw",
        "z_pre",
        "delta_raw",
        "delta",
    }
    assert fusion._c1_last_diagnostics["delta"]["sum_sq"] == 0.0


def test_pre_projector_cross_attention_uses_dec12_patch_only_before_mm_projector():
    torch.manual_seed(7)
    harness = EncodeHarness("pre_projector_cross_attention_patch_only")
    dec12 = torch.randn(1, 9, 3)
    sidecar = {
        "camera_tokens": torch.full((1, 2, 99), float("nan")),
        "cut3r_dec_layers": {"12": {"patch_tokens": dec12}},
    }
    output = harness.encode_images(torch.zeros(1, 3, 4, 4), spatial_features=[sidecar])

    assert output.shape == (1, 4, 6)
    assert torch.isfinite(output).all()
    assert harness._base.mm_projector.input.shape == (1, 4, 4)
    metrics = harness._base._last_pre_projector_cross_attention_patch_only_metrics
    assert metrics["fusion_stage"] == "pre_mm_projector"
    assert metrics["cut3r_source_layer"] == 12
    assert metrics["geometry_tokens"] == "patch_only"
    assert metrics["camera_token_count"] == 0
    assert metrics["visual_query_shape"] == [1, 4, 4]
    assert metrics["raw_patch_shape"] == [1, 9, 3]
    assert metrics["geometry_kv_shape"] == [1, 4, 3]
    assert metrics["mm_projector_input_shape"] == [1, 4, 4]
    assert metrics["mm_projector_output_shape"] == [1, 4, 6]


def test_pre_projector_cross_attention_preserves_original_module_math_after_alignment():
    torch.manual_seed(11)
    controlled = PreProjectorCrossAttentionPatchOnlyFusion(
        4, 3, 4, num_heads=2, source_layer=12
    ).eval()
    original = CrossAttentionFusion(4, 3, 4, num_heads=2).eval()
    original.load_state_dict(controlled.state_dict(), strict=True)
    clip = torch.randn(2, 4, 4)
    patches = torch.randn(2, 9, 3)
    aligned = align_patch_tokens_to_visual_grid(patches, target_tokens=4)

    actual, actual_weights = controlled(clip, patches)
    expected, expected_weights = original(clip, aligned)
    assert torch.equal(actual, expected)
    assert torch.equal(actual_weights, expected_weights)


def test_builder_constructs_explicit_patch_only_pre_projector_cross_attention():
    config = SimpleNamespace(
        fusion_block="pre_projector_cross_attention_patch_only",
        mm_hidden_size=36,
        hidden_size=48,
        spatial_feature_dim=12,
        pre_projector_cross_attention_source_layer=12,
    )
    fusion = build_multimodal_fusion_block(config)
    assert isinstance(fusion, PreProjectorCrossAttentionPatchOnlyFusion)
    assert fusion.cross_attention.num_heads == 18
    assert fusion.source_layer == 12


def test_a_prime_pre_sft_spec_is_default_initialized_patch_only_pre_projector_control():
    spec = controlled_fusion_spec("A_prime")
    assert spec is A_PRIME_PRE_SFT_SPEC
    assert controlled_fusion_spec_for_variant("controlled_a_prime") is spec
    assert spec.architecture == "pre_projector_cross_attention_patch_only"
    assert spec.cut3r_source_layers == (12,)
    assert spec.llm_injection_layers == ()
    assert spec.fusion_type == "pre_projector_cross_attention_patch_only"
