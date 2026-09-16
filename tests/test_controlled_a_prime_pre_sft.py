import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
PROBING_ROOT = REPO_ROOT / "scripts" / "probing"
if str(PROBING_ROOT) not in sys.path:
    sys.path.insert(0, str(PROBING_ROOT))

from llava.model.controlled_fusion_pre_sft import (
    CONTROLLED_A_PRIME_PRE_SFT_SPEC,
    CONTROLLED_FUSION_PRE_SFT_SPECS,
    controlled_fusion_spec_for_variant,
)
from scripts.diagnose_layerwise_spatial_hidden_scan import (
    install_pre_sft_fusion,
    module_state_sha256,
    seeded_fusion_initialization,
)
from scripts.probing.extract_depth_probe_features import assert_first_controlled_a_prime_runtime
from scripts.probing.prepare_controlled_a_prime_pre_sft import fresh_a_prime_state


def test_a_prime_is_non_c1_and_does_not_expand_frozen_controlled_roster():
    spec = CONTROLLED_A_PRIME_PRE_SFT_SPEC
    assert tuple(CONTROLLED_FUSION_PRE_SFT_SPECS) == ("B", "C", "D", "E", "H")
    assert controlled_fusion_spec_for_variant("CONTROLLED_A_PRIME") is spec
    assert spec.identifier == "A_prime"
    assert spec.architecture == "pre_projector_cross_attention_patch_only"
    assert spec.cut3r_source_layers == (12,)
    assert spec.llm_injection_layers == ()


def test_canonical_module_state_hash_tracks_seed_and_preserves_rng():
    torch.manual_seed(99)
    expected_after = torch.rand(3)
    torch.manual_seed(99)
    with seeded_fusion_initialization(42):
        first = nn.Linear(7, 5)
    observed_after = torch.rand(3)
    with seeded_fusion_initialization(42):
        second = nn.Linear(7, 5)
    with seeded_fusion_initialization(43):
        third = nn.Linear(7, 5)
    assert torch.equal(observed_after, expected_after)
    assert module_state_sha256(first) == module_state_sha256(second)
    assert module_state_sha256(first) != module_state_sha256(third)


def test_official_a_prime_constructor_hash_is_deterministic():
    first_hash, first_count = fresh_a_prime_state()
    second_hash, second_count = fresh_a_prime_state()
    assert first_hash == second_hash
    assert len(first_hash) == 64
    assert first_count == second_count
    assert first_count > 0


class _InstallBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.fusion_block = None
        self.spatial_tower = None

    def get_fusion_block(self):
        return self.fusion_block

    def get_cut3r_spatialstack_merger(self):
        return None

    def get_geometry_aware_projection(self):
        return None


class _InstallHarness(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(mm_hidden_size=36, hidden_size=48)
        self.base = _InstallBase()

    @property
    def dtype(self):
        return torch.float32

    def get_model(self):
        return self.base


def test_a_prime_install_is_fresh_non_c1_patch_only_fusion():
    model = _InstallHarness()
    metadata = install_pre_sft_fusion(model, "controlled_a_prime", fusion_init_seed=42)
    assert model.config.fusion_block == "pre_projector_cross_attention_patch_only"
    assert model.config.use_cut3r_spatialstack is False
    assert model.config.spatial_tower_select_feature == "patch_tokens"
    assert model.config.pre_projector_cross_attention_source_layer == 12
    assert metadata["controlled_fusion_id"] == "A_prime"
    assert metadata["c1_enabled"] is False
    assert metadata["fresh_fusion_state_sha256"] == metadata["runtime_fusion_state_sha256"]
    assert all(not parameter.requires_grad for parameter in model.base.fusion_block.parameters())


class _RuntimeHarness:
    def __init__(self, metrics):
        fusion = SimpleNamespace(
            cross_attention=SimpleNamespace(num_heads=18, dropout=0.0),
            dropout=SimpleNamespace(p=0.1),
            c1_enabled=torch.tensor(False),
        )
        self.base = SimpleNamespace(
            _last_pre_projector_cross_attention_patch_only_metrics=metrics,
            get_fusion_block=lambda: fusion,
        )
        self.config = SimpleNamespace(use_cut3r_spatialstack=False)

    def get_model(self):
        return self.base


def test_a_prime_runtime_assertion_checks_exact_unpooled_shapes():
    metrics = {
        "fusion_type": "pre_projector_cross_attention_patch_only",
        "fusion_stage": "pre_mm_projector",
        "cut3r_source_layer": 12,
        "geometry_tokens": "patch_only",
        "camera_token_count": 0,
        "visual_query_shape": [32, 729, 1152],
        "raw_patch_shape": [32, 729, 768],
        "geometry_kv_shape": [32, 729, 768],
        "fused_shape": [32, 729, 1152],
        "mm_projector_input_shape": [32, 729, 1152],
        "mm_projector_output_shape": [32, 729, 3584],
        "finite": True,
        "cut3r_detached": True,
    }
    result = assert_first_controlled_a_prime_runtime(
        model=_RuntimeHarness(metrics),
        hidden_states=[None] * 29,
        metadata={"visual_frame_ids": torch.arange(32)},
        selected_frames=[0, 31],
        model_forward_inputs={
            "spatial_features": True,
            "point_maps": False,
            "geometry_spatial_features": False,
        },
    )
    assert result["assessment"] == "PASS"
    assert result["architecture"] == "A_prime"
    assert result["cut3r_detached"] is True
    assert result["spatialstack_disabled"] is True


def test_a_prime_runtime_assertion_rejects_camera_tokens():
    metrics = {
        "fusion_type": "pre_projector_cross_attention_patch_only",
        "fusion_stage": "pre_mm_projector",
        "cut3r_source_layer": 12,
        "geometry_tokens": "patch_only",
        "camera_token_count": 1,
        "visual_query_shape": [32, 729, 1152],
        "raw_patch_shape": [32, 729, 768],
        "geometry_kv_shape": [32, 729, 768],
        "fused_shape": [32, 729, 1152],
        "mm_projector_input_shape": [32, 729, 1152],
        "mm_projector_output_shape": [32, 729, 3584],
        "finite": True,
        "cut3r_detached": True,
    }
    try:
        assert_first_controlled_a_prime_runtime(
            model=_RuntimeHarness(metrics),
            hidden_states=[None] * 29,
            metadata={"visual_frame_ids": torch.arange(32)},
            selected_frames=[0, 31],
            model_forward_inputs={
                "spatial_features": True,
                "point_maps": False,
                "geometry_spatial_features": False,
            },
        )
    except RuntimeError as exc:
        assert "camera_token_count" in str(exc)
    else:
        raise AssertionError("A-prime runtime assertion accepted a camera token")
