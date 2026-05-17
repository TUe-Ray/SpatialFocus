import torch

from llava.model.llava_arch import pool_point_map_to_tokens
from llava.model.multimodal_fusion_block.builder import CrossAttentionFusion, GeoRoPEFusionCrossAttention, GeoRoPEFusionRotary


def copy_baseline_weights(baseline, geo_rope_fusion_block):
    geo_rope_fusion_block.clip_norm.load_state_dict(baseline.clip_norm.state_dict())
    geo_rope_fusion_block.spatial_encoder_norm.load_state_dict(baseline.spatial_encoder_norm.state_dict())
    geo_rope_fusion_block.clip_query_proj.load_state_dict(baseline.clip_query_proj.state_dict())
    geo_rope_fusion_block.spatial_encoder_key_proj.load_state_dict(baseline.spatial_encoder_key_proj.state_dict())
    geo_rope_fusion_block.spatial_encoder_value_proj.load_state_dict(baseline.spatial_encoder_value_proj.state_dict())
    geo_rope_fusion_block.out_proj.load_state_dict(baseline.out_proj.state_dict())
    geo_rope_fusion_block.out_norm.load_state_dict(baseline.out_norm.state_dict())

    in_weight = baseline.cross_attention.in_proj_weight
    in_bias = baseline.cross_attention.in_proj_bias
    d_attn = geo_rope_fusion_block.d_attn
    geo_rope_fusion_block.attn_query_proj.weight.data.copy_(in_weight[:d_attn])
    geo_rope_fusion_block.attn_key_proj.weight.data.copy_(in_weight[d_attn:2 * d_attn])
    geo_rope_fusion_block.attn_value_proj.weight.data.copy_(in_weight[2 * d_attn:])
    geo_rope_fusion_block.attn_query_proj.bias.data.copy_(in_bias[:d_attn])
    geo_rope_fusion_block.attn_key_proj.bias.data.copy_(in_bias[d_attn:2 * d_attn])
    geo_rope_fusion_block.attn_value_proj.bias.data.copy_(in_bias[2 * d_attn:])
    geo_rope_fusion_block.attn_out_proj.load_state_dict(baseline.cross_attention.out_proj.state_dict())


def check_mode(mode, group_split):
    torch.manual_seed(7)
    baseline = CrossAttentionFusion(
        d_clip=54,
        d_spatial_encoder=40,
        d_attn=54,
        num_heads=3,
    )
    block = GeoRoPEFusionCrossAttention(
        d_clip=54,
        d_spatial_encoder=40,
        d_attn=54,
        num_heads=3,
        geo_rope_fusion_mode=mode,
        group_split=group_split,
        log_stats=True,
        dropout_rate=0.0,
    )
    copy_baseline_weights(baseline, block)
    baseline.eval()
    block.eval()

    clip_features = torch.randn(2, 16, 54)
    spatial_features = torch.randn(2, 25, 40)
    pos_clip = torch.randn(2, 16, 3)
    pos_spatial = torch.randn(2, 25, 3)

    assert block.geo_rope_fusion_gate_q.item() == 0.0
    assert block.geo_rope_fusion_gate_k.item() == 0.0

    out_a, attn = block(clip_features, spatial_features, pos_clip, pos_spatial)
    out_b, _ = block(clip_features, spatial_features, pos_clip + 10.0, pos_spatial - 10.0)
    out_base, _ = baseline(clip_features, spatial_features)

    assert out_a.shape == clip_features.shape
    assert attn.shape == (2, 16, 25)
    assert torch.allclose(out_a, out_b, atol=0.0, rtol=0.0)
    assert torch.allclose(out_a, out_base, atol=1e-5, rtol=1e-5)
    assert block.last_geo_rope_fusion_stats["q_geo_rope_fusion_minus_q"]["std"] > 0.0
    assert block.last_geo_rope_fusion_stats["group_split"] == group_split


def expect_value_error(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError("Expected ValueError")


def check_default_depth_extension_matches_original():
    torch.manual_seed(11)
    x = torch.randn(2, 3, 5, 36)
    pos = torch.randn(2, 5, 3) * 4.0

    original = GeoRoPEFusionRotary(
        head_dim=36,
        mode="spherical",
        max_depth=10.0,
        group_split="2,1,2",
    )
    explicit_default = GeoRoPEFusionRotary(
        head_dim=36,
        mode="spherical",
        max_depth=10.0,
        train_max_depth=10.0,
        eval_max_depth=None,
        ntk_scaling=False,
        group_split="2,1,2",
    )
    original.eval()
    explicit_default.eval()

    out_original = original(x, pos)
    out_default = explicit_default(x, pos)
    assert torch.allclose(out_original, out_default, atol=0.0, rtol=0.0)


def check_eval_max_depth_ignored_in_training_mode():
    torch.manual_seed(12)
    x = torch.randn(1, 2, 4, 36)
    pos = torch.tensor(
        [[[0.0, 0.0, 3.0], [1.0, 0.5, 8.0], [2.0, -1.0, 18.0], [0.0, 3.0, 30.0]]],
        dtype=torch.float32,
    )

    baseline = GeoRoPEFusionRotary(
        head_dim=36,
        mode="spherical",
        max_depth=10.0,
        train_max_depth=10.0,
        eval_max_depth=None,
        group_split="2,1,2",
    )
    extended = GeoRoPEFusionRotary(
        head_dim=36,
        mode="spherical",
        max_depth=10.0,
        train_max_depth=10.0,
        eval_max_depth=30.0,
        ntk_scaling=True,
        group_split="2,1,2",
    )
    baseline.train()
    extended.train()

    out_baseline = baseline(x, pos)
    out_extended = extended(x, pos)
    assert torch.allclose(out_baseline, out_extended, atol=0.0, rtol=0.0)


def check_eval_max_depth_changes_far_eval_behavior():
    torch.manual_seed(13)
    x = torch.randn(1, 2, 4, 36)
    pos = torch.tensor(
        [[[0.0, 0.0, 4.0], [0.0, 0.0, 9.0], [0.0, 0.0, 15.0], [0.0, 0.0, 24.0]]],
        dtype=torch.float32,
    )

    clamped = GeoRoPEFusionRotary(
        head_dim=36,
        mode="spherical",
        max_depth=10.0,
        train_max_depth=10.0,
        eval_max_depth=None,
        group_split="2,1,2",
    )
    extended = GeoRoPEFusionRotary(
        head_dim=36,
        mode="spherical",
        max_depth=10.0,
        train_max_depth=10.0,
        eval_max_depth=30.0,
        ntk_scaling=False,
        group_split="2,1,2",
    )
    clamped.eval()
    extended.eval()

    out_clamped = clamped(x, pos)
    out_extended = extended(x, pos)
    assert not torch.allclose(out_clamped, out_extended, atol=1e-7, rtol=1e-7)


def check_ntk_toggle_changes_eval_rotary_frequencies():
    torch.manual_seed(14)
    x = torch.randn(1, 2, 4, 36)
    pos = torch.tensor(
        [[[0.0, 0.0, 12.0], [1.0, 0.0, 20.0], [0.0, -1.0, 30.0], [2.0, 1.0, 35.0]]],
        dtype=torch.float32,
    )

    no_ntk = GeoRoPEFusionRotary(
        head_dim=36,
        mode="spherical",
        max_depth=10.0,
        train_max_depth=10.0,
        eval_max_depth=30.0,
        ntk_scaling=False,
        group_split="2,1,2",
    )
    with_ntk = GeoRoPEFusionRotary(
        head_dim=36,
        mode="spherical",
        max_depth=10.0,
        train_max_depth=10.0,
        eval_max_depth=30.0,
        ntk_scaling=True,
        group_split="2,1,2",
    )
    no_ntk.eval()
    with_ntk.eval()

    assert no_ntk.depth_group_index == 2
    out_no_ntk = no_ntk(x, pos)
    out_with_ntk = with_ntk(x, pos)
    assert not torch.allclose(out_no_ntk, out_with_ntk, atol=1e-7, rtol=1e-7)


def main():
    point_maps = torch.randn(2, 8, 8, 3)
    token_pos = pool_point_map_to_tokens(point_maps, 16)
    assert token_pos.shape == (2, 16, 3)

    point_maps_chw = point_maps.permute(0, 3, 1, 2)
    token_pos_chw = pool_point_map_to_tokens(point_maps_chw, 16)
    assert torch.allclose(token_pos, token_pos_chw)

    valid_cases = (
        ("depth", "1"),
        ("xyz", "1,1,1"),
        ("xyz", "2,1,2"),
        ("spherical", "1,1,1"),
        ("spherical", "2,1,2"),
        ("spherical", "3,1,3"),
    )
    for mode, group_split in valid_cases:
        check_mode(mode, group_split)

    check_default_depth_extension_matches_original()
    check_eval_max_depth_ignored_in_training_mode()
    check_eval_max_depth_changes_far_eval_behavior()
    check_ntk_toggle_changes_eval_rotary_frequencies()

    expect_value_error(lambda: GeoRoPEFusionRotary(head_dim=18, mode="depth", group_split="2,1,2"))
    expect_value_error(lambda: GeoRoPEFusionRotary(head_dim=18, mode="xyz", group_split="1"))
    expect_value_error(lambda: GeoRoPEFusionRotary(head_dim=18, mode="spherical", group_split="1,1"))
    expect_value_error(lambda: GeoRoPEFusionRotary(head_dim=18, mode="xyz", group_split="1,0,1"))
    expect_value_error(lambda: GeoRoPEFusionRotary(head_dim=18, mode="spherical", group_split="-1,1,1"))
    expect_value_error(lambda: GeoRoPEFusionRotary(head_dim=18, mode="spherical", eval_max_depth=-1.0))
    print("GeoRoPE Fusion sanity checks passed.")


if __name__ == "__main__":
    main()
