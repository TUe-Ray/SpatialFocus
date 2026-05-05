import torch

from llava.model.llava_arch import pool_point_map_to_tokens
from llava.model.multimodal_fusion_block.builder import CrossAttentionFusion, CrossAttentionFusion3DRoPE, GeometryRoPE


def copy_baseline_weights(baseline, rope_block):
    rope_block.clip_norm.load_state_dict(baseline.clip_norm.state_dict())
    rope_block.spatial_encoder_norm.load_state_dict(baseline.spatial_encoder_norm.state_dict())
    rope_block.clip_query_proj.load_state_dict(baseline.clip_query_proj.state_dict())
    rope_block.spatial_encoder_key_proj.load_state_dict(baseline.spatial_encoder_key_proj.state_dict())
    rope_block.spatial_encoder_value_proj.load_state_dict(baseline.spatial_encoder_value_proj.state_dict())
    rope_block.out_proj.load_state_dict(baseline.out_proj.state_dict())
    rope_block.out_norm.load_state_dict(baseline.out_norm.state_dict())

    in_weight = baseline.cross_attention.in_proj_weight
    in_bias = baseline.cross_attention.in_proj_bias
    d_attn = rope_block.d_attn
    rope_block.attn_query_proj.weight.data.copy_(in_weight[:d_attn])
    rope_block.attn_key_proj.weight.data.copy_(in_weight[d_attn:2 * d_attn])
    rope_block.attn_value_proj.weight.data.copy_(in_weight[2 * d_attn:])
    rope_block.attn_query_proj.bias.data.copy_(in_bias[:d_attn])
    rope_block.attn_key_proj.bias.data.copy_(in_bias[d_attn:2 * d_attn])
    rope_block.attn_value_proj.bias.data.copy_(in_bias[2 * d_attn:])
    rope_block.attn_out_proj.load_state_dict(baseline.cross_attention.out_proj.state_dict())


def check_mode(mode, group_split):
    torch.manual_seed(7)
    baseline = CrossAttentionFusion(
        d_clip=54,
        d_spatial_encoder=40,
        d_attn=54,
        num_heads=3,
    )
    block = CrossAttentionFusion3DRoPE(
        d_clip=54,
        d_spatial_encoder=40,
        d_attn=54,
        num_heads=3,
        rope_mode=mode,
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

    assert block.rope_gate_q.item() == 0.0
    assert block.rope_gate_k.item() == 0.0

    out_a, attn = block(clip_features, spatial_features, pos_clip, pos_spatial)
    out_b, _ = block(clip_features, spatial_features, pos_clip + 10.0, pos_spatial - 10.0)
    out_base, _ = baseline(clip_features, spatial_features)

    assert out_a.shape == clip_features.shape
    assert attn.shape == (2, 16, 25)
    assert torch.allclose(out_a, out_b, atol=0.0, rtol=0.0)
    assert torch.allclose(out_a, out_base, atol=1e-5, rtol=1e-5)
    assert block.last_geometry_rope_stats["q_rope_minus_q"]["std"] > 0.0
    assert block.last_geometry_rope_stats["group_split"] == group_split


def expect_value_error(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError("Expected ValueError")


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

    expect_value_error(lambda: GeometryRoPE(head_dim=18, mode="depth", group_split="2,1,2"))
    expect_value_error(lambda: GeometryRoPE(head_dim=18, mode="xyz", group_split="1"))
    expect_value_error(lambda: GeometryRoPE(head_dim=18, mode="spherical", group_split="1,1"))
    expect_value_error(lambda: GeometryRoPE(head_dim=18, mode="xyz", group_split="1,0,1"))
    expect_value_error(lambda: GeometryRoPE(head_dim=18, mode="spherical", group_split="-1,1,1"))
    print("Geometry-RoPE fusion sanity checks passed.")


if __name__ == "__main__":
    main()
