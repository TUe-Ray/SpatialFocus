from pathlib import Path

import torch

from scripts.extraction.convert_vggt_l23 import (
    build_output_payload,
    frame_idx_sha256,
    validate_output_payload,
    validate_source_payload,
)


def _source_payload(shape=(2, 3, 4)):
    frames, _, features = shape
    return {
        "frames": {
            "aggregated_tokens": {
                layer: torch.arange(frames * shape[1] * features, dtype=torch.bfloat16).reshape(shape)
                for layer in ("11", "17", "23")
            },
            "frame_idx": torch.tensor([0, 9], dtype=torch.int64),
        },
        "meta": {
            "num_frames": frames,
            "input_size": 518,
            "model_image_hw": (518, 518),
            "patch_size": 14,
            "patch_start_idx": 5,
            "feature_dim": features,
            "intermediate_layer_idx": [4, 11, 17, 23],
            "token_dtype": "bfloat16",
            "schema": "vggt_aggregated_tokens_v1",
            "source_video": "/source/example.mp4",
            "vggt_weights_path": "/weights/model.safetensors",
        },
    }


def test_l23_payload_preserves_tensor_frame_idx_and_metadata():
    shape = (2, 3, 4)
    source = _source_payload(shape)
    validate_source_payload(source, Path("source.pt"), expected_shape=shape)
    output = build_output_payload(source)
    validate_output_payload(output, Path("output.pt"), source, expected_shape=shape)

    assert set(output["frames"]["aggregated_tokens"]) == {"23"}
    assert output["frames"]["aggregated_tokens"]["23"] is source["frames"]["aggregated_tokens"]["23"]
    assert output["frames"]["frame_idx"] is source["frames"]["frame_idx"]
    assert output["meta"]["intermediate_layer_idx"] == [23]
    assert {k: v for k, v in output["meta"].items() if k != "intermediate_layer_idx"} == {
        k: v for k, v in source["meta"].items() if k != "intermediate_layer_idx"
    }


def test_frame_idx_hash_is_stable_and_order_sensitive():
    first = torch.tensor([0, 5, 9], dtype=torch.int64)
    same = torch.tensor([0, 5, 9], dtype=torch.int64)
    reordered = torch.tensor([9, 5, 0], dtype=torch.int64)
    assert frame_idx_sha256(first) == frame_idx_sha256(same)
    assert frame_idx_sha256(first) != frame_idx_sha256(reordered)


def test_output_rejects_extra_layer():
    shape = (2, 3, 4)
    source = _source_payload(shape)
    output = build_output_payload(source)
    output["frames"]["aggregated_tokens"]["11"] = source["frames"]["aggregated_tokens"]["11"]
    try:
        validate_output_payload(output, Path("output.pt"), source, expected_shape=shape)
    except ValueError as exc:
        assert "output layer keys" in str(exc)
    else:
        raise AssertionError("extra output layer was accepted")
