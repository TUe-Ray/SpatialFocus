"""CPU-only checks for the controlled-fusion post-SFT probe runner."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
PROBING_DIR = REPO_ROOT / "scripts" / "probing"
if str(PROBING_DIR) not in sys.path:
    sys.path.insert(0, str(PROBING_DIR))

from local_depth_probe_cache import assert_baseline_or_zero_spatial_forward_contract  # noqa: E402


class _Model:
    def __init__(self, **config):
        self.config = SimpleNamespace(**config)


class ControlledFusionPostSftProbeTests(unittest.TestCase):
    def test_local_token_contract_accepts_controlled_pre_projector_add(self) -> None:
        model = _Model(
            spatial_tower="cut3r",
            fusion_block="pre_projector_add",
            pre_projector_add_source_layer=12,
            use_cut3r_spatialstack=False,
        )
        assert_baseline_or_zero_spatial_forward_contract(model)

    def test_local_token_contract_rejects_wrong_pre_projector_source(self) -> None:
        model = _Model(
            spatial_tower="cut3r",
            fusion_block="pre_projector_add",
            pre_projector_add_source_layer=9,
            use_cut3r_spatialstack=False,
        )
        with self.assertRaises(RuntimeError):
            assert_baseline_or_zero_spatial_forward_contract(model)

    def test_local_token_contract_accepts_a_prime_patch_only_cross_attention(self) -> None:
        model = _Model(
            spatial_tower="cut3r",
            fusion_block="pre_projector_cross_attention_patch_only",
            pre_projector_cross_attention_source_layer=12,
            use_cut3r_spatialstack=False,
        )
        assert_baseline_or_zero_spatial_forward_contract(model)

    def test_local_token_contract_rejects_a_prime_wrong_source(self) -> None:
        model = _Model(
            spatial_tower="cut3r",
            fusion_block="pre_projector_cross_attention_patch_only",
            pre_projector_cross_attention_source_layer=9,
            use_cut3r_spatialstack=False,
        )
        with self.assertRaises(RuntimeError):
            assert_baseline_or_zero_spatial_forward_contract(model)

    def test_runner_uses_full_policy_and_rolling_cache(self) -> None:
        runner = (PROBING_DIR / "run_controlled_fusion_post_sft_depth_probe_local.sh").read_text()
        self.assertIn('PRE_LLM_FEATURES="fusion_output,projected_features"', runner)
        self.assertIn('COMMON_PROBE_LAYER_LEVELS_CSV', runner)
        self.assertIn('CANDIDATES=(A_prime B C D E H)', runner)
        self.assertIn("A_prime:label) printf 'controlled_a_prime_post_sft'", runner)
        self.assertIn('SPATIAL_SUBDIR="12:spatial_features"', runner)
        self.assertIn('RECYCLE_FULL_CACHE="${RECYCLE_FULL_CACHE:-1}"', runner)
        self.assertIn('--assert-first-video --resume', runner)


if __name__ == "__main__":
    unittest.main()
