"""Immutable definitions for completing the seven legacy pre-SFT depth probes.

These candidates already have historical partial depth-probe artifacts.  This
module records the exact fresh pre-SFT construction and C1 inputs required for
a new, complete 15-representation run.  It deliberately excludes the two
loss-only depth-supervision variants: they do not alter the forward
representation and must instead be handled by a separate forward-equivalence
and gradient-proxy analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from scripts.probing.probe_layer_policy import COMMON_PROBE_LAYERS, PRE_SFT_PRE_LLM_FEATURES


@dataclass(frozen=True)
class CompletionCandidate:
    identifier: str
    label: str
    display_name: str
    fusion_variant: str
    c1_artifact: Path
    c1_architecture: str
    spatial_features_subdir: str
    spatialstack_cut3r_layers: str = ""
    spatialstack_llm_layers: str = ""
    geometry_c1_activation: Path | None = None
    geometry_architecture: str | None = None
    uses_eomt_selective_gate: bool = False


REPO_OUTPUTS = Path("/home/shaoruei/probe_outputs")
VLM3R_C1 = REPO_OUTPUTS / "c1_vlm3r_v1/official/vlm3r.json"

LEGACY_PARTIAL_CANDIDATES: tuple[CompletionCandidate, ...] = (
    CompletionCandidate(
        "SS012",
        "c1_spatialstack_add",
        "Hierarchical Add @ L0/1/2 (SpatialStack)",
        "c1_ss_add",
        REPO_OUTPUTS / "c1_additive_v1/official/spatialstack_add.json",
        "spatialstack_add",
        "6:spatial_features_dec_6,9:spatial_features_dec_9,12:spatial_features",
        "6,9,12",
        "0,1,2",
    ),
    CompletionCandidate(
        "SS123",
        "c1_spatialstack_add_123",
        "Hierarchical Add @ L1/2/3",
        "c1_ss_add",
        REPO_OUTPUTS / "c1_ss_add_123/official/spatialstack_add.json",
        "spatialstack_add",
        "6:spatial_features_dec_6,9:spatial_features_dec_9,12:spatial_features",
        "6,9,12",
        "1,2,3",
    ),
    CompletionCandidate(
        "SS036",
        "c1_spatialstack_add_036",
        "Hierarchical Add @ L0/3/6",
        "c1_ss_add",
        REPO_OUTPUTS / "c1_ss_add_036/official/spatialstack_add.json",
        "spatialstack_add",
        "6:spatial_features_dec_6,9:spatial_features_dec_9,12:spatial_features",
        "6,9,12",
        "0,3,6",
    ),
    CompletionCandidate(
        "SSCROSS",
        "c1_spatialstack_cross_attn_v1",
        "Hierarchical Cross-Attn @ L0/1/2",
        "c1_ss_cross_attn_v1",
        REPO_OUTPUTS / "c1_ss_cross_attn_v1/official/spatialstack_cross_attn_v1.json",
        "spatialstack_cross_attn_v1",
        "6:spatial_features_dec_6,9:spatial_features_dec_9,12:spatial_features",
        "6,9,12",
        "0,1,2",
    ),
    CompletionCandidate(
        "VLM3R",
        "c1_vlm3r",
        "Pre-Projector Cross-Attn (VLM3R)",
        "c1_vlm3r",
        VLM3R_C1,
        "vlm3r",
        "spatial_features",
    ),
    CompletionCandidate(
        "GEOROPE",
        "c1_geo_rope_fusion",
        "Geometry-RoPE Fusion",
        "c1_geo_rope_fusion",
        VLM3R_C1,
        "vlm3r",
        "spatial_features",
        geometry_c1_activation=REPO_OUTPUTS / "c1_geometry_pre_sft_v1/geo_rope_fusion/c1_activation.json",
        geometry_architecture="geo_rope_fusion",
    ),
    CompletionCandidate(
        "SELECTIVE",
        "c1_vlm3r_eomt_selective",
        "Selective Geometry Fusion (EoMT K/V gate)",
        "c1_vlm3r",
        VLM3R_C1,
        "vlm3r",
        "spatial_features",
        uses_eomt_selective_gate=True,
    ),
)

BY_IDENTIFIER = {candidate.identifier: candidate for candidate in LEGACY_PARTIAL_CANDIDATES}
FULL_FEATURE_LEVELS = tuple(
    [*PRE_SFT_PRE_LLM_FEATURES, *(f"layer_{layer}" for layer in COMMON_PROBE_LAYERS)]
)
EXPECTED_VALIDATION_TOKENS = 75_656
