# Controlled A-prime non-C1 pre-SFT depth probe

This is a separate representation probe for `A_prime`; it does not modify the
frozen five-candidate C1 proxy study or the controlled B/C/D/E/H C1 artifacts.

## Architecture and initialization

`controlled_a_prime` installs
`pre_projector_cross_attention_patch_only` once on the plain pretrained base
VLM. SigLIP patch features are Q, and detached CUT3R decoder-12 patch tokens
are K/V. Camera tokens are excluded. The inherited `CrossAttentionFusion`
uses pre-LN, 1152-dimensional attention with 18 heads, default PyTorch MHA
dropout 0.0, an output projection and LayerNorm, residual addition to SigLIP,
and final dropout 0.1. Fusion occurs before the pretrained `mm_projector`;
SpatialStack is disabled.

This candidate is **not C1 calibrated**. The fusion block uses ordinary
PyTorch initialization under the isolated constructor seed 42. The canonical
SHA-256 over sorted state-dict tensor names, dtypes, shapes, and raw bytes is
frozen in the experiment manifest and repeated in extraction provenance. No
post-SFT adapter, trained fusion/projector state, LoRA, or optimizer update is
permitted for the candidate model. The downstream linear depth probes do use
their own optimizer; that optimizer never owns or updates candidate weights.

The first forward must attest these raw runtime shapes before feature pooling:

```text
SigLIP Q                 [32, 729, 1152]
CUT3R decoder-12 patch KV [32, 729, 768]
Residual fused visual    [32, 729, 1152]
mm_projector output      [32, 729, 3584]
```

It also proves source layer 12, patch-only K/V, zero camera tokens, finite
output, matching Q/KV frame-token axes, and fusion immediately entering the
projector.

## Formal policy

Both the fresh plain-base comparator and A-prime extract all 15 required
representations in one forward:

```text
siglip_output fusion_output projected_features
layer_0 layer_1 layer_2 layer_3 layer_6 layer_9 layer_12
layer_15 layer_18 layer_21 layer_24 layer_27
```

The formal split is locked to SHA-256
`d478cb684958dfc25066821ec83d5216469577c9e282e33bdf87d3c88b200d8e`:
1,199 ScanNet videos, 2,398 selected target frames, and 75,656 validation
tokens per fitted probe. Model forward still uses all 32 RGB frames.

## Local execution

On `mps-edu-06`, after committing the reviewed code so provenance can prove a
clean worktree:

```bash
scripts/probing/run_controlled_a_prime_pre_sft_local.sh preflight
scripts/probing/run_controlled_a_prime_pre_sft_local.sh smoke
scripts/probing/run_controlled_a_prime_pre_sft_local.sh full
```

The smoke always runs the plain pre-SFT baseline first, then A-prime, using one
train and one validation video. Its verified marker gates the full sweep. The
full run again executes baseline first, preserves probe metrics and extraction
provenance, and recycles regeneratable feature tensors after each candidate.

Default namespaces are:

- cache: `/home/shaoruei/probe_cache/controlled_a_prime_pre_sft_v1`
- durable outputs: `/home/shaoruei/probe_outputs/controlled_a_prime_pre_sft_v1`
- logs: `logs/controlled_a_prime_pre_sft_v1`
