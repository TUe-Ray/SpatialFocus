# Legacy pre-SFT depth-probe completion

This campaign completes the current 15-representation depth-probe policy for
seven candidates whose retained historical results used only a partial layer
set. It is separate from the existing five-candidate zero-cost-proxy
conclusions.

The candidates are Hierarchical Add at L0/1/2, L1/2/3, and L0/3/6;
Hierarchical Cross-Attn at L0/1/2; pre-projector Cross-Attn (C1 VLM3R);
Geometry-RoPE Fusion; and Selective Geometry Fusion (the C1 VLM3R EoMT K/V
gate).

The two depth-supervision entries are deliberately excluded. They are
loss-only variants, not distinct forward representations. A depth probe may
reuse the matching parent representation only after numerical forward
equivalence has been verified; their loss/gradient proxies remain separate.

On `mps-edu-06`, prepare and gate the campaign with:

```bash
scripts/probing/run_legacy_pre_sft_completion_local.sh preflight
scripts/probing/run_legacy_pre_sft_completion_local.sh smoke
```

Only after the baseline-first smoke marker passes may the seven sequential
full extractions and two-GPU probe fits run:

```bash
scripts/probing/run_legacy_pre_sft_completion_local.sh full
```

The wrapper locks the immutable C1 artifact hashes, the authoritative 1,199
ScanNet split, full 32-frame RGB forward inputs, all 15 feature levels, and
the relevant GeoRoPE or EoMT sidecar identities. It saves compact results and
provenance before recycling only the regenerated feature tensors under its own
cache root.

Current asset audit:

- Geometry-RoPE has 1,199 full ScanNet CUT3R point-map sidecars at
  `/mnt/DATA_SSD/shaoruei/probing_data/cut3r_point_maps_32_v1`, using
  `point_maps_ref`, plus a frozen no-training C1 activation.
- Selective fusion has a PASS EoMT consumer grid with 1,199 scenes and 3,597
  payload files.
- Object Token has the same complete EoMT consumer grid and can be prepared
  as a separate full-policy campaign. Its validation still records actual-VLM
  forward parity as pending, so it must pass an architecture-specific smoke.
- Visual GeoRoPE has the same 1,199 full point-map sidecars and a frozen
  no-training C1 activation. It lacks a complete 15-feature pre-SFT run, not
  an input asset.
