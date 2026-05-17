# GeoRoPE Depth-Range Probes

This diagnostic tests whether Room Size degradation is tied to depth-range clipping or radial-depth normalization in `GeoRoPEFusionRotary`.

## Config Flags

- `geo_rope_fusion_max_depth`: original GeoRoPE depth/radius clamp and normalization scale. Default is `10.0`.
- `geo_rope_fusion_train_max_depth`: training-time depth/radius normalization scale. If unset, defaults to `geo_rope_fusion_max_depth`.
- `geo_rope_fusion_eval_max_depth`: optional eval-time clamp. If unset, eval uses `geo_rope_fusion_train_max_depth`, preserving old behavior.
- `geo_rope_fusion_ntk_scaling`: optional eval-time NTK-style theta scaling for the depth-bearing rotary group.

Backward-compatible aliases are also accepted for older checkpoints and eval configs:

- `geometry_rope_max_depth`
- `geometry_rope_train_max_depth`
- `geometry_rope_eval_max_depth`
- `geometry_rope_ntk_scaling`

## Behavior

Training mode ignores `eval_max_depth` and keeps the original train-depth behavior:

```python
clamp_max_depth = train_max_depth
normalization_max_depth = train_max_depth
```

Eval mode uses the decoupled path only when `eval_max_depth` is set:

```python
clamp_max_depth = eval_max_depth
normalization_max_depth = train_max_depth
```

For spherical mode, this changes only the log-radius coordinate. NTK scaling is applied only to the depth-bearing group: group `0` for `depth`, group `2` for `xyz`, and group `2` for `spherical`.

## Scripts

- Probe 0: `python scripts/analysis/analyze_room_size_depth.py`
- Eval summary: `python scripts/analysis/summarize_vsibench_depth_probes.py --variant original <run>/vsibench.json 10 false`
- Slurm matrix launcher: `DRY_RUN=True bash submit_depth_range_probe_evals.sh`
