# Snellius VGGT L23-only cache

This workflow converts the audited 2,693-file three-layer VGGT cache into a
separate cache containing only aggregated layer 23. It never runs VGGT and
never modifies the source cache.

## Paths

- Source: `/scratch-shared/geusdd/shaoruei/VLM3R/spatial_features/vggt`
- Restartable staging: `/scratch-shared/geusdd/shaoruei/VLM3R/spatial_features/vggt_l23_staging`
- Published cache: `/scratch-shared/geusdd/shaoruei/VLM3R/spatial_features/vggt_l23`
- Inventory/logs/reports: `/scratch-shared/geusdd/shaoruei/VLM3R/spatial_features/vggt_l23_artifacts`

The canonical union is derived from the three VLM-3R training JSON manifests
and the union of `test_pruned.parquet` and `test_debiased.parquet`. Publication
requires exactly 2,693 records and the audited per-dataset/scope counts.

## Run

```bash
bash snellius/vggt_l23/prepare.sh
sbatch snellius/vggt_l23/smoke.sbatch
# Submit only after the smoke completes successfully:
sbatch snellius/vggt_l23/full.sbatch
```

The full job writes each sidecar through a temporary file, reloads it, checks
exact tensor and frame-index equality with the source, computes the output
SHA-256, and then atomically renames it. Completed valid staged outputs are
safe to reuse after a restart. The source tree is protected by filename,
size, and mtime fingerprints before and after the run; no full source-sidecar
SHA sweep is performed.

The required first-stage deliverable is:

`vggt_l23_artifacts/cached_vggt_l23_inventory.json`

It contains 2,693 canonical records with source/output paths, output SHA-256,
and a canonical SHA-256 of the unchanged frame indices. The Candidate A
loader manifest is intentionally deferred until its authoritative existing
manifest path is available; no `frame_positions` field is inferred.
