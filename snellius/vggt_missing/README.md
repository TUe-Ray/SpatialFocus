# Snellius VGGT missing-feature extraction

This directory reproduces the Leonardo VGGT extraction contract for the fixed
412-sidecar repair/evaluation scope. The only feature-definition change is the
handoff-authorized serialization selection `--layer-indices 11,17,23`.

## Fixed paths and identities

- Repository: `/gpfs/home4/geusdd/shuang/SpatialFocus`
- VGGT source: `third_party/VGGT` at
  `44b3afbd1869d8bde4894dd8ea1e293112dd5eba`
- Checkpoint snapshot:
  `/home/geusdd/.cache/huggingface/hub/models--facebook--VGGT-1B/snapshots/860abec7937da0a4c03c41d3c269c366e82abdf9`
- Checkpoint SHA-256:
  `f164acf60724910d8fe1578bb499d800850c7bb0948db7555c413f9fbe60467e`
- Training media: `/scratch-shared/geusdd/VLM3R/data/vlm3r/{dataset}/videos`
- Evaluation media: `/scratch-shared/geusdd/VLM3R/hf_cache/vsibench/{dataset}`
- Staging: `/scratch-shared/geusdd/shaoruei/VLM3R/spatial_features/vggt_missing_staging`
- Final cache: `/scratch-shared/geusdd/shaoruei/VLM3R/spatial_features/vggt`

The jobs set `HF_HUB_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`, and
`TRANSFORMERS_OFFLINE=1`. They pass the local snapshot directory directly to
the extractor, so model weights are never downloaded.

## Commands

Prepare and verify the exact symlink views:

```bash
bash snellius/vggt_missing/prepare.sh
```

Run the mandatory three-dataset smoke, inspect its report, and only then
submit the independent full jobs:

```bash
sbatch snellius/vggt_missing/smoke.sbatch
sbatch snellius/vggt_missing/extract_scannet.sbatch
sbatch snellius/vggt_missing/extract_scannetpp.sbatch
sbatch snellius/vggt_missing/extract_arkitscenes.sbatch
```

Each full job performs its own `torch.load` validation. A separate complete
412-file validation is available with:

```bash
bash snellius/vggt_missing/validate_staging.sh
```

Do not merge until the Leonardo rsync has populated the exact 2,281 expected
pre-existing sidecars. Record a final-tree fingerprint, wait at least ten
minutes, then merge only if it is unchanged:

```bash
/home/geusdd/.conda/envs/vlm3r-snellius/bin/python \
  scripts/extraction/merge_vggt_missing_staging.py --record-stability
# wait at least 10 minutes
/home/geusdd/.conda/envs/vlm3r-snellius/bin/python \
  scripts/extraction/merge_vggt_missing_staging.py --merge
```

The merge uses atomic, non-overwriting hard links because staging and final
are on the same filesystem. An unexpected corrupt `dd685be466.pt` is moved to
a timestamped quarantine name before replacement; any other invalid collision
aborts the merge.

Finally submit the CPU-only audit. It loads all 2,693 unique sidecars and
checks every stored frame index against the corresponding raw video:

```bash
sbatch snellius/vggt_missing/final_audit.sbatch
```

## Compatibility note

The transferred Leonardo payloads contain only tensor keys `11`, `17`, and
`23`, but retain the historical metadata value
`meta.intermediate_layer_idx=[4,11,17,23]`. Newly extracted sidecars correctly
record `[11,17,23]`. Validation permits the stale value only for transferred
reference/final files; it never permits a serialized Layer-4 tensor.
