# Agent Instructions

## Repository Rules

- Check `git status --short` before edits.
- Do not modify user-changed files unless the task requires it.
- Do not edit files under `third_party/` unless explicitly requested.
- Prefer small, experiment-specific wrapper scripts over broad training-script changes.

## HPC / Slurm

- Do not run GPU-heavy training directly on the login node.
- For smoke tests or jobs under 30 minutes, use:
  `#SBATCH --qos=boost_qos_dbg`
  with `#SBATCH --time=00:30:00` or less.
- When submitting smoke-test training or eval jobs, make the Slurm job name start
  with `SMOKE`.
- When submitting debug jobs that use `#SBATCH --qos=boost_qos_dbg`, make the
  Slurm job name start with `DBG`, unless the job is a smoke test that already
  starts with `SMOKE`.
- For official training or eval runs, remove debug/smoke prefixes such as
  `SMOKE` or `DBG` from the Slurm job name.
- Do not stop, cancel, kill, or otherwise interrupt any run unless its Slurm job
  name starts with `DBG` or `SMOKE`. For all non-`DBG`/`SMOKE` runs, ask the
  user for explicit permission before using commands such as `scancel`, `kill`,
  or `pkill`, or before changing scripts in a way that would stop the run.
- For normal runs, use:
  `#SBATCH --qos=normal`
- For interactive GPU debugging, use `srun`.
- Before submitting long jobs, show the intended command or script and expected output path.
- Write Slurm logs under `logs/` with experiment-specific names.
- Keep Slurm stdout and stderr separated. Prefer `#SBATCH --output=...%x_%j.out`
  and `#SBATCH --error=...%x_%j.err`; do not merge training stderr into stdout
  with `2>&1 | tee` unless the user explicitly asks for a combined log.

## Environment

- Use the existing project environment unless instructed otherwise.
- For geometry projection tests, use:
  `conda run -n vlm3r python tests/test_metric_grounded_geometry_projection.py`
- Do not install or upgrade dependencies without asking.

## Preextracted Spatial Sidecars

- Extraction utilities live under `scripts/extraction/`. Use that path in new
  Slurm wrappers, for example:
  `python scripts/extraction/extract_cut3r_point_maps.py`.
- The provenance scripts are mainly under `logs/chore/` and
  `logs/chore/archived/`. Treat those as the record of what was extracted.

### CUT3R Token Features

- Baseline CUT3R token sidecars use subdir `spatial_features`.
- Schema: `.pt` dict with `camera_tokens` and `patch_tokens`.
  `camera_tokens` is frame-level CUT3R camera token data; `patch_tokens` is the
  729-token spatial patch grid. Feature dim is 768.
- Verified locations:
  `/leonardo_scratch/fast/EUHPC_D32_006/data/vlm3r/{scannet,scannetpp,arkitscenes}/spatial_features`
  and the training mirror
  `/leonardo_work/EUHPC_D32_006/FAST/train_data/vlm3r/{scannet,scannetpp}/spatial_features`.
- Use for CUT3R cross-attention or feature-alignment baselines with:
  `SPATIAL_FEATURES_ROOT=/leonardo_scratch/fast/EUHPC_D32_006/data/vlm3r`
  or `/leonardo_work/EUHPC_D32_006/FAST/train_data/vlm3r`,
  `SPATIAL_FEATURES_SUBDIR=spatial_features`,
  `MODEL_SPATIAL_TOWER=cut3r`,
  `MODEL_SPATIAL_TOWER_SELECT_FEATURE=all_tokens`,
  `MODEL_SPATIAL_FEATURE_DIM=768`.

### CUT3R Point Maps

- CUT3R point-map sidecars use subdir `spatial_features_points`.
- Schema: `.pt` dict with `point_maps_ref`, `point_maps_cam`, `camera_pose`,
  and `metadata`.
- `point_maps_ref` / `pts3d_in_other_view` means CUT3R reference/anchor-frame
  coordinates. `point_maps_cam` / `pts3d_in_self_view` means per-frame camera
  coordinates. Keep the selected coordinate source identical between training
  and evaluation for a checkpoint.
- Verified train/large root:
  `/leonardo_scratch/large/userexternal/shuang00/VLM_3R_cut3r_pointmaps/{scannet,scannetpp,arkitscenes}/spatial_features_points`.
- Verified fast/eval-style root:
  `/leonardo_scratch/fast/EUHPC_D32_006/data/vlm3r/{scannet,scannetpp,arkitscenes}/spatial_features_points`.
- Use for CUT3R Metric-Grounded Geometry Projection or GeoRoPE point-map
  geometry with:
  `SPATIAL_FEATURES_ROOT=<one of the roots above>`,
  `SPATIAL_FEATURES_SUBDIR=spatial_features_points`,
  `MODEL_SPATIAL_ENCODER_TYPE=cut3r` or `DATA_SPATIAL_TOWER_TYPE=cut3r`.

### CUT3R Decoder-Layer Features

- Decoder-layer ablation sidecars are separate from the baseline `spatial_features`.
- Verified root:
  `/leonardo_work/EUHPC_D32_006/VLM_3R_cut3r_min2N4_features/{scannet,scannetpp,arkitscenes}/`.
- Available subdirs:
  `spatial_features_dec_m2` for CUT3R decoder layer `-2`;
  `spatial_features_dec_m4` for CUT3R decoder layer `-4`.
- Schema matches CUT3R token sidecars: `camera_tokens`, `patch_tokens`,
  `metadata`; feature dim is 768.
- Use through `train_cut3r_layer_ablation.sh` with `CUT3R_LAYER=-2` or
  `CUT3R_LAYER=-4`, or directly set:
  `SPATIAL_FEATURES_ROOT=/leonardo_work/EUHPC_D32_006/VLM_3R_cut3r_min2N4_features`
  and `SPATIAL_FEATURES_SUBDIR=spatial_features_dec_m2` or
  `spatial_features_dec_m4`.

### Pi3X Decoded Features

- Current Pi3X training sidecars are decoded-feature sidecars, not pre-sliced
  camera-token sidecars.
- Verified root:
  `/leonardo_work/EUHPC_D32_006/VLM_3R_pi3x_features/{scannet,scannetpp,arkitscenes}/*.pt`.
- Schema: `.pt` dict with `frames.decoded_features`, `frames.frame_idx`, and
  `meta.decoded_pos_template`, `meta.patch_start_idx`, `meta.num_frames`.
  `decoded_features` has shape `[F, T, 2048]`.
- Camera tokens must be computed at runtime from
  `pi3.camera_decoder(decoded_features, xpos=decoded_pos)`; do not use legacy
  flat `camera_tokens` Pi3X payloads.
- Use with:
  `SPATIAL_FEATURES_ROOT=/leonardo_work/EUHPC_D32_006/VLM_3R_pi3x_features`,
  `SPATIAL_FEATURES_SUBDIR=.`,
  `DATA_SPATIAL_TOWER_TYPE=pi3x`,
  `MODEL_SPATIAL_TOWER=pi3x`,
  `MODEL_SPATIAL_FEATURE_DIM=2048`.
- `train_geo_rope_fusion_cut3r_pi3x_pos.sh` keeps CUT3R as fusion/KV features
  while using Pi3X decoded features as the geometry provider via
  `GEOMETRY_SPATIAL_FEATURES_ROOT=/leonardo_work/EUHPC_D32_006/VLM_3R_pi3x_features`
  and `GEOMETRY_SPATIAL_FEATURES_SUBDIR=.`.

### Pi3X Point Maps

- Pi3X point-map sidecars are world-space point maps decoded from the Pi3X
  decoded-feature root.
- Verified train/large root:
  `/leonardo_scratch/large/userexternal/shuang00/VLM_3R_pi3x_pointmaps/{scannet,scannetpp,arkitscenes}/*.pt`.
- Verified VSI-Bench eval root:
  `/leonardo_work/EUHPC_D32_006/VLM_3R_pi3x_vsibench_eval_pointmaps/{scannet,scannetpp,arkitscenes}/*.pt`.
- Schema: `.pt` dict with `point_map` `[F,518,518,3]`, `camera_pose`
  `[F,4,4]`, `frame_idx`, and `meta`. `meta.coordinate_frame` is `world`;
  schema is `pi3x_world_point_map_v1`.
- If consuming these directly, use root `.../VLM_3R_pi3x_pointmaps` and
  subdir `.` because files live directly under each dataset directory.

### VGGT Features And Diagnostics

- Current VGGT feature sidecars are aggregated-token sidecars, not depth-only
  sidecars.
- Verified roots:
  `/leonardo_work/EUHPC_D32_006/VLM_3R_vggt_features/{scannet,scannetpp,arkitscenes}/*.pt`
  and
  `/leonardo_scratch/large/userexternal/shuang00/VLM_3R_vggt_features/{scannet,scannetpp,arkitscenes}/*.pt`.
- Schema: `.pt` dict with `frames.aggregated_tokens`, `frames.frame_idx`, and
  `meta`. `meta.schema` is `vggt_aggregated_tokens_v1`; feature dim is 2048;
  extracted VGGT intermediate layers are `[4, 11, 17, 23]`.
- Use with:
  `SPATIAL_FEATURES_ROOT=/leonardo_work/EUHPC_D32_006/VLM_3R_vggt_features`
  or `/leonardo_scratch/large/userexternal/shuang00/VLM_3R_vggt_features`,
  `SPATIAL_FEATURES_SUBDIR=.`,
  `MODEL_SPATIAL_ENCODER_TYPE=vggt`.
- `scripts/extraction/export_vggt_point_cloud.py` is a diagnostic/export tool:
  it writes a PLY and manifest from an image folder, and only writes
  `depth_map` / `depth_conf` tensors when run with `--save-pt`.

### Spatial Rank Head / P_geo

- `scripts/extraction/extract_spatial_rank_head.py` does not produce dataset
  sidecars. It extracts `spatial_rank_head.*` weights from a trained checkpoint
  into a small state dict, often called `p_geo.bin`.
- Use:
  `python scripts/extraction/extract_spatial_rank_head.py --checkpoint <ckpt> --output <p_geo.bin>`.

## Training / Evaluation Scripts

- Do not modify `train_vsi.sh` unless explicitly requested.
- Prefer creating or editing dedicated wrapper scripts for new experiments.
- Keep train/eval wrapper names descriptive, for example:
  `train_<feature>_<backbone>.sh`
  `eval_<feature>_<benchmark>.sh`

## Geometry / RoPE Design

- Preserve the Metric-Grounded Geometry Projection invariant:
  Q/K/V come from 2D visual tokens.
  Geometry only rotates Q/K through Geometry-RoPE.
  Geometry is not used as K/V and is not concatenated into LLM tokens.
- Keep GeoRoPE Fusion and Metric-Grounded Geometry Projection conceptually separate.
- GeoRoPE point-map coordinates must be train/eval consistent. If training uses
  `point_maps_ref` / `pts3d_in_other_view` (CUT3R reference/anchor-frame
  coordinates), evaluation must use the same keys. If training uses
  `point_maps_cam` / `pts3d_in_self_view` (per-frame camera coordinates),
  evaluation must use the same keys. Never add an eval-only alias such as
  `point_maps = point_maps_cam` unless the matching training job used that same
  coordinate source.
- Default new experimental features to disabled unless requested.
- Avoid changing baseline behavior unless the wrapper or config explicitly enables the feature.

## Verification

- For Python changes, run:
  `python -m py_compile <changed files>`
- For geometry projection changes, run:
  `conda run -n vlm3r python tests/test_metric_grounded_geometry_projection.py`
- For Slurm or shell wrapper changes, run:
  `bash -n <changed scripts>`
- If a dependency or environment is unavailable, report that clearly instead of silently skipping.

## Git

- Never revert user changes unless explicitly asked.
- When asked to commit, commit relevant files together and split unrelated changes into separate commits.
- Use Conventional Commits:
  `<type>[optional scope]: <description>`
