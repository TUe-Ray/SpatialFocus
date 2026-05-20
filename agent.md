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
