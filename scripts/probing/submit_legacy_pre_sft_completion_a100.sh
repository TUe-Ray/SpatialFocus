#!/usr/bin/env bash
# Submit one baseline-first smoke, seven parallel formal probes, then summary.
set -euo pipefail

REPO_ROOT="${SPATIALFOCUS_ROOT:-/home/shuang/SpatialFocus}"
DURABLE_ROOT="${DURABLE_ROOT:-/home/shuang/proxy_outputs/legacy_pre_sft_completion_a100_v1}"
GPU_SCRIPT="$REPO_ROOT/scripts/probing/slurm_legacy_pre_sft_completion_a100.sbatch"
SUMMARY_SCRIPT="$REPO_ROOT/scripts/probing/slurm_legacy_pre_sft_completion_summary.sbatch"
CANDIDATES=(SS012 SS123 SS036 SSCROSS VLM3R GEOROPE SELECTIVE)
A100_SBU_PER_GPU_HOUR=128
SMOKE_HOURS=4
FORMAL_HOURS=48
SUMMARY_SBU=1

mkdir -p "$DURABLE_ROOT/logs" "$DURABLE_ROOT/provenance"
bash "$REPO_ROOT/scripts/probing/run_legacy_pre_sft_completion_a100.sh" preflight

# Fail closed on Snellius credit. gpu_a100 is billed at 128 SBU/GPU-hour;
# queued-job accounting can reserve the declared walltime, so gate the full
# dependency graph rather than assuming every job will finish early.
required_sbu=$((A100_SBU_PER_GPU_HOUR * (SMOKE_HOURS + ${#CANDIDATES[@]} * FORMAL_HOURS) + SUMMARY_SBU))
remaining_sbu="$(/home/shuang/miniconda3/envs/vlm3r_a100_proxy/bin/python - <<'PY'
import json
import subprocess

payload = json.loads(subprocess.check_output(["accinfo", "--json2"], text=True))
budgets = [
    item for item in payload.get("budgets", [])
    if "gpu_a100" in item.get("products", [])
]
if not budgets:
    raise SystemExit("No active accinfo budget includes gpu_a100")
print(sum(float(item["acc_balance"]) for item in budgets))
PY
)"
"/home/shuang/miniconda3/envs/vlm3r_a100_proxy/bin/python" - "$remaining_sbu" "$required_sbu" <<'PY'
import sys

remaining, required = map(float, sys.argv[1:])
print({"remaining_sbu": remaining, "declared_maximum_sbu": required})
if remaining < required:
    raise SystemExit(
        f"Insufficient Snellius budget: {remaining:.2f} SBU available, "
        f"{required:.2f} SBU required by declared job walltimes"
    )
PY

smoke_job="$(sbatch --parsable --job-name=legacy-presft-smoke --time=04:00:00 --export=ALL,PHASE=smoke "$GPU_SCRIPT")"
echo -e "SMOKE\t$smoke_job\tgpu_a100\t$DURABLE_ROOT"

formal_jobs=()
for candidate in "${CANDIDATES[@]}"; do
  job="$(sbatch --parsable --dependency="afterok:$smoke_job" --job-name="legacy-presft-${candidate,,}" \
    --export="ALL,PHASE=candidate,CANDIDATE=$candidate" "$GPU_SCRIPT")"
  formal_jobs+=("$job")
  echo -e "$candidate\t$job\tgpu_a100\t$DURABLE_ROOT/results/$candidate"
done

dependency="$(IFS=:; echo "${formal_jobs[*]}")"
summary_job="$(sbatch --parsable --dependency="afterok:$dependency" --job-name=legacy-presft-summary "$SUMMARY_SCRIPT")"
echo -e "SUMMARY\t$summary_job\trome\t$DURABLE_ROOT/summary"
