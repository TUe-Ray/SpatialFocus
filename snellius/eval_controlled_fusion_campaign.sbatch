#!/usr/bin/env bash
# Official VSI-Bench evaluation for controlled B/C/D/E/H and reference A/F/G.
# Submit each task with TRAIN_ARRAY_JOB_ID and
# --dependency=afterany:<training-array-id>_<task-index>.  `afterany` only
# releases the evaluation; the checkpoint-completeness gate below refuses to
# evaluate a failed or partial training output.  This keeps a final model
# evaluable when distributed teardown reports a non-zero Slurm state after all
# training artifacts were durably saved.
#SBATCH --job-name=eval_controlled_fusion_BCDEH
#SBATCH --account=tesei2748
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --qos=normal
#SBATCH --time=06:00:00
#SBATCH --mem=0
#SBATCH --array=0-4%5
#SBATCH --output=/gpfs/home4/geusdd/shuang/SpatialFocus/logs/eval/%x_%A_%a.out
#SBATCH --error=/gpfs/home4/geusdd/shuang/SpatialFocus/logs/eval/%x_%A_%a.err

set -Eeuo pipefail

ACTIVE_REPO_DIR="/gpfs/home4/geusdd/shuang/SpatialFocus"
SNELLIUS_REFERENCE_REPO="/home/geusdd/VLM-3R"
# shellcheck source=/dev/null
source "$SNELLIUS_REFERENCE_REPO/snellius/paths.env"
REPO_DIR="$ACTIVE_REPO_DIR"
export REPO_DIR

TRAIN_ARRAY_JOB_ID="${TRAIN_ARRAY_JOB_ID:?Set TRAIN_ARRAY_JOB_ID to the official training array job ID.}"
[[ "$TRAIN_ARRAY_JOB_ID" =~ ^[1-9][0-9]*$ ]] || {
    echo "TRAIN_ARRAY_JOB_ID must be numeric, got: $TRAIN_ARRAY_JOB_ID" >&2
    exit 2
}

ARRAY_INDEX="${SLURM_ARRAY_TASK_ID:?Submit this launcher as an array task.}"
CAMPAIGN_KIND="${FUSION_CAMPAIGN_KIND:-controlled}"
if [[ "$CAMPAIGN_KIND" == controlled ]]; then
    ARCH_IDS=(B C D E H)
elif [[ "$CAMPAIGN_KIND" == reference_half ]]; then
    ARCH_IDS=(A F G)
else
    echo "Unsupported fusion campaign kind: $CAMPAIGN_KIND" >&2
    exit 2
fi
if (( ARRAY_INDEX < 0 || ARRAY_INDEX >= ${#ARCH_IDS[@]} )); then
    echo "Unsupported array index: $ARRAY_INDEX" >&2
    exit 2
fi
ARCH_ID="${ARCH_IDS[$ARRAY_INDEX]}"

# Reuse the canonical architecture naming instead of maintaining a second
# topology table in the Snellius wrapper.
if [[ "$CAMPAIGN_KIND" == controlled ]]; then
    export CONTROLLED_FUSION_ID="$ARCH_ID"
    export CONTROLLED_FUSION_CONFIG_ONLY=True
    # shellcheck source=/dev/null
    source "$REPO_DIR/train_controlled_fusion_variant.sh"
    unset CONTROLLED_FUSION_CONFIG_ONLY
    CAMPAIGN_CHECKPOINT_ROOT="$TRAIN_SAVE_ROOT/controlled_fusion_official"
    CAMPAIGN_EVAL_ROOT=controlled_fusion_official
else
    export REFERENCE_FUSION_ID="$ARCH_ID"
    export REFERENCE_FUSION_CONFIG_ONLY=True
    # shellcheck source=/dev/null
    source "$REPO_DIR/train_reference_fusion_variant.sh"
    unset REFERENCE_FUSION_CONFIG_ONLY
    CAMPAIGN_CHECKPOINT_ROOT="$TRAIN_SAVE_ROOT/reference_fusion_half_official"
    CAMPAIGN_EVAL_ROOT=reference_fusion_half_official
fi

TRAIN_OUTPUT_ROOT="$CAMPAIGN_CHECKPOINT_ROOT/official_${SUFFIX}_4n4g_16gpu_${TRAIN_ARRAY_JOB_ID}_${ARRAY_INDEX}"
EVAL_STAGE="${FUSION_EVAL_STAGE:-final}"
case "$EVAL_STAGE" in
    final) EXPECTED_RATIO=""; EXPECTED_STEP=1622; PRETRAINED_LOCAL="$TRAIN_OUTPUT_ROOT" ;;
    p01) EXPECTED_RATIO=0.01; EXPECTED_STEP=17; PRETRAINED_LOCAL="$TRAIN_OUTPUT_ROOT/checkpoint-17" ;;
    p05) EXPECTED_RATIO=0.05; EXPECTED_STEP=82; PRETRAINED_LOCAL="$TRAIN_OUTPUT_ROOT/checkpoint-82" ;;
    p25) EXPECTED_RATIO=0.25; EXPECTED_STEP=406; PRETRAINED_LOCAL="$TRAIN_OUTPUT_ROOT/checkpoint-406" ;;
    p50) EXPECTED_RATIO=0.50; EXPECTED_STEP=811; PRETRAINED_LOCAL="$TRAIN_OUTPUT_ROOT/checkpoint-811" ;;
    *) echo "Unsupported FUSION_EVAL_STAGE=$EVAL_STAGE (expected final/p01/p05/p25/p50)." >&2; exit 2 ;;
esac
export PRETRAINED_LOCAL
case "${CONTROLLED_FUSION_EVAL_CONFIG_ONLY:-False}" in
    1|[Tt][Rr][Uu][Ee]|[Yy][Ee][Ss]|[Oo][Nn])
        echo "[OFFICIAL EVAL PLAN] campaign=$CAMPAIGN_KIND arch=$ARCH_ID stage=$EVAL_STAGE training_array=$TRAIN_ARRAY_JOB_ID task=$ARRAY_INDEX"
        echo "[OFFICIAL EVAL PLAN] checkpoint=$PRETRAINED_LOCAL"
        exit 0
        ;;
esac
for required_file in config.json adapter_config.json non_lora_trainables.bin trainer_state.json; do
    [[ -s "$PRETRAINED_LOCAL/$required_file" ]] || {
        echo "Missing or empty official checkpoint artifact: $PRETRAINED_LOCAL/$required_file" >&2
        exit 3
    }
done
if [[ ! -s "$PRETRAINED_LOCAL/adapter_model.bin" && ! -s "$PRETRAINED_LOCAL/adapter_model.safetensors" ]]; then
    echo "Missing LoRA adapter under $PRETRAINED_LOCAL" >&2
    exit 3
fi
[[ -s "$TRAIN_OUTPUT_ROOT/checkpoint_milestones.json" ]] || {
    echo "Missing milestone manifest: $TRAIN_OUTPUT_ROOT/checkpoint_milestones.json" >&2
    exit 3
}
if [[ "$EVAL_STAGE" == final ]]; then
    [[ -s "$PRETRAINED_LOCAL/generation_config.json" ]] || {
        echo "Missing final generation config: $PRETRAINED_LOCAL/generation_config.json" >&2
        exit 3
    }
fi

module purge
module load 2023
module load CUDA/12.1.1

VLM3R_CONDA_PREFIX="/home/geusdd/.conda/envs/vlm3r-snellius"
[[ -x "$VLM3R_CONDA_PREFIX/bin/python" ]] || {
    echo "Missing validated Snellius environment: $VLM3R_CONDA_PREFIX" >&2
    exit 2
}
export CONDA_PREFIX="$VLM3R_CONDA_PREFIX"
export PATH="$VLM3R_CONDA_PREFIX/bin:$PATH"
export CONDA_BASE="/sw/arch/RHEL9/EB_production/2024/software/Anaconda3/2024.06-1"
export CONDA_ENV="$VLM3R_CONDA_PREFIX"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$REPO_DIR:$SNELLIUS_REFERENCE_REPO/third_party/CUT3R${PYTHONPATH:+:$PYTHONPATH}"
export CUT3R_SOURCE_ROOT="$SNELLIUS_REFERENCE_REPO/third_party/CUT3R"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_HOME="${CUDA_HOME:?CUDA module did not set CUDA_HOME}"
if [[ "${SLURM_JOB_PARTITION:-}" == gpu_h100 ]]; then
    export TORCH_CUDA_ARCH_LIST=9.0
else
    export TORCH_CUDA_ARCH_LIST=8.0
fi

# `afterany` handles the known multi-node torchrun teardown race, but must not
# make an incomplete checkpoint eligible for evaluation. A milestone eval only
# requires its own protected checkpoint; a final eval requires the completed
# root plus every milestone.
"$VLM3R_CONDA_PREFIX/bin/python" - \
    "$TRAIN_OUTPUT_ROOT" "$PRETRAINED_LOCAL" "$EVAL_STAGE" "$EXPECTED_RATIO" "$EXPECTED_STEP" <<'PY'
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
checkpoint = Path(sys.argv[2])
stage = sys.argv[3]
expected_ratio = float(sys.argv[4]) if sys.argv[4] else None
requested_step = int(sys.argv[5])
trainer_state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
manifest = json.loads((root / "checkpoint_milestones.json").read_text(encoding="utf-8"))

max_steps = manifest.get("max_steps")
if max_steps != 1622:
    raise SystemExit(f"official campaign expected max_steps=1622, got {max_steps!r}")
if trainer_state.get("max_steps") != max_steps:
    raise SystemExit(
        "trainer/manifest max-step mismatch: "
        f"trainer={trainer_state.get('max_steps')!r}, manifest={max_steps!r}"
    )
required_global_step = max_steps if stage == "final" else requested_step
if trainer_state.get("global_step") != required_global_step:
    raise SystemExit(
        "checkpoint did not reach its declared optimizer step: "
        f"global_step={trainer_state.get('global_step')!r}, expected={required_global_step}"
    )

expected_ratios = (0.01, 0.05, 0.25, 0.50)
entries = manifest.get("milestones")
if not isinstance(entries, list) or len(entries) != len(expected_ratios):
    raise SystemExit(f"unexpected milestone manifest: {manifest}")
for entry, ratio in zip(entries, expected_ratios):
    milestone_step = math.ceil(ratio * max_steps)
    expected_checkpoint = f"checkpoint-{milestone_step}"
    if entry.get("ratio") != ratio or entry.get("step") != milestone_step:
        raise SystemExit(
            "milestone mismatch: "
            f"expected ratio={ratio}, step={milestone_step}; got {entry!r}"
        )
    if entry.get("checkpoint") != expected_checkpoint:
        raise SystemExit(f"milestone checkpoint name mismatch: {entry!r}")
    if (stage == "final" or ratio == expected_ratio) and entry.get("saved") is not True:
        raise SystemExit(f"milestone was not durably saved: {entry!r}")
    if stage == "final" or ratio == expected_ratio:
        milestone = root / expected_checkpoint
        for name in ("config.json", "adapter_config.json", "non_lora_trainables.bin", "trainer_state.json"):
            if not (milestone / name).is_file():
                raise SystemExit(f"missing milestone artifact: {milestone / name}")
        if not any((milestone / name).is_file() for name in ("adapter_model.bin", "adapter_model.safetensors")):
            raise SystemExit(f"missing milestone LoRA adapter: {milestone}")

if stage != "final":
    target = next((entry for entry in entries if entry.get("ratio") == expected_ratio), None)
    if target is None or target.get("step") != requested_step or target.get("saved") is not True:
        raise SystemExit(f"requested milestone is unavailable: stage={stage}, entry={target!r}")
    if checkpoint != root / f"checkpoint-{requested_step}":
        raise SystemExit(f"resolved checkpoint path mismatch: {checkpoint}")

print(
    "[CHECKPOINT COMPLETENESS PASS] "
    f"root={root} stage={stage} checkpoint={checkpoint} global_step={required_global_step}"
)
PY

if [[ "$CAMPAIGN_KIND" == reference_half && "$ARCH_ID" == A ]]; then
    "$VLM3R_CONDA_PREFIX/bin/python" - "$PRETRAINED_LOCAL/config.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    config = json.load(handle)
expected = {
    "spatial_tower": "cut3r",
    "fusion_block": "cross_attention",
    "use_cut3r_spatialstack": False,
    "tune_fusion_block": True,
    "tune_mm_mlp_adapter": True,
}
for key, value in expected.items():
    # Legacy VLM3R configs predate the explicit SpatialStack flag; absence is
    # the backward-compatible False value used by model construction.
    actual = config.get(key, False) if key == "use_cut3r_spatialstack" else config.get(key)
    if actual != value:
        raise SystemExit(
            f"reference A config mismatch {key}: expected={value!r}, got={actual!r}"
        )
print("[REFERENCE A CONFIG PASS] legacy CUT3R cross-attention topology verified")
PY
fi

case "${FUSION_EVAL_PREFLIGHT_ONLY:-False}" in
    1|[Tt][Rr][Uu][Ee]|[Yy][Ee][Ss]|[Oo][Nn])
        echo "[OFFICIAL EVAL PREFLIGHT PASS] campaign=$CAMPAIGN_KIND arch=$ARCH_ID stage=$EVAL_STAGE checkpoint=$PRETRAINED_LOCAL"
        exit 0
        ;;
esac

JOB_TMP_ROOT="/scratch-local/${SLURM_JOB_ID}/fusion_eval_${CAMPAIGN_KIND}_${ARRAY_INDEX}_${EVAL_STAGE}"
export TMPDIR="$JOB_TMP_ROOT/tmp"
export XDG_CACHE_HOME="$JOB_TMP_ROOT/xdg_cache"
export TRITON_CACHE_DIR="$JOB_TMP_ROOT/triton_cache"
export TORCH_EXTENSIONS_DIR="$JOB_TMP_ROOT/torch_extensions"
TASK_DIR="$JOB_TMP_ROOT/vsibench_task"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" "$TASK_DIR"

SOURCE_TASK_DIR="$REPO_DIR/thinking-in-space/lmms_eval/tasks/vsibench_leonardo_offline"
sed \
    's|cache_dir = os.path.join(base_cache_dir, cache_name)|cache_dir = os.path.expanduser(os.getenv("VSI_MEDIA_ROOT", os.path.join(base_cache_dir, cache_name)))|' \
    "$SOURCE_TASK_DIR/utils.py" > "$TASK_DIR/utils.py"
sed \
    -e "s|/leonardo_scratch/fast/EUHPC_D32_006/vsibench|$VSI_BENCH_EVAL_ROOT|g" \
    -e "s|cache_dir: vsibench|cache_dir: $HF_DATASETS_CACHE/vsibench|" \
    "$SOURCE_TASK_DIR/vsibench.yaml" > "$TASK_DIR/vsibench.yaml"
if grep -En '/leonardo_|cache_dir: vsibench$' "$TASK_DIR/vsibench.yaml"; then
    echo "Rendered task still contains a Leonardo or relative dataset path." >&2
    exit 2
fi

EVAL_TOKEN="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}_${SLURM_ARRAY_TASK_ID}"
OUTPUT_PATH="$VLM3R_LOG_ROOT/eval/$CAMPAIGN_EVAL_ROOT/${SUFFIX}_${TRAIN_ARRAY_JOB_ID}_${ARRAY_INDEX}/${EVAL_STAGE}_${EVAL_TOKEN}"
RUNTIME_ROOT="$JOB_TMP_ROOT/runtime"
RUN_NAME="eval_official_${SUFFIX}_${EVAL_STAGE}_${TRAIN_ARRAY_JOB_ID}_${ARRAY_INDEX}_${EVAL_TOKEN}"
export OUTPUT_PATH RUNTIME_ROOT RUN_NAME TASK_DIR
export TASK_FILE="$TASK_DIR/vsibench.yaml"
export SUBMODULE_DIR="$REPO_DIR/thinking-in-space"
export MODEL_BASE_LOCAL="$LOCAL_MODEL_BASE"
export SIGLIP_LOCAL="$LOCAL_SIGLIP"
export FAST_ROOT="$VLM3R_ROOT"
export VSI_ROOT="$VSI_BENCH_EVAL_ROOT"
export VSI_MEDIA_ROOT="$VSI_BENCH_EVAL_MEDIA_ROOT"
if [[ "$CAMPAIGN_KIND" == controlled ]]; then
    export SPATIAL_FEATURES_ROOT="$CUT3R_DEC12_ROOT"
    export SPATIAL_FEATURES_SUBDIR="12:$CUT3R_DEC12_ROOT:spatial_features"
else
    export SPATIAL_FEATURES_ROOT="$CUT3R_FEATURE_ROOT"
    export SPATIAL_FEATURES_SUBDIR="6:$CUT3R_DEC6_ROOT:spatial_features_dec_6;9:$CUT3R_DEC9_ROOT:spatial_features_dec_9;12:$CUT3R_DEC12_ROOT:spatial_features"
fi
export NUM_PROCESSES=4
export BATCH_SIZE=1
export LIMIT=0
export MAX_FRAMES_NUM=32
export CHECK_SPATIAL_SIDECARS=True
export MODEL_ATTN_IMPLEMENTATION=sdpa

echo "[OFFICIAL EVAL] campaign=$CAMPAIGN_KIND arch=$ARCH_ID stage=$EVAL_STAGE training_array=$TRAIN_ARRAY_JOB_ID task=$ARRAY_INDEX"
echo "[OFFICIAL EVAL] checkpoint=$PRETRAINED_LOCAL"
echo "[OFFICIAL EVAL] output=$OUTPUT_PATH"

if [[ "$CAMPAIGN_KIND" == controlled ]]; then
    case "$ARCH_ID" in
        B) EVAL_WRAPPER="eval_controlled_fusion_B_pre_projector_add_dec12_vsibench.sh" ;;
        C) EVAL_WRAPPER="eval_controlled_fusion_C_cross_attn_dec12_llm0_vsibench.sh" ;;
        D) EVAL_WRAPPER="eval_controlled_fusion_D_add_dec12_llm0_vsibench.sh" ;;
        E) EVAL_WRAPPER="eval_controlled_fusion_E_add_dec12x3_llm0_1_2_vsibench.sh" ;;
        H) EVAL_WRAPPER="eval_controlled_fusion_H_cross_attn_dec12x3_llm0_1_2_vsibench.sh" ;;
    esac
elif [[ "$ARCH_ID" == A ]]; then
    COMPAT_VSI_ROOT="$JOB_TMP_ROOT/vsi_root"
    mkdir -p "$COMPAT_VSI_ROOT"
    for parquet in test_pruned.parquet test_debiased.parquet; do
        ln -sfn "$VSI_BENCH_EVAL_ROOT/$parquet" "$COMPAT_VSI_ROOT/$parquet"
    done
    for split in scannet arkitscenes scannetpp; do
        ln -sfn "$VSI_BENCH_EVAL_MEDIA_ROOT/$split" "$COMPAT_VSI_ROOT/$split"
    done
    export VSI_ROOT="$COMPAT_VSI_ROOT"
    export EXTRA_MODEL_ARGS=overwrite=False
    EVAL_WRAPPER="eval_vlm3r_orig.sh"
else
    export REFERENCE_FUSION_ID="$ARCH_ID"
    EVAL_WRAPPER="eval_reference_fusion_variant_vsibench.sh"
fi
bash "$REPO_DIR/$EVAL_WRAPPER"

"$VLM3R_CONDA_PREFIX/bin/python" - "$OUTPUT_PATH" <<'PY'
import json
import math
import sys
from pathlib import Path

output = Path(sys.argv[1])
result_paths = list(output.rglob("results.json"))
sample_paths = list(output.rglob("vsibench.json"))
if not result_paths or not sample_paths:
    raise SystemExit(
        f"Missing evaluation artifacts under {output}: results={result_paths}, samples={sample_paths}"
    )
effective = 0
metrics = []
for path in result_paths:
    payload = json.loads(path.read_text(encoding="utf-8"))
    effective += int(payload.get("n-samples", {}).get("vsibench", {}).get("effective", 0))
    metrics.append(payload.get("results", {}).get("vsibench", {}))
if effective <= 0:
    raise SystemExit(f"Evaluation completed without effective samples: {effective}")
for metric in metrics:
    for name, value in metric.items():
        if isinstance(value, (int, float)) and not math.isfinite(float(value)):
            raise SystemExit(f"Non-finite evaluation metric: {name}={value}")
print(f"[OFFICIAL EVAL PASS] effective={effective} metrics={metrics}")
PY

echo "[OFFICIAL EVAL PASS] campaign=$CAMPAIGN_KIND arch=$ARCH_ID stage=$EVAL_STAGE output=$OUTPUT_PATH"
