#!/usr/bin/env bash
# Full-policy post-SFT ScanNet depth probes for controlled B/C/D/E/H.
# Extraction is sequential and feature caches are recycled only after durable
# probe metrics/checkpoints/provenance pass completeness verification.
set -euo pipefail

MODE="${1:-}"
CANDIDATE="${2:-}"
if [[ ! "$MODE" =~ ^(preflight|smoke-one|smoke-all|run-one|run-all|summarize|status)$ ]]; then
  echo "Usage: $0 {preflight|smoke-one <B|C|D|E|H>|smoke-all|run-one <B|C|D|E|H>|run-all|summarize|status}" >&2
  exit 2
fi

REPO_ROOT="${REPO_ROOT:-/home/shaoruei/SpatialFocus}"
ENV_NAME="${ENV_NAME:-vlm3r}"
VLM3R_PYTHON="${VLM3R_PYTHON:-/home/shaoruei/miniconda3/envs/vlm3r/bin/python}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/mnt/DATA_SSD/shaoruei/models/vlm3r_runs/controlled_fusion_official}"
BASE_MODEL="${BASE_MODEL:-/mnt/DATA_SSD/shaoruei/models/base/LLaVA-NeXT-Video-7B-Qwen2}"
SIGLIP_MODEL="${SIGLIP_MODEL:-/mnt/DATA_SSD/shaoruei/models/base/siglip-so400m-patch14-384}"
FORWARD_ROOT="${FORWARD_ROOT:-/mnt/DATA_SSD/shaoruei/probing_data/forward_frames_32_v1}"
TARGET_ROOT="${TARGET_ROOT:-/mnt/DATA_SSD/shaoruei/probing_data/probe_targets_2f_v1}"
FEATURE_ROOT="${FEATURE_ROOT:-/mnt/DATA_SSD/shaoruei/probing_data/cut3r_features}"
SAMPLE_INDICES="${SAMPLE_INDICES:-/home/shaoruei/probe_provenance/scannet_baseline_L6/scannet_baseline_L6_depth_provenance/splits/semantic_probe_scannet_final_usable_sample_indices.json}"
DATA_YAML="${DATA_YAML:-$REPO_ROOT/scripts/probing/scannet_depth_probe_local_data.yaml}"
CACHE_ROOT="${CACHE_ROOT:-/home/shaoruei/probe_cache/controlled_fusion_post_sft_depth_v1}"
DURABLE_ROOT="${DURABLE_ROOT:-/home/shaoruei/probe_outputs/controlled_fusion_post_sft_depth_v1}"
LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/logs/controlled_fusion_post_sft_depth_v1}"
SMOKE_MANIFEST="$DURABLE_ROOT/provenance/smoke_1train_1val.json"
RECYCLE_FULL_CACHE="${RECYCLE_FULL_CACHE:-1}"
SPATIAL_SUBDIR="12:spatial_features"
GPU_WEIGHT_BUDGETS="${GPU_WEIGHT_BUDGETS:-6GiB,10GiB}"
CPU_WEIGHT_BUDGET="${CPU_WEIGHT_BUDGET:-40GiB}"
CANDIDATES=(B C D E H)

source "$REPO_ROOT/scripts/probing/common_probe_layers.sh"
PRE_LLM_FEATURES="fusion_output,projected_features"
FEATURE_LEVELS="${PRE_LLM_FEATURES},${COMMON_PROBE_LAYER_LEVELS_CSV}"
LEVELS=(fusion_output projected_features)
for layer in "${COMMON_PROBE_LAYERS[@]}"; do LEVELS+=("layer_$layer"); done

mkdir -p "$CACHE_ROOT" "$DURABLE_ROOT/provenance" "$DURABLE_ROOT/probes" "$LOG_ROOT"

log() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$*"; }

candidate_field() {
  local id="$1" field="$2"
  case "$id:$field" in
    B:checkpoint) printf 'official_controlled_B_pre_projector_add_dec12_once_4n4g_16gpu_26484327_0' ;;
    C:checkpoint) printf 'official_controlled_C_cross_attn_dec12_llm0_once_4n4g_16gpu_26484327_1' ;;
    D:checkpoint) printf 'official_controlled_D_add_dec12_llm0_once_4n4g_16gpu_26484327_2' ;;
    E:checkpoint) printf 'official_controlled_E_add_dec12x3_llm0_1_2_repeat_siteproj_4n4g_16gpu_26484327_3' ;;
    H:checkpoint) printf 'official_controlled_H_cross_attn_dec12x3_llm0_1_2_repeat_4n4g_16gpu_26484327_4' ;;
    B:label) printf 'controlled_b_post_sft' ;; C:label) printf 'controlled_c_post_sft' ;;
    D:label) printf 'controlled_d_post_sft' ;; E:label) printf 'controlled_e_post_sft' ;;
    H:label) printf 'controlled_h_post_sft' ;;
    B:preset) printf 'original' ;;
    C:preset|D:preset|E:preset|H:preset) printf 'spatialstack' ;;
    *) echo "Unsupported controlled candidate/field: $id/$field" >&2; return 2 ;;
  esac
}

validate_candidate() {
  [[ "$1" =~ ^(B|C|D|E|H)$ ]] || {
    echo "Expected candidate B, C, D, E, or H; got ${1:-<empty>}" >&2
    exit 2
  }
}

checkpoint_path() { printf '%s/%s\n' "$CHECKPOINT_ROOT" "$(candidate_field "$1" checkpoint)"; }

gpu_readiness() {
  local purpose="$1" gpu
  for gpu in 0 1; do
    env CUDA_VISIBLE_DEVICES="$gpu" conda run --no-capture-output -n "$ENV_NAME" python -u \
      "$REPO_ROOT/scripts/probing/verify_titan_v_readiness.py" \
      --physical-gpu-id "$gpu" --output "$LOG_ROOT/${purpose}_gpu${gpu}_readiness.json"
  done
}

require_idle_gpus() {
  local apps
  apps="$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits)"
  if [[ -n "${apps//[[:space:]]/}" ]]; then
    echo "Refusing to contend with existing GPU compute processes:" >&2
    printf '%s\n' "$apps" >&2
    exit 1
  fi
}

preflight() {
  "$VLM3R_PYTHON" - "$CHECKPOINT_ROOT" "$SAMPLE_INDICES" "$FORWARD_ROOT" "$TARGET_ROOT" \
    "$FEATURE_ROOT" "$BASE_MODEL" "$SIGLIP_MODEL" "$LOG_ROOT/preflight.json" "$FEATURE_LEVELS" <<'PY'
import hashlib
import json
import shutil
import sys
from pathlib import Path

import torch

(checkpoint_root, split_path, forward_root, target_root, feature_root,
 base_model, siglip_model, report_path) = map(Path, sys.argv[1:9])
feature_levels = sys.argv[9].split(",")
required_files = {
    "adapter_model.bin", "non_lora_trainables.bin", "adapter_config.json",
    "config.json", "generation_config.json",
}
specs = {
    "B": {
        "directory": "official_controlled_B_pre_projector_add_dec12_once_4n4g_16gpu_26484327_0",
        "config": {"use_cut3r_spatialstack": None, "fusion_block": "pre_projector_add",
                   "pre_projector_add_source_layer": 12, "spatial_tower_select_feature": "patch_tokens"},
        "required_keys": ("fusion_block.spatial_proj_in.weight", "fusion_block.spatial_proj_out.weight",
                          "mm_projector.0.weight", "mm_projector.2.weight"),
    },
    "C": {
        "directory": "official_controlled_C_cross_attn_dec12_llm0_once_4n4g_16gpu_26484327_1",
        "config": {"use_cut3r_spatialstack": True, "cut3r_spatialstack_layers": "12",
                   "cut3r_spatialstack_llm_layers": "0", "cut3r_spatialstack_fusion_type": "cross_attn",
                   "cut3r_spatialstack_projector_binding": "source_specific"},
        "required_keys": ("cross_attn_blocks.0.q_proj.weight", "cross_attn_blocks.0.k_proj.weight"),
    },
    "D": {
        "directory": "official_controlled_D_add_dec12_llm0_once_4n4g_16gpu_26484327_2",
        "config": {"use_cut3r_spatialstack": True, "cut3r_spatialstack_layers": "12",
                   "cut3r_spatialstack_llm_layers": "0", "cut3r_spatialstack_fusion_type": "add",
                   "cut3r_spatialstack_projector_binding": "source_specific"},
        "required_keys": ("branches.12.proj_in.weight", "branches.12.proj_out.weight"),
    },
    "E": {
        "directory": "official_controlled_E_add_dec12x3_llm0_1_2_repeat_siteproj_4n4g_16gpu_26484327_3",
        "config": {"use_cut3r_spatialstack": True, "cut3r_spatialstack_layers": "12,12,12",
                   "cut3r_spatialstack_llm_layers": "0,1,2", "cut3r_spatialstack_fusion_type": "add",
                   "cut3r_spatialstack_projector_binding": "site_specific"},
        "required_keys": tuple(f"branches.{i}.proj_in.weight" for i in range(3)),
    },
    "H": {
        "directory": "official_controlled_H_cross_attn_dec12x3_llm0_1_2_repeat_4n4g_16gpu_26484327_4",
        "config": {"use_cut3r_spatialstack": True, "cut3r_spatialstack_layers": "12,12,12",
                   "cut3r_spatialstack_llm_layers": "0,1,2", "cut3r_spatialstack_fusion_type": "cross_attn",
                   "cut3r_spatialstack_projector_binding": "source_specific"},
        "required_keys": tuple(f"cross_attn_blocks.{i}.q_proj.weight" for i in range(3)),
    },
}

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

failures = []
report = {"schema_version": "controlled_fusion_post_sft_depth_preflight_v1", "models": {}}
actual_dirs = sorted(path.name for path in checkpoint_root.iterdir() if path.is_dir())
expected_dirs = sorted(spec["directory"] for spec in specs.values())
if actual_dirs != expected_dirs:
    failures.append(f"checkpoint directories differ: expected={expected_dirs}, actual={actual_dirs}")
all_checkpoint_files = list(checkpoint_root.glob("*/*"))
if len([path for path in all_checkpoint_files if path.is_file()]) != 25:
    failures.append("controlled checkpoint root does not contain exactly 25 files")
for candidate, spec in specs.items():
    root = checkpoint_root / spec["directory"]
    actual_files = {path.name for path in root.iterdir() if path.is_file()} if root.is_dir() else set()
    if actual_files != required_files:
        failures.append(f"{candidate} file set mismatch: {sorted(actual_files)}")
        continue
    config = json.loads((root / "config.json").read_text())
    mismatches = {key: {"expected": expected, "actual": config.get(key)}
                  for key, expected in spec["config"].items() if config.get(key) != expected}
    try:
        state = torch.load(root / "non_lora_trainables.bin", map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(root / "non_lora_trainables.bin", map_location="cpu")
    keys = tuple(state)
    missing_signatures = [fragment for fragment in spec["required_keys"]
                          if not any(fragment in key for key in keys)]
    if mismatches or missing_signatures:
        failures.append(f"{candidate} config/signature mismatch")
    report["models"][candidate] = {
        "checkpoint": str(root), "config_mismatches": mismatches,
        "missing_state_signatures": missing_signatures,
        "files": {name: {"bytes": (root / name).stat().st_size, "sha256": sha256(root / name)}
                  for name in sorted(required_files)},
    }
    del state

for forbidden in ("optimizer.pt", "zero_to_fp32.py"):
    if any(checkpoint_root.rglob(forbidden)):
        failures.append(f"forbidden training-state artifact present: {forbidden}")
if any(path.is_dir() for path in checkpoint_root.rglob("checkpoint-*")):
    failures.append("checkpoint milestone directory present")

split_hash = sha256(split_path)
split = json.loads(split_path.read_text())
videos = split.get("videos", [])
scene_ids = {str(video.get("scene_id")) for video in videos if video.get("source_dataset") == "scannet"}
forward_count = sum(1 for _ in (forward_root / "frames/scannet").glob("*.pt"))
missing_sidecars = sorted(scene for scene in scene_ids
                          if not (feature_root / "scannet/spatial_features" / f"{scene}.pt").is_file())
if (split_hash != "d478cb684958dfc25066821ec83d5216469577c9e282e33bdf87d3c88b200d8e"
        or len(videos) != 1199 or len(scene_ids) != 1199
        or int(split.get("train_videos", -1)) != 1006 or int(split.get("val_videos", -1)) != 193):
    failures.append("authoritative ScanNet split identity mismatch")
if forward_count != 1199 or missing_sidecars:
    failures.append(f"input cache incomplete: forward={forward_count}, missing_sidecars={missing_sidecars[:5]}")
for path in (target_root, base_model / "config.json", siglip_model / "config.json"):
    if not path.exists(): failures.append(f"missing input: {path}")

disk = shutil.disk_usage(Path.home())
report.update({
    "split_sha256": split_hash, "videos": len(videos), "train_videos": split.get("train_videos"),
    "val_videos": split.get("val_videos"), "forward_scannet_files": forward_count,
    "cut3r_dec12_missing": len(missing_sidecars), "feature_levels": feature_levels,
    "home_free_bytes": disk.free, "rolling_cache_policy": True,
    "assessment": "PASS" if not failures else "FAIL", "failures": failures,
})
report_path.parent.mkdir(parents=True, exist_ok=True)
report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(json.dumps({k: report[k] for k in ("assessment", "videos", "train_videos", "val_videos",
                                         "forward_scannet_files", "cut3r_dec12_missing",
                                         "home_free_bytes", "feature_levels", "failures")}, indent=2))
if failures: raise SystemExit(1)
PY
}

make_smoke_manifest() {
  [[ -f "$SMOKE_MANIFEST" ]] && return 0
  conda run --no-capture-output -n "$ENV_NAME" python -u "$REPO_ROOT/scripts/probing/make_depth_probe_smoke_manifest.py" \
    --sample-indices "$SAMPLE_INDICES" --output "$SMOKE_MANIFEST" --train-videos 1 --val-videos 1
}

extract_one() {
  local id="$1" namespace="$2" manifest="$3" root checkpoint label preset log_file
  checkpoint="$(checkpoint_path "$id")"
  label="$(candidate_field "$id" label)"
  preset="$(candidate_field "$id" preset)"
  root="$CACHE_ROOT/$namespace/$id"
  log_file="$LOG_ROOT/${namespace}_${id}_extract.log"
  mkdir -p "$root"
  log "Extraction candidate=$id label=$label checkpoint=$checkpoint GPUs=$CUDA_DEVICES output=$root log=$log_file"
  env CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" SPATIALFOCUS_CPU_MERGE_LORA=1 \
    SPATIALFOCUS_CPU_MERGE_GPU_BUDGETS="$GPU_WEIGHT_BUDGETS" \
    SPATIALFOCUS_CPU_MERGE_CPU_BUDGET="$CPU_WEIGHT_BUDGET" \
    MPLCONFIGDIR=/tmp/controlled_post_sft_mpl \
    conda run --no-capture-output -n "$ENV_NAME" python -u "$REPO_ROOT/scripts/probing/extract_depth_probe_features.py" \
      --model-label "$label" --model-loading-mode adapter --model-path "$checkpoint" \
      --feature-preset "$preset" --model-base "$BASE_MODEL" --model-name vlm-3r-llava-qwen2-lora \
      --siglip-path "$SIGLIP_MODEL" --output-root "$root" --sample-indices "$manifest" \
      --data-yaml "$DATA_YAML" --feature-root "$FEATURE_ROOT" --spatial-features-subdir "$SPATIAL_SUBDIR" \
      --forward-frames-root "$FORWARD_ROOT" --probe-targets-root "$TARGET_ROOT" \
      --video-folder "$FORWARD_ROOT" --image-folder "$FORWARD_ROOT" --frames-upbound 32 \
      --skip-spatial-tower-load true --device cuda:0 --device-map auto --dtype float16 --cache-dtype float16 \
      --feature-levels "$FEATURE_LEVELS" --runtime-root "$root/runtime" \
      --assert-first-video --resume 2>&1 | tee -a "$log_file"
}

train_level() {
  local root="$1" manifest="$2" label="$3" level="$4" gpu="$5" epochs="$6" allow_partial="$7" log_file="$8"
  local partial=()
  [[ "$allow_partial" == 1 ]] && partial=(--allow-partial)
  env CUDA_VISIBLE_DEVICES="$gpu" MPLCONFIGDIR=/tmp/controlled_post_sft_mpl \
    conda run --no-capture-output -n "$ENV_NAME" python -u "$REPO_ROOT/scripts/probing/train_depth_probes.py" \
      --output-root "$root" --sample-indices "$manifest" --probe-subdir probes \
      --model-labels "$label" --feature-levels "$level" --epochs "$epochs" --batch-size 32 \
      --lr 1e-3 --early-stop-patience 10 --num-workers 0 --probe-seed 0 \
      --experiment-variant controlled_fusion_post_sft --device cuda:0 --no-write-aggregate \
      --skip-existing "${partial[@]}" >>"$log_file" 2>&1
}

train_all_levels() {
  local id="$1" namespace="$2" manifest="$3" epochs="$4" allow_partial="$5"
  local root="$CACHE_ROOT/$namespace/$id" label log_file index first second pid0 pid1
  label="$(candidate_field "$id" label)"
  log_file="$LOG_ROOT/${namespace}_${id}_probes.log"
  index=0
  while (( index < ${#LEVELS[@]} )); do
    first="${LEVELS[$index]}"
    train_level "$root" "$manifest" "$label" "$first" 0 "$epochs" "$allow_partial" "$log_file" & pid0=$!
    index=$((index + 1))
    if (( index < ${#LEVELS[@]} )); then
      second="${LEVELS[$index]}"
      train_level "$root" "$manifest" "$label" "$second" 1 "$epochs" "$allow_partial" "$log_file" & pid1=$!
      wait "$pid0"
      wait "$pid1"
      index=$((index + 1))
    else
      wait "$pid0"
    fi
  done
}

verify_full() {
  local id="$1" require_probes="$2" root label suffix args=()
  root="$CACHE_ROOT/full/$id"
  label="$(candidate_field "$id" label)"
  suffix=features
  if [[ "$require_probes" == 1 ]]; then args=(--require-probes); suffix=probes; fi
  conda run --no-capture-output -n "$ENV_NAME" python -u "$REPO_ROOT/scripts/probing/verify_post_sft_depth_probe.py" \
    --output-root "$root" --model-label "$label" --sample-indices "$SAMPLE_INDICES" \
    "${args[@]}" --output "$DURABLE_ROOT/provenance/$label/${suffix}_completeness.json"
}

smoke_one() {
  local id="$1" root label marker
  validate_candidate "$id"
  label="$(candidate_field "$id" label)"
  marker="$DURABLE_ROOT/provenance/$label/smoke_verification.json"
  if [[ -f "$marker" ]] && [[ "$(jq -r '.assessment // empty' "$marker")" == PASS ]]; then
    log "Smoke already PASS for $id ($marker)"
    return 0
  fi
  preflight
  require_idle_gpus
  gpu_readiness "smoke_${id}"
  make_smoke_manifest
  extract_one "$id" smoke "$SMOKE_MANIFEST"
  train_all_levels "$id" smoke "$SMOKE_MANIFEST" 2 1
  root="$CACHE_ROOT/smoke/$id"
  mkdir -p "$(dirname "$marker")"
  conda run --no-capture-output -n "$ENV_NAME" python -u "$REPO_ROOT/scripts/probing/verify_scannet_final_layerwise_smoke.py" \
    --output-root "$root" --model-label "$label" --feature-levels "$FEATURE_LEVELS" \
    --manifest "$SMOKE_MANIFEST" --report "$marker"
  log "Smoke PASS candidate=$id"
}

preserve_and_recycle() {
  local id="$1" root label checkpoint destination
  root="$CACHE_ROOT/full/$id"
  label="$(candidate_field "$id" label)"
  checkpoint="$(checkpoint_path "$id")"
  destination="$DURABLE_ROOT/probes/$label"
  mkdir -p "$destination" "$DURABLE_ROOT/provenance/$label"
  cp -a "$root/probes/$label/." "$destination/"
  cp -a "$root/features/$label/extraction_provenance.json" "$DURABLE_ROOT/provenance/$label/"
  sha256sum "$checkpoint"/{adapter_model.bin,non_lora_trainables.bin,adapter_config.json,config.json,generation_config.json} \
    >"$DURABLE_ROOT/provenance/$label/checkpoint_sha256.txt"
  if [[ "$RECYCLE_FULL_CACHE" == 1 ]]; then
    case "$root" in
      "$CACHE_ROOT"/full/[BCDEH]) rm -rf -- "$root" ;;
      *) echo "Refusing unexpected cache cleanup target: $root" >&2; exit 1 ;;
    esac
    log "Verified durable results retained; recycled $root"
  fi
}

run_one() {
  local id="$1" label marker
  validate_candidate "$id"
  label="$(candidate_field "$id" label)"
  marker="$DURABLE_ROOT/provenance/$label/probes_completeness.json"
  if [[ -f "$marker" ]] && [[ "$(jq -r '.assessment // empty' "$marker")" == PASS ]]; then
    log "Full probe already PASS for $id ($marker)"
    return 0
  fi
  smoke_one "$id"
  require_idle_gpus
  gpu_readiness "full_${id}"
  extract_one "$id" full "$SAMPLE_INDICES"
  verify_full "$id" 0
  train_all_levels "$id" full "$SAMPLE_INDICES" 50 0
  verify_full "$id" 1
  preserve_and_recycle "$id"
  log "Full probe PASS candidate=$id"
}

summarize() {
  local labels=() id labels_csv
  for id in "${CANDIDATES[@]}"; do labels+=("$(candidate_field "$id" label)"); done
  labels_csv="$(IFS=,; echo "${labels[*]}")"
  conda run --no-capture-output -n "$ENV_NAME" python -u "$REPO_ROOT/scripts/probing/train_depth_probes.py" \
    --output-root "$DURABLE_ROOT" --sample-indices "$SAMPLE_INDICES" --probe-subdir probes \
    --model-labels "$labels_csv" --feature-levels "$FEATURE_LEVELS" --skip-existing \
    --result-stem controlled_fusion_post_sft_depth_probe --device cpu
}

status() {
  local id label smoke full
  printf 'candidate\tlabel\tsmoke\tfull\n'
  for id in "${CANDIDATES[@]}"; do
    label="$(candidate_field "$id" label)"
    smoke="$(jq -r '.assessment // "MISSING"' "$DURABLE_ROOT/provenance/$label/smoke_verification.json" 2>/dev/null || echo MISSING)"
    full="$(jq -r '.assessment // "MISSING"' "$DURABLE_ROOT/provenance/$label/probes_completeness.json" 2>/dev/null || echo MISSING)"
    printf '%s\t%s\t%s\t%s\n' "$id" "$label" "$smoke" "$full"
  done
}

case "$MODE" in
  preflight) preflight ;;
  smoke-one) validate_candidate "$CANDIDATE"; smoke_one "$CANDIDATE" ;;
  smoke-all) for CANDIDATE in "${CANDIDATES[@]}"; do smoke_one "$CANDIDATE"; done ;;
  run-one) validate_candidate "$CANDIDATE"; run_one "$CANDIDATE"; summarize ;;
  run-all)
    preflight
    for CANDIDATE in "${CANDIDATES[@]}"; do smoke_one "$CANDIDATE"; done
    for CANDIDATE in "${CANDIDATES[@]}"; do run_one "$CANDIDATE"; done
    summarize
    ;;
  summarize) summarize ;;
  status) status ;;
esac
