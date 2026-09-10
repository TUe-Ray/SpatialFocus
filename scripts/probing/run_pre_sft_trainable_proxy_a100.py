#!/usr/bin/env python
"""One-A100 primary pre-SFT trainable-scope proxy and explicit extensions.

This is deliberately narrower than the historical fusion-only evaluator.  It
scores the candidate's exact QA-reachable SFT groups using a single ordinary
supervised QA CE backward and no optimizer.  Extensions remain explicitly
labelled and never alter the authoritative five-candidate roster.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
PROBING_DIR = REPO_ROOT / "scripts" / "probing"
if str(PROBING_DIR) not in sys.path:
    sys.path.insert(0, str(PROBING_DIR))

import evaluate_pre_sft_zero_cost_proxies as common  # noqa: E402
from depth_probe_common import write_json  # noqa: E402
from extract_depth_probe_features import (  # noqa: E402
    build_dataset,
    full_geometry_point_maps,
    load_eomt_consumer_cache,
)
from local_depth_probe_cache import install_forward_frame_loader  # noqa: E402
from scripts.diagnose_layerwise_spatial_hidden_scan import load_model  # noqa: E402
from llava.model.c1_structured_isometry import (  # noqa: E402
    apply_c1_calibration_artifact,
    apply_geometry_c1_calibration_artifact,
)
from llava.model.controlled_fusion_pre_sft import (  # noqa: E402
    CONTROLLED_FUSION_PRE_SFT_SPECS,
    controlled_fusion_artifact_metadata,
)


SCHEMA_VERSION = "pre_sft_trainable_proxy_a100_v1"
BASELINE_ID = "c1_vlm3r_native"
ALL_IDS = tuple(item.identifier for item in common.CANDIDATES)
EXTENSION_IDS = ("ss_depth", "baseline_depth", "base_vlm_zero_spatial")
CONTROLLED_IDS = tuple(f"controlled_{identifier.lower()}" for identifier in CONTROLLED_FUSION_PRE_SFT_SPECS)
VSI_MISSING_IDS = ("geo_rope_fusion", "visual_geo_rope", "extra_object_token", "selective_fusion")
ADDITIONAL_IDS = CONTROLLED_IDS + VSI_MISSING_IDS
EXPECTED_BASELINE_COUNTS = {
    "lora": 322_961_408,
    "fusion_block": 9_747_456,
    "mm_projector": 16_980_992,
    "total": 349_689_856,
}
EXPECTED_C1_SHA256 = {
    "c1_ss_add_012": "29a54005d90ea083d14418dca67e41aaf95947f2575c14910287ce1cf2fc80dc",
    "c1_ss_add_036": "f51409a74fd0735e9b782ccd7fe67da8265b8d80ca55f69035a090145a7e42d9",
    "c1_ss_add_123": "8ff199bfb49cd4fbcfae0a49fd48b4ef3c4b6a36796b25833a38fb8c18b2a150",
    "c1_ss_cross_attn_012": "d1080b6f8a9f0b983aae36867ed560c84fc02c81a321a7aa8158fdbbc675a520",
    "c1_vlm3r_native": "edb6ab3c255d0875e37cf6a18de078511fcfb61a37e3b102541e3f6219548f9c",
}
C1_FILENAMES = {
    "c1_ss_add_012": ("c1_additive_v1", "spatialstack_add.json"),
    "c1_ss_add_036": ("c1_ss_add_036", "spatialstack_add.json"),
    "c1_ss_add_123": ("c1_ss_add_123", "spatialstack_add.json"),
    "c1_ss_cross_attn_012": ("c1_ss_cross_attn_v1", "spatialstack_cross_attn_v1.json"),
    "c1_vlm3r_native": ("c1_vlm3r_v1", "vlm3r.json"),
}


@dataclass(frozen=True)
class ExtensionCandidateSpec:
    """A pre-SFT extension whose provenance is separate from the C1-five roster.

    The two auxiliary-supervision variants reuse an audited C1 interface.
    Their fresh supervision heads are deliberately not in the primary scope:
    the mandated ``L_proxy = L_QA`` never executes them.  The base control has
    no spatial interface at all, so it uses the plain-base loader and retains
    only fresh LoRA plus the SFT-tuned projector.
    """

    identifier: str
    vsi_model: str
    vsi_avg: float
    construction: str
    source_c1_identifier: str | None = None
    auxiliary_only_module: str | None = None


EXTENSION_CANDIDATES = {
    "ss_depth": ExtensionCandidateSpec(
        "ss_depth", "SS + depth", 61.3, "c1_auxiliary", "c1_ss_add_012", "pointmap_head"
    ),
    "baseline_depth": ExtensionCandidateSpec(
        "baseline_depth", "Baseline + depth", 59.6, "c1_auxiliary", "c1_vlm3r_native", "depth_head"
    ),
    "base_vlm_zero_spatial": ExtensionCandidateSpec(
        "base_vlm_zero_spatial", "0 spatial / Base VLM", 56.4, "plain_base"
    ),
}


@dataclass(frozen=True)
class ArchitectureCandidateSpec:
    """A separately labelled pre-SFT architecture extension."""

    identifier: str
    vsi_model: str
    vsi_avg: float | None
    construction: str
    fusion_variant: str
    cut3r_source_layers: tuple[int, ...]
    llm_injection_layers: tuple[int, ...]
    interface_kind: str
    include_mm_projector: bool
    controlled_id: str | None = None
    source_c1_identifier: str = BASELINE_ID
    geometry_activation_name: str | None = None
    requires_geometry: bool = False
    eomt_mode: str | None = None
    auxiliary_only_prefix: str | None = None


ARCHITECTURE_CANDIDATES: dict[str, ArchitectureCandidateSpec] = {}
for _controlled_id, _controlled in CONTROLLED_FUSION_PRE_SFT_SPECS.items():
    _identifier = f"controlled_{_controlled_id.lower()}"
    ARCHITECTURE_CANDIDATES[_identifier] = ArchitectureCandidateSpec(
        identifier=_identifier,
        vsi_model=_controlled.display_name,
        vsi_avg=None,
        construction="controlled_fusion_c1",
        fusion_variant=_controlled.pre_sft_variant,
        cut3r_source_layers=_controlled.cut3r_source_layers,
        llm_injection_layers=_controlled.llm_injection_layers,
        interface_kind="fusion_block",
        include_mm_projector=_controlled_id == "B",
        controlled_id=_controlled_id,
    )

ARCHITECTURE_CANDIDATES.update(
    {
        "geo_rope_fusion": ArchitectureCandidateSpec(
            "geo_rope_fusion", "GeoRope Fusion", 60.2, "geometry_c1", "c1_geo_rope_fusion",
            (12,), (), "fusion_block", True, geometry_activation_name="geo_rope_fusion",
            requires_geometry=True,
        ),
        "visual_geo_rope": ArchitectureCandidateSpec(
            "visual_geo_rope", "Visual geo-RoPE", 57.9, "geometry_c1", "c1_visual_geo_rope",
            (), (), "geometry_aware_projection", True, geometry_activation_name="visual_geo_rope",
            requires_geometry=True, auxiliary_only_prefix="aux_head.",
        ),
        "extra_object_token": ArchitectureCandidateSpec(
            "extra_object_token", "Extra Object token", 58.6, "eomt_c1", "c1_eomt_object",
            (12,), (), "fusion_block", True, eomt_mode="object",
        ),
        "selective_fusion": ArchitectureCandidateSpec(
            "selective_fusion", "selective fusion", 57.2, "eomt_c1", "c1_vlm3r",
            (12,), (), "fusion_block", True, eomt_mode="selective",
        ),
    }
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, (Path, torch.device, torch.dtype)):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    return value


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def candidate_specs(c1_root: Path) -> dict[str, common.CandidateSpec]:
    output: dict[str, common.CandidateSpec] = {}
    for candidate in common.CANDIDATES:
        unit, filename = C1_FILENAMES[candidate.identifier]
        output[candidate.identifier] = replace(
            candidate, calibration_artifact=c1_root / unit / "official" / filename
        )
    return output


def parse_ids(value: str, allowed: tuple[str, ...]) -> list[str]:
    values = [part.strip() for part in value.split(",") if part.strip()]
    unknown = sorted(set(values).difference(allowed))
    if unknown:
        raise ValueError(f"Unknown candidate IDs: {unknown}; expected {list(allowed)}")
    if len(values) != len(set(values)):
        raise ValueError("Candidate IDs must not be repeated")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=(
            "preflight", "smoke", "formal", "extension-smoke", "extension-formal",
            "additional-smoke", "additional-formal",
        ),
        required=True,
    )
    parser.add_argument("--candidates", default=None)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--siglip-model", type=Path, required=True)
    parser.add_argument("--forward-frames-root", type=Path, required=True)
    parser.add_argument("--probe-targets-root", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--data-yaml", type=Path, required=True)
    parser.add_argument("--sample-indices", type=Path, required=True)
    parser.add_argument("--c1-root", type=Path, required=True)
    parser.add_argument("--vsi-csv", type=Path, default=REPO_ROOT / "VSI result.csv")
    parser.add_argument(
        "--controlled-c1-root",
        type=Path,
        default=Path("/home/shuang/c1_artifacts/controlled_fusion_pre_sft_v1/c1"),
    )
    parser.add_argument(
        "--geometry-c1-root",
        type=Path,
        default=Path("/home/shuang/c1_artifacts/c1_geometry_pre_sft_v1"),
    )
    parser.add_argument(
        "--geometry-feature-root",
        type=Path,
        default=Path("/home/shuang/probing_data/cut3r_point_maps_32_v1"),
    )
    parser.add_argument(
        "--eomt-cache-root",
        type=Path,
        default=Path("/home/shuang/probing_data/eomt_consumer_grid_v2"),
    )
    parser.add_argument("--device", default="cuda:0")
    # ``make_load_args`` is shared with the historical evaluator and reads
    # this legacy placement field before ``run_candidate`` asserts the direct
    # one-A100 policy.  It is deliberately fixed to the validated placement;
    # this runner never accepts the historical auto/CPU-offload modes.
    parser.add_argument("--device-map", choices=("cuda:0",), default="cuda:0")
    parser.add_argument("--dtype", choices=("float16",), default="float16")
    parser.add_argument("--attn-implementation", default=None)
    # Retained solely because the shared load-argument constructor carries
    # the historical TITAN-V placement fields.  This A100 runner forces
    # ``device_map=cuda:0`` and never reads these values for dispatch.
    parser.add_argument("--pre-sft-gpu-weight-budget", default=None)
    parser.add_argument("--pre-sft-gpu-weight-budgets", default=None)
    parser.add_argument("--pre-sft-cpu-offload-budget", default=None)
    # ``load_calibration_records`` is also shared with the historical
    # evaluator.  The Baseline protocol fixes this at one minibatch.
    parser.add_argument("--calibration-batches", type=int, choices=(1,), default=1)
    parser.add_argument("--rng-seed", type=int, default=42)
    args = parser.parse_args()
    if args.candidates is None:
        args.candidates = {
            "smoke": BASELINE_ID,
            "formal": ",".join(ALL_IDS),
            "extension-smoke": "ss_depth",
            "extension-formal": ",".join(EXTENSION_IDS),
            "additional-smoke": CONTROLLED_IDS[0],
            "additional-formal": ",".join(ADDITIONAL_IDS),
        }.get(args.mode, BASELINE_ID)
    if args.mode == "preflight":
        allowed = ALL_IDS + EXTENSION_IDS + ADDITIONAL_IDS
    elif args.mode.startswith("extension-"):
        allowed = EXTENSION_IDS
    elif args.mode.startswith("additional-"):
        allowed = ADDITIONAL_IDS
    else:
        allowed = ALL_IDS
    args.candidate_ids = parse_ids(args.candidates, allowed)
    if args.mode == "smoke" and args.candidate_ids != [BASELINE_ID]:
        raise ValueError("The migration smoke must be exactly the C1 VLM3R Baseline")
    if args.mode == "formal" and set(args.candidate_ids) != set(ALL_IDS):
        raise ValueError("The formal run must contain exactly the audited five-candidate roster")
    if args.mode == "extension-smoke" and args.candidate_ids != ["ss_depth"]:
        raise ValueError("The extension smoke must be exactly SS + depth")
    if args.mode == "extension-formal" and set(args.candidate_ids) != set(EXTENSION_IDS):
        raise ValueError("The extension formal run must contain exactly the currently constructible extension roster")
    if args.mode == "additional-smoke" and len(args.candidate_ids) != 1:
        raise ValueError("The additional-architecture smoke must contain exactly one candidate")
    if args.mode == "additional-formal" and not args.candidate_ids:
        raise ValueError("The additional-architecture formal run must contain at least one explicit candidate")
    return args


def required_sidecars(candidate: common.CandidateSpec, feature_root: Path) -> list[Path]:
    scene = "scene0384_00.pt"
    if candidate.fusion_variant == "c1_vlm3r":
        return [feature_root / "scannet" / "spatial_features" / scene]
    return [
        feature_root / "scannet" / "spatial_features_dec_6" / scene,
        feature_root / "scannet" / "spatial_features_dec_9" / scene,
        feature_root / "scannet" / "spatial_features" / scene,
    ]


def c1_source_candidate(
    candidate: common.CandidateSpec | ExtensionCandidateSpec | ArchitectureCandidateSpec,
    specs: dict[str, common.CandidateSpec],
) -> common.CandidateSpec | None:
    if isinstance(candidate, common.CandidateSpec):
        return candidate
    if candidate.source_c1_identifier is None:
        return None
    return specs[candidate.source_c1_identifier]


def controlled_c1_artifact(args: argparse.Namespace, candidate: ArchitectureCandidateSpec) -> Path:
    if candidate.controlled_id is None:
        raise ValueError(f"{candidate.identifier} is not a controlled-fusion candidate")
    return args.controlled_c1_root / candidate.controlled_id / "c1.json"


def geometry_activation_artifact(args: argparse.Namespace, candidate: ArchitectureCandidateSpec) -> Path | None:
    if candidate.geometry_activation_name is None:
        return None
    return args.geometry_c1_root / candidate.geometry_activation_name / "c1_activation.json"


def architecture_c1_artifact(
    args: argparse.Namespace,
    candidate: ArchitectureCandidateSpec,
    specs: dict[str, common.CandidateSpec],
) -> Path:
    if candidate.controlled_id is not None:
        return controlled_c1_artifact(args, candidate)
    return specs[candidate.source_c1_identifier].calibration_artifact


def architecture_schedule(candidate: ArchitectureCandidateSpec) -> tuple[str, str]:
    return (
        ",".join(str(value) for value in candidate.cut3r_source_layers),
        ",".join(str(value) for value in candidate.llm_injection_layers),
    )


def validate_controlled_artifact(
    args: argparse.Namespace,
    candidate: ArchitectureCandidateSpec,
    artifact_path: Path,
) -> dict[str, Any]:
    manifest_path = args.controlled_c1_root / "artifact_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing locked controlled-fusion C1 manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != "controlled_fusion_c1_manifest_v1"
        or manifest.get("post_sft_state_loaded") is not False
        or manifest.get("no_optimizer_step") is not True
    ):
        raise RuntimeError("Controlled-fusion C1 manifest does not prove the no-training pre-SFT contract")
    entry = (manifest.get("artifacts") or {}).get(candidate.controlled_id)
    actual_sha = sha256(artifact_path)
    if not isinstance(entry, dict) or entry.get("sha256") != actual_sha:
        raise RuntimeError(f"{candidate.identifier} C1 SHA-256 does not match the locked manifest")
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    controlled_spec = CONTROLLED_FUSION_PRE_SFT_SPECS[str(candidate.controlled_id)]
    if (
        artifact.get("schema_version") != "c1_calibration_v1"
        or artifact.get("no_training") is not True
        or artifact.get("controlled_fusion") != controlled_fusion_artifact_metadata(controlled_spec)
    ):
        raise RuntimeError(f"{candidate.identifier} is not the canonical no-training controlled C1 artifact")
    return {
        "path": str(artifact_path),
        "sha256": actual_sha,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "artifact_source_git_commit": manifest.get("git_commit"),
        "source_c1_identifier": candidate.controlled_id,
    }


def validate_geometry_activation(
    candidate: ArchitectureCandidateSpec,
    activation_path: Path,
    reference_sha256: str,
) -> dict[str, Any]:
    activation = json.loads(activation_path.read_text(encoding="utf-8"))
    expected_architecture = (
        "geo_rope_fusion" if candidate.identifier == "geo_rope_fusion" else "visual_geo_rope"
    )
    if (
        activation.get("schema_version") != "c1_geometry_activation_v1"
        or activation.get("architecture") != expected_architecture
        or activation.get("reference_c1_artifact_sha256") != reference_sha256
    ):
        raise RuntimeError(f"{candidate.identifier} geometry activation provenance is incompatible")
    achieved = float((activation.get("achieved_ratio") or {}).get("median", float("nan")))
    target = float(activation.get("r0", float("nan")))
    if not math.isfinite(achieved) or not math.isfinite(target) or abs(achieved - target) > 2e-4:
        raise RuntimeError(f"{candidate.identifier} geometry activation did not meet its frozen C1 target")
    return {"path": str(activation_path), "sha256": sha256(activation_path)}


def additional_required_files(
    args: argparse.Namespace,
    candidate: ArchitectureCandidateSpec,
) -> list[Path]:
    scene = "scene0384_00.pt"
    paths: list[Path] = []
    if candidate.cut3r_source_layers:
        paths.append(args.feature_root / "scannet" / "spatial_features" / scene)
    if candidate.requires_geometry:
        paths.append(
            args.geometry_feature_root / "scannet" / "spatial_features_points" / scene
        )
    if candidate.eomt_mode is not None:
        mask_name = "object_masks" if candidate.eomt_mode == "object" else "selective_masks"
        paths.extend(
            [
                args.eomt_cache_root / "validation.json",
                args.eomt_cache_root / "checksums.json",
                args.eomt_cache_root / "class_logits" / "scannet" / scene,
                args.eomt_cache_root / mask_name / "scannet" / scene,
            ]
        )
    return paths


def validate(args: argparse.Namespace, specs: dict[str, common.CandidateSpec]) -> dict[str, Any]:
    required = (
        args.base_model / "config.json", args.siglip_model / "config.json", args.forward_frames_root,
        args.probe_targets_root, args.feature_root, args.data_yaml, args.sample_indices, args.vsi_csv,
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required proxy inputs: {missing}")
    forbidden = [
        path for name in ("adapter_config.json", "adapter_model.bin", "non_lora_trainables.bin")
        for path in args.base_model.rglob(name)
    ]
    if forbidden:
        raise RuntimeError(f"Base model contains forbidden adapter/checkpoint artifacts: {[str(x) for x in forbidden]}")
    with args.vsi_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        vsi_scores = {row["Model"]: float(row["Avg."]) for row in csv.DictReader(handle)}
    for candidate in ARCHITECTURE_CANDIDATES.values():
        if candidate.vsi_avg is None:
            continue
        actual_vsi = vsi_scores.get(candidate.vsi_model)
        if actual_vsi != candidate.vsi_avg:
            raise RuntimeError(
                f"Frozen VSI mismatch for {candidate.identifier}: runner={candidate.vsi_avg}, csv={actual_vsi}"
            )
    artifacts: dict[str, dict[str, Any]] = {}
    for identifier in args.candidate_ids:
        candidate: common.CandidateSpec | ExtensionCandidateSpec | ArchitectureCandidateSpec
        if identifier in ARCHITECTURE_CANDIDATES:
            candidate = ARCHITECTURE_CANDIDATES[identifier]
        elif identifier in EXTENSION_CANDIDATES:
            candidate = EXTENSION_CANDIDATES[identifier]
        else:
            candidate = specs[identifier]
        if isinstance(candidate, ArchitectureCandidateSpec):
            artifact_path = architecture_c1_artifact(args, candidate, specs)
            paths = [artifact_path, *additional_required_files(args, candidate)]
            if candidate.controlled_id is not None:
                paths.append(args.controlled_c1_root / "artifact_manifest.json")
            activation_path = geometry_activation_artifact(args, candidate)
            if activation_path is not None:
                paths.append(activation_path)
            absent = [str(path) for path in paths if not path.is_file()]
            if absent:
                raise FileNotFoundError(f"{identifier} is missing pre-SFT inputs: {absent}")
            if candidate.controlled_id is not None:
                artifacts[identifier] = validate_controlled_artifact(args, candidate, artifact_path)
            else:
                actual = sha256(artifact_path)
                expected = EXPECTED_C1_SHA256[candidate.source_c1_identifier]
                if actual != expected:
                    raise RuntimeError(f"{identifier} reference C1 hash mismatch: {actual} != {expected}")
                artifacts[identifier] = {
                    "path": str(artifact_path),
                    "sha256": actual,
                    "source_c1_identifier": candidate.source_c1_identifier,
                }
            if activation_path is not None:
                artifacts[identifier]["geometry_activation"] = validate_geometry_activation(
                    candidate, activation_path, artifacts[identifier]["sha256"]
                )
            continue
        source = c1_source_candidate(candidate, specs)
        if source is None:
            artifacts[identifier] = {
                "path": None,
                "sha256": None,
                "construction": "plain pretrained base VLM; no spatial sidecar or C1 artifact",
            }
            continue
        paths = [source.calibration_artifact, *required_sidecars(source, args.feature_root)]
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"{identifier} is missing C1/sidecar inputs: {missing}")
        actual = sha256(source.calibration_artifact)
        if actual != EXPECTED_C1_SHA256[source.identifier]:
            raise RuntimeError(f"{identifier} C1 hash mismatch: {actual} != {EXPECTED_C1_SHA256[source.identifier]}")
        payload = json.loads(source.calibration_artifact.read_text(encoding="utf-8"))
        if payload.get("schema_version") != "c1_calibration_v1" or payload.get("no_training") is not True:
            raise RuntimeError(f"{identifier} is not a verified no-training C1 calibration artifact")
        artifacts[identifier] = {
            "path": str(source.calibration_artifact), "sha256": actual, "source_c1_identifier": source.identifier,
        }
    if args.mode != "preflight" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for an A100 proxy execution")
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty output root: {args.output_root}")
    return {
        "base_model": str(args.base_model),
        "base_model_config_sha256": sha256(args.base_model / "config.json"),
        "base_adapter_artifacts": [],
        "c1_artifacts": artifacts,
        "sample_indices": str(args.sample_indices),
        "sample_indices_sha256": sha256(args.sample_indices),
        "vsi_csv": str(args.vsi_csv),
        "vsi_csv_sha256": sha256(args.vsi_csv),
    }


def runtime_metadata(model: torch.nn.Module) -> dict[str, Any]:
    gpu = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        gpu.append({"logical_index": index, "name": properties.name, "total_memory_bytes": int(properties.total_memory)})
    config = getattr(model, "config", None)
    return {
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "gpu": gpu,
        "dtype": "float16",
        "tf32": {
            "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        },
        "attention_implementation": getattr(config, "_attn_implementation", None),
        "gradient_checkpointing": bool(getattr(model, "is_gradient_checkpointing", False)),
        "hf_device_map": jsonable(getattr(model, "hf_device_map", None)),
        "cpu_or_meta_offload": False,
    }


def selected_residency(groups: dict[str, list[torch.nn.Parameter]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, parameters in groups.items():
        devices = sorted({"meta" if parameter.is_meta else str(parameter.device) for parameter in parameters})
        invalid = [parameter for parameter in parameters if parameter.is_meta or parameter.device.type != "cuda"]
        result[name] = {
            "devices": devices,
            "parameter_elements": common.parameter_count(parameters),
            "invalid_parameter_elements": common.parameter_count(invalid),
        }
    return result


def finite_scores(scope: dict[str, Any]) -> bool:
    groups = scope.get("proxy_groups", {})
    return all(math.isfinite(float(groups[name][metric])) for name in groups for metric in ("gradnorm", "snip", "fisher"))


def base_load_args(args: argparse.Namespace, baseline: common.CandidateSpec) -> Any:
    """Reuse the audited loader namespace, changing only to the plain-base mode."""
    cut3r_layers, llm_layers, _artifact = common.candidate_schedule(baseline)
    load_args = common.make_load_args(args, baseline, cut3r_layers, llm_layers)
    load_args.model_label = "pre_sft_base_vlm"
    load_args.model_loading_mode = "pre_sft_base_vlm"
    load_args.pre_sft_fusion_variant = None
    load_args.architecture = "base"
    load_args.feature_preset = "original"
    load_args.zero_spatial_features = False
    return load_args


def architecture_load_args(
    args: argparse.Namespace,
    candidate: ArchitectureCandidateSpec,
    specs: dict[str, common.CandidateSpec],
) -> tuple[Any, dict[str, Any], dict[str, Any] | None]:
    """Build the shared loader Namespace for an explicit architecture extension."""
    artifact_path = architecture_c1_artifact(args, candidate, specs)
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    cut3r_layers, llm_layers = architecture_schedule(candidate)
    # ``install_pre_sft_fusion`` parses these legacy SpatialStack fields before
    # branching to the native fusion implementation.  GeoRoPE, EoMT object,
    # and selective VLM3R consume decoder-12 through their fusion block rather
    # than injecting a SpatialStack residual into an LLM layer, but the parser
    # still requires equal-length lists.  Supply an ignored, explicitly
    # recorded placeholder solely for that parser; the canonical candidate
    # schedule above and the actual non-SpatialStack forward remain unchanged.
    loader_llm_layers = llm_layers
    legacy_schedule_placeholder = None
    if candidate.controlled_id is None and cut3r_layers and not llm_layers:
        loader_llm_layers = ",".join("0" for _ in candidate.cut3r_source_layers)
        legacy_schedule_placeholder = {
            "spatialstack_cut3r_layers": cut3r_layers,
            "spatialstack_llm_layers": loader_llm_layers,
            "used_by_candidate_forward": False,
        }
    adapter = common.CandidateSpec(
        candidate.identifier,
        candidate.identifier,
        candidate.fusion_variant,
        artifact_path,
        candidate.vsi_model,
        float(candidate.vsi_avg) if candidate.vsi_avg is not None else float("nan"),
    )
    load_args = common.make_load_args(args, adapter, cut3r_layers, loader_llm_layers)
    load_args.proxy_legacy_schedule_placeholder = legacy_schedule_placeholder
    load_args.architecture = None
    load_args.feature_preset = "original"
    load_args.spatial_features_subdir = (
        "12:spatial_features" if candidate.controlled_id is not None else "spatial_features"
    )
    load_args.geometry_spatial_features_root = (
        str(args.geometry_feature_root) if candidate.requires_geometry else None
    )
    load_args.geometry_spatial_features_subdir = (
        "spatial_features_points" if candidate.requires_geometry else None
    )
    load_args.geometry_point_map_key = "point_maps_ref"
    load_args.post_sft_architecture = None
    load_args.eomt_consumer_cache_root = str(args.eomt_cache_root)
    load_args.eomt_cache_validation = str(args.eomt_cache_root / "validation.json")
    load_args.verify_eomt_file_checksum = candidate.eomt_mode is not None
    load_args.eomt_selective_kv_gate = candidate.eomt_mode == "selective"
    activation_path = geometry_activation_artifact(args, candidate)
    activation = (
        json.loads(activation_path.read_text(encoding="utf-8"))
        if activation_path is not None else None
    )
    return load_args, artifact, activation


def architecture_interface_module(
    model: torch.nn.Module,
    candidate: ArchitectureCandidateSpec,
) -> torch.nn.Module:
    get_base_model = getattr(model, "get_base_model", None)
    base_model = get_base_model() if hasattr(model, "peft_config") and callable(get_base_model) else model
    base = base_model.get_model()
    if candidate.interface_kind == "fusion_block":
        module = common.fusion_module(model)
    elif candidate.interface_kind == "geometry_aware_projection":
        module = base.get_geometry_aware_projection()
    else:
        raise ValueError(f"Unsupported interface kind: {candidate.interface_kind}")
    if not isinstance(module, torch.nn.Module):
        raise RuntimeError(f"{candidate.identifier} did not construct {candidate.interface_kind}")
    return module


def primary_trainable_groups(
    model: torch.nn.Module,
    *,
    include_fusion: bool,
    interface_module: torch.nn.Module | None = None,
    interface_group_name: str = "fusion_block",
    include_mm_projector: bool = True,
    exclude_interface_prefix: str | None = None,
) -> dict[str, list[torch.nn.Parameter]]:
    """Return only trainable tensors reachable from the primary QA CE loss."""
    lora = [
        parameter for name, parameter in model.named_parameters()
        if ".lora_A." in name or ".lora_B." in name
    ]
    if not lora:
        raise RuntimeError("Fresh SFT LoRA construction produced no lora_A/lora_B parameters")
    get_base_model = getattr(model, "get_base_model", None)
    base_model = get_base_model() if hasattr(model, "peft_config") and callable(get_base_model) else model
    projector = getattr(base_model.get_model(), "mm_projector", None)
    if not isinstance(projector, torch.nn.Module):
        raise RuntimeError("SFT recipe requires a materialized mm_projector")
    groups: dict[str, list[torch.nn.Parameter]] = {"lora": lora}
    if include_fusion:
        module = interface_module if interface_module is not None else common.fusion_module(model)
        fusion = [
            parameter for name, parameter in module.named_parameters()
            if exclude_interface_prefix is None or not name.startswith(exclude_interface_prefix)
        ]
        if not fusion:
            raise RuntimeError("Candidate-specific fusion interface is unexpectedly empty")
        groups[interface_group_name] = fusion
    if include_mm_projector:
        groups["mm_projector"] = list(projector.parameters())
    identities = [id(parameter) for parameters in groups.values() for parameter in parameters]
    if len(identities) != len(set(identities)):
        raise RuntimeError("Primary LoRA/interface/projector scopes must be disjoint")
    if not all(parameters for parameters in groups.values()):
        empty = [name for name, parameters in groups.items() if not parameters]
        raise RuntimeError(f"Primary trainable group is empty: {empty}")
    return groups


def initialize_auxiliary_only_module(
    model: torch.nn.Module,
    module_name: str | None,
    *,
    device: torch.device,
    seed: int,
) -> dict[str, Any] | None:
    """Materialize a recipe auxiliary head without letting it enter ``L_QA``.

    The head is a real candidate module and is retained for provenance, but
    the primary study explicitly uses CE alone.  Turning its corresponding
    supervision switch off is loss selection, not an architecture change:
    neither head is called by the QA forward.
    """
    if module_name is None:
        return None
    common.reset_proxy_rng(seed)
    if module_name == "depth_head":
        module = model.initialize_depth_head(device=device, dtype=torch.float16)
        model.config.use_depth_supervision = False
        loss_flag = "use_depth_supervision"
    elif module_name == "pointmap_head":
        module = model.initialize_pointmap_head(device=device, dtype=torch.float16)
        model.config.use_pointmap_supervision = False
        loss_flag = "use_pointmap_supervision"
    else:
        raise ValueError(f"Unsupported auxiliary-only module: {module_name}")
    parameters = list(module.parameters())
    if not parameters or any(parameter.is_meta or parameter.device.type != "cuda" for parameter in parameters):
        raise RuntimeError(f"Auxiliary-only {module_name} was not fully materialized on CUDA")
    return {
        "module": module_name,
        "parameter_elements": common.parameter_count(parameters),
        "construction_seed": int(seed),
        "devices": sorted({str(parameter.device) for parameter in parameters}),
        "qa_reachable": False,
        "primary_scope_included": False,
        "primary_loss": "L_QA only",
        "loss_switch": loss_flag,
        "loss_switch_value_during_proxy": False,
        "reason": "supervision-only head is not executed by the ordinary QA CE forward",
    }


def existing_auxiliary_only_module(
    module: torch.nn.Module,
    prefix: str | None,
) -> dict[str, Any] | None:
    if prefix is None:
        return None
    parameters = [
        parameter for name, parameter in module.named_parameters() if name.startswith(prefix)
    ]
    if not parameters:
        raise RuntimeError(f"Expected auxiliary-only interface prefix {prefix!r}, but it is empty")
    return {
        "module": prefix.rstrip("."),
        "parameter_elements": common.parameter_count(parameters),
        "devices": sorted({"meta" if p.is_meta else str(p.device) for p in parameters}),
        "qa_reachable": False,
        "primary_scope_included": False,
        "primary_loss": "L_QA only",
        "reason": "architecture auxiliary head is not reached by ordinary QA CE",
    }


def configure_architecture_runtime(
    model: torch.nn.Module,
    candidate: ArchitectureCandidateSpec,
    artifact: dict[str, Any],
    activation: dict[str, Any] | None,
    *,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any] | None, dict[str, Any] | None]:
    interface = architecture_interface_module(model, candidate)
    interface.to(device=device, dtype=torch.float16)
    apply_c1_calibration_artifact(model, artifact)
    if activation is not None:
        apply_geometry_c1_calibration_artifact(model, activation)
    selective_settings = None
    if candidate.eomt_mode == "selective":
        from llava.model.multimodal_eomt import configure_selective_kv_gate

        selective_settings = configure_selective_kv_gate(model.get_model().config, enabled=True)
    if candidate.identifier == "visual_geo_rope":
        # The canonical architecture contains an auxiliary geometry head, but
        # the shared primary comparison is strictly L_QA.
        model.config.use_auxiliary_geometry_loss = False
    auxiliary = existing_auxiliary_only_module(interface, candidate.auxiliary_only_prefix)
    return interface, auxiliary, selective_settings


def add_architecture_batch_inputs(
    args: argparse.Namespace,
    load_args: Any,
    candidate: ArchitectureCandidateSpec,
    record: dict[str, Any],
    batch: dict[str, Any],
    device: torch.device,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    evidence: dict[str, Any] = {
        "geometry_spatial_features": False,
        "geometry_point_map_shape": None,
    }
    if candidate.requires_geometry:
        if "geometry_spatial_features" not in batch:
            raise RuntimeError(
                f"{candidate.identifier} requires its separate full-frame geometry sidecar"
            )
        geometry_point_maps = full_geometry_point_maps(
            batch["geometry_spatial_features"],
            load_args.geometry_point_map_key,
            expected_frames=32,
        )
        evidence["geometry_spatial_features"] = True
        evidence["geometry_point_map_shape"] = list(geometry_point_maps.shape)
        # This is the same adapter contract used by the validated geometry
        # feature extractor: Visual geo-RoPE consumes the full point map as
        # visual-token positions, while GeoRoPE fusion reads the separately
        # retained geometry_spatial_features payload.
        if candidate.identifier == "visual_geo_rope":
            batch["point_maps"] = geometry_point_maps
    if "geometry_spatial_features" in batch:
        batch["geometry_spatial_features"] = common.move_value(
            batch["geometry_spatial_features"], device
        )
    if "point_maps" in batch:
        batch["point_maps"] = common.move_value(batch["point_maps"], device)
    eomt_payload = None
    if candidate.eomt_mode is not None:
        eomt_payload = load_eomt_consumer_cache(load_args, record)
        if eomt_payload is None:
            raise RuntimeError(f"{candidate.identifier} did not resolve its required EoMT cache")
        batch["eomt_cached_outputs"] = [eomt_payload]
        evidence["eomt_cache_scene"] = eomt_payload.get("scene_id")
    return eomt_payload, evidence


def architecture_forward_evidence(
    model: torch.nn.Module,
    candidate: ArchitectureCandidateSpec,
    batch_evidence: dict[str, Any],
) -> dict[str, Any]:
    """Prove architecture-specific inputs were exercised by the shared QA forward."""
    evidence = dict(batch_evidence)
    if candidate.eomt_mode == "object":
        debug = getattr(model, "_last_eomt_object_debug", None)
        if not isinstance(debug, list) or not debug:
            raise RuntimeError("Extra Object Token forward did not consume cached EoMT outputs")
        selected = sum(int(item.get("selected_count", 0)) for item in debug)
        inserted = sum(int(item.get("object_block_token_count", 0)) for item in debug)
        if selected <= 0 or inserted <= 0:
            raise RuntimeError("Extra Object Token forward selected no QA-reachable object tokens")
        evidence["eomt_forward_debug_entries"] = len(debug)
        evidence["eomt_selected_objects"] = selected
        evidence["eomt_inserted_tokens"] = inserted
    elif candidate.eomt_mode == "selective":
        debug = getattr(model, "_last_eomt_selective_debug", None)
        if not isinstance(debug, list) or len(debug) != 32:
            raise RuntimeError("Selective fusion forward did not execute its 32-frame cached gate")
        evidence["eomt_forward_debug_entries"] = len(debug)
    return evidence


def run_candidate(
    args: argparse.Namespace,
    candidate: common.CandidateSpec | ExtensionCandidateSpec | ArchitectureCandidateSpec,
    specs: dict[str, common.CandidateSpec],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    architecture = candidate if isinstance(candidate, ArchitectureCandidateSpec) else None
    source = c1_source_candidate(candidate, specs)
    extension = candidate if isinstance(candidate, ExtensionCandidateSpec) else None
    architecture_activation = None
    if architecture is not None:
        load_args, artifact, architecture_activation = architecture_load_args(
            args, architecture, specs
        )
    elif source is None:
        load_args = base_load_args(args, specs[BASELINE_ID])
        artifact = None
    else:
        cut3r_layers, llm_layers, artifact = common.candidate_schedule(source)
        load_args = common.make_load_args(args, source, cut3r_layers, llm_layers)
    # Direct one-A100 placement.  Never activate the TITAN-V auto-map or CPU
    # offload/deferred dispatch path for this protocol.
    if load_args.device_map != "cuda:0":
        raise RuntimeError(f"A100 proxy requires direct cuda:0 placement, got {load_args.device_map!r}")
    load_args.pre_sft_defer_dispatch = False
    device = torch.device(args.device)
    common.reset_proxy_rng(args.rng_seed)
    tokenizer, model, image_processor = load_model(load_args, device, torch.float16)
    try:
        existing_lora = [name for name, _ in model.named_parameters() if "lora_" in name]
        if existing_lora:
            raise RuntimeError("Plain base unexpectedly contains existing LoRA parameters")
        # The shared loader constructs the candidate fusion module after the
        # pretrained checkpoint dispatch, so this *new* C1 module otherwise
        # retains PyTorch's CPU construction device.  Move precisely that
        # intended SFT-trainable module to the direct A100 placement before
        # C1 calibration; do not alter any pretrained model weights or the
        # frozen-backbone placement.
        interface_module = None
        architecture_auxiliary = None
        selective_settings = None
        if architecture is not None:
            interface_module, architecture_auxiliary, selective_settings = configure_architecture_runtime(
                model,
                architecture,
                artifact,
                architecture_activation,
                device=device,
            )
        elif source is not None:
            common.fusion_module(model).to(device=device, dtype=torch.float16)
            apply_c1_calibration_artifact(model, artifact)
        auxiliary_only = initialize_auxiliary_only_module(
            model,
            extension.auxiliary_only_module if extension is not None else None,
            device=device,
            seed=args.rng_seed,
        )
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        model, lora_recipe = common.attach_intended_sft_lora(model, seed=args.rng_seed)
        lora_initialization = common.lora_initialization_summary(model)
        groups = primary_trainable_groups(
            model,
            include_fusion=source is not None or architecture is not None,
            interface_module=interface_module,
            interface_group_name=(
                "geometry_aware_projection"
                if architecture is not None and architecture.interface_kind == "geometry_aware_projection"
                else "fusion_block"
            ),
            include_mm_projector=(
                architecture.include_mm_projector if architecture is not None else True
            ),
            exclude_interface_prefix=(
                architecture.auxiliary_only_prefix if architecture is not None else None
            ),
        )
        selected = common.configure_intended_sft_trainable_parameters(model, groups)
        residency = selected_residency(groups)
        if any(info["invalid_parameter_elements"] for info in residency.values()):
            raise RuntimeError(f"Intended trainable parameters are not fully CUDA resident: {residency}")
        if {id(parameter) for parameter in selected} != {id(parameter) for values in groups.values() for parameter in values}:
            raise RuntimeError("Selected scope does not equal disjoint intended SFT group union")
        if candidate.identifier == BASELINE_ID:
            actual_counts = {name: common.parameter_count(parameters) for name, parameters in groups.items()}
            actual_counts["total"] = common.parameter_count(selected)
            if actual_counts != EXPECTED_BASELINE_COUNTS:
                raise RuntimeError(f"Baseline trainable count gate failed: {actual_counts} != {EXPECTED_BASELINE_COUNTS}")
        install_forward_frame_loader(args.forward_frames_root)
        dataset, collator, by_video = build_dataset(load_args, tokenizer, image_processor)
        records = common.load_calibration_records(args, by_video)
        if len(records) != 1:
            raise RuntimeError(f"Expected exactly one calibration minibatch, got {len(records)}")
        record = records[0]
        batch = common.prepare_batch(dataset, collator, by_video[str(record["video_path"])], model, device, torch.float16)
        eomt_payload = None
        architecture_input_evidence: dict[str, Any] = {}
        if architecture is not None:
            eomt_payload, architecture_input_evidence = add_architecture_batch_inputs(
                args, load_args, architecture, record, batch, device
            )
        if source is None:
            forbidden = [key for key in ("spatial_features", "point_maps", "geometry_spatial_features") if key in batch]
            if forbidden:
                raise RuntimeError(f"Plain Base VLM received forbidden spatial inputs: {forbidden}")
        batch_info = common.batch_metadata(batch, record)
        if batch_info.get("scene_id") != "scene0384_00" or batch_info.get("input_ids_shape") != [1, 424] or batch_info.get("supervised_label_tokens") != 13:
            raise RuntimeError(f"Fixed calibration contract failed: {batch_info}")
        started = time.perf_counter()
        scope = common.run_grouped_backward_scope(model, batch, "sft_trainable", groups, rng_seed=args.rng_seed)
        wall_seconds = time.perf_counter() - started
        scope["requires_grad_flags_restored"] = bool(getattr(model, "_last_proxy_requires_grad_restored", False))
        scope["selected_parameter_elements"] = common.parameter_count(selected)
        if scope["status"] != "PASS":
            raise RuntimeError(f"{candidate.identifier} primary scope did not pass: {scope['status']}")
        if not finite_scores(scope):
            raise RuntimeError(f"{candidate.identifier} produced non-finite proxy scores")
        architecture_execution_evidence = (
            architecture_forward_evidence(model, architecture, architecture_input_evidence)
            if architecture is not None else None
        )
        runtime = runtime_metadata(model)
        runtime["cpu_or_meta_offload"] = any(info["invalid_parameter_elements"] for info in residency.values())
        return {
            "schema_version": SCHEMA_VERSION,
            "candidate": {
                **asdict(candidate),
                "calibration_artifact": (
                    str(architecture_c1_artifact(args, architecture, specs))
                    if architecture is not None
                    else str(source.calibration_artifact) if source is not None else None
                ),
                "c1_artifact_sha256": provenance["c1_artifacts"][candidate.identifier]["sha256"],
                "post_sft_weights_loaded": False,
            },
            "provenance": provenance,
            "runtime": runtime,
            "calibration": {
                **batch_info,
                "sample_indices": str(args.sample_indices),
                "sample_indices_sha256": provenance["sample_indices_sha256"],
                "calibration_minibatches": 1,
                "loss_definition": "L_proxy = L_QA: ordinary supervised causal-LM next-token cross entropy",
            },
            "lora_initialization": {
                "peft_version": package_version("peft"),
                "recipe": jsonable(lora_recipe),
                "actual_state": lora_initialization,
            },
            "scope_definition": (
                " + ".join(groups)
                if source is not None or architecture is not None
                else "fresh LoRA + mm_projector (plain Base VLM; no spatial interface)"
            ),
            "selected_residency": residency,
            "auxiliary_only": architecture_auxiliary or auxiliary_only,
            "architecture_runtime": {
                "interface_kind": architecture.interface_kind if architecture is not None else None,
                "mm_projector_in_primary_scope": architecture.include_mm_projector if architecture is not None else True,
                "geometry_activation": provenance["c1_artifacts"][candidate.identifier].get("geometry_activation"),
                "eomt_mode": architecture.eomt_mode if architecture is not None else None,
                "eomt_cache_scene": eomt_payload.get("scene_id") if eomt_payload is not None else None,
                "eomt_selective_settings": selective_settings,
                "legacy_loader_schedule_placeholder": (
                    getattr(load_args, "proxy_legacy_schedule_placeholder", None)
                ),
                "forward_evidence": architecture_execution_evidence,
            },
            "primary_scope": scope,
            "wall_clock_seconds": wall_seconds,
            "no_training": {"optimizer_constructed": False, "optimizer_step_called": False, "parameter_updates": False},
        }
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def rankdata(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    rank = np.empty(array.size, dtype=np.float64)
    rank[order] = np.arange(array.size, dtype=np.float64)
    _, inverse, counts = np.unique(array, return_inverse=True, return_counts=True)
    sums = np.zeros(counts.size, dtype=np.float64)
    np.add.at(sums, inverse, rank)
    return sums[inverse] / counts[inverse]


def spearman(left: list[float], right: list[float]) -> float:
    if len(left) < 3 or len(set(left)) < 2 or len(set(right)) < 2:
        return float("nan")
    return float(np.corrcoef(rankdata(left), rankdata(right))[0, 1])


def kendall_tau_b(left: list[float], right: list[float]) -> float:
    concordant = discordant = ties_left = ties_right = 0
    for first in range(len(left)):
        for second in range(first + 1, len(left)):
            dx, dy = left[first] - left[second], right[first] - right[second]
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                ties_left += 1
            elif dy == 0:
                ties_right += 1
            elif (dx > 0) == (dy > 0):
                concordant += 1
            else:
                discordant += 1
    denominator = math.sqrt((concordant + discordant + ties_left) * (concordant + discordant + ties_right))
    return float("nan") if denominator == 0 else (concordant - discordant) / denominator


def pairwise_accuracy(left: list[float], right: list[float]) -> float:
    correct = total = 0
    for first in range(len(left)):
        for second in range(first + 1, len(left)):
            dx, dy = left[first] - left[second], right[first] - right[second]
            if dx == 0 or dy == 0:
                continue
            total += 1
            correct += int((dx > 0) == (dy > 0))
    return float("nan") if total == 0 else correct / total


def flattened(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        groups = result["primary_scope"]["proxy_groups"]
        total = groups["total"]
        peak = max((item["peak_allocated_bytes"] for item in result["primary_scope"]["peak_gpu_memory"]), default=0)
        row = {
            "candidate": result["candidate"]["identifier"],
            "display_name": result["candidate"]["vsi_model"],
            "vsi_avg": result["candidate"]["vsi_avg"],
            "selected_params": total["parameter_elements"],
            "gradient_covered_params": total["parameters_with_gradient"],
            "ce_loss": result["primary_scope"]["loss"],
            "gradnorm": total["gradnorm"], "snip": total["snip"], "fisher": total["fisher"],
            "peak_vram_bytes": peak, "runtime_seconds": result["wall_clock_seconds"],
        }
        for group, values in groups.items():
            for metric in ("parameter_elements", "parameters_with_gradient", "gradnorm", "snip", "fisher"):
                row[f"{group}_{metric}"] = values[metric]
        rows.append(row)
    return rows


def analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scored_rows = [row for row in rows if row.get("vsi_avg") is not None]
    vsi = [float(row["vsi_avg"]) for row in scored_rows]
    metrics: dict[str, Any] = {}
    for name in ("gradnorm", "snip", "fisher"):
        values = [float(row[name]) for row in scored_rows]
        metrics[name] = {
            "spearman_rho": spearman(values, vsi), "kendall_tau_b": kendall_tau_b(values, vsi),
            "pairwise_ordering_accuracy": pairwise_accuracy(values, vsi),
            "proxy_ranking_descending": [row["candidate"] for row in sorted(scored_rows, key=lambda row: row[name], reverse=True)],
        }
    return {
        "label": f"preliminary/pilot statistics; n={len(scored_rows)} with frozen VSI values",
        "unscored_candidates": [row["candidate"] for row in rows if row.get("vsi_avg") is None],
        "vsi_ranking_descending": [row["candidate"] for row in sorted(scored_rows, key=lambda row: row["vsi_avg"], reverse=True)],
        "metrics": metrics,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)


def write_summary(path: Path, rows: list[dict[str, Any]], ranking: dict[str, Any] | None) -> None:
    lines = ["# A100 pre-SFT primary trainable-scope proxy", "", "One fixed QA minibatch, one forward/backward per candidate, and no optimizer/update.", "", "| Candidate | Selected params | Gradient-covered params | CE loss | GradNorm | SNIP | Fisher | Peak VRAM bytes | Runtime s |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['display_name']} | {row['selected_params']} | {row['gradient_covered_params']} | {row['ce_loss']:.8g} | {row['gradnorm']:.8g} | {row['snip']:.8g} | {row['fisher']:.8g} | {row['peak_vram_bytes']} | {row['runtime_seconds']:.3f} |")
    if ranking is not None:
        lines.extend(["", f"## {ranking['label']}", "", f"VSI ranking: `{', '.join(ranking['vsi_ranking_descending'])}`", "", "| Proxy | Spearman rho | Kendall tau-b | Pairwise ordering accuracy | Proxy ranking |", "|---|---:|---:|---:|---|"])
        for name, values in ranking["metrics"].items():
            lines.append(f"| {name} | {values['spearman_rho']:.6g} | {values['kendall_tau_b']:.6g} | {values['pairwise_ordering_accuracy']:.6g} | {', '.join(values['proxy_ranking_descending'])} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    # The migration gate preserves the audited fp16/TF32-off numerical path.
    # Set these before any CUDA model construction and record them per run.
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    specs = candidate_specs(args.c1_root)
    provenance = validate(args, specs)
    provenance.update({
        "git_commit": git_commit(), "rng_seed": args.rng_seed,
        "environment": {"python": sys.version, "torch": torch.__version__, "cuda_runtime": torch.version.cuda, "transformers": package_version("transformers"), "peft": package_version("peft"), "accelerate": package_version("accelerate")},
        "requested": {"mode": args.mode, "candidate_ids": args.candidate_ids, "device": args.device, "dtype": args.dtype, "attn_implementation": args.attn_implementation},
    })
    args.output_root.mkdir(parents=True, exist_ok=False)
    if args.mode == "preflight":
        write_json(args.output_root / "preflight.json", {"schema_version": SCHEMA_VERSION, "status": "PASS", "provenance": provenance})
        print(json.dumps({"status": "PASS", "output_root": str(args.output_root)})); return
    results: list[dict[str, Any]] = []
    for identifier in args.candidate_ids:
        candidate: common.CandidateSpec | ExtensionCandidateSpec | ArchitectureCandidateSpec
        if identifier in ARCHITECTURE_CANDIDATES:
            candidate = ARCHITECTURE_CANDIDATES[identifier]
        elif identifier in EXTENSION_CANDIDATES:
            candidate = EXTENSION_CANDIDATES[identifier]
        else:
            candidate = specs[identifier]
        result = run_candidate(args, candidate, specs, provenance)
        results.append(result)
        candidate_dir = args.output_root / "per_candidate" / identifier
        candidate_dir.mkdir(parents=True, exist_ok=False)
        write_json(candidate_dir / "provenance.json", result)
    rows = flattened(results)
    ranking = analysis(rows) if args.mode in {"formal", "extension-formal", "additional-formal"} else None
    payload = {"schema_version": SCHEMA_VERSION, "status": "PASS", "provenance": provenance, "results": results, "ranking_analysis": ranking}
    write_json(args.output_root / "results.json", payload)
    write_csv(args.output_root / "results.csv", rows)
    write_summary(args.output_root / "summary.md", rows, ranking)
    print(json.dumps({"status": "PASS", "output_root": str(args.output_root), "candidates": args.candidate_ids}))


if __name__ == "__main__":
    main()
