#!/usr/bin/env python
"""Run the normal extractor with an auditable loss-only config toggle.

The toggle is applied only after the verified pre-SFT model is constructed.
It is used on a one-video numerical equivalence smoke; no trained auxiliary
head or post-SFT state is loaded.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.probing import extract_depth_probe_features as extractor


def forwarded_value(arguments: list[str], flag: str) -> str:
    try:
        return arguments[arguments.index(flag) + 1]
    except (ValueError, IndexError) as exc:
        raise RuntimeError(f"Attestation extractor requires {flag}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--attestation-loss", choices=("depth", "pointmap"), required=True)
    known, forwarded = parser.parse_known_args()
    original_load_model = extractor.load_model
    runtime_attestation: dict[str, Any] = {}

    def load_model_with_toggle(*args: Any, **kwargs: Any) -> Any:
        result = original_load_model(*args, **kwargs)
        model = result[1]
        config = model.get_model().config
        if known.attestation_loss == "depth":
            config.use_depth_supervision = True
            config.depth_head_source = "llm_output"
            config.lambda_depth = 0.05
            auxiliary_head = model.initialize_depth_head(
                device=next(model.parameters()).device,
                dtype=next(model.parameters()).dtype,
            )
        else:
            config.use_pointmap_supervision = True
            config.pointmap_head_source = "llm_output"
            config.lambda_pointmap = 0.1
            auxiliary_head = model.initialize_pointmap_head(
                device=next(model.parameters()).device,
                dtype=next(model.parameters()).dtype,
            )
        auxiliary_head.eval()
        runtime_attestation.update({
            "loss": known.attestation_loss,
            "auxiliary_head_class": type(auxiliary_head).__name__,
            "auxiliary_head_parameters": sum(parameter.numel() for parameter in auxiliary_head.parameters()),
            "auxiliary_head_freshly_initialized": True,
        })
        return result

    extractor.load_model = load_model_with_toggle
    sys.argv = [extractor.__file__, *forwarded]
    extractor.main()

    output_root = Path(forwarded_value(forwarded, "--output-root")).resolve()
    model_label = forwarded_value(forwarded, "--model-label")
    provenance_path = output_root / "features" / model_label / "extraction_provenance.json"
    payload = json.loads(provenance_path.read_text(encoding="utf-8"))
    payload["loss_only_forward_equivalence_attestation"] = {
        "loss": known.attestation_loss,
        "config_override_after_verified_pre_sft_construction": {
            "use_depth_supervision": known.attestation_loss == "depth",
            "depth_head_source": "llm_output" if known.attestation_loss == "depth" else None,
            "lambda_depth": 0.05 if known.attestation_loss == "depth" else None,
            "use_pointmap_supervision": known.attestation_loss == "pointmap",
            "pointmap_head_source": "llm_output" if known.attestation_loss == "pointmap" else None,
            "lambda_pointmap": 0.1 if known.attestation_loss == "pointmap" else None,
        },
        "no_optimizer": True,
        "no_post_sft_state": True,
        "fresh_auxiliary_head_required": True,
        **runtime_attestation,
    }
    provenance_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
