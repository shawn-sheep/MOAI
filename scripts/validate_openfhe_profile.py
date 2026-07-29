#!/usr/bin/env python3
"""Validate every frozen effective paper_compat OpenFHE CryptoProfile payload."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "config" / "paper_compat.json"
DEFAULT_FEATURE_CONFIG = REPO_ROOT / "config" / "paper_compat_feature_packed.json"
EXPECTED_M3 = {
    "hash": "965ded1a76b064c738f55b87b151df60c605d62eaeaad55f6cc753e149f8af59",
    "bootstrap_slots": 16,
    "bootstrap_precision": 10,
    "levels_available_after_bootstrap": 12,
    "multiplicative_depth": 31,
}
EXPECTED_FEATURE = {
    "hash": "94f30e628e21f02146ce7ed9820194eabba3820f6e1e17176a31f8c5acf8b0be",
    "bootstrap_slots": 1024,
    "bootstrap_precision": 14,
    "levels_available_after_bootstrap": 28,
    "multiplicative_depth": 47,
}
EXPECTED_FEATURE_SOFTMAX_OVERRIDE = {
    "scope": (
        "M4/M5 1024-slot multi-head path only; the M3 generic Softmax contract "
        "is unchanged"
    ),
    "exponential_coefficient_sha256": (
        "6eda4377151897e8c4ca4d72f2a918db0b888fc6771f8ee5cf1d950b378de76f"
    ),
    "exponential_at_zero": 1.000000000011056,
    "inactive_slot_cleanup": (
        "subtract the frozen P_exp(0) plaintext only from public inactive slots"
    ),
    "pre_bootstrap_required_depth": 6,
    "post_bootstrap_required_depth": 11,
    "attention_output_cleanup": (
        "native bootstrap followed by one public 768-prefix Ct-Pt mask"
    ),
    "gate": {
        "inactive_max_absolute": 1e-6,
        "threshold_policy": "may tighten; never silently relax",
    },
}
EXPECTED_WARNING = (
    "Research reproduction parameters only. Do not claim 128-bit security."
)
EXPECTED_FEATURE_VALIDATION_STATUS = (
    "five-token layer-1 server-only encoder gate passed; 12-layer encoder gate required"
)


def canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--feature-config", type=Path, default=DEFAULT_FEATURE_CONFIG
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    with arguments.config.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if (
        config.get("profile_id") != "paper_compat"
        or config.get("security_claim") != "none"
        or config.get("warning") != EXPECTED_WARNING
    ):
        raise RuntimeError("paper_compat identity/security contract drifted")
    with arguments.feature_config.open("r", encoding="utf-8") as handle:
        feature_config = json.load(handle)
    if (
        feature_config.get("runtime_id") != "paper_compat_feature_packed_m4_m5"
        or feature_config.get("profile_id") != "paper_compat"
        or feature_config.get("security_claim") != "none"
        or feature_config.get("warning") != EXPECTED_WARNING
        or feature_config.get("validation_status")
        != EXPECTED_FEATURE_VALIDATION_STATUS
    ):
        raise RuntimeError("feature-packed paper_compat identity/security contract drifted")
    if (
        feature_config.get("feature_packed_softmax_override")
        != EXPECTED_FEATURE_SOFTMAX_OVERRIDE
    ):
        raise RuntimeError("feature-packed Softmax override contract drifted")

    runtime_sources = {
        "m3_smoke_effective_runtime": (
            config["openfhe_translation"]["m3_smoke_effective_runtime"],
            EXPECTED_M3,
            config["openfhe_translation"]["m3_smoke_effective_runtime"][
                "parameters"
            ],
        ),
        "m4_m5_feature_packed_effective_runtime": (
            feature_config,
            EXPECTED_FEATURE,
            feature_config["effective_profile"],
        ),
    }
    calculated_hashes: dict[str, str] = {}
    for runtime_name, (runtime, expected, parameters) in runtime_sources.items():
        payload = canonical_json(parameters)
        calculated = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        if (
            runtime.get("canonical_encoding")
            != "UTF-8 JSON with lexicographically sorted keys and compact separators"
            or runtime["effective_profile_sha256"] != expected["hash"]
            or calculated != expected["hash"]
            or parameters.get("bootstrap_slots") != expected["bootstrap_slots"]
            or parameters.get("bootstrap_precision") != expected["bootstrap_precision"]
            or parameters.get("levels_available_after_bootstrap")
            != expected["levels_available_after_bootstrap"]
            or parameters.get("multiplicative_depth")
            != expected["multiplicative_depth"]
            or parameters.get("security_claim") != "none"
            or parameters.get("security_level") != "HEStd_NotSet"
        ):
            raise RuntimeError(
                f"{runtime_name} effective CryptoProfile drifted: "
                f"frozen={runtime['effective_profile_sha256']} calculated={calculated}"
            )
        calculated_hashes[runtime_name] = calculated
    print(
        json.dumps(
            {
                "test": "validate_openfhe_profile",
                "profile": "paper_compat",
                "security_claim": "none",
                "parameter_sha256": calculated_hashes[
                    "m4_m5_feature_packed_effective_runtime"
                ],
                "effective_profiles": calculated_hashes,
                "passed": True,
            },
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"validate_openfhe_profile failed: {error}", file=sys.stderr)
        sys.exit(1)
