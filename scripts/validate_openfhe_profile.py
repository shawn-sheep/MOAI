#!/usr/bin/env python3
"""Validate every frozen effective paper_compat OpenFHE CryptoProfile payload."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "config" / "paper_compat.json"
DEFAULT_FEATURE_CONFIG = REPO_ROOT / "config" / "paper_compat_feature_packed.json"
DEFAULT_APPROXIMATION_CONFIG = REPO_ROOT / "config" / "openfhe_approximations.json"
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
EXPECTED_M5_PROTOTYPE_ACCEPTANCE = {
    "contract_id": "moai_observability_compatible_prototype_v1",
    "status": "user_authorized_2026-08-01",
    "scope": (
        "M5 exact-three and full-12 encrypted encoder runtime only; M2 packing, "
        "M3 nonlinear, and M4 single-layer gates remain unchanged"
    ),
    "legacy_basis": (
        "legacy MOAI has no inactive/cross-lane numerical assertion and observes "
        "only active slots; 1e-3 is a new prototype engineering bound, not a legacy "
        "MOAI threshold"
    ),
    "thresholds": {
        "per_layer_relative_l2_max": 0.05,
        "per_layer_cosine_min": 0.99,
        "final_relative_l2_max": 0.05,
        "final_cosine_min": 0.99,
        "inactive_or_cross_lane_max_absolute": 1e-3,
        "finite_required": True,
    },
    "active_quality_policy": "unchanged from the original M5 contract",
    "sentinel_range_policy": (
        "registered polynomial intervals and inactive guard range checks remain "
        "unchanged"
    ),
    "claim_boundary": (
        "prototype_only; MOAI-observability-compatible; not strict numerical parity "
        "and not evidence that 1e-3 was used by legacy MOAI"
    ),
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
        "scope": (
            "M4 single-layer and standalone FeaturePacked regression only; M5 "
            "exact-three/full-12 is governed by /m5_prototype_acceptance"
        ),
        "inactive_max_absolute": 1e-6,
        "threshold_policy": "may tighten; never silently relax",
    },
}
EXPECTED_FEATURE_LAYERNORM_OVERRIDE = {
    "scope": "M4/M5 FeaturePacked LayerNorm only; generic M3 LayerNorm is unchanged",
    "fixed_shape": (
        "2 sites x 12 layers x 5 public token rows; 768 active features in 1024 slots"
    ),
    "trace_scale_contract": {
        "source_path": "config/openfhe_approximations.json",
        "json_locator": "operators.layernorm.feature_packed_trace_scale_contract",
        "contract_id": "layernorm_layer_token_power2_scale_v1",
        "contract_sha256": (
            "b493b032e461e15d7436efe7fff5948436afa8e302646daa97db78fa1d59be3e"
        ),
        "values_sha256": (
            "9436f05ce80b427de47700d924869b0dd6f13cc515e586446fcd2dc56faec28a"
        ),
        "raw_variance_sha256": (
            "940f92de81915c2121b427b72827e4ed87d072fda0847266d0cb5521d60f93b6"
        ),
    },
    "scale_selection": (
        "pre-registered D=2^roundTiesToEven(log2(64/raw_population_variance)) "
        "selected only by public site/layer/token coordinates"
    ),
    "runtime_activation_dependency": "none",
    "active_epsilon_placement": (
        "public epsilon is multiplied by the selected D and added to the active "
        "768-slot prefix before native bootstrap"
    ),
    "bootstrap": "OpenFHE native CKKS bootstrap",
    "bootstrap_preconditioner": 2048,
    "pre_bootstrap_values": ("active u/2048 and inactive public guard 1/2048"),
    "post_bootstrap_restore": (
        "one uniform all-1024-slot public multiplication by 2048 on each ciphertext"
    ),
    "post_inverse_cleanup": (
        "one public first-768-active/last-256-zero Ct-Pt mask and explicit rescale "
        "before the centered Ct-Ct merge"
    ),
    "gamma_application": (
        "per-token public gamma*sqrt(D) Ct-Pt multiplication on the centered "
        "branch before the inverse-sqrt merge"
    ),
    "sites_per_encoder_layer": 2,
    "ciphertexts_per_site": 5,
    "post_bootstrap_required_depth": 12,
    "schedule_status": (
        "two-layer live M5 candidate recorded but unsealed; exact-three, live M4 "
        "regression, and 12-layer MOAI-observability-compatible prototype gates "
        "remain required"
    ),
    "gate": {
        "scope": (
            "M4 single-layer and standalone FeaturePacked regression only; M5 "
            "exact-three/full-12 is governed by /m5_prototype_acceptance"
        ),
        "normalized_variance_interval": [0.5, 1536.0],
        "inactive_guard_in_interval_required": True,
        "inactive_guard_deviation_from_one": "diagnostic_only",
        "output_inactive_max_absolute": 1e-6,
        "threshold_policy": "may tighten; never silently relax",
    },
}
EXPECTED_M5_SCHEDULE_CANDIDATE = {
    "status": "candidate_unsealed",
    "source": (
        "reviewed live two-layer metadata calibration output; no sealed artifact "
        "identity claimed"
    ),
    "formal_schedule_sealed": False,
    "calibrated_layers": [0, 1],
    "metadata_tuple_order": ["level", "noise_scale_degree", "remaining_levels"],
    "used_level_delta_order": [
        "input_to_softmax_checkpoint_net_recovered",
        "softmax_checkpoint_to_attention_output_consumed",
        "attention_output_to_ln1_checkpoint_net_recovered",
        "ln1_checkpoint_to_ln1_output_consumed",
        "ln1_output_to_ffn_output_consumed",
        "ffn_output_to_ln2_checkpoint_net_recovered",
        "ln2_checkpoint_to_raw_output_consumed",
        "previous_raw_output_to_input_recovered",
    ],
    "layer_regimes": {
        "layer0": {
            "observed_layer_ids": [0],
            "checkpoint_metadata": {
                "input": [29, 1, 18],
                "softmax_denominator": [18, 2, 28],
                "attention_output": [40, 2, 6],
                "ln1_variance": [19, 2, 27],
                "ln1_output": [32, 2, 14],
                "ffn_output": [45, 2, 1],
                "ln2_variance": [19, 2, 27],
                "raw_output": [30, 2, 16],
            },
            "used_level_deltas": [10, 22, 21, 13, 13, 26, 11, None],
        },
        "later_layers": {
            "observed_layer_ids": [1],
            "candidate_layer_ids": list(range(1, 12)),
            "checkpoint_metadata": {
                "input": [19, 2, 27],
                "softmax_denominator": [18, 2, 28],
                "attention_output": [31, 2, 15],
                "ln1_variance": [19, 2, 27],
                "ln1_output": [30, 2, 16],
                "ffn_output": [43, 2, 3],
                "ln2_variance": [19, 2, 27],
                "raw_output": [30, 2, 16],
            },
            "used_level_deltas": [1, 13, 12, 11, 13, 24, 11, 11],
        },
    },
    "per_encoder_layer_operation_counts": {
        "rotations": 6300,
        "ct_pt_multiplications": 51885,
        "ct_ct_multiplications": 95,
        "explicit_rescale_requests": 810,
        "chebyshev_evaluations": 55,
        "estimated_polynomial_multiplications": 1150,
        "bootstraps": 25,
        "bootstrap_iterations": 50,
    },
    "per_inter_layer_refresh_operation_counts": {
        "rotations": 0,
        "ct_pt_multiplications": 5,
        "ct_ct_multiplications": 0,
        "explicit_rescale_requests": 5,
        "chebyshev_evaluations": 0,
        "estimated_polynomial_multiplications": 0,
        "bootstraps": 5,
        "bootstrap_iterations": 10,
    },
    "two_layer_cumulative_operation_counts": {
        "rotations": 12600,
        "ct_pt_multiplications": 103775,
        "ct_ct_multiplications": 190,
        "explicit_rescale_requests": 1625,
        "chebyshev_evaluations": 110,
        "estimated_polynomial_multiplications": 2300,
        "bootstraps": 55,
        "bootstrap_iterations": 110,
    },
    "sealing_requirements": {
        "minimum_exact_diagnostic_layers": 3,
        "exact_three_purpose": (
            "validate the second inter-layer handoff and later-layer steady-state regime"
        ),
        "live_m4_regression_required": True,
        "full_12_layer_prototype_validation_required": True,
    },
}
EXPECTED_WARNING = (
    "Research reproduction parameters only. Do not claim 128-bit security."
)
EXPECTED_FEATURE_VALIDATION_STATUS = (
    "two-layer live M5 schedule recorded as candidate; formal schedule unsealed until "
    "exact-three succeeds; live M4 regression and 12-layer MOAI-observability-compatible "
    "prototype gates remain required"
)
EXPECTED_FEATURE_VALIDATION_STATUS_SEALED = (
    "exact-three live M5 metadata schedule sealed; live M4 regression and full "
    "12-layer MOAI-observability-compatible prototype gates remain required"
)
EXPECTED_M5_SCHEDULE_SEALED_STATUS = "sealed_exact3"
EXPECTED_M5_SCHEDULE_SEALED_SOURCE = (
    "approved exact-three-layer live OpenFHE schedule evidence"
)
EXPECTED_PROFILE_PRESEAL_TRANSITION = (
    "candidate profile captured by the exact3 sealer before the one-way "
    "schedule-seal transition"
)
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+:-]{0,127}$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def validate_m5_prototype_acceptance(feature_config: dict[str, object]) -> None:
    contract = feature_config.get("m5_prototype_acceptance")
    if contract != EXPECTED_M5_PROTOTYPE_ACCEPTANCE:
        raise RuntimeError("M5 prototype acceptance contract drifted")


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
    parser.add_argument("--feature-config", type=Path, default=DEFAULT_FEATURE_CONFIG)
    parser.add_argument(
        "--approximation-config",
        type=Path,
        default=DEFAULT_APPROXIMATION_CONFIG,
    )
    return parser.parse_args()


def candidate_used_levels(metadata: object, label: str) -> int:
    if (
        not isinstance(metadata, list)
        or len(metadata) != 3
        or any(type(value) is not int for value in metadata)
    ):
        raise RuntimeError(
            f"M5 schedule candidate contract drifted: {label} tuple is invalid"
        )
    level, noise_scale_degree, remaining_levels = (
        int(metadata[0]),
        int(metadata[1]),
        int(metadata[2]),
    )
    used_levels = level + noise_scale_degree - 1
    if (
        noise_scale_degree <= 0
        or used_levels >= 47
        or remaining_levels != 47 - used_levels
    ):
        raise RuntimeError(
            f"M5 schedule candidate contract drifted: {label} depth is inconsistent"
        )
    return used_levels


def derive_candidate_deltas(
    regime: object,
    previous_raw_output_used_levels: int | None,
    label: str,
) -> list[int | None]:
    if not isinstance(regime, dict):
        raise RuntimeError(
            f"M5 schedule candidate contract drifted: {label} regime is missing"
        )
    checkpoints = regime.get("checkpoint_metadata")
    if not isinstance(checkpoints, dict):
        raise RuntimeError(
            f"M5 schedule candidate contract drifted: {label} checkpoints are missing"
        )
    used = {
        checkpoint: candidate_used_levels(metadata, f"{label} {checkpoint}")
        for checkpoint, metadata in checkpoints.items()
    }
    required = {
        "input",
        "softmax_denominator",
        "attention_output",
        "ln1_variance",
        "ln1_output",
        "ffn_output",
        "ln2_variance",
        "raw_output",
    }
    if set(used) != required:
        raise RuntimeError(
            f"M5 schedule candidate contract drifted: {label} checkpoints drifted"
        )
    deltas: list[int | None] = [
        used["input"] - used["softmax_denominator"],
        used["attention_output"] - used["softmax_denominator"],
        used["attention_output"] - used["ln1_variance"],
        used["ln1_output"] - used["ln1_variance"],
        used["ffn_output"] - used["ln1_output"],
        used["ffn_output"] - used["ln2_variance"],
        used["raw_output"] - used["ln2_variance"],
        (
            None
            if previous_raw_output_used_levels is None
            else previous_raw_output_used_levels - used["input"]
        ),
    ]
    if any(value is not None and value < 0 for value in deltas):
        raise RuntimeError(
            f"M5 schedule candidate contract drifted: {label} delta is negative"
        )
    return deltas


def validate_exact3_seal(value: object) -> None:
    if not isinstance(value, dict) or set(value) != {
        "source",
        "evidence",
        "pre_seal_profile",
    }:
        raise RuntimeError("M5 schedule candidate exact3 seal contract drifted")
    if value["source"] != EXPECTED_M5_SCHEDULE_SEALED_SOURCE:
        raise RuntimeError("M5 schedule candidate exact3 seal source drifted")

    evidence = value["evidence"]
    evidence_keys = {
        "run_id",
        "relative_path",
        "manifest_sha256",
        "sha256sums_sha256",
        "layer_count",
        "artifact_eligible",
        "exact3_gate_passed",
        "schedule_evidence_eligible",
        "formal_schedule_sealed",
    }
    if not isinstance(evidence, dict) or set(evidence) != evidence_keys:
        raise RuntimeError("M5 schedule candidate exact3 evidence contract drifted")
    run_id = evidence["run_id"]
    if not isinstance(run_id, str) or RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise RuntimeError("M5 schedule candidate exact3 evidence run_id drifted")
    if evidence["relative_path"] != f"results/openfhe/{run_id}":
        raise RuntimeError("M5 schedule candidate exact3 evidence path drifted")
    for key in ("manifest_sha256", "sha256sums_sha256"):
        digest = evidence[key]
        if not isinstance(digest, str) or SHA256_PATTERN.fullmatch(digest) is None:
            raise RuntimeError(
                f"M5 schedule candidate exact3 evidence {key} drifted"
            )
    expected_evidence = {
        "layer_count": 3,
        "artifact_eligible": False,
        "exact3_gate_passed": True,
        "schedule_evidence_eligible": True,
        "formal_schedule_sealed": True,
    }
    if any(type(evidence[key]) is not type(expected) or evidence[key] != expected
           for key, expected in expected_evidence.items()):
        raise RuntimeError("M5 schedule candidate exact3 evidence state drifted")

    preseal = value["pre_seal_profile"]
    if not isinstance(preseal, dict) or set(preseal) != {
        "path",
        "sha256",
        "size_bytes",
        "transition",
    }:
        raise RuntimeError("M5 schedule candidate pre-seal binding drifted")
    if (
        preseal["path"] != "config/paper_compat_feature_packed.json"
        or not isinstance(preseal["sha256"], str)
        or SHA256_PATTERN.fullmatch(preseal["sha256"]) is None
        or type(preseal["size_bytes"]) is not int
        or preseal["size_bytes"] <= 0
        or preseal["transition"] != EXPECTED_PROFILE_PRESEAL_TRANSITION
    ):
        raise RuntimeError("M5 schedule candidate pre-seal binding drifted")


def validate_m5_schedule_candidate(feature_config: dict[str, object]) -> bool:
    candidate = feature_config.get("m5_schedule_candidate")
    if not isinstance(candidate, dict):
        raise RuntimeError("M5 schedule candidate contract drifted: object is missing")
    layernorm = feature_config.get("feature_packed_layernorm_override")
    layernorm_status = (
        layernorm.get("schedule_status") if isinstance(layernorm, dict) else None
    )
    status = candidate.get("status")
    sealed = status == EXPECTED_M5_SCHEDULE_SEALED_STATUS
    if status == EXPECTED_M5_SCHEDULE_CANDIDATE["status"]:
        if (
            feature_config.get("validation_status")
            != EXPECTED_FEATURE_VALIDATION_STATUS
            or layernorm_status
            != EXPECTED_FEATURE_LAYERNORM_OVERRIDE["schedule_status"]
        ):
            raise RuntimeError("M5 schedule candidate state is internally inconsistent")
        normalized_candidate = candidate
    elif sealed:
        if (
            feature_config.get("validation_status")
            != EXPECTED_FEATURE_VALIDATION_STATUS_SEALED
            or layernorm_status != "sealed"
            or candidate.get("source") != EXPECTED_M5_SCHEDULE_SEALED_SOURCE
            or candidate.get("formal_schedule_sealed") is not True
            or candidate.get("calibrated_layers") != [0, 1, 2]
        ):
            raise RuntimeError("M5 sealed schedule state is internally inconsistent")
        validate_exact3_seal(candidate.get("exact3_seal"))
        normalized_candidate = dict(candidate)
        normalized_candidate.pop("exact3_seal", None)
        normalized_candidate["status"] = EXPECTED_M5_SCHEDULE_CANDIDATE["status"]
        normalized_candidate["source"] = EXPECTED_M5_SCHEDULE_CANDIDATE["source"]
        normalized_candidate["formal_schedule_sealed"] = False
        normalized_candidate["calibrated_layers"] = [0, 1]
    else:
        raise RuntimeError("M5 schedule candidate status drifted")

    candidate = normalized_candidate
    if candidate.get("formal_schedule_sealed") is not False:
        raise RuntimeError(
            "M5 schedule candidate contract drifted: unvalidated candidate claims sealed"
        )

    regimes = candidate.get("layer_regimes")
    if not isinstance(regimes, dict):
        raise RuntimeError(
            "M5 schedule candidate contract drifted: regimes are missing"
        )
    layer0 = regimes.get("layer0")
    later_layers = regimes.get("later_layers")
    layer0_deltas = derive_candidate_deltas(layer0, None, "layer0")
    if not isinstance(layer0, dict):
        raise RuntimeError("M5 schedule candidate contract drifted: layer0 is missing")
    layer0_checkpoints = layer0.get("checkpoint_metadata")
    if not isinstance(layer0_checkpoints, dict):
        raise RuntimeError(
            "M5 schedule candidate contract drifted: layer0 checkpoints are missing"
        )
    previous_raw_used = candidate_used_levels(
        layer0_checkpoints.get("raw_output"),
        "layer0 raw_output",
    )
    later_deltas = derive_candidate_deltas(
        later_layers,
        previous_raw_used,
        "later_layers",
    )
    if (
        layer0.get("used_level_deltas") != layer0_deltas
        or not isinstance(later_layers, dict)
        or later_layers.get("used_level_deltas") != later_deltas
    ):
        raise RuntimeError(
            "M5 schedule candidate contract drifted: used-level deltas are inconsistent"
        )

    count_fields = {
        "rotations",
        "ct_pt_multiplications",
        "ct_ct_multiplications",
        "explicit_rescale_requests",
        "chebyshev_evaluations",
        "estimated_polynomial_multiplications",
        "bootstraps",
        "bootstrap_iterations",
    }
    per_layer = candidate.get("per_encoder_layer_operation_counts")
    per_refresh = candidate.get("per_inter_layer_refresh_operation_counts")
    cumulative = candidate.get("two_layer_cumulative_operation_counts")
    for label, counts in (
        ("per-layer", per_layer),
        ("per-refresh", per_refresh),
        ("two-layer cumulative", cumulative),
    ):
        if (
            not isinstance(counts, dict)
            or set(counts) != count_fields
            or any(
                type(counts[field]) is not int or counts[field] < 0 for field in counts
            )
        ):
            raise RuntimeError(
                f"M5 schedule candidate contract drifted: {label} counts are invalid"
            )
    derived_cumulative = {
        field: 2 * per_layer[field] + per_refresh[field] for field in count_fields
    }
    if cumulative != derived_cumulative:
        raise RuntimeError(
            "M5 schedule candidate contract drifted: cumulative counts are inconsistent"
        )
    if candidate != EXPECTED_M5_SCHEDULE_CANDIDATE:
        raise RuntimeError("M5 schedule candidate contract drifted")
    return sealed


def validate_feature_layernorm_override(
    feature_config: dict[str, object],
    approximation_config: dict[str, object],
    sealed_schedule: bool,
) -> None:
    override = feature_config.get("feature_packed_layernorm_override")
    if not isinstance(override, dict):
        raise RuntimeError("feature-packed LayerNorm override contract drifted")
    expected_override = dict(EXPECTED_FEATURE_LAYERNORM_OVERRIDE)
    if sealed_schedule:
        expected_override["schedule_status"] = "sealed"
    if override != expected_override:
        raise RuntimeError("feature-packed LayerNorm override contract drifted")

    binding = override["trace_scale_contract"]
    try:
        contract = approximation_config["operators"]["layernorm"][
            "feature_packed_trace_scale_contract"
        ]
    except (KeyError, TypeError) as error:
        raise RuntimeError(
            "FeaturePacked LayerNorm scale source is incomplete"
        ) from error
    if not isinstance(contract, dict):
        raise RuntimeError("FeaturePacked LayerNorm scale source is incomplete")

    fixed_fields = {
        "contract_id": binding["contract_id"],
        "shape": [2, 12, 5],
        "axis_order": ["site", "layer", "token"],
        "site_order": ["ln1", "ln2"],
        "layer_order": list(range(12)),
        "token_order": list(range(5)),
        "packing": {
            "hidden_features": 768,
            "feature_slots": 1024,
            "trace_tokens": 5,
            "inactive_feature_slots": 256,
        },
        "selection": {
            "target": 64.0,
            "formula": ("D=2^roundTiesToEven(log2(64/raw_population_variance))"),
            "rounding": "IEEE-style round ties to even",
            "power_of_two_only": True,
        },
    }
    for key, expected in fixed_fields.items():
        if contract.get(key) != expected:
            raise RuntimeError(f"FeaturePacked LayerNorm scale contract {key} drifted")
    scope = contract.get("scope")
    if (
        not isinstance(scope, dict)
        or scope.get("non_generalizable") is not True
        or scope.get("runtime_activation_dependency") != "none"
        or scope.get("public_indices") != ["site", "layer", "token"]
    ):
        raise RuntimeError("FeaturePacked LayerNorm scale scope drifted")
    epsilon = contract.get("epsilon")
    if not isinstance(epsilon, dict) or epsilon.get("value") != 1e-12:
        raise RuntimeError("FeaturePacked LayerNorm epsilon contract drifted")
    bootstrap = contract.get("bootstrap_execution")
    if (
        not isinstance(bootstrap, dict)
        or bootstrap.get("preconditioner") != 2048.0
        or bootstrap.get("active_pre_bootstrap_value") != "u/2048"
        or bootstrap.get("inactive_pre_bootstrap_guard") != "1/2048"
        or bootstrap.get("post_bootstrap_restore")
        != "multiply all 1024 slots uniformly by 2048"
    ):
        raise RuntimeError("FeaturePacked LayerNorm bootstrap contract drifted")
    inactive_guard = contract.get("inactive_guard")
    if (
        not isinstance(inactive_guard, dict)
        or inactive_guard.get("exact_identity_gate") is not False
        or inactive_guard.get("post_restore_target") != 1.0
        or inactive_guard.get("validation")
        != (
            "finite and inside the inverse-sqrt hard interval; deviation "
            "from one is diagnostic only"
        )
    ):
        raise RuntimeError("FeaturePacked LayerNorm inactive guard drifted")
    normalized_range = contract.get("normalized_variance_range")
    if not isinstance(normalized_range, dict) or normalized_range.get(
        "hard_interval"
    ) != [0.5, 1536.0]:
        raise RuntimeError("FeaturePacked LayerNorm interval contract drifted")

    values = contract.get("values")
    if (
        not isinstance(values, list)
        or len(values) != 2
        or any(not isinstance(site, list) or len(site) != 12 for site in values)
        or any(
            not isinstance(row, list) or len(row) != 5
            for site in values
            for row in site
        )
    ):
        raise RuntimeError("FeaturePacked LayerNorm scale tensor shape drifted")
    flattened: list[float] = []
    for site in values:
        for row in site:
            for value in row:
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or value <= 0.0
                    or math.frexp(float(value))[0] != 0.5
                ):
                    raise RuntimeError(
                        "FeaturePacked LayerNorm scale is not a positive power of two"
                    )
                flattened.append(float(value))
    values_sha256 = hashlib.sha256(
        b"".join(struct.pack("<d", value) for value in flattened)
    ).hexdigest()
    if (
        values_sha256 != binding["values_sha256"]
        or contract.get("values_sha256") != values_sha256
        or contract.get("raw_variance_sha256") != binding["raw_variance_sha256"]
    ):
        raise RuntimeError("FeaturePacked LayerNorm scale provenance drifted")

    contract_payload = dict(contract)
    contract_sha256 = contract_payload.pop("contract_sha256", None)
    calculated_contract_sha256 = hashlib.sha256(
        canonical_json(contract_payload).encode("utf-8")
    ).hexdigest()
    if (
        contract_sha256 != binding["contract_sha256"]
        or calculated_contract_sha256 != binding["contract_sha256"]
    ):
        raise RuntimeError("FeaturePacked LayerNorm scale contract hash drifted")


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
    with arguments.approximation_config.open("r", encoding="utf-8") as handle:
        approximation_config = json.load(handle)
    if (
        feature_config.get("runtime_id") != "paper_compat_feature_packed_m4_m5"
        or feature_config.get("profile_id") != "paper_compat"
        or feature_config.get("security_claim") != "none"
        or feature_config.get("warning") != EXPECTED_WARNING
        or feature_config.get("validation_status")
        not in {
            EXPECTED_FEATURE_VALIDATION_STATUS,
            EXPECTED_FEATURE_VALIDATION_STATUS_SEALED,
        }
    ):
        raise RuntimeError(
            "feature-packed paper_compat identity/security contract drifted"
        )
    if (
        feature_config.get("feature_packed_softmax_override")
        != EXPECTED_FEATURE_SOFTMAX_OVERRIDE
    ):
        raise RuntimeError("feature-packed Softmax override contract drifted")
    validate_m5_prototype_acceptance(feature_config)
    sealed_schedule = validate_m5_schedule_candidate(feature_config)
    validate_feature_layernorm_override(
        feature_config,
        approximation_config,
        sealed_schedule,
    )

    runtime_sources = {
        "m3_smoke_effective_runtime": (
            config["openfhe_translation"]["m3_smoke_effective_runtime"],
            EXPECTED_M3,
            config["openfhe_translation"]["m3_smoke_effective_runtime"]["parameters"],
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
