#!/usr/bin/env python3
"""Freeze and validate the complete five-token, 12-layer MOAI encoder trace.

This is a plaintext fixture validator.  It binds every CSV byte-for-byte, checks
the trace's exact BERT relations, and replays the frozen OpenFHE polynomial oracle.
It does not execute OpenFHE and does not make a security claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - environment failure
    raise SystemExit(
        "NumPy is required; use the MOAI Python 3.10/3.11 validation environment."
    ) from exc


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = REPO_ROOT / "data"
DEFAULT_MANIFEST = REPO_ROOT / "config" / "moai_encoder_trace.json"
SCALE_MANIFEST_PATH = REPO_ROOT / "config" / "moai_trace_channel_scales.json"
APPROXIMATION_PATH = REPO_ROOT / "config" / "openfhe_approximations.json"
PASS_DECISION = "PASS_OPENFHE_ENCODER_TRACE_CONTRACT"
FAIL_DECISION = "FAIL_OPENFHE_ENCODER_TRACE_CONTRACT"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
FEATURE_PACKED_LAYERNORM_CONTRACT_ID = "layernorm_layer_token_power2_scale_v1"
FEATURE_PACKED_LAYERNORM_CONTRACT_SHA256 = (
    "b493b032e461e15d7436efe7fff5948436afa8e302646daa97db78fa1d59be3e"
)
FEATURE_PACKED_LAYERNORM_VALUES_SHA256 = (
    "9436f05ce80b427de47700d924869b0dd6f13cc515e586446fcd2dc56faec28a"
)
FEATURE_PACKED_LAYERNORM_RAW_VARIANCE_SHA256 = (
    "940f92de81915c2121b427b72827e4ed87d072fda0847266d0cb5521d60f93b6"
)
FEATURE_PACKED_LAYERNORM_SCALE_TARGET = 64.0
FEATURE_PACKED_LAYERNORM_BOOTSTRAP_PRECONDITIONER = 2048.0
FEATURE_PACKED_LAYERNORM_SHAPE = (2, 12, 5)
FEATURE_PACKED_LAYERNORM_SITE_ORDER = ("ln1", "ln2")
FEATURE_PACKED_LAYERNORM_SELECTION_FORMULA = (
    "D=2^roundTiesToEven(log2(64/raw_population_variance))"
)

SCOPE = {
    "fixture_only": True,
    "source_trace_token_count": 5,
    "source_trace_kind": "server-only BERT-base encoder trace replay",
    "runtime_activation_metadata_allowed": False,
    "production_generalization": "forbidden",
    "excluded": [
        "GPU",
        "Discrete CKKS/FBT",
        "QDQ",
        "tokenizer",
        "classifier",
        "task-level end-to-end inference",
    ],
}

DIMENSIONS = {
    "encoder_layers": 12,
    "trace_token_rows": 5,
    "packed_query_rows": 128,
    "attention_heads": 12,
    "head_size": 64,
    "hidden_size": 768,
    "intermediate_size": 3072,
}

# These gates describe CSV-to-equation parity, so they are intentionally much
# tighter than the future encrypted gates recorded below.
TRACE_RELATION_THRESHOLDS = {
    "query_projection": {"max_absolute": 5e-6, "relative_l2": 5e-7, "cosine": 0.999999},
    "key_projection": {"max_absolute": 5e-6, "relative_l2": 5e-7, "cosine": 0.999999},
    "value_projection": {"max_absolute": 5e-6, "relative_l2": 5e-7, "cosine": 0.999999},
    "qkt_scaled_by_eight": {
        "max_absolute": 5e-6,
        "relative_l2": 5e-7,
        "cosine": 0.999999,
    },
    "exact_softmax": {"max_absolute": 2e-6, "relative_l2": 1e-6, "cosine": 0.999999},
    "attention_times_value": {
        "max_absolute": 2e-6,
        "relative_l2": 1e-6,
        "cosine": 0.999999,
    },
    "attention_snapshot_link": {
        "max_absolute": 5e-6,
        "relative_l2": 1e-6,
        "cosine": 0.999999,
    },
    "self_projection": {"max_absolute": 3e-6, "relative_l2": 1e-6, "cosine": 0.999999},
    "self_residual": {"max_absolute": 3e-6, "relative_l2": 5e-7, "cosine": 0.999999},
    "self_layernorm_exact": {
        "max_absolute": 3e-5,
        "relative_l2": 5e-7,
        "cosine": 0.999999,
    },
    "self_to_ffn_link": {"max_absolute": 1e-5, "relative_l2": 2e-7, "cosine": 0.999999},
    "intermediate_projection_scaled": {
        "max_absolute": 3e-5,
        "relative_l2": 4e-7,
        "cosine": 0.999999,
    },
    "exact_gelu": {"max_absolute": 1e-5, "relative_l2": 8e-7, "cosine": 0.999999},
    "gelu_snapshot_link": {
        "max_absolute": 1e-5,
        "relative_l2": 2e-7,
        "cosine": 0.999999,
    },
    "final_projection_unscaled": {
        "max_absolute": 1.5e-4,
        "relative_l2": 5e-7,
        "cosine": 0.999999,
    },
    "final_residual": {"max_absolute": 3e-5, "relative_l2": 5e-8, "cosine": 0.999999},
    "final_layernorm_exact": {
        "max_absolute": 3e-5,
        "relative_l2": 5e-7,
        "cosine": 0.999999,
    },
}

THRESHOLDS = {
    "trace_relations": TRACE_RELATION_THRESHOLDS,
    "chained_polynomial_oracle_per_layer": {
        "max_absolute": 0.1,
        "relative_l2": 0.012,
        "cosine": 0.9999,
    },
    "chained_polynomial_oracle_final": {
        "max_absolute": 0.1,
        "relative_l2": 0.05,
        "cosine": 0.99,
    },
    "hard_intervals": {
        "gelu_input": [-80.0, 128.0],
        "softmax_shifted_logits": [-16.0, 5.0],
        "softmax_denominator": [0.01, 80.0],
        "layernorm_normalized_variance": [0.5, 1536.0],
    },
    "future_encrypted_acceptance": {
        "inactive_or_cross_lane_max_absolute": 1e-6,
        "linear_relative_l2": 1e-4,
        "linear_cosine": 0.99999,
        "nonlinear_or_single_layer_relative_l2": 1e-2,
        "nonlinear_or_single_layer_cosine": 0.999,
        "final_relative_l2": 5e-2,
        "final_cosine": 0.99,
        "nan_or_inf_allowed": False,
    },
    "policy": "thresholds may be tightened; relaxation requires a contract revision",
}

# The orientation text is part of the public contract, not just documentation.
FILE_SPECS: dict[str, dict[str, Any]] = {
    "key_output": {
        "path": "Attention/BertSelfAttention/allresults/K.csv",
        "shape": [5, 768],
        "orientation": "token_by_hidden",
    },
    "query_output": {
        "path": "Attention/BertSelfAttention/allresults/Q.csv",
        "shape": [5, 768],
        "orientation": "token_by_hidden",
    },
    "qkt_output": {
        "path": "Attention/BertSelfAttention/allresults/QKT.csv",
        "shape": [5, 60],
        "orientation": "query_by_head_then_key; QK^T/sqrt(64)",
    },
    "value_output": {
        "path": "Attention/BertSelfAttention/allresults/V.csv",
        "shape": [5, 768],
        "orientation": "token_by_hidden",
    },
    "softmax_output": {
        "path": "Attention/BertSelfAttention/allresults/aftsoftmax.csv",
        "shape": [5, 60],
        "orientation": "query_by_head_then_key",
    },
    "embedded_inputs": {
        "path": "Attention/BertSelfAttention/allresults/embedded_inputs.csv",
        "shape": [5, 768],
        "orientation": "token_by_hidden",
    },
    "token_ids": {
        "path": "Attention/BertSelfAttention/allresults/inputs.csv",
        "shape": [5],
        "orientation": "five_integer_token_ids",
    },
    "attention_output": {
        "path": "Attention/BertSelfAttention/allresults/real_attention.csv",
        "shape": [5, 768],
        "orientation": "token_by_head_then_feature",
    },
    "key_bias": {
        "path": "Attention/BertSelfAttention/parms/key_bias.csv",
        "shape": [768],
        "orientation": "output_hidden",
    },
    "key_weight": {
        "path": "Attention/BertSelfAttention/parms/key_weight.csv",
        "shape": [768, 768],
        "orientation": "output_by_input",
    },
    "query_bias": {
        "path": "Attention/BertSelfAttention/parms/query_bias.csv",
        "shape": [768],
        "orientation": "output_hidden",
    },
    "query_weight": {
        "path": "Attention/BertSelfAttention/parms/query_weight.csv",
        "shape": [768, 768],
        "orientation": "output_by_input",
    },
    "value_bias": {
        "path": "Attention/BertSelfAttention/parms/value_bias.csv",
        "shape": [768],
        "orientation": "output_hidden",
    },
    "value_weight": {
        "path": "Attention/BertSelfAttention/parms/value_weight.csv",
        "shape": [768, 768],
        "orientation": "output_by_input",
    },
    "self_output": {
        "path": "Attention/SelfOutput/allresults/real_self_output.csv",
        "shape": [5, 768],
        "orientation": "token_by_hidden",
    },
    "self_after_linear": {
        "path": "Attention/SelfOutput/allresults/self_output_after_linear.csv",
        "shape": [5, 768],
        "orientation": "token_by_hidden",
    },
    "self_inputs": {
        "path": "Attention/SelfOutput/allresults/self_output_inputs.csv",
        "shape": [5, 768],
        "orientation": "token_by_hidden",
    },
    "self_before_layernorm": {
        "path": (
            "Attention/SelfOutput/allresults/"
            "self_output_residual_connection_before_layernorm.csv"
        ),
        "shape": [5, 768],
        "orientation": "token_by_hidden",
    },
    "self_residual_operand": {
        "path": "Attention/SelfOutput/allresults/self_output_usedby_residual.csv",
        "shape": [5, 768],
        "orientation": "token_by_hidden",
    },
    "self_layernorm_bias": {
        "path": "Attention/SelfOutput/parms/self_output_LayerNorm_bias.csv",
        "shape": [768],
        "orientation": "hidden",
    },
    "self_layernorm_weight": {
        "path": "Attention/SelfOutput/parms/self_output_LayerNorm_weight.csv",
        "shape": [768],
        "orientation": "hidden",
    },
    "self_dense_bias": {
        "path": "Attention/SelfOutput/parms/self_output_dense_bias.csv",
        "shape": [768],
        "orientation": "output_hidden",
    },
    "self_dense_weight": {
        "path": "Attention/SelfOutput/parms/self_output_dense_weight.csv",
        "shape": [768, 768],
        "orientation": "output_by_input",
    },
    "intermediate_inputs": {
        "path": "Intermediate/allresults/intermediate_inputs.csv",
        "shape": [5, 768],
        "orientation": "token_by_hidden",
    },
    "intermediate_after_linear": {
        "path": "Intermediate/allresults/intermediate_output_after_linear.csv",
        "shape": [5, 3072],
        "orientation": "token_by_intermediate_channel_after_sparse_scale",
    },
    "intermediate_output": {
        "path": "Intermediate/allresults/real_intermediate_output.csv",
        "shape": [5, 3072],
        "orientation": "token_by_intermediate_channel",
    },
    "intermediate_bias": {
        "path": "Intermediate/parms/intermediate_dense_bias.csv",
        "shape": [3072],
        "orientation": "output_intermediate_channel_before_sparse_scale",
    },
    "intermediate_weight": {
        "path": "Intermediate/parms/intermediate_dense_weight.csv",
        "shape": [3072, 768],
        "orientation": "output_by_input_before_sparse_scale",
    },
    "final_after_linear": {
        "path": "Output/allresults/final_output_after_linear.csv",
        "shape": [5, 768],
        "orientation": "token_by_hidden",
    },
    "final_inputs": {
        "path": "Output/allresults/final_output_inputs.csv",
        "shape": [5, 3072],
        "orientation": "token_by_intermediate_channel",
    },
    "final_before_layernorm": {
        "path": "Output/allresults/final_output_residual_connection_before_layernorm.csv",
        "shape": [5, 768],
        "orientation": "token_by_hidden",
    },
    "final_residual_operand": {
        "path": "Output/allresults/final_output_usedby_residual.csv",
        "shape": [5, 768],
        "orientation": "token_by_hidden",
    },
    "final_output": {
        "path": "Output/allresults/real_final_output.csv",
        "shape": [5, 768],
        "orientation": "token_by_hidden",
    },
    "final_layernorm_bias": {
        "path": "Output/parms/final_output_LayerNorm_bias.csv",
        "shape": [768],
        "orientation": "hidden",
    },
    "final_layernorm_weight": {
        "path": "Output/parms/final_output_LayerNorm_weight.csv",
        "shape": [768],
        "orientation": "hidden",
    },
    "final_dense_bias": {
        "path": "Output/parms/final_output_dense_bias.csv",
        "shape": [768],
        "orientation": "output_hidden",
    },
    "final_dense_weight": {
        "path": "Output/parms/final_output_dense_weight.csv",
        "shape": [768, 3072],
        "orientation": "output_by_input_with_sparse_scale_division",
    },
}

TOP_LEVEL_KEYS = {
    "schema_version",
    "contract_id",
    "validator_version",
    "profile_id",
    "security_claim",
    "hash_algorithm",
    "scope",
    "dimensions",
    "sources",
    "approximation_binding",
    "thresholds",
    "required_files",
    "layers",
    "static_trace_summary",
    "chained_oracle_summary",
}


class ContractError(RuntimeError):
    """Raised when the manifest, source contracts, or trace fails closed."""


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ContractError(f"JSON source is not a file: {path}")
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream, object_pairs_hook=_no_duplicate_keys)
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read strict JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"JSON root must be an object: {path}")
    return value


def require_exact_keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractError(f"{label} must be an object")
    actual = set(value)
    if actual != expected:
        raise ContractError(
            f"{label} keys differ: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ContractError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def tensor_sha256(values: np.ndarray) -> str:
    encoded = np.asarray(values, dtype="<f8").tobytes(order="C")
    return hashlib.sha256(encoded).hexdigest()


def coefficient_sha256(values: np.ndarray) -> str:
    return tensor_sha256(values)


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def openfhe_chebyshev_coefficients(
    function: Callable[[float], float], lower: float, upper: float, degree: int
) -> np.ndarray:
    """Match OpenFHE EvalChebyshevCoefficients, including doubled raw c0."""
    if degree <= 0 or not lower < upper:
        raise ContractError("invalid Chebyshev degree or interval")
    count = degree + 1
    half_width = 0.5 * (upper - lower)
    midpoint = 0.5 * (upper + lower)
    pi_by_count = math.pi / count
    points = np.empty(count, dtype=np.float64)
    for index in range(count):
        node = math.cos(pi_by_count * (index + 0.5)) * half_width + midpoint
        points[index] = function(node)
    coefficients = np.zeros(count, dtype=np.float64)
    for coefficient_index in range(count):
        accumulator = 0.0
        for point_index in range(count):
            accumulator += points[point_index] * math.cos(
                pi_by_count * coefficient_index * (point_index + 0.5)
            )
        coefficients[coefficient_index] = accumulator * (2.0 / count)
    return coefficients


def evaluate_openfhe_chebyshev(
    values: np.ndarray, coefficients: np.ndarray, lower: float, upper: float
) -> np.ndarray:
    normalized = (2.0 * values - lower - upper) / (upper - lower)
    evaluator_coefficients = coefficients.copy()
    evaluator_coefficients[0] *= 0.5
    return np.polynomial.chebyshev.chebval(normalized, evaluator_coefficients)


def exact_gelu_scalar(value: float) -> float:
    return 0.5 * value * (1.0 + math.erf(value / math.sqrt(2.0)))


def exact_gelu(values: np.ndarray) -> np.ndarray:
    result = np.fromiter(
        (exact_gelu_scalar(float(value)) for value in values.flat),
        dtype=np.float64,
        count=values.size,
    )
    return result.reshape(values.shape)


def constrain_zero_at_origin(
    coefficients: np.ndarray, lower: float, upper: float
) -> np.ndarray:
    raw_at_zero = float(
        evaluate_openfhe_chebyshev(
            np.array([0.0], dtype=np.float64), coefficients, lower, upper
        )[0]
    )
    constrained = coefficients.copy()
    constrained[0] -= 2.0 * raw_at_zero
    return constrained


def quality(actual: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    actual_flat = np.asarray(actual, dtype=np.float64).reshape(-1)
    expected_flat = np.asarray(expected, dtype=np.float64).reshape(-1)
    if actual_flat.shape != expected_flat.shape:
        raise ContractError("quality tensors have different shapes")
    if not np.isfinite(actual_flat).all() or not np.isfinite(expected_flat).all():
        raise ContractError("quality tensors contain NaN or Inf")
    delta = actual_flat - expected_flat
    expected_norm = float(np.linalg.norm(expected_flat))
    actual_norm = float(np.linalg.norm(actual_flat))
    delta_norm = float(np.linalg.norm(delta))
    if expected_norm == 0.0 or actual_norm == 0.0:
        raise ContractError("quality tensors must have nonzero norms")
    return {
        "relative_l2": delta_norm / expected_norm,
        "cosine": float(
            np.vdot(actual_flat, expected_flat) / (actual_norm * expected_norm)
        ),
        "max_absolute": float(np.max(np.abs(delta))),
    }


def require_metric_gate(
    label: str, metrics: dict[str, float], gate: dict[str, float]
) -> None:
    if (
        metrics["relative_l2"] > gate["relative_l2"]
        or metrics["cosine"] < gate["cosine"]
        or metrics["max_absolute"] > gate["max_absolute"]
    ):
        raise ContractError(
            f"{label} gate failed: rel-L2={metrics['relative_l2']:.9e} "
            f"(max {gate['relative_l2']:.9e}), cosine={metrics['cosine']:.12f} "
            f"(min {gate['cosine']:.12f}), max-abs={metrics['max_absolute']:.9e} "
            f"(max {gate['max_absolute']:.9e})"
        )


def require_range(label: str, values: np.ndarray, interval: list[float]) -> list[float]:
    observed = [float(np.min(values)), float(np.max(values))]
    if observed[0] < interval[0] or observed[1] > interval[1]:
        raise ContractError(
            f"{label} range {observed} is outside hard interval {interval}; no clipping allowed"
        )
    return observed


def exact_layernorm(
    values: np.ndarray, weight: np.ndarray, bias: np.ndarray, epsilon: float
) -> np.ndarray:
    centered = values - np.mean(values, axis=-1, keepdims=True)
    variance = np.mean(centered * centered, axis=-1, keepdims=True) + epsilon
    return weight * centered / np.sqrt(variance) + bias


def polynomial_layernorm(
    values: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    epsilon: float,
    variance_scales: np.ndarray,
    inverse_sqrt_coefficients: np.ndarray,
    interval: list[float],
    range_label: str,
) -> tuple[np.ndarray, list[float]]:
    scales = np.asarray(variance_scales, dtype=np.float64)
    if values.shape[0] != 5 or scales.shape != (5,):
        raise ContractError(
            "FeaturePacked LayerNorm requires exactly five pre-registered "
            "per-token variance scales"
        )
    if not np.isfinite(scales).all() or np.any(scales <= 0.0):
        raise ContractError(
            "FeaturePacked LayerNorm variance scales must be finite and positive"
        )
    centered = values - np.mean(values, axis=-1, keepdims=True)
    variance = np.mean(centered * centered, axis=-1, keepdims=True) + epsilon
    normalized_variance = scales[:, None] * variance
    observed = require_range(range_label, normalized_variance, interval)
    inverse = np.sqrt(scales)[:, None] * evaluate_openfhe_chebyshev(
        normalized_variance, inverse_sqrt_coefficients, interval[0], interval[1]
    )
    return weight * centered * inverse + bias, observed


def selected_polynomial(value: dict[str, Any]) -> dict[str, Any]:
    keys = {
        "function",
        "degree",
        "interval",
        "coefficient_count",
        "coefficient_sha256",
        "coefficient_encoding",
        "c0_convention",
    }
    selected = {key: value[key] for key in keys}
    if "constraint" in value:
        selected["constraint"] = value["constraint"]
    return selected


def selected_feature_packed_layernorm_binding(
    contract: dict[str, Any],
) -> dict[str, Any]:
    return {
        "contract_id": contract["contract_id"],
        "contract_sha256": contract["contract_sha256"],
        "shape": contract["shape"],
        "axis_order": contract["axis_order"],
        "site_order": contract["site_order"],
        "layer_order": contract["layer_order"],
        "token_order": contract["token_order"],
        "scope": {
            "non_generalizable": contract["scope"]["non_generalizable"],
            "public_indices": contract["scope"]["public_indices"],
            "runtime_activation_dependency": contract["scope"][
                "runtime_activation_dependency"
            ],
        },
        "epsilon": contract["epsilon"],
        "selection": contract["selection"],
        "values_sha256": contract["values_sha256"],
        "raw_variance_sha256": contract["raw_variance_sha256"],
        "bootstrap_preconditioner": contract["bootstrap_execution"]["preconditioner"],
        "inactive_guard": contract["inactive_guard"],
    }


def validate_feature_packed_layernorm_contract(
    contract: Any,
    generic_layernorm_epsilon: Any,
) -> np.ndarray:
    value = require_exact_keys(
        contract,
        {
            "status",
            "contract_id",
            "scope",
            "shape",
            "axis_order",
            "site_order",
            "layer_order",
            "token_order",
            "packing",
            "epsilon",
            "selection",
            "values",
            "values_sha256",
            "values_sha256_encoding",
            "raw_variance_sha256",
            "raw_variance_sha256_encoding",
            "raw_variance_observed_range",
            "normalization_formula",
            "normalized_variance_range",
            "bootstrap_execution",
            "inactive_guard",
            "sites",
            "trace_validation",
            "contract_hash_encoding",
            "contract_sha256",
        },
        "FeaturePacked LayerNorm trace-scale contract",
    )
    fixed_values = {
        "status": "plaintext_trace_gate_passed",
        "contract_id": FEATURE_PACKED_LAYERNORM_CONTRACT_ID,
        "shape": list(FEATURE_PACKED_LAYERNORM_SHAPE),
        "axis_order": ["site", "layer", "token"],
        "site_order": list(FEATURE_PACKED_LAYERNORM_SITE_ORDER),
        "layer_order": list(range(DIMENSIONS["encoder_layers"])),
        "token_order": list(range(DIMENSIONS["trace_token_rows"])),
        "packing": {
            "hidden_features": DIMENSIONS["hidden_size"],
            "feature_slots": 1024,
            "trace_tokens": DIMENSIONS["trace_token_rows"],
            "inactive_feature_slots": 256,
        },
        "scope": {
            "workload": "fixed five-token 12-layer encoder trace replay only",
            "non_generalizable": True,
            "visibility": "public pre-registered workload metadata",
            "public_indices": ["site", "layer", "token"],
            "runtime_activation_dependency": "none",
            "derivation_source": "bundled plaintext residual tensors, used offline only",
            "forbidden": (
                "deriving or selecting scales from runtime plaintext or "
                "ciphertext activations, client-private inputs, or values "
                "outside the registered site/layer/token trace coordinates"
            ),
        },
        "epsilon": {
            "value": 1e-12,
            "placement": (
                "population variance + epsilon before multiplication by the "
                "selected D[site,layer,token]"
            ),
        },
        "selection": {
            "target": FEATURE_PACKED_LAYERNORM_SCALE_TARGET,
            "formula": FEATURE_PACKED_LAYERNORM_SELECTION_FORMULA,
            "rounding": "IEEE-style round ties to even",
            "power_of_two_only": True,
        },
        "normalization_formula": {
            "variance": "v=mean_i((x_i-mean(x))^2)+1e-12",
            "polynomial_input": "u=D[site,layer,token]*v",
            "inverse_standard_deviation": ("sqrt(D[site,layer,token])*ChebInvSqrt(u)"),
            "sqrt_D_compensation": (
                "fuse the per-site/layer/token sqrt(D) into public gamma"
            ),
            "output": ("gamma_i*(x_i-mean(x))*inverse_standard_deviation+beta_i"),
        },
        "bootstrap_execution": {
            "placement": (
                "after preconditioned normalized variance is formed, before "
                "inverse sqrt"
            ),
            "preconditioner": FEATURE_PACKED_LAYERNORM_BOOTSTRAP_PRECONDITIONER,
            "active_pre_bootstrap_value": "u/2048",
            "inactive_pre_bootstrap_guard": "1/2048",
            "post_bootstrap_restore": "multiply all 1024 slots uniformly by 2048",
            "post_restore_active_semantics": "active slots approximate u",
            "post_restore_inactive_semantics": (
                "inactive slots are an interval-checked diagnostic guard near "
                "one, not an exact identity assertion"
            ),
            "post_inverse_mask": (
                "after inverse-sqrt evaluation, multiply by the public "
                "first-768-active/last-256-zero mask and rescale before the "
                "centered ciphertext-ciphertext product"
            ),
        },
        "inactive_guard": {
            "pre_bootstrap_value": "1/2048",
            "post_restore_target": 1.0,
            "validation": (
                "finite and inside the inverse-sqrt hard interval; deviation "
                "from one is diagnostic only"
            ),
            "exact_identity_gate": False,
            "post_inverse_mask": (
                "the public first-768-active/last-256-zero mask removes "
                "inactive inverse-sqrt slots before the centered product"
            ),
        },
        "values_sha256_encoding": (
            "site-major ln1/ln2, then layer 0..11, then token 0..4; "
            "raw IEEE-754 binary64 little-endian bytes"
        ),
        "raw_variance_sha256_encoding": (
            "site-major ln1/ln2, then layer 0..11, then token 0..4; "
            "raw IEEE-754 binary64 little-endian bytes"
        ),
        "contract_hash_encoding": (
            "SHA-256 of canonical sorted-key compact JSON for this object, "
            "excluding only contract_sha256"
        ),
    }
    for key, expected in fixed_values.items():
        if value[key] != expected:
            raise ContractError(
                f"FeaturePacked LayerNorm trace-scale contract {key} changed"
            )
    if (
        type(generic_layernorm_epsilon) is not float
        or generic_layernorm_epsilon != value["epsilon"]["value"]
    ):
        raise ContractError(
            "FeaturePacked LayerNorm epsilon differs from the generic "
            "LayerNorm approximation contract"
        )

    try:
        scales = np.asarray(value["values"], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ContractError(
            "FeaturePacked LayerNorm scale values are not a binary64 tensor"
        ) from exc
    if scales.shape != FEATURE_PACKED_LAYERNORM_SHAPE:
        raise ContractError(
            "FeaturePacked LayerNorm scale values shape changed: "
            f"expected {FEATURE_PACKED_LAYERNORM_SHAPE}, got {scales.shape}"
        )
    fractions, _ = np.frexp(scales)
    if (
        not np.isfinite(scales).all()
        or np.any(scales <= 0.0)
        or not np.all(fractions == 0.5)
    ):
        raise ContractError(
            "FeaturePacked LayerNorm scale values must be positive powers of two"
        )
    values_hash = tensor_sha256(scales)
    if (
        value["values_sha256"] != FEATURE_PACKED_LAYERNORM_VALUES_SHA256
        or values_hash != value["values_sha256"]
    ):
        raise ContractError(
            "FeaturePacked LayerNorm scale values matrix or SHA-256 changed"
        )
    if (
        not isinstance(value["raw_variance_sha256"], str)
        or SHA256_PATTERN.fullmatch(value["raw_variance_sha256"]) is None
        or value["raw_variance_sha256"] != FEATURE_PACKED_LAYERNORM_RAW_VARIANCE_SHA256
    ):
        raise ContractError(
            "FeaturePacked LayerNorm raw population-variance SHA-256 changed"
        )
    normalized_range = require_exact_keys(
        value["normalized_variance_range"],
        {"hard_interval", "observed_trace_range", "action_on_violation"},
        "FeaturePacked LayerNorm normalized-variance range",
    )
    if (
        normalized_range["hard_interval"]
        != THRESHOLDS["hard_intervals"]["layernorm_normalized_variance"]
    ):
        raise ContractError(
            "FeaturePacked LayerNorm normalized-variance hard interval changed"
        )
    if normalized_range["action_on_violation"] != (
        "stop; do not clip or derive a runtime replacement scale"
    ):
        raise ContractError(
            "FeaturePacked LayerNorm runtime scale-derivation policy changed"
        )
    try:
        contract_payload = dict(value)
        del contract_payload["contract_sha256"]
        contract_hash = hashlib.sha256(
            canonical_json(contract_payload).encode("utf-8")
        ).hexdigest()
    except (TypeError, ValueError) as exc:
        raise ContractError(
            "FeaturePacked LayerNorm contract is not canonical finite JSON"
        ) from exc
    if (
        value["contract_sha256"] != FEATURE_PACKED_LAYERNORM_CONTRACT_SHA256
        or contract_hash != value["contract_sha256"]
    ):
        raise ContractError(
            "FeaturePacked LayerNorm contract matrix or contract SHA-256 changed"
        )
    return scales


def extract_approximation_binding(source: dict[str, Any]) -> dict[str, Any]:
    try:
        operators = source["operators"]
        softmax = operators["softmax"]
        layernorm = operators["layernorm"]
        feature_packed_layernorm = layernorm["feature_packed_trace_scale_contract"]
        shift = softmax["shift_contract"]
        return {
            "softmax_shift": {
                "kind": shift["kind"],
                "axis_order": shift["axis_order"],
                "shape": shift["shape"],
                "shared_across_all_query_rows": shift["shared_across_all_query_rows"],
                "runtime_activation_dependency": shift["runtime_activation_dependency"],
                "values_sha256": shift["values_sha256"],
            },
            "polynomials": {
                "gelu": selected_polynomial(operators["gelu"]["polynomial"]),
                "exp": selected_polynomial(softmax["exponential"]),
                "reciprocal": selected_polynomial(softmax["reciprocal"]),
                "inverse_sqrt": selected_polynomial(layernorm["inverse_sqrt"]),
            },
            "layernorm": {
                "epsilon": layernorm["epsilon"]["value"],
                "ln1_variance_scale_D_s": layernorm["sites"]["ln1"][
                    "variance_scale_D_s"
                ],
                "ln2_variance_scale_D_s": layernorm["sites"]["ln2"][
                    "variance_scale_D_s"
                ],
                "feature_packed_trace_scale_contract": (
                    selected_feature_packed_layernorm_binding(feature_packed_layernorm)
                ),
            },
        }
    except (KeyError, TypeError) as exc:
        raise ContractError(
            f"approximation source is missing required fields: {exc}"
        ) from exc


def approximation_runtime(
    source: dict[str, Any], binding: dict[str, Any]
) -> dict[str, Any]:
    extracted = extract_approximation_binding(source)
    if binding != extracted:
        raise ContractError(
            "approximation_binding differs from the bound approximation source"
        )
    try:
        layernorm = source["operators"]["layernorm"]
        trace_scale_contract = layernorm["feature_packed_trace_scale_contract"]
    except (KeyError, TypeError) as exc:
        raise ContractError(
            "approximation source is missing the FeaturePacked LayerNorm "
            f"trace-scale contract: {exc}"
        ) from exc
    trace_scales = validate_feature_packed_layernorm_contract(
        trace_scale_contract,
        layernorm["epsilon"]["value"],
    )
    shifts = np.asarray(
        source["operators"]["softmax"]["shift_contract"]["values"], dtype=np.float64
    )
    if (
        shifts.shape != (12, 12)
        or tensor_sha256(shifts) != binding["softmax_shift"]["values_sha256"]
    ):
        raise ContractError("softmax shift tensor shape or binary64 SHA-256 changed")

    functions: dict[str, Callable[[float], float]] = {
        "gelu": exact_gelu_scalar,
        "exp": math.exp,
        "reciprocal": lambda value: 1.0 / value,
        "inverse_sqrt": lambda value: 1.0 / math.sqrt(value),
    }
    coefficients: dict[str, np.ndarray] = {}
    for name, polynomial in binding["polynomials"].items():
        interval = polynomial["interval"]
        values = openfhe_chebyshev_coefficients(
            functions[name],
            interval["minimum"],
            interval["maximum"],
            polynomial["degree"],
        )
        if name == "gelu":
            values = constrain_zero_at_origin(
                values, interval["minimum"], interval["maximum"]
            )
        if len(values) != polynomial["coefficient_count"]:
            raise ContractError(f"{name} coefficient count changed")
        actual_hash = coefficient_sha256(values)
        if actual_hash != polynomial["coefficient_sha256"]:
            raise ContractError(
                f"{name} coefficient SHA-256 changed: expected "
                f"{polynomial['coefficient_sha256']}, got {actual_hash}"
            )
        coefficients[name] = values
    return {
        "shifts": shifts,
        "coefficients": coefficients,
        "feature_packed_layernorm_scales": trace_scales,
        "feature_packed_layernorm_contract": binding["layernorm"][
            "feature_packed_trace_scale_contract"
        ],
    }


def extract_scale_layers(source: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        layers = source["layers"]
    except (KeyError, TypeError) as exc:
        raise ContractError("channel-scale source has no layers") from exc
    if not isinstance(layers, list) or len(layers) != 12:
        raise ContractError("channel-scale source must contain 12 layers")
    result = []
    for layer_id, layer in enumerate(layers):
        if layer.get("layer_id") != layer_id:
            raise ContractError("channel-scale layers must be ordered 0..11")
        result.append(
            {
                "layer_id": layer_id,
                "channel_scale_default": layer["channel_scale_default"],
                "channel_scale_overrides": layer["channel_scale_overrides"],
            }
        )
    return result


def validate_scales(layer: dict[str, Any], expected: dict[str, Any]) -> np.ndarray:
    if layer["layer_id"] != expected["layer_id"]:
        raise ContractError("layer scale id differs from source")
    for key in ("channel_scale_default", "channel_scale_overrides"):
        if layer[key] != expected[key]:
            raise ContractError(
                f"layer {layer['layer_id']} {key} differs from scale source"
            )
    if (
        type(layer["channel_scale_default"]) is not int
        or layer["channel_scale_default"] != 1
    ):
        raise ContractError("channel_scale_default must be integer one")
    scales = np.ones(3072, dtype=np.float64)
    overrides = layer["channel_scale_overrides"]
    if not isinstance(overrides, dict):
        raise ContractError("channel_scale_overrides must be an object")
    for raw_index, scale in overrides.items():
        try:
            index = int(raw_index)
        except (TypeError, ValueError) as exc:
            raise ContractError(f"invalid channel-scale index {raw_index!r}") from exc
        if str(index) != raw_index or not 0 <= index < 3072:
            raise ContractError(f"invalid channel-scale index {raw_index!r}")
        if type(scale) is not int or scale <= 1:
            raise ContractError(f"channel scale {raw_index} must be integer > 1")
        scales[index] = scale
    return scales


def load_layer(
    data_root: Path, layer: dict[str, Any], verify_hashes: bool
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    layer_id = layer["layer_id"]
    root = data_root / f"layer_{layer_id}"
    if not root.is_dir():
        raise ContractError(f"layer directory is missing: {root}")
    arrays: dict[str, np.ndarray] = {}
    digests: dict[str, str] = {}
    for file_id, spec in FILE_SPECS.items():
        path = root / spec["path"]
        if not path.is_file():
            raise ContractError(f"layer {layer_id}: missing {spec['path']}")
        digest = sha256_file(path)
        digests[file_id] = digest
        if verify_hashes and digest != layer["sha256"][file_id]:
            raise ContractError(
                f"layer {layer_id}: SHA-256 mismatch for {spec['path']}: "
                f"expected {layer['sha256'][file_id]}, got {digest}"
            )
        try:
            values = np.loadtxt(path, delimiter=",", dtype=np.float64)
        except (OSError, ValueError) as exc:
            raise ContractError(
                f"layer {layer_id}: cannot parse {spec['path']}: {exc}"
            ) from exc
        expected_shape = tuple(spec["shape"])
        if values.shape != expected_shape:
            raise ContractError(
                f"layer {layer_id}: {spec['path']} shape {values.shape}, "
                f"expected {expected_shape}"
            )
        if not np.isfinite(values).all():
            raise ContractError(f"layer {layer_id}: {spec['path']} contains NaN or Inf")
        arrays[file_id] = values
    token_ids = arrays["token_ids"]
    if np.any(token_ids < 0.0) or not np.array_equal(token_ids, np.rint(token_ids)):
        raise ContractError(f"layer {layer_id}: token ids must be nonnegative integers")
    return arrays, digests


def layer_scales(layer: dict[str, Any]) -> np.ndarray:
    scales = np.ones(3072, dtype=np.float64)
    for raw_index, scale in layer["channel_scale_overrides"].items():
        scales[int(raw_index)] = scale
    return scales


def static_relations(
    arrays: dict[str, np.ndarray], scales: np.ndarray
) -> dict[str, Any]:
    embedded = arrays["embedded_inputs"]
    query = embedded @ arrays["query_weight"].T + arrays["query_bias"]
    key = embedded @ arrays["key_weight"].T + arrays["key_bias"]
    value = embedded @ arrays["value_weight"].T + arrays["value_bias"]
    query_heads = arrays["query_output"].reshape(5, 12, 64)
    key_heads = arrays["key_output"].reshape(5, 12, 64)
    qkt = np.einsum("qhd,khd->qhk", query_heads, key_heads) / 8.0
    stored_qkt = arrays["qkt_output"].reshape(5, 12, 5)
    stable = stored_qkt - np.max(stored_qkt, axis=-1, keepdims=True)
    softmax = np.exp(stable)
    softmax /= np.sum(softmax, axis=-1, keepdims=True)
    attention = np.einsum(
        "qhk,khd->qhd",
        arrays["softmax_output"].reshape(5, 12, 5),
        arrays["value_output"].reshape(5, 12, 64),
    ).reshape(5, 768)
    self_linear = arrays["self_inputs"] @ arrays["self_dense_weight"].T
    self_linear += arrays["self_dense_bias"]
    self_before_ln = arrays["self_after_linear"] + arrays["self_residual_operand"]
    self_ln = exact_layernorm(
        arrays["self_before_layernorm"],
        arrays["self_layernorm_weight"],
        arrays["self_layernorm_bias"],
        1e-12,
    )
    intermediate = arrays["intermediate_inputs"] @ arrays["intermediate_weight"].T
    intermediate = (intermediate + arrays["intermediate_bias"]) * scales
    gelu = exact_gelu(arrays["intermediate_after_linear"])
    final = (
        arrays["intermediate_output"]
        @ (arrays["final_dense_weight"] / scales[np.newaxis, :]).T
    )
    final += arrays["final_dense_bias"]
    final_before_ln = arrays["final_after_linear"] + arrays["final_residual_operand"]
    final_ln = exact_layernorm(
        arrays["final_before_layernorm"],
        arrays["final_layernorm_weight"],
        arrays["final_layernorm_bias"],
        1e-12,
    )
    comparisons = {
        "query_projection": (query, arrays["query_output"]),
        "key_projection": (key, arrays["key_output"]),
        "value_projection": (value, arrays["value_output"]),
        "qkt_scaled_by_eight": (qkt, stored_qkt),
        "exact_softmax": (softmax, arrays["softmax_output"].reshape(5, 12, 5)),
        "attention_times_value": (attention, arrays["attention_output"]),
        "attention_snapshot_link": (arrays["attention_output"], arrays["self_inputs"]),
        "self_projection": (self_linear, arrays["self_after_linear"]),
        "self_residual": (self_before_ln, arrays["self_before_layernorm"]),
        "self_layernorm_exact": (self_ln, arrays["self_output"]),
        "self_to_ffn_link": (arrays["self_output"], arrays["intermediate_inputs"]),
        "intermediate_projection_scaled": (
            intermediate,
            arrays["intermediate_after_linear"],
        ),
        "exact_gelu": (gelu, arrays["intermediate_output"]),
        "gelu_snapshot_link": (arrays["intermediate_output"], arrays["final_inputs"]),
        "final_projection_unscaled": (final, arrays["final_after_linear"]),
        "final_residual": (final_before_ln, arrays["final_before_layernorm"]),
        "final_layernorm_exact": (final_ln, arrays["final_output"]),
    }
    metrics = {
        name: quality(actual, expected)
        for name, (actual, expected) in comparisons.items()
    }
    for name, value in metrics.items():
        require_metric_gate(f"static {name}", value, TRACE_RELATION_THRESHOLDS[name])
    return metrics


def chained_layer(
    layer_id: int,
    values: np.ndarray,
    arrays: dict[str, np.ndarray],
    scales: np.ndarray,
    approximation: dict[str, Any],
    binding: dict[str, Any],
) -> tuple[np.ndarray, dict[str, list[float]]]:
    coefficients = approximation["coefficients"]
    polynomials = binding["polynomials"]
    query = values @ arrays["query_weight"].T + arrays["query_bias"]
    key = values @ arrays["key_weight"].T + arrays["key_bias"]
    projected_value = values @ arrays["value_weight"].T + arrays["value_bias"]
    qkt = (
        np.einsum("qhd,khd->qhk", query.reshape(5, 12, 64), key.reshape(5, 12, 64))
        / 8.0
    )
    shifted = qkt - approximation["shifts"][layer_id][None, :, None]
    shifted_range = require_range(
        f"layer {layer_id} softmax shifted logits",
        shifted,
        THRESHOLDS["hard_intervals"]["softmax_shifted_logits"],
    )
    exp_interval = polynomials["exp"]["interval"]
    numerators = evaluate_openfhe_chebyshev(
        shifted,
        coefficients["exp"],
        exp_interval["minimum"],
        exp_interval["maximum"],
    )
    denominators = np.sum(numerators, axis=-1)
    denominator_range = require_range(
        f"layer {layer_id} softmax denominator",
        denominators,
        THRESHOLDS["hard_intervals"]["softmax_denominator"],
    )
    reciprocal_interval = polynomials["reciprocal"]["interval"]
    inverse_denominators = evaluate_openfhe_chebyshev(
        denominators,
        coefficients["reciprocal"],
        reciprocal_interval["minimum"],
        reciprocal_interval["maximum"],
    )
    probabilities = numerators * inverse_denominators[:, :, None]
    attention = np.einsum(
        "qhk,khd->qhd", probabilities, projected_value.reshape(5, 12, 64)
    ).reshape(5, 768)
    self_linear = attention @ arrays["self_dense_weight"].T + arrays["self_dense_bias"]
    self_before_ln = self_linear + values
    inverse_sqrt = polynomials["inverse_sqrt"]
    layernorm = binding["layernorm"]
    registered_trace_scales = approximation["feature_packed_layernorm_scales"]
    self_output, ln1_range = polynomial_layernorm(
        self_before_ln,
        arrays["self_layernorm_weight"],
        arrays["self_layernorm_bias"],
        layernorm["epsilon"],
        registered_trace_scales[0, layer_id, :],
        coefficients["inverse_sqrt"],
        [inverse_sqrt["interval"]["minimum"], inverse_sqrt["interval"]["maximum"]],
        f"layer {layer_id} LN1 normalized variance",
    )
    intermediate = self_output @ arrays["intermediate_weight"].T
    intermediate = (intermediate + arrays["intermediate_bias"]) * scales
    gelu_range = require_range(
        f"layer {layer_id} GELU input",
        intermediate,
        THRESHOLDS["hard_intervals"]["gelu_input"],
    )
    gelu_polynomial = polynomials["gelu"]
    activated = evaluate_openfhe_chebyshev(
        intermediate,
        coefficients["gelu"],
        gelu_polynomial["interval"]["minimum"],
        gelu_polynomial["interval"]["maximum"],
    )
    final_linear = activated @ (arrays["final_dense_weight"] / scales[np.newaxis, :]).T
    final_linear += arrays["final_dense_bias"]
    final_before_ln = final_linear + self_output
    output, ln2_range = polynomial_layernorm(
        final_before_ln,
        arrays["final_layernorm_weight"],
        arrays["final_layernorm_bias"],
        layernorm["epsilon"],
        registered_trace_scales[1, layer_id, :],
        coefficients["inverse_sqrt"],
        [inverse_sqrt["interval"]["minimum"], inverse_sqrt["interval"]["maximum"]],
        f"layer {layer_id} LN2 normalized variance",
    )
    return output, {
        "softmax_shifted_logits": shifted_range,
        "softmax_denominator": denominator_range,
        "ln1_normalized_variance": ln1_range,
        "gelu_input": gelu_range,
        "ln2_normalized_variance": ln2_range,
    }


def aggregate_relation_metrics(
    per_layer: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for name in TRACE_RELATION_THRESHOLDS:
        values = [entry["relations"][name] for entry in per_layer]
        result[name] = {
            "worst_relative_l2": max(item["relative_l2"] for item in values),
            "minimum_cosine": min(item["cosine"] for item in values),
            "maximum_absolute": max(item["max_absolute"] for item in values),
        }
    return result


def aggregate_ranges(per_layer: list[dict[str, Any]]) -> dict[str, list[float]]:
    names = next(iter(per_layer))["ranges"]
    return {
        name: [
            min(item["ranges"][name][0] for item in per_layer),
            max(item["ranges"][name][1] for item in per_layer),
        ]
        for name in names
    }


def offline_feature_packed_layernorm_scale(
    raw_population_variance: float,
) -> float:
    if not math.isfinite(raw_population_variance) or raw_population_variance <= 0.0:
        raise ContractError(
            "static FeaturePacked LayerNorm raw population variance must be "
            "finite and positive"
        )
    exponent = round(
        math.log2(FEATURE_PACKED_LAYERNORM_SCALE_TARGET / raw_population_variance)
    )
    return math.ldexp(1.0, exponent)


def validate_static_layernorm_trace_scales(
    raw_variances: np.ndarray,
    approximation: dict[str, Any],
) -> dict[str, Any]:
    if raw_variances.shape != FEATURE_PACKED_LAYERNORM_SHAPE:
        raise ContractError("static FeaturePacked LayerNorm raw-variance shape changed")
    if not np.isfinite(raw_variances).all() or np.any(raw_variances <= 0.0):
        raise ContractError(
            "static FeaturePacked LayerNorm raw variances must be finite and positive"
        )
    contract = approximation["feature_packed_layernorm_contract"]
    raw_hash = tensor_sha256(raw_variances)
    if raw_hash != contract["raw_variance_sha256"]:
        raise ContractError(
            "static residual CSV population-variance SHA-256 changed: "
            f"expected {contract['raw_variance_sha256']}, got {raw_hash}"
        )
    selected_scales = np.fromiter(
        (
            offline_feature_packed_layernorm_scale(float(raw_variance))
            for raw_variance in raw_variances.flat
        ),
        dtype=np.float64,
        count=raw_variances.size,
    ).reshape(FEATURE_PACKED_LAYERNORM_SHAPE)
    registered_scales = approximation["feature_packed_layernorm_scales"]
    if not np.array_equal(selected_scales, registered_scales):
        differing = np.argwhere(selected_scales != registered_scales)
        coordinate = tuple(int(index) for index in differing[0])
        raise ContractError(
            "static residual CSV scale selection differs from the "
            "pre-registered [site,layer,token] matrix at "
            f"{coordinate}: selected={selected_scales[coordinate]}, "
            f"registered={registered_scales[coordinate]}"
        )
    selected_hash = tensor_sha256(selected_scales)
    if selected_hash != contract["values_sha256"]:
        raise ContractError(
            "static residual CSV selected-scale SHA-256 changed: "
            f"expected {contract['values_sha256']}, got {selected_hash}"
        )
    return {
        "source": (
            "registered self_before_layernorm/final_before_layernorm residual CSVs only"
        ),
        "raw_population_variance_shape": list(raw_variances.shape),
        "raw_population_variance_sha256": raw_hash,
        "selection_formula": FEATURE_PACKED_LAYERNORM_SELECTION_FORMULA,
        "selected_scale_values_sha256": selected_hash,
        "registered_matrix_exact_match": True,
        "runtime_activation_dependency": "none",
        "passed": True,
    }


def run_trace(
    data_root: Path,
    layers: list[dict[str, Any]],
    approximation: dict[str, Any],
    binding: dict[str, Any],
    verify_hashes: bool,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, str]]]:
    if not data_root.is_dir():
        raise ContractError(f"data root is not a directory: {data_root}")
    static_layers: list[dict[str, Any]] = []
    chained_layers: list[dict[str, Any]] = []
    observed_hashes: list[dict[str, str]] = []
    previous_output_digest: str | None = None
    first_token_digest: str | None = None
    chained_values: np.ndarray | None = None
    static_layernorm_raw_variances = np.empty(
        FEATURE_PACKED_LAYERNORM_SHAPE,
        dtype=np.float64,
    )
    for layer_id, layer in enumerate(layers):
        arrays, digests = load_layer(data_root, layer, verify_hashes)
        observed_hashes.append(digests)
        if digests["embedded_inputs"] != digests["self_residual_operand"]:
            raise ContractError(
                f"layer {layer_id}: attention residual is not byte-identical"
            )
        if digests["intermediate_inputs"] != digests["final_residual_operand"]:
            raise ContractError(f"layer {layer_id}: FFN residual is not byte-identical")
        if (
            previous_output_digest is not None
            and previous_output_digest != digests["embedded_inputs"]
        ):
            raise ContractError(
                f"layer {layer_id - 1}->{layer_id}: chain is not byte-identical"
            )
        previous_output_digest = digests["final_output"]
        if first_token_digest is None:
            first_token_digest = digests["token_ids"]
        elif first_token_digest != digests["token_ids"]:
            raise ContractError(f"layer {layer_id}: token-id bytes differ from layer 0")

        for site_index, file_id in enumerate(
            ("self_before_layernorm", "final_before_layernorm")
        ):
            residual = arrays[file_id]
            centered_residual = residual - np.mean(residual, axis=1, keepdims=True)
            static_layernorm_raw_variances[site_index, layer_id, :] = np.mean(
                centered_residual * centered_residual,
                axis=1,
            )
        scales = layer_scales(layer)
        relations = static_relations(arrays, scales)
        static_layers.append({"layer_id": layer_id, "relations": relations})
        if chained_values is None:
            chained_values = arrays["embedded_inputs"].copy()
        chained_values, ranges = chained_layer(
            layer_id, chained_values, arrays, scales, approximation, binding
        )
        metrics = quality(chained_values, arrays["final_output"])
        require_metric_gate(
            f"layer {layer_id} chained polynomial oracle",
            metrics,
            THRESHOLDS["chained_polynomial_oracle_per_layer"],
        )
        chained_layers.append(
            {"layer_id": layer_id, "metrics": metrics, "ranges": ranges}
        )
        del arrays

    if chained_values is None:
        raise ContractError("no encoder layers were processed")
    trace_scale_summary = validate_static_layernorm_trace_scales(
        static_layernorm_raw_variances,
        approximation,
    )
    final_metrics = chained_layers[-1]["metrics"]
    require_metric_gate(
        "12-layer chained polynomial oracle",
        final_metrics,
        THRESHOLDS["chained_polynomial_oracle_final"],
    )
    static_summary = {
        "file_count": 12 * len(FILE_SPECS),
        "relation_count_per_layer": len(TRACE_RELATION_THRESHOLDS),
        "byte_equality_checks": {
            "attention_residual": 12,
            "ffn_residual": 12,
            "cross_layer": 11,
            "token_ids_across_layers": 11,
        },
        "feature_packed_layernorm_scale_registry": trace_scale_summary,
        "aggregates": aggregate_relation_metrics(static_layers),
        "per_layer": static_layers,
        "passed": True,
    }
    chained_summary = {
        "oracle": "frozen OpenFHE-order Chebyshev plaintext replay",
        "feature_packed_layernorm_scale_selection": {
            "contract_id": approximation["feature_packed_layernorm_contract"][
                "contract_id"
            ],
            "contract_sha256": approximation["feature_packed_layernorm_contract"][
                "contract_sha256"
            ],
            "coordinate_order": ["site", "layer", "token"],
            "selection": (
                "pre-registered public coordinate lookup only; chained "
                "activations are never inspected to select D"
            ),
            "runtime_activation_dependency": "none",
        },
        "per_layer_gate": THRESHOLDS["chained_polynomial_oracle_per_layer"],
        "final_gate": THRESHOLDS["chained_polynomial_oracle_final"],
        "worst_relative_l2": max(
            item["metrics"]["relative_l2"] for item in chained_layers
        ),
        "minimum_cosine": min(item["metrics"]["cosine"] for item in chained_layers),
        "maximum_absolute": max(
            item["metrics"]["max_absolute"] for item in chained_layers
        ),
        "observed_ranges": aggregate_ranges(chained_layers),
        "per_layer": chained_layers,
        "final": {"layer_id": 11, **final_metrics},
        "passed": True,
    }
    return static_summary, chained_summary, observed_hashes


def source_contracts() -> tuple[dict[str, Any], dict[str, Any]]:
    return load_json(SCALE_MANIFEST_PATH), load_json(APPROXIMATION_PATH)


def source_entries() -> dict[str, dict[str, str]]:
    return {
        "channel_scales": {
            "path": "config/moai_trace_channel_scales.json",
            "sha256": sha256_file(SCALE_MANIFEST_PATH),
        },
        "approximations": {
            "path": "config/openfhe_approximations.json",
            "sha256": sha256_file(APPROXIMATION_PATH),
        },
    }


def build_manifest(data_root: Path) -> dict[str, Any]:
    scale_source, approximation_source = source_contracts()
    scale_layers = extract_scale_layers(scale_source)
    binding = extract_approximation_binding(approximation_source)
    approximation = approximation_runtime(approximation_source, binding)
    layers = [
        {
            **scale_layer,
            "sha256": {file_id: "0" * 64 for file_id in FILE_SPECS},
        }
        for scale_layer in scale_layers
    ]
    # First pass freezes all file bytes.  The measurement pass deliberately uses
    # the same loader again so candidate generation exercises the validation path.
    for layer in layers:
        _, digests = load_layer(data_root, layer, verify_hashes=False)
        layer["sha256"] = digests
    static_summary, chained_summary, _ = run_trace(
        data_root, layers, approximation, binding, verify_hashes=False
    )
    return {
        "schema_version": 1,
        "contract_id": "moai-openfhe-bert-base-five-token-encoder-trace-v1",
        "validator_version": 1,
        "profile_id": "paper_compat",
        "security_claim": "none",
        "hash_algorithm": "sha256",
        "scope": SCOPE,
        "dimensions": DIMENSIONS,
        "sources": source_entries(),
        "approximation_binding": binding,
        "thresholds": THRESHOLDS,
        "required_files": FILE_SPECS,
        "layers": layers,
        "static_trace_summary": static_summary,
        "chained_oracle_summary": chained_summary,
    }


def validate_source_entry(
    entry: Any, expected_path: str, actual_path: Path, label: str
) -> None:
    value = require_exact_keys(entry, {"path", "sha256"}, f"sources.{label}")
    if value["path"] != expected_path:
        raise ContractError(f"sources.{label}.path changed")
    digest = value["sha256"]
    if not isinstance(digest, str) or SHA256_PATTERN.fullmatch(digest) is None:
        raise ContractError(f"sources.{label}.sha256 is invalid")
    actual = sha256_file(actual_path)
    if actual != digest:
        raise ContractError(
            f"sources.{label} SHA-256 mismatch: expected {digest}, got {actual}"
        )


def validate_manifest_structure(
    manifest: dict[str, Any],
    scale_source: dict[str, Any],
    approximation_source: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    require_exact_keys(manifest, TOP_LEVEL_KEYS, "manifest")
    fixed = {
        "schema_version": 1,
        "contract_id": "moai-openfhe-bert-base-five-token-encoder-trace-v1",
        "validator_version": 1,
        "profile_id": "paper_compat",
        "security_claim": "none",
        "hash_algorithm": "sha256",
        "scope": SCOPE,
        "dimensions": DIMENSIONS,
        "thresholds": THRESHOLDS,
        "required_files": FILE_SPECS,
    }
    for key, expected in fixed.items():
        if manifest[key] != expected:
            raise ContractError(
                f"manifest.{key} differs from the locked validator contract"
            )
    sources = require_exact_keys(
        manifest["sources"], {"channel_scales", "approximations"}, "sources"
    )
    validate_source_entry(
        sources["channel_scales"],
        "config/moai_trace_channel_scales.json",
        SCALE_MANIFEST_PATH,
        "channel_scales",
    )
    validate_source_entry(
        sources["approximations"],
        "config/openfhe_approximations.json",
        APPROXIMATION_PATH,
        "approximations",
    )
    binding = extract_approximation_binding(approximation_source)
    if manifest["approximation_binding"] != binding:
        raise ContractError("manifest approximation_binding changed")
    approximation = approximation_runtime(approximation_source, binding)
    scale_layers = extract_scale_layers(scale_source)
    layers = manifest["layers"]
    if not isinstance(layers, list) or len(layers) != 12:
        raise ContractError("manifest.layers must contain exactly 12 entries")
    for layer_id, layer in enumerate(layers):
        value = require_exact_keys(
            layer,
            {"layer_id", "channel_scale_default", "channel_scale_overrides", "sha256"},
            f"layers[{layer_id}]",
        )
        if type(value["layer_id"]) is not int or value["layer_id"] != layer_id:
            raise ContractError(
                f"layers[{layer_id}].layer_id must be integer {layer_id}"
            )
        validate_scales(value, scale_layers[layer_id])
        hashes = require_exact_keys(
            value["sha256"], set(FILE_SPECS), f"layers[{layer_id}].sha256"
        )
        for file_id, digest in hashes.items():
            if not isinstance(digest, str) or SHA256_PATTERN.fullmatch(digest) is None:
                raise ContractError(f"layer {layer_id}: invalid SHA-256 for {file_id}")
    if not isinstance(manifest["static_trace_summary"], dict):
        raise ContractError("static_trace_summary must be an object")
    if not isinstance(manifest["chained_oracle_summary"], dict):
        raise ContractError("chained_oracle_summary must be an object")
    return layers, binding, approximation


def compare_frozen(actual: Any, expected: Any, label: str) -> None:
    """Compare summaries while allowing only last-bit BLAS variation."""
    if isinstance(expected, bool) or expected is None or isinstance(expected, str):
        if actual != expected:
            raise ContractError(
                f"{label} changed: expected {expected!r}, got {actual!r}"
            )
        return
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if not isinstance(actual, (int, float)) or isinstance(actual, bool):
            raise ContractError(f"{label} numeric type changed")
        if not math.isfinite(float(actual)) or not math.isclose(
            float(actual), float(expected), rel_tol=1e-11, abs_tol=1e-13
        ):
            raise ContractError(f"{label} changed: expected {expected}, got {actual}")
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ContractError(f"{label} list shape changed")
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            compare_frozen(actual_item, expected_item, f"{label}[{index}]")
        return
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ContractError(f"{label} object keys changed")
        for key in expected:
            compare_frozen(actual[key], expected[key], f"{label}.{key}")
        return
    raise ContractError(f"{label} contains unsupported JSON value")


def validate_contract(data_root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    scale_source, approximation_source = source_contracts()
    layers, binding, approximation = validate_manifest_structure(
        manifest, scale_source, approximation_source
    )
    static_summary, chained_summary, _ = run_trace(
        data_root, layers, approximation, binding, verify_hashes=True
    )
    compare_frozen(
        static_summary, manifest["static_trace_summary"], "static_trace_summary"
    )
    compare_frozen(
        chained_summary, manifest["chained_oracle_summary"], "chained_oracle_summary"
    )
    return {"static": static_summary, "chained": chained_summary}


def print_report(report: dict[str, Any]) -> None:
    chained = report["chained"]
    for item in chained["per_layer"]:
        metrics = item["metrics"]
        print(
            f"layer={item['layer_id']:02d} chained_rel_l2={metrics['relative_l2']:.9e} "
            f"cosine={metrics['cosine']:.12f} max_abs={metrics['max_absolute']:.9e}"
        )
    final = chained["final"]
    print(
        f"final_rel_l2={final['relative_l2']:.9e} final_cosine={final['cosine']:.12f} "
        f"final_max_abs={final['max_absolute']:.9e}"
    )
    print(f"validated_files={report['static']['file_count']}")


def write_manifest_atomic(path: Path, manifest: dict[str, Any]) -> Path:
    if path.is_symlink():
        raise ContractError(f"refusing to replace a symlink: {path}")
    target = path.resolve(strict=False)
    repository = REPO_ROOT.resolve()
    if target != repository and repository not in target.parents:
        raise ContractError(
            f"generated manifest output must stay inside the MOAI repository: {target}"
        )
    if not target.parent.is_dir():
        raise ContractError(
            f"generated manifest parent directory does not exist: {target.parent}"
        )
    if target.exists() and not target.is_file():
        raise ContractError(
            f"generated manifest target is not a regular file: {target}"
        )
    mode = target.stat().st_mode & 0o777 if target.exists() else 0o644
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                manifest,
                stream,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, target)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return target


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate/freeze the full MOAI five-token 12-layer encoder trace."
    )
    parser.add_argument(
        "--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="MOAI trace data root"
    )
    parser.add_argument(
        "--manifest", type=Path, default=DEFAULT_MANIFEST, help="frozen trace contract"
    )
    parser.add_argument(
        "--emit-config",
        action="store_true",
        help="emit a newly measured contract to stdout or the explicit --output path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "atomically write --emit-config output inside the MOAI repository; "
            "refuses symlinks and non-regular targets"
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.output is not None and not args.emit_config:
            raise ContractError("--output requires --emit-config")
        if args.emit_config:
            manifest = build_manifest(args.data_root)
            if args.output is None:
                json.dump(
                    manifest,
                    sys.stdout,
                    sort_keys=True,
                    indent=2,
                    allow_nan=False,
                )
                sys.stdout.write("\n")
            else:
                output = write_manifest_atomic(args.output, manifest)
                print(f"wrote_manifest={output}")
            return 0
        manifest = load_json(args.manifest)
        report = validate_contract(args.data_root, manifest)
        print_report(report)
        print(f"DECISION={PASS_DECISION}")
        return 0
    except (ContractError, KeyError, OSError, TypeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print(f"DECISION={FAIL_DECISION}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
