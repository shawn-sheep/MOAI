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
import re
import sys
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
        "cosine": float(np.vdot(actual_flat, expected_flat) / (actual_norm * expected_norm)),
        "max_absolute": float(np.max(np.abs(delta))),
    }


def require_metric_gate(label: str, metrics: dict[str, float], gate: dict[str, float]) -> None:
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
    variance_scale: float,
    inverse_sqrt_coefficients: np.ndarray,
    interval: list[float],
    range_label: str,
) -> tuple[np.ndarray, list[float]]:
    centered = values - np.mean(values, axis=-1, keepdims=True)
    variance = np.mean(centered * centered, axis=-1, keepdims=True) + epsilon
    normalized_variance = variance_scale * variance
    observed = require_range(range_label, normalized_variance, interval)
    inverse = math.sqrt(variance_scale) * evaluate_openfhe_chebyshev(
        normalized_variance,
        inverse_sqrt_coefficients,
        interval[0],
        interval[1],
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


def extract_approximation_binding(source: dict[str, Any]) -> dict[str, Any]:
    try:
        operators = source["operators"]
        softmax = operators["softmax"]
        layernorm = operators["layernorm"]
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
                "ln1_variance_scale_D_s": layernorm["sites"]["ln1"]["variance_scale_D_s"],
                "ln2_variance_scale_D_s": layernorm["sites"]["ln2"]["variance_scale_D_s"],
            },
        }
    except (KeyError, TypeError) as exc:
        raise ContractError(f"approximation source is missing required fields: {exc}") from exc


def approximation_runtime(
    source: dict[str, Any], binding: dict[str, Any]
) -> dict[str, Any]:
    extracted = extract_approximation_binding(source)
    if binding != extracted:
        raise ContractError("approximation_binding differs from the bound approximation source")
    shifts = np.asarray(
        source["operators"]["softmax"]["shift_contract"]["values"], dtype=np.float64
    )
    if shifts.shape != (12, 12) or tensor_sha256(shifts) != binding["softmax_shift"][
        "values_sha256"
    ]:
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
            functions[name], interval["minimum"], interval["maximum"], polynomial["degree"]
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
    return {"shifts": shifts, "coefficients": coefficients}


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
            raise ContractError(f"layer {layer['layer_id']} {key} differs from scale source")
    if type(layer["channel_scale_default"]) is not int or layer["channel_scale_default"] != 1:
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
            raise ContractError(f"layer {layer_id}: cannot parse {spec['path']}: {exc}") from exc
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


def static_relations(arrays: dict[str, np.ndarray], scales: np.ndarray) -> dict[str, Any]:
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
    final = arrays["intermediate_output"] @ (
        arrays["final_dense_weight"] / scales[np.newaxis, :]
    ).T
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
    metrics = {name: quality(actual, expected) for name, (actual, expected) in comparisons.items()}
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
    qkt = np.einsum(
        "qhd,khd->qhk", query.reshape(5, 12, 64), key.reshape(5, 12, 64)
    ) / 8.0
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
    self_output, ln1_range = polynomial_layernorm(
        self_before_ln,
        arrays["self_layernorm_weight"],
        arrays["self_layernorm_bias"],
        layernorm["epsilon"],
        layernorm["ln1_variance_scale_D_s"],
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
    final_linear = activated @ (
        arrays["final_dense_weight"] / scales[np.newaxis, :]
    ).T
    final_linear += arrays["final_dense_bias"]
    final_before_ln = final_linear + self_output
    output, ln2_range = polynomial_layernorm(
        final_before_ln,
        arrays["final_layernorm_weight"],
        arrays["final_layernorm_bias"],
        layernorm["epsilon"],
        layernorm["ln2_variance_scale_D_s"],
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
    per_layer: list[dict[str, Any]]
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
    for layer_id, layer in enumerate(layers):
        arrays, digests = load_layer(data_root, layer, verify_hashes)
        observed_hashes.append(digests)
        if digests["embedded_inputs"] != digests["self_residual_operand"]:
            raise ContractError(f"layer {layer_id}: attention residual is not byte-identical")
        if digests["intermediate_inputs"] != digests["final_residual_operand"]:
            raise ContractError(f"layer {layer_id}: FFN residual is not byte-identical")
        if previous_output_digest is not None and previous_output_digest != digests[
            "embedded_inputs"
        ]:
            raise ContractError(f"layer {layer_id - 1}->{layer_id}: chain is not byte-identical")
        previous_output_digest = digests["final_output"]
        if first_token_digest is None:
            first_token_digest = digests["token_ids"]
        elif first_token_digest != digests["token_ids"]:
            raise ContractError(f"layer {layer_id}: token-id bytes differ from layer 0")

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
        chained_layers.append({"layer_id": layer_id, "metrics": metrics, "ranges": ranges})
        del arrays

    if chained_values is None:
        raise ContractError("no encoder layers were processed")
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
        "aggregates": aggregate_relation_metrics(static_layers),
        "per_layer": static_layers,
        "passed": True,
    }
    chained_summary = {
        "oracle": "frozen OpenFHE-order Chebyshev plaintext replay",
        "per_layer_gate": THRESHOLDS["chained_polynomial_oracle_per_layer"],
        "final_gate": THRESHOLDS["chained_polynomial_oracle_final"],
        "worst_relative_l2": max(item["metrics"]["relative_l2"] for item in chained_layers),
        "minimum_cosine": min(item["metrics"]["cosine"] for item in chained_layers),
        "maximum_absolute": max(item["metrics"]["max_absolute"] for item in chained_layers),
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
    manifest: dict[str, Any], scale_source: dict[str, Any], approximation_source: dict[str, Any]
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
            raise ContractError(f"manifest.{key} differs from the locked validator contract")
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
            raise ContractError(f"layers[{layer_id}].layer_id must be integer {layer_id}")
        validate_scales(value, scale_layers[layer_id])
        hashes = require_exact_keys(value["sha256"], set(FILE_SPECS), f"layers[{layer_id}].sha256")
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
            raise ContractError(f"{label} changed: expected {expected!r}, got {actual!r}")
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
    compare_frozen(static_summary, manifest["static_trace_summary"], "static_trace_summary")
    compare_frozen(chained_summary, manifest["chained_oracle_summary"], "chained_oracle_summary")
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
        help="emit a newly measured contract to stdout; never writes files",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.emit_config:
            manifest = build_manifest(args.data_root)
            json.dump(manifest, sys.stdout, sort_keys=True, indent=2, allow_nan=False)
            sys.stdout.write("\n")
            return 0
        manifest = load_json(args.manifest)
        report = validate_contract(args.data_root, manifest)
        print_report(report)
        print(f"DECISION={PASS_DECISION}")
        return 0
    except (ContractError, KeyError, TypeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print(f"DECISION={FAIL_DECISION}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
