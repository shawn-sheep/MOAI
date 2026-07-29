#!/usr/bin/env python3
"""Freeze and validate the plaintext M3 nonlinear approximation contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Callable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "config" / "openfhe_approximations.json"
DEFAULT_DATA_ROOT = REPO_ROOT / "data"
DATA_ROOT = DEFAULT_DATA_ROOT
LAYER_COUNT = 12
QUERY_ROWS = 5
PACKED_QUERY_ROWS = 128
ATTENTION_HEADS = 12
HEAD_SIZE = 64
HIDDEN_SIZE = 768
RELATIVE_L2_LIMIT = 1e-2
COSINE_LIMIT = 0.999
TWO_ITERATION_BOOTSTRAP_MAX_ABS = 1e-6
SOFTMAX_DENOMINATOR_NOISE_RATIO_MIN = 100.0


def load_trace(layer: int, relative_path: str) -> np.ndarray:
    path = DATA_ROOT / f"layer_{layer}" / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"missing calibration trace: {path}")
    return np.loadtxt(path, delimiter=",", dtype=np.float64)


def openfhe_chebyshev_coefficients(
    function: Callable[[float], float],
    lower: float,
    upper: float,
    degree: int,
) -> np.ndarray:
    """Match OpenFHE EvalChebyshevCoefficients, including its doubled c0."""
    if degree == 0:
        raise ValueError("OpenFHE rejects a zero approximation degree")
    coefficient_count = degree + 1
    half_width = 0.5 * (upper - lower)
    midpoint = 0.5 * (upper + lower)
    pi_by_count = math.pi / coefficient_count
    function_points = np.empty(coefficient_count, dtype=np.float64)
    for index in range(coefficient_count):
        node = math.cos(pi_by_count * (index + 0.5)) * half_width + midpoint
        function_points[index] = function(node)

    multiplier = 2.0 / coefficient_count
    coefficients = np.zeros(coefficient_count, dtype=np.float64)
    for coefficient_index in range(coefficient_count):
        accumulator = 0.0
        for point_index in range(coefficient_count):
            accumulator += function_points[point_index] * math.cos(
                pi_by_count * coefficient_index * (point_index + 0.5)
            )
        coefficients[coefficient_index] = accumulator * multiplier
    return coefficients


def evaluate_openfhe_chebyshev(
    values: np.ndarray,
    coefficients: np.ndarray,
    lower: float,
    upper: float,
) -> np.ndarray:
    """Plaintext equivalent of EvalChebyshevSeries (the evaluator halves c0)."""
    normalized = (2.0 * values - lower - upper) / (upper - lower)
    evaluator_coefficients = coefficients.copy()
    evaluator_coefficients[0] *= 0.5
    return np.polynomial.chebyshev.chebval(normalized, evaluator_coefficients)


def coefficient_sha256(coefficients: np.ndarray) -> str:
    encoded = np.asarray(coefficients, dtype="<f8").tobytes(order="C")
    return hashlib.sha256(encoded).hexdigest()


def tensor_sha256(values: np.ndarray) -> str:
    encoded = np.asarray(values, dtype="<f8").tobytes(order="C")
    return hashlib.sha256(encoded).hexdigest()


def ps_contract(degree: int, interval: tuple[float, float]) -> dict[str, int]:
    ranges = (
        (2, 1),
        (11, 2),
        (13, 3),
        (17, 2),
        (55, 3),
        (59, 4),
        (76, 3),
        (239, 4),
        (247, 5),
        (284, 4),
        (991, 5),
        (1007, 6),
        (1083, 5),
        (2015, 6),
        (2031, 7),
        (2204, 6),
    )
    for upper_degree, m_value in ranges:
        if degree <= upper_degree:
            break
    else:
        raise ValueError("this validator only supports OpenFHE PS degrees <= 2204")
    k_value = math.floor(degree / ((1 << m_value) - 1)) + 1
    polynomial_depth = math.ceil(math.log2(k_value)) + m_value
    interval_levels = 0 if interval == (-1.0, 1.0) else 1
    multiplication_count = k_value + 2 * m_value + (1 << (m_value - 1)) - 4
    return {
        "k": k_value,
        "m": m_value,
        "polynomial_depth": polynomial_depth,
        "interval_transform_levels": interval_levels,
        "required_depth": polynomial_depth + interval_levels,
        "estimated_multiplications": multiplication_count,
    }


def quality(actual: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    actual_flat = np.asarray(actual, dtype=np.float64).reshape(-1)
    expected_flat = np.asarray(expected, dtype=np.float64).reshape(-1)
    if actual_flat.shape != expected_flat.shape:
        raise ValueError("quality tensors have different shapes")
    if not np.all(np.isfinite(actual_flat)):
        raise ValueError("approximation output contains NaN or Inf")
    expected_norm = np.linalg.norm(expected_flat)
    actual_norm = np.linalg.norm(actual_flat)
    if expected_norm == 0.0 or actual_norm == 0.0:
        raise ValueError("quality tensors must have nonzero norms")
    relative_l2 = np.linalg.norm(actual_flat - expected_flat) / expected_norm
    cosine = np.vdot(actual_flat, expected_flat) / (actual_norm * expected_norm)
    return {
        "relative_l2": float(relative_l2),
        "cosine": float(cosine),
        "max_absolute": float(np.max(np.abs(actual_flat - expected_flat))),
    }


def summarize_layer_metrics(per_layer: list[dict[str, float]]) -> dict[str, object]:
    worst_relative_l2 = max(item["relative_l2"] for item in per_layer)
    minimum_cosine = min(item["cosine"] for item in per_layer)
    maximum_absolute = max(item["max_absolute"] for item in per_layer)
    return {
        "layer_count": len(per_layer),
        "worst_relative_l2": worst_relative_l2,
        "minimum_cosine": minimum_cosine,
        "maximum_absolute": maximum_absolute,
        "gate": {
            "relative_l2_max": RELATIVE_L2_LIMIT,
            "cosine_min": COSINE_LIMIT,
            "passed": (
                worst_relative_l2 <= RELATIVE_L2_LIMIT
                and minimum_cosine >= COSINE_LIMIT
            ),
        },
        "per_layer": per_layer,
    }


def require_quality_gate(name: str, summary: dict[str, object]) -> None:
    gate = summary["gate"]
    if not gate["passed"]:
        raise RuntimeError(
            f"{name} gate failed: rel-L2={summary['worst_relative_l2']}, "
            f"cosine={summary['minimum_cosine']}"
        )


def require_range(name: str, observed: tuple[float, float], hard: tuple[float, float]) -> None:
    if observed[0] < hard[0] or observed[1] > hard[1]:
        raise RuntimeError(
            f"{name} range {observed} is outside the registered interval {hard}"
        )


def polynomial_contract(
    function_name: str,
    coefficients: np.ndarray,
    interval: tuple[float, float],
) -> dict[str, object]:
    degree = len(coefficients) - 1
    return {
        "function": function_name,
        "degree": degree,
        "interval": {"minimum": interval[0], "maximum": interval[1]},
        "coefficient_count": len(coefficients),
        "coefficient_sha256": coefficient_sha256(coefficients),
        "coefficient_encoding": "raw OpenFHE-order IEEE-754 binary64 little-endian",
        "c0_convention": "raw c0 is doubled; EvalChebyshevSeries adds c0/2",
        "openfhe_ps": ps_contract(degree, interval),
    }


def exact_gelu(value: float) -> float:
    return 0.5 * value * (1.0 + math.erf(value / math.sqrt(2.0)))


def constrain_chebyshev_zero_at_origin(
    coefficients: np.ndarray,
    interval: tuple[float, float],
) -> tuple[np.ndarray, dict[str, float | str]]:
    if interval[0] > 0.0 or interval[1] < 0.0:
        raise ValueError("zero-at-origin constraint requires an interval containing zero")
    raw = np.asarray(coefficients, dtype=np.float64)
    raw_at_origin = float(
        evaluate_openfhe_chebyshev(
            np.array([0.0], dtype=np.float64),
            raw,
            *interval,
        )[0]
    )
    constrained = raw.copy()
    constrained[0] -= 2.0 * raw_at_origin
    constrained_at_origin = float(
        evaluate_openfhe_chebyshev(
            np.array([0.0], dtype=np.float64),
            constrained,
            *interval,
        )[0]
    )
    if abs(constrained_at_origin) > 1e-12:
        raise RuntimeError(
            "zero-at-origin Chebyshev correction exceeds the 1e-12 oracle gate"
        )
    return constrained, {
        "kind": "zero_at_origin_raw_c0_correction",
        "formula": "c0_zero=c0_raw-2*P_raw(0)",
        "raw_value_at_origin": raw_at_origin,
        "constrained_value_at_origin": constrained_at_origin,
        "raw_coefficient_sha256": coefficient_sha256(raw),
        "raw_c0": float(raw[0]),
        "constrained_c0": float(constrained[0]),
    }


def calibrate_gelu() -> dict[str, object]:
    interval = (-80.0, 128.0)
    raw_coefficients = openfhe_chebyshev_coefficients(
        exact_gelu,
        *interval,
        319,
    )
    coefficients, origin_constraint = constrain_chebyshev_zero_at_origin(
        raw_coefficients,
        interval,
    )
    per_layer = []
    observed_minimum = math.inf
    observed_maximum = -math.inf
    for layer in range(LAYER_COUNT):
        inputs = load_trace(
            layer,
            "Intermediate/allresults/intermediate_output_after_linear.csv",
        )
        expected = load_trace(
            layer,
            "Intermediate/allresults/real_intermediate_output.csv",
        )
        observed_minimum = min(observed_minimum, float(np.min(inputs)))
        observed_maximum = max(observed_maximum, float(np.max(inputs)))
        actual = evaluate_openfhe_chebyshev(
            inputs,
            coefficients,
            *interval,
        )
        per_layer.append(quality(actual, expected))
    trace_validation = summarize_layer_metrics(per_layer)
    require_quality_gate("GELU", trace_validation)
    require_range(
        "GELU input",
        (observed_minimum, observed_maximum),
        interval,
    )
    grid = np.linspace(interval[0], interval[1], 200_001)
    exact_grid = np.array([exact_gelu(float(value)) for value in grid])
    approximate_grid = evaluate_openfhe_chebyshev(
        grid,
        coefficients,
        *interval,
    )
    polynomial = polynomial_contract(
        "exact_erf_gelu_zero_at_origin",
        coefficients,
        interval,
    )
    polynomial["constraint"] = origin_constraint
    raw_approximate_grid = evaluate_openfhe_chebyshev(
        grid,
        raw_coefficients,
        *interval,
    )
    return {
        "status": "plaintext_trace_gate_passed",
        "definition": (
            "zero-preserving Chebyshev oracle calibrated against "
            "0.5*x*(1+erf(x/sqrt(2)))"
        ),
        "polynomial": polynomial,
        "input_guard": {
            "hard_interval": list(interval),
            "observed_trace_range": [observed_minimum, observed_maximum],
            "action_on_violation": "stop; do not clip; recalibrate",
        },
        "uniform_grid_max_absolute_error": float(
            np.max(np.abs(approximate_grid - exact_grid))
        ),
        "unconstrained_uniform_grid_max_absolute_error": float(
            np.max(np.abs(raw_approximate_grid - exact_grid))
        ),
        "bootstrap_placement": {
            "placement": (
                "before GELU polynomial when the "
                "pre-activation has fewer than 12 usable levels"
            ),
            "preactivation_to_ffn_output_required_levels": 12,
            "raw_ffn_input_to_output_required_levels": 13,
            "levels_required_after_bootstrap": 12,
            "gelu_polynomial_levels": 10,
            "final_mask_ct_pt_level": 1,
            "ffn_input_ct_pt_level": 1,
            "ffn_final_ct_pt_level": 1,
        },
        "trace_validation": trace_validation,
    }


def stable_logsumexp(values: np.ndarray) -> np.ndarray:
    maximum = np.max(values, axis=-1, keepdims=True)
    return maximum + np.log(np.sum(np.exp(values - maximum), axis=-1, keepdims=True))


def calibrate_softmax() -> dict[str, object]:
    exp_interval = (-16.0, 5.0)
    reciprocal_interval = (0.01, 80.0)
    rejected_reciprocal_interval = (0.9, 1.1)
    exp_coefficients = openfhe_chebyshev_coefficients(
        math.exp,
        *exp_interval,
        27,
    )
    reciprocal_coefficients = openfhe_chebyshev_coefficients(
        lambda value: 1.0 / value,
        *reciprocal_interval,
        383,
    )
    rejected_reciprocal_coefficients = openfhe_chebyshev_coefficients(
        lambda value: 1.0 / value,
        *rejected_reciprocal_interval,
        5,
    )

    shifts = np.zeros((LAYER_COUNT, ATTENTION_HEADS), dtype=np.float64)
    shifted_minimum = math.inf
    shifted_maximum = -math.inf
    denominator_minimum = math.inf
    denominator_maximum = -math.inf
    row_sum_minimum = math.inf
    row_sum_maximum = -math.inf
    per_layer = []
    attention_per_layer = []

    rejected_shift_minimum = math.inf
    rejected_shift_maximum = -math.inf
    rejected_denominator_minimum = math.inf
    rejected_denominator_maximum = -math.inf
    rejected_per_layer = []
    rejected_attention_per_layer = []

    for layer in range(LAYER_COUNT):
        qkt = load_trace(
            layer,
            "Attention/BertSelfAttention/allresults/QKT.csv",
        ).reshape(QUERY_ROWS, ATTENTION_HEADS, QUERY_ROWS)
        expected = load_trace(
            layer,
            "Attention/BertSelfAttention/allresults/aftsoftmax.csv",
        ).reshape(QUERY_ROWS, ATTENTION_HEADS, QUERY_ROWS)

        query_head_logsumexp = stable_logsumexp(qkt).squeeze(-1)
        minimum_per_head_shift = np.min(query_head_logsumexp, axis=0)
        maximum_per_head_shift = np.max(query_head_logsumexp, axis=0)
        per_head_shift = 0.5 * (
            minimum_per_head_shift + maximum_per_head_shift
        )
        shifts[layer, :] = per_head_shift
        shifted = qkt - per_head_shift[None, :, None]
        numerators = evaluate_openfhe_chebyshev(
            shifted,
            exp_coefficients,
            *exp_interval,
        )
        denominators = np.sum(numerators, axis=-1)
        inverse_denominators = evaluate_openfhe_chebyshev(
            denominators,
            reciprocal_coefficients,
            *reciprocal_interval,
        )
        actual = numerators * inverse_denominators[:, :, None]
        per_layer.append(quality(actual, expected))

        values = load_trace(
            layer,
            "Attention/BertSelfAttention/allresults/V.csv",
        ).reshape(QUERY_ROWS, ATTENTION_HEADS, HEAD_SIZE)
        actual_attention = np.einsum("rhk,khd->rhd", actual, values)
        expected_attention = load_trace(
            layer,
            "Attention/BertSelfAttention/allresults/real_attention.csv",
        ).reshape(QUERY_ROWS, ATTENTION_HEADS, HEAD_SIZE)
        attention_per_layer.append(quality(actual_attention, expected_attention))

        shifted_minimum = min(shifted_minimum, float(np.min(shifted)))
        shifted_maximum = max(shifted_maximum, float(np.max(shifted)))
        denominator_minimum = min(
            denominator_minimum,
            float(np.min(denominators)),
        )
        denominator_maximum = max(
            denominator_maximum,
            float(np.max(denominators)),
        )
        row_sums = np.sum(actual, axis=-1)
        row_sum_minimum = min(row_sum_minimum, float(np.min(row_sums)))
        row_sum_maximum = max(row_sum_maximum, float(np.max(row_sums)))

        rejected_shifted = qkt - query_head_logsumexp[:, :, None]
        rejected_numerators = evaluate_openfhe_chebyshev(
            rejected_shifted,
            exp_coefficients,
            *exp_interval,
        )
        rejected_denominators = np.sum(rejected_numerators, axis=-1)
        rejected_inverse = evaluate_openfhe_chebyshev(
            rejected_denominators,
            rejected_reciprocal_coefficients,
            *rejected_reciprocal_interval,
        )
        rejected_actual = rejected_numerators * rejected_inverse[:, :, None]
        rejected_per_layer.append(quality(rejected_actual, expected))
        rejected_attention = np.einsum(
            "rhk,khd->rhd",
            rejected_actual,
            values,
        )
        rejected_attention_per_layer.append(
            quality(rejected_attention, expected_attention)
        )
        rejected_shift_minimum = min(
            rejected_shift_minimum,
            float(np.min(rejected_shifted)),
        )
        rejected_shift_maximum = max(
            rejected_shift_maximum,
            float(np.max(rejected_shifted)),
        )
        rejected_denominator_minimum = min(
            rejected_denominator_minimum,
            float(np.min(rejected_denominators)),
        )
        rejected_denominator_maximum = max(
            rejected_denominator_maximum,
            float(np.max(rejected_denominators)),
        )

    trace_validation = summarize_layer_metrics(per_layer)
    attention_validation = summarize_layer_metrics(attention_per_layer)
    rejected_trace_validation = summarize_layer_metrics(rejected_per_layer)
    rejected_attention_validation = summarize_layer_metrics(
        rejected_attention_per_layer
    )
    require_quality_gate("Softmax", trace_validation)
    require_quality_gate("Softmax times V", attention_validation)
    require_range(
        "Softmax shifted logits",
        (shifted_minimum, shifted_maximum),
        exp_interval,
    )
    require_range(
        "Softmax denominator",
        (denominator_minimum, denominator_maximum),
        reciprocal_interval,
    )
    denominator_noise_ratio = (
        denominator_minimum / TWO_ITERATION_BOOTSTRAP_MAX_ABS
    )
    if denominator_noise_ratio < SOFTMAX_DENOMINATOR_NOISE_RATIO_MIN:
        raise RuntimeError(
            "Softmax denominator noise-margin gate failed: "
            f"ratio={denominator_noise_ratio} < "
            f"{SOFTMAX_DENOMINATOR_NOISE_RATIO_MIN}"
        )
    denominator_after_negative_noise = (
        denominator_minimum - TWO_ITERATION_BOOTSTRAP_MAX_ABS
    )
    if denominator_after_negative_noise < reciprocal_interval[0]:
        raise RuntimeError(
            "Softmax denominator may leave the registered reciprocal interval "
            "under the two-iteration bootstrap error budget"
        )

    reciprocal_grid = np.linspace(
        reciprocal_interval[0],
        reciprocal_interval[1],
        200_001,
    )
    reciprocal_grid_actual = evaluate_openfhe_chebyshev(
        reciprocal_grid,
        reciprocal_coefficients,
        *reciprocal_interval,
    )
    reciprocal_uniform_relative = float(
        np.max(np.abs(reciprocal_grid * reciprocal_grid_actual - 1.0))
    )
    return {
        "status": "plaintext_trace_gate_passed",
        "scope": (
            "fixed five-token trace replay only; each offline-calibrated public "
            "shift balances the bundled query-row logsumexp extrema in log space, "
            "is shared by every query row in one layer/head, and is not "
            "generalizable beyond the bundled trace contract"
        ),
        "shift_contract": {
            "kind": "fixed_layer_head_scalar_offline_calibrated",
            "calibration_rule": "query_row_logsumexp_extrema_midpoint",
            "axis_order": ["layer", "head"],
            "shape": list(shifts.shape),
            "shared_across_all_query_rows": True,
            "packed_query_rows": PACKED_QUERY_ROWS,
            "values": shifts.tolist(),
            "values_sha256": tensor_sha256(shifts),
            "offline_derivation": (
                "0.5*(min_query_row(log(sum_j(exp(QKT))))+"
                "max_query_row(log(sum_j(exp(QKT)))))"
            ),
            "derivation_scope": "bundled five-token plaintext calibration trace",
            "visibility": "public pre-registered workload metadata",
            "runtime_activation_dependency": "none",
            "forbidden": (
                "runtime derivation from plaintext or ciphertext activations, "
                "client-private inputs, or query-row-specific shift vectors"
            ),
            "packing_application": (
                "broadcast one public scalar for the selected layer/head to every "
                "query-row slot, all 128 QK diagonal ciphertexts, and all batch lanes"
            ),
        },
        "padding_mask_contract": {
            "order": "evaluate exp, then multiply by the public active-slot mask",
            "inactive_numerator": 0.0,
            "inactive_denominator_before_reciprocal": 1.0,
            "quality_scope": "active trace rows only",
            "metadata_visibility": "public fixture/model metadata",
        },
        "exponential": polynomial_contract(
            "exp",
            exp_coefficients,
            exp_interval,
        ),
        "reciprocal": polynomial_contract(
            "1/x",
            reciprocal_coefficients,
            reciprocal_interval,
        ),
        "reciprocal_uniform_grid_max_relative_error": (
            reciprocal_uniform_relative
        ),
        "range_guards": {
            "shifted_logits": {
                "hard_interval": list(exp_interval),
                "observed_trace_range": [shifted_minimum, shifted_maximum],
            },
            "active_denominator": {
                "hard_interval": list(reciprocal_interval),
                "observed_trace_range": [
                    denominator_minimum,
                    denominator_maximum,
                ],
            },
            "action_on_violation": "stop; do not clip or add epsilon",
        },
        "epsilon": {
            "value": 0.0,
            "contract": "no denominator epsilon; inactive denominators are set to 1",
        },
        "bootstrap_placement": {
            "placement": "after masked exp sum and before reciprocal",
            "count_per_head": 1,
            "count_per_layer": ATTENTION_HEADS,
            "pre_bootstrap_required_levels": 7,
            "post_bootstrap_required_levels": 12,
            "reciprocal_levels": 10,
            "normalization_ct_ct_level": 1,
            "final_mask_ct_pt_level": 1,
            "reference_scope": (
                "the M3 two-iteration precision-10 acceptance budget is used for "
                "a conservative denominator noise-margin gate; this plaintext "
                "contract is not a full-slot ciphertext guarantee"
            ),
            "registered_denominator_margin_to_interval_edge": min(
                denominator_minimum - reciprocal_interval[0],
                reciprocal_interval[1] - denominator_maximum,
            ),
            "denominator_noise_margin_gate": {
                "two_iteration_bootstrap_max_absolute_error": (
                    TWO_ITERATION_BOOTSTRAP_MAX_ABS
                ),
                "minimum_denominator_to_noise_ratio": denominator_noise_ratio,
                "required_minimum_ratio": (
                    SOFTMAX_DENOMINATOR_NOISE_RATIO_MIN
                ),
                "minimum_denominator_after_negative_noise": (
                    denominator_after_negative_noise
                ),
                "reciprocal_interval_minimum": reciprocal_interval[0],
                "passed": True,
            },
        },
        "observed_output_row_sum_range": [
            row_sum_minimum,
            row_sum_maximum,
        ],
        "trace_validation": trace_validation,
        "downstream_attention_times_v_validation": attention_validation,
        "rejected_row_specific_shift_candidate": {
            "status": "rejected_trust_boundary",
            "shift_kind": "activation_dependent_layer_head_query_row_logsumexp",
            "candidate_derivation": (
                "log(sum_j(exp(QKT[layer,query_row,head,j])))"
            ),
            "runtime_plaintext_activation_required": True,
            "shifted_logit_observed_range": [
                rejected_shift_minimum,
                rejected_shift_maximum,
            ],
            "denominator_observed_range": [
                rejected_denominator_minimum,
                rejected_denominator_maximum,
            ],
            "reciprocal": polynomial_contract(
                "1/x",
                rejected_reciprocal_coefficients,
                rejected_reciprocal_interval,
            ),
            "plaintext_trace_validation": rejected_trace_validation,
            "downstream_attention_times_v_validation": (
                rejected_attention_validation
            ),
            "rejection_reason": (
                "the shift varies with each query row and requires runtime "
                "activation-derived logsumexp values; numerical quality cannot "
                "override the server-only trust boundary"
            ),
            "allowed_use": "offline diagnostic comparison only",
        },
    }


def layernorm_paths(site: str) -> tuple[str, str, str, str]:
    if site == "ln1":
        stem = "Attention/SelfOutput"
        prefix = "self_output"
    elif site == "ln2":
        stem = "Output"
        prefix = "final_output"
    else:
        raise ValueError(f"unknown LayerNorm site: {site}")
    return (
        f"{stem}/allresults/{prefix}_residual_connection_before_layernorm.csv",
        f"{stem}/allresults/real_{prefix}.csv",
        f"{stem}/parms/{prefix}_LayerNorm_weight.csv",
        f"{stem}/parms/{prefix}_LayerNorm_bias.csv",
    )


def calibrate_layernorm() -> dict[str, object]:
    interval = (0.5, 1536.0)
    coefficients = openfhe_chebyshev_coefficients(
        lambda value: 1.0 / math.sqrt(value),
        *interval,
        159,
    )
    site_scales = {"ln1": 64.0, "ln2": 1.0}
    site_results: dict[str, object] = {}
    combined_per_layer = []
    for site, variance_scale in site_scales.items():
        input_path, expected_path, gamma_path, beta_path = layernorm_paths(site)
        raw_minimum = math.inf
        raw_maximum = -math.inf
        normalized_minimum = math.inf
        normalized_maximum = -math.inf
        per_layer = []
        exact_per_layer = []
        for layer in range(LAYER_COUNT):
            inputs = load_trace(layer, input_path)
            expected = load_trace(layer, expected_path)
            gamma = load_trace(layer, gamma_path).reshape(-1)
            beta = load_trace(layer, beta_path).reshape(-1)
            if inputs.shape[1] != HIDDEN_SIZE:
                raise RuntimeError(f"{site} hidden dimension is not 768")
            centered = inputs - np.mean(inputs, axis=1, keepdims=True)
            raw_variance = np.mean(centered * centered, axis=1, keepdims=True)
            variance_with_epsilon = raw_variance + 1e-12
            normalized_variance = variance_scale * variance_with_epsilon
            approximate_inverse_std = math.sqrt(variance_scale) * (
                evaluate_openfhe_chebyshev(
                    normalized_variance,
                    coefficients,
                    *interval,
                )
            )
            actual = centered * approximate_inverse_std * gamma + beta
            exact = centered / np.sqrt(variance_with_epsilon) * gamma + beta
            per_layer.append(quality(actual, expected))
            exact_per_layer.append(quality(exact, expected))
            raw_minimum = min(raw_minimum, float(np.min(raw_variance)))
            raw_maximum = max(raw_maximum, float(np.max(raw_variance)))
            normalized_minimum = min(
                normalized_minimum,
                float(np.min(normalized_variance)),
            )
            normalized_maximum = max(
                normalized_maximum,
                float(np.max(normalized_variance)),
            )

        summary = summarize_layer_metrics(per_layer)
        exact_summary = summarize_layer_metrics(exact_per_layer)
        require_quality_gate(f"LayerNorm {site}", summary)
        require_range(
            f"LayerNorm {site} normalized variance",
            (normalized_minimum, normalized_maximum),
            interval,
        )
        combined_per_layer.extend(per_layer)
        site_results[site] = {
            "variance_scale_D_s": variance_scale,
            "inverse_output_compensation_sqrt_D_s": math.sqrt(variance_scale),
            "raw_population_variance_observed_range": [
                raw_minimum,
                raw_maximum,
            ],
            "normalized_variance_observed_range": [
                normalized_minimum,
                normalized_maximum,
            ],
            "trace_validation": summary,
            "exact_formula_trace_parity": exact_summary,
        }

    combined_summary = summarize_layer_metrics(combined_per_layer)
    require_quality_gate("LayerNorm combined", combined_summary)
    return {
        "status": "plaintext_trace_gate_passed",
        "definition": (
            "mean over 768 hidden values; population variance; "
            "epsilon added before public D_s normalization"
        ),
        "epsilon": {
            "value": 1e-12,
            "placement": "population_variance + epsilon before multiplication by D_s",
            "trace_parity": "BERT LayerNorm epsilon; exact formula matches CSV traces",
        },
        "normalization_formula": {
            "variance": "v=mean_i((x_i-mean(x))^2)+1e-12",
            "polynomial_input": "u=D_s*v",
            "inverse_standard_deviation": "sqrt(D_s)*ChebInvSqrt(u)",
            "output": "gamma_i*(x_i-mean(x))*inverse_standard_deviation+beta_i",
            "implementation_note": "sqrt(D_s) may be fused into public gamma",
        },
        "inverse_sqrt": polynomial_contract(
            "1/sqrt(x)",
            coefficients,
            interval,
        ),
        "normalized_variance_guard": {
            "hard_interval": list(interval),
            "action_on_violation": "stop; do not clip; recalibrate D_s or interval",
        },
        "sites": site_results,
        "bootstrap_placement": {
            "placement": "after normalized variance u is formed, before inverse sqrt",
            "pre_bootstrap_required_levels": 4,
            "post_bootstrap_required_levels": 11,
            "input_mask_ct_pt_level": 1,
            "mean_ct_pt_level": 1,
            "centered_square_ct_ct_level": 1,
            "variance_scale_ct_pt_level": 1,
            "inverse_sqrt_levels": 9,
            "final_centered_times_inverse_level": 1,
            "final_gamma_ct_pt_level": 1,
        },
        "combined_trace_validation": combined_summary,
    }


def build_contract() -> dict[str, object]:
    return {
        "schema_version": 1,
        "profile_id": "paper_compat",
        "security_claim": "none",
        "scope": {
            "backend": "OpenFHE CKKS CPU",
            "workload": "server-only 12-layer fixed BERT-base encoder trace replay",
            "status": "plaintext calibration only; ciphertext validation required",
            "non_generalizable": (
                "Softmax uses only frozen layer/head scalar shifts and LayerNorm "
                "uses frozen D_s values calibrated for the bundled traces; no "
                "runtime activation-derived shift metadata is permitted"
            ),
        },
        "chebyshev_convention": {
            "producer": "OpenFHE EvalChebyshevCoefficients-compatible DCT-II",
            "nodes": (
                "cos(pi/(degree+1)*(i+0.5))*(b-a)/2+(b+a)/2"
            ),
            "raw_c0": "doubled; OpenFHE evaluator consumes c0/2",
            "coefficient_hash_encoding": (
                "raw coefficient order as IEEE-754 binary64 little-endian bytes"
            ),
            "openfhe_source_reference": (
                "src/core/lib/math/chebyshev.cpp:"
                "EvalChebyshevCoefficients/EvalChebyshevFunctionPtxt"
            ),
            "ps_source_reference": (
                "src/pke/lib/scheme/ckksrns/ckksrns-utils.cpp:ComputeDegreesPS"
            ),
        },
        "global_trace_gate": {
            "relative_l2_max": RELATIVE_L2_LIMIT,
            "cosine_min": COSINE_LIMIT,
            "range_policy": "all observed inputs must remain inside hard intervals",
            "threshold_policy": "thresholds may be tightened, never silently relaxed",
        },
        "ordinary_ckks_bootstrap": {
            "implementation": "OpenFHE v1.5.1 native EvalBootstrap",
            "iterations": 2,
            "precision_bits": 10,
            "level_budget": [4, 4],
            "bsgs_dimension": [0, 0],
            "correction_factor": 0,
            "slots_to_coefficients_first": False,
            "levels_available_after_bootstrap": 12,
            "depth_formula": (
                "12 + GetBootstrapDepth([4,4], SPARSE_TERNARY) + "
                "(iterations-1)"
            ),
            "precision_derivation": (
                "floor(-log2(5.6e-4))=10 from the conservative observed "
                "single-bootstrap max-error scale"
            ),
        },
        "operators": {
            "gelu": calibrate_gelu(),
            "softmax": calibrate_softmax(),
            "layernorm": calibrate_layernorm(),
        },
    }


def canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def validation_summary(contract: dict[str, object]) -> dict[str, object]:
    operators = contract["operators"]
    return {
        "test": "calibrate_openfhe_nonlinear",
        "profile": contract["profile_id"],
        "security_claim": contract["security_claim"],
        "contract_sha256": hashlib.sha256(
            canonical_json(contract).encode("utf-8")
        ).hexdigest(),
        "gelu": {
            "worst_relative_l2": operators["gelu"]["trace_validation"][
                "worst_relative_l2"
            ],
            "minimum_cosine": operators["gelu"]["trace_validation"][
                "minimum_cosine"
            ],
            "observed_input_range": operators["gelu"]["input_guard"][
                "observed_trace_range"
            ],
        },
        "softmax": {
            "shift_kind": operators["softmax"]["shift_contract"]["kind"],
            "shift_calibration_rule": operators["softmax"]["shift_contract"][
                "calibration_rule"
            ],
            "worst_relative_l2": operators["softmax"]["trace_validation"][
                "worst_relative_l2"
            ],
            "minimum_cosine": operators["softmax"]["trace_validation"][
                "minimum_cosine"
            ],
            "shifted_logit_range": operators["softmax"]["range_guards"][
                "shifted_logits"
            ]["observed_trace_range"],
            "denominator_range": operators["softmax"]["range_guards"][
                "active_denominator"
            ]["observed_trace_range"],
            "downstream_attention_worst_relative_l2": operators["softmax"][
                "downstream_attention_times_v_validation"
            ]["worst_relative_l2"],
            "downstream_attention_minimum_cosine": operators["softmax"][
                "downstream_attention_times_v_validation"
            ]["minimum_cosine"],
            "minimum_denominator_to_noise_ratio": operators["softmax"][
                "bootstrap_placement"
            ]["denominator_noise_margin_gate"][
                "minimum_denominator_to_noise_ratio"
            ],
            "rejected_row_specific_status": operators["softmax"][
                "rejected_row_specific_shift_candidate"
            ]["status"],
        },
        "layernorm": {
            "worst_relative_l2": operators["layernorm"][
                "combined_trace_validation"
            ]["worst_relative_l2"],
            "minimum_cosine": operators["layernorm"][
                "combined_trace_validation"
            ]["minimum_cosine"],
            "ln1_D_s": operators["layernorm"]["sites"]["ln1"][
                "variance_scale_D_s"
            ],
            "ln2_D_s": operators["layernorm"]["sites"]["ln2"][
                "variance_scale_D_s"
            ],
        },
        "passed": True,
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="frozen approximation contract to validate",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="MOAI trace data root read in place",
    )
    parser.add_argument(
        "--emit-config",
        type=Path,
        nargs="?",
        const=Path("-"),
        help=(
            "emit a freshly calibrated JSON contract to stdout, or write PATH"
        ),
    )
    return parser.parse_args()


def main() -> int:
    global DATA_ROOT
    arguments = parse_arguments()
    DATA_ROOT = arguments.data_root.resolve()
    generated = build_contract()
    if arguments.emit_config is not None:
        serialized = json.dumps(
            generated,
            separators=(",", ":"),
            allow_nan=False,
        )
        if arguments.emit_config == Path("-"):
            print(serialized)
        else:
            arguments.emit_config.write_text(serialized + "\n", encoding="utf-8")
            print(json.dumps(validation_summary(generated), allow_nan=False))
        return 0

    if not arguments.config.is_file():
        raise FileNotFoundError(
            f"frozen calibration config does not exist: {arguments.config}"
        )
    with arguments.config.open("r", encoding="utf-8") as handle:
        frozen = json.load(handle)
    if canonical_json(frozen) != canonical_json(generated):
        frozen_hash = hashlib.sha256(
            canonical_json(frozen).encode("utf-8")
        ).hexdigest()
        generated_hash = hashlib.sha256(
            canonical_json(generated).encode("utf-8")
        ).hexdigest()
        raise RuntimeError(
            "frozen nonlinear contract drifted: "
            f"frozen={frozen_hash} generated={generated_hash}"
        )
    print(json.dumps(validation_summary(generated), allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"calibrate_openfhe_nonlinear failed: {error}", file=sys.stderr)
        sys.exit(1)
