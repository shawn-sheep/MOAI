#!/usr/bin/env python3
"""Validate the checked-in five-token MOAI FFN trace contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - environment failure
    raise SystemExit(
        "NumPy is required. Activate the MOAI validation environment before running."
    ) from exc


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "config" / "moai_trace_channel_scales.json"
PASS_DECISION = "PASS_TRACE_PARITY_FIXTURE"
FAIL_DECISION = "FAIL_TRACE_PARITY_FIXTURE"

EXPECTED_TOP_LEVEL_KEYS = {
    "schema_version",
    "contract_id",
    "profile_id",
    "security_claim",
    "hash_algorithm",
    "scope",
    "dimensions",
    "tolerances",
    "required_files",
    "layers",
}
EXPECTED_SCOPE = {
    "fixture_only": True,
    "source_trace_token_count": 5,
    "scale_provenance": "reverse_engineered_from_existing_five_token_trace",
    "runtime_scale_inference_allowed": False,
    "production_generalization": "forbidden",
    "source_statement": (
        "Channel scales were inferred solely from the checked-in five-token trace."
    ),
}
EXPECTED_DIMENSIONS = {
    "encoder_layers": 12,
    "trace_token_rows": 5,
    "hidden_size": 768,
    "intermediate_size": 3072,
}
EXPECTED_FILES = {
    "embedded_inputs": (
        "Attention/BertSelfAttention/allresults/embedded_inputs.csv",
        (5, 768),
    ),
    "intermediate_inputs": (
        "Intermediate/allresults/intermediate_inputs.csv",
        (5, 768),
    ),
    "intermediate_weight": (
        "Intermediate/parms/intermediate_dense_weight.csv",
        (3072, 768),
    ),
    "intermediate_bias": (
        "Intermediate/parms/intermediate_dense_bias.csv",
        (3072,),
    ),
    "intermediate_after_linear": (
        "Intermediate/allresults/intermediate_output_after_linear.csv",
        (5, 3072),
    ),
    "real_intermediate_output": (
        "Intermediate/allresults/real_intermediate_output.csv",
        (5, 3072),
    ),
    "final_output_inputs": (
        "Output/allresults/final_output_inputs.csv",
        (5, 3072),
    ),
    "final_weight": (
        "Output/parms/final_output_dense_weight.csv",
        (768, 3072),
    ),
    "final_bias": (
        "Output/parms/final_output_dense_bias.csv",
        (768,),
    ),
    "final_after_linear": (
        "Output/allresults/final_output_after_linear.csv",
        (5, 768),
    ),
    "final_residual": (
        "Output/allresults/final_output_usedby_residual.csv",
        (5, 768),
    ),
    "final_before_layernorm": (
        "Output/allresults/final_output_residual_connection_before_layernorm.csv",
        (5, 768),
    ),
    "real_final_output": (
        "Output/allresults/real_final_output.csv",
        (5, 768),
    ),
}
LOCKED_TOLERANCES = {
    "ffn_input_snapshot": {"max_abs": 1e-5, "rel_l2": 2e-7},
    "intermediate_linear": {"max_abs": 3e-5, "rel_l2": 4e-7},
    "exact_gelu": {"max_abs": 1e-5, "rel_l2": 8e-7},
    "final_linear": {"max_abs": 1.5e-4, "rel_l2": 5e-7},
    "residual_add": {"max_abs": 3e-5, "rel_l2": 5e-8},
    "cross_layer": {
        "max_abs": 0.0,
        "rel_l2": 0.0,
        "require_byte_equal": True,
    },
}
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class ContractError(RuntimeError):
    """Raised when the manifest or its fixture violates the locked contract."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the MOAI five-token FFN trace, hashes, channel scales, "
            "and layer chaining."
        )
    )
    parser.add_argument(
        "--data-root",
        required=True,
        type=Path,
        help="Path to the existing MOAI data directory; files are read in place.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=f"Trace manifest (default: {DEFAULT_MANIFEST}).",
    )
    return parser.parse_args()


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ContractError(f"manifest is not a file: {path}")
    try:
        with path.open("r", encoding="utf-8") as stream:
            manifest = json.load(stream, object_pairs_hook=_no_duplicate_keys)
    except json.JSONDecodeError as exc:
        raise ContractError(f"invalid manifest JSON: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ContractError("manifest root must be a JSON object")
    return manifest


def require_exact_keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractError(f"{label} must be an object")
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ContractError(f"{label} keys differ: missing={missing}, extra={extra}")
    return value


def validate_manifest(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    require_exact_keys(manifest, EXPECTED_TOP_LEVEL_KEYS, "manifest")
    if manifest["schema_version"] != 1:
        raise ContractError("schema_version must be 1")
    if manifest["contract_id"] != "moai-bert-base-five-token-ffn-trace-v1":
        raise ContractError("unexpected contract_id")
    if manifest["profile_id"] != "paper_compat":
        raise ContractError("profile_id must be paper_compat")
    if manifest["security_claim"] != "none":
        raise ContractError("security_claim must be none")
    if manifest["hash_algorithm"] != "sha256":
        raise ContractError("hash_algorithm must be sha256")
    if manifest["scope"] != EXPECTED_SCOPE:
        raise ContractError(
            "scope must lock fixture-only provenance and forbid production generalization"
        )
    if manifest["dimensions"] != EXPECTED_DIMENSIONS:
        raise ContractError(f"dimensions must equal {EXPECTED_DIMENSIONS}")
    if manifest["tolerances"] != LOCKED_TOLERANCES:
        raise ContractError("tolerances differ from the validator's locked thresholds")

    file_specs = require_exact_keys(
        manifest["required_files"], set(EXPECTED_FILES), "required_files"
    )
    for file_id, (expected_path, expected_shape) in EXPECTED_FILES.items():
        spec = require_exact_keys(
            file_specs[file_id], {"path", "shape"}, f"required_files.{file_id}"
        )
        if spec["path"] != expected_path or spec["shape"] != list(expected_shape):
            raise ContractError(f"required_files.{file_id} path or shape changed")

    layers = manifest["layers"]
    if not isinstance(layers, list) or len(layers) != 12:
        raise ContractError("layers must contain exactly 12 entries")
    layer_ids = [layer.get("layer_id") if isinstance(layer, dict) else None for layer in layers]
    if layer_ids != list(range(12)):
        raise ContractError(f"layer ids must be ordered 0..11, got {layer_ids}")

    for layer_id, layer in enumerate(layers):
        layer = require_exact_keys(
            layer,
            {
                "layer_id",
                "channel_scale_default",
                "channel_scale_overrides",
                "sha256",
            },
            f"layers[{layer_id}]",
        )
        if type(layer["layer_id"]) is not int or layer["layer_id"] != layer_id:
            raise ContractError(f"layers[{layer_id}].layer_id is invalid")
        if type(layer["channel_scale_default"]) is not int:
            raise ContractError(f"layer {layer_id}: channel_scale_default must be integer")
        if layer["channel_scale_default"] != 1:
            raise ContractError(f"layer {layer_id}: channel_scale_default must be 1")

        overrides = layer["channel_scale_overrides"]
        if not isinstance(overrides, dict):
            raise ContractError(f"layer {layer_id}: channel_scale_overrides must be object")
        for raw_index, scale in overrides.items():
            try:
                index = int(raw_index)
            except ValueError as exc:
                raise ContractError(
                    f"layer {layer_id}: invalid channel index {raw_index!r}"
                ) from exc
            if str(index) != raw_index or not 0 <= index < 3072:
                raise ContractError(f"layer {layer_id}: invalid channel index {raw_index!r}")
            if type(scale) is not int or scale <= 1:
                raise ContractError(
                    f"layer {layer_id}: scale for channel {index} must be integer > 1"
                )

        hashes = require_exact_keys(
            layer["sha256"], set(EXPECTED_FILES), f"layers[{layer_id}].sha256"
        )
        for file_id, digest in hashes.items():
            if not isinstance(digest, str) or SHA256_PATTERN.fullmatch(digest) is None:
                raise ContractError(f"layer {layer_id}: invalid SHA-256 for {file_id}")
    return layers


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_layer(
    data_root: Path, layer: dict[str, Any]
) -> tuple[dict[str, np.ndarray], dict[str, Path]]:
    layer_id = layer["layer_id"]
    layer_root = data_root / f"layer_{layer_id}"
    if not layer_root.is_dir():
        raise ContractError(f"layer directory is missing: {layer_root}")

    arrays: dict[str, np.ndarray] = {}
    paths: dict[str, Path] = {}
    for file_id, (relative_path, expected_shape) in EXPECTED_FILES.items():
        path = layer_root / relative_path
        if not path.is_file():
            raise ContractError(f"layer {layer_id}: required file missing: {relative_path}")
        actual_digest = sha256_file(path)
        expected_digest = layer["sha256"][file_id]
        if actual_digest != expected_digest:
            raise ContractError(
                f"layer {layer_id}: SHA-256 mismatch for {relative_path}: "
                f"expected {expected_digest}, got {actual_digest}"
            )
        try:
            values = np.loadtxt(path, delimiter=",", dtype=np.float64)
        except (OSError, ValueError) as exc:
            raise ContractError(f"layer {layer_id}: cannot parse {relative_path}: {exc}") from exc
        if values.shape != expected_shape:
            raise ContractError(
                f"layer {layer_id}: {relative_path} shape {values.shape}, "
                f"expected {expected_shape}"
            )
        if not np.isfinite(values).all():
            raise ContractError(f"layer {layer_id}: {relative_path} contains NaN or Inf")
        arrays[file_id] = values
        paths[file_id] = path
    return arrays, paths


def exact_gelu(values: np.ndarray) -> np.ndarray:
    erf_values = np.fromiter(
        (math.erf(float(value) / math.sqrt(2.0)) for value in values.flat),
        dtype=np.float64,
        count=values.size,
    ).reshape(values.shape)
    return 0.5 * values * (1.0 + erf_values)


def error_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    delta = actual - predicted
    denominator = float(np.linalg.norm(actual.ravel()))
    numerator = float(np.linalg.norm(delta.ravel()))
    relative_l2 = numerator / denominator if denominator else (0.0 if numerator == 0 else math.inf)
    return {
        "max_abs": float(np.max(np.abs(delta))),
        "rel_l2": relative_l2,
    }


def metric_failure(
    layer_label: str,
    relation: str,
    metrics: dict[str, float],
    tolerance: dict[str, float],
) -> str | None:
    if (
        metrics["max_abs"] > tolerance["max_abs"]
        or metrics["rel_l2"] > tolerance["rel_l2"]
    ):
        return (
            f"{layer_label} {relation}: max_abs={metrics['max_abs']:.9e} "
            f"(limit {tolerance['max_abs']:.9e}), rel_l2={metrics['rel_l2']:.9e} "
            f"(limit {tolerance['rel_l2']:.9e})"
        )
    return None


def validate_trace(data_root: Path, layers: list[dict[str, Any]]) -> int:
    if not data_root.is_dir():
        raise ContractError(f"data root is not a directory: {data_root}")

    failures: list[str] = []
    reports: list[dict[str, dict[str, float]]] = []
    final_outputs: dict[int, np.ndarray] = {}
    embedded_inputs: dict[int, np.ndarray] = {}
    chain_paths: dict[int, dict[str, Path]] = {}

    relation_tolerances = {
        "ffn_input_snapshot": LOCKED_TOLERANCES["ffn_input_snapshot"],
        "intermediate_linear": LOCKED_TOLERANCES["intermediate_linear"],
        "exact_gelu": LOCKED_TOLERANCES["exact_gelu"],
        "final_linear": LOCKED_TOLERANCES["final_linear"],
        "residual_add": LOCKED_TOLERANCES["residual_add"],
    }

    for layer in layers:
        layer_id = layer["layer_id"]
        arrays, paths = load_layer(data_root, layer)
        scales = np.ones(3072, dtype=np.float64)
        for raw_index, scale in layer["channel_scale_overrides"].items():
            scales[int(raw_index)] = float(scale)

        predicted_intermediate = (
            arrays["intermediate_inputs"] @ arrays["intermediate_weight"].T
            + arrays["intermediate_bias"]
        ) * scales
        predicted_final = (
            arrays["real_intermediate_output"]
            @ (arrays["final_weight"] / scales).T
            + arrays["final_bias"]
        )
        metrics = {
            "ffn_input_snapshot": error_metrics(
                arrays["real_intermediate_output"], arrays["final_output_inputs"]
            ),
            "intermediate_linear": error_metrics(
                arrays["intermediate_after_linear"], predicted_intermediate
            ),
            "exact_gelu": error_metrics(
                arrays["real_intermediate_output"],
                exact_gelu(arrays["intermediate_after_linear"]),
            ),
            "final_linear": error_metrics(
                arrays["final_after_linear"], predicted_final
            ),
            "residual_add": error_metrics(
                arrays["final_before_layernorm"],
                arrays["final_after_linear"] + arrays["final_residual"],
            ),
        }
        reports.append(metrics)
        final_outputs[layer_id] = arrays["real_final_output"].copy()
        embedded_inputs[layer_id] = arrays["embedded_inputs"].copy()
        chain_paths[layer_id] = paths

        for relation, relation_metrics in metrics.items():
            failure = metric_failure(
                f"L{layer_id:02d}",
                relation,
                relation_metrics,
                relation_tolerances[relation],
            )
            if failure is not None:
                failures.append(failure)

        print(
            f"L{layer_id:02d} "
            f"snapshot={metrics['ffn_input_snapshot']['max_abs']:.9e} "
            f"pre_linear={metrics['intermediate_linear']['max_abs']:.9e} "
            f"gelu={metrics['exact_gelu']['max_abs']:.9e} "
            f"post_linear={metrics['final_linear']['max_abs']:.9e} "
            f"residual={metrics['residual_add']['max_abs']:.9e}"
        )

    chain_tolerance = LOCKED_TOLERANCES["cross_layer"]
    for layer_id in range(11):
        metrics = error_metrics(final_outputs[layer_id], embedded_inputs[layer_id + 1])
        left_path = chain_paths[layer_id]["real_final_output"]
        right_path = chain_paths[layer_id + 1]["embedded_inputs"]
        byte_equal = left_path.read_bytes() == right_path.read_bytes()
        failure = metric_failure(
            f"L{layer_id:02d}->L{layer_id + 1:02d}",
            "cross_layer",
            metrics,
            chain_tolerance,
        )
        if failure is not None:
            failures.append(failure)
        if chain_tolerance["require_byte_equal"] and not byte_equal:
            failures.append(
                f"L{layer_id:02d}->L{layer_id + 1:02d} cross_layer: bytes differ"
            )
        print(
            f"L{layer_id:02d}->L{layer_id + 1:02d} "
            f"cross_layer={metrics['max_abs']:.9e} "
            f"bytes={'match' if byte_equal else 'DIFFER'}"
        )

    relation_names = tuple(relation_tolerances)
    maxima = {
        relation: max(report[relation]["max_abs"] for report in reports)
        for relation in relation_names
    }
    print(
        "GLOBAL_MAX "
        + " ".join(f"{name}={value:.9e}" for name, value in maxima.items())
    )

    if failures:
        for failure in failures:
            print(f"ERROR {failure}", file=sys.stderr)
        print(f"DECISION={FAIL_DECISION}")
        return 1
    print(f"DECISION={PASS_DECISION}")
    return 0


def main() -> int:
    args = parse_args()
    try:
        manifest = load_manifest(args.manifest.resolve())
        layers = validate_manifest(manifest)
        print(f"CONTRACT={manifest['contract_id']}")
        print(f"MANIFEST={args.manifest.resolve()}")
        print(f"DATA_ROOT={args.data_root.resolve()}")
        return validate_trace(args.data_root.resolve(), layers)
    except (ContractError, OSError) as exc:
        print(f"ERROR {exc}", file=sys.stderr)
        print(f"DECISION={FAIL_DECISION}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
