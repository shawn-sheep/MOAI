#!/usr/bin/env python3
"""Validate the fail-closed schema-v4 M5 OpenFHE evidence bundle.

This validator is intentionally separate from the frozen schema-v2 M3/M4
validator.  It accepts exactly one server-only, twelve-layer ciphertext-chain
correctness run.  Timing fields are retained only as non-benchmark diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "results" / "openfhe"
DEFAULT_SCHEMA = REPO_ROOT / "docs" / "openfhe-m5-artifact-schema.json"
SCHEMA_URI = "https://json-schema.org/draft/2020-12/schema"
SCHEMA_ID = "https://local.moai/openfhe-m5-artifact-schema-v4.json"
SCHEMA_TITLE = "MOAI OpenFHE M5 evidence manifest v4"
M5_SCHEMA_VERSION = 4
CANONICALIZATION = "MOAI-json-sort-keys-compact-utf8-v1"
PROFILE_PATH = "config/paper_compat_feature_packed.json"
PROFILE_LOCATOR = "/effective_profile"
PROFILE_SHA256 = "94f30e628e21f02146ce7ed9820194eabba3820f6e1e17176a31f8c5acf8b0be"
WARNING = "Research reproduction parameters only. Do not claim 128-bit security."
BRANCH = "refactor/openfhe-cpu"
TRACKING_REF = "refs/remotes/origin/refactor/openfhe-cpu"
REMOTE_NAME = "origin"
REMOTE_REF = "refs/heads/refactor/openfhe-cpu"
EXECUTABLE_PATH = "build-openfhe/openfhe_encoder_12_layer_smoke"
VALIDATOR_PATH = REPO_ROOT / "scripts" / "validate_openfhe_m5_artifact.py"
CONFIG_INPUTS = (
    "config/moai_encoder_trace.json",
    "config/moai_trace_channel_scales.json",
    "config/openfhe_approximations.json",
    PROFILE_PATH,
)
LAYER_IDS = tuple(range(12))
TRACE_SHAPE = [5, 768]
TRACE_VALUE_COUNT = 5 * 768
FEATURE_BLOCK_SIZE = 1024
CSV_FILE_COUNT = 444
INPUT_FILE_COUNT = CSV_FILE_COUNT + len(CONFIG_INPUTS)
WEIGHT_FILES_PER_LAYER = 16
TRACE_FILES_PER_LAYER = 21

# These operation-count and metadata contracts are intentionally centralized.
# The metadata tuples were frozen from the exit-0 two-layer r6 calibration;
# formal evidence must still pass the exact seam and full 12-layer gates.
OPERATION_COUNT_KEYS = (
    "rotations",
    "ct_pt_multiplications",
    "ct_ct_multiplications",
    "explicit_rescale_requests",
    "chebyshev_evaluations",
    "estimated_polynomial_multiplications",
    "bootstraps",
    "bootstrap_iterations",
)
ZERO_COUNTS = dict.fromkeys(OPERATION_COUNT_KEYS, 0)
LAYER_COUNTS = {
    "rotations": 6300,
    "ct_pt_multiplications": 51865,
    "ct_ct_multiplications": 95,
    "explicit_rescale_requests": 800,
    "chebyshev_evaluations": 55,
    "estimated_polynomial_multiplications": 1150,
    "bootstraps": 25,
    "bootstrap_iterations": 50,
}
REFRESH_COUNTS = {
    "rotations": 0,
    "ct_pt_multiplications": 5,
    "ct_ct_multiplications": 0,
    "explicit_rescale_requests": 5,
    "chebyshev_evaluations": 0,
    "estimated_polynomial_multiplications": 0,
    "bootstraps": 5,
    "bootstrap_iterations": 10,
}
TOTAL_COUNTS = {
    "rotations": 75600,
    "ct_pt_multiplications": 622435,
    "ct_ct_multiplications": 1140,
    "explicit_rescale_requests": 9655,
    "chebyshev_evaluations": 660,
    "estimated_polynomial_multiplications": 13800,
    "bootstraps": 355,
    "bootstrap_iterations": 710,
}
# Live crypto preflight: actual scale_bits=50.000000079945785.  Freeze the
# schedule at 2^50 while retaining the 1e-3 tolerance for future live values.
LAYER0_INPUT_METADATA: dict[str, Any] | None = {
    "level": 29,
    "noise_scale_degree": 1,
    "remaining_levels": 18,
    "scale_bits": 50,
    "expected_scale_bits": 50,
    "ciphertext_count": 5,
}
LAYER_HANDOFF_INPUT_METADATA: dict[str, Any] | None = {
    "level": 19,
    "noise_scale_degree": 2,
    "remaining_levels": 27,
    "scale_bits": 100,
    "expected_scale_bits": 100,
    "ciphertext_count": 5,
}
RAW_OUTPUT_METADATA = {
    "level": 29,
    "noise_scale_degree": 2,
    "remaining_levels": 17,
    "scale_bits": 100,
    "expected_scale_bits": 100,
    "ciphertext_count": 5,
}
POLYNOMIAL_CHECKPOINT_METADATA: dict[str, Any] | None = {
    "level": 18,
    "noise_scale_degree": 2,
    "remaining_levels": 28,
    "scale_bits": 100,
    "expected_scale_bits": 100,
    "ciphertext_count": 5,
}
LAYER0_ATTENTION_OUTPUT_METADATA = {
    "level": 40,
    "noise_scale_degree": 2,
    "remaining_levels": 6,
    "scale_bits": 100,
    "expected_scale_bits": 100,
    "ciphertext_count": 5,
}
POST_REFRESH_ATTENTION_OUTPUT_METADATA = {
    "level": 31,
    "noise_scale_degree": 2,
    "remaining_levels": 15,
    "scale_bits": 100,
    "expected_scale_bits": 100,
    "ciphertext_count": 5,
}
LAYER0_LN1_OUTPUT_METADATA = {
    "level": 32,
    "noise_scale_degree": 2,
    "remaining_levels": 14,
    "scale_bits": 100,
    "expected_scale_bits": 100,
    "ciphertext_count": 5,
}
POST_REFRESH_LN1_OUTPUT_METADATA = {
    "level": 29,
    "noise_scale_degree": 2,
    "remaining_levels": 17,
    "scale_bits": 100,
    "expected_scale_bits": 100,
    "ciphertext_count": 5,
}
LAYER0_FFN_OUTPUT_METADATA = {
    "level": 45,
    "noise_scale_degree": 2,
    "remaining_levels": 1,
    "scale_bits": 100,
    "expected_scale_bits": 100,
    "ciphertext_count": 5,
}
POST_REFRESH_FFN_OUTPUT_METADATA = {
    "level": 42,
    "noise_scale_degree": 2,
    "remaining_levels": 4,
    "scale_bits": 100,
    "expected_scale_bits": 100,
    "ciphertext_count": 5,
}
RELATIVE_USED_LEVEL_DELTAS = {
    "attention_output_from_layer_input": {
        "layer_0": 12,
        "layers_1_to_11": 12,
    },
    "ln1_output_from_ln1_variance": {
        "layer_0": 14,
        "layers_1_to_11": 11,
    },
    "ffn_output_from_ln1_output": 13,
    "raw_output_from_ln2_variance": 11,
}
CHECKPOINT_METADATA_FIELDS = (
    "input_metadata",
    "softmax_denominator_metadata",
    "attention_output_metadata",
    "ln1_variance_metadata",
    "ln1_output_metadata",
    "ffn_output_metadata",
    "ln2_variance_metadata",
    "raw_output_metadata",
)
CHECKPOINT_METADATA_HASH_FIELDS = ("layer_id", *CHECKPOINT_METADATA_FIELDS)
CRYPTO_PREFLIGHT_KEYS = {
    "test",
    "profile",
    "security_claim",
    "input_metadata",
    "server_private_key_present",
    "server_decryptions",
    "server_plaintext_activations",
    "additive_he_operations",
    "passed",
}
POLYNOMIAL_INTERVALS = {
    "softmax_shifted_logits": [-16.0, 5.0],
    "softmax_denominator": [0.01, 80.0],
    "ln1_normalized_variance": [0.5, 1536.0],
    "gelu_input": [-80.0, 128.0],
    "ln2_normalized_variance": [0.5, 1536.0],
}
THRESHOLDS = {
    "per_layer_relative_l2_max": 5e-2,
    "per_layer_cosine_min": 0.99,
    "final_relative_l2_max": 5e-2,
    "final_cosine_min": 0.99,
    "inactive_max_abs": 1e-6,
    "inactive_sentinel_in_interval_required": True,
    "finite_required": True,
}
INACTIVE_POLYNOMIAL_SENTINEL_INTERVALS = {
    name: POLYNOMIAL_INTERVALS[name]
    for name in (
        "softmax_denominator",
        "ln1_normalized_variance",
        "ln2_normalized_variance",
    )
}
EXPECTED_PROFILE_FIELDS = {
    "effective_profile_schema_version": 1,
    "ring_dimension": 65536,
    "slot_count": 32768,
    "scaling_modulus_bits": 50,
    "first_modulus_bits": 55,
    "multiplicative_depth": 47,
    "levels_available_after_bootstrap": 28,
    "bootstrap_slots": 1024,
    "bootstrap_iterations": 2,
    "bootstrap_precision": 14,
    "scaling_technique": "FLEXIBLEAUTO",
    "security_claim": "none",
    "security_level": "HEStd_NotSet",
}
EXPECTED_EXECUTION = {
    "mode": "server-only",
    "encoder_layers": 12,
    "layer_ids": list(LAYER_IDS),
    "trace_shape": TRACE_SHAPE,
    "feature_block_size": FEATURE_BLOCK_SIZE,
    "chain_mode": "ciphertext_output_to_next_input",
    "client_encrypt_calls": 1,
    "encrypted_input_ciphertexts": 5,
    "plaintext_activation_resets": 0,
    "server_layer_evaluations": 12,
    "inter_layer_refreshes": 11,
    "checkpoint_decryption_owner": "client",
    "final_decryption_owner": "client",
    "server_private_key_present": False,
    "server_decryptions": 0,
    "server_plaintext_activations": False,
    "approximation_range_status": "all_client_validated",
    "quality_reference": (
        "12-layer chained frozen polynomial oracle from "
        "config/moai_encoder_trace.json"
    ),
}
M5_CTEST_PATTERN = (
    "^(moai_trace_contract|openfhe_(server_trust_boundary|profile_contract|"
    "profile_validator_contract|artifact_schema_contract|"
    "artifact_validator_contract|encoder_artifact_runner_contract|"
    "m5_artifact_schema_contract|m5_artifact_validator_contract|"
    "encoder12_artifact_runner_contract|evaluation_key_bundle_smoke|"
    "feature_packed_smoke|feature_packed_attention_smoke|"
    "feature_bootstrap_smoke|encoder_fixture_contract|"
    "encoder_plaintext_oracle_smoke|encoder_12_layer_preflight|"
    "encoder_12_layer_crypto_preflight|"
    "encoder_trace_contract|encoder_trace_validator_contract))$"
)
CSV_FIELDS = (
    "run",
    "exit_code",
    "elapsed_seconds",
    "peak_rss_kib",
    "fixture_load_oracle_ms",
    "setup_keygen_ms",
    "client_encrypt_ms",
    "server_online_diagnostic_ms",
    "client_checkpoint_validate_ms",
    "final_rel_l2",
    "final_cosine",
    "final_max_absolute",
    "inactive_max_abs",
    "inactive_sentinel_max_error",
    "checkpoint_metadata_sha256",
    "final_level",
    "final_noise_scale_degree",
    "final_remaining_levels",
    "final_scale_bits",
    "final_ciphertext_count",
    *OPERATION_COUNT_KEYS,
    "timing_claim",
    "latency_kind",
    "finite",
    "passed",
)
CLAIM_BOUNDARY = [
    "Five-token M5 12-layer encoder trace replay only; not task-level inference.",
    "paper_compat OpenFHE CKKS CPU parameters with security_claim=none.",
    "Timing is a non-benchmark diagnostic; no speedup claim.",
    "GPU, Discrete CKKS/FBT, QDQ, tokenizer, and classifier are excluded.",
]
REQUIRED_TOP_LEVEL = {
    "schema_version",
    "run_id",
    "milestone",
    "started_at",
    "finished_at",
    "git",
    "commands",
    "environment",
    "profile",
    "inputs",
    "workload",
    "contracts",
    "metrics",
    "gate",
    "artifacts",
    "claim_boundary",
    "verdict",
}
SUPPORTED_SCHEMA_KEYWORDS = {
    "$schema",
    "$id",
    "$defs",
    "$ref",
    "title",
    "description",
    "type",
    "additionalProperties",
    "required",
    "properties",
    "const",
    "enum",
    "pattern",
    "format",
    "minLength",
    "maxLength",
    "minimum",
    "maximum",
    "minItems",
    "maxItems",
    "uniqueItems",
    "items",
    "prefixItems",
    "allOf",
    "minProperties",
}


class ValidationError(RuntimeError):
    """Raised when an M5 schema, manifest, or evidence file drifts."""


def _expected_metadata_schedule_contract() -> dict[str, Any]:
    if LAYER0_INPUT_METADATA is None:
        raise ValidationError(
            "M5 layer-0 input metadata is not sealed by the crypto preflight"
        )
    if LAYER_HANDOFF_INPUT_METADATA is None:
        raise ValidationError(
            "M5 layer-handoff input metadata is not sealed by the two-layer seam run"
        )
    if POLYNOMIAL_CHECKPOINT_METADATA is None:
        raise ValidationError(
            "M5 polynomial checkpoint metadata is not sealed by the two-layer seam run"
        )
    schedule = {
        "multiplicative_depth": EXPECTED_PROFILE_FIELDS["multiplicative_depth"],
        "used_level_formula": "level + noise_scale_degree - 1",
        "remaining_levels_formula": "multiplicative_depth - used_level",
        "initial_layer_input": dict(LAYER0_INPUT_METADATA),
        "post_refresh_layer_input": dict(LAYER_HANDOFF_INPUT_METADATA),
        "post_bootstrap_polynomial_checkpoint": dict(
            POLYNOMIAL_CHECKPOINT_METADATA
        ),
        "raw_layer_output": dict(RAW_OUTPUT_METADATA),
        "attention_output_by_layer_regime": {
            "layer_0": dict(LAYER0_ATTENTION_OUTPUT_METADATA),
            "layers_1_to_11": dict(POST_REFRESH_ATTENTION_OUTPUT_METADATA),
        },
        "ln1_output_by_layer_regime": {
            "layer_0": dict(LAYER0_LN1_OUTPUT_METADATA),
            "layers_1_to_11": dict(POST_REFRESH_LN1_OUTPUT_METADATA),
        },
        "ffn_output_by_layer_regime": {
            "layer_0": dict(LAYER0_FFN_OUTPUT_METADATA),
            "layers_1_to_11": dict(POST_REFRESH_FFN_OUTPUT_METADATA),
        },
        "relative_used_level_deltas": dict(RELATIVE_USED_LEVEL_DELTAS),
        "checkpoint_metadata_digest": {
            "algorithm": "SHA-256",
            "canonicalization": CANONICALIZATION,
            "layer_order": "0 through 11",
            "fields": list(CHECKPOINT_METADATA_HASH_FIELDS),
        },
    }
    for name in (
        "initial_layer_input",
        "post_refresh_layer_input",
        "post_bootstrap_polynomial_checkpoint",
        "raw_layer_output",
    ):
        _require_metadata(schedule[name], schedule[name], f"metadata_schedule.{name}")
    for field in (
        "attention_output_by_layer_regime",
        "ln1_output_by_layer_regime",
        "ffn_output_by_layer_regime",
    ):
        for regime, metadata in schedule[field].items():
            _require_metadata(
                metadata,
                metadata,
                f"metadata_schedule.{field}.{regime}",
            )
    for regime, input_field in (
        ("layer_0", "initial_layer_input"),
        ("layers_1_to_11", "post_refresh_layer_input"),
    ):
        _require_used_level_delta(
            schedule["attention_output_by_layer_regime"][regime],
            schedule[input_field],
            RELATIVE_USED_LEVEL_DELTAS[
                "attention_output_from_layer_input"
            ][regime],
            f"metadata_schedule.{regime}_attention",
        )
        _require_used_level_delta(
            schedule["ln1_output_by_layer_regime"][regime],
            schedule["post_bootstrap_polynomial_checkpoint"],
            RELATIVE_USED_LEVEL_DELTAS[
                "ln1_output_from_ln1_variance"
            ][regime],
            f"metadata_schedule.{regime}_ln1",
        )
        _require_used_level_delta(
            schedule["ffn_output_by_layer_regime"][regime],
            schedule["ln1_output_by_layer_regime"][regime],
            RELATIVE_USED_LEVEL_DELTAS["ffn_output_from_ln1_output"],
            f"metadata_schedule.{regime}_ffn",
        )
    _require_used_level_delta(
        schedule["raw_layer_output"],
        schedule["post_bootstrap_polynomial_checkpoint"],
        RELATIVE_USED_LEVEL_DELTAS["raw_output_from_ln2_variance"],
        "metadata_schedule.raw_output",
    )
    return schedule


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_non_finite(token: str) -> None:
    raise ValidationError(f"non-finite JSON number is forbidden: {token}")


def load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(
                handle,
                object_pairs_hook=_object_without_duplicates,
                parse_constant=_reject_non_finite,
            )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValidationError(f"cannot read JSON {path}: {error}") from error


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValidationError(f"value cannot be canonicalized: {error}") from error


def checkpoint_metadata_payload(layers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(layers) != 12:
        raise ValidationError(
            "checkpoint metadata digest requires exactly 12 ordered layer records"
        )
    payload: list[dict[str, Any]] = []
    for layer_id, layer in enumerate(layers):
        if layer.get("layer_id") != layer_id:
            raise ValidationError(
                "checkpoint metadata digest requires layer order 0 through 11"
            )
        missing = set(CHECKPOINT_METADATA_FIELDS) - set(layer)
        if missing:
            raise ValidationError(
                f"checkpoint metadata digest is missing fields: {sorted(missing)}"
            )
        payload.append(
            {
                "layer_id": layer_id,
                **{field: layer[field] for field in CHECKPOINT_METADATA_FIELDS},
            }
        )
    return payload


def checkpoint_metadata_sha256(layers: list[dict[str, Any]]) -> str:
    return hashlib.sha256(
        canonical_json_bytes(checkpoint_metadata_payload(layers))
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _json_equal(left: Any, right: Any) -> bool:
    return canonical_json_bytes(left) == canonical_json_bytes(right)


def _resolve_ref(root: dict[str, Any], reference: str) -> dict[str, Any]:
    if not reference.startswith("#/"):
        raise ValidationError(f"only local JSON Schema references are supported: {reference}")
    current: Any = root
    for encoded in reference[2:].split("/"):
        part = encoded.replace("~1", "/").replace("~0", "~")
        if not isinstance(current, dict) or part not in current:
            raise ValidationError(f"unresolvable JSON Schema reference: {reference}")
        current = current[part]
    if not isinstance(current, dict):
        raise ValidationError(f"JSON Schema reference is not an object: {reference}")
    return current


def _type_matches(value: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
        )
    raise ValidationError(f"unsupported JSON Schema type: {expected}")


def _parse_timestamp(value: str, label: str) -> datetime:
    if not (value.endswith("Z") or re.search(r"[+-][0-9]{2}:[0-9]{2}$", value)):
        raise ValidationError(f"{label}: date-time must include a timezone")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValidationError(f"{label}: invalid RFC 3339 date-time") from error
    if result.tzinfo is None:
        raise ValidationError(f"{label}: date-time must include a timezone")
    return result


def validate_instance(
    instance: Any,
    schema: dict[str, Any],
    root: dict[str, Any],
    label: str = "$",
) -> None:
    if "$ref" in schema:
        validate_instance(instance, _resolve_ref(root, schema["$ref"]), root, label)
        return
    for subschema in schema.get("allOf", []):
        validate_instance(instance, subschema, root, label)
    expected_type = schema.get("type")
    if expected_type is not None and not _type_matches(instance, expected_type):
        raise ValidationError(
            f"{label}: expected {expected_type}, got {type(instance).__name__}"
        )
    if "const" in schema and not _json_equal(instance, schema["const"]):
        raise ValidationError(f"{label}: expected constant {schema['const']!r}")
    if "enum" in schema and not any(
        _json_equal(instance, candidate) for candidate in schema["enum"]
    ):
        raise ValidationError(f"{label}: value is not in the allowed enum")
    if isinstance(instance, dict):
        if len(instance) < schema.get("minProperties", 0):
            raise ValidationError(f"{label}: object has too few properties")
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        for key in required:
            if key not in instance:
                raise ValidationError(f"{label}: missing required property {key!r}")
        for key, value in instance.items():
            if key in properties:
                validate_instance(value, properties[key], root, f"{label}.{key}")
            elif schema.get("additionalProperties", True) is False:
                raise ValidationError(f"{label}: unexpected property {key!r}")
    if isinstance(instance, list):
        if len(instance) < schema.get("minItems", 0):
            raise ValidationError(f"{label}: array is shorter than minItems")
        if "maxItems" in schema and len(instance) > schema["maxItems"]:
            raise ValidationError(f"{label}: array is longer than maxItems")
        if schema.get("uniqueItems", False):
            serialized = [canonical_json_bytes(item) for item in instance]
            if len(serialized) != len(set(serialized)):
                raise ValidationError(f"{label}: array items must be unique")
        prefix_schemas = schema.get("prefixItems", [])
        for index, prefix_schema in enumerate(prefix_schemas[: len(instance)]):
            validate_instance(instance[index], prefix_schema, root, f"{label}[{index}]")
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(instance[len(prefix_schemas) :], len(prefix_schemas)):
                validate_instance(item, item_schema, root, f"{label}[{index}]")
    if isinstance(instance, str):
        if len(instance) < schema.get("minLength", 0):
            raise ValidationError(f"{label}: string is shorter than minLength")
        if "maxLength" in schema and len(instance) > schema["maxLength"]:
            raise ValidationError(f"{label}: string is longer than maxLength")
        if "pattern" in schema and re.search(schema["pattern"], instance) is None:
            raise ValidationError(f"{label}: string does not match the required pattern")
        if schema.get("format") == "date-time":
            _parse_timestamp(instance, label)
    if isinstance(instance, (int, float)) and not isinstance(instance, bool):
        if not math.isfinite(instance):
            raise ValidationError(f"{label}: non-finite number is forbidden")
        if "minimum" in schema and instance < schema["minimum"]:
            raise ValidationError(f"{label}: value is below minimum")
        if "maximum" in schema and instance > schema["maximum"]:
            raise ValidationError(f"{label}: value is above maximum")


def _walk_schema(node: Any, root: dict[str, Any], label: str = "$schema") -> None:
    if not isinstance(node, dict):
        raise ValidationError(f"{label}: every schema node must be an object")
    unknown = set(node) - SUPPORTED_SCHEMA_KEYWORDS
    if unknown:
        raise ValidationError(f"{label}: unsupported schema keywords: {sorted(unknown)}")
    if "$ref" in node:
        if set(node) != {"$ref"}:
            raise ValidationError(f"{label}: $ref must not have sibling keywords")
        if not isinstance(node["$ref"], str):
            raise ValidationError(f"{label}.$ref must be a string")
        _resolve_ref(root, node["$ref"])
    if "type" in node:
        if not isinstance(node["type"], str):
            raise ValidationError(f"{label}.type must be a string")
        _type_matches(None, node["type"])
    if "enum" in node:
        choices = node["enum"]
        if not isinstance(choices, list) or not choices:
            raise ValidationError(f"{label}.enum must be a non-empty array")
        serialized = [canonical_json_bytes(choice) for choice in choices]
        if len(serialized) != len(set(serialized)):
            raise ValidationError(f"{label}.enum values must be unique")
    if "pattern" in node:
        if not isinstance(node["pattern"], str):
            raise ValidationError(f"{label}.pattern must be a string")
        try:
            re.compile(node["pattern"])
        except (TypeError, re.error) as error:
            raise ValidationError(f"{label}.pattern is invalid: {error}") from error
    if "format" in node and node["format"] != "date-time":
        raise ValidationError(f"{label}.format is unsupported")
    for keyword in (
        "minLength",
        "maxLength",
        "minItems",
        "maxItems",
        "minProperties",
    ):
        if keyword in node and (
            not isinstance(node[keyword], int)
            or isinstance(node[keyword], bool)
            or node[keyword] < 0
        ):
            raise ValidationError(f"{label}.{keyword} must be a nonnegative integer")
    for keyword in ("minimum", "maximum"):
        if keyword in node and (
            not isinstance(node[keyword], (int, float))
            or isinstance(node[keyword], bool)
            or not math.isfinite(node[keyword])
        ):
            raise ValidationError(f"{label}.{keyword} must be finite")
    for minimum_key, maximum_key in (
        ("minimum", "maximum"),
        ("minLength", "maxLength"),
        ("minItems", "maxItems"),
    ):
        if (
            minimum_key in node
            and maximum_key in node
            and node[minimum_key] > node[maximum_key]
        ):
            raise ValidationError(
                f"{label}.{minimum_key} must not exceed {maximum_key}"
            )
    if "uniqueItems" in node and not isinstance(node["uniqueItems"], bool):
        raise ValidationError(f"{label}.uniqueItems must be boolean")
    if "additionalProperties" in node and not isinstance(
        node["additionalProperties"], bool
    ):
        raise ValidationError(f"{label}.additionalProperties must be boolean")
    all_of = node.get("allOf", [])
    if not isinstance(all_of, list) or not all_of:
        if "allOf" in node:
            raise ValidationError(f"{label}.allOf must be a non-empty array")
    else:
        for index, child in enumerate(all_of):
            _walk_schema(child, root, f"{label}.allOf[{index}]")
    properties = node.get("properties", {})
    required = node.get("required", [])
    if not isinstance(properties, dict):
        raise ValidationError(f"{label}.properties must be an object")
    if not isinstance(required, list) or not all(isinstance(item, str) for item in required):
        raise ValidationError(f"{label}.required must be an array of strings")
    if len(required) != len(set(required)) or set(required) - set(properties):
        raise ValidationError(f"{label}.required is duplicated or lacks property schemas")
    for key, child in properties.items():
        _walk_schema(child, root, f"{label}.properties.{key}")
    definitions = node.get("$defs", {})
    if not isinstance(definitions, dict):
        raise ValidationError(f"{label}.$defs must be an object")
    for key, child in definitions.items():
        _walk_schema(child, root, f"{label}.$defs.{key}")
    item_schema = node.get("items")
    if item_schema is not None:
        _walk_schema(item_schema, root, f"{label}.items")
    prefix_items = node.get("prefixItems", [])
    if not isinstance(prefix_items, list) or not prefix_items:
        if "prefixItems" in node:
            raise ValidationError(f"{label}.prefixItems must be a non-empty array")
    else:
        if "maxItems" in node and len(prefix_items) > node["maxItems"]:
            raise ValidationError(
                f"{label}.prefixItems must not exceed maxItems"
            )
        for index, child in enumerate(prefix_items):
            _walk_schema(child, root, f"{label}.prefixItems[{index}]")


def validate_schema(schema: Any) -> dict[str, Any]:
    if not isinstance(schema, dict):
        raise ValidationError("schema root must be an object")
    if (
        schema.get("$schema") != SCHEMA_URI
        or schema.get("$id") != SCHEMA_ID
        or schema.get("title") != SCHEMA_TITLE
    ):
        raise ValidationError("schema identity drifted")
    if schema.get("type") != "object" or schema.get("additionalProperties") is not False:
        raise ValidationError("schema root must be a closed object")
    if set(schema.get("required", [])) != REQUIRED_TOP_LEVEL:
        raise ValidationError("schema top-level required fields drifted")
    properties = schema.get("properties", {})
    if properties.get("schema_version", {}).get("const") != M5_SCHEMA_VERSION:
        raise ValidationError("M5 schema_version must be frozen at 4")
    if properties.get("milestone", {}).get("const") != "M5":
        raise ValidationError("schema-v4 milestone must be exactly M5")
    inputs = properties.get("inputs", {})
    if inputs.get("minItems") != INPUT_FILE_COUNT or inputs.get("maxItems") != INPUT_FILE_COUNT:
        raise ValidationError("M5 schema must require exactly 448 inputs")
    layers = properties.get("metrics", {}).get("properties", {}).get("layers", {})
    prefix_items = layers.get("prefixItems") if isinstance(layers, dict) else None
    if (
        not isinstance(layers, dict)
        or layers.get("minItems") != len(LAYER_IDS)
        or layers.get("maxItems") != len(LAYER_IDS)
        or not isinstance(prefix_items, list)
        or len(prefix_items) != len(LAYER_IDS)
        or "items" in layers
    ):
        raise ValidationError(
            "M5 schema metrics.layers must use exactly 12 closed prefixItems"
        )
    _walk_schema(schema, schema)
    return schema


def _require_exact_keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValidationError(f"{label} must be an object")
    if set(value) != expected:
        raise ValidationError(
            f"{label} keys differ: missing={sorted(expected - set(value))} "
            f"extra={sorted(set(value) - expected)}"
        )
    return value


def _require_finite(
    value: Any,
    label: str,
    minimum: float = -math.inf,
    maximum: float = math.inf,
) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value < minimum
        or value > maximum
    ):
        raise ValidationError(f"{label} violates [{minimum},{maximum}]")
    return float(value)


def _require_typed_equal(actual: Any, expected: Any, label: str) -> None:
    if not _json_equal(actual, expected):
        raise ValidationError(
            f"{label} mismatch: expected={expected!r} actual={actual!r}"
        )


def _resolve_confined_file(base: Path, relative: str, label: str) -> Path:
    raw = Path(relative)
    if (
        raw.is_absolute()
        or "\\" in relative
        or ".." in raw.parts
        or raw.as_posix() != relative
    ):
        raise ValidationError(f"{label}: path must be normalized and relative: {relative}")
    resolved_base = base.resolve()
    resolved = (resolved_base / raw).resolve()
    try:
        resolved.relative_to(resolved_base)
    except ValueError as error:
        raise ValidationError(f"{label}: path escapes its evidence root") from error
    if not resolved.is_file():
        raise ValidationError(f"{label}: file does not exist: {resolved}")
    return resolved


def _verify_file_record(path: Path, record: dict[str, Any], label: str) -> None:
    if path.stat().st_size != record["bytes"]:
        raise ValidationError(f"{label}: byte count mismatch for {path}")
    if sha256_file(path) != record["sha256"]:
        raise ValidationError(f"{label}: SHA-256 mismatch for {path}")


def _decode_pointer(document: Any, pointer: str) -> Any:
    if not pointer.startswith("/"):
        raise ValidationError("effective profile locator must be an absolute JSON pointer")
    current = document
    for encoded in pointer[1:].split("/"):
        part = encoded.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif isinstance(current, list) and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
        else:
            raise ValidationError("effective profile locator does not resolve")
    return current


def _verify_profile(
    manifest: dict[str, Any], input_records: dict[str, dict[str, Any]]
) -> None:
    profile = manifest["profile"]
    record = input_records[PROFILE_PATH]
    if (
        profile["source_config_sha256"] != record["sha256"]
        or profile["source_config_bytes"] != record["bytes"]
    ):
        raise ValidationError("profile source record differs from the required input")
    source = load_json(REPO_ROOT / PROFILE_PATH)
    effective = _decode_pointer(source, profile["effective_profile_locator"])
    if not _json_equal(effective, profile["effective_profile_payload"]):
        raise ValidationError("effective profile payload differs from its source config")
    digest = hashlib.sha256(canonical_json_bytes(effective)).hexdigest()
    if digest != PROFILE_SHA256 or profile["effective_profile_sha256"] != digest:
        raise ValidationError("effective profile SHA-256 differs from paper_compat")
    mismatches = {
        key: {"expected": expected, "actual": effective.get(key)}
        for key, expected in EXPECTED_PROFILE_FIELDS.items()
        if effective.get(key) != expected
    }
    if mismatches:
        raise ValidationError(f"effective paper_compat profile fields drifted: {mismatches}")


def _bundle_identity(records: list[dict[str, Any]]) -> str:
    payload = [
        {"path": record["path"], "sha256": record["sha256"], "bytes": record["bytes"]}
        for record in sorted(records, key=lambda item: item["path"])
    ]
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _expected_trace_inputs(
    trace_contract: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    try:
        required_files = trace_contract["required_files"]
        layers = trace_contract["layers"]
    except (KeyError, TypeError) as error:
        raise ValidationError(f"encoder trace contract is incomplete: {error}") from error
    if not isinstance(required_files, dict) or len(required_files) != 37:
        raise ValidationError("encoder trace contract must map exactly 37 logical files")
    if not isinstance(layers, list) or len(layers) != 12:
        raise ValidationError("encoder trace contract must contain exactly 12 layers")
    if any(not isinstance(layer, dict) for layer in layers):
        raise ValidationError("encoder trace layers must be objects")
    ordered_layer_ids = [layer.get("layer_id") for layer in layers]
    if ordered_layer_ids != list(LAYER_IDS):
        raise ValidationError("encoder trace layer ids must be ordered identities 0 through 11")
    by_id = {layer["layer_id"]: layer for layer in layers}

    expected: dict[str, dict[str, Any]] = {}
    groups: list[dict[str, Any]] = []
    for layer_id in LAYER_IDS:
        layer = by_id[layer_id]
        hashes = layer.get("sha256")
        if not isinstance(hashes, dict) or set(hashes) != set(required_files):
            raise ValidationError(f"trace layer {layer_id} does not bind all 37 files")
        group = {"weights": [], "trace": []}
        for logical_name in sorted(required_files):
            specification = required_files[logical_name]
            if not isinstance(specification, dict) or not isinstance(
                specification.get("path"), str
            ):
                raise ValidationError(f"trace mapping {logical_name!r} is malformed")
            contract_path = specification["path"]
            raw = Path(contract_path)
            if (
                raw.is_absolute()
                or "\\" in contract_path
                or ".." in raw.parts
                or raw.as_posix() != contract_path
            ):
                raise ValidationError(f"trace mapping {logical_name!r} is not normalized")
            path = (Path("data") / f"layer_{layer_id}" / raw).as_posix()
            if path in expected:
                raise ValidationError(f"encoder trace paths collide at {path}")
            role = "weights" if "/parms/" in f"/{contract_path}" else "trace"
            expected[path] = {
                "sha256": hashes[logical_name],
                "media_type": "text/csv",
                "role": role,
                "layer_id": layer_id,
            }
            group[role].append(path)
        if (
            len(group["weights"]) != WEIGHT_FILES_PER_LAYER
            or len(group["trace"]) != TRACE_FILES_PER_LAYER
        ):
            raise ValidationError(f"trace layer {layer_id} weight/trace split drifted")
        groups.append(group)
    if len(expected) != CSV_FILE_COUNT:
        raise ValidationError("encoder trace contract does not resolve to exactly 444 CSVs")
    return expected, groups


def _verify_inputs_and_contracts(
    manifest: dict[str, Any], repository_root: Path
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    paths = [record["path"] for record in manifest["inputs"]]
    if paths != sorted(paths):
        raise ValidationError("M5 inputs must be ordered lexicographically by path")
    for index, record in enumerate(manifest["inputs"]):
        path = record["path"]
        if path in records:
            raise ValidationError(f"duplicate input path: {path}")
        resolved = _resolve_confined_file(repository_root, path, f"inputs[{index}].path")
        _verify_file_record(resolved, record, f"inputs[{index}]")
        records[path] = record

    trace_contract = load_json(repository_root / "config/moai_encoder_trace.json")
    try:
        if (
            trace_contract["profile_id"] != "paper_compat"
            or trace_contract["security_claim"] != "none"
            or trace_contract["dimensions"]["encoder_layers"] != 12
            or trace_contract["dimensions"]["trace_token_rows"] != 5
            or trace_contract["dimensions"]["hidden_size"] != 768
        ):
            raise ValidationError("encoder trace contract identity drifted")
    except (KeyError, TypeError) as error:
        raise ValidationError(f"encoder trace identity is incomplete: {error}") from error

    expected_trace, groups = _expected_trace_inputs(trace_contract)
    expected_paths = set(CONFIG_INPUTS) | set(expected_trace)
    if set(records) != expected_paths or len(records) != INPUT_FILE_COUNT:
        raise ValidationError(
            "M5 inputs differ from four configs plus 444 CSVs: "
            f"missing={sorted(expected_paths - set(records))} "
            f"extra={sorted(set(records) - expected_paths)}"
        )
    for path in CONFIG_INPUTS:
        record = records[path]
        if record["media_type"] != "application/json" or record["role"] != "configuration":
            raise ValidationError(f"M5 config input {path} has the wrong media type or role")
    for path, expected in expected_trace.items():
        record = records[path]
        for key in ("sha256", "media_type", "role"):
            if record[key] != expected[key]:
                raise ValidationError(
                    f"M5 trace input {path} has wrong {key}: "
                    f"expected={expected[key]!r} actual={record[key]!r}"
                )

    contracts = manifest["contracts"]
    bindings = {
        "approximation_config": "config/openfhe_approximations.json",
        "channel_scales_config": "config/moai_trace_channel_scales.json",
        "encoder_trace_contract": "config/moai_encoder_trace.json",
        "profile_config": PROFILE_PATH,
    }
    for prefix, expected_path in bindings.items():
        if contracts[f"{prefix}_path"] != expected_path:
            raise ValidationError(f"contracts.{prefix}_path drifted")
        if contracts[f"{prefix}_sha256"] != records[expected_path]["sha256"]:
            raise ValidationError(f"contracts.{prefix}_sha256 differs from inputs")

    try:
        sources = trace_contract["sources"]
        if set(sources) != {"approximations", "channel_scales"}:
            raise ValidationError("encoder trace source identities drifted")
        source_paths = {
            "approximations": "config/openfhe_approximations.json",
            "channel_scales": "config/moai_trace_channel_scales.json",
        }
        for name, path in source_paths.items():
            if sources[name]["path"] != path or sources[name]["sha256"] != records[path]["sha256"]:
                raise ValidationError(f"encoder trace source {name} differs from inputs")
    except (KeyError, TypeError) as error:
        raise ValidationError(f"encoder trace source binding is incomplete: {error}") from error

    expected_identities: list[dict[str, Any]] = []
    content_identities = {"weights": set(), "trace": set()}
    for layer_id, group in enumerate(groups):
        weight_records = [records[path] for path in group["weights"]]
        trace_records = [records[path] for path in group["trace"]]
        expected_identities.append(
            {
                "layer_id": layer_id,
                "weight_file_count": WEIGHT_FILES_PER_LAYER,
                "trace_file_count": TRACE_FILES_PER_LAYER,
                "weight_bundle_sha256": _bundle_identity(weight_records),
                "trace_bundle_sha256": _bundle_identity(trace_records),
            }
        )
        layer_prefix = Path("data") / f"layer_{layer_id}"
        for role, role_records in (
            ("weights", weight_records),
            ("trace", trace_records),
        ):
            content_payload = [
                {
                    "contract_path": Path(record["path"])
                    .relative_to(layer_prefix)
                    .as_posix(),
                    "sha256": record["sha256"],
                    "bytes": record["bytes"],
                }
                for record in sorted(role_records, key=lambda item: item["path"])
            ]
            content_identities[role].add(
                hashlib.sha256(canonical_json_bytes(content_payload)).hexdigest()
            )
    if not _json_equal(contracts["layer_input_identities"], expected_identities):
        raise ValidationError("contracts.layer_input_identities differs from 444 inputs")
    weight_identities = {
        identity["weight_bundle_sha256"] for identity in expected_identities
    }
    trace_identities = {
        identity["trace_bundle_sha256"] for identity in expected_identities
    }
    if len(weight_identities) != 12 or len(trace_identities) != 12:
        raise ValidationError("all 12 weight and trace bundle identities must be distinct")
    if any(len(identities) != 12 for identities in content_identities.values()):
        raise ValidationError(
            "all 12 weight and trace bundles must also have distinct content identities"
        )

    if not _json_equal(contracts["execution"], EXPECTED_EXECUTION):
        raise ValidationError("contracts.execution differs from the frozen ciphertext chain")
    _require_crypto_preflight_record(
        contracts["crypto_preflight"],
        "contracts.crypto_preflight",
    )
    if not _json_equal(
        contracts["metadata_schedule"],
        _expected_metadata_schedule_contract(),
    ):
        raise ValidationError(
            "contracts.metadata_schedule differs from the frozen level schedule"
        )
    if not _json_equal(contracts["thresholds"], THRESHOLDS):
        raise ValidationError("contracts.thresholds differs from the frozen M5 gate")
    if not _json_equal(contracts["polynomial_intervals"], POLYNOMIAL_INTERVALS):
        raise ValidationError("contracts.polynomial_intervals differs from the frozen ranges")

    approximation = load_json(repository_root / "config/openfhe_approximations.json")
    try:
        operators = approximation["operators"]
        config_intervals = {
            "softmax_shifted_logits": [
                operators["softmax"]["exponential"]["interval"]["minimum"],
                operators["softmax"]["exponential"]["interval"]["maximum"],
            ],
            "softmax_denominator": [
                operators["softmax"]["reciprocal"]["interval"]["minimum"],
                operators["softmax"]["reciprocal"]["interval"]["maximum"],
            ],
            "ln1_normalized_variance": [
                operators["layernorm"]["inverse_sqrt"]["interval"]["minimum"],
                operators["layernorm"]["inverse_sqrt"]["interval"]["maximum"],
            ],
            "gelu_input": [
                operators["gelu"]["polynomial"]["interval"]["minimum"],
                operators["gelu"]["polynomial"]["interval"]["maximum"],
            ],
            "ln2_normalized_variance": [
                operators["layernorm"]["inverse_sqrt"]["interval"]["minimum"],
                operators["layernorm"]["inverse_sqrt"]["interval"]["maximum"],
            ],
        }
    except (KeyError, TypeError) as error:
        raise ValidationError(f"approximation range contract is incomplete: {error}") from error
    if not _json_equal(config_intervals, POLYNOMIAL_INTERVALS):
        raise ValidationError("checked-in approximation intervals drifted")

    _verify_profile(manifest, records)
    return records


def _command_tokens(record: Any, label: str, *, timed: bool) -> list[str]:
    keys = {"command", "cwd", "exit_code", "phase"}
    if timed:
        keys.update({"started_at", "finished_at"})
    value = _require_exact_keys(record, keys, label)
    if value["cwd"] != str(REPO_ROOT):
        raise ValidationError(f"{label}.cwd must be {REPO_ROOT}")
    if value["exit_code"] != 0 or value["phase"] != "artifact_generation":
        raise ValidationError(f"{label} must be a successful artifact-generation command")
    if timed:
        started = _parse_timestamp(value["started_at"], f"{label}.started_at")
        finished = _parse_timestamp(value["finished_at"], f"{label}.finished_at")
        if finished < started:
            raise ValidationError(f"{label}.finished_at precedes started_at")
    try:
        tokens = shlex.split(value["command"])
    except ValueError as error:
        raise ValidationError(f"{label}.command is invalid shell syntax") from error
    if not tokens:
        raise ValidationError(f"{label}.command is empty")
    return tokens


def _verify_command_transcript(manifest: dict[str, Any], manifest_path: Path) -> None:
    commands = manifest["commands"]
    expected_git = (
        ["git", "status", "--porcelain=v1", "--untracked-files=normal"],
        ["git", "branch", "--show-current"],
        ["git", "rev-parse", "HEAD"],
        ["git", "rev-parse", TRACKING_REF],
        ["git", "ls-remote", "--exit-code", REMOTE_NAME, REMOTE_REF],
    )
    for index, expected in enumerate(expected_git):
        actual = _command_tokens(
            commands[index], f"commands[{index}] Git preflight", timed=False
        )
        if actual != expected:
            raise ValidationError(f"commands[{index}] differs from the frozen Git preflight")

    build_root = REPO_ROOT / "build-openfhe"
    executable = REPO_ROOT / EXECUTABLE_PATH
    if _command_tokens(commands[5], "commands[5] clean-first build", timed=True) != [
        "cmake",
        "--build",
        str(build_root),
        "--clean-first",
        "-j",
        "4",
    ]:
        raise ValidationError("commands[5] differs from the frozen clean-first build")
    if _command_tokens(commands[6], "commands[6] narrow CTest", timed=True) != [
        "ctest",
        "--test-dir",
        str(build_root),
        "--output-on-failure",
        "--verbose",
        "--no-tests=error",
        "-R",
        M5_CTEST_PATTERN,
    ]:
        raise ValidationError("commands[6] differs from the frozen M5 narrow CTest")
    if _command_tokens(commands[7], "commands[7] OpenFHE linkage", timed=True) != [
        "ldd",
        str(executable),
    ]:
        raise ValidationError("commands[7] differs from the frozen ldd check")

    workload = _command_tokens(commands[8], "commands[8] M5 workload", timed=True)
    if len(workload) != 6 or workload[:2] != ["/usr/bin/time", "--format=%M"]:
        raise ValidationError("commands[8] must use the frozen GNU time invocation")
    if re.fullmatch(r"--output=/tmp/moai-m5-time-[A-Za-z0-9_.-]+\.txt", workload[2]) is None:
        raise ValidationError("commands[8] GNU time output is not a fixed temporary path")
    if workload[3:] != [str(executable), "--data-root", str(REPO_ROOT / "data")]:
        raise ValidationError("commands[8] M5 executable invocation drifted")

    validator = _command_tokens(commands[9], "commands[9] M5 validator", timed=False)
    if validator != [
        sys.executable,
        str(VALIDATOR_PATH),
        "--manifest",
        str(manifest_path.resolve()),
        "--verify-git",
    ]:
        raise ValidationError("commands[9] differs from the frozen M5 validator")


def _run_git(repository_root: Path, arguments: list[str]) -> str:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValidationError(f"Git verification failed: git {' '.join(arguments)}") from error
    return completed.stdout.strip()


def _verify_git(manifest: dict[str, Any], repository_root: Path, live: bool) -> None:
    record = manifest["git"]
    commits = {
        record["local_commit"],
        record["tracking_commit"],
        record["remote_commit"],
    }
    if len(commits) != 1:
        raise ValidationError("manifest local, tracking, and live-remote commits differ")
    if not record["clean"]:
        raise ValidationError("M5 GO evidence requires git.clean=true")
    if not live:
        raise ValidationError("M5 GO verdict requires --verify-git live verification")
    if _run_git(repository_root, ["rev-parse", "HEAD"]) != record["local_commit"]:
        raise ValidationError("live local HEAD differs from the manifest")
    if _run_git(repository_root, ["branch", "--show-current"]) != BRANCH:
        raise ValidationError("live branch differs from refactor/openfhe-cpu")
    if _run_git(repository_root, ["status", "--porcelain=v1", "--untracked-files=normal"]):
        raise ValidationError("live repository is not clean")
    if _run_git(repository_root, ["rev-parse", TRACKING_REF]) != record["tracking_commit"]:
        raise ValidationError("live tracking ref differs from the manifest")
    remote_output = _run_git(
        repository_root,
        ["ls-remote", "--exit-code", REMOTE_NAME, REMOTE_REF],
    )
    remote_lines = [line.split() for line in remote_output.splitlines() if line.strip()]
    if (
        len(remote_lines) != 1
        or len(remote_lines[0]) != 2
        or remote_lines[0]
        != [record["remote_commit"], REMOTE_REF]
    ):
        raise ValidationError("live remote ref differs from the manifest")


LAYER_STDOUT_KEYS = {
    "test",
    "profile",
    "security_claim",
    "layer_id",
    "weights_layer_id",
    "chain_input_source",
    "input_metadata",
    "raw_output_metadata",
    "softmax_denominator_metadata",
    "attention_output_metadata",
    "ln1_variance_metadata",
    "ln1_output_metadata",
    "ffn_output_metadata",
    "ln2_variance_metadata",
    "input_quality",
    "output_quality",
    "exact_trace_diagnostic",
    "inactive_max_abs",
    "inactive_zero_checkpoint_count",
    "inactive_sentinel_max_error",
    "inactive_polynomial_sentinel_ranges",
    "inactive_sentinel_range_status",
    "encrypted_polynomial_input_ranges",
    "handoff_refresh_performed",
    "refresh_operation_counts",
    "layer_operation_counts",
    "cumulative_operation_counts",
    "range_validation_owner",
    "checkpoint_decryption_owner",
    "server_decryptions",
    "server_plaintext_activations",
    "finite",
    "range_status",
}
SUMMARY_STDOUT_KEYS = {
    "test",
    "profile",
    "security_claim",
    "parameter_sha256",
    "execution_mode",
    "claim_scope",
    "encoder_layers",
    "chain_mode",
    "client_encrypt_calls",
    "encrypted_input_ciphertexts",
    "plaintext_activation_resets",
    "server_layer_evaluations",
    "inter_layer_refreshes",
    "checkpoint_decryption_owner",
    "final_decryption_owner",
    "server_private_key_present",
    "server_decryptions",
    "server_plaintext_activations",
    "approximation_range_status",
    "fixture_load_oracle_ms",
    "setup_keygen_ms",
    "client_encrypt_ms",
    "server_online_diagnostic_ms",
    "client_checkpoint_validate_ms",
    "relative_l2",
    "cosine",
    "max_absolute",
    "inactive_max_abs",
    "inactive_sentinel_max_error",
    "inactive_sentinel_range_status",
    "final_metadata",
    "operation_counts",
    "multiplicative_depth",
    "max_observed_level",
    "max_polynomial_depth",
    "peak_rss_bytes",
    "timing_claim",
    "latency_kind",
    "finite",
    "passed",
}
LAYER_METRIC_KEYS = {
    "layer_id",
    "weights_layer_id",
    "chain_input_source",
    "input_metadata",
    "raw_output_metadata",
    "softmax_denominator_metadata",
    "attention_output_metadata",
    "ln1_variance_metadata",
    "ln1_output_metadata",
    "ffn_output_metadata",
    "ln2_variance_metadata",
    "input_quality",
    "output_quality",
    "exact_trace_diagnostic",
    "inactive_max_abs",
    "inactive_zero_checkpoint_count",
    "inactive_sentinel_max_error",
    "inactive_polynomial_sentinel_ranges",
    "inactive_sentinel_range_status",
    "encrypted_polynomial_input_ranges",
    "handoff_refresh_performed",
    "refresh_operation_counts",
    "layer_operation_counts",
    "cumulative_operation_counts",
    "finite",
    "range_status",
}


def _parse_stdout_records(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise ValidationError(f"cannot read stdout.log: {error}") from error
    relevant: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        start = line.find("{")
        if start < 0:
            continue
        try:
            candidate = json.loads(
                line[start:],
                object_pairs_hook=_object_without_duplicates,
                parse_constant=_reject_non_finite,
            )
        except (json.JSONDecodeError, ValidationError) as error:
            raise ValidationError(
                f"stdout.log line {line_number} contains malformed JSON evidence"
            ) from error
        if not isinstance(candidate, dict):
            raise ValidationError(
                f"stdout.log line {line_number} contains non-object JSON evidence"
            )
        test_name = candidate.get("test")
        if test_name in {
            "openfhe_encoder_12_layer_layer",
            "openfhe_encoder_12_layer",
        }:
            relevant.append(candidate)
            continue
        if test_name == "openfhe_security_disclosure":
            disclosure = _require_exact_keys(
                candidate,
                {"test", "security_claim"},
                f"stdout.log security disclosure line {line_number}",
            )
            _require_typed_equal(
                disclosure["security_claim"],
                "none",
                f"stdout.log security disclosure line {line_number}.security_claim",
            )
            continue
        raise ValidationError(
            f"stdout.log line {line_number} contains unexpected JSON evidence "
            f"namespace {test_name!r}"
        )
    if len(relevant) != 13:
        raise ValidationError("stdout must contain exactly 12 layer records then one summary")
    layers = relevant[:12]
    summary = relevant[12]
    if any(record.get("test") != "openfhe_encoder_12_layer_layer" for record in layers):
        raise ValidationError("stdout M5 layer records are not ordered before the summary")
    if summary.get("test") != "openfhe_encoder_12_layer":
        raise ValidationError("stdout M5 final record is missing or out of order")
    return layers, summary


def _metadata_used_level(value: dict[str, Any], label: str) -> int:
    used = value["level"] + value["noise_scale_degree"] - 1
    depth = EXPECTED_PROFILE_FIELDS["multiplicative_depth"]
    if used < 0 or used >= depth:
        raise ValidationError(
            f"{label} used level {used} is outside the depth-{depth} schedule"
        )
    expected_remaining = depth - used
    if value["remaining_levels"] != expected_remaining:
        raise ValidationError(
            f"{label}.remaining_levels drifted: expected={expected_remaining} "
            f"from used level {used}, actual={value['remaining_levels']!r}"
        )
    return used


def _require_metadata(value: Any, expected: dict[str, Any], label: str) -> None:
    record = _require_exact_keys(value, set(expected), label)
    for key in (
        "level",
        "noise_scale_degree",
        "remaining_levels",
        "expected_scale_bits",
        "ciphertext_count",
    ):
        if not isinstance(record[key], int) or isinstance(record[key], bool):
            raise ValidationError(f"{label}.{key} must be an integer")
    _metadata_used_level(record, label)
    for key, expected_value in expected.items():
        if key == "scale_bits":
            target_scale = float(expected["expected_scale_bits"])
            actual_scale = _require_finite(
                record[key],
                f"{label}.scale_bits",
                target_scale - 1e-3,
                target_scale + 1e-3,
            )
            if abs(actual_scale - target_scale) > 1e-3:
                raise ValidationError(
                    f"{label}.scale_bits differs from expected_scale_bits "
                    "by more than 1e-3"
                )
            continue
        _require_typed_equal(record[key], expected_value, f"{label}.{key}")


def _require_used_level_delta(
    output: dict[str, Any],
    anchor: dict[str, Any],
    expected_delta: int,
    label: str,
) -> None:
    actual_delta = _metadata_used_level(
        output,
        f"{label} output",
    ) - _metadata_used_level(anchor, f"{label} anchor")
    if actual_delta != expected_delta:
        raise ValidationError(
            f"{label} used-level delta drifted: expected={expected_delta} "
            f"actual={actual_delta}"
        )


def _require_polynomial_checkpoint_metadata(value: Any, label: str) -> None:
    if POLYNOMIAL_CHECKPOINT_METADATA is None:
        raise ValidationError(
            "M5 polynomial checkpoint metadata is not sealed by the two-layer seam run"
        )
    _require_metadata(value, POLYNOMIAL_CHECKPOINT_METADATA, label)


def _require_layer0_input_metadata(value: Any, label: str) -> None:
    if LAYER0_INPUT_METADATA is None:
        raise ValidationError(
            "M5 layer-0 input metadata is not sealed by the crypto preflight"
        )
    _require_metadata(value, LAYER0_INPUT_METADATA, label)


def _require_crypto_preflight_record(value: Any, label: str) -> dict[str, Any]:
    record = _require_exact_keys(value, CRYPTO_PREFLIGHT_KEYS, label)
    expected_scalars = {
        "test": "openfhe_encoder_12_layer_crypto_preflight",
        "profile": "paper_compat",
        "security_claim": "none",
        "server_private_key_present": False,
        "server_decryptions": 0,
        "server_plaintext_activations": False,
        "additive_he_operations": 0,
        "passed": True,
    }
    for key, expected in expected_scalars.items():
        _require_typed_equal(record[key], expected, f"{label}.{key}")
    _require_layer0_input_metadata(
        record["input_metadata"],
        f"{label}.input_metadata",
    )
    return record


def _require_handoff_input_metadata(value: Any, label: str) -> None:
    if LAYER_HANDOFF_INPUT_METADATA is None:
        raise ValidationError(
            "M5 layer-handoff input metadata is not sealed by the two-layer seam run"
        )
    _require_metadata(value, LAYER_HANDOFF_INPUT_METADATA, label)


def _require_quality(value: Any, label: str) -> dict[str, Any]:
    quality = _require_exact_keys(
        value,
        {"relative_l2", "cosine", "max_absolute"},
        label,
    )
    _require_finite(
        quality["relative_l2"],
        f"{label}.relative_l2",
        0.0,
        THRESHOLDS["per_layer_relative_l2_max"],
    )
    _require_finite(
        quality["cosine"],
        f"{label}.cosine",
        THRESHOLDS["per_layer_cosine_min"],
        1.000000000001,
    )
    _require_finite(quality["max_absolute"], f"{label}.max_absolute", 0.0)
    return quality


def _require_counts(value: Any, expected: dict[str, int], label: str) -> None:
    counts = _require_exact_keys(value, set(OPERATION_COUNT_KEYS), label)
    for key in OPERATION_COUNT_KEYS:
        count = counts[key]
        if not isinstance(count, int) or isinstance(count, bool) or count != expected[key]:
            raise ValidationError(
                f"{label}.{key} must be {expected[key]}, got {count!r}"
            )


def _cumulative_counts(layer_id: int) -> dict[str, int]:
    layer_multiplier = layer_id + 1
    refresh_multiplier = min(layer_id + 1, 11)
    return {
        key: LAYER_COUNTS[key] * layer_multiplier
        + REFRESH_COUNTS[key] * refresh_multiplier
        for key in OPERATION_COUNT_KEYS
    }


def _verify_range_evidence(value: Any, label: str) -> None:
    ranges = _require_exact_keys(value, set(POLYNOMIAL_INTERVALS), label)
    for name, declared in POLYNOMIAL_INTERVALS.items():
        observed = ranges[name]
        if not isinstance(observed, list) or len(observed) != 2:
            raise ValidationError(f"{label}.{name} must be a [minimum,maximum] array")
        minimum = _require_finite(observed[0], f"{label}.{name}[0]")
        maximum = _require_finite(observed[1], f"{label}.{name}[1]")
        if minimum > maximum:
            raise ValidationError(f"{label}.{name} minimum exceeds maximum")
        if minimum < declared[0] or maximum > declared[1]:
            raise ValidationError(
                f"{label}.{name} escaped pre-registered interval {declared}"
            )


def _verify_inactive_polynomial_sentinel_ranges(value: Any, label: str) -> None:
    ranges = _require_exact_keys(
        value,
        set(INACTIVE_POLYNOMIAL_SENTINEL_INTERVALS),
        label,
    )
    for name, declared in INACTIVE_POLYNOMIAL_SENTINEL_INTERVALS.items():
        observed = ranges[name]
        if not isinstance(observed, list) or len(observed) != 2:
            raise ValidationError(f"{label}.{name} must be a [minimum,maximum] array")
        minimum = _require_finite(observed[0], f"{label}.{name}[0]")
        maximum = _require_finite(observed[1], f"{label}.{name}[1]")
        if minimum > maximum:
            raise ValidationError(f"{label}.{name} minimum exceeds maximum")
        if minimum < declared[0] or maximum > declared[1]:
            raise ValidationError(
                f"{label}.{name} escaped pre-registered interval {declared}"
            )


def _verify_layer_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    manifest_layers: list[dict[str, Any]] = []
    for layer_id, record in enumerate(records):
        label = f"stdout layer record {layer_id}"
        regime = "layer_0" if layer_id == 0 else "layers_1_to_11"
        _require_exact_keys(record, LAYER_STDOUT_KEYS, label)
        expected_scalars = {
            "test": "openfhe_encoder_12_layer_layer",
            "profile": "paper_compat",
            "security_claim": "none",
            "layer_id": layer_id,
            "weights_layer_id": layer_id,
            "chain_input_source": (
                "client_encrypted_trace_input"
                if layer_id == 0
                else "previous_ciphertext_output_after_refresh"
            ),
            "handoff_refresh_performed": layer_id < 11,
            "range_validation_owner": "client",
            "checkpoint_decryption_owner": "client",
            "server_decryptions": 0,
            "server_plaintext_activations": False,
            "finite": True,
            "range_status": "passed",
            "inactive_sentinel_range_status": "passed",
            "inactive_zero_checkpoint_count": 28,
        }
        for key, expected in expected_scalars.items():
            _require_typed_equal(record[key], expected, f"{label}.{key}")
        if layer_id == 0:
            _require_layer0_input_metadata(
                record["input_metadata"], f"{label}.input_metadata"
            )
        else:
            _require_handoff_input_metadata(
                record["input_metadata"], f"{label}.input_metadata"
            )
        _require_metadata(
            record["raw_output_metadata"],
            RAW_OUTPUT_METADATA,
            f"{label}.raw_output_metadata",
        )
        for field in (
            "softmax_denominator_metadata",
            "ln1_variance_metadata",
            "ln2_variance_metadata",
        ):
            _require_polynomial_checkpoint_metadata(record[field], f"{label}.{field}")
        _require_metadata(
            record["attention_output_metadata"],
            (
                LAYER0_ATTENTION_OUTPUT_METADATA
                if layer_id == 0
                else POST_REFRESH_ATTENTION_OUTPUT_METADATA
            ),
            f"{label}.attention_output_metadata",
        )
        _require_metadata(
            record["ln1_output_metadata"],
            (
                LAYER0_LN1_OUTPUT_METADATA
                if layer_id == 0
                else POST_REFRESH_LN1_OUTPUT_METADATA
            ),
            f"{label}.ln1_output_metadata",
        )
        _require_metadata(
            record["ffn_output_metadata"],
            (
                LAYER0_FFN_OUTPUT_METADATA
                if layer_id == 0
                else POST_REFRESH_FFN_OUTPUT_METADATA
            ),
            f"{label}.ffn_output_metadata",
        )
        _require_used_level_delta(
            record["attention_output_metadata"],
            record["input_metadata"],
            RELATIVE_USED_LEVEL_DELTAS[
                "attention_output_from_layer_input"
            ][regime],
            f"{label}.attention_output_from_layer_input",
        )
        _require_used_level_delta(
            record["ln1_output_metadata"],
            record["ln1_variance_metadata"],
            RELATIVE_USED_LEVEL_DELTAS[
                "ln1_output_from_ln1_variance"
            ][regime],
            f"{label}.ln1_output_from_ln1_variance",
        )
        _require_used_level_delta(
            record["ffn_output_metadata"],
            record["ln1_output_metadata"],
            RELATIVE_USED_LEVEL_DELTAS["ffn_output_from_ln1_output"],
            f"{label}.ffn_output_from_ln1_output",
        )
        _require_used_level_delta(
            record["raw_output_metadata"],
            record["ln2_variance_metadata"],
            RELATIVE_USED_LEVEL_DELTAS["raw_output_from_ln2_variance"],
            f"{label}.raw_output_from_ln2_variance",
        )
        for field in ("input_quality", "output_quality", "exact_trace_diagnostic"):
            _require_quality(record[field], f"{label}.{field}")
        _require_finite(
            record["inactive_max_abs"],
            f"{label}.inactive_max_abs",
            0.0,
            THRESHOLDS["inactive_max_abs"],
        )
        _require_finite(
            record["inactive_sentinel_max_error"],
            f"{label}.inactive_sentinel_max_error",
            0.0,
        )
        _verify_inactive_polynomial_sentinel_ranges(
            record["inactive_polynomial_sentinel_ranges"],
            f"{label}.inactive_polynomial_sentinel_ranges",
        )
        _verify_range_evidence(
            record["encrypted_polynomial_input_ranges"],
            f"{label}.encrypted_polynomial_input_ranges",
        )
        _require_counts(
            record["refresh_operation_counts"],
            REFRESH_COUNTS if layer_id < 11 else ZERO_COUNTS,
            f"{label}.refresh_operation_counts",
        )
        _require_counts(
            record["layer_operation_counts"],
            LAYER_COUNTS,
            f"{label}.layer_operation_counts",
        )
        _require_counts(
            record["cumulative_operation_counts"],
            _cumulative_counts(layer_id),
            f"{label}.cumulative_operation_counts",
        )
        manifest_layers.append({key: record[key] for key in LAYER_METRIC_KEYS})
    return manifest_layers


def _verify_summary(
    summary: dict[str, Any], layers: list[dict[str, Any]]
) -> dict[str, Any]:
    _require_exact_keys(summary, SUMMARY_STDOUT_KEYS, "stdout M5 summary")
    expected_scalars = {
        "test": "openfhe_encoder_12_layer",
        "profile": "paper_compat",
        "security_claim": "none",
        "parameter_sha256": PROFILE_SHA256,
        "execution_mode": "server-only",
        "claim_scope": "m5_12_layer_correctness",
        "encoder_layers": 12,
        "chain_mode": "ciphertext_output_to_next_input",
        "client_encrypt_calls": 1,
        "encrypted_input_ciphertexts": 5,
        "plaintext_activation_resets": 0,
        "server_layer_evaluations": 12,
        "inter_layer_refreshes": 11,
        "checkpoint_decryption_owner": "client",
        "final_decryption_owner": "client",
        "server_private_key_present": False,
        "server_decryptions": 0,
        "server_plaintext_activations": False,
        "approximation_range_status": "all_client_validated",
        "multiplicative_depth": 47,
        "max_observed_level": 45,
        "max_polynomial_depth": 10,
        "inactive_sentinel_range_status": "all_client_validated",
        "timing_claim": False,
        "latency_kind": "non_benchmark_diagnostic",
        "finite": True,
        "passed": True,
    }
    for key, expected in expected_scalars.items():
        _require_typed_equal(summary[key], expected, f"stdout M5 summary.{key}")
    for key in (
        "fixture_load_oracle_ms",
        "setup_keygen_ms",
        "client_encrypt_ms",
        "client_checkpoint_validate_ms",
    ):
        _require_finite(summary[key], f"stdout M5 summary.{key}", 0.0)
    _require_finite(
        summary["server_online_diagnostic_ms"],
        "stdout M5 summary.server_online_diagnostic_ms",
        1e-9,
    )
    _require_finite(
        summary["relative_l2"],
        "stdout M5 summary.relative_l2",
        0.0,
        THRESHOLDS["final_relative_l2_max"],
    )
    _require_finite(
        summary["cosine"],
        "stdout M5 summary.cosine",
        THRESHOLDS["final_cosine_min"],
        1.000000000001,
    )
    _require_finite(summary["max_absolute"], "stdout M5 summary.max_absolute", 0.0)
    _require_finite(
        summary["inactive_max_abs"],
        "stdout M5 summary.inactive_max_abs",
        0.0,
        THRESHOLDS["inactive_max_abs"],
    )
    _require_finite(
        summary["inactive_sentinel_max_error"],
        "stdout M5 summary.inactive_sentinel_max_error",
        0.0,
    )
    peak_rss_bytes = summary["peak_rss_bytes"]
    if (
        not isinstance(peak_rss_bytes, int)
        or isinstance(peak_rss_bytes, bool)
        or peak_rss_bytes <= 0
    ):
        raise ValidationError("stdout M5 summary.peak_rss_bytes must be positive")
    _require_metadata(summary["final_metadata"], RAW_OUTPUT_METADATA, "stdout final_metadata")
    _require_counts(summary["operation_counts"], TOTAL_COUNTS, "stdout operation_counts")

    final_layer = layers[-1]
    for summary_key, quality_key in (
        ("relative_l2", "relative_l2"),
        ("cosine", "cosine"),
        ("max_absolute", "max_absolute"),
    ):
        if not math.isclose(
            float(summary[summary_key]),
            float(final_layer["output_quality"][quality_key]),
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            raise ValidationError(
                f"stdout final {summary_key} differs from layer-11 output_quality"
            )
    for field in ("inactive_max_abs", "inactive_sentinel_max_error"):
        expected_maximum = max(float(layer[field]) for layer in layers)
        if not math.isclose(
            float(summary[field]),
            expected_maximum,
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            raise ValidationError(
                f"stdout final {field} differs from the 12-layer maximum"
            )
    if not _json_equal(summary["final_metadata"], final_layer["raw_output_metadata"]):
        raise ValidationError("stdout final metadata differs from layer-11 raw output")
    if not _json_equal(summary["operation_counts"], final_layer["cumulative_operation_counts"]):
        raise ValidationError("stdout final counts differ from layer-11 cumulative counts")
    return summary


def _close(actual: Any, expected: Any, label: str) -> None:
    try:
        actual_number = float(actual)
        expected_number = float(expected)
    except (TypeError, ValueError) as error:
        raise ValidationError(f"{label} is not numeric") from error
    if not math.isfinite(actual_number) or not math.isclose(
        actual_number,
        expected_number,
        rel_tol=1e-12,
        abs_tol=1e-15,
    ):
        raise ValidationError(f"{label} differs: expected={expected!r} actual={actual!r}")


def _csv_integer(value: Any, expected: int, label: str) -> int:
    if not isinstance(value, str) or re.fullmatch(r"0|[1-9][0-9]*", value) is None:
        raise ValidationError(f"{label} must be a canonical nonnegative decimal integer")
    actual = int(value)
    if actual != expected:
        raise ValidationError(f"{label} must be exactly {expected}, got {actual}")
    return actual


def _read_metrics_csv(
    path: Path,
    summary: dict[str, Any],
    layers: list[dict[str, Any]],
) -> tuple[float, int, dict[str, float]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != CSV_FIELDS:
                raise ValidationError(
                    "metrics.csv header differs from the frozen M5 field order: "
                    f"expected={CSV_FIELDS!r} actual={tuple(reader.fieldnames or ())!r}"
                )
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as error:
        raise ValidationError(f"cannot read metrics.csv: {error}") from error
    if len(rows) != 1:
        raise ValidationError("metrics.csv must contain exactly one correctness row")
    row = rows[0]
    if set(row) != set(CSV_FIELDS) or None in row:
        raise ValidationError("metrics.csv row columns differ from the frozen header")
    _csv_integer(row["run"], 1, "metrics.csv run")
    _csv_integer(row["exit_code"], 0, "metrics.csv exit_code")
    expected_checkpoint_hash = checkpoint_metadata_sha256(layers)
    if row["checkpoint_metadata_sha256"] != expected_checkpoint_hash:
        raise ValidationError(
            "metrics.csv checkpoint_metadata_sha256 differs from stdout"
        )
    try:
        elapsed = float(row["elapsed_seconds"])
    except (TypeError, ValueError) as error:
        raise ValidationError(f"metrics.csv elapsed_seconds is invalid: {error}") from error
    peak_text = row["peak_rss_kib"]
    if (
        not isinstance(peak_text, str)
        or re.fullmatch(r"[1-9][0-9]*", peak_text) is None
    ):
        raise ValidationError("metrics.csv peak_rss_kib must be a canonical positive integer")
    peak_rss_kib = int(peak_text)
    if not math.isfinite(elapsed) or elapsed <= 0.0 or peak_rss_kib <= 0:
        raise ValidationError("metrics.csv resource fields must be finite and positive")

    phase_fields = {
        key: summary[key]
        for key in (
            "fixture_load_oracle_ms",
            "setup_keygen_ms",
            "client_encrypt_ms",
            "server_online_diagnostic_ms",
            "client_checkpoint_validate_ms",
        )
    }
    for key, expected in phase_fields.items():
        _close(row[key], expected, f"metrics.csv {key}")
    for column, expected in {
        "final_rel_l2": summary["relative_l2"],
        "final_cosine": summary["cosine"],
        "final_max_absolute": summary["max_absolute"],
        "inactive_max_abs": summary["inactive_max_abs"],
        "inactive_sentinel_max_error": summary["inactive_sentinel_max_error"],
        "final_scale_bits": summary["final_metadata"]["scale_bits"],
    }.items():
        _close(row[column], expected, f"metrics.csv {column}")
    discrete = {
        "final_level": summary["final_metadata"]["level"],
        "final_noise_scale_degree": summary["final_metadata"]["noise_scale_degree"],
        "final_remaining_levels": summary["final_metadata"]["remaining_levels"],
        "final_ciphertext_count": summary["final_metadata"]["ciphertext_count"],
        **summary["operation_counts"],
    }
    for column, expected in discrete.items():
        _csv_integer(row[column], expected, f"metrics.csv {column}")
    expected_literals = {
        "timing_claim": "false",
        "latency_kind": "non_benchmark_diagnostic",
        "finite": "true",
        "passed": "true",
    }
    for key, expected in expected_literals.items():
        if row[key] != expected:
            raise ValidationError(
                f"metrics.csv {key} must be {expected!r}, got {row[key]!r}"
            )
    if summary["peak_rss_bytes"] > peak_rss_kib * 1024:
        raise ValidationError("stdout peak RSS exceeds GNU time metrics.csv peak RSS")
    return elapsed, peak_rss_kib, {key: float(value) for key, value in phase_fields.items()}


def _verify_manifest_metrics(
    manifest: dict[str, Any],
    layer_metrics: list[dict[str, Any]],
    summary: dict[str, Any],
    elapsed: float,
    peak_rss_kib: int,
    phases: dict[str, float],
) -> None:
    metrics = manifest["metrics"]
    expected_keys = {
        "repeat_count",
        "successful_repeats",
        "warmup_count",
        "timing_claim",
        "latency_kind",
        "elapsed_seconds",
        "peak_rss_kib",
        "checkpoint_metadata_sha256",
        "phase_latency_ms",
        "layers",
        "final_quality",
        "final_metadata",
        "operation_counts",
        "finite",
    }
    _require_exact_keys(metrics, expected_keys, "metrics")
    expected_scalars = {
        "repeat_count": 1,
        "successful_repeats": 1,
        "warmup_count": 0,
        "timing_claim": False,
        "latency_kind": "non_benchmark_diagnostic",
        "peak_rss_kib": peak_rss_kib,
        "finite": True,
    }
    for key, expected in expected_scalars.items():
        _require_typed_equal(metrics[key], expected, f"metrics.{key}")
    _close(metrics["elapsed_seconds"], elapsed, "metrics.elapsed_seconds")
    expected_checkpoint_hash = checkpoint_metadata_sha256(layer_metrics)
    if metrics["checkpoint_metadata_sha256"] != expected_checkpoint_hash:
        raise ValidationError(
            "metrics.checkpoint_metadata_sha256 differs from stdout"
        )
    for key, expected in phases.items():
        _close(metrics["phase_latency_ms"][key], expected, f"metrics.phase_latency_ms.{key}")
    if not _json_equal(metrics["layers"], layer_metrics):
        raise ValidationError("metrics.layers differs from the 12 stdout layer records")
    expected_final_quality = {
        "relative_l2": summary["relative_l2"],
        "cosine": summary["cosine"],
        "max_absolute": summary["max_absolute"],
        "inactive_max_abs": summary["inactive_max_abs"],
        "inactive_sentinel_max_error": summary["inactive_sentinel_max_error"],
        "inactive_sentinel_range_status": summary[
            "inactive_sentinel_range_status"
        ],
    }
    if not _json_equal(metrics["final_quality"], expected_final_quality):
        raise ValidationError("metrics.final_quality differs from stdout")
    if not _json_equal(metrics["final_metadata"], summary["final_metadata"]):
        raise ValidationError("metrics.final_metadata differs from stdout")
    if not _json_equal(metrics["operation_counts"], summary["operation_counts"]):
        raise ValidationError("metrics.operation_counts differs from stdout")


def _verify_checksum_evidence(
    artifact_root: Path, artifact_records: dict[str, dict[str, Any]]
) -> None:
    try:
        lines = (artifact_root / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise ValidationError(f"cannot read SHA256SUMS: {error}") from error
    entries: dict[str, str] = {}
    for line in lines:
        match = re.fullmatch(r"([0-9a-f]{64})  ([A-Za-z0-9._+-]+)", line)
        if match is None or match.group(2) in entries:
            raise ValidationError(f"invalid or duplicate SHA256SUMS line: {line!r}")
        entries[match.group(2)] = match.group(1)
    if set(entries) != {"stdout.log", "metrics.csv"}:
        raise ValidationError("SHA256SUMS must cover exactly stdout.log and metrics.csv")
    for path, digest in entries.items():
        if artifact_records[path]["sha256"] != digest:
            raise ValidationError(f"SHA256SUMS digest disagrees for {path}")


def _verify_artifacts_and_runtime(
    manifest_path: Path,
    manifest: dict[str, Any],
    repository_root: Path,
) -> None:
    executable = _resolve_confined_file(
        repository_root,
        manifest["workload"]["executable_path"],
        "workload.executable_path",
    )
    _verify_file_record(
        executable,
        {
            "bytes": manifest["workload"]["executable_bytes"],
            "sha256": manifest["workload"]["executable_sha256"],
        },
        "workload executable",
    )

    artifact_root = manifest_path.resolve().parent
    expected_inventory = {
        "manifest.json",
        "stdout.log",
        "metrics.csv",
        "SHA256SUMS",
    }
    try:
        actual_inventory = {entry.name for entry in artifact_root.iterdir()}
    except OSError as error:
        raise ValidationError(f"cannot inspect M5 artifact directory: {error}") from error
    if actual_inventory != expected_inventory:
        raise ValidationError(
            "M5 artifact directory must contain exactly the sealed four-file bundle: "
            f"missing={sorted(expected_inventory - actual_inventory)} "
            f"extra={sorted(actual_inventory - expected_inventory)}"
        )
    required = {
        "stdout.log": ("stdout", "text/plain"),
        "metrics.csv": ("metrics", "text/csv"),
        "SHA256SUMS": ("checksum", "text/plain"),
    }
    if [record["path"] for record in manifest["artifacts"]] != list(required):
        raise ValidationError("artifacts must be ordered stdout.log, metrics.csv, SHA256SUMS")
    records: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(manifest["artifacts"]):
        path = record["path"]
        if path in records:
            raise ValidationError(f"duplicate artifact path: {path}")
        expected_role, expected_media_type = required[path]
        if record["role"] != expected_role or record["media_type"] != expected_media_type:
            raise ValidationError(f"artifact {path} role or media type drifted")
        resolved = _resolve_confined_file(artifact_root, path, f"artifacts[{index}].path")
        if resolved == manifest_path.resolve():
            raise ValidationError("manifest.json cannot hash itself")
        _verify_file_record(resolved, record, f"artifacts[{index}]")
        records[path] = record

    layers, summary = _parse_stdout_records(artifact_root / "stdout.log")
    layer_metrics = _verify_layer_records(layers)
    _verify_summary(summary, layers)
    elapsed, peak_rss_kib, phases = _read_metrics_csv(
        artifact_root / "metrics.csv",
        summary,
        layers,
    )
    _verify_manifest_metrics(
        manifest,
        layer_metrics,
        summary,
        elapsed,
        peak_rss_kib,
        phases,
    )
    _verify_checksum_evidence(artifact_root, records)


def validate_manifest(
    manifest_path: Path,
    manifest: Any,
    schema: dict[str, Any],
    verify_git: bool,
) -> None:
    if (
        isinstance(manifest, dict)
        and manifest.get("schema_version") != M5_SCHEMA_VERSION
    ):
        raise ValidationError("M5 validator accepts only schema_version=4")
    validate_instance(manifest, schema, schema)
    assert isinstance(manifest, dict)
    started = _parse_timestamp(manifest["started_at"], "$.started_at")
    finished = _parse_timestamp(manifest["finished_at"], "$.finished_at")
    if finished < started:
        raise ValidationError("finished_at precedes started_at")
    repository_root = Path(manifest["git"]["repository_root"])
    if repository_root.resolve() != REPO_ROOT.resolve():
        raise ValidationError(f"manifest repository_root must be {REPO_ROOT}")
    resolved_manifest = manifest_path.resolve()
    artifact_root = resolved_manifest.parent
    if resolved_manifest.name != "manifest.json":
        raise ValidationError("M5 manifest filename must be manifest.json")
    if artifact_root.name != manifest["run_id"]:
        raise ValidationError("M5 artifact directory name must equal manifest.run_id")
    if artifact_root.parent.resolve() != OUTPUT_ROOT.resolve():
        raise ValidationError(
            f"M5 artifact directory must be a direct child of {OUTPUT_ROOT.resolve()}"
        )
    if manifest["claim_boundary"] != CLAIM_BOUNDARY:
        raise ValidationError("claim_boundary differs from the frozen M5 scope")
    _verify_inputs_and_contracts(manifest, repository_root)
    _verify_command_transcript(manifest, manifest_path)
    _verify_artifacts_and_runtime(manifest_path, manifest, repository_root)
    _verify_git(manifest, repository_root, verify_git)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--verify-git",
        action="store_true",
        help="verify clean local HEAD, tracking ref, and live remote ref",
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    if arguments.verify_git and arguments.manifest is None:
        raise ValidationError("--verify-git requires --manifest")
    schema = validate_schema(load_json(arguments.schema))
    result: dict[str, Any] = {
        "test": "validate_openfhe_m5_artifact",
        "schema": str(arguments.schema.resolve()),
        "schema_version": M5_SCHEMA_VERSION,
        "schema_valid": True,
    }
    if arguments.manifest is not None:
        manifest = load_json(arguments.manifest)
        validate_manifest(arguments.manifest, manifest, schema, arguments.verify_git)
        result.update(
            {
                "manifest": str(arguments.manifest.resolve()),
                "run_id": manifest["run_id"],
                "manifest_valid": True,
                "git_verified_live": arguments.verify_git,
            }
        )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValidationError, OSError) as error:
        print(f"validate_openfhe_m5_artifact failed: {error}", file=sys.stderr)
        sys.exit(1)
