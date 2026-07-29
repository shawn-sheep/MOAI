#!/usr/bin/env python3
"""Validate the MOAI OpenFHE artifact schema and schema-v2 manifests.

The validator intentionally uses only the Python standard library. With no
``--manifest`` argument it checks that the checked-in schema still describes
the frozen v2 contract. With ``--manifest`` it additionally validates the
manifest structure, canonical effective-profile hash, repository inputs, and
run-directory artifacts. ``--verify-git`` adds live local/remote ref checks.

Schema v2 dispatches two non-interchangeable payloads: the frozen M3 nonlinear
smoke and the M4 server-only encoder layer. M5 and M6 are not accepted until
separate contracts are defined.

Schema-version-1 manifests are immutable legacy evidence. They are not upgraded
or accepted by this validator.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shlex
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCHEMA = REPO_ROOT / "docs" / "openfhe-artifact-schema.json"
SCHEMA_URI = "https://json-schema.org/draft/2020-12/schema"
CANONICALIZATION = "MOAI-json-sort-keys-compact-utf8-v1"
REQUIRED_M3_INPUTS = {
    "config/openfhe_approximations.json",
    "config/moai_trace_channel_scales.json",
    "config/paper_compat.json",
}
M3_POLYNOMIAL_LOCATORS = {
    "gelu": ("gelu", "polynomial"),
    "softmax_exponential": ("softmax", "exponential"),
    "softmax_reciprocal": ("softmax", "reciprocal"),
    "layernorm_inverse_sqrt": ("layernorm", "inverse_sqrt"),
}
M3_RUNTIME_POLYNOMIAL_PREFIXES = {
    "gelu": "gelu",
    "softmax_exponential": "softmax_exp",
    "softmax_reciprocal": "softmax_reciprocal",
    "layernorm_inverse_sqrt": "layernorm_invsqrt",
}
M3_EXPECTED_POLYNOMIALS = {
    "gelu": {
        "degree": 319,
        "interval": {"minimum": -80.0, "maximum": 128.0},
        "required_depth": 10,
        "estimated_multiplications": 33,
        "coefficient_sha256": (
            "35d68b2f56f267f27e8f962c3bd54cddd90892349b77405172f48987864eaaa3"
        ),
    },
    "softmax_exponential": {
        "degree": 27,
        "interval": {"minimum": -16.0, "maximum": 5.0},
        "required_depth": 6,
        "estimated_multiplications": 10,
        "coefficient_sha256": (
            "6eda4377151897e8c4ca4d72f2a918db0b888fc6771f8ee5cf1d950b378de76f"
        ),
    },
    "softmax_reciprocal": {
        "degree": 383,
        "interval": {"minimum": 0.01, "maximum": 80.0},
        "required_depth": 10,
        "estimated_multiplications": 35,
        "coefficient_sha256": (
            "fa97f298751bca97f40eed3b6de949262d1b3971cda57013b55420d5c9fbbeb3"
        ),
    },
    "layernorm_inverse_sqrt": {
        "degree": 159,
        "interval": {"minimum": 0.5, "maximum": 1536.0},
        "required_depth": 9,
        "estimated_multiplications": 23,
        "coefficient_sha256": (
            "28d0ffd36436228da4ee6023e0aad7641be995492a4314375b5a1aafe68465be"
        ),
    },
}
M3_EXPECTED_SOFTMAX_SHIFT = (
    1,
    2,
    "41ecf6ade53f674096c9afc752ccfe99834876ea4768bcf4bf3ea25a8f502432",
)
M3_EXPECTED_THRESHOLDS = {
    "relative_l2_max": 1e-2,
    "cosine_min": 0.999,
    "inactive_max_abs": 1e-6,
    "denominator_post_max_abs": 1e-6,
}
M3_EXPECTED_OPERATION_COUNTS = {
    "rotations": 0,
    "ct_pt_multiplications": 19,
    "ct_ct_multiplications": 10,
    "rescale_operations": 28,
    "bootstraps": 3,
    "bootstrap_iterations": 6,
    "chebyshev_evaluations": 5,
    "estimated_polynomial_multiplications": 111,
    "max_polynomial_depth": 10,
}
REQUIRED_ENCODER_INPUTS = {
    "config/moai_encoder_trace.json",
    "config/moai_trace_channel_scales.json",
    "config/openfhe_approximations.json",
    "config/paper_compat_feature_packed.json",
}
M4_EXPECTED_THRESHOLDS = {
    "relative_l2_max": 1e-2,
    "cosine_min": 0.999,
    "inactive_max_abs": 1e-6,
}
M4_CHECKPOINT_NAMES = (
    "attention_output",
    "self_layernorm_output",
    "ffn_output",
    "encoder_output",
)
M4_MULTIPLICATIVE_DEPTH = 47
M4_MAX_OBSERVED_LEVEL = 45
M4_MAX_POLYNOMIAL_DEPTH = 10
M4_EXPECTED_CHECKPOINTS = (
    {
        "name": "attention_output",
        "level": 40,
        "noise_scale_degree": 2,
        "remaining_levels": 6,
        "scale_bits": 100,
        "ciphertext_count": 5,
        "decryption_owner": "client",
    },
    {
        "name": "self_layernorm_output",
        "level": 32,
        "noise_scale_degree": 2,
        "remaining_levels": 14,
        "scale_bits": 100,
        "ciphertext_count": 5,
        "decryption_owner": "client",
    },
    {
        "name": "ffn_output",
        "level": 45,
        "noise_scale_degree": 2,
        "remaining_levels": 1,
        "scale_bits": 100,
        "ciphertext_count": 5,
        "decryption_owner": "client",
    },
    {
        "name": "encoder_output",
        "level": 29,
        "noise_scale_degree": 2,
        "remaining_levels": 17,
        "scale_bits": 100,
        "ciphertext_count": 5,
        "decryption_owner": "client",
    },
)
M4_CHECKPOINT_METADATA_SHA256 = (
    "c4c1c85e52154215784b9fa93a584d824dde94a644e5af997882486aac01a6db"
)
M4_PHASE_LATENCY_FIELDS = (
    "fixture_load_ms",
    "setup_keygen_ms",
    "client_encrypt_ms",
    "server_online_ms",
    "client_decrypt_validate_ms",
)
M4_TOKENS_PER_BATCH = 5
M4_DIAGNOSTIC_KEYS = {
    "test",
    "profile",
    "security_claim",
    "parameter_sha256",
    "layer",
    "input_level",
    "tokens",
    "hidden_size",
    "intermediate_size",
    "feature_block",
    *M4_PHASE_LATENCY_FIELDS,
    "relative_l2",
    "cosine",
    "max_absolute",
    "exact_trace_relative_l2",
    "exact_trace_cosine",
    "exact_trace_max_absolute",
    "inactive_max_absolute",
    "peak_rss_bytes",
    "rotations",
    "ct_pt_multiplications",
    "ct_ct_multiplications",
    "explicit_rescale_requests",
    "chebyshev_evaluations",
    "estimated_polynomial_multiplications",
    "bootstraps",
    "bootstrap_iterations",
    "final_level",
    "final_remaining_levels",
    "multiplicative_depth",
    "max_observed_level",
    "max_polynomial_depth",
}
ENCODER_TRACE_SHAPE = [5, 768]
ENCODER_TRACE_VALUE_COUNT = 5 * 768
ENCODER_FEATURE_BLOCK_SIZE = 1024
ENCODER_BOOTSTRAP_ITERATIONS_PER_CALL = 2
ENCODER_FEATURE_PROFILE_PATH = "config/paper_compat_feature_packed.json"
ENCODER_FEATURE_PROFILE_LOCATOR = "/effective_profile"
ENCODER_FEATURE_PROFILE_SHA256 = (
    "94f30e628e21f02146ce7ed9820194eabba3820f6e1e17176a31f8c5acf8b0be"
)
ENCODER_PROFILE_FIELDS = {
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
M4_LAYER_ID = 1
M4_INPUT_LEVEL = 29
M4_OUTPUT_LEVEL = 29
M4_REMAINING_LEVELS = 17
ENCODER_BOOTSTRAPS_PER_LAYER = 25
ENCODER_BOOTSTRAP_ITERATIONS_PER_LAYER = 50
ENCODER_ROTATIONS_PER_LAYER = 6300
ENCODER_CT_PT_MULTIPLICATIONS_PER_LAYER = 51865
ENCODER_CT_CT_MULTIPLICATIONS_PER_LAYER = 95
ENCODER_EXPLICIT_RESCALE_REQUESTS_PER_LAYER = 800
ENCODER_CHEBYSHEV_EVALUATIONS_PER_LAYER = 55
ENCODER_ESTIMATED_POLYNOMIAL_MULTIPLICATIONS_PER_LAYER = 1150
M4_WORKLOAD_EXECUTABLE_PATH = "build-openfhe/openfhe_encoder_layer_smoke"
M4_CTEST_PATTERN = (
    "^(moai_trace_contract|openfhe_(server_trust_boundary|profile_contract|"
    "artifact_schema_contract|artifact_validator_contract|"
    "encoder_artifact_runner_contract|evaluation_key_bundle_smoke|"
    "feature_packed_smoke|"
    "feature_packed_attention_smoke|feature_bootstrap_smoke|"
    "encoder_fixture_contract|"
    "encoder_plaintext_oracle_smoke|encoder_trace_contract|"
    "encoder_trace_validator_contract))$"
)
ENCODER_OPERATION_COUNT_KEYS = {
    "rotations",
    "ct_pt_multiplications",
    "ct_ct_multiplications",
    "explicit_rescale_requests",
    "chebyshev_evaluations",
    "estimated_polynomial_multiplications",
    "bootstraps",
    "bootstrap_iterations",
}
M4_EXECUTION_KEYS = {
    "mode",
    "encoder_layers",
    "layer_id",
    "trace_shape",
    "feature_block_size",
    "checkpoint_decryption_owner",
    "server_private_key_present",
    "server_decryptions",
    "server_plaintext_activations",
    "multiplicative_depth",
    "max_observed_level",
    "max_polynomial_depth",
    "required_checkpoints",
    "quality_reference",
}
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
    "minProperties",
    "maxProperties",
    "oneOf",
}


class ValidationError(RuntimeError):
    """Raised for a malformed schema, manifest, or evidence file."""


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
        text = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ValidationError(f"value cannot be canonicalized as JSON: {error}") from error
    return text.encode("utf-8")


def checkpoint_metadata_sha256(checkpoints: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(checkpoints)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _json_equal(left: Any, right: Any) -> bool:
    return canonical_json_bytes(left) == canonical_json_bytes(right)


def _resolve_ref(root_schema: dict[str, Any], reference: str) -> dict[str, Any]:
    if not reference.startswith("#/"):
        raise ValidationError(f"only local JSON Schema references are supported: {reference}")
    current: Any = root_schema
    for encoded_part in reference[2:].split("/"):
        part = encoded_part.replace("~1", "/").replace("~0", "~")
        if not isinstance(current, dict) or part not in current:
            raise ValidationError(f"unresolvable JSON Schema reference: {reference}")
        current = current[part]
    if not isinstance(current, dict):
        raise ValidationError(f"JSON Schema reference is not an object: {reference}")
    return current


def _type_matches(instance: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(instance, dict)
    if expected == "array":
        return isinstance(instance, list)
    if expected == "string":
        return isinstance(instance, str)
    if expected == "boolean":
        return isinstance(instance, bool)
    if expected == "integer":
        return isinstance(instance, int) and not isinstance(instance, bool)
    if expected == "number":
        return (
            isinstance(instance, (int, float))
            and not isinstance(instance, bool)
            and math.isfinite(instance)
        )
    if expected == "null":
        return instance is None
    raise ValidationError(f"unsupported JSON Schema type: {expected}")


def _parse_timestamp(value: str, location: str) -> datetime:
    if not (value.endswith("Z") or re.search(r"[+-][0-9]{2}:[0-9]{2}$", value)):
        raise ValidationError(f"{location}: date-time must include a timezone")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValidationError(f"{location}: invalid RFC 3339 date-time: {value}") from error
    if parsed.tzinfo is None:
        raise ValidationError(f"{location}: date-time must include a timezone")
    return parsed


def validate_instance(
    instance: Any,
    schema: dict[str, Any],
    root_schema: dict[str, Any],
    location: str = "$",
) -> None:
    if "$ref" in schema:
        validate_instance(
            instance,
            _resolve_ref(root_schema, schema["$ref"]),
            root_schema,
            location,
        )
        return

    if "oneOf" in schema:
        alternatives = schema["oneOf"]
        matches = 0
        failures: list[str] = []
        for alternative in alternatives:
            try:
                validate_instance(instance, alternative, root_schema, location)
                matches += 1
            except ValidationError as error:
                failures.append(str(error))
        if matches != 1:
            detail = failures[0] if failures else "multiple alternatives matched"
            raise ValidationError(
                f"{location}: expected exactly one schema alternative, matched={matches}; "
                f"first_failure={detail}"
            )

    if "type" in schema and not _type_matches(instance, schema["type"]):
        raise ValidationError(
            f"{location}: expected {schema['type']}, got {type(instance).__name__}"
        )
    if "const" in schema and not _json_equal(instance, schema["const"]):
        raise ValidationError(f"{location}: expected constant {schema['const']!r}")
    if "enum" in schema and not any(_json_equal(instance, item) for item in schema["enum"]):
        raise ValidationError(f"{location}: value {instance!r} is not in the allowed enum")

    if isinstance(instance, dict):
        minimum = schema.get("minProperties")
        maximum = schema.get("maxProperties")
        if minimum is not None and len(instance) < minimum:
            raise ValidationError(f"{location}: expected at least {minimum} properties")
        if maximum is not None and len(instance) > maximum:
            raise ValidationError(f"{location}: expected at most {maximum} properties")
        for required in schema.get("required", []):
            if required not in instance:
                raise ValidationError(f"{location}: missing required property {required!r}")
        properties = schema.get("properties", {})
        additional = schema.get("additionalProperties", True)
        for key, value in instance.items():
            child_location = f"{location}.{key}"
            if key in properties:
                validate_instance(value, properties[key], root_schema, child_location)
            elif additional is False:
                raise ValidationError(f"{location}: unexpected property {key!r}")
            elif isinstance(additional, dict):
                validate_instance(value, additional, root_schema, child_location)

    if isinstance(instance, list):
        minimum = schema.get("minItems")
        maximum = schema.get("maxItems")
        if minimum is not None and len(instance) < minimum:
            raise ValidationError(f"{location}: expected at least {minimum} items")
        if maximum is not None and len(instance) > maximum:
            raise ValidationError(f"{location}: expected at most {maximum} items")
        if schema.get("uniqueItems", False):
            serialized = [canonical_json_bytes(item) for item in instance]
            if len(serialized) != len(set(serialized)):
                raise ValidationError(f"{location}: array items must be unique")
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, value in enumerate(instance):
                validate_instance(value, item_schema, root_schema, f"{location}[{index}]")

    if isinstance(instance, str):
        if "minLength" in schema and len(instance) < schema["minLength"]:
            raise ValidationError(f"{location}: string is shorter than minLength")
        if "maxLength" in schema and len(instance) > schema["maxLength"]:
            raise ValidationError(f"{location}: string is longer than maxLength")
        if "pattern" in schema and re.search(schema["pattern"], instance) is None:
            raise ValidationError(f"{location}: string does not match {schema['pattern']!r}")
        if schema.get("format") == "date-time":
            _parse_timestamp(instance, location)

    if isinstance(instance, (int, float)) and not isinstance(instance, bool):
        if not math.isfinite(instance):
            raise ValidationError(f"{location}: non-finite number is forbidden")
        if "minimum" in schema and instance < schema["minimum"]:
            raise ValidationError(f"{location}: value is below minimum {schema['minimum']}")
        if "maximum" in schema and instance > schema["maximum"]:
            raise ValidationError(f"{location}: value is above maximum {schema['maximum']}")


def _walk_schema(node: Any, root_schema: dict[str, Any], location: str = "$schema") -> None:
    if not isinstance(node, dict):
        raise ValidationError(f"{location}: every schema node must be an object")
    unknown = set(node) - SUPPORTED_SCHEMA_KEYWORDS
    if unknown:
        raise ValidationError(f"{location}: unsupported schema keywords: {sorted(unknown)}")
    if "$ref" in node:
        if not isinstance(node["$ref"], str):
            raise ValidationError(f"{location}.$ref must be a string")
        _resolve_ref(root_schema, node["$ref"])
    if "type" in node:
        if not isinstance(node["type"], str):
            raise ValidationError(f"{location}.type must be a string")
        _type_matches(None, node["type"])
    if "enum" in node:
        enum = node["enum"]
        if not isinstance(enum, list) or not enum:
            raise ValidationError(f"{location}.enum must be a non-empty array")
        serialized = [canonical_json_bytes(item) for item in enum]
        if len(serialized) != len(set(serialized)):
            raise ValidationError(f"{location}.enum values must be unique")
    if "pattern" in node:
        if not isinstance(node["pattern"], str):
            raise ValidationError(f"{location}.pattern must be a string")
        try:
            re.compile(node["pattern"])
        except re.error as error:
            raise ValidationError(f"{location}.pattern is invalid: {error}") from error
    if "format" in node and node["format"] != "date-time":
        raise ValidationError(f"{location}.format is unsupported: {node['format']!r}")
    for keyword in (
        "minLength",
        "maxLength",
        "minItems",
        "maxItems",
        "minProperties",
        "maxProperties",
    ):
        if keyword in node and (
            not isinstance(node[keyword], int)
            or isinstance(node[keyword], bool)
            or node[keyword] < 0
        ):
            raise ValidationError(f"{location}.{keyword} must be a non-negative integer")
    for keyword in ("minimum", "maximum"):
        if keyword in node and (
            not isinstance(node[keyword], (int, float))
            or isinstance(node[keyword], bool)
            or not math.isfinite(node[keyword])
        ):
            raise ValidationError(f"{location}.{keyword} must be a finite number")
    if "uniqueItems" in node and not isinstance(node["uniqueItems"], bool):
        raise ValidationError(f"{location}.uniqueItems must be boolean")
    properties = node.get("properties", {})
    if not isinstance(properties, dict):
        raise ValidationError(f"{location}.properties must be an object")
    required = node.get("required", [])
    if not isinstance(required, list) or not all(isinstance(item, str) for item in required):
        raise ValidationError(f"{location}.required must be an array of strings")
    if len(required) != len(set(required)):
        raise ValidationError(f"{location}.required entries must be unique")
    missing_definitions = set(required) - set(properties)
    if missing_definitions:
        raise ValidationError(
            f"{location}: required properties have no schema: {sorted(missing_definitions)}"
        )
    for key, child in properties.items():
        _walk_schema(child, root_schema, f"{location}.properties.{key}")
    definitions = node.get("$defs", {})
    if not isinstance(definitions, dict):
        raise ValidationError(f"{location}.$defs must be an object")
    for key, child in definitions.items():
        _walk_schema(child, root_schema, f"{location}.$defs.{key}")
    alternatives = node.get("oneOf", [])
    if not isinstance(alternatives, list) or (
        "oneOf" in node and len(alternatives) < 2
    ):
        raise ValidationError(f"{location}.oneOf must contain at least two schemas")
    for index, child in enumerate(alternatives):
        _walk_schema(child, root_schema, f"{location}.oneOf[{index}]")
    for keyword in ("items", "additionalProperties"):
        child = node.get(keyword)
        if isinstance(child, dict):
            _walk_schema(child, root_schema, f"{location}.{keyword}")
        elif keyword == "items" and child is not None:
            raise ValidationError(f"{location}.items must be an object")
        elif keyword == "additionalProperties" and not isinstance(child, (bool, type(None))):
            raise ValidationError(f"{location}.additionalProperties must be boolean or object")


def validate_schema(schema: Any) -> dict[str, Any]:
    if not isinstance(schema, dict):
        raise ValidationError("schema root must be an object")
    if schema.get("$schema") != SCHEMA_URI:
        raise ValidationError(f"schema must declare {SCHEMA_URI}")
    if schema.get("type") != "object" or schema.get("additionalProperties") is not False:
        raise ValidationError("schema root must be a closed object")
    if set(schema.get("required", [])) != REQUIRED_TOP_LEVEL:
        raise ValidationError("schema top-level required fields drifted")
    version_schema = schema.get("properties", {}).get("schema_version", {})
    if version_schema.get("const") != 2:
        raise ValidationError("schema_version must be frozen at 2")
    milestone_schema = schema.get("properties", {}).get("milestone", {})
    if milestone_schema.get("enum") != ["M3", "M4"]:
        raise ValidationError("schema-v2 milestone dispatch must be exactly M3/M4")
    if not isinstance(schema.get("oneOf"), list) or len(schema["oneOf"]) != 2:
        raise ValidationError("schema root must contain two milestone alternatives")
    _walk_schema(schema, schema)
    return schema


def _resolve_confined_file(base: Path, relative_path: str, label: str) -> Path:
    raw_path = Path(relative_path)
    if raw_path.is_absolute() or "\\" in relative_path or ".." in raw_path.parts:
        raise ValidationError(f"{label}: path must be normalized and relative: {relative_path}")
    resolved_base = base.resolve()
    resolved = (resolved_base / raw_path).resolve()
    try:
        resolved.relative_to(resolved_base)
    except ValueError as error:
        raise ValidationError(
            f"{label}: path escapes its evidence root: {relative_path}"
        ) from error
    if not resolved.is_file():
        raise ValidationError(f"{label}: file does not exist: {resolved}")
    return resolved


def _verify_file_record(path: Path, record: dict[str, Any], label: str) -> None:
    actual_bytes = path.stat().st_size
    if actual_bytes != record["bytes"]:
        raise ValidationError(
            f"{label}: byte count mismatch for {path}: expected={record['bytes']} "
            f"actual={actual_bytes}"
        )
    actual_hash = sha256_file(path)
    if actual_hash != record["sha256"]:
        raise ValidationError(
            f"{label}: SHA-256 mismatch for {path}: expected={record['sha256']} "
            f"actual={actual_hash}"
        )


def _decode_json_pointer(document: Any, pointer: str) -> Any:
    current = document
    for encoded_part in pointer[1:].split("/"):
        part = encoded_part.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif isinstance(current, list) and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
        else:
            raise ValidationError(f"effective_profile_locator does not resolve: {pointer}")
    return current


def _verify_profile(manifest: dict[str, Any], repository_root: Path) -> None:
    profile = manifest["profile"]
    config_path = _resolve_confined_file(
        repository_root, profile["source_config_path"], "profile.source_config_path"
    )
    actual_bytes = config_path.stat().st_size
    if actual_bytes != profile["source_config_bytes"]:
        raise ValidationError(
            "profile source-config byte count mismatch: "
            f"expected={profile['source_config_bytes']} actual={actual_bytes}"
        )
    actual_source_hash = sha256_file(config_path)
    if actual_source_hash != profile["source_config_sha256"]:
        raise ValidationError(
            "profile source-config SHA-256 mismatch: "
            f"expected={profile['source_config_sha256']} actual={actual_source_hash}"
        )
    if profile["canonicalization"] != CANONICALIZATION:
        raise ValidationError("unsupported effective-profile canonicalization")
    effective_bytes = canonical_json_bytes(profile["effective_profile_payload"])
    effective_hash = hashlib.sha256(effective_bytes).hexdigest()
    if effective_hash != profile["effective_profile_sha256"]:
        raise ValidationError(
            "effective-profile SHA-256 mismatch: "
            f"expected={profile['effective_profile_sha256']} actual={effective_hash}"
        )

    config = load_json(config_path)
    located_payload = _decode_json_pointer(config, profile["effective_profile_locator"])
    if not _json_equal(located_payload, profile["effective_profile_payload"]):
        raise ValidationError(
            "effective_profile_payload does not match the source config at "
            f"{profile['effective_profile_locator']}"
        )
    parent_pointer, _, _ = profile["effective_profile_locator"].rpartition("/")
    located_parent = _decode_json_pointer(config, parent_pointer) if parent_pointer else config
    if isinstance(located_parent, dict) and "effective_profile_sha256" in located_parent:
        if located_parent["effective_profile_sha256"] != effective_hash:
            raise ValidationError(
                "source config effective_profile_sha256 does not match its canonical payload"
            )
    if isinstance(config, dict):
        if config.get("profile_id") != profile["id"]:
            raise ValidationError("profile id does not match source config")
        if config.get("security_claim") != profile["security_claim"]:
            raise ValidationError("security claim does not match source config")


def _run_git(repository_root: Path, arguments: list[str]) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise ValidationError(f"git {' '.join(arguments)} failed: {detail}")
    return completed.stdout.strip()


def _verify_git(manifest: dict[str, Any], repository_root: Path, live: bool) -> None:
    git_record = manifest["git"]
    remote_fields = ("remote_name", "remote_ref", "remote_commit")
    present = [field in git_record for field in remote_fields]
    if any(present) and not all(present):
        raise ValidationError("git remote_name, remote_ref, and remote_commit are all-or-none")
    if all(present) and git_record["local_commit"] != git_record["remote_commit"]:
        raise ValidationError("manifest local_commit and remote_commit differ")

    verdict = manifest["verdict"]
    passed = manifest["gate"]["passed"]
    checks = manifest["gate"]["checks"]
    if verdict == "GO":
        if not live:
            raise ValidationError("GO verdict requires --verify-git live verification")
        if not passed:
            raise ValidationError("GO verdict requires gate.passed=true")
        if not git_record["clean"]:
            raise ValidationError("GO verdict requires git.clean=true")
        if not all(present):
            raise ValidationError("GO verdict requires verified remote git fields")
        failed_commands = [
            command["command"]
            for command in manifest["commands"]
            if command["exit_code"] != 0
        ]
        if failed_commands:
            raise ValidationError(
                f"GO verdict contains nonzero command exits: {failed_commands}"
            )
        incomplete = {key: value for key, value in checks.items() if value != "PASS"}
        if incomplete:
            raise ValidationError(f"GO verdict contains incomplete gate checks: {incomplete}")
    elif verdict in {"NO-GO", "BLOCKED"} and passed:
        raise ValidationError(f"{verdict} verdict requires gate.passed=false")

    if not live:
        return
    if not (repository_root / ".git").exists():
        raise ValidationError(f"not a Git worktree: {repository_root}")
    local_head = _run_git(repository_root, ["rev-parse", "HEAD"])
    if local_head != git_record["local_commit"]:
        raise ValidationError(
            f"live local HEAD differs: manifest={git_record['local_commit']} actual={local_head}"
        )
    branch = _run_git(repository_root, ["branch", "--show-current"])
    if branch != git_record["branch"]:
        raise ValidationError(
            f"live branch differs: manifest={git_record['branch']} actual={branch}"
        )
    clean = not _run_git(repository_root, ["status", "--porcelain=v1"])
    if clean != git_record["clean"]:
        raise ValidationError(
            f"live clean state differs: manifest={git_record['clean']} actual={clean}"
        )
    if all(present):
        remote_output = _run_git(
            repository_root,
            ["ls-remote", "--exit-code", git_record["remote_name"], git_record["remote_ref"]],
        )
        remote_lines = [line.split() for line in remote_output.splitlines() if line.strip()]
        remote_shas = [parts[0] for parts in remote_lines if len(parts) >= 2]
        if remote_shas != [git_record["remote_commit"]]:
            raise ValidationError(
                "live remote ref differs: "
                f"manifest={git_record['remote_commit']} actual={remote_shas}"
            )


def _parse_runtime_records(
    stdout_path: Path, test_name: str = "openfhe_nonlinear_smoke"
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        lines = stdout_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise ValidationError(f"cannot read runtime stdout {stdout_path}: {error}") from error
    for line in lines:
        start = line.find("{")
        if start < 0:
            continue
        try:
            candidate = json.loads(
                line[start:],
                object_pairs_hook=_object_without_duplicates,
                parse_constant=_reject_non_finite,
            )
        except (json.JSONDecodeError, ValidationError):
            continue
        if isinstance(candidate, dict) and candidate.get("test") == test_name:
            records.append(candidate)
    return records


def _command_tokens(
    record: Any,
    label: str,
    *,
    timed: bool,
) -> list[str]:
    expected_keys = {"command", "cwd", "exit_code", "phase"}
    if timed:
        expected_keys.update({"started_at", "finished_at"})
    value = _require_exact_object_keys(record, expected_keys, label)
    if value["cwd"] != str(REPO_ROOT):
        raise ValidationError(f"{label}.cwd must be {REPO_ROOT}")
    if value["exit_code"] != 0 or value["phase"] != "artifact_generation":
        raise ValidationError(
            f"{label} must record exit_code=0 and phase=artifact_generation"
        )
    if timed:
        started = _parse_timestamp(value["started_at"], f"{label}.started_at")
        finished = _parse_timestamp(value["finished_at"], f"{label}.finished_at")
        if finished < started:
            raise ValidationError(f"{label}.finished_at precedes started_at")
    try:
        tokens = shlex.split(value["command"])
    except ValueError as error:
        raise ValidationError(f"{label}.command is not valid shell syntax: {error}") from error
    if not tokens:
        raise ValidationError(f"{label}.command must not be empty")
    return tokens


def _verify_m4_go_command_transcript(
    manifest: dict[str, Any],
    manifest_path: Path,
) -> None:
    if manifest["milestone"] != "M4" or manifest["verdict"] != "GO":
        return
    commands = manifest["commands"]
    if len(commands) != 14:
        raise ValidationError(
            "M4 GO commands transcript must contain exactly 14 ordered records"
        )
    expected_git = [
        ["git", "status", "--porcelain=v1", "--untracked-files=normal"],
        ["git", "branch", "--show-current"],
        ["git", "rev-parse", "HEAD"],
        [
            "git",
            "ls-remote",
            "--exit-code",
            "origin",
            "refs/heads/refactor/openfhe-cpu",
        ],
    ]
    for index, expected in enumerate(expected_git):
        actual = _command_tokens(
            commands[index], f"commands[{index}] Git preflight", timed=False
        )
        if actual != expected:
            raise ValidationError(
                f"commands[{index}] differs from the frozen Git preflight"
            )

    build_root = REPO_ROOT / "build-openfhe"
    executable = REPO_ROOT / M4_WORKLOAD_EXECUTABLE_PATH
    build = _command_tokens(commands[4], "commands[4] clean-first build", timed=True)
    if build != [
        "cmake",
        "--build",
        str(build_root),
        "--clean-first",
        "-j",
        "4",
    ]:
        raise ValidationError("commands[4] differs from the frozen clean-first build")
    ctest = _command_tokens(commands[5], "commands[5] narrow CTest", timed=True)
    if ctest != [
        "ctest",
        "--test-dir",
        str(build_root),
        "--output-on-failure",
        "--no-tests=error",
        "-R",
        M4_CTEST_PATTERN,
    ]:
        raise ValidationError("commands[5] differs from the frozen narrow CTest")
    linkage = _command_tokens(commands[6], "commands[6] OpenFHE linkage", timed=True)
    if linkage != ["ldd", str(executable)]:
        raise ValidationError("commands[6] differs from the frozen ldd check")

    expected_workload_tail = [
        str(executable),
        "--data-root",
        str(REPO_ROOT / "data"),
        "--layer",
        "1",
        "--input-level",
        "29",
    ]
    time_outputs: set[str] = set()
    normalized_workloads: list[list[str]] = []
    for index in range(7, 13):
        workload = _command_tokens(
            commands[index], f"commands[{index}] encoder workload", timed=True
        )
        if len(workload) != 10:
            raise ValidationError(f"commands[{index}] encoder workload shape drifted")
        if workload[:2] != ["/usr/bin/time", "--format=%M"]:
            raise ValidationError(f"commands[{index}] must use frozen GNU time")
        output_option = workload[2]
        if re.fullmatch(r"--output=/tmp/moai-m4-time-[A-Za-z0-9_.-]+\.txt", output_option) is None:
            raise ValidationError(
                f"commands[{index}] GNU time output path is not the frozen temporary form"
            )
        if workload[3:] != expected_workload_tail:
            raise ValidationError(f"commands[{index}] encoder invocation drifted")
        time_outputs.add(output_option)
        normalized_workloads.append([*workload[:2], "--output=<temporary>", *workload[3:]])
    if len(time_outputs) != 6 or any(
        workload != normalized_workloads[0] for workload in normalized_workloads[1:]
    ):
        raise ValidationError(
            "commands[7:13] must contain six identical frozen workload invocations "
            "with distinct GNU time outputs"
        )

    validator_command = _command_tokens(
        commands[13], "commands[13] artifact validator", timed=False
    )
    expected_validator = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "validate_openfhe_artifact.py"),
        "--manifest",
        str(manifest_path.resolve()),
        "--verify-git",
    ]
    if validator_command != expected_validator:
        raise ValidationError("commands[13] differs from the frozen artifact validator")


def _verify_m3_contract_bindings(
    manifest: dict[str, Any],
    repository_root: Path,
    input_records: dict[str, dict[str, Any]],
) -> None:
    if manifest["milestone"] != "M3":
        return

    missing_inputs = REQUIRED_M3_INPUTS - set(input_records)
    if missing_inputs:
        raise ValidationError(
            f"M3 manifest is missing frozen config inputs: {sorted(missing_inputs)}"
        )

    contracts = manifest["contracts"]
    contract_input_fields = {
        "approximation_config": "config/openfhe_approximations.json",
        "trace_contract": "config/moai_trace_channel_scales.json",
        "profile_config": "config/paper_compat.json",
    }
    for prefix, expected_path in contract_input_fields.items():
        if contracts[f"{prefix}_path"] != expected_path:
            raise ValidationError(
                f"contracts.{prefix}_path must be {expected_path}"
            )
        expected_hash = input_records[expected_path]["sha256"]
        if contracts[f"{prefix}_sha256"] != expected_hash:
            raise ValidationError(
                f"contracts.{prefix}_sha256 does not match inputs record"
            )

    profile = manifest["profile"]
    if profile["source_config_path"] != "config/paper_compat.json":
        raise ValidationError("M3 profile must be sourced from config/paper_compat.json")
    profile_input = input_records["config/paper_compat.json"]
    if (
        profile["source_config_sha256"] != profile_input["sha256"]
        or profile["source_config_bytes"] != profile_input["bytes"]
    ):
        raise ValidationError("profile source record does not match the required input record")

    approximation_path = repository_root / "config/openfhe_approximations.json"
    approximation = load_json(approximation_path)
    try:
        operators = approximation["operators"]
        for name, (operator_name, contract_name) in M3_POLYNOMIAL_LOCATORS.items():
            config_contract = operators[operator_name][contract_name]
            manifest_contract = contracts["polynomials"][name]
            expected_contract = M3_EXPECTED_POLYNOMIALS[name]
            config_frozen_fields = {
                "degree": config_contract["degree"],
                "interval": config_contract["interval"],
                "required_depth": config_contract["openfhe_ps"]["required_depth"],
                "estimated_multiplications": config_contract["openfhe_ps"][
                    "estimated_multiplications"
                ],
                "coefficient_sha256": config_contract["coefficient_sha256"],
            }
            if config_frozen_fields != expected_contract:
                raise ValidationError(f"frozen {name} config contract drifted")
            if manifest_contract != expected_contract:
                raise ValidationError(f"manifest {name} polynomial contract drifted")

        shift_config = operators["softmax"]["shift_contract"]
        expected_layer, expected_head, expected_shift_hash = M3_EXPECTED_SOFTMAX_SHIFT
        shift_contract = contracts["softmax_shift"]
        if shift_config["values_sha256"] != expected_shift_hash:
            raise ValidationError("frozen Softmax shift config contract drifted")
        if (
            shift_contract["layer"] != expected_layer
            or shift_contract["head"] != expected_head
            or shift_contract["values_sha256"] != expected_shift_hash
        ):
            raise ValidationError("manifest Softmax shift contract drifted")

        trace_gate = approximation["global_trace_gate"]
        if (
            trace_gate["relative_l2_max"] != M3_EXPECTED_THRESHOLDS["relative_l2_max"]
            or trace_gate["cosine_min"] != M3_EXPECTED_THRESHOLDS["cosine_min"]
        ):
            raise ValidationError("frozen global trace thresholds drifted")
    except (KeyError, TypeError) as error:
        raise ValidationError(
            f"openfhe approximation config is missing an M3 contract field: {error}"
        ) from error

    if contracts["thresholds"] != M3_EXPECTED_THRESHOLDS:
        raise ValidationError("manifest M3 thresholds do not match the frozen gate")


def _require_exact_object_keys(
    value: Any, expected: set[str], label: str
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValidationError(f"{label} must be an object")
    if set(value) != expected:
        raise ValidationError(
            f"{label} keys differ: missing={sorted(expected - set(value))} "
            f"extra={sorted(set(value) - expected)}"
        )
    return value


def _expected_encoder_trace_inputs(
    trace_contract: dict[str, Any],
) -> dict[str, dict[str, str]]:
    try:
        required_files = trace_contract["required_files"]
        layers = trace_contract["layers"]
    except (KeyError, TypeError) as error:
        raise ValidationError(
            f"encoder trace contract is missing file bindings: {error}"
        ) from error
    if not isinstance(required_files, dict) or len(required_files) != 37:
        raise ValidationError("encoder trace contract must map exactly 37 logical files")
    if not isinstance(layers, list) or len(layers) != 12:
        raise ValidationError("encoder trace contract must contain exactly 12 layers")

    layer_by_id: dict[int, dict[str, Any]] = {}
    for index, layer in enumerate(layers):
        if not isinstance(layer, dict):
            raise ValidationError(f"encoder trace layer {index} must be an object")
        layer_id = layer.get("layer_id")
        if (
            not isinstance(layer_id, int)
            or isinstance(layer_id, bool)
            or layer_id not in range(12)
            or layer_id in layer_by_id
        ):
            raise ValidationError("encoder trace layer ids must be exactly 0 through 11")
        layer_by_id[layer_id] = layer
    if set(layer_by_id) != set(range(12)):
        raise ValidationError("encoder trace layer ids must be exactly 0 through 11")

    layer_ids = [M4_LAYER_ID]
    expected: dict[str, dict[str, str]] = {}
    for layer_id in layer_ids:
        hashes = layer_by_id[layer_id].get("sha256")
        if not isinstance(hashes, dict) or set(hashes) != set(required_files):
            raise ValidationError(
                f"encoder trace layer {layer_id} must bind all 37 logical files"
            )
        for logical_name in sorted(required_files):
            specification = required_files[logical_name]
            if not isinstance(specification, dict):
                raise ValidationError(
                    f"encoder trace file mapping {logical_name!r} must be an object"
                )
            contract_path = specification.get("path")
            if not isinstance(contract_path, str):
                raise ValidationError(
                    f"encoder trace file mapping {logical_name!r} lacks a path"
                )
            raw_path = Path(contract_path)
            if (
                raw_path.is_absolute()
                or "\\" in contract_path
                or ".." in raw_path.parts
                or raw_path.as_posix() != contract_path
            ):
                raise ValidationError(
                    f"encoder trace file mapping {logical_name!r} is not normalized"
                )
            repository_path = (
                Path("data") / f"layer_{layer_id}" / raw_path
            ).as_posix()
            if repository_path in expected:
                raise ValidationError(
                    f"encoder trace file mappings collide at {repository_path}"
                )
            expected[repository_path] = {
                "sha256": hashes[logical_name],
                "media_type": "text/csv",
                "role": "weights" if "/parms/" in f"/{contract_path}" else "trace",
            }

    expected_count = 37
    if len(expected) != expected_count:
        raise ValidationError(
            f"M4 encoder trace must bind exactly {expected_count} data files"
        )
    return expected


def _verify_encoder_contract_bindings(
    manifest: dict[str, Any],
    repository_root: Path,
    input_records: dict[str, dict[str, Any]],
) -> None:
    milestone = manifest["milestone"]
    missing_inputs = REQUIRED_ENCODER_INPUTS - set(input_records)
    if missing_inputs:
        raise ValidationError(
            f"{milestone} manifest is missing frozen config inputs: "
            f"{sorted(missing_inputs)}"
        )
    for path in REQUIRED_ENCODER_INPUTS:
        record = input_records[path]
        if record["media_type"] != "application/json" or record["role"] != "configuration":
            raise ValidationError(
                f"{milestone} config input {path} must be application/json configuration"
            )

    contracts = _require_exact_object_keys(
        manifest["contracts"],
        {
            "approximation_config_path",
            "approximation_config_sha256",
            "encoder_trace_contract_path",
            "encoder_trace_contract_sha256",
            "profile_config_path",
            "profile_config_sha256",
            "execution",
            "thresholds",
        },
        f"{milestone} contracts",
    )
    contract_inputs = {
        "approximation_config": "config/openfhe_approximations.json",
        "encoder_trace_contract": "config/moai_encoder_trace.json",
        "profile_config": ENCODER_FEATURE_PROFILE_PATH,
    }
    for prefix, expected_path in contract_inputs.items():
        if contracts[f"{prefix}_path"] != expected_path:
            raise ValidationError(f"contracts.{prefix}_path must be {expected_path}")
        if contracts[f"{prefix}_sha256"] != input_records[expected_path]["sha256"]:
            raise ValidationError(
                f"contracts.{prefix}_sha256 does not match inputs record"
            )

    profile = manifest["profile"]
    profile_input = input_records[ENCODER_FEATURE_PROFILE_PATH]
    if profile["source_config_path"] != ENCODER_FEATURE_PROFILE_PATH or (
        profile["source_config_sha256"] != profile_input["sha256"]
        or profile["source_config_bytes"] != profile_input["bytes"]
    ):
        raise ValidationError(
            f"{milestone} profile source does not match "
            f"{ENCODER_FEATURE_PROFILE_PATH}"
        )
    if (
        profile["effective_profile_locator"] != ENCODER_FEATURE_PROFILE_LOCATOR
        or profile["effective_profile_sha256"] != ENCODER_FEATURE_PROFILE_SHA256
    ):
        raise ValidationError(
            f"{milestone} profile must bind {ENCODER_FEATURE_PROFILE_LOCATOR} "
            f"with SHA-256 {ENCODER_FEATURE_PROFILE_SHA256}"
        )
    effective = profile["effective_profile_payload"]
    mismatches = {
        key: {"expected": expected, "actual": effective.get(key)}
        for key, expected in ENCODER_PROFILE_FIELDS.items()
        if effective.get(key) != expected
    }
    if mismatches:
        raise ValidationError(
            "M4 evidence requires the frozen 1024-slot/depth-47 "
            f"encoder profile: {mismatches}"
        )

    trace_contract = load_json(repository_root / "config/moai_encoder_trace.json")
    try:
        if (
            trace_contract["profile_id"] != "paper_compat"
            or trace_contract["security_claim"] != "none"
            or trace_contract["dimensions"]["encoder_layers"] != 12
            or trace_contract["dimensions"]["trace_token_rows"] != 5
            or trace_contract["dimensions"]["hidden_size"] != 768
        ):
            raise ValidationError("frozen encoder trace contract identity drifted")
    except (KeyError, TypeError) as error:
        raise ValidationError(
            f"encoder trace contract is missing a required identity field: {error}"
        ) from error

    try:
        sources = trace_contract["sources"]
        if set(sources) != {"approximations", "channel_scales"}:
            raise ValidationError("encoder trace contract source bindings drifted")
        source_inputs = {
            "approximations": "config/openfhe_approximations.json",
            "channel_scales": "config/moai_trace_channel_scales.json",
        }
        for source_name, expected_path in source_inputs.items():
            source = sources[source_name]
            if (
                source["path"] != expected_path
                or source["sha256"] != input_records[expected_path]["sha256"]
            ):
                raise ValidationError(
                    f"encoder trace contract source {source_name} does not match inputs"
                )
    except (KeyError, TypeError) as error:
        raise ValidationError(
            f"encoder trace contract is missing a source binding: {error}"
        ) from error

    expected_trace_inputs = _expected_encoder_trace_inputs(trace_contract)
    expected_input_paths = REQUIRED_ENCODER_INPUTS | set(expected_trace_inputs)
    actual_input_paths = set(input_records)
    if actual_input_paths != expected_input_paths:
        raise ValidationError(
            f"{milestone} inputs differ from the frozen encoder trace set: "
            f"missing={sorted(expected_input_paths - actual_input_paths)} "
            f"extra={sorted(actual_input_paths - expected_input_paths)}"
        )
    for path, expected_record in expected_trace_inputs.items():
        record = input_records[path]
        for key, expected_value in expected_record.items():
            if record[key] != expected_value:
                raise ValidationError(
                    f"{milestone} encoder trace input {path} has wrong {key}: "
                    f"expected={expected_value!r} actual={record[key]!r}"
                )

    if manifest["workload"]["executable_path"] != M4_WORKLOAD_EXECUTABLE_PATH:
        raise ValidationError(
            f"M4 workload.executable_path must be {M4_WORKLOAD_EXECUTABLE_PATH}"
        )

    execution = contracts["execution"]
    common_values = {
        "mode": "server-only",
        "trace_shape": ENCODER_TRACE_SHAPE,
        "feature_block_size": ENCODER_FEATURE_BLOCK_SIZE,
        "checkpoint_decryption_owner": "client",
        "server_private_key_present": False,
        "server_decryptions": 0,
        "server_plaintext_activations": False,
        "multiplicative_depth": M4_MULTIPLICATIVE_DEPTH,
        "max_observed_level": M4_MAX_OBSERVED_LEVEL,
        "max_polynomial_depth": M4_MAX_POLYNOMIAL_DEPTH,
    }
    _require_exact_object_keys(execution, M4_EXECUTION_KEYS, "M4 execution")
    expected_values = {
        **common_values,
        "quality_reference": (
            "config/moai_encoder_trace.json single-layer frozen polynomial oracle"
        ),
        "encoder_layers": 1,
        "layer_id": M4_LAYER_ID,
        "required_checkpoints": list(M4_CHECKPOINT_NAMES),
    }
    for key, expected in expected_values.items():
        if execution.get(key) != expected:
            raise ValidationError(
                f"{milestone} execution.{key} mismatch: expected={expected!r} "
                f"actual={execution.get(key)!r}"
            )
    if contracts["thresholds"] != M4_EXPECTED_THRESHOLDS:
        raise ValidationError(
            f"manifest {milestone} thresholds do not match the frozen gate"
        )


def _verify_contract_bindings(
    manifest: dict[str, Any],
    repository_root: Path,
    input_records: dict[str, dict[str, Any]],
) -> None:
    milestone = manifest["milestone"]
    if milestone == "M3":
        _verify_m3_contract_bindings(manifest, repository_root, input_records)
    elif milestone == "M4":
        _verify_encoder_contract_bindings(manifest, repository_root, input_records)
    else:
        raise ValidationError(f"unsupported schema-v2 milestone: {milestone}")


def _require_finite_number(
    value: Any,
    label: str,
    minimum: float,
    maximum: float,
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


def _verify_encoder_operation_counts(
    value: Any,
    label: str,
    expected_rotations: int,
    expected_bootstraps: int,
    expected_iterations: int,
) -> dict[str, int]:
    counts = _require_exact_object_keys(
        value, ENCODER_OPERATION_COUNT_KEYS, label
    )
    for key, count in counts.items():
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ValidationError(f"{label}.{key} must be a nonnegative integer")
    expected_counts = {
        "rotations": expected_rotations,
        "ct_pt_multiplications": ENCODER_CT_PT_MULTIPLICATIONS_PER_LAYER,
        "ct_ct_multiplications": ENCODER_CT_CT_MULTIPLICATIONS_PER_LAYER,
        "explicit_rescale_requests": ENCODER_EXPLICIT_RESCALE_REQUESTS_PER_LAYER,
        "chebyshev_evaluations": ENCODER_CHEBYSHEV_EVALUATIONS_PER_LAYER,
        "estimated_polynomial_multiplications": (
            ENCODER_ESTIMATED_POLYNOMIAL_MULTIPLICATIONS_PER_LAYER
        ),
        "bootstraps": expected_bootstraps,
        "bootstrap_iterations": expected_iterations,
    }
    for key, expected in expected_counts.items():
        if counts[key] != expected:
            raise ValidationError(
                f"{label}.{key} must be {expected}, got {counts[key]}"
            )
    return counts


def _verify_encoder_runtime_identity(
    record: dict[str, Any],
    profile: dict[str, Any],
    execution: dict[str, Any],
    label: str,
) -> None:
    expected = {
        "profile": profile["id"],
        "security_claim": profile["security_claim"],
        "parameter_sha256": profile["effective_profile_sha256"],
        "execution_mode": "server-only",
        "actual_trace_shape": ENCODER_TRACE_SHAPE,
        "actual_trace_value_count": ENCODER_TRACE_VALUE_COUNT,
        "feature_block_size": ENCODER_FEATURE_BLOCK_SIZE,
        "checkpoint_decryption_owner": "client",
        "server_private_key_present": False,
        "server_decryptions": 0,
        "server_plaintext_activations": False,
        "multiplicative_depth": M4_MULTIPLICATIVE_DEPTH,
        "max_observed_level": M4_MAX_OBSERVED_LEVEL,
        "max_polynomial_depth": M4_MAX_POLYNOMIAL_DEPTH,
    }
    for key, expected_value in expected.items():
        if record.get(key) != expected_value:
            raise ValidationError(
                f"{label}: {key} mismatch: expected={expected_value!r} "
                f"actual={record.get(key)!r}"
            )
    if execution["trace_shape"] != record["actual_trace_shape"]:
        raise ValidationError(f"{label}: actual trace shape differs from contract")


def _verify_checkpoint(
    checkpoint: Any,
    expected: dict[str, Any],
    label: str,
) -> dict[str, Any]:
    value = _require_exact_object_keys(
        checkpoint,
        {
            "name",
            "level",
            "noise_scale_degree",
            "remaining_levels",
            "scale_bits",
            "ciphertext_count",
            "decryption_owner",
        },
        label,
    )
    if value["name"] != expected["name"]:
        raise ValidationError(
            f"{label}.name must be {expected['name']!r}, got {value['name']!r}"
        )
    level = value["level"]
    if (
        not isinstance(level, int)
        or isinstance(level, bool)
        or level != expected["level"]
    ):
        raise ValidationError(f"{label}.level must be {expected['level']}")
    _require_finite_number(value["scale_bits"], f"{label}.scale_bits", 1.0, 100.0)
    noise_scale_degree = value["noise_scale_degree"]
    if (
        not isinstance(noise_scale_degree, int)
        or isinstance(noise_scale_degree, bool)
        or noise_scale_degree <= 0
    ):
        raise ValidationError(f"{label}.noise_scale_degree must be a positive integer")
    remaining_levels = value["remaining_levels"]
    if (
        not isinstance(remaining_levels, int)
        or isinstance(remaining_levels, bool)
        or remaining_levels <= 0
    ):
        raise ValidationError(f"{label}.remaining_levels must be a positive integer")
    ciphertext_count = value["ciphertext_count"]
    if (
        not isinstance(ciphertext_count, int)
        or isinstance(ciphertext_count, bool)
        or ciphertext_count != 5
    ):
        raise ValidationError(f"{label}.ciphertext_count must be 5")
    if value["decryption_owner"] != "client":
        raise ValidationError(f"{label}.decryption_owner must be client")
    if not _json_equal(value, expected):
        raise ValidationError(
            f"{label} metadata differs from the frozen M4 checkpoint contract"
        )
    return value


def _verify_m4_diagnostic_record(
    diagnostic: Any,
    target: dict[str, Any],
    label: str,
) -> dict[str, Any]:
    value = _require_exact_object_keys(diagnostic, M4_DIAGNOSTIC_KEYS, label)
    expected = {
        "test": "openfhe_encoder_layer_smoke",
        "profile": target["profile"],
        "security_claim": target["security_claim"],
        "parameter_sha256": target["parameter_sha256"],
        "layer": target["layer_id"],
        "input_level": target["input_level"],
        "tokens": target["actual_trace_shape"][0],
        "hidden_size": target["actual_trace_shape"][1],
        "intermediate_size": 3072,
        "feature_block": target["feature_block_size"],
        "final_level": target["output_level"],
        "final_remaining_levels": target["remaining_levels"],
        "multiplicative_depth": target["multiplicative_depth"],
        "max_observed_level": target["max_observed_level"],
        "max_polynomial_depth": target["max_polynomial_depth"],
    }
    for key, expected_value in expected.items():
        if value[key] != expected_value:
            raise ValidationError(
                f"{label}.{key} mismatch: expected={expected_value!r} "
                f"actual={value[key]!r}"
            )
    for field in M4_PHASE_LATENCY_FIELDS:
        _require_finite_number(value[field], f"{label}.{field}", 0.0, math.inf)
    for field in (
        "relative_l2",
        "max_absolute",
        "exact_trace_relative_l2",
        "exact_trace_max_absolute",
        "inactive_max_absolute",
    ):
        _require_finite_number(value[field], f"{label}.{field}", 0.0, math.inf)
    for field in ("cosine", "exact_trace_cosine"):
        _require_finite_number(
            value[field], f"{label}.{field}", -1.000000000001, 1.000000000001
        )
    peak_rss_bytes = value["peak_rss_bytes"]
    if (
        not isinstance(peak_rss_bytes, int)
        or isinstance(peak_rss_bytes, bool)
        or peak_rss_bytes <= 0
    ):
        raise ValidationError(f"{label}.peak_rss_bytes must be a positive integer")
    for field in (
        "rotations",
        "ct_pt_multiplications",
        "ct_ct_multiplications",
        "explicit_rescale_requests",
        "chebyshev_evaluations",
        "estimated_polynomial_multiplications",
        "bootstraps",
        "bootstrap_iterations",
    ):
        count = value[field]
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise ValidationError(f"{label}.{field} must be a positive integer")

    for diagnostic_key, target_key in {
        "relative_l2": "relative_l2",
        "cosine": "cosine",
        "inactive_max_absolute": "inactive_max_abs",
    }.items():
        if not math.isclose(
            float(value[diagnostic_key]),
            float(target[target_key]),
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            raise ValidationError(
                f"{label}.{diagnostic_key} differs from the target runtime record"
            )
    for key in ENCODER_OPERATION_COUNT_KEYS:
        if value[key] != target["operation_counts"][key]:
            raise ValidationError(
                f"{label}.{key} differs from the target runtime record"
            )
    return value


def _read_encoder_metrics_csv(
    path: Path,
    repeat_count: int,
    runtime_records: list[dict[str, Any]],
    m4_diagnostics: list[dict[str, Any]],
) -> tuple[list[float], list[int], dict[str, list[float]]]:
    expected_columns = {
        "run",
        "exit_code",
        "elapsed_seconds",
        "peak_rss_kib",
        "final_rel_l2",
        "final_cosine",
        "inactive_max_abs",
        "multiplicative_depth",
        "max_observed_level",
        "max_polynomial_depth",
        "checkpoint_metadata_sha256",
        "bootstraps",
        "bootstrap_iterations",
        *M4_PHASE_LATENCY_FIELDS,
        "batch_total_ms",
        "batch_amortized_ms_per_token",
        "server_amortized_ms_per_token",
    }
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or set(reader.fieldnames) != expected_columns:
                actual = set(reader.fieldnames or [])
                raise ValidationError(
                    f"metrics.csv columns differ: missing={sorted(expected_columns - actual)} "
                    f"extra={sorted(actual - expected_columns)}"
                )
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as error:
        raise ValidationError(f"cannot read metrics.csv: {error}") from error
    if len(rows) != repeat_count:
        raise ValidationError("metrics.csv row count differs from workload.repeat_count")

    elapsed_values: list[float] = []
    peak_rss_values: list[int] = []
    m4_latency_values = {
        **{field: [] for field in M4_PHASE_LATENCY_FIELDS},
        "batch_total_ms": [],
        "batch_amortized_ms_per_token": [],
        "server_amortized_ms_per_token": [],
    }
    for index, (row, runtime) in enumerate(zip(rows, runtime_records), start=1):
        counts = runtime["operation_counts"]
        expected = {
            "final_rel_l2": runtime["relative_l2"],
            "final_cosine": runtime["cosine"],
            "inactive_max_abs": runtime["inactive_max_abs"],
            "bootstraps": counts["bootstraps"],
            "bootstrap_iterations": counts["bootstrap_iterations"],
        }
        try:
            if int(row["run"]) != index or int(row["exit_code"]) != 0:
                raise ValueError("run sequence or exit code")
            elapsed = float(row["elapsed_seconds"])
            peak_rss = int(row["peak_rss_kib"])
            if not math.isfinite(elapsed) or elapsed <= 0.0 or peak_rss <= 0:
                raise ValueError("resource metrics")
            for key, expected_value in expected.items():
                actual_value = float(row[key])
                if not math.isfinite(actual_value) or not math.isclose(
                    actual_value,
                    float(expected_value),
                    rel_tol=1e-12,
                    abs_tol=1e-15,
                ):
                    raise ValueError(f"{key} differs from stdout")
            for key in (
                "multiplicative_depth",
                "max_observed_level",
                "max_polynomial_depth",
            ):
                if int(row[key]) != runtime[key]:
                    raise ValueError(f"{key} differs from stdout")
            expected_checkpoint_hash = checkpoint_metadata_sha256(
                runtime["checkpoints"]
            )
            if row["checkpoint_metadata_sha256"] != expected_checkpoint_hash:
                raise ValueError(
                    "checkpoint_metadata_sha256 differs from stdout"
                )
            if expected_checkpoint_hash != M4_CHECKPOINT_METADATA_SHA256:
                raise ValueError("stdout checkpoint metadata hash is not frozen")
            diagnostic = m4_diagnostics[index - 1]
            if diagnostic["peak_rss_bytes"] > peak_rss * 1024:
                raise ValueError("stdout diagnostic peak RSS exceeds peak_rss_kib")
            phase_values: dict[str, float] = {}
            for field in M4_PHASE_LATENCY_FIELDS:
                phase_value = float(row[field])
                if not math.isfinite(phase_value) or phase_value < 0.0:
                    raise ValueError(f"{field} is not finite and nonnegative")
                if not math.isclose(
                    phase_value,
                    float(diagnostic[field]),
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                ):
                    raise ValueError(f"{field} differs from stdout diagnostic")
                phase_values[field] = phase_value
            derived_values = {
                "batch_total_ms": elapsed * 1000.0,
                "batch_amortized_ms_per_token": (
                    elapsed * 1000.0 / M4_TOKENS_PER_BATCH
                ),
                "server_amortized_ms_per_token": (
                    phase_values["server_online_ms"] / M4_TOKENS_PER_BATCH
                ),
            }
            for field, expected_value in derived_values.items():
                actual_value = float(row[field])
                if (
                    not math.isfinite(actual_value)
                    or actual_value < 0.0
                    or not math.isclose(
                        actual_value,
                        expected_value,
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    )
                ):
                    raise ValueError(f"{field} differs from its source metric")
                m4_latency_values[field].append(actual_value)
            for field, value in phase_values.items():
                m4_latency_values[field].append(value)
        except (KeyError, TypeError, ValueError) as error:
            raise ValidationError(f"metrics.csv row {index} is invalid: {error}") from error
        elapsed_values.append(elapsed)
        peak_rss_values.append(peak_rss)
    return elapsed_values, peak_rss_values, m4_latency_values


def _verify_timing_and_repeat_metrics(
    manifest: dict[str, Any],
    elapsed_values: list[float],
    peak_rss_values: list[int],
) -> None:
    workload = manifest["workload"]
    metrics = manifest["metrics"]
    if workload["repeat_count"] != 5:
        raise ValidationError(
            f"{manifest['milestone']} evidence requires exactly five measured repeats"
        )
    if (
        metrics["repeat_count"] != 5
        or metrics["successful_repeats"] != 5
        or metrics["warmup_count"] != 1
    ):
        raise ValidationError("encoder metrics require one warm-up and five successful repeats")
    expected_timing = {
        "minimum": min(elapsed_values),
        "median": statistics.median(elapsed_values),
        "maximum": max(elapsed_values),
    }
    for key, expected in expected_timing.items():
        actual = metrics["elapsed_seconds"][key]
        if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-15):
            raise ValidationError(f"metrics.elapsed_seconds.{key} differs from metrics.csv")
    if metrics["peak_rss_kib_max"] != max(peak_rss_values):
        raise ValidationError("metrics.peak_rss_kib_max differs from metrics.csv")


def _verify_latency_summary(
    value: Any,
    samples: list[float],
    label: str,
) -> None:
    summary = _require_exact_object_keys(
        value, {"minimum", "median", "maximum"}, label
    )
    expected = {
        "minimum": min(samples),
        "median": statistics.median(samples),
        "maximum": max(samples),
    }
    for key, expected_value in expected.items():
        actual_value = _require_finite_number(
            summary[key], f"{label}.{key}", 0.0, math.inf
        )
        if not math.isclose(
            actual_value,
            expected_value,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValidationError(f"{label}.{key} differs from metrics.csv")


def _verify_m4_latency_metrics(
    metrics: dict[str, Any],
    latency_values: dict[str, list[float]],
) -> None:
    if metrics["tokens_per_batch"] != M4_TOKENS_PER_BATCH:
        raise ValidationError(
            f"M4 metrics.tokens_per_batch must be {M4_TOKENS_PER_BATCH}"
        )
    phase_metrics = _require_exact_object_keys(
        metrics["phase_latency_ms"],
        set(M4_PHASE_LATENCY_FIELDS),
        "M4 metrics.phase_latency_ms",
    )
    for field in M4_PHASE_LATENCY_FIELDS:
        _verify_latency_summary(
            phase_metrics[field],
            latency_values[field],
            f"M4 metrics.phase_latency_ms.{field}",
        )
    for field in (
        "batch_total_ms",
        "batch_amortized_ms_per_token",
        "server_amortized_ms_per_token",
    ):
        _verify_latency_summary(
            metrics[field], latency_values[field], f"M4 metrics.{field}"
        )


def _verify_m4_runtime_evidence(
    manifest: dict[str, Any],
    artifact_root: Path,
) -> None:
    workload = manifest["workload"]
    records = _parse_runtime_records(
        artifact_root / "stdout.log", "openfhe_encoder_layer"
    )
    diagnostics = _parse_runtime_records(
        artifact_root / "stdout.log", "openfhe_encoder_layer_smoke"
    )
    if len(records) != workload["repeat_count"]:
        raise ValidationError(
            "stdout M4 record count differs from workload.repeat_count: "
            f"records={len(records)} repeats={workload['repeat_count']}"
        )
    if len(diagnostics) != workload["repeat_count"]:
        raise ValidationError(
            "stdout M4 diagnostic count differs from workload.repeat_count: "
            f"records={len(diagnostics)} repeats={workload['repeat_count']}"
        )
    profile = manifest["profile"]
    execution = manifest["contracts"]["execution"]

    record_keys = {
        "test",
        "profile",
        "security_claim",
        "parameter_sha256",
        "execution_mode",
        "encoder_layers",
        "layer_id",
        "input_level",
        "output_level",
        "remaining_levels",
        "multiplicative_depth",
        "max_observed_level",
        "max_polynomial_depth",
        "actual_trace_shape",
        "actual_trace_value_count",
        "feature_block_size",
        "checkpoint_decryption_owner",
        "server_private_key_present",
        "server_decryptions",
        "server_plaintext_activations",
        "relative_l2",
        "cosine",
        "inactive_max_abs",
        "checkpoints",
        "operation_counts",
    }
    per_record_checkpoints: list[list[dict[str, Any]]] = []
    first_counts: dict[str, int] | None = None
    for index, record in enumerate(records, start=1):
        label = f"stdout M4 runtime record {index}"
        _require_exact_object_keys(record, record_keys, label)
        _verify_encoder_runtime_identity(record, profile, execution, label)
        if (
            record["encoder_layers"] != 1
            or not isinstance(record["layer_id"], int)
            or isinstance(record["layer_id"], bool)
            or record["layer_id"] != M4_LAYER_ID
        ):
            raise ValidationError(f"{label}: layer identity differs from M4 contract")
        expected_levels = {
            "input_level": M4_INPUT_LEVEL,
            "output_level": M4_OUTPUT_LEVEL,
            "remaining_levels": M4_REMAINING_LEVELS,
        }
        for key, expected in expected_levels.items():
            if (
                not isinstance(record[key], int)
                or isinstance(record[key], bool)
                or record[key] != expected
            ):
                raise ValidationError(
                    f"{label}.{key} must be {expected}, got {record[key]!r}"
                )
        _require_finite_number(
            record["relative_l2"],
            f"{label}.relative_l2",
            0.0,
            M4_EXPECTED_THRESHOLDS["relative_l2_max"],
        )
        _require_finite_number(
            record["cosine"],
            f"{label}.cosine",
            M4_EXPECTED_THRESHOLDS["cosine_min"],
            1.000000000001,
        )
        _require_finite_number(
            record["inactive_max_abs"],
            f"{label}.inactive_max_abs",
            0.0,
            M4_EXPECTED_THRESHOLDS["inactive_max_abs"],
        )
        checkpoints = record["checkpoints"]
        if not isinstance(checkpoints, list) or len(checkpoints) != len(
            M4_CHECKPOINT_NAMES
        ):
            raise ValidationError(f"{label}: checkpoints must contain the four frozen sites")
        verified_checkpoints = [
            _verify_checkpoint(
                checkpoint,
                expected_checkpoint,
                f"{label}.checkpoints[{checkpoint_index}]",
            )
            for checkpoint_index, (checkpoint, expected_checkpoint) in enumerate(
                zip(checkpoints, M4_EXPECTED_CHECKPOINTS)
            )
        ]
        checkpoint_hash = checkpoint_metadata_sha256(verified_checkpoints)
        if checkpoint_hash != M4_CHECKPOINT_METADATA_SHA256:
            raise ValidationError(
                f"{label}.checkpoints canonical SHA-256 differs from the frozen contract"
            )
        per_record_checkpoints.append(verified_checkpoints)
        counts = _verify_encoder_operation_counts(
            record["operation_counts"],
            f"{label}.operation_counts",
            ENCODER_ROTATIONS_PER_LAYER,
            ENCODER_BOOTSTRAPS_PER_LAYER,
            ENCODER_BOOTSTRAP_ITERATIONS_PER_LAYER,
        )
        if first_counts is None:
            first_counts = counts
        elif counts != first_counts:
            raise ValidationError("M4 operation counts differ across repeats")
        _verify_m4_diagnostic_record(
            diagnostics[index - 1], record, f"stdout M4 diagnostic record {index}"
        )

    elapsed, peak_rss, latency_values = _read_encoder_metrics_csv(
        artifact_root / "metrics.csv",
        workload["repeat_count"],
        records,
        diagnostics,
    )
    _verify_timing_and_repeat_metrics(manifest, elapsed, peak_rss)
    metrics = _require_exact_object_keys(
        manifest["metrics"],
        {
            "repeat_count",
            "successful_repeats",
            "warmup_count",
            "elapsed_seconds",
            "tokens_per_batch",
            "phase_latency_ms",
            "batch_total_ms",
            "batch_amortized_ms_per_token",
            "server_amortized_ms_per_token",
            "peak_rss_kib_max",
            "actual_trace_shape",
            "actual_trace_value_count",
            "feature_block_size",
            "input_level",
            "output_level",
            "remaining_levels",
            "multiplicative_depth",
            "max_observed_level",
            "max_polynomial_depth",
            "quality",
            "checkpoints",
            "checkpoint_metadata_sha256",
            "operation_counts",
        },
        "M4 metrics",
    )
    _verify_m4_latency_metrics(metrics, latency_values)
    if (
        metrics["actual_trace_shape"] != ENCODER_TRACE_SHAPE
        or metrics["actual_trace_value_count"] != ENCODER_TRACE_VALUE_COUNT
        or metrics["feature_block_size"] != ENCODER_FEATURE_BLOCK_SIZE
    ):
        raise ValidationError("M4 metrics trace shape or feature block changed")
    expected_metric_levels = {
        "input_level": M4_INPUT_LEVEL,
        "output_level": M4_OUTPUT_LEVEL,
        "remaining_levels": M4_REMAINING_LEVELS,
    }
    for key, expected in expected_metric_levels.items():
        if metrics[key] != expected or any(record[key] != expected for record in records):
            raise ValidationError(f"M4 metrics.{key} violates the frozen schedule")
    expected_depth_metrics = {
        "multiplicative_depth": M4_MULTIPLICATIVE_DEPTH,
        "max_observed_level": M4_MAX_OBSERVED_LEVEL,
        "max_polynomial_depth": M4_MAX_POLYNOMIAL_DEPTH,
    }
    for key, expected in expected_depth_metrics.items():
        if metrics[key] != expected or any(record[key] != expected for record in records):
            raise ValidationError(f"M4 metrics.{key} violates the frozen depth contract")
    expected_quality = {
        "relative_l2_max": max(record["relative_l2"] for record in records),
        "cosine_min": min(record["cosine"] for record in records),
        "inactive_max_abs_max": max(record["inactive_max_abs"] for record in records),
    }
    if not _json_equal(metrics["quality"], expected_quality):
        raise ValidationError("M4 metrics.quality differs from stdout")
    expected_checkpoints = []
    for checkpoint_index, name in enumerate(M4_CHECKPOINT_NAMES):
        values = [items[checkpoint_index] for items in per_record_checkpoints]
        counts = {item["ciphertext_count"] for item in values}
        if len(counts) != 1:
            raise ValidationError(f"M4 checkpoint {name} ciphertext count changed")
        expected_checkpoints.append(
            {
                "name": name,
                "level_min": min(item["level"] for item in values),
                "level_max": max(item["level"] for item in values),
                "noise_scale_degree_min": min(
                    item["noise_scale_degree"] for item in values
                ),
                "noise_scale_degree_max": max(
                    item["noise_scale_degree"] for item in values
                ),
                "remaining_levels_min": min(
                    item["remaining_levels"] for item in values
                ),
                "remaining_levels_max": max(
                    item["remaining_levels"] for item in values
                ),
                "scale_bits_min": min(item["scale_bits"] for item in values),
                "scale_bits_max": max(item["scale_bits"] for item in values),
                "ciphertext_count": next(iter(counts)),
                "decryption_owner": "client",
            }
        )
    if not _json_equal(metrics["checkpoints"], expected_checkpoints):
        raise ValidationError("M4 metrics.checkpoints differs from stdout")
    if metrics["checkpoint_metadata_sha256"] != M4_CHECKPOINT_METADATA_SHA256:
        raise ValidationError(
            "M4 metrics.checkpoint_metadata_sha256 violates the frozen contract"
        )
    assert first_counts is not None
    if metrics["operation_counts"] != first_counts:
        raise ValidationError("M4 metrics.operation_counts differs from stdout")


def _verify_checksum_evidence(
    artifact_root: Path, artifact_records: dict[str, dict[str, Any]]
) -> None:
    checksum_path = artifact_root / "SHA256SUMS"
    checksum_entries: dict[str, str] = {}
    try:
        checksum_lines = checksum_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise ValidationError(f"cannot read SHA256SUMS: {error}") from error
    for line in checksum_lines:
        match = re.fullmatch(r"([0-9a-f]{64})  ([A-Za-z0-9._+-]+)", line)
        if match is None or match.group(2) in checksum_entries:
            raise ValidationError(f"invalid or duplicate SHA256SUMS line: {line!r}")
        checksum_entries[match.group(2)] = match.group(1)
    if set(checksum_entries) != {"stdout.log", "metrics.csv"}:
        raise ValidationError(
            "SHA256SUMS must cover exactly stdout.log and metrics.csv, not itself/manifest"
        )
    for path in checksum_entries:
        if checksum_entries[path] != artifact_records[path]["sha256"]:
            raise ValidationError(f"SHA256SUMS digest disagrees for {path}")


def _verify_runtime_evidence(
    manifest: dict[str, Any],
    repository_root: Path,
    artifact_root: Path,
    artifact_records: dict[str, dict[str, Any]],
) -> None:
    workload = manifest["workload"]
    executable = _resolve_confined_file(
        repository_root,
        workload["executable_path"],
        "workload.executable_path",
    )
    _verify_file_record(
        executable,
        {
            "bytes": workload["executable_bytes"],
            "sha256": workload["executable_sha256"],
        },
        "workload executable",
    )
    if manifest["milestone"] == "M4":
        _verify_m4_runtime_evidence(manifest, artifact_root)
        _verify_checksum_evidence(artifact_root, artifact_records)
        return
    if manifest["milestone"] == "M3" and workload["repeat_count"] < 3:
        raise ValidationError("M3 evidence requires at least three independent repeats")

    stdout_path = artifact_root / "stdout.log"
    runtime_records = _parse_runtime_records(stdout_path)
    if len(runtime_records) != workload["repeat_count"]:
        raise ValidationError(
            "stdout nonlinear-smoke record count differs from workload.repeat_count: "
            f"records={len(runtime_records)} repeats={workload['repeat_count']}"
        )
    profile = manifest["profile"]
    effective = profile["effective_profile_payload"]
    contracts = manifest["contracts"]
    shift_contract = contracts["softmax_shift"]
    polynomial_contracts = contracts["polynomials"]
    for index, record in enumerate(runtime_records, start=1):
        prefix = f"stdout runtime record {index}"
        expected_pairs = {
            "profile": profile["id"],
            "security_claim": profile["security_claim"],
            "parameter_sha256": profile["effective_profile_sha256"],
            "effective_profile_schema_version": effective[
                "effective_profile_schema_version"
            ],
            "ring_dimension": effective["ring_dimension"],
            "slot_count": effective["slot_count"],
            "multiplicative_depth": effective["multiplicative_depth"],
            "levels_available_after_bootstrap": effective[
                "levels_available_after_bootstrap"
            ],
            "encoded_slots": effective["bootstrap_slots"],
            "bootstrap_iterations_per_call": effective["bootstrap_iterations"],
            "bootstrap_precision": effective["bootstrap_precision"],
            "packing_active_slots": effective["bootstrap_slots"],
            "logical_active_slots": 4,
            "gelu_degree": polynomial_contracts["gelu"]["degree"],
            "softmax_exp_degree": polynomial_contracts["softmax_exponential"]["degree"],
            "softmax_reciprocal_degree": polynomial_contracts[
                "softmax_reciprocal"
            ]["degree"],
            "softmax_shift_layer": shift_contract["layer"],
            "softmax_shift_head": shift_contract["head"],
            "softmax_shift_sha256": shift_contract["values_sha256"],
            "layernorm_invsqrt_degree": polynomial_contracts[
                "layernorm_inverse_sqrt"
            ]["degree"],
            **M3_EXPECTED_OPERATION_COUNTS,
        }
        for contract_name, runtime_prefix in M3_RUNTIME_POLYNOMIAL_PREFIXES.items():
            polynomial = polynomial_contracts[contract_name]
            expected_pairs.update(
                {
                    f"{runtime_prefix}_interval_min": polynomial["interval"]["minimum"],
                    f"{runtime_prefix}_interval_max": polynomial["interval"]["maximum"],
                    f"{runtime_prefix}_required_depth": polynomial["required_depth"],
                    f"{runtime_prefix}_estimated_multiplications": polynomial[
                        "estimated_multiplications"
                    ],
                }
            )
        for key, expected in expected_pairs.items():
            actual = record.get(key)
            expected_is_integer = isinstance(expected, int) and not isinstance(
                expected, bool
            )
            integer_type_mismatch = expected_is_integer and (
                not isinstance(actual, int) or isinstance(actual, bool)
            )
            if integer_type_mismatch or actual != expected:
                raise ValidationError(
                    f"{prefix}: {key} mismatch: expected={expected!r} "
                    f"actual={actual!r}"
                )
        bootstraps = record.get("bootstraps")
        iterations = record.get("bootstrap_iterations")
        if (
            not isinstance(bootstraps, int)
            or isinstance(bootstraps, bool)
            or not isinstance(iterations, int)
            or isinstance(iterations, bool)
            or iterations != bootstraps * effective["bootstrap_iterations"]
        ):
            raise ValidationError(f"{prefix}: logical/iterative bootstrap counts disagree")
        numeric_gates = {
            "gelu_rel_l2": (0.0, contracts["thresholds"]["relative_l2_max"]),
            "ffn_rel_l2": (0.0, contracts["thresholds"]["relative_l2_max"]),
            "softmax_rel_l2": (0.0, contracts["thresholds"]["relative_l2_max"]),
            "layernorm_rel_l2": (0.0, contracts["thresholds"]["relative_l2_max"]),
            "inactive_max_abs": (0.0, contracts["thresholds"]["inactive_max_abs"]),
            "denominator_post_max_abs": (
                0.0,
                contracts["thresholds"]["denominator_post_max_abs"],
            ),
        }
        for key, (minimum, maximum) in numeric_gates.items():
            value = record.get(key)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value < minimum
                or value > maximum
            ):
                raise ValidationError(f"{prefix}: {key} violates [{minimum},{maximum}]")
        for key in (
            "gelu_cosine",
            "ffn_cosine",
            "softmax_cosine",
            "layernorm_cosine",
        ):
            value = record.get(key)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value < contracts["thresholds"]["cosine_min"]
                or value > 1.000000000001
            ):
                raise ValidationError(f"{prefix}: {key} violates the cosine gate")

    metrics_path = artifact_root / "metrics.csv"
    try:
        with metrics_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, UnicodeError, csv.Error) as error:
        raise ValidationError(f"cannot read metrics.csv: {error}") from error
    required_columns = {
        "run",
        "exit_code",
        "elapsed_seconds",
        "peak_rss_kib",
        "inactive_max_abs",
        "softmax_rel_l2",
        "softmax_cosine",
        "bootstraps",
        "bootstrap_iterations",
    }
    if len(rows) != workload["repeat_count"] or not rows:
        raise ValidationError("metrics.csv row count differs from workload.repeat_count")
    if not required_columns.issubset(rows[0]):
        raise ValidationError(
            f"metrics.csv is missing columns: {sorted(required_columns - set(rows[0]))}"
        )
    elapsed_values: list[float] = []
    peak_rss_values: list[int] = []
    for index, (row, runtime) in enumerate(zip(rows, runtime_records), start=1):
        try:
            if int(row["run"]) != index or int(row["exit_code"]) != 0:
                raise ValueError("run sequence or exit code")
            elapsed_seconds = float(row["elapsed_seconds"])
            peak_rss_kib = int(row["peak_rss_kib"])
            if (
                peak_rss_kib <= 0
                or not math.isfinite(elapsed_seconds)
                or elapsed_seconds <= 0.0
            ):
                raise ValueError("resource metrics")
            elapsed_values.append(elapsed_seconds)
            peak_rss_values.append(peak_rss_kib)
            for key in (
                "inactive_max_abs",
                "softmax_rel_l2",
                "softmax_cosine",
                "bootstraps",
                "bootstrap_iterations",
            ):
                if not math.isclose(
                    float(row[key]),
                    float(runtime[key]),
                    rel_tol=1e-12,
                    abs_tol=1e-15,
                ):
                    raise ValueError(f"{key} differs from stdout")
        except (KeyError, TypeError, ValueError) as error:
            raise ValidationError(f"metrics.csv row {index} is invalid: {error}") from error

    manifest_metrics = manifest["metrics"]
    if (
        manifest_metrics["repeat_count"] != workload["repeat_count"]
        or manifest_metrics["successful_repeats"] != workload["repeat_count"]
    ):
        raise ValidationError("manifest repeat metrics do not match the workload")

    expected_timing = {
        "minimum": min(elapsed_values),
        "median": statistics.median(elapsed_values),
        "maximum": max(elapsed_values),
    }
    for key, expected in expected_timing.items():
        actual = manifest_metrics["elapsed_seconds"][key]
        if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-15):
            raise ValidationError(
                f"metrics.elapsed_seconds.{key} differs from metrics.csv"
            )
    if manifest_metrics["peak_rss_kib_max"] != max(peak_rss_values):
        raise ValidationError("metrics.peak_rss_kib_max differs from metrics.csv")

    quality_fields = {
        "gelu_rel_l2_max": ("gelu_rel_l2", max),
        "ffn_rel_l2_max": ("ffn_rel_l2", max),
        "softmax_rel_l2_max": ("softmax_rel_l2", max),
        "layernorm_rel_l2_max": ("layernorm_rel_l2", max),
        "gelu_cosine_min": ("gelu_cosine", min),
        "ffn_cosine_min": ("ffn_cosine", min),
        "softmax_cosine_min": ("softmax_cosine", min),
        "layernorm_cosine_min": ("layernorm_cosine", min),
        "inactive_max_abs_max": ("inactive_max_abs", max),
        "denominator_post_max_abs_max": ("denominator_post_max_abs", max),
    }
    for summary_key, (runtime_key, reducer) in quality_fields.items():
        expected = reducer(record[runtime_key] for record in runtime_records)
        actual = manifest_metrics["quality"][summary_key]
        if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-15):
            raise ValidationError(f"metrics.quality.{summary_key} differs from stdout")

    if manifest_metrics["operation_counts"] != M3_EXPECTED_OPERATION_COUNTS:
        raise ValidationError("manifest operation counts violate the frozen M3 contract")
    maximum_observed_level = max(
        record["max_observed_level"] for record in runtime_records
    )
    if (
        manifest_metrics["max_observed_level_max"] != maximum_observed_level
        or maximum_observed_level > effective["multiplicative_depth"]
    ):
        raise ValidationError("manifest maximum observed level is invalid")

    _verify_checksum_evidence(artifact_root, artifact_records)


def validate_manifest(
    manifest_path: Path,
    manifest: Any,
    schema: dict[str, Any],
    verify_git: bool,
) -> None:
    if isinstance(manifest, dict) and manifest.get("schema_version") == 1:
        raise ValidationError(
            "schema_version=1 is legacy evidence; validate only new schema_version=2 manifests"
        )
    validate_instance(manifest, schema, schema)
    assert isinstance(manifest, dict)

    started_at = _parse_timestamp(manifest["started_at"], "$.started_at")
    finished_at = _parse_timestamp(manifest["finished_at"], "$.finished_at")
    if finished_at < started_at:
        raise ValidationError("finished_at precedes started_at")

    repository_root = Path(manifest["git"]["repository_root"])
    if repository_root.resolve() != REPO_ROOT.resolve():
        raise ValidationError(
            f"manifest repository_root must be {REPO_ROOT}, got {repository_root}"
        )
    _verify_profile(manifest, repository_root)

    input_records: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(manifest["inputs"]):
        if record["path"] in input_records:
            raise ValidationError(f"duplicate input path: {record['path']}")
        input_records[record["path"]] = record
        input_path = _resolve_confined_file(
            repository_root, record["path"], f"inputs[{index}].path"
        )
        _verify_file_record(input_path, record, f"inputs[{index}]")
    _verify_contract_bindings(manifest, repository_root, input_records)
    _verify_m4_go_command_transcript(manifest, manifest_path)

    artifact_root = manifest_path.resolve().parent
    artifact_paths: set[str] = set()
    artifact_records: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(manifest["artifacts"]):
        if record["path"] in artifact_paths:
            raise ValidationError(f"duplicate artifact path: {record['path']}")
        artifact_paths.add(record["path"])
        artifact_records[record["path"]] = record
        artifact_path = _resolve_confined_file(
            artifact_root, record["path"], f"artifacts[{index}].path"
        )
        if artifact_path == manifest_path.resolve():
            raise ValidationError("manifest.json cannot hash itself as an artifact")
        _verify_file_record(artifact_path, record, f"artifacts[{index}]")

    required_artifacts = {
        "stdout.log": "stdout",
        "metrics.csv": "metrics",
        "SHA256SUMS": "checksum",
    }
    if artifact_paths != set(required_artifacts):
        raise ValidationError(
            "artifacts must contain exactly stdout.log, metrics.csv, and SHA256SUMS"
        )
    for path, role in required_artifacts.items():
        record = artifact_records.get(path)
        if record is None or record["role"] != role:
            raise ValidationError(
                f"required artifact {path} must be present with role={role}"
            )

    _verify_runtime_evidence(
        manifest,
        repository_root,
        artifact_root,
        artifact_records,
    )

    _verify_git(manifest, repository_root, verify_git)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--verify-git",
        action="store_true",
        help="verify current local HEAD/branch/clean state and query the recorded remote ref",
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    if arguments.verify_git and arguments.manifest is None:
        raise ValidationError("--verify-git requires --manifest")
    schema = validate_schema(load_json(arguments.schema))
    result: dict[str, Any] = {
        "test": "validate_openfhe_artifact",
        "schema": str(arguments.schema.resolve()),
        "schema_version": 2,
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
        print(f"validate_openfhe_artifact failed: {error}", file=sys.stderr)
        sys.exit(1)
