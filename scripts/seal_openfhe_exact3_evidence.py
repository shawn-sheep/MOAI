#!/usr/bin/env python3
"""Fail-closed launcher and sealer for one OpenFHE exact-three diagnostic run.

This script owns the sole HE child launch.  It snapshots source, configuration,
the executable, and Git/worktree state immediately before launch, runs exactly
one frozen command, requires the same bytes and filesystem identities after the
child exits, independently parses the raw logs, and publishes one six-file
bundle by an atomic directory rename.
"""

from __future__ import annotations

import argparse
import ctypes
import csv
import errno
import fcntl
import hashlib
import io
import json
import math
import os
import re
import shutil
import shlex
import stat
import struct
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "results" / "openfhe"
RAW_NAMES = ("stdout.log", "stderr.log", "time.log")
OUTPUT_NAMES = ("metrics.csv", "manifest.json", "SHA256SUMS")
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+:-]{0,127}$")
EXACT3_MANIFEST_SCHEMA_ID = "moai.openfhe.diagnostic.exact3.schedule-evidence.v2"
EXACT3_MANIFEST_SCHEMA_VERSION = 2
SECURITY_WARNING = (
    "Research reproduction parameters only. Do not claim 128-bit security."
)
PROFILE_WARNING = (
    f'profile=paper_compat security_claim=none warning="{SECURITY_WARNING}"'
)
EXPECTED_RUNTIME_ARGV = (
    "./build-openfhe/openfhe_encoder_12_layer_smoke",
    "--data-root",
    str(REPO_ROOT / "data"),
    "--diagnostic-layer-count",
    "3",
)
TIME_EXECUTABLE = "/usr/bin/time"
RUNTIME_ENVIRONMENT = {
    "PATH": "/usr/bin:/bin",
    "LANG": "C",
    "LC_ALL": "C",
}
PROVENANCE_SNAPSHOT_SEMANTICS = (
    "source, config, executable, trace-scale, and git/worktree provenance "
    "captured immediately before the sole exact3 child launch and rechecked "
    "byte-for-byte and identity-for-identity after child exit"
)
GIT_SNAPSHOT_SEMANTICS = (
    "Git HEAD, branch, porcelain status, tracked diff, and untracked source "
    "manifest captured immediately before the sole exact3 child launch and "
    "rechecked byte-for-byte after child exit"
)
PROFILE_TRANSITION_CANONICALIZATION = "MOAI-json-sort-keys-compact-utf8-v1"
PROFILE_TRANSITION_ALLOWED_JSON_POINTERS = (
    "/validation_status",
    "/m5_schedule_candidate/status",
    "/m5_schedule_candidate/source",
    "/m5_schedule_candidate/formal_schedule_sealed",
    "/m5_schedule_candidate/calibrated_layers",
    "/m5_schedule_candidate/exact3_seal",
    "/feature_packed_layernorm_override/schedule_status",
)
PROFILE_TRANSITION_PRE_STATE_VALUES = {
    "/validation_status": (
        "two-layer live M5 schedule recorded as candidate; formal schedule "
        "unsealed until exact-three succeeds; live M4 regression and 12-layer "
        "MOAI-observability-compatible prototype gates remain required"
    ),
    "/m5_schedule_candidate/status": "candidate_unsealed",
    "/m5_schedule_candidate/source": (
        "reviewed live two-layer metadata calibration output; no sealed "
        "artifact identity claimed"
    ),
    "/m5_schedule_candidate/formal_schedule_sealed": False,
    "/m5_schedule_candidate/calibrated_layers": [0, 1],
    "/feature_packed_layernorm_override/schedule_status": (
        "two-layer live M5 candidate recorded but unsealed; exact-three, live "
        "M4 regression, and 12-layer MOAI-observability-compatible prototype "
        "gates remain required"
    ),
}
PROFILE_TRANSITION_PRE_STATE_ABSENT = (
    "/m5_schedule_candidate/exact3_seal",
)
PROFILE_SCHEDULE_SEALED_STATUS = "sealed"
PROFILE_CANDIDATE_SEALED_STATUS = "sealed_exact3"
PROFILE_SCHEDULE_SEALED_SOURCE = (
    "approved exact-three-layer live OpenFHE schedule evidence"
)
PROFILE_VALIDATION_STATUS_SEALED = (
    "exact-three live M5 metadata schedule sealed; live M4 regression and "
    "full 12-layer MOAI-observability-compatible prototype gates remain required"
)
PROFILE_PRESEAL_TRANSITION = (
    "candidate profile captured by the exact3 sealer before the one-way "
    "schedule-seal transition"
)
PROFILE_TRANSITION_APPLICATION_ORDER = (
    "candidate pre-state and immutable bytes are verified before and after the "
    "sole exact3 child; the fsynced six-file bundle is published before the "
    "one-way profile transition is applied"
)
PROFILE_TRANSITION_POST_STATE_VALUES = {
    "/validation_status": PROFILE_VALIDATION_STATUS_SEALED,
    "/m5_schedule_candidate/status": PROFILE_CANDIDATE_SEALED_STATUS,
    "/m5_schedule_candidate/source": PROFILE_SCHEDULE_SEALED_SOURCE,
    "/m5_schedule_candidate/formal_schedule_sealed": True,
    "/m5_schedule_candidate/calibrated_layers": [0, 1, 2],
    "/feature_packed_layernorm_override/schedule_status": (
        PROFILE_SCHEDULE_SEALED_STATUS
    ),
}
PROFILE_SHA256 = "94f30e628e21f02146ce7ed9820194eabba3820f6e1e17176a31f8c5acf8b0be"
TRACE_SCALE_SOURCE_PATH = "config/openfhe_approximations.json"
TRACE_SCALE_LOCATOR = "operators.layernorm.feature_packed_trace_scale_contract"
TRACE_SCALE_CONTRACT_ID = "layernorm_layer_token_power2_scale_v1"
TRACE_SCALE_CONTRACT_SHA256 = (
    "b493b032e461e15d7436efe7fff5948436afa8e302646daa97db78fa1d59be3e"
)
TRACE_SCALE_VALUES_SHA256 = (
    "9436f05ce80b427de47700d924869b0dd6f13cc515e586446fcd2dc56faec28a"
)
TRACE_SCALE_RAW_VARIANCE_SHA256 = (
    "940f92de81915c2121b427b72827e4ed87d072fda0847266d0cb5521d60f93b6"
)
TRACE_SCALE_SHAPE = (2, 12, 5)
PROFILE_PATH = "config/paper_compat_feature_packed.json"
CONFIG_PATHS = (
    "config/moai_encoder_trace.json",
    "config/moai_trace_channel_scales.json",
    TRACE_SCALE_SOURCE_PATH,
    PROFILE_PATH,
)
SOURCE_FIXED_PATHS = (
    "CMakeLists.txt",
    "scripts/seal_openfhe_exact3_evidence.py",
    "tests/openfhe_encoder_12_layer_smoke.cpp",
    "tests/support/moai_encoder_fixture.cpp",
    "tests/support/moai_encoder_fixture.hpp",
    "tests/support/moai_encoder_plaintext_oracle.cpp",
    "tests/support/moai_encoder_plaintext_oracle.hpp",
)
SOURCE_GLOBS = ("include/moai/openfhe/*.hpp", "src/openfhe/*.cpp")
EXECUTABLE_PATH = "build-openfhe/openfhe_encoder_12_layer_smoke"
GIT_ENVIRONMENT = {
    "PATH": "/usr/bin:/bin",
    "LANG": "C",
    "LC_ALL": "C",
}

MAX_RAW_BYTES = {
    "stdout.log": 2 * 1024 * 1024,
    "stderr.log": 64 * 1024,
    "time.log": 64 * 1024,
}
MAX_STDOUT_LINE_BYTES = 512 * 1024
METADATA_SCALE_TOLERANCE = 1e-3
RELATIVE_L2_MAX = 5e-2
COSINE_MIN = 0.99
INACTIVE_MAX_ABS = 1e-3
M5_PROTOTYPE_CONTRACT_ID = "moai_observability_compatible_prototype_v1"
M5_PROTOTYPE_CONTRACT_STATUS = "user_authorized_2026-08-01"
M5_PROTOTYPE_ACCEPTANCE = {
    "contract_id": M5_PROTOTYPE_CONTRACT_ID,
    "status": M5_PROTOTYPE_CONTRACT_STATUS,
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
        "per_layer_relative_l2_max": RELATIVE_L2_MAX,
        "per_layer_cosine_min": COSINE_MIN,
        "final_relative_l2_max": RELATIVE_L2_MAX,
        "final_cosine_min": COSINE_MIN,
        "inactive_or_cross_lane_max_absolute": INACTIVE_MAX_ABS,
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
EXPECTED_INACTIVE_ZERO_CHECKPOINTS = 28
POLYNOMIAL_INTERVALS = {
    "softmax_shifted_logits": (-16.0, 5.0),
    "softmax_denominator": (0.01, 80.0),
    "ln1_normalized_variance": (0.5, 1536.0),
    "gelu_input": (-80.0, 128.0),
    "ln2_normalized_variance": (0.5, 1536.0),
}
INACTIVE_SENTINEL_INTERVALS = {
    key: POLYNOMIAL_INTERVALS[key]
    for key in (
        "softmax_denominator",
        "ln1_normalized_variance",
        "ln2_normalized_variance",
    )
}
COUNT_KEYS = (
    "rotations",
    "ct_pt_multiplications",
    "ct_ct_multiplications",
    "explicit_rescale_requests",
    "chebyshev_evaluations",
    "estimated_polynomial_multiplications",
    "bootstraps",
    "bootstrap_iterations",
)
LAYER_COUNTS = {
    "rotations": 6300,
    "ct_pt_multiplications": 51885,
    "ct_ct_multiplications": 95,
    "explicit_rescale_requests": 810,
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
ZERO_COUNTS = dict.fromkeys(COUNT_KEYS, 0)
FINAL_COUNTS = {
    "rotations": 18900,
    "ct_pt_multiplications": 155665,
    "ct_ct_multiplications": 285,
    "explicit_rescale_requests": 2440,
    "chebyshev_evaluations": 165,
    "estimated_polynomial_multiplications": 3450,
    "bootstraps": 85,
    "bootstrap_iterations": 170,
}


def _metadata(
    level: int,
    noise_scale_degree: int,
    remaining_levels: int,
    scale_bits: int,
) -> dict[str, int]:
    return {
        "level": level,
        "noise_scale_degree": noise_scale_degree,
        "remaining_levels": remaining_levels,
        "scale_bits": scale_bits,
        "expected_scale_bits": scale_bits,
        "ciphertext_count": 5,
    }


LAYER0_METADATA = {
    "input_metadata": _metadata(29, 1, 18, 50),
    "raw_output_metadata": _metadata(30, 2, 16, 100),
    "softmax_denominator_metadata": _metadata(18, 2, 28, 100),
    "ln1_variance_metadata": _metadata(19, 2, 27, 100),
    "ln2_variance_metadata": _metadata(19, 2, 27, 100),
    "attention_output_metadata": _metadata(40, 2, 6, 100),
    "ln1_output_metadata": _metadata(32, 2, 14, 100),
    "ffn_output_metadata": _metadata(45, 2, 1, 100),
}
LATER_LAYER_METADATA = {
    "input_metadata": _metadata(19, 2, 27, 100),
    "raw_output_metadata": _metadata(30, 2, 16, 100),
    "softmax_denominator_metadata": _metadata(18, 2, 28, 100),
    "ln1_variance_metadata": _metadata(19, 2, 27, 100),
    "ln2_variance_metadata": _metadata(19, 2, 27, 100),
    "attention_output_metadata": _metadata(31, 2, 15, 100),
    "ln1_output_metadata": _metadata(30, 2, 16, 100),
    "ffn_output_metadata": _metadata(43, 2, 3, 100),
}
METADATA_FIELDS = tuple(LAYER0_METADATA)
LAYER0_USED_LEVELS = {
    "input": 29,
    "softmax_denominator": 19,
    "attention_output": 41,
    "ln1_variance": 20,
    "ln1_output": 33,
    "ffn_output": 46,
    "ln2_variance": 20,
    "raw_output": 31,
}
LATER_USED_LEVELS = {
    "input": 20,
    "softmax_denominator": 19,
    "attention_output": 32,
    "ln1_variance": 20,
    "ln1_output": 31,
    "ffn_output": 44,
    "ln2_variance": 20,
    "raw_output": 31,
}
LAYER0_USED_LEVEL_DELTAS = {
    "input_to_softmax_checkpoint_net_recovered": 10,
    "softmax_checkpoint_to_attention_output_consumed": 22,
    "attention_output_to_ln1_checkpoint_net_recovered": 21,
    "ln1_checkpoint_to_ln1_output_consumed": 13,
    "ln1_output_to_ffn_output_consumed": 13,
    "ffn_output_to_ln2_checkpoint_net_recovered": 26,
    "ln2_checkpoint_to_raw_output_consumed": 11,
    "previous_raw_output_to_input_recovered": None,
}
LATER_USED_LEVEL_DELTAS = {
    "input_to_softmax_checkpoint_net_recovered": 1,
    "softmax_checkpoint_to_attention_output_consumed": 13,
    "attention_output_to_ln1_checkpoint_net_recovered": 12,
    "ln1_checkpoint_to_ln1_output_consumed": 11,
    "ln1_output_to_ffn_output_consumed": 13,
    "ffn_output_to_ln2_checkpoint_net_recovered": 24,
    "ln2_checkpoint_to_raw_output_consumed": 11,
    "previous_raw_output_to_input_recovered": 11,
}
QUALITY_KEYS = {"relative_l2", "cosine", "max_absolute"}
LAYER_KEYS = {
    "test",
    "profile",
    "security_claim",
    "layer_id",
    "claim_scope",
    "artifact_eligible",
    "formal_schedule_sealed",
    "metadata_validation_mode",
    "weights_layer_id",
    "handoff_refresh_performed",
    "chain_input_source",
    *METADATA_FIELDS,
    "metadata_used_levels",
    "metadata_used_level_deltas",
    "input_quality",
    "output_quality",
    "exact_trace_diagnostic",
    "inactive_max_abs",
    "inactive_zero_checkpoint_count",
    "inactive_sentinel_max_error",
    "inactive_polynomial_sentinel_ranges",
    "inactive_sentinel_range_status",
    "encrypted_polynomial_input_ranges",
    "refresh_operation_counts",
    "layer_operation_counts",
    "cumulative_operation_counts",
    "range_validation_owner",
    "checkpoint_decryption_owner",
    "server_decryptions",
    "server_plaintext_activations",
    "finite",
    "range_status",
    "diagnostic_gates_passed",
}
SUMMARY_KEYS = {
    "test",
    "profile",
    "security_claim",
    "parameter_sha256",
    "execution_mode",
    "claim_scope",
    "artifact_eligible",
    "formal_schedule_sealed",
    "metadata_validation_mode",
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
    "diagnostic_gates_passed",
}


class SealError(RuntimeError):
    """Raised when raw evidence or repository provenance is unsafe or invalid."""


@dataclass(frozen=True)
class RawSnapshot:
    name: str
    data: bytes
    device: int
    inode: int
    mode: int
    link_count: int
    size: int
    mtime_ns: int
    ctime_ns: int

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.data).hexdigest()


@dataclass(frozen=True)
class ParsedEvidence:
    layers: tuple[dict[str, Any], ...]
    summary: dict[str, Any]


@dataclass(frozen=True)
class TimeEvidence:
    command_text: str
    elapsed_text: str
    elapsed_seconds: float
    peak_rss_kib: int
    exit_status: int
    swaps: int


@dataclass(frozen=True)
class ProvenanceSnapshot:
    source_files: tuple[dict[str, Any], ...]
    config_files: tuple[dict[str, Any], ...]
    source_manifest_sha256: str
    config_manifest_sha256: str
    executable: dict[str, Any]
    trace_scale: dict[str, str]
    profile_transition: dict[str, Any]


@dataclass(frozen=True)
class GitSnapshot:
    manifest: dict[str, Any]
    fingerprint: tuple[str, ...]


@dataclass(frozen=True)
class LaunchEvidence:
    argv: tuple[str, ...]
    started_at: str
    finished_at: str
    returncode: int


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise SealError(f"value cannot be canonicalized as JSON: {error}") from error


def _pretty_json_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise SealError(f"manifest cannot be encoded as JSON: {error}") from error


def _duplicate_rejecting_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SealError(f"duplicate JSON key is forbidden: {key!r}")
        result[key] = value
    return result


def _reject_constant(token: str) -> None:
    raise SealError(f"non-finite JSON number is forbidden: {token}")


def _parse_finite_float(token: str) -> float:
    value = float(token)
    if not math.isfinite(value):
        raise SealError(f"non-finite JSON number is forbidden: {token}")
    return value


def _strict_json_loads(text: str, label: str) -> Any:
    try:
        value = json.loads(
            text,
            object_pairs_hook=_duplicate_rejecting_object,
            parse_constant=_reject_constant,
            parse_float=_parse_finite_float,
        )
    except (json.JSONDecodeError, UnicodeError, ValueError) as error:
        raise SealError(f"{label} is malformed strict JSON: {error}") from error
    return value


def _strict_json_file(path: Path) -> Any:
    try:
        data = path.read_bytes()
    except OSError as error:
        raise SealError(f"cannot read repository JSON file {path}: {error}") from error
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise SealError(f"repository JSON file is not UTF-8: {path}") from error
    return _strict_json_loads(text, str(path))


def _json_pointer_tokens(pointer: str) -> tuple[str, ...]:
    if not pointer.startswith("/") or pointer == "/":
        raise SealError(f"profile transition JSON pointer is malformed: {pointer!r}")
    tokens: list[str] = []
    for encoded in pointer[1:].split("/"):
        token = encoded.replace("~1", "/").replace("~0", "~")
        if not token:
            raise SealError(
                f"profile transition JSON pointer has an empty token: {pointer!r}"
            )
        tokens.append(token)
    return tuple(tokens)


def _json_pointer_parent(
    document: dict[str, Any], pointer: str
) -> tuple[dict[str, Any], str]:
    tokens = _json_pointer_tokens(pointer)
    current: Any = document
    for token in tokens[:-1]:
        if not isinstance(current, dict) or token not in current:
            raise SealError(f"profile transition pointer is missing: {pointer}")
        current = current[token]
    if not isinstance(current, dict):
        raise SealError(f"profile transition pointer parent is not an object: {pointer}")
    return current, tokens[-1]


def _profile_transition_record(
    profile: Any,
    profile_file: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(profile, dict):
        raise SealError("feature profile must be a JSON object")
    acceptance = profile.get("m5_prototype_acceptance")
    if not isinstance(acceptance, dict):
        raise SealError("M5 prototype acceptance contract is missing")
    if acceptance != M5_PROTOTYPE_ACCEPTANCE:
        raise SealError("M5 prototype acceptance contract drifted")
    for pointer, expected in PROFILE_TRANSITION_PRE_STATE_VALUES.items():
        parent, key = _json_pointer_parent(profile, pointer)
        if key not in parent:
            raise SealError(f"candidate profile transition field is missing: {pointer}")
        _require_exact(
            parent[key],
            expected,
            f"candidate profile transition field {pointer}",
        )
    for pointer in PROFILE_TRANSITION_PRE_STATE_ABSENT:
        parent, key = _json_pointer_parent(profile, pointer)
        if key in parent:
            raise SealError(
                f"candidate profile transition field must be absent: {pointer}"
            )

    immutable_sha256 = _profile_immutable_projection_sha256(profile)
    return {
        "path": PROFILE_PATH,
        "canonicalization": PROFILE_TRANSITION_CANONICALIZATION,
        "allowed_json_pointers": list(PROFILE_TRANSITION_ALLOWED_JSON_POINTERS),
        "pre_state": {
            "values": dict(PROFILE_TRANSITION_PRE_STATE_VALUES),
            "absent": list(PROFILE_TRANSITION_PRE_STATE_ABSENT),
        },
        "pre_seal_file_sha256": profile_file["sha256"],
        "pre_seal_size_bytes": profile_file["size_bytes"],
        "immutable_projection_sha256": immutable_sha256,
        "application_order": PROFILE_TRANSITION_APPLICATION_ORDER,
        "post_state_contract": {
            "values": dict(PROFILE_TRANSITION_POST_STATE_VALUES),
            "exact3_seal_pointer": "/m5_schedule_candidate/exact3_seal",
            "exact3_seal_keys": [
                "source",
                "evidence",
                "pre_seal_profile",
            ],
        },
    }


def _profile_immutable_projection_sha256(profile: Any) -> str:
    if not isinstance(profile, dict):
        raise SealError("feature profile must be a JSON object")
    projection = _strict_json_loads(
        _canonical_json_bytes(profile).decode("utf-8"),
        "profile immutable projection",
    )
    if not isinstance(projection, dict):  # pragma: no cover - guarded above
        raise SealError("profile immutable projection is not an object")
    for pointer in PROFILE_TRANSITION_ALLOWED_JSON_POINTERS:
        parent, key = _json_pointer_parent(projection, pointer)
        parent.pop(key, None)
    return hashlib.sha256(_canonical_json_bytes(projection)).hexdigest()


def _escaped_json_pointer_token(token: str) -> str:
    return token.replace("~", "~0").replace("/", "~1")


def _profile_changed_pointers(
    before: Any,
    after: Any,
    pointer: str = "",
) -> set[str]:
    if type(before) is not type(after):
        return {pointer}
    if isinstance(before, dict):
        changed: set[str] = set()
        for key in set(before) | set(after):
            child_pointer = f"{pointer}/{_escaped_json_pointer_token(key)}"
            if key not in before or key not in after:
                changed.add(child_pointer)
            else:
                changed.update(
                    _profile_changed_pointers(
                        before[key],
                        after[key],
                        child_pointer,
                    )
                )
        return changed
    if _canonical_json_bytes(before) != _canonical_json_bytes(after):
        return {pointer}
    return set()


def _validate_profile_evidence(value: Any) -> dict[str, Any]:
    evidence = _require_exact_keys(
        value,
        {
            "run_id",
            "relative_path",
            "manifest_sha256",
            "sha256sums_sha256",
            "layer_count",
            "artifact_eligible",
            "exact3_gate_passed",
            "schedule_evidence_eligible",
            "formal_schedule_sealed",
        },
        "exact3 profile evidence",
    )
    run_id = evidence["run_id"]
    if not isinstance(run_id, str) or RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise SealError("exact3 profile evidence run_id is malformed")
    _require_exact(
        evidence["relative_path"],
        f"results/openfhe/{run_id}",
        "exact3 profile evidence relative_path",
    )
    for key in ("manifest_sha256", "sha256sums_sha256"):
        digest = evidence[key]
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise SealError(f"exact3 profile evidence {key} is malformed")
    expected = {
        "layer_count": 3,
        "artifact_eligible": False,
        "exact3_gate_passed": True,
        "schedule_evidence_eligible": True,
        "formal_schedule_sealed": True,
    }
    for key, expected_value in expected.items():
        _require_exact(
            evidence[key],
            expected_value,
            f"exact3 profile evidence {key}",
        )
    return dict(evidence)


def _sealed_profile_value(
    candidate_profile: Any,
    transition: dict[str, Any],
    evidence: Any,
) -> dict[str, Any]:
    if not isinstance(candidate_profile, dict):
        raise SealError("candidate feature profile must be a JSON object")
    approved_evidence = _validate_profile_evidence(evidence)
    sealed = _strict_json_loads(
        _canonical_json_bytes(candidate_profile).decode("utf-8"),
        "sealed profile staging value",
    )
    if not isinstance(sealed, dict):  # pragma: no cover - guarded above
        raise SealError("sealed profile staging value is not an object")
    for pointer, value in PROFILE_TRANSITION_POST_STATE_VALUES.items():
        parent, key = _json_pointer_parent(sealed, pointer)
        if key not in parent:
            raise SealError(f"sealed profile target field is missing: {pointer}")
        parent[key] = value
    candidate = sealed.get("m5_schedule_candidate")
    if not isinstance(candidate, dict):
        raise SealError("sealed profile m5_schedule_candidate is not an object")
    candidate["exact3_seal"] = {
        "source": PROFILE_SCHEDULE_SEALED_SOURCE,
        "evidence": approved_evidence,
        "pre_seal_profile": {
            "path": PROFILE_PATH,
            "sha256": transition["pre_seal_file_sha256"],
            "size_bytes": transition["pre_seal_size_bytes"],
            "transition": PROFILE_PRESEAL_TRANSITION,
        },
    }
    changed = _profile_changed_pointers(candidate_profile, sealed)
    expected_changed = set(PROFILE_TRANSITION_ALLOWED_JSON_POINTERS)
    if changed != expected_changed:
        raise SealError(
            "sealed profile changed pointers outside the exact transition contract: "
            f"expected={sorted(expected_changed)!r} actual={sorted(changed)!r}"
        )
    before_projection = _profile_immutable_projection_sha256(candidate_profile)
    after_projection = _profile_immutable_projection_sha256(sealed)
    if (
        before_projection != transition["immutable_projection_sha256"]
        or after_projection != transition["immutable_projection_sha256"]
    ):
        raise SealError("sealed profile immutable projection changed")
    return sealed


def _json_literal(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise SealError(f"profile transition value cannot be encoded: {error}") from error


def _replace_profile_line_once(
    data: bytes,
    before: str,
    after: str,
    label: str,
) -> bytes:
    try:
        before_bytes = before.encode("utf-8")
        after_bytes = after.encode("utf-8")
    except UnicodeEncodeError as error:  # pragma: no cover - constants are UTF-8
        raise SealError(f"profile transition line is not UTF-8: {label}") from error
    if data.count(before_bytes) != 1:
        raise SealError(
            f"candidate profile textual contract drifted for {label}; "
            "refusing a broad rewrite"
        )
    return data.replace(before_bytes, after_bytes, 1)


def _exact3_seal_block(value: dict[str, Any]) -> str:
    try:
        lines = json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ).splitlines()
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise SealError(f"exact3 profile seal cannot be encoded: {error}") from error
    if len(lines) < 2 or lines[0] != "{" or lines[-1] != "}":
        raise SealError("exact3 profile seal renderer produced an invalid object")
    rendered = [f'    "exact3_seal": {lines[0]}']
    rendered.extend(f"    {line}" for line in lines[1:])
    rendered[-1] += ","
    return "\n".join(rendered) + "\n"


def _render_sealed_profile_bytes(
    candidate_bytes: bytes,
    candidate: dict[str, Any],
    sealed: dict[str, Any],
) -> bytes:
    """Patch only the seven authorized JSON paths in the frozen source text."""

    rendered = candidate_bytes
    replacements = (
        (
            "  ",
            "validation_status",
            PROFILE_TRANSITION_PRE_STATE_VALUES["/validation_status"],
            PROFILE_TRANSITION_POST_STATE_VALUES["/validation_status"],
        ),
        (
            "    ",
            "status",
            PROFILE_TRANSITION_PRE_STATE_VALUES["/m5_schedule_candidate/status"],
            PROFILE_TRANSITION_POST_STATE_VALUES["/m5_schedule_candidate/status"],
        ),
        (
            "    ",
            "source",
            PROFILE_TRANSITION_PRE_STATE_VALUES["/m5_schedule_candidate/source"],
            PROFILE_TRANSITION_POST_STATE_VALUES["/m5_schedule_candidate/source"],
        ),
        (
            "    ",
            "formal_schedule_sealed",
            PROFILE_TRANSITION_PRE_STATE_VALUES[
                "/m5_schedule_candidate/formal_schedule_sealed"
            ],
            PROFILE_TRANSITION_POST_STATE_VALUES[
                "/m5_schedule_candidate/formal_schedule_sealed"
            ],
        ),
        (
            "    ",
            "calibrated_layers",
            PROFILE_TRANSITION_PRE_STATE_VALUES[
                "/m5_schedule_candidate/calibrated_layers"
            ],
            PROFILE_TRANSITION_POST_STATE_VALUES[
                "/m5_schedule_candidate/calibrated_layers"
            ],
        ),
        (
            "    ",
            "schedule_status",
            PROFILE_TRANSITION_PRE_STATE_VALUES[
                "/feature_packed_layernorm_override/schedule_status"
            ],
            PROFILE_TRANSITION_POST_STATE_VALUES[
                "/feature_packed_layernorm_override/schedule_status"
            ],
        ),
    )
    for indentation, key, old_value, new_value in replacements:
        before = f'{indentation}"{key}": {_json_literal(old_value)},\n'
        after = f'{indentation}"{key}": {_json_literal(new_value)},\n'
        if key == "calibrated_layers":
            seal = sealed["m5_schedule_candidate"]["exact3_seal"]
            if not isinstance(seal, dict):  # pragma: no cover - constructed above
                raise SealError("sealed profile exact3_seal is not an object")
            after += _exact3_seal_block(seal)
        rendered = _replace_profile_line_once(
            rendered,
            before,
            after,
            key,
        )
    try:
        rendered_text = rendered.decode("utf-8")
    except UnicodeDecodeError as error:
        raise SealError("rendered sealed profile is not UTF-8") from error
    parsed = _strict_json_loads(rendered_text, "rendered sealed feature profile")
    if _canonical_json_bytes(parsed) != _canonical_json_bytes(sealed):
        raise SealError("textual profile transition differs from the sealed JSON value")
    if _profile_changed_pointers(candidate, parsed) != set(
        PROFILE_TRANSITION_ALLOWED_JSON_POINTERS
    ):
        raise SealError("textual profile transition changed an unauthorized JSON path")
    return rendered


def _require_exact_keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        actual = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise SealError(
            f"{label} keys differ: expected={sorted(expected)!r} actual={actual!r}"
        )
    return value


def _require_exact(value: Any, expected: Any, label: str) -> None:
    if type(value) is not type(expected) or value != expected:
        raise SealError(f"{label} must be exactly {expected!r}, got {value!r}")


def _finite_number(value: Any, label: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SealError(f"{label} must be a finite number")
    converted = float(value)
    if not math.isfinite(converted):
        raise SealError(f"{label} must be finite")
    if minimum is not None and converted < minimum:
        raise SealError(f"{label} must be at least {minimum}")
    return converted


def _validate_quality(value: Any, label: str) -> dict[str, Any]:
    quality = _require_exact_keys(value, QUALITY_KEYS, label)
    relative_l2 = _finite_number(
        quality["relative_l2"], f"{label}.relative_l2", minimum=0
    )
    cosine = _finite_number(quality["cosine"], f"{label}.cosine")
    maximum_absolute = _finite_number(
        quality["max_absolute"], f"{label}.max_absolute", minimum=0
    )
    if relative_l2 > RELATIVE_L2_MAX:
        raise SealError(f"{label}.relative_l2 exceeds {RELATIVE_L2_MAX}")
    if cosine < COSINE_MIN or cosine > 1.000000000001:
        raise SealError(f"{label}.cosine is outside [{COSINE_MIN}, 1]")
    if not math.isfinite(maximum_absolute):  # pragma: no cover - guarded above
        raise SealError(f"{label}.max_absolute must be finite")
    return quality


def _validate_metadata(value: Any, expected: dict[str, int], label: str) -> None:
    metadata = _require_exact_keys(value, set(expected), label)
    for key in (
        "level",
        "noise_scale_degree",
        "remaining_levels",
        "expected_scale_bits",
        "ciphertext_count",
    ):
        _require_exact(metadata[key], expected[key], f"{label}.{key}")
    scale_bits = _finite_number(metadata["scale_bits"], f"{label}.scale_bits")
    if abs(scale_bits - expected["expected_scale_bits"]) > METADATA_SCALE_TOLERANCE:
        raise SealError(f"{label}.scale_bits differs from its frozen scale")
    used = metadata["level"] + metadata["noise_scale_degree"] - 1
    if used >= 47 or metadata["remaining_levels"] != 47 - used:
        raise SealError(f"{label} violates the depth-47 used-level formula")


def _validate_counts(value: Any, expected: dict[str, int], label: str) -> None:
    counts = _require_exact_keys(value, set(COUNT_KEYS), label)
    for key in COUNT_KEYS:
        _require_exact(counts[key], expected[key], f"{label}.{key}")


def _scaled_counts(value: dict[str, int], multiplier: int) -> dict[str, int]:
    return {key: value[key] * multiplier for key in COUNT_KEYS}


def _added_counts(*values: dict[str, int]) -> dict[str, int]:
    return {key: sum(value[key] for value in values) for key in COUNT_KEYS}


def _validate_range_map(
    value: Any,
    intervals: dict[str, tuple[float, float]],
    label: str,
) -> dict[str, Any]:
    ranges = _require_exact_keys(value, set(intervals), label)
    for name, (declared_minimum, declared_maximum) in intervals.items():
        observed = ranges[name]
        if not isinstance(observed, list) or len(observed) != 2:
            raise SealError(f"{label}.{name} must be a two-number array")
        minimum = _finite_number(observed[0], f"{label}.{name}[0]")
        maximum = _finite_number(observed[1], f"{label}.{name}[1]")
        if (
            minimum > maximum
            or minimum < declared_minimum
            or maximum > declared_maximum
        ):
            raise SealError(
                f"{label}.{name} escaped [{declared_minimum}, {declared_maximum}]"
            )
    return ranges


def _validate_layer(record: Any, layer_id: int) -> dict[str, Any]:
    layer = _require_exact_keys(record, LAYER_KEYS, f"stdout layer {layer_id}")
    expected_scalars = {
        "test": "openfhe_encoder_exact_prefix_layer",
        "profile": "paper_compat",
        "security_claim": "none",
        "layer_id": layer_id,
        "claim_scope": "diagnostic_prefix_exact_schedule",
        "artifact_eligible": False,
        "formal_schedule_sealed": False,
        "metadata_validation_mode": "exact",
        "weights_layer_id": layer_id,
        "handoff_refresh_performed": layer_id < 2,
        "chain_input_source": (
            "client_encrypted_trace_input"
            if layer_id == 0
            else "previous_ciphertext_output_after_refresh"
        ),
        "range_validation_owner": "client",
        "checkpoint_decryption_owner": "client",
        "server_decryptions": 0,
        "server_plaintext_activations": False,
        "finite": True,
        "range_status": "passed",
        "diagnostic_gates_passed": True,
        "inactive_zero_checkpoint_count": EXPECTED_INACTIVE_ZERO_CHECKPOINTS,
        "inactive_sentinel_range_status": "passed",
    }
    for key, expected in expected_scalars.items():
        _require_exact(layer[key], expected, f"stdout layer {layer_id}.{key}")

    expected_metadata = LAYER0_METADATA if layer_id == 0 else LATER_LAYER_METADATA
    for name, expected in expected_metadata.items():
        _validate_metadata(layer[name], expected, f"stdout layer {layer_id}.{name}")
    expected_used = LAYER0_USED_LEVELS if layer_id == 0 else LATER_USED_LEVELS
    _require_exact(
        layer["metadata_used_levels"],
        expected_used,
        f"stdout layer {layer_id}.metadata_used_levels",
    )
    expected_deltas = (
        LAYER0_USED_LEVEL_DELTAS if layer_id == 0 else LATER_USED_LEVEL_DELTAS
    )
    _require_exact(
        layer["metadata_used_level_deltas"],
        expected_deltas,
        f"stdout layer {layer_id}.metadata_used_level_deltas",
    )

    for quality_name in ("input_quality", "output_quality", "exact_trace_diagnostic"):
        _validate_quality(
            layer[quality_name], f"stdout layer {layer_id}.{quality_name}"
        )
    inactive_maximum = _finite_number(
        layer["inactive_max_abs"],
        f"stdout layer {layer_id}.inactive_max_abs",
        minimum=0,
    )
    if inactive_maximum > INACTIVE_MAX_ABS:
        raise SealError(
            f"stdout layer {layer_id}.inactive_max_abs exceeds "
            f"{INACTIVE_MAX_ABS:.17g}"
        )
    sentinel_maximum = _finite_number(
        layer["inactive_sentinel_max_error"],
        f"stdout layer {layer_id}.inactive_sentinel_max_error",
        minimum=0,
    )
    sentinel_ranges = _validate_range_map(
        layer["inactive_polynomial_sentinel_ranges"],
        INACTIVE_SENTINEL_INTERVALS,
        f"stdout layer {layer_id}.inactive_polynomial_sentinel_ranges",
    )
    derived_sentinel_maximum = max(
        abs(endpoint - 1.0)
        for observed in sentinel_ranges.values()
        for endpoint in observed
    )
    if not math.isclose(
        sentinel_maximum,
        derived_sentinel_maximum,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise SealError(
            f"stdout layer {layer_id}.inactive_sentinel_max_error is inconsistent"
        )
    _validate_range_map(
        layer["encrypted_polynomial_input_ranges"],
        POLYNOMIAL_INTERVALS,
        f"stdout layer {layer_id}.encrypted_polynomial_input_ranges",
    )

    refresh = REFRESH_COUNTS if layer_id < 2 else ZERO_COUNTS
    _validate_counts(
        layer["refresh_operation_counts"],
        refresh,
        f"stdout layer {layer_id}.refresh_operation_counts",
    )
    _validate_counts(
        layer["layer_operation_counts"],
        LAYER_COUNTS,
        f"stdout layer {layer_id}.layer_operation_counts",
    )
    refresh_count = min(layer_id + 1, 2)
    expected_cumulative = _added_counts(
        _scaled_counts(LAYER_COUNTS, layer_id + 1),
        _scaled_counts(REFRESH_COUNTS, refresh_count),
    )
    _validate_counts(
        layer["cumulative_operation_counts"],
        expected_cumulative,
        f"stdout layer {layer_id}.cumulative_operation_counts",
    )
    return layer


def _validate_summary(value: Any, layers: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    summary = _require_exact_keys(value, SUMMARY_KEYS, "stdout final summary")
    expected_scalars = {
        "test": "openfhe_encoder_exact_prefix",
        "profile": "paper_compat",
        "security_claim": "none",
        "parameter_sha256": PROFILE_SHA256,
        "execution_mode": "server-only",
        "claim_scope": "diagnostic_prefix_exact_schedule",
        "artifact_eligible": False,
        "formal_schedule_sealed": True,
        "metadata_validation_mode": "exact",
        "encoder_layers": 3,
        "chain_mode": "ciphertext_output_to_next_input",
        "client_encrypt_calls": 1,
        "encrypted_input_ciphertexts": 5,
        "plaintext_activation_resets": 0,
        "server_layer_evaluations": 3,
        "inter_layer_refreshes": 2,
        "checkpoint_decryption_owner": "client",
        "final_decryption_owner": "client",
        "server_private_key_present": False,
        "server_decryptions": 0,
        "server_plaintext_activations": False,
        "approximation_range_status": "all_client_validated",
        "inactive_sentinel_range_status": "all_client_validated",
        "multiplicative_depth": 47,
        "max_observed_level": 45,
        "max_polynomial_depth": 10,
        "timing_claim": False,
        "latency_kind": "non_benchmark_diagnostic",
        "finite": True,
        "passed": True,
        "diagnostic_gates_passed": True,
    }
    for key, expected in expected_scalars.items():
        _require_exact(summary[key], expected, f"stdout final summary.{key}")

    for field in (
        "fixture_load_oracle_ms",
        "setup_keygen_ms",
        "client_encrypt_ms",
        "client_checkpoint_validate_ms",
    ):
        _finite_number(summary[field], f"stdout final summary.{field}", minimum=0)
    _finite_number(
        summary["server_online_diagnostic_ms"],
        "stdout final summary.server_online_diagnostic_ms",
        minimum=0,
    )
    if summary["server_online_diagnostic_ms"] <= 0:
        raise SealError(
            "stdout final summary.server_online_diagnostic_ms must be positive"
        )

    final_quality = _validate_quality(
        {
            "relative_l2": summary["relative_l2"],
            "cosine": summary["cosine"],
            "max_absolute": summary["max_absolute"],
        },
        "stdout final summary quality",
    )
    if final_quality != layers[2]["output_quality"]:
        raise SealError("stdout final quality differs from layer 2 output_quality")
    inactive_maximum = _finite_number(
        summary["inactive_max_abs"],
        "stdout final summary.inactive_max_abs",
        minimum=0,
    )
    if inactive_maximum > INACTIVE_MAX_ABS:
        raise SealError(
            "stdout final summary.inactive_max_abs exceeds "
            f"{INACTIVE_MAX_ABS:.17g}"
        )
    if inactive_maximum != max(layer["inactive_max_abs"] for layer in layers):
        raise SealError("stdout final inactive_max_abs differs from the layer maximum")
    sentinel_maximum = _finite_number(
        summary["inactive_sentinel_max_error"],
        "stdout final summary.inactive_sentinel_max_error",
        minimum=0,
    )
    if sentinel_maximum != max(
        layer["inactive_sentinel_max_error"] for layer in layers
    ):
        raise SealError(
            "stdout final inactive_sentinel_max_error differs from the layer maximum"
        )
    _validate_metadata(
        summary["final_metadata"],
        LATER_LAYER_METADATA["raw_output_metadata"],
        "stdout final summary.final_metadata",
    )
    if summary["final_metadata"] != layers[2]["raw_output_metadata"]:
        raise SealError("stdout final metadata differs from layer 2 raw output")
    _validate_counts(
        summary["operation_counts"],
        FINAL_COUNTS,
        "stdout final summary.operation_counts",
    )
    _finite_number(
        summary["peak_rss_bytes"],
        "stdout final summary.peak_rss_bytes",
        minimum=1,
    )
    return summary


def _parse_stdout(data: bytes) -> ParsedEvidence:
    if not data:
        raise SealError("stdout.log is empty")
    if not data.endswith(b"\n") or b"\r" in data or b"\x00" in data:
        raise SealError("stdout.log must be NUL-free UTF-8 with LF line endings")
    lines_bytes = data[:-1].split(b"\n")
    if len(lines_bytes) != 5 or any(not line for line in lines_bytes):
        raise SealError(
            "stdout.log must contain one warning and exactly four JSON lines"
        )
    if any(len(line) > MAX_STDOUT_LINE_BYTES for line in lines_bytes):
        raise SealError("stdout.log contains an over-sized line")
    try:
        lines = [line.decode("utf-8") for line in lines_bytes]
    except UnicodeDecodeError as error:
        raise SealError("stdout.log is not UTF-8") from error
    if lines[0] != PROFILE_WARNING:
        raise SealError("stdout.log first line is not the frozen profile warning")
    records: list[dict[str, Any]] = []
    for index, line in enumerate(lines[1:], start=2):
        if not line.startswith("{") or not line.endswith("}"):
            raise SealError(f"stdout.log line {index} is not a JSON object")
        value = _strict_json_loads(line, f"stdout.log line {index}")
        if not isinstance(value, dict):
            raise SealError(f"stdout.log line {index} must be a JSON object")
        records.append(value)
    layers = tuple(_validate_layer(records[index], index) for index in range(3))
    if (
        layers[1]["metadata_used_level_deltas"][
            "previous_raw_output_to_input_recovered"
        ]
        != 11
        or layers[2]["metadata_used_level_deltas"][
            "previous_raw_output_to_input_recovered"
        ]
        != 11
    ):
        raise SealError(
            "the two exact-prefix handoffs must each recover 11 used levels"
        )
    for field in METADATA_FIELDS:
        for key in set(LATER_LAYER_METADATA[field]) - {"scale_bits"}:
            if layers[1][field][key] != layers[2][field][key]:
                raise SealError(
                    f"layer 2 is not in the later-layer steady-state for {field}.{key}"
                )
    summary = _validate_summary(records[3], layers)
    return ParsedEvidence(layers, summary)


def _parse_elapsed(value: str) -> float:
    parts = value.split(":")
    if len(parts) not in (2, 3) or any(not part for part in parts):
        raise SealError(f"GNU time elapsed value is malformed: {value!r}")
    try:
        seconds = float(parts[-1])
        minutes = int(parts[-2])
        hours = int(parts[0]) if len(parts) == 3 else 0
    except ValueError as error:
        raise SealError(f"GNU time elapsed value is malformed: {value!r}") from error
    if (
        not math.isfinite(seconds)
        or seconds < 0
        or seconds >= 60
        or minutes < 0
        or (len(parts) == 3 and minutes >= 60)
        or hours < 0
    ):
        raise SealError(f"GNU time elapsed value is out of range: {value!r}")
    elapsed = hours * 3600 + minutes * 60 + seconds
    if elapsed <= 0 or not math.isfinite(elapsed):
        raise SealError("GNU time elapsed duration must be positive and finite")
    return elapsed


def _parse_time(data: bytes) -> TimeEvidence:
    if not data:
        raise SealError("time.log is empty")
    if not data.endswith(b"\n") or b"\r" in data or b"\x00" in data:
        raise SealError("time.log must be NUL-free UTF-8 with LF line endings")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise SealError("time.log is not UTF-8") from error
    lines = text.splitlines()
    if len(lines) > 64:
        raise SealError("time.log contains too many lines")
    fields = {
        "Command being timed": None,
        "Elapsed (wall clock) time (h:mm:ss or m:ss)": None,
        "Maximum resident set size (kbytes)": None,
        "Swaps": None,
        "Exit status": None,
    }
    for line in lines:
        stripped = line.strip()
        for key in fields:
            prefix = f"{key}: "
            if stripped.startswith(prefix):
                if fields[key] is not None:
                    raise SealError(f"time.log repeats GNU time field {key!r}")
                fields[key] = stripped[len(prefix) :]
    missing = [key for key, value in fields.items() if value is None]
    if missing:
        raise SealError(f"time.log is missing GNU time fields: {missing!r}")

    command_field = fields["Command being timed"]
    assert command_field is not None
    if len(command_field) < 2 or command_field[0] != '"' or command_field[-1] != '"':
        raise SealError("GNU time command must use its quoted verbose format")
    command_text = command_field[1:-1]
    try:
        command_argv = tuple(shlex.split(command_text))
    except ValueError as error:
        raise SealError("GNU time command is not shell-parseable") from error
    if command_argv != EXPECTED_RUNTIME_ARGV:
        raise SealError(
            f"GNU time command differs from the exact3 command: {command_argv!r}"
        )

    elapsed_text = fields["Elapsed (wall clock) time (h:mm:ss or m:ss)"]
    peak_text = fields["Maximum resident set size (kbytes)"]
    swaps_text = fields["Swaps"]
    exit_text = fields["Exit status"]
    assert elapsed_text is not None
    assert peak_text is not None
    assert swaps_text is not None
    assert exit_text is not None
    elapsed_seconds = _parse_elapsed(elapsed_text)
    try:
        peak_rss_kib = int(peak_text)
        swaps = int(swaps_text)
        exit_status = int(exit_text)
    except ValueError as error:
        raise SealError("GNU time integer field is malformed") from error
    if peak_rss_kib <= 0:
        raise SealError("GNU time peak RSS must be positive")
    if swaps != 0:
        raise SealError("GNU time Swaps must be exactly 0")
    if exit_status != 0:
        raise SealError("GNU time Exit status must be exactly 0")
    return TimeEvidence(
        command_text,
        elapsed_text,
        elapsed_seconds,
        peak_rss_kib,
        exit_status,
        swaps,
    )


def _snapshot_from_stat(name: str, data: bytes, result: os.stat_result) -> RawSnapshot:
    return RawSnapshot(
        name=name,
        data=data,
        device=result.st_dev,
        inode=result.st_ino,
        mode=result.st_mode,
        link_count=result.st_nlink,
        size=result.st_size,
        mtime_ns=result.st_mtime_ns,
        ctime_ns=result.st_ctime_ns,
    )


def _read_raw_file(directory_fd: int, name: str) -> RawSnapshot:
    try:
        before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except OSError as error:
        raise SealError(f"cannot stat raw log {name}: {error}") from error
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise SealError(f"raw log must be one ordinary, non-linked file: {name}")
    maximum = MAX_RAW_BYTES[name]
    if before.st_size > maximum:
        raise SealError(f"raw log exceeds {maximum} bytes: {name}")
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        file_descriptor = os.open(name, flags, dir_fd=directory_fd)
    except OSError as error:
        raise SealError(f"cannot safely open raw log {name}: {error}") from error
    try:
        opened = os.fstat(file_descriptor)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_nlink,
        )
        identity_opened = (
            opened.st_dev,
            opened.st_ino,
            opened.st_mode,
            opened.st_nlink,
        )
        if identity_before != identity_opened:
            raise SealError(f"raw log changed while opening: {name}")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(file_descriptor, min(1024 * 1024, maximum + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > maximum:
                raise SealError(f"raw log exceeds {maximum} bytes: {name}")
        after = os.fstat(file_descriptor)
        stable_fields = (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_nlink",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        if any(
            getattr(opened, field) != getattr(after, field) for field in stable_fields
        ):
            raise SealError(f"raw log changed while being read: {name}")
        data = b"".join(chunks)
        if len(data) != after.st_size:
            raise SealError(f"raw log size changed while being read: {name}")
        return _snapshot_from_stat(name, data, after)
    finally:
        os.close(file_descriptor)


def _require_initial_directory(directory_fd: int) -> None:
    try:
        names = set(os.listdir(directory_fd))
    except OSError as error:
        raise SealError(f"cannot list run directory: {error}") from error
    expected = set(RAW_NAMES)
    if names != expected:
        raise SealError(
            f"run directory must initially contain only {sorted(expected)!r}, "
            f"got {sorted(names)!r}"
        )


def _open_run_root(
    run_root: Path,
    *,
    require_public_run_id: bool = True,
) -> tuple[Path, int]:
    output_root = OUTPUT_ROOT.resolve(strict=True)
    lexical = Path(os.path.abspath(os.fspath(run_root)))
    if lexical.parent != output_root or (
        require_public_run_id and RUN_ID_PATTERN.fullmatch(lexical.name) is None
    ):
        raise SealError(
            f"--run-root must be one direct results/openfhe/<run-id> directory: {run_root}"
        )
    try:
        resolved = lexical.resolve(strict=True)
        root_stat = lexical.lstat()
    except OSError as error:
        raise SealError(f"cannot resolve run root {run_root}: {error}") from error
    if resolved != lexical or resolved.parent != output_root:
        raise SealError(f"run root must not use symlinks or escape {output_root}")
    if not stat.S_ISDIR(root_stat.st_mode) or stat.S_ISLNK(root_stat.st_mode):
        raise SealError(f"run root is not an ordinary directory: {run_root}")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(lexical, flags)
    except OSError as error:
        raise SealError(f"cannot safely open run root {run_root}: {error}") from error
    opened = os.fstat(descriptor)
    if (
        opened.st_dev != root_stat.st_dev
        or opened.st_ino != root_stat.st_ino
        or not stat.S_ISDIR(opened.st_mode)
    ):
        os.close(descriptor)
        raise SealError("run root changed while being opened")
    return lexical, descriptor


def _verify_run_root_identity(run_root: Path, directory_fd: int) -> None:
    try:
        path_stat = run_root.stat(follow_symlinks=False)
        opened = os.fstat(directory_fd)
        resolved = run_root.resolve(strict=True)
    except OSError as error:
        raise SealError(f"cannot revalidate run root identity: {error}") from error
    if (
        resolved != run_root
        or path_stat.st_dev != opened.st_dev
        or path_stat.st_ino != opened.st_ino
        or not stat.S_ISDIR(path_stat.st_mode)
    ):
        raise SealError("run root path changed during sealing")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise SealError(f"cannot hash {path}: {error}") from error
    return digest.hexdigest()


def _repository_file_record(relative_path: str) -> dict[str, Any]:
    path = REPO_ROOT / relative_path
    try:
        resolved = path.resolve(strict=True)
        before = path.lstat()
    except OSError as error:
        raise SealError(
            f"repository input is missing: {relative_path}: {error}"
        ) from error
    try:
        resolved.relative_to(REPO_ROOT.resolve())
    except ValueError as error:
        raise SealError(
            f"repository input escapes the repository: {relative_path}"
        ) from error
    if resolved != path or not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise SealError(
            f"repository input must be an ordinary non-linked file: {relative_path}"
        )
    digest = _sha256_file(path)
    try:
        after = path.lstat()
    except OSError as error:
        raise SealError(
            f"repository input disappeared while hashing: {relative_path}: {error}"
        ) from error
    identity_fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_nlink",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if any(
        getattr(before, field) != getattr(after, field) for field in identity_fields
    ):
        raise SealError(f"repository input changed while hashing: {relative_path}")
    return {
        "path": relative_path,
        "sha256": digest,
        "size_bytes": after.st_size,
        "device": after.st_dev,
        "inode": after.st_ino,
        "mode": after.st_mode,
        "link_count": after.st_nlink,
        "mtime_ns": after.st_mtime_ns,
        "ctime_ns": after.st_ctime_ns,
    }


def _source_paths() -> tuple[str, ...]:
    paths = set(SOURCE_FIXED_PATHS)
    for pattern in SOURCE_GLOBS:
        matches = [
            path.relative_to(REPO_ROOT).as_posix()
            for path in REPO_ROOT.glob(pattern)
            if path.is_file()
        ]
        if not matches:
            raise SealError(f"source provenance glob has no matches: {pattern}")
        paths.update(matches)
    return tuple(sorted(paths))


def _binary64_tensor_sha256(value: Any, shape: tuple[int, ...], label: str) -> str:
    digest = hashlib.sha256()

    def visit(
        node: Any, dimensions: tuple[int, ...], coordinate: tuple[int, ...]
    ) -> None:
        if not dimensions:
            number = _finite_number(node, f"{label}{coordinate}")
            digest.update(struct.pack("<d", number))
            return
        if not isinstance(node, list) or len(node) != dimensions[0]:
            raise SealError(f"{label}{coordinate} differs from shape {shape}")
        for index, child in enumerate(node):
            visit(child, dimensions[1:], (*coordinate, index))

    visit(value, shape, ())
    return digest.hexdigest()


def _trace_scale_provenance() -> dict[str, str]:
    approximation = _strict_json_file(REPO_ROOT / TRACE_SCALE_SOURCE_PATH)
    try:
        contract = approximation["operators"]["layernorm"][
            "feature_packed_trace_scale_contract"
        ]
    except (KeyError, TypeError) as error:
        raise SealError(
            f"trace-scale locator is missing: {TRACE_SCALE_LOCATOR}"
        ) from error
    if not isinstance(contract, dict):
        raise SealError("trace-scale contract must be a JSON object")
    payload = {
        key: value for key, value in contract.items() if key != "contract_sha256"
    }
    contract_hash = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
    values_hash = _binary64_tensor_sha256(
        contract.get("values"), TRACE_SCALE_SHAPE, "trace-scale values"
    )
    provenance = {
        "source_path": TRACE_SCALE_SOURCE_PATH,
        "json_locator": TRACE_SCALE_LOCATOR,
        "contract_id": TRACE_SCALE_CONTRACT_ID,
        "contract_sha256": contract_hash,
        "values_sha256": values_hash,
        "raw_variance_sha256": TRACE_SCALE_RAW_VARIANCE_SHA256,
    }
    if (
        contract.get("contract_id") != TRACE_SCALE_CONTRACT_ID
        or tuple(contract.get("shape", ())) != TRACE_SCALE_SHAPE
        or contract_hash != TRACE_SCALE_CONTRACT_SHA256
        or contract.get("contract_sha256") != TRACE_SCALE_CONTRACT_SHA256
        or values_hash != TRACE_SCALE_VALUES_SHA256
        or contract.get("values_sha256") != TRACE_SCALE_VALUES_SHA256
        or contract.get("raw_variance_sha256") != TRACE_SCALE_RAW_VARIANCE_SHA256
    ):
        raise SealError("trace-scale identity or one of its three hashes drifted")
    try:
        hard_interval = contract["normalized_variance_range"]["hard_interval"]
        runtime_dependency = contract["scope"]["runtime_activation_dependency"]
        exact_identity_gate = contract["inactive_guard"]["exact_identity_gate"]
    except (KeyError, TypeError) as error:
        raise SealError("trace-scale semantic gate is incomplete") from error
    if (
        hard_interval != [0.5, 1536.0]
        or runtime_dependency != "none"
        or exact_identity_gate is not False
    ):
        raise SealError("trace-scale semantic gate drifted")
    profile = _strict_json_file(REPO_ROOT / PROFILE_PATH)
    try:
        binding = profile["feature_packed_layernorm_override"]["trace_scale_contract"]
    except (KeyError, TypeError) as error:
        raise SealError(
            "feature profile lacks the trace-scale cross-binding"
        ) from error
    if binding != provenance:
        raise SealError("feature profile trace-scale cross-binding drifted")
    return provenance


def _provenance_snapshot() -> ProvenanceSnapshot:
    source_files = tuple(_repository_file_record(path) for path in _source_paths())
    config_files = tuple(_repository_file_record(path) for path in CONFIG_PATHS)
    source_manifest_sha256 = hashlib.sha256(
        _canonical_json_bytes(source_files)
    ).hexdigest()
    config_manifest_sha256 = hashlib.sha256(
        _canonical_json_bytes(config_files)
    ).hexdigest()
    executable = _repository_file_record(EXECUTABLE_PATH)
    executable_path = REPO_ROOT / EXECUTABLE_PATH
    if not os.access(executable_path, os.X_OK):
        raise SealError(f"exact3 executable is not executable: {EXECUTABLE_PATH}")
    profile_file = next(
        record for record in config_files if record["path"] == PROFILE_PATH
    )
    profile_transition = _profile_transition_record(
        _strict_json_file(REPO_ROOT / PROFILE_PATH),
        profile_file,
    )
    return ProvenanceSnapshot(
        source_files,
        config_files,
        source_manifest_sha256,
        config_manifest_sha256,
        executable,
        _trace_scale_provenance(),
        profile_transition,
    )


def _run_git(
    arguments: list[str], *, allow_failure: bool = False
) -> subprocess.CompletedProcess:
    completed = subprocess.run(
        ["/usr/bin/git", *arguments],
        cwd=REPO_ROOT,
        env=GIT_ENVIRONMENT,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0 and not allow_failure:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise SealError(f"git {' '.join(arguments)} failed: {detail}")
    return completed


def _untracked_source_entries(output: bytes) -> tuple[dict[str, Any], ...]:
    paths = output.split(b"\x00")
    if paths and paths[-1] == b"":
        paths.pop()
    entries: list[dict[str, Any]] = []
    for encoded in paths:
        try:
            relative = encoded.decode("utf-8")
        except UnicodeDecodeError as error:
            raise SealError("untracked source path is not UTF-8") from error
        if not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise SealError(f"unsafe untracked source path: {relative!r}")
        entries.append(_repository_file_record(relative))
    return tuple(entries)


def _git_snapshot() -> GitSnapshot:
    head = _run_git(["rev-parse", "HEAD"]).stdout.decode("ascii").strip()
    if re.fullmatch(r"[0-9a-f]{40,64}", head) is None:
        raise SealError(f"git HEAD is malformed: {head!r}")
    branch_result = _run_git(
        ["symbolic-ref", "--quiet", "--short", "HEAD"], allow_failure=True
    )
    if branch_result.returncode not in (0, 1):
        raise SealError("git symbolic-ref failed unexpectedly")
    branch = (
        branch_result.stdout.decode("utf-8").strip()
        if branch_result.returncode == 0
        else None
    )
    status = _run_git(
        ["status", "--porcelain=v1", "-z", "--untracked-files=all"]
    ).stdout
    tracked_diff = _run_git(["diff", "--binary", "HEAD", "--", "."]).stdout
    untracked_output = _run_git(
        [
            "ls-files",
            "-z",
            "--others",
            "--exclude-standard",
            "--",
            "CMakeLists.txt",
            "config",
            "docs",
            "include",
            "scripts",
            "src",
            "tests",
        ]
    ).stdout
    untracked_entries = _untracked_source_entries(untracked_output)
    untracked_hash = hashlib.sha256(
        _canonical_json_bytes(untracked_entries)
    ).hexdigest()
    status_hash = hashlib.sha256(status).hexdigest()
    diff_hash = hashlib.sha256(tracked_diff).hexdigest()
    manifest = {
        "head": head,
        "branch": branch,
        "detached": branch is None,
        "clean": not status,
        "snapshot_semantics": GIT_SNAPSHOT_SEMANTICS,
        "status_porcelain_v1_z_sha256": status_hash,
        "tracked_diff": {
            "command": "git diff --binary HEAD -- .",
            "sha256": diff_hash,
            "size_bytes": len(tracked_diff),
        },
        "untracked_source_manifest": {
            "selection": (
                "git ls-files --others --exclude-standard under "
                "CMakeLists.txt, config, docs, include, scripts, src, tests"
            ),
            "canonicalization": "sorted-key compact UTF-8 JSON",
            "sha256": untracked_hash,
            "entries": list(untracked_entries),
        },
    }
    fingerprint = (
        head,
        "" if branch is None else branch,
        status_hash,
        diff_hash,
        untracked_hash,
    )
    return GitSnapshot(manifest, fingerprint)


def _raw_manifest(snapshot: RawSnapshot) -> dict[str, Any]:
    return {
        "path": snapshot.name,
        "sha256": snapshot.sha256,
        "size_bytes": snapshot.size,
        "filesystem_mtime_ns": snapshot.mtime_ns,
        "filesystem_ctime_ns": snapshot.ctime_ns,
    }


def _metrics_bytes(
    evidence: ParsedEvidence,
    timing: TimeEvidence,
) -> bytes:
    output = io.StringIO(newline="")
    fieldnames = [
        "record_kind",
        "layer_id",
        "relative_l2",
        "cosine",
        "max_absolute",
        "inactive_max_abs",
        "inactive_sentinel_max_error",
        *COUNT_KEYS,
        "elapsed_seconds",
        "peak_rss_bytes",
        "passed",
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for layer_id, layer in enumerate(evidence.layers):
        writer.writerow(
            {
                "record_kind": "layer",
                "layer_id": layer_id,
                **layer["output_quality"],
                "inactive_max_abs": layer["inactive_max_abs"],
                "inactive_sentinel_max_error": layer["inactive_sentinel_max_error"],
                **layer["cumulative_operation_counts"],
                "passed": True,
            }
        )
    summary = evidence.summary
    writer.writerow(
        {
            "record_kind": "final",
            "layer_id": "",
            "relative_l2": summary["relative_l2"],
            "cosine": summary["cosine"],
            "max_absolute": summary["max_absolute"],
            "inactive_max_abs": summary["inactive_max_abs"],
            "inactive_sentinel_max_error": summary["inactive_sentinel_max_error"],
            **summary["operation_counts"],
            "elapsed_seconds": timing.elapsed_seconds,
            "peak_rss_bytes": timing.peak_rss_kib * 1024,
            "passed": True,
        }
    )
    return output.getvalue().encode("utf-8")


def _manifest(
    run_root: Path,
    raw: dict[str, RawSnapshot],
    evidence: ParsedEvidence,
    timing: TimeEvidence,
    launch: LaunchEvidence,
    git: GitSnapshot,
    provenance: ProvenanceSnapshot,
    metrics: bytes,
) -> dict[str, Any]:
    sealed_at = datetime.now().astimezone().isoformat(timespec="seconds")
    return {
        "schema": "diagnostic",
        "schema_id": EXACT3_MANIFEST_SCHEMA_ID,
        "schema_version": EXACT3_MANIFEST_SCHEMA_VERSION,
        "milestone": "M5",
        "artifact_kind": "exact3_schedule_evidence",
        "milestone_artifact_eligible": False,
        "schedule_evidence_eligible": True,
        "exact3_gate_passed": True,
        "formal_schedule_sealed": True,
        "timing_claim": False,
        "security_claim": "none",
        "run_id": run_root.name,
        "timestamp_semantics": {
            "run_id_label": run_root.name,
            "run_id_label_semantics": (
                "operator-assigned directory label; not treated as a measured "
                "process start timestamp"
            ),
            "execution_started_at": launch.started_at,
            "execution_finished_at": launch.finished_at,
            "execution_timestamp_semantics": (
                "launcher wall clocks sampled immediately before the sole child "
                "invocation and immediately after it returned; GNU time supplies "
                "the independently parsed elapsed duration"
            ),
            "raw_log_filesystem_timestamp_semantics": (
                "mtime_ns and ctime_ns are filesystem metadata, not process "
                "start or finish timestamps"
            ),
            "sealed_at": sealed_at,
            "sealed_at_semantics": (
                "sealer wall clock sampled while preparing derived outputs after "
                "initial validation, rounded to whole seconds; not an execution "
                "finish timestamp"
            ),
        },
        "git": git.manifest,
        "command": {
            "argv": list(launch.argv),
            "runtime_argv": list(EXPECTED_RUNTIME_ARGV),
            "gnu_time_reported_command": timing.command_text,
            "cwd": str(REPO_ROOT),
            "environment": dict(RUNTIME_ENVIRONMENT),
            "stdin": "DEVNULL",
            "attempt_count": 1,
            "exit_code": launch.returncode,
            "started_at": launch.started_at,
            "finished_at": launch.finished_at,
        },
        "profile": {
            "id": "paper_compat",
            "parameter_sha256": PROFILE_SHA256,
            "security_claim": "none",
            "warning": SECURITY_WARNING,
        },
        "provenance": {
            "snapshot_semantics": PROVENANCE_SNAPSHOT_SEMANTICS,
            "source_files": list(provenance.source_files),
            "source_manifest_sha256": provenance.source_manifest_sha256,
            "config_files": list(provenance.config_files),
            "config_manifest_sha256": provenance.config_manifest_sha256,
            "executable": provenance.executable,
            "trace_scale": provenance.trace_scale,
            "profile_transition": provenance.profile_transition,
        },
        "raw_logs": {name: _raw_manifest(raw[name]) for name in RAW_NAMES},
        "timing": {
            "kind": "non_benchmark_diagnostic",
            "timing_claim": False,
            "gnu_time_elapsed_text": timing.elapsed_text,
            "elapsed_seconds": timing.elapsed_seconds,
            "peak_rss_kib": timing.peak_rss_kib,
            "peak_rss_bytes": timing.peak_rss_kib * 1024,
            "swaps": timing.swaps,
            "exit_status": timing.exit_status,
        },
        "gate_contract": {
            "prototype_acceptance": M5_PROTOTYPE_ACCEPTANCE,
            "raw_input_bounds": {
                "maximum_bytes": MAX_RAW_BYTES,
                "stdout_nonempty_line_count": 5,
                "stdout_maximum_line_bytes": MAX_STDOUT_LINE_BYTES,
            },
            "quality": {
                "relative_l2_max": RELATIVE_L2_MAX,
                "cosine_min": COSINE_MIN,
                "finite_required": True,
            },
            "inactive": {
                "inactive_max_abs": INACTIVE_MAX_ABS,
                "zero_checkpoint_count": EXPECTED_INACTIVE_ZERO_CHECKPOINTS,
                "sentinel_deviation_from_one": "diagnostic_only",
                "sentinel_ranges_must_stay_in_registered_intervals": True,
            },
            "polynomial_intervals": {
                key: list(value) for key, value in POLYNOMIAL_INTERVALS.items()
            },
        },
        "execution": {
            "mode": "diagnostic_exact_prefix",
            "backend": "OpenFHE CKKS CPU",
            "encoder_layers": 3,
            "layer_ids": [0, 1, 2],
            "chain_mode": "ciphertext_output_to_next_input",
            "inter_layer_refreshes": 2,
            "checkpoint_decryption_owner": "client",
            "final_decryption_owner": "client",
            "server_private_key_present": False,
            "server_decryptions": 0,
            "server_plaintext_activations": False,
            "plaintext_activation_resets": 0,
            "artifact_eligible": False,
            "formal_schedule_sealed": True,
            "diagnostic_gates_passed": True,
        },
        "evidence": {
            "layers": list(evidence.layers),
            "final_summary": evidence.summary,
            "later_layer_steady_state_verified": True,
            "second_handoff_previous_raw_output_to_input_recovered": 11,
        },
        "derived_artifacts": {
            "metrics.csv": {
                "sha256": hashlib.sha256(metrics).hexdigest(),
                "size_bytes": len(metrics),
            }
        },
        "verdict": {
            "exact3_gate_passed": True,
            "schedule_evidence_eligible": True,
            "formal_schedule_sealed": True,
            "milestone_artifact_eligible": False,
            "timing_claim": False,
            "security_claim": "none",
        },
    }


def _write_all(file_descriptor: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = os.write(file_descriptor, view)
        if written <= 0:  # pragma: no cover - defensive OS guard
            raise SealError("short write while staging evidence")
        view = view[written:]


def _repository_record_identity(record: dict[str, Any]) -> tuple[int, ...]:
    try:
        return (
            record["device"],
            record["inode"],
            record["mode"],
            record["link_count"],
            record["size_bytes"],
            record["mtime_ns"],
            record["ctime_ns"],
        )
    except KeyError as error:
        raise SealError(
            f"repository provenance record lacks identity field: {error}"
        ) from error


def _stat_identity(result: os.stat_result) -> tuple[int, ...]:
    return (
        result.st_dev,
        result.st_ino,
        result.st_mode,
        result.st_nlink,
        result.st_size,
        result.st_mtime_ns,
        result.st_ctime_ns,
    )


def _read_expected_repository_file(
    relative_path: str,
    expected: dict[str, Any],
) -> bytes:
    _require_exact(expected.get("path"), relative_path, "repository record path")
    path = REPO_ROOT / relative_path
    try:
        resolved = path.resolve(strict=True)
        before = path.lstat()
    except OSError as error:
        raise SealError(f"cannot inspect repository file {relative_path}: {error}") from error
    if (
        resolved != path
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or _stat_identity(before) != _repository_record_identity(expected)
    ):
        raise SealError(
            f"repository file identity drifted before profile transition: {relative_path}"
        )
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise SealError(
            f"cannot safely open repository file {relative_path}: {error}"
        ) from error
    try:
        opened = os.fstat(descriptor)
        if _stat_identity(opened) != _repository_record_identity(expected):
            raise SealError(
                f"repository file changed while opening: {relative_path}"
            )
        chunks: list[bytes] = []
        remaining = expected["size_bytes"]
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise SealError(
                    f"repository file was truncated while reading: {relative_path}"
                )
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise SealError(
                f"repository file grew while reading: {relative_path}"
            )
        after = os.fstat(descriptor)
        if _stat_identity(after) != _repository_record_identity(expected):
            raise SealError(
                f"repository file changed while reading: {relative_path}"
            )
    finally:
        os.close(descriptor)
    data = b"".join(chunks)
    if hashlib.sha256(data).hexdigest() != expected.get("sha256"):
        raise SealError(f"repository file hash drifted: {relative_path}")
    try:
        path_after = path.lstat()
    except OSError as error:
        raise SealError(
            f"repository file disappeared after reading: {relative_path}: {error}"
        ) from error
    if _stat_identity(path_after) != _repository_record_identity(expected):
        raise SealError(f"repository file path changed after reading: {relative_path}")
    return data


def _profile_file_record(provenance: ProvenanceSnapshot) -> dict[str, Any]:
    matches = [
        record for record in provenance.config_files if record.get("path") == PROFILE_PATH
    ]
    if len(matches) != 1:
        raise SealError("exact3 provenance must contain exactly one feature profile")
    return matches[0]


def _stage_profile_bytes(
    directory_fd: int,
    name: str,
    data: bytes,
    mode: int,
) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(name, flags, mode, dir_fd=directory_fd)
    try:
        os.fchmod(descriptor, mode)
        _write_all(descriptor, data)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_profile_after_replace(
    directory_fd: int,
    name: str,
) -> tuple[os.stat_result, bytes]:
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(name, flags, dir_fd=directory_fd)
    try:
        opened = os.fstat(descriptor)
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    if _stat_identity(opened) != _stat_identity(after) or _stat_identity(
        after
    ) != _stat_identity(current):
        raise SealError("sealed feature profile changed during post-replace verification")
    return after, b"".join(chunks)


def _apply_profile_transition(
    provenance: ProvenanceSnapshot,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    """Apply the sole allowed post-run mutation by one same-directory replace."""

    profile_record = _profile_file_record(provenance)
    candidate_bytes = _read_expected_repository_file(PROFILE_PATH, profile_record)
    try:
        candidate_text = candidate_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise SealError("candidate feature profile is not UTF-8") from error
    candidate = _strict_json_loads(candidate_text, "candidate feature profile")
    transition = _profile_transition_record(candidate, profile_record)
    if transition != provenance.profile_transition:
        raise SealError("candidate feature profile transition record drifted")
    sealed = _sealed_profile_value(candidate, transition, evidence)
    sealed_bytes = _render_sealed_profile_bytes(
        candidate_bytes,
        candidate,
        sealed,
    )

    profile_path = REPO_ROOT / PROFILE_PATH
    config_directory = profile_path.parent
    try:
        resolved_directory = config_directory.resolve(strict=True)
        directory_stat = config_directory.lstat()
    except OSError as error:
        raise SealError(f"cannot inspect profile directory: {error}") from error
    if (
        resolved_directory != config_directory
        or not stat.S_ISDIR(directory_stat.st_mode)
        or stat.S_ISLNK(directory_stat.st_mode)
    ):
        raise SealError("profile directory must be an ordinary non-symlink directory")
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        directory_flags |= os.O_NOFOLLOW
    try:
        directory_fd = os.open(config_directory, directory_flags)
    except OSError as error:
        raise SealError(f"cannot safely open profile directory: {error}") from error

    source_fd: int | None = None
    temporary_name: str | None = None
    rollback_name: str | None = None
    preserve_rollback_for_recovery = False
    final_bytes: bytes | None = None
    try:
        source_flags = os.O_RDONLY | os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            source_flags |= os.O_NOFOLLOW
        source_fd = os.open(profile_path.name, source_flags, dir_fd=directory_fd)
        fcntl.flock(source_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if _stat_identity(os.fstat(source_fd)) != _repository_record_identity(
            profile_record
        ):
            raise SealError("candidate feature profile identity drifted before replace")
        if _read_expected_repository_file(PROFILE_PATH, profile_record) != candidate_bytes:
            raise SealError("candidate feature profile bytes drifted before replace")

        token = hashlib.sha256(
            f"{os.getpid()}-{datetime.now().timestamp()}".encode("ascii")
        ).hexdigest()[:16]
        temporary_name = f".{profile_path.name}.exact3-seal-{token}.tmp"
        rollback_name = f".{profile_path.name}.exact3-rollback-{token}.tmp"
        mode = stat.S_IMODE(profile_record["mode"])
        _stage_profile_bytes(
            directory_fd,
            temporary_name,
            sealed_bytes,
            mode,
        )
        _stage_profile_bytes(
            directory_fd,
            rollback_name,
            candidate_bytes,
            mode,
        )

        current = os.stat(
            profile_path.name,
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(current.st_mode)
            or current.st_nlink != 1
            or _stat_identity(current) != _repository_record_identity(profile_record)
        ):
            raise SealError("candidate feature profile drifted before atomic replace")
        os.replace(
            temporary_name,
            profile_path.name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
        temporary_name = None
        try:
            os.fsync(directory_fd)
            final_stat, final_bytes = _read_profile_after_replace(
                directory_fd,
                profile_path.name,
            )
            if (
                not stat.S_ISREG(final_stat.st_mode)
                or final_stat.st_nlink != 1
                or stat.S_IMODE(final_stat.st_mode) != mode
                or final_bytes != sealed_bytes
            ):
                raise SealError(
                    "sealed feature profile failed post-replace verification"
                )
        except (OSError, SealError) as post_replace_error:
            try:
                os.replace(
                    rollback_name,
                    profile_path.name,
                    src_dir_fd=directory_fd,
                    dst_dir_fd=directory_fd,
                )
            except OSError as rollback_error:
                preserve_rollback_for_recovery = True
                raise SealError(
                    "exact3 profile replace failed and candidate rollback also "
                    f"failed; recovery temp {rollback_name!r} was retained: "
                    f"replace={post_replace_error}; rollback={rollback_error}"
                ) from post_replace_error
            rollback_name = None
            try:
                os.fsync(directory_fd)
            except OSError as rollback_fsync_error:
                raise SealError(
                    "exact3 profile replace failed; candidate bytes were restored "
                    "but the rollback directory fsync failed: "
                    f"replace={post_replace_error}; fsync={rollback_fsync_error}"
                ) from post_replace_error
            raise SealError(
                "exact3 profile replace failed after publication; the candidate "
                f"profile was restored: {post_replace_error}"
            ) from post_replace_error
        try:
            os.unlink(rollback_name, dir_fd=directory_fd)
            rollback_name = None
        except OSError as cleanup_error:
            try:
                os.replace(
                    rollback_name,
                    profile_path.name,
                    src_dir_fd=directory_fd,
                    dst_dir_fd=directory_fd,
                )
            except OSError as rollback_error:
                preserve_rollback_for_recovery = True
                raise SealError(
                    "sealed profile rollback-temp cleanup failed and candidate "
                    f"rollback also failed; recovery temp {rollback_name!r} was "
                    f"retained: cleanup={cleanup_error}; rollback={rollback_error}"
                ) from cleanup_error
            rollback_name = None
            try:
                os.fsync(directory_fd)
            except OSError as rollback_fsync_error:
                raise SealError(
                    "sealed profile rollback-temp cleanup failed; candidate bytes "
                    "were restored but the rollback directory fsync failed: "
                    f"cleanup={cleanup_error}; fsync={rollback_fsync_error}"
                ) from cleanup_error
            raise SealError(
                "sealed profile rollback-temp cleanup failed; the candidate "
                f"profile was restored: {cleanup_error}"
            ) from cleanup_error
        try:
            os.fsync(directory_fd)
        except OSError:
            # The prior directory fsync already made the profile replace durable.
            # A failed cleanup fsync can at worst resurrect the private rollback
            # temp after a crash; it cannot make the sealed profile reference
            # unpublished evidence.
            pass
    except (BlockingIOError, OSError, SealError) as error:
        if isinstance(error, SealError):
            raise
        raise SealError(f"cannot atomically apply exact3 profile transition: {error}") from error
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
        if rollback_name is not None and not preserve_rollback_for_recovery:
            try:
                os.unlink(rollback_name, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
        if source_fd is not None:
            os.close(source_fd)
        os.close(directory_fd)

    if final_bytes is None:  # pragma: no cover - successful path always assigns
        raise SealError("sealed feature profile verification did not complete")
    return {
        "path": PROFILE_PATH,
        "sha256": hashlib.sha256(final_bytes).hexdigest(),
        "size_bytes": len(final_bytes),
        "immutable_projection_sha256": _profile_immutable_projection_sha256(sealed),
        "changed_json_pointers": sorted(PROFILE_TRANSITION_ALLOWED_JSON_POINTERS),
        "evidence": dict(evidence),
    }


def _atomic_install(
    run_root: Path,
    directory_fd: int,
    outputs: dict[str, bytes],
) -> None:
    token = hashlib.sha256(
        f"{os.getpid()}-{datetime.now().timestamp()}".encode("ascii")
    ).hexdigest()[:16]
    temporary_names = {
        name: f".exact3-seal-{os.getpid()}-{token}-{name}.tmp" for name in outputs
    }
    installed: list[str] = []
    staged: list[str] = []
    try:
        for name, data in outputs.items():
            temporary = temporary_names[name]
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            descriptor = os.open(temporary, flags, 0o644, dir_fd=directory_fd)
            staged.append(temporary)
            try:
                _write_all(descriptor, data)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        expected_names = set(RAW_NAMES) | set(temporary_names.values())
        actual_names = set(os.listdir(directory_fd))
        if actual_names != expected_names:
            raise SealError("run directory changed while derived evidence was staged")
        for name in OUTPUT_NAMES:
            temporary = temporary_names[name]
            os.link(
                temporary,
                name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
                follow_symlinks=False,
            )
            installed.append(name)
            os.unlink(temporary, dir_fd=directory_fd)
            staged.remove(temporary)
        _verify_run_root_identity(run_root, directory_fd)
        if set(os.listdir(directory_fd)) != set(RAW_NAMES) | set(OUTPUT_NAMES):
            raise SealError(
                "run directory changed while derived evidence was installed"
            )
        os.fsync(directory_fd)
    except (OSError, SealError) as error:
        for name in installed:
            try:
                os.unlink(name, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
        for name in staged:
            try:
                os.unlink(name, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
        try:
            os.fsync(directory_fd)
        except OSError:
            pass
        if isinstance(error, SealError):
            raise
        raise SealError(
            f"cannot atomically install derived evidence: {error}"
        ) from error


def _verify_raw_unchanged(
    directory_fd: int,
    original: dict[str, RawSnapshot],
) -> None:
    _require_initial_directory(directory_fd)
    for name in RAW_NAMES:
        current = _read_raw_file(directory_fd, name)
        if current != original[name]:
            raise SealError(f"raw log changed during sealing: {name}")


def _canonical_new_run_root(run_root: Path) -> Path:
    try:
        output_root = OUTPUT_ROOT.resolve(strict=True)
    except OSError as error:
        raise SealError(f"cannot resolve exact3 output root: {error}") from error
    lexical = Path(os.path.abspath(os.fspath(run_root)))
    if lexical.parent != output_root or RUN_ID_PATTERN.fullmatch(lexical.name) is None:
        raise SealError(
            f"--run-root must be one direct results/openfhe/<run-id> directory: {run_root}"
        )
    try:
        lexical.lstat()
    except FileNotFoundError:
        return lexical
    except OSError as error:
        raise SealError(f"cannot inspect requested run root {run_root}: {error}") from error
    raise SealError(f"requested exact3 run root already exists: {run_root}")


def _create_raw_file(path: Path) -> int:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        return os.open(path, flags, 0o600)
    except OSError as error:
        raise SealError(f"cannot create exact3 raw log {path.name}: {error}") from error


def _execute_exact3_once(staging_root: Path) -> LaunchEvidence:
    if not Path(TIME_EXECUTABLE).is_file() or not os.access(TIME_EXECUTABLE, os.X_OK):
        raise SealError(f"GNU time executable is unavailable: {TIME_EXECUTABLE}")
    stdout_path = staging_root / "stdout.log"
    stderr_path = staging_root / "stderr.log"
    time_path = staging_root / "time.log"
    open_descriptors: list[int] = []
    try:
        stdout_fd = _create_raw_file(stdout_path)
        open_descriptors.append(stdout_fd)
        stderr_fd = _create_raw_file(stderr_path)
        open_descriptors.append(stderr_fd)
        time_fd = _create_raw_file(time_path)
        os.close(time_fd)
        argv = (
            TIME_EXECUTABLE,
            "-v",
            "-o",
            str(time_path),
            "--",
            *EXPECTED_RUNTIME_ARGV,
        )
        started_at = datetime.now().astimezone().isoformat(timespec="microseconds")
        try:
            completed = subprocess.run(
                argv,
                cwd=REPO_ROOT,
                env=RUNTIME_ENVIRONMENT,
                stdin=subprocess.DEVNULL,
                stdout=stdout_fd,
                stderr=stderr_fd,
                check=False,
                close_fds=True,
            )
        except OSError as error:
            raise SealError(f"cannot launch the exact3 child: {error}") from error
        finished_at = datetime.now().astimezone().isoformat(timespec="microseconds")
        for descriptor in (stdout_fd, stderr_fd):
            os.fsync(descriptor)
        try:
            synced_time_fd = os.open(time_path, os.O_RDONLY | os.O_CLOEXEC)
        except OSError as error:
            raise SealError(f"cannot reopen GNU time output: {error}") from error
        try:
            os.fsync(synced_time_fd)
        finally:
            os.close(synced_time_fd)
        return LaunchEvidence(
            argv=argv,
            started_at=started_at,
            finished_at=finished_at,
            returncode=completed.returncode,
        )
    finally:
        for descriptor in open_descriptors:
            os.close(descriptor)


def _validate_launch_evidence(
    launch: LaunchEvidence,
    staging_root: Path,
) -> None:
    expected_argv = (
        TIME_EXECUTABLE,
        "-v",
        "-o",
        str(staging_root / "time.log"),
        "--",
        *EXPECTED_RUNTIME_ARGV,
    )
    if launch.argv != expected_argv:
        raise SealError(
            f"exact3 launcher argv differs from the frozen command: {launch.argv!r}"
        )
    if (
        not isinstance(launch.returncode, int)
        or isinstance(launch.returncode, bool)
        or launch.returncode != 0
    ):
        raise SealError(
            f"the sole exact3 child exited unsuccessfully: {launch.returncode!r}"
        )
    for label, value in (
        ("started_at", launch.started_at),
        ("finished_at", launch.finished_at),
    ):
        if not isinstance(value, str) or not value:
            raise SealError(f"exact3 launch {label} is missing")
        try:
            datetime.fromisoformat(value)
        except ValueError as error:
            raise SealError(f"exact3 launch {label} is malformed") from error
    if datetime.fromisoformat(launch.finished_at) < datetime.fromisoformat(
        launch.started_at
    ):
        raise SealError("exact3 launch finished_at precedes started_at")


def _require_bracket_unchanged(
    git_before: GitSnapshot,
    provenance_before: ProvenanceSnapshot,
) -> None:
    git_after = _git_snapshot()
    provenance_after = _provenance_snapshot()
    if git_after.fingerprint != git_before.fingerprint:
        raise SealError("git/worktree changed after the exact3 pre-launch snapshot")
    if provenance_after != provenance_before:
        raise SealError(
            "source, config, executable, trace-scale, or file identity changed "
            "after the exact3 pre-launch snapshot"
        )


def _remove_private_staging(staging_root: Path) -> None:
    try:
        output_root = OUTPUT_ROOT.resolve(strict=True)
        lexical = Path(os.path.abspath(os.fspath(staging_root)))
    except OSError:
        return
    if (
        lexical.parent != output_root
        or not lexical.name.startswith(".exact3-stage-")
        or not lexical.exists()
        or lexical.is_symlink()
    ):
        return
    shutil.rmtree(lexical)


def _rename_noreplace(source: Path, destination: Path) -> None:
    """Atomically rename one directory without ever replacing the destination."""

    if source.parent != destination.parent:
        raise SealError("exact3 publication source and destination have different parents")
    parent = source.parent
    try:
        resolved_parent = parent.resolve(strict=True)
        parent_stat = parent.lstat()
        expected_parent = OUTPUT_ROOT.resolve(strict=True)
    except OSError as error:
        raise SealError(f"cannot inspect exact3 publication parent: {error}") from error
    if (
        resolved_parent != parent
        or resolved_parent != expected_parent
        or not stat.S_ISDIR(parent_stat.st_mode)
        or stat.S_ISLNK(parent_stat.st_mode)
    ):
        raise SealError("exact3 publication parent identity is unsafe")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        parent_fd = os.open(parent, flags)
    except OSError as error:
        raise SealError(f"cannot open exact3 publication parent: {error}") from error
    try:
        opened_parent = os.fstat(parent_fd)
        if (
            opened_parent.st_dev != parent_stat.st_dev
            or opened_parent.st_ino != parent_stat.st_ino
            or not stat.S_ISDIR(opened_parent.st_mode)
        ):
            raise SealError("exact3 publication parent changed while opening")
        try:
            library = ctypes.CDLL(None, use_errno=True)
        except OSError as error:
            raise SealError(f"cannot load renameat2 support: {error}") from error
        renameat2 = getattr(library, "renameat2", None)
        if renameat2 is None:
            raise SealError(
                "renameat2(RENAME_NOREPLACE) is unavailable; refusing unsafe publication"
            )
        renameat2.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        renameat2.restype = ctypes.c_int
        result = renameat2(
            parent_fd,
            os.fsencode(source.name),
            parent_fd,
            os.fsencode(destination.name),
            1,  # RENAME_NOREPLACE from linux/fs.h
        )
        if result != 0:
            error_number = ctypes.get_errno()
            if error_number == errno.EEXIST:
                raise SealError(
                    f"requested exact3 run root appeared during launch: {destination}"
                )
            raise SealError(
                "cannot atomically publish exact3 bundle without replacement: "
                f"{os.strerror(error_number)}"
            )
        try:
            os.fsync(parent_fd)
        except OSError as error:
            raise SealError(
                "exact3 bundle was renamed but publication-parent fsync failed; "
                f"the unapproved orphan bundle is retained: {error}"
            ) from error
    finally:
        os.close(parent_fd)


def _published_profile_evidence(final_root: Path) -> dict[str, Any]:
    expected_names = set(RAW_NAMES) | set(OUTPUT_NAMES)
    try:
        actual_names = {path.name for path in final_root.iterdir()}
    except OSError as error:
        raise SealError(f"cannot inspect published exact3 bundle: {error}") from error
    if actual_names != expected_names:
        raise SealError(
            "published exact3 bundle inventory differs from the six-file contract"
        )
    live: dict[str, bytes] = {}
    for name in (*RAW_NAMES, *OUTPUT_NAMES):
        path = final_root / name
        try:
            file_stat = path.lstat()
            data = path.read_bytes()
            after = path.lstat()
        except OSError as error:
            raise SealError(
                f"cannot verify published exact3 bundle file {name}: {error}"
            ) from error
        if (
            not stat.S_ISREG(file_stat.st_mode)
            or file_stat.st_nlink != 1
            or _stat_identity(file_stat) != _stat_identity(after)
            or len(data) != after.st_size
        ):
            raise SealError(
                f"published exact3 bundle file is linked or unstable: {name}"
            )
        live[name] = data
    expected_sums = "".join(
        f"{hashlib.sha256(live[name]).hexdigest()}  {name}\n"
        for name in (
            "stdout.log",
            "stderr.log",
            "time.log",
            "metrics.csv",
            "manifest.json",
        )
    ).encode("ascii")
    if live["SHA256SUMS"] != expected_sums:
        raise SealError("published exact3 SHA256SUMS does not bind the five payloads")
    try:
        manifest_text = live["manifest.json"].decode("utf-8")
    except UnicodeDecodeError as error:
        raise SealError("published exact3 manifest is not UTF-8") from error
    manifest = _strict_json_loads(manifest_text, "published exact3 manifest")
    if not isinstance(manifest, dict):
        raise SealError("published exact3 manifest must be a JSON object")
    expected_manifest_values = {
        "schema_id": EXACT3_MANIFEST_SCHEMA_ID,
        "schema_version": EXACT3_MANIFEST_SCHEMA_VERSION,
        "run_id": final_root.name,
        "artifact_kind": "exact3_schedule_evidence",
        "milestone_artifact_eligible": False,
        "schedule_evidence_eligible": True,
        "exact3_gate_passed": True,
        "formal_schedule_sealed": True,
    }
    for key, expected in expected_manifest_values.items():
        if key not in manifest:
            raise SealError(f"published exact3 manifest lacks {key}")
        _require_exact(manifest[key], expected, f"published exact3 manifest {key}")
    return _validate_profile_evidence(
        {
            "run_id": final_root.name,
            "relative_path": (
                final_root.relative_to(REPO_ROOT).as_posix()
            ),
            "manifest_sha256": hashlib.sha256(live["manifest.json"]).hexdigest(),
            "sha256sums_sha256": hashlib.sha256(live["SHA256SUMS"]).hexdigest(),
            "layer_count": 3,
            "artifact_eligible": False,
            "exact3_gate_passed": True,
            "schedule_evidence_eligible": True,
            "formal_schedule_sealed": True,
        }
    )


def run_and_seal(run_root: Path) -> Path:
    final_root = _canonical_new_run_root(run_root)
    staging_root = Path(
        tempfile.mkdtemp(
            prefix=f".exact3-stage-{final_root.name}-",
            dir=OUTPUT_ROOT.resolve(strict=True),
        )
    )
    directory_fd: int | None = None
    published = False
    try:
        canonical_staging, directory_fd = _open_run_root(
            staging_root,
            require_public_run_id=False,
        )
        if os.listdir(directory_fd):
            raise SealError("private exact3 staging directory is not empty")

        git_before = _git_snapshot()
        provenance_before = _provenance_snapshot()
        try:
            launch = _execute_exact3_once(canonical_staging)
        except SealError:
            _require_bracket_unchanged(git_before, provenance_before)
            raise
        _require_bracket_unchanged(git_before, provenance_before)
        _validate_launch_evidence(launch, canonical_staging)

        _require_initial_directory(directory_fd)
        raw = {name: _read_raw_file(directory_fd, name) for name in RAW_NAMES}
        if raw["stderr.log"].data:
            raise SealError("stderr.log must be byte-empty")
        evidence = _parse_stdout(raw["stdout.log"].data)
        timing = _parse_time(raw["time.log"].data)
        if timing.exit_status != launch.returncode:
            raise SealError("GNU time exit status differs from the sole child result")
        if evidence.summary["peak_rss_bytes"] != timing.peak_rss_kib * 1024:
            raise SealError("stdout peak_rss_bytes differs from GNU time peak RSS")

        metrics = _metrics_bytes(evidence, timing)
        manifest_value = _manifest(
            final_root,
            raw,
            evidence,
            timing,
            launch,
            git_before,
            provenance_before,
            metrics,
        )
        manifest = _pretty_json_bytes(manifest_value)
        checksummed = {
            "stdout.log": raw["stdout.log"].data,
            "stderr.log": raw["stderr.log"].data,
            "time.log": raw["time.log"].data,
            "metrics.csv": metrics,
            "manifest.json": manifest,
        }
        sums = "".join(
            f"{hashlib.sha256(checksummed[name]).hexdigest()}  {name}\n"
            for name in (
                "stdout.log",
                "stderr.log",
                "time.log",
                "metrics.csv",
                "manifest.json",
            )
        ).encode("ascii")

        _verify_raw_unchanged(directory_fd, raw)
        _require_bracket_unchanged(git_before, provenance_before)
        _verify_run_root_identity(canonical_staging, directory_fd)
        _atomic_install(
            canonical_staging,
            directory_fd,
            {
                "metrics.csv": metrics,
                "manifest.json": manifest,
                "SHA256SUMS": sums,
            },
        )
        os.close(directory_fd)
        directory_fd = None
        _rename_noreplace(canonical_staging, final_root)
        published = True
        profile_evidence = _published_profile_evidence(final_root)
        _require_bracket_unchanged(git_before, provenance_before)
        _apply_profile_transition(provenance_before, profile_evidence)
        return final_root / "manifest.json"
    finally:
        if directory_fd is not None:
            os.close(directory_fd)
        if not published:
            _remove_private_staging(staging_root)


def seal_run(run_root: Path) -> Path:
    del run_root
    raise SealError(
        "post-run-only exact3 sealing is retired; use run_and_seal so provenance "
        "snapshots bracket the sole child launch"
    )


def _parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch exactly one OpenFHE exact-three run and seal its evidence"
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        required=True,
        help="New canonical repo/results/openfhe/<run-id> directory",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = _parse_arguments(argv)
    try:
        manifest = run_and_seal(arguments.run_root)
    except SealError as error:
        print(f"exact3 evidence sealing failed: {error}", file=sys.stderr)
        return 2
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
