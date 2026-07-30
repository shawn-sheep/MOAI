#!/usr/bin/env python3
"""Run and seal the frozen M5 OpenFHE 12-layer correctness evidence bundle.

This runner is intentionally not a benchmark harness.  It builds and validates
the clean M5 target, runs the frozen plaintext/contract preflight, then performs
exactly one complete server-only ciphertext chain through all 12 encoder layers.
There is no warm-up and no repeat: timing remains a non-benchmark diagnostic
until M6.  Prefix diagnostics are deliberately not exposed by this interface.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXECUTABLE = REPO_ROOT / "build-openfhe" / "openfhe_encoder_12_layer_smoke"
DEFAULT_DATA_ROOT = REPO_ROOT / "data"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "openfhe"
DEFAULT_OPENFHE_PREFIX = Path("/home/shawnsheep/opt/openfhe_v1_5_1")
VALIDATOR_PATH = REPO_ROOT / "scripts" / "validate_openfhe_m5_artifact.py"
TIME_EXECUTABLE = Path("/usr/bin/time")
REMOTE_NAME = "origin"
BRANCH = "refactor/openfhe-cpu"
TRACKING_REF = "refs/remotes/origin/refactor/openfhe-cpu"
REMOTE_REF = "refs/heads/refactor/openfhe-cpu"
PROFILE_PATH = "config/paper_compat_feature_packed.json"
PROFILE_LOCATOR = "/effective_profile"
PROFILE_SHA256 = "94f30e628e21f02146ce7ed9820194eabba3820f6e1e17176a31f8c5acf8b0be"
M5_SCHEMA_VERSION = 4
WARNING = "Research reproduction parameters only. Do not claim 128-bit security."
CANONICALIZATION = "MOAI-json-sort-keys-compact-utf8-v1"
REQUIRED_INPUTS = (
    "config/moai_encoder_trace.json",
    "config/moai_trace_channel_scales.json",
    "config/openfhe_approximations.json",
    PROFILE_PATH,
)
ENCODER_LAYERS = 12
FILES_PER_LAYER = 37
WEIGHT_FILES_PER_LAYER = 16
TRACE_FILES_PER_LAYER = 21
EXPECTED_INPUT_COUNT = len(REQUIRED_INPUTS) + ENCODER_LAYERS * FILES_PER_LAYER
EXPECTED_MULTIPLICATIVE_DEPTH = 47
EXPECTED_MAX_OBSERVED_LEVEL = 45
EXPECTED_MAX_POLYNOMIAL_DEPTH = 10
EXPECTED_LAYER_COUNTS = {
    "rotations": 6300,
    "ct_pt_multiplications": 51865,
    "ct_ct_multiplications": 95,
    "explicit_rescale_requests": 800,
    "chebyshev_evaluations": 55,
    "estimated_polynomial_multiplications": 1150,
    "bootstraps": 25,
    "bootstrap_iterations": 50,
}
EXPECTED_REFRESH_COUNTS = {
    "rotations": 0,
    "ct_pt_multiplications": 5,
    "ct_ct_multiplications": 0,
    "explicit_rescale_requests": 5,
    "chebyshev_evaluations": 0,
    "estimated_polynomial_multiplications": 0,
    "bootstraps": 5,
    "bootstrap_iterations": 10,
}
ZERO_COUNTS = {key: 0 for key in EXPECTED_LAYER_COUNTS}
EXPECTED_TOTAL_COUNTS = {
    key: EXPECTED_LAYER_COUNTS[key] * ENCODER_LAYERS
    + EXPECTED_REFRESH_COUNTS[key] * (ENCODER_LAYERS - 1)
    for key in EXPECTED_LAYER_COUNTS
}
# The complete layer-0/post-refresh metadata schedule is frozen from the live r6
# two-layer seam calibration.  Formal execution still keeps explicit ``None``
# guards so any accidental removal of a sealed tuple fails before the HE child
# process starts.
LAYER0_INPUT_METADATA: dict[str, object] | None = {
    "level": 29,
    "noise_scale_degree": 1,
    "remaining_levels": 18,
    "scale_bits": 50,
    "expected_scale_bits": 50,
    "ciphertext_count": 5,
}
LAYER_HANDOFF_INPUT_METADATA: dict[str, object] | None = {
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
POLYNOMIAL_CHECKPOINT_METADATA: dict[str, object] | None = {
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
COUNT_KEYS = set(EXPECTED_LAYER_COUNTS)
METADATA_KEYS = {
    "level",
    "noise_scale_degree",
    "remaining_levels",
    "scale_bits",
    "expected_scale_bits",
    "ciphertext_count",
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
QUALITY_KEYS = {"relative_l2", "cosine", "max_absolute"}
RANGE_KEYS = {
    "softmax_shifted_logits",
    "softmax_denominator",
    "ln1_normalized_variance",
    "gelu_input",
    "ln2_normalized_variance",
}
INACTIVE_SENTINEL_RANGE_KEYS = {
    "softmax_denominator",
    "ln1_normalized_variance",
    "ln2_normalized_variance",
}
FROZEN_RANGES = {
    "softmax_shifted_logits": (-16.0, 5.0),
    "softmax_denominator": (0.01, 80.0),
    "ln1_normalized_variance": (0.5, 1536.0),
    "gelu_input": (-80.0, 128.0),
    "ln2_normalized_variance": (0.5, 1536.0),
}
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
LAYER_RECORD_KEYS = {
    "test",
    "profile",
    "security_claim",
    "layer_id",
    "weights_layer_id",
    "chain_input_source",
    "handoff_refresh_performed",
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
FINAL_RECORD_KEYS = {
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
FORMAL_RUNTIME_TEST_NAMES = frozenset(
    {
        "openfhe_encoder_12_layer_layer",
        "openfhe_encoder_12_layer",
    }
)
FORMAL_NON_EVIDENCE_JSON_ALLOWLIST = {
    "openfhe_security_disclosure": frozenset({"test", "security_claim"}),
}
DIAGNOSTIC_EVIDENCE_TEST_PREFIXES = (
    "openfhe_encoder_metadata_calibration",
    "openfhe_encoder_exact_prefix",
)
DIAGNOSTIC_ONLY_EVIDENCE_FIELDS = frozenset(
    {
        "artifact_eligible",
        "formal_schedule_sealed",
        "metadata_validation_mode",
        "metadata_used_levels",
        "metadata_used_level_deltas",
        "diagnostic_gates_passed",
    }
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
    "rotations",
    "ct_pt_multiplications",
    "ct_ct_multiplications",
    "explicit_rescale_requests",
    "chebyshev_evaluations",
    "estimated_polynomial_multiplications",
    "bootstraps",
    "bootstrap_iterations",
    "timing_claim",
    "latency_kind",
    "finite",
    "passed",
)
# This narrow set intentionally includes the executable's --preflight-only CTest
# and excludes the full openfhe_encoder_12_layer_smoke workload.
M5_CTEST_PATTERN = (
    "^(moai_trace_contract|openfhe_(server_trust_boundary|profile_contract|"
    "profile_validator_contract|artifact_schema_contract|artifact_validator_contract|"
    "encoder_artifact_runner_contract|m5_artifact_schema_contract|"
    "m5_artifact_validator_contract|encoder12_artifact_runner_contract|"
    "evaluation_key_bundle_smoke|feature_packed_smoke|"
    "feature_packed_attention_smoke|feature_bootstrap_smoke|"
    "encoder_fixture_contract|encoder_plaintext_oracle_smoke|"
    "encoder_12_layer_preflight|encoder_12_layer_crypto_preflight|"
    "encoder_trace_contract|"
    "encoder_trace_validator_contract))$"
)
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+:-]{0,127}$")
SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")


class ArtifactRunnerError(RuntimeError):
    """Raised when an M5 artifact cannot be produced without weakening a gate."""


def _metadata_schedule_contract() -> dict[str, object]:
    if (
        LAYER0_INPUT_METADATA is None
        or LAYER_HANDOFF_INPUT_METADATA is None
        or POLYNOMIAL_CHECKPOINT_METADATA is None
    ):
        raise ArtifactRunnerError(
            "M5 metadata_schedule cannot be emitted before all seam tuples are sealed"
        )
    schedule = {
        "multiplicative_depth": EXPECTED_MULTIPLICATIVE_DEPTH,
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
        _validate_metadata(schedule[name], schedule[name], f"metadata_schedule.{name}")
    regime_fields = (
        "attention_output_by_layer_regime",
        "ln1_output_by_layer_regime",
        "ffn_output_by_layer_regime",
    )
    for field in regime_fields:
        for regime, metadata in schedule[field].items():
            _validate_metadata(
                metadata,
                metadata,
                f"metadata_schedule.{field}.{regime}",
            )
    for regime, input_field in (
        ("layer_0", "initial_layer_input"),
        ("layers_1_to_11", "post_refresh_layer_input"),
    ):
        _validate_used_level_delta(
            schedule["attention_output_by_layer_regime"][regime],
            schedule[input_field],
            RELATIVE_USED_LEVEL_DELTAS[
                "attention_output_from_layer_input"
            ][regime],
            f"metadata_schedule.attention_output_by_layer_regime.{regime}",
        )
        _validate_used_level_delta(
            schedule["ln1_output_by_layer_regime"][regime],
            schedule["post_bootstrap_polynomial_checkpoint"],
            RELATIVE_USED_LEVEL_DELTAS[
                "ln1_output_from_ln1_variance"
            ][regime],
            f"metadata_schedule.ln1_output_by_layer_regime.{regime}",
        )
        _validate_used_level_delta(
            schedule["ffn_output_by_layer_regime"][regime],
            schedule["ln1_output_by_layer_regime"][regime],
            RELATIVE_USED_LEVEL_DELTAS["ffn_output_from_ln1_output"],
            f"metadata_schedule.ffn_output_by_layer_regime.{regime}",
        )
    _validate_used_level_delta(
        schedule["raw_layer_output"],
        schedule["post_bootstrap_polynomial_checkpoint"],
        RELATIVE_USED_LEVEL_DELTAS["raw_output_from_ln2_variance"],
        "metadata_schedule.raw_output",
    )
    return schedule


@dataclass(frozen=True)
class GitState:
    head: str
    tracking_head: str
    remote_head: str
    commands: tuple[dict[str, object], ...]


@dataclass(frozen=True)
class M5Preflight:
    commands: tuple[dict[str, object], ...]
    crypto_preflight: dict[str, Any]


@dataclass(frozen=True)
class RunnerConfig:
    executable: Path
    data_root: Path
    output_root: Path
    run_id: str
    openfhe_prefix: Path = DEFAULT_OPENFHE_PREFIX


@dataclass(frozen=True)
class RunSample:
    stdout: str
    layers: tuple[dict[str, Any], ...]
    final: dict[str, Any]
    elapsed_seconds: float
    peak_rss_kib: int
    command: dict[str, object]


def _timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ArtifactRunnerError(f"value is not canonical JSON: {error}") from error


def _checkpoint_metadata_payload(
    layers: tuple[dict[str, Any], ...] | list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if len(layers) != ENCODER_LAYERS:
        raise ArtifactRunnerError(
            "checkpoint metadata digest requires exactly 12 ordered layer records"
        )
    payload: list[dict[str, Any]] = []
    for layer_id, layer in enumerate(layers):
        if layer.get("layer_id") != layer_id:
            raise ArtifactRunnerError(
                "checkpoint metadata digest requires layer order 0 through 11"
            )
        missing = set(CHECKPOINT_METADATA_FIELDS) - set(layer)
        if missing:
            raise ArtifactRunnerError(
                f"checkpoint metadata digest is missing fields: {sorted(missing)}"
            )
        payload.append(
            {
                "layer_id": layer_id,
                **{field: layer[field] for field in CHECKPOINT_METADATA_FIELDS},
            }
        )
    return payload


def _checkpoint_metadata_sha256(
    layers: tuple[dict[str, Any], ...] | list[dict[str, Any]],
) -> str:
    return hashlib.sha256(
        _canonical_json_bytes(_checkpoint_metadata_payload(layers))
    ).hexdigest()


def _load_json(path: Path) -> Any:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ArtifactRunnerError(f"duplicate JSON key in {path}: {key!r}")
            result[key] = value
        return result

    def reject_non_finite(token: str) -> None:
        raise ArtifactRunnerError(f"non-finite JSON number in {path}: {token}")

    try:
        with path.open("r", encoding="utf-8") as stream:
            return json.load(
                stream,
                object_pairs_hook=reject_duplicate_keys,
                parse_constant=reject_non_finite,
            )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ArtifactRunnerError(f"cannot read JSON {path}: {error}") from error


def _resolve_repository_file(path: Path, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(REPO_ROOT.resolve())
    except ValueError as error:
        raise ArtifactRunnerError(f"{label} must stay inside {REPO_ROOT}: {path}") from error
    if not resolved.is_file():
        raise ArtifactRunnerError(f"{label} is not a file: {resolved}")
    return resolved


def _lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _resolve_m5_executable(path: Path, *, require_exists: bool = True) -> Path:
    candidate = _lexical_absolute(path)
    expected = _lexical_absolute(DEFAULT_EXECUTABLE)
    if candidate != expected:
        raise ArtifactRunnerError(
            f"M5 executable must be the fixed build target {DEFAULT_EXECUTABLE}"
        )
    try:
        relative_parts = candidate.relative_to(REPO_ROOT).parts
    except ValueError as error:  # pragma: no cover - fixed target is repository-local
        raise ArtifactRunnerError(f"M5 executable must stay inside {REPO_ROOT}") from error
    cursor = REPO_ROOT
    for part in relative_parts:
        cursor /= part
        if cursor.is_symlink():
            raise ArtifactRunnerError(f"M5 executable path must not contain symlinks: {cursor}")
    if not require_exists:
        if candidate.exists() and not candidate.is_file():
            raise ArtifactRunnerError(f"M5 executable is not a regular file: {candidate}")
        return candidate
    if not candidate.is_file() or not os.access(candidate, os.X_OK):
        raise ArtifactRunnerError(f"M5 executable is not an executable file: {candidate}")
    return candidate


def _resolve_output_root(path: Path) -> Path:
    resolved = path.resolve()
    allowed = DEFAULT_OUTPUT_ROOT.resolve()
    if resolved != allowed:
        raise ArtifactRunnerError(f"M5 output root must be exactly {allowed}: {path}")
    return resolved


def _command_record(command: list[str], returncode: int, started: str | None = None,
                    finished: str | None = None) -> dict[str, object]:
    record: dict[str, object] = {
        "command": shlex.join(command),
        "cwd": str(REPO_ROOT),
        "exit_code": returncode,
        "phase": "artifact_generation",
    }
    if started is not None and finished is not None:
        record["started_at"] = started
        record["finished_at"] = finished
    return record


def _run_git(arguments: list[str]) -> tuple[str, dict[str, object]]:
    command = ["git", *arguments]
    completed = subprocess.run(
        command, cwd=REPO_ROOT, check=False, capture_output=True, text=True
    )
    record = _command_record(command, completed.returncode)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise ArtifactRunnerError(f"{shlex.join(command)} failed: {detail}")
    return completed.stdout.strip(), record


def _preflight_git() -> GitState:
    commands: list[dict[str, object]] = []
    status, record = _run_git(["status", "--porcelain=v1", "--untracked-files=normal"])
    commands.append(record)
    if status:
        raise ArtifactRunnerError("repository is not clean; refusing to run M5 evidence")

    branch, record = _run_git(["branch", "--show-current"])
    commands.append(record)
    if branch != BRANCH:
        raise ArtifactRunnerError(f"branch must be {BRANCH}, got {branch!r}")

    head, record = _run_git(["rev-parse", "HEAD"])
    commands.append(record)
    if SHA_PATTERN.fullmatch(head) is None:
        raise ArtifactRunnerError(f"local HEAD is not a full Git SHA: {head!r}")

    tracking_head, record = _run_git(["rev-parse", TRACKING_REF])
    commands.append(record)
    if SHA_PATTERN.fullmatch(tracking_head) is None:
        raise ArtifactRunnerError(f"tracking ref is not a full Git SHA: {tracking_head!r}")

    remote_output, record = _run_git(
        ["ls-remote", "--exit-code", REMOTE_NAME, REMOTE_REF]
    )
    commands.append(record)
    remote_lines = [line.split() for line in remote_output.splitlines() if line.strip()]
    if len(remote_lines) != 1 or len(remote_lines[0]) < 2:
        raise ArtifactRunnerError(f"remote ref lookup was not unique: {remote_output!r}")
    remote_head, remote_ref = remote_lines[0][0], remote_lines[0][1]
    if SHA_PATTERN.fullmatch(remote_head) is None or remote_ref != REMOTE_REF:
        raise ArtifactRunnerError(f"remote ref lookup was malformed: {remote_output!r}")
    if len({head, tracking_head, remote_head}) != 1:
        raise ArtifactRunnerError(
            "local, tracking, and live remote commits differ: "
            f"local={head} tracking={tracking_head} remote={remote_head}"
        )
    return GitState(head, tracking_head, remote_head, tuple(commands))


def _read_openfhe_version(openfhe_prefix: Path) -> str:
    version_file = openfhe_prefix / "lib" / "OpenFHE" / "OpenFHEConfigVersion.cmake"
    try:
        contents = version_file.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ArtifactRunnerError(f"cannot read OpenFHE version file: {error}") from error
    match = re.search(r'^set\(PACKAGE_VERSION "([0-9]+\.[0-9]+\.[0-9]+)"\)$', contents, re.M)
    if match is None or match.group(1) != "1.5.1":
        raise ArtifactRunnerError(f"OpenFHE package version must be 1.5.1: {version_file}")
    return match.group(1)


def _verify_build_configuration(build_root: Path, openfhe_prefix: Path) -> None:
    if build_root.resolve() != DEFAULT_EXECUTABLE.parent.resolve():
        raise ArtifactRunnerError(
            f"M5 build root must be {DEFAULT_EXECUTABLE.parent}, got {build_root}"
        )
    cache_path = build_root / "CMakeCache.txt"
    try:
        cache = cache_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ArtifactRunnerError(f"cannot read M5 CMake cache: {error}") from error
    expected_lines = {
        f"CMAKE_HOME_DIRECTORY:INTERNAL={REPO_ROOT.resolve()}",
        "CMAKE_BUILD_TYPE:STRING=Release",
        "BUILD_TESTING:BOOL=ON",
        f"OpenFHE_DIR:PATH={openfhe_prefix.resolve() / 'lib' / 'OpenFHE'}",
    }
    missing = sorted(expected_lines - set(cache.splitlines()))
    if missing:
        raise ArtifactRunnerError(f"M5 CMake cache binding drifted: missing={missing}")
    _read_openfhe_version(openfhe_prefix.resolve())


def _run_checked_command(
    command: list[str], label: str
) -> tuple[dict[str, object], str]:
    started = _timestamp()
    completed = subprocess.run(
        command, cwd=REPO_ROOT, check=False, capture_output=True, text=True
    )
    finished = _timestamp()
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise ArtifactRunnerError(
            f"{label} failed with {completed.returncode}: {detail[-4000:]}"
        )
    return (
        _command_record(command, completed.returncode, started, finished),
        completed.stdout,
    )


def _verify_openfhe_linkage(executable: Path, openfhe_prefix: Path) -> dict[str, object]:
    command = ["ldd", str(executable)]
    started = _timestamp()
    completed = subprocess.run(
        command, cwd=REPO_ROOT, check=False, capture_output=True, text=True
    )
    finished = _timestamp()
    if completed.returncode != 0:
        raise ArtifactRunnerError(f"cannot inspect M5 OpenFHE linkage: {completed.stderr.strip()}")
    linked_paths: list[Path] = []
    for line in completed.stdout.splitlines():
        if "libOPENFHE" not in line or "=>" not in line:
            continue
        target = line.split("=>", maxsplit=1)[1].strip().split(maxsplit=1)[0]
        if target == "not":
            raise ArtifactRunnerError(f"M5 OpenFHE library is unresolved: {line.strip()}")
        linked_paths.append(Path(target).resolve())
    if len(linked_paths) < 2:
        raise ArtifactRunnerError("M5 executable does not expose expected OpenFHE linkage")
    library_root = (openfhe_prefix.resolve() / "lib").resolve()
    for linked_path in linked_paths:
        try:
            linked_path.relative_to(library_root)
        except ValueError as error:
            raise ArtifactRunnerError(
                f"M5 executable links OpenFHE outside {library_root}: {linked_path}"
            ) from error
    return _command_record(command, completed.returncode, started, finished)


def _run_m5_preflight(build_root: Path, executable: Path,
                      openfhe_prefix: Path) -> M5Preflight:
    _verify_build_configuration(build_root, openfhe_prefix)
    build_command = [
        "cmake",
        "--build",
        str(build_root),
        "--clean-first",
        "-j",
        "4",
    ]
    test_command = [
        "ctest",
        "--test-dir",
        str(build_root),
        "--output-on-failure",
        "--verbose",
        "--no-tests=error",
        "-R",
        M5_CTEST_PATTERN,
    ]
    build_record, _ = _run_checked_command(build_command, "clean-commit M5 build")
    rebuilt = _resolve_m5_executable(executable)
    test_record, test_stdout = _run_checked_command(
        test_command, "clean-commit M5 narrow gates"
    )
    crypto_preflight = _parse_crypto_preflight_output(test_stdout)
    linkage_record = _verify_openfhe_linkage(rebuilt, openfhe_prefix)
    return M5Preflight(
        (build_record, test_record, linkage_record),
        crypto_preflight,
    )


def _version_line(command: list[str], label: str) -> str:
    completed = subprocess.run(
        command, cwd=REPO_ROOT, check=False, capture_output=True, text=True
    )
    if completed.returncode != 0 or not completed.stdout.strip():
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise ArtifactRunnerError(f"cannot identify {label}: {detail}")
    return completed.stdout.strip().splitlines()[0]


def _environment(openfhe_prefix: Path) -> dict[str, object]:
    prefix = openfhe_prefix.resolve()
    if not prefix.is_dir():
        raise ArtifactRunnerError(f"OpenFHE prefix is not a directory: {prefix}")
    release = platform.release()
    return {
        "os": f"{platform.system()} {release}",
        "architecture": platform.machine(),
        "compiler": _version_line(["c++", "--version"], "C++ compiler"),
        "cmake": _version_line(["cmake", "--version"], "CMake"),
        "python": platform.python_version(),
        "openfhe_version": _read_openfhe_version(prefix),
        "openfhe_prefix": str(prefix),
        "wsl": "microsoft" in release.lower() or "WSL_DISTRO_NAME" in os.environ,
    }


def _repository_record(path: str) -> dict[str, object]:
    resolved = _resolve_repository_file(REPO_ROOT / path, f"input {path}")
    return {
        "path": path,
        "sha256": _sha256(resolved),
        "bytes": resolved.stat().st_size,
        "media_type": "application/json",
        "role": "configuration",
    }


def _trace_input_records(
    data_root: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if data_root.resolve() != DEFAULT_DATA_ROOT.resolve():
        raise ArtifactRunnerError(f"M5 data root must be {DEFAULT_DATA_ROOT.resolve()}")
    contract = _load_json(REPO_ROOT / "config" / "moai_encoder_trace.json")
    try:
        required_files = contract["required_files"]
        layers = contract["layers"]
    except (KeyError, TypeError) as error:
        raise ArtifactRunnerError("encoder trace contract has an invalid shape") from error
    if not isinstance(required_files, dict) or len(required_files) != FILES_PER_LAYER:
        raise ArtifactRunnerError("M5 trace contract must bind exactly 37 files per layer")
    if not isinstance(layers, list) or [item.get("layer_id") for item in layers] != list(range(12)):
        raise ArtifactRunnerError("M5 trace contract must contain ordered layer ids 0..11")

    records: list[dict[str, object]] = []
    identities: list[dict[str, object]] = []
    for layer in layers:
        layer_id = layer["layer_id"]
        expected_hashes = layer.get("sha256")
        if not isinstance(expected_hashes, dict) or set(expected_hashes) != set(required_files):
            raise ArtifactRunnerError(f"layer {layer_id} trace hash map drifted")
        weight_bundle: list[dict[str, object]] = []
        trace_bundle: list[dict[str, object]] = []
        for logical_name in sorted(required_files):
            specification = required_files[logical_name]
            if not isinstance(specification, dict) or not isinstance(
                specification.get("path"), str
            ):
                raise ArtifactRunnerError(f"trace file mapping is invalid: {logical_name}")
            contract_path = specification["path"]
            relative = Path("data") / f"layer_{layer_id}" / contract_path
            resolved = _resolve_repository_file(
                REPO_ROOT / relative, f"layer-{layer_id} trace input {logical_name}"
            )
            actual_hash = _sha256(resolved)
            if actual_hash != expected_hashes[logical_name]:
                raise ArtifactRunnerError(
                    f"layer-{layer_id} trace hash mismatch for {logical_name}: "
                    f"expected={expected_hashes[logical_name]} actual={actual_hash}"
                )
            is_weight = "/parms/" in "/" + contract_path
            record = {
                "path": relative.as_posix(),
                "sha256": actual_hash,
                "bytes": resolved.stat().st_size,
                "media_type": "text/csv",
                "role": "weights" if is_weight else "trace",
            }
            records.append(record)
            bundle_record = {key: record[key] for key in ("path", "sha256", "bytes")}
            (weight_bundle if is_weight else trace_bundle).append(bundle_record)
        weight_bundle.sort(key=lambda item: str(item["path"]))
        trace_bundle.sort(key=lambda item: str(item["path"]))
        if (
            len(weight_bundle) != WEIGHT_FILES_PER_LAYER
            or len(trace_bundle) != TRACE_FILES_PER_LAYER
        ):
            raise ArtifactRunnerError(
                f"layer {layer_id} must bind 16 weight and 21 trace files"
            )
        identities.append(
            {
                "layer_id": layer_id,
                "weight_file_count": len(weight_bundle),
                "trace_file_count": len(trace_bundle),
                "weight_bundle_sha256": hashlib.sha256(
                    _canonical_json_bytes(weight_bundle)
                ).hexdigest(),
                "trace_bundle_sha256": hashlib.sha256(
                    _canonical_json_bytes(trace_bundle)
                ).hexdigest(),
            }
        )
    if len(records) != ENCODER_LAYERS * FILES_PER_LAYER:
        raise ArtifactRunnerError("M5 trace input count must be exactly 444")
    return records, identities


def _profile_record() -> dict[str, object]:
    path = _resolve_repository_file(REPO_ROOT / PROFILE_PATH, "feature profile")
    config = _load_json(path)
    if not isinstance(config, dict) or not isinstance(config.get("effective_profile"), dict):
        raise ArtifactRunnerError("feature profile lacks /effective_profile")
    effective = config["effective_profile"]
    effective_hash = hashlib.sha256(_canonical_json_bytes(effective)).hexdigest()
    if effective_hash != PROFILE_SHA256 or config.get("effective_profile_sha256") != PROFILE_SHA256:
        raise ArtifactRunnerError(
            f"feature profile hash drifted: expected={PROFILE_SHA256} actual={effective_hash}"
        )
    return {
        "id": "paper_compat",
        "security_claim": "none",
        "warning": WARNING,
        "source_config_path": PROFILE_PATH,
        "source_config_sha256": _sha256(path),
        "source_config_bytes": path.stat().st_size,
        "effective_profile_locator": PROFILE_LOCATOR,
        "canonicalization": CANONICALIZATION,
        "effective_profile_payload": effective,
        "effective_profile_sha256": effective_hash,
    }


def _require_executable_hash(executable: Path, expected_sha256: str, phase: str) -> None:
    actual = _sha256(executable)
    if actual != expected_sha256:
        raise ArtifactRunnerError(
            f"M5 executable changed {phase}: expected={expected_sha256} actual={actual}"
        )


def _require_input_hashes(inputs: list[dict[str, object]], phase: str) -> None:
    for record in inputs:
        path = record.get("path")
        expected_sha256 = record.get("sha256")
        expected_bytes = record.get("bytes")
        if (
            not isinstance(path, str)
            or not isinstance(expected_sha256, str)
            or not isinstance(expected_bytes, int)
        ):
            raise ArtifactRunnerError(f"invalid frozen input record {phase}: {record!r}")
        resolved = _resolve_repository_file(REPO_ROOT / path, f"frozen input {path}")
        actual_sha256 = _sha256(resolved)
        actual_bytes = resolved.stat().st_size
        if actual_sha256 != expected_sha256 or actual_bytes != expected_bytes:
            raise ArtifactRunnerError(
                f"frozen input changed {phase}: {path} expected_sha256={expected_sha256} "
                f"actual_sha256={actual_sha256} expected_bytes={expected_bytes} "
                f"actual_bytes={actual_bytes}"
            )


def _extract_records(stdout: str, test_name: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    formal_stdout = test_name in FORMAL_RUNTIME_TEST_NAMES

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key!r}")
            result[key] = value
        return result

    def reject_non_finite(token: str) -> None:
        raise ValueError(f"non-finite JSON number: {token}")

    for line in stdout.splitlines():
        start = line.find("{")
        if start < 0:
            continue
        try:
            candidate = json.loads(
                line[start:],
                object_pairs_hook=reject_duplicate_keys,
                parse_constant=reject_non_finite,
            )
        except (json.JSONDecodeError, ValueError) as error:
            if formal_stdout:
                raise ArtifactRunnerError(
                    "formal M5 stdout contains a malformed JSON-bearing line: "
                    f"{line[:400]!r} ({error})"
                ) from error
            continue
        if not isinstance(candidate, dict):
            if formal_stdout:
                raise ArtifactRunnerError(
                    "formal M5 stdout JSON lines must be objects"
                )
            continue
        candidate_test = candidate.get("test")
        if not isinstance(candidate_test, str):
            if formal_stdout:
                raise ArtifactRunnerError(
                    "formal M5 stdout JSON object lacks a string test discriminator"
                )
            continue
        diagnostic_fields = DIAGNOSTIC_ONLY_EVIDENCE_FIELDS.intersection(candidate)
        diagnostic_namespace = candidate_test.startswith(
            DIAGNOSTIC_EVIDENCE_TEST_PREFIXES
        )
        diagnostic_claim_scope = "claim_scope" in candidate and not (
            candidate_test == "openfhe_encoder_12_layer"
            and candidate["claim_scope"] == "m5_12_layer_correctness"
        )
        if diagnostic_namespace or diagnostic_fields or diagnostic_claim_scope:
            raise ArtifactRunnerError(
                "formal evidence stdout contains diagnostic-only JSON: "
                f"test={candidate_test!r} fields={sorted(diagnostic_fields)}"
            )
        allowed_non_evidence_keys = FORMAL_NON_EVIDENCE_JSON_ALLOWLIST.get(
            candidate_test
        )
        if allowed_non_evidence_keys is not None:
            if (
                set(candidate) != allowed_non_evidence_keys
                or candidate.get("security_claim") != "none"
            ):
                raise ArtifactRunnerError(
                    f"allowed non-evidence JSON {candidate_test!r} drifted"
                )
            continue
        if formal_stdout and candidate_test not in FORMAL_RUNTIME_TEST_NAMES:
            raise ArtifactRunnerError(
                "formal M5 stdout contains an unexpected JSON evidence record: "
                f"{candidate_test!r}"
            )
        if candidate_test == test_name:
            records.append(candidate)
    return records


def _finite_number(value: Any, label: str, minimum: float, maximum: float) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or not minimum <= value <= maximum
    ):
        raise ArtifactRunnerError(f"{label} violates [{minimum}, {maximum}]")
    return float(value)


def _typed_equal(actual: Any, expected: Any) -> bool:
    """Compare frozen JSON scalars without Python's bool/int equivalence."""
    return type(actual) is type(expected) and actual == expected


def _used_level(value: dict[str, Any], label: str) -> int:
    used = value["level"] + value["noise_scale_degree"] - 1
    if used < 0 or used >= EXPECTED_MULTIPLICATIVE_DEPTH:
        raise ArtifactRunnerError(
            f"{label} used level {used} is outside the depth-47 schedule"
        )
    expected_remaining = EXPECTED_MULTIPLICATIVE_DEPTH - used
    if value["remaining_levels"] != expected_remaining:
        raise ArtifactRunnerError(
            f"{label}.remaining_levels drifted: expected={expected_remaining} "
            f"from used level {used}, actual={value['remaining_levels']!r}"
        )
    return used


def _validate_metadata(value: Any, expected: dict[str, Any], label: str) -> None:
    if not isinstance(value, dict) or set(value) != METADATA_KEYS:
        raise ArtifactRunnerError(f"{label} metadata shape drifted")
    for key in (
        "level",
        "noise_scale_degree",
        "remaining_levels",
        "expected_scale_bits",
        "ciphertext_count",
    ):
        if not isinstance(value[key], int) or isinstance(value[key], bool):
            raise ArtifactRunnerError(f"{label}.{key} must be an integer")
    _used_level(value, label)
    for key in (
        "level",
        "noise_scale_degree",
        "remaining_levels",
        "expected_scale_bits",
        "ciphertext_count",
    ):
        if not _typed_equal(value[key], expected[key]):
            raise ArtifactRunnerError(
                f"{label}.{key} drifted: expected={expected[key]!r} "
                f"actual={value[key]!r}"
            )
    scale_bits = _finite_number(
        value["scale_bits"], f"{label}.scale_bits", 1.0, math.inf
    )
    if abs(scale_bits - float(value["expected_scale_bits"])) > 1e-3:
        raise ArtifactRunnerError(
            f"{label}.scale_bits differs from expected_scale_bits by more than 1e-3"
        )


def _validate_used_level_delta(
    output: dict[str, Any],
    anchor: dict[str, Any],
    expected_delta: int,
    label: str,
) -> None:
    actual_delta = _used_level(output, f"{label} output") - _used_level(
        anchor,
        f"{label} anchor",
    )
    if actual_delta != expected_delta:
        raise ArtifactRunnerError(
            f"{label} used-level delta drifted: expected={expected_delta} "
            f"actual={actual_delta}"
        )


def _validate_polynomial_checkpoint_metadata(value: Any, label: str) -> None:
    if POLYNOMIAL_CHECKPOINT_METADATA is None:
        raise ArtifactRunnerError(
            "M5 polynomial checkpoint metadata is not sealed by the two-layer seam run"
        )
    _validate_metadata(value, POLYNOMIAL_CHECKPOINT_METADATA, label)


def _validate_layer0_input_metadata(value: Any, label: str) -> None:
    if LAYER0_INPUT_METADATA is None:
        raise ArtifactRunnerError(
            "M5 layer-0 input metadata is not sealed by the crypto preflight"
        )
    _validate_metadata(value, LAYER0_INPUT_METADATA, label)


def _validate_crypto_preflight_record(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != CRYPTO_PREFLIGHT_KEYS:
        raise ArtifactRunnerError(f"{label} keys differ from the frozen contract")
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
        if not _typed_equal(value[key], expected):
            raise ArtifactRunnerError(
                f"{label}.{key} mismatch: expected={expected!r} actual={value[key]!r}"
            )
    _validate_layer0_input_metadata(
        value["input_metadata"], f"{label}.input_metadata"
    )
    return value


def _parse_crypto_preflight_output(stdout: str) -> dict[str, Any]:
    records = _extract_records(
        stdout,
        "openfhe_encoder_12_layer_crypto_preflight",
    )
    if len(records) != 1:
        raise ArtifactRunnerError(
            "M5 narrow CTest must emit exactly one crypto-preflight JSON record, "
            f"got {len(records)}"
        )
    return _validate_crypto_preflight_record(
        records[0],
        "M5 narrow CTest crypto preflight",
    )


def _validate_handoff_input_metadata(value: Any, label: str) -> None:
    if LAYER_HANDOFF_INPUT_METADATA is None:
        raise ArtifactRunnerError(
            "M5 handoff input metadata is not sealed by the two-layer seam run"
        )
    _validate_metadata(value, LAYER_HANDOFF_INPUT_METADATA, label)


def _validate_quality(value: Any, label: str) -> None:
    if not isinstance(value, dict) or set(value) != QUALITY_KEYS:
        raise ArtifactRunnerError(f"{label} quality shape drifted")
    _finite_number(value["relative_l2"], f"{label}.relative_l2", 0.0, 5e-2)
    _finite_number(value["cosine"], f"{label}.cosine", 0.99, 1.000000000001)
    _finite_number(value["max_absolute"], f"{label}.max_absolute", 0.0, math.inf)


def _expected_cumulative(layer_id: int) -> dict[str, int]:
    completed_layers = layer_id + 1
    completed_refreshes = min(completed_layers, ENCODER_LAYERS - 1)
    return {
        key: EXPECTED_LAYER_COUNTS[key] * completed_layers
        + EXPECTED_REFRESH_COUNTS[key] * completed_refreshes
        for key in EXPECTED_LAYER_COUNTS
    }


def _validate_counts(value: Any, expected: dict[str, int], label: str) -> None:
    if not isinstance(value, dict) or set(value) != COUNT_KEYS:
        raise ArtifactRunnerError(f"{label} operation-count shape drifted")
    if any(
        not isinstance(item, int) or isinstance(item, bool) or item < 0
        for item in value.values()
    ):
        raise ArtifactRunnerError(f"{label} operation counts must be non-negative integers")
    if value != expected:
        raise ArtifactRunnerError(f"{label} operation schedule drifted")


def _validate_layer_record(record: dict[str, Any], layer_id: int) -> None:
    label = f"layer[{layer_id}]"
    regime = "layer_0" if layer_id == 0 else "layers_1_to_11"
    if set(record) != LAYER_RECORD_KEYS:
        raise ArtifactRunnerError(
            f"{label} keys differ: missing={sorted(LAYER_RECORD_KEYS - set(record))} "
            f"extra={sorted(set(record) - LAYER_RECORD_KEYS)}"
        )
    expected_scalars = {
        "test": "openfhe_encoder_12_layer_layer",
        "profile": "paper_compat",
        "security_claim": "none",
        "layer_id": layer_id,
        "weights_layer_id": layer_id,
        "chain_input_source": (
            "client_encrypted_trace_input" if layer_id == 0
            else "previous_ciphertext_output_after_refresh"
        ),
        "handoff_refresh_performed": layer_id < ENCODER_LAYERS - 1,
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
        if not _typed_equal(record.get(key), expected):
            raise ArtifactRunnerError(
                f"{label}.{key} mismatch: expected={expected!r} actual={record.get(key)!r}"
            )
    output_metadata = RAW_OUTPUT_METADATA
    if layer_id == 0:
        _validate_layer0_input_metadata(
            record["input_metadata"], f"{label}.input_metadata"
        )
    else:
        _validate_handoff_input_metadata(
            record["input_metadata"], f"{label}.input_metadata"
        )
    _validate_metadata(
        record["raw_output_metadata"], output_metadata, f"{label}.raw_output_metadata"
    )
    _validate_polynomial_checkpoint_metadata(
        record["softmax_denominator_metadata"],
        f"{label}.softmax_denominator_metadata",
    )
    _validate_metadata(
        record["attention_output_metadata"],
        (
            LAYER0_ATTENTION_OUTPUT_METADATA
            if layer_id == 0
            else POST_REFRESH_ATTENTION_OUTPUT_METADATA
        ),
        f"{label}.attention_output_metadata",
    )
    _validate_polynomial_checkpoint_metadata(
        record["ln1_variance_metadata"],
        f"{label}.ln1_variance_metadata",
    )
    _validate_metadata(
        record["ln1_output_metadata"],
        (
            LAYER0_LN1_OUTPUT_METADATA
            if layer_id == 0
            else POST_REFRESH_LN1_OUTPUT_METADATA
        ),
        f"{label}.ln1_output_metadata",
    )
    _validate_metadata(
        record["ffn_output_metadata"],
        (
            LAYER0_FFN_OUTPUT_METADATA
            if layer_id == 0
            else POST_REFRESH_FFN_OUTPUT_METADATA
        ),
        f"{label}.ffn_output_metadata",
    )
    _validate_polynomial_checkpoint_metadata(
        record["ln2_variance_metadata"],
        f"{label}.ln2_variance_metadata",
    )
    _validate_used_level_delta(
        record["attention_output_metadata"],
        record["input_metadata"],
        RELATIVE_USED_LEVEL_DELTAS["attention_output_from_layer_input"][regime],
        f"{label}.attention_output_from_layer_input",
    )
    _validate_used_level_delta(
        record["ln1_output_metadata"],
        record["ln1_variance_metadata"],
        RELATIVE_USED_LEVEL_DELTAS["ln1_output_from_ln1_variance"][regime],
        f"{label}.ln1_output_from_ln1_variance",
    )
    _validate_used_level_delta(
        record["ffn_output_metadata"],
        record["ln1_output_metadata"],
        RELATIVE_USED_LEVEL_DELTAS["ffn_output_from_ln1_output"],
        f"{label}.ffn_output_from_ln1_output",
    )
    _validate_used_level_delta(
        record["raw_output_metadata"],
        record["ln2_variance_metadata"],
        RELATIVE_USED_LEVEL_DELTAS["raw_output_from_ln2_variance"],
        f"{label}.raw_output_from_ln2_variance",
    )
    for key in ("input_quality", "output_quality", "exact_trace_diagnostic"):
        _validate_quality(record[key], f"{label}.{key}")
    _finite_number(record["inactive_max_abs"], f"{label}.inactive_max_abs", 0.0, 1e-6)
    _finite_number(
        record["inactive_sentinel_max_error"],
        f"{label}.inactive_sentinel_max_error",
        0.0,
        math.inf,
    )
    sentinel_ranges = record["inactive_polynomial_sentinel_ranges"]
    if (
        not isinstance(sentinel_ranges, dict)
        or set(sentinel_ranges) != INACTIVE_SENTINEL_RANGE_KEYS
    ):
        raise ArtifactRunnerError(
            f"{label}.inactive_polynomial_sentinel_ranges drifted"
        )
    for key in sorted(INACTIVE_SENTINEL_RANGE_KEYS):
        observed = sentinel_ranges[key]
        frozen = FROZEN_RANGES[key]
        if not isinstance(observed, list) or len(observed) != 2:
            raise ArtifactRunnerError(
                f"{label}.inactive_polynomial_sentinel_ranges.{key} "
                "must contain [min,max]"
            )
        minimum = _finite_number(
            observed[0],
            f"{label}.inactive_polynomial_sentinel_ranges.{key}[0]",
            frozen[0],
            frozen[1],
        )
        maximum = _finite_number(
            observed[1],
            f"{label}.inactive_polynomial_sentinel_ranges.{key}[1]",
            frozen[0],
            frozen[1],
        )
        if minimum > maximum:
            raise ArtifactRunnerError(
                f"{label}.inactive_polynomial_sentinel_ranges.{key} is reversed"
            )
    ranges = record["encrypted_polynomial_input_ranges"]
    if not isinstance(ranges, dict) or set(ranges) != RANGE_KEYS:
        raise ArtifactRunnerError(f"{label}.encrypted_polynomial_input_ranges drifted")
    for key, frozen in FROZEN_RANGES.items():
        observed = ranges[key]
        if not isinstance(observed, list) or len(observed) != 2:
            raise ArtifactRunnerError(f"{label}.ranges.{key} must contain [min,max]")
        minimum = _finite_number(observed[0], f"{label}.ranges.{key}[0]", frozen[0], frozen[1])
        maximum = _finite_number(observed[1], f"{label}.ranges.{key}[1]", frozen[0], frozen[1])
        if minimum > maximum:
            raise ArtifactRunnerError(f"{label}.ranges.{key} is reversed")
    _validate_counts(
        record["refresh_operation_counts"],
        EXPECTED_REFRESH_COUNTS
        if layer_id < ENCODER_LAYERS - 1
        else ZERO_COUNTS,
        f"{label}.refresh_operation_counts",
    )
    _validate_counts(
        record["layer_operation_counts"], EXPECTED_LAYER_COUNTS,
        f"{label}.layer_operation_counts"
    )
    _validate_counts(
        record["cumulative_operation_counts"], _expected_cumulative(layer_id),
        f"{label}.cumulative_operation_counts"
    )


def _validate_final_record(final: dict[str, Any], layers: list[dict[str, Any]]) -> None:
    if set(final) != FINAL_RECORD_KEYS:
        raise ArtifactRunnerError(
            f"final record keys differ: missing={sorted(FINAL_RECORD_KEYS - set(final))} "
            f"extra={sorted(set(final) - FINAL_RECORD_KEYS)}"
        )
    expected = {
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
        "inactive_sentinel_range_status": "all_client_validated",
        "multiplicative_depth": EXPECTED_MULTIPLICATIVE_DEPTH,
        "max_observed_level": EXPECTED_MAX_OBSERVED_LEVEL,
        "max_polynomial_depth": EXPECTED_MAX_POLYNOMIAL_DEPTH,
        "timing_claim": False,
        "latency_kind": "non_benchmark_diagnostic",
        "finite": True,
        "passed": True,
    }
    for key, value in expected.items():
        if not _typed_equal(final.get(key), value):
            raise ArtifactRunnerError(
                f"final.{key} mismatch: expected={value!r} actual={final.get(key)!r}"
            )
    for field in (
        "fixture_load_oracle_ms",
        "setup_keygen_ms",
        "client_encrypt_ms",
        "server_online_diagnostic_ms",
        "client_checkpoint_validate_ms",
    ):
        _finite_number(final[field], f"final.{field}", 0.0, math.inf)
    if final["server_online_diagnostic_ms"] <= 0.0:
        raise ArtifactRunnerError("final.server_online_diagnostic_ms must be positive")
    _finite_number(final["relative_l2"], "final.relative_l2", 0.0, 5e-2)
    _finite_number(final["cosine"], "final.cosine", 0.99, 1.000000000001)
    _finite_number(final["max_absolute"], "final.max_absolute", 0.0, math.inf)
    _finite_number(final["inactive_max_abs"], "final.inactive_max_abs", 0.0, 1e-6)
    _finite_number(
        final["inactive_sentinel_max_error"],
        "final.inactive_sentinel_max_error",
        0.0,
        math.inf,
    )
    _validate_metadata(
        final["final_metadata"], RAW_OUTPUT_METADATA, "final.final_metadata"
    )
    if final["final_metadata"] != layers[-1]["raw_output_metadata"]:
        raise ArtifactRunnerError("final.final_metadata differs from layer 11 raw output")
    _validate_counts(final["operation_counts"], EXPECTED_TOTAL_COUNTS, "final.operation_counts")
    peak_rss = final["peak_rss_bytes"]
    if not isinstance(peak_rss, int) or isinstance(peak_rss, bool) or peak_rss <= 0:
        raise ArtifactRunnerError("final.peak_rss_bytes must be a positive integer")
    bindings = {
        "relative_l2": "relative_l2",
        "cosine": "cosine",
        "max_absolute": "max_absolute",
    }
    for final_key, layer_key in bindings.items():
        if not math.isclose(
            float(final[final_key]),
            float(layers[-1]["output_quality"][layer_key]),
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            raise ArtifactRunnerError(f"final.{final_key} differs from layer 11 output")
    for field in ("inactive_max_abs", "inactive_sentinel_max_error"):
        layer_maximum = max(float(layer[field]) for layer in layers)
        if not math.isclose(
            float(final[field]),
            layer_maximum,
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            raise ArtifactRunnerError(f"final.{field} differs from 12-layer maximum")


def _runtime_command(config: RunnerConfig, resource_file: Path) -> list[str]:
    command = [
        str(TIME_EXECUTABLE),
        "--format=%M",
        f"--output={resource_file}",
        str(config.executable),
        "--data-root",
        str(config.data_root),
    ]
    forbidden = {
        "--diagnostic-layer-count",
        "--metadata-calibration",
        "--preflight-only",
        "--crypto-preflight-only",
        "--layer",
        "--input-level",
    }
    if forbidden.intersection(command):  # pragma: no cover - immutable command guard
        raise ArtifactRunnerError("M5 evidence command contains a diagnostic/prefix option")
    return command


def _run_once(config: RunnerConfig) -> RunSample:
    if (
        LAYER0_INPUT_METADATA is None
        or LAYER_HANDOFF_INPUT_METADATA is None
        or POLYNOMIAL_CHECKPOINT_METADATA is None
    ):
        raise ArtifactRunnerError(
            "M5 layer-0/handoff/checkpoint metadata is not sealed by the crypto "
            "preflight and two-layer seam run"
        )
    descriptor, resource_name = tempfile.mkstemp(
        prefix="moai-m5-time-", suffix=".txt", dir="/tmp"
    )
    os.close(descriptor)
    resource_file = Path(resource_name)
    command = _runtime_command(config, resource_file)
    started = _timestamp()
    monotonic_start = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
            env={**os.environ, "LC_ALL": "C"},
        )
        elapsed_seconds = time.perf_counter() - monotonic_start
        finished = _timestamp()
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise ArtifactRunnerError(
                f"M5 correctness run exited with {completed.returncode}: {detail[:4000]}"
            )
        if completed.stderr.strip():
            raise ArtifactRunnerError(
                "M5 correctness run emitted stderr despite exit_code=0: "
                f"{completed.stderr.strip()[:4000]}"
            )
        try:
            resource_lines = resource_file.read_text(encoding="utf-8").splitlines()
            if len(resource_lines) != 1:
                raise ValueError("expected one GNU time output line")
            peak_rss_kib = int(resource_lines[0])
        except (OSError, UnicodeError, ValueError) as error:
            raise ArtifactRunnerError(f"cannot parse M5 peak RSS: {error}") from error
        if elapsed_seconds <= 0.0 or peak_rss_kib <= 0:
            raise ArtifactRunnerError("M5 run produced invalid resource metrics")
        layers = _extract_records(completed.stdout, "openfhe_encoder_12_layer_layer")
        finals = _extract_records(completed.stdout, "openfhe_encoder_12_layer")
        if len(layers) != ENCODER_LAYERS:
            raise ArtifactRunnerError(
                f"M5 run must emit exactly 12 layer records, got {len(layers)}"
            )
        if len(finals) != 1:
            raise ArtifactRunnerError(
                f"M5 run must emit exactly one final record, got {len(finals)}"
            )
        for layer_id, record in enumerate(layers):
            _validate_layer_record(record, layer_id)
        _validate_final_record(finals[0], layers)
        if finals[0]["peak_rss_bytes"] > peak_rss_kib * 1024:
            raise ArtifactRunnerError("runtime peak RSS exceeds GNU time peak RSS")
        return RunSample(
            completed.stdout,
            tuple(layers),
            finals[0],
            elapsed_seconds,
            peak_rss_kib,
            _command_record(command, completed.returncode, started, finished),
        )
    finally:
        resource_file.unlink(missing_ok=True)


def _write_metrics(path: Path, sample: RunSample) -> None:
    final = sample.final
    metadata = final["final_metadata"]
    counts = final["operation_counts"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerow(
            {
                "run": 1,
                "exit_code": 0,
                "elapsed_seconds": format(sample.elapsed_seconds, ".17g"),
                "peak_rss_kib": sample.peak_rss_kib,
                **{
                    field: format(float(final[field]), ".17g")
                    for field in (
                        "fixture_load_oracle_ms",
                        "setup_keygen_ms",
                        "client_encrypt_ms",
                        "server_online_diagnostic_ms",
                        "client_checkpoint_validate_ms",
                    )
                },
                "final_rel_l2": format(float(final["relative_l2"]), ".17g"),
                "final_cosine": format(float(final["cosine"]), ".17g"),
                "final_max_absolute": format(float(final["max_absolute"]), ".17g"),
                "inactive_max_abs": format(float(final["inactive_max_abs"]), ".17g"),
                "inactive_sentinel_max_error": format(
                    float(final["inactive_sentinel_max_error"]), ".17g"
                ),
                "checkpoint_metadata_sha256": _checkpoint_metadata_sha256(
                    sample.layers
                ),
                "final_level": metadata["level"],
                "final_noise_scale_degree": metadata["noise_scale_degree"],
                "final_remaining_levels": metadata["remaining_levels"],
                "final_scale_bits": format(float(metadata["scale_bits"]), ".17g"),
                "final_ciphertext_count": metadata["ciphertext_count"],
                **counts,
                "timing_claim": "false",
                "latency_kind": "non_benchmark_diagnostic",
                "finite": "true",
                "passed": "true",
            }
        )


def _artifact_record(root: Path, path: str, role: str, media_type: str) -> dict[str, object]:
    resolved = root / path
    return {
        "path": path,
        "sha256": _sha256(resolved),
        "bytes": resolved.stat().st_size,
        "media_type": media_type,
        "role": role,
    }


def _validator_command(manifest_path: Path) -> list[str]:
    return [
        sys.executable,
        str(VALIDATOR_PATH),
        "--manifest",
        str(manifest_path),
        "--verify-git",
    ]


def _validate_artifact(manifest_path: Path) -> None:
    command = _validator_command(manifest_path)
    completed = subprocess.run(
        command, cwd=REPO_ROOT, check=False, capture_output=True, text=True
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise ArtifactRunnerError(f"artifact validator rejected the M5 run: {detail}")


def _metrics_manifest(sample: RunSample) -> dict[str, object]:
    final = sample.final
    omitted_layer_fields = {
        "test",
        "profile",
        "security_claim",
        "range_validation_owner",
        "checkpoint_decryption_owner",
        "server_decryptions",
        "server_plaintext_activations",
    }
    return {
        "repeat_count": 1,
        "successful_repeats": 1,
        "warmup_count": 0,
        "elapsed_seconds": sample.elapsed_seconds,
        "peak_rss_kib": sample.peak_rss_kib,
        "timing_claim": False,
        "latency_kind": "non_benchmark_diagnostic",
        "checkpoint_metadata_sha256": _checkpoint_metadata_sha256(sample.layers),
        "phase_latency_ms": {
            key: final[key]
            for key in (
                "fixture_load_oracle_ms",
                "setup_keygen_ms",
                "client_encrypt_ms",
                "server_online_diagnostic_ms",
                "client_checkpoint_validate_ms",
            )
        },
        "layers": [
            {
                key: value
                for key, value in layer.items()
                if key not in omitted_layer_fields
            }
            for layer in sample.layers
        ],
        "final_quality": {
            "relative_l2": final["relative_l2"],
            "cosine": final["cosine"],
            "max_absolute": final["max_absolute"],
            "inactive_max_abs": final["inactive_max_abs"],
            "inactive_sentinel_max_error": final["inactive_sentinel_max_error"],
            "inactive_sentinel_range_status": final[
                "inactive_sentinel_range_status"
            ],
        },
        "final_metadata": final["final_metadata"],
        "operation_counts": final["operation_counts"],
        "finite": True,
    }


def generate_m5_artifact(config: RunnerConfig) -> Path:
    if RUN_ID_PATTERN.fullmatch(config.run_id) is None:
        raise ArtifactRunnerError(f"invalid run id: {config.run_id!r}")
    started_at = _timestamp()
    executable = _resolve_m5_executable(config.executable, require_exists=False)
    data_root = config.data_root.resolve()
    if not data_root.is_dir() or data_root != DEFAULT_DATA_ROOT.resolve():
        raise ArtifactRunnerError(f"M5 data root must be {DEFAULT_DATA_ROOT.resolve()}")
    if not TIME_EXECUTABLE.is_file() or not os.access(TIME_EXECUTABLE, os.X_OK):
        raise ArtifactRunnerError(f"GNU time is unavailable: {TIME_EXECUTABLE}")

    output_root = _resolve_output_root(config.output_root)
    git_state = _preflight_git()
    preflight = _run_m5_preflight(
        executable.parent, executable, config.openfhe_prefix.resolve()
    )
    executable = _resolve_m5_executable(DEFAULT_EXECUTABLE)
    executable_sha256 = _sha256(executable)
    environment = _environment(config.openfhe_prefix)
    profile = _profile_record()
    trace_inputs, layer_identities = _trace_input_records(data_root)
    inputs = sorted(
        [*(_repository_record(path) for path in REQUIRED_INPUTS), *trace_inputs],
        key=lambda record: str(record["path"]),
    )
    if (
        len(inputs) != EXPECTED_INPUT_COUNT
        or len({item["path"] for item in inputs}) != EXPECTED_INPUT_COUNT
    ):
        raise ArtifactRunnerError("M5 inputs must contain exactly 448 unique records")
    input_by_path = {record["path"]: record for record in inputs}

    output_root.mkdir(parents=True, exist_ok=True)
    run_root = output_root / config.run_id
    try:
        run_root.mkdir(exist_ok=False)
    except FileExistsError as error:
        raise ArtifactRunnerError(f"run directory already exists: {run_root}") from error

    remove_unvalidated = True
    try:
        _require_executable_hash(executable, executable_sha256, "before correctness run")
        _require_input_hashes(inputs, "before correctness run")
        sample = _run_once(
            RunnerConfig(executable, data_root, output_root, config.run_id, config.openfhe_prefix)
        )
        _require_executable_hash(executable, executable_sha256, "after correctness run")
        _require_input_hashes(inputs, "after correctness run")

        stdout_path = run_root / "stdout.log"
        metrics_path = run_root / "metrics.csv"
        checksum_path = run_root / "SHA256SUMS"
        manifest_path = run_root / "manifest.json"
        stdout_path.write_text(
            sample.stdout if sample.stdout.endswith("\n") else sample.stdout + "\n",
            encoding="utf-8",
        )
        _write_metrics(metrics_path, sample)
        checksum_path.write_text(
            f"{_sha256(stdout_path)}  stdout.log\n{_sha256(metrics_path)}  metrics.csv\n",
            encoding="utf-8",
        )

        validator_command = _validator_command(manifest_path)
        commands = [
            *git_state.commands,
            *preflight.commands,
            sample.command,
            _command_record(validator_command, 0),
        ]
        if len(commands) != 10:
            raise ArtifactRunnerError(
                f"M5 command contract must contain 10 records, got {len(commands)}"
            )
        executable_path = executable.relative_to(REPO_ROOT.resolve()).as_posix()
        manifest: dict[str, object] = {
            "schema_version": M5_SCHEMA_VERSION,
            "run_id": config.run_id,
            "milestone": "M5",
            "started_at": started_at,
            "finished_at": _timestamp(),
            "git": {
                "repository_root": str(REPO_ROOT),
                "branch": BRANCH,
                "local_commit": git_state.head,
                "clean": True,
                "tracking_ref": TRACKING_REF,
                "tracking_commit": git_state.tracking_head,
                "remote_name": REMOTE_NAME,
                "remote_ref": REMOTE_REF,
                "remote_commit": git_state.remote_head,
            },
            "commands": commands,
            "environment": environment,
            "profile": profile,
            "inputs": inputs,
            "workload": {
                "scope": (
                    "M5 server-only five-token BERT-base 12-layer encoder "
                    "ciphertext-chain trace replay"
                ),
                "backend": "OpenFHE CKKS CPU",
                "executable_path": executable_path,
                "executable_sha256": executable_sha256,
                "executable_bytes": executable.stat().st_size,
                "repeat_count": 1,
                "warmup_count": 0,
                "timing_claim": False,
                "latency_kind": "non_benchmark_diagnostic",
            },
            "contracts": {
                "approximation_config_path": "config/openfhe_approximations.json",
                "approximation_config_sha256": input_by_path[
                    "config/openfhe_approximations.json"
                ]["sha256"],
                "channel_scales_config_path": "config/moai_trace_channel_scales.json",
                "channel_scales_config_sha256": input_by_path[
                    "config/moai_trace_channel_scales.json"
                ]["sha256"],
                "encoder_trace_contract_path": "config/moai_encoder_trace.json",
                "encoder_trace_contract_sha256": input_by_path[
                    "config/moai_encoder_trace.json"
                ]["sha256"],
                "profile_config_path": PROFILE_PATH,
                "profile_config_sha256": input_by_path[PROFILE_PATH]["sha256"],
                "execution": {
                    "mode": "server-only",
                    "encoder_layers": 12,
                    "layer_ids": list(range(12)),
                    "chain_mode": "ciphertext_output_to_next_input",
                    "client_encrypt_calls": 1,
                    "encrypted_input_ciphertexts": 5,
                    "plaintext_activation_resets": 0,
                    "server_layer_evaluations": 12,
                    "inter_layer_refreshes": 11,
                    "trace_shape": [5, 768],
                    "feature_block_size": 1024,
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
                },
                "metadata_schedule": _metadata_schedule_contract(),
                "crypto_preflight": preflight.crypto_preflight,
                "layer_input_identities": layer_identities,
                "thresholds": {
                    "per_layer_relative_l2_max": 5e-2,
                    "per_layer_cosine_min": 0.99,
                    "final_relative_l2_max": 5e-2,
                    "final_cosine_min": 0.99,
                    "inactive_max_abs": 1e-6,
                    "inactive_sentinel_in_interval_required": True,
                    "finite_required": True,
                },
                "polynomial_intervals": {
                    key: [minimum, maximum]
                    for key, (minimum, maximum) in FROZEN_RANGES.items()
                },
            },
            "metrics": _metrics_manifest(sample),
            "gate": {
                "passed": True,
                "decision": "PASS_M5_12_LAYER_OPENFHE_ENCODER",
                "checks": {
                    "build": "PASS",
                    "trust_boundary": "PASS",
                    "input_integrity": "PASS",
                    "ciphertext_chain": "PASS",
                    "approximation_ranges": "PASS",
                    "correctness": "PASS",
                    "artifact_integrity": "PASS",
                    "remote_sha": "PASS",
                },
            },
            "artifacts": [
                _artifact_record(run_root, "stdout.log", "stdout", "text/plain"),
                _artifact_record(run_root, "metrics.csv", "metrics", "text/csv"),
                _artifact_record(run_root, "SHA256SUMS", "checksum", "text/plain"),
            ],
            "claim_boundary": [
                "Five-token M5 12-layer encoder trace replay only; not task-level inference.",
                "paper_compat OpenFHE CKKS CPU parameters with security_claim=none.",
                "Timing is a non-benchmark diagnostic; no speedup claim.",
                "GPU, Discrete CKKS/FBT, QDQ, tokenizer, and classifier are excluded.",
            ],
            "verdict": "GO",
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        _validate_artifact(manifest_path)
        remove_unvalidated = False
        return run_root
    finally:
        if remove_unvalidated:
            try:
                shutil.rmtree(run_root)
            except OSError as cleanup_error:
                raise ArtifactRunnerError(
                    f"failed to remove unvalidated run directory {run_root}: {cleanup_error}"
                ) from cleanup_error


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, default=DEFAULT_EXECUTABLE)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id")
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    git_sha_hint = "unknown"
    try:
        git_sha_hint = subprocess.run(
            ["git", "rev-parse", "--short=7", "HEAD"],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        ).stdout.strip() or "unknown"
        run_id = arguments.run_id or (
            f"{datetime.now().astimezone().strftime('%Y%m%dT%H%M%S%z')}"
            f"-m5-{git_sha_hint}"
        )
        run_root = generate_m5_artifact(
            RunnerConfig(
                arguments.executable,
                arguments.data_root,
                arguments.output_root,
                run_id,
            )
        )
    except (ArtifactRunnerError, OSError, ValueError) as error:
        print(f"run_openfhe_encoder12_artifact failed: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "artifact_root": str(run_root),
                "manifest": str(run_root / "manifest.json"),
                "milestone": "M5",
                "run_id": run_root.name,
                "git_sha_hint": git_sha_hint,
                "validated": True,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
