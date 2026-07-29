#!/usr/bin/env python3
"""Run and seal the frozen M4 OpenFHE encoder-layer evidence bundle.

This runner is deliberately narrow: it supports only the server-only M4 layer-1
trace replay at input level 29.  It performs one unrecorded warm-up followed by
five measured executions, then seals the measured stdout and resource metrics
into the schema-v2 artifact contract.  M5 is intentionally not implemented.
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
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXECUTABLE = REPO_ROOT / "build-openfhe" / "openfhe_encoder_layer_smoke"
DEFAULT_DATA_ROOT = REPO_ROOT / "data"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "openfhe"
DEFAULT_OPENFHE_PREFIX = Path("/home/shawnsheep/opt/openfhe_v1_5_1")
VALIDATOR_PATH = REPO_ROOT / "scripts" / "validate_openfhe_artifact.py"
TIME_EXECUTABLE = Path("/usr/bin/time")
REMOTE_NAME = "origin"
REMOTE_REF = "refs/heads/refactor/openfhe-cpu"
BRANCH = "refactor/openfhe-cpu"
PROFILE_PATH = "config/paper_compat_feature_packed.json"
PROFILE_LOCATOR = "/effective_profile"
PROFILE_SHA256 = "94f30e628e21f02146ce7ed9820194eabba3820f6e1e17176a31f8c5acf8b0be"
WARNING = "Research reproduction parameters only. Do not claim 128-bit security."
CANONICALIZATION = "MOAI-json-sort-keys-compact-utf8-v1"
REQUIRED_INPUTS = (
    "config/moai_encoder_trace.json",
    "config/moai_trace_channel_scales.json",
    "config/openfhe_approximations.json",
    PROFILE_PATH,
)
CHECKPOINT_NAMES = (
    "attention_output",
    "self_layernorm_output",
    "ffn_output",
    "encoder_output",
)
EXPECTED_MULTIPLICATIVE_DEPTH = 47
EXPECTED_MAX_OBSERVED_LEVEL = 45
EXPECTED_MAX_POLYNOMIAL_DEPTH = 10
EXPECTED_CHECKPOINTS = (
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
EXPECTED_CHECKPOINT_METADATA_SHA256 = (
    "c4c1c85e52154215784b9fa93a584d824dde94a644e5af997882486aac01a6db"
)
OPERATION_COUNT_KEYS = {
    "rotations",
    "ct_pt_multiplications",
    "ct_ct_multiplications",
    "explicit_rescale_requests",
    "chebyshev_evaluations",
    "estimated_polynomial_multiplications",
    "bootstraps",
    "bootstrap_iterations",
}
EXPECTED_OPERATION_COUNTS = {
    "rotations": 6300,
    "ct_pt_multiplications": 51865,
    "ct_ct_multiplications": 95,
    "explicit_rescale_requests": 800,
    "chebyshev_evaluations": 55,
    "estimated_polynomial_multiplications": 1150,
    "bootstraps": 25,
    "bootstrap_iterations": 50,
}
TARGET_RECORD_KEYS = {
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
DIAGNOSTIC_RECORD_KEYS = {
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
    "fixture_load_ms",
    "setup_keygen_ms",
    "client_encrypt_ms",
    "server_online_ms",
    "client_decrypt_validate_ms",
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
PHASE_LATENCY_FIELDS = (
    "fixture_load_ms",
    "setup_keygen_ms",
    "client_encrypt_ms",
    "server_online_ms",
    "client_decrypt_validate_ms",
)
TOKENS_PER_BATCH = 5
M4_CTEST_PATTERN = (
    "^(moai_trace_contract|openfhe_(server_trust_boundary|profile_contract|"
    "profile_validator_contract|"
    "artifact_schema_contract|artifact_validator_contract|"
    "encoder_artifact_runner_contract|evaluation_key_bundle_smoke|"
    "feature_packed_smoke|"
    "feature_packed_attention_smoke|feature_bootstrap_smoke|"
    "encoder_fixture_contract|"
    "encoder_plaintext_oracle_smoke|encoder_trace_contract|"
    "encoder_trace_validator_contract))$"
)
CSV_FIELDS = (
    "run",
    "exit_code",
    "elapsed_seconds",
    "peak_rss_kib",
    *PHASE_LATENCY_FIELDS,
    "batch_total_ms",
    "batch_amortized_ms_per_token",
    "server_amortized_ms_per_token",
    "final_rel_l2",
    "final_cosine",
    "inactive_max_abs",
    "multiplicative_depth",
    "max_observed_level",
    "max_polynomial_depth",
    "checkpoint_metadata_sha256",
    "bootstraps",
    "bootstrap_iterations",
)
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+:-]{0,127}$")
SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")


class ArtifactRunnerError(RuntimeError):
    """Raised when an M4 artifact cannot be produced without weakening a gate."""


@dataclass(frozen=True)
class GitState:
    head: str
    remote_head: str
    commands: tuple[dict[str, object], ...]


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
    record: dict[str, Any]
    diagnostic: dict[str, Any]
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
        raise ArtifactRunnerError(f"effective profile is not canonical JSON: {error}") from error


def _checkpoint_metadata_sha256(checkpoints: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(checkpoints)).hexdigest()


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
    """Return an absolute path without following its final or parent symlinks."""

    return Path(os.path.abspath(os.fspath(path)))


def _resolve_m4_executable(path: Path, *, require_exists: bool = True) -> Path:
    candidate = _lexical_absolute(path)
    expected = _lexical_absolute(DEFAULT_EXECUTABLE)
    if candidate != expected:
        raise ArtifactRunnerError(
            f"M4 executable must be the fixed build target {DEFAULT_EXECUTABLE}"
        )
    try:
        relative_parts = candidate.relative_to(REPO_ROOT).parts
    except ValueError as error:  # pragma: no cover - fixed target is repository-local
        raise ArtifactRunnerError(f"M4 executable must stay inside {REPO_ROOT}") from error
    cursor = REPO_ROOT
    for part in relative_parts:
        cursor /= part
        if cursor.is_symlink():
            raise ArtifactRunnerError(f"M4 executable path must not contain symlinks: {cursor}")
    if not require_exists:
        if candidate.exists() and not candidate.is_file():
            raise ArtifactRunnerError(f"M4 executable is not a regular file: {candidate}")
        return candidate
    if not candidate.is_file():
        raise ArtifactRunnerError(f"M4 executable is not a file: {candidate}")
    if not os.access(candidate, os.X_OK):
        raise ArtifactRunnerError(f"M4 executable is not executable: {candidate}")
    return candidate


def _resolve_output_root(path: Path) -> Path:
    resolved = path.resolve()
    allowed = DEFAULT_OUTPUT_ROOT.resolve()
    try:
        resolved.relative_to(allowed)
    except ValueError as error:
        raise ArtifactRunnerError(f"output root must stay inside {allowed}: {path}") from error
    return resolved


def _run_git(arguments: list[str]) -> tuple[str, dict[str, object]]:
    command = ["git", *arguments]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    record: dict[str, object] = {
        "command": shlex.join(command),
        "cwd": str(REPO_ROOT),
        "exit_code": completed.returncode,
        "phase": "artifact_generation",
    }
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise ArtifactRunnerError(f"{shlex.join(command)} failed: {detail}")
    return completed.stdout.strip(), record


def _run_checked_command(command: list[str], label: str) -> dict[str, object]:
    started_at = _timestamp()
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    finished_at = _timestamp()
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        if len(detail) > 4000:
            detail = detail[-4000:]
        raise ArtifactRunnerError(
            f"{label} failed with {completed.returncode}: {detail}"
        )
    return {
        "command": shlex.join(command),
        "cwd": str(REPO_ROOT),
        "exit_code": completed.returncode,
        "phase": "artifact_generation",
        "started_at": started_at,
        "finished_at": finished_at,
    }


def _read_openfhe_version(openfhe_prefix: Path) -> str:
    version_file = openfhe_prefix / "lib" / "OpenFHE" / "OpenFHEConfigVersion.cmake"
    try:
        contents = version_file.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ArtifactRunnerError(f"cannot read OpenFHE version file: {error}") from error
    match = re.search(r'^set\(PACKAGE_VERSION "([0-9]+\.[0-9]+\.[0-9]+)"\)$', contents, re.M)
    if match is None or match.group(1) != "1.5.1":
        raise ArtifactRunnerError(
            f"OpenFHE package version must be 1.5.1: {version_file}"
        )
    return match.group(1)


def _verify_build_configuration(build_root: Path, openfhe_prefix: Path) -> None:
    if build_root.resolve() != DEFAULT_EXECUTABLE.parent.resolve():
        raise ArtifactRunnerError(
            f"M4 build root must be {DEFAULT_EXECUTABLE.parent}, got {build_root}"
        )
    cache_path = build_root / "CMakeCache.txt"
    try:
        cache = cache_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ArtifactRunnerError(f"cannot read M4 CMake cache: {error}") from error
    expected_lines = {
        f"CMAKE_HOME_DIRECTORY:INTERNAL={REPO_ROOT.resolve()}",
        "CMAKE_BUILD_TYPE:STRING=Release",
        "BUILD_TESTING:BOOL=ON",
        f"OpenFHE_DIR:PATH={openfhe_prefix.resolve() / 'lib' / 'OpenFHE'}",
    }
    actual_lines = set(cache.splitlines())
    missing = sorted(expected_lines - actual_lines)
    if missing:
        raise ArtifactRunnerError(f"M4 CMake cache binding drifted: missing={missing}")
    _read_openfhe_version(openfhe_prefix.resolve())


def _verify_openfhe_linkage(
    executable: Path,
    openfhe_prefix: Path,
) -> dict[str, object]:
    command = ["ldd", str(executable)]
    started_at = _timestamp()
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    finished_at = _timestamp()
    if completed.returncode != 0:
        raise ArtifactRunnerError(f"cannot inspect M4 OpenFHE linkage: {completed.stderr.strip()}")
    linked_paths: list[Path] = []
    for line in completed.stdout.splitlines():
        if "libOPENFHE" not in line or "=>" not in line:
            continue
        target = line.split("=>", maxsplit=1)[1].strip().split(maxsplit=1)[0]
        if target == "not":
            raise ArtifactRunnerError(f"M4 OpenFHE library is unresolved: {line.strip()}")
        linked_paths.append(Path(target).resolve())
    if len(linked_paths) < 2:
        raise ArtifactRunnerError("M4 executable does not expose the expected OpenFHE linkage")
    library_root = (openfhe_prefix.resolve() / "lib").resolve()
    for linked_path in linked_paths:
        try:
            linked_path.relative_to(library_root)
        except ValueError as error:
            raise ArtifactRunnerError(
                f"M4 executable links OpenFHE outside {library_root}: {linked_path}"
            ) from error
    return {
        "command": shlex.join(command),
        "cwd": str(REPO_ROOT),
        "exit_code": completed.returncode,
        "phase": "artifact_generation",
        "started_at": started_at,
        "finished_at": finished_at,
    }


def _run_m4_preflight(
    build_root: Path,
    executable: Path,
    openfhe_prefix: Path,
) -> tuple[dict[str, object], ...]:
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
        "--no-tests=error",
        "-R",
        M4_CTEST_PATTERN,
    ]
    build_record = _run_checked_command(build_command, "clean-commit M4 build")
    rebuilt_executable = _resolve_m4_executable(executable)
    test_record = _run_checked_command(test_command, "clean-commit M4 narrow gates")
    linkage_record = _verify_openfhe_linkage(rebuilt_executable, openfhe_prefix)
    return (build_record, test_record, linkage_record)


def _preflight_git() -> GitState:
    commands: list[dict[str, object]] = []
    status, record = _run_git(["status", "--porcelain=v1", "--untracked-files=normal"])
    commands.append(record)
    if status:
        raise ArtifactRunnerError("repository is not clean; refusing to run M4 evidence")

    branch, record = _run_git(["branch", "--show-current"])
    commands.append(record)
    if branch != BRANCH:
        raise ArtifactRunnerError(f"branch must be {BRANCH}, got {branch!r}")

    head, record = _run_git(["rev-parse", "HEAD"])
    commands.append(record)
    if SHA_PATTERN.fullmatch(head) is None:
        raise ArtifactRunnerError(f"local HEAD is not a full Git SHA: {head!r}")

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
    if head != remote_head:
        raise ArtifactRunnerError(
            f"local HEAD {head} differs from {REMOTE_NAME}/{BRANCH} {remote_head}"
        )
    return GitState(head=head, remote_head=remote_head, commands=tuple(commands))


def _version_line(command: list[str], label: str) -> str:
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise ArtifactRunnerError(f"cannot identify {label}: {detail}")
    lines = completed.stdout.strip().splitlines()
    if not lines:
        raise ArtifactRunnerError(f"cannot identify {label}: empty version output")
    return lines[0]


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


def _extract_record(stdout: str, test_name: str, label: str) -> dict[str, Any]:
    records: list[dict[str, Any]] = []

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
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(candidate, dict) and candidate.get("test") == test_name:
            records.append(candidate)
    if len(records) != 1:
        raise ArtifactRunnerError(
            f"{label} must emit exactly one {test_name} record, got {len(records)}"
        )
    return records[0]


def _finite_number(value: Any, label: str, minimum: float, maximum: float) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or not minimum <= value <= maximum
    ):
        raise ArtifactRunnerError(f"{label} violates [{minimum}, {maximum}]")
    return float(value)


def _precheck_target_record(record: dict[str, Any], label: str) -> None:
    if set(record) != TARGET_RECORD_KEYS:
        raise ArtifactRunnerError(
            f"{label} record keys differ: "
            f"missing={sorted(TARGET_RECORD_KEYS - set(record))} "
            f"extra={sorted(set(record) - TARGET_RECORD_KEYS)}"
        )
    expected = {
        "test": "openfhe_encoder_layer",
        "profile": "paper_compat",
        "security_claim": "none",
        "parameter_sha256": PROFILE_SHA256,
        "execution_mode": "server-only",
        "encoder_layers": 1,
        "layer_id": 1,
        "input_level": 29,
        "output_level": 29,
        "remaining_levels": 17,
        "multiplicative_depth": EXPECTED_MULTIPLICATIVE_DEPTH,
        "max_observed_level": EXPECTED_MAX_OBSERVED_LEVEL,
        "max_polynomial_depth": EXPECTED_MAX_POLYNOMIAL_DEPTH,
        "actual_trace_shape": [5, 768],
        "actual_trace_value_count": 3840,
        "feature_block_size": 1024,
        "checkpoint_decryption_owner": "client",
        "server_private_key_present": False,
        "server_decryptions": 0,
        "server_plaintext_activations": False,
    }
    for key, expected_value in expected.items():
        if record.get(key) != expected_value:
            raise ArtifactRunnerError(
                f"{label}.{key} mismatch: expected={expected_value!r} "
                f"actual={record.get(key)!r}"
            )
    _finite_number(record["relative_l2"], f"{label}.relative_l2", 0.0, 1e-2)
    _finite_number(record["cosine"], f"{label}.cosine", 0.999, 1.000000000001)
    _finite_number(record["inactive_max_abs"], f"{label}.inactive_max_abs", 0.0, 1e-6)

    checkpoints = record["checkpoints"]
    if not isinstance(checkpoints, list) or len(checkpoints) != len(CHECKPOINT_NAMES):
        raise ArtifactRunnerError(f"{label}.checkpoints must contain four records")
    for index, (checkpoint, expected_checkpoint) in enumerate(
        zip(checkpoints, EXPECTED_CHECKPOINTS)
    ):
        checkpoint_label = f"{label}.checkpoints[{index}]"
        expected_keys = {
            "name",
            "level",
            "noise_scale_degree",
            "remaining_levels",
            "scale_bits",
            "ciphertext_count",
            "decryption_owner",
        }
        if not isinstance(checkpoint, dict) or set(checkpoint) != expected_keys:
            raise ArtifactRunnerError(f"{checkpoint_label} has an invalid shape")
        if (
            checkpoint["name"] != expected_checkpoint["name"]
            or checkpoint["decryption_owner"] != "client"
        ):
            raise ArtifactRunnerError(f"{checkpoint_label} identity drifted")
        level = checkpoint["level"]
        noise_scale_degree = checkpoint["noise_scale_degree"]
        remaining_levels = checkpoint["remaining_levels"]
        ciphertext_count = checkpoint["ciphertext_count"]
        if not isinstance(level, int) or isinstance(level, bool) or not 0 <= level <= 47:
            raise ArtifactRunnerError(f"{checkpoint_label}.level is invalid")
        if (
            not isinstance(noise_scale_degree, int)
            or isinstance(noise_scale_degree, bool)
            or noise_scale_degree <= 0
        ):
            raise ArtifactRunnerError(
                f"{checkpoint_label}.noise_scale_degree must be positive"
            )
        if (
            not isinstance(remaining_levels, int)
            or isinstance(remaining_levels, bool)
            or remaining_levels <= 0
        ):
            raise ArtifactRunnerError(
                f"{checkpoint_label}.remaining_levels must be positive"
            )
        if (
            not isinstance(ciphertext_count, int)
            or isinstance(ciphertext_count, bool)
            or ciphertext_count != 5
        ):
            raise ArtifactRunnerError(
                f"{checkpoint_label}.ciphertext_count must be the five-token packing"
            )
        _finite_number(checkpoint["scale_bits"], f"{checkpoint_label}.scale_bits", 1.0, 100.0)
        if checkpoint != expected_checkpoint:
            raise ArtifactRunnerError(
                f"{checkpoint_label} metadata drifted: "
                f"expected={expected_checkpoint} actual={checkpoint}"
            )

    checkpoint_hash = _checkpoint_metadata_sha256(checkpoints)
    if checkpoint_hash != EXPECTED_CHECKPOINT_METADATA_SHA256:
        raise ArtifactRunnerError(
            f"{label}.checkpoints canonical SHA-256 drifted: {checkpoint_hash}"
        )

    counts = record["operation_counts"]
    if not isinstance(counts, dict) or set(counts) != OPERATION_COUNT_KEYS:
        raise ArtifactRunnerError(f"{label}.operation_counts has an invalid shape")
    for key, value in counts.items():
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ArtifactRunnerError(f"{label}.operation_counts.{key} must be positive")
    if counts != EXPECTED_OPERATION_COUNTS:
        raise ArtifactRunnerError(
            f"{label} encoder operation schedule drifted: "
            f"expected={EXPECTED_OPERATION_COUNTS} actual={counts}"
        )


def _precheck_diagnostic_record(
    diagnostic: dict[str, Any],
    target: dict[str, Any],
    label: str,
) -> None:
    if set(diagnostic) != DIAGNOSTIC_RECORD_KEYS:
        raise ArtifactRunnerError(
            f"{label} diagnostic keys differ: "
            f"missing={sorted(DIAGNOSTIC_RECORD_KEYS - set(diagnostic))} "
            f"extra={sorted(set(diagnostic) - DIAGNOSTIC_RECORD_KEYS)}"
        )
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
        if diagnostic.get(key) != expected_value:
            raise ArtifactRunnerError(
                f"{label} diagnostic.{key} mismatch: expected={expected_value!r} "
                f"actual={diagnostic.get(key)!r}"
            )

    for field in PHASE_LATENCY_FIELDS:
        _finite_number(diagnostic[field], f"{label} diagnostic.{field}", 0.0, math.inf)
    for field in (
        "relative_l2",
        "max_absolute",
        "exact_trace_relative_l2",
        "exact_trace_max_absolute",
        "inactive_max_absolute",
    ):
        _finite_number(diagnostic[field], f"{label} diagnostic.{field}", 0.0, math.inf)
    for field in ("cosine", "exact_trace_cosine"):
        _finite_number(
            diagnostic[field],
            f"{label} diagnostic.{field}",
            -1.000000000001,
            1.000000000001,
        )
    peak_rss_bytes = diagnostic["peak_rss_bytes"]
    if (
        not isinstance(peak_rss_bytes, int)
        or isinstance(peak_rss_bytes, bool)
        or peak_rss_bytes <= 0
    ):
        raise ArtifactRunnerError(f"{label} diagnostic.peak_rss_bytes must be positive")
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
        value = diagnostic[field]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ArtifactRunnerError(f"{label} diagnostic.{field} must be positive")

    numeric_bindings = {
        "relative_l2": "relative_l2",
        "cosine": "cosine",
        "inactive_max_absolute": "inactive_max_abs",
    }
    for diagnostic_key, target_key in numeric_bindings.items():
        if not math.isclose(
            float(diagnostic[diagnostic_key]),
            float(target[target_key]),
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            raise ArtifactRunnerError(
                f"{label} diagnostic.{diagnostic_key} differs from target.{target_key}"
            )
    for key in OPERATION_COUNT_KEYS:
        if diagnostic[key] != target["operation_counts"][key]:
            raise ArtifactRunnerError(
                f"{label} diagnostic.{key} differs from target.operation_counts.{key}"
            )


def _run_once(config: RunnerConfig, phase: str) -> RunSample:
    resource_file_descriptor, resource_file_name = tempfile.mkstemp(
        prefix="moai-m4-time-", suffix=".txt", dir="/tmp"
    )
    os.close(resource_file_descriptor)
    resource_file = Path(resource_file_name)
    command = [
        str(TIME_EXECUTABLE),
        "--format=%M",
        f"--output={resource_file}",
        str(config.executable),
        "--data-root",
        str(config.data_root),
        "--layer",
        "1",
        "--input-level",
        "29",
    ]
    started_at = _timestamp()
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
        finished_at = _timestamp()
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            if len(detail) > 2000:
                detail = detail[:2000] + "..."
            raise ArtifactRunnerError(
                f"{phase} exited with {completed.returncode}: {detail}"
            )
        try:
            resource_lines = resource_file.read_text(encoding="utf-8").splitlines()
            if len(resource_lines) != 1:
                raise ValueError("expected one GNU time output line")
            peak_rss_kib = int(resource_lines[0])
        except (OSError, UnicodeError, ValueError) as error:
            raise ArtifactRunnerError(f"cannot parse {phase} peak RSS: {error}") from error
        if elapsed_seconds <= 0.0 or peak_rss_kib <= 0:
            raise ArtifactRunnerError(f"{phase} produced invalid resource metrics")
        record = _extract_record(completed.stdout, "openfhe_encoder_layer", phase)
        diagnostic = _extract_record(
            completed.stdout, "openfhe_encoder_layer_smoke", phase
        )
        _precheck_target_record(record, phase)
        _precheck_diagnostic_record(diagnostic, record, phase)
        if diagnostic["peak_rss_bytes"] > peak_rss_kib * 1024:
            raise ArtifactRunnerError(
                f"{phase} diagnostic.peak_rss_bytes exceeds GNU time peak RSS: "
                f"diagnostic={diagnostic['peak_rss_bytes']} "
                f"gnu_time={peak_rss_kib * 1024}"
            )
        command_record: dict[str, object] = {
            "command": shlex.join(command),
            "cwd": str(REPO_ROOT),
            "exit_code": completed.returncode,
            "phase": "artifact_generation",
            "started_at": started_at,
            "finished_at": finished_at,
        }
        return RunSample(
            stdout=completed.stdout,
            record=record,
            diagnostic=diagnostic,
            elapsed_seconds=elapsed_seconds,
            peak_rss_kib=peak_rss_kib,
            command=command_record,
        )
    finally:
        resource_file.unlink(missing_ok=True)


def _require_executable_hash(executable: Path, expected_sha256: str, phase: str) -> None:
    actual_sha256 = _sha256(executable)
    if actual_sha256 != expected_sha256:
        raise ArtifactRunnerError(
            f"M4 executable changed {phase}: "
            f"expected={expected_sha256} actual={actual_sha256}"
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
        actual_bytes = resolved.stat().st_size
        actual_sha256 = _sha256(resolved)
        if actual_bytes != expected_bytes or actual_sha256 != expected_sha256:
            raise ArtifactRunnerError(
                f"frozen input changed {phase}: {path} "
                f"expected_sha256={expected_sha256} actual_sha256={actual_sha256} "
                f"expected_bytes={expected_bytes} actual_bytes={actual_bytes}"
            )


def _repository_record(path: str) -> dict[str, object]:
    resolved = _resolve_repository_file(REPO_ROOT / path, f"input {path}")
    return {
        "path": path,
        "sha256": _sha256(resolved),
        "bytes": resolved.stat().st_size,
        "media_type": "application/json",
        "role": "configuration",
    }


def _trace_input_records(data_root: Path, layer_id: int) -> list[dict[str, object]]:
    contract_path = REPO_ROOT / "config" / "moai_encoder_trace.json"
    contract = _load_json(contract_path)
    try:
        required_files = contract["required_files"]
        layers = contract["layers"]
        layer = next(item for item in layers if item["layer_id"] == layer_id)
        expected_hashes = layer["sha256"]
    except (KeyError, TypeError, StopIteration) as error:
        raise ArtifactRunnerError(
            f"encoder trace contract lacks layer {layer_id} file hashes"
        ) from error
    if (
        not isinstance(required_files, dict)
        or not isinstance(expected_hashes, dict)
        or set(required_files) != set(expected_hashes)
        or len(required_files) != 37
    ):
        raise ArtifactRunnerError("layer-1 trace contract must bind exactly 37 files")

    records: list[dict[str, object]] = []
    for logical_name in sorted(required_files):
        specification = required_files[logical_name]
        if not isinstance(specification, dict) or not isinstance(
            specification.get("path"), str
        ):
            raise ArtifactRunnerError(f"trace file mapping is invalid: {logical_name}")
        relative_data_path = Path("data") / f"layer_{layer_id}" / specification["path"]
        resolved = _resolve_repository_file(
            REPO_ROOT / relative_data_path,
            f"layer-{layer_id} trace input {logical_name}",
        )
        expected_hash = expected_hashes[logical_name]
        actual_hash = _sha256(resolved)
        if actual_hash != expected_hash:
            raise ArtifactRunnerError(
                f"layer-{layer_id} trace hash mismatch for {logical_name}: "
                f"expected={expected_hash} actual={actual_hash}"
            )
        records.append(
            {
                "path": relative_data_path.as_posix(),
                "sha256": actual_hash,
                "bytes": resolved.stat().st_size,
                "media_type": "text/csv",
                "role": "weights" if "/parms/" in specification["path"] else "trace",
            }
        )
    expected_data_root = DEFAULT_DATA_ROOT.resolve()
    if data_root.resolve() != expected_data_root:
        raise ArtifactRunnerError(
            f"M4 data root must be the frozen path {expected_data_root}, got {data_root}"
        )
    return records


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


def _write_stdout(path: Path, samples: list[RunSample]) -> None:
    chunks = [
        sample.stdout if sample.stdout.endswith("\n") else sample.stdout + "\n"
        for sample in samples
    ]
    path.write_text("".join(chunks), encoding="utf-8")


def _write_metrics(path: Path, samples: list[RunSample]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for index, sample in enumerate(samples, start=1):
            counts = sample.record["operation_counts"]
            batch_total_ms = sample.elapsed_seconds * 1000.0
            writer.writerow(
                {
                    "run": index,
                    "exit_code": 0,
                    "elapsed_seconds": format(sample.elapsed_seconds, ".17g"),
                    "peak_rss_kib": sample.peak_rss_kib,
                    **{
                        field: format(sample.diagnostic[field], ".17g")
                        for field in PHASE_LATENCY_FIELDS
                    },
                    "batch_total_ms": format(batch_total_ms, ".17g"),
                    "batch_amortized_ms_per_token": format(
                        batch_total_ms / TOKENS_PER_BATCH, ".17g"
                    ),
                    "server_amortized_ms_per_token": format(
                        sample.diagnostic["server_online_ms"] / TOKENS_PER_BATCH,
                        ".17g",
                    ),
                    "final_rel_l2": format(sample.record["relative_l2"], ".17g"),
                    "final_cosine": format(sample.record["cosine"], ".17g"),
                    "inactive_max_abs": format(sample.record["inactive_max_abs"], ".17g"),
                    "multiplicative_depth": sample.record["multiplicative_depth"],
                    "max_observed_level": sample.record["max_observed_level"],
                    "max_polynomial_depth": sample.record["max_polynomial_depth"],
                    "checkpoint_metadata_sha256": _checkpoint_metadata_sha256(
                        sample.record["checkpoints"]
                    ),
                    "bootstraps": counts["bootstraps"],
                    "bootstrap_iterations": counts["bootstrap_iterations"],
                }
            )


def _summarize_metrics(samples: list[RunSample]) -> dict[str, object]:
    if len(samples) != 5:
        raise ArtifactRunnerError(f"M4 requires exactly five measured samples, got {len(samples)}")
    records = [sample.record for sample in samples]
    first_counts = records[0]["operation_counts"]
    if any(record["operation_counts"] != first_counts for record in records[1:]):
        raise ArtifactRunnerError("operation counts differ across measured repeats")

    checkpoints: list[dict[str, object]] = []
    for index, name in enumerate(CHECKPOINT_NAMES):
        values = [record["checkpoints"][index] for record in records]
        ciphertext_counts = {value["ciphertext_count"] for value in values}
        if len(ciphertext_counts) != 1:
            raise ArtifactRunnerError(f"checkpoint {name} ciphertext count differs across repeats")
        checkpoints.append(
            {
                "name": name,
                "level_min": min(value["level"] for value in values),
                "level_max": max(value["level"] for value in values),
                "noise_scale_degree_min": min(
                    value["noise_scale_degree"] for value in values
                ),
                "noise_scale_degree_max": max(
                    value["noise_scale_degree"] for value in values
                ),
                "remaining_levels_min": min(
                    value["remaining_levels"] for value in values
                ),
                "remaining_levels_max": max(
                    value["remaining_levels"] for value in values
                ),
                "scale_bits_min": min(value["scale_bits"] for value in values),
                "scale_bits_max": max(value["scale_bits"] for value in values),
                "ciphertext_count": next(iter(ciphertext_counts)),
                "decryption_owner": "client",
            }
        )
    elapsed = [sample.elapsed_seconds for sample in samples]
    batch_total_ms = [value * 1000.0 for value in elapsed]
    phase_latency_ms = {
        field: {
            "minimum": min(sample.diagnostic[field] for sample in samples),
            "median": statistics.median(
                sample.diagnostic[field] for sample in samples
            ),
            "maximum": max(sample.diagnostic[field] for sample in samples),
        }
        for field in PHASE_LATENCY_FIELDS
    }

    def summarize(values: list[float]) -> dict[str, float]:
        return {
            "minimum": min(values),
            "median": statistics.median(values),
            "maximum": max(values),
        }

    return {
        "repeat_count": 5,
        "successful_repeats": 5,
        "warmup_count": 1,
        "elapsed_seconds": {
            "minimum": min(elapsed),
            "median": statistics.median(elapsed),
            "maximum": max(elapsed),
        },
        "tokens_per_batch": TOKENS_PER_BATCH,
        "phase_latency_ms": phase_latency_ms,
        "batch_total_ms": summarize(batch_total_ms),
        "batch_amortized_ms_per_token": summarize(
            [value / TOKENS_PER_BATCH for value in batch_total_ms]
        ),
        "server_amortized_ms_per_token": summarize(
            [
                sample.diagnostic["server_online_ms"] / TOKENS_PER_BATCH
                for sample in samples
            ]
        ),
        "peak_rss_kib_max": max(sample.peak_rss_kib for sample in samples),
        "actual_trace_shape": [5, 768],
        "actual_trace_value_count": 3840,
        "feature_block_size": 1024,
        "input_level": 29,
        "output_level": 29,
        "remaining_levels": 17,
        "multiplicative_depth": EXPECTED_MULTIPLICATIVE_DEPTH,
        "max_observed_level": EXPECTED_MAX_OBSERVED_LEVEL,
        "max_polynomial_depth": EXPECTED_MAX_POLYNOMIAL_DEPTH,
        "quality": {
            "relative_l2_max": max(record["relative_l2"] for record in records),
            "cosine_min": min(record["cosine"] for record in records),
            "inactive_max_abs_max": max(record["inactive_max_abs"] for record in records),
        },
        "checkpoints": checkpoints,
        "checkpoint_metadata_sha256": EXPECTED_CHECKPOINT_METADATA_SHA256,
        "operation_counts": first_counts,
    }


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
        command,
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise ArtifactRunnerError(f"artifact validator rejected the M4 run: {detail}")


def generate_m4_artifact(config: RunnerConfig) -> Path:
    if RUN_ID_PATTERN.fullmatch(config.run_id) is None:
        raise ArtifactRunnerError(f"invalid run id: {config.run_id!r}")
    started_at = _timestamp()
    executable = _resolve_m4_executable(config.executable, require_exists=False)
    data_root = config.data_root.resolve()
    if not data_root.is_dir():
        raise ArtifactRunnerError(f"data root is not a directory: {data_root}")
    if data_root != DEFAULT_DATA_ROOT.resolve():
        raise ArtifactRunnerError(
            f"M4 data root must be the frozen path {DEFAULT_DATA_ROOT.resolve()}"
        )
    if not TIME_EXECUTABLE.is_file() or not os.access(TIME_EXECUTABLE, os.X_OK):
        raise ArtifactRunnerError(f"GNU time is unavailable: {TIME_EXECUTABLE}")

    output_root = _resolve_output_root(config.output_root)
    git_state = _preflight_git()
    preflight_commands = _run_m4_preflight(
        executable.parent,
        executable,
        config.openfhe_prefix.resolve(),
    )
    executable = _resolve_m4_executable(DEFAULT_EXECUTABLE)
    executable_sha256 = _sha256(executable)
    environment = _environment(config.openfhe_prefix)
    profile = _profile_record()
    inputs = [
        *(_repository_record(path) for path in REQUIRED_INPUTS),
        *_trace_input_records(data_root, 1),
    ]
    input_by_path = {record["path"]: record for record in inputs}
    output_root.mkdir(parents=True, exist_ok=True)
    run_root = output_root / config.run_id
    try:
        run_root.mkdir(exist_ok=False)
    except FileExistsError as error:
        raise ArtifactRunnerError(f"run directory already exists: {run_root}") from error

    created_run_root = True
    try:
        _require_executable_hash(executable, executable_sha256, "before warm-up")
        _require_input_hashes(inputs, "before warm-up")
        warmup = _run_once(
            RunnerConfig(
                executable=executable,
                data_root=data_root,
                output_root=output_root,
                run_id=config.run_id,
                openfhe_prefix=config.openfhe_prefix,
            ),
            "warm-up",
        )
        _require_executable_hash(executable, executable_sha256, "after warm-up")
        _require_input_hashes(inputs, "after warm-up")
        samples: list[RunSample] = []
        for index in range(1, 6):
            _require_executable_hash(
                executable,
                executable_sha256,
                f"before measured run {index}",
            )
            _require_input_hashes(inputs, f"before measured run {index}")
            samples.append(_run_once(
                RunnerConfig(
                    executable=executable,
                    data_root=data_root,
                    output_root=output_root,
                    run_id=config.run_id,
                    openfhe_prefix=config.openfhe_prefix,
                ),
                f"measured run {index}",
            ))
            _require_executable_hash(
                executable,
                executable_sha256,
                f"after measured run {index}",
            )
            _require_input_hashes(inputs, f"after measured run {index}")
        stdout_path = run_root / "stdout.log"
        metrics_path = run_root / "metrics.csv"
        checksum_path = run_root / "SHA256SUMS"
        manifest_path = run_root / "manifest.json"
        _write_stdout(stdout_path, samples)
        _write_metrics(metrics_path, samples)
        checksum_path.write_text(
            f"{_sha256(stdout_path)}  stdout.log\n{_sha256(metrics_path)}  metrics.csv\n",
            encoding="utf-8",
        )

        executable_path = executable.relative_to(REPO_ROOT.resolve()).as_posix()
        validator_command = _validator_command(manifest_path)
        commands = [
            *git_state.commands,
            *preflight_commands,
            warmup.command,
            *(sample.command for sample in samples),
            {
                "command": shlex.join(validator_command),
                "cwd": str(REPO_ROOT),
                "exit_code": 0,
                "phase": "artifact_generation",
            },
        ]
        manifest: dict[str, object] = {
            "schema_version": 2,
            "run_id": config.run_id,
            "milestone": "M4",
            "started_at": started_at,
            "finished_at": _timestamp(),
            "git": {
                "repository_root": str(REPO_ROOT),
                "branch": BRANCH,
                "local_commit": git_state.head,
                "clean": True,
                "remote_name": REMOTE_NAME,
                "remote_ref": REMOTE_REF,
                "remote_commit": git_state.remote_head,
            },
            "commands": commands,
            "environment": environment,
            "profile": profile,
            "inputs": inputs,
            "workload": {
                "scope": "M4 server-only five-token BERT-base encoder layer-1 trace replay",
                "backend": "OpenFHE CKKS CPU",
                "executable_path": executable_path,
                "executable_sha256": executable_sha256,
                "executable_bytes": executable.stat().st_size,
                "repeat_count": 5,
            },
            "contracts": {
                "approximation_config_path": "config/openfhe_approximations.json",
                "approximation_config_sha256": input_by_path[
                    "config/openfhe_approximations.json"
                ]["sha256"],
                "encoder_trace_contract_path": "config/moai_encoder_trace.json",
                "encoder_trace_contract_sha256": input_by_path[
                    "config/moai_encoder_trace.json"
                ]["sha256"],
                "profile_config_path": PROFILE_PATH,
                "profile_config_sha256": input_by_path[PROFILE_PATH]["sha256"],
                "execution": {
                    "mode": "server-only",
                    "quality_reference": (
                        "config/moai_encoder_trace.json "
                        "single-layer frozen polynomial oracle"
                    ),
                    "encoder_layers": 1,
                    "layer_id": 1,
                    "trace_shape": [5, 768],
                    "feature_block_size": 1024,
                    "checkpoint_decryption_owner": "client",
                    "server_private_key_present": False,
                    "server_decryptions": 0,
                    "server_plaintext_activations": False,
                    "multiplicative_depth": EXPECTED_MULTIPLICATIVE_DEPTH,
                    "max_observed_level": EXPECTED_MAX_OBSERVED_LEVEL,
                    "max_polynomial_depth": EXPECTED_MAX_POLYNOMIAL_DEPTH,
                    "required_checkpoints": list(CHECKPOINT_NAMES),
                },
                "thresholds": {
                    "relative_l2_max": 1e-2,
                    "cosine_min": 0.999,
                    "inactive_max_abs": 1e-6,
                },
            },
            "metrics": _summarize_metrics(samples),
            "gate": {
                "passed": True,
                "decision": "PASS_M4_SERVER_ONLY_OPENFHE_ENCODER_LAYER",
                "checks": {
                    "build": "PASS",
                    "trust_boundary": "PASS",
                    "correctness": "PASS",
                    "repeatability": "PASS",
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
                "Five-token M4 layer-1 trace replay only; not task-level inference.",
                "paper_compat OpenFHE CKKS CPU parameters with security_claim=none.",
                "GPU, Discrete CKKS/FBT, QDQ, tokenizer, and classifier are excluded.",
            ],
            "verdict": "GO",
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        _validate_artifact(manifest_path)
        created_run_root = False
        return run_root
    finally:
        if created_run_root:
            try:
                shutil.rmtree(run_root)
            except OSError as cleanup_error:
                raise ArtifactRunnerError(
                    f"failed to remove unvalidated run directory {run_root}: "
                    f"{cleanup_error}"
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
            f"-m4-{git_sha_hint}"
        )
        run_root = generate_m4_artifact(
            RunnerConfig(
                executable=arguments.executable,
                data_root=arguments.data_root,
                output_root=arguments.output_root,
                run_id=run_id,
            )
        )
    except (ArtifactRunnerError, OSError, ValueError) as error:
        print(f"run_openfhe_encoder_artifact failed: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "artifact_root": str(run_root),
                "manifest": str(run_root / "manifest.json"),
                "milestone": "M4",
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
