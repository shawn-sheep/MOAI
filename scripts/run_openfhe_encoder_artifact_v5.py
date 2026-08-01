#!/usr/bin/env python3
"""Run and seal the frozen M4 OpenFHE encoder-layer schema-v5 evidence bundle.

This runner is deliberately narrow: it supports only the server-only M4 layer-1
trace replay at input level 29.  It performs one recorded warm-up followed by
five measured executions, then seals raw stdout, successful stderr, GNU time,
and independently derived metrics into the schema-v5 artifact contract.
Historical schema-v2 files are deliberately not imported, replayed, or modified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import platform
import re
import shlex
import shutil
import statistics
import struct
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXECUTABLE = REPO_ROOT / "build-openfhe" / "openfhe_encoder_layer_smoke"
DEFAULT_DATA_ROOT = REPO_ROOT / "data"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "openfhe"
DEFAULT_OPENFHE_PREFIX = Path("/home/shawnsheep/opt/openfhe_v1_5_1")
SYSTEM_CMAKE = Path("/usr/bin/cmake")
SYSTEM_CTEST = Path("/usr/bin/ctest")
SYSTEM_CXX = Path("/usr/bin/c++")
SYSTEM_GIT = Path("/usr/bin/git")
SYSTEM_LDD = Path("/usr/bin/ldd")
SYSTEM_MAKE = Path("/usr/bin/gmake")
SYSTEM_ENV = Path("/usr/bin/env")
CONFIGURE_ENV_UNSET = (
    "CC",
    "CXX",
    "CFLAGS",
    "CXXFLAGS",
    "CPPFLAGS",
    "LDFLAGS",
    "LD_PRELOAD",
    "LD_LIBRARY_PATH",
    "CMAKE_PREFIX_PATH",
    "CMAKE_TOOLCHAIN_FILE",
    "CPATH",
    "CPLUS_INCLUDE_PATH",
    "C_INCLUDE_PATH",
    "OBJC_INCLUDE_PATH",
    "LIBRARY_PATH",
    "LD_AUDIT",
    "LD_DEBUG",
    "GCC_EXEC_PREFIX",
    "COMPILER_PATH",
    "MAKEFLAGS",
    "MFLAGS",
    "BASH_ENV",
    "ENV",
    "PYTHONHOME",
    "PYTHONPATH",
    "CMAKE_GENERATOR",
    "CMAKE_GENERATOR_PLATFORM",
    "CMAKE_GENERATOR_TOOLSET",
    "CMAKE_GENERATOR_INSTANCE",
    "CMAKE_BUILD_PARALLEL_LEVEL",
    "CTEST_PARALLEL_LEVEL",
    "CTEST_TEST_LOAD",
    "CTEST_RESOURCE_SPEC_FILE",
)
FORCED_SUBPROCESS_ENVIRONMENT = {"LANG": "C", "LC_ALL": "C"}
OPENFHE_CMAKE_PACKAGE_FILES = (
    "OpenFHEConfig.cmake",
    "OpenFHEConfigVersion.cmake",
    "OpenFHETargets.cmake",
    "OpenFHETargets-release.cmake",
)
OPENFHE_LINKED_LIBRARY_SONAMES = (
    "libOPENFHEbinfhe.so.1",
    "libOPENFHEcore.so.1",
    "libOPENFHEpke.so.1",
)
OPENFHE_LINKED_LIBRARY_BASENAMES = {
    "libOPENFHEbinfhe.so.1": "libOPENFHEbinfhe.so.1.5.1",
    "libOPENFHEcore.so.1": "libOPENFHEcore.so.1.5.1",
    "libOPENFHEpke.so.1": "libOPENFHEpke.so.1.5.1",
}
OPENFHE_INCLUDE_TREE_CANONICALIZATION = (
    "relative-posix-path-nul-size-nul-sha256-newline-v1"
)
SCHEMA_PATH = REPO_ROOT / "docs" / "openfhe-m4-artifact-schema-v5.json"
VALIDATOR_PATH = REPO_ROOT / "scripts" / "validate_openfhe_m4_artifact_v5.py"
RUNNER_PATH = REPO_ROOT / "scripts" / "run_openfhe_encoder_artifact_v5.py"
SCHEMA_ID = "https://local.moai/openfhe-m4-artifact-schema-v5.json"
SCHEMA_TITLE = "MOAI OpenFHE M4 evidence manifest v5"
SCHEMA_VERSION = 5
SCHEMA_RELATIVE_PATH = "docs/openfhe-m4-artifact-schema-v5.json"
VALIDATOR_RELATIVE_PATH = "scripts/validate_openfhe_m4_artifact_v5.py"
RUNNER_RELATIVE_PATH = "scripts/run_openfhe_encoder_artifact_v5.py"
TIME_EXECUTABLE = Path("/usr/bin/time")
REMOTE_NAME = "origin"
REMOTE_URL = "https://github.com/shawn-sheep/MOAI.git"
REMOTE_REF = "refs/heads/refactor/openfhe-cpu"
BRANCH = "refactor/openfhe-cpu"
PROFILE_PATH = "config/paper_compat_feature_packed.json"
PROFILE_LOCATOR = "/effective_profile"
PROFILE_SHA256 = "94f30e628e21f02146ce7ed9820194eabba3820f6e1e17176a31f8c5acf8b0be"
WARNING = "Research reproduction parameters only. Do not claim 128-bit security."
CLAIM_BOUNDARY = (
    "M4 covers only single-layer layer-1 correctness on the fixed five-token "
    "BERT-base encoder trace.",
    "The server-only claim is an API/target trust boundary: server code receives "
    "only ciphertexts, public model weights, and evaluation keys; it is not a "
    "process-isolation claim.",
    "paper_compat OpenFHE CKKS CPU parameters are research-reproduction parameters "
    "with security_claim=none.",
    "This artifact makes no 128-bit-security or production-security claim.",
    "Timing is diagnostic only; this artifact makes no benchmark or speedup claim.",
    "GPU, Discrete CKKS/FBT, QDQ, tokenizer, classifier, and task-level inference "
    "are excluded.",
)
CHECKSUM_EVIDENCE_PATHS = (
    "stdout.log",
    "stderr.log",
    "time.log",
    "metrics.csv",
    "manifest.json",
)
MANIFEST_ARTIFACTS = {
    "stdout.log": ("stdout", "text/plain"),
    "stderr.log": ("stderr", "text/plain"),
    "time.log": ("resource_metrics", "text/plain"),
    "metrics.csv": ("metrics", "text/csv"),
}
PROFILE_WARNING = (
    'profile=paper_compat security_claim=none warning="'
    f'{WARNING}"'
)
RAW_STDOUT_MAX_BYTES = 16 * 1024 * 1024
RAW_STDOUT_MAX_LINE_BYTES = 2 * 1024 * 1024
RAW_STDERR_MAX_BYTES = 64 * 1024
RAW_TIME_MAX_BYTES = 64 * 1024
TIME_RECORD_PREFIX = "M4_TIME_V1"
TIME_PHASES = ("warmup", *(f"measured-{index}" for index in range(1, 6)))
TIME_ELAPSED_PATTERN = re.compile(r"(?:0|[1-9][0-9]*)\.[0-9]{2}")
TIME_RSS_PATTERN = re.compile(r"[1-9][0-9]*")
M4_SOURCE_PATHS = (
    "CMakeLists.txt",
    "cmake/ValidateServerBoundary.cmake",
    "include/moai/openfhe/approximation_registry.hpp",
    "include/moai/openfhe/client_runtime.hpp",
    "include/moai/openfhe/context_factory.hpp",
    "include/moai/openfhe/encoder_layer.hpp",
    "include/moai/openfhe/evaluation_key_registry.hpp",
    "include/moai/openfhe/feature_packed_attention.hpp",
    "include/moai/openfhe/feature_packed_ops.hpp",
    "include/moai/openfhe/linear_ops.hpp",
    "include/moai/openfhe/nonlinear_ops.hpp",
    "include/moai/openfhe/packing.hpp",
    "include/moai/openfhe/server_runtime.hpp",
    "include/moai/openfhe/types.hpp",
    "src/openfhe/approximation_registry.cpp",
    "src/openfhe/client_runtime.cpp",
    "src/openfhe/context_factory.cpp",
    "src/openfhe/encoder_layer.cpp",
    "src/openfhe/evaluation_key_registry.cpp",
    "src/openfhe/feature_packed_attention.cpp",
    "src/openfhe/feature_packed_ops.cpp",
    "src/openfhe/linear_ops.cpp",
    "src/openfhe/nonlinear_ops.cpp",
    "src/openfhe/packing.cpp",
    "src/openfhe/server_runtime.cpp",
    "tests/openfhe_encoder_layer_smoke.cpp",
    "tests/support/moai_encoder_fixture.cpp",
    "tests/support/moai_encoder_fixture.hpp",
    "tests/support/moai_encoder_plaintext_oracle.cpp",
    "tests/support/moai_encoder_plaintext_oracle.hpp",
)
M4_BUILD_PRODUCT_PATHS = (
    "build-openfhe/CMakeFiles/moai_openfhe_common.dir/build.make",
    "build-openfhe/CMakeFiles/moai_openfhe_common.dir/flags.make",
    "build-openfhe/CMakeFiles/moai_openfhe_common.dir/link.txt",
    "build-openfhe/CMakeFiles/moai_openfhe_common.dir/src/openfhe/approximation_registry.cpp.o",
    "build-openfhe/CMakeFiles/moai_openfhe_common.dir/src/openfhe/context_factory.cpp.o",
    "build-openfhe/CMakeFiles/moai_openfhe_common.dir/src/openfhe/evaluation_key_registry.cpp.o",
    "build-openfhe/CMakeFiles/moai_openfhe_common.dir/src/openfhe/packing.cpp.o",
    "build-openfhe/CMakeFiles/moai_openfhe_server.dir/build.make",
    "build-openfhe/CMakeFiles/moai_openfhe_server.dir/flags.make",
    "build-openfhe/CMakeFiles/moai_openfhe_server.dir/link.txt",
    "build-openfhe/CMakeFiles/moai_openfhe_server.dir/src/openfhe/encoder_layer.cpp.o",
    "build-openfhe/CMakeFiles/moai_openfhe_server.dir/src/openfhe/feature_packed_attention.cpp.o",
    "build-openfhe/CMakeFiles/moai_openfhe_server.dir/src/openfhe/feature_packed_ops.cpp.o",
    "build-openfhe/CMakeFiles/moai_openfhe_server.dir/src/openfhe/linear_ops.cpp.o",
    "build-openfhe/CMakeFiles/moai_openfhe_server.dir/src/openfhe/nonlinear_ops.cpp.o",
    "build-openfhe/CMakeFiles/moai_openfhe_server.dir/src/openfhe/server_runtime.cpp.o",
    "build-openfhe/CMakeFiles/moai_openfhe_client.dir/build.make",
    "build-openfhe/CMakeFiles/moai_openfhe_client.dir/flags.make",
    "build-openfhe/CMakeFiles/moai_openfhe_client.dir/link.txt",
    "build-openfhe/CMakeFiles/moai_openfhe_client.dir/src/openfhe/client_runtime.cpp.o",
    "build-openfhe/CMakeFiles/openfhe_encoder_layer_smoke.dir/build.make",
    "build-openfhe/CMakeFiles/openfhe_encoder_layer_smoke.dir/flags.make",
    "build-openfhe/CMakeFiles/openfhe_encoder_layer_smoke.dir/link.txt",
    "build-openfhe/CMakeFiles/openfhe_encoder_layer_smoke.dir/tests/openfhe_encoder_layer_smoke.cpp.o",
    "build-openfhe/CMakeFiles/openfhe_encoder_layer_smoke.dir/tests/support/moai_encoder_fixture.cpp.o",
    "build-openfhe/CMakeFiles/openfhe_encoder_layer_smoke.dir/tests/support/moai_encoder_plaintext_oracle.cpp.o",
    "build-openfhe/libmoai_openfhe_common.a",
    "build-openfhe/libmoai_openfhe_server.a",
    "build-openfhe/libmoai_openfhe_client.a",
)
CANONICALIZATION = "MOAI-json-sort-keys-compact-utf8-v1"
TRACE_SCALE_SOURCE_PATH = "config/openfhe_approximations.json"
TRACE_SCALE_JSON_LOCATOR = "operators.layernorm.feature_packed_trace_scale_contract"
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
LAYERNORM_REGISTERED_INTERVAL = (0.5, 1536.0)
M4_SCHEDULE_SEALED_STATUS = "sealed"
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
EXPECTED_POST_BOOTSTRAP_MASK = "single_normal_scale_mask"
EXPECTED_INTERNAL_CHECKPOINTS = {
    "softmax_denominator": {
        "level": 18,
        "noise_scale_degree": 2,
        "remaining_levels": 28,
    },
    "layernorm_normalized_variance": {
        "level": 19,
        "noise_scale_degree": 2,
        "remaining_levels": 27,
    },
}
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
    "ct_pt_multiplications": 51885,
    "ct_ct_multiplications": 95,
    "explicit_rescale_requests": 810,
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
    "layernorm_inactive_guard_max_error",
    "inactive_polynomial_sentinel_ranges",
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
    "m4_v5_artifact_schema_contract|m4_v5_artifact_validator_contract|"
    "m4_v5_encoder_artifact_runner_contract|"
    "feature_packed_smoke|"
    "feature_packed_attention_smoke|feature_bootstrap_smoke|"
    "feature_layernorm_smoke|"
    "encoder_fixture_contract|"
    "encoder_plaintext_oracle_smoke|encoder_trace_contract|"
    "encoder_trace_validator_contract))$"
)
M4_CTEST_EXPECTED_TESTS = (
    "openfhe_server_trust_boundary",
    "openfhe_profile_contract",
    "openfhe_profile_validator_contract",
    "openfhe_artifact_schema_contract",
    "openfhe_artifact_validator_contract",
    "openfhe_encoder_artifact_runner_contract",
    "openfhe_m4_v5_artifact_schema_contract",
    "openfhe_m4_v5_artifact_validator_contract",
    "openfhe_m4_v5_encoder_artifact_runner_contract",
    "moai_trace_contract",
    "openfhe_encoder_trace_contract",
    "openfhe_encoder_trace_validator_contract",
    "openfhe_feature_packed_smoke",
    "openfhe_feature_packed_attention_smoke",
    "openfhe_encoder_fixture_contract",
    "openfhe_encoder_plaintext_oracle_smoke",
    "openfhe_evaluation_key_bundle_smoke",
    "openfhe_feature_layernorm_smoke",
    "openfhe_feature_bootstrap_smoke",
)
M4_V5_REQUIRED_CTEST_CONTRACTS = (
    "openfhe_m4_v5_artifact_schema_contract",
    "openfhe_m4_v5_artifact_validator_contract",
    "openfhe_m4_v5_encoder_artifact_runner_contract",
    "openfhe_feature_layernorm_smoke",
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
    "layernorm_inactive_guard_max_error",
    "softmax_sentinel_minimum",
    "softmax_sentinel_maximum",
    "self_layernorm_sentinel_minimum",
    "self_layernorm_sentinel_maximum",
    "output_layernorm_sentinel_minimum",
    "output_layernorm_sentinel_maximum",
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
    remote_url: str
    remote_head: str
    commands: tuple[dict[str, object], ...]


@dataclass(frozen=True)
class M4Preflight:
    commands: tuple[dict[str, object], ...]
    build_configuration: dict[str, object]


@dataclass(frozen=True)
class RunnerConfig:
    executable: Path
    data_root: Path
    output_root: Path
    run_id: str
    openfhe_prefix: Path = DEFAULT_OPENFHE_PREFIX


@dataclass(frozen=True)
class RunSample:
    stdout: bytes
    stderr: bytes
    record: dict[str, Any]
    diagnostic: dict[str, Any]
    elapsed_seconds: float
    peak_rss_kib: int
    command: dict[str, object]


def _timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _artifact_environment() -> dict[str, str]:
    environment = os.environ.copy()
    for name in CONFIGURE_ENV_UNSET:
        environment.pop(name, None)
    environment.update(FORCED_SUBPROCESS_ENVIRONMENT)
    return environment


def _git_environment() -> dict[str, str]:
    environment = _artifact_environment()
    for name in tuple(environment):
        if name.startswith("GIT_"):
            environment.pop(name, None)
    environment["PATH"] = "/usr/bin:/bin"
    return environment


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
        raise ArtifactRunnerError(
            f"effective profile is not canonical JSON: {error}"
        ) from error


def _checkpoint_metadata_sha256(checkpoints: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(checkpoints)).hexdigest()


def _contract_schema_binding(
    repository_root: Path = REPO_ROOT,
    head: str | None = None,
) -> dict[str, object]:
    """Bind v5 contract files to both working-tree bytes and the current HEAD blob."""

    root = repository_root.resolve()
    if head is None:
        completed = subprocess.run(
            [str(SYSTEM_GIT), "rev-parse", "HEAD"],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
            env=_git_environment(),
        )
        if completed.returncode != 0:
            raise ArtifactRunnerError(
                "cannot resolve HEAD while binding schema-v5 contract files"
            )
        head = completed.stdout.strip()
    if SHA_PATTERN.fullmatch(head) is None:
        raise ArtifactRunnerError(f"schema-v5 binding HEAD is invalid: {head!r}")

    records: dict[str, dict[str, object]] = {}
    for role, relative in (
        ("schema", SCHEMA_RELATIVE_PATH),
        ("validator", VALIDATOR_RELATIVE_PATH),
        ("runner", RUNNER_RELATIVE_PATH),
    ):
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise ArtifactRunnerError(
                f"schema-v5 {role} must be a regular file: {path}"
            )
        completed = subprocess.run(
            [str(SYSTEM_GIT), "show", f"{head}:{relative}"],
            cwd=root,
            check=False,
            capture_output=True,
            env=_git_environment(),
        )
        if completed.returncode != 0:
            raise ArtifactRunnerError(
                f"schema-v5 {role} is not committed at HEAD: {relative}"
            )
        actual = path.read_bytes()
        if actual != completed.stdout:
            raise ArtifactRunnerError(
                f"schema-v5 {role} working-tree bytes differ from HEAD blob: {relative}"
            )
        records[role] = {
            "path": relative,
            "sha256": hashlib.sha256(actual).hexdigest(),
        }

    schema_document = _load_json(root / SCHEMA_RELATIVE_PATH)
    if not isinstance(schema_document, dict):
        raise ArtifactRunnerError("schema-v5 document must be an object")
    identity = {
        "id": schema_document.get("$id"),
        "title": schema_document.get("title"),
        "version": schema_document.get("properties", {})
        .get("schema_version", {})
        .get("const"),
    }
    expected_identity = {
        "id": SCHEMA_ID,
        "title": SCHEMA_TITLE,
        "version": SCHEMA_VERSION,
    }
    if identity != expected_identity:
        raise ArtifactRunnerError(
            f"schema-v5 identity drifted: expected={expected_identity} actual={identity}"
        )
    return {
        "schema": {**identity, **records["schema"]},
        "validator": records["validator"],
        "runner": records["runner"],
    }


def _precheck_sentinel_ranges(value: Any, label: str) -> dict[str, dict[str, float]]:
    expected_keys = {
        "softmax_denominator",
        "attention_layernorm_normalized_variance",
        "output_layernorm_normalized_variance",
    }
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise ArtifactRunnerError(
            f"{label} has an invalid shape: expected keys={sorted(expected_keys)}"
        )
    result: dict[str, dict[str, float]] = {}
    for name in sorted(expected_keys):
        observed = value[name]
        if not isinstance(observed, dict) or set(observed) != {"minimum", "maximum"}:
            raise ArtifactRunnerError(f"{label}.{name} must contain minimum/maximum")
        minimum_limit, maximum_limit = (
            (0.01, 80.0)
            if name == "softmax_denominator"
            else LAYERNORM_REGISTERED_INTERVAL
        )
        minimum = _finite_number(
            observed["minimum"], f"{label}.{name}.minimum", minimum_limit, maximum_limit
        )
        maximum = _finite_number(
            observed["maximum"], f"{label}.{name}.maximum", minimum_limit, maximum_limit
        )
        if minimum > maximum:
            raise ArtifactRunnerError(f"{label}.{name} minimum exceeds maximum")
        result[name] = {"minimum": minimum, "maximum": maximum}
    return result


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

    def parse_finite_float(token: str) -> float:
        value = float(token)
        if not math.isfinite(value):
            raise ArtifactRunnerError(
                f"non-finite JSON number in {path}: {token}"
            )
        return value

    try:
        with path.open("r", encoding="utf-8") as stream:
            return json.load(
                stream,
                object_pairs_hook=reject_duplicate_keys,
                parse_constant=reject_non_finite,
                parse_float=parse_finite_float,
            )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ArtifactRunnerError(f"cannot read JSON {path}: {error}") from error


def _binary64_tensor_sha256(
    value: Any,
    expected_shape: tuple[int, ...],
    label: str,
) -> str:
    digest = hashlib.sha256()

    def update(node: Any, dimensions: tuple[int, ...], location: str) -> None:
        if not dimensions:
            if (
                not isinstance(node, (int, float))
                or isinstance(node, bool)
                or not math.isfinite(node)
            ):
                raise ArtifactRunnerError(f"{location} must be a finite number")
            digest.update(struct.pack("<d", float(node)))
            return
        if not isinstance(node, list) or len(node) != dimensions[0]:
            raise ArtifactRunnerError(f"{location} must have dimension {dimensions[0]}")
        for index, child in enumerate(node):
            update(child, dimensions[1:], f"{location}[{index}]")

    update(value, expected_shape, label)
    return digest.hexdigest()


def _trace_scale_contract_provenance(
    repository_root: Path = REPO_ROOT,
) -> dict[str, str]:
    approximation_path = repository_root / TRACE_SCALE_SOURCE_PATH
    profile_path = repository_root / PROFILE_PATH
    approximation = _load_json(approximation_path)
    profile_config = _load_json(profile_path)
    try:
        contract = approximation["operators"]["layernorm"][
            "feature_packed_trace_scale_contract"
        ]
        profile_binding = profile_config["feature_packed_layernorm_override"][
            "trace_scale_contract"
        ]
        runtime_dependency = contract["scope"]["runtime_activation_dependency"]
        exact_identity_gate = contract["inactive_guard"]["exact_identity_gate"]
        hard_interval = contract["normalized_variance_range"]["hard_interval"]
        shape = contract["shape"]
        values = contract["values"]
    except (KeyError, TypeError) as error:
        raise ArtifactRunnerError(
            f"feature-packed LayerNorm trace-scale contract is incomplete: {error}"
        ) from error
    if not isinstance(contract, dict):
        raise ArtifactRunnerError(
            "feature-packed LayerNorm trace-scale contract must be an object"
        )

    canonical_contract = dict(contract)
    stored_contract_sha256 = canonical_contract.pop("contract_sha256", None)
    computed_contract_sha256 = hashlib.sha256(
        _canonical_json_bytes(canonical_contract)
    ).hexdigest()
    computed_values_sha256 = _binary64_tensor_sha256(
        values,
        TRACE_SCALE_SHAPE,
        f"{TRACE_SCALE_JSON_LOCATOR}.values",
    )
    expected_provenance = {
        "source_path": TRACE_SCALE_SOURCE_PATH,
        "json_locator": TRACE_SCALE_JSON_LOCATOR,
        "contract_id": TRACE_SCALE_CONTRACT_ID,
        "contract_sha256": TRACE_SCALE_CONTRACT_SHA256,
        "values_sha256": TRACE_SCALE_VALUES_SHA256,
        "raw_variance_sha256": TRACE_SCALE_RAW_VARIANCE_SHA256,
    }
    source_provenance = {
        "source_path": TRACE_SCALE_SOURCE_PATH,
        "json_locator": TRACE_SCALE_JSON_LOCATOR,
        "contract_id": contract.get("contract_id"),
        "contract_sha256": stored_contract_sha256,
        "values_sha256": contract.get("values_sha256"),
        "raw_variance_sha256": contract.get("raw_variance_sha256"),
    }
    if computed_contract_sha256 != TRACE_SCALE_CONTRACT_SHA256:
        raise ArtifactRunnerError(
            "feature-packed LayerNorm trace-scale canonical SHA-256 drifted: "
            f"expected={TRACE_SCALE_CONTRACT_SHA256} "
            f"actual={computed_contract_sha256}"
        )
    if computed_values_sha256 != TRACE_SCALE_VALUES_SHA256:
        raise ArtifactRunnerError(
            "feature-packed LayerNorm trace-scale values SHA-256 drifted: "
            f"expected={TRACE_SCALE_VALUES_SHA256} actual={computed_values_sha256}"
        )
    if source_provenance != expected_provenance:
        raise ArtifactRunnerError(
            "feature-packed LayerNorm trace-scale provenance drifted: "
            f"expected={expected_provenance} actual={source_provenance}"
        )
    if profile_binding != expected_provenance:
        raise ArtifactRunnerError(
            "feature profile trace-scale provenance does not match the "
            "approximation contract"
        )
    if (
        shape != list(TRACE_SCALE_SHAPE)
        or runtime_dependency != "none"
        or exact_identity_gate is not False
        or hard_interval != list(LAYERNORM_REGISTERED_INTERVAL)
    ):
        raise ArtifactRunnerError(
            "feature-packed LayerNorm trace-scale semantic contract drifted"
        )
    return expected_provenance


def _require_m4_schedule_sealed(repository_root: Path = REPO_ROOT) -> None:
    profile_config = _load_json(repository_root / PROFILE_PATH)
    try:
        schedule_status = profile_config["feature_packed_layernorm_override"][
            "schedule_status"
        ]
    except (KeyError, TypeError) as error:
        raise ArtifactRunnerError(
            "feature profile lacks the M4 LayerNorm schedule status"
        ) from error
    if schedule_status != M4_SCHEDULE_SEALED_STATUS:
        raise ArtifactRunnerError(
            "M4 metadata schedule is not sealed; refusing homomorphic execution: "
            f"schedule_status={schedule_status!r}"
        )


def _resolve_repository_file(path: Path, label: str) -> Path:
    candidate = Path(os.path.abspath(os.fspath(path)))
    try:
        relative_parts = candidate.relative_to(REPO_ROOT).parts
    except ValueError as error:
        raise ArtifactRunnerError(
            f"{label} must stay inside {REPO_ROOT}: {path}"
        ) from error
    cursor = REPO_ROOT
    for part in relative_parts:
        cursor /= part
        if cursor.is_symlink():
            raise ArtifactRunnerError(f"{label} path contains a symlink: {cursor}")
    resolved = candidate.resolve()
    try:
        resolved.relative_to(REPO_ROOT.resolve())
    except ValueError as error:
        raise ArtifactRunnerError(
            f"{label} must stay inside {REPO_ROOT}: {path}"
        ) from error
    if not resolved.is_file():
        raise ArtifactRunnerError(f"{label} is not a file: {resolved}")
    return resolved


def _relative_file_record(relative_path: str, label: str) -> dict[str, object]:
    path = _resolve_repository_file(REPO_ROOT / relative_path, label)
    return {
        "path": relative_path,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _head_bound_source_records(head: str) -> list[dict[str, object]]:
    if SHA_PATTERN.fullmatch(head) is None:
        raise ArtifactRunnerError(f"source provenance HEAD is invalid: {head!r}")
    records: list[dict[str, object]] = []
    for relative_path in M4_SOURCE_PATHS:
        path = _resolve_repository_file(
            REPO_ROOT / relative_path,
            f"M4 source provenance {relative_path}",
        )
        completed = subprocess.run(
            [str(SYSTEM_GIT), "show", f"{head}:{relative_path}"],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            env=_git_environment(),
        )
        if completed.returncode != 0:
            raise ArtifactRunnerError(
                f"M4 source provenance is not committed at HEAD: {relative_path}"
            )
        actual = path.read_bytes()
        if actual != completed.stdout:
            raise ArtifactRunnerError(
                f"M4 source provenance differs from HEAD blob: {relative_path}"
            )
        digest = hashlib.sha256(actual).hexdigest()
        records.append(
            {
                "path": relative_path,
                "bytes": len(actual),
                "sha256": digest,
                "head_blob_sha256": digest,
            }
        )
    return records


def _source_build_provenance(
    head: str,
    executable: Path,
) -> dict[str, object]:
    executable = _resolve_m4_executable(executable)
    executable_relative = executable.relative_to(REPO_ROOT).as_posix()
    return {
        "source_files": _head_bound_source_records(head),
        "build_products": [
            _relative_file_record(
                relative_path,
                f"M4 build provenance {relative_path}",
            )
            for relative_path in M4_BUILD_PRODUCT_PATHS
        ],
        "executable": _relative_file_record(
            executable_relative,
            "M4 executable provenance",
        ),
    }


def _require_source_build_provenance(
    expected: dict[str, object],
    head: str,
    executable: Path,
    phase: str,
) -> None:
    actual = _source_build_provenance(head, executable)
    if actual != expected:
        raise ArtifactRunnerError(
            f"M4 source/build provenance changed {phase}: "
            f"expected={expected} actual={actual}"
        )


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
        raise ArtifactRunnerError(
            f"M4 executable must stay inside {REPO_ROOT}"
        ) from error
    cursor = REPO_ROOT
    for part in relative_parts:
        cursor /= part
        if cursor.is_symlink():
            raise ArtifactRunnerError(
                f"M4 executable path must not contain symlinks: {cursor}"
            )
    if not require_exists:
        if candidate.exists() and not candidate.is_file():
            raise ArtifactRunnerError(
                f"M4 executable is not a regular file: {candidate}"
            )
        return candidate
    if not candidate.is_file():
        raise ArtifactRunnerError(f"M4 executable is not a file: {candidate}")
    if not os.access(candidate, os.X_OK):
        raise ArtifactRunnerError(f"M4 executable is not executable: {candidate}")
    return candidate


def _resolve_output_root(path: Path) -> Path:
    resolved = path.resolve()
    allowed = DEFAULT_OUTPUT_ROOT.resolve()
    if resolved != allowed:
        raise ArtifactRunnerError(
            f"output root must be the frozen artifact root {allowed}: {path}"
        )
    if path.is_symlink() or (path.exists() and not path.is_dir()):
        raise ArtifactRunnerError(
            f"output root must be a canonical directory or absent: {path}"
        )
    return resolved


def _preflight_run_root(output_root: Path, run_id: str) -> Path:
    resolved_output = _resolve_output_root(output_root)
    run_root = resolved_output / run_id
    if run_root.exists() or run_root.is_symlink():
        raise ArtifactRunnerError(f"run directory already exists: {run_root}")
    return run_root


def _run_git(arguments: list[str]) -> tuple[str, dict[str, object]]:
    command = [str(SYSTEM_GIT), *arguments]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=_git_environment(),
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


def _run_checked_command(
    command: list[str], label: str
) -> tuple[dict[str, object], str]:
    started_at = _timestamp()
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=_artifact_environment(),
    )
    finished_at = _timestamp()
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        if len(detail) > 4000:
            detail = detail[-4000:]
        raise ArtifactRunnerError(
            f"{label} failed with {completed.returncode}: {detail}"
        )
    return (
        {
            "command": shlex.join(command),
            "cwd": str(REPO_ROOT),
            "exit_code": completed.returncode,
            "phase": "artifact_generation",
            "started_at": started_at,
            "finished_at": finished_at,
        },
        completed.stdout,
    )


def _require_ctest_passed_tests(
    stdout: str,
    expected_tests: tuple[str, ...],
    label: str,
) -> None:
    summary_pattern = re.compile(
        r"^\s*(\d+)/(\d+)\s+Test\s+#(\d+):\s+([A-Za-z0-9_.+-]+)"
        r"\s+\.+\s+(.+?)\s+[0-9]+(?:\.[0-9]+)?\s+sec\s*$",
        re.MULTILINE,
    )
    summaries = summary_pattern.findall(stdout)
    observed_tests = tuple(name for _, _, _, name, _ in summaries)
    if len(set(observed_tests)) != len(observed_tests):
        raise ArtifactRunnerError(
            f"{label} CTest summary contains duplicate test-name records: "
            f"{observed_tests}"
        )
    test_ids = tuple(int(test_id) for _, _, test_id, _, _ in summaries)
    if len(set(test_ids)) != len(test_ids):
        raise ArtifactRunnerError(
            f"{label} CTest summary contains duplicate test-id records: {test_ids}"
        )
    expected_count = len(expected_tests)
    ordinals = {int(ordinal) for ordinal, _, _, _, _ in summaries}
    totals = {int(total) for _, total, _, _, _ in summaries}
    statuses = tuple(status.strip() for _, _, _, _, status in summaries)
    footer_pattern = re.compile(
        rf"^100% tests passed, 0 tests failed out of {expected_count}$",
        re.MULTILINE,
    )
    footer_count = len(footer_pattern.findall(stdout))
    if (
        set(observed_tests) != set(expected_tests)
        or len(observed_tests) != expected_count
        or ordinals != set(range(1, expected_count + 1))
        or totals != {expected_count}
        or statuses != ("Passed",) * expected_count
        or footer_count != 1
    ):
        missing = [name for name in expected_tests if name not in observed_tests]
        unexpected = [name for name in observed_tests if name not in expected_tests]
        raise ArtifactRunnerError(
            f"{label} CTest did not enumerate and pass the exact frozen test set: "
            f"missing={missing} unexpected={unexpected} "
            f"ordinals={sorted(ordinals)} totals={sorted(totals)} "
            f"statuses={statuses} footer_count={footer_count} "
            f"actual={observed_tests} expected={expected_tests}"
        )


def _read_openfhe_version(openfhe_prefix: Path) -> str:
    version_file = openfhe_prefix / "lib" / "OpenFHE" / "OpenFHEConfigVersion.cmake"
    try:
        contents = version_file.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ArtifactRunnerError(
            f"cannot read OpenFHE version file: {error}"
        ) from error
    match = re.search(
        r'^set\(PACKAGE_VERSION "([0-9]+\.[0-9]+\.[0-9]+)"\)$', contents, re.M
    )
    if match is None or match.group(1) != "1.5.1":
        raise ArtifactRunnerError(
            f"OpenFHE package version must be 1.5.1: {version_file}"
        )
    return match.group(1)


def _require_frozen_openfhe_prefix(openfhe_prefix: Path) -> Path:
    prefix = Path(os.path.abspath(os.fspath(openfhe_prefix)))
    expected = DEFAULT_OPENFHE_PREFIX
    if prefix != expected or prefix.resolve() != expected:
        raise ArtifactRunnerError(
            f"OpenFHE prefix must be the canonical path {expected}, got {openfhe_prefix}"
        )
    return prefix


def _configure_command(build_root: Path, openfhe_prefix: Path) -> list[str]:
    prefix = _require_frozen_openfhe_prefix(openfhe_prefix)
    if Path(os.path.abspath(os.fspath(build_root))) != DEFAULT_EXECUTABLE.parent:
        raise ArtifactRunnerError(
            f"M4 build root must be {DEFAULT_EXECUTABLE.parent}, got {build_root}"
        )
    return [
        str(SYSTEM_ENV),
        *(f"--unset={name}" for name in CONFIGURE_ENV_UNSET),
        str(SYSTEM_CMAKE),
        "--fresh",
        "-S",
        str(REPO_ROOT),
        "-B",
        str(DEFAULT_EXECUTABLE.parent),
        "-G",
        "Unix Makefiles",
        f"-DOpenFHE_DIR:PATH={prefix / 'lib' / 'OpenFHE'}",
        "-DCMAKE_BUILD_TYPE:STRING=Release",
        "-DBUILD_TESTING:BOOL=ON",
        f"-DCMAKE_CXX_COMPILER:FILEPATH={SYSTEM_CXX}",
        f"-DCMAKE_MAKE_PROGRAM:FILEPATH={SYSTEM_MAKE}",
        "-DCMAKE_CXX_FLAGS:STRING=",
        "-DCMAKE_CXX_FLAGS_RELEASE:STRING=-O3 -DNDEBUG",
        "-DCMAKE_EXE_LINKER_FLAGS:STRING=",
        "-DCMAKE_EXE_LINKER_FLAGS_RELEASE:STRING=",
        "-DCMAKE_SHARED_LINKER_FLAGS:STRING=",
        "-DCMAKE_MODULE_LINKER_FLAGS:STRING=",
        "-DCMAKE_STATIC_LINKER_FLAGS:STRING=",
    ]


def _parse_cmake_cache(cache_path: Path) -> dict[str, tuple[str, str]]:
    if (
        cache_path.is_symlink()
        or not cache_path.is_file()
        or cache_path.resolve() != cache_path
    ):
        raise ArtifactRunnerError(
            f"M4 CMake cache must be a regular file: {cache_path}"
        )
    try:
        lines = cache_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise ArtifactRunnerError(f"cannot read M4 CMake cache: {error}") from error
    entries: dict[str, tuple[str, str]] = {}
    for line in lines:
        if not line or line.startswith(("//", "#")) or "=" not in line:
            continue
        key_and_type, value = line.split("=", maxsplit=1)
        if ":" not in key_and_type:
            continue
        key, entry_type = key_and_type.rsplit(":", maxsplit=1)
        if key in entries:
            raise ArtifactRunnerError(f"M4 CMake cache repeats key {key}")
        entries[key] = (entry_type, value)
    return entries


def _verify_build_configuration(
    build_root: Path,
    openfhe_prefix: Path,
) -> dict[str, object]:
    if Path(os.path.abspath(os.fspath(build_root))) != DEFAULT_EXECUTABLE.parent:
        raise ArtifactRunnerError(
            f"M4 build root must be {DEFAULT_EXECUTABLE.parent}, got {build_root}"
        )
    prefix = _require_frozen_openfhe_prefix(openfhe_prefix)
    cache_path = build_root / "CMakeCache.txt"
    entries = _parse_cmake_cache(cache_path)
    expected_entries = {
        "CMAKE_GENERATOR": ("INTERNAL", "Unix Makefiles"),
        "CMAKE_HOME_DIRECTORY": ("INTERNAL", str(REPO_ROOT)),
        "CMAKE_BUILD_TYPE": ("STRING", "Release"),
        "BUILD_TESTING": ("BOOL", "ON"),
        "OpenFHE_DIR": ("PATH", str(prefix / "lib" / "OpenFHE")),
        # CMake normalizes this known cache entry from the command-line
        # FILEPATH hint to STRING while project(... LANGUAGES CXX) initializes.
        "CMAKE_CXX_COMPILER": ("STRING", str(SYSTEM_CXX)),
        "CMAKE_MAKE_PROGRAM": ("FILEPATH", str(SYSTEM_MAKE)),
        "CMAKE_CXX_FLAGS": ("STRING", ""),
        "CMAKE_CXX_FLAGS_RELEASE": ("STRING", "-O3 -DNDEBUG"),
        "CMAKE_EXE_LINKER_FLAGS": ("STRING", ""),
        "CMAKE_EXE_LINKER_FLAGS_RELEASE": ("STRING", ""),
        "CMAKE_SHARED_LINKER_FLAGS": ("STRING", ""),
        "CMAKE_MODULE_LINKER_FLAGS": ("STRING", ""),
        "CMAKE_STATIC_LINKER_FLAGS": ("STRING", ""),
    }
    drift = {
        key: {"expected": expected, "actual": entries.get(key)}
        for key, expected in expected_entries.items()
        if entries.get(key) != expected
    }
    forbidden_nonempty = {
        key: entries[key][1]
        for key in (
            "CMAKE_TOOLCHAIN_FILE",
            "CMAKE_CXX_COMPILER_LAUNCHER",
            "CMAKE_CXX_LINKER_LAUNCHER",
            "CMAKE_PROJECT_TOP_LEVEL_INCLUDES",
            "CMAKE_PROJECT_INCLUDE",
            "CMAKE_PROJECT_INCLUDE_BEFORE",
            "CMAKE_USER_MAKE_RULES_OVERRIDE",
            "CMAKE_USER_MAKE_RULES_OVERRIDE_CXX",
        )
        if key in entries and entries[key][1]
    }
    if drift or forbidden_nonempty:
        raise ArtifactRunnerError(
            "M4 CMake cache binding drifted: "
            f"critical={drift} forbidden={forbidden_nonempty}"
        )
    _read_openfhe_version(prefix)
    return {
        "generator": "Unix Makefiles",
        "source_root": str(REPO_ROOT),
        "build_root": str(DEFAULT_EXECUTABLE.parent),
        "openfhe_dir": str(prefix / "lib" / "OpenFHE"),
        "build_type": "Release",
        "build_testing": True,
        "cxx_compiler": str(SYSTEM_CXX),
        "make_program": str(SYSTEM_MAKE),
        "cxx_flags": "",
        "cxx_flags_release": "-O3 -DNDEBUG",
        "exe_linker_flags": "",
        "exe_linker_flags_release": "",
        "shared_linker_flags": "",
        "module_linker_flags": "",
        "static_linker_flags": "",
        "environment_inheritance": "ambient_minus_cleared_variables",
        "cleared_environment_variables": list(CONFIGURE_ENV_UNSET),
        "forced_environment_variables": dict(FORCED_SUBPROCESS_ENVIRONMENT),
        "cmake_cache": {
            "path": "build-openfhe/CMakeCache.txt",
            "bytes": cache_path.stat().st_size,
            "sha256": _sha256(cache_path),
        },
    }


def _regular_file_record(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file() or path.resolve() != path:
        raise ArtifactRunnerError(f"provenance input must be a regular file: {path}")
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _openfhe_package_records(openfhe_prefix: Path) -> list[dict[str, object]]:
    prefix = _require_frozen_openfhe_prefix(openfhe_prefix)
    package_root = prefix / "lib" / "OpenFHE"
    return [
        _regular_file_record(package_root / name)
        for name in OPENFHE_CMAKE_PACKAGE_FILES
    ]


def _openfhe_include_tree_record(openfhe_prefix: Path) -> dict[str, object]:
    prefix = _require_frozen_openfhe_prefix(openfhe_prefix)
    root = prefix / "include" / "openfhe"
    if root.is_symlink() or not root.is_dir() or root.resolve() != root:
        raise ArtifactRunnerError(
            f"OpenFHE include tree must be a canonical directory: {root}"
        )
    files: list[Path] = []
    for candidate in root.rglob("*"):
        if candidate.is_symlink():
            raise ArtifactRunnerError(
                f"OpenFHE include tree must not contain symlinks: {candidate}"
            )
        if candidate.is_file():
            files.append(candidate)
    files.sort(key=lambda path: path.relative_to(root).as_posix())
    if not files:
        raise ArtifactRunnerError(f"OpenFHE include tree is empty: {root}")
    digest = hashlib.sha256()
    total_bytes = 0
    for path in files:
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        total_bytes += size
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        digest.update(_sha256(path).encode("ascii"))
        digest.update(b"\n")
    return {
        "root": str(root),
        "file_count": len(files),
        "bytes": total_bytes,
        "algorithm": "SHA-256",
        "canonicalization": OPENFHE_INCLUDE_TREE_CANONICALIZATION,
        "sha256": digest.hexdigest(),
    }


def _inspect_openfhe_linkage(
    executable: Path,
    openfhe_prefix: Path,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    command = [str(SYSTEM_LDD), str(executable)]
    started_at = _timestamp()
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=_artifact_environment(),
    )
    finished_at = _timestamp()
    if completed.returncode != 0:
        raise ArtifactRunnerError(
            f"cannot inspect M4 OpenFHE linkage: {completed.stderr.strip()}"
        )
    linked: dict[str, dict[str, object]] = {}
    for line in completed.stdout.splitlines():
        if "libOPENFHE" not in line:
            continue
        match = re.fullmatch(
            r"\s*(libOPENFHE\S+)\s+=>\s+(\S+)\s+\(0x[0-9a-fA-F]+\)\s*",
            line,
        )
        if match is None:
            raise ArtifactRunnerError(f"malformed M4 OpenFHE ldd line: {line.strip()}")
        soname, target = match.groups()
        if target == "not" or soname not in OPENFHE_LINKED_LIBRARY_SONAMES:
            raise ArtifactRunnerError(
                f"M4 OpenFHE library is unresolved: {line.strip()}"
            )
        if soname in linked:
            raise ArtifactRunnerError(f"M4 OpenFHE SONAME is duplicated: {soname}")
        target_path = Path(target)
        if not target_path.is_absolute():
            raise ArtifactRunnerError(f"M4 OpenFHE ldd path is not absolute: {target}")
        try:
            resolved = target_path.resolve(strict=True)
        except OSError as error:
            raise ArtifactRunnerError(
                f"M4 OpenFHE library cannot be resolved: {target}"
            ) from error
        library_root = _require_frozen_openfhe_prefix(openfhe_prefix) / "lib"
        try:
            resolved.relative_to(library_root)
        except ValueError as error:
            raise ArtifactRunnerError(
                f"M4 executable links OpenFHE outside {library_root}: {resolved}"
            ) from error
        if resolved.is_symlink() or not resolved.is_file():
            raise ArtifactRunnerError(
                f"M4 resolved OpenFHE library is not a regular file: {resolved}"
            )
        if resolved.name != OPENFHE_LINKED_LIBRARY_BASENAMES[soname]:
            raise ArtifactRunnerError(
                f"M4 OpenFHE versioned path drifted for {soname}: {resolved}"
            )
        linked[soname] = {
            "soname": soname,
            **_regular_file_record(resolved),
        }
    if set(linked) != set(OPENFHE_LINKED_LIBRARY_SONAMES):
        raise ArtifactRunnerError(
            "M4 executable must resolve exactly three OpenFHE SONAMEs: "
            f"expected={list(OPENFHE_LINKED_LIBRARY_SONAMES)} "
            f"actual={sorted(linked)}"
        )
    records = [linked[soname] for soname in OPENFHE_LINKED_LIBRARY_SONAMES]
    return (
        {
            "command": shlex.join(command),
            "cwd": str(REPO_ROOT),
            "exit_code": completed.returncode,
            "phase": "artifact_generation",
            "started_at": started_at,
            "finished_at": finished_at,
        },
        records,
    )


def _build_configuration_snapshot(
    build_root: Path,
    executable: Path,
    openfhe_prefix: Path,
    linked_libraries: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    result = _verify_build_configuration(build_root, openfhe_prefix)
    if linked_libraries is None:
        _, linked_libraries = _inspect_openfhe_linkage(executable, openfhe_prefix)
    return {
        **result,
        "openfhe_cmake_package_files": _openfhe_package_records(openfhe_prefix),
        "openfhe_include_tree": _openfhe_include_tree_record(openfhe_prefix),
        "openfhe_linked_libraries": linked_libraries,
    }


def _require_build_configuration(
    expected: dict[str, object],
    build_root: Path,
    executable: Path,
    openfhe_prefix: Path,
    phase: str,
) -> None:
    actual = _build_configuration_snapshot(build_root, executable, openfhe_prefix)
    if actual != expected:
        raise ArtifactRunnerError(
            f"M4 build/OpenFHE provenance changed {phase}: "
            f"expected={expected} actual={actual}"
        )


def _run_m4_preflight(
    build_root: Path,
    executable: Path,
    openfhe_prefix: Path,
) -> M4Preflight:
    missing_v5_contracts = [
        name
        for name in M4_V5_REQUIRED_CTEST_CONTRACTS
        if re.fullmatch(M4_CTEST_PATTERN, name) is None
    ]
    if missing_v5_contracts:
        raise ArtifactRunnerError(
            "M4 v5 narrow CTest pattern omits required contracts: "
            f"{missing_v5_contracts}"
        )
    configure_command = _configure_command(build_root, openfhe_prefix)
    build_command = [
        str(SYSTEM_CMAKE),
        "--build",
        str(build_root),
        "--clean-first",
        "-j",
        "4",
    ]
    test_command = [
        str(SYSTEM_CTEST),
        "--test-dir",
        str(build_root),
        "--output-on-failure",
        "--no-tests=error",
        "-R",
        M4_CTEST_PATTERN,
    ]
    configure_record, _ = _run_checked_command(
        configure_command,
        "fresh fixed-configuration M4 configure",
    )
    configured = _verify_build_configuration(build_root, openfhe_prefix)
    build_record, _ = _run_checked_command(build_command, "clean-commit M4 build")
    if _verify_build_configuration(build_root, openfhe_prefix) != configured:
        raise ArtifactRunnerError("M4 CMake configuration changed during the build")
    rebuilt_executable = _resolve_m4_executable(executable)
    linkage_record, linked_libraries = _inspect_openfhe_linkage(
        rebuilt_executable,
        openfhe_prefix,
    )
    build_configuration = _build_configuration_snapshot(
        build_root,
        rebuilt_executable,
        openfhe_prefix,
        linked_libraries,
    )
    test_record, test_stdout = _run_checked_command(
        test_command, "clean-commit M4 narrow gates"
    )
    _require_ctest_passed_tests(
        test_stdout,
        M4_CTEST_EXPECTED_TESTS,
        "M4 narrow gates",
    )
    _require_build_configuration(
        build_configuration,
        build_root,
        rebuilt_executable,
        openfhe_prefix,
        "after M4 narrow gates",
    )
    return M4Preflight(
        (configure_record, build_record, linkage_record, test_record),
        build_configuration,
    )


def _preflight_git() -> GitState:
    commands: list[dict[str, object]] = []
    status, record = _run_git(["status", "--porcelain=v1", "--untracked-files=normal"])
    commands.append(record)
    if status:
        raise ArtifactRunnerError(
            "repository is not clean; refusing to run M4 evidence"
        )

    branch, record = _run_git(["branch", "--show-current"])
    commands.append(record)
    if branch != BRANCH:
        raise ArtifactRunnerError(f"branch must be {BRANCH}, got {branch!r}")

    head, record = _run_git(["rev-parse", "HEAD"])
    commands.append(record)
    if SHA_PATTERN.fullmatch(head) is None:
        raise ArtifactRunnerError(f"local HEAD is not a full Git SHA: {head!r}")

    remote_url, record = _run_git(
        ["remote", "get-url", "--all", REMOTE_NAME]
    )
    commands.append(record)
    remote_urls = [line for line in remote_url.splitlines() if line]
    if remote_urls != [REMOTE_URL]:
        raise ArtifactRunnerError(
            f"{REMOTE_NAME} URL must be {REMOTE_URL}, got {remote_urls!r}"
        )

    remote_output, record = _run_git(
        ["ls-remote", "--exit-code", REMOTE_URL, REMOTE_REF]
    )
    commands.append(record)
    remote_lines = [line.split() for line in remote_output.splitlines() if line.strip()]
    if len(remote_lines) != 1 or len(remote_lines[0]) < 2:
        raise ArtifactRunnerError(
            f"remote ref lookup was not unique: {remote_output!r}"
        )
    remote_head, remote_ref = remote_lines[0][0], remote_lines[0][1]
    if SHA_PATTERN.fullmatch(remote_head) is None or remote_ref != REMOTE_REF:
        raise ArtifactRunnerError(f"remote ref lookup was malformed: {remote_output!r}")
    if head != remote_head:
        raise ArtifactRunnerError(
            f"local HEAD {head} differs from {REMOTE_NAME}/{BRANCH} {remote_head}"
        )
    return GitState(
        head=head,
        remote_url=REMOTE_URL,
        remote_head=remote_head,
        commands=tuple(commands),
    )


def _version_line(command: list[str], label: str) -> str:
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=_artifact_environment(),
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
        "compiler": _version_line([str(SYSTEM_CXX), "--version"], "C++ compiler"),
        "cmake": _version_line([str(SYSTEM_CMAKE), "--version"], "CMake"),
        "python": platform.python_version(),
        "openfhe_version": _read_openfhe_version(prefix),
        "openfhe_prefix": str(prefix),
        "wsl": "microsoft" in release.lower() or "WSL_DISTRO_NAME" in os.environ,
    }


def _load_json_line(line: str, label: str) -> dict[str, Any]:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ArtifactRunnerError(
                    f"{label} repeats JSON key {key!r}"
                )
            result[key] = value
        return result

    def reject_non_finite(token: str) -> None:
        raise ArtifactRunnerError(
            f"{label} contains non-finite JSON number {token}"
        )

    def parse_finite_float(token: str) -> float:
        value = float(token)
        if not math.isfinite(value):
            raise ArtifactRunnerError(
                f"{label} contains non-finite JSON number {token}"
            )
        return value

    try:
        value = json.loads(
            line,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_non_finite,
            parse_float=parse_finite_float,
        )
    except json.JSONDecodeError as error:
        raise ArtifactRunnerError(f"{label} is malformed JSON: {error}") from error
    if not isinstance(value, dict):
        raise ArtifactRunnerError(f"{label} must be a JSON object")
    return value


def _parse_stdout_run(
    stdout: bytes | str,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    data = stdout.encode("utf-8") if isinstance(stdout, str) else stdout
    if not data or len(data) > RAW_STDOUT_MAX_BYTES:
        raise ArtifactRunnerError(f"{label} stdout size is outside the frozen bound")
    if not data.endswith(b"\n") or b"\r" in data or b"\0" in data:
        raise ArtifactRunnerError(
            f"{label} stdout must be NUL-free UTF-8 with LF line endings"
        )
    raw_lines = data[:-1].split(b"\n")
    if not raw_lines or any(not line for line in raw_lines):
        raise ArtifactRunnerError(f"{label} stdout contains an empty line")
    if any(len(line) > RAW_STDOUT_MAX_LINE_BYTES for line in raw_lines):
        raise ArtifactRunnerError(f"{label} stdout contains an oversized line")
    try:
        lines = [line.decode("utf-8") for line in raw_lines]
    except UnicodeDecodeError as error:
        raise ArtifactRunnerError(f"{label} stdout is not UTF-8") from error
    if lines[0] != PROFILE_WARNING or PROFILE_WARNING in lines[1:]:
        raise ArtifactRunnerError(
            f"{label} must start with exactly one frozen profile warning"
        )

    selected: list[dict[str, Any]] = []
    for index, line in enumerate(lines[1:], start=2):
        if not line.startswith("{"):
            raise ArtifactRunnerError(
                f"{label} stdout line {index} is neither the warning nor JSON"
            )
        record = _load_json_line(line, f"{label} stdout line {index}")
        if record.get("test") in {
            "openfhe_encoder_layer_smoke",
            "openfhe_encoder_layer",
        }:
            selected.append(record)
    selected_tests = [record["test"] for record in selected]
    if selected_tests != [
        "openfhe_encoder_layer_smoke",
        "openfhe_encoder_layer",
    ]:
        raise ArtifactRunnerError(
            f"{label} stdout target records drifted: {selected_tests!r}"
        )
    diagnostic, target = selected
    return target, diagnostic


def _extract_record(stdout: bytes | str, test_name: str, label: str) -> dict[str, Any]:
    target, diagnostic = _parse_stdout_run(stdout, label)
    if test_name == "openfhe_encoder_layer":
        return target
    if test_name == "openfhe_encoder_layer_smoke":
        return diagnostic
    raise ArtifactRunnerError(f"{label} requested unknown runtime record {test_name!r}")


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
    _finite_number(
        record["layernorm_inactive_guard_max_error"],
        f"{label}.layernorm_inactive_guard_max_error",
        0.0,
        math.inf,
    )
    _precheck_sentinel_ranges(
        record["inactive_polynomial_sentinel_ranges"],
        f"{label}.inactive_polynomial_sentinel_ranges",
    )

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
        if (
            not isinstance(level, int)
            or isinstance(level, bool)
            or not 0 <= level <= 47
        ):
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
        _finite_number(
            checkpoint["scale_bits"], f"{checkpoint_label}.scale_bits", 1.0, 100.0
        )
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
            raise ArtifactRunnerError(
                f"{label}.operation_counts.{key} must be positive"
            )
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


def _runtime_argv(config: RunnerConfig) -> list[str]:
    return [
        str(config.executable),
        "--data-root",
        str(config.data_root),
        "--layer",
        "1",
        "--input-level",
        "29",
    ]


def _time_format(phase: str) -> str:
    if phase not in TIME_PHASES:
        raise ArtifactRunnerError(f"unknown M4 evidence phase: {phase!r}")
    return f"{TIME_RECORD_PREFIX}\t{phase}\t%e\t%M\t%x\t%C"


def _parse_time_record_line(
    line: bytes,
    phase: str,
    runtime_argv: list[str],
    label: str,
) -> dict[str, object]:
    if b"\r" in line or b"\0" in line or not line.endswith(b"\n"):
        raise ArtifactRunnerError(
            f"{label} GNU time record must be NUL-free UTF-8 with LF"
        )
    try:
        text = line[:-1].decode("utf-8")
    except UnicodeDecodeError as error:
        raise ArtifactRunnerError(f"{label} GNU time record is not UTF-8") from error
    fields = text.split("\t", maxsplit=5)
    if len(fields) != 6 or fields[:2] != [TIME_RECORD_PREFIX, phase]:
        raise ArtifactRunnerError(f"{label} GNU time record shape drifted")
    if (
        TIME_ELAPSED_PATTERN.fullmatch(fields[2]) is None
        or TIME_RSS_PATTERN.fullmatch(fields[3]) is None
        or fields[4] != "0"
    ):
        raise ArtifactRunnerError(
            f"{label} GNU time resource field is not canonical C-locale output"
        )
    try:
        elapsed_seconds = float(fields[2])
        peak_rss_kib = int(fields[3])
        exit_status = int(fields[4])
    except ValueError as error:
        raise ArtifactRunnerError(
            f"{label} GNU time resource field is malformed"
        ) from error
    if (
        not math.isfinite(elapsed_seconds)
        or elapsed_seconds <= 0.0
        or peak_rss_kib <= 0
        or exit_status != 0
    ):
        raise ArtifactRunnerError(
            f"{label} GNU time requires positive elapsed/RSS and exit status 0"
        )
    expected_command = " ".join(runtime_argv)
    if fields[5] != expected_command:
        raise ArtifactRunnerError(
            f"{label} GNU time command differs: "
            f"expected={expected_command!r} actual={fields[5]!r}"
        )
    return {
        "phase": phase,
        "elapsed_seconds": elapsed_seconds,
        "peak_rss_kib": peak_rss_kib,
        "exit_status": exit_status,
        "command": fields[5],
    }


def _run_once(config: RunnerConfig, phase: str, time_path: Path) -> RunSample:
    runtime_argv = _runtime_argv(config)
    command = [
        str(TIME_EXECUTABLE),
        "--append",
        f"--format={_time_format(phase)}",
        f"--output={time_path}",
        *runtime_argv,
    ]
    started_at = _timestamp()
    before_size = time_path.stat().st_size
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        env=_artifact_environment(),
    )
    finished_at = _timestamp()
    time_bytes = time_path.read_bytes()
    if len(time_bytes) <= before_size or len(time_bytes) > RAW_TIME_MAX_BYTES:
        raise ArtifactRunnerError(f"{phase} produced invalid GNU time evidence")
    time_record = _parse_time_record_line(
        time_bytes[before_size:],
        phase,
        runtime_argv,
        phase,
    )
    if completed.returncode != time_record["exit_status"]:
        raise ArtifactRunnerError(
            f"{phase} subprocess/GNU time exit status disagree: "
            f"subprocess={completed.returncode} time={time_record['exit_status']}"
        )
    if completed.returncode != 0:
        detail_bytes = completed.stderr.strip() or completed.stdout.strip()
        detail = detail_bytes[:2000].decode("utf-8", errors="replace")
        raise ArtifactRunnerError(
            f"{phase} exited with {completed.returncode}: {detail}"
        )
    if len(completed.stderr) > RAW_STDERR_MAX_BYTES:
        raise ArtifactRunnerError(f"{phase} stderr exceeds the frozen bound")
    record, diagnostic = _parse_stdout_run(completed.stdout, phase)
    _precheck_target_record(record, phase)
    _precheck_diagnostic_record(diagnostic, record, phase)
    peak_rss_kib = int(time_record["peak_rss_kib"])
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
        stderr=completed.stderr,
        record=record,
        diagnostic=diagnostic,
        elapsed_seconds=float(time_record["elapsed_seconds"]),
        peak_rss_kib=peak_rss_kib,
        command=command_record,
    )


def _require_executable_hash(
    executable: Path, expected_sha256: str, phase: str
) -> None:
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
            raise ArtifactRunnerError(
                f"invalid frozen input record {phase}: {record!r}"
            )
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
    if not isinstance(config, dict) or not isinstance(
        config.get("effective_profile"), dict
    ):
        raise ArtifactRunnerError("feature profile lacks /effective_profile")
    effective = config["effective_profile"]
    effective_hash = hashlib.sha256(_canonical_json_bytes(effective)).hexdigest()
    if (
        effective_hash != PROFILE_SHA256
        or config.get("effective_profile_sha256") != PROFILE_SHA256
    ):
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
    data = b"".join(sample.stdout for sample in samples)
    if not data or len(data) > RAW_STDOUT_MAX_BYTES:
        raise ArtifactRunnerError("combined stdout.log is outside the frozen bound")
    path.write_bytes(data)


def _write_stderr(path: Path, samples: list[RunSample]) -> None:
    data = b"".join(sample.stderr for sample in samples)
    if len(data) > RAW_STDERR_MAX_BYTES:
        raise ArtifactRunnerError("combined stderr.log exceeds the frozen bound")
    path.write_bytes(data)


def _metrics_bytes(samples: list[RunSample]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=CSV_FIELDS,
        lineterminator="\n",
    )
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
                "inactive_max_abs": format(
                    sample.record["inactive_max_abs"], ".17g"
                ),
                "layernorm_inactive_guard_max_error": format(
                    sample.record["layernorm_inactive_guard_max_error"], ".17g"
                ),
                "softmax_sentinel_minimum": format(
                    sample.record["inactive_polynomial_sentinel_ranges"][
                        "softmax_denominator"
                    ]["minimum"],
                    ".17g",
                ),
                "softmax_sentinel_maximum": format(
                    sample.record["inactive_polynomial_sentinel_ranges"][
                        "softmax_denominator"
                    ]["maximum"],
                    ".17g",
                ),
                "self_layernorm_sentinel_minimum": format(
                    sample.record["inactive_polynomial_sentinel_ranges"][
                        "attention_layernorm_normalized_variance"
                    ]["minimum"],
                    ".17g",
                ),
                "self_layernorm_sentinel_maximum": format(
                    sample.record["inactive_polynomial_sentinel_ranges"][
                        "attention_layernorm_normalized_variance"
                    ]["maximum"],
                    ".17g",
                ),
                "output_layernorm_sentinel_minimum": format(
                    sample.record["inactive_polynomial_sentinel_ranges"][
                        "output_layernorm_normalized_variance"
                    ]["minimum"],
                    ".17g",
                ),
                "output_layernorm_sentinel_maximum": format(
                    sample.record["inactive_polynomial_sentinel_ranges"][
                        "output_layernorm_normalized_variance"
                    ]["maximum"],
                    ".17g",
                ),
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
    return stream.getvalue().encode("utf-8")


def _write_metrics(path: Path, samples: list[RunSample]) -> None:
    path.write_bytes(_metrics_bytes(samples))


def _summarize_metrics(samples: list[RunSample]) -> dict[str, object]:
    if len(samples) != 5:
        raise ArtifactRunnerError(
            f"M4 requires exactly five measured samples, got {len(samples)}"
        )
    records = [sample.record for sample in samples]
    first_counts = records[0]["operation_counts"]
    if any(record["operation_counts"] != first_counts for record in records[1:]):
        raise ArtifactRunnerError("operation counts differ across measured repeats")

    checkpoints: list[dict[str, object]] = []
    for index, name in enumerate(CHECKPOINT_NAMES):
        values = [record["checkpoints"][index] for record in records]
        ciphertext_counts = {value["ciphertext_count"] for value in values}
        if len(ciphertext_counts) != 1:
            raise ArtifactRunnerError(
                f"checkpoint {name} ciphertext count differs across repeats"
            )
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
            "median": statistics.median(sample.diagnostic[field] for sample in samples),
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
            "inactive_max_abs_max": max(
                record["inactive_max_abs"] for record in records
            ),
            "layernorm_inactive_guard_max_error": max(
                record["layernorm_inactive_guard_max_error"] for record in records
            ),
        },
        "inactive_polynomial_sentinel_ranges": {
            name: {
                "minimum": min(
                    record["inactive_polynomial_sentinel_ranges"][name]["minimum"]
                    for record in records
                ),
                "maximum": max(
                    record["inactive_polynomial_sentinel_ranges"][name]["maximum"]
                    for record in records
                ),
            }
            for name in (
                "softmax_denominator",
                "attention_layernorm_normalized_variance",
                "output_layernorm_normalized_variance",
            )
        },
        "checkpoints": checkpoints,
        "checkpoint_metadata_sha256": EXPECTED_CHECKPOINT_METADATA_SHA256,
        "operation_counts": first_counts,
    }


def _artifact_record(
    root: Path, path: str, role: str, media_type: str
) -> dict[str, object]:
    resolved = root / path
    return {
        "path": path,
        "sha256": _sha256(resolved),
        "bytes": resolved.stat().st_size,
        "media_type": media_type,
        "role": role,
    }


def _write_checksum_evidence(path: Path, root: Path) -> None:
    try:
        lines = [
            f"{_sha256(root / relative_path)}  {relative_path}\n"
            for relative_path in CHECKSUM_EVIDENCE_PATHS
        ]
        path.write_text("".join(lines), encoding="utf-8")
    except OSError as error:
        raise ArtifactRunnerError(
            f"cannot write complete M4 checksum evidence: {error}"
        ) from error


def _validator_command(manifest_path: Path) -> list[str]:
    return [
        sys.executable,
        str(VALIDATOR_PATH),
        "--schema",
        str(SCHEMA_PATH),
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
        env=_artifact_environment(),
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise ArtifactRunnerError(f"artifact validator rejected the M4 run: {detail}")


def generate_m4_artifact(config: RunnerConfig) -> Path:
    if RUN_ID_PATTERN.fullmatch(config.run_id) is None:
        raise ArtifactRunnerError(f"invalid run id: {config.run_id!r}")
    trace_scale_contract = _trace_scale_contract_provenance()
    _require_m4_schedule_sealed()
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

    run_root = _preflight_run_root(config.output_root, config.run_id)
    output_root = run_root.parent
    _require_frozen_openfhe_prefix(config.openfhe_prefix)
    profile = _profile_record()
    inputs = [
        *(_repository_record(path) for path in REQUIRED_INPUTS),
        *_trace_input_records(data_root, 1),
    ]
    input_by_path = {record["path"]: record for record in inputs}
    git_state = _preflight_git()
    schema_binding = _contract_schema_binding(REPO_ROOT, git_state.head)
    preflight = _run_m4_preflight(
        executable.parent,
        executable,
        config.openfhe_prefix.resolve(),
    )
    executable = _resolve_m4_executable(DEFAULT_EXECUTABLE)
    executable_sha256 = _sha256(executable)
    source_build_provenance = _source_build_provenance(
        git_state.head,
        executable,
    )
    environment = _environment(config.openfhe_prefix)
    output_root.mkdir(parents=True, exist_ok=True)
    try:
        run_root.mkdir(exist_ok=False)
    except FileExistsError as error:
        raise ArtifactRunnerError(
            f"run directory already exists: {run_root}"
        ) from error

    created_run_root = True
    try:
        time_path = run_root / "time.log"
        time_path.write_bytes(b"")
        _require_build_configuration(
            preflight.build_configuration,
            executable.parent,
            executable,
            config.openfhe_prefix,
            "before warm-up",
        )
        _require_executable_hash(executable, executable_sha256, "before warm-up")
        _require_input_hashes(inputs, "before warm-up")
        _require_source_build_provenance(
            source_build_provenance,
            git_state.head,
            executable,
            "before warm-up",
        )
        warmup = _run_once(
            RunnerConfig(
                executable=executable,
                data_root=data_root,
                output_root=output_root,
                run_id=config.run_id,
                openfhe_prefix=config.openfhe_prefix,
            ),
            "warmup",
            time_path,
        )
        _require_executable_hash(executable, executable_sha256, "after warm-up")
        _require_input_hashes(inputs, "after warm-up")
        _require_build_configuration(
            preflight.build_configuration,
            executable.parent,
            executable,
            config.openfhe_prefix,
            "after warm-up",
        )
        _require_source_build_provenance(
            source_build_provenance,
            git_state.head,
            executable,
            "after warm-up",
        )
        samples: list[RunSample] = []
        for index in range(1, 6):
            _require_build_configuration(
                preflight.build_configuration,
                executable.parent,
                executable,
                config.openfhe_prefix,
                f"before measured run {index}",
            )
            _require_executable_hash(
                executable,
                executable_sha256,
                f"before measured run {index}",
            )
            _require_input_hashes(inputs, f"before measured run {index}")
            _require_source_build_provenance(
                source_build_provenance,
                git_state.head,
                executable,
                f"before measured run {index}",
            )
            samples.append(
                _run_once(
                    RunnerConfig(
                        executable=executable,
                        data_root=data_root,
                        output_root=output_root,
                        run_id=config.run_id,
                        openfhe_prefix=config.openfhe_prefix,
                    ),
                    f"measured-{index}",
                    time_path,
                )
            )
            _require_executable_hash(
                executable,
                executable_sha256,
                f"after measured run {index}",
            )
            _require_input_hashes(inputs, f"after measured run {index}")
            _require_build_configuration(
                preflight.build_configuration,
                executable.parent,
                executable,
                config.openfhe_prefix,
                f"after measured run {index}",
            )
            _require_source_build_provenance(
                source_build_provenance,
                git_state.head,
                executable,
                f"after measured run {index}",
            )
        if _contract_schema_binding(REPO_ROOT, git_state.head) != schema_binding:
            raise ArtifactRunnerError(
                "schema-v5 contract binding changed during homomorphic execution"
            )
        stdout_path = run_root / "stdout.log"
        stderr_path = run_root / "stderr.log"
        metrics_path = run_root / "metrics.csv"
        checksum_path = run_root / "SHA256SUMS"
        manifest_path = run_root / "manifest.json"
        all_samples = [warmup, *samples]
        _write_stdout(stdout_path, all_samples)
        _write_stderr(stderr_path, all_samples)
        _write_metrics(metrics_path, samples)

        executable_path = executable.relative_to(REPO_ROOT.resolve()).as_posix()
        validator_command = _validator_command(manifest_path)
        commands = [
            *git_state.commands,
            *preflight.commands,
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
            "schema_version": SCHEMA_VERSION,
            "schema_binding": schema_binding,
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
                "remote_url": git_state.remote_url,
                "remote_ref": REMOTE_REF,
                "remote_commit": git_state.remote_head,
            },
            "commands": commands,
            "build_configuration": preflight.build_configuration,
            "source_build_provenance": source_build_provenance,
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
                "feature_packed_trace_scale_contract": trace_scale_contract,
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
                    "post_bootstrap_mask": EXPECTED_POST_BOOTSTRAP_MASK,
                    "internal_checkpoint_metadata": EXPECTED_INTERNAL_CHECKPOINTS,
                    "required_checkpoints": list(CHECKPOINT_NAMES),
                    "required_ctest_contracts": list(M4_V5_REQUIRED_CTEST_CONTRACTS),
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
                "decision": "PASS_M4_V5_SERVER_ONLY_OPENFHE_ENCODER_LAYER",
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
                _artifact_record(run_root, path, role, media_type)
                for path, (role, media_type) in MANIFEST_ARTIFACTS.items()
            ],
            "claim_boundary": list(CLAIM_BOUNDARY),
            "verdict": "GO",
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        _write_checksum_evidence(checksum_path, run_root)
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
    try:
        run_id = arguments.run_id or (
            f"{datetime.now().astimezone().strftime('%Y%m%dT%H%M%S%z')}"
            "-m4"
        )
        run_root = generate_m4_artifact(
            RunnerConfig(
                executable=arguments.executable,
                data_root=arguments.data_root,
                output_root=arguments.output_root,
                run_id=run_id,
            )
        )
        manifest = _load_json(run_root / "manifest.json")
        git_sha_hint = str(manifest["git"]["local_commit"])[:7]
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
