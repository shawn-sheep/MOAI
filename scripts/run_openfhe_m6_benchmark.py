#!/usr/bin/env python3
"""Generate the fail-closed MOAI OpenFHE M6 CPU benchmark artifact.

The runner accepts only a clean, pushed ``refactor/openfhe-cpu`` commit.  It
verifies the frozen M5 r27 prerequisite, performs a fresh fixed build and narrow
CTest preflight, then launches six independent 12-layer processes: one warm-up
and five measured samples.  M5 evidence code is imported but never modified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import re
import shlex
import statistics
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_openfhe_encoder12_artifact_v6 as m5  # noqa: E402


SCHEMA_PATH = REPO_ROOT / "docs" / "openfhe-m6-benchmark-schema-v1.json"
RUNNER_PATH = REPO_ROOT / "scripts" / "run_openfhe_m6_benchmark.py"
VALIDATOR_PATH = REPO_ROOT / "scripts" / "validate_openfhe_m6_benchmark.py"
DEFAULT_EXECUTABLE = REPO_ROOT / "build-openfhe" / "openfhe_encoder_12_layer_smoke"
DEFAULT_DATA_ROOT = REPO_ROOT / "data"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "openfhe"
DEFAULT_OPENFHE_PREFIX = Path("/home/shawnsheep/opt/openfhe_v1_5_1")
TIME_EXECUTABLE = Path("/usr/bin/time")
SYSTEM_CTEST = Path("/usr/bin/ctest")
SYSTEM_GIT = Path("/usr/bin/git")

SCHEMA_ID = "https://local.moai/openfhe-m6-benchmark-schema-v1.json"
SCHEMA_VERSION = 1
RUNNER_ID = "moai.openfhe.m6.benchmark-runner.v1"
VALIDATOR_ID = "moai.openfhe.m6.benchmark-validator.v1"
BRANCH = "refactor/openfhe-cpu"
TRACKING_REF = "refs/remotes/origin/refactor/openfhe-cpu"
REMOTE_REF = "refs/heads/refactor/openfhe-cpu"
REMOTE_NAME = "origin"
REMOTE_URL = "https://github.com/shawn-sheep/MOAI.git"
PROFILE_ID = "paper_compat"
SECURITY_CLAIM = "none"
PROFILE_SHA256 = "94f30e628e21f02146ce7ed9820194eabba3820f6e1e17176a31f8c5acf8b0be"
PROFILE_WARNING = "Research reproduction parameters only. Do not claim 128-bit security."
SERIALIZATION_FORMAT = "openfhe_binary_archive_component_sum_v1"
LOGICAL_CPU_COUNT = 16
FORCED_THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "16",
    "OMP_DYNAMIC": "FALSE",
}
THREAD_ENVIRONMENT_PREFIXES = (
    "OMP_",
    "OPENBLAS_",
    "MKL_",
    "GOTO_",
    "BLIS_",
    "VECLIB_",
    "NUMEXPR_",
)

M5_RUN_ID = "20260801T224242+0900-m5-runnable-prototype-v6-534f582-r27"
M5_RELATIVE_PATH = f"results/openfhe/{M5_RUN_ID}"
M5_MANIFEST_SHA256 = "d77425290cb15f6265527f362f016ee51c277e3a8e89da74592b79c353b44d95"
M5_SHA256SUMS_SHA256 = "48abf4300188388499fff60dd2617a1009e546314537e320c4bb9c5cbb148fd8"
M5_DECISION = "PASS_M5_RUNNABLE_PROTOTYPE"
M5_VERDICT = "GO_PROTOTYPE"

WARMUP_COUNT = 1
MEASURED_COUNT = 5
TOKEN_COUNT = 5
ENCODER_LAYERS = 12
EXTERNAL_WALL_TOLERANCE_MS = 100.0
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+:-]{0,127}$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

TIMING_FIELDS = (
    "fixture_load_oracle",
    "setup_keygen",
    "client_encrypt",
    "server_online",
    "client_decrypt",
    "client_validate",
    "serialized_size_measurement",
    "online_batch",
    "end_to_end_batch",
)
SAMPLE_TIMING_FIELDS = (
    *TIMING_FIELDS,
    "online_batch_amortized_per_token",
    "end_to_end_amortized_per_token",
    "server_online_amortized_per_token",
)
COUNT_FIELDS = (
    "rotations",
    "ct_pt_multiplications",
    "ct_ct_multiplications",
    "explicit_rescale_requests",
    "chebyshev_evaluations",
    "estimated_polynomial_multiplications",
    "bootstraps",
    "bootstrap_iterations",
)
EXPECTED_COUNTS = {
    "rotations": 75600,
    "ct_pt_multiplications": 622675,
    "ct_ct_multiplications": 1140,
    "explicit_rescale_requests": 9775,
    "chebyshev_evaluations": 660,
    "estimated_polynomial_multiplications": 13800,
    "bootstraps": 355,
    "bootstrap_iterations": 710,
}
KEY_SIZE_FIELDS = (
    "context_bytes",
    "public_key_bytes",
    "private_key_bytes",
    "evaluation_multiplication_key_bytes",
    "evaluation_automorphism_key_bytes",
    "server_key_bundle_component_sum_bytes",
)
M6_CTEST_PATTERN = (
    "^(migration_contract|openfhe_server_trust_boundary|openfhe_profile_contract|"
    "openfhe_encoder_12_layer_preflight|openfhe_encoder_12_layer_crypto_preflight|"
    "openfhe_serialized_size_smoke|openfhe_m6_benchmark_schema_contract|"
    "openfhe_m6_benchmark_validator_contract|"
    "openfhe_m6_benchmark_runner_contract)$"
)
M6_CTEST_EXPECTED_TESTS = (
    "migration_contract",
    "openfhe_server_trust_boundary",
    "openfhe_profile_contract",
    "openfhe_encoder_12_layer_preflight",
    "openfhe_encoder_12_layer_crypto_preflight",
    "openfhe_serialized_size_smoke",
    "openfhe_m6_benchmark_schema_contract",
    "openfhe_m6_benchmark_validator_contract",
    "openfhe_m6_benchmark_runner_contract",
)
SCHEMA_BINDING_PATHS = (
    "docs/openfhe-m6-benchmark-schema-v1.json",
    "scripts/run_openfhe_m6_benchmark.py",
    "scripts/validate_openfhe_m6_benchmark.py",
    "docs/openfhe-m5-artifact-schema-v6.json",
    "scripts/run_openfhe_encoder12_artifact_v6.py",
    "scripts/validate_openfhe_m5_artifact_v6.py",
)
ARTIFACT_LAYOUT = (
    ("stdout.log", "stdout", "text/plain"),
    ("stderr.log", "stderr", "text/plain"),
    ("time.log", "resource_timing", "text/plain"),
    ("metrics.csv", "metrics", "text/csv"),
    ("SHA256SUMS", "checksum", "text/plain"),
)
CHECKSUM_PATHS = tuple(item[0] for item in ARTIFACT_LAYOUT[:-1])
CLAIM_BOUNDARY = (
    "OpenFHE CKKS CPU server-only 12-layer encoder trace replay benchmark only.",
    "paper_compat uses security_claim=none; no 128-bit security claim.",
    "Five fixed trace tokens are one packed batch, not five task-level samples.",
    "No tokenizer, classifier, task accuracy, or task-level end-to-end inference claim.",
    "No GPU, MOAI_GPU, Discrete CKKS, FBT, or QDQ claim.",
    "Legacy SEAL was not run and is noncomparable under the fixed benchmark contract.",
    "No SEAL speedup is computed or claimed.",
)


class BenchmarkRunnerError(RuntimeError):
    """Raised when any fail-closed M6 runner contract is violated."""


@dataclass(frozen=True)
class RunnerConfig:
    executable: Path = DEFAULT_EXECUTABLE
    data_root: Path = DEFAULT_DATA_ROOT
    output_root: Path = DEFAULT_OUTPUT_ROOT
    run_id: str = ""
    openfhe_prefix: Path = DEFAULT_OPENFHE_PREFIX


@dataclass(frozen=True)
class ExecutedSample:
    phase: str
    index: int
    sample: dict[str, Any]
    stdout: str
    stderr: str
    time_line: str
    external_wall_seconds: float
    external_peak_rss_bytes: int
    command: dict[str, Any]

    def manifest_record(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "index": self.index,
            "sample": self.sample,
            "external_wall_seconds": self.external_wall_seconds,
            "external_peak_rss_bytes": self.external_peak_rss_bytes,
        }


def _timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _progress(phase: str, index: int | None, status: str) -> None:
    suffix = "" if index is None else f" index={index}"
    print(
        f"M6_PROGRESS phase={phase}{suffix} status={status} at={_timestamp()}",
        file=sys.stderr,
        flush=True,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise BenchmarkRunnerError(f"cannot canonicalize JSON: {error}") from error


def _reject_json_constant(token: str) -> Any:
    raise ValueError(f"non-finite JSON constant is forbidden: {token}")


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key is forbidden: {key}")
        value[key] = child
    return value


def _strict_json_loads(text: str) -> Any:
    return json.loads(
        text,
        object_pairs_hook=_strict_json_object,
        parse_constant=_reject_json_constant,
    )


def _finite_number(value: Any, label: str, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BenchmarkRunnerError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise BenchmarkRunnerError(f"{label} must be finite and >= {minimum}")
    return result


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise BenchmarkRunnerError(f"{label} must be a positive integer")
    return value


def _require_exact_keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        actual = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise BenchmarkRunnerError(
            f"{label} keys differ: expected={sorted(expected)} actual={actual}"
        )
    return value


def _close(actual: float, expected: float) -> bool:
    return math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-6)


def _validate_benchmark_sample(value: Any, label: str = "sample") -> dict[str, Any]:
    required = {
        "test",
        "backend",
        "profile",
        "security_claim",
        "parameter_sha256",
        "benchmark_sample",
        "timing_claim",
        "latency_kind",
        "claim_scope",
        "execution_mode",
        "encoder_layers",
        "token_count",
        "chain_mode",
        "client_encrypt_calls",
        "plaintext_activation_resets",
        "server_layer_evaluations",
        "inter_layer_refreshes",
        "checkpoint_decryption_owner",
        "observer_present_during_server_online",
        "checkpoint_decryptions",
        "final_decryption_owner",
        "server_private_key_present",
        "server_decryptions",
        "server_plaintext_activations",
        "approximation_range_status",
        "timing_ms",
        "correctness",
        "serialized_sizes",
        "final_metadata",
        "operation_counts",
        "multiplicative_depth",
        "max_observed_level",
        "max_polynomial_depth",
        "peak_rss_bytes",
        "peak_rss_scope",
        "passed",
    }
    sample = _require_exact_keys(value, required, label)
    frozen = {
        "test": "openfhe_encoder_12_layer_benchmark_sample",
        "backend": "OpenFHE CKKS CPU",
        "profile": PROFILE_ID,
        "security_claim": SECURITY_CLAIM,
        "parameter_sha256": PROFILE_SHA256,
        "benchmark_sample": True,
        "timing_claim": True,
        "latency_kind": "benchmark",
        "claim_scope": "m6_12_layer_benchmark_sample",
        "execution_mode": "server-only",
        "encoder_layers": ENCODER_LAYERS,
        "token_count": TOKEN_COUNT,
        "chain_mode": "ciphertext_output_to_next_input",
        "client_encrypt_calls": 1,
        "plaintext_activation_resets": 0,
        "server_layer_evaluations": 12,
        "inter_layer_refreshes": 11,
        "checkpoint_decryption_owner": "client",
        "observer_present_during_server_online": False,
        "checkpoint_decryptions": 0,
        "final_decryption_owner": "client",
        "server_private_key_present": False,
        "server_decryptions": 0,
        "server_plaintext_activations": False,
        "approximation_range_status": (
            "prevalidated_by_bound_m5_artifact_not_observed_in_sample"
        ),
        "multiplicative_depth": 47,
        "max_observed_level": 45,
        "max_polynomial_depth": 10,
        "peak_rss_scope": "process_high_water_mark",
        "passed": True,
    }
    for key, expected in frozen.items():
        if sample[key] != expected or type(sample[key]) is not type(expected):
            raise BenchmarkRunnerError(f"{label}.{key} differs from {expected!r}")

    timing = _require_exact_keys(sample["timing_ms"], set(SAMPLE_TIMING_FIELDS), f"{label}.timing_ms")
    parsed_timing = {
        key: _finite_number(timing[key], f"{label}.timing_ms.{key}")
        for key in SAMPLE_TIMING_FIELDS
    }
    if not _close(
        parsed_timing["online_batch"],
        parsed_timing["client_encrypt"]
        + parsed_timing["server_online"]
        + parsed_timing["client_decrypt"],
    ):
        raise BenchmarkRunnerError(f"{label} online batch is not encrypt+server+decrypt")
    if not _close(
        parsed_timing["online_batch_amortized_per_token"],
        parsed_timing["online_batch"] / TOKEN_COUNT,
    ):
        raise BenchmarkRunnerError(f"{label} online amortization is not batch/5")
    if not _close(
        parsed_timing["end_to_end_amortized_per_token"],
        parsed_timing["end_to_end_batch"] / TOKEN_COUNT,
    ):
        raise BenchmarkRunnerError(f"{label} end-to-end amortization is not batch/5")
    if not _close(
        parsed_timing["server_online_amortized_per_token"],
        parsed_timing["server_online"] / TOKEN_COUNT,
    ):
        raise BenchmarkRunnerError(f"{label} server amortization is not batch/5")
    if not _close(
        parsed_timing["end_to_end_batch"],
        parsed_timing["setup_keygen"] + parsed_timing["online_batch"]
    ):
        raise BenchmarkRunnerError(
            f"{label} end-to-end batch is not setup/keygen plus online batch"
        )

    correctness = _require_exact_keys(
        sample["correctness"],
        {"relative_l2", "cosine", "max_absolute", "inactive_max_abs", "finite", "passed"},
        f"{label}.correctness",
    )
    relative_l2 = _finite_number(correctness["relative_l2"], f"{label}.relative_l2")
    cosine = _finite_number(correctness["cosine"], f"{label}.cosine", -1.0)
    inactive = _finite_number(correctness["inactive_max_abs"], f"{label}.inactive")
    _finite_number(correctness["max_absolute"], f"{label}.max_absolute")
    if (
        relative_l2 > 5e-2
        or cosine < 0.99
        or cosine > 1.0
        or inactive > 1e-3
        or correctness["finite"] is not True
        or correctness["passed"] is not True
    ):
        raise BenchmarkRunnerError(f"{label} failed the frozen correctness gate")

    sizes = _require_exact_keys(
        sample["serialized_sizes"],
        {"keys", "encrypted_input", "final_output"},
        f"{label}.serialized_sizes",
    )
    keys = _require_exact_keys(
        sizes["keys"],
        {"serialization_format", *KEY_SIZE_FIELDS},
        f"{label}.serialized_sizes.keys",
    )
    if keys["serialization_format"] != SERIALIZATION_FORMAT:
        raise BenchmarkRunnerError(f"{label} key serialization format drifted")
    for key in KEY_SIZE_FIELDS:
        _positive_int(keys[key], f"{label}.serialized_sizes.keys.{key}")
    server_sum = sum(
        keys[key]
        for key in (
            "context_bytes",
            "public_key_bytes",
            "evaluation_multiplication_key_bytes",
            "evaluation_automorphism_key_bytes",
        )
    )
    if keys["server_key_bundle_component_sum_bytes"] != server_sum:
        raise BenchmarkRunnerError(f"{label} server key-bundle component sum drifted")
    for tensor_name in ("encrypted_input", "final_output"):
        tensor = _require_exact_keys(
            sizes[tensor_name],
            {"serialization_format", "ciphertext_count", "ciphertext_component_sum_bytes"},
            f"{label}.serialized_sizes.{tensor_name}",
        )
        if tensor["serialization_format"] != SERIALIZATION_FORMAT:
            raise BenchmarkRunnerError(f"{label} {tensor_name} serialization format drifted")
        if tensor["ciphertext_count"] != 5 or type(tensor["ciphertext_count"]) is not int:
            raise BenchmarkRunnerError(f"{label} {tensor_name} must contain five ciphertexts")
        _positive_int(
            tensor["ciphertext_component_sum_bytes"],
            f"{label}.serialized_sizes.{tensor_name}.ciphertext_component_sum_bytes",
        )

    counts = _require_exact_keys(sample["operation_counts"], set(COUNT_FIELDS), f"{label}.operation_counts")
    if counts != EXPECTED_COUNTS:
        raise BenchmarkRunnerError(f"{label} operation counts drifted")
    metadata = _require_exact_keys(
        sample["final_metadata"],
        {"level", "noise_scale_degree", "remaining_levels", "scale_bits", "expected_scale_bits", "ciphertext_count"},
        f"{label}.final_metadata",
    )
    expected_metadata = {
        "level": 30,
        "noise_scale_degree": 2,
        "remaining_levels": 16,
        "expected_scale_bits": 100,
        "ciphertext_count": 5,
    }
    for key, expected in expected_metadata.items():
        if metadata[key] != expected or type(metadata[key]) is not int:
            raise BenchmarkRunnerError(f"{label}.final_metadata.{key} drifted")
    scale_bits = _finite_number(metadata["scale_bits"], f"{label}.scale_bits")
    if not 99.99 <= scale_bits <= 100.01:
        raise BenchmarkRunnerError(f"{label} final scale bits drifted")
    _positive_int(sample["peak_rss_bytes"], f"{label}.peak_rss_bytes")
    return sample


def _extract_unique_sample(stdout: str, label: str) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        stripped = line.strip()
        if not stripped.startswith("{"):
            continue
        try:
            value = _strict_json_loads(stripped)
        except (json.JSONDecodeError, ValueError) as error:
            raise BenchmarkRunnerError(f"{label} emitted malformed JSON: {error}") from error
        if isinstance(value, dict) and value.get("test") == "openfhe_encoder_12_layer_benchmark_sample":
            records.append(value)
    if len(records) != 1:
        raise BenchmarkRunnerError(f"{label} must emit exactly one benchmark sample JSON, got {len(records)}")
    return _validate_benchmark_sample(records[0], label)


def _summary(values: list[float]) -> dict[str, Any]:
    if len(values) != MEASURED_COUNT or any(not math.isfinite(value) or value < 0 for value in values):
        raise BenchmarkRunnerError("statistics require exactly five finite nonnegative samples")
    median = float(statistics.median(values))
    return {
        "samples": list(values),
        "median": median,
        "mad": float(statistics.median(abs(value - median) for value in values)),
        "minimum": float(min(values)),
        "maximum": float(max(values)),
    }


def _timing_summary(values: list[float]) -> dict[str, Any]:
    return {
        "batch": _summary(values),
        "amortized_per_token": _summary([value / TOKEN_COUNT for value in values]),
    }


def _metrics(measured: list[ExecutedSample]) -> dict[str, Any]:
    if len(measured) != MEASURED_COUNT:
        raise BenchmarkRunnerError("M6 requires exactly five measured samples")
    samples = [item.sample for item in measured]
    sizes = samples[0]["serialized_sizes"]
    counts = samples[0]["operation_counts"]
    if any(item["serialized_sizes"] != sizes for item in samples[1:]):
        raise BenchmarkRunnerError("five measured serialized-size records differ")
    if any(item["operation_counts"] != counts for item in samples[1:]):
        raise BenchmarkRunnerError("five measured operation-count records differ")
    timing = {
        key: _timing_summary([float(sample["timing_ms"][key]) for sample in samples])
        for key in TIMING_FIELDS
    }
    correctness = {
        key: _summary([float(sample["correctness"][key]) for sample in samples])
        for key in ("relative_l2", "cosine", "max_absolute", "inactive_max_abs")
    }
    correctness.update({"all_finite": True, "all_passed": True})
    return {
        "measured_count": MEASURED_COUNT,
        "token_count": TOKEN_COUNT,
        "timing_ms": timing,
        "external_wall_seconds": _timing_summary(
            [item.external_wall_seconds for item in measured]
        ),
        "peak_rss_bytes": _summary(
            [float(item.external_peak_rss_bytes) for item in measured]
        ),
        "serialized_sizes": sizes,
        "operation_counts": counts,
        "correctness": correctness,
    }


def _artifact_environment() -> dict[str, str]:
    environment = os.environ.copy()
    cleared = set(m5.CONFIGURE_ENV_UNSET)
    cleared.update(
        name
        for name in environment
        if name.startswith(THREAD_ENVIRONMENT_PREFIXES)
    )
    for name in cleared:
        environment.pop(name, None)
    environment.update(m5.FORCED_SUBPROCESS_ENVIRONMENT)
    environment.update(FORCED_THREAD_ENVIRONMENT)
    return environment


def _install_environment_contract() -> tuple[str, ...]:
    if os.cpu_count() != LOGICAL_CPU_COUNT:
        raise BenchmarkRunnerError(
            f"M6 benchmark host must expose {LOGICAL_CPU_COUNT} logical CPUs, "
            f"got {os.cpu_count()}"
        )
    cleared = tuple(
        sorted(
            {
                *m5.CONFIGURE_ENV_UNSET,
                *(
                    name
                    for name in os.environ
                    if name.startswith(THREAD_ENVIRONMENT_PREFIXES)
                ),
            }
        )
    )
    m5.CONFIGURE_ENV_UNSET = cleared
    m5.FORCED_SUBPROCESS_ENVIRONMENT = {
        "LANG": "C",
        "LC_ALL": "C",
        **FORCED_THREAD_ENVIRONMENT,
    }
    return cleared


def _command_record(argv: list[str], returncode: int, started: str | None = None, finished: str | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {"argv": argv, "cwd": str(REPO_ROOT), "returncode": returncode}
    if started is not None and finished is not None:
        record.update({"started_at": started, "finished_at": finished})
    return record


def _convert_m5_command(record: dict[str, Any]) -> dict[str, Any]:
    command = record.get("command")
    if not isinstance(command, str):
        raise BenchmarkRunnerError("M5 command transcript lacks a command string")
    converted = _command_record(shlex.split(command), int(record.get("exit_code", -1)))
    if "started_at" in record and "finished_at" in record:
        converted.update({"started_at": record["started_at"], "finished_at": record["finished_at"]})
    return converted


def _require_m5_prerequisite() -> dict[str, Any]:
    root = REPO_ROOT / M5_RELATIVE_PATH
    manifest_path = root / "manifest.json"
    checksum_path = root / "SHA256SUMS"
    if root.is_symlink() or not root.is_dir() or root.resolve() != root:
        raise BenchmarkRunnerError(f"fixed M5 r27 artifact is missing: {root}")
    if _sha256(manifest_path) != M5_MANIFEST_SHA256 or _sha256(checksum_path) != M5_SHA256SUMS_SHA256:
        raise BenchmarkRunnerError("fixed M5 r27 manifest or SHA256SUMS hash drifted")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise BenchmarkRunnerError(f"cannot read fixed M5 r27 manifest: {error}") from error
    if (
        manifest.get("run_id") != M5_RUN_ID
        or manifest.get("schema_version") != 6
        or manifest.get("gate", {}).get("decision") != M5_DECISION
        or manifest.get("gate", {}).get("passed") is not True
        or manifest.get("verdict") != M5_VERDICT
    ):
        raise BenchmarkRunnerError("fixed M5 r27 semantic prerequisite drifted")
    lines = checksum_path.read_text(encoding="ascii").splitlines()
    if len(lines) != 2:
        raise BenchmarkRunnerError("fixed M5 r27 SHA256SUMS must contain two entries")
    seen: set[str] = set()
    for line in lines:
        match = re.fullmatch(r"([0-9a-f]{64})  ([A-Za-z0-9_.-]+)", line)
        if match is None:
            raise BenchmarkRunnerError("fixed M5 r27 SHA256SUMS is malformed")
        expected, name = match.groups()
        if name not in {"stdout.log", "metrics.csv"} or name in seen:
            raise BenchmarkRunnerError("fixed M5 r27 checksum paths drifted")
        path = root / name
        if path.resolve().parent != root or _sha256(path) != expected:
            raise BenchmarkRunnerError(f"fixed M5 r27 checksum failed: {name}")
        seen.add(name)
    return {
        "run_id": M5_RUN_ID,
        "relative_path": M5_RELATIVE_PATH,
        "manifest_sha256": M5_MANIFEST_SHA256,
        "sha256sums_sha256": M5_SHA256SUMS_SHA256,
        "decision": M5_DECISION,
        "verdict": M5_VERDICT,
        "validated": True,
    }


def _git_blob_bytes(head: str, relative_path: str) -> bytes:
    completed = subprocess.run(
        [str(SYSTEM_GIT), "show", f"{head}:{relative_path}"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        env=m5._git_environment(),
        stdin=subprocess.DEVNULL,
    )
    if completed.returncode != 0:
        raise BenchmarkRunnerError(f"M6 binding file is not committed at HEAD: {relative_path}")
    return completed.stdout


def _schema_binding(head: str) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for relative_path in SCHEMA_BINDING_PATHS:
        path = REPO_ROOT / relative_path
        if path.is_symlink() or not path.is_file() or path.resolve() != path:
            raise BenchmarkRunnerError(f"M6 binding file must be canonical: {relative_path}")
        contents = path.read_bytes()
        blob = _git_blob_bytes(head, relative_path)
        if contents != blob:
            raise BenchmarkRunnerError(f"M6 binding working bytes differ from HEAD: {relative_path}")
        records.append(
            {
                "path": relative_path,
                "bytes": len(contents),
                "sha256": hashlib.sha256(contents).hexdigest(),
                "git_blob_sha256": hashlib.sha256(blob).hexdigest(),
            }
        )
    return {"canonicalization": "git-blob-and-working-tree-sha256-v1", "files": records}


def _require_m6_build_configuration(
    expected: dict[str, object],
    executable: Path,
    openfhe_prefix: Path,
    phase: str,
) -> None:
    m5._require_build_configuration(
        expected,
        executable.parent,
        executable,
        openfhe_prefix,
        phase,
    )
    entries = m5._parse_cmake_cache(executable.parent / "CMakeCache.txt")
    if entries.get("MOAI_ENABLE_SEAL_REFERENCE") != ("BOOL", "OFF"):
        raise BenchmarkRunnerError(
            f"M6 requires default MOAI_ENABLE_SEAL_REFERENCE=OFF {phase}"
        )


def _run_m6_preflight(
    executable: Path,
    openfhe_prefix: Path,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    preflight = m5._run_m5_preflight(executable.parent, executable, openfhe_prefix)
    _require_m6_build_configuration(
        preflight.build_configuration,
        executable,
        openfhe_prefix,
        "before M6 narrow CTest",
    )
    command = [
        str(SYSTEM_CTEST),
        "--test-dir",
        str(executable.parent),
        "--output-on-failure",
        "--verbose",
        "--no-tests=error",
        "-R",
        M6_CTEST_PATTERN,
    ]
    record, stdout = m5._run_checked_command(command, "M6 narrow CTest")
    m5._require_ctest_passed_tests(stdout, M6_CTEST_EXPECTED_TESTS, "M6 narrow CTest")
    _require_m6_build_configuration(
        preflight.build_configuration,
        executable,
        openfhe_prefix,
        "after M6 narrow CTest",
    )
    return preflight, _convert_m5_command(record), {
        "fresh_configure": True,
        "clean_build": True,
        "narrow_ctest": True,
        "passed_tests": list(M6_CTEST_EXPECTED_TESTS),
        "schedule_preflight": preflight.schedule_preflight,
        "crypto_preflight": preflight.crypto_preflight,
    }


def _runtime_command(config: RunnerConfig, resource_path: Path, phase: str, index: int) -> list[str]:
    if phase not in {"warmup", "measured"} or index < 0 or index > MEASURED_COUNT:
        raise BenchmarkRunnerError("invalid M6 sample phase/index")
    return [
        str(TIME_EXECUTABLE),
        "--format",
        f'{{"phase":"{phase}","index":{index},"elapsed_seconds":%e,"max_rss_kib":%M,"exit_status":%x}}',
        "--output",
        str(resource_path),
        str(config.executable),
        "--data-root",
        str(config.data_root),
        "--benchmark-sample",
    ]


def _parse_time_line(text: str, phase: str, index: int) -> tuple[str, float, int]:
    lines = text.splitlines()
    if len(lines) != 1:
        raise BenchmarkRunnerError("GNU time must emit exactly one line per sample")
    try:
        value = _strict_json_loads(lines[0])
    except (json.JSONDecodeError, ValueError) as error:
        raise BenchmarkRunnerError(f"malformed GNU time JSON: {error}") from error
    if set(value) != {"phase", "index", "elapsed_seconds", "max_rss_kib", "exit_status"}:
        raise BenchmarkRunnerError("GNU time JSON keys drifted")
    if value["phase"] != phase or value["index"] != index or value["exit_status"] != 0:
        raise BenchmarkRunnerError("GNU time phase/index/status drifted")
    elapsed = _finite_number(value["elapsed_seconds"], "GNU time elapsed", 0.000001)
    rss_bytes = _positive_int(value["max_rss_kib"], "GNU time max RSS KiB") * 1024
    return lines[0], elapsed, rss_bytes


def _phase_raw_paths(staging: Path, phase: str, index: int) -> tuple[Path, Path, Path]:
    prefix = staging / f".raw-{phase}-{index}"
    return (
        Path(f"{prefix}.stdout.log"),
        Path(f"{prefix}.stderr.log"),
        Path(f"{prefix}.time.log"),
    )


def _remove_phase_raw(staging: Path, phase: str, index: int) -> None:
    for path in _phase_raw_paths(staging, phase, index):
        path.unlink(missing_ok=True)


def _run_sample(config: RunnerConfig, staging: Path, phase: str, index: int) -> ExecutedSample:
    stdout_path, stderr_path, resource_path = _phase_raw_paths(staging, phase, index)
    command = _runtime_command(config, resource_path, phase, index)
    started = _timestamp()
    # Stream the active multi-hour sample directly into staging.  If the runner
    # is interrupted, the outer failure handler can preserve partial evidence
    # instead of losing everything that the child emitted before termination.
    with stdout_path.open("wb") as stdout_handle, stderr_path.open("wb") as stderr_handle:
        try:
            completed = subprocess.run(
                command,
                cwd=REPO_ROOT,
                check=False,
                stdout=stdout_handle,
                stderr=stderr_handle,
                env=_artifact_environment(),
                stdin=subprocess.DEVNULL,
            )
        finally:
            for handle in (stdout_handle, stderr_handle):
                handle.flush()
                os.fsync(handle.fileno())
    finished = _timestamp()
    try:
        stdout = stdout_path.read_text(encoding="utf-8")
        stderr = stderr_path.read_text(encoding="utf-8")
        time_text = resource_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise BenchmarkRunnerError(
            f"cannot read M6 {phase} {index} raw evidence: {error}"
        ) from error
    if completed.returncode != 0:
        detail = stderr.strip() or stdout.strip()
        raise BenchmarkRunnerError(
            f"M6 {phase} {index} exited with {completed.returncode}: {detail[-4000:]}"
        )
    time_line, external_wall, external_rss = _parse_time_line(time_text, phase, index)
    sample = _extract_unique_sample(stdout, f"M6 {phase} {index}")
    if external_rss < sample["peak_rss_bytes"]:
        raise BenchmarkRunnerError("GNU time peak RSS is below the in-process high-water mark")
    if (
        external_wall * 1000.0 + EXTERNAL_WALL_TOLERANCE_MS
        < sample["timing_ms"]["end_to_end_batch"]
    ):
        raise BenchmarkRunnerError("external wall time is below the in-process end-to-end time")
    return ExecutedSample(
        phase=phase,
        index=index,
        sample=sample,
        stdout=stdout,
        stderr=stderr,
        time_line=time_line,
        external_wall_seconds=external_wall,
        external_peak_rss_bytes=external_rss,
        command=_command_record(command, completed.returncode, started, finished),
    )


def _log_blocks(samples: list[ExecutedSample], attribute: str) -> str:
    chunks: list[str] = []
    for item in samples:
        chunks.append(f"=== phase={item.phase} index={item.index} ===\n")
        value = getattr(item, attribute)
        chunks.append(value)
        if value and not value.endswith("\n"):
            chunks.append("\n")
    return "".join(chunks)


def _time_log(samples: list[ExecutedSample]) -> str:
    return "\n".join(item.time_line for item in samples) + "\n"


def _refresh_partial_logs(staging: Path, samples: list[ExecutedSample]) -> None:
    if not samples:
        return
    _atomic_write(staging / "stdout.log", _log_blocks(samples, "stdout").encode("utf-8"))
    _atomic_write(staging / "stderr.log", _log_blocks(samples, "stderr").encode("utf-8"))
    _atomic_write(staging / "time.log", _time_log(samples).encode("utf-8"))


CSV_FIELDS = (
    "sample_index",
    *[f"{name}_ms" for name in TIMING_FIELDS],
    "external_wall_seconds",
    "external_peak_rss_bytes",
    "relative_l2",
    "cosine",
    "max_absolute",
    "inactive_max_abs",
    *KEY_SIZE_FIELDS,
    "input_ciphertext_bytes",
    "final_ciphertext_bytes",
    *COUNT_FIELDS,
    "multiplicative_depth",
    "max_observed_level",
    "max_polynomial_depth",
)


def _metrics_csv(measured: list[ExecutedSample]) -> str:
    from io import StringIO

    output = StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    for item in measured:
        sample = item.sample
        row: dict[str, Any] = {"sample_index": item.index}
        row.update({f"{key}_ms": sample["timing_ms"][key] for key in TIMING_FIELDS})
        row.update(
            {
                "external_wall_seconds": item.external_wall_seconds,
                "external_peak_rss_bytes": item.external_peak_rss_bytes,
                **{
                    key: sample["correctness"][key]
                    for key in ("relative_l2", "cosine", "max_absolute", "inactive_max_abs")
                },
                **{key: sample["serialized_sizes"]["keys"][key] for key in KEY_SIZE_FIELDS},
                "input_ciphertext_bytes": sample["serialized_sizes"]["encrypted_input"]["ciphertext_component_sum_bytes"],
                "final_ciphertext_bytes": sample["serialized_sizes"]["final_output"]["ciphertext_component_sum_bytes"],
                **sample["operation_counts"],
                "multiplicative_depth": sample["multiplicative_depth"],
                "max_observed_level": sample["max_observed_level"],
                "max_polynomial_depth": sample["max_polynomial_depth"],
            }
        )
        writer.writerow(row)
    return output.getvalue()


def _atomic_write(path: Path, data: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _preserve_failure_artifact(
    source: Path,
    failed_root: Path,
    error: BaseException,
    completed_samples: list[ExecutedSample],
) -> None:
    if failed_root.exists() or failed_root.is_symlink():
        raise BenchmarkRunnerError(f"M6 failure artifact already exists: {failed_root}")
    detail = " ".join(str(error).splitlines())[-4000:]
    completed = ",".join(f"{item.phase}-{item.index}" for item in completed_samples)
    evidence = (
        "artifact_eligible=false\n"
        "status=FAILED_INCOMPLETE\n"
        f"failed_at={_timestamp()}\n"
        f"error_type={type(error).__name__}\n"
        f"completed_samples={completed}\n"
        f"error={detail}\n"
    )
    _atomic_write(source / "failure.txt", evidence.encode("utf-8"))
    _fsync_directory(source)
    os.replace(source, failed_root)
    _fsync_directory(failed_root.parent)


def _artifact_record(root: Path, name: str, role: str, media_type: str) -> dict[str, Any]:
    path = root / name
    return {
        "path": name,
        "role": role,
        "media_type": media_type,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _input_records(data_root: Path) -> list[dict[str, Any]]:
    trace_inputs, _ = m5._trace_input_records(data_root)
    records = sorted(
        [*(m5._repository_record(path) for path in m5.REQUIRED_INPUTS), *trace_inputs],
        key=lambda record: str(record["path"]),
    )
    if len(records) != m5.EXPECTED_INPUT_COUNT or len({item["path"] for item in records}) != len(records):
        raise BenchmarkRunnerError("M6 input provenance differs from the frozen M5 input set")
    return [
        {key: record[key] for key in ("path", "bytes", "sha256")}
        for record in records
    ]


def _seal_comparison() -> dict[str, Any]:
    return {
        "seal_reference": {
            "status": "not_run",
            "comparability": "noncomparable",
            "reason": "No fixed-parameter, fixed-packing, fixed-precision SEAL benchmark was run.",
            "speedup": None,
        }
    }


def _load_validator_module() -> Any:
    spec = importlib.util.spec_from_file_location("validate_openfhe_m6_benchmark", VALIDATOR_PATH)
    if spec is None or spec.loader is None:
        raise BenchmarkRunnerError(f"cannot import M6 validator: {VALIDATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def generate_artifact(config: RunnerConfig) -> Path:
    if RUN_ID_PATTERN.fullmatch(config.run_id) is None:
        raise BenchmarkRunnerError(f"invalid M6 run id: {config.run_id!r}")
    if config.executable.resolve() != DEFAULT_EXECUTABLE.resolve():
        raise BenchmarkRunnerError(f"M6 executable must be {DEFAULT_EXECUTABLE}")
    if config.data_root.resolve() != DEFAULT_DATA_ROOT.resolve():
        raise BenchmarkRunnerError(f"M6 data root must be {DEFAULT_DATA_ROOT}")
    if config.output_root.resolve() != DEFAULT_OUTPUT_ROOT.resolve():
        raise BenchmarkRunnerError(f"M6 output root must be {DEFAULT_OUTPUT_ROOT}")
    if config.openfhe_prefix.resolve() != DEFAULT_OPENFHE_PREFIX:
        raise BenchmarkRunnerError(f"M6 OpenFHE prefix must be {DEFAULT_OPENFHE_PREFIX}")
    if not TIME_EXECUTABLE.is_file() or not os.access(TIME_EXECUTABLE, os.X_OK):
        raise BenchmarkRunnerError(f"GNU time is unavailable: {TIME_EXECUTABLE}")

    started_at = _timestamp()
    cleared_thread_environment = _install_environment_contract()
    try:
        git_state = m5._preflight_git()
    except m5.ArtifactRunnerError as error:
        raise BenchmarkRunnerError(str(error).replace("M5", "M6")) from error
    prerequisite = _require_m5_prerequisite()
    binding = _schema_binding(git_state.head)
    output_root = config.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    final_root = output_root / config.run_id
    failed_root = output_root / f"{config.run_id}-failed"
    if final_root.exists() or final_root.is_symlink():
        raise BenchmarkRunnerError(f"M6 artifact already exists: {final_root}")
    if failed_root.exists() or failed_root.is_symlink():
        raise BenchmarkRunnerError(f"M6 failure artifact already exists: {failed_root}")
    staging = Path(tempfile.mkdtemp(prefix=f".{config.run_id}.staging-", dir=output_root))
    published = False
    all_samples: list[ExecutedSample] = []
    active_phase = "preflight"
    active_index: int | None = None
    try:
        _progress(active_phase, active_index, "begin")
        preflight, m6_ctest_command, preflight_record = _run_m6_preflight(
            DEFAULT_EXECUTABLE, config.openfhe_prefix
        )
        executable = m5._resolve_m5_executable(DEFAULT_EXECUTABLE)
        executable_hash = _sha256(executable)
        inputs = _input_records(config.data_root)
        environment = {
            **m5._environment(config.openfhe_prefix),
            "logical_cpu_count": LOGICAL_CPU_COUNT,
            "threading": {
                "forced": dict(FORCED_THREAD_ENVIRONMENT),
                "cleared_inherited_variables": list(cleared_thread_environment),
                "applies_to": (
                    "configure, build, CTest, warm-up, measured samples, validator"
                ),
            },
        }
        profile = {
            "id": PROFILE_ID,
            "security_claim": SECURITY_CLAIM,
            "effective_profile_sha256": PROFILE_SHA256,
            "warning": PROFILE_WARNING,
        }
        _progress(active_phase, active_index, "success")

        for phase, index in (("warmup", 0), *( ("measured", i) for i in range(1, 6) )):
            active_phase = phase
            active_index = index
            _progress(active_phase, active_index, "begin")
            _require_m6_build_configuration(
                preflight.build_configuration,
                executable,
                config.openfhe_prefix,
                f"before M6 {phase} {index}",
            )
            m5._require_executable_hash(executable, executable_hash, f"before M6 {phase} {index}")
            m5._require_input_hashes(inputs, f"before M6 {phase} {index}")
            item = _run_sample(config, staging, phase, index)
            m5._require_executable_hash(executable, executable_hash, f"after M6 {phase} {index}")
            m5._require_input_hashes(inputs, f"after M6 {phase} {index}")
            _require_m6_build_configuration(
                preflight.build_configuration,
                executable,
                config.openfhe_prefix,
                f"after M6 {phase} {index}",
            )
            all_samples.append(item)
            _refresh_partial_logs(staging, all_samples)
            _remove_phase_raw(staging, phase, index)
            _progress(active_phase, active_index, "success")
        warmup = all_samples[0]
        measured = all_samples[1:]
        metrics = _metrics(measured)

        if list(staging.glob(".raw-*")):
            raise BenchmarkRunnerError("temporary per-phase raw files remain before sealing")
        active_phase = "seal"
        active_index = None
        _progress(active_phase, active_index, "begin")
        _atomic_write(staging / "metrics.csv", _metrics_csv(measured).encode("utf-8"))
        checksums = "".join(f"{_sha256(staging / name)}  {name}\n" for name in CHECKSUM_PATHS)
        _atomic_write(staging / "SHA256SUMS", checksums.encode("ascii"))

        validator_argv = [
            sys.executable,
            str(VALIDATOR_PATH),
            "--schema",
            str(SCHEMA_PATH),
            "--manifest",
            str(final_root / "manifest.json"),
            "--verify-git",
        ]
        commands = [
            *(_convert_m5_command(record) for record in git_state.commands),
            *(_convert_m5_command(record) for record in preflight.commands),
            m6_ctest_command,
            *(item.command for item in all_samples),
            _command_record(validator_argv, 0),
        ]
        if len(commands) != 18:
            raise BenchmarkRunnerError(
                f"M6 command contract must contain 18 records, got {len(commands)}"
            )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "schema_binding": binding,
            "run_id": config.run_id,
            "milestone": "M6",
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
                "remote_url": git_state.remote_url,
                "remote_ref": REMOTE_REF,
                "remote_commit": git_state.remote_head,
            },
            "m5_prerequisite": prerequisite,
            "profile": profile,
            "build_configuration": preflight.build_configuration,
            "workload": {
                "backend": "OpenFHE CKKS CPU",
                "scope": "server-only five-token BERT-base 12-layer encoder trace replay",
                "executable_path": "build-openfhe/openfhe_encoder_12_layer_smoke",
                "executable_sha256": executable_hash,
                "executable_bytes": executable.stat().st_size,
                "warmup_count": WARMUP_COUNT,
                "measured_count": MEASURED_COUNT,
                "independent_processes": True,
                "token_count": TOKEN_COUNT,
                "encoder_layers": ENCODER_LAYERS,
                "timing_kind": "benchmark",
            },
            "preflight": preflight_record,
            "commands": commands,
            "environment": environment,
            "inputs": inputs,
            "samples": {
                "warmup": warmup.manifest_record(),
                "measured": [item.manifest_record() for item in measured],
            },
            "metrics": metrics,
            "comparison": _seal_comparison(),
            "gate": {
                "passed": True,
                "decision": "PASS_M6_OPENFHE_BENCHMARK",
                "checks": {
                    "git": "PASS",
                    "m5_prerequisite": "PASS",
                    "build": "PASS",
                    "narrow_ctest": "PASS",
                    "repeat_contract": "PASS",
                    "correctness": "PASS",
                    "bytes_and_counts": "PASS",
                    "statistics": "PASS",
                    "artifact_integrity": "PASS",
                    "seal_comparability": "PASS_NONCOMPARABLE",
                },
            },
            "artifacts": [
                _artifact_record(staging, name, role, media_type)
                for name, role, media_type in ARTIFACT_LAYOUT
            ],
            "claim_boundary": list(CLAIM_BOUNDARY),
            "verdict": "GO_M6_OPENFHE_BENCHMARK",
        }
        _atomic_write(
            staging / "manifest.json",
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n",
        )

        validator = _load_validator_module()
        schema = validator.validate_schema(validator.load_json(SCHEMA_PATH))
        validator.validate_bundle(
            staging / "manifest.json",
            validator.load_json(staging / "manifest.json"),
            schema,
            verify_git=True,
            expected_run_id=config.run_id,
            allow_staging_root=True,
        )
        _fsync_directory(staging)
        os.replace(staging, final_root)
        published = True
        _fsync_directory(output_root)

        active_phase = "independent-validator"
        active_index = None
        _progress(active_phase, active_index, "begin")
        completed = subprocess.run(
            validator_argv,
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
            env=_artifact_environment(),
            stdin=subprocess.DEVNULL,
        )
        if completed.returncode != 0:
            raise BenchmarkRunnerError(
                "published M6 artifact failed independent validation: "
                f"{(completed.stderr or completed.stdout)[-4000:]}"
            )
        _progress(active_phase, active_index, "success")
        return final_root
    except BaseException as error:
        _progress(active_phase, active_index, "failed")
        source = final_root if published else staging
        try:
            _preserve_failure_artifact(source, failed_root, error, all_samples)
        except BaseException as preservation_error:
            raise BenchmarkRunnerError(
                "M6 failed and its incomplete evidence could not be atomically preserved: "
                f"original={error!r} preservation={preservation_error!r}"
            ) from preservation_error
        print(
            f"M6_FAILURE_ARTIFACT path={failed_root} artifact_eligible=false",
            file=sys.stderr,
            flush=True,
        )
        raise


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, default=DEFAULT_EXECUTABLE)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--openfhe-prefix", type=Path, default=DEFAULT_OPENFHE_PREFIX)
    parser.add_argument("--schema", type=Path, default=SCHEMA_PATH)
    parser.add_argument("--run-id")
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    try:
        if arguments.schema.resolve() != SCHEMA_PATH.resolve():
            raise BenchmarkRunnerError(f"M6 runner requires --schema {SCHEMA_PATH}")
        short = subprocess.run(
            [str(SYSTEM_GIT), "rev-parse", "--short=7", "HEAD"],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
            env=m5._git_environment(),
            stdin=subprocess.DEVNULL,
        ).stdout.strip() or "unknown"
        run_id = arguments.run_id or (
            f"{datetime.now().astimezone().strftime('%Y%m%dT%H%M%S%z')}-m6-benchmark-{short}"
        )
        root = generate_artifact(
            RunnerConfig(
                executable=arguments.executable,
                data_root=arguments.data_root,
                output_root=arguments.output_root,
                run_id=run_id,
                openfhe_prefix=arguments.openfhe_prefix,
            )
        )
    except (BenchmarkRunnerError, m5.ArtifactRunnerError, OSError, ValueError) as error:
        print(f"run_openfhe_m6_benchmark failed: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "artifact_root": str(root),
                "manifest": str(root / "manifest.json"),
                "milestone": "M6",
                "run_id": root.name,
                "validated": True,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
