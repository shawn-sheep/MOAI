#!/usr/bin/env python3
"""Independently validate MOAI OpenFHE M6 benchmark schema and artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import validate_openfhe_m5_artifact_v6 as m5_validator  # noqa: E402


DEFAULT_SCHEMA = REPO_ROOT / "docs" / "openfhe-m6-benchmark-schema-v1.json"
OUTPUT_ROOT = REPO_ROOT / "results" / "openfhe"
SYSTEM_GIT = Path("/usr/bin/git")
SCHEMA_URI = "https://json-schema.org/draft/2020-12/schema"
SCHEMA_ID = "https://local.moai/openfhe-m6-benchmark-schema-v1.json"
SCHEMA_TITLE = "MOAI OpenFHE M6 benchmark artifact v1"
SCHEMA_VERSION = 1
BRANCH = "refactor/openfhe-cpu"
TRACKING_REF = "refs/remotes/origin/refactor/openfhe-cpu"
REMOTE_REF = "refs/heads/refactor/openfhe-cpu"
REMOTE_URL = "https://github.com/shawn-sheep/MOAI.git"
PROFILE_SHA256 = "94f30e628e21f02146ce7ed9820194eabba3820f6e1e17176a31f8c5acf8b0be"
SERIALIZATION_FORMAT = "openfhe_binary_archive_component_sum_v1"
LOGICAL_CPU_COUNT = 16
FORCED_THREAD_ENVIRONMENT = {"OMP_NUM_THREADS": "16", "OMP_DYNAMIC": "FALSE"}
THREAD_ENVIRONMENT_PREFIXES = (
    "OMP_",
    "OPENBLAS_",
    "MKL_",
    "GOTO_",
    "BLIS_",
    "VECLIB_",
    "NUMEXPR_",
)
BASE_CONFIGURE_ENV_UNSET = tuple(m5_validator.CONFIGURE_ENV_UNSET)
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
M5_RUN_ID = "20260801T224242+0900-m5-runnable-prototype-v6-534f582-r27"
M5_RELATIVE_PATH = f"results/openfhe/{M5_RUN_ID}"
M5_MANIFEST_SHA256 = "d77425290cb15f6265527f362f016ee51c277e3a8e89da74592b79c353b44d95"
M5_SHA256SUMS_SHA256 = "48abf4300188388499fff60dd2617a1009e546314537e320c4bb9c5cbb148fd8"
M5_DECISION = "PASS_M5_RUNNABLE_PROTOTYPE"
M5_VERDICT = "GO_PROTOTYPE"
TOKEN_COUNT = 5
MEASURED_COUNT = 5
EXTERNAL_WALL_TOLERANCE_MS = 100.0
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
CLAIM_BOUNDARY = [
    "OpenFHE CKKS CPU server-only 12-layer encoder trace replay benchmark only.",
    "paper_compat uses security_claim=none; no 128-bit security claim.",
    "Five fixed trace tokens are one packed batch, not five task-level samples.",
    "No tokenizer, classifier, task accuracy, or task-level end-to-end inference claim.",
    "No GPU, MOAI_GPU, Discrete CKKS, FBT, or QDQ claim.",
    "Legacy SEAL was not run and is noncomparable under the fixed benchmark contract.",
    "No SEAL speedup is computed or claimed.",
]
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+:-]{0,127}$")
LOG_MARKER = re.compile(r"^=== phase=(warmup|measured) index=([0-5]) ===$")


class ValidationError(RuntimeError):
    """Raised for any schema, semantic, transcript, or hash violation."""


def load_json(path: Path) -> Any:
    try:
        return m5_validator.load_json(path)
    except m5_validator.ValidationError as error:
        raise ValidationError(str(error)) from error


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
        raise ValidationError(f"cannot canonicalize JSON: {error}") from error


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise ValidationError(f"cannot hash {path}: {error}") from error
    return digest.hexdigest()


def validate_schema(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValidationError("M6 schema root must be an object")
    if (
        value.get("$schema") != SCHEMA_URI
        or value.get("$id") != SCHEMA_ID
        or value.get("title") != SCHEMA_TITLE
        or value.get("type") != "object"
        or value.get("additionalProperties") is not False
    ):
        raise ValidationError("M6 schema identity or closed-root contract drifted")
    required = {
        "schema_version",
        "schema_binding",
        "run_id",
        "milestone",
        "started_at",
        "finished_at",
        "git",
        "m5_prerequisite",
        "profile",
        "build_configuration",
        "workload",
        "preflight",
        "commands",
        "environment",
        "inputs",
        "samples",
        "metrics",
        "comparison",
        "gate",
        "artifacts",
        "claim_boundary",
        "verdict",
    }
    if set(value.get("required", [])) != required:
        raise ValidationError("M6 schema required top-level fields drifted")
    properties = value.get("properties", {})
    if (
        properties.get("schema_version", {}).get("const") != SCHEMA_VERSION
        or properties.get("milestone", {}).get("const") != "M6"
        or properties.get("verdict", {}).get("const") != "GO_M6_OPENFHE_BENCHMARK"
    ):
        raise ValidationError("M6 schema milestone/version/verdict drifted")
    try:
        m5_validator._walk_schema(value, value, "$")
    except m5_validator.ValidationError as error:
        raise ValidationError(str(error)) from error
    return value


def _validate_instance(instance: Any, schema: dict[str, Any]) -> None:
    try:
        m5_validator.validate_instance(instance, schema, schema)
    except m5_validator.ValidationError as error:
        raise ValidationError(str(error)) from error


def _finite(value: Any, label: str, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ValidationError(f"{label} must be finite and >= {minimum}")
    return result


def _close(actual: float, expected: float) -> bool:
    return math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-6)


def _parse_timestamp(value: str, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError) as error:
        raise ValidationError(f"{label} is not an ISO-8601 timestamp") from error
    if parsed.tzinfo is None:
        raise ValidationError(f"{label} lacks a timezone")
    return parsed


def _summary(values: list[float]) -> dict[str, Any]:
    if len(values) != MEASURED_COUNT:
        raise ValidationError("statistics require exactly five measured values")
    parsed = [_finite(value, "statistic sample") for value in values]
    median = float(statistics.median(parsed))
    return {
        "samples": parsed,
        "median": median,
        "mad": float(statistics.median(abs(value - median) for value in parsed)),
        "minimum": min(parsed),
        "maximum": max(parsed),
    }


def _timing_summary(values: list[float]) -> dict[str, Any]:
    return {
        "batch": _summary(values),
        "amortized_per_token": _summary([value / TOKEN_COUNT for value in values]),
    }


def _validate_sample(sample: dict[str, Any], label: str) -> None:
    timing = sample["timing_ms"]
    if set(timing) != set(SAMPLE_TIMING_FIELDS):
        raise ValidationError(f"{label} timing fields drifted")
    parsed = {key: _finite(timing[key], f"{label}.{key}") for key in timing}
    if not _close(
        parsed["online_batch"],
        parsed["client_encrypt"]
        + parsed["server_online"]
        + parsed["client_decrypt"],
    ):
        raise ValidationError(f"{label} online batch is not encrypt+server+decrypt")
    if not _close(
        parsed["online_batch_amortized_per_token"],
        parsed["online_batch"] / TOKEN_COUNT,
    ):
        raise ValidationError(f"{label} online amortization differs from batch/5")
    if not _close(
        parsed["end_to_end_amortized_per_token"],
        parsed["end_to_end_batch"] / TOKEN_COUNT,
    ):
        raise ValidationError(f"{label} end-to-end amortization differs from batch/5")
    if not _close(
        parsed["server_online_amortized_per_token"],
        parsed["server_online"] / TOKEN_COUNT,
    ):
        raise ValidationError(f"{label} server amortization differs from batch/5")
    if not _close(
        parsed["end_to_end_batch"],
        parsed["setup_keygen"] + parsed["online_batch"],
    ):
        raise ValidationError(
            f"{label} end-to-end batch is not setup/keygen plus online batch"
        )
    correctness = sample["correctness"]
    if (
        _finite(correctness["relative_l2"], f"{label}.relative_l2") > 5e-2
        or _finite(correctness["cosine"], f"{label}.cosine", -1.0) < 0.99
        or _finite(correctness["inactive_max_abs"], f"{label}.inactive") > 1e-3
        or correctness["finite"] is not True
        or correctness["passed"] is not True
    ):
        raise ValidationError(f"{label} correctness gate failed")
    sizes = sample["serialized_sizes"]
    keys = sizes["keys"]
    if any(sizes[name]["serialization_format"] != SERIALIZATION_FORMAT for name in sizes):
        raise ValidationError(f"{label} serialization format drifted")
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
        raise ValidationError(f"{label} server key-bundle byte sum drifted")
    if sample["operation_counts"] != EXPECTED_COUNTS:
        raise ValidationError(f"{label} operation counts drifted")


def _expected_phases() -> list[tuple[str, int]]:
    return [("warmup", 0), *( ("measured", index) for index in range(1, 6) )]


def _manifest_samples(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    records = [manifest["samples"]["warmup"], *manifest["samples"]["measured"]]
    observed = [(record["phase"], record["index"]) for record in records]
    if observed != _expected_phases():
        raise ValidationError(f"sample repeat/order contract drifted: {observed}")
    for index, record in enumerate(records):
        _validate_sample(record["sample"], f"samples[{index}]")
        external_wall = _finite(record["external_wall_seconds"], f"samples[{index}].wall")
        external_rss = record["external_peak_rss_bytes"]
        if not isinstance(external_rss, int) or isinstance(external_rss, bool) or external_rss <= 0:
            raise ValidationError(f"samples[{index}] external RSS must be a positive integer")
        if external_rss < record["sample"]["peak_rss_bytes"]:
            raise ValidationError(f"samples[{index}] external RSS is below process RSS")
        if (
            external_wall * 1000.0 + EXTERNAL_WALL_TOLERANCE_MS
            < record["sample"]["timing_ms"]["end_to_end_batch"]
        ):
            raise ValidationError(f"samples[{index}] external wall is below end-to-end time")
    return records


def _parse_log_blocks(path: Path) -> list[tuple[str, int, str]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    except (OSError, UnicodeError) as error:
        raise ValidationError(f"cannot read {path.name}: {error}") from error
    blocks: list[tuple[str, int, str]] = []
    phase: str | None = None
    index: int | None = None
    contents: list[str] = []
    for line in lines:
        match = LOG_MARKER.fullmatch(line.rstrip("\r\n"))
        if match is not None:
            if phase is not None and index is not None:
                blocks.append((phase, index, "".join(contents)))
            phase, raw_index = match.groups()
            index = int(raw_index)
            contents = []
        else:
            if phase is None:
                raise ValidationError(f"{path.name} has bytes before its first sample marker")
            contents.append(line)
    if phase is not None and index is not None:
        blocks.append((phase, index, "".join(contents)))
    if [(phase, index) for phase, index, _ in blocks] != _expected_phases():
        raise ValidationError(f"{path.name} sample block order/count drifted")
    return blocks


def _sample_json_from_block(text: str, label: str) -> dict[str, Any]:
    matches: list[dict[str, Any]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("{"):
            continue
        try:
            value = _strict_json_loads(stripped)
        except (json.JSONDecodeError, ValueError) as error:
            raise ValidationError(f"{label} contains malformed JSON") from error
        if isinstance(value, dict) and value.get("test") == "openfhe_encoder_12_layer_benchmark_sample":
            matches.append(value)
    if len(matches) != 1:
        raise ValidationError(f"{label} must contain exactly one sample JSON")
    return matches[0]


def _verify_raw_logs(root: Path, records: list[dict[str, Any]]) -> None:
    stdout_blocks = _parse_log_blocks(root / "stdout.log")
    stderr_blocks = _parse_log_blocks(root / "stderr.log")
    if [(a, b) for a, b, _ in stderr_blocks] != _expected_phases():
        raise ValidationError("stderr.log phase order drifted")
    for manifest_record, (phase, index, text) in zip(records, stdout_blocks, strict=True):
        raw_sample = _sample_json_from_block(text, f"stdout {phase}-{index}")
        if canonical_json_bytes(raw_sample) != canonical_json_bytes(manifest_record["sample"]):
            raise ValidationError(f"stdout {phase}-{index} differs from manifest sample")
    try:
        time_lines = (root / "time.log").read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise ValidationError(f"cannot read time.log: {error}") from error
    if len(time_lines) != 6:
        raise ValidationError("time.log must contain exactly six records")
    for manifest_record, line, (phase, index) in zip(records, time_lines, _expected_phases(), strict=True):
        try:
            value = _strict_json_loads(line)
        except (json.JSONDecodeError, ValueError) as error:
            raise ValidationError("time.log contains malformed JSON") from error
        if (
            set(value) != {"phase", "index", "elapsed_seconds", "max_rss_kib", "exit_status"}
            or value["phase"] != phase
            or value["index"] != index
            or value["exit_status"] != 0
            or not _close(float(value["elapsed_seconds"]), manifest_record["external_wall_seconds"])
            or value["max_rss_kib"] * 1024 != manifest_record["external_peak_rss_bytes"]
        ):
            raise ValidationError(f"time.log {phase}-{index} differs from manifest")


def _verify_metrics(manifest: dict[str, Any], measured: list[dict[str, Any]]) -> None:
    samples = [record["sample"] for record in measured]
    first_sizes = samples[0]["serialized_sizes"]
    first_counts = samples[0]["operation_counts"]
    if any(sample["serialized_sizes"] != first_sizes for sample in samples[1:]):
        raise ValidationError("measured serialized-size records are not identical")
    if any(sample["operation_counts"] != first_counts for sample in samples[1:]):
        raise ValidationError("measured operation counts are not identical")
    expected_timing = {
        key: _timing_summary([float(sample["timing_ms"][key]) for sample in samples])
        for key in TIMING_FIELDS
    }
    expected_quality = {
        key: _summary([float(sample["correctness"][key]) for sample in samples])
        for key in ("relative_l2", "cosine", "max_absolute", "inactive_max_abs")
    }
    expected_quality.update({"all_finite": True, "all_passed": True})
    expected = {
        "measured_count": 5,
        "token_count": 5,
        "timing_ms": expected_timing,
        "external_wall_seconds": _timing_summary(
            [float(record["external_wall_seconds"]) for record in measured]
        ),
        "peak_rss_bytes": _summary(
            [float(record["external_peak_rss_bytes"]) for record in measured]
        ),
        "serialized_sizes": first_sizes,
        "operation_counts": first_counts,
        "correctness": expected_quality,
    }
    if canonical_json_bytes(manifest["metrics"]) != canonical_json_bytes(expected):
        raise ValidationError("manifest metrics/statistics differ from five measured samples")


def _verify_metrics_csv(root: Path, measured: list[dict[str, Any]]) -> None:
    try:
        with (root / "metrics.csv").open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != CSV_FIELDS:
                raise ValidationError("metrics.csv header drifted")
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as error:
        raise ValidationError(f"cannot parse metrics.csv: {error}") from error
    if len(rows) != 5:
        raise ValidationError("metrics.csv must contain exactly five measured rows")
    for manifest_record, row in zip(measured, rows, strict=True):
        sample = manifest_record["sample"]
        expected: dict[str, Any] = {"sample_index": manifest_record["index"]}
        expected.update({f"{key}_ms": sample["timing_ms"][key] for key in TIMING_FIELDS})
        expected.update(
            {
                "external_wall_seconds": manifest_record["external_wall_seconds"],
                "external_peak_rss_bytes": manifest_record["external_peak_rss_bytes"],
                **{key: sample["correctness"][key] for key in ("relative_l2", "cosine", "max_absolute", "inactive_max_abs")},
                **{key: sample["serialized_sizes"]["keys"][key] for key in KEY_SIZE_FIELDS},
                "input_ciphertext_bytes": sample["serialized_sizes"]["encrypted_input"]["ciphertext_component_sum_bytes"],
                "final_ciphertext_bytes": sample["serialized_sizes"]["final_output"]["ciphertext_component_sum_bytes"],
                **sample["operation_counts"],
                "multiplicative_depth": sample["multiplicative_depth"],
                "max_observed_level": sample["max_observed_level"],
                "max_polynomial_depth": sample["max_polynomial_depth"],
            }
        )
        for key, value in expected.items():
            if isinstance(value, int):
                if row[key] != str(value):
                    raise ValidationError(f"metrics.csv {key} differs from manifest")
            elif not _close(float(row[key]), float(value)):
                raise ValidationError(f"metrics.csv {key} differs from manifest")


def _confined_file(root: Path, relative: str) -> Path:
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValidationError(f"artifact path is not confined: {relative}")
    path = root / relative
    if path.is_symlink() or not path.is_file() or path.resolve().parent != root.resolve():
        raise ValidationError(f"artifact must be a regular direct child: {relative}")
    return path


def _verify_artifacts(root: Path, manifest: dict[str, Any]) -> None:
    expected_files = {"manifest.json", *(item[0] for item in ARTIFACT_LAYOUT)}
    try:
        observed_files = {path.name for path in root.iterdir()}
    except OSError as error:
        raise ValidationError(f"cannot enumerate M6 artifact directory: {error}") from error
    if observed_files != expected_files:
        raise ValidationError(
            "M6 artifact directory contains missing or extra files: "
            f"expected={sorted(expected_files)} actual={sorted(observed_files)}"
        )
    artifacts = manifest["artifacts"]
    if [(record["path"], record["role"], record["media_type"]) for record in artifacts] != list(ARTIFACT_LAYOUT):
        raise ValidationError("artifact layout/order/role/media type drifted")
    for record in artifacts:
        path = _confined_file(root, record["path"])
        if path.stat().st_size != record["bytes"] or _sha256(path) != record["sha256"]:
            raise ValidationError(f"artifact record hash/size failed: {record['path']}")
    lines = (root / "SHA256SUMS").read_text(encoding="ascii").splitlines()
    if len(lines) != len(CHECKSUM_PATHS):
        raise ValidationError("SHA256SUMS entry count drifted")
    for line, expected_name in zip(lines, CHECKSUM_PATHS, strict=True):
        match = re.fullmatch(r"([0-9a-f]{64})  ([A-Za-z0-9_.-]+)", line)
        if match is None or match.group(2) != expected_name:
            raise ValidationError("SHA256SUMS path/order/format drifted")
        if _sha256(root / expected_name) != match.group(1):
            raise ValidationError(f"SHA256SUMS failed: {expected_name}")


def _verify_m5_prerequisite(manifest: dict[str, Any]) -> None:
    expected = {
        "run_id": M5_RUN_ID,
        "relative_path": M5_RELATIVE_PATH,
        "manifest_sha256": M5_MANIFEST_SHA256,
        "sha256sums_sha256": M5_SHA256SUMS_SHA256,
        "decision": M5_DECISION,
        "verdict": M5_VERDICT,
        "validated": True,
    }
    if manifest["m5_prerequisite"] != expected:
        raise ValidationError("M5 r27 prerequisite record drifted")
    root = REPO_ROOT / M5_RELATIVE_PATH
    if (
        _sha256(root / "manifest.json") != M5_MANIFEST_SHA256
        or _sha256(root / "SHA256SUMS") != M5_SHA256SUMS_SHA256
    ):
        raise ValidationError("live M5 r27 prerequisite hashes drifted")
    prerequisite = load_json(root / "manifest.json")
    if (
        prerequisite.get("run_id") != M5_RUN_ID
        or prerequisite.get("gate", {}).get("decision") != M5_DECISION
        or prerequisite.get("gate", {}).get("passed") is not True
        or prerequisite.get("verdict") != M5_VERDICT
    ):
        raise ValidationError("live M5 r27 prerequisite semantics drifted")


def _validated_cleared_environment(manifest: dict[str, Any]) -> tuple[str, ...]:
    record = manifest["build_configuration"]
    cleared = record.get("cleared_environment_variables")
    if (
        not isinstance(cleared, list)
        or any(not isinstance(name, str) or not name for name in cleared)
        or cleared != sorted(set(cleared))
        or not set(BASE_CONFIGURE_ENV_UNSET).issubset(cleared)
    ):
        raise ValidationError("build_configuration cleared environment contract drifted")
    extras = set(cleared) - set(BASE_CONFIGURE_ENV_UNSET)
    if any(not name.startswith(THREAD_ENVIRONMENT_PREFIXES) for name in extras):
        raise ValidationError("build_configuration cleared an unapproved extra variable")
    expected_forced = {"LANG": "C", "LC_ALL": "C", **FORCED_THREAD_ENVIRONMENT}
    if record.get("forced_environment_variables") != expected_forced:
        raise ValidationError("build_configuration forced environment drifted")
    return tuple(cleared)


def _verify_build_configuration(manifest: dict[str, Any]) -> None:
    cleared = _validated_cleared_environment(manifest)
    old_cleared = m5_validator.CONFIGURE_ENV_UNSET
    old_forced = m5_validator.FORCED_SUBPROCESS_ENVIRONMENT
    m5_validator.CONFIGURE_ENV_UNSET = cleared
    m5_validator.FORCED_SUBPROCESS_ENVIRONMENT = {
        "LANG": "C",
        "LC_ALL": "C",
        **FORCED_THREAD_ENVIRONMENT,
    }
    try:
        m5_validator._verify_build_configuration(manifest)
    except m5_validator.ValidationError as error:
        raise ValidationError(str(error)) from error
    finally:
        m5_validator.CONFIGURE_ENV_UNSET = old_cleared
        m5_validator.FORCED_SUBPROCESS_ENVIRONMENT = old_forced
    try:
        entries = m5_validator._parse_cmake_cache(REPO_ROOT / "build-openfhe/CMakeCache.txt")
    except m5_validator.ValidationError as error:
        raise ValidationError(str(error)) from error
    if entries.get("MOAI_ENABLE_SEAL_REFERENCE") != ("BOOL", "OFF"):
        raise ValidationError("live CMake cache does not prove default SEAL reference OFF")


def _verify_preflight(manifest: dict[str, Any]) -> None:
    preflight = manifest["preflight"]
    if preflight["passed_tests"] != list(M6_CTEST_EXPECTED_TESTS):
        raise ValidationError("M6 preflight exact passed-test list drifted")
    try:
        m5_validator._require_schedule_preflight_record(
            preflight["schedule_preflight"],
            "preflight.schedule_preflight",
        )
        m5_validator._require_crypto_preflight_record(
            preflight["crypto_preflight"],
            "preflight.crypto_preflight",
        )
    except m5_validator.ValidationError as error:
        raise ValidationError(str(error)) from error


def _verify_workload(manifest: dict[str, Any]) -> None:
    workload = manifest["workload"]
    executable = REPO_ROOT / workload["executable_path"]
    if (
        executable.is_symlink()
        or not executable.is_file()
        or executable.resolve() != executable
        or executable.stat().st_size != workload["executable_bytes"]
        or _sha256(executable) != workload["executable_sha256"]
    ):
        raise ValidationError("live benchmark executable hash/size provenance drifted")


def _verify_inputs(manifest: dict[str, Any]) -> None:
    seen: set[str] = set()
    for record in manifest["inputs"]:
        relative = record["path"]
        if relative in seen or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValidationError(f"duplicate or unsafe input path: {relative}")
        seen.add(relative)
        path = REPO_ROOT / relative
        try:
            path.resolve().relative_to(REPO_ROOT.resolve())
        except ValueError as error:
            raise ValidationError(f"input escapes repository: {relative}") from error
        if path.is_symlink() or not path.is_file():
            raise ValidationError(f"input is not a regular file: {relative}")
        if path.stat().st_size != record["bytes"] or _sha256(path) != record["sha256"]:
            raise ValidationError(f"input hash/size drifted: {relative}")


def _git_blob(commit: str, path: str) -> bytes:
    completed = subprocess.run(
        [str(SYSTEM_GIT), "show", f"{commit}:{path}"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        stdin=subprocess.DEVNULL,
    )
    if completed.returncode != 0:
        raise ValidationError(f"schema-binding path is absent from manifest commit: {path}")
    return completed.stdout


def _verify_schema_binding(manifest: dict[str, Any]) -> None:
    binding = manifest["schema_binding"]
    if binding["canonicalization"] != "git-blob-and-working-tree-sha256-v1":
        raise ValidationError("schema binding canonicalization drifted")
    records = binding["files"]
    if [record["path"] for record in records] != list(SCHEMA_BINDING_PATHS):
        raise ValidationError("schema binding paths/order drifted")
    commit = manifest["git"]["local_commit"]
    for record in records:
        path = REPO_ROOT / record["path"]
        contents = path.read_bytes()
        blob = _git_blob(commit, record["path"])
        if (
            contents != blob
            or len(contents) != record["bytes"]
            or hashlib.sha256(contents).hexdigest() != record["sha256"]
            or hashlib.sha256(blob).hexdigest() != record["git_blob_sha256"]
        ):
            raise ValidationError(f"schema binding drifted: {record['path']}")


def _run_git(arguments: list[str]) -> str:
    completed = subprocess.run(
        [str(SYSTEM_GIT), *arguments],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
    )
    if completed.returncode != 0:
        raise ValidationError(f"git {' '.join(arguments)} failed: {completed.stderr.strip()}")
    return completed.stdout.strip()


def _verify_git(manifest: dict[str, Any], live: bool) -> None:
    record = manifest["git"]
    if len({record["local_commit"], record["tracking_commit"], record["remote_commit"]}) != 1:
        raise ValidationError("manifest local/tracking/remote commits differ")
    if not live:
        return
    if _run_git(["status", "--porcelain=v1", "--untracked-files=normal"]):
        raise ValidationError("repository is not clean during live M6 validation")
    if _run_git(["branch", "--show-current"]) != BRANCH:
        raise ValidationError("live branch differs from refactor/openfhe-cpu")
    head = _run_git(["rev-parse", "HEAD"])
    tracking = _run_git(["rev-parse", TRACKING_REF])
    remote = _run_git(["ls-remote", "--exit-code", REMOTE_URL, REMOTE_REF])
    parts = remote.split()
    if len(parts) != 2 or parts[1] != REMOTE_REF:
        raise ValidationError("live remote lookup is malformed")
    if (head, tracking, parts[0]) != (
        record["local_commit"],
        record["tracking_commit"],
        record["remote_commit"],
    ):
        raise ValidationError("live Git state differs from the M6 manifest")


def _verify_environment(manifest: dict[str, Any]) -> None:
    environment = manifest["environment"]
    threading = environment.get("threading")
    cleared = _validated_cleared_environment(manifest)
    if (
        environment.get("logical_cpu_count") != LOGICAL_CPU_COUNT
        or not isinstance(threading, dict)
        or set(threading)
        != {"forced", "cleared_inherited_variables", "applies_to"}
        or threading.get("forced") != FORCED_THREAD_ENVIRONMENT
        or threading.get("cleared_inherited_variables") != list(cleared)
        or threading.get("applies_to")
        != "configure, build, CTest, warm-up, measured samples, validator"
    ):
        raise ValidationError("M6 fixed 16-thread environment contract drifted")


def _verify_commands(manifest: dict[str, Any]) -> None:
    commands = manifest["commands"]
    if len(commands) != 18:
        raise ValidationError("M6 command transcript must contain exactly 18 commands")
    expected_git = [
        ["/usr/bin/git", "status", "--porcelain=v1", "--untracked-files=normal"],
        ["/usr/bin/git", "branch", "--show-current"],
        ["/usr/bin/git", "rev-parse", "HEAD"],
        ["/usr/bin/git", "rev-parse", TRACKING_REF],
        ["/usr/bin/git", "config", "--local", "--get-all", "remote.origin.url"],
        ["/usr/bin/git", "ls-remote", "--exit-code", REMOTE_URL, REMOTE_REF],
    ]
    if [command["argv"] for command in commands[:6]] != expected_git:
        raise ValidationError("M6 Git preflight command transcript drifted")

    cleared = manifest["build_configuration"]["cleared_environment_variables"]
    expected_configure = [
        "/usr/bin/env",
        *(f"--unset={name}" for name in cleared),
        "/usr/bin/cmake",
        "--fresh",
        "-S",
        str(REPO_ROOT),
        "-B",
        str(REPO_ROOT / "build-openfhe"),
        "-G",
        "Unix Makefiles",
        "-DOpenFHE_DIR:PATH=/home/shawnsheep/opt/openfhe_v1_5_1/lib/OpenFHE",
        "-DCMAKE_BUILD_TYPE:STRING=Release",
        "-DBUILD_TESTING:BOOL=ON",
        "-DCMAKE_CXX_COMPILER:FILEPATH=/usr/bin/c++",
        "-DCMAKE_MAKE_PROGRAM:FILEPATH=/usr/bin/gmake",
        "-DCMAKE_CXX_FLAGS:STRING=",
        "-DCMAKE_CXX_FLAGS_RELEASE:STRING=-O3 -DNDEBUG",
        "-DCMAKE_EXE_LINKER_FLAGS:STRING=",
        "-DCMAKE_EXE_LINKER_FLAGS_RELEASE:STRING=",
        "-DCMAKE_SHARED_LINKER_FLAGS:STRING=",
        "-DCMAKE_MODULE_LINKER_FLAGS:STRING=",
        "-DCMAKE_STATIC_LINKER_FLAGS:STRING=",
    ]
    expected_preflight = [
        expected_configure,
        [
            "/usr/bin/cmake",
            "--build",
            str(REPO_ROOT / "build-openfhe"),
            "--clean-first",
            "-j",
            "4",
        ],
        [
            "/usr/bin/ldd",
            str(REPO_ROOT / "build-openfhe/openfhe_encoder_12_layer_smoke"),
        ],
        [
            "/usr/bin/ctest",
            "--test-dir",
            str(REPO_ROOT / "build-openfhe"),
            "--output-on-failure",
            "--verbose",
            "--no-tests=error",
            "-R",
            m5_validator.M5_CTEST_PATTERN,
        ],
        [
            "/usr/bin/ctest",
            "--test-dir",
            str(REPO_ROOT / "build-openfhe"),
            "--output-on-failure",
            "--verbose",
            "--no-tests=error",
            "-R",
            M6_CTEST_PATTERN,
        ],
    ]
    if [command["argv"] for command in commands[6:11]] != expected_preflight:
        raise ValidationError("M6 configure/build/linkage/CTest transcript drifted")

    runtime: list[tuple[str, int]] = []
    for command in commands[11:17]:
        argv = command["argv"]
        expected_tail = [
            "/home/shawnsheep/MOAI/build-openfhe/openfhe_encoder_12_layer_smoke",
            "--data-root",
            "/home/shawnsheep/MOAI/data",
            "--benchmark-sample",
        ]
        if argv[0] != "/usr/bin/time" or argv[-4:] != expected_tail:
            raise ValidationError("M6 benchmark command argv drifted")
        try:
            format_value = argv[argv.index("--format") + 1]
            descriptor = _strict_json_loads(
                format_value.replace("%e", "1.0")
                .replace("%M", "1")
                .replace("%x", "0")
            )
        except (ValueError, IndexError, json.JSONDecodeError) as error:
            raise ValidationError("M6 GNU time format is malformed") from error
        runtime.append((descriptor.get("phase"), descriptor.get("index")))
    if runtime != _expected_phases():
        raise ValidationError(f"M6 command transcript is not one warm-up plus five measured: {runtime}")
    validator_argv = commands[17]["argv"]
    expected_validator_tail = [
        str(REPO_ROOT / "scripts/validate_openfhe_m6_benchmark.py"),
        "--schema",
        str(DEFAULT_SCHEMA),
        "--manifest",
        str(OUTPUT_ROOT / manifest["run_id"] / "manifest.json"),
        "--verify-git",
    ]
    if (
        len(validator_argv) != len(expected_validator_tail) + 1
        or not Path(validator_argv[0]).is_absolute()
        or validator_argv[1:] != expected_validator_tail
    ):
        raise ValidationError("M6 command transcript lacks one live independent validator")


def _reject_speedup_claim(value: Any, path: str = "$") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if "speedup" in key.lower() and not (
                child_path == "$.comparison.seal_reference.speedup" and child is None
            ):
                raise ValidationError(f"speedup field/claim is forbidden: {child_path}")
            _reject_speedup_claim(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_speedup_claim(child, f"{path}[{index}]")


def validate_bundle(
    manifest_path: Path,
    manifest: Any,
    schema: dict[str, Any],
    verify_git: bool,
    *,
    expected_run_id: str | None = None,
    allow_staging_root: bool = False,
) -> None:
    if not isinstance(manifest, dict):
        raise ValidationError("M6 manifest must be an object")
    _validate_instance(manifest, schema)
    run_id = manifest["run_id"]
    if RUN_ID_PATTERN.fullmatch(run_id) is None or (
        expected_run_id is not None and run_id != expected_run_id
    ):
        raise ValidationError("M6 run id is invalid or differs from the staging contract")
    root = manifest_path.resolve().parent
    if manifest_path.name != "manifest.json":
        raise ValidationError("M6 manifest filename must be manifest.json")
    if allow_staging_root:
        if root.parent != OUTPUT_ROOT.resolve() or not root.name.startswith(f".{run_id}.staging-"):
            raise ValidationError("M6 staging directory is not a confined atomic staging root")
    elif root.parent != OUTPUT_ROOT.resolve() or root.name != run_id:
        raise ValidationError("M6 artifact must be a direct run-id child of results/openfhe")
    started = _parse_timestamp(manifest["started_at"], "started_at")
    finished = _parse_timestamp(manifest["finished_at"], "finished_at")
    if finished < started:
        raise ValidationError("M6 finished_at precedes started_at")
    if manifest["claim_boundary"] != CLAIM_BOUNDARY:
        raise ValidationError("M6 claim boundary drifted")
    if manifest["comparison"] != {
        "seal_reference": {
            "status": "not_run",
            "comparability": "noncomparable",
            "reason": "No fixed-parameter, fixed-packing, fixed-precision SEAL benchmark was run.",
            "speedup": None,
        }
    }:
        raise ValidationError("SEAL must remain not-run/noncomparable without speedup")
    _reject_speedup_claim(manifest)
    records = _manifest_samples(manifest)
    measured = records[1:]
    _verify_metrics(manifest, measured)
    _verify_artifacts(root, manifest)
    _verify_raw_logs(root, records)
    _verify_metrics_csv(root, measured)
    _verify_m5_prerequisite(manifest)
    _verify_build_configuration(manifest)
    _verify_preflight(manifest)
    _verify_workload(manifest)
    _verify_inputs(manifest)
    _verify_environment(manifest)
    _verify_commands(manifest)
    _verify_git(manifest, verify_git)
    _verify_schema_binding(manifest)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--verify-git", action="store_true")
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    if arguments.schema.resolve() != DEFAULT_SCHEMA.resolve():
        raise ValidationError(f"M6 validator requires --schema {DEFAULT_SCHEMA}")
    if arguments.verify_git and arguments.manifest is None:
        raise ValidationError("--verify-git requires --manifest")
    schema = validate_schema(load_json(arguments.schema))
    result: dict[str, Any] = {
        "test": "validate_openfhe_m6_benchmark",
        "schema": str(arguments.schema.resolve()),
        "schema_version": SCHEMA_VERSION,
        "schema_valid": True,
    }
    if arguments.manifest is not None:
        manifest = load_json(arguments.manifest)
        validate_bundle(arguments.manifest, manifest, schema, arguments.verify_git)
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
        print(f"validate_openfhe_m6_benchmark failed: {error}", file=sys.stderr)
        sys.exit(1)
