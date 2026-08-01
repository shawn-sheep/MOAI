#!/usr/bin/env python3
"""Unit tests for the fail-closed M5 12-layer artifact runner."""

from __future__ import annotations

import copy
import csv
import hashlib
import importlib.util
import io
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = REPO_ROOT / "scripts" / "run_openfhe_encoder12_artifact_v6.py"
VALIDATOR_PATH = REPO_ROOT / "scripts" / "validate_openfhe_m5_artifact_v6.py"
RESULTS_ROOT = REPO_ROOT / "results" / "openfhe"

spec = importlib.util.spec_from_file_location(
    "run_openfhe_encoder12_artifact_v6", RUNNER_PATH
)
if spec is None or spec.loader is None:  # pragma: no cover - import machinery failure
    raise RuntimeError(f"cannot import {RUNNER_PATH}")
runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)

validator_spec = importlib.util.spec_from_file_location(
    "validate_openfhe_m5_artifact_v6", VALIDATOR_PATH
)
if validator_spec is None or validator_spec.loader is None:  # pragma: no cover
    raise RuntimeError(f"cannot import {VALIDATOR_PATH}")
validator = importlib.util.module_from_spec(validator_spec)
sys.modules[validator_spec.name] = validator
validator_spec.loader.exec_module(validator)
ORIGINAL_SCHEMA_BINDING_SNAPSHOT = runner._schema_binding_snapshot

TEST_LAYER0_INPUT_METADATA = {
    "level": 29,
    "noise_scale_degree": 1,
    "remaining_levels": 18,
    "scale_bits": 50,
    "expected_scale_bits": 50,
    "ciphertext_count": 5,
}
TEST_SOFTMAX_CHECKPOINT_METADATA = {
    "level": 18,
    "noise_scale_degree": 2,
    "remaining_levels": 28,
    "scale_bits": 100,
    "expected_scale_bits": 100,
    "ciphertext_count": 5,
}
TEST_LAYERNORM_CHECKPOINT_METADATA = {
    "level": 19,
    "noise_scale_degree": 2,
    "remaining_levels": 27,
    "scale_bits": 100,
    "expected_scale_bits": 100,
    "ciphertext_count": 5,
}
TEST_HANDOFF_INPUT_METADATA = {
    "level": 19,
    "noise_scale_degree": 2,
    "remaining_levels": 27,
    "scale_bits": 100,
    "expected_scale_bits": 100,
    "ciphertext_count": 5,
}


def _crypto_preflight_record(
    scale_bits: float = 50.000000079945785,
) -> dict[str, object]:
    metadata = dict(TEST_LAYER0_INPUT_METADATA)
    metadata["scale_bits"] = scale_bits
    return {
        "test": "openfhe_encoder_12_layer_crypto_preflight",
        "profile": "paper_compat",
        "security_claim": "none",
        "input_metadata": metadata,
        "server_private_key_present": False,
        "server_decryptions": 0,
        "server_plaintext_activations": False,
        "additive_he_operations": 0,
        "passed": True,
    }


def _schedule_preflight_record(
    *,
    formal_schedule_sealed: bool = False,
    he_metadata_verified_by_preflight: bool = False,
) -> dict[str, object]:
    return {
        "test": "openfhe_encoder_12_layer_preflight",
        "profile": "paper_compat",
        "security_claim": "none",
        "encoder_layers": 12,
        "layer_ids": list(range(12)),
        "chain_mode": "plaintext_output_to_next_oracle_input",
        "expected_fixture_files": 444,
        "metadata_schedule_preflight": "synthetic_structure_only",
        "formal_metadata_schedule_source": (
            "two_layer_live_candidate_requires_exact_three"
        ),
        "formal_schedule_sealed": formal_schedule_sealed,
        "he_metadata_verified_by_preflight": he_metadata_verified_by_preflight,
        "trace_hash_gate": "separate_fail_closed_validator_required",
        "finite_and_in_range": True,
        "final_relative_l2": 0.001,
        "final_cosine": 0.99999,
        "final_max_absolute": 0.002,
        "load_and_oracle_ms": 1.0,
    }


def _trace_scale_provenance() -> dict[str, str]:
    return {
        "source_path": runner.TRACE_SCALE_SOURCE_PATH,
        "json_locator": runner.TRACE_SCALE_LOCATOR,
        "contract_id": runner.TRACE_SCALE_CONTRACT_ID,
        "contract_sha256": runner.TRACE_SCALE_CONTRACT_SHA256,
        "values_sha256": runner.TRACE_SCALE_VALUES_SHA256,
        "raw_variance_sha256": runner.TRACE_SCALE_RAW_VARIANCE_SHA256,
    }


def _exact3_evidence() -> dict[str, object]:
    run_id = "synthetic-exact3-static-contract"
    return {
        "run_id": run_id,
        "relative_path": f"results/openfhe/{run_id}",
        "manifest_sha256": "d" * 64,
        "sha256sums_sha256": "e" * 64,
        "layer_count": 3,
        "artifact_eligible": False,
        "exact3_gate_passed": True,
        "schedule_evidence_eligible": True,
        "formal_schedule_sealed": True,
    }


def _canonical_test_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _reseal_exact3_test_bundle(
    bundle_root: Path,
    evidence: dict[str, object],
) -> None:
    manifest_path = bundle_root / "manifest.json"
    evidence["manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    sums = "".join(
        f"{hashlib.sha256((bundle_root / name).read_bytes()).hexdigest()}  {name}\n"
        for name in runner.EXACT3_CHECKSUM_PATHS
    )
    checksum_path = bundle_root / "SHA256SUMS"
    checksum_path.write_text(sums, encoding="ascii")
    evidence["sha256sums_sha256"] = hashlib.sha256(
        checksum_path.read_bytes()
    ).hexdigest()


def _refresh_exact3_manifest_file(bundle_root: Path, name: str) -> None:
    manifest_path = bundle_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    path = bundle_root / name
    if name in ("stdout.log", "stderr.log", "time.log"):
        result = path.stat()
        manifest["raw_logs"][name] = {
            "path": name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": result.st_size,
            "filesystem_mtime_ns": result.st_mtime_ns,
            "filesystem_ctime_ns": result.st_ctime_ns,
        }
    elif name == "metrics.csv":
        manifest["derived_artifacts"]["metrics.csv"] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size,
        }
    else:  # pragma: no cover - test helper misuse
        raise AssertionError(name)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _refresh_exact3_profile_evidence(
    repository_root: Path,
    evidence: dict[str, object],
) -> None:
    path = repository_root / runner.PROFILE_PATH
    profile = json.loads(path.read_text(encoding="utf-8"))
    profile["m5_schedule_candidate"]["exact3_seal"]["evidence"] = copy.deepcopy(
        evidence
    )
    path.write_text(
        json.dumps(profile, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_exact3_test_bundle(
    repository_root: Path,
    run_id: str = "synthetic-exact3-bundle",
) -> tuple[dict[str, object], Path]:
    bundle_root = repository_root / "results" / "openfhe" / run_id
    bundle_root.mkdir(parents=True)
    raw_contents, layers, final_summary, timing = _exact3_fixture(repository_root)
    for name, data in raw_contents.items():
        (bundle_root / name).write_bytes(data)

    def write_provenance_file(
        relative_path: str,
        payload: bytes,
        *,
        executable: bool = False,
    ) -> dict[str, object]:
        path = repository_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        if executable:
            path.chmod(0o755)
        result = path.stat()
        return {
            "path": relative_path,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
            "device": result.st_dev,
            "inode": result.st_ino,
            "mode": result.st_mode,
            "link_count": result.st_nlink,
            "mtime_ns": result.st_mtime_ns,
            "ctime_ns": result.st_ctime_ns,
        }

    source_files = [
        write_provenance_file(path, f"synthetic source: {path}\n".encode())
        for path in runner.EXACT3_SOURCE_PATHS
    ]
    candidate_profile = {
        "validation_status": runner.PROFILE_TRANSITION_PRE_STATE_VALUES[
            "/validation_status"
        ],
        "m5_schedule_candidate": {
            "status": runner.PROFILE_TRANSITION_PRE_STATE_VALUES[
                "/m5_schedule_candidate/status"
            ],
            "source": runner.PROFILE_TRANSITION_PRE_STATE_VALUES[
                "/m5_schedule_candidate/source"
            ],
            "formal_schedule_sealed": False,
            "calibrated_layers": [0, 1],
        },
        "feature_packed_layernorm_override": {
            "schedule_status": runner.PROFILE_TRANSITION_PRE_STATE_VALUES[
                "/feature_packed_layernorm_override/schedule_status"
            ]
        },
    }
    config_files = []
    for path in runner.EXACT3_CONFIG_PATHS:
        payload = (
            json.dumps(candidate_profile, indent=2, sort_keys=True).encode() + b"\n"
            if path == runner.PROFILE_PATH
            else f"synthetic config: {path}\n".encode()
        )
        config_files.append(write_provenance_file(path, payload))
    executable_record = write_provenance_file(
        runner.EXACT3_EXECUTABLE_PATH,
        b"synthetic exact3 executable\n",
        executable=True,
    )
    raw_logs = {
        name: {
            "path": name,
            "sha256": hashlib.sha256((bundle_root / name).read_bytes()).hexdigest(),
            "size_bytes": (bundle_root / name).stat().st_size,
            "filesystem_mtime_ns": (bundle_root / name).stat().st_mtime_ns,
            "filesystem_ctime_ns": (bundle_root / name).stat().st_ctime_ns,
        }
        for name in ("stdout.log", "stderr.log", "time.log")
    }
    empty_hash = hashlib.sha256(b"").hexdigest()
    manifest = {
        "schema": "diagnostic",
        "schema_id": runner.EXACT3_MANIFEST_SCHEMA_ID,
        "schema_version": runner.EXACT3_MANIFEST_SCHEMA_VERSION,
        "milestone": "M5",
        "artifact_kind": "exact3_schedule_evidence",
        "milestone_artifact_eligible": False,
        "schedule_evidence_eligible": True,
        "exact3_gate_passed": True,
        "formal_schedule_sealed": True,
        "timing_claim": False,
        "security_claim": "none",
        "run_id": run_id,
        "timestamp_semantics": {
            "run_id_label": run_id,
            "run_id_label_semantics": runner.EXACT3_RUN_ID_LABEL_SEMANTICS,
            "execution_started_at": "2026-07-31T11:59:58.000000+09:00",
            "execution_finished_at": "2026-07-31T11:59:59.000000+09:00",
            "execution_timestamp_semantics": (
                runner.EXACT3_EXECUTION_TIMESTAMP_SEMANTICS
            ),
            "raw_log_filesystem_timestamp_semantics": (
                runner.EXACT3_RAW_TIMESTAMP_SEMANTICS
            ),
            "sealed_at": "2026-07-31T12:00:00+09:00",
            "sealed_at_semantics": runner.EXACT3_SEALED_AT_SEMANTICS,
        },
        "git": {
            "head": "a" * 40,
            "branch": runner.BRANCH,
            "detached": False,
            "clean": True,
            "snapshot_semantics": runner.EXACT3_GIT_SNAPSHOT_SEMANTICS,
            "status_porcelain_v1_z_sha256": empty_hash,
            "tracked_diff": {
                "command": "git diff --binary HEAD -- .",
                "sha256": empty_hash,
                "size_bytes": 0,
            },
            "untracked_source_manifest": {
                "selection": runner.EXACT3_GIT_UNTRACKED_SELECTION,
                "canonicalization": runner.EXACT3_GIT_CANONICALIZATION,
                "sha256": hashlib.sha256(_canonical_test_json([])).hexdigest(),
                "entries": [],
            },
        },
        "command": {
            "argv": [
                "/usr/bin/time",
                "-v",
                "-o",
                str(
                    repository_root.resolve()
                    / "results"
                    / "openfhe"
                    / f".exact3-stage-{run_id}-fixture"
                    / "time.log"
                ),
                "--",
                f"./{runner.EXACT3_EXECUTABLE_PATH}",
                "--data-root",
                str(repository_root.resolve() / "data"),
                "--diagnostic-layer-count",
                "3",
            ],
            "runtime_argv": [
                f"./{runner.EXACT3_EXECUTABLE_PATH}",
                "--data-root",
                str(repository_root.resolve() / "data"),
                "--diagnostic-layer-count",
                "3",
            ],
            "gnu_time_reported_command": timing["command_text"],
            "cwd": str(repository_root.resolve()),
            "environment": {"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"},
            "stdin": "DEVNULL",
            "attempt_count": 1,
            "exit_code": 0,
            "started_at": "2026-07-31T11:59:58.000000+09:00",
            "finished_at": "2026-07-31T11:59:59.000000+09:00",
        },
        "profile": {
            "id": "paper_compat",
            "parameter_sha256": runner.PROFILE_SHA256,
            "security_claim": "none",
            "warning": runner.WARNING,
        },
        "provenance": {
            "snapshot_semantics": runner.EXACT3_PROVENANCE_SNAPSHOT_SEMANTICS,
            "source_files": source_files,
            "source_manifest_sha256": hashlib.sha256(
                _canonical_test_json(source_files)
            ).hexdigest(),
            "config_files": config_files,
            "config_manifest_sha256": hashlib.sha256(
                _canonical_test_json(config_files)
            ).hexdigest(),
            "executable": executable_record,
            "trace_scale": _trace_scale_provenance(),
            "profile_transition": {
                "path": runner.PROFILE_PATH,
                "canonicalization": runner.PROFILE_TRANSITION_CANONICALIZATION,
                "allowed_json_pointers": list(
                    runner.PROFILE_TRANSITION_ALLOWED_JSON_POINTERS
                ),
                "pre_state": {
                    "values": copy.deepcopy(
                        runner.PROFILE_TRANSITION_PRE_STATE_VALUES
                    ),
                    "absent": list(runner.PROFILE_TRANSITION_PRE_STATE_ABSENT),
                },
                "pre_seal_file_sha256": next(
                    record["sha256"]
                    for record in config_files
                    if record["path"] == runner.PROFILE_PATH
                ),
                "pre_seal_size_bytes": next(
                    record["size_bytes"]
                    for record in config_files
                    if record["path"] == runner.PROFILE_PATH
                ),
                "immutable_projection_sha256": (
                    runner._profile_immutable_projection_sha256(candidate_profile)
                ),
                "application_order": runner.PROFILE_TRANSITION_APPLICATION_ORDER,
                "post_state_contract": {
                    "values": copy.deepcopy(
                        runner.PROFILE_TRANSITION_POST_STATE_VALUES
                    ),
                    "exact3_seal_pointer": "/m5_schedule_candidate/exact3_seal",
                    "exact3_seal_keys": [
                        "source",
                        "evidence",
                        "pre_seal_profile",
                    ],
                },
            },
        },
        "raw_logs": raw_logs,
        "timing": {
            "kind": "non_benchmark_diagnostic",
            "timing_claim": False,
            "gnu_time_elapsed_text": timing["elapsed_text"],
            "elapsed_seconds": timing["elapsed_seconds"],
            "peak_rss_kib": timing["peak_rss_kib"],
            "peak_rss_bytes": timing["peak_rss_bytes"],
            "swaps": timing["swaps"],
            "exit_status": timing["exit_status"],
        },
        "gate_contract": copy.deepcopy(runner.EXACT3_GATE_CONTRACT),
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
            "layers": layers,
            "final_summary": final_summary,
            "later_layer_steady_state_verified": True,
            "second_handoff_previous_raw_output_to_input_recovered": 11,
        },
        "derived_artifacts": {
            "metrics.csv": {
                "sha256": hashlib.sha256(
                    (bundle_root / "metrics.csv").read_bytes()
                ).hexdigest(),
                "size_bytes": (bundle_root / "metrics.csv").stat().st_size,
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
    manifest_path = bundle_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    evidence = {
        "run_id": run_id,
        "relative_path": f"results/openfhe/{run_id}",
        "manifest_sha256": "",
        "sha256sums_sha256": "",
        "layer_count": 3,
        "artifact_eligible": False,
        "exact3_gate_passed": True,
        "schedule_evidence_eligible": True,
        "formal_schedule_sealed": True,
    }
    _reseal_exact3_test_bundle(bundle_root, evidence)
    preseal_profile = next(
        record for record in config_files if record["path"] == runner.PROFILE_PATH
    )
    sealed_profile = {
        "validation_status": runner.PROFILE_VALIDATION_STATUS_SEALED,
        "m5_schedule_candidate": {
            "status": runner.PROFILE_CANDIDATE_SEALED_STATUS,
            "source": runner.PROFILE_SCHEDULE_SEALED_SOURCE,
            "formal_schedule_sealed": True,
            "calibrated_layers": [0, 1, 2],
            "exact3_seal": {
                "source": runner.PROFILE_SCHEDULE_SEALED_SOURCE,
                "evidence": copy.deepcopy(evidence),
                "pre_seal_profile": {
                    "path": preseal_profile["path"],
                    "sha256": preseal_profile["sha256"],
                    "size_bytes": preseal_profile["size_bytes"],
                    "transition": runner.PROFILE_PRESEAL_TRANSITION,
                },
            },
        },
        "feature_packed_layernorm_override": {"schedule_status": "sealed"},
    }
    (repository_root / runner.PROFILE_PATH).write_text(
        json.dumps(sealed_profile, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return evidence, bundle_root


def _crypto_preflight_ctest_output(*records: dict[str, object]) -> str:
    total = len(runner.M5_CTEST_EXPECTED_TESTS)
    passed = "".join(
        f"{index}/{total} Test #{index}: {name} .......   Passed    0.01 sec\n"
        for index, name in enumerate(runner.M5_CTEST_EXPECTED_TESTS, start=1)
    )
    evidence = "".join(
        f"28: {json.dumps(record, sort_keys=True)}\n" for record in records
    )
    return f"{evidence}{passed}100% tests passed, 0 tests failed out of {total}\n"


def _m5_preflight_ctest_output(*records: dict[str, object]) -> str:
    return _crypto_preflight_ctest_output(
        _schedule_preflight_record(),
        *records,
    )


def _quality(relative_l2: float = 0.004, cosine: float = 0.9999) -> dict[str, float]:
    return {
        "relative_l2": relative_l2,
        "cosine": cosine,
        "max_absolute": 0.02,
    }


def _layer_record(layer_id: int) -> dict[str, object]:
    return {
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
        "input_metadata": (
            dict(TEST_LAYER0_INPUT_METADATA)
            if layer_id == 0
            else dict(TEST_HANDOFF_INPUT_METADATA)
        ),
        "raw_output_metadata": dict(runner.RAW_OUTPUT_METADATA),
        "softmax_denominator_metadata": dict(TEST_SOFTMAX_CHECKPOINT_METADATA),
        "attention_output_metadata": dict(
            runner.LAYER0_ATTENTION_OUTPUT_METADATA
            if layer_id == 0
            else runner.POST_REFRESH_ATTENTION_OUTPUT_METADATA
        ),
        "ln1_variance_metadata": dict(TEST_LAYERNORM_CHECKPOINT_METADATA),
        "ln1_output_metadata": dict(
            runner.LAYER0_LN1_OUTPUT_METADATA
            if layer_id == 0
            else runner.POST_REFRESH_LN1_OUTPUT_METADATA
        ),
        "ffn_output_metadata": dict(
            runner.LAYER0_FFN_OUTPUT_METADATA
            if layer_id == 0
            else runner.POST_REFRESH_FFN_OUTPUT_METADATA
        ),
        "ln2_variance_metadata": dict(TEST_LAYERNORM_CHECKPOINT_METADATA),
        "input_quality": _quality(0.003),
        "output_quality": _quality(0.004),
        "exact_trace_diagnostic": _quality(0.005),
        "inactive_max_abs": 3e-7 if layer_id == 3 else 2e-7,
        "inactive_zero_checkpoint_count": 28,
        "inactive_sentinel_max_error": 0.04 if layer_id == 5 else 0.01,
        "inactive_polynomial_sentinel_ranges": {
            "softmax_denominator": [0.98, 1.02],
            "ln1_normalized_variance": [45.0, 90.0],
            "ln2_normalized_variance": [32.0, 128.0],
        },
        "inactive_sentinel_range_status": "passed",
        "encrypted_polynomial_input_ranges": {
            "softmax_shifted_logits": [-10.0, 1.0],
            "softmax_denominator": [1.0, 60.0],
            "ln1_normalized_variance": [1.0, 70.0],
            "gelu_input": [-60.0, 120.0],
            "ln2_normalized_variance": [0.8, 1200.0],
        },
        "refresh_operation_counts": (
            dict(runner.EXPECTED_REFRESH_COUNTS)
            if layer_id < 11
            else dict(runner.ZERO_COUNTS)
        ),
        "layer_operation_counts": dict(runner.EXPECTED_LAYER_COUNTS),
        "cumulative_operation_counts": runner._expected_cumulative(layer_id),
        "range_validation_owner": "client",
        "checkpoint_decryption_owner": "client",
        "server_decryptions": 0,
        "server_plaintext_activations": False,
        "finite": True,
        "range_status": "passed",
    }


def _validate_layer_record(
    record: dict[str, object],
    layer_id: int,
    previous_raw_output_metadata: dict[str, object] | None = None,
) -> None:
    if layer_id > 0 and previous_raw_output_metadata is None:
        previous_raw_output_metadata = dict(runner.RAW_OUTPUT_METADATA)
    runner._validate_layer_record(
        record,
        layer_id,
        previous_raw_output_metadata,
    )


def _final_record(layers: list[dict[str, object]]) -> dict[str, object]:
    output = layers[-1]["output_quality"]
    assert isinstance(output, dict)
    return {
        "test": "openfhe_encoder_12_layer",
        "profile": "paper_compat",
        "security_claim": "none",
        "parameter_sha256": runner.PROFILE_SHA256,
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
        "fixture_load_oracle_ms": 100.0,
        "setup_keygen_ms": 200.0,
        "client_encrypt_ms": 20.0,
        "server_online_diagnostic_ms": 1000.0,
        "client_checkpoint_validate_ms": 300.0,
        "relative_l2": output["relative_l2"],
        "cosine": output["cosine"],
        "max_absolute": output["max_absolute"],
        "inactive_max_abs": max(float(layer["inactive_max_abs"]) for layer in layers),
        "inactive_sentinel_max_error": max(
            float(layer["inactive_sentinel_max_error"]) for layer in layers
        ),
        "final_metadata": dict(runner.RAW_OUTPUT_METADATA),
        "operation_counts": dict(runner.EXPECTED_TOTAL_COUNTS),
        "multiplicative_depth": 47,
        "max_observed_level": 45,
        "max_polynomial_depth": 10,
        "peak_rss_bytes": 1024 * 1024,
        "timing_claim": False,
        "latency_kind": "non_benchmark_diagnostic",
        "finite": True,
        "passed": True,
    }


def _exact3_fixture(
    repository_root: Path,
) -> tuple[
    dict[str, bytes],
    list[dict[str, object]],
    dict[str, object],
    dict[str, object],
]:
    layers: list[dict[str, object]] = []
    previous: dict[str, object] | None = None
    for layer_id in range(3):
        layer = _layer_record(layer_id)
        layer.update(
            {
                "test": "openfhe_encoder_exact_prefix_layer",
                "claim_scope": "diagnostic_prefix_exact_schedule",
                "artifact_eligible": False,
                "formal_schedule_sealed": False,
                "metadata_validation_mode": "exact",
                "handoff_refresh_performed": layer_id < 2,
                "diagnostic_gates_passed": True,
            }
        )
        metadata_fields = {
            "input": "input_metadata",
            "softmax_denominator": "softmax_denominator_metadata",
            "attention_output": "attention_output_metadata",
            "ln1_variance": "ln1_variance_metadata",
            "ln1_output": "ln1_output_metadata",
            "ffn_output": "ffn_output_metadata",
            "ln2_variance": "ln2_variance_metadata",
            "raw_output": "raw_output_metadata",
        }
        used = {
            name: int(layer[field]["level"])
            + int(layer[field]["noise_scale_degree"])
            - 1
            for name, field in metadata_fields.items()
        }
        previous_recovered = (
            None
            if previous is None
            else int(previous["raw_output_metadata"]["level"])
            + int(previous["raw_output_metadata"]["noise_scale_degree"])
            - 1
            - used["input"]
        )
        layer["metadata_used_levels"] = used
        layer["metadata_used_level_deltas"] = {
            "input_to_softmax_checkpoint_net_recovered": (
                used["input"] - used["softmax_denominator"]
            ),
            "softmax_checkpoint_to_attention_output_consumed": (
                used["attention_output"] - used["softmax_denominator"]
            ),
            "attention_output_to_ln1_checkpoint_net_recovered": (
                used["attention_output"] - used["ln1_variance"]
            ),
            "ln1_checkpoint_to_ln1_output_consumed": (
                used["ln1_output"] - used["ln1_variance"]
            ),
            "ln1_output_to_ffn_output_consumed": (
                used["ffn_output"] - used["ln1_output"]
            ),
            "ffn_output_to_ln2_checkpoint_net_recovered": (
                used["ffn_output"] - used["ln2_variance"]
            ),
            "ln2_checkpoint_to_raw_output_consumed": (
                used["raw_output"] - used["ln2_variance"]
            ),
            "previous_raw_output_to_input_recovered": previous_recovered,
        }
        sentinel_ranges = layer["inactive_polynomial_sentinel_ranges"]
        layer["inactive_sentinel_max_error"] = max(
            abs(float(endpoint) - 1.0)
            for observed in sentinel_ranges.values()
            for endpoint in observed
        )
        layer["refresh_operation_counts"] = (
            dict(runner.EXPECTED_REFRESH_COUNTS)
            if layer_id < 2
            else dict(runner.ZERO_COUNTS)
        )
        layer["cumulative_operation_counts"] = runner._exact3_expected_counts(
            layer_id
        )
        layers.append(layer)
        previous = layer

    summary = _final_record(layers)
    summary.update(
        {
            "test": "openfhe_encoder_exact_prefix",
            "claim_scope": "diagnostic_prefix_exact_schedule",
            "artifact_eligible": False,
            "formal_schedule_sealed": True,
            "metadata_validation_mode": "exact",
            "encoder_layers": 3,
            "server_layer_evaluations": 3,
            "inter_layer_refreshes": 2,
            "operation_counts": dict(runner.EXACT3_FINAL_COUNTS),
            "peak_rss_bytes": 2048 * 1024,
            "diagnostic_gates_passed": True,
        }
    )
    summary["relative_l2"] = layers[-1]["output_quality"]["relative_l2"]
    summary["cosine"] = layers[-1]["output_quality"]["cosine"]
    summary["max_absolute"] = layers[-1]["output_quality"]["max_absolute"]
    summary["inactive_max_abs"] = max(
        float(layer["inactive_max_abs"]) for layer in layers
    )
    summary["inactive_sentinel_max_error"] = max(
        float(layer["inactive_sentinel_max_error"]) for layer in layers
    )
    summary["final_metadata"] = copy.deepcopy(layers[-1]["raw_output_metadata"])
    command_argv = [
        f"./{runner.EXACT3_EXECUTABLE_PATH}",
        "--data-root",
        str(repository_root.resolve() / "data"),
        "--diagnostic-layer-count",
        "3",
    ]
    command_text = shlex.join(command_argv)
    timing: dict[str, object] = {
        "command_text": command_text,
        "elapsed_text": "0:01.00",
        "elapsed_seconds": 1.0,
        "peak_rss_kib": 2048,
        "peak_rss_bytes": 2048 * 1024,
        "swaps": 0,
        "exit_status": 0,
    }
    stdout = (
        runner.EXACT3_PROFILE_WARNING
        + "\n"
        + "\n".join(
            json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
            for record in [*layers, summary]
        )
        + "\n"
    ).encode()
    time_log = (
        f'\tCommand being timed: "{command_text}"\n'
        "\tElapsed (wall clock) time (h:mm:ss or m:ss): 0:01.00\n"
        "\tMaximum resident set size (kbytes): 2048\n"
        "\tSwaps: 0\n"
        "\tExit status: 0\n"
    ).encode()
    metrics = runner._exact3_metrics_bytes(layers, summary, timing)
    return (
        {
            "stdout.log": stdout,
            "stderr.log": b"",
            "time.log": time_log,
            "metrics.csv": metrics,
        },
        layers,
        summary,
        timing,
    )


def _sample() -> object:
    layers = [_layer_record(layer_id) for layer_id in range(12)]
    final = _final_record(layers)
    stdout = (
        "\n".join(
            json.dumps(record, sort_keys=True, separators=(",", ":"))
            for record in [*layers, final]
        )
        + "\n"
    )
    return runner.RunSample(
        stdout=stdout,
        layers=tuple(layers),
        final=final,
        elapsed_seconds=2.5,
        peak_rss_kib=2048,
        command={
            "command": "synthetic M5 correctness command",
            "cwd": str(REPO_ROOT),
            "exit_code": 0,
            "phase": "artifact_generation",
        },
    )


class M5ArtifactRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
        self.temporary_directory = tempfile.TemporaryDirectory(
            prefix="moai-m5-runner-", dir=RESULTS_ROOT
        )
        self.root = Path(self.temporary_directory.name)
        self.output_root = self.root / "artifacts"
        self.executable = self.root / "openfhe_encoder_12_layer_smoke"
        self.executable.write_text("synthetic executable\n", encoding="utf-8")
        self.executable.chmod(0o755)
        self.head = "a" * 40
        self.binding_patcher = mock.patch.object(
            runner,
            "_schema_binding_snapshot",
            return_value=self._schema_binding(),
        )
        self.binding_mock = self.binding_patcher.start()
        self.provenance_patcher = mock.patch.object(
            runner,
            "_require_build_configuration",
        )
        self.provenance_mock = self.provenance_patcher.start()
        self.static_contracts_patcher = mock.patch.object(
            runner,
            "_sealed_static_contracts",
            return_value=(_trace_scale_provenance(), _exact3_evidence()),
        )
        self.static_contracts_mock = self.static_contracts_patcher.start()

    def tearDown(self) -> None:
        self.static_contracts_patcher.stop()
        self.provenance_patcher.stop()
        self.binding_patcher.stop()
        self.temporary_directory.cleanup()

    @staticmethod
    def _schema_binding() -> dict[str, object]:
        return {
            component: {
                "id": component_id,
                "path": relative_path,
                "version": 6,
                "sha256": runner._sha256(REPO_ROOT / relative_path),
            }
            for component, (
                component_id,
                relative_path,
            ) in runner.SCHEMA_BINDING_SPECS.items()
        } | {
            "retired_unpublished_drafts": [
                "https://local.moai/openfhe-m5-artifact-schema-v4.json"
            ]
        }

    @staticmethod
    def _command_record(name: str) -> dict[str, object]:
        return {
            "command": name,
            "cwd": str(REPO_ROOT),
            "exit_code": 0,
            "phase": "artifact_generation",
        }

    def _git_state(self) -> object:
        return runner.GitState(
            self.head,
            self.head,
            runner.REMOTE_URL,
            self.head,
            tuple(self._command_record(f"git command {index}") for index in range(6)),
        )

    def test_runner_and_validator_reject_full_prototype_acceptance_drift(self) -> None:
        base = json.loads(
            (REPO_ROOT / runner.PROFILE_PATH).read_text(encoding="utf-8")
        )
        self.assertEqual(
            base["m5_prototype_acceptance"],
            runner.M5_PROTOTYPE_ACCEPTANCE,
        )
        self.assertEqual(
            runner.M5_PROTOTYPE_ACCEPTANCE,
            validator.M5_PROTOTYPE_ACCEPTANCE,
        )
        repository_root = self.root / "profile-contract-repo"
        profile_path = repository_root / runner.PROFILE_PATH
        profile_path.parent.mkdir(parents=True)
        for label, key, value in (
            ("legacy", "legacy_basis", "legacy MOAI used 1e-3"),
            ("scope", "scope", "all M5 diagnostics"),
            ("active", "active_quality_policy", "relaxed"),
            ("sentinel", "sentinel_range_policy", "disabled"),
            ("extra", "unexpected", True),
        ):
            with self.subTest(label=label):
                drifted = copy.deepcopy(base)
                drifted["m5_prototype_acceptance"][key] = value
                profile_path.write_text(
                    json.dumps(drifted, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                with (
                    mock.patch.object(runner, "REPO_ROOT", repository_root),
                    self.assertRaisesRegex(
                        runner.ArtifactRunnerError,
                        "prototype acceptance contract drifted",
                    ),
                ):
                    runner._profile_record()

                payload = profile_path.read_bytes()
                input_record = {
                    "path": validator.PROFILE_PATH,
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
                manifest = {
                    "profile": {
                        "source_config_sha256": input_record["sha256"],
                        "source_config_bytes": input_record["bytes"],
                        "effective_profile_locator": "/effective_profile",
                        "effective_profile_payload": drifted["effective_profile"],
                        "effective_profile_sha256": validator.PROFILE_SHA256,
                    }
                }
                with (
                    mock.patch.object(validator, "REPO_ROOT", repository_root),
                    self.assertRaisesRegex(
                        validator.ValidationError,
                        "prototype acceptance contract drifted",
                    ),
                ):
                    validator._verify_profile(
                        manifest,
                        {validator.PROFILE_PATH: input_record},
                    )

    def _preflight_commands(self) -> tuple[dict[str, object], ...]:
        timestamp = "2026-07-31T10:00:00+09:00"
        return (
            runner._command_record(
                runner._configure_command(
                    runner.DEFAULT_EXECUTABLE.parent,
                    runner.DEFAULT_OPENFHE_PREFIX,
                ),
                0,
                timestamp,
                timestamp,
            ),
            runner._command_record(
                [
                    str(runner.SYSTEM_CMAKE),
                    "--build",
                    str(runner.DEFAULT_EXECUTABLE.parent),
                    "--clean-first",
                    "-j",
                    "4",
                ],
                0,
                timestamp,
                timestamp,
            ),
            runner._command_record(
                [str(runner.SYSTEM_LDD), str(runner.DEFAULT_EXECUTABLE)],
                0,
                timestamp,
                timestamp,
            ),
            runner._command_record(
                [
                    str(runner.SYSTEM_CTEST),
                    "--test-dir",
                    str(runner.DEFAULT_EXECUTABLE.parent),
                    "--output-on-failure",
                    "--verbose",
                    "--no-tests=error",
                    "-R",
                    runner.M5_CTEST_PATTERN,
                ],
                0,
                timestamp,
                timestamp,
            ),
        )

    @staticmethod
    def _build_configuration() -> dict[str, object]:
        return {
            "generator": "Unix Makefiles",
            "source_root": str(REPO_ROOT),
            "build_root": str(runner.DEFAULT_EXECUTABLE.parent),
            "openfhe_dir": str(runner.DEFAULT_OPENFHE_PREFIX / "lib" / "OpenFHE"),
            "build_type": "Release",
            "build_testing": True,
            "cxx_compiler": str(runner.SYSTEM_CXX),
            "make_program": str(runner.SYSTEM_MAKE),
            "cxx_flags": "",
            "cxx_flags_release": "-O3 -DNDEBUG",
            "exe_linker_flags": "",
            "exe_linker_flags_release": "",
            "shared_linker_flags": "",
            "module_linker_flags": "",
            "static_linker_flags": "",
            "environment_inheritance": "ambient_minus_cleared_variables",
            "cleared_environment_variables": list(runner.CONFIGURE_ENV_UNSET),
            "forced_environment_variables": dict(runner.FORCED_SUBPROCESS_ENVIRONMENT),
            "cmake_cache": {
                "path": "build-openfhe/CMakeCache.txt",
                "bytes": 1,
                "sha256": "a" * 64,
            },
            "openfhe_cmake_package_files": [
                {
                    "path": str(
                        runner.DEFAULT_OPENFHE_PREFIX / "lib" / "OpenFHE" / name
                    ),
                    "bytes": 1,
                    "sha256": f"{index + 1:064x}",
                }
                for index, name in enumerate(runner.OPENFHE_CMAKE_PACKAGE_FILES)
            ],
            "openfhe_include_tree": {
                "root": str(runner.DEFAULT_OPENFHE_PREFIX / "include" / "openfhe"),
                "file_count": 1,
                "bytes": 1,
                "algorithm": "SHA-256",
                "canonicalization": runner.OPENFHE_INCLUDE_TREE_CANONICALIZATION,
                "sha256": "b" * 64,
            },
            "openfhe_linked_libraries": [
                {
                    "soname": soname,
                    "path": str(
                        runner.DEFAULT_OPENFHE_PREFIX
                        / "lib"
                        / runner.OPENFHE_LINKED_LIBRARY_BASENAMES[soname]
                    ),
                    "bytes": 1,
                    "sha256": f"{index + 10:064x}",
                }
                for index, soname in enumerate(runner.OPENFHE_LINKED_LIBRARY_SONAMES)
            ],
        }

    @staticmethod
    def _trace_inputs() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        records = [
            {
                "path": f"synthetic/layer_{index // 37}/input_{index:03}.csv",
                "sha256": f"{index:064x}"[-64:],
                "bytes": index + 1,
                "media_type": "text/csv",
                "role": "weights" if index % 37 < 16 else "trace",
            }
            for index in range(444)
        ]
        identities = [
            {
                "layer_id": layer_id,
                "weight_file_count": 16,
                "trace_file_count": 21,
                "weight_bundle_sha256": f"{1000 + layer_id:064x}",
                "trace_bundle_sha256": f"{2000 + layer_id:064x}",
            }
            for layer_id in range(12)
        ]
        return records, identities

    @staticmethod
    def _repository_record(path: str) -> dict[str, object]:
        return {
            "path": path,
            "sha256": "f" * 64,
            "bytes": 1,
            "media_type": "application/json",
            "role": "configuration",
        }

    @staticmethod
    def _profile() -> dict[str, object]:
        return {
            "id": "paper_compat",
            "security_claim": "none",
            "warning": runner.WARNING,
            "source_config_path": runner.PROFILE_PATH,
            "source_config_sha256": "f" * 64,
            "source_config_bytes": 1,
            "effective_profile_locator": runner.PROFILE_LOCATOR,
            "canonicalization": runner.CANONICALIZATION,
            "effective_profile_payload": {"profile_id": "paper_compat"},
            "effective_profile_sha256": runner.PROFILE_SHA256,
        }

    @staticmethod
    def _environment() -> dict[str, object]:
        return {
            "os": "Linux test",
            "architecture": "x86_64",
            "compiler": "c++ test",
            "cmake": "cmake test",
            "python": "3.test",
            "openfhe_version": "1.5.1",
            "openfhe_prefix": str(runner.DEFAULT_OPENFHE_PREFIX),
            "wsl": True,
        }

    def _config(self, run_id: str) -> object:
        return runner.RunnerConfig(
            self.executable,
            runner.DEFAULT_DATA_ROOT,
            self.output_root,
            run_id,
        )

    def _generation_patches(self, sample: object | None = None) -> tuple[object, ...]:
        return (
            mock.patch.object(
                runner, "_resolve_output_root", return_value=self.output_root
            ),
            mock.patch.object(
                runner, "_resolve_m5_executable", return_value=self.executable
            ),
            mock.patch.object(runner, "_preflight_git", return_value=self._git_state()),
            mock.patch.object(
                runner,
                "_run_m5_preflight",
                return_value=runner.M5Preflight(
                    self._preflight_commands(),
                    _crypto_preflight_record(),
                    self._build_configuration(),
                    _schedule_preflight_record(),
                ),
            ),
            mock.patch.object(runner, "_environment", return_value=self._environment()),
            mock.patch.object(runner, "_profile_record", return_value=self._profile()),
            mock.patch.object(
                runner, "_trace_input_records", return_value=self._trace_inputs()
            ),
            mock.patch.object(
                runner, "_repository_record", side_effect=self._repository_record
            ),
            mock.patch.object(runner, "_require_input_hashes"),
            mock.patch.object(runner, "_run_once", return_value=sample or _sample()),
            mock.patch.object(runner, "_validate_artifact"),
        )

    def test_success_runs_once_without_warmup_and_seals_four_files(self) -> None:
        patches = self._generation_patches()
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            patches[6],
            patches[7],
            patches[8] as input_hashes,
            patches[9] as run_once,
            patches[10] as validate_artifact,
        ):
            run_root = runner.generate_m5_artifact(self._config("unit-m5-success"))

        run_once.assert_called_once()
        self.assertEqual(input_hashes.call_count, 2)
        validate_artifact.assert_called_once_with(run_root / "manifest.json")
        self.assertEqual(self.binding_mock.call_count, 2)
        self.assertEqual(
            {path.name for path in run_root.iterdir()},
            {"manifest.json", "stdout.log", "metrics.csv", "SHA256SUMS"},
        )
        manifest = json.loads((run_root / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["schema_version"], 6)
        self.assertEqual(manifest["schema_binding"], self._schema_binding())
        self.assertEqual(manifest["milestone"], "M5")
        self.assertEqual(len(manifest["commands"]), 12)
        self.assertEqual(
            manifest["build_configuration"],
            self._build_configuration(),
        )
        self.assertEqual(len(manifest["inputs"]), 448)
        self.assertEqual(len(manifest["contracts"]["layer_input_identities"]), 12)
        self.assertEqual(manifest["workload"]["warmup_count"], 0)
        self.assertEqual(manifest["workload"]["repeat_count"], 1)
        self.assertFalse(manifest["workload"]["timing_claim"])
        self.assertEqual(
            manifest["workload"]["latency_kind"], "non_benchmark_diagnostic"
        )
        self.assertEqual(len(manifest["metrics"]["layers"]), 12)
        self.assertEqual(
            manifest["contracts"]["crypto_preflight"],
            _crypto_preflight_record(),
        )
        self.assertEqual(
            manifest["contracts"]["schedule_preflight"],
            _schedule_preflight_record(),
        )
        self.assertEqual(
            manifest["contracts"]["feature_packed_trace_scale_contract"],
            _trace_scale_provenance(),
        )
        self.assertEqual(
            manifest["contracts"]["exact3_evidence"],
            _exact3_evidence(),
        )
        self.assertNotIn("test", manifest["metrics"]["layers"][0])
        self.assertIn("softmax_denominator_metadata", manifest["metrics"]["layers"][0])
        self.assertIn("attention_output_metadata", manifest["metrics"]["layers"][0])
        self.assertIn("ln1_output_metadata", manifest["metrics"]["layers"][0])
        self.assertIn("ffn_output_metadata", manifest["metrics"]["layers"][0])
        self.assertIn(
            "inactive_polynomial_sentinel_ranges",
            manifest["metrics"]["layers"][0],
        )
        self.assertEqual(
            manifest["metrics"]["final_quality"]["inactive_sentinel_range_status"],
            "all_client_validated",
        )
        thresholds = manifest["contracts"]["thresholds"]
        self.assertNotIn("inactive_sentinel_max_error", thresholds)
        self.assertNotIn("layernorm_inactive_identity_max_error", thresholds)
        self.assertTrue(thresholds["inactive_sentinel_in_interval_required"])
        schedule = manifest["contracts"]["metadata_schedule"]
        self.assertEqual(
            schedule["used_level_formula"],
            "level + noise_scale_degree - 1",
        )
        self.assertEqual(
            schedule["attention_output_by_layer_regime"]["layer_0"]["level"],
            40,
        )
        self.assertEqual(
            schedule["attention_output_by_layer_regime"]["layers_1_to_11"]["level"],
            31,
        )
        self.assertEqual(
            schedule["ln1_output_by_layer_regime"]["layer_0"]["level"],
            32,
        )
        self.assertEqual(
            schedule["ln1_output_by_layer_regime"]["layers_1_to_11"]["level"],
            30,
        )
        self.assertEqual(
            schedule["ffn_output_by_layer_regime"]["layer_0"]["level"],
            45,
        )
        self.assertEqual(
            schedule["ffn_output_by_layer_regime"]["layers_1_to_11"]["level"],
            43,
        )
        self.assertNotIn("ln1_output", schedule)
        self.assertNotIn("ffn_output", schedule)
        self.assertEqual(
            manifest["metrics"]["checkpoint_metadata_sha256"],
            runner._checkpoint_metadata_sha256(_sample().layers),
        )

        with (run_root / "metrics.csv").open(
            "r", encoding="utf-8", newline=""
        ) as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 1)
        self.assertEqual(tuple(rows[0]), runner.CSV_FIELDS)
        self.assertEqual(rows[0]["run"], "1")
        self.assertEqual(rows[0]["timing_claim"], "false")
        self.assertEqual(rows[0]["latency_kind"], "non_benchmark_diagnostic")
        self.assertEqual(int(rows[0]["bootstraps"]), 355)
        self.assertEqual(float(rows[0]["inactive_sentinel_max_error"]), 0.04)
        self.assertEqual(
            rows[0]["checkpoint_metadata_sha256"],
            manifest["metrics"]["checkpoint_metadata_sha256"],
        )

    def test_unsealed_profile_stops_before_git_build_or_he(self) -> None:
        profile = runner._load_json(runner.REPO_ROOT / runner.PROFILE_PATH)
        profile["feature_packed_layernorm_override"]["schedule_status"] = "candidate"
        self.static_contracts_patcher.stop()
        try:
            with (
                mock.patch.object(runner, "_load_json", return_value=profile),
                mock.patch.object(
                    runner,
                    "EXACT3_EVIDENCE",
                    _exact3_evidence(),
                ),
                mock.patch.object(runner, "_preflight_git") as git_preflight,
                mock.patch.object(runner, "_run_m5_preflight") as m5_preflight,
                mock.patch.object(runner, "_run_once") as run_once,
                self.assertRaisesRegex(
                    runner.ArtifactRunnerError,
                    "metadata schedule is not sealed",
                ),
            ):
                runner.generate_m5_artifact(self._config("unit-m5-unsealed-profile"))
            git_preflight.assert_not_called()
            m5_preflight.assert_not_called()
            run_once.assert_not_called()
        finally:
            self.static_contracts_mock = self.static_contracts_patcher.start()

    def test_current_sealed_profile_and_exact3_bundle_are_approved(self) -> None:
        profile_contract = runner._require_profile_schedule_sealed(
            evidence=runner.EXACT3_EVIDENCE
        )
        self.assertEqual(profile_contract["status"], "sealed")
        self.assertEqual(
            profile_contract["exact3_evidence"], runner.EXACT3_EVIDENCE
        )
        self.assertEqual(
            runner._validate_exact3_evidence(
                runner.EXACT3_EVIDENCE, "EXACT3_EVIDENCE"
            ),
            runner.EXACT3_EVIDENCE,
        )

    def test_missing_exact3_stops_before_git_build_ctest_or_he(self) -> None:
        self.static_contracts_patcher.stop()
        try:
            with (
                mock.patch.object(
                    runner,
                    "_require_profile_schedule_sealed",
                    return_value={"status": "sealed"},
                ),
                mock.patch.object(
                    runner,
                    "_trace_scale_contract_provenance",
                    return_value=_trace_scale_provenance(),
                ),
                mock.patch.object(runner, "EXACT3_EVIDENCE", None),
                mock.patch.object(runner, "_preflight_git") as git_preflight,
                mock.patch.object(runner, "_run_m5_preflight") as m5_preflight,
                mock.patch.object(runner, "_run_once") as run_once,
                self.assertRaisesRegex(
                    runner.ArtifactRunnerError,
                    "evidence is missing",
                ),
            ):
                runner.generate_m5_artifact(self._config("unit-m5-missing-exact3"))
            git_preflight.assert_not_called()
            m5_preflight.assert_not_called()
            run_once.assert_not_called()
        finally:
            self.static_contracts_mock = self.static_contracts_patcher.start()

    def test_missing_exact3_and_retired_r16_evidence_fail_closed(self) -> None:
        with (
            mock.patch.object(
                runner,
                "_load_json",
                return_value={"feature_packed_layernorm_override": {}},
            ),
            self.assertRaisesRegex(
                runner.ArtifactRunnerError,
                "metadata schedule is not sealed",
            ),
        ):
            runner._require_profile_schedule_sealed()
        with (
            mock.patch.object(runner, "EXACT3_EVIDENCE", None),
            self.assertRaisesRegex(runner.ArtifactRunnerError, "evidence is missing"),
        ):
            runner._require_exact3_evidence()
        retired = {
            **runner.LEGACY_R16_CALIBRATION_EVIDENCE,
            "relative_path": (
                f"results/openfhe/{runner.LEGACY_R16_CALIBRATION_EVIDENCE['run_id']}"
            ),
            "layer_count": 3,
            "artifact_eligible": False,
            "exact3_gate_passed": True,
            "schedule_evidence_eligible": True,
            "formal_schedule_sealed": True,
        }
        with self.assertRaisesRegex(runner.ArtifactRunnerError, "retired r16"):
            runner._validate_exact3_evidence(retired, "retired")
        renamed = {
            **retired,
            "run_id": "renamed-retired-r16",
            "relative_path": "results/openfhe/renamed-retired-r16",
        }
        with self.assertRaisesRegex(runner.ArtifactRunnerError, "reuses retired r16"):
            runner._validate_exact3_evidence(renamed, "renamed")

    def test_trace_scale_provenance_recomputes_all_three_hashes(self) -> None:
        self.assertEqual(
            runner._trace_scale_contract_provenance(),
            _trace_scale_provenance(),
        )
        for key in ("contract_sha256", "values_sha256", "raw_variance_sha256"):
            malformed = _trace_scale_provenance()
            malformed[key] = "0" * 63
            self.assertIsNone(runner.SHA256_PATTERN.fullmatch(malformed[key]))

    def test_exact3_evidence_role_booleans_and_shape_fail_closed(self) -> None:
        invalid_values = {
            "artifact_eligible": True,
            "exact3_gate_passed": False,
            "schedule_evidence_eligible": False,
            "formal_schedule_sealed": 1,
        }
        for key, invalid in invalid_values.items():
            evidence = _exact3_evidence()
            evidence[key] = invalid
            with (
                self.subTest(key=key),
                self.assertRaisesRegex(runner.ArtifactRunnerError, key),
            ):
                runner._validate_exact3_evidence(evidence, "exact3")
        missing = _exact3_evidence()
        missing.pop("schedule_evidence_eligible")
        with self.assertRaisesRegex(runner.ArtifactRunnerError, "keys differ"):
            runner._validate_exact3_evidence(missing, "exact3")

    def test_exact3_bundle_is_dereferenced_and_verified_before_use(self) -> None:
        with tempfile.TemporaryDirectory(prefix="m5-exact3-runner-") as directory:
            repository_root = Path(directory)
            evidence, _ = _write_exact3_test_bundle(repository_root)
            validated = runner._validate_exact3_evidence(evidence, "exact3")
            self.assertEqual(
                runner._verify_exact3_evidence_bundle(validated, repository_root),
                evidence,
            )
            with mock.patch.object(runner, "EXACT3_EVIDENCE", evidence):
                self.assertEqual(
                    runner._require_exact3_evidence(repository_root),
                    evidence,
                )

    def test_exact3_bundle_rejects_path_inventory_symlink_hash_and_sums_drift(
        self,
    ) -> None:
        evidence = _exact3_evidence()
        for relative_path in (
            "/absolute/exact3",
            "results/openfhe/../escape",
            "results/openfhe/not-the-run-id",
        ):
            with self.subTest(relative_path=relative_path):
                tampered = copy.deepcopy(evidence)
                tampered["relative_path"] = relative_path
                with self.assertRaisesRegex(
                    runner.ArtifactRunnerError,
                    "relative_path",
                ):
                    runner._validate_exact3_evidence(tampered, "exact3")

        for case in (
            "missing",
            "extra",
            "bundle_symlink",
            "file_symlink",
            "manifest_hash",
            "file_hash",
            "live_source",
            "live_config",
            "sums_duplicate",
            "sums_order",
            "sums_path",
        ):
            with (
                self.subTest(case=case),
                tempfile.TemporaryDirectory(prefix="m5-exact3-runner-") as directory,
            ):
                repository_root = Path(directory)
                approved, bundle_root = _write_exact3_test_bundle(repository_root)
                if case == "missing":
                    (bundle_root / "time.log").unlink()
                elif case == "extra":
                    (bundle_root / "extra.log").write_text("extra\n", encoding="utf-8")
                elif case == "bundle_symlink":
                    outside = repository_root / "outside-bundle"
                    bundle_root.rename(outside)
                    bundle_root.symlink_to(outside, target_is_directory=True)
                elif case == "file_symlink":
                    time_path = bundle_root / "time.log"
                    outside = repository_root / "outside-time.log"
                    outside.write_bytes(time_path.read_bytes())
                    time_path.unlink()
                    time_path.symlink_to(outside)
                elif case == "manifest_hash":
                    approved["manifest_sha256"] = "0" * 64
                elif case == "file_hash":
                    (bundle_root / "stdout.log").write_bytes(b"tampered stdout\n")
                elif case == "live_source":
                    (repository_root / runner.EXACT3_SOURCE_PATHS[0]).write_bytes(
                        b"tampered source\n"
                    )
                elif case == "live_config":
                    (repository_root / runner.EXACT3_CONFIG_PATHS[0]).write_bytes(
                        b"tampered config\n"
                    )
                else:
                    checksum_path = bundle_root / "SHA256SUMS"
                    lines = checksum_path.read_text(encoding="ascii").splitlines()
                    if case == "sums_duplicate":
                        lines[-1] = lines[0]
                    elif case == "sums_order":
                        lines[0], lines[1] = lines[1], lines[0]
                    else:
                        digest = lines[-1].split("  ", maxsplit=1)[0]
                        lines[-1] = f"{digest}  unexpected.log"
                    checksum_path.write_text(
                        "\n".join(lines) + "\n",
                        encoding="ascii",
                    )
                    approved["sha256sums_sha256"] = hashlib.sha256(
                        checksum_path.read_bytes()
                    ).hexdigest()
                with self.assertRaises(runner.ArtifactRunnerError):
                    runner._verify_exact3_evidence_bundle(
                        approved,
                        repository_root,
                    )

    def test_exact3_historical_executable_need_not_match_fresh_m5_build(self) -> None:
        with tempfile.TemporaryDirectory(prefix="m5-exact3-runner-") as directory:
            repository_root = Path(directory)
            approved, _ = _write_exact3_test_bundle(repository_root)
            executable = repository_root / runner.EXACT3_EXECUTABLE_PATH
            executable.write_bytes(b"a different fresh M5 executable\n")
            executable.chmod(0o755)
            self.assertEqual(
                runner._verify_exact3_evidence_bundle(approved, repository_root),
                approved,
            )

    def test_exact3_manifest_claim_gate_trust_count_and_json_tamper_are_rejected(
        self,
    ) -> None:
        cases = (
            (("exact3_gate_passed",), False),
            (("formal_schedule_sealed",), False),
            (("timing_claim",), True),
            (("security_claim",), "128-bit"),
            (("command", "exit_code"), 1),
            (("timing", "kind"), "benchmark"),
            (("timing", "elapsed_seconds"), 2.0),
            (("timing", "peak_rss_bytes"), 1),
            (("gate_contract", "quality", "cosine_min"), 0.98),
            (("execution", "diagnostic_gates_passed"), False),
            (("execution", "server_decryptions"), 1),
            (("evidence", "final_summary", "server_private_key_present"), True),
            (("evidence", "final_summary", "encoder_layers"), 2),
            (
                ("evidence", "final_summary", "operation_counts", "rotations"),
                runner.EXACT3_FINAL_COUNTS["rotations"] + 1,
            ),
            (("evidence", "layers", 2, "layer_id"), 1),
            (("evidence", "second_handoff_previous_raw_output_to_input_recovered"), 10),
            (("verdict", "schedule_evidence_eligible"), False),
            (("run_id",), "different-run-id"),
            (("raw_logs", "stdout.log", "sha256"), "0" * 64),
            (("derived_artifacts", "metrics.csv", "size_bytes"), 0),
            (("provenance", "snapshot_semantics"), "unbound snapshot"),
            (("provenance", "source_files", 0, "path"), "src/openfhe/fake.cpp"),
            (("provenance", "source_files", 0, "inode"), 0),
            (("provenance", "source_files", 0, "mtime_ns"), -1),
            (("provenance", "source_manifest_sha256"), "0" * 64),
            (("provenance", "executable", "path"), "build-openfhe/not-approved"),
            (("provenance", "executable", "mode"), 0o100644),
            (("provenance", "trace_scale", "values_sha256"), "0" * 64),
            (
                ("provenance", "profile_transition", "allowed_json_pointers"),
                list(runner.PROFILE_TRANSITION_ALLOWED_JSON_POINTERS[:-1]),
            ),
            (
                (
                    "provenance",
                    "profile_transition",
                    "pre_state",
                    "values",
                    "/validation_status",
                ),
                "forged candidate status",
            ),
            (
                ("provenance", "profile_transition", "pre_seal_file_sha256"),
                "0" * 64,
            ),
            (
                ("provenance", "profile_transition", "pre_seal_size_bytes"),
                1,
            ),
            (
                ("provenance", "profile_transition", "immutable_projection_sha256"),
                "0" * 64,
            ),
            (
                ("provenance", "profile_transition", "application_order"),
                "forged application order",
            ),
            (
                (
                    "provenance",
                    "profile_transition",
                    "post_state_contract",
                    "exact3_seal_keys",
                ),
                ["evidence", "source", "pre_seal_profile"],
            ),
            (
                ("timestamp_semantics", "execution_finished_at"),
                "2026-07-31T11:59:57.000000+09:00",
            ),
            (("command", "started_at"), "2026-07-31T11:59:57.000000+09:00"),
            (("git", "snapshot_semantics"), "unbound Git snapshot"),
        )
        for path, value in cases:
            with (
                self.subTest(path=path),
                tempfile.TemporaryDirectory(prefix="m5-exact3-runner-") as directory,
            ):
                repository_root = Path(directory)
                approved, bundle_root = _write_exact3_test_bundle(repository_root)
                manifest_path = bundle_root / "manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                current = manifest
                for key in path[:-1]:
                    current = current[key]
                current[path[-1]] = value
                manifest_path.write_text(
                    json.dumps(
                        manifest,
                        indent=2,
                        sort_keys=True,
                        allow_nan=False,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                _reseal_exact3_test_bundle(bundle_root, approved)
                with self.assertRaises(runner.ArtifactRunnerError):
                    runner._verify_exact3_evidence_bundle(
                        approved,
                        repository_root,
                    )

    def test_exact3_sealed_profile_immutable_projection_drift_is_rejected(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="m5-exact3-profile-runner-") as directory:
            repository_root = Path(directory)
            approved, _ = _write_exact3_test_bundle(repository_root)
            profile_path = repository_root / runner.PROFILE_PATH
            profile = json.loads(profile_path.read_text(encoding="utf-8"))
            profile["unapproved_immutable_field"] = "drift"
            profile_path.write_text(
                json.dumps(profile, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                runner.ArtifactRunnerError,
                "immutable projection",
            ):
                runner._verify_exact3_evidence_bundle(approved, repository_root)

    def test_exact3_raw_logs_metrics_time_and_git_are_independently_rejected(
        self,
    ) -> None:
        cases = (
            "one_line_stdout",
            "nan_stdout",
            "stderr",
            "metrics",
            "time",
            "git",
        )
        for case in cases:
            with (
                self.subTest(case=case),
                tempfile.TemporaryDirectory(prefix="m5-exact3-raw-runner-") as directory,
            ):
                repository_root = Path(directory)
                approved, bundle_root = _write_exact3_test_bundle(repository_root)
                manifest_path = bundle_root / "manifest.json"
                if case == "one_line_stdout":
                    (bundle_root / "stdout.log").write_bytes(b"synthetic one line\n")
                    _refresh_exact3_manifest_file(bundle_root, "stdout.log")
                elif case == "nan_stdout":
                    stdout_path = bundle_root / "stdout.log"
                    text = stdout_path.read_text(encoding="utf-8")
                    text = text.replace('"relative_l2":0.004', '"relative_l2":NaN', 1)
                    stdout_path.write_text(text, encoding="utf-8")
                    _refresh_exact3_manifest_file(bundle_root, "stdout.log")
                elif case == "stderr":
                    (bundle_root / "stderr.log").write_bytes(b"unexpected error\n")
                    _refresh_exact3_manifest_file(bundle_root, "stderr.log")
                elif case == "metrics":
                    (bundle_root / "metrics.csv").write_bytes(
                        b"record_kind,passed\nfinal,true\n"
                    )
                    _refresh_exact3_manifest_file(bundle_root, "metrics.csv")
                elif case == "time":
                    time_path = bundle_root / "time.log"
                    time_path.write_text(
                        time_path.read_text(encoding="utf-8").replace(
                            "0:01.00",
                            "0:02.00",
                            1,
                        ),
                        encoding="utf-8",
                    )
                    _refresh_exact3_manifest_file(bundle_root, "time.log")
                else:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    manifest["git"]["branch"] = "main"
                    manifest_path.write_text(
                        json.dumps(
                            manifest,
                            indent=2,
                            sort_keys=True,
                            allow_nan=False,
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                _reseal_exact3_test_bundle(bundle_root, approved)
                _refresh_exact3_profile_evidence(repository_root, approved)
                with self.assertRaises(runner.ArtifactRunnerError):
                    runner._verify_exact3_evidence_bundle(
                        approved,
                        repository_root,
                    )

        for case in ("duplicate_key", "nan", "overflow"):
            with (
                self.subTest(case=case),
                tempfile.TemporaryDirectory(prefix="m5-exact3-runner-") as directory,
            ):
                repository_root = Path(directory)
                approved, bundle_root = _write_exact3_test_bundle(repository_root)
                manifest_path = bundle_root / "manifest.json"
                text = manifest_path.read_text(encoding="utf-8")
                if case == "duplicate_key":
                    text = text.replace(
                        '"schema": "diagnostic",',
                        '"schema": "diagnostic",\n  "schema": "diagnostic",',
                        1,
                    )
                elif case == "nan":
                    text = text.replace(
                        '"schema_version": 2', '"schema_version": NaN', 1
                    )
                else:
                    text = text.replace(
                        '"elapsed_seconds": 1.0',
                        '"elapsed_seconds": 1e999',
                        1,
                    )
                manifest_path.write_text(text, encoding="utf-8")
                _reseal_exact3_test_bundle(bundle_root, approved)
                with self.assertRaises(runner.ArtifactRunnerError):
                    runner._verify_exact3_evidence_bundle(
                        approved,
                        repository_root,
                    )

    def test_m5_claim_boundary_is_exact_and_schema_closed(self) -> None:
        expected = [
            "Five-token M5 12-layer encoder trace replay only; not task-level inference.",
            "server-only is an API/target trust boundary, not operating-system process "
            "isolation; correctness executable links client observer for checkpoints",
            "paper_compat OpenFHE CKKS CPU parameters with security_claim=none.",
            "M5 correctness is prototype-only with inactive/cross-lane <=1e-3; legacy "
            "MOAI defined no such threshold and strict numerical parity is not claimed.",
            "Timing is a non-benchmark diagnostic; no speedup claim.",
            "GPU, Discrete CKKS/FBT, QDQ, tokenizer, and classifier are excluded.",
        ]
        self.assertEqual(list(runner.CLAIM_BOUNDARY), expected)
        schema = json.loads(
            (REPO_ROOT / "docs" / "openfhe-m5-artifact-schema-v6.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            schema["properties"]["claim_boundary"],
            {"type": "array", "const": expected},
        )

    def test_schema_alone_rejects_metadata_schedule_and_count_tamper(self) -> None:
        patches = self._generation_patches()
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            patches[6],
            patches[7],
            patches[8],
            patches[9],
            patches[10],
        ):
            run_root = runner.generate_m5_artifact(
                self._config("unit-m5-schema-const-tamper")
            )
        manifest = json.loads((run_root / "manifest.json").read_text(encoding="utf-8"))
        manifest["workload"]["executable_path"] = (
            "build-openfhe/openfhe_encoder_12_layer_smoke"
        )
        schema = validator.validate_schema(
            validator.load_json(REPO_ROOT / "docs/openfhe-m5-artifact-schema-v6.json")
        )
        validator.validate_instance(manifest, schema, schema)

        metadata_paths = (
            ("initial_layer_input",),
            ("post_refresh_layer_input",),
            ("post_bootstrap_softmax_denominator_checkpoint",),
            ("post_bootstrap_canonicalized_layernorm_variance_checkpoint",),
            ("raw_layer_output",),
            ("attention_output_by_layer_regime", "layer_0"),
            ("attention_output_by_layer_regime", "layers_1_to_11"),
            ("ln1_output_by_layer_regime", "layer_0"),
            ("ln1_output_by_layer_regime", "layers_1_to_11"),
            ("ffn_output_by_layer_regime", "layer_0"),
            ("ffn_output_by_layer_regime", "layers_1_to_11"),
        )
        for path in metadata_paths:
            tampered = copy.deepcopy(manifest)
            metadata = tampered["contracts"]["metadata_schedule"]
            for part in path:
                metadata = metadata[part]
            metadata["level"] += 1
            with (
                self.subTest(metadata_path=path),
                self.assertRaises(validator.ValidationError),
            ):
                validator.validate_instance(tampered, schema, schema)

        for delta_name in runner.RELATIVE_USED_LEVEL_DELTAS:
            tampered = copy.deepcopy(manifest)
            tampered["contracts"]["metadata_schedule"]["relative_used_level_deltas"][
                delta_name
            ]["layers_1_to_11"] += 1
            with (
                self.subTest(delta_name=delta_name),
                self.assertRaises(validator.ValidationError),
            ):
                validator.validate_instance(tampered, schema, schema)

        layer0_null_tamper = copy.deepcopy(manifest)
        layer0_null_tamper["contracts"]["metadata_schedule"][
            "relative_used_level_deltas"
        ]["previous_raw_output_to_input_recovered"]["layer_0"] = 0
        with self.assertRaises(validator.ValidationError):
            validator.validate_instance(layer0_null_tamper, schema, schema)

        exact3_tamper = copy.deepcopy(manifest)
        exact3_tamper["contracts"]["exact3_evidence"]["manifest_sha256"] = "0" * 63
        with self.assertRaises(validator.ValidationError):
            validator.validate_instance(exact3_tamper, schema, schema)

        count_paths = (
            ("metrics", "operation_counts"),
            ("metrics", "layers", 0, "layer_operation_counts"),
            ("metrics", "layers", 0, "refresh_operation_counts"),
            ("metrics", "layers", 5, "cumulative_operation_counts"),
        )
        for path in count_paths:
            tampered = copy.deepcopy(manifest)
            counts = tampered
            for part in path:
                counts = counts[part]
            counts["rotations"] += 1
            with (
                self.subTest(count_path=path),
                self.assertRaises(validator.ValidationError),
            ):
                validator.validate_instance(tampered, schema, schema)

        checkpoint_count_tamper = copy.deepcopy(manifest)
        checkpoint_count_tamper["metrics"]["layers"][5][
            "inactive_zero_checkpoint_count"
        ] = 27
        with self.assertRaises(validator.ValidationError):
            validator.validate_instance(checkpoint_count_tamper, schema, schema)

        for label, metadata_path in (
            (
                "crypto preflight input",
                ("contracts", "crypto_preflight", "input_metadata"),
            ),
            ("final output", ("metrics", "final_metadata")),
        ):
            tampered = copy.deepcopy(manifest)
            metadata = tampered
            for part in metadata_path:
                metadata = metadata[part]
            metadata["level"] += 1
            metadata["remaining_levels"] -= 1
            with (
                self.subTest(metadata=label),
                self.assertRaises(validator.ValidationError),
            ):
                validator.validate_instance(tampered, schema, schema)

    def test_output_root_rejects_nested_custom_directory(self) -> None:
        self.assertEqual(
            runner._resolve_output_root(runner.DEFAULT_OUTPUT_ROOT),
            runner.DEFAULT_OUTPUT_ROOT.resolve(),
        )
        with self.assertRaisesRegex(runner.ArtifactRunnerError, "must be exactly"):
            runner._resolve_output_root(self.output_root)

    def test_generated_manifest_passes_schema_and_semantic_validator_contract(
        self,
    ) -> None:
        timestamp = runner._timestamp()
        git_commands = (
            [
                str(runner.SYSTEM_GIT),
                "status",
                "--porcelain=v1",
                "--untracked-files=normal",
            ],
            [str(runner.SYSTEM_GIT), "branch", "--show-current"],
            [str(runner.SYSTEM_GIT), "rev-parse", "HEAD"],
            [str(runner.SYSTEM_GIT), "rev-parse", runner.TRACKING_REF],
            [
                str(runner.SYSTEM_GIT),
                "config",
                "--local",
                "--get-all",
                f"remote.{runner.REMOTE_NAME}.url",
            ],
            [
                str(runner.SYSTEM_GIT),
                "ls-remote",
                "--exit-code",
                runner.REMOTE_URL,
                runner.REMOTE_REF,
            ],
        )
        git_state = runner.GitState(
            self.head,
            self.head,
            runner.REMOTE_URL,
            self.head,
            tuple(runner._command_record(command, 0) for command in git_commands),
        )
        build_root = runner.DEFAULT_EXECUTABLE.parent
        preflight_commands = (
            runner._command_record(
                runner._configure_command(
                    build_root,
                    runner.DEFAULT_OPENFHE_PREFIX,
                ),
                0,
                timestamp,
                timestamp,
            ),
            runner._command_record(
                [
                    str(runner.SYSTEM_CMAKE),
                    "--build",
                    str(build_root),
                    "--clean-first",
                    "-j",
                    "4",
                ],
                0,
                timestamp,
                timestamp,
            ),
            runner._command_record(
                [str(runner.SYSTEM_LDD), str(runner.DEFAULT_EXECUTABLE)],
                0,
                timestamp,
                timestamp,
            ),
            runner._command_record(
                [
                    str(runner.SYSTEM_CTEST),
                    "--test-dir",
                    str(build_root),
                    "--output-on-failure",
                    "--verbose",
                    "--no-tests=error",
                    "-R",
                    runner.M5_CTEST_PATTERN,
                ],
                0,
                timestamp,
                timestamp,
            ),
        )
        run_id = f"{self.root.name}-validator"
        planned_run_root = RESULTS_ROOT / run_id
        self.addCleanup(shutil.rmtree, planned_run_root, ignore_errors=True)
        config = runner.RunnerConfig(
            runner.DEFAULT_EXECUTABLE,
            runner.DEFAULT_DATA_ROOT,
            runner.DEFAULT_OUTPUT_ROOT,
            run_id,
        )
        sample = _sample()
        sample = runner.RunSample(
            sample.stdout,
            sample.layers,
            sample.final,
            sample.elapsed_seconds,
            sample.peak_rss_kib,
            runner._command_record(
                runner._runtime_command(
                    config, Path("/tmp/moai-m5-time-validator-unit.txt")
                ),
                0,
                timestamp,
                timestamp,
            ),
        )
        with (
            mock.patch.object(
                runner, "_resolve_output_root", return_value=RESULTS_ROOT
            ),
            mock.patch.object(runner, "_preflight_git", return_value=git_state),
            mock.patch.object(
                runner,
                "_run_m5_preflight",
                return_value=runner.M5Preflight(
                    preflight_commands,
                    _crypto_preflight_record(),
                    self._build_configuration(),
                    _schedule_preflight_record(),
                ),
            ),
            mock.patch.object(runner, "_environment", return_value=self._environment()),
            mock.patch.object(runner, "_run_once", return_value=sample),
            mock.patch.object(runner, "_validate_artifact"),
        ):
            run_root = runner.generate_m5_artifact(config)

        manifest_path = run_root / "manifest.json"
        schema = validator.validate_schema(
            validator.load_json(
                REPO_ROOT / "docs" / "openfhe-m5-artifact-schema-v6.json"
            )
        )
        with (
            mock.patch.object(validator, "_verify_schema_binding"),
            mock.patch.object(validator, "_verify_build_configuration"),
            mock.patch.object(validator, "_require_profile_schedule_sealed"),
            mock.patch.object(
                validator,
                "EXACT3_EVIDENCE",
                _exact3_evidence(),
            ),
            mock.patch.object(
                validator,
                "_verify_exact3_evidence_bundle",
                return_value=_exact3_evidence(),
            ),
            mock.patch.object(validator, "_verify_git") as verify_git,
        ):
            validator.validate_manifest(
                manifest_path,
                validator.load_json(manifest_path),
                schema,
                verify_git=True,
            )
        verify_git.assert_called_once()

    def test_runtime_command_is_full_chain_only(self) -> None:
        resource_file = Path("/tmp/m5-unit-resource.txt")
        command = runner._runtime_command(self._config("unit-command"), resource_file)
        self.assertEqual(
            command,
            [
                str(runner.TIME_EXECUTABLE),
                "--format=%M",
                f"--output={resource_file}",
                str(self.executable),
                "--data-root",
                str(runner.DEFAULT_DATA_ROOT),
            ],
        )
        for forbidden in (
            "--diagnostic-layer-count",
            "--metadata-calibration",
            "--preflight-only",
            "--crypto-preflight-only",
            "--layer",
            "--input-level",
        ):
            self.assertNotIn(forbidden, command)

    def test_validator_command_explicitly_binds_v6_schema(self) -> None:
        manifest = Path("/tmp/m5-v6-manifest.json")
        self.assertEqual(
            runner._validator_command(manifest),
            [
                sys.executable,
                str(runner.VALIDATOR_PATH),
                "--schema",
                str(runner.SCHEMA_PATH),
                "--manifest",
                str(manifest),
                "--verify-git",
            ],
        )

    def test_executable_target_is_fixed(self) -> None:
        with self.assertRaisesRegex(runner.ArtifactRunnerError, "fixed build target"):
            runner._resolve_m5_executable(self.executable)

    def test_cli_rejects_prefix_diagnostic_options(self) -> None:
        for arguments in (
            ["--diagnostic-layer-count", "2"],
            ["--metadata-calibration"],
            ["--preflight-only"],
            ["--crypto-preflight-only"],
        ):
            with (
                self.subTest(arguments=arguments),
                mock.patch.object(sys, "argv", [str(RUNNER_PATH), *arguments]),
                mock.patch.object(sys, "stderr", new=io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                runner.parse_arguments()

    def test_formal_parser_rejects_mixed_calibration_and_exact_prefix_evidence(
        self,
    ) -> None:
        formal_stdout = _sample().stdout
        diagnostic_records = (
            {
                "test": "openfhe_encoder_metadata_calibration_layer",
                "claim_scope": "diagnostic_metadata_calibration_only",
                "artifact_eligible": False,
                "formal_schedule_sealed": False,
                "metadata_validation_mode": "calibration",
                "diagnostic_gates_passed": True,
            },
            {
                "test": "openfhe_encoder_exact_prefix_layer",
                "claim_scope": "diagnostic_exact_prefix_only",
                "artifact_eligible": False,
                "formal_schedule_sealed": True,
                "metadata_validation_mode": "exact",
                "diagnostic_gates_passed": True,
            },
        )
        for record in diagnostic_records:
            mixed_stdout = formal_stdout + json.dumps(
                record, sort_keys=True, separators=(",", ":")
            )
            with (
                self.subTest(test=record["test"]),
                self.assertRaisesRegex(
                    runner.ArtifactRunnerError,
                    "diagnostic-only JSON",
                ),
            ):
                runner._extract_records(
                    mixed_stdout,
                    "openfhe_encoder_12_layer_layer",
                )

    def test_formal_parser_allows_only_explicit_non_evidence_json(self) -> None:
        disclosure = {
            "test": "openfhe_security_disclosure",
            "security_claim": "none",
        }
        stdout = json.dumps(disclosure, sort_keys=True) + "\n" + _sample().stdout
        self.assertEqual(
            len(runner._extract_records(stdout, "openfhe_encoder_12_layer_layer")),
            12,
        )
        self.assertEqual(
            len(runner._extract_records(stdout, "openfhe_encoder_12_layer")),
            1,
        )

        for record in (
            {**disclosure, "security_claim": "128-bit"},
            {"test": "unknown_json_evidence", "passed": True},
        ):
            with (
                self.subTest(record=record),
                self.assertRaises(runner.ArtifactRunnerError),
            ):
                runner._extract_records(
                    json.dumps(record) + "\n" + _sample().stdout,
                    "openfhe_encoder_12_layer_layer",
                )

    def test_formal_parser_rejects_diagnostic_fields_before_semantic_validation(
        self,
    ) -> None:
        for field, value in (
            ("artifact_eligible", False),
            ("claim_scope", "diagnostic_exact_prefix_only"),
        ):
            formal_layer_with_diagnostic_fields = _layer_record(0)
            formal_layer_with_diagnostic_fields[field] = value
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(
                    runner.ArtifactRunnerError,
                    "diagnostic-only JSON",
                ),
            ):
                runner._extract_records(
                    json.dumps(formal_layer_with_diagnostic_fields),
                    "openfhe_encoder_12_layer_layer",
                )

        formal_layer_with_diagnostic_fields = _layer_record(0)
        formal_layer_with_diagnostic_fields["artifact_eligible"] = False
        with self.assertRaisesRegex(runner.ArtifactRunnerError, "keys differ"):
            _validate_layer_record(formal_layer_with_diagnostic_fields, 0)

        formal_layers = [_layer_record(index) for index in range(12)]
        formal_summary_with_diagnostic_fields = _final_record(formal_layers)
        formal_summary_with_diagnostic_fields["metadata_validation_mode"] = (
            "calibration"
        )
        with self.assertRaisesRegex(runner.ArtifactRunnerError, "diagnostic-only JSON"):
            runner._extract_records(
                json.dumps(formal_summary_with_diagnostic_fields),
                "openfhe_encoder_12_layer",
            )
        formal_summary_with_diagnostic_scope = _final_record(formal_layers)
        formal_summary_with_diagnostic_scope["claim_scope"] = (
            "diagnostic_metadata_calibration_only"
        )
        with self.assertRaisesRegex(runner.ArtifactRunnerError, "diagnostic-only JSON"):
            runner._extract_records(
                json.dumps(formal_summary_with_diagnostic_scope),
                "openfhe_encoder_12_layer",
            )
        with self.assertRaisesRegex(runner.ArtifactRunnerError, "keys differ"):
            runner._validate_final_record(
                formal_summary_with_diagnostic_fields,
                formal_layers,
            )

    def test_ctest_contract_contains_preflight_and_excludes_full_workload(self) -> None:
        required_artifact_contracts = (
            "openfhe_m4_v5_artifact_schema_contract",
            "openfhe_m4_v5_artifact_validator_contract",
            "openfhe_m4_v5_encoder_artifact_runner_contract",
            "openfhe_m5_v6_artifact_schema_contract",
            "openfhe_m5_v6_artifact_validator_contract",
            "openfhe_m5_v6_encoder12_artifact_runner_contract",
        )
        for target in required_artifact_contracts:
            with self.subTest(target=target):
                self.assertIsNotNone(
                    __import__("re").fullmatch(runner.M5_CTEST_PATTERN, target)
                )
        self.assertIsNotNone(
            __import__("re").fullmatch(
                runner.M5_CTEST_PATTERN, "openfhe_encoder_12_layer_preflight"
            )
        )
        self.assertIsNotNone(
            __import__("re").fullmatch(
                runner.M5_CTEST_PATTERN,
                "openfhe_encoder_12_layer_crypto_preflight",
            )
        )
        self.assertIsNotNone(
            __import__("re").fullmatch(
                runner.M5_CTEST_PATTERN,
                "openfhe_feature_layernorm_smoke",
            )
        )
        self.assertIsNotNone(
            __import__("re").fullmatch(
                runner.M5_CTEST_PATTERN,
                "openfhe_encoder_layer_smoke",
            )
        )
        self.assertIsNone(
            __import__("re").fullmatch(
                runner.M5_CTEST_PATTERN, "openfhe_encoder_12_layer_smoke"
            )
        )

    def test_runner_and_validator_frozen_constants_match(self) -> None:
        self.assertEqual(runner.M5_SCHEMA_VERSION, validator.M5_SCHEMA_VERSION)
        self.assertEqual(runner.M5_CTEST_PATTERN, validator.M5_CTEST_PATTERN)
        self.assertEqual(runner.CSV_FIELDS, validator.CSV_FIELDS)
        self.assertEqual(runner.EXPECTED_LAYER_COUNTS, validator.LAYER_COUNTS)
        self.assertEqual(runner.EXPECTED_REFRESH_COUNTS, validator.REFRESH_COUNTS)
        self.assertEqual(runner.EXPECTED_TOTAL_COUNTS, validator.TOTAL_COUNTS)
        self.assertEqual(
            runner.LEGACY_R16_CALIBRATION_EVIDENCE,
            validator.LEGACY_R16_CALIBRATION_EVIDENCE,
        )
        self.assertEqual(runner.EXACT3_EVIDENCE, validator.EXACT3_EVIDENCE)
        self.assertEqual(runner.LAYER0_INPUT_METADATA, validator.LAYER0_INPUT_METADATA)
        self.assertEqual(
            runner.LAYER_HANDOFF_INPUT_METADATA,
            validator.LAYER_HANDOFF_INPUT_METADATA,
        )
        self.assertEqual(
            runner.SOFTMAX_CHECKPOINT_METADATA,
            validator.SOFTMAX_CHECKPOINT_METADATA,
        )
        self.assertEqual(
            runner.LAYERNORM_CHECKPOINT_METADATA,
            validator.LAYERNORM_CHECKPOINT_METADATA,
        )
        self.assertEqual(runner.RAW_OUTPUT_METADATA, validator.RAW_OUTPUT_METADATA)
        self.assertEqual(
            runner.LAYER0_ATTENTION_OUTPUT_METADATA,
            validator.LAYER0_ATTENTION_OUTPUT_METADATA,
        )
        self.assertEqual(
            runner.POST_REFRESH_ATTENTION_OUTPUT_METADATA,
            validator.POST_REFRESH_ATTENTION_OUTPUT_METADATA,
        )
        self.assertEqual(
            runner.LAYER0_LN1_OUTPUT_METADATA,
            validator.LAYER0_LN1_OUTPUT_METADATA,
        )
        self.assertEqual(
            runner.POST_REFRESH_LN1_OUTPUT_METADATA,
            validator.POST_REFRESH_LN1_OUTPUT_METADATA,
        )
        self.assertEqual(
            runner.LAYER0_FFN_OUTPUT_METADATA,
            validator.LAYER0_FFN_OUTPUT_METADATA,
        )
        self.assertEqual(
            runner.POST_REFRESH_FFN_OUTPUT_METADATA,
            validator.POST_REFRESH_FFN_OUTPUT_METADATA,
        )
        self.assertEqual(
            runner.RELATIVE_USED_LEVEL_DELTAS,
            validator.RELATIVE_USED_LEVEL_DELTAS,
        )
        self.assertEqual(runner.SCHEMA_BINDING_SPECS, validator.SCHEMA_BINDING_SPECS)
        self.assertEqual(
            runner.CHECKPOINT_METADATA_HASH_FIELDS,
            validator.CHECKPOINT_METADATA_HASH_FIELDS,
        )
        self.assertEqual(
            runner.CRYPTO_PREFLIGHT_KEYS,
            validator.CRYPTO_PREFLIGHT_KEYS,
        )
        self.assertEqual(
            runner.SCHEDULE_PREFLIGHT_KEYS,
            validator.SCHEDULE_PREFLIGHT_KEYS,
        )
        self.assertEqual(
            {key: list(value) for key, value in runner.FROZEN_RANGES.items()},
            validator.POLYNOMIAL_INTERVALS,
        )

    def test_operation_counts_and_formal_cumulative_counts_are_frozen(self) -> None:
        self.assertEqual(
            runner.EXPECTED_LAYER_COUNTS,
            {
                "rotations": 6300,
                "ct_pt_multiplications": 51885,
                "ct_ct_multiplications": 95,
                "explicit_rescale_requests": 810,
                "chebyshev_evaluations": 55,
                "estimated_polynomial_multiplications": 1150,
                "bootstraps": 25,
                "bootstrap_iterations": 50,
            },
        )
        self.assertEqual(
            runner.EXPECTED_TOTAL_COUNTS,
            {
                "rotations": 75600,
                "ct_pt_multiplications": 622675,
                "ct_ct_multiplications": 1140,
                "explicit_rescale_requests": 9775,
                "chebyshev_evaluations": 660,
                "estimated_polynomial_multiplications": 13800,
                "bootstraps": 355,
                "bootstrap_iterations": 710,
            },
        )
        for layer_id in range(11):
            cumulative = runner._expected_cumulative(layer_id)
            self.assertEqual(
                cumulative["ct_pt_multiplications"],
                (layer_id + 1) * 51890,
            )
            self.assertEqual(
                cumulative["explicit_rescale_requests"],
                (layer_id + 1) * 815,
            )
        self.assertEqual(runner._expected_cumulative(11), runner.EXPECTED_TOTAL_COUNTS)

    def test_preflight_commands_are_frozen(self) -> None:
        record = self._command_record("synthetic preflight")
        build_configuration = self._build_configuration()
        with (
            mock.patch.object(
                runner,
                "_verify_build_configuration",
                return_value={"cmake_cache": "stable"},
            ) as build_config,
            mock.patch.object(
                runner,
                "_run_checked_command",
                side_effect=[
                    (record, "synthetic configure output\n"),
                    (record, "synthetic build output\n"),
                    (
                        record,
                        _m5_preflight_ctest_output(_crypto_preflight_record()),
                    ),
                ],
            ) as checked,
            mock.patch.object(
                runner, "_resolve_m5_executable", return_value=self.executable
            ),
            mock.patch.object(
                runner,
                "_inspect_openfhe_linkage",
                return_value=(record, build_configuration["openfhe_linked_libraries"]),
            ) as linkage,
            mock.patch.object(
                runner,
                "_build_configuration_snapshot",
                return_value=build_configuration,
            ),
        ):
            preflight = runner._run_m5_preflight(
                runner.DEFAULT_EXECUTABLE.parent,
                runner.DEFAULT_EXECUTABLE,
                runner.DEFAULT_OPENFHE_PREFIX,
            )
        self.assertEqual(len(preflight.commands), 4)
        self.assertEqual(preflight.schedule_preflight, _schedule_preflight_record())
        self.assertEqual(preflight.crypto_preflight, _crypto_preflight_record())
        self.assertEqual(preflight.build_configuration, build_configuration)
        self.assertEqual(build_config.call_count, 2)
        self.assertEqual(
            checked.call_args_list[0].args[0],
            runner._configure_command(
                runner.DEFAULT_EXECUTABLE.parent,
                runner.DEFAULT_OPENFHE_PREFIX,
            ),
        )
        self.assertIn("--fresh", checked.call_args_list[0].args[0])
        self.assertEqual(
            checked.call_args_list[1].args[0],
            [
                str(runner.SYSTEM_CMAKE),
                "--build",
                str(runner.DEFAULT_EXECUTABLE.parent),
                "--clean-first",
                "-j",
                "4",
            ],
        )
        self.assertEqual(
            checked.call_args_list[2].args[0],
            [
                str(runner.SYSTEM_CTEST),
                "--test-dir",
                str(runner.DEFAULT_EXECUTABLE.parent),
                "--output-on-failure",
                "--verbose",
                "--no-tests=error",
                "-R",
                runner.M5_CTEST_PATTERN,
            ],
        )
        linkage.assert_called_once_with(self.executable, runner.DEFAULT_OPENFHE_PREFIX)

    def test_configure_command_clears_injection_environment_and_requires_fresh(
        self,
    ) -> None:
        command = runner._configure_command(
            runner.DEFAULT_EXECUTABLE.parent,
            runner.DEFAULT_OPENFHE_PREFIX,
        )
        cmake_index = command.index(str(runner.SYSTEM_CMAKE))
        self.assertEqual(
            command[:cmake_index],
            [
                str(runner.SYSTEM_ENV),
                *(f"--unset={name}" for name in runner.CONFIGURE_ENV_UNSET),
            ],
        )
        self.assertEqual(command[cmake_index + 1], "--fresh")
        self.assertEqual(command.count("--fresh"), 1)
        self.assertIn(
            f"-DCMAKE_CXX_COMPILER:FILEPATH={runner.SYSTEM_CXX}",
            command,
        )
        self.assertIn("-DCMAKE_CXX_FLAGS:STRING=", command)
        self.assertIn(
            "-DCMAKE_CXX_FLAGS_RELEASE:STRING=-O3 -DNDEBUG",
            command,
        )

    def test_artifact_environment_clears_all_injection_surfaces(self) -> None:
        poisoned = {name: f"poison-{name}" for name in runner.CONFIGURE_ENV_UNSET}
        poisoned["MOAI_SAFE_SENTINEL"] = "preserved"
        with mock.patch.dict(os.environ, poisoned, clear=False):
            environment = runner._artifact_environment()
        for name in runner.CONFIGURE_ENV_UNSET:
            with self.subTest(name=name):
                self.assertNotIn(name, environment)
        self.assertEqual(
            {name: environment[name] for name in runner.FORCED_SUBPROCESS_ENVIRONMENT},
            runner.FORCED_SUBPROCESS_ENVIRONMENT,
        )
        self.assertEqual(environment["MOAI_SAFE_SENTINEL"], "preserved")

    def test_preflight_rejects_pattern_without_m4_layer_regression(self) -> None:
        missing_m4 = runner.M5_CTEST_PATTERN.replace(
            "encoder_layer_smoke|",
            "",
        )
        with (
            mock.patch.object(runner, "M5_CTEST_PATTERN", missing_m4),
            mock.patch.object(runner, "_run_checked_command") as checked,
            self.assertRaisesRegex(
                runner.ArtifactRunnerError,
                "omits openfhe_encoder_layer_smoke",
            ),
        ):
            runner._run_m5_preflight(
                runner.DEFAULT_EXECUTABLE.parent,
                runner.DEFAULT_EXECUTABLE,
                runner.DEFAULT_OPENFHE_PREFIX,
            )
        checked.assert_not_called()

    def test_ctest_summary_rejects_skipped_test_and_missing_footer(self) -> None:
        output = _crypto_preflight_ctest_output(_crypto_preflight_record())
        runner._require_ctest_passed_tests(
            output,
            runner.M5_CTEST_EXPECTED_TESTS,
            "M5 test",
        )
        skipped = output.replace("Passed", "***Skipped", 1)
        with self.assertRaisesRegex(runner.ArtifactRunnerError, "statuses="):
            runner._require_ctest_passed_tests(
                skipped,
                runner.M5_CTEST_EXPECTED_TESTS,
                "M5 test",
            )
        without_footer = output.replace(
            (
                "100% tests passed, 0 tests failed out of "
                f"{len(runner.M5_CTEST_EXPECTED_TESTS)}\n"
            ),
            "",
        )
        with self.assertRaisesRegex(runner.ArtifactRunnerError, "footer_count=0"):
            runner._require_ctest_passed_tests(
                without_footer,
                runner.M5_CTEST_EXPECTED_TESTS,
                "M5 test",
            )

    def test_preflight_rejects_missing_executed_layer_gate_before_crypto_parse(
        self,
    ) -> None:
        record = self._command_record("synthetic preflight")
        build_configuration = self._build_configuration()
        missing_layer_gate = tuple(
            name
            for name in runner.M5_CTEST_EXPECTED_TESTS
            if name != "openfhe_encoder_layer_smoke"
        )
        total = len(runner.M5_CTEST_EXPECTED_TESTS)
        ctest_output = (
            f"28: {json.dumps(_crypto_preflight_record(), sort_keys=True)}\n"
            + "".join(
                f"{index}/{total} Test #{index}: {name} .......   Passed    0.01 sec\n"
                for index, name in enumerate(missing_layer_gate, start=1)
            )
        )
        with (
            mock.patch.object(
                runner,
                "_verify_build_configuration",
                return_value={"cmake_cache": "stable"},
            ),
            mock.patch.object(
                runner,
                "_run_checked_command",
                side_effect=[
                    (record, "synthetic configure output\n"),
                    (record, "synthetic build output\n"),
                    (record, ctest_output),
                ],
            ),
            mock.patch.object(
                runner, "_resolve_m5_executable", return_value=self.executable
            ),
            mock.patch.object(
                runner,
                "_inspect_openfhe_linkage",
                return_value=(
                    record,
                    build_configuration["openfhe_linked_libraries"],
                ),
            ),
            mock.patch.object(
                runner,
                "_build_configuration_snapshot",
                return_value=build_configuration,
            ),
            mock.patch.object(
                runner,
                "_parse_crypto_preflight_output",
            ) as parse_crypto,
            mock.patch.object(
                runner,
                "_parse_schedule_preflight_output",
            ) as parse_schedule,
            self.assertRaisesRegex(
                runner.ArtifactRunnerError,
                "did not enumerate and pass.*openfhe_encoder_layer_smoke",
            ),
        ):
            runner._run_m5_preflight(
                runner.DEFAULT_EXECUTABLE.parent,
                runner.DEFAULT_EXECUTABLE,
                runner.DEFAULT_OPENFHE_PREFIX,
            )
        parse_crypto.assert_not_called()
        parse_schedule.assert_not_called()

    def test_build_configuration_rejects_extra_flags_and_old_cache(self) -> None:
        build_root = self.root / "cache-fixture"
        prefix = self.root / "openfhe-prefix"
        package_root = prefix / "lib" / "OpenFHE"
        package_root.mkdir(parents=True)
        (package_root / "OpenFHEConfigVersion.cmake").write_text(
            'set(PACKAGE_VERSION "1.5.1")\n',
            encoding="utf-8",
        )
        executable = build_root / "openfhe_encoder_12_layer_smoke"
        build_root.mkdir()

        def write_cache(
            *,
            cxx_flags: str = "",
            compiler_type: str = "STRING",
            compiler: Path = runner.SYSTEM_CXX,
            home: Path = runner.REPO_ROOT,
            extra: str = "",
        ) -> None:
            lines = [
                "CMAKE_GENERATOR:INTERNAL=Unix Makefiles",
                f"CMAKE_HOME_DIRECTORY:INTERNAL={home}",
                "CMAKE_BUILD_TYPE:STRING=Release",
                "BUILD_TESTING:BOOL=ON",
                f"OpenFHE_DIR:PATH={package_root}",
                f"CMAKE_CXX_COMPILER:{compiler_type}={compiler}",
                f"CMAKE_MAKE_PROGRAM:FILEPATH={runner.SYSTEM_MAKE}",
                f"CMAKE_CXX_FLAGS:STRING={cxx_flags}",
                "CMAKE_CXX_FLAGS_RELEASE:STRING=-O3 -DNDEBUG",
                "CMAKE_EXE_LINKER_FLAGS:STRING=",
                "CMAKE_EXE_LINKER_FLAGS_RELEASE:STRING=",
                "CMAKE_SHARED_LINKER_FLAGS:STRING=",
                "CMAKE_MODULE_LINKER_FLAGS:STRING=",
                "CMAKE_STATIC_LINKER_FLAGS:STRING=",
            ]
            if extra:
                lines.append(extra)
            (build_root / "CMakeCache.txt").write_text(
                "\n".join(lines) + "\n",
                encoding="utf-8",
            )

        with (
            mock.patch.object(runner, "DEFAULT_EXECUTABLE", executable),
            mock.patch.object(runner, "DEFAULT_OPENFHE_PREFIX", prefix),
        ):
            write_cache()
            runner._verify_build_configuration(build_root, prefix)

            write_cache(compiler_type="FILEPATH")
            with self.assertRaisesRegex(
                runner.ArtifactRunnerError,
                "CMake cache binding drifted",
            ):
                runner._verify_build_configuration(build_root, prefix)

            write_cache(compiler=Path("/tmp/injected-c++"))
            with self.assertRaisesRegex(
                runner.ArtifactRunnerError,
                "CMake cache binding drifted",
            ):
                runner._verify_build_configuration(build_root, prefix)

            write_cache(cxx_flags="-march=native")
            with self.assertRaisesRegex(
                runner.ArtifactRunnerError,
                "CMake cache binding drifted",
            ):
                runner._verify_build_configuration(build_root, prefix)

            write_cache(home=self.root / "stale-checkout")
            with self.assertRaisesRegex(
                runner.ArtifactRunnerError,
                "CMake cache binding drifted",
            ):
                runner._verify_build_configuration(build_root, prefix)

            write_cache(extra="CMAKE_TOOLCHAIN_FILE:FILEPATH=/tmp/injected.cmake")
            with self.assertRaisesRegex(
                runner.ArtifactRunnerError,
                "forbidden",
            ):
                runner._verify_build_configuration(build_root, prefix)

    def test_linkage_requires_exact_three_sonames_and_canonical_versioned_files(
        self,
    ) -> None:
        prefix = self.root / "linked-prefix"
        library_root = prefix / "lib"
        library_root.mkdir(parents=True)
        resolved_paths: dict[str, Path] = {}
        for soname in runner.OPENFHE_LINKED_LIBRARY_SONAMES:
            path = library_root / runner.OPENFHE_LINKED_LIBRARY_BASENAMES[soname]
            path.write_bytes(f"{soname}\n".encode())
            resolved_paths[soname] = path

        def output(lines: list[tuple[str, Path]]) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(
                ["ldd", str(self.executable)],
                0,
                stdout="".join(
                    f"\t{soname} => {path} (0x00000001)\n" for soname, path in lines
                ),
                stderr="",
            )

        valid_lines = [
            (soname, resolved_paths[soname])
            for soname in runner.OPENFHE_LINKED_LIBRARY_SONAMES
        ]
        with (
            mock.patch.object(runner, "DEFAULT_OPENFHE_PREFIX", prefix),
            mock.patch.object(
                runner.subprocess,
                "run",
                return_value=output(valid_lines),
            ),
        ):
            _, records = runner._inspect_openfhe_linkage(
                self.executable,
                prefix,
            )
        self.assertEqual(
            [record["soname"] for record in records],
            list(runner.OPENFHE_LINKED_LIBRARY_SONAMES),
        )

        cases = {
            "missing": valid_lines[:-1],
            "duplicate": [*valid_lines, valid_lines[0]],
            "extra": [
                *valid_lines,
                ("libOPENFHEunexpected.so.1", resolved_paths[valid_lines[0][0]]),
            ],
        }
        for name, lines in cases.items():
            with (
                self.subTest(name=name),
                mock.patch.object(runner, "DEFAULT_OPENFHE_PREFIX", prefix),
                mock.patch.object(
                    runner.subprocess,
                    "run",
                    return_value=output(lines),
                ),
                self.assertRaises(runner.ArtifactRunnerError),
            ):
                runner._inspect_openfhe_linkage(self.executable, prefix)

        outside = (
            self.root / runner.OPENFHE_LINKED_LIBRARY_BASENAMES["libOPENFHEcore.so.1"]
        )
        outside.write_bytes(b"outside\n")
        out_of_bounds = [
            (
                soname,
                outside if soname == "libOPENFHEcore.so.1" else resolved_paths[soname],
            )
            for soname in runner.OPENFHE_LINKED_LIBRARY_SONAMES
        ]
        with (
            mock.patch.object(runner, "DEFAULT_OPENFHE_PREFIX", prefix),
            mock.patch.object(
                runner.subprocess,
                "run",
                return_value=output(out_of_bounds),
            ),
            self.assertRaisesRegex(runner.ArtifactRunnerError, "outside"),
        ):
            runner._inspect_openfhe_linkage(self.executable, prefix)

        core = resolved_paths["libOPENFHEcore.so.1"]
        core.unlink()
        core.symlink_to(outside)
        with (
            mock.patch.object(runner, "DEFAULT_OPENFHE_PREFIX", prefix),
            mock.patch.object(
                runner.subprocess,
                "run",
                return_value=output(valid_lines),
            ),
            self.assertRaisesRegex(runner.ArtifactRunnerError, "outside"),
        ):
            runner._inspect_openfhe_linkage(self.executable, prefix)

    def test_build_provenance_byte_drift_is_fail_closed(self) -> None:
        expected = self._build_configuration()
        drifted = copy.deepcopy(expected)
        drifted["openfhe_linked_libraries"][0]["sha256"] = "0" * 64
        self.provenance_patcher.stop()
        try:
            with (
                mock.patch.object(
                    runner,
                    "_build_configuration_snapshot",
                    return_value=drifted,
                ),
                self.assertRaisesRegex(
                    runner.ArtifactRunnerError,
                    "provenance changed after workload",
                ),
            ):
                runner._require_build_configuration(
                    expected,
                    runner.DEFAULT_EXECUTABLE.parent,
                    runner.DEFAULT_EXECUTABLE,
                    runner.DEFAULT_OPENFHE_PREFIX,
                    "after workload",
                )
        finally:
            self.provenance_mock = self.provenance_patcher.start()

    def test_crypto_preflight_parser_rejects_missing_record(self) -> None:
        with self.assertRaisesRegex(runner.ArtifactRunnerError, "exactly one"):
            runner._parse_crypto_preflight_output("17: unrelated CTest output\n")

    def test_crypto_preflight_parser_rejects_duplicate_record(self) -> None:
        record = _crypto_preflight_record()
        with self.assertRaisesRegex(runner.ArtifactRunnerError, "got 2"):
            runner._parse_crypto_preflight_output(
                _crypto_preflight_ctest_output(record, record)
            )

    def test_crypto_preflight_parser_rejects_tampered_contract(self) -> None:
        missing = _crypto_preflight_record()
        missing.pop("passed")
        with self.assertRaisesRegex(runner.ArtifactRunnerError, "keys differ"):
            runner._parse_crypto_preflight_output(
                _crypto_preflight_ctest_output(missing)
            )

        tampered = _crypto_preflight_record()
        tampered["additive_he_operations"] = 1
        with self.assertRaisesRegex(
            runner.ArtifactRunnerError, "additive_he_operations"
        ):
            runner._parse_crypto_preflight_output(
                _crypto_preflight_ctest_output(tampered)
            )

    def test_crypto_preflight_parser_rejects_actual_scale_outside_tolerance(
        self,
    ) -> None:
        with self.assertRaisesRegex(runner.ArtifactRunnerError, "more than 1e-3"):
            runner._parse_crypto_preflight_output(
                _crypto_preflight_ctest_output(
                    _crypto_preflight_record(scale_bits=50.0011)
                )
            )

    def test_schedule_preflight_parser_requires_exactly_one_record(self) -> None:
        with self.assertRaisesRegex(runner.ArtifactRunnerError, "got 0"):
            runner._parse_schedule_preflight_output(
                _crypto_preflight_ctest_output(_crypto_preflight_record())
            )
        record = _schedule_preflight_record()
        with self.assertRaisesRegex(runner.ArtifactRunnerError, "got 2"):
            runner._parse_schedule_preflight_output(
                _crypto_preflight_ctest_output(record, record)
            )

    def test_schedule_preflight_parser_rejects_unsealed_or_unverified(self) -> None:
        for key in ("formal_schedule_sealed", "he_metadata_verified_by_preflight"):
            record = _schedule_preflight_record()
            record[key] = True
            with (
                self.subTest(key=key),
                self.assertRaisesRegex(runner.ArtifactRunnerError, key),
            ):
                runner._parse_schedule_preflight_output(
                    _crypto_preflight_ctest_output(record)
                )
        unsealed = _schedule_preflight_record()
        unsealed["formal_metadata_schedule_source"] = (
            "forged_formal_schedule_source"
        )
        with self.assertRaisesRegex(
            runner.ArtifactRunnerError,
            "formal_metadata_schedule_source",
        ):
            runner._parse_schedule_preflight_output(
                _crypto_preflight_ctest_output(unsealed)
            )

    def test_command_count_drift_is_fail_closed_and_removes_run_directory(self) -> None:
        patches = list(self._generation_patches())
        patches[3] = mock.patch.object(
            runner,
            "_run_m5_preflight",
            return_value=runner.M5Preflight(
                self._preflight_commands()[:2],
                _crypto_preflight_record(),
                self._build_configuration(),
                _schedule_preflight_record(),
            ),
        )
        config = self._config("unit-m5-command-drift")
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            patches[6],
            patches[7],
            patches[8],
            patches[9],
            patches[10],
            self.assertRaisesRegex(runner.ArtifactRunnerError, "12 records, got 10"),
        ):
            runner.generate_m5_artifact(config)
        self.assertFalse((self.output_root / config.run_id).exists())

    def test_input_hash_drift_after_run_is_fail_closed(self) -> None:
        patches = list(self._generation_patches())
        patches[8] = mock.patch.object(
            runner,
            "_require_input_hashes",
            side_effect=[
                None,
                runner.ArtifactRunnerError("synthetic input hash drift"),
            ],
        )
        config = self._config("unit-m5-input-hash-drift")
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            patches[6],
            patches[7],
            patches[8],
            patches[9],
            patches[10],
            self.assertRaisesRegex(runner.ArtifactRunnerError, "input hash drift"),
        ):
            runner.generate_m5_artifact(config)
        self.assertFalse((self.output_root / config.run_id).exists())

    def test_executable_hash_drift_after_run_is_fail_closed(self) -> None:
        patches = list(self._generation_patches())
        executable_hash = runner._sha256(self.executable)
        patches.append(
            mock.patch.object(
                runner,
                "_require_executable_hash",
                side_effect=[
                    None,
                    runner.ArtifactRunnerError("synthetic executable hash drift"),
                ],
            )
        )
        config = self._config("unit-m5-executable-hash-drift")
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            patches[6],
            patches[7],
            patches[8],
            patches[9],
            patches[10],
            patches[11],
            self.assertRaisesRegex(runner.ArtifactRunnerError, "executable hash drift"),
        ):
            runner.generate_m5_artifact(config)
        self.assertEqual(len(executable_hash), 64)
        self.assertFalse((self.output_root / config.run_id).exists())

    def test_validator_failure_is_fail_closed(self) -> None:
        patches = list(self._generation_patches())
        patches[10] = mock.patch.object(
            runner,
            "_validate_artifact",
            side_effect=runner.ArtifactRunnerError("synthetic validator rejection"),
        )
        config = self._config("unit-m5-validator-rejection")
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            patches[6],
            patches[7],
            patches[8],
            patches[9],
            patches[10],
            self.assertRaisesRegex(runner.ArtifactRunnerError, "validator rejection"),
        ):
            runner.generate_m5_artifact(config)
        self.assertFalse((self.output_root / config.run_id).exists())

    def test_existing_run_directory_is_preserved(self) -> None:
        patches = self._generation_patches()
        config = self._config("unit-m5-existing")
        run_root = self.output_root / config.run_id
        run_root.mkdir(parents=True)
        sentinel = run_root / "sentinel"
        sentinel.write_text("preserve", encoding="utf-8")
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            patches[6],
            patches[7],
            patches[8],
            patches[9],
            patches[10],
            self.assertRaisesRegex(runner.ArtifactRunnerError, "already exists"),
        ):
            runner.generate_m5_artifact(config)
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "preserve")

    def test_git_preflight_rejects_dirty_branch_and_sha_drift_without_network(
        self,
    ) -> None:
        record = self._command_record("synthetic git command")
        remote_sha = "b" * 40
        cases = (
            (
                "dirty",
                [(" M scripts/run_openfhe_encoder12_artifact_v6.py", record)],
                "repository is not clean",
            ),
            (
                "wrong_branch",
                [("", record), ("main", record)],
                "branch must be",
            ),
            (
                "local_tracking_mismatch",
                [
                    ("", record),
                    (runner.BRANCH, record),
                    (self.head, record),
                    (remote_sha, record),
                    (runner.REMOTE_URL, record),
                    (f"{self.head}\t{runner.REMOTE_REF}", record),
                ],
                "commits differ",
            ),
            (
                "origin_url_spoof",
                [
                    ("", record),
                    (runner.BRANCH, record),
                    (self.head, record),
                    (self.head, record),
                    ("https://example.invalid/spoof.git", record),
                ],
                "URL must be exactly",
            ),
            (
                "tracking_live_mismatch",
                [
                    ("", record),
                    (runner.BRANCH, record),
                    (self.head, record),
                    (self.head, record),
                    (runner.REMOTE_URL, record),
                    (f"{remote_sha}\t{runner.REMOTE_REF}", record),
                ],
                "commits differ",
            ),
            (
                "local_live_mismatch",
                [
                    ("", record),
                    (runner.BRANCH, record),
                    (self.head, record),
                    (remote_sha, record),
                    (runner.REMOTE_URL, record),
                    (f"{remote_sha}\t{runner.REMOTE_REF}", record),
                ],
                "commits differ",
            ),
        )
        for name, side_effect, message in cases:
            with (
                self.subTest(case=name),
                mock.patch.object(
                    runner, "_run_git", side_effect=side_effect
                ) as run_git,
                self.assertRaisesRegex(runner.ArtifactRunnerError, message),
            ):
                runner._preflight_git()
            self.assertLessEqual(run_git.call_count, 6)

    def test_git_subprocess_is_absolute_and_scrubs_ambient_git_environment(
        self,
    ) -> None:
        poisoned = {
            "PATH": "/tmp/attacker-bin",
            "GIT_DIR": "/tmp/attacker-repository",
            "GIT_WORK_TREE": "/tmp/attacker-worktree",
            "GIT_CONFIG_GLOBAL": "/tmp/attacker-gitconfig",
            "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": "url.https://example.invalid/.insteadOf",
            "GIT_CONFIG_VALUE_0": "https://github.com/",
        }
        completed = subprocess.CompletedProcess(
            [str(runner.SYSTEM_GIT), "status"],
            0,
            stdout="clean\n",
            stderr="",
        )
        with (
            mock.patch.dict(os.environ, poisoned, clear=False),
            mock.patch.object(
                runner.subprocess,
                "run",
                return_value=completed,
            ) as run,
        ):
            output, _ = runner._run_git(["status"])
        self.assertEqual(output, "clean")
        command = run.call_args.args[0]
        environment = run.call_args.kwargs["env"]
        self.assertEqual(command[0], "/usr/bin/git")
        self.assertEqual(environment["PATH"], "/usr/bin:/bin")
        self.assertEqual(environment["LANG"], "C")
        self.assertEqual(environment["LC_ALL"], "C")
        self.assertFalse(any(name.startswith("GIT_") for name in environment))
        self.assertIs(run.call_args.kwargs["stdin"], subprocess.DEVNULL)

    def test_schema_binding_snapshot_rejects_head_blob_drift(self) -> None:
        def git_output(arguments: list[str]) -> tuple[str, dict[str, object]]:
            if arguments[0] == "rev-parse":
                return "c" * 40, {}
            if arguments[0] == "hash-object":
                return "d" * 40, {}
            raise AssertionError(arguments)

        with (
            mock.patch.object(runner, "_run_git", side_effect=git_output),
            self.assertRaisesRegex(runner.ArtifactRunnerError, "HEAD blob"),
        ):
            ORIGINAL_SCHEMA_BINDING_SNAPSHOT(self.head, "synthetic preflight")

    def test_schema_binding_change_during_run_removes_bundle(self) -> None:
        initial = self._schema_binding()
        changed = copy.deepcopy(initial)
        changed["runner"]["sha256"] = "0" * 64
        self.binding_mock.side_effect = [initial, changed]
        config = self._config("unit-m5-v6-binding-drift")
        patches = self._generation_patches()
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            patches[6],
            patches[7],
            patches[8],
            patches[9],
            patches[10],
            self.assertRaisesRegex(runner.ArtifactRunnerError, "binding changed"),
        ):
            runner.generate_m5_artifact(config)
        self.assertFalse((self.output_root / config.run_id).exists())

    def test_actual_trace_inventory_binds_448_inputs_and_unique_layer_bundles(
        self,
    ) -> None:
        records, identities = runner._trace_input_records(runner.DEFAULT_DATA_ROOT)
        self.assertEqual(len(records), 444)
        self.assertEqual(len({record["path"] for record in records}), 444)
        self.assertEqual(len(identities), 12)
        self.assertEqual(
            {
                (item["weight_file_count"], item["trace_file_count"])
                for item in identities
            },
            {(16, 21)},
        )
        self.assertEqual(len({item["weight_bundle_sha256"] for item in identities}), 12)
        self.assertEqual(len({item["trace_bundle_sha256"] for item in identities}), 12)

    def test_runtime_records_are_strict_and_fail_closed(self) -> None:
        layers = [_layer_record(layer_id) for layer_id in range(12)]
        with (
            mock.patch.object(
                runner,
                "SOFTMAX_CHECKPOINT_METADATA",
                TEST_SOFTMAX_CHECKPOINT_METADATA,
            ),
            mock.patch.object(
                runner,
                "LAYER0_INPUT_METADATA",
                TEST_LAYER0_INPUT_METADATA,
            ),
            mock.patch.object(
                runner,
                "LAYER_HANDOFF_INPUT_METADATA",
                TEST_HANDOFF_INPUT_METADATA,
            ),
        ):
            for layer_id, record in enumerate(layers):
                _validate_layer_record(record, layer_id)
            runner._validate_final_record(_final_record(layers), layers)

            missing_checkpoint_count = copy.deepcopy(layers[5])
            missing_checkpoint_count.pop("inactive_zero_checkpoint_count")
            with self.assertRaisesRegex(runner.ArtifactRunnerError, "keys differ"):
                _validate_layer_record(missing_checkpoint_count, 5)

            for invalid_count in (27, True):
                invalid_checkpoint_count = copy.deepcopy(layers[5])
                invalid_checkpoint_count["inactive_zero_checkpoint_count"] = (
                    invalid_count
                )
                with (
                    self.subTest(inactive_zero_checkpoint_count=invalid_count),
                    self.assertRaisesRegex(
                        runner.ArtifactRunnerError,
                        "inactive_zero_checkpoint_count",
                    ),
                ):
                    _validate_layer_record(invalid_checkpoint_count, 5)

        bad_layer = copy.deepcopy(layers[4])
        bad_layer["chain_input_source"] = "trace_reset"
        with (
            mock.patch.object(
                runner,
                "SOFTMAX_CHECKPOINT_METADATA",
                TEST_SOFTMAX_CHECKPOINT_METADATA,
            ),
            mock.patch.object(
                runner,
                "LAYER_HANDOFF_INPUT_METADATA",
                TEST_HANDOFF_INPUT_METADATA,
            ),
            self.assertRaisesRegex(runner.ArtifactRunnerError, "chain_input_source"),
        ):
            _validate_layer_record(bad_layer, 4)

        bad_range = copy.deepcopy(layers[7])
        bad_range["encrypted_polynomial_input_ranges"]["gelu_input"][1] = 128.1
        with (
            mock.patch.object(
                runner,
                "SOFTMAX_CHECKPOINT_METADATA",
                TEST_SOFTMAX_CHECKPOINT_METADATA,
            ),
            mock.patch.object(
                runner,
                "LAYER_HANDOFF_INPUT_METADATA",
                TEST_HANDOFF_INPUT_METADATA,
            ),
            self.assertRaisesRegex(runner.ArtifactRunnerError, "gelu_input"),
        ):
            _validate_layer_record(bad_range, 7)

        bad_final = _final_record(layers)
        bad_final["timing_claim"] = True
        with self.assertRaisesRegex(runner.ArtifactRunnerError, "timing_claim"):
            runner._validate_final_record(bad_final, layers)

        bad_final = _final_record(layers)
        bad_final["inactive_sentinel_max_error"] = layers[-1][
            "inactive_sentinel_max_error"
        ]
        with self.assertRaisesRegex(runner.ArtifactRunnerError, "12-layer maximum"):
            runner._validate_final_record(bad_final, layers)

        bad_final = _final_record(layers)
        bad_final["inactive_sentinel_range_status"] = "unchecked"
        with self.assertRaisesRegex(
            runner.ArtifactRunnerError, "inactive_sentinel_range_status"
        ):
            runner._validate_final_record(bad_final, layers)

    def test_zero_inactive_and_polynomial_sentinel_contracts_are_distinct(self) -> None:
        layer = _layer_record(5)
        self.assertGreater(float(layer["inactive_sentinel_max_error"]), 1e-6)
        with (
            mock.patch.object(
                runner,
                "SOFTMAX_CHECKPOINT_METADATA",
                TEST_SOFTMAX_CHECKPOINT_METADATA,
            ),
            mock.patch.object(
                runner,
                "LAYER_HANDOFF_INPUT_METADATA",
                TEST_HANDOFF_INPUT_METADATA,
            ),
        ):
            _validate_layer_record(layer, 5)

            relaxed_prototype_tail = copy.deepcopy(layer)
            relaxed_prototype_tail["inactive_max_abs"] = 5e-4
            _validate_layer_record(relaxed_prototype_tail, 5)

            bad_zero_tail = copy.deepcopy(layer)
            bad_zero_tail["inactive_max_abs"] = 1.1e-3
            with self.assertRaisesRegex(runner.ArtifactRunnerError, "inactive_max_abs"):
                _validate_layer_record(bad_zero_tail, 5)

            for invalid_diagnostic in (-1.0, float("inf")):
                bad_diagnostic = copy.deepcopy(layer)
                bad_diagnostic["inactive_sentinel_max_error"] = invalid_diagnostic
                with (
                    self.subTest(invalid_diagnostic=invalid_diagnostic),
                    self.assertRaisesRegex(
                        runner.ArtifactRunnerError, "inactive_sentinel_max_error"
                    ),
                ):
                    _validate_layer_record(bad_diagnostic, 5)

            for key in sorted(runner.INACTIVE_SENTINEL_RANGE_KEYS):
                for endpoint, invalid_value in (
                    (0, runner.FROZEN_RANGES[key][0] - 0.1),
                    (1, runner.FROZEN_RANGES[key][1] + 0.1),
                ):
                    bad_interval = copy.deepcopy(layer)
                    bad_interval["inactive_polynomial_sentinel_ranges"][key][
                        endpoint
                    ] = invalid_value
                    with (
                        self.subTest(checkpoint=key, endpoint=endpoint),
                        self.assertRaisesRegex(
                            runner.ArtifactRunnerError,
                            f"inactive_polynomial_sentinel_ranges.{key}",
                        ),
                    ):
                        _validate_layer_record(bad_interval, 5)

            bad_status = copy.deepcopy(layer)
            bad_status["inactive_sentinel_range_status"] = "unchecked"
            with self.assertRaisesRegex(
                runner.ArtifactRunnerError, "inactive_sentinel_range_status"
            ):
                _validate_layer_record(bad_status, 5)

    def test_frozen_scalars_reject_python_bool_integer_equivalence(self) -> None:
        layer = _layer_record(0)
        layer["server_decryptions"] = False
        with (
            mock.patch.object(
                runner,
                "SOFTMAX_CHECKPOINT_METADATA",
                TEST_SOFTMAX_CHECKPOINT_METADATA,
            ),
            mock.patch.object(
                runner,
                "LAYER0_INPUT_METADATA",
                TEST_LAYER0_INPUT_METADATA,
            ),
            self.assertRaisesRegex(runner.ArtifactRunnerError, "server_decryptions"),
        ):
            _validate_layer_record(layer, 0)

        layers = [_layer_record(layer_id) for layer_id in range(12)]
        final = _final_record(layers)
        final["client_encrypt_calls"] = True
        with self.assertRaisesRegex(runner.ArtifactRunnerError, "client_encrypt_calls"):
            runner._validate_final_record(final, layers)

        metadata = dict(runner.RAW_OUTPUT_METADATA)
        metadata["level"] = False
        with self.assertRaisesRegex(runner.ArtifactRunnerError, "must be an integer"):
            runner._validate_metadata(metadata, runner.RAW_OUTPUT_METADATA, "synthetic")

    def test_metadata_scale_is_measured_with_a_frozen_expectation(self) -> None:
        observed = dict(runner.RAW_OUTPUT_METADATA)
        observed["scale_bits"] = 100.0005
        runner._validate_metadata(observed, runner.RAW_OUTPUT_METADATA, "synthetic")

        observed["scale_bits"] = 100.0011
        with self.assertRaisesRegex(runner.ArtifactRunnerError, "more than 1e-3"):
            runner._validate_metadata(observed, runner.RAW_OUTPUT_METADATA, "synthetic")

        observed = dict(runner.RAW_OUTPUT_METADATA)
        observed["expected_scale_bits"] = 99
        with self.assertRaisesRegex(runner.ArtifactRunnerError, "expected_scale_bits"):
            runner._validate_metadata(observed, runner.RAW_OUTPUT_METADATA, "synthetic")

    def test_metadata_schedule_uses_layer0_and_post_refresh_regimes(self) -> None:
        layer0 = _layer_record(0)
        layer1 = _layer_record(1)
        _validate_layer_record(layer0, 0)
        _validate_layer_record(layer1, 1)
        self.assertEqual(layer0["attention_output_metadata"]["level"], 40)
        self.assertEqual(layer1["attention_output_metadata"]["level"], 31)
        self.assertEqual(layer0["ln1_output_metadata"]["level"], 32)
        self.assertEqual(layer1["ln1_output_metadata"]["level"], 30)
        self.assertEqual(layer0["ffn_output_metadata"]["level"], 45)
        self.assertEqual(layer1["ffn_output_metadata"]["level"], 43)
        self.assertEqual(layer0["raw_output_metadata"]["level"], 30)
        self.assertEqual(layer1["raw_output_metadata"]["level"], 30)

        schedule = runner._metadata_schedule_contract()
        self.assertEqual(
            schedule["relative_used_level_deltas"],
            {
                "input_to_softmax_checkpoint_net_recovered": {
                    "layer_0": 10,
                    "layers_1_to_11": 1,
                },
                "softmax_checkpoint_to_attention_output_consumed": {
                    "layer_0": 22,
                    "layers_1_to_11": 13,
                },
                "attention_output_to_ln1_checkpoint_net_recovered": {
                    "layer_0": 21,
                    "layers_1_to_11": 12,
                },
                "ln1_checkpoint_to_ln1_output_consumed": {
                    "layer_0": 13,
                    "layers_1_to_11": 11,
                },
                "ln1_output_to_ffn_output_consumed": {
                    "layer_0": 13,
                    "layers_1_to_11": 13,
                },
                "ffn_output_to_ln2_checkpoint_net_recovered": {
                    "layer_0": 26,
                    "layers_1_to_11": 24,
                },
                "ln2_checkpoint_to_raw_output_consumed": {
                    "layer_0": 11,
                    "layers_1_to_11": 11,
                },
                "previous_raw_output_to_input_recovered": {
                    "layer_0": None,
                    "layers_1_to_11": 11,
                },
            },
        )

    def test_interlayer_recovery_requires_previous_raw_output_metadata(self) -> None:
        layer1 = _layer_record(1)
        with self.assertRaisesRegex(
            runner.ArtifactRunnerError,
            "previous_raw_output_metadata is required",
        ):
            runner._validate_layer_record(layer1, 1)
        _validate_layer_record(
            layer1,
            1,
            dict(_layer_record(0)["raw_output_metadata"]),
        )

        for field, layer0_metadata in (
            ("attention_output_metadata", runner.LAYER0_ATTENTION_OUTPUT_METADATA),
            ("ln1_output_metadata", runner.LAYER0_LN1_OUTPUT_METADATA),
            ("ffn_output_metadata", runner.LAYER0_FFN_OUTPUT_METADATA),
        ):
            with self.subTest(field=field):
                forged = copy.deepcopy(layer1)
                forged[field] = dict(layer0_metadata)
                with self.assertRaisesRegex(
                    runner.ArtifactRunnerError,
                    rf"{field}\.level drifted",
                ):
                    _validate_layer_record(forged, 1)

    def test_metadata_remaining_levels_is_derived_from_used_level(self) -> None:
        forged = copy.deepcopy(_layer_record(2))
        forged["ffn_output_metadata"]["remaining_levels"] = 2
        with self.assertRaisesRegex(
            runner.ArtifactRunnerError,
            "remaining_levels drifted",
        ):
            _validate_layer_record(forged, 2)

    def test_each_relative_used_level_delta_rejects_self_consistent_tamper(
        self,
    ) -> None:
        layer = _layer_record(1)
        cases = (
            (
                "input_to_softmax_checkpoint_net_recovered",
                "input_metadata",
                "softmax_denominator_metadata",
            ),
            (
                "softmax_checkpoint_to_attention_output_consumed",
                "attention_output_metadata",
                "softmax_denominator_metadata",
            ),
            (
                "attention_output_to_ln1_checkpoint_net_recovered",
                "attention_output_metadata",
                "ln1_variance_metadata",
            ),
            (
                "ln1_checkpoint_to_ln1_output_consumed",
                "ln1_output_metadata",
                "ln1_variance_metadata",
            ),
            (
                "ln1_output_to_ffn_output_consumed",
                "ffn_output_metadata",
                "ln1_output_metadata",
            ),
            (
                "ffn_output_to_ln2_checkpoint_net_recovered",
                "ffn_output_metadata",
                "ln2_variance_metadata",
            ),
            (
                "ln2_checkpoint_to_raw_output_consumed",
                "raw_output_metadata",
                "ln2_variance_metadata",
            ),
            (
                "previous_raw_output_to_input_recovered",
                "raw_output_metadata",
                "input_metadata",
            ),
        )
        for delta_name, output_name, anchor_name in cases:
            with self.subTest(delta=delta_name):
                output = copy.deepcopy(layer[output_name])
                output["level"] -= 1
                output["remaining_levels"] += 1
                runner._validate_metadata(output, output, "self-consistent tamper")
                with self.assertRaisesRegex(
                    runner.ArtifactRunnerError,
                    "used-level delta drifted",
                ):
                    expected_delta = runner.RELATIVE_USED_LEVEL_DELTAS[delta_name]
                    if isinstance(expected_delta, dict):
                        expected_delta = expected_delta["layers_1_to_11"]
                    runner._validate_used_level_delta(
                        output,
                        layer[anchor_name],
                        expected_delta,
                        delta_name,
                    )

    def test_post_bootstrap_checkpoint_anchor_drift_is_rejected(self) -> None:
        forged = copy.deepcopy(_layer_record(3))
        forged["softmax_denominator_metadata"]["level"] = 19
        forged["softmax_denominator_metadata"]["remaining_levels"] = 27
        with self.assertRaisesRegex(
            runner.ArtifactRunnerError,
            "softmax_denominator_metadata.level drifted",
        ):
            _validate_layer_record(forged, 3)

    def test_checkpoint_metadata_sha256_covers_every_declared_field(self) -> None:
        layers = [_layer_record(layer_id) for layer_id in range(12)]
        baseline = runner._checkpoint_metadata_sha256(layers)
        self.assertEqual(
            runner._metadata_schedule_contract()["checkpoint_metadata_digest"][
                "fields"
            ],
            list(runner.CHECKPOINT_METADATA_HASH_FIELDS),
        )
        for field in runner.CHECKPOINT_METADATA_FIELDS:
            with self.subTest(field=field):
                tampered = copy.deepcopy(layers)
                tampered[4][field]["scale_bits"] += 1e-7
                self.assertNotEqual(
                    runner._checkpoint_metadata_sha256(tampered),
                    baseline,
                )

    def test_nonzero_subprocess_is_fail_closed(self) -> None:
        completed = subprocess.CompletedProcess(
            args=[str(self.executable)],
            returncode=7,
            stdout="",
            stderr="synthetic child failure",
        )
        with (
            mock.patch.object(
                runner,
                "SOFTMAX_CHECKPOINT_METADATA",
                TEST_SOFTMAX_CHECKPOINT_METADATA,
            ),
            mock.patch.object(
                runner,
                "LAYER0_INPUT_METADATA",
                TEST_LAYER0_INPUT_METADATA,
            ),
            mock.patch.object(
                runner,
                "LAYER_HANDOFF_INPUT_METADATA",
                TEST_HANDOFF_INPUT_METADATA,
            ),
            mock.patch.object(runner.subprocess, "run", return_value=completed),
            self.assertRaisesRegex(runner.ArtifactRunnerError, "exited with 7"),
        ):
            runner._run_once(self._config("unit-m5-child-failure"))

    def test_successful_subprocess_stderr_is_fail_closed(self) -> None:
        completed = subprocess.CompletedProcess(
            args=[str(self.executable)],
            returncode=0,
            stdout="",
            stderr="synthetic OpenFHE warning",
        )
        with (
            mock.patch.object(
                runner,
                "SOFTMAX_CHECKPOINT_METADATA",
                TEST_SOFTMAX_CHECKPOINT_METADATA,
            ),
            mock.patch.object(
                runner,
                "LAYER0_INPUT_METADATA",
                TEST_LAYER0_INPUT_METADATA,
            ),
            mock.patch.object(
                runner,
                "LAYER_HANDOFF_INPUT_METADATA",
                TEST_HANDOFF_INPUT_METADATA,
            ),
            mock.patch.object(runner.subprocess, "run", return_value=completed),
            self.assertRaisesRegex(
                runner.ArtifactRunnerError, "stderr despite exit_code=0"
            ),
        ):
            runner._run_once(self._config("unit-m5-child-stderr"))

    def test_child_stderr_failure_removes_unsealed_artifact_directory(self) -> None:
        patches = list(self._generation_patches())
        patches[9] = mock.patch.object(
            runner,
            "_run_once",
            side_effect=runner.ArtifactRunnerError(
                "M5 correctness run emitted stderr despite exit_code=0"
            ),
        )
        config = self._config("unit-m5-child-stderr-cleanup")
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            patches[6],
            patches[7],
            patches[8],
            patches[9],
            patches[10],
            self.assertRaisesRegex(runner.ArtifactRunnerError, "stderr"),
        ):
            runner.generate_m5_artifact(config)
        self.assertFalse((self.output_root / config.run_id).exists())

    def test_missing_runtime_records_are_fail_closed(self) -> None:
        def synthetic_run(command: list[str], **_: object) -> object:
            output_argument = next(
                item for item in command if item.startswith("--output=")
            )
            Path(output_argument.split("=", maxsplit=1)[1]).write_text(
                "2048\n", encoding="utf-8"
            )
            return subprocess.CompletedProcess(
                args=command,
                returncode=0,
                stdout='{"test":"wrong_record"}\n',
                stderr="",
            )

        with (
            mock.patch.object(
                runner,
                "SOFTMAX_CHECKPOINT_METADATA",
                TEST_SOFTMAX_CHECKPOINT_METADATA,
            ),
            mock.patch.object(
                runner,
                "LAYER0_INPUT_METADATA",
                TEST_LAYER0_INPUT_METADATA,
            ),
            mock.patch.object(
                runner,
                "LAYER_HANDOFF_INPUT_METADATA",
                TEST_HANDOFF_INPUT_METADATA,
            ),
            mock.patch.object(runner.subprocess, "run", side_effect=synthetic_run),
            self.assertRaisesRegex(
                runner.ArtifactRunnerError,
                "unexpected JSON evidence record",
            ),
        ):
            runner._run_once(self._config("unit-m5-missing-records"))

    def test_each_unsealed_metadata_contract_stops_before_subprocess(self) -> None:
        sealed_schedule = {
            "LAYER0_INPUT_METADATA": TEST_LAYER0_INPUT_METADATA,
            "LAYER_HANDOFF_INPUT_METADATA": TEST_HANDOFF_INPUT_METADATA,
            "SOFTMAX_CHECKPOINT_METADATA": TEST_SOFTMAX_CHECKPOINT_METADATA,
            "LAYERNORM_CHECKPOINT_METADATA": TEST_LAYERNORM_CHECKPOINT_METADATA,
        }
        for unsealed_name in sealed_schedule:
            schedule = dict(sealed_schedule)
            schedule[unsealed_name] = None
            with (
                self.subTest(unsealed=unsealed_name),
                mock.patch.multiple(runner, **schedule),
                mock.patch.object(runner.subprocess, "run") as child,
                self.assertRaisesRegex(runner.ArtifactRunnerError, "not sealed"),
            ):
                runner._run_once(self._config(f"unit-m5-unsealed-{unsealed_name}"))
            child.assert_not_called()


if __name__ == "__main__":
    unittest.main()
