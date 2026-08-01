#!/usr/bin/env python3
"""Positive and tamper-negative tests for the independent M5 validator."""

from __future__ import annotations

import copy
import csv
import hashlib
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
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import validate_openfhe_m5_artifact_v6 as validator  # noqa: E402


ORIGINAL_SHA256_FILE = validator.sha256_file
TRACE_CONTRACT = validator.load_json(REPO_ROOT / "config/moai_encoder_trace.json")
TRACE_HASHES: dict[str, str] = {}
for _layer in TRACE_CONTRACT["layers"]:
    for _name, _specification in TRACE_CONTRACT["required_files"].items():
        _path = (
            Path("data") / f"layer_{_layer['layer_id']}" / _specification["path"]
        ).as_posix()
        TRACE_HASHES[_path] = _layer["sha256"][_name]


def _crypto_preflight_record(
    scale_bits: float = 50.000000079945785,
) -> dict[str, object]:
    if validator.LAYER0_INPUT_METADATA is None:  # pragma: no cover - production gate
        raise RuntimeError("layer-0 metadata must be sealed for the validator fixture")
    metadata = copy.deepcopy(validator.LAYER0_INPUT_METADATA)
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
        "source_path": validator.TRACE_SCALE_SOURCE_PATH,
        "json_locator": validator.TRACE_SCALE_LOCATOR,
        "contract_id": validator.TRACE_SCALE_CONTRACT_ID,
        "contract_sha256": validator.TRACE_SCALE_CONTRACT_SHA256,
        "values_sha256": validator.TRACE_SCALE_VALUES_SHA256,
        "raw_variance_sha256": validator.TRACE_SCALE_RAW_VARIANCE_SHA256,
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
        for name in validator.EXACT3_CHECKSUM_PATHS
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
    path = repository_root / validator.PROFILE_PATH
    profile = json.loads(path.read_text(encoding="utf-8"))
    profile["m5_schedule_candidate"]["exact3_seal"]["evidence"] = copy.deepcopy(
        evidence
    )
    path.write_text(
        json.dumps(profile, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


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
        metadata = {
            "input_metadata": copy.deepcopy(
                validator.LAYER0_INPUT_METADATA
                if layer_id == 0
                else validator.LAYER_HANDOFF_INPUT_METADATA
            ),
            "raw_output_metadata": copy.deepcopy(validator.RAW_OUTPUT_METADATA),
            "softmax_denominator_metadata": copy.deepcopy(
                validator.SOFTMAX_CHECKPOINT_METADATA
            ),
            "attention_output_metadata": copy.deepcopy(
                validator.LAYER0_ATTENTION_OUTPUT_METADATA
                if layer_id == 0
                else validator.POST_REFRESH_ATTENTION_OUTPUT_METADATA
            ),
            "ln1_variance_metadata": copy.deepcopy(
                validator.LAYERNORM_CHECKPOINT_METADATA
            ),
            "ln1_output_metadata": copy.deepcopy(
                validator.LAYER0_LN1_OUTPUT_METADATA
                if layer_id == 0
                else validator.POST_REFRESH_LN1_OUTPUT_METADATA
            ),
            "ffn_output_metadata": copy.deepcopy(
                validator.LAYER0_FFN_OUTPUT_METADATA
                if layer_id == 0
                else validator.POST_REFRESH_FFN_OUTPUT_METADATA
            ),
            "ln2_variance_metadata": copy.deepcopy(
                validator.LAYERNORM_CHECKPOINT_METADATA
            ),
        }
        metadata_names = {
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
            name: int(metadata[field]["level"])
            + int(metadata[field]["noise_scale_degree"])
            - 1
            for name, field in metadata_names.items()
        }
        previous_recovered = (
            None
            if previous is None
            else int(previous["raw_output_metadata"]["level"])
            + int(previous["raw_output_metadata"]["noise_scale_degree"])
            - 1
            - used["input"]
        )
        quality = {
            "relative_l2": 0.004 + layer_id * 0.0001,
            "cosine": 0.9999,
            "max_absolute": 0.02,
        }
        sentinel_ranges = {
            "softmax_denominator": [0.98, 1.02],
            "ln1_normalized_variance": [45.0, 90.0],
            "ln2_normalized_variance": [32.0, 128.0],
        }
        sentinel_maximum = max(
            abs(float(endpoint) - 1.0)
            for observed in sentinel_ranges.values()
            for endpoint in observed
        )
        layer = {
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
            **metadata,
            "metadata_used_levels": used,
            "metadata_used_level_deltas": {
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
            },
            "input_quality": copy.deepcopy(quality),
            "output_quality": copy.deepcopy(quality),
            "exact_trace_diagnostic": copy.deepcopy(quality),
            "inactive_max_abs": 2e-7 + layer_id * 1e-8,
            "inactive_zero_checkpoint_count": 28,
            "inactive_sentinel_max_error": sentinel_maximum,
            "inactive_polynomial_sentinel_ranges": sentinel_ranges,
            "inactive_sentinel_range_status": "passed",
            "encrypted_polynomial_input_ranges": {
                "softmax_shifted_logits": [-10.0, 1.0],
                "softmax_denominator": [1.0, 60.0],
                "ln1_normalized_variance": [1.0, 70.0],
                "gelu_input": [-60.0, 120.0],
                "ln2_normalized_variance": [0.8, 1200.0],
            },
            "refresh_operation_counts": copy.deepcopy(
                validator.REFRESH_COUNTS if layer_id < 2 else validator.ZERO_COUNTS
            ),
            "layer_operation_counts": copy.deepcopy(validator.LAYER_COUNTS),
            "cumulative_operation_counts": validator._exact3_expected_counts(
                layer_id
            ),
            "range_validation_owner": "client",
            "checkpoint_decryption_owner": "client",
            "server_decryptions": 0,
            "server_plaintext_activations": False,
            "finite": True,
            "range_status": "passed",
            "diagnostic_gates_passed": True,
        }
        layers.append(layer)
        previous = layer
    last = layers[-1]
    summary: dict[str, object] = {
        "test": "openfhe_encoder_exact_prefix",
        "profile": "paper_compat",
        "security_claim": "none",
        "parameter_sha256": validator.PROFILE_SHA256,
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
        "fixture_load_oracle_ms": 1.0,
        "setup_keygen_ms": 2.0,
        "client_encrypt_ms": 3.0,
        "server_online_diagnostic_ms": 4.0,
        "client_checkpoint_validate_ms": 5.0,
        "relative_l2": last["output_quality"]["relative_l2"],
        "cosine": last["output_quality"]["cosine"],
        "max_absolute": last["output_quality"]["max_absolute"],
        "inactive_max_abs": max(layer["inactive_max_abs"] for layer in layers),
        "inactive_sentinel_max_error": max(
            layer["inactive_sentinel_max_error"] for layer in layers
        ),
        "inactive_sentinel_range_status": "all_client_validated",
        "final_metadata": copy.deepcopy(last["raw_output_metadata"]),
        "operation_counts": copy.deepcopy(validator.EXACT3_FINAL_COUNTS),
        "multiplicative_depth": 47,
        "max_observed_level": 45,
        "max_polynomial_depth": 10,
        "peak_rss_bytes": 2048 * 1024,
        "timing_claim": False,
        "latency_kind": "non_benchmark_diagnostic",
        "finite": True,
        "passed": True,
        "diagnostic_gates_passed": True,
    }
    command_argv = [
        f"./{validator.EXACT3_EXECUTABLE_PATH}",
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
        validator.EXACT3_PROFILE_WARNING
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
    metrics = validator._exact3_metrics_bytes(layers, summary, timing)
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
        for path in validator.EXACT3_SOURCE_PATHS
    ]
    candidate_profile = {
        "validation_status": validator.PROFILE_TRANSITION_PRE_STATE_VALUES[
            "/validation_status"
        ],
        "m5_schedule_candidate": {
            "status": validator.PROFILE_TRANSITION_PRE_STATE_VALUES[
                "/m5_schedule_candidate/status"
            ],
            "source": validator.PROFILE_TRANSITION_PRE_STATE_VALUES[
                "/m5_schedule_candidate/source"
            ],
            "formal_schedule_sealed": False,
            "calibrated_layers": [0, 1],
        },
        "feature_packed_layernorm_override": {
            "schedule_status": validator.PROFILE_TRANSITION_PRE_STATE_VALUES[
                "/feature_packed_layernorm_override/schedule_status"
            ]
        },
    }
    config_files = []
    for path in validator.EXACT3_CONFIG_PATHS:
        payload = (
            json.dumps(candidate_profile, indent=2, sort_keys=True).encode() + b"\n"
            if path == validator.PROFILE_PATH
            else f"synthetic config: {path}\n".encode()
        )
        config_files.append(write_provenance_file(path, payload))
    executable_record = write_provenance_file(
        validator.EXACT3_EXECUTABLE_PATH,
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
        "schema_id": validator.EXACT3_MANIFEST_SCHEMA_ID,
        "schema_version": validator.EXACT3_MANIFEST_SCHEMA_VERSION,
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
            "run_id_label_semantics": validator.EXACT3_RUN_ID_LABEL_SEMANTICS,
            "execution_started_at": "2026-07-31T11:59:58.000000+09:00",
            "execution_finished_at": "2026-07-31T11:59:59.000000+09:00",
            "execution_timestamp_semantics": (
                validator.EXACT3_EXECUTION_TIMESTAMP_SEMANTICS
            ),
            "raw_log_filesystem_timestamp_semantics": (
                validator.EXACT3_RAW_TIMESTAMP_SEMANTICS
            ),
            "sealed_at": "2026-07-31T12:00:00+09:00",
            "sealed_at_semantics": validator.EXACT3_SEALED_AT_SEMANTICS,
        },
        "git": {
            "head": "a" * 40,
            "branch": validator.BRANCH,
            "detached": False,
            "clean": True,
            "snapshot_semantics": validator.EXACT3_GIT_SNAPSHOT_SEMANTICS,
            "status_porcelain_v1_z_sha256": empty_hash,
            "tracked_diff": {
                "command": "git diff --binary HEAD -- .",
                "sha256": empty_hash,
                "size_bytes": 0,
            },
            "untracked_source_manifest": {
                "selection": validator.EXACT3_GIT_UNTRACKED_SELECTION,
                "canonicalization": validator.EXACT3_GIT_CANONICALIZATION,
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
                f"./{validator.EXACT3_EXECUTABLE_PATH}",
                "--data-root",
                str(repository_root.resolve() / "data"),
                "--diagnostic-layer-count",
                "3",
            ],
            "runtime_argv": [
                f"./{validator.EXACT3_EXECUTABLE_PATH}",
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
            "parameter_sha256": validator.PROFILE_SHA256,
            "security_claim": "none",
            "warning": validator.WARNING,
        },
        "provenance": {
            "snapshot_semantics": validator.EXACT3_PROVENANCE_SNAPSHOT_SEMANTICS,
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
                "path": validator.PROFILE_PATH,
                "canonicalization": validator.PROFILE_TRANSITION_CANONICALIZATION,
                "allowed_json_pointers": list(
                    validator.PROFILE_TRANSITION_ALLOWED_JSON_POINTERS
                ),
                "pre_state": {
                    "values": copy.deepcopy(
                        validator.PROFILE_TRANSITION_PRE_STATE_VALUES
                    ),
                    "absent": list(validator.PROFILE_TRANSITION_PRE_STATE_ABSENT),
                },
                "pre_seal_file_sha256": next(
                    record["sha256"]
                    for record in config_files
                    if record["path"] == validator.PROFILE_PATH
                ),
                "pre_seal_size_bytes": next(
                    record["size_bytes"]
                    for record in config_files
                    if record["path"] == validator.PROFILE_PATH
                ),
                "immutable_projection_sha256": (
                    validator._profile_immutable_projection_sha256(candidate_profile)
                ),
                "application_order": (
                    validator.PROFILE_TRANSITION_APPLICATION_ORDER
                ),
                "post_state_contract": {
                    "values": copy.deepcopy(
                        validator.PROFILE_TRANSITION_POST_STATE_VALUES
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
        "gate_contract": copy.deepcopy(validator.EXACT3_GATE_CONTRACT),
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
        record for record in config_files if record["path"] == validator.PROFILE_PATH
    )
    sealed_profile = {
        "validation_status": validator.PROFILE_VALIDATION_STATUS_SEALED,
        "m5_schedule_candidate": {
            "status": validator.PROFILE_CANDIDATE_SEALED_STATUS,
            "source": validator.PROFILE_SCHEDULE_SEALED_SOURCE,
            "formal_schedule_sealed": True,
            "calibrated_layers": [0, 1, 2],
            "exact3_seal": {
                "source": validator.PROFILE_SCHEDULE_SEALED_SOURCE,
                "evidence": copy.deepcopy(evidence),
                "pre_seal_profile": {
                    "path": preseal_profile["path"],
                    "sha256": preseal_profile["sha256"],
                    "size_bytes": preseal_profile["size_bytes"],
                    "transition": validator.PROFILE_PRESEAL_TRANSITION,
                },
            },
        },
        "feature_packed_layernorm_override": {"schedule_status": "sealed"},
    }
    (repository_root / validator.PROFILE_PATH).write_text(
        json.dumps(sealed_profile, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return evidence, bundle_root


def _sha256_without_large_csv_reads(path: Path) -> str:
    try:
        relative = path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        relative = ""
    if relative in TRACE_HASHES:
        return TRACE_HASHES[relative]
    return ORIGINAL_SHA256_FILE(path)


class M5ArtifactValidatorContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory(
            prefix="moai-m5-artifact-validator-"
        )
        self.output_root = Path(self.temporary_directory.name)
        self.run_id = "synthetic-m5-artifact-validator"
        self.artifact_root = self.output_root / self.run_id
        self.artifact_root.mkdir()
        self.manifest_path = self.artifact_root / "manifest.json"
        self.output_root_patcher = mock.patch.object(
            validator, "OUTPUT_ROOT", self.output_root
        )
        self.output_root_patcher.start()
        self.sha_patcher = mock.patch.object(
            validator, "sha256_file", side_effect=_sha256_without_large_csv_reads
        )
        self.sha_patcher.start()
        self.schema = validator.validate_schema(
            validator.load_json(REPO_ROOT / "docs/openfhe-m5-artifact-schema-v6.json")
        )
        self.layer_records = self._make_layer_records()
        self.summary = self._make_summary()
        self._write_stdout()
        self._write_metrics()
        self.manifest = self._make_manifest()
        self._reseal_artifacts()
        self._write_manifest()

    def tearDown(self) -> None:
        self.sha_patcher.stop()
        self.output_root_patcher.stop()
        self.temporary_directory.cleanup()

    @staticmethod
    def _metadata(value: dict[str, object]) -> dict[str, object]:
        return copy.deepcopy(value)

    @staticmethod
    def _quality(layer_id: int, offset: float = 0.0) -> dict[str, float]:
        return {
            "relative_l2": 0.001 + layer_id * 0.0001 + offset,
            "cosine": 0.99999 - layer_id * 0.000001,
            "max_absolute": 0.002 + layer_id * 0.0001,
        }

    def _make_layer_records(self) -> list[dict[str, object]]:
        records: list[dict[str, object]] = []
        for layer_id in validator.LAYER_IDS:
            refresh = (
                validator.REFRESH_COUNTS if layer_id < 11 else validator.ZERO_COUNTS
            )
            records.append(
                {
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
                    "input_metadata": self._metadata(
                        validator.LAYER0_INPUT_METADATA
                        if layer_id == 0
                        else validator.LAYER_HANDOFF_INPUT_METADATA
                    ),
                    "raw_output_metadata": self._metadata(
                        validator.RAW_OUTPUT_METADATA
                    ),
                    "softmax_denominator_metadata": self._metadata(
                        validator.SOFTMAX_CHECKPOINT_METADATA
                    ),
                    "attention_output_metadata": self._metadata(
                        validator.LAYER0_ATTENTION_OUTPUT_METADATA
                        if layer_id == 0
                        else validator.POST_REFRESH_ATTENTION_OUTPUT_METADATA
                    ),
                    "ln1_variance_metadata": self._metadata(
                        validator.LAYERNORM_CHECKPOINT_METADATA
                    ),
                    "ln1_output_metadata": self._metadata(
                        validator.LAYER0_LN1_OUTPUT_METADATA
                        if layer_id == 0
                        else validator.POST_REFRESH_LN1_OUTPUT_METADATA
                    ),
                    "ffn_output_metadata": self._metadata(
                        validator.LAYER0_FFN_OUTPUT_METADATA
                        if layer_id == 0
                        else validator.POST_REFRESH_FFN_OUTPUT_METADATA
                    ),
                    "ln2_variance_metadata": self._metadata(
                        validator.LAYERNORM_CHECKPOINT_METADATA
                    ),
                    "input_quality": self._quality(layer_id),
                    "output_quality": self._quality(layer_id, 0.0002),
                    "exact_trace_diagnostic": self._quality(layer_id, 0.0003),
                    "inactive_max_abs": 1e-7 + layer_id * 1e-9,
                    "inactive_zero_checkpoint_count": 28,
                    # Diagnostic only: a public polynomial sentinel is not an
                    # inactive zero slot and is not required to remain near 1.
                    "inactive_sentinel_max_error": 10.0 + layer_id,
                    "inactive_polynomial_sentinel_ranges": {
                        "softmax_denominator": [0.1, 70.0],
                        "ln1_normalized_variance": [45.0, 90.0],
                        "ln2_normalized_variance": [32.0, 128.0],
                    },
                    "inactive_sentinel_range_status": "passed",
                    "encrypted_polynomial_input_ranges": {
                        "softmax_shifted_logits": [-15.0, 4.0],
                        "softmax_denominator": [0.1, 70.0],
                        "ln1_normalized_variance": [0.6, 1400.0],
                        "gelu_input": [-70.0, 120.0],
                        "ln2_normalized_variance": [0.7, 1300.0],
                    },
                    "handoff_refresh_performed": layer_id < 11,
                    "refresh_operation_counts": copy.deepcopy(refresh),
                    "layer_operation_counts": copy.deepcopy(validator.LAYER_COUNTS),
                    "cumulative_operation_counts": validator._cumulative_counts(
                        layer_id
                    ),
                    "range_validation_owner": "client",
                    "checkpoint_decryption_owner": "client",
                    "server_decryptions": 0,
                    "server_plaintext_activations": False,
                    "finite": True,
                    "range_status": "passed",
                }
            )
        return records

    def _make_summary(self) -> dict[str, object]:
        final = self.layer_records[-1]
        return {
            "test": "openfhe_encoder_12_layer",
            "profile": "paper_compat",
            "security_claim": "none",
            "parameter_sha256": validator.PROFILE_SHA256,
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
            "fixture_load_oracle_ms": 100.0,
            "setup_keygen_ms": 200.0,
            "client_encrypt_ms": 30.0,
            "server_online_diagnostic_ms": 4000.0,
            "client_checkpoint_validate_ms": 500.0,
            "relative_l2": final["output_quality"]["relative_l2"],
            "cosine": final["output_quality"]["cosine"],
            "max_absolute": final["output_quality"]["max_absolute"],
            "inactive_max_abs": max(
                record["inactive_max_abs"] for record in self.layer_records
            ),
            "inactive_sentinel_max_error": max(
                record["inactive_sentinel_max_error"] for record in self.layer_records
            ),
            "inactive_sentinel_range_status": "all_client_validated",
            "final_metadata": copy.deepcopy(validator.RAW_OUTPUT_METADATA),
            "operation_counts": copy.deepcopy(validator.TOTAL_COUNTS),
            "multiplicative_depth": 47,
            "max_observed_level": 45,
            "max_polynomial_depth": 10,
            "peak_rss_bytes": 2_000_000,
            "timing_claim": False,
            "latency_kind": "non_benchmark_diagnostic",
            "finite": True,
            "passed": True,
        }

    def _write_stdout(self) -> None:
        records = [
            {"test": "openfhe_security_disclosure", "security_claim": "none"},
            *self.layer_records,
            self.summary,
        ]
        (self.artifact_root / "stdout.log").write_text(
            "\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n",
            encoding="utf-8",
        )

    def _write_metrics(self) -> None:
        final = self.summary["final_metadata"]
        row = {
            "run": 1,
            "exit_code": 0,
            "elapsed_seconds": 5.0,
            "peak_rss_kib": 2048,
            "fixture_load_oracle_ms": self.summary["fixture_load_oracle_ms"],
            "setup_keygen_ms": self.summary["setup_keygen_ms"],
            "client_encrypt_ms": self.summary["client_encrypt_ms"],
            "server_online_diagnostic_ms": self.summary["server_online_diagnostic_ms"],
            "client_checkpoint_validate_ms": self.summary[
                "client_checkpoint_validate_ms"
            ],
            "final_rel_l2": self.summary["relative_l2"],
            "final_cosine": self.summary["cosine"],
            "final_max_absolute": self.summary["max_absolute"],
            "inactive_max_abs": self.summary["inactive_max_abs"],
            "inactive_sentinel_max_error": self.summary["inactive_sentinel_max_error"],
            "checkpoint_metadata_sha256": validator.checkpoint_metadata_sha256(
                self.layer_records
            ),
            "final_level": final["level"],
            "final_noise_scale_degree": final["noise_scale_degree"],
            "final_remaining_levels": final["remaining_levels"],
            "final_scale_bits": final["scale_bits"],
            "final_ciphertext_count": final["ciphertext_count"],
            **self.summary["operation_counts"],
            "timing_claim": "false",
            "latency_kind": "non_benchmark_diagnostic",
            "finite": "true",
            "passed": "true",
        }
        with (self.artifact_root / "metrics.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=validator.CSV_FIELDS)
            writer.writeheader()
            writer.writerow(row)

    @staticmethod
    def _input_record(
        path: str, role: str, sha256: str | None = None
    ) -> dict[str, object]:
        resolved = REPO_ROOT / path
        return {
            "path": path,
            "sha256": sha256 or validator.sha256_file(resolved),
            "bytes": resolved.stat().st_size,
            "media_type": "application/json" if role == "configuration" else "text/csv",
            "role": role,
        }

    @staticmethod
    def _artifact_record(root: Path, path: str, role: str) -> dict[str, object]:
        resolved = root / path
        return {
            "path": path,
            "sha256": validator.sha256_file(resolved),
            "bytes": resolved.stat().st_size,
            "media_type": "text/csv" if path.endswith(".csv") else "text/plain",
            "role": role,
        }

    def _inputs_and_identities(
        self,
    ) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        records: list[dict[str, object]] = [
            self._input_record(path, "configuration")
            for path in validator.CONFIG_INPUTS
        ]
        required_files = TRACE_CONTRACT["required_files"]
        grouped: list[dict[str, list[dict[str, object]]]] = [
            {"weights": [], "trace": []} for _ in validator.LAYER_IDS
        ]
        for layer in TRACE_CONTRACT["layers"]:
            layer_id = layer["layer_id"]
            for logical_name, specification in required_files.items():
                path = (
                    Path("data") / f"layer_{layer_id}" / specification["path"]
                ).as_posix()
                role = (
                    "weights" if "/parms/" in f"/{specification['path']}" else "trace"
                )
                record = self._input_record(path, role, layer["sha256"][logical_name])
                records.append(record)
                grouped[layer_id][role].append(record)
        records.sort(key=lambda record: record["path"])
        identities = [
            {
                "layer_id": layer_id,
                "weight_file_count": validator.WEIGHT_FILES_PER_LAYER,
                "trace_file_count": validator.TRACE_FILES_PER_LAYER,
                "weight_bundle_sha256": validator._bundle_identity(
                    grouped[layer_id]["weights"]
                ),
                "trace_bundle_sha256": validator._bundle_identity(
                    grouped[layer_id]["trace"]
                ),
            }
            for layer_id in validator.LAYER_IDS
        ]
        return records, identities

    @staticmethod
    def _command(tokens: list[str], timed: bool = False) -> dict[str, object]:
        record: dict[str, object] = {
            "command": shlex.join(tokens),
            "cwd": str(REPO_ROOT),
            "exit_code": 0,
            "phase": "artifact_generation",
        }
        if timed:
            record.update(
                {
                    "started_at": "2026-07-30T10:00:00+09:00",
                    "finished_at": "2026-07-30T10:01:00+09:00",
                }
            )
        return record

    def _commands(self) -> list[dict[str, object]]:
        build = REPO_ROOT / "build-openfhe"
        executable = REPO_ROOT / validator.EXECUTABLE_PATH
        return [
            self._command(
                [
                    validator.SYSTEM_GIT,
                    "status",
                    "--porcelain=v1",
                    "--untracked-files=normal",
                ]
            ),
            self._command([validator.SYSTEM_GIT, "branch", "--show-current"]),
            self._command([validator.SYSTEM_GIT, "rev-parse", "HEAD"]),
            self._command([validator.SYSTEM_GIT, "rev-parse", validator.TRACKING_REF]),
            self._command(
                [
                    validator.SYSTEM_GIT,
                    "config",
                    "--local",
                    "--get-all",
                    f"remote.{validator.REMOTE_NAME}.url",
                ]
            ),
            self._command(
                [
                    validator.SYSTEM_GIT,
                    "ls-remote",
                    "--exit-code",
                    validator.REMOTE_URL,
                    validator.REMOTE_REF,
                ]
            ),
            self._command(
                validator._expected_configure_command(),
                timed=True,
            ),
            self._command(
                [
                    validator.SYSTEM_CMAKE,
                    "--build",
                    str(build),
                    "--clean-first",
                    "-j",
                    "4",
                ],
                timed=True,
            ),
            self._command([validator.SYSTEM_LDD, str(executable)], timed=True),
            self._command(
                [
                    validator.SYSTEM_CTEST,
                    "--test-dir",
                    str(build),
                    "--output-on-failure",
                    "--verbose",
                    "--no-tests=error",
                    "-R",
                    validator.M5_CTEST_PATTERN,
                ],
                timed=True,
            ),
            self._command(
                [
                    "/usr/bin/time",
                    "--format=%M",
                    "--output=/tmp/moai-m5-time-synthetic.txt",
                    str(executable),
                    "--data-root",
                    str(REPO_ROOT / "data"),
                ],
                timed=True,
            ),
            self._command(
                [
                    sys.executable,
                    str(validator.VALIDATOR_PATH),
                    "--schema",
                    str(validator.DEFAULT_SCHEMA),
                    "--manifest",
                    str(self.manifest_path.resolve()),
                    "--verify-git",
                ]
            ),
        ]

    @staticmethod
    def _build_configuration() -> dict[str, object]:
        package_root = validator.OPENFHE_PREFIX / "lib" / "OpenFHE"
        return {
            "generator": "Unix Makefiles",
            "source_root": str(REPO_ROOT),
            "build_root": str(validator.BUILD_ROOT),
            "openfhe_dir": str(package_root),
            "build_type": "Release",
            "build_testing": True,
            "cxx_compiler": validator.SYSTEM_CXX,
            "make_program": validator.SYSTEM_MAKE,
            "cxx_flags": "",
            "cxx_flags_release": "-O3 -DNDEBUG",
            "exe_linker_flags": "",
            "exe_linker_flags_release": "",
            "shared_linker_flags": "",
            "module_linker_flags": "",
            "static_linker_flags": "",
            "environment_inheritance": "ambient_minus_cleared_variables",
            "cleared_environment_variables": list(validator.CONFIGURE_ENV_UNSET),
            "forced_environment_variables": dict(
                validator.FORCED_SUBPROCESS_ENVIRONMENT
            ),
            "cmake_cache": {
                "path": "build-openfhe/CMakeCache.txt",
                "bytes": 1,
                "sha256": "1" * 64,
            },
            "openfhe_cmake_package_files": [
                {
                    "path": str(package_root / name),
                    "bytes": 1,
                    "sha256": f"{index + 2:064x}",
                }
                for index, name in enumerate(validator.OPENFHE_CMAKE_PACKAGE_FILES)
            ],
            "openfhe_include_tree": {
                "root": str(validator.OPENFHE_PREFIX / "include" / "openfhe"),
                "file_count": 1,
                "bytes": 1,
                "algorithm": "SHA-256",
                "canonicalization": (validator.OPENFHE_INCLUDE_TREE_CANONICALIZATION),
                "sha256": "2" * 64,
            },
            "openfhe_linked_libraries": [
                {
                    "soname": soname,
                    "path": str(
                        validator.OPENFHE_PREFIX
                        / "lib"
                        / validator.OPENFHE_LINKED_LIBRARY_BASENAMES[soname]
                    ),
                    "bytes": 1,
                    "sha256": f"{index + 20:064x}",
                }
                for index, soname in enumerate(validator.OPENFHE_LINKED_LIBRARY_SONAMES)
            ],
        }

    def _make_manifest(self) -> dict[str, object]:
        inputs, identities = self._inputs_and_identities()
        profile_path = REPO_ROOT / validator.PROFILE_PATH
        profile_config = validator.load_json(profile_path)
        effective = profile_config["effective_profile"]
        executable = REPO_ROOT / validator.EXECUTABLE_PATH
        commit = "a" * 40
        layer_metrics = [
            {key: record[key] for key in validator.LAYER_METRIC_KEYS}
            for record in self.layer_records
        ]
        return {
            "schema_version": 6,
            "schema_binding": {
                component: {
                    "id": component_id,
                    "path": relative_path,
                    "version": 6,
                    "sha256": validator.sha256_file(REPO_ROOT / relative_path),
                }
                for component, (
                    component_id,
                    relative_path,
                ) in validator.SCHEMA_BINDING_SPECS.items()
            }
            | {
                "retired_unpublished_drafts": [
                    "https://local.moai/openfhe-m5-artifact-schema-v4.json"
                ]
            },
            "run_id": self.run_id,
            "milestone": "M5",
            "started_at": "2026-07-30T10:00:00+09:00",
            "finished_at": "2026-07-30T10:01:00+09:00",
            "git": {
                "repository_root": str(REPO_ROOT),
                "branch": validator.BRANCH,
                "local_commit": commit,
                "clean": True,
                "tracking_ref": validator.TRACKING_REF,
                "tracking_commit": commit,
                "remote_name": validator.REMOTE_NAME,
                "remote_url": validator.REMOTE_URL,
                "remote_ref": validator.REMOTE_REF,
                "remote_commit": commit,
            },
            "commands": self._commands(),
            "build_configuration": self._build_configuration(),
            "environment": {
                "os": "synthetic WSL",
                "architecture": "x86_64",
                "compiler": "synthetic",
                "cmake": "synthetic",
                "python": sys.version.split()[0],
                "openfhe_version": "1.5.1",
                "openfhe_prefix": "/home/shawnsheep/opt/openfhe_v1_5_1",
                "wsl": True,
            },
            "profile": {
                "id": "paper_compat",
                "security_claim": "none",
                "warning": validator.WARNING,
                "source_config_path": validator.PROFILE_PATH,
                "source_config_sha256": validator.sha256_file(profile_path),
                "source_config_bytes": profile_path.stat().st_size,
                "effective_profile_locator": validator.PROFILE_LOCATOR,
                "canonicalization": validator.CANONICALIZATION,
                "effective_profile_payload": effective,
                "effective_profile_sha256": validator.PROFILE_SHA256,
            },
            "inputs": inputs,
            "workload": {
                "scope": (
                    "M5 server-only five-token BERT-base 12-layer encoder "
                    "ciphertext-chain trace replay"
                ),
                "backend": "OpenFHE CKKS CPU",
                "executable_path": validator.EXECUTABLE_PATH,
                "executable_sha256": validator.sha256_file(executable),
                "executable_bytes": executable.stat().st_size,
                "repeat_count": 1,
                "warmup_count": 0,
                "timing_claim": False,
                "latency_kind": "non_benchmark_diagnostic",
            },
            "contracts": {
                "approximation_config_path": "config/openfhe_approximations.json",
                "approximation_config_sha256": next(
                    record["sha256"]
                    for record in inputs
                    if record["path"] == "config/openfhe_approximations.json"
                ),
                "channel_scales_config_path": "config/moai_trace_channel_scales.json",
                "channel_scales_config_sha256": next(
                    record["sha256"]
                    for record in inputs
                    if record["path"] == "config/moai_trace_channel_scales.json"
                ),
                "encoder_trace_contract_path": "config/moai_encoder_trace.json",
                "encoder_trace_contract_sha256": next(
                    record["sha256"]
                    for record in inputs
                    if record["path"] == "config/moai_encoder_trace.json"
                ),
                "profile_config_path": validator.PROFILE_PATH,
                "profile_config_sha256": validator.sha256_file(profile_path),
                "execution": copy.deepcopy(validator.EXPECTED_EXECUTION),
                "crypto_preflight": _crypto_preflight_record(),
                "schedule_preflight": _schedule_preflight_record(),
                "feature_packed_trace_scale_contract": (_trace_scale_provenance()),
                "exact3_evidence": _exact3_evidence(),
                "metadata_schedule": copy.deepcopy(
                    validator._expected_metadata_schedule_contract()
                ),
                "thresholds": copy.deepcopy(validator.THRESHOLDS),
                "polynomial_intervals": copy.deepcopy(validator.POLYNOMIAL_INTERVALS),
                "layer_input_identities": identities,
            },
            "metrics": {
                "repeat_count": 1,
                "successful_repeats": 1,
                "warmup_count": 0,
                "timing_claim": False,
                "latency_kind": "non_benchmark_diagnostic",
                "elapsed_seconds": 5.0,
                "peak_rss_kib": 2048,
                "checkpoint_metadata_sha256": (
                    validator.checkpoint_metadata_sha256(self.layer_records)
                ),
                "phase_latency_ms": {
                    key: self.summary[key]
                    for key in (
                        "fixture_load_oracle_ms",
                        "setup_keygen_ms",
                        "client_encrypt_ms",
                        "server_online_diagnostic_ms",
                        "client_checkpoint_validate_ms",
                    )
                },
                "layers": layer_metrics,
                "final_quality": {
                    "relative_l2": self.summary["relative_l2"],
                    "cosine": self.summary["cosine"],
                    "max_absolute": self.summary["max_absolute"],
                    "inactive_max_abs": self.summary["inactive_max_abs"],
                    "inactive_sentinel_max_error": self.summary[
                        "inactive_sentinel_max_error"
                    ],
                    "inactive_sentinel_range_status": self.summary[
                        "inactive_sentinel_range_status"
                    ],
                },
                "final_metadata": copy.deepcopy(validator.RAW_OUTPUT_METADATA),
                "operation_counts": copy.deepcopy(validator.TOTAL_COUNTS),
                "finite": True,
            },
            "gate": {
                "passed": True,
                "decision": "PASS_M5_RUNNABLE_PROTOTYPE",
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
            "artifacts": [],
            "claim_boundary": copy.deepcopy(validator.CLAIM_BOUNDARY),
            "verdict": "GO_PROTOTYPE",
        }

    def _reseal_artifacts(self) -> None:
        stdout = self.artifact_root / "stdout.log"
        metrics = self.artifact_root / "metrics.csv"
        (self.artifact_root / "SHA256SUMS").write_text(
            f"{validator.sha256_file(stdout)}  stdout.log\n"
            f"{validator.sha256_file(metrics)}  metrics.csv\n",
            encoding="utf-8",
        )
        self.manifest["artifacts"] = [
            self._artifact_record(self.artifact_root, "stdout.log", "stdout"),
            self._artifact_record(self.artifact_root, "metrics.csv", "metrics"),
            self._artifact_record(self.artifact_root, "SHA256SUMS", "checksum"),
        ]

    def _write_manifest(self) -> None:
        self.manifest_path.write_text(
            json.dumps(self.manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _git_output(self, _root: Path, arguments: list[str]) -> str:
        commit = self.manifest["git"]["local_commit"]
        if arguments == ["rev-parse", "HEAD"]:
            return commit
        if (
            len(arguments) == 2
            and arguments[0] == "rev-parse"
            and arguments[1].startswith(f"{commit}:")
        ):
            return "c" * 40
        if arguments[:2] == ["hash-object", "--"]:
            return "c" * 40
        if arguments == ["branch", "--show-current"]:
            return validator.BRANCH
        if arguments == ["status", "--porcelain=v1", "--untracked-files=normal"]:
            return ""
        if arguments == ["rev-parse", validator.TRACKING_REF]:
            return commit
        if arguments == [
            "config",
            "--local",
            "--get-all",
            f"remote.{validator.REMOTE_NAME}.url",
        ]:
            return validator.REMOTE_URL
        if arguments == [
            "ls-remote",
            "--exit-code",
            validator.REMOTE_URL,
            validator.REMOTE_REF,
        ]:
            return f"{commit}\t{validator.REMOTE_REF}"
        raise AssertionError(arguments)

    def _validate(self) -> None:
        with (
            mock.patch.object(validator, "_run_git", side_effect=self._git_output),
            mock.patch.object(validator, "_verify_build_configuration"),
            mock.patch.object(
                validator,
                "_require_profile_schedule_sealed",
                return_value={
                    "source_path": validator.PROFILE_PATH,
                    "json_locator": (
                        "feature_packed_layernorm_override.schedule_status"
                    ),
                    "status": validator.PROFILE_SCHEDULE_SEALED_STATUS,
                },
            ),
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
        ):
            validator.validate_manifest(
                self.manifest_path,
                validator.load_json(self.manifest_path),
                self.schema,
                True,
            )

    def _assert_invalid(self, pattern: str) -> None:
        self._write_manifest()
        with self.assertRaisesRegex(validator.ValidationError, pattern):
            self._validate()

    def test_accepts_frozen_m5_bundle(self) -> None:
        self._validate()

    def test_artifact_environment_and_schema_are_fail_closed(self) -> None:
        poisoned = {name: f"poison-{name}" for name in validator.CONFIGURE_ENV_UNSET}
        poisoned["MOAI_SAFE_SENTINEL"] = "preserved"
        with mock.patch.dict(os.environ, poisoned, clear=False):
            environment = validator._artifact_environment()
        for name in validator.CONFIGURE_ENV_UNSET:
            with self.subTest(name=name):
                self.assertNotIn(name, environment)
        self.assertEqual(
            {
                name: environment[name]
                for name in validator.FORCED_SUBPROCESS_ENVIRONMENT
            },
            validator.FORCED_SUBPROCESS_ENVIRONMENT,
        )
        self.assertEqual(environment["MOAI_SAFE_SENTINEL"], "preserved")

        build_schema = self.schema["$defs"]["build_configuration"]
        record = self._build_configuration()
        validator.validate_instance(record, build_schema, self.schema)
        for name, replacement in (
            ("environment_inheritance", "ambient"),
            ("cleared_environment_variables", ["CXXFLAGS"]),
            ("forced_environment_variables", {"LANG": "C"}),
        ):
            with self.subTest(field=name):
                tampered = copy.deepcopy(record)
                tampered[name] = replacement
                with self.assertRaises(validator.ValidationError):
                    validator.validate_instance(
                        tampered,
                        build_schema,
                        self.schema,
                    )

    def test_cmake_cache_requires_cmake_normalized_compiler_type(self) -> None:
        cache_path = self.output_root / "CMakeCache.txt"

        def write_cache(
            compiler_type: str,
            compiler: Path = validator.SYSTEM_CXX,
        ) -> None:
            cache_path.write_text(
                "\n".join(
                    [
                        "CMAKE_GENERATOR:INTERNAL=Unix Makefiles",
                        f"CMAKE_HOME_DIRECTORY:INTERNAL={validator.REPO_ROOT}",
                        "CMAKE_BUILD_TYPE:STRING=Release",
                        "BUILD_TESTING:BOOL=ON",
                        "OpenFHE_DIR:PATH="
                        f"{validator.OPENFHE_PREFIX / 'lib' / 'OpenFHE'}",
                        f"CMAKE_CXX_COMPILER:{compiler_type}={compiler}",
                        f"CMAKE_MAKE_PROGRAM:FILEPATH={validator.SYSTEM_MAKE}",
                        "CMAKE_CXX_FLAGS:STRING=",
                        "CMAKE_CXX_FLAGS_RELEASE:STRING=-O3 -DNDEBUG",
                        "CMAKE_EXE_LINKER_FLAGS:STRING=",
                        "CMAKE_EXE_LINKER_FLAGS_RELEASE:STRING=",
                        "CMAKE_SHARED_LINKER_FLAGS:STRING=",
                        "CMAKE_MODULE_LINKER_FLAGS:STRING=",
                        "CMAKE_STATIC_LINKER_FLAGS:STRING=",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

        write_cache("STRING")
        validator._verify_cmake_cache(cache_path)
        write_cache("FILEPATH")
        with self.assertRaisesRegex(
            validator.ValidationError,
            "CMake cache build configuration drifted",
        ):
            validator._verify_cmake_cache(cache_path)
        write_cache("STRING", Path("/tmp/injected-c++"))
        with self.assertRaisesRegex(
            validator.ValidationError,
            "CMake cache build configuration drifted",
        ):
            validator._verify_cmake_cache(cache_path)

    def test_schema_binding_freezes_v6_and_retires_v4_draft(self) -> None:
        binding = self.manifest["schema_binding"]
        self.assertEqual(
            binding["retired_unpublished_drafts"],
            ["https://local.moai/openfhe-m5-artifact-schema-v4.json"],
        )
        for component, (_, relative_path) in validator.SCHEMA_BINDING_SPECS.items():
            self.assertEqual(binding[component]["version"], 6)
            self.assertEqual(binding[component]["path"], relative_path)
            self.assertEqual(
                binding[component]["sha256"],
                validator.sha256_file(REPO_ROOT / relative_path),
            )
        self._validate()

    def test_rejects_missing_or_tampered_schema_binding(self) -> None:
        missing = copy.deepcopy(self.manifest)
        missing.pop("schema_binding")
        with self.assertRaisesRegex(validator.ValidationError, "schema_binding"):
            validator.validate_instance(missing, self.schema, self.schema)

        self.manifest["schema_binding"]["runner"]["sha256"] = "0" * 64
        self._assert_invalid("schema_binding.runner.sha256")

    def test_rejects_schema_binding_head_blob_drift(self) -> None:
        def git_output(root: Path, arguments: list[str]) -> str:
            if arguments[:2] == ["hash-object", "--"] and arguments[-1].endswith(
                "run_openfhe_encoder12_artifact_v6.py"
            ):
                return "d" * 40
            return self._git_output(root, arguments)

        self._write_manifest()
        with (
            mock.patch.object(validator, "_run_git", side_effect=git_output),
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
            self.assertRaisesRegex(validator.ValidationError, "HEAD blob"),
        ):
            validator.validate_manifest(
                self.manifest_path,
                validator.load_json(self.manifest_path),
                self.schema,
                True,
            )

    def test_accepts_sentinel_diagnostic_above_zero_inactive_threshold(self) -> None:
        self.assertGreater(self.summary["inactive_sentinel_max_error"], 1e-6)
        self.assertNotIn(
            "layernorm_inactive_identity_max_error",
            validator.THRESHOLDS,
        )
        self.assertEqual(
            validator.INACTIVE_POLYNOMIAL_SENTINEL_INTERVALS["ln1_normalized_variance"],
            [0.5, 1536.0],
        )
        self.assertEqual(
            validator.INACTIVE_POLYNOMIAL_SENTINEL_INTERVALS["ln2_normalized_variance"],
            [0.5, 1536.0],
        )
        self._validate()

    def test_schema_contract_is_independent_v6_m5(self) -> None:
        self.assertEqual(self.schema["properties"]["schema_version"]["const"], 6)
        self.assertTrue(self.schema["$id"].endswith("schema-v6.json"))
        self.assertEqual(self.schema["title"], validator.SCHEMA_TITLE)
        self.assertEqual(self.schema["properties"]["milestone"]["const"], "M5")
        self.assertEqual(self.schema["properties"]["inputs"]["minItems"], 448)
        self.assertIn(
            "crypto_preflight", self.schema["properties"]["contracts"]["required"]
        )
        self.assertIn(
            "schedule_preflight",
            self.schema["properties"]["contracts"]["required"],
        )
        self.assertIn(
            "feature_packed_trace_scale_contract",
            self.schema["properties"]["contracts"]["required"],
        )
        self.assertIn(
            "exact3_evidence",
            self.schema["properties"]["contracts"]["required"],
        )

    def test_exact3_evidence_rejects_retired_r16_and_bad_sha_length(self) -> None:
        retired = {
            **validator.LEGACY_R16_CALIBRATION_EVIDENCE,
            "relative_path": (
                f"results/openfhe/{validator.LEGACY_R16_CALIBRATION_EVIDENCE['run_id']}"
            ),
            "layer_count": 3,
            "artifact_eligible": False,
            "exact3_gate_passed": True,
            "schedule_evidence_eligible": True,
            "formal_schedule_sealed": True,
        }
        with self.assertRaisesRegex(validator.ValidationError, "retired r16"):
            validator._validate_exact3_evidence(retired, "retired")
        renamed = {
            **retired,
            "run_id": "renamed-retired-r16",
            "relative_path": "results/openfhe/renamed-retired-r16",
        }
        with self.assertRaisesRegex(validator.ValidationError, "reuses retired r16"):
            validator._validate_exact3_evidence(renamed, "renamed")
        self.manifest["contracts"]["exact3_evidence"]["sha256sums_sha256"] = "0" * 63
        self._assert_invalid("exact3_evidence")

    def test_current_profile_and_exact3_bundle_are_sealed(self) -> None:
        profile_contract = validator._require_profile_schedule_sealed(
            evidence=validator.EXACT3_EVIDENCE
        )
        self.assertEqual(profile_contract["status"], "sealed")
        self.assertEqual(
            profile_contract["exact3_evidence"], validator.EXACT3_EVIDENCE
        )
        self.assertEqual(
            validator._validate_exact3_evidence(
                validator.EXACT3_EVIDENCE, "EXACT3_EVIDENCE"
            ),
            validator.EXACT3_EVIDENCE,
        )
        with (
            mock.patch.object(validator, "EXACT3_EVIDENCE", None),
            self.assertRaisesRegex(validator.ValidationError, "evidence is missing"),
        ):
            validator._require_exact3_evidence()

    def test_exact3_evidence_role_booleans_and_shape_fail_closed(self) -> None:
        invalid_values = {
            "artifact_eligible": True,
            "exact3_gate_passed": False,
            "schedule_evidence_eligible": False,
            "formal_schedule_sealed": 1,
        }
        for key, invalid in invalid_values.items():
            self.manifest["contracts"]["exact3_evidence"][key] = invalid
            with self.subTest(key=key):
                self._assert_invalid(f"exact3_evidence.{key}")
            self.manifest["contracts"]["exact3_evidence"] = _exact3_evidence()
        self.manifest["contracts"]["exact3_evidence"].pop("schedule_evidence_eligible")
        self._assert_invalid("exact3_evidence")

    def test_exact3_bundle_is_independently_dereferenced_and_verified(self) -> None:
        with tempfile.TemporaryDirectory(prefix="m5-exact3-validator-") as directory:
            repository_root = Path(directory)
            evidence, _ = _write_exact3_test_bundle(repository_root)
            validated = validator._validate_exact3_evidence(evidence, "exact3")
            self.assertEqual(
                validator._verify_exact3_evidence_bundle(
                    validated,
                    repository_root,
                ),
                evidence,
            )
            with mock.patch.object(validator, "EXACT3_EVIDENCE", evidence):
                self.assertEqual(
                    validator._require_exact3_evidence(repository_root),
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
                    validator.ValidationError,
                    "relative_path",
                ):
                    validator._validate_exact3_evidence(tampered, "exact3")

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
                tempfile.TemporaryDirectory(prefix="m5-exact3-validator-") as directory,
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
                    (repository_root / validator.EXACT3_SOURCE_PATHS[0]).write_bytes(
                        b"tampered source\n"
                    )
                elif case == "live_config":
                    (repository_root / validator.EXACT3_CONFIG_PATHS[0]).write_bytes(
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
                with self.assertRaises(validator.ValidationError):
                    validator._verify_exact3_evidence_bundle(
                        approved,
                        repository_root,
                    )

    def test_exact3_historical_executable_need_not_match_fresh_m5_build(self) -> None:
        with tempfile.TemporaryDirectory(prefix="m5-exact3-validator-") as directory:
            repository_root = Path(directory)
            approved, _ = _write_exact3_test_bundle(repository_root)
            executable = repository_root / validator.EXACT3_EXECUTABLE_PATH
            executable.write_bytes(b"a different fresh M5 executable\n")
            executable.chmod(0o755)
            self.assertEqual(
                validator._verify_exact3_evidence_bundle(
                    approved,
                    repository_root,
                ),
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
                validator.EXACT3_FINAL_COUNTS["rotations"] + 1,
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
                list(validator.PROFILE_TRANSITION_ALLOWED_JSON_POINTERS[:-1]),
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
                tempfile.TemporaryDirectory(prefix="m5-exact3-validator-") as directory,
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
                with self.assertRaises(validator.ValidationError):
                    validator._verify_exact3_evidence_bundle(
                        approved,
                        repository_root,
                    )

    def test_exact3_sealed_profile_immutable_projection_drift_is_rejected(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(
            prefix="m5-exact3-profile-validator-"
        ) as directory:
            repository_root = Path(directory)
            approved, _ = _write_exact3_test_bundle(repository_root)
            profile_path = repository_root / validator.PROFILE_PATH
            profile = json.loads(profile_path.read_text(encoding="utf-8"))
            profile["unapproved_immutable_field"] = "drift"
            profile_path.write_text(
                json.dumps(profile, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                validator.ValidationError,
                "immutable projection",
            ):
                validator._verify_exact3_evidence_bundle(approved, repository_root)

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
                tempfile.TemporaryDirectory(
                    prefix="m5-exact3-raw-validator-"
                ) as directory,
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
                with self.assertRaises(validator.ValidationError):
                    validator._verify_exact3_evidence_bundle(
                        approved,
                        repository_root,
                    )

        for case in ("duplicate_key", "nan", "overflow"):
            with (
                self.subTest(case=case),
                tempfile.TemporaryDirectory(prefix="m5-exact3-validator-") as directory,
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
                with self.assertRaises(validator.ValidationError):
                    validator._verify_exact3_evidence_bundle(
                        approved,
                        repository_root,
                    )

    def test_m5_claim_boundary_is_exact_schema_const(self) -> None:
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
        self.assertEqual(validator.CLAIM_BOUNDARY, expected)
        self.assertEqual(
            self.schema["properties"]["claim_boundary"],
            {"type": "array", "const": expected},
        )
        validator.validate_instance(
            expected,
            self.schema["properties"]["claim_boundary"],
            self.schema,
        )
        for index in range(len(expected)):
            with self.subTest(index=index):
                tampered = copy.deepcopy(expected)
                tampered[index] += " expanded"
                manifest = copy.deepcopy(self.manifest)
                manifest["claim_boundary"] = tampered
                with self.assertRaises(validator.ValidationError):
                    validator.validate_instance(manifest, self.schema, self.schema)

    def test_trace_scale_provenance_is_independently_recomputed(self) -> None:
        self.assertEqual(
            validator._trace_scale_contract_provenance(),
            _trace_scale_provenance(),
        )
        for key in ("contract_sha256", "values_sha256", "raw_variance_sha256"):
            self.manifest["contracts"]["feature_packed_trace_scale_contract"][key] = (
                "0" * 63
            )
            with self.subTest(key=key):
                self._assert_invalid("feature_packed_trace_scale_contract")
            self.manifest["contracts"]["feature_packed_trace_scale_contract"] = (
                _trace_scale_provenance()
            )

    def test_schedule_preflight_is_formal_and_he_verified(self) -> None:
        for key in ("formal_schedule_sealed", "he_metadata_verified_by_preflight"):
            self.manifest["contracts"]["schedule_preflight"][key] = True
            with self.subTest(key=key):
                self._assert_invalid(f"schedule_preflight.{key}")
            self.manifest["contracts"]["schedule_preflight"] = (
                _schedule_preflight_record()
            )
        self.manifest["contracts"]["schedule_preflight"][
            "metadata_schedule_preflight"
        ] = "forged_formal_metadata_schedule"
        self._assert_invalid("schedule_preflight.metadata_schedule_preflight")

    def test_operation_counts_and_formal_cumulative_counts_are_frozen(self) -> None:
        self.assertEqual(validator.LAYER_COUNTS["ct_pt_multiplications"], 51885)
        self.assertEqual(validator.LAYER_COUNTS["explicit_rescale_requests"], 810)
        self.assertEqual(validator.TOTAL_COUNTS["ct_pt_multiplications"], 622675)
        self.assertEqual(validator.TOTAL_COUNTS["explicit_rescale_requests"], 9775)
        for layer_id in range(11):
            cumulative = validator._cumulative_counts(layer_id)
            self.assertEqual(
                cumulative["ct_pt_multiplications"],
                (layer_id + 1) * 51890,
            )
            self.assertEqual(
                cumulative["explicit_rescale_requests"],
                (layer_id + 1) * 815,
            )
        self.assertEqual(validator._cumulative_counts(11), validator.TOTAL_COUNTS)

    def test_metadata_schedule_freezes_all_layer_regimes(self) -> None:
        schedule = validator._expected_metadata_schedule_contract()
        for field in (
            "attention_output_by_layer_regime",
            "ln1_output_by_layer_regime",
            "ffn_output_by_layer_regime",
        ):
            self.assertEqual(
                set(schedule[field]),
                {"layer_0", "layers_1_to_11"},
            )
        self.assertEqual(
            schedule["post_refresh_layer_input"],
            validator.LAYER_HANDOFF_INPUT_METADATA,
        )
        self.assertEqual(
            schedule["post_bootstrap_softmax_denominator_checkpoint"],
            validator.SOFTMAX_CHECKPOINT_METADATA,
        )
        self.assertEqual(
            schedule["post_bootstrap_canonicalized_layernorm_variance_checkpoint"],
            validator.LAYERNORM_CHECKPOINT_METADATA,
        )
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

    def test_schema_rejects_each_r16_delta_tamper(self) -> None:
        validator.validate_instance(self.manifest, self.schema, self.schema)
        for delta_name in validator.RELATIVE_USED_LEVEL_DELTAS:
            tampered = copy.deepcopy(self.manifest)
            tampered["contracts"]["metadata_schedule"]["relative_used_level_deltas"][
                delta_name
            ]["layers_1_to_11"] += 1
            with (
                self.subTest(delta_name=delta_name),
                self.assertRaises(validator.ValidationError),
            ):
                validator.validate_instance(tampered, self.schema, self.schema)

        layer0_null_tamper = copy.deepcopy(self.manifest)
        layer0_null_tamper["contracts"]["metadata_schedule"][
            "relative_used_level_deltas"
        ]["previous_raw_output_to_input_recovered"]["layer_0"] = 0
        with self.assertRaises(validator.ValidationError):
            validator.validate_instance(
                layer0_null_tamper,
                self.schema,
                self.schema,
            )

    def test_schema_and_validator_freeze_v5_v6_ctest_contracts(self) -> None:
        required_artifact_contracts = (
            "openfhe_m4_v5_artifact_schema_contract",
            "openfhe_m4_v5_artifact_validator_contract",
            "openfhe_m4_v5_encoder_artifact_runner_contract",
            "openfhe_m5_v6_artifact_schema_contract",
            "openfhe_m5_v6_artifact_validator_contract",
            "openfhe_m5_v6_encoder12_artifact_runner_contract",
        )
        command = self.manifest["commands"][9]["command"]
        pattern = shlex.split(command)[-1]
        for target in required_artifact_contracts:
            with self.subTest(target=target):
                self.assertIsNotNone(validator.re.fullmatch(pattern, target))
        self.assertIsNotNone(
            validator.re.fullmatch(pattern, "openfhe_feature_layernorm_smoke")
        )
        self.assertIsNotNone(
            validator.re.fullmatch(pattern, "openfhe_encoder_layer_smoke")
        )

        schema_command = self.schema["properties"]["commands"]["prefixItems"][9][
            "allOf"
        ][1]["properties"]["command"]["const"]
        self.assertEqual(schema_command, command)
        validator.validate_instance(self.manifest, self.schema, self.schema)
        validator._verify_command_transcript(self.manifest, self.manifest_path)

        tampered = copy.deepcopy(self.manifest)
        tokens = shlex.split(tampered["commands"][9]["command"])
        tokens[-1] = tokens[-1].replace("m5_v6_artifact_schema_contract|", "")
        tampered["commands"][9]["command"] = shlex.join(tokens)
        with self.assertRaises(validator.ValidationError):
            validator.validate_instance(tampered, self.schema, self.schema)
        with self.assertRaisesRegex(validator.ValidationError, "commands\\[9\\]"):
            validator._verify_command_transcript(tampered, self.manifest_path)

    def test_configure_transcript_rejects_missing_fresh_and_extra_flags(self) -> None:
        schema_command = self.schema["properties"]["commands"]["prefixItems"][6][
            "allOf"
        ][1]["properties"]["command"]["const"]
        self.assertEqual(
            schema_command,
            self.manifest["commands"][6]["command"],
        )
        missing_fresh = copy.deepcopy(self.manifest)
        tokens = shlex.split(missing_fresh["commands"][6]["command"])
        tokens.remove("--fresh")
        missing_fresh["commands"][6]["command"] = shlex.join(tokens)
        with self.assertRaisesRegex(validator.ValidationError, "commands\\[6\\]"):
            validator._verify_command_transcript(
                missing_fresh,
                self.manifest_path,
            )

        extra_flags = copy.deepcopy(self.manifest)
        tokens = shlex.split(extra_flags["commands"][6]["command"])
        tokens.append("-DCMAKE_CXX_FLAGS:STRING=-march=native")
        extra_flags["commands"][6]["command"] = shlex.join(tokens)
        with self.assertRaisesRegex(validator.ValidationError, "commands\\[6\\]"):
            validator._verify_command_transcript(
                extra_flags,
                self.manifest_path,
            )

    def test_schema_rejects_linked_library_set_and_path_tamper(self) -> None:
        linked = self.manifest["build_configuration"]["openfhe_linked_libraries"]
        cases: list[tuple[str, list[dict[str, object]]]] = []
        cases.append(("missing", copy.deepcopy(linked[:-1])))
        cases.append(("duplicate", copy.deepcopy([linked[0], linked[0], linked[2]])))
        cases.append(("extra", copy.deepcopy([*linked, linked[0]])))
        outside = copy.deepcopy(linked)
        outside[0]["path"] = "/tmp/libOPENFHEbinfhe.so.1.5.1"
        cases.append(("outside", outside))
        for name, replacement in cases:
            tampered = copy.deepcopy(self.manifest)
            tampered["build_configuration"]["openfhe_linked_libraries"] = replacement
            with (
                self.subTest(name=name),
                self.assertRaises(validator.ValidationError),
            ):
                validator.validate_instance(tampered, self.schema, self.schema)

    def test_semantic_build_provenance_rejects_library_byte_drift(self) -> None:
        record = self.manifest["build_configuration"]
        package_records = record["openfhe_cmake_package_files"]
        linked_records = record["openfhe_linked_libraries"]
        with (
            mock.patch.object(validator, "_verify_file_record"),
            mock.patch.object(validator, "_verify_cmake_cache"),
            mock.patch.object(
                validator,
                "_absolute_regular_file_record",
                side_effect=copy.deepcopy(package_records),
            ),
            mock.patch.object(
                validator,
                "_openfhe_include_tree_record",
                return_value=copy.deepcopy(record["openfhe_include_tree"]),
            ),
            mock.patch.object(
                validator,
                "_live_openfhe_linked_libraries",
                return_value=copy.deepcopy(linked_records),
            ),
        ):
            validator._verify_build_configuration(self.manifest)

    def test_semantic_build_provenance_rejects_cache_package_and_include_drift(
        self,
    ) -> None:
        record = self.manifest["build_configuration"]
        package_records = record["openfhe_cmake_package_files"]
        include_record = record["openfhe_include_tree"]
        linked_records = record["openfhe_linked_libraries"]

        def common_patches(
            *,
            packages: object = package_records,
            include: object = include_record,
        ) -> tuple[mock._patch, ...]:
            return (
                mock.patch.object(validator, "_verify_cmake_cache"),
                mock.patch.object(
                    validator,
                    "_absolute_regular_file_record",
                    side_effect=copy.deepcopy(packages),
                ),
                mock.patch.object(
                    validator,
                    "_openfhe_include_tree_record",
                    return_value=copy.deepcopy(include),
                ),
                mock.patch.object(
                    validator,
                    "_live_openfhe_linked_libraries",
                    return_value=copy.deepcopy(linked_records),
                ),
            )

        cache_patches = common_patches()
        with (
            mock.patch.object(
                validator,
                "_verify_file_record",
                side_effect=validator.ValidationError(
                    "build_configuration.cmake_cache: SHA-256 mismatch"
                ),
            ),
            cache_patches[0],
            cache_patches[1],
            cache_patches[2],
            cache_patches[3],
            self.assertRaisesRegex(
                validator.ValidationError,
                "cmake_cache.*SHA-256 mismatch",
            ),
        ):
            validator._verify_build_configuration(self.manifest)

        drifted_packages = copy.deepcopy(package_records)
        drifted_packages[0]["sha256"] = "0" * 64
        package_patches = common_patches(packages=drifted_packages)
        with (
            mock.patch.object(validator, "_verify_file_record"),
            package_patches[0],
            package_patches[1],
            package_patches[2],
            package_patches[3],
            self.assertRaisesRegex(
                validator.ValidationError,
                "CMake package file provenance drifted",
            ),
        ):
            validator._verify_build_configuration(self.manifest)

        drifted_include = copy.deepcopy(include_record)
        drifted_include["sha256"] = "0" * 64
        include_patches = common_patches(include=drifted_include)
        with (
            mock.patch.object(validator, "_verify_file_record"),
            include_patches[0],
            include_patches[1],
            include_patches[2],
            include_patches[3],
            self.assertRaisesRegex(
                validator.ValidationError,
                "include tree provenance drifted",
            ),
        ):
            validator._verify_build_configuration(self.manifest)

        drifted = copy.deepcopy(linked_records)
        drifted[0]["sha256"] = "0" * 64
        with (
            mock.patch.object(validator, "_verify_file_record"),
            mock.patch.object(validator, "_verify_cmake_cache"),
            mock.patch.object(
                validator,
                "_absolute_regular_file_record",
                side_effect=copy.deepcopy(package_records),
            ),
            mock.patch.object(
                validator,
                "_openfhe_include_tree_record",
                return_value=copy.deepcopy(record["openfhe_include_tree"]),
            ),
            mock.patch.object(
                validator,
                "_live_openfhe_linked_libraries",
                return_value=drifted,
            ),
            self.assertRaisesRegex(
                validator.ValidationError,
                "linked-library provenance drifted",
            ),
        ):
            validator._verify_build_configuration(self.manifest)

    def test_schema_alone_rejects_layer_position_swaps(self) -> None:
        validator.validate_instance(self.manifest, self.schema, self.schema)

        metadata_swap = copy.deepcopy(self.manifest)
        metadata_layers = metadata_swap["metrics"]["layers"]
        metadata_layers[0]["input_metadata"], metadata_layers[1]["input_metadata"] = (
            metadata_layers[1]["input_metadata"],
            metadata_layers[0]["input_metadata"],
        )

        refresh_swap = copy.deepcopy(self.manifest)
        refresh_layers = refresh_swap["metrics"]["layers"]
        (
            refresh_layers[0]["refresh_operation_counts"],
            refresh_layers[11]["refresh_operation_counts"],
        ) = (
            refresh_layers[11]["refresh_operation_counts"],
            refresh_layers[0]["refresh_operation_counts"],
        )

        cumulative_swap = copy.deepcopy(self.manifest)
        cumulative_layers = cumulative_swap["metrics"]["layers"]
        (
            cumulative_layers[0]["cumulative_operation_counts"],
            cumulative_layers[1]["cumulative_operation_counts"],
        ) = (
            cumulative_layers[1]["cumulative_operation_counts"],
            cumulative_layers[0]["cumulative_operation_counts"],
        )

        position_swap = copy.deepcopy(self.manifest)
        position_layers = position_swap["metrics"]["layers"]
        position_layers[0], position_layers[11] = (
            position_layers[11],
            position_layers[0],
        )

        for label, tampered in (
            ("metadata", metadata_swap),
            ("refresh", refresh_swap),
            ("cumulative", cumulative_swap),
            ("whole position", position_swap),
        ):
            with (
                self.subTest(label=label),
                self.assertRaises(validator.ValidationError),
            ):
                validator.validate_instance(tampered, self.schema, self.schema)

    def test_schema_alone_rejects_checkpoint_count_and_boundary_metadata_tamper(
        self,
    ) -> None:
        validator.validate_instance(self.manifest, self.schema, self.schema)

        checkpoint_count_tamper = copy.deepcopy(self.manifest)
        checkpoint_count_tamper["metrics"]["layers"][5][
            "inactive_zero_checkpoint_count"
        ] = 27

        crypto_metadata_tamper = copy.deepcopy(self.manifest)
        crypto_metadata = crypto_metadata_tamper["contracts"]["crypto_preflight"][
            "input_metadata"
        ]
        crypto_metadata["level"] += 1
        crypto_metadata["remaining_levels"] -= 1

        final_metadata_tamper = copy.deepcopy(self.manifest)
        final_metadata = final_metadata_tamper["metrics"]["final_metadata"]
        final_metadata["level"] += 1
        final_metadata["remaining_levels"] -= 1

        for label, tampered in (
            ("checkpoint count", checkpoint_count_tamper),
            ("crypto preflight input", crypto_metadata_tamper),
            ("final output", final_metadata_tamper),
        ):
            with (
                self.subTest(label=label),
                self.assertRaises(validator.ValidationError),
            ):
                validator.validate_instance(tampered, self.schema, self.schema)

    def test_rejects_missing_crypto_preflight_evidence(self) -> None:
        self.manifest["contracts"].pop("crypto_preflight")
        self._assert_invalid("missing required property 'crypto_preflight'")

    def test_rejects_tampered_crypto_preflight_evidence(self) -> None:
        self.manifest["contracts"]["crypto_preflight"]["additive_he_operations"] = 1
        self._assert_invalid("expected constant 0")

    def test_rejects_crypto_preflight_actual_scale_outside_tolerance(self) -> None:
        self.manifest["contracts"]["crypto_preflight"] = _crypto_preflight_record(
            scale_bits=50.0011
        )
        self._assert_invalid(
            r"value is above maximum|violates \[49\.999,50\.001\]|more than 1e-3"
        )

    def test_schema_identity_rejects_retired_v4_title(self) -> None:
        schema = copy.deepcopy(self.schema)
        schema["title"] = "MOAI OpenFHE M5 evidence manifest v4"
        with self.assertRaisesRegex(
            validator.ValidationError, "schema identity drifted"
        ):
            validator.validate_schema(schema)

    def test_schema_engine_rejects_malformed_defensive_keywords(self) -> None:
        duplicate_enum = copy.deepcopy(self.schema)
        duplicate_enum["$defs"]["input_file"]["properties"]["role"]["enum"].append(
            "trace"
        )

        inverted_length = copy.deepcopy(self.schema)
        inverted_length["properties"]["run_id"]["maxLength"] = 0

        non_boolean_unique = copy.deepcopy(self.schema)
        non_boolean_unique["properties"]["inputs"]["uniqueItems"] = "true"

        non_boolean_additional = copy.deepcopy(self.schema)
        non_boolean_additional["$defs"]["quality"]["additionalProperties"] = {}

        scalar_all_of = copy.deepcopy(self.schema)
        scalar_all_of["$defs"]["quality"]["allOf"] = "not-an-array"

        empty_all_of = copy.deepcopy(self.schema)
        empty_all_of["$defs"]["quality"]["allOf"] = []

        scalar_prefix_items = copy.deepcopy(self.schema)
        scalar_prefix_items["properties"]["commands"]["prefixItems"] = {}

        non_schema_prefix_item = copy.deepcopy(self.schema)
        non_schema_prefix_item["properties"]["commands"]["prefixItems"] = [False]

        cases = (
            ("duplicate enum", duplicate_enum, "enum values must be unique"),
            ("inverted min/max", inverted_length, "minLength must not exceed"),
            ("non-boolean uniqueItems", non_boolean_unique, "must be boolean"),
            (
                "non-boolean additionalProperties",
                non_boolean_additional,
                "additionalProperties must be boolean",
            ),
            ("scalar allOf", scalar_all_of, "allOf must be a non-empty array"),
            ("empty allOf", empty_all_of, "allOf must be a non-empty array"),
            (
                "scalar prefixItems",
                scalar_prefix_items,
                "prefixItems must be a non-empty array",
            ),
            (
                "non-schema prefix item",
                non_schema_prefix_item,
                "every schema node must be an object",
            ),
        )
        for label, malformed, pattern in cases:
            with self.subTest(label=label):
                with self.assertRaisesRegex(validator.ValidationError, pattern):
                    validator.validate_schema(malformed)

    def test_unsealed_checkpoint_metadata_fails_closed(self) -> None:
        for name in ("SOFTMAX_CHECKPOINT_METADATA", "LAYERNORM_CHECKPOINT_METADATA"):
            with self.subTest(name=name), mock.patch.object(validator, name, None):
                self._assert_invalid("not sealed by the live seam run")

    def test_unsealed_layer0_metadata_fails_closed(self) -> None:
        with mock.patch.object(validator, "LAYER0_INPUT_METADATA", None):
            self._assert_invalid("layer-0 input metadata is not sealed")

    def test_unsealed_handoff_metadata_fails_closed(self) -> None:
        with mock.patch.object(validator, "LAYER_HANDOFF_INPUT_METADATA", None):
            self._assert_invalid("layer-handoff input metadata is not sealed")

    def test_rejects_old_schema_version(self) -> None:
        self.manifest["schema_version"] = 4
        self._assert_invalid("accepts only schema_version=6")

    def test_requires_live_git_verification(self) -> None:
        with (
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
            self.assertRaisesRegex(validator.ValidationError, "requires --verify-git"),
        ):
            validator.validate_manifest(
                self.manifest_path,
                validator.load_json(self.manifest_path),
                self.schema,
                False,
            )

    def test_rejects_local_tracking_remote_sha_drift(self) -> None:
        self.manifest["git"]["tracking_commit"] = "b" * 40
        self._assert_invalid("local, tracking, and live-remote commits differ")

    def test_rejects_live_remote_wrong_ref_or_extra_line(self) -> None:
        commit = self.manifest["git"]["remote_commit"]
        remote_arguments = [
            "ls-remote",
            "--exit-code",
            validator.REMOTE_URL,
            validator.REMOTE_REF,
        ]
        for remote_output in (
            f"{commit}\trefs/heads/wrong",
            f"{commit}\t{validator.REMOTE_REF}\n{commit}\trefs/heads/extra",
        ):
            with self.subTest(remote_output=remote_output):

                def git_output(root: Path, arguments: list[str]) -> str:
                    if arguments == remote_arguments:
                        return remote_output
                    return self._git_output(root, arguments)

                self._write_manifest()
                with (
                    mock.patch.object(validator, "_run_git", side_effect=git_output),
                    mock.patch.object(validator, "_verify_build_configuration"),
                    mock.patch.object(
                        validator,
                        "_require_profile_schedule_sealed",
                    ),
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
                ):
                    with self.assertRaisesRegex(
                        validator.ValidationError, "live remote ref differs"
                    ):
                        validator.validate_manifest(
                            self.manifest_path,
                            validator.load_json(self.manifest_path),
                            self.schema,
                            True,
                        )

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
            [validator.SYSTEM_GIT, "status"],
            0,
            stdout="clean\n",
            stderr="",
        )
        with (
            mock.patch.dict(os.environ, poisoned, clear=False),
            mock.patch.object(
                validator.subprocess,
                "run",
                return_value=completed,
            ) as run,
        ):
            output = validator._run_git(REPO_ROOT, ["status"])
        self.assertEqual(output, "clean")
        command = run.call_args.args[0]
        environment = run.call_args.kwargs["env"]
        self.assertEqual(command[0], "/usr/bin/git")
        self.assertEqual(environment["PATH"], "/usr/bin:/bin")
        self.assertEqual(environment["LANG"], "C")
        self.assertEqual(environment["LC_ALL"], "C")
        self.assertFalse(any(name.startswith("GIT_") for name in environment))
        self.assertIs(run.call_args.kwargs["stdin"], subprocess.DEVNULL)

    def test_rejects_live_origin_url_spoof(self) -> None:
        def git_output(root: Path, arguments: list[str]) -> str:
            if arguments == [
                "config",
                "--local",
                "--get-all",
                f"remote.{validator.REMOTE_NAME}.url",
            ]:
                return "https://example.invalid/spoof.git"
            return self._git_output(root, arguments)

        with (
            mock.patch.object(validator, "_run_git", side_effect=git_output),
            self.assertRaisesRegex(
                validator.ValidationError,
                "live origin URL differs",
            ),
        ):
            validator._verify_git(self.manifest, REPO_ROOT, True)

    def test_rejects_command_drift(self) -> None:
        self.manifest["commands"][9]["command"] += " -L m5"
        self._assert_invalid("commands\\[9\\]\\.command|narrow CTest")

    def test_rejects_input_path_escape(self) -> None:
        self.manifest["inputs"][0]["path"] = "../escape.csv"
        self._assert_invalid("required pattern|normalized and relative")

    def test_rejects_duplicate_input_path(self) -> None:
        self.manifest["inputs"][1]["path"] = self.manifest["inputs"][0]["path"]
        self.manifest["inputs"].sort(key=lambda record: record["path"])
        self._assert_invalid("duplicate input path")

    def test_rejects_trace_hash_tamper(self) -> None:
        trace_record = next(
            record for record in self.manifest["inputs"] if record["role"] == "trace"
        )
        trace_record["sha256"] = "0" * 64
        self._assert_invalid("SHA-256 mismatch")

    def test_rejects_layer_identity_tamper(self) -> None:
        self.manifest["contracts"]["layer_input_identities"][3][
            "weight_bundle_sha256"
        ] = "0" * 64
        self._assert_invalid("layer_input_identities")

    def test_rejects_reordered_trace_contract_layers(self) -> None:
        contract = copy.deepcopy(TRACE_CONTRACT)
        contract["layers"][0], contract["layers"][1] = (
            contract["layers"][1],
            contract["layers"][0],
        )
        with self.assertRaisesRegex(validator.ValidationError, "must be ordered"):
            validator._expected_trace_inputs(contract)

    def test_rejects_plaintext_activation_reset(self) -> None:
        self.manifest["contracts"]["execution"]["plaintext_activation_resets"] = 1
        self._assert_invalid("expected constant 0")

    def test_rejects_metadata_schedule_contract_tamper(self) -> None:
        self.manifest["contracts"]["metadata_schedule"]["initial_layer_input"][
            "scale_bits"
        ] = 50.0005
        self._assert_invalid(
            "metadata_schedule.initial_layer_input|contracts.metadata_schedule differs"
        )

    def test_rejects_layer_order_tamper(self) -> None:
        self.layer_records[0]["layer_id"] = 1
        self._write_stdout()
        self._reseal_artifacts()
        self._assert_invalid("layer_id mismatch")

    def test_rejects_chain_source_tamper(self) -> None:
        self.layer_records[4]["chain_input_source"] = "client_encrypted_trace_input"
        self._write_stdout()
        self._reseal_artifacts()
        self._assert_invalid("chain_input_source mismatch")

    def test_rejects_diagnostic_calibration_namespace_in_formal_record(self) -> None:
        self.layer_records[0]["metadata_validation_mode"] = "calibration"
        self._write_stdout()
        self._reseal_artifacts()
        self._assert_invalid("keys differ")

    def test_rejects_mixed_formal_and_diagnostic_stdout_namespaces(self) -> None:
        diagnostics = (
            {
                "test": "openfhe_encoder_metadata_calibration_layer",
                "metadata_validation_mode": "calibration",
            },
            {
                "test": "openfhe_encoder_exact_prefix",
                "claim_scope": "diagnostic_prefix_exact_schedule",
            },
        )
        for diagnostic in diagnostics:
            with self.subTest(test=diagnostic["test"]):
                self._write_stdout()
                with (self.artifact_root / "stdout.log").open(
                    "a", encoding="utf-8"
                ) as handle:
                    handle.write(json.dumps(diagnostic, sort_keys=True) + "\n")
                self._reseal_artifacts()
                self._assert_invalid("unexpected JSON evidence namespace")

    def test_rejects_handoff_refresh_placement_tamper(self) -> None:
        self.layer_records[11]["handoff_refresh_performed"] = True
        self._write_stdout()
        self._reseal_artifacts()
        self._assert_invalid("handoff_refresh_performed mismatch")

    def test_rejects_refresh_count_tamper(self) -> None:
        self.layer_records[0]["refresh_operation_counts"]["bootstraps"] = 0
        self._write_stdout()
        self._reseal_artifacts()
        self._assert_invalid(
            "refresh_operation_counts.*allowed enum|refresh_operation_counts.bootstraps"
        )

    def test_rejects_cumulative_count_tamper(self) -> None:
        self.layer_records[6]["cumulative_operation_counts"]["rotations"] += 1
        self._write_stdout()
        self._reseal_artifacts()
        self._assert_invalid(
            "cumulative_operation_counts.*allowed enum|"
            "cumulative_operation_counts.rotations"
        )

    def test_rejects_metadata_tamper(self) -> None:
        self.layer_records[2]["input_metadata"]["scale_bits"] = 99
        self._write_stdout()
        self._reseal_artifacts()
        self._assert_invalid("input_metadata.scale_bits")

    def test_metadata_accepts_measured_scale_within_tolerance(self) -> None:
        metadata = copy.deepcopy(validator.RAW_OUTPUT_METADATA)
        metadata["scale_bits"] = 100.0005
        validator._require_metadata(
            metadata, validator.RAW_OUTPUT_METADATA, "synthetic metadata"
        )

    def test_metadata_rejects_wrong_expected_scale(self) -> None:
        metadata = copy.deepcopy(validator.RAW_OUTPUT_METADATA)
        metadata["expected_scale_bits"] = 99
        with self.assertRaisesRegex(validator.ValidationError, "expected_scale_bits"):
            validator._require_metadata(
                metadata, validator.RAW_OUTPUT_METADATA, "synthetic metadata"
            )

    def test_attention_metadata_uses_two_exact_layer_regimes(self) -> None:
        self.assertEqual(
            self.layer_records[0]["attention_output_metadata"]["level"], 40
        )
        self.assertEqual(
            self.layer_records[1]["attention_output_metadata"]["level"], 31
        )
        self._validate()

        self.layer_records[1]["attention_output_metadata"] = copy.deepcopy(
            validator.LAYER0_ATTENTION_OUTPUT_METADATA
        )
        self._write_stdout()
        self._reseal_artifacts()
        self._assert_invalid("attention_output_metadata.level")

    def test_ln1_and_ffn_metadata_use_two_exact_layer_regimes(self) -> None:
        self.assertEqual(self.layer_records[0]["ln1_output_metadata"]["level"], 32)
        self.assertEqual(self.layer_records[1]["ln1_output_metadata"]["level"], 30)
        self.assertEqual(self.layer_records[0]["ffn_output_metadata"]["level"], 45)
        self.assertEqual(self.layer_records[1]["ffn_output_metadata"]["level"], 43)
        self.assertEqual(self.layer_records[0]["raw_output_metadata"]["level"], 30)
        self.assertEqual(self.layer_records[1]["raw_output_metadata"]["level"], 30)
        self._validate()

        for field, wrong_metadata in (
            ("ln1_output_metadata", validator.LAYER0_LN1_OUTPUT_METADATA),
            ("ffn_output_metadata", validator.LAYER0_FFN_OUTPUT_METADATA),
        ):
            with self.subTest(field=field):
                original = copy.deepcopy(self.layer_records[1][field])
                self.layer_records[1][field] = copy.deepcopy(wrong_metadata)
                self._write_stdout()
                self._reseal_artifacts()
                self._write_manifest()
                with self.assertRaisesRegex(
                    validator.ValidationError,
                    rf"{field}.level",
                ):
                    self._validate()
                self.layer_records[1][field] = original

    def test_rejects_metadata_remaining_levels_inconsistent_with_used_level(
        self,
    ) -> None:
        self.layer_records[2]["ffn_output_metadata"]["remaining_levels"] = 2
        self._write_stdout()
        self._reseal_artifacts()
        self._assert_invalid(
            "remaining_levels.*expected constant 3|remaining_levels drifted"
        )

    def test_each_relative_used_level_delta_rejects_self_consistent_tamper(
        self,
    ) -> None:
        layer = self.layer_records[1]
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
                output = copy.deepcopy(
                    self.layer_records[0][output_name]
                    if delta_name == "previous_raw_output_to_input_recovered"
                    else layer[output_name]
                )
                output["level"] -= 1
                output["remaining_levels"] += 1
                validator._require_metadata(output, output, "self-consistent tamper")
                with self.assertRaisesRegex(
                    validator.ValidationError,
                    "used-level delta drifted",
                ):
                    validator._require_used_level_delta(
                        output,
                        layer[anchor_name],
                        (
                            validator.RELATIVE_USED_LEVEL_DELTAS[delta_name][
                                "layers_1_to_11"
                            ]
                            if isinstance(
                                validator.RELATIVE_USED_LEVEL_DELTAS[delta_name],
                                dict,
                            )
                            else validator.RELATIVE_USED_LEVEL_DELTAS[delta_name]
                        ),
                        delta_name,
                    )

    def test_interlayer_recovery_anchors_the_previous_layer_raw_output(self) -> None:
        original = validator._require_used_level_delta
        observed: list[tuple[bool, bool]] = []

        def inspect_anchor(
            output: dict[str, object],
            anchor: dict[str, object],
            expected_delta: int,
            label: str,
        ) -> None:
            if label == (
                "stdout layer record 1.previous_raw_output_to_input_recovered"
            ):
                observed.append(
                    (
                        output is self.layer_records[0]["raw_output_metadata"],
                        output is self.layer_records[1]["raw_output_metadata"],
                    )
                )
            original(output, anchor, expected_delta, label)

        with mock.patch.object(
            validator,
            "_require_used_level_delta",
            side_effect=inspect_anchor,
        ):
            validator._verify_layer_records(self.layer_records)
        self.assertEqual(observed, [(True, False)])

    def test_rejects_post_bootstrap_checkpoint_anchor_drift(self) -> None:
        checkpoint = self.layer_records[3]["softmax_denominator_metadata"]
        checkpoint["level"] = 19
        checkpoint["remaining_levels"] = 27
        self._write_stdout()
        self._reseal_artifacts()
        self._assert_invalid("softmax_denominator_metadata.level")

    def test_checkpoint_metadata_sha256_covers_every_declared_field(self) -> None:
        baseline = validator.checkpoint_metadata_sha256(self.layer_records)
        self.assertEqual(
            self.manifest["contracts"]["metadata_schedule"][
                "checkpoint_metadata_digest"
            ]["fields"],
            list(validator.CHECKPOINT_METADATA_HASH_FIELDS),
        )
        for field in validator.CHECKPOINT_METADATA_FIELDS:
            with self.subTest(field=field):
                tampered = copy.deepcopy(self.layer_records)
                tampered[4][field]["scale_bits"] += 1e-7
                self.assertNotEqual(
                    validator.checkpoint_metadata_sha256(tampered),
                    baseline,
                )

    def test_rejects_manifest_internal_metadata_copy_mismatch(self) -> None:
        self.manifest["metrics"]["layers"][4]["ln1_output_metadata"]["scale_bits"] += (
            1e-7
        )
        self._assert_invalid("metrics.layers differs")

    def test_rejects_manifest_checkpoint_metadata_hash_tamper(self) -> None:
        self.manifest["metrics"]["checkpoint_metadata_sha256"] = "0" * 64
        self._assert_invalid("checkpoint_metadata_sha256 differs from stdout")

    def test_rejects_csv_checkpoint_metadata_hash_tamper(self) -> None:
        metrics = self.artifact_root / "metrics.csv"
        with metrics.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        rows[0]["checkpoint_metadata_sha256"] = "0" * 64
        with metrics.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=validator.CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        self._reseal_artifacts()
        self._assert_invalid("metrics.csv checkpoint_metadata_sha256 differs")

    def test_rejects_quality_threshold_tamper(self) -> None:
        self.layer_records[5]["output_quality"]["relative_l2"] = 0.051
        self._write_stdout()
        self._reseal_artifacts()
        self._assert_invalid("output_quality.relative_l2")

    def test_rejects_zero_inactive_slot_above_threshold(self) -> None:
        self.layer_records[5]["inactive_max_abs"] = 1.1e-3
        self._write_stdout()
        self._reseal_artifacts()
        self._assert_invalid("inactive_max_abs")

    def test_accepts_zero_inactive_slot_between_old_and_prototype_bounds(self) -> None:
        records = copy.deepcopy(self.layer_records)
        records[5]["inactive_max_abs"] = 5e-4
        verified = validator._verify_layer_records(records)
        summary = copy.deepcopy(self.summary)
        summary["inactive_max_abs"] = 5e-4
        validator._verify_summary(summary, verified)

    def test_rejects_missing_inactive_zero_checkpoint_count(self) -> None:
        self.layer_records[5].pop("inactive_zero_checkpoint_count")
        self._write_stdout()
        self._reseal_artifacts()
        self._assert_invalid("keys differ")

    def test_rejects_wrong_inactive_zero_checkpoint_count(self) -> None:
        self.layer_records[5]["inactive_zero_checkpoint_count"] = 27
        self._write_stdout()
        self._reseal_artifacts()
        self._assert_invalid("inactive_zero_checkpoint_count")

    def test_rejects_boolean_inactive_zero_checkpoint_count(self) -> None:
        self.layer_records[5]["inactive_zero_checkpoint_count"] = True
        self._write_stdout()
        self._reseal_artifacts()
        self._assert_invalid("inactive_zero_checkpoint_count")

    def test_rejects_negative_sentinel_diagnostic(self) -> None:
        self.layer_records[5]["inactive_sentinel_max_error"] = -1.0
        self._write_stdout()
        self._reseal_artifacts()
        self._assert_invalid("inactive_sentinel_max_error")

    def test_rejects_non_finite_sentinel_diagnostic(self) -> None:
        self.layer_records[5]["inactive_sentinel_max_error"] = float("inf")
        self._write_stdout()
        self._reseal_artifacts()
        self._assert_invalid("malformed JSON evidence")

    def test_rejects_range_escape(self) -> None:
        self.layer_records[8]["encrypted_polynomial_input_ranges"]["gelu_input"] = [
            -80.1,
            120.0,
        ]
        self._write_stdout()
        self._reseal_artifacts()
        self._assert_invalid("gelu_input escaped")

    def test_rejects_each_inactive_polynomial_sentinel_range_escape(self) -> None:
        escaped_ranges = (
            ("softmax denominator lower", "softmax_denominator", [0.009, 70.0]),
            ("softmax denominator upper", "softmax_denominator", [0.1, 80.1]),
            ("LN1 lower", "ln1_normalized_variance", [0.4999, 90.0]),
            ("LN1 upper", "ln1_normalized_variance", [45.0, 1536.1]),
            ("LN2 lower", "ln2_normalized_variance", [0.4999, 128.0]),
            ("LN2 upper", "ln2_normalized_variance", [32.0, 1536.1]),
        )
        original = copy.deepcopy(
            self.layer_records[8]["inactive_polynomial_sentinel_ranges"]
        )
        for case, name, escaped in escaped_ranges:
            with self.subTest(case=case):
                self.layer_records[8]["inactive_polynomial_sentinel_ranges"] = (
                    copy.deepcopy(original)
                )
                self.layer_records[8]["inactive_polynomial_sentinel_ranges"][name] = (
                    escaped
                )
                self._write_stdout()
                self._reseal_artifacts()
                self._write_manifest()
                with self.assertRaisesRegex(
                    validator.ValidationError,
                    rf"{name}.*(below minimum|above maximum|escaped)",
                ):
                    self._validate()

    def test_rejects_layer_sentinel_range_status_tamper(self) -> None:
        self.layer_records[4]["inactive_sentinel_range_status"] = "failed"
        self._write_stdout()
        self._reseal_artifacts()
        self._assert_invalid("inactive_sentinel_range_status mismatch")

    def test_rejects_server_decryption(self) -> None:
        self.layer_records[7]["server_decryptions"] = 1
        self._write_stdout()
        self._reseal_artifacts()
        self._assert_invalid("server_decryptions mismatch")

    def test_rejects_bool_disguised_as_omitted_integer_trust_field(self) -> None:
        self.layer_records[0]["server_decryptions"] = False
        self._write_stdout()
        self._reseal_artifacts()
        self._assert_invalid("server_decryptions mismatch")

    def test_rejects_final_summary_mismatch(self) -> None:
        self.summary["relative_l2"] += 0.001
        self._write_stdout()
        self._write_metrics()
        self._reseal_artifacts()
        self._assert_invalid("differs from layer-11")

    def test_rejects_global_inactive_sentinel_summary_mismatch(self) -> None:
        self.summary["inactive_sentinel_max_error"] = 1e-9
        self._write_stdout()
        self._write_metrics()
        self._reseal_artifacts()
        self._assert_invalid("differs from the 12-layer maximum")

    def test_rejects_final_sentinel_range_status_tamper(self) -> None:
        self.summary["inactive_sentinel_range_status"] = "failed"
        self._write_stdout()
        self._write_metrics()
        self._reseal_artifacts()
        self._assert_invalid("inactive_sentinel_range_status mismatch")

    def test_rejects_manifest_layer_sentinel_range_mismatch(self) -> None:
        self.manifest["metrics"]["layers"][2]["inactive_polynomial_sentinel_ranges"][
            "softmax_denominator"
        ] = [0.2, 60.0]
        self._assert_invalid("metrics.layers differs")

    def test_rejects_manifest_final_sentinel_status_mismatch(self) -> None:
        self.manifest["metrics"]["final_quality"]["inactive_sentinel_range_status"] = (
            "failed"
        )
        self._assert_invalid("expected constant|final_quality differs")

    def test_rejects_metrics_csv_sentinel_diagnostic_mismatch(self) -> None:
        metrics = self.artifact_root / "metrics.csv"
        with metrics.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        rows[0]["inactive_sentinel_max_error"] = "999.0"
        with metrics.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=validator.CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        self._reseal_artifacts()
        self._assert_invalid("inactive_sentinel_max_error differs")

    def test_rejects_metrics_csv_mismatch(self) -> None:
        metrics = self.artifact_root / "metrics.csv"
        text = metrics.read_text(encoding="utf-8")
        metrics.write_text(
            text.replace("non_benchmark_diagnostic", "benchmark"), encoding="utf-8"
        )
        self._reseal_artifacts()
        self._assert_invalid("latency_kind")

    def test_rejects_fractional_operation_count_after_reseal(self) -> None:
        metrics = self.artifact_root / "metrics.csv"
        with metrics.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        rows[0]["bootstraps"] = "355.0000000001"
        with metrics.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=validator.CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        self._reseal_artifacts()
        self._assert_invalid("bootstraps must be a canonical.*integer")

    def test_rejects_noncanonical_run_integer_after_reseal(self) -> None:
        metrics = self.artifact_root / "metrics.csv"
        with metrics.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        rows[0]["run"] = "01"
        with metrics.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=validator.CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        self._reseal_artifacts()
        self._assert_invalid("run must be a canonical.*integer")

    def test_rejects_extra_csv_row_column_after_reseal(self) -> None:
        metrics = self.artifact_root / "metrics.csv"
        lines = metrics.read_text(encoding="utf-8").splitlines()
        lines[1] += ",forged"
        metrics.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self._reseal_artifacts()
        self._assert_invalid("row columns differ")

    def test_rejects_extra_artifact_file(self) -> None:
        (self.artifact_root / "unexpected.txt").write_text("extra\n", encoding="utf-8")
        self._assert_invalid("exactly the sealed four-file bundle")

    def test_rejects_artifact_directory_name_run_id_mismatch(self) -> None:
        self.manifest["run_id"] = "different-run-id"
        self._assert_invalid("directory name must equal manifest.run_id")

    def test_rejects_nested_artifact_directory(self) -> None:
        nested_parent = self.output_root / "custom"
        nested_parent.mkdir()
        nested_root = nested_parent / self.run_id
        shutil.move(str(self.artifact_root), str(nested_root))
        self.artifact_root = nested_root
        self.manifest_path = nested_root / "manifest.json"
        self.manifest["commands"] = self._commands()
        self._assert_invalid("must be a direct child")

    def test_rejects_unsealed_artifact_tamper(self) -> None:
        with (self.artifact_root / "stdout.log").open("a", encoding="utf-8") as handle:
            handle.write("tamper\n")
        self._assert_invalid("byte count mismatch")

    def test_rejects_checksum_tamper_after_reseal(self) -> None:
        checksum = self.artifact_root / "SHA256SUMS"
        checksum.write_text(f"{'0' * 64}  stdout.log\n", encoding="utf-8")
        self.manifest["artifacts"][2] = self._artifact_record(
            self.artifact_root, "SHA256SUMS", "checksum"
        )
        self._assert_invalid("must cover exactly")

    def test_load_json_rejects_duplicate_keys(self) -> None:
        duplicate = self.artifact_root / "duplicate.json"
        duplicate.write_text(
            '{"schema_version":6,"schema_version":6}\n', encoding="utf-8"
        )
        with self.assertRaisesRegex(validator.ValidationError, "duplicate JSON key"):
            validator.load_json(duplicate)

    def test_load_json_rejects_non_finite(self) -> None:
        non_finite = self.artifact_root / "non-finite.json"
        non_finite.write_text('{"value":NaN}\n', encoding="utf-8")
        with self.assertRaisesRegex(validator.ValidationError, "non-finite"):
            validator.load_json(non_finite)

    def test_schema_rejects_unknown_keyword(self) -> None:
        schema = copy.deepcopy(self.schema)
        schema["unknownKeyword"] = True
        with self.assertRaisesRegex(validator.ValidationError, "unsupported schema"):
            validator.validate_schema(schema)

    def test_schema_rejects_unknown_type(self) -> None:
        schema = copy.deepcopy(self.schema)
        schema["properties"]["run_id"]["type"] = "finite-string"
        with self.assertRaisesRegex(
            validator.ValidationError, "unsupported JSON Schema type"
        ):
            validator.validate_schema(schema)

    def test_schema_rejects_ref_sibling_keyword(self) -> None:
        schema = copy.deepcopy(self.schema)
        schema["properties"]["started_at"]["description"] = "ignored sibling"
        with self.assertRaisesRegex(validator.ValidationError, "must not have sibling"):
            validator.validate_schema(schema)


if __name__ == "__main__":
    unittest.main()
