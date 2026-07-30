#!/usr/bin/env python3
"""Positive and tamper-negative tests for the independent M5 validator."""

from __future__ import annotations

import copy
import csv
import json
import shlex
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import validate_openfhe_m5_artifact as validator  # noqa: E402


ORIGINAL_SHA256_FILE = validator.sha256_file
TRACE_CONTRACT = validator.load_json(REPO_ROOT / "config/moai_encoder_trace.json")
TRACE_HASHES: dict[str, str] = {}
for _layer in TRACE_CONTRACT["layers"]:
    for _name, _specification in TRACE_CONTRACT["required_files"].items():
        _path = (
            Path("data")
            / f"layer_{_layer['layer_id']}"
            / _specification["path"]
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
            validator.load_json(REPO_ROOT / "docs/openfhe-m5-artifact-schema.json")
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
                validator.REFRESH_COUNTS
                if layer_id < 11
                else validator.ZERO_COUNTS
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
                        validator.POLYNOMIAL_CHECKPOINT_METADATA
                    ),
                    "attention_output_metadata": self._metadata(
                        validator.LAYER0_ATTENTION_OUTPUT_METADATA
                        if layer_id == 0
                        else validator.POST_REFRESH_ATTENTION_OUTPUT_METADATA
                    ),
                    "ln1_variance_metadata": self._metadata(
                        validator.POLYNOMIAL_CHECKPOINT_METADATA
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
                        validator.POLYNOMIAL_CHECKPOINT_METADATA
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
                        "ln1_normalized_variance": [0.6, 1400.0],
                        "ln2_normalized_variance": [0.7, 1300.0],
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
                    "layer_operation_counts": copy.deepcopy(
                        validator.LAYER_COUNTS
                    ),
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
                record["inactive_sentinel_max_error"]
                for record in self.layer_records
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
            "\n".join(json.dumps(record, sort_keys=True) for record in records)
            + "\n",
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
            "server_online_diagnostic_ms": self.summary[
                "server_online_diagnostic_ms"
            ],
            "client_checkpoint_validate_ms": self.summary[
                "client_checkpoint_validate_ms"
            ],
            "final_rel_l2": self.summary["relative_l2"],
            "final_cosine": self.summary["cosine"],
            "final_max_absolute": self.summary["max_absolute"],
            "inactive_max_abs": self.summary["inactive_max_abs"],
            "inactive_sentinel_max_error": self.summary[
                "inactive_sentinel_max_error"
            ],
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
    def _input_record(path: str, role: str, sha256: str | None = None) -> dict[str, object]:
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
                    Path("data")
                    / f"layer_{layer_id}"
                    / specification["path"]
                ).as_posix()
                role = (
                    "weights"
                    if "/parms/" in f"/{specification['path']}"
                    else "trace"
                )
                record = self._input_record(
                    path, role, layer["sha256"][logical_name]
                )
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
                ["git", "status", "--porcelain=v1", "--untracked-files=normal"]
            ),
            self._command(["git", "branch", "--show-current"]),
            self._command(["git", "rev-parse", "HEAD"]),
            self._command(["git", "rev-parse", validator.TRACKING_REF]),
            self._command(
                [
                    "git",
                    "ls-remote",
                    "--exit-code",
                    validator.REMOTE_NAME,
                    validator.REMOTE_REF,
                ]
            ),
            self._command(
                ["cmake", "--build", str(build), "--clean-first", "-j", "4"],
                timed=True,
            ),
            self._command(
                [
                    "ctest",
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
            self._command(["ldd", str(executable)], timed=True),
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
                    "--manifest",
                    str(self.manifest_path.resolve()),
                    "--verify-git",
                ]
            ),
        ]

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
            "schema_version": 4,
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
                "remote_ref": validator.REMOTE_REF,
                "remote_commit": commit,
            },
            "commands": self._commands(),
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
                "metadata_schedule": copy.deepcopy(
                    validator._expected_metadata_schedule_contract()
                ),
                "thresholds": copy.deepcopy(validator.THRESHOLDS),
                "polynomial_intervals": copy.deepcopy(
                    validator.POLYNOMIAL_INTERVALS
                ),
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
            "artifacts": [],
            "claim_boundary": copy.deepcopy(validator.CLAIM_BOUNDARY),
            "verdict": "GO",
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
        if arguments == ["branch", "--show-current"]:
            return validator.BRANCH
        if arguments == ["status", "--porcelain=v1", "--untracked-files=normal"]:
            return ""
        if arguments == ["rev-parse", validator.TRACKING_REF]:
            return commit
        if arguments == [
            "ls-remote",
            "--exit-code",
            validator.REMOTE_NAME,
            validator.REMOTE_REF,
        ]:
            return f"{commit}\t{validator.REMOTE_REF}"
        raise AssertionError(arguments)

    def _validate(self) -> None:
        with mock.patch.object(validator, "_run_git", side_effect=self._git_output):
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

    def test_accepts_sentinel_diagnostic_above_zero_inactive_threshold(self) -> None:
        self.assertGreater(self.summary["inactive_sentinel_max_error"], 1e-6)
        self._validate()

    def test_schema_contract_is_independent_v4_m5(self) -> None:
        self.assertEqual(self.schema["properties"]["schema_version"]["const"], 4)
        self.assertTrue(self.schema["$id"].endswith("schema-v4.json"))
        self.assertEqual(self.schema["title"], validator.SCHEMA_TITLE)
        self.assertEqual(self.schema["properties"]["milestone"]["const"], "M5")
        self.assertEqual(self.schema["properties"]["inputs"]["minItems"], 448)
        self.assertIn("crypto_preflight", self.schema["properties"]["contracts"]["required"])

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
            schedule["post_bootstrap_polynomial_checkpoint"],
            validator.POLYNOMIAL_CHECKPOINT_METADATA,
        )
        self.assertEqual(
            schedule["relative_used_level_deltas"][
                "ln1_output_from_ln1_variance"
            ],
            {"layer_0": 14, "layers_1_to_11": 11},
        )

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

    def test_schema_identity_rejects_stale_v3_title(self) -> None:
        schema = copy.deepcopy(self.schema)
        schema["title"] = "MOAI OpenFHE M5 evidence manifest v3"
        with self.assertRaisesRegex(validator.ValidationError, "schema identity drifted"):
            validator.validate_schema(schema)

    def test_schema_engine_rejects_malformed_defensive_keywords(self) -> None:
        duplicate_enum = copy.deepcopy(self.schema)
        duplicate_enum["$defs"]["input_file"]["properties"]["role"][
            "enum"
        ].append("trace")

        inverted_length = copy.deepcopy(self.schema)
        inverted_length["properties"]["run_id"]["maxLength"] = 0

        non_boolean_unique = copy.deepcopy(self.schema)
        non_boolean_unique["properties"]["claim_boundary"]["uniqueItems"] = (
            "true"
        )

        non_boolean_additional = copy.deepcopy(self.schema)
        non_boolean_additional["$defs"]["quality"]["additionalProperties"] = {}

        scalar_all_of = copy.deepcopy(self.schema)
        scalar_all_of["$defs"]["quality"]["allOf"] = "not-an-array"

        empty_all_of = copy.deepcopy(self.schema)
        empty_all_of["$defs"]["quality"]["allOf"] = []

        scalar_prefix_items = copy.deepcopy(self.schema)
        scalar_prefix_items["properties"]["claim_boundary"]["prefixItems"] = {}

        non_schema_prefix_item = copy.deepcopy(self.schema)
        non_schema_prefix_item["properties"]["claim_boundary"]["prefixItems"] = [
            False
        ]

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

    def test_unsealed_polynomial_metadata_fails_closed(self) -> None:
        with mock.patch.object(validator, "POLYNOMIAL_CHECKPOINT_METADATA", None):
            self._assert_invalid("not sealed by the two-layer seam run")

    def test_unsealed_layer0_metadata_fails_closed(self) -> None:
        with mock.patch.object(validator, "LAYER0_INPUT_METADATA", None):
            self._assert_invalid("layer-0 input metadata is not sealed")

    def test_unsealed_handoff_metadata_fails_closed(self) -> None:
        with mock.patch.object(validator, "LAYER_HANDOFF_INPUT_METADATA", None):
            self._assert_invalid("layer-handoff input metadata is not sealed")

    def test_rejects_old_schema_version(self) -> None:
        self.manifest["schema_version"] = 3
        self._assert_invalid("accepts only schema_version=4")

    def test_requires_live_git_verification(self) -> None:
        with self.assertRaisesRegex(validator.ValidationError, "requires --verify-git"):
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
            validator.REMOTE_NAME,
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
                with mock.patch.object(
                    validator, "_run_git", side_effect=git_output
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

    def test_rejects_command_drift(self) -> None:
        self.manifest["commands"][6]["command"] += " -L m5"
        self._assert_invalid("narrow CTest")

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
            "refresh_operation_counts.*allowed enum|"
            "refresh_operation_counts.bootstraps"
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
        self.assertEqual(self.layer_records[0]["attention_output_metadata"]["level"], 40)
        self.assertEqual(self.layer_records[1]["attention_output_metadata"]["level"], 31)
        self._validate()

        self.layer_records[1]["attention_output_metadata"] = copy.deepcopy(
            validator.LAYER0_ATTENTION_OUTPUT_METADATA
        )
        self._write_stdout()
        self._reseal_artifacts()
        self._assert_invalid("attention_output_metadata.level")

    def test_ln1_and_ffn_metadata_use_two_exact_layer_regimes(self) -> None:
        self.assertEqual(self.layer_records[0]["ln1_output_metadata"]["level"], 32)
        self.assertEqual(self.layer_records[1]["ln1_output_metadata"]["level"], 29)
        self.assertEqual(self.layer_records[0]["ffn_output_metadata"]["level"], 45)
        self.assertEqual(self.layer_records[1]["ffn_output_metadata"]["level"], 42)
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

    def test_rejects_metadata_remaining_levels_inconsistent_with_used_level(self) -> None:
        self.layer_records[2]["ffn_output_metadata"]["remaining_levels"] = 2
        self._write_stdout()
        self._reseal_artifacts()
        self._assert_invalid(
            "remaining_levels.*expected constant 4|remaining_levels drifted"
        )

    def test_each_relative_used_level_delta_rejects_self_consistent_tamper(self) -> None:
        layer = self.layer_records[1]
        cases = (
            (
                "attention_output_from_layer_input",
                "attention_output_metadata",
                "input_metadata",
            ),
            (
                "ln1_output_from_ln1_variance",
                "ln1_output_metadata",
                "ln1_variance_metadata",
            ),
            (
                "ffn_output_from_ln1_output",
                "ffn_output_metadata",
                "ln1_output_metadata",
            ),
            (
                "raw_output_from_ln2_variance",
                "raw_output_metadata",
                "ln2_variance_metadata",
            ),
        )
        for delta_name, output_name, anchor_name in cases:
            with self.subTest(delta=delta_name):
                output = copy.deepcopy(layer[output_name])
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
        self.manifest["metrics"]["layers"][4]["ln1_output_metadata"][
            "scale_bits"
        ] += 1e-7
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
        self.layer_records[5]["inactive_max_abs"] = 1.1e-6
        self._write_stdout()
        self._reseal_artifacts()
        self._assert_invalid("inactive_max_abs")

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
            ("LN1 lower", "ln1_normalized_variance", [0.49, 1400.0]),
            ("LN1 upper", "ln1_normalized_variance", [0.6, 1536.1]),
            ("LN2 lower", "ln2_normalized_variance", [0.49, 1300.0]),
            ("LN2 upper", "ln2_normalized_variance", [0.7, 1536.1]),
        )
        original = copy.deepcopy(
            self.layer_records[8]["inactive_polynomial_sentinel_ranges"]
        )
        for case, name, escaped in escaped_ranges:
            with self.subTest(case=case):
                self.layer_records[8]["inactive_polynomial_sentinel_ranges"] = (
                    copy.deepcopy(original)
                )
                self.layer_records[8]["inactive_polynomial_sentinel_ranges"][
                    name
                ] = escaped
                self._write_stdout()
                self._reseal_artifacts()
                self._write_manifest()
                with self.assertRaisesRegex(
                    validator.ValidationError,
                    rf"{name} escaped",
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
        self.manifest["metrics"]["layers"][2][
            "inactive_polynomial_sentinel_ranges"
        ]["softmax_denominator"] = [0.2, 60.0]
        self._assert_invalid("metrics.layers differs")

    def test_rejects_manifest_final_sentinel_status_mismatch(self) -> None:
        self.manifest["metrics"]["final_quality"][
            "inactive_sentinel_range_status"
        ] = "failed"
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
        metrics.write_text(text.replace("non_benchmark_diagnostic", "benchmark"), encoding="utf-8")
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
        duplicate.write_text('{"schema_version":4,"schema_version":4}\n', encoding="utf-8")
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
        with self.assertRaisesRegex(validator.ValidationError, "unsupported JSON Schema type"):
            validator.validate_schema(schema)

    def test_schema_rejects_ref_sibling_keyword(self) -> None:
        schema = copy.deepcopy(self.schema)
        schema["properties"]["started_at"]["description"] = "ignored sibling"
        with self.assertRaisesRegex(validator.ValidationError, "must not have sibling"):
            validator.validate_schema(schema)


if __name__ == "__main__":
    unittest.main()
