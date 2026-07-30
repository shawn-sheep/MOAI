#!/usr/bin/env python3
"""Unit tests for the fail-closed M5 12-layer artifact runner."""

from __future__ import annotations

import copy
import csv
import importlib.util
import io
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = REPO_ROOT / "scripts" / "run_openfhe_encoder12_artifact.py"
VALIDATOR_PATH = REPO_ROOT / "scripts" / "validate_openfhe_m5_artifact.py"
RESULTS_ROOT = REPO_ROOT / "results" / "openfhe"

spec = importlib.util.spec_from_file_location("run_openfhe_encoder12_artifact", RUNNER_PATH)
if spec is None or spec.loader is None:  # pragma: no cover - import machinery failure
    raise RuntimeError(f"cannot import {RUNNER_PATH}")
runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)

validator_spec = importlib.util.spec_from_file_location(
    "validate_openfhe_m5_artifact", VALIDATOR_PATH
)
if validator_spec is None or validator_spec.loader is None:  # pragma: no cover
    raise RuntimeError(f"cannot import {VALIDATOR_PATH}")
validator = importlib.util.module_from_spec(validator_spec)
sys.modules[validator_spec.name] = validator
validator_spec.loader.exec_module(validator)

TEST_LAYER0_INPUT_METADATA = {
    "level": 29,
    "noise_scale_degree": 1,
    "remaining_levels": 18,
    "scale_bits": 50,
    "expected_scale_bits": 50,
    "ciphertext_count": 5,
}
TEST_POLYNOMIAL_CHECKPOINT_METADATA = {
    "level": 18,
    "noise_scale_degree": 2,
    "remaining_levels": 28,
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


def _crypto_preflight_ctest_output(*records: dict[str, object]) -> str:
    return "".join(f"17: {json.dumps(record, sort_keys=True)}\n" for record in records)


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
        "softmax_denominator_metadata": dict(
            TEST_POLYNOMIAL_CHECKPOINT_METADATA
        ),
        "attention_output_metadata": dict(
            runner.LAYER0_ATTENTION_OUTPUT_METADATA
            if layer_id == 0
            else runner.POST_REFRESH_ATTENTION_OUTPUT_METADATA
        ),
        "ln1_variance_metadata": dict(TEST_POLYNOMIAL_CHECKPOINT_METADATA),
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
        "ln2_variance_metadata": dict(TEST_POLYNOMIAL_CHECKPOINT_METADATA),
        "input_quality": _quality(0.003),
        "output_quality": _quality(0.004),
        "exact_trace_diagnostic": _quality(0.005),
        "inactive_max_abs": 3e-7 if layer_id == 3 else 2e-7,
        "inactive_zero_checkpoint_count": 28,
        "inactive_sentinel_max_error": 0.04 if layer_id == 5 else 0.01,
        "inactive_polynomial_sentinel_ranges": {
            "softmax_denominator": [0.98, 1.02],
            "ln1_normalized_variance": [0.97, 1.03],
            "ln2_normalized_variance": [0.96, 1.04],
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


def _sample() -> object:
    layers = [_layer_record(layer_id) for layer_id in range(12)]
    final = _final_record(layers)
    stdout = "\n".join(
        json.dumps(record, sort_keys=True, separators=(",", ":"))
        for record in [*layers, final]
    ) + "\n"
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

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

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
            self.head,
            tuple(self._command_record(f"git command {index}") for index in range(5)),
        )

    def _preflight_commands(self) -> tuple[dict[str, object], ...]:
        return tuple(
            self._command_record(name) for name in ("build", "ctest", "ldd")
        )

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
            mock.patch.object(runner, "_resolve_output_root", return_value=self.output_root),
            mock.patch.object(runner, "_resolve_m5_executable", return_value=self.executable),
            mock.patch.object(runner, "_preflight_git", return_value=self._git_state()),
            mock.patch.object(
                runner,
                "_run_m5_preflight",
                return_value=runner.M5Preflight(
                    self._preflight_commands(),
                    _crypto_preflight_record(),
                ),
            ),
            mock.patch.object(runner, "_environment", return_value=self._environment()),
            mock.patch.object(runner, "_profile_record", return_value=self._profile()),
            mock.patch.object(
                runner, "_trace_input_records", return_value=self._trace_inputs()
            ),
            mock.patch.object(runner, "_repository_record", side_effect=self._repository_record),
            mock.patch.object(runner, "_require_input_hashes"),
            mock.patch.object(runner, "_run_once", return_value=sample or _sample()),
            mock.patch.object(runner, "_validate_artifact"),
        )

    def test_success_runs_once_without_warmup_and_seals_four_files(self) -> None:
        patches = self._generation_patches()
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], \
                patches[6], patches[7], patches[8] as input_hashes, \
                patches[9] as run_once, patches[10] as validate_artifact:
            run_root = runner.generate_m5_artifact(self._config("unit-m5-success"))

        run_once.assert_called_once()
        self.assertEqual(input_hashes.call_count, 2)
        validate_artifact.assert_called_once_with(run_root / "manifest.json")
        self.assertEqual(
            {path.name for path in run_root.iterdir()},
            {"manifest.json", "stdout.log", "metrics.csv", "SHA256SUMS"},
        )
        manifest = json.loads((run_root / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["schema_version"], 4)
        self.assertEqual(manifest["milestone"], "M5")
        self.assertEqual(len(manifest["commands"]), 10)
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
            schedule["attention_output_by_layer_regime"]["layers_1_to_11"][
                "level"
            ],
            31,
        )
        self.assertEqual(
            schedule["ln1_output_by_layer_regime"]["layer_0"]["level"],
            32,
        )
        self.assertEqual(
            schedule["ln1_output_by_layer_regime"]["layers_1_to_11"]["level"],
            29,
        )
        self.assertEqual(
            schedule["ffn_output_by_layer_regime"]["layer_0"]["level"],
            45,
        )
        self.assertEqual(
            schedule["ffn_output_by_layer_regime"]["layers_1_to_11"]["level"],
            42,
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

    def test_schema_alone_rejects_metadata_schedule_and_count_tamper(self) -> None:
        patches = self._generation_patches()
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], \
                patches[6], patches[7], patches[8], patches[9], patches[10]:
            run_root = runner.generate_m5_artifact(
                self._config("unit-m5-schema-const-tamper")
            )
        manifest = json.loads((run_root / "manifest.json").read_text(encoding="utf-8"))
        manifest["workload"][
            "executable_path"
        ] = "build-openfhe/openfhe_encoder_12_layer_smoke"
        schema = validator.validate_schema(
            validator.load_json(REPO_ROOT / "docs/openfhe-m5-artifact-schema.json")
        )
        validator.validate_instance(manifest, schema, schema)

        metadata_paths = (
            ("initial_layer_input",),
            ("post_refresh_layer_input",),
            ("post_bootstrap_polynomial_checkpoint",),
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

    def test_generated_manifest_passes_schema_and_semantic_validator_contract(self) -> None:
        timestamp = runner._timestamp()
        git_commands = (
            ["git", "status", "--porcelain=v1", "--untracked-files=normal"],
            ["git", "branch", "--show-current"],
            ["git", "rev-parse", "HEAD"],
            ["git", "rev-parse", runner.TRACKING_REF],
            ["git", "ls-remote", "--exit-code", runner.REMOTE_NAME, runner.REMOTE_REF],
        )
        git_state = runner.GitState(
            self.head,
            self.head,
            self.head,
            tuple(runner._command_record(command, 0) for command in git_commands),
        )
        build_root = runner.DEFAULT_EXECUTABLE.parent
        preflight_commands = (
            runner._command_record(
                [
                    "cmake",
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
                [
                    "ctest",
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
            runner._command_record(
                ["ldd", str(runner.DEFAULT_EXECUTABLE)],
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
                ),
            ),
            mock.patch.object(runner, "_environment", return_value=self._environment()),
            mock.patch.object(runner, "_run_once", return_value=sample),
            mock.patch.object(runner, "_validate_artifact"),
        ):
            run_root = runner.generate_m5_artifact(config)

        manifest_path = run_root / "manifest.json"
        schema = validator.validate_schema(
            validator.load_json(REPO_ROOT / "docs" / "openfhe-m5-artifact-schema.json")
        )
        with mock.patch.object(validator, "_verify_git") as verify_git:
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
            runner._validate_layer_record(formal_layer_with_diagnostic_fields, 0)

        formal_layers = [_layer_record(index) for index in range(12)]
        formal_summary_with_diagnostic_fields = _final_record(formal_layers)
        formal_summary_with_diagnostic_fields["metadata_validation_mode"] = "calibration"
        with self.assertRaisesRegex(runner.ArtifactRunnerError, "diagnostic-only JSON"):
            runner._extract_records(
                json.dumps(formal_summary_with_diagnostic_fields),
                "openfhe_encoder_12_layer",
            )
        formal_summary_with_diagnostic_scope = _final_record(formal_layers)
        formal_summary_with_diagnostic_scope[
            "claim_scope"
        ] = "diagnostic_metadata_calibration_only"
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
        self.assertEqual(runner.LAYER0_INPUT_METADATA, validator.LAYER0_INPUT_METADATA)
        self.assertEqual(
            runner.LAYER_HANDOFF_INPUT_METADATA,
            validator.LAYER_HANDOFF_INPUT_METADATA,
        )
        self.assertEqual(
            runner.POLYNOMIAL_CHECKPOINT_METADATA,
            validator.POLYNOMIAL_CHECKPOINT_METADATA,
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
        self.assertEqual(
            runner.CHECKPOINT_METADATA_HASH_FIELDS,
            validator.CHECKPOINT_METADATA_HASH_FIELDS,
        )
        self.assertEqual(
            runner.CRYPTO_PREFLIGHT_KEYS,
            validator.CRYPTO_PREFLIGHT_KEYS,
        )
        self.assertEqual(
            {key: list(value) for key, value in runner.FROZEN_RANGES.items()},
            validator.POLYNOMIAL_INTERVALS,
        )

    def test_preflight_commands_are_frozen(self) -> None:
        record = self._command_record("synthetic preflight")
        with (
            mock.patch.object(runner, "_verify_build_configuration") as build_config,
            mock.patch.object(
                runner,
                "_run_checked_command",
                side_effect=[
                    (record, "synthetic build output\n"),
                    (
                        record,
                        _crypto_preflight_ctest_output(
                            _crypto_preflight_record()
                        ),
                    ),
                ],
            ) as checked,
            mock.patch.object(
                runner, "_resolve_m5_executable", return_value=self.executable
            ),
            mock.patch.object(
                runner, "_verify_openfhe_linkage", return_value=record
            ) as linkage,
        ):
            preflight = runner._run_m5_preflight(
                runner.DEFAULT_EXECUTABLE.parent,
                runner.DEFAULT_EXECUTABLE,
                runner.DEFAULT_OPENFHE_PREFIX,
            )
        self.assertEqual(len(preflight.commands), 3)
        self.assertEqual(preflight.crypto_preflight, _crypto_preflight_record())
        build_config.assert_called_once_with(
            runner.DEFAULT_EXECUTABLE.parent, runner.DEFAULT_OPENFHE_PREFIX
        )
        self.assertEqual(
            checked.call_args_list[0].args[0],
            [
                "cmake",
                "--build",
                str(runner.DEFAULT_EXECUTABLE.parent),
                "--clean-first",
                "-j",
                "4",
            ],
        )
        self.assertEqual(
            checked.call_args_list[1].args[0],
            [
                "ctest",
                "--test-dir",
                str(runner.DEFAULT_EXECUTABLE.parent),
                "--output-on-failure",
                "--verbose",
                "--no-tests=error",
                "-R",
                runner.M5_CTEST_PATTERN,
            ],
        )
        linkage.assert_called_once_with(
            self.executable, runner.DEFAULT_OPENFHE_PREFIX
        )

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
        with self.assertRaisesRegex(runner.ArtifactRunnerError, "additive_he_operations"):
            runner._parse_crypto_preflight_output(
                _crypto_preflight_ctest_output(tampered)
            )

    def test_crypto_preflight_parser_rejects_actual_scale_outside_tolerance(self) -> None:
        with self.assertRaisesRegex(runner.ArtifactRunnerError, "more than 1e-3"):
            runner._parse_crypto_preflight_output(
                _crypto_preflight_ctest_output(
                    _crypto_preflight_record(scale_bits=50.0011)
                )
            )

    def test_command_count_drift_is_fail_closed_and_removes_run_directory(self) -> None:
        patches = list(self._generation_patches())
        patches[3] = mock.patch.object(
            runner,
            "_run_m5_preflight",
            return_value=runner.M5Preflight(
                self._preflight_commands()[:2],
                _crypto_preflight_record(),
            ),
        )
        config = self._config("unit-m5-command-drift")
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], \
                patches[6], patches[7], patches[8], patches[9], patches[10], \
                self.assertRaisesRegex(runner.ArtifactRunnerError, "10 records"):
            runner.generate_m5_artifact(config)
        self.assertFalse((self.output_root / config.run_id).exists())

    def test_input_hash_drift_after_run_is_fail_closed(self) -> None:
        patches = list(self._generation_patches())
        patches[8] = mock.patch.object(
            runner,
            "_require_input_hashes",
            side_effect=[None, runner.ArtifactRunnerError("synthetic input hash drift")],
        )
        config = self._config("unit-m5-input-hash-drift")
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], \
                patches[6], patches[7], patches[8], patches[9], patches[10], \
                self.assertRaisesRegex(runner.ArtifactRunnerError, "input hash drift"):
            runner.generate_m5_artifact(config)
        self.assertFalse((self.output_root / config.run_id).exists())

    def test_executable_hash_drift_after_run_is_fail_closed(self) -> None:
        patches = list(self._generation_patches())
        executable_hash = runner._sha256(self.executable)
        patches.append(
            mock.patch.object(
                runner,
                "_require_executable_hash",
                side_effect=[None, runner.ArtifactRunnerError("synthetic executable hash drift")],
            )
        )
        config = self._config("unit-m5-executable-hash-drift")
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], \
                patches[6], patches[7], patches[8], patches[9], patches[10], patches[11], \
                self.assertRaisesRegex(runner.ArtifactRunnerError, "executable hash drift"):
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
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], \
                patches[6], patches[7], patches[8], patches[9], patches[10], \
                self.assertRaisesRegex(runner.ArtifactRunnerError, "validator rejection"):
            runner.generate_m5_artifact(config)
        self.assertFalse((self.output_root / config.run_id).exists())

    def test_existing_run_directory_is_preserved(self) -> None:
        patches = self._generation_patches()
        config = self._config("unit-m5-existing")
        run_root = self.output_root / config.run_id
        run_root.mkdir(parents=True)
        sentinel = run_root / "sentinel"
        sentinel.write_text("preserve", encoding="utf-8")
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], \
                patches[6], patches[7], patches[8], patches[9], patches[10], \
                self.assertRaisesRegex(runner.ArtifactRunnerError, "already exists"):
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
                [(" M scripts/run_openfhe_encoder12_artifact.py", record)],
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
                    (f"{self.head}\t{runner.REMOTE_REF}", record),
                ],
                "commits differ",
            ),
            (
                "tracking_live_mismatch",
                [
                    ("", record),
                    (runner.BRANCH, record),
                    (self.head, record),
                    (self.head, record),
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
                    (f"{remote_sha}\t{runner.REMOTE_REF}", record),
                ],
                "commits differ",
            ),
        )
        for name, side_effect, message in cases:
            with (
                self.subTest(case=name),
                mock.patch.object(runner, "_run_git", side_effect=side_effect) as run_git,
                self.assertRaisesRegex(runner.ArtifactRunnerError, message),
            ):
                runner._preflight_git()
            self.assertLessEqual(run_git.call_count, 5)

    def test_actual_trace_inventory_binds_448_inputs_and_unique_layer_bundles(self) -> None:
        records, identities = runner._trace_input_records(runner.DEFAULT_DATA_ROOT)
        self.assertEqual(len(records), 444)
        self.assertEqual(len({record["path"] for record in records}), 444)
        self.assertEqual(len(identities), 12)
        self.assertEqual(
            {(item["weight_file_count"], item["trace_file_count"]) for item in identities},
            {(16, 21)},
        )
        self.assertEqual(len({item["weight_bundle_sha256"] for item in identities}), 12)
        self.assertEqual(len({item["trace_bundle_sha256"] for item in identities}), 12)

    def test_runtime_records_are_strict_and_fail_closed(self) -> None:
        layers = [_layer_record(layer_id) for layer_id in range(12)]
        with mock.patch.object(
            runner,
            "POLYNOMIAL_CHECKPOINT_METADATA",
            TEST_POLYNOMIAL_CHECKPOINT_METADATA,
        ), mock.patch.object(
            runner,
            "LAYER0_INPUT_METADATA",
            TEST_LAYER0_INPUT_METADATA,
        ), mock.patch.object(
            runner,
            "LAYER_HANDOFF_INPUT_METADATA",
            TEST_HANDOFF_INPUT_METADATA,
        ):
            for layer_id, record in enumerate(layers):
                runner._validate_layer_record(record, layer_id)
            runner._validate_final_record(_final_record(layers), layers)

            missing_checkpoint_count = copy.deepcopy(layers[5])
            missing_checkpoint_count.pop("inactive_zero_checkpoint_count")
            with self.assertRaisesRegex(runner.ArtifactRunnerError, "keys differ"):
                runner._validate_layer_record(missing_checkpoint_count, 5)

            for invalid_count in (27, True):
                invalid_checkpoint_count = copy.deepcopy(layers[5])
                invalid_checkpoint_count["inactive_zero_checkpoint_count"] = invalid_count
                with (
                    self.subTest(inactive_zero_checkpoint_count=invalid_count),
                    self.assertRaisesRegex(
                        runner.ArtifactRunnerError,
                        "inactive_zero_checkpoint_count",
                    ),
                ):
                    runner._validate_layer_record(invalid_checkpoint_count, 5)

        bad_layer = copy.deepcopy(layers[4])
        bad_layer["chain_input_source"] = "trace_reset"
        with (
            mock.patch.object(
                runner,
                "POLYNOMIAL_CHECKPOINT_METADATA",
                TEST_POLYNOMIAL_CHECKPOINT_METADATA,
            ),
            mock.patch.object(
                runner,
                "LAYER_HANDOFF_INPUT_METADATA",
                TEST_HANDOFF_INPUT_METADATA,
            ),
            self.assertRaisesRegex(runner.ArtifactRunnerError, "chain_input_source"),
        ):
            runner._validate_layer_record(bad_layer, 4)

        bad_range = copy.deepcopy(layers[7])
        bad_range["encrypted_polynomial_input_ranges"]["gelu_input"][1] = 128.1
        with (
            mock.patch.object(
                runner,
                "POLYNOMIAL_CHECKPOINT_METADATA",
                TEST_POLYNOMIAL_CHECKPOINT_METADATA,
            ),
            mock.patch.object(
                runner,
                "LAYER_HANDOFF_INPUT_METADATA",
                TEST_HANDOFF_INPUT_METADATA,
            ),
            self.assertRaisesRegex(runner.ArtifactRunnerError, "gelu_input"),
        ):
            runner._validate_layer_record(bad_range, 7)

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
                "POLYNOMIAL_CHECKPOINT_METADATA",
                TEST_POLYNOMIAL_CHECKPOINT_METADATA,
            ),
            mock.patch.object(
                runner,
                "LAYER_HANDOFF_INPUT_METADATA",
                TEST_HANDOFF_INPUT_METADATA,
            ),
        ):
            runner._validate_layer_record(layer, 5)

            bad_zero_tail = copy.deepcopy(layer)
            bad_zero_tail["inactive_max_abs"] = 1.1e-6
            with self.assertRaisesRegex(runner.ArtifactRunnerError, "inactive_max_abs"):
                runner._validate_layer_record(bad_zero_tail, 5)

            for invalid_diagnostic in (-1.0, float("inf")):
                bad_diagnostic = copy.deepcopy(layer)
                bad_diagnostic["inactive_sentinel_max_error"] = invalid_diagnostic
                with (
                    self.subTest(invalid_diagnostic=invalid_diagnostic),
                    self.assertRaisesRegex(
                        runner.ArtifactRunnerError, "inactive_sentinel_max_error"
                    ),
                ):
                    runner._validate_layer_record(bad_diagnostic, 5)

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
                        runner._validate_layer_record(bad_interval, 5)

            bad_status = copy.deepcopy(layer)
            bad_status["inactive_sentinel_range_status"] = "unchecked"
            with self.assertRaisesRegex(
                runner.ArtifactRunnerError, "inactive_sentinel_range_status"
            ):
                runner._validate_layer_record(bad_status, 5)

    def test_frozen_scalars_reject_python_bool_integer_equivalence(self) -> None:
        layer = _layer_record(0)
        layer["server_decryptions"] = False
        with (
            mock.patch.object(
                runner,
                "POLYNOMIAL_CHECKPOINT_METADATA",
                TEST_POLYNOMIAL_CHECKPOINT_METADATA,
            ),
            mock.patch.object(
                runner,
                "LAYER0_INPUT_METADATA",
                TEST_LAYER0_INPUT_METADATA,
            ),
            self.assertRaisesRegex(runner.ArtifactRunnerError, "server_decryptions"),
        ):
            runner._validate_layer_record(layer, 0)

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
        runner._validate_layer_record(layer0, 0)
        runner._validate_layer_record(layer1, 1)
        self.assertEqual(layer0["attention_output_metadata"]["level"], 40)
        self.assertEqual(layer1["attention_output_metadata"]["level"], 31)
        self.assertEqual(layer0["ln1_output_metadata"]["level"], 32)
        self.assertEqual(layer1["ln1_output_metadata"]["level"], 29)
        self.assertEqual(layer0["ffn_output_metadata"]["level"], 45)
        self.assertEqual(layer1["ffn_output_metadata"]["level"], 42)

        schedule = runner._metadata_schedule_contract()
        self.assertEqual(
            schedule["relative_used_level_deltas"][
                "attention_output_from_layer_input"
            ],
            {"layer_0": 12, "layers_1_to_11": 12},
        )
        self.assertEqual(
            schedule["relative_used_level_deltas"][
                "ln1_output_from_ln1_variance"
            ],
            {"layer_0": 14, "layers_1_to_11": 11},
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
                    runner._validate_layer_record(forged, 1)

    def test_metadata_remaining_levels_is_derived_from_used_level(self) -> None:
        forged = copy.deepcopy(_layer_record(2))
        forged["ffn_output_metadata"]["remaining_levels"] = 2
        with self.assertRaisesRegex(
            runner.ArtifactRunnerError,
            "remaining_levels drifted",
        ):
            runner._validate_layer_record(forged, 2)

    def test_each_relative_used_level_delta_rejects_self_consistent_tamper(self) -> None:
        layer = _layer_record(1)
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
            runner._validate_layer_record(forged, 3)

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
                "POLYNOMIAL_CHECKPOINT_METADATA",
                TEST_POLYNOMIAL_CHECKPOINT_METADATA,
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
                "POLYNOMIAL_CHECKPOINT_METADATA",
                TEST_POLYNOMIAL_CHECKPOINT_METADATA,
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
            self.assertRaisesRegex(runner.ArtifactRunnerError, "stderr despite exit_code=0"),
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
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], \
                patches[6], patches[7], patches[8], patches[9], patches[10], \
                self.assertRaisesRegex(runner.ArtifactRunnerError, "stderr"):
            runner.generate_m5_artifact(config)
        self.assertFalse((self.output_root / config.run_id).exists())

    def test_missing_runtime_records_are_fail_closed(self) -> None:
        def synthetic_run(command: list[str], **_: object) -> object:
            output_argument = next(item for item in command if item.startswith("--output="))
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
                "POLYNOMIAL_CHECKPOINT_METADATA",
                TEST_POLYNOMIAL_CHECKPOINT_METADATA,
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
            "POLYNOMIAL_CHECKPOINT_METADATA": TEST_POLYNOMIAL_CHECKPOINT_METADATA,
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
