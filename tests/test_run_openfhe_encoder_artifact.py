#!/usr/bin/env python3
"""Unit tests for the fail-closed M4 encoder artifact runner."""

from __future__ import annotations

import copy
import csv
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = REPO_ROOT / "scripts" / "run_openfhe_encoder_artifact.py"
RESULTS_ROOT = REPO_ROOT / "results" / "openfhe"
BUILD_ROOT = REPO_ROOT / "build-openfhe"

spec = importlib.util.spec_from_file_location("run_openfhe_encoder_artifact", RUNNER_PATH)
if spec is None or spec.loader is None:  # pragma: no cover - import machinery failure
    raise RuntimeError(f"cannot import {RUNNER_PATH}")
runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)
ORIGINAL_RESOLVE_M4_EXECUTABLE = runner._resolve_m4_executable
ORIGINAL_TRACE_INPUT_RECORDS = runner._trace_input_records
ORIGINAL_REQUIRE_INPUT_HASHES = runner._require_input_hashes
ORIGINAL_RUN_M4_PREFLIGHT = runner._run_m4_preflight
ORIGINAL_VERIFY_BUILD_CONFIGURATION = runner._verify_build_configuration

sys.path.insert(0, str(REPO_ROOT / "scripts"))
import validate_openfhe_artifact as validator  # noqa: E402


def _contract_trace_records() -> list[dict[str, object]]:
    contract = json.loads(
        (REPO_ROOT / "config" / "moai_encoder_trace.json").read_text(
            encoding="utf-8"
        )
    )
    layer = next(item for item in contract["layers"] if item["layer_id"] == 1)
    records: list[dict[str, object]] = []
    for logical_name, specification in sorted(contract["required_files"].items()):
        relative = Path("data") / "layer_1" / specification["path"]
        resolved = REPO_ROOT / relative
        records.append(
            {
                "path": relative.as_posix(),
                "sha256": layer["sha256"][logical_name],
                "bytes": resolved.stat().st_size,
                "media_type": "text/csv",
                "role": "weights" if "/parms/" in specification["path"] else "trace",
            }
        )
    return records


def _runtime_record() -> dict[str, object]:
    return {
        "test": "openfhe_encoder_layer",
        "profile": "paper_compat",
        "security_claim": "none",
        "parameter_sha256": runner.PROFILE_SHA256,
        "execution_mode": "server-only",
        "encoder_layers": 1,
        "layer_id": 1,
        "input_level": 29,
        "output_level": 29,
        "remaining_levels": 17,
        "multiplicative_depth": runner.EXPECTED_MULTIPLICATIVE_DEPTH,
        "max_observed_level": runner.EXPECTED_MAX_OBSERVED_LEVEL,
        "max_polynomial_depth": runner.EXPECTED_MAX_POLYNOMIAL_DEPTH,
        "actual_trace_shape": [5, 768],
        "actual_trace_value_count": 3840,
        "feature_block_size": 1024,
        "checkpoint_decryption_owner": "client",
        "server_private_key_present": False,
        "server_decryptions": 0,
        "server_plaintext_activations": False,
        "relative_l2": 0.004,
        "cosine": 0.9999,
        "inactive_max_abs": 2e-7,
        "checkpoints": [
            {
                "name": name,
                "level": (40, 32, 45, 29)[index],
                "noise_scale_degree": 2,
                "remaining_levels": (6, 14, 1, 17)[index],
                "scale_bits": 100,
                "ciphertext_count": (5, 5, 5, 5)[index],
                "decryption_owner": "client",
            }
            for index, name in enumerate(runner.CHECKPOINT_NAMES)
        ],
        "operation_counts": {
            "rotations": 6300,
            "ct_pt_multiplications": 51865,
            "ct_ct_multiplications": 95,
            "explicit_rescale_requests": 800,
            "chebyshev_evaluations": 55,
            "estimated_polynomial_multiplications": 1150,
            "bootstraps": 25,
            "bootstrap_iterations": 50,
        },
    }


def _diagnostic_record(target: dict[str, object]) -> dict[str, object]:
    counts = target["operation_counts"]
    return {
        "test": "openfhe_encoder_layer_smoke",
        "profile": target["profile"],
        "security_claim": target["security_claim"],
        "parameter_sha256": target["parameter_sha256"],
        "layer": target["layer_id"],
        "input_level": target["input_level"],
        "tokens": 5,
        "hidden_size": 768,
        "intermediate_size": 3072,
        "feature_block": 1024,
        "fixture_load_ms": 10.0,
        "setup_keygen_ms": 100.0,
        "client_encrypt_ms": 20.0,
        "server_online_ms": 500.0,
        "client_decrypt_validate_ms": 30.0,
        "relative_l2": target["relative_l2"],
        "cosine": target["cosine"],
        "max_absolute": 0.01,
        "exact_trace_relative_l2": 0.006,
        "exact_trace_cosine": 0.9998,
        "exact_trace_max_absolute": 0.02,
        "inactive_max_absolute": target["inactive_max_abs"],
        "peak_rss_bytes": 1024,
        "rotations": counts["rotations"],
        "ct_pt_multiplications": counts["ct_pt_multiplications"],
        "ct_ct_multiplications": counts["ct_ct_multiplications"],
        "explicit_rescale_requests": counts["explicit_rescale_requests"],
        "chebyshev_evaluations": counts["chebyshev_evaluations"],
        "estimated_polynomial_multiplications": counts[
            "estimated_polynomial_multiplications"
        ],
        "bootstraps": counts["bootstraps"],
        "bootstrap_iterations": counts["bootstrap_iterations"],
        "final_level": target["output_level"],
        "final_remaining_levels": target["remaining_levels"],
        "multiplicative_depth": target["multiplicative_depth"],
        "max_observed_level": target["max_observed_level"],
        "max_polynomial_depth": target["max_polynomial_depth"],
    }


class EncoderArtifactRunnerTest(unittest.TestCase):
    def setUp(self) -> None:
        RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
        BUILD_ROOT.mkdir(parents=True, exist_ok=True)
        self.output_directory = tempfile.TemporaryDirectory(
            prefix="runner-unit-", dir=RESULTS_ROOT
        )
        self.fake_directory = tempfile.TemporaryDirectory(
            prefix="runner-unit-", dir=BUILD_ROOT
        )
        self.output_root = Path(self.output_directory.name)
        self.fake_root = Path(self.fake_directory.name)
        self.executable = self.fake_root / "openfhe_encoder_layer_smoke"
        self.head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        self.preflight_patcher = mock.patch.object(
            runner,
            "_run_m4_preflight",
            return_value=(self._command_record("synthetic M4 build and narrow gates"),),
        )
        self.preflight_patcher.start()
        self.addCleanup(self.preflight_patcher.stop)
        self.executable_patcher = mock.patch.object(
            runner,
            "_resolve_m4_executable",
            side_effect=lambda _path, **_kwargs: self.executable,
        )
        self.executable_resolver = self.executable_patcher.start()
        self.addCleanup(self.executable_patcher.stop)
        self.trace_patcher = mock.patch.object(
            runner,
            "_trace_input_records",
            side_effect=lambda _data_root, _layer_id: _contract_trace_records(),
        )
        self.trace_patcher.start()
        self.addCleanup(self.trace_patcher.stop)
        self.input_recheck_patcher = mock.patch.object(
            runner,
            "_require_input_hashes",
            return_value=None,
        )
        self.input_recheck_patcher.start()
        self.addCleanup(self.input_recheck_patcher.stop)

    def tearDown(self) -> None:
        self.fake_directory.cleanup()
        self.output_directory.cleanup()

    @staticmethod
    def _command_record(command: str) -> dict[str, object]:
        return {
            "command": command,
            "cwd": str(REPO_ROOT),
            "exit_code": 0,
            "phase": "artifact_generation",
        }

    def _git_state(self) -> object:
        return runner.GitState(
            head=self.head,
            remote_head=self.head,
            commands=(self._command_record("synthetic clean pushed SHA preflight"),),
        )

    def _write_fake(self, mode: str = "success") -> None:
        record = repr(_runtime_record())
        source = f"""#!/usr/bin/env python3
import json
import resource
import sys
from pathlib import Path

counter_path = Path(__file__).with_suffix('.count')
count = int(counter_path.read_text()) + 1 if counter_path.exists() else 1
counter_path.write_text(str(count))
record = {record}
record['relative_l2'] += count * 0.00001
record['cosine'] -= count * 0.000001
record['inactive_max_abs'] += count * 0.00000001
diagnostic = {{
    'test': 'openfhe_encoder_layer_smoke',
    'profile': record['profile'],
    'security_claim': record['security_claim'],
    'parameter_sha256': record['parameter_sha256'],
    'layer': record['layer_id'],
    'input_level': record['input_level'],
    'tokens': 5,
    'hidden_size': 768,
    'intermediate_size': 3072,
    'feature_block': 1024,
    'fixture_load_ms': 10.0 + count,
    'setup_keygen_ms': 100.0 + count,
    'client_encrypt_ms': 20.0 + count,
    'server_online_ms': 500.0 + count,
    'client_decrypt_validate_ms': 30.0 + count,
    'relative_l2': record['relative_l2'],
    'cosine': record['cosine'],
    'max_absolute': 0.01,
    'exact_trace_relative_l2': 0.006,
    'exact_trace_cosine': 0.9998,
    'exact_trace_max_absolute': 0.02,
    'inactive_max_absolute': record['inactive_max_abs'],
    'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    'rotations': record['operation_counts']['rotations'],
    'ct_pt_multiplications': record['operation_counts']['ct_pt_multiplications'],
    'ct_ct_multiplications': record['operation_counts']['ct_ct_multiplications'],
    'explicit_rescale_requests': record['operation_counts']['explicit_rescale_requests'],
    'chebyshev_evaluations': record['operation_counts']['chebyshev_evaluations'],
    'estimated_polynomial_multiplications': record['operation_counts']['estimated_polynomial_multiplications'],
    'bootstraps': record['operation_counts']['bootstraps'],
    'bootstrap_iterations': record['operation_counts']['bootstrap_iterations'],
    'final_level': record['output_level'],
    'final_remaining_levels': record['remaining_levels'],
    'multiplicative_depth': record['multiplicative_depth'],
    'max_observed_level': record['max_observed_level'],
    'max_polynomial_depth': record['max_polynomial_depth'],
}}
if {mode!r} == 'diagnostic_mismatch':
    diagnostic['relative_l2'] += 0.0001
if {mode!r} == 'nonfinite_diagnostic':
    diagnostic['setup_keygen_ms'] = float('nan')
if {mode!r} == 'failure':
    print('synthetic child failure', file=sys.stderr)
    raise SystemExit(7)
if {mode!r} != 'missing_diagnostic':
    print(json.dumps(diagnostic, sort_keys=True, separators=(',', ':')))
    if {mode!r} == 'duplicate_diagnostic':
        print(json.dumps(diagnostic, sort_keys=True, separators=(',', ':')))
if {mode!r} == 'missing':
    print(json.dumps({{'test': 'wrong_record'}}, sort_keys=True))
else:
    print(json.dumps(record, sort_keys=True, separators=(',', ':')))
    if {mode!r} == 'duplicate':
        print(json.dumps(record, sort_keys=True, separators=(',', ':')))
"""
        self.executable.write_text(source, encoding="utf-8")
        self.executable.chmod(0o755)

    def _config(self, run_id: str) -> object:
        return runner.RunnerConfig(
            executable=self.executable,
            data_root=REPO_ROOT / "data",
            output_root=self.output_root,
            run_id=run_id,
        )

    def _assert_semantically_valid_without_live_git(self, manifest_path: Path) -> None:
        schema = validator.validate_schema(
            validator.load_json(REPO_ROOT / "docs" / "openfhe-artifact-schema.json")
        )
        manifest = validator.load_json(manifest_path)
        manifest["verdict"] = "SOURCE_AUDIT_ONLY"
        manifest["gate"]["passed"] = False
        fixed_executable = runner.DEFAULT_EXECUTABLE
        manifest["workload"]["executable_path"] = fixed_executable.relative_to(
            REPO_ROOT
        ).as_posix()
        manifest["workload"]["executable_sha256"] = hashlib.sha256(
            fixed_executable.read_bytes()
        ).hexdigest()
        manifest["workload"]["executable_bytes"] = fixed_executable.stat().st_size
        validator.validate_manifest(
            manifest_path,
            manifest,
            schema,
            verify_git=False,
        )

    def test_success_runs_one_warmup_and_five_measured_and_seals_artifact(self) -> None:
        self._write_fake()
        with (
            mock.patch.object(runner, "_preflight_git", return_value=self._git_state()),
            mock.patch.object(runner, "_validate_artifact") as validate_artifact,
        ):
            run_root = runner.generate_m4_artifact(self._config("unit-m4-success"))

        self.assertEqual(
            (self.executable.with_suffix(".count")).read_text(encoding="utf-8"),
            "6",
        )
        self.assertGreaterEqual(self.executable_resolver.call_count, 2)
        validate_artifact.assert_called_once_with(run_root / "manifest.json")
        self.assertEqual(
            {path.name for path in run_root.iterdir()},
            {"manifest.json", "stdout.log", "metrics.csv", "SHA256SUMS"},
        )
        stdout = (run_root / "stdout.log").read_text(encoding="utf-8")
        self.assertEqual(stdout.count('"test":"openfhe_encoder_layer"'), 5)
        self.assertEqual(stdout.count('"test":"openfhe_encoder_layer_smoke"'), 5)
        self.assertNotIn('"relative_l2":0.00401', stdout)

        with (run_root / "metrics.csv").open(
            "r", encoding="utf-8", newline=""
        ) as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 5)
        self.assertEqual([int(row["run"]) for row in rows], [1, 2, 3, 4, 5])
        self.assertTrue(all(int(row["exit_code"]) == 0 for row in rows))
        self.assertEqual(
            set(rows[0]),
            set(runner.CSV_FIELDS),
        )
        for row in rows:
            self.assertAlmostEqual(
                float(row["batch_total_ms"]),
                float(row["elapsed_seconds"]) * 1000.0,
            )
            self.assertAlmostEqual(
                float(row["batch_amortized_ms_per_token"]),
                float(row["batch_total_ms"]) / 5.0,
            )
            self.assertAlmostEqual(
                float(row["server_amortized_ms_per_token"]),
                float(row["server_online_ms"]) / 5.0,
            )
            self.assertEqual(
                row["checkpoint_metadata_sha256"],
                runner.EXPECTED_CHECKPOINT_METADATA_SHA256,
            )
            self.assertEqual(
                int(row["multiplicative_depth"]),
                runner.EXPECTED_MULTIPLICATIVE_DEPTH,
            )
            self.assertEqual(
                int(row["max_observed_level"]),
                runner.EXPECTED_MAX_OBSERVED_LEVEL,
            )
            self.assertEqual(
                int(row["max_polynomial_depth"]),
                runner.EXPECTED_MAX_POLYNOMIAL_DEPTH,
            )

        manifest = json.loads((run_root / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["milestone"], "M4")
        self.assertEqual(manifest["workload"]["repeat_count"], 5)
        self.assertEqual(manifest["metrics"]["warmup_count"], 1)
        self.assertEqual(manifest["metrics"]["tokens_per_batch"], 5)
        self.assertEqual(
            manifest["metrics"]["operation_counts"]["explicit_rescale_requests"],
            800,
        )
        self.assertEqual(
            manifest["metrics"]["checkpoints"][0]["noise_scale_degree_min"],
            2,
        )
        self.assertEqual(
            manifest["metrics"]["checkpoints"][0]["remaining_levels_min"],
            6,
        )
        self.assertEqual(
            manifest["metrics"]["checkpoint_metadata_sha256"],
            runner.EXPECTED_CHECKPOINT_METADATA_SHA256,
        )
        self.assertEqual(
            manifest["metrics"]["multiplicative_depth"],
            runner.EXPECTED_MULTIPLICATIVE_DEPTH,
        )
        self.assertEqual(
            manifest["metrics"]["max_observed_level"],
            runner.EXPECTED_MAX_OBSERVED_LEVEL,
        )
        self.assertEqual(
            manifest["metrics"]["max_polynomial_depth"],
            runner.EXPECTED_MAX_POLYNOMIAL_DEPTH,
        )
        self.assertEqual(
            manifest["contracts"]["execution"]["multiplicative_depth"],
            runner.EXPECTED_MULTIPLICATIVE_DEPTH,
        )
        self.assertEqual(
            manifest["contracts"]["execution"]["max_observed_level"],
            runner.EXPECTED_MAX_OBSERVED_LEVEL,
        )
        self.assertEqual(
            manifest["contracts"]["execution"]["max_polynomial_depth"],
            runner.EXPECTED_MAX_POLYNOMIAL_DEPTH,
        )
        self.assertEqual(
            set(manifest["metrics"]["phase_latency_ms"]),
            set(runner.PHASE_LATENCY_FIELDS),
        )
        self.assertEqual(manifest["git"]["local_commit"], self.head)
        self.assertEqual(manifest["git"]["remote_commit"], self.head)
        self.assertTrue(manifest["git"]["clean"])
        self.assertEqual(manifest["verdict"], "GO")

        checksum_lines = (run_root / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
        expected_checksums = {
            path: hashlib.sha256((run_root / path).read_bytes()).hexdigest()
            for path in ("stdout.log", "metrics.csv")
        }
        self.assertEqual(
            checksum_lines,
            [
                f"{expected_checksums['stdout.log']}  stdout.log",
                f"{expected_checksums['metrics.csv']}  metrics.csv",
            ],
        )
        self._assert_semantically_valid_without_live_git(run_root / "manifest.json")

    def test_target_checkpoint_metadata_is_strict_and_positive(self) -> None:
        baseline = _runtime_record()
        cases = (
            ("noise_scale_degree", 0, "noise_scale_degree must be positive"),
            ("noise_scale_degree", 1.0, "noise_scale_degree must be positive"),
            ("remaining_levels", 0, "remaining_levels must be positive"),
            ("remaining_levels", float("nan"), "remaining_levels must be positive"),
            ("scale_bits", 0.0, "scale_bits violates"),
            ("scale_bits", float("nan"), "scale_bits violates"),
            ("scale_bits", 101.0, "scale_bits violates"),
        )
        for field, replacement, message in cases:
            with self.subTest(field=field, replacement=replacement):
                record = copy.deepcopy(baseline)
                record["checkpoints"][0][field] = replacement
                with self.assertRaisesRegex(runner.ArtifactRunnerError, message):
                    runner._precheck_target_record(record, "synthetic target")

    def test_target_operation_schedule_is_exact(self) -> None:
        for key, expected in runner.EXPECTED_OPERATION_COUNTS.items():
            with self.subTest(key=key):
                record = _runtime_record()
                record["operation_counts"][key] = expected - 1
                with self.assertRaisesRegex(
                    runner.ArtifactRunnerError,
                    "encoder operation schedule drifted",
                ):
                    runner._precheck_target_record(record, "synthetic target")

    def test_target_depth_schedule_is_exact(self) -> None:
        expected = {
            "multiplicative_depth": runner.EXPECTED_MULTIPLICATIVE_DEPTH,
            "max_observed_level": runner.EXPECTED_MAX_OBSERVED_LEVEL,
            "max_polynomial_depth": runner.EXPECTED_MAX_POLYNOMIAL_DEPTH,
        }
        for key, value in expected.items():
            with self.subTest(key=key):
                record = _runtime_record()
                record[key] = value - 1
                with self.assertRaisesRegex(
                    runner.ArtifactRunnerError,
                    rf"{key} mismatch",
                ):
                    runner._precheck_target_record(record, "synthetic target")

    def test_diagnostic_depth_schedule_is_exact(self) -> None:
        target = _runtime_record()
        expected = {
            "multiplicative_depth": runner.EXPECTED_MULTIPLICATIVE_DEPTH,
            "max_observed_level": runner.EXPECTED_MAX_OBSERVED_LEVEL,
            "max_polynomial_depth": runner.EXPECTED_MAX_POLYNOMIAL_DEPTH,
        }
        for key, value in expected.items():
            with self.subTest(key=key):
                diagnostic = _diagnostic_record(target)
                diagnostic[key] = value - 1
                with self.assertRaisesRegex(
                    runner.ArtifactRunnerError,
                    rf"diagnostic.{key} mismatch",
                ):
                    runner._precheck_diagnostic_record(
                        diagnostic, target, "synthetic diagnostic"
                    )

    def test_every_checkpoint_metadata_field_is_exact(self) -> None:
        replacements = {
            "name": "wrong_checkpoint",
            "level": 41,
            "noise_scale_degree": 1,
            "remaining_levels": 4,
            "scale_bits": 99.0,
            "ciphertext_count": 4,
            "decryption_owner": "server",
        }
        for key, value in replacements.items():
            with self.subTest(key=key):
                record = _runtime_record()
                record["checkpoints"][0][key] = value
                with self.assertRaises(runner.ArtifactRunnerError):
                    runner._precheck_target_record(record, "synthetic target")

    def test_checkpoint_metadata_hash_is_canonical_and_frozen(self) -> None:
        record = _runtime_record()
        self.assertEqual(
            runner._checkpoint_metadata_sha256(record["checkpoints"]),
            runner.EXPECTED_CHECKPOINT_METADATA_SHA256,
        )
        record["checkpoints"][0]["level"] -= 1
        self.assertNotEqual(
            runner._checkpoint_metadata_sha256(record["checkpoints"]),
            runner.EXPECTED_CHECKPOINT_METADATA_SHA256,
        )

    def test_nonzero_child_is_fail_closed_and_removes_new_run_directory(self) -> None:
        self._write_fake("failure")
        config = self._config("unit-m4-child-failure")
        with (
            mock.patch.object(runner, "_preflight_git", return_value=self._git_state()),
            mock.patch.object(runner, "_validate_artifact") as validate_artifact,
            self.assertRaisesRegex(runner.ArtifactRunnerError, "exited with 7"),
        ):
            runner.generate_m4_artifact(config)
        self.assertFalse((self.output_root / config.run_id).exists())
        validate_artifact.assert_not_called()

    def test_missing_or_duplicate_target_record_is_fail_closed(self) -> None:
        for mode in ("missing", "duplicate"):
            with self.subTest(mode=mode):
                self._write_fake(mode)
                config = self._config(f"unit-m4-{mode}")
                with (
                    mock.patch.object(
                        runner, "_preflight_git", return_value=self._git_state()
                    ),
                    mock.patch.object(runner, "_validate_artifact") as validate_artifact,
                    self.assertRaisesRegex(
                        runner.ArtifactRunnerError,
                        "must emit exactly one openfhe_encoder_layer record",
                    ),
                ):
                    runner.generate_m4_artifact(config)
                self.assertFalse((self.output_root / config.run_id).exists())
                validate_artifact.assert_not_called()
                self.executable.with_suffix(".count").unlink(missing_ok=True)

    def test_missing_duplicate_nonfinite_or_mismatched_diagnostic_is_fail_closed(
        self,
    ) -> None:
        cases = {
            "missing_diagnostic": "exactly one openfhe_encoder_layer_smoke record",
            "duplicate_diagnostic": "exactly one openfhe_encoder_layer_smoke record",
            "nonfinite_diagnostic": "exactly one openfhe_encoder_layer_smoke record",
            "diagnostic_mismatch": "differs from target.relative_l2",
        }
        for mode, message in cases.items():
            with self.subTest(mode=mode):
                self._write_fake(mode)
                config = self._config(f"unit-m4-{mode}")
                with (
                    mock.patch.object(
                        runner, "_preflight_git", return_value=self._git_state()
                    ),
                    mock.patch.object(runner, "_validate_artifact") as validate_artifact,
                    self.assertRaisesRegex(runner.ArtifactRunnerError, message),
                ):
                    runner.generate_m4_artifact(config)
                self.assertFalse((self.output_root / config.run_id).exists())
                validate_artifact.assert_not_called()
                self.executable.with_suffix(".count").unlink(missing_ok=True)

    def test_validator_failure_is_fail_closed_and_removes_new_run_directory(self) -> None:
        self._write_fake()
        config = self._config("unit-m4-validator-failure")
        with (
            mock.patch.object(runner, "_preflight_git", return_value=self._git_state()),
            mock.patch.object(
                runner,
                "_validate_artifact",
                side_effect=runner.ArtifactRunnerError("synthetic validator failure"),
            ),
            self.assertRaisesRegex(runner.ArtifactRunnerError, "validator failure"),
        ):
            runner.generate_m4_artifact(config)
        self.assertFalse((self.output_root / config.run_id).exists())

    def test_existing_run_directory_is_never_overwritten(self) -> None:
        self._write_fake()
        config = self._config("unit-m4-existing")
        run_root = self.output_root / config.run_id
        run_root.mkdir()
        sentinel = run_root / "sentinel"
        sentinel.write_text("preserve", encoding="utf-8")
        with (
            mock.patch.object(runner, "_preflight_git", return_value=self._git_state()),
            mock.patch.object(runner, "_validate_artifact") as validate_artifact,
            self.assertRaisesRegex(runner.ArtifactRunnerError, "already exists"),
        ):
            runner.generate_m4_artifact(config)
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "preserve")
        validate_artifact.assert_not_called()

    def test_git_preflight_rejects_dirty_tree_and_remote_mismatch(self) -> None:
        clean_record = self._command_record("synthetic git command")
        with (
            mock.patch.object(
                runner,
                "_run_git",
                return_value=(" M tracked.cpp", clean_record),
            ),
            self.assertRaisesRegex(runner.ArtifactRunnerError, "not clean"),
        ):
            runner._preflight_git()

        remote_sha = "f" * 40 if self.head != "f" * 40 else "e" * 40
        side_effect = [
            ("", clean_record),
            (runner.BRANCH, clean_record),
            (self.head, clean_record),
            (f"{remote_sha}\t{runner.REMOTE_REF}", clean_record),
        ]
        with (
            mock.patch.object(runner, "_run_git", side_effect=side_effect),
            self.assertRaisesRegex(runner.ArtifactRunnerError, "differs from"),
        ):
            runner._preflight_git()

    def test_build_configuration_requires_testing_enabled(self) -> None:
        build_root = self.fake_root / "build-contract"
        build_root.mkdir()
        cache_path = build_root / "CMakeCache.txt"
        required_lines = [
            f"CMAKE_HOME_DIRECTORY:INTERNAL={REPO_ROOT.resolve()}",
            "CMAKE_BUILD_TYPE:STRING=Release",
            (
                "OpenFHE_DIR:PATH="
                f"{runner.DEFAULT_OPENFHE_PREFIX.resolve() / 'lib' / 'OpenFHE'}"
            ),
        ]
        cache_path.write_text("\n".join(required_lines) + "\n", encoding="utf-8")
        with (
            mock.patch.object(
                runner, "DEFAULT_EXECUTABLE", build_root / "encoder_smoke"
            ),
            mock.patch.object(runner, "_read_openfhe_version", return_value="1.5.1"),
            self.assertRaisesRegex(runner.ArtifactRunnerError, "BUILD_TESTING:BOOL=ON"),
        ):
            ORIGINAL_VERIFY_BUILD_CONFIGURATION(
                build_root, runner.DEFAULT_OPENFHE_PREFIX
            )

        cache_path.write_text(
            "\n".join([*required_lines, "BUILD_TESTING:BOOL=ON"]) + "\n",
            encoding="utf-8",
        )
        with (
            mock.patch.object(
                runner, "DEFAULT_EXECUTABLE", build_root / "encoder_smoke"
            ),
            mock.patch.object(runner, "_read_openfhe_version", return_value="1.5.1"),
        ):
            ORIGINAL_VERIFY_BUILD_CONFIGURATION(
                build_root, runner.DEFAULT_OPENFHE_PREFIX
            )

    def test_ctest_preflight_uses_no_tests_error_and_zero_matches_fail_closed(
        self,
    ) -> None:
        commands: list[list[str]] = []

        def run_checked(command: list[str], _label: str) -> dict[str, object]:
            commands.append(command)
            if command[0] == "ctest":
                raise runner.ArtifactRunnerError("synthetic zero tests matched")
            return self._command_record("synthetic successful build")

        with (
            mock.patch.object(runner, "_verify_build_configuration"),
            mock.patch.object(runner, "_run_checked_command", side_effect=run_checked),
            mock.patch.object(
                runner, "_resolve_m4_executable", return_value=self.executable
            ),
            mock.patch.object(runner, "_verify_openfhe_linkage"),
            self.assertRaisesRegex(runner.ArtifactRunnerError, "zero tests matched"),
        ):
            ORIGINAL_RUN_M4_PREFLIGHT(
                runner.DEFAULT_EXECUTABLE.parent,
                runner.DEFAULT_EXECUTABLE,
                runner.DEFAULT_OPENFHE_PREFIX,
            )
        ctest_command = next(command for command in commands if command[0] == "ctest")
        self.assertIn("--no-tests=error", ctest_command)

    def test_fixed_executable_path_rejects_another_same_named_binary(self) -> None:
        self._write_fake()
        with self.assertRaisesRegex(runner.ArtifactRunnerError, "fixed build target"):
            ORIGINAL_RESOLVE_M4_EXECUTABLE(self.executable)

    def test_fixed_executable_rejects_symlink_to_stale_binary(self) -> None:
        stale = self.fake_root / "stale_encoder"
        stale.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        stale.chmod(0o755)
        self.executable.symlink_to(stale.name)
        with (
            mock.patch.object(runner, "DEFAULT_EXECUTABLE", self.executable),
            self.assertRaisesRegex(runner.ArtifactRunnerError, "must not contain symlinks"),
        ):
            ORIGINAL_RESOLVE_M4_EXECUTABLE(self.executable)

    def test_frozen_input_hash_is_rechecked(self) -> None:
        frozen = self.fake_root / "frozen-input.json"
        frozen.write_text("before", encoding="utf-8")
        record = {
            "path": frozen.relative_to(REPO_ROOT).as_posix(),
            "sha256": hashlib.sha256(frozen.read_bytes()).hexdigest(),
            "bytes": frozen.stat().st_size,
        }
        frozen.write_text("after!", encoding="utf-8")
        with self.assertRaisesRegex(runner.ArtifactRunnerError, "frozen input changed"):
            ORIGINAL_REQUIRE_INPUT_HASHES([record], "after measured run 1")

    def test_trace_inputs_bind_all_layer_one_hashes(self) -> None:
        self._write_fake()
        expected_hash = "a" * 64
        required_files = {
            f"file_{index}": {
                "path": (
                    f"Attention/parms/file_{index}.csv"
                    if index % 2 == 0
                    else f"Attention/allresults/file_{index}.csv"
                )
            }
            for index in range(37)
        }
        contract = {
            "required_files": required_files,
            "layers": [
                {
                    "layer_id": 1,
                    "sha256": {name: expected_hash for name in required_files},
                }
            ],
        }
        with (
            mock.patch.object(runner, "_load_json", return_value=contract),
            mock.patch.object(
                runner,
                "_resolve_repository_file",
                return_value=self.executable,
            ),
            mock.patch.object(runner, "_sha256", return_value=expected_hash),
        ):
            records = ORIGINAL_TRACE_INPUT_RECORDS(runner.DEFAULT_DATA_ROOT, 1)
        self.assertEqual(len(records), 37)
        self.assertEqual({record["media_type"] for record in records}, {"text/csv"})
        self.assertEqual(
            {record["role"] for record in records},
            {"trace", "weights"},
        )

        with (
            mock.patch.object(runner, "_load_json", return_value=contract),
            mock.patch.object(
                runner,
                "_resolve_repository_file",
                return_value=self.executable,
            ),
            mock.patch.object(runner, "_sha256", return_value="b" * 64),
            self.assertRaisesRegex(runner.ArtifactRunnerError, "trace hash mismatch"),
        ):
            ORIGINAL_TRACE_INPUT_RECORDS(runner.DEFAULT_DATA_ROOT, 1)

if __name__ == "__main__":
    unittest.main()
