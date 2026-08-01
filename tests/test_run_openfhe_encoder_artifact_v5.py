#!/usr/bin/env python3
"""Contract tests for the independent M4 schema-v5 artifact runner."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = REPO_ROOT / "scripts" / "run_openfhe_encoder_artifact_v5.py"
spec = importlib.util.spec_from_file_location(
    "run_openfhe_encoder_artifact_v5", RUNNER_PATH
)
if spec is None or spec.loader is None:  # pragma: no cover
    raise RuntimeError(f"cannot import {RUNNER_PATH}")
runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)


def sentinel_ranges() -> dict[str, dict[str, float]]:
    return {
        "softmax_denominator": {"minimum": 0.75, "maximum": 1.25},
        "attention_layernorm_normalized_variance": {
            "minimum": 45.0,
            "maximum": 90.0,
        },
        "output_layernorm_normalized_variance": {
            "minimum": 32.0,
            "maximum": 128.0,
        },
    }


def runtime_record() -> dict[str, object]:
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
        "multiplicative_depth": 47,
        "max_observed_level": 45,
        "max_polynomial_depth": 10,
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
        "layernorm_inactive_guard_max_error": 0.25,
        "inactive_polynomial_sentinel_ranges": sentinel_ranges(),
        "checkpoints": copy.deepcopy(list(runner.EXPECTED_CHECKPOINTS)),
        "operation_counts": copy.deepcopy(runner.EXPECTED_OPERATION_COUNTS),
    }


def diagnostic_record(target: dict[str, object]) -> dict[str, object]:
    counts = target["operation_counts"]
    assert isinstance(counts, dict)
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
        "peak_rss_bytes": 4096,
        **counts,
        "final_level": target["output_level"],
        "final_remaining_levels": target["remaining_levels"],
        "multiplicative_depth": target["multiplicative_depth"],
        "max_observed_level": target["max_observed_level"],
        "max_polynomial_depth": target["max_polynomial_depth"],
    }


def runtime_stdout() -> bytes:
    target = runtime_record()
    lines = (
        runner.PROFILE_WARNING,
        json.dumps(
            diagnostic_record(target),
            sort_keys=True,
            separators=(",", ":"),
        ),
        json.dumps(target, sort_keys=True, separators=(",", ":")),
    )
    return ("\n".join(lines) + "\n").encode("utf-8")


def sample(index: int = 0) -> runner.RunSample:
    target = runtime_record()
    target["relative_l2"] = 0.004 + index * 0.0001
    diagnostic = diagnostic_record(target)
    return runner.RunSample(
        stdout=b"",
        stderr=b"",
        record=target,
        diagnostic=diagnostic,
        elapsed_seconds=1.0 + index,
        peak_rss_kib=1024 + index,
        command={},
    )


def ctest_summary(names: tuple[str, ...]) -> str:
    total = len(runner.M4_CTEST_EXPECTED_TESTS)
    summaries = "".join(
        f"{index}/{total} Test #{index}: {name} .......   Passed    0.01 sec\n"
        for index, name in enumerate(names, start=1)
    )
    return f"{summaries}100% tests passed, 0 tests failed out of {total}\n"


class EncoderArtifactV5RunnerTest(unittest.TestCase):
    def test_claim_boundary_and_checksum_bundle_are_frozen(self) -> None:
        expected_claim_boundary = (
            "M4 covers only single-layer layer-1 correctness on the fixed five-token "
            "BERT-base encoder trace.",
            "The server-only claim is an API/target trust boundary: server code "
            "receives only ciphertexts, public model weights, and evaluation keys; "
            "it is not a process-isolation claim.",
            "paper_compat OpenFHE CKKS CPU parameters are research-reproduction "
            "parameters with security_claim=none.",
            "This artifact makes no 128-bit-security or production-security claim.",
            "Timing is diagnostic only; this artifact makes no benchmark or speedup "
            "claim.",
            "GPU, Discrete CKKS/FBT, QDQ, tokenizer, classifier, and task-level "
            "inference are excluded.",
        )
        self.assertEqual(runner.CLAIM_BOUNDARY, expected_claim_boundary)
        self.assertEqual(
            runner.CHECKSUM_EVIDENCE_PATHS,
            (
                "stdout.log",
                "stderr.log",
                "time.log",
                "metrics.csv",
                "manifest.json",
            ),
        )

        with tempfile.TemporaryDirectory(prefix="m4-v5-checksum-") as directory:
            root = Path(directory)
            for relative_path, content in (
                ("stdout.log", "stdout\n"),
                ("stderr.log", ""),
                ("time.log", "time\n"),
                ("metrics.csv", "metric,value\n"),
                ("manifest.json", '{"schema_version": 5}\n'),
            ):
                (root / relative_path).write_text(content, encoding="utf-8")
            checksum_path = root / "SHA256SUMS"
            runner._write_checksum_evidence(checksum_path, root)
            expected_lines = [
                f"{hashlib.sha256((root / path).read_bytes()).hexdigest()}  {path}"
                for path in runner.CHECKSUM_EVIDENCE_PATHS
            ]
            self.assertEqual(
                checksum_path.read_text(encoding="utf-8").splitlines(),
                expected_lines,
            )

    def test_success_stderr_is_preserved_byte_for_byte_without_truncation(self) -> None:
        first = sample()
        second = sample(1)
        first = dataclasses.replace(first, stderr=b"first warning\n")
        second = dataclasses.replace(second, stderr=b"\x00second diagnostic\n")
        with tempfile.TemporaryDirectory(prefix="m4-v5-stderr-") as directory:
            path = Path(directory) / "stderr.log"
            runner._write_stderr(path, [first, second])
            self.assertEqual(
                path.read_bytes(),
                b"first warning\n\x00second diagnostic\n",
            )

            oversized = dataclasses.replace(
                first,
                stderr=b"x" * (runner.RAW_STDERR_MAX_BYTES + 1),
            )
            with self.assertRaisesRegex(
                runner.ArtifactRunnerError,
                "exceeds the frozen bound",
            ):
                runner._write_stderr(path, [oversized])

    def test_narrow_ctest_pattern_explicitly_contains_all_v5_contracts(self) -> None:
        self.assertEqual(
            runner.M4_V5_REQUIRED_CTEST_CONTRACTS,
            (
                "openfhe_m4_v5_artifact_schema_contract",
                "openfhe_m4_v5_artifact_validator_contract",
                "openfhe_m4_v5_encoder_artifact_runner_contract",
                "openfhe_feature_layernorm_smoke",
            ),
        )
        for name in runner.M4_V5_REQUIRED_CTEST_CONTRACTS:
            with self.subTest(name=name):
                self.assertIsNotNone(re.fullmatch(runner.M4_CTEST_PATTERN, name))
                self.assertIsNone(
                    re.fullmatch(runner.M4_CTEST_PATTERN, f"{name}_not_frozen")
                )

    def test_preflight_rejects_historical_only_ctest_pattern_before_build(self) -> None:
        historical_only = runner.M4_CTEST_PATTERN
        for fragment in (
            "m4_v5_artifact_schema_contract|",
            "m4_v5_artifact_validator_contract|",
            "m4_v5_encoder_artifact_runner_contract|",
            "feature_layernorm_smoke|",
        ):
            historical_only = historical_only.replace(fragment, "")
        with (
            mock.patch.object(runner, "M4_CTEST_PATTERN", historical_only),
            mock.patch.object(runner, "_verify_build_configuration") as verify_build,
        ):
            with self.assertRaisesRegex(
                runner.ArtifactRunnerError, "omits required contracts"
            ):
                runner._run_m4_preflight(
                    Path("/tmp/build-openfhe"),
                    Path("/tmp/openfhe_encoder_layer_smoke"),
                    Path("/tmp/openfhe"),
                )
            verify_build.assert_not_called()

    def test_configure_command_is_fresh_fixed_and_environment_scrubbed(self) -> None:
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
        self.assertIn("-DCMAKE_CXX_FLAGS:STRING=", command)
        self.assertIn("-DCMAKE_EXE_LINKER_FLAGS:STRING=", command)

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

    def test_git_uses_fixed_binary_and_scrubs_all_git_environment(self) -> None:
        poisoned = {
            "PATH": "/tmp/fake-bin:/usr/bin",
            "GIT_DIR": "/tmp/fake.git",
            "GIT_WORK_TREE": "/tmp/fake-worktree",
            "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": "remote.origin.url",
            "GIT_CONFIG_VALUE_0": "file:///tmp/fake.git",
        }
        with mock.patch.dict(os.environ, poisoned, clear=False):
            environment = runner._git_environment()
        self.assertEqual(environment["PATH"], "/usr/bin:/bin")
        self.assertFalse(any(name.startswith("GIT_") for name in environment))

        completed = subprocess.CompletedProcess(
            [str(runner.SYSTEM_GIT), "rev-parse", "HEAD"],
            0,
            stdout="a" * 40 + "\n",
            stderr="",
        )
        with mock.patch.object(
            runner.subprocess,
            "run",
            return_value=completed,
        ) as run:
            output, record = runner._run_git(["rev-parse", "HEAD"])
        self.assertEqual(output, "a" * 40)
        self.assertEqual(run.call_args.args[0][0], str(runner.SYSTEM_GIT))
        self.assertFalse(
            any(
                name.startswith("GIT_")
                for name in run.call_args.kwargs["env"]
            )
        )
        self.assertTrue(record["command"].startswith("/usr/bin/git "))

    def test_git_preflight_freezes_origin_url_and_remote_ref(self) -> None:
        head = "a" * 40
        valid = [
            ("", {}),
            (runner.BRANCH, {}),
            (head, {}),
            (runner.REMOTE_URL, {}),
            (f"{head}\t{runner.REMOTE_REF}", {}),
        ]
        with mock.patch.object(
            runner,
            "_run_git",
            side_effect=valid,
        ) as run_git:
            state = runner._preflight_git()
        self.assertEqual(state.remote_url, runner.REMOTE_URL)
        self.assertEqual(
            run_git.call_args_list[-1].args[0],
            [
                "ls-remote",
                "--exit-code",
                runner.REMOTE_URL,
                runner.REMOTE_REF,
            ],
        )

        poisoned = [*valid]
        poisoned[3] = ("file:///tmp/fake.git", {})
        with (
            mock.patch.object(runner, "_run_git", side_effect=poisoned),
            self.assertRaisesRegex(runner.ArtifactRunnerError, "URL must be"),
        ):
            runner._preflight_git()

    def test_runtime_stdout_is_strict_for_duplicate_nan_malformed_and_extra(self) -> None:
        target, diagnostic = runner._parse_stdout_run(runtime_stdout(), "run")
        self.assertEqual(target["test"], "openfhe_encoder_layer")
        self.assertEqual(diagnostic["test"], "openfhe_encoder_layer_smoke")

        lines = runtime_stdout().decode("utf-8").splitlines()
        adversarial = (
            '{"unknown":1,"unknown":2}',
            '{"unknown":NaN}',
            '{"unknown":',
            json.dumps(runtime_record(), separators=(",", ":")),
        )
        for injected in adversarial:
            with self.subTest(injected=injected):
                tampered = (
                    "\n".join([lines[0], injected, *lines[1:]]) + "\n"
                ).encode("utf-8")
                with self.assertRaises(runner.ArtifactRunnerError):
                    runner._parse_stdout_run(tampered, "tampered")

    def test_json_files_reject_duplicate_nan_and_malformed(self) -> None:
        with tempfile.TemporaryDirectory(prefix="m4-v5-json-") as directory:
            path = Path(directory) / "input.json"
            for content in (
                '{"key":1,"key":2}\n',
                '{"key":NaN}\n',
                '{"key":\n',
            ):
                with self.subTest(content=content):
                    path.write_text(content, encoding="utf-8")
                    with self.assertRaises(runner.ArtifactRunnerError):
                        runner._load_json(path)

    def test_gnu_time_record_is_raw_and_strict(self) -> None:
        config = runner.RunnerConfig(
            executable=runner.DEFAULT_EXECUTABLE,
            data_root=runner.DEFAULT_DATA_ROOT,
            output_root=runner.DEFAULT_OUTPUT_ROOT,
            run_id="m4-time-test",
        )
        argv = runner._runtime_argv(config)
        line = (
            f"{runner.TIME_RECORD_PREFIX}\twarmup\t1.25\t2048\t0\t"
            f"{' '.join(argv)}\n"
        ).encode("utf-8")
        record = runner._parse_time_record_line(
            line,
            "warmup",
            argv,
            "time",
        )
        self.assertEqual(record["elapsed_seconds"], 1.25)
        self.assertEqual(record["peak_rss_kib"], 2048)
        for replacement in ("NaN", "0", "-1", "01.25", "1.250"):
            with self.subTest(replacement=replacement):
                fields = line.decode("utf-8").split("\t")
                fields[2] = replacement
                with self.assertRaises(runner.ArtifactRunnerError):
                    runner._parse_time_record_line(
                        "\t".join(fields).encode("utf-8"),
                        "warmup",
                        argv,
                        "time",
                    )
        for rss in ("+2048", "02048", "0"):
            with self.subTest(rss=rss):
                fields = line.decode("utf-8").split("\t")
                fields[3] = rss
                with self.assertRaises(runner.ArtifactRunnerError):
                    runner._parse_time_record_line(
                        "\t".join(fields).encode("utf-8"),
                        "warmup",
                        argv,
                        "time",
                    )

    def test_existing_run_directory_fails_before_git_build_or_he(self) -> None:
        with tempfile.TemporaryDirectory(prefix="m4-v5-run-root-") as directory:
            output_root = Path(directory) / "results" / "openfhe"
            run_root = output_root / "existing"
            run_root.mkdir(parents=True)
            with (
                mock.patch.object(runner, "DEFAULT_OUTPUT_ROOT", output_root),
                mock.patch.object(
                    runner,
                    "_trace_scale_contract_provenance",
                    return_value={},
                ),
                mock.patch.object(runner, "_require_m4_schedule_sealed"),
                mock.patch.object(runner, "_preflight_git") as preflight_git,
                mock.patch.object(runner, "_run_m4_preflight") as build_preflight,
                mock.patch.object(runner, "_run_once") as run_once,
            ):
                with self.assertRaisesRegex(
                    runner.ArtifactRunnerError,
                    "run directory already exists",
                ):
                    runner.generate_m4_artifact(
                        runner.RunnerConfig(
                            executable=runner.DEFAULT_EXECUTABLE,
                            data_root=runner.DEFAULT_DATA_ROOT,
                            output_root=output_root,
                            run_id="existing",
                        )
                    )
            preflight_git.assert_not_called()
            build_preflight.assert_not_called()
            run_once.assert_not_called()

    def test_invalid_run_id_fails_before_git_build_or_he(self) -> None:
        with (
            mock.patch.object(runner, "_preflight_git") as preflight_git,
            mock.patch.object(runner, "_run_m4_preflight") as build_preflight,
            mock.patch.object(runner, "_run_once") as run_once,
        ):
            with self.assertRaisesRegex(runner.ArtifactRunnerError, "invalid run id"):
                runner.generate_m4_artifact(
                    runner.RunnerConfig(
                        executable=runner.DEFAULT_EXECUTABLE,
                        data_root=runner.DEFAULT_DATA_ROOT,
                        output_root=runner.DEFAULT_OUTPUT_ROOT,
                        run_id="../escape",
                    )
                )
        preflight_git.assert_not_called()
        build_preflight.assert_not_called()
        run_once.assert_not_called()

    def test_ctest_summary_requires_exact_frozen_passed_set(self) -> None:
        runner._require_ctest_passed_tests(
            ctest_summary(runner.M4_CTEST_EXPECTED_TESTS),
            runner.M4_CTEST_EXPECTED_TESTS,
            "M4 test",
        )
        runner._require_ctest_passed_tests(
            ctest_summary(tuple(reversed(runner.M4_CTEST_EXPECTED_TESTS))),
            runner.M4_CTEST_EXPECTED_TESTS,
            "M4 reordered test",
        )

        missing = tuple(
            name
            for name in runner.M4_CTEST_EXPECTED_TESTS
            if name != "openfhe_feature_layernorm_smoke"
        )
        with self.assertRaisesRegex(
            runner.ArtifactRunnerError,
            "did not enumerate and pass.*openfhe_feature_layernorm_smoke",
        ):
            runner._require_ctest_passed_tests(
                ctest_summary(missing),
                runner.M4_CTEST_EXPECTED_TESTS,
                "M4 test",
            )

        duplicate = (
            runner.M4_CTEST_EXPECTED_TESTS[0],
            *runner.M4_CTEST_EXPECTED_TESTS,
        )
        with self.assertRaisesRegex(runner.ArtifactRunnerError, "duplicate"):
            runner._require_ctest_passed_tests(
                ctest_summary(duplicate),
                runner.M4_CTEST_EXPECTED_TESTS,
                "M4 test",
            )

        skipped = ctest_summary(runner.M4_CTEST_EXPECTED_TESTS).replace(
            "Passed",
            "***Skipped",
            1,
        )
        with self.assertRaisesRegex(runner.ArtifactRunnerError, "statuses="):
            runner._require_ctest_passed_tests(
                skipped,
                runner.M4_CTEST_EXPECTED_TESTS,
                "M4 test",
            )

        without_footer = ctest_summary(runner.M4_CTEST_EXPECTED_TESTS).replace(
            "100% tests passed, 0 tests failed out of 19\n", ""
        )
        with self.assertRaisesRegex(runner.ArtifactRunnerError, "footer_count=0"):
            runner._require_ctest_passed_tests(
                without_footer,
                runner.M4_CTEST_EXPECTED_TESTS,
                "M4 test",
            )

    def test_frozen_schedule_counts_and_digest(self) -> None:
        self.assertEqual(runner.SCHEMA_VERSION, 5)
        self.assertEqual(runner.EXPECTED_OPERATION_COUNTS["rotations"], 6300)
        self.assertEqual(
            runner.EXPECTED_OPERATION_COUNTS["ct_pt_multiplications"], 51885
        )
        self.assertEqual(
            runner.EXPECTED_OPERATION_COUNTS["explicit_rescale_requests"], 810
        )
        self.assertEqual(runner.EXPECTED_MAX_OBSERVED_LEVEL, 45)
        self.assertEqual(runner.EXPECTED_CHECKPOINTS[-1]["level"], 29)
        self.assertEqual(runner.EXPECTED_CHECKPOINTS[-1]["remaining_levels"], 17)
        self.assertEqual(
            runner.EXPECTED_POST_BOOTSTRAP_MASK, "single_normal_scale_mask"
        )
        self.assertEqual(
            runner.EXPECTED_INTERNAL_CHECKPOINTS,
            {
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
            },
        )
        self.assertEqual(
            runner._checkpoint_metadata_sha256(list(runner.EXPECTED_CHECKPOINTS)),
            "c4c1c85e52154215784b9fa93a584d824dde94a644e5af997882486aac01a6db",
        )

    def test_target_and_diagnostic_records_pass(self) -> None:
        target = runtime_record()
        runner._precheck_target_record(target, "target")
        runner._precheck_diagnostic_record(
            diagnostic_record(target), target, "diagnostic"
        )

    def test_old_checkpoint_metadata_is_rejected(self) -> None:
        target = runtime_record()
        target["checkpoints"][-1]["level"] = 30
        target["checkpoints"][-1]["remaining_levels"] = 16
        with self.assertRaisesRegex(runner.ArtifactRunnerError, "metadata drifted"):
            runner._precheck_target_record(target, "target")

    def test_operation_count_drift_is_rejected(self) -> None:
        for key in runner.EXPECTED_OPERATION_COUNTS:
            with self.subTest(key=key):
                target = runtime_record()
                target["operation_counts"][key] += 1
                with self.assertRaisesRegex(
                    runner.ArtifactRunnerError, "operation schedule drifted"
                ):
                    runner._precheck_target_record(target, "target")

    def test_layernorm_guard_uses_registered_interval_not_identity(self) -> None:
        target = runtime_record()
        target["layernorm_inactive_guard_max_error"] = 100.0
        runner._precheck_target_record(target, "target")

        for site, field, value in (
            ("attention_layernorm_normalized_variance", "minimum", 0.499999),
            ("attention_layernorm_normalized_variance", "maximum", 1536.0001),
            ("output_layernorm_normalized_variance", "minimum", float("nan")),
            ("output_layernorm_normalized_variance", "maximum", float("inf")),
        ):
            with self.subTest(site=site, field=field):
                target = runtime_record()
                target["inactive_polynomial_sentinel_ranges"][site][field] = value
                with self.assertRaises(runner.ArtifactRunnerError):
                    runner._precheck_target_record(target, "target")
        for invalid_guard in (-1e-12, float("nan"), float("inf")):
            with self.subTest(invalid_guard=invalid_guard):
                target = runtime_record()
                target["layernorm_inactive_guard_max_error"] = invalid_guard
                with self.assertRaisesRegex(
                    runner.ArtifactRunnerError,
                    "layernorm_inactive_guard_max_error",
                ):
                    runner._precheck_target_record(target, "target")

        target = runtime_record()
        target["inactive_max_abs"] = 1.000001e-6
        with self.assertRaisesRegex(runner.ArtifactRunnerError, "inactive_max_abs"):
            runner._precheck_target_record(target, "target")

        target = runtime_record()
        target["inactive_polynomial_sentinel_ranges"][
            "attention_layernorm_normalized_variance"
        ] = {"minimum": 64.0, "maximum": 63.0}
        with self.assertRaisesRegex(runner.ArtifactRunnerError, "minimum exceeds"):
            runner._precheck_target_record(target, "target")

    def test_softmax_uses_reciprocal_interval_not_identity_interval(self) -> None:
        target = runtime_record()
        target["inactive_polynomial_sentinel_ranges"]["softmax_denominator"] = {
            "minimum": 0.01,
            "maximum": 80.0,
        }
        runner._precheck_target_record(target, "target")
        target["inactive_polynomial_sentinel_ranges"]["softmax_denominator"][
            "minimum"
        ] = 0.009999
        with self.assertRaises(runner.ArtifactRunnerError):
            runner._precheck_target_record(target, "target")

    def test_metrics_summary_keeps_guard_diagnostic_and_output_gate(self) -> None:
        metrics = runner._summarize_metrics([sample(index) for index in range(5)])
        self.assertEqual(metrics["output_level"], 29)
        self.assertEqual(metrics["remaining_levels"], 17)
        self.assertEqual(metrics["quality"]["layernorm_inactive_guard_max_error"], 0.25)
        self.assertLessEqual(metrics["quality"]["inactive_max_abs_max"], 1e-6)
        self.assertEqual(
            metrics["inactive_polynomial_sentinel_ranges"], sentinel_ranges()
        )

    def test_trace_scale_provenance_recomputes_source_hashes(self) -> None:
        provenance = runner._trace_scale_contract_provenance()
        self.assertEqual(
            provenance,
            {
                "source_path": runner.TRACE_SCALE_SOURCE_PATH,
                "json_locator": runner.TRACE_SCALE_JSON_LOCATOR,
                "contract_id": runner.TRACE_SCALE_CONTRACT_ID,
                "contract_sha256": runner.TRACE_SCALE_CONTRACT_SHA256,
                "values_sha256": runner.TRACE_SCALE_VALUES_SHA256,
                "raw_variance_sha256": runner.TRACE_SCALE_RAW_VARIANCE_SHA256,
            },
        )
        approximation = json.loads(
            (REPO_ROOT / runner.TRACE_SCALE_SOURCE_PATH).read_text(encoding="utf-8")
        )
        contract = approximation["operators"]["layernorm"][
            "feature_packed_trace_scale_contract"
        ]
        self.assertEqual(
            runner._binary64_tensor_sha256(
                contract["values"],
                runner.TRACE_SCALE_SHAPE,
                "values",
            ),
            runner.TRACE_SCALE_VALUES_SHA256,
        )

        profile = json.loads(
            (REPO_ROOT / runner.PROFILE_PATH).read_text(encoding="utf-8")
        )
        for field in (
            "contract_sha256",
            "values_sha256",
            "raw_variance_sha256",
        ):
            with self.subTest(field=field):
                tampered_approximation = copy.deepcopy(approximation)
                tampered_profile = copy.deepcopy(profile)
                tampered_approximation["operators"]["layernorm"][
                    "feature_packed_trace_scale_contract"
                ][field] = "0" * 64
                tampered_profile["feature_packed_layernorm_override"][
                    "trace_scale_contract"
                ][field] = "0" * 64
                with mock.patch.object(
                    runner,
                    "_load_json",
                    side_effect=[tampered_approximation, tampered_profile],
                ):
                    with self.assertRaisesRegex(
                        runner.ArtifactRunnerError,
                        "trace-scale",
                    ):
                        runner._trace_scale_contract_provenance()

    def test_unsealed_schedule_stops_before_git_build_or_he(self) -> None:
        with (
            mock.patch.object(runner, "_preflight_git") as preflight_git,
            mock.patch.object(runner, "_run_m4_preflight") as build_preflight,
            mock.patch.object(runner, "_run_once") as run_once,
        ):
            with self.assertRaisesRegex(
                runner.ArtifactRunnerError,
                "metadata schedule is not sealed",
            ):
                runner.generate_m4_artifact(
                    runner.RunnerConfig(
                        executable=runner.DEFAULT_EXECUTABLE,
                        data_root=runner.DEFAULT_DATA_ROOT,
                        output_root=runner.DEFAULT_OUTPUT_ROOT,
                        run_id="m4-unsealed-fail-closed",
                    )
                )
        preflight_git.assert_not_called()
        build_preflight.assert_not_called()
        run_once.assert_not_called()

    def test_validator_command_explicitly_binds_versioned_schema(self) -> None:
        command = runner._validator_command(Path("/tmp/manifest.json"))
        self.assertEqual(
            command,
            [
                sys.executable,
                str(runner.VALIDATOR_PATH),
                "--schema",
                str(runner.SCHEMA_PATH),
                "--manifest",
                "/tmp/manifest.json",
                "--verify-git",
            ],
        )

    def test_head_blob_binding_uses_actual_bytes_and_detects_tamper(self) -> None:
        with tempfile.TemporaryDirectory(prefix="m4-v5-runner-binding-") as directory:
            root = Path(directory)
            files = {
                runner.SCHEMA_RELATIVE_PATH: json.dumps(
                    {
                        "$id": runner.SCHEMA_ID,
                        "title": runner.SCHEMA_TITLE,
                        "properties": {
                            "schema_version": {"const": runner.SCHEMA_VERSION}
                        },
                    }
                )
                + "\n",
                runner.VALIDATOR_RELATIVE_PATH: "# validator\n",
                runner.RUNNER_RELATIVE_PATH: "# runner\n",
            }
            for relative, content in files.items():
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            subprocess.run(
                ["git", "config", "user.email", "tests@example.invalid"],
                cwd=root,
                check=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "MOAI tests"], cwd=root, check=True
            )
            subprocess.run(["git", "add", "."], cwd=root, check=True)
            subprocess.run(["git", "commit", "-qm", "fixture"], cwd=root, check=True)
            head = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            binding = runner._contract_schema_binding(root, head)
            for role, relative in (
                ("schema", runner.SCHEMA_RELATIVE_PATH),
                ("validator", runner.VALIDATOR_RELATIVE_PATH),
                ("runner", runner.RUNNER_RELATIVE_PATH),
            ):
                self.assertEqual(binding[role]["path"], relative)
                self.assertEqual(
                    binding[role]["sha256"],
                    hashlib.sha256((root / relative).read_bytes()).hexdigest(),
                )
            (root / runner.RUNNER_RELATIVE_PATH).write_text(
                "# tampered runner\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(
                runner.ArtifactRunnerError, "differ from HEAD blob"
            ):
                runner._contract_schema_binding(root, head)

    def test_build_configuration_rejects_flags_stale_source_and_toolchain(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="m4-v5-cache-") as directory:
            root = Path(directory)
            build_root = root / "build-openfhe"
            prefix = root / "openfhe-prefix"
            package_root = prefix / "lib" / "OpenFHE"
            package_root.mkdir(parents=True)
            (package_root / "OpenFHEConfigVersion.cmake").write_text(
                'set(PACKAGE_VERSION "1.5.1")\n',
                encoding="utf-8",
            )
            executable = build_root / "openfhe_encoder_layer_smoke"
            build_root.mkdir()

            def write_cache(
                *,
                cxx_flags: str = "",
                source_root: Path = runner.REPO_ROOT,
                extra: str = "",
            ) -> None:
                lines = [
                    "CMAKE_GENERATOR:INTERNAL=Unix Makefiles",
                    f"CMAKE_HOME_DIRECTORY:INTERNAL={source_root}",
                    "CMAKE_BUILD_TYPE:STRING=Release",
                    "BUILD_TESTING:BOOL=ON",
                    f"OpenFHE_DIR:PATH={package_root}",
                    f"CMAKE_CXX_COMPILER:FILEPATH={runner.SYSTEM_CXX}",
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
                record = runner._verify_build_configuration(build_root, prefix)
                self.assertEqual(
                    record["environment_inheritance"],
                    "ambient_minus_cleared_variables",
                )
                self.assertEqual(
                    record["forced_environment_variables"],
                    runner.FORCED_SUBPROCESS_ENVIRONMENT,
                )

                write_cache(cxx_flags="-march=native")
                with self.assertRaisesRegex(
                    runner.ArtifactRunnerError,
                    "CMake cache binding drifted",
                ):
                    runner._verify_build_configuration(build_root, prefix)

                write_cache(source_root=root / "stale-checkout")
                with self.assertRaisesRegex(
                    runner.ArtifactRunnerError,
                    "CMake cache binding drifted",
                ):
                    runner._verify_build_configuration(build_root, prefix)

                write_cache(extra="CMAKE_TOOLCHAIN_FILE:FILEPATH=/tmp/injected.cmake")
                with self.assertRaisesRegex(runner.ArtifactRunnerError, "forbidden"):
                    runner._verify_build_configuration(build_root, prefix)

    def test_linkage_requires_exact_frozen_sonames_and_files(self) -> None:
        with tempfile.TemporaryDirectory(prefix="m4-v5-linkage-") as directory:
            root = Path(directory)
            prefix = root / "openfhe-prefix"
            library_root = prefix / "lib"
            library_root.mkdir(parents=True)
            executable = root / "openfhe_encoder_layer_smoke"
            executable.write_bytes(b"synthetic\n")
            paths: dict[str, Path] = {}
            for soname in runner.OPENFHE_LINKED_LIBRARY_SONAMES:
                path = library_root / runner.OPENFHE_LINKED_LIBRARY_BASENAMES[soname]
                path.write_bytes(f"{soname}\n".encode())
                paths[soname] = path

            def completed(
                lines: list[tuple[str, Path]],
            ) -> subprocess.CompletedProcess[str]:
                return subprocess.CompletedProcess(
                    [str(runner.SYSTEM_LDD), str(executable)],
                    0,
                    stdout="".join(
                        f"\t{soname} => {path} (0x00000001)\n" for soname, path in lines
                    ),
                    stderr="",
                )

            valid = [
                (soname, paths[soname])
                for soname in runner.OPENFHE_LINKED_LIBRARY_SONAMES
            ]
            with (
                mock.patch.object(runner, "DEFAULT_OPENFHE_PREFIX", prefix),
                mock.patch.object(
                    runner.subprocess, "run", return_value=completed(valid)
                ),
            ):
                _, records = runner._inspect_openfhe_linkage(executable, prefix)
            self.assertEqual(
                [record["soname"] for record in records],
                list(runner.OPENFHE_LINKED_LIBRARY_SONAMES),
            )

            for name, lines in (
                ("missing", valid[:-1]),
                ("duplicate", [*valid, valid[0]]),
            ):
                with (
                    self.subTest(name=name),
                    mock.patch.object(runner, "DEFAULT_OPENFHE_PREFIX", prefix),
                    mock.patch.object(
                        runner.subprocess, "run", return_value=completed(lines)
                    ),
                    self.assertRaises(runner.ArtifactRunnerError),
                ):
                    runner._inspect_openfhe_linkage(executable, prefix)

            outside = (
                root / runner.OPENFHE_LINKED_LIBRARY_BASENAMES["libOPENFHEcore.so.1"]
            )
            outside.write_bytes(b"outside\n")
            outside_lines = [
                (
                    soname,
                    outside if soname == "libOPENFHEcore.so.1" else paths[soname],
                )
                for soname in runner.OPENFHE_LINKED_LIBRARY_SONAMES
            ]
            with (
                mock.patch.object(runner, "DEFAULT_OPENFHE_PREFIX", prefix),
                mock.patch.object(
                    runner.subprocess,
                    "run",
                    return_value=completed(outside_lines),
                ),
                self.assertRaisesRegex(runner.ArtifactRunnerError, "outside"),
            ):
                runner._inspect_openfhe_linkage(executable, prefix)

            core = paths["libOPENFHEcore.so.1"]
            core.unlink()
            core.symlink_to(outside)
            with (
                mock.patch.object(runner, "DEFAULT_OPENFHE_PREFIX", prefix),
                mock.patch.object(
                    runner.subprocess, "run", return_value=completed(valid)
                ),
                self.assertRaisesRegex(runner.ArtifactRunnerError, "outside"),
            ):
                runner._inspect_openfhe_linkage(executable, prefix)

    def test_build_provenance_byte_drift_is_fail_closed(self) -> None:
        expected = {
            "cmake_cache": {"sha256": "a" * 64},
            "openfhe_cmake_package_files": [{"sha256": "b" * 64}],
            "openfhe_include_tree": {"sha256": "c" * 64},
            "openfhe_linked_libraries": [{"sha256": "d" * 64}],
        }
        drifted = copy.deepcopy(expected)
        drifted["openfhe_cmake_package_files"][0]["sha256"] = "0" * 64
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


if __name__ == "__main__":
    unittest.main()
